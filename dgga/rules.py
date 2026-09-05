# -*- coding: utf-8 -*-
"""dgga.rules - 规则控制引擎（梯队3；rules.c 逐字复刻）。

范围（源文件:行号在各方法注释）：
- 解析：newrule/newpremise/newaction/newpriority（rules.c:552-808） -
  从 Net.rules_raw 的 [RULES] 原始行重建前提/动作/优先级；数值一律存用户原值
  （checkvalue 把当前状态换算到用户单位再比，rules.c:939-1036）；
- 求值：evalpremises 的短路布尔（AND 链短路 / OR 恢复，rules.c:810-836）、
  checktime（:849-912）、checkstatus（:914-937）、checkvalue（:939-1036）；
- 仲裁：updateactionlist 头插 + onactionlist 的 priority 替换（:1038-1105）；
- 执行：takeactions → setlinkstatus/setlinksetting（hydraul.c:377-462，含
  resetpumpflow :1103-1116 恒功率泵复位）。

引擎持 EpsDriver 引用读运行态（Htime/S/K/q/H/NodeDemand/Dsystem/tankV）。
float64。不支持的对象/变量组合照 EPANET 报错语义抛异常。
"""

import numpy as np

try:
    from dgga.parse import Net, _hour
    from dgga.solver import MISSING, TINY
except ImportError:  # pragma: no cover
    from parse import Net, _hour
    from solver import MISSING, TINY

SECperDAY = 86400          # types.h:83

# Rulewords（rules.c:27-38）
_RULEWORDS = ["RULE", "IF", "AND", "OR", "THEN", "ELSE", "PRIORITY"]
r_RULE, r_IF, r_AND, r_OR, r_THEN, r_ELSE, r_PRIORITY = range(7)

# Varwords（rules.c:40-57）
_VARWORDS = ["DEMAND", "HEAD", "GRADE", "LEVEL", "PRESSURE", "FLOW",
             "STATUS", "SETTING", "POWER", "TIME", "CLOCKTIME",
             "FILLTIME", "DRAINTIME"]
(r_DEMAND, r_HEAD, r_GRADE, r_LEVEL, r_PRESSURE, r_FLOW, r_STATUS,
 r_SETTING, r_POWER, r_TIME, r_CLOCKTIME, r_FILLTIME, r_DRAINTIME) = range(13)

# Objects（rules.c:59-71）
_OBJECTS = ["JUNC", "RESER", "TANK", "PIPE", "PUMP", "VALVE", "NODE",
            "LINK", "SYSTEM"]
(r_JUNC, r_RESERV, r_TANK, r_PIPE, r_PUMP, r_VALVE, r_NODE, r_LINK,
 r_SYSTEM) = range(9)

# Operators（rules.c:73-76；"<="/">=" 必须排在 "<"/">" 前）
_OPERATORS = ["=", "<>", "<=", ">=", "<", ">", "IS", "NOT", "BELOW", "ABOVE"]
EQ, NE, LE, GE, LT, GT, IS, NOT, BELOW, ABOVE = range(10)

# Values（rules.c:78-79）
_VALUES = ["XXXX", "OPEN", "CLOSED", "ACTIVE"]
IS_NUMBER, IS_OPEN, IS_CLOSED, IS_ACTIVE = range(4)

# EN_LinkType（与 solver 一致）
_CVPIPE, _PIPE, _PUMP, _PRV, _PSV, _PBV, _FCV, _TCV, _GPV = range(9)


def _match(s, substr):
    """match（input2.c:643-672）：substr 为 s 的前缀（大小写不敏感）。"""
    if not substr:
        return False
    s = s.lstrip(" ")
    return s.upper().startswith(substr.upper())


def _findmatch(tok, words):
    """findmatch（input2.c:623-641）：返回首个 match 的词索引，无则 -1。"""
    for i, w in enumerate(words):
        if _match(tok, w):
            return i
    return -1


def _getfloat(tok):
    """getfloat：strtod 语义；失败返回 None。"""
    try:
        return float(tok)
    except ValueError:
        return None


class _Premise:
    __slots__ = ("logop", "object", "index", "variable", "relop", "status", "value")

    def __init__(self, logop, obj, index, variable, relop, status, value):
        self.logop = logop        # r_AND / r_OR
        self.object = obj         # r_NODE / r_LINK / r_SYSTEM
        self.index = index        # 0 基节点/管段索引；SYSTEM 为 -1
        self.variable = variable  # Varwords
        self.relop = relop        # Operators（归一化后 EQ/NE/LT/LE/GT/GE）
        self.status = status      # IS_OPEN/IS_CLOSED/IS_ACTIVE 或 0
        self.value = value        # 用户单位数值（时间为秒）；MISSING=无


class _Action:
    __slots__ = ("link", "status", "setting")

    def __init__(self, link, status, setting):
        self.link = link          # 0 基管段索引
        self.status = status      # IS_OPEN/IS_CLOSED 或 -1
        self.setting = setting    # 用户单位设定；MISSING=无


class _Rule:
    __slots__ = ("label", "premises", "then_actions", "else_actions", "priority")

    def __init__(self, label):
        self.label = label
        self.premises = []
        self.then_actions = []
        self.else_actions = []
        self.priority = 0.0       # newrule（rules.c:566）


class RuleEngine:
    """RuleEngine(net, solver, driver)。driver 供运行态：Htime/Tstart/S/K/q/H/
    node_dem（EN_DEMAND 口径 NodeDemand）/Dsystem/tankV。"""

    def __init__(self, net: Net, solver, driver):
        self.net = net
        self.s = solver
        self.drv = driver
        self.rules = []
        self._node_index = {nid: i for i, nid in enumerate(net.node_id)}
        self._link_index = {lid: k for k, lid in enumerate(net.link_id)}
        self._tank_of_node = {int(n): j for j, n in enumerate(
            np.asarray(net.tank_node, dtype=np.int64))}
        self._parse(net.rules_raw)

    # ------------------------------------------------------------- 解析
    def _parse(self, raw_lines):
        """ruledata（rules.c:184-288）状态机：RULE/IF/AND/OR/THEN/ELSE/PRIORITY。"""
        state = r_PRIORITY                       # initrules（rules.c:110）
        cur = None
        last_list = None                         # 当前动作挂载列表（THEN/ELSE）
        for raw in raw_lines:
            body = raw.split(";", 1)[0].strip()
            if not body:
                continue
            tok = body.split()
            key = _findmatch(tok[0], _RULEWORDS)
            if key == r_RULE:                    # rules.c:209-220
                cur = _Rule(tok[1] if len(tok) >= 2 else "")
                self.rules.append(cur)
                state = r_RULE
            elif key == r_IF:                    # rules.c:222-230
                if state != r_RULE:
                    raise ValueError(f"[RULES] 错置 IF（错误 221）: {raw!r}")
                state = r_IF
                cur.premises.append(self._new_premise(tok, r_AND))
            elif key == r_AND:                   # rules.c:232-239
                if state == r_IF:
                    cur.premises.append(self._new_premise(tok, r_AND))
                elif state in (r_THEN, r_ELSE):
                    last_list.append(self._new_action(tok))
                else:
                    raise ValueError(f"[RULES] 错置 AND（错误 221）: {raw!r}")
            elif key == r_OR:                    # rules.c:241-244
                if state != r_IF:
                    raise ValueError(f"[RULES] 错置 OR（错误 221）: {raw!r}")
                cur.premises.append(self._new_premise(tok, r_OR))
            elif key == r_THEN:                  # rules.c:246-254
                if state != r_IF:
                    raise ValueError(f"[RULES] 错置 THEN（错误 221）: {raw!r}")
                state = r_THEN
                last_list = cur.then_actions
                last_list.append(self._new_action(tok))
            elif key == r_ELSE:                  # rules.c:256-264
                if state != r_THEN:
                    raise ValueError(f"[RULES] 错置 ELSE（错误 221）: {raw!r}")
                state = r_ELSE
                last_list = cur.else_actions
                last_list.append(self._new_action(tok))
            elif key == r_PRIORITY:              # rules.c:266-274
                if state not in (r_THEN, r_ELSE):
                    raise ValueError(f"[RULES] 错置 PRIORITY（错误 221）: {raw!r}")
                state = r_PRIORITY
                x = _getfloat(tok[1])            # newpriority（rules.c:795-808）
                if x is None:
                    raise ValueError(f"[RULES] PRIORITY 非数（错误 202）: {raw!r}")
                cur.priority = x
            else:
                raise ValueError(f"[RULES] 未知关键词（错误 201）: {raw!r}")

    def _new_premise(self, tok, logop):
        """newpremise（rules.c:572-715）。"""
        net = self.net
        ntok = len(tok)
        if ntok not in (5, 6):
            raise ValueError(f"[RULES] 前提字段数非法（错误 201）: {tok}")
        i = _findmatch(tok[1], _OBJECTS)
        if i == r_SYSTEM:                        # rules.c:594-599
            j = -1
            v = _findmatch(tok[2], _VARWORDS)
            if v not in (r_DEMAND, r_TIME, r_CLOCKTIME):
                raise ValueError(f"[RULES] SYSTEM 变量非法（错误 201）: {tok}")
        else:
            v = _findmatch(tok[3], _VARWORDS)
            if v < 0:
                raise ValueError(f"[RULES] 变量非法（错误 201）: {tok}")
            if i in (r_NODE, r_JUNC, r_RESERV, r_TANK):    # rules.c:604-611
                i = r_NODE
            elif i in (r_LINK, r_PIPE, r_PUMP, r_VALVE):   # rules.c:612-617
                i = r_LINK
            else:
                raise ValueError(f"[RULES] 对象非法（错误 201）: {tok}")
            if i == r_NODE:
                j = self._node_index.get(tok[2])
                if j is None:
                    raise ValueError(f"[RULES] 未知节点（错误 203）: {tok}")
                if v in (r_FILLTIME, r_DRAINTIME):         # rules.c:634-637
                    if self.net.node_type[j] == 0:
                        raise ValueError(f"[RULES] FILLTIME 用于 junction（201）: {tok}")
                elif v not in (r_DEMAND, r_HEAD, r_GRADE, r_LEVEL, r_PRESSURE):
                    raise ValueError(f"[RULES] 节点变量非法（错误 201）: {tok}")
            else:
                j = self._link_index.get(tok[2])
                if j is None:
                    raise ValueError(f"[RULES] 未知管段（错误 204）: {tok}")
                if v not in (r_FLOW, r_STATUS, r_SETTING):  # rules.c:646-654
                    raise ValueError(f"[RULES] 管段变量非法（错误 201）: {tok}")
        # 关系算子（rules.c:658-679）
        m = 3 if i == r_SYSTEM else 4
        k = _findmatch(tok[m], _OPERATORS)
        if k < 0:
            raise ValueError(f"[RULES] 算子非法（错误 201）: {tok}")
        if k == IS:
            r = EQ
        elif k == NOT:
            r = NE
        elif k == BELOW:
            r = LT
        elif k == ABOVE:
            r = GT
        else:
            r = k
        # 状态或数值（rules.c:681-696）
        s = 0
        x = MISSING
        if v in (r_TIME, r_CLOCKTIME):
            if ntok == 6:
                x = _hour(tok[4], tok[5]) * 3600.0     # rules.c:686
            else:
                x = _hour(tok[4], "") * 3600.0         # rules.c:687
            if x < 0.0:
                raise ValueError(f"[RULES] 时间非法（错误 202）: {tok}")
        else:
            k = _findmatch(tok[ntok - 1], _VALUES)
            if k > IS_NUMBER:                          # rules.c:690
                s = k
            else:
                x = _getfloat(tok[ntok - 1])           # rules.c:693
                if x is None:
                    raise ValueError(f"[RULES] 数值非法（错误 202）: {tok}")
                if v in (r_FILLTIME, r_DRAINTIME):
                    x = x * 3600.0                     # rules.c:695
        return _Premise(logop, i, j, v, r, s, x)

    def _new_action(self, tok):
        """newaction（rules.c:717-793）。格式 THEN/ELSE/AND LINK <id> <var> IS <值>。"""
        if len(tok) != 6:
            raise ValueError(f"[RULES] 动作字段数非法（错误 201）: {tok}")
        j = self._link_index.get(tok[2])
        if j is None:
            raise ValueError(f"[RULES] 动作未知管段（错误 204）: {tok}")
        lt = int(self.net.link_type[j])
        if lt == _CVPIPE:                              # rules.c:741
            raise ValueError(f"[RULES] 不能控制 CV 管（错误 207）: {tok}")
        s = -1
        x = MISSING
        k = _findmatch(tok[5], _VALUES)
        if k > IS_NUMBER:                              # rules.c:746
            s = k
        else:
            x = _getfloat(tok[5])                      # rules.c:749
            if x is None or x < 0.0:
                raise ValueError(f"[RULES] 动作数值非法（错误 202）: {tok}")
        if x != MISSING and lt == _GPV:                # rules.c:754
            raise ValueError(f"[RULES] GPV 不能改设定（错误 202）: {tok}")
        if x != MISSING and lt == _PIPE:               # rules.c:757-762
            s = IS_CLOSED if x == 0.0 else IS_OPEN
            x = MISSING
        return _Action(j, s, x)

    # ------------------------------------------------------------- 求值
    def checkrules(self, dt):
        """checkrules（rules.c:511-550）：返回实际执行的动作数。"""
        drv = self.drv
        self._time1 = drv.Htime - dt + 1               # rules.c:524
        action_list = []                               # (action, rule_idx) 头插链
        for i, rule in enumerate(self.rules):          # rules.c:528-544
            if self._evalpremises(rule):
                self._updateactionlist(i, rule.then_actions, action_list)
            else:
                if rule.else_actions:
                    self._updateactionlist(i, rule.else_actions, action_list)
        n = 0
        if action_list:
            n = self._takeactions(action_list)         # rules.c:547
        return n

    def _evalpremises(self, rule):
        """evalpremises（rules.c:810-836）：AND 短路 / OR 恢复。"""
        result = True
        for p in rule.premises:
            if p.logop == r_OR:                        # rules.c:824-827
                if not result:
                    result = self._checkpremise(p)
            else:
                if not result:                         # rules.c:830
                    return False
                result = self._checkpremise(p)
        return result

    def _checkpremise(self, p):
        """checkpremise（rules.c:838-847）。"""
        if p.variable in (r_TIME, r_CLOCKTIME):
            return self._checktime(p)
        if p.status > IS_NUMBER:
            return self._checkstatus(p)
        return self._checkvalue(p)

    def _checktime(self, p):
        """checktime（rules.c:849-912）。"""
        drv = self.drv
        if p.variable == r_TIME:                       # rules.c:861-865
            t1 = self._time1
            t2 = drv.Htime
        elif p.variable == r_CLOCKTIME:                # rules.c:866-870
            t1 = (self._time1 + drv.Tstart) % SECperDAY
            t2 = (drv.Htime + drv.Tstart) % SECperDAY
        else:
            return 0
        x = int(p.value)                               # rules.c:874
        if p.relop == LT:
            if t2 >= x:
                return 0                               # rules.c:879
        elif p.relop == LE:
            if t2 > x:
                return 0                               # rules.c:882
        elif p.relop == GT:
            if t2 <= x:
                return 0                               # rules.c:885
        elif p.relop == GE:
            if t2 < x:
                return 0                               # rules.c:888
        elif p.relop in (EQ, NE):                      # rules.c:892-907
            flag = False
            if t2 < t1:                                # 跨午夜区间
                if x >= t1 or x <= t2:
                    flag = True
            else:
                if t1 <= x <= t2:
                    flag = True
            if p.relop == EQ and not flag:
                return 0
            if p.relop == NE and flag:
                return 0
        return 1                                       # rules.c:911

    def _checkstatus(self, p):
        """checkstatus（rules.c:914-937）。"""
        s = self.s
        if p.status in (IS_OPEN, IS_CLOSED, IS_ACTIVE):
            i = int(self.drv.S[p.index])               # rules.c:929
            if i <= s.ST_CLOSED:
                j = IS_CLOSED                          # rules.c:930
            elif i == s.ST_ACTIVE:
                j = IS_ACTIVE                          # rules.c:931
            else:
                j = IS_OPEN                            # rules.c:932
            if j == p.status and p.relop == EQ:
                return 1                               # rules.c:933
            if j != p.status and p.relop == NE:
                return 1                               # rules.c:934
        return 0

    def _checkvalue(self, p):
        """checkvalue（rules.c:939-1036）：用户单位、容差 1e-3。"""
        net = self.net
        s = self.s
        drv = self.drv
        tol = 1.0e-3                                   # rules.c:950
        i = p.index
        v = p.variable
        if v == r_DEMAND:                              # rules.c:965-968
            if p.object == r_SYSTEM:
                x = drv.Dsystem * s.ucf_flow           # Ucf[DEMAND]=qcf（input1.c:492）
            else:
                x = drv.node_dem[i] * s.ucf_flow
        elif v in (r_HEAD, r_GRADE):                   # rules.c:970-973
            x = drv.H[i] * s.ucf_head
        elif v == r_PRESSURE:                          # rules.c:975-977
            x = (drv.H[i] - net.elev_ft[i]) * s.ucf_pressure
        elif v == r_LEVEL:                             # rules.c:979-981
            x = (drv.H[i] - net.elev_ft[i]) * s.ucf_head
        elif v == r_FLOW:                              # rules.c:983-985
            x = abs(drv.q[i]) * s.ucf_flow
        elif v == r_SETTING:                           # rules.c:987-1003
            if drv.K[i] == MISSING:
                return 0
            x = float(drv.K[i])
            lt = int(net.link_type[i])
            if lt in (_PRV, _PSV, _PBV):
                x = x * s.ucf_pressure
            elif lt == _FCV:
                x = x * s.ucf_flow
        elif v == r_FILLTIME:                          # rules.c:1005-1011
            j = self._tank_of_node.get(i)
            if j is None:
                return 0
            if drv.node_dem[i] <= TINY:
                return 0
            x = (net.tank_vmax[j] - drv.tankV[j]) / drv.node_dem[i]
        elif v == r_DRAINTIME:                         # rules.c:1013-1019
            j = self._tank_of_node.get(i)
            if j is None:
                return 0
            if drv.node_dem[i] >= -TINY:
                return 0
            x = (net.tank_vmin[j] - drv.tankV[j]) / drv.node_dem[i]
        else:
            return 0                                   # rules.c:1021-1022
        # 与前提值比较（rules.c:1026-1034）
        if p.relop == EQ:
            if abs(x - p.value) > tol:
                return 0
        elif p.relop == NE:
            if abs(x - p.value) < tol:
                return 0
        elif p.relop == LT:
            if x > p.value + tol:
                return 0
        elif p.relop == LE:
            if x > p.value - tol:
                return 0
        elif p.relop == GT:
            if x < p.value - tol:
                return 0
        elif p.relop == GE:
            if x < p.value + tol:
                return 0
        return 1

    # ------------------------------------------------------------- 仲裁与执行
    def _updateactionlist(self, i, actions, action_list):
        """updateactionlist（rules.c:1038-1066）：头插；同管段按 priority 替换
        （onactionlist rules.c:1068-1105）。"""
        for a in actions:
            on_list = False
            for item in action_list:                   # onactionlist 自表头扫描
                a1, i1 = item
                if a.link == a1.link:                  # rules.c:1088
                    if self.rules[i].priority > self.rules[i1].priority:  # :1091
                        item[0] = a
                        item[1] = i
                    on_list = True                     # rules.c:1098
                    break
            if not on_list:
                action_list.insert(0, [a, i])          # rules.c:1060-1061 头插

    def _takeactions(self, action_list):
        """takeactions（rules.c:1107-1186）：按链表序（头插後进先出）执行。"""
        net = self.net
        s = self.s
        drv = self.drv
        tol = 1.0e-3                                   # rules.c:1119
        n = 0
        for a, _i in action_list:
            flag = False
            k = a.link
            st = int(drv.S[k])
            v = float(drv.K[k])
            x = a.setting
            if a.status == IS_OPEN and st <= s.ST_CLOSED:      # rules.c:1135-1139
                self._setlinkstatus(k, 1)
                flag = True
            elif a.status == IS_CLOSED and st > s.ST_CLOSED:   # rules.c:1142-1146
                self._setlinkstatus(k, 0)
                flag = True
            elif x != MISSING:                                 # rules.c:1149-1170
                lt = int(net.link_type[k])
                if lt in (_PRV, _PSV, _PBV):
                    x = x / s.ucf_pressure                     # rules.c:1156
                elif lt == _FCV:
                    x = x / s.ucf_flow                         # rules.c:1159
                if abs(x - v) > tol:                           # rules.c:1164
                    self._setlinksetting(k, x)
                    flag = True
            if flag:
                n += 1
        return n

    def _setlinkstatus(self, k, value):
        """setlinkstatus（hydraul.c:377-415）。"""
        s = self.s
        drv = self.drv
        t = int(self.net.link_type[k])
        if value == 1:                                 # hydraul.c:394-405
            if t == _PUMP:
                drv.K[k] = 1.0                         # :399
                if int(drv.S[k]) == s.ST_CLOSED:       # :401
                    self._resetpumpflow(k)
            if t > _PUMP and t != _GPV:
                drv.K[k] = MISSING                     # :403
            drv.S[k] = s.ST_OPEN                       # :404
        elif value == 0:                               # hydraul.c:408-414
            if t == _PUMP:
                drv.K[k] = 0.0                         # :411
            if t > _PUMP and t != _GPV:
                drv.K[k] = MISSING                     # :412
            drv.S[k] = s.ST_CLOSED                     # :413

    def _setlinksetting(self, k, value):
        """setlinksetting（hydraul.c:418-462）。"""
        s = self.s
        drv = self.drv
        t = int(self.net.link_type[k])
        if t == _PUMP:                                 # hydraul.c:437-447
            drv.K[k] = value
            if value > 0 and int(drv.S[k]) <= s.ST_CLOSED:
                self._resetpumpflow(k)
                drv.S[k] = s.ST_OPEN
            if value == 0 and int(drv.S[k]) > s.ST_CLOSED:
                drv.S[k] = s.ST_CLOSED
        elif t == _FCV:                                # hydraul.c:450-454
            drv.K[k] = value
            drv.S[k] = s.ST_ACTIVE
        else:                                          # hydraul.c:457-461
            if drv.K[k] == MISSING and int(drv.S[k]) <= s.ST_CLOSED:
                drv.S[k] = s.ST_OPEN
            drv.K[k] = value

    def _resetpumpflow(self, k):
        """resetpumpflow（hydraul.c:1103-1116）：仅恒功率泵把流量复位为 Q0。"""
        net = self.net
        pl = np.asarray(net.pump_link, dtype=np.int64)
        j = int(np.where(pl == k)[0][0])
        if net.pump_ptype[j] == 0:                     # CONST_HP
            self.drv.q[k] = float(net.pump_q0[j])

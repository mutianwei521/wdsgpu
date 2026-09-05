# -*- coding: utf-8 -*-
"""ctypes 封装 wntr 自带的双精度 EPANET 2.2 DLL（epanet22.dll）。

契约见 dgga/CONTRACT.md：
- 所有对外返回值均换算为 EPANET 内部单位（长度/水头 ft、流量 cfs），float64；
- 返回码 >100 抛 RuntimeError（附 EN_geterror 文本），1..100 收集为警告；
- 函数原型逐条核对 ref/epanet2.2_toolkit/epanet2_2.h，
  枚举值逐条抄自 ref/epanet2.2_toolkit/epanet2_enums.h（注释标行号）。
"""

import ctypes
import os
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# 单位换算常数（逐字抄自 ref/EPANET2.2-2.2.0/SRC_engines/types.h:68-83）
# TODO: chain-1 交付 dgga/units.py 后切换为 `from dgga.units import ...`
# ---------------------------------------------------------------------------
try:  # pragma: no cover - 优先用 chain-1 的常数，保持全工程同一常数体系
    from dgga.units import (GPMperCFS, LPSperCFS, MperFT, PSIperFT,  # type: ignore
                            KPAperPSI, FLOW_UCF)
except ImportError:  # chain-1 尚未交付时的本地兜底（数值与 types.h 完全一致）
    GPMperCFS = 448.831   # types.h:68
    LPSperCFS = 28.317    # types.h:72
    MperFT = 0.3048       # types.h:79
    PSIperFT = 0.4333     # types.h:80
    KPAperPSI = 6.895     # types.h:81
    FLOW_UCF = {"CFS": 1.0, "GPM": 448.831, "MGD": 0.64632, "IMGD": 0.5382,
                "AFD": 1.9837, "LPS": 28.317, "LPM": 1699.0, "MLD": 2.4466,
                "CMH": 101.94, "CMD": 2446.6}   # types.h:68-76

# ---------------------------------------------------------------------------
# 枚举值（抄自 ref/epanet2.2_toolkit/epanet2_enums.h，注释为行号）
# ---------------------------------------------------------------------------
EN_MAXID = 31    # epanet2_enums.h:29  ID 最大字符数
EN_MAXMSG = 255  # epanet2_enums.h:30  消息最大字符数

# EN_NodeProperty
EN_ELEVATION = 0  # epanet2_enums.h:39  高程（reservoir 为定水头）
EN_BASEDEMAND = 1  # epanet2_enums.h:40  首类需水基值
EN_EMITTER = 3    # epanet2_enums.h:42  emitter 流量系数（用户单位原值）
EN_DEMAND = 9     # epanet2_enums.h:48  当前计算需水（只读）
EN_HEAD = 10      # epanet2_enums.h:49  当前计算水头（只读）
EN_PRESSURE = 11  # epanet2_enums.h:50  当前计算压力（只读）

# EN_LinkProperty
EN_DIAMETER = 0     # epanet2_enums.h:75  管/阀直径
EN_LENGTH = 1       # epanet2_enums.h:76  管长
EN_ROUGHNESS = 2    # epanet2_enums.h:77  粗糙系数（H-W 的 C，无量纲）
EN_MINORLOSS = 3    # epanet2_enums.h:78  局损系数（无量纲）
EN_INITSTATUS = 4   # epanet2_enums.h:79  初始状态
EN_INITSETTING = 5  # epanet2_enums.h:80  初始泵速/阀设定（用户原值）
EN_FLOW = 8      # epanet2_enums.h:83  当前计算流量（只读）
EN_STATUS = 11   # epanet2_enums.h:86  当前管段状态
EN_SETTING = 12  # epanet2_enums.h:87  当前管段设定

# EN_Option
EN_TRIALS = 0    # epanet2_enums.h:298  最大迭代数
EN_ACCURACY = 1  # epanet2_enums.h:299  水力收敛精度（EN_setoption 不做 1e-5 钳位，
                 # 对照 input3.c:2014-2019 仅 INP 解析路径钳位）

# EN_AnalysisStatistic
EN_ITERATIONS = 0     # epanet2_enums.h:132  水力迭代次数
EN_RELATIVEERROR = 1  # epanet2_enums.h:133  流量相对误差

# EN_CountType
EN_NODECOUNT = 0     # epanet2_enums.h:159
EN_TANKCOUNT = 1     # epanet2_enums.h:160  水池+水库数
EN_LINKCOUNT = 2     # epanet2_enums.h:161
EN_PATCOUNT = 3      # epanet2_enums.h:162
EN_CURVECOUNT = 4    # epanet2_enums.h:163
EN_CONTROLCOUNT = 5  # epanet2_enums.h:164
EN_RULECOUNT = 6     # epanet2_enums.h:165

# EN_NodeType
EN_JUNCTION = 0   # epanet2_enums.h:173
EN_RESERVOIR = 1  # epanet2_enums.h:174
EN_TANK = 2       # epanet2_enums.h:175

# EN_LinkType
EN_CVPIPE = 0  # epanet2_enums.h:183
EN_PIPE = 1    # epanet2_enums.h:184
EN_PUMP = 2    # epanet2_enums.h:185
EN_PRV = 3     # epanet2_enums.h:186
EN_PSV = 4     # epanet2_enums.h:187
EN_PBV = 5     # epanet2_enums.h:188
EN_FCV = 6     # epanet2_enums.h:189
EN_TCV = 7     # epanet2_enums.h:190
EN_GPV = 8     # epanet2_enums.h:191

# EN_FlowUnits（0..4 为 US 制，5..9 为 SI 制）
EN_CFS = 0   # epanet2_enums.h:264
EN_GPM = 1   # epanet2_enums.h:265
EN_MGD = 2   # epanet2_enums.h:266
EN_IMGD = 3  # epanet2_enums.h:267
EN_AFD = 4   # epanet2_enums.h:268
EN_LPS = 5   # epanet2_enums.h:269
EN_LPM = 6   # epanet2_enums.h:270
EN_MLD = 7   # epanet2_enums.h:271
EN_CMH = 8   # epanet2_enums.h:272
EN_CMD = 9   # epanet2_enums.h:273

# EN_InitHydOption
EN_NOSAVE = 0         # epanet2_enums.h:369  不存水力文件、不重置流量
EN_SAVE = 1           # epanet2_enums.h:370
EN_SAVE_AND_INIT = 11  # epanet2_enums.h:372

# EN_Option
EN_SP_GRAVITY = 12  # epanet2_enums.h:310  比重（压力换算系数含此因子，见 input1.c:451-452）


def _dll_path() -> str:
    """按契约用 wntr 包目录定位双精度 EPANET 2.2 共享库。

    wntr 为每个平台各带一份，目录名与扩展名都不同。位级复刻的基准是
    Windows 构建（msvcrt 的 pow/log）；其它平台可加载并可运行，但
    Section "Limitations" 第 (2) 条说明的位级断言不适用于它们。
    """
    import wntr

    base = os.path.join(os.path.dirname(wntr.__file__), "epanet", "libepanet")
    if sys.platform.startswith("win"):
        sub, name = "windows-x64", "epanet22.dll"
    elif sys.platform == "darwin":
        sub, name = "darwin-x64", "libepanet22.dylib"
    else:
        sub, name = "linux-x64", "libepanet22.so"
    path = os.path.join(base, sub, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "EPANET 2.2 shared library not found for platform %r at %s"
            % (sys.platform, path))
    return path


def _load_lib() -> ctypes.CDLL:
    """加载 DLL 并逐个声明函数原型（原型核对自 ref/epanet2.2_toolkit/epanet2_2.h）。

    要点：EN_Project 用 c_void_p；时间参数 long*（Windows LLP64 下 c_long=32 位，
    与 DLL 的 long 一致）；数值输出 double*；字符串缓冲 char*。
    """
    lib = ctypes.CDLL(_dll_path())
    P = ctypes.c_void_p
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_long_p = ctypes.POINTER(ctypes.c_long)
    c_dbl_p = ctypes.POINTER(ctypes.c_double)
    c_char_pp = ctypes.c_char_p

    protos = {
        # epanet2_2.h:64  int EN_createproject(EN_Project *ph)
        "EN_createproject": ([ctypes.POINTER(P)],),
        # epanet2_2.h:73  int EN_deleteproject(EN_Project ph)
        "EN_deleteproject": ([P],),
        # epanet2_2.h:126 int EN_open(ph, inpFile, rptFile, outFile)
        "EN_open": ([P, c_char_pp, c_char_pp, c_char_pp],),
        # epanet2_2.h:195 int EN_close(ph)
        "EN_close": ([P],),
        # epanet2_2.h:258 int EN_openH(ph)
        "EN_openH": ([P],),
        # epanet2_2.h:286 int EN_initH(ph, int initFlag)
        "EN_initH": ([P, ctypes.c_int],),
        # epanet2_2.h:302 int EN_runH(ph, long *currentTime)
        "EN_runH": ([P, c_long_p],),
        # epanet2_2.h:336 int EN_nextH(ph, long *tStep)
        "EN_nextH": ([P, c_long_p],),
        # epanet2_2.h:375 int EN_closeH(ph)
        "EN_closeH": ([P],),
        # epanet2_2.h:176 int EN_getcount(ph, int object, int *count)
        "EN_getcount": ([P, ctypes.c_int, c_int_p],),
        # epanet2_2.h:811 int EN_getnodeid(ph, int index, char *out_id)
        "EN_getnodeid": ([P, ctypes.c_int, c_char_pp],),
        # epanet2_2.h:1135 int EN_getlinkid(ph, int index, char *out_id)
        "EN_getlinkid": ([P, ctypes.c_int, c_char_pp],),
        # epanet2_2.h:844 int EN_getnodevalue(ph, int index, int property, double *value)
        "EN_getnodevalue": ([P, ctypes.c_int, ctypes.c_int, c_dbl_p],),
        # epanet2_2.h:1202 int EN_getlinkvalue(ph, int index, int property, double *value)
        "EN_getlinkvalue": ([P, ctypes.c_int, ctypes.c_int, c_dbl_p],),
        # epanet2_2.h:831 int EN_getnodetype(ph, int index, int *nodeType)
        "EN_getnodetype": ([P, ctypes.c_int, c_int_p],),
        # epanet2_2.h:1155 int EN_getlinktype(ph, int index, int *linkType)
        "EN_getlinktype": ([P, ctypes.c_int, c_int_p],),
        # epanet2_2.h:1180 int EN_getlinknodes(ph, int index, int *node1, int *node2)
        "EN_getlinknodes": ([P, ctypes.c_int, c_int_p, c_int_p],),
        # epanet2_2.h:691 int EN_getflowunits(ph, int *units)
        "EN_getflowunits": ([P, c_int_p],),
        # epanet2_2.h:670 int EN_getoption(ph, int option, double *value)
        "EN_getoption": ([P, ctypes.c_int, c_dbl_p],),
        # epanet2_2.h:639 int EN_getstatistic(ph, int type, double* value)
        "EN_getstatistic": ([P, ctypes.c_int, c_dbl_p],),
        # epanet2_2.h:229 int EN_solveH(EN_Project ph)
        "EN_solveH": ([P],),
        # epanet2_2.h:856 int EN_setnodevalue(ph, int index, int property, double value)
        "EN_setnodevalue": ([P, ctypes.c_int, ctypes.c_int, ctypes.c_double],),
        # epanet2_2.h:680 int EN_setoption(ph, int option, double value)
        "EN_setoption": ([P, ctypes.c_int, ctypes.c_double],),
        # epanet2_2.h:1214 int EN_setlinkvalue(ph, int index, int property, double value)
        "EN_setlinkvalue": ([P, ctypes.c_int, ctypes.c_int, ctypes.c_double],),
        # epanet2_2.h:619 int EN_getversion(int *version) - 无项目句柄
        "EN_getversion": ([c_int_p],),
        # epanet2_2.h:630 int EN_geterror(int errcode, char *out_errmsg, int maxLen)
        "EN_geterror": ([ctypes.c_int, c_char_pp, ctypes.c_int],),
    }
    for name, (argtypes,) in protos.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
    return lib


def _errmsg(lib: ctypes.CDLL, code: int) -> str:
    """取 EN_geterror 文本。"""
    buf = ctypes.create_string_buffer(EN_MAXMSG + 1)
    lib.EN_geterror(code, buf, EN_MAXMSG)
    return buf.value.decode("ascii", errors="replace")


class Epanet:
    """双精度 EN_* API 封装。上下文管理器；对外一律返回内部单位 ft/cfs。"""

    def __init__(self, inp_path: str, rpt_path: str = None):
        self.lib = _load_lib()
        self.warnings = []  # [(阶段/时刻描述, 警告码, 文本)]
        self._ph = ctypes.c_void_p()
        self._tmp_rpt = None  # 需要在 close 时删除的临时 rpt

        inp_path = os.path.abspath(inp_path)
        if rpt_path is None:
            fd, rpt_path = tempfile.mkstemp(suffix=".rpt", prefix="epanet_ref_")
            os.close(fd)
            self._tmp_rpt = rpt_path
        self._rpt_path = rpt_path

        self._check(self.lib.EN_createproject(ctypes.byref(self._ph)), "EN_createproject")
        # 路径必须是 ASCII bytes（契约：无中文路径进 EN_open）
        self._check(
            self.lib.EN_open(
                self._ph,
                inp_path.encode("ascii"),
                rpt_path.encode("ascii"),
                b"",
            ),
            "EN_open",
        )
        # 判定单位制：EN_getflowunits，0..4=US（水头 ft、压力 psi），5..9=SI（水头 m、压力 m）
        fu = ctypes.c_int()
        self._check(self.lib.EN_getflowunits(self._ph, ctypes.byref(fu)), "EN_getflowunits")
        self.flow_units = fu.value
        # 比重：EPANET 压力换算系数 Ucf[PRESSURE] 含 SpGrav 因子（input1.c:451-452），
        # 还原成内部 ft 水头（= H − El）必须一并除掉，否则 city_d（SpGrav=0.998）会残留 0.2% 偏差。
        sg = ctypes.c_double()
        self._check(self.lib.EN_getoption(self._ph, EN_SP_GRAVITY, ctypes.byref(sg)), "EN_getoption(SP_GRAVITY)")
        self.sp_grav = sg.value
        # EN_FlowUnits 枚举序 = FLOW_UCF 键序（epanet2_enums.h:264-273）：
        # 0..4 US（CFS GPM MGD IMGD AFD），5..9 SI（LPS LPM MLD CMH CMD）
        _FU_NAMES = ("CFS", "GPM", "MGD", "IMGD", "AFD",
                     "LPS", "LPM", "MLD", "CMH", "CMD")
        if not (0 <= self.flow_units <= 9):
            raise RuntimeError(f"未支持的流量单位枚举值: {self.flow_units}")
        fu_name = _FU_NAMES[self.flow_units]
        is_si = self.flow_units >= EN_LPS         # input1.c:255-265
        self._ucf_flow = FLOW_UCF[fu_name]        # initunits input1.c:443-448/:469-474
        self._ucf_head = MperFT if is_si else 1.0  # input1.c:450/:475
        if is_si:
            # Pressflag：SI 默认 METERS（input1.c:268-269），INP 可设 KPA
            # （input3.c:1780-1782） - 扫 [OPTIONS] 原文判定。
            press = "METERS"
            try:
                with open(inp_path, "r", encoding="latin-1") as f:
                    sec = None
                    for raw in f:
                        s0 = raw.strip()
                        if not s0:
                            continue
                        if s0.startswith("["):
                            sec = s0[1:s0.find("]")].strip().upper()
                            continue
                        if sec == "OPTIONS":
                            tk = s0.split(";", 1)[0].split()
                            if len(tk) >= 2 and tk[0].upper().startswith("PRESSURE") \
                                    and tk[1].upper() in ("PSI", "KPA", "METERS"):
                                press = tk[1].upper()
            except OSError:
                pass
            self._ucf_press = (KPAperPSI * PSIperFT * self.sp_grav if press == "KPA"
                               else MperFT * self.sp_grav)   # input1.c:451-452
        else:
            self._ucf_press = PSIperFT * self.sp_grav        # input1.c:476

    # ------------------------------------------------------------------ 基础设施
    def _check(self, code: int, where: str) -> None:
        """返回码检查：>100 抛异常，1..100 记警告。"""
        if code > 100:
            raise RuntimeError(f"{where} 失败（错误码 {code}）: {_errmsg(self.lib, code)}")
        if code > 0:
            self.warnings.append((where, code, _errmsg(self.lib, code)))

    def close(self) -> None:
        if self._ph:
            self.lib.EN_close(self._ph)
            self.lib.EN_deleteproject(self._ph)
            self._ph = ctypes.c_void_p()
        if self._tmp_rpt is not None:
            try:
                os.remove(self._tmp_rpt)
            except OSError:
                pass
            self._tmp_rpt = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ------------------------------------------------------------------ 查询
    def version(self) -> int:
        v = ctypes.c_int()
        self._check(self.lib.EN_getversion(ctypes.byref(v)), "EN_getversion")
        return v.value

    def counts(self) -> dict:
        keys = {
            "nodes": EN_NODECOUNT,
            "tanks": EN_TANKCOUNT,       # 水池+水库
            "links": EN_LINKCOUNT,
            "patterns": EN_PATCOUNT,
            "curves": EN_CURVECOUNT,
            "controls": EN_CONTROLCOUNT,
            "rules": EN_RULECOUNT,
        }
        out = {}
        c = ctypes.c_int()
        for name, obj in keys.items():
            self._check(self.lib.EN_getcount(self._ph, obj, ctypes.byref(c)), f"EN_getcount({name})")
            out[name] = c.value
        return out

    def node_ids(self) -> list:
        n = self.counts()["nodes"]
        buf = ctypes.create_string_buffer(EN_MAXID + 1)
        ids = []
        for i in range(1, n + 1):  # EPANET 索引 1 起始
            self._check(self.lib.EN_getnodeid(self._ph, i, buf), f"EN_getnodeid({i})")
            ids.append(buf.value.decode("ascii"))
        return ids

    def link_ids(self) -> list:
        n = self.counts()["links"]
        buf = ctypes.create_string_buffer(EN_MAXID + 1)
        ids = []
        for i in range(1, n + 1):
            self._check(self.lib.EN_getlinkid(self._ph, i, buf), f"EN_getlinkid({i})")
            ids.append(buf.value.decode("ascii"))
        return ids

    def node_types(self) -> np.ndarray:
        """int8[N]：0=junction 1=reservoir 2=tank（枚举值同 epanet2_enums.h:173-175）。"""
        n = self.counts()["nodes"]
        t = ctypes.c_int()
        out = np.empty(n, dtype=np.int8)
        for i in range(1, n + 1):
            self._check(self.lib.EN_getnodetype(self._ph, i, ctypes.byref(t)), f"EN_getnodetype({i})")
            out[i - 1] = t.value
        return out

    def link_types(self) -> np.ndarray:
        """int8[L]：EN_LinkType 枚举值（epanet2_enums.h:183-191）。"""
        n = self.counts()["links"]
        t = ctypes.c_int()
        out = np.empty(n, dtype=np.int8)
        for i in range(1, n + 1):
            self._check(self.lib.EN_getlinktype(self._ph, i, ctypes.byref(t)), f"EN_getlinktype({i})")
            out[i - 1] = t.value
        return out

    def link_nodes(self) -> np.ndarray:
        """int32[L,2]：每条管段的起止节点索引（0 起始）。"""
        n = self.counts()["links"]
        n1 = ctypes.c_int()
        n2 = ctypes.c_int()
        out = np.empty((n, 2), dtype=np.int32)
        for i in range(1, n + 1):
            self._check(
                self.lib.EN_getlinknodes(self._ph, i, ctypes.byref(n1), ctypes.byref(n2)),
                f"EN_getlinknodes({i})",
            )
            out[i - 1, 0] = n1.value - 1
            out[i - 1, 1] = n2.value - 1
        return out

    # ------------------------------------------------------------------ setter
    # 单位换算与 getter 互逆：getter 是 用户值 ÷ Ucf → 内部值，setter 反向
    # 内部值 × Ucf → 用户值再传给 EN_set*（EN_set* 一律吃用户单位，epanet2_2.h:854）。
    # 无量纲/用户原值属性（EMITTER 系数、ROUGHNESS、SETTING、STATUS 等）不换算。
    def _node_ucf(self, prop: int) -> float:
        if prop in (EN_BASEDEMAND, EN_DEMAND):
            return self._ucf_flow
        if prop in (EN_ELEVATION, EN_HEAD):
            return self._ucf_head
        if prop == EN_PRESSURE:
            return self._ucf_press
        return 1.0        # EN_EMITTER 等：用户原值直传（内部 Ke 换算见 input1.c:567-573）

    def _link_ucf(self, prop: int) -> float:
        if prop == EN_DIAMETER:
            # Ucf[DIAM]：SI mm = ft×1000·MperFT（input1.c:443）；US in = ft×12（:469）
            return 1000.0 * MperFT if self.flow_units >= EN_LPS else 12.0
        if prop == EN_LENGTH:
            return self._ucf_head
        if prop == EN_FLOW:
            return self._ucf_flow
        return 1.0        # ROUGHNESS/MINORLOSS/INITSETTING/STATUS：用户原值

    def set_node_value(self, index: int, prop: int, value_internal: float) -> None:
        """EN_setnodevalue（index 1 基）。value_internal 为内部单位（ft/cfs），
        无量纲属性（如 EN_EMITTER 的用户系数 C）直传。"""
        self._check(
            self.lib.EN_setnodevalue(self._ph, index, prop,
                                     float(value_internal) * self._node_ucf(prop)),
            f"EN_setnodevalue({prop},{index})")

    def set_link_value(self, index: int, prop: int, value_internal: float) -> None:
        """EN_setlinkvalue（index 1 基）。内部单位入参，无量纲属性直传。"""
        self._check(
            self.lib.EN_setlinkvalue(self._ph, index, prop,
                                     float(value_internal) * self._link_ucf(prop)),
            f"EN_setlinkvalue({prop},{index})")

    def set_option(self, option: int, value: float) -> None:
        """EN_setoption。选项值均为无量纲/用户原值（如 EN_ACCURACY），不换算。
        注意 API 路径不做 input3.c:2014-2019 的 [1e-5,1e-1] 钳位，可设更小精度。"""
        self._check(self.lib.EN_setoption(self._ph, option, float(value)),
                    "EN_setoption")

    def solve_single(self) -> dict:
        """EN_solveH 单次完整水力求解 + 读回当前值（内部单位）。
        用于 setter 后的快照对拍（外部 FD 场景）。"""
        self._check(self.lib.EN_solveH(self._ph), "EN_solveH")
        cnt = self.counts()
        val = ctypes.c_double()
        n_nodes, n_links = cnt["nodes"], cnt["links"]
        h = np.empty(n_nodes)
        dem = np.empty(n_nodes)
        q = np.empty(n_links)
        for i in range(1, n_nodes + 1):
            self._check(self.lib.EN_getnodevalue(self._ph, i, EN_HEAD, ctypes.byref(val)),
                        f"EN_getnodevalue(HEAD,{i})")
            h[i - 1] = val.value / self._ucf_head
            self._check(self.lib.EN_getnodevalue(self._ph, i, EN_DEMAND, ctypes.byref(val)),
                        f"EN_getnodevalue(DEMAND,{i})")
            dem[i - 1] = val.value / self._ucf_flow
        for i in range(1, n_links + 1):
            self._check(self.lib.EN_getlinkvalue(self._ph, i, EN_FLOW, ctypes.byref(val)),
                        f"EN_getlinkvalue(FLOW,{i})")
            q[i - 1] = val.value / self._ucf_flow
        stat = ctypes.c_double()
        self._check(self.lib.EN_getstatistic(self._ph, EN_ITERATIONS, ctypes.byref(stat)),
                    "EN_getstatistic(ITERATIONS)")
        iters = int(stat.value)
        self._check(self.lib.EN_getstatistic(self._ph, EN_RELATIVEERROR, ctypes.byref(stat)),
                    "EN_getstatistic(RELATIVEERROR)")
        return {"head_ft": h, "demand_out_cfs": dem, "flow_cfs": q,
                "iterations": iters, "relerr": float(stat.value)}

    # ------------------------------------------------------------------ EPS 求解
    def solve_eps(self) -> dict:
        """完整 EPS：EN_openH → EN_initH(EN_NOSAVE) → {EN_runH → 读值 → EN_nextH} 直到 tstep==0。

        返回 dict（契约表）：t_sec/head_ft/pressure_ft/demand_out_cfs/flow_cfs/
        status/setting/iterations/relerr/warnings，全部内部单位 ft/cfs。
        """
        cnt = self.counts()
        n_nodes, n_links = cnt["nodes"], cnt["links"]

        t_list = []
        head, press, dem = [], [], []
        flow, status, setting = [], [], []
        iters, relerr = [], []
        run_warnings = []

        t = ctypes.c_long()
        tstep = ctypes.c_long()
        val = ctypes.c_double()
        stat = ctypes.c_double()

        self._check(self.lib.EN_openH(self._ph), "EN_openH")
        try:
            self._check(self.lib.EN_initH(self._ph, EN_NOSAVE), "EN_initH")
            while True:
                rc = self.lib.EN_runH(self._ph, ctypes.byref(t))
                if rc > 100:
                    raise RuntimeError(
                        f"EN_runH 失败（t={t.value}s，错误码 {rc}）: {_errmsg(self.lib, rc)}"
                    )
                if rc > 0:  # 1..100 = 运行警告（如水力不平衡、负压）
                    run_warnings.append([int(t.value), int(rc), _errmsg(self.lib, rc)])

                # ---- 逐节点读值（用户单位 → 内部单位：除以 Ucf）----
                h_row = np.empty(n_nodes, dtype=np.float64)
                p_row = np.empty(n_nodes, dtype=np.float64)
                d_row = np.empty(n_nodes, dtype=np.float64)
                for i in range(1, n_nodes + 1):
                    self._check(self.lib.EN_getnodevalue(self._ph, i, EN_HEAD, ctypes.byref(val)),
                                f"EN_getnodevalue(HEAD,{i})")
                    h_row[i - 1] = val.value / self._ucf_head
                    self._check(self.lib.EN_getnodevalue(self._ph, i, EN_PRESSURE, ctypes.byref(val)),
                                f"EN_getnodevalue(PRESSURE,{i})")
                    p_row[i - 1] = val.value / self._ucf_press
                    self._check(self.lib.EN_getnodevalue(self._ph, i, EN_DEMAND, ctypes.byref(val)),
                                f"EN_getnodevalue(DEMAND,{i})")
                    d_row[i - 1] = val.value / self._ucf_flow

                # ---- 逐管段读值 ----
                q_row = np.empty(n_links, dtype=np.float64)
                s_row = np.empty(n_links, dtype=np.int8)
                st_row = np.empty(n_links, dtype=np.float64)
                for i in range(1, n_links + 1):
                    self._check(self.lib.EN_getlinkvalue(self._ph, i, EN_FLOW, ctypes.byref(val)),
                                f"EN_getlinkvalue(FLOW,{i})")
                    q_row[i - 1] = val.value / self._ucf_flow  # 注意：关闭管被 API 置 0
                    self._check(self.lib.EN_getlinkvalue(self._ph, i, EN_STATUS, ctypes.byref(val)),
                                f"EN_getlinkvalue(STATUS,{i})")
                    s_row[i - 1] = int(val.value)
                    self._check(self.lib.EN_getlinkvalue(self._ph, i, EN_SETTING, ctypes.byref(val)),
                                f"EN_getlinkvalue(SETTING,{i})")
                    st_row[i - 1] = val.value  # 原值不换算（契约）

                # ---- 本步求解统计 ----
                self._check(self.lib.EN_getstatistic(self._ph, EN_ITERATIONS, ctypes.byref(stat)),
                            "EN_getstatistic(ITERATIONS)")
                iters.append(int(stat.value))
                self._check(self.lib.EN_getstatistic(self._ph, EN_RELATIVEERROR, ctypes.byref(stat)),
                            "EN_getstatistic(RELATIVEERROR)")
                relerr.append(float(stat.value))

                t_list.append(int(t.value))
                head.append(h_row)
                press.append(p_row)
                dem.append(d_row)
                flow.append(q_row)
                status.append(s_row)
                setting.append(st_row)

                self._check(self.lib.EN_nextH(self._ph, ctypes.byref(tstep)), "EN_nextH")
                if tstep.value == 0:
                    break
        finally:
            self.lib.EN_closeH(self._ph)

        return {
            "t_sec": np.asarray(t_list, dtype=np.int64),
            "head_ft": np.vstack(head).astype(np.float64),
            "pressure_ft": np.vstack(press).astype(np.float64),
            "demand_out_cfs": np.vstack(dem).astype(np.float64),
            "flow_cfs": np.vstack(flow).astype(np.float64),
            "status": np.vstack(status).astype(np.int8),
            "setting": np.vstack(setting).astype(np.float64),
            "iterations": np.asarray(iters, dtype=np.int32),
            "relerr": np.asarray(relerr, dtype=np.float64),
            "warnings": run_warnings,  # [[t_sec, code, msg], ...]（存 meta）
        }


if __name__ == "__main__":
    # 冒烟测试：EXA6（GPM/US 制）
    inp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "networks", "InpData", "EXA6.inp")
    with Epanet(inp) as en:
        print("EN_getversion =", en.version())
        print("counts =", en.counts())
        print("flow_units 枚举 =", en.flow_units, "(1=GPM)")
        ids_n = en.node_ids()
        ids_l = en.link_ids()
        print("首 3 个节点 ID:", ids_n[:3], " 首 3 个管段 ID:", ids_l[:3])
        res = en.solve_eps()
    T = len(res["t_sec"])
    print(f"帧数 T={T}, t_sec[0..2]={res['t_sec'][:3]}, t_sec[-1]={res['t_sec'][-1]}")
    print("head_ft min/max = %.3f / %.3f" % (res["head_ft"].min(), res["head_ft"].max()))
    print("max|Q| cfs = %.4f" % np.abs(res["flow_cfs"]).max())
    print("iterations =", res["iterations"].tolist())
    print("relerr max = %.2e" % res["relerr"].max())
    print("warnings =", res["warnings"])

    # ---- setter 冒烟：EN_setnodevalue(EN_EMITTER) 改一个 junction 后重解 ----
    with Epanet(inp) as en:
        nt = en.node_types()
        j = int(np.where(nt == EN_JUNCTION)[0][0]) + 1     # 1 基
        base = en.solve_single()
        en.set_node_value(j, EN_EMITTER, 5.0)              # 用户系数 C=5（原 0）
        pert = en.solve_single()
        dH = np.abs(pert["head_ft"] - base["head_ft"]).max()
        dD = pert["demand_out_cfs"][j - 1] - base["demand_out_cfs"][j - 1]
        print(f"setter 冒烟: junction#{j} 加 emitter C=5 后 max|dH|={dH:.4g} ft, "
              f"该点出流增量={dD:.4g} cfs (应>0)")
        assert dH > 1e-6 and dD > 1e-6, "EN_setnodevalue(EN_EMITTER) 未生效"
        # EN_setoption(EN_ACCURACY) 冒烟：收敛更紧 → 迭代数不应减少
        en.set_option(EN_ACCURACY, 1e-8)
        tight = en.solve_single()
        print(f"setter 冒烟: ACCURACY 1e-8 后 iters={tight['iterations']} "
              f"(原 {pert['iterations']}), relerr={tight['relerr']:.2e}")
        assert tight["iterations"] >= pert["iterations"] and tight["relerr"] < 1e-7
        # EN_setlinkvalue(EN_ROUGHNESS) 冒烟：加糙 → 水头变化
        en.set_link_value(1, EN_ROUGHNESS, 60.0)
        rough = en.solve_single()
        print(f"setter 冒烟: link#1 C=60 后 max|dH|={np.abs(rough['head_ft']-tight['head_ft']).max():.4g} ft")
    print("epanet_ref.py setter 冒烟测试通过")

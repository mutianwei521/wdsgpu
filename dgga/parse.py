# -*- coding: utf-8 -*-
"""dgga.parse - 用 wntr 1.5 解析 INP，产出契约规定的 Net 数据类（chain-1）。

要点（与契约/EPANET 源码逐条对齐）：
- 一切落盘数值为 EPANET 内部单位：长度/水头 ft、流量 cfs，dtype 一律 float64。
- wntr 内部是 SI（m、m³/s）。换算回内部单位必须走"SI → 用户原值 → ÷types.h 常数"链：
    * 长度/水头：m ÷ MperFT（wntr 的 m↔ft 因子恰为 0.3048，与 types.h 一致，链路无损）。
    * 流量：先 si ÷ wntr因子 还原用户原值（LPS 因子=1e-3，GPM 因子=6.30901964e-5），
      再 ÷ LPSperCFS 或 ÷ GPMperCFS。LPS 情形等价于契约给的 ×1000/LPSperCFS；
      GPM 情形必须走这条链 - wntr 的加仑因子与 EPANET 的 448.831 体系差 ~1.4e-5
      相对误差，直接 ×1000/LPSperCFS 会通不过 1e-9 cfs 的需水对拍。
- 节点/管段顺序 = INP 文件物理出现顺序（即 EPANET 逐行读入的索引顺序）。
  注意 EXA4 的段序是 [PUMPS]→[VALVES]→[PIPES]，与 wntr 的 pipe→pump→valve
  注册顺序不同，因此顺序一律从 INP 文本抽取，wntr 只按 ID 查属性。
- [DEMANDS] 段会**替换** [JUNCTIONS] 行的需水（EPANET input3.c:761-813），
  wntr 1.5 已复刻该语义（city_d 节点 162→3 类、888→4 类，均与 [DEMANDS] 行一致）。
- 需水类别无 pattern（None 或 ''）时用全局默认 pattern：
  [OPTIONS] 的 PATTERN 选项（wn.options.hydraulic.pattern）；若未设或找不到，
  按 EPANET input1.c 的 DEFPATID 规则退到 ID 为 "1" 的 pattern；再没有则常数 1.0。
- pattern 阶梯取值：F[ floor((t+pat_start)/pat_step) mod len ]，不插值。
"""

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import wntr
from wntr.epanet.util import FlowUnits
from wntr.network.elements import Junction, Reservoir, Tank, Pipe, Pump, Valve

try:  # 包内导入；直接跑 python dgga/parse.py 时退化为同目录导入
    from dgga.units import (UCF, MperFT, LPSperCFS, GPMperCFS, PSIperFT,
                            KPAperPSI, KWperHP, FLOW_UCF, SI_FLOW_UNITS)
except ImportError:  # pragma: no cover
    from units import (UCF, MperFT, LPSperCFS, GPMperCFS, PSIperFT,
                       KPAperPSI, KWperHP, FLOW_UCF, SI_FLOW_UNITS)

# ---- EPANET 常数（逐字抄源码）----
BIG = 1.0e10          # types.h:48   #define BIG 1.E10
TINY = 1.0e-6         # types.h:49   #define TINY 1.E-6
MISSING = -1.0e10     # types.h:50   #define MISSING -1.E10
PI_EPANET = 3.141592654  # types.h:60（wntr 的 epanet22.dll 构建未定义 M_PI，走回退字面量）

# 位级复刻续用既有基建：msvcrt 的 pow/log 与 epanet22.dll（MinGW 系静态 CRT）
# 位级一致（solver.py 同策略）；非 Windows 退回 math.pow/math.log。
try:
    import ctypes as _ct
    _crt = _ct.CDLL("msvcrt.dll")
    _pow_crt = _crt.pow
    _pow_crt.restype = _ct.c_double
    _pow_crt.argtypes = [_ct.c_double, _ct.c_double]
    _log_crt = _crt.log
    _log_crt.restype = _ct.c_double
    _log_crt.argtypes = [_ct.c_double]
except OSError:  # pragma: no cover
    _pow_crt = math.pow
    _log_crt = math.log

# 泵类型 PumpType 枚举（types.h:172-177）
CONST_HP = 0
POWER_FUNC = 1
CUSTOM = 2
NOCURVE = 3

# 控制类型 ControlType 枚举（types.h:186-191）
LOWLEVEL, HILEVEL, TIMER, TIMEOFDAY = 0, 1, 2, 3
# 内部状态 StatusType 枚举（types.h:193-205）
ST_XHEAD, ST_TEMPCLOSED, ST_CLOSED, ST_OPEN, ST_ACTIVE = 0, 1, 2, 3, 4

# EN_LinkType 枚举值，逐字抄自 ref/epanet2.2_toolkit/epanet2_enums.h:183-191
EN_CVPIPE = 0   # epanet2_enums.h:183 Pipe with check valve
EN_PIPE = 1     # epanet2_enums.h:184 Pipe
EN_PUMP = 2     # epanet2_enums.h:185 Pump
_VALVE_TYPE_ENUM = {
    "PRV": 3,   # epanet2_enums.h:186
    "PSV": 4,   # epanet2_enums.h:187
    "PBV": 5,   # epanet2_enums.h:188
    "FCV": 6,   # epanet2_enums.h:189
    "TCV": 7,   # epanet2_enums.h:190
    "GPV": 8,   # epanet2_enums.h:191
}

# meta.json 中的标量键（Net.meta 持有的部分）
_META_SCALAR_KEYS = (
    "flow_units", "headloss", "trials", "accuracy", "demand_model",
    "emitter_exponent", "qexp", "duration_sec", "hyd_step_sec",
    "pat_step_sec", "pat_start_sec", "demand_multiplier",
)
# B2 增补标量键（老 meta.json 缺键时按 EPANET 源码默认值补齐；见 _META_B2_DEFAULTS）
_META_B2_DEFAULTS = {
    "extra_iter": -1,          # input1.c:115 hyd->ExtraIter = -1（UNBALANCED STOP）
    "checkfreq": 2,            # input1.c:36  #define CHECKFREQ 2
    "maxcheck": 10,            # input1.c:37  #define MAXCHECK 10
    "damp_limit": 0.0,         # input1.c:38  #define DAMPLIMIT 0
    "tstart_sec": 0,           # input1.c:144 time->Tstart = 0（Start ClockTime）
    "report_step_sec": 3600,   # input1.c:149 time->Rstep = 3600
    "report_start_sec": 0,     # input1.c:151 time->Rstart = 0
    # ---- 梯队3 增补（PRV/PSV 内部设定换算 + 规则引擎）----
    "spgrav": 1.0,             # input1.c:121 hyd->SpGrav = SPGRAV(=1.0)
    "press_units": "",         # Pressflag 用户原词（PSI/KPA/METERS；空=按流量单位默认）
    "rule_step_sec": 0,        # input1.c:150 time->Rulestep = 0（0→adjustdata Hstep/10）
    # ---- 任务 D 增补（D-W 支持；老 meta 缺键按源码默认补齐）----
    "viscosity": 1.1e-5,       # types.h:53 VISCOS（内部运动黏度 sq ft/sec）
}

_NODE_SECS = ("JUNCTIONS", "RESERVOIRS", "TANKS")
_LINK_SECS = ("PIPES", "PUMPS", "VALVES")


def _scan_inp(path):
    """扫描 INP 文本：抽取节点/管段 ID 的物理出现顺序 + [CONTROLS]/[RULES] 原始行
    + [EMITTERS] 原始 token 行 + [OPTIONS] 原始 token 行（Ke 换算需要 PRESSURE 单位）。

    EPANET 逐行读文件、边读边给节点/管段编号（input2.c readdata→newline），
    所以索引顺序就是各段在文件中的物理顺序，与段名无关。
    """
    node_ids, link_ids, controls, rules = [], [], [], []
    emitters, options = [], []
    tanks, pumps, curves, times = [], [], [], []
    sec = None
    with open(path, "r", encoding="latin-1") as f:  # ID 均为 ASCII；latin-1 永不解码失败
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("["):
                end = s.find("]")
                sec = (s[1:end] if end > 0 else s[1:]).strip().upper()
                continue
            if sec == "CONTROLS":
                if not s.startswith(";"):
                    controls.append(s)
                continue
            if sec == "RULES":
                rules.append(s)
                continue
            body = s.split(";", 1)[0].strip()  # 去掉行内注释
            if not body:
                continue
            tok = body.split()[0]
            if sec in _NODE_SECS:
                node_ids.append(tok)
            elif sec in _LINK_SECS:
                link_ids.append(tok)
            elif sec == "EMITTERS":
                emitters.append(body.split())   # [node, Ke_user]（input3.c:996-1026）
            elif sec == "OPTIONS":
                options.append(body.split())
            elif sec == "TIMES":
                times.append(body.split())      # timedata（input3.c:1656-1707）
            if sec == "TANKS":
                tanks.append(body.split())      # tankdata（input3.c:123-257）
            elif sec == "PUMPS":
                pumps.append(body.split())      # pumpdata（input3.c:345-464）
            elif sec == "CURVES":
                curves.append(body.split())     # [id, x, y] 逐行
    return (node_ids, link_ids, controls, rules, emitters, options,
            tanks, pumps, curves, times)


def _stair_index(t_sec, pat_start, pat_step, plen):
    """EPANET 的 pattern 阶梯索引：floor((t+Pstart)/Pstep) mod len，不插值。"""
    return int(((int(t_sec) + int(pat_start)) // int(pat_step)) % int(plen))


def _hour(time_tok, units_tok):
    """hour()（input2.c:748-803）：时间 token → 小时数；非法返回 -1。"""
    parts = time_tok.split(":")
    y = [0.0, 0.0, 0.0]
    try:
        for n, s in enumerate(parts[:3]):
            y[n] = float(s)
    except ValueError:
        return -1.0
    n = len(parts)
    u = units_tok.upper()
    if n == 1:                                # 小数时间 + 可选单位（:775-782）
        if not u:
            return y[0]
        if u.startswith("SEC"):
            return y[0] / 3600.0
        if u.startswith("MIN"):
            return y[0] / 60.0
        if u.startswith("HOU"):
            return y[0]
        if u.startswith("DAY"):
            return y[0] * 24.0
    if n > 1:                                 # hh:mm:ss → 小数小时（:785）
        y[0] = y[0] + y[1] / 60.0 + y[2] / 3600.0
    if not u:
        return y[0]
    if u.startswith("AM"):                    # 12am=0 点（:790-795）
        if y[0] >= 13.0:
            return -1.0
        return y[0] - 12.0 if y[0] >= 12.0 else y[0]
    if u.startswith("PM"):                    # 12pm=正午（:796-801）
        if y[0] >= 13.0:
            return -1.0
        return y[0] if y[0] >= 12.0 else y[0] + 12.0
    return -1.0


@dataclass
class Net:
    """契约规定的管网数据类。数组按 INP 文件出现顺序、0 起始索引，float64/int8/int32。"""

    # ---- 节点（meta 存 node_id）----
    node_id: list                 # list[str]
    node_type: np.ndarray         # int8[N] 0=junction 1=reservoir 2=tank
    elev_ft: np.ndarray           # float64[N]（reservoir 存其定水头 base）
    node_ke: np.ndarray           # float64[N] emitter 内部系数（本期全 0 占位）
    res_head_pat: np.ndarray      # int32[N] reservoir 水头模式索引（-1=无）
    # ---- 管段 ----
    link_id: list                 # list[str]
    link_type: np.ndarray         # int8[L] EN_LinkType
    link_n1: np.ndarray           # int32[L]
    link_n2: np.ndarray           # int32[L]
    diam_ft: np.ndarray           # float64[L]（泵 0）
    len_ft: np.ndarray            # float64[L]（泵/阀 0）
    roughness: np.ndarray         # float64[L] H-W 的 C 原值；非管道 0
    km_int: np.ndarray            # float64[L] 0.02517*K/D_ft^4
    r_hw: np.ndarray              # float64[L] 4.727*len_ft/C^1.852/diam_ft^4.871；非管道 0
    init_status: np.ndarray       # int8[L] 0=Closed 1=Open 2=Active
    valve_setting_user: np.ndarray  # float64[L] 阀设定用户原值；泵=转速比
    # ---- 需水装配（多类别展开表）----
    dem_node: np.ndarray          # int32[K] 类别所属节点索引
    dem_base_cfs: np.ndarray      # float64[K] 类别基准需水（cfs，未乘 Dmult）
    dem_pat: np.ndarray           # int32[K] 类别 pattern 索引（-1=常数 1.0）
    # ---- pattern 与元数据 ----
    pattern_ids: list             # list[str]，dem_pat/res_head_pat 的索引基准
    patterns: list                # list[np.ndarray]，与 pattern_ids 对齐
    meta: dict                    # _META_SCALAR_KEYS 各标量
    controls_raw: list = field(default_factory=list)
    rules_raw: list = field(default_factory=list)
    # ---- B2 增补：水池（仅真实储水池，柱形；顺序 = tank 节点文件序）----
    tank_node: np.ndarray = None      # int32[Nt] 节点索引
    tank_h0: np.ndarray = None        # float64[Nt] 初始水头（绝对 ft，El+lvl/Ucf）
    tank_hmin: np.ndarray = None      # float64[Nt] 最低水头（绝对 ft）
    tank_hmax: np.ndarray = None      # float64[Nt] 最高水头（绝对 ft）
    tank_area: np.ndarray = None      # float64[Nt] 截面积 ft²（PI*SQR(D/dcf?)/4，input1.c:584）
    tank_vmin: np.ndarray = None      # float64[Nt] 最小体积 ft³
    tank_v0: np.ndarray = None        # float64[Nt] 初始体积 ft³
    tank_vmax: np.ndarray = None      # float64[Nt] 最大体积 ft³
    tank_overflow: np.ndarray = None  # int8[Nt] CanOverflow（input3.c:207-212）
    # ---- B2 增补：泵（顺序 = 泵管段文件序；系数为内部单位）----
    pump_link: np.ndarray = None      # int32[Np] 管段索引
    pump_ptype: np.ndarray = None     # int8[Np] PumpType（0=CONST_HP 1=POWER_FUNC 3=NOCURVE）
    pump_h0: np.ndarray = None        # float64[Np] -截止扬程（内部 ft；POWER_FUNC 为负）
    pump_r: np.ndarray = None         # float64[Np] 阻力系数（内部；CONST_HP 为 -8.814P）
    pump_n: np.ndarray = None         # float64[Np] 流量指数（CONST_HP 恒 -1）
    pump_q0: np.ndarray = None        # float64[Np] 设计流量 cfs（CONST_HP=1.0）
    pump_qmax: np.ndarray = None      # float64[Np] 最大流量 cfs（CONST_HP=BIG）
    pump_hmax: np.ndarray = None      # float64[Np] 最大扬程 ft（CONST_HP=BIG）
    pump_upat: np.ndarray = None      # int32[Np] 转速 pattern 索引（-1=无）
    # ---- CUSTOM 多段泵曲线点（用户原值 untransformed，hydcoeffs.c:810；
    #      ragged 扁平化：泵 j 的点区间 = [ptr[j], ptr[j+1])；非 CUSTOM 泵为空）----
    pump_curve_q: np.ndarray = None   # float64[ΣnptsC] 曲线流量（用户单位）
    pump_curve_h: np.ndarray = None   # float64[ΣnptsC] 曲线扬程（用户单位）
    pump_curve_ptr: np.ndarray = None  # int32[Np+1] 偏移指针（无泵时 [0]）
    # ---- B2 增补：[CONTROLS] 结构化（顺序 = 文件序）----
    ctl_link: np.ndarray = None       # int32[C] 受控管段索引
    ctl_node: np.ndarray = None       # int32[C] 监测节点索引（-1=时间控制）
    ctl_type: np.ndarray = None       # int8[C] 0=LOWLEVEL 1=HILEVEL 2=TIMER 3=TIMEOFDAY
    ctl_status: np.ndarray = None     # int8[C] 目标内部状态（2=CLOSED 3=OPEN 4=ACTIVE）
    ctl_setting: np.ndarray = None    # float64[C] 目标设定（内部单位；MISSING=-1e10）
    ctl_grade: np.ndarray = None      # float64[C] 阈值绝对水头 ft（input1.c:684-687）
    ctl_time: np.ndarray = None       # int64[C] 触发时刻 s（TIMER/TIMEOFDAY）

    def __post_init__(self):
        """B2 新键的向后兼容默认：老 npz 缺键 → 空数组。"""
        empt = {
            "tank_node": np.int32, "tank_h0": np.float64, "tank_hmin": np.float64,
            "tank_hmax": np.float64, "tank_area": np.float64, "tank_vmin": np.float64,
            "tank_v0": np.float64, "tank_vmax": np.float64, "tank_overflow": np.int8,
            "pump_link": np.int32, "pump_ptype": np.int8, "pump_h0": np.float64,
            "pump_r": np.float64, "pump_n": np.float64, "pump_q0": np.float64,
            "pump_qmax": np.float64, "pump_hmax": np.float64, "pump_upat": np.int32,
            "pump_curve_q": np.float64, "pump_curve_h": np.float64,
            "pump_curve_ptr": np.int32,
            "ctl_link": np.int32, "ctl_node": np.int32, "ctl_type": np.int8,
            "ctl_status": np.int8, "ctl_setting": np.float64, "ctl_grade": np.float64,
            "ctl_time": np.int64,
        }
        for k, dt in empt.items():
            if getattr(self, k) is None:
                setattr(self, k, np.zeros(0, dtype=dt))
        for k, dv in _META_B2_DEFAULTS.items():
            self.meta.setdefault(k, dv)

    # ------------------------------------------------------------------
    @property
    def N(self):
        return len(self.node_id)

    @property
    def L(self):
        return len(self.link_id)

    # ------------------------------------------------------------------
    def _pattern_factors_at(self, t_sec):
        """返回 float64[npat+1]：各 pattern 在 t 时刻的阶梯因子，末位=常数 1.0。"""
        npat = len(self.patterns)
        fac = np.empty(npat + 1, dtype=np.float64)
        ps, pt = self.meta["pat_start_sec"], self.meta["pat_step_sec"]
        for i, F in enumerate(self.patterns):
            fac[i] = F[_stair_index(t_sec, ps, pt, len(F))]
        fac[npat] = 1.0
        return fac

    def demand_cfs_at(self, t_sec):
        """t 时刻全网名义需水 float64[N]（cfs）：多类别求和 × 全局 Demand Multiplier；
        tank/reservoir 无类别自然为 0。"""
        out = np.zeros(self.N, dtype=np.float64)
        if self.dem_node.size == 0:
            return out
        fac = self._pattern_factors_at(t_sec)
        npat = len(self.patterns)
        pidx = np.where(self.dem_pat < 0, npat, self.dem_pat)
        w = self.dem_base_cfs * fac[pidx] * float(self.meta["demand_multiplier"])
        out = np.bincount(self.dem_node, weights=w, minlength=self.N).astype(np.float64)
        return out

    def reservoir_head_ft_at(self, t_sec):
        """t 时刻各 reservoir 水头 float64[N]（ft）；非 reservoir 置 nan。
        有模式则 base × F[t]（阶梯），无模式即 base。"""
        out = np.full(self.N, np.nan, dtype=np.float64)
        fac = self._pattern_factors_at(t_sec)
        for i in np.where(self.node_type == 1)[0]:
            p = int(self.res_head_pat[i])
            out[i] = self.elev_ft[i] * (fac[p] if p >= 0 else 1.0)
        return out

    # ------------------------------------------------------------------
    def save(self, dir_path, stem):
        """写 <stem>_net.npz（数值数组）+ <stem>_meta.json（ID/标量/pattern/控制行）。"""
        os.makedirs(dir_path, exist_ok=True)
        np.savez_compressed(
            os.path.join(dir_path, f"{stem}_net.npz"),
            node_type=self.node_type, elev_ft=self.elev_ft, node_ke=self.node_ke,
            res_head_pat=self.res_head_pat,
            link_type=self.link_type, link_n1=self.link_n1, link_n2=self.link_n2,
            diam_ft=self.diam_ft, len_ft=self.len_ft, roughness=self.roughness,
            km_int=self.km_int, r_hw=self.r_hw, init_status=self.init_status,
            valve_setting_user=self.valve_setting_user,
            dem_node=self.dem_node, dem_base_cfs=self.dem_base_cfs, dem_pat=self.dem_pat,
            # ---- B2 增补键（load 对缺键向后兼容）----
            tank_node=self.tank_node, tank_h0=self.tank_h0, tank_hmin=self.tank_hmin,
            tank_hmax=self.tank_hmax, tank_area=self.tank_area,
            tank_vmin=self.tank_vmin, tank_v0=self.tank_v0, tank_vmax=self.tank_vmax,
            tank_overflow=self.tank_overflow,
            pump_link=self.pump_link, pump_ptype=self.pump_ptype,
            pump_h0=self.pump_h0, pump_r=self.pump_r, pump_n=self.pump_n,
            pump_q0=self.pump_q0, pump_qmax=self.pump_qmax, pump_hmax=self.pump_hmax,
            pump_upat=self.pump_upat,
            pump_curve_q=self.pump_curve_q, pump_curve_h=self.pump_curve_h,
            pump_curve_ptr=self.pump_curve_ptr,
            ctl_link=self.ctl_link, ctl_node=self.ctl_node, ctl_type=self.ctl_type,
            ctl_status=self.ctl_status, ctl_setting=self.ctl_setting,
            ctl_grade=self.ctl_grade, ctl_time=self.ctl_time,
        )
        meta_out = {k: self.meta[k] for k in _META_SCALAR_KEYS}
        for k in _META_B2_DEFAULTS:
            meta_out[k] = self.meta.get(k, _META_B2_DEFAULTS[k])
        meta_out["node_id"] = list(self.node_id)
        meta_out["link_id"] = list(self.link_id)
        meta_out["pattern_ids"] = list(self.pattern_ids)
        meta_out["patterns"] = {pid: [float(x) for x in F]
                                for pid, F in zip(self.pattern_ids, self.patterns)}
        meta_out["controls_raw"] = list(self.controls_raw)
        meta_out["rules_raw"] = list(self.rules_raw)
        with open(os.path.join(dir_path, f"{stem}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False, indent=1)

    @staticmethod
    def load(dir_path, stem):
        """save 的逆操作。"""
        with np.load(os.path.join(dir_path, f"{stem}_net.npz")) as z:
            arrays = {k: z[k] for k in z.files}
        with open(os.path.join(dir_path, f"{stem}_meta.json"), "r", encoding="utf-8") as f:
            m = json.load(f)
        pattern_ids = list(m["pattern_ids"])
        patterns = [np.asarray(m["patterns"][pid], dtype=np.float64) for pid in pattern_ids]
        meta = {k: m[k] for k in _META_SCALAR_KEYS}
        for k, dv in _META_B2_DEFAULTS.items():
            meta[k] = m.get(k, dv)
        b2 = {k: arrays.get(k) for k in (
            "tank_node", "tank_h0", "tank_hmin", "tank_hmax", "tank_area",
            "tank_vmin", "tank_v0", "tank_vmax", "tank_overflow",
            "pump_link", "pump_ptype", "pump_h0", "pump_r", "pump_n",
            "pump_q0", "pump_qmax", "pump_hmax", "pump_upat",
            "pump_curve_q", "pump_curve_h", "pump_curve_ptr",
            "ctl_link", "ctl_node", "ctl_type", "ctl_status", "ctl_setting",
            "ctl_grade", "ctl_time")}
        return Net(
            node_id=list(m["node_id"]),
            node_type=arrays["node_type"], elev_ft=arrays["elev_ft"],
            node_ke=arrays["node_ke"], res_head_pat=arrays["res_head_pat"],
            link_id=list(m["link_id"]),
            link_type=arrays["link_type"], link_n1=arrays["link_n1"],
            link_n2=arrays["link_n2"], diam_ft=arrays["diam_ft"],
            len_ft=arrays["len_ft"], roughness=arrays["roughness"],
            km_int=arrays["km_int"], r_hw=arrays["r_hw"],
            init_status=arrays["init_status"],
            valve_setting_user=arrays["valve_setting_user"],
            dem_node=arrays["dem_node"], dem_base_cfs=arrays["dem_base_cfs"],
            dem_pat=arrays["dem_pat"],
            pattern_ids=pattern_ids, patterns=patterns,
            meta=meta,
            controls_raw=list(m.get("controls_raw", [])),
            rules_raw=list(m.get("rules_raw", [])),
            **b2,
        )


# ----------------------------------------------------------------------
def _status_to_int(st):
    """wntr LinkStatus → 契约 init_status：Closed=0 Open=1 Active=2；CV(3) 视为 Open。"""
    v = int(st)
    return 1 if v == 3 else v


def parse_inp(path):
    """解析 INP → Net。顺序以 INP 文本为准，属性以 wntr 为准，单位换算走 types.h 常数。"""
    (node_ids, link_ids, controls_raw, rules_raw, emitters_raw, options_raw,
     tanks_raw, pumps_raw, curves_raw, times_raw) = _scan_inp(path)
    wn = wntr.network.WaterNetworkModel(path)
    opt = wn.options

    # ---- 基本一致性校验：文本抽取的 ID 集合必须与 wntr 完全一致 ----
    if len(node_ids) != wn.num_nodes or set(node_ids) != set(wn.node_name_list):
        raise RuntimeError(f"{path}: INP 文本节点序抽取与 wntr 不一致 "
                           f"({len(node_ids)} vs {wn.num_nodes})")
    if len(link_ids) != wn.num_links or set(link_ids) != set(wn.link_name_list):
        raise RuntimeError(f"{path}: INP 文本管段序抽取与 wntr 不一致 "
                           f"({len(link_ids)} vs {wn.num_links})")

    # ---- 单位体系（全 10 种流量单位；initunits input1.c:443-448/:469-474）----
    fu = str(opt.hydraulic.inpfile_units).upper()
    if fu not in FLOW_UCF:
        raise ValueError(f"{path}: 流量单位 {fu} 不在支持范围 {sorted(FLOW_UCF)}")
    is_si = fu in SI_FLOW_UNITS               # input1.c:255-265 Unitsflag 判定
    headloss = str(opt.hydraulic.headloss).upper()
    if headloss not in ("H-W", "D-W", "C-M"):
        raise ValueError(f"{path}: 水损公式 {headloss} 不在 H-W/D-W/C-M 范围")
    # Hexp：HW=1.852，否则 2.0（input1.c:294-295）
    hexp_v = 1.852 if headloss == "H-W" else 2.0
    fu_factor = FlowUnits[fu].factor          # wntr 的 用户流量单位→m³/s 因子
    flow_per_cfs = FLOW_UCF[fu]               # types.h 常数（qcf）

    def si_flow_to_cfs(x_m3s):
        """m³/s → cfs：先还原用户原值（÷wntr 因子），再 ÷types.h 常数。
        LPS 情形恒等于契约公式 ×1000/LPSperCFS。"""
        return float(x_m3s) / fu_factor / flow_per_cfs

    def m_to_ft(x_m):
        return float(x_m) / MperFT

    # ---- emitter 换算所需标量 ----
    gamma = float(opt.hydraulic.emitter_exponent)   # 用户 γ（[OPTIONS] EMITTER EXPONENT）
    qexp_v = 1.0 / gamma                            # input3.c:2029  hyd->Qexp = 1.0/y
    spgrav = float(opt.hydraulic.specific_gravity)  # input1.c:121 默认 1.0
    # Ucf[PRESSURE]：Pressflag 默认 PSI（input1.c:99）；非 SI 强制 PSI、SI 且 PSI→METERS
    # （input1.c:268-269）。SI: METERS→MperFT*SpGrav，KPA→KPAperPSI*PSIperFT*SpGrav
    # （input1.c:451-452）；US: PSIperFT*SpGrav（input1.c:476）。
    press = "METERS" if is_si else "PSI"
    for tk in options_raw:      # [OPTIONS] PRESSURE PSI/KPA/METERS（input3.c:1780-1782）
        if len(tk) >= 2 and tk[0].upper().startswith("PRESSURE") \
                and tk[1].upper() in ("PSI", "KPA", "METERS"):
            press = tk[1].upper()
    if is_si:
        pcf = KPAperPSI * PSIperFT * spgrav if press == "KPA" else MperFT * spgrav
    else:
        pcf = PSIperFT * spgrav

    # ---- VISCOSITY（input3.c:2011 → input1.c:271-282）：内部 sq ft/sec ----
    # ucf = 1.0；SI 时 SQR(MperFT)（input1.c:272-273）。未设 → VISCOS=1.1e-5
    # （types.h:53）；>1e-3 → 倍数×VISCOS（:278-281）；否则实际值/ucf（:282）。
    VISCOS = 1.1e-5                            # types.h:53
    visc_raw = None
    for tk in options_raw:
        if tk[0].upper().startswith("VISC") and len(tk) >= 2:  # w_VISCOSITY "VISC"
            visc_raw = float(tk[1])
    if visc_raw is None:
        viscos = VISCOS                        # input1.c:274-277
    elif visc_raw > 1.0e-3:
        viscos = visc_raw * VISCOS             # input1.c:278-281
    else:
        viscos = visc_raw / ((MperFT * MperFT) if is_si else 1.0)  # input1.c:282

    # ---- pattern 表（以 wntr 注册顺序为索引基准）----
    pattern_ids = list(wn.pattern_name_list)
    pat_index = {pid: i for i, pid in enumerate(pattern_ids)}
    patterns = [np.asarray(wn.get_pattern(pid).multipliers, dtype=np.float64)
                for pid in pattern_ids]
    if any(len(F) == 0 for F in patterns):
        raise RuntimeError(f"{path}: 存在空 pattern")

    # 全局默认 pattern：PATTERN 选项 → 否则 EPANET 的 DEFPATID("1") → 否则常数
    dp = opt.hydraulic.pattern
    if dp and str(dp) in pat_index:
        default_pat = str(dp)
    elif "1" in pat_index:
        default_pat = "1"
    else:
        default_pat = None

    # ---- 节点数组 ----
    N = len(node_ids)
    node_type = np.zeros(N, dtype=np.int8)
    elev_ft = np.zeros(N, dtype=np.float64)
    node_ke = np.zeros(N, dtype=np.float64)      # 本期无 emitter，全 0 占位
    res_head_pat = np.full(N, -1, dtype=np.int32)
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    dem_node, dem_base_cfs, dem_pat = [], [], []
    for i, nid in enumerate(node_ids):
        nd = wn.get_node(nid)
        if isinstance(nd, Junction):
            node_type[i] = 0
            elev_ft[i] = m_to_ft(nd.elevation)
            for ts in nd.demand_timeseries_list:  # 多类别逐条展开
                pname = ts.pattern_name
                if not pname:                     # None 或 '' → 全局默认 pattern
                    pname = default_pat
                if pname is None:
                    pidx = -1                     # 常数 1.0
                elif pname in pat_index:
                    pidx = pat_index[pname]
                else:
                    raise RuntimeError(f"{path}: 节点 {nid} 需水引用未知 pattern {pname!r}")
                dem_node.append(i)
                dem_base_cfs.append(si_flow_to_cfs(ts.base_value))
                dem_pat.append(pidx)
        elif isinstance(nd, Reservoir):
            node_type[i] = 1
            elev_ft[i] = m_to_ft(nd.base_head)    # reservoir 存定水头 base
            hp = nd.head_pattern_name
            if hp:
                if hp not in pat_index:
                    raise RuntimeError(f"{path}: reservoir {nid} 引用未知 pattern {hp!r}")
                res_head_pat[i] = pat_index[hp]
        elif isinstance(nd, Tank):
            node_type[i] = 2
            elev_ft[i] = m_to_ft(nd.elevation)    # tank 底高程
        else:
            raise RuntimeError(f"{path}: 未知节点类型 {type(nd).__name__} ({nid})")

    # ---- [EMITTERS]：用户系数 C → 内部 Ke（水损系数）----
    # 读入语义照抄 input3.c:996-1026：节点必须存在（:1018 否则 203 错），
    # 非 junction 静默忽略（:1019 `if (j > net->Njuncs) return 0;`），
    # C<0 报错（:1023），同节点多行后行覆盖前行（:1024 直接赋值）。
    # 换算照抄 input1.c:567-573：
    #   ucf = pow(Ucf[FLOW], Qexp) / Ucf[PRESSURE]   （input1.c:568）
    #   Ke  = ucf / pow(C, Qexp)，仅 C>0 时换算       （input1.c:572）
    for tk in emitters_raw:
        if len(tk) < 2:
            raise RuntimeError(f"{path}: [EMITTERS] 行缺字段: {tk}")   # input3.c:1017
        nid, c = tk[0], float(tk[1])
        if c < 0.0:
            raise ValueError(f"{path}: emitter 系数为负 ({nid}: {c})")  # input3.c:1023
        if nid not in node_index:
            raise RuntimeError(f"{path}: [EMITTERS] 引用未知节点 {nid}")  # input3.c:1018
        i = node_index[nid]
        if node_type[i] != 0:
            continue                                   # 非 junction 忽略（input3.c:1019）
        node_ke[i] = (flow_per_cfs ** qexp_v / pcf) / c ** qexp_v if c > 0.0 else 0.0

    # ---- 管段数组 ----
    L = len(link_ids)
    link_type = np.zeros(L, dtype=np.int8)
    link_n1 = np.zeros(L, dtype=np.int32)
    link_n2 = np.zeros(L, dtype=np.int32)
    diam_ft = np.zeros(L, dtype=np.float64)
    len_ft = np.zeros(L, dtype=np.float64)
    roughness = np.zeros(L, dtype=np.float64)
    km_int = np.zeros(L, dtype=np.float64)
    r_hw = np.zeros(L, dtype=np.float64)
    init_status = np.zeros(L, dtype=np.int8)
    valve_setting_user = np.zeros(L, dtype=np.float64)

    for k, lid in enumerate(link_ids):
        lk = wn.get_link(lid)
        link_n1[k] = node_index[lk.start_node_name]
        link_n2[k] = node_index[lk.end_node_name]
        if isinstance(lk, Pipe):
            # wntr 的 check_valve 管 → EN_CVPIPE(0)；普通管 → EN_PIPE(1)
            link_type[k] = EN_CVPIPE if lk.check_valve else EN_PIPE
            d = m_to_ft(lk.diameter)
            ln = m_to_ft(lk.length)
            c = float(lk.roughness)
            if headloss == "D-W":
                # D-W 粗糙度内部值：Kc_user/(1000*Ucf[ELEV])（input1.c:613）。
                # wntr 已存 m（SI mm×1e-3 / US 毫英尺×0.3048e-3），两种单位制
                # 内部 ft 值均 = wntr_m/MperFT。
                c = m_to_ft(lk.roughness)
            diam_ft[k] = d
            len_ft[k] = ln
            roughness[k] = c        # H-W:C 原值；D-W:内部 ft 的 Kc；C-M:曼宁 n
            # 内部阻力 R（resistcoeff hydcoeffs.c:92-103）：仅管道（含 CV 管）
            if headloss == "H-W":
                r_hw[k] = 4.727 * ln / (c ** 1.852) / (d ** 4.871)   # :95
            elif headloss == "D-W":
                # :98  R = L/2.0/32.2/d/SQR(PI*SQR(d)/4.0)（f 在解算时并入）
                r_hw[k] = ln / 2.0 / 32.2 / d / ((PI_EPANET * (d * d) / 4.0) ** 2)
            else:  # C-M（:101-102）
                r_hw[k] = ((4.0 * c / (1.49 * PI_EPANET * (d * d))) ** 2
                           * (d / 4.0) ** -1.333 * ln)
            km_int[k] = 0.02517 * float(lk.minor_loss) / (d ** 4)
            init_status[k] = _status_to_int(lk.initial_status)  # [STATUS] 关闭已反映
        elif isinstance(lk, Pump):
            link_type[k] = EN_PUMP
            valve_setting_user[k] = float(lk.base_speed)  # 泵：转速比
            init_status[k] = _status_to_int(lk.initial_status)
        elif isinstance(lk, Valve):
            vt = str(lk.valve_type).upper()
            if vt not in _VALVE_TYPE_ENUM:
                raise RuntimeError(f"{path}: 阀 {lid} 未知类型 {vt}")
            link_type[k] = _VALVE_TYPE_ENUM[vt]
            d = m_to_ft(lk.diameter)
            diam_ft[k] = d
            km_int[k] = 0.02517 * float(lk.minor_loss) / (d ** 4)
            # 阀设定还原为用户原值（本期不换算到内部单位）。
            # 【wntr 往返 1 ULP 口径，发布审计欠账 d 项】US 制网（psi 设定）
            # 经 wntr 内部存 m 水柱再由下式还原 psi，与 INP 原文 strtod 值可差
            # ≤1 ULP（×0.4333/0.3048 再除回不是恒等舍入）。实测（scratch
            # ulp_probe，2026-08-24）：BWSN_Network_1(GPM) 8 阀中 3 阀差
            # 1 ULP（~3e-15 psi 相对量级），Richmond_standard/D-Town/ky10 为
            # 0；SI 网（如 L-TOWN，米设定）此路是恒等，0 ULP - 主线网不受
            # 影响。该 1 ULP 进 init_setting（÷ucf_pressure，solver.py:1317）
            # 后远小于 htol=5e-4 ft，不会改变 prvstatus 判定；但要与 EPANET
            # DLL 做**位级**对拍时（correct_ltown §1 / check_prv_release ①），
            # US 制 PRV 网不入位级口径（本仓位级判据只钉在 SI 主线网上）。
            # 修复方案（未做，留作论文口径说明）：像 _apply_exact_props 复刻
            # diam/len/r_hw 那样从 INP 原文 token 直读 setting。
            if vt in ("PRV", "PSV", "PBV"):
                s_m = float(lk.initial_setting)   # wntr 存 m 水柱
                valve_setting_user[k] = s_m if is_si else s_m / MperFT * PSIperFT
            elif vt == "FCV":
                valve_setting_user[k] = float(lk.initial_setting) / fu_factor  # m³/s→用户流量
            elif vt == "TCV":
                valve_setting_user[k] = float(lk.initial_setting)  # 无量纲局损系数
            else:  # GPV 设定是曲线，记 0
                valve_setting_user[k] = 0.0
            # 阀默认 Active(2)，被 [STATUS] 固定的为 0/1（wntr 已反映）
            init_status[k] = _status_to_int(lk.initial_status)
        else:
            raise RuntimeError(f"{path}: 未知管段类型 {type(lk).__name__} ({lid})")

    # ================= B2 增补：水池 / 泵 / 曲线 / 控制 =================
    # 换算因子（input1.c:443-450 SI / :469-477 US；逐字对应）
    if is_si:
        hcf = MperFT                      # Ucf[HEAD]=Ucf[ELEV]（input1.c:450）
        qcf = FLOW_UCF[fu]                # Ucf[FLOW]（input1.c:444-448）
        wcf = KWperHP                     # Ucf[POWER]（input1.c:453）
    else:
        hcf = 1.0                         # input1.c:475
        qcf = FLOW_UCF[fu]                # input1.c:470-474
        wcf = 1.0                         # input1.c:477
    vol_ucf = hcf * hcf * hcf             # Ucf[VOLUME] = hcf*hcf*hcf（input1.c:506）

    # ---- 曲线表（[CURVES] 逐行 id x y，同 id 追加）----
    curve_pts = {}
    for tk in curves_raw:
        if len(tk) < 3:
            raise RuntimeError(f"{path}: [CURVES] 行缺字段: {tk}")
        curve_pts.setdefault(tk[0], []).append((float(tk[1]), float(tk[2])))

    # ---- 水池（tankdata input3.c:123-257 + convertunits input1.c:575-592）----
    t_node, t_h0, t_hmin, t_hmax, t_area = [], [], [], [], []
    t_vmin, t_v0, t_vmax, t_ovf = [], [], [], []
    for tk in tanks_raw:
        if len(tk) <= 3:
            continue                       # n<=3 是 reservoir 行（input3.c:175）
        if tk[0] not in node_index:
            raise RuntimeError(f"{path}: [TANKS] 未知节点 {tk[0]}")
        i = node_index[tk[0]]
        el = float(tk[1])
        initlevel = float(tk[2])           # input3.c:189
        minlevel = float(tk[3])            # input3.c:190
        maxlevel = float(tk[4])            # input3.c:191
        diam = float(tk[5])                # input3.c:192
        minvol = float(tk[6]) if len(tk) >= 7 else 0.0   # input3.c:193
        if len(tk) >= 8 and tk[7] and tk[7] != "*":
            raise NotImplementedError(f"{path}: 水池 {tk[0]} 带体积曲线（本期不支持）")
        overflow = 0
        if len(tk) >= 9 and tk[8].upper().startswith("YES"):
            overflow = 1                   # input3.c:209
        if diam == 0.0:
            continue                       # diam==0 是 reservoir（input3.c:231）
        # 用户单位体积（input3.c:247-251；PI 用 DLL 回退字面量）
        area_u = PI_EPANET * (diam * diam) / 4.0
        vmin_u = area_u * minlevel
        if minvol > 0.0:
            vmin_u = minvol                # input3.c:249
        v0_u = vmin_u + area_u * (initlevel - minlevel)   # input3.c:250
        vmax_u = vmin_u + area_u * (maxlevel - minlevel)  # input3.c:251
        # convertunits：El 先换算（input1.c:549），水位折绝对水头（:581-583），
        # 面积（:584），体积（:585-587）
        el_int = el / hcf
        elev_ft[i] = el_int                # 覆写 wntr 链，保证位级
        t_node.append(i)
        t_h0.append(el_int + initlevel / hcf)    # input1.c:581
        t_hmin.append(el_int + minlevel / hcf)   # input1.c:582
        t_hmax.append(el_int + maxlevel / hcf)   # input1.c:583
        t_area.append(PI_EPANET * ((diam / hcf) * (diam / hcf)) / 4.0)  # :584
        t_v0.append(v0_u / vol_ucf)              # input1.c:585
        t_vmin.append(vmin_u / vol_ucf)          # input1.c:586
        t_vmax.append(vmax_u / vol_ucf)          # input1.c:587
        t_ovf.append(overflow)

    # ---- 泵（pumpdata input3.c:345-464 → getpumpcurve/updatepumpparams
    #      input3.c:2030-2099 / input2.c:372-462 → convertunits input1.c:625-648）----
    p_link, p_ptype, p_h0, p_r, p_n = [], [], [], [], []
    p_q0, p_qmax, p_hmax, p_upat = [], [], [], []
    p_cpts = []                            # 逐泵曲线点（CUSTOM 用户原值；其余空）
    pump_rows = {tk[0]: tk for tk in pumps_raw}
    link_index = {lid: k for k, lid in enumerate(link_ids)}
    for k, lid in enumerate(link_ids):
        if link_type[k] != EN_PUMP:
            continue
        tk = pump_rows.get(lid)
        if tk is None:
            raise RuntimeError(f"{path}: 泵 {lid} 无 [PUMPS] 原始行")
        ptype = NOCURVE                    # input3.c:409 占位
        power = 0.0
        hcurve = None
        upat = -1
        cu_pts = []                        # CUSTOM 曲线点（用户原值，untransformed）
        h0 = h1 = h2 = q1 = q2 = 0.0
        # 判定 1.x 数值格式 vs 2.x 关键字格式（input3.c:417-419）
        is_v1 = False
        if len(tk) >= 4:
            try:
                float(tk[3])
                is_v1 = True
            except ValueError:
                is_v1 = False
        if is_v1:
            X = [float(t) for t in tk[3:]]
            m = len(X)
            if m == 1:                     # 恒功率（input3.c:2059-2064）
                if X[0] <= 0.0:
                    raise ValueError(f"{path}: 泵 {lid} 功率非正")
                ptype = CONST_HP
                power = X[0]
            elif m == 2:                   # 1 点曲线（input3.c:2070-2077）
                q1 = X[1]; h1 = X[0]
                h0 = 1.33334 * h1          # input3.c:2074
                q2 = 2.0 * q1              # input3.c:2075
                h2 = 0.0
                ptype = POWER_FUNC
            elif m >= 5:                   # 3 点曲线（input3.c:2080-2087）
                h0, h1, q1, h2, q2 = X[0], X[1], X[2], X[3], X[4]
                ptype = POWER_FUNC
            else:
                raise RuntimeError(f"{path}: 泵 {lid} 1.x 格式字段数非法")
        else:
            m = 4
            while m < len(tk):             # 关键字对（input3.c:432-462）
                kw = tk[m - 1].upper()
                if kw.startswith("POWER"):
                    y = float(tk[m])
                    if y <= 0.0:
                        raise ValueError(f"{path}: 泵 {lid} POWER 非正")
                    ptype = CONST_HP       # input3.c:439
                    power = y
                elif kw.startswith("HEAD"):
                    hcurve = tk[m]         # input3.c:444-446
                elif kw.startswith("PATTERN"):
                    if tk[m] not in pat_index:
                        raise RuntimeError(f"{path}: 泵 {lid} 引用未知 pattern {tk[m]!r}")
                    upat = pat_index[tk[m]]  # input3.c:450-452
                elif kw.startswith("SPEED"):
                    pass                   # Kc 已由 wntr base_speed 提供（input3.c:456-458）
                else:
                    raise RuntimeError(f"{path}: 泵 {lid} 未知关键字 {tk[m-1]!r}")
                m += 2
        # updatepumpparams（input2.c:372-462）：曲线泵在用户单位拟合
        if ptype == NOCURVE and hcurve is not None:
            pts = curve_pts.get(hcurve)
            if not pts:
                raise RuntimeError(f"{path}: 泵 {lid} 引用未知曲线 {hcurve!r}")
            npts = len(pts)
            if npts == 1:                  # input2.c:412-420
                q1, h1 = pts[0]
                h0 = 1.33334 * h1          # input2.c:417
                q2 = 2.0 * q1              # input2.c:418
                h2 = 0.0
                ptype = POWER_FUNC
            elif npts == 3 and pts[0][0] == 0.0:   # input2.c:423-431
                h0 = pts[0][1]
                q1, h1 = pts[1]
                q2, h2 = pts[2]
                ptype = POWER_FUNC
            else:
                # CUSTOM 多段曲线（input2.c:433-444）：形状参数 + 曲线点原值。
                # 曲线点保持用户单位（untransformed，hydcoeffs.c:810 注释），
                # 求解期 curvecoeff（hydcoeffs.c:794-831）逐段取局部线性模型。
                for m_ in range(1, npts):
                    if pts[m_][1] >= pts[m_ - 1][1]:
                        raise RuntimeError(f"{path}: 泵 {lid} CUSTOM 曲线非降序"
                                           f"（input2.c:439 错误 227）")
                ptype = CUSTOM
                cu_qmax = pts[npts - 1][0]         # input2.c:441
                cu_q0 = (pts[0][0] + cu_qmax) / 2.0   # input2.c:442
                cu_hmax = pts[0][1]                # input2.c:443
                cu_pts = list(pts)                 # curvecoeff 所需原始点
        if ptype == CONST_HP:              # input2.c:392-400
            P_h0, P_n = 0.0, -1.0
            P_r = -8.814 * power           # input2.c:395
            P_hmax, P_qmax, P_q0 = BIG, BIG, 1.0   # input2.c:397-399
            if is_si:
                P_r = P_r / wcf            # input1.c:633（SI: kw→hp）
        elif ptype == CUSTOM:              # input2.c:436-444 + input1.c:644-647
            P_h0, P_r, P_n = 0.0, 0.0, 1.0     # 曲线取段留接口，形状参数不用
            P_q0 = cu_q0 / qcf             # input1.c:645
            P_qmax = cu_qmax / qcf         # input1.c:646
            P_hmax = cu_hmax / hcf         # input1.c:647
        elif ptype == POWER_FUNC:
            # powercurve（input3.c:2101-2129），log/pow 走 CRT
            if (h0 < TINY or h0 - h1 < TINY or h1 - h2 < TINY or
                    q1 < TINY or q2 - q1 < TINY):
                raise RuntimeError(f"{path}: 泵 {lid} 曲线非法（powercurve 前置检查）")
            a = h0                         # input3.c:2121
            h4 = h0 - h1                   # input3.c:2122
            h5 = h0 - h2                   # input3.c:2123
            c = _log_crt(h5 / h4) / _log_crt(q2 / q1)   # input3.c:2124
            if c <= 0.0 or c > 20.0:
                raise RuntimeError(f"{path}: 泵 {lid} 曲线指数非法 c={c}")
            b = -h4 / _pow_crt(q1, c)      # input3.c:2126
            if b >= 0.0:
                raise RuntimeError(f"{path}: 泵 {lid} 曲线系数非法 b={b}")
            P_h0 = -a                      # input3.c:2091 / input2.c:452
            P_r = -b                       # input3.c:2092 / input2.c:453
            P_n = c                        # input3.c:2093 / input2.c:454
            P_q0 = q1                      # input3.c:2094 / input2.c:455
            P_qmax = _pow_crt((-a / b), (1.0 / c))   # input3.c:2095 / input2.c:456
            P_hmax = h0                    # input3.c:2096 / input2.c:457
            # convertunits（input1.c:638-647）
            P_h0 = P_h0 / hcf              # input1.c:640
            P_r = P_r * (_pow_crt(qcf, P_n) / hcf)   # input1.c:641
            P_q0 = P_q0 / qcf              # input1.c:645
            P_qmax = P_qmax / qcf          # input1.c:646
            P_hmax = P_hmax / hcf          # input1.c:647
        else:
            raise RuntimeError(f"{path}: 泵 {lid} 无曲线也无功率（Ptype=NOCURVE，错误 226）")
        p_link.append(k); p_ptype.append(ptype)
        p_h0.append(P_h0); p_r.append(P_r); p_n.append(P_n)
        p_q0.append(P_q0); p_qmax.append(P_qmax); p_hmax.append(P_hmax)
        p_upat.append(upat); p_cpts.append(cu_pts)

    # CUSTOM 曲线点扁平化（ragged → 值数组 + 偏移指针；npz 可存）
    p_cptr = [0]
    p_cq, p_ch = [], []
    for cu_pts in p_cpts:
        for qx, hy in cu_pts:
            p_cq.append(qx); p_ch.append(hy)
        p_cptr.append(len(p_cq))

    # ---- [CONTROLS] 结构化（controldata input3.c:817-926 + input1.c:672-706）----
    c_link, c_node, c_type, c_status = [], [], [], []
    c_setting, c_grade, c_time = [], [], []
    for line in controls_raw:
        body = line.split(";", 1)[0].strip()
        if not body:
            continue
        tk = body.split()
        if len(tk) < 6:
            raise RuntimeError(f"{path}: [CONTROLS] 行字段不足: {line!r}")
        if tk[1] not in link_index:
            raise RuntimeError(f"{path}: [CONTROLS] 未知管段 {tk[1]!r}")
        k = link_index[tk[1]]
        ltype = int(link_type[k])
        if ltype == EN_CVPIPE:
            raise RuntimeError(f"{path}: [CONTROLS] 不能控制 CV 管（input3.c:856）")
        setting = MISSING                  # input3.c:838
        status = ST_ACTIVE                 # input3.c:841
        # match(str,substr)=大小写不敏感前缀匹配（input2.c:643-672）
        u2 = tk[2].upper()
        if u2.startswith("OPEN"):
            status = ST_OPEN               # input3.c:859-864
            if ltype == EN_PUMP:
                setting = 1.0
        elif u2.startswith("CLOSED"):
            status = ST_CLOSED             # input3.c:865-870
            if ltype == EN_PUMP:
                setting = 0.0
        else:
            setting = float(tk[2])         # input3.c:872
        # 泵/管道带数值设定时的状态（input3.c:876-884）
        if ltype in (EN_PUMP, EN_PIPE) and setting != MISSING:
            if setting < 0.0:
                raise ValueError(f"{path}: [CONTROLS] 设定为负: {line!r}")
            status = ST_CLOSED if setting == 0.0 else ST_OPEN
        # 控制类型（input3.c:886-896）
        u4 = tk[4].upper()
        node_i = -1
        grade = 0.0
        tsec = 0
        if u4.startswith("TIME"):          # match(Tok[4], w_TIME)（input3.c:887）
            ctype = TIMER
        elif u4.startswith("CLOCKTIME"):   # input3.c:888
            ctype = TIMEOFDAY
        else:
            if len(tk) < 8:
                raise RuntimeError(f"{path}: [CONTROLS] 行字段不足: {line!r}")
            if tk[5] not in node_index:
                raise RuntimeError(f"{path}: [CONTROLS] 未知节点 {tk[5]!r}")
            node_i = node_index[tk[5]]
            u6 = tk[6].upper()
            if u6.startswith("BELOW"):
                ctype = LOWLEVEL           # input3.c:893
            elif u6.startswith("ABOVE"):
                ctype = HILEVEL            # input3.c:894
            else:
                raise RuntimeError(f"{path}: [CONTROLS] 非法条件词 {tk[6]!r}")
        # 阈值/时刻（input3.c:899-911）
        if ctype in (TIMER, TIMEOFDAY):
            hrs = _hour(tk[5], tk[6] if len(tk) >= 7 else "")
            if hrs < 0.0:
                raise RuntimeError(f"{path}: [CONTROLS] 非法时间 {line!r}")
            tsec = int(3600.0 * hrs)       # input3.c:922
            if ctype == TIMEOFDAY:
                tsec %= 86400              # input3.c:923 SECperDAY
        else:
            grade = float(tk[7])           # input3.c:909
            # 阈值折算绝对水头（input1.c:678-688）
            if node_type[node_i] != 0:     # tank/reservoir：El + grade/Ucf[ELEV]
                grade = elev_ft[node_i] + grade / hcf     # input1.c:684
            else:                          # junction：El + grade/Ucf[PRESSURE]
                grade = elev_ft[node_i] + grade / pcf     # input1.c:687
        # 阀设定换算（input1.c:691-705）；泵转速不换算
        if setting != MISSING:
            if ltype in (3, 4, 5):         # PRV/PSV/PBV
                setting = setting / pcf    # input1.c:698
            elif ltype == 6:               # FCV
                setting = setting / flow_per_cfs   # input1.c:701
        c_link.append(k); c_node.append(node_i); c_type.append(ctype)
        c_status.append(status); c_setting.append(setting)
        c_grade.append(grade); c_time.append(tsec)

    # ---- B2 增补 [OPTIONS]/[TIMES] 标量 ----
    extra_iter = -1                        # 默认 UNBALANCED STOP（input1.c:115）
    checkfreq, maxcheck, damp_limit = 2, 10, 0.0
    # 状态机/收敛容差默认值（setdefaults input1.c:105-109,125；均可被 INP 覆盖）
    htol, qtol, rqtol = 0.0005, 0.0001, 1.0e-7   # HTOL/QTOL/RQTOL（input1.c:27,28,35）
    headerror, flowchange = 0.0, 0.0             # input1.c:108-109 默认 0
    for tk_ in options_raw:
        t0 = tk_[0].upper()
        if t0.startswith("UNBALANCED") and len(tk_) >= 2:
            if tk_[1].upper().startswith("STOP"):
                extra_iter = -1            # input3.c:1854
            elif tk_[1].upper().startswith("CONTINUE"):
                extra_iter = int(tk_[2]) if len(tk_) >= 3 else 0   # input3.c:1857-1858
        elif t0.startswith("CHECKFREQ") and len(tk_) >= 2:
            checkfreq = int(float(tk_[1]))
        elif t0.startswith("MAXCHECK") and len(tk_) >= 2:
            maxcheck = int(float(tk_[1]))
        elif t0.startswith("DAMPLIMIT") and len(tk_) >= 2:
            damp_limit = float(tk_[1])     # input3.c:1957-1960（可为 0）
        elif t0.startswith("HEADERROR") and len(tk_) >= 2:
            y = float(tk_[1])              # input3.c:1972-1976
            if y < 0.0:
                raise ValueError(f"{path}: HEADERROR 为负（错误 213）")
            headerror = y
        elif t0.startswith("FLOWCHANGE") and len(tk_) >= 2:
            y = float(tk_[1])              # input3.c:1964-1969
            if y < 0.0:
                raise ValueError(f"{path}: FLOWCHANGE 为负（错误 213）")
            flowchange = y
        elif t0.startswith("HTOL") and len(tk_) >= 2:
            y = float(tk_[1])              # input3.c:2020 hyd->Htol = y
            if y <= 0.0:
                raise ValueError(f"{path}: HTOL 非正（input3.c:2008 错误 213）")
            htol = y
        elif t0.startswith("QTOL") and len(tk_) >= 2:
            y = float(tk_[1])              # input3.c:2021 hyd->Qtol = y
            if y <= 0.0:
                raise ValueError(f"{path}: QTOL 非正（input3.c:2008 错误 213）")
            qtol = y
        elif t0.startswith("RQTOL") and len(tk_) >= 2:
            y = float(tk_[1])              # input3.c:2022-2026
            if y <= 0.0:
                raise ValueError(f"{path}: RQTOL 非正（input3.c:2008 错误 213）")
            if y >= 1.0:
                raise ValueError(f"{path}: RQTOL >= 1.0（input3.c:2024 错误 213）")
            rqtol = y

    # ---- [TIMES] RULE TIMESTEP（timedata input3.c:1667-1676 数值链 + :1683）----
    # wntr 对缺省 rule_timestep 给自身默认 360，无法区分"未设"；EPANET 未设时
    # Rulestep=0 → adjustdata 取 Hstep/10（input1.c:240），故从 INP 原文判定。
    rule_step = 0
    for tk_ in times_raw:
        if tk_[0].upper().startswith("RULE"):       # match(Tok[0], w_RULE)（input3.c:1683）
            n_ = len(tk_) - 1
            try:
                y_ = float(tk_[n_])                 # getfloat（input3.c:1667）
            except ValueError:
                y_ = _hour(tk_[n_], "")             # input3.c:1669
                if y_ < 0.0 and n_ >= 1:
                    y_ = _hour(tk_[n_ - 1], tk_[n_])   # input3.c:1671
                if y_ < 0.0:
                    raise ValueError(f"{path}: RULE TIMESTEP 非法（错误 213）")
            rule_step = int(3600.0 * y_ + 0.5)      # input3.c:1677 t=(long)(3600.0*y+0.5)

    # ---- 元数据标量 ----
    meta = {
        "flow_units": fu,
        "headloss": headloss,
        "viscosity": float(viscos),         # 内部 sq ft/sec（input1.c:271-282）
        "trials": int(opt.hydraulic.trials),
        "accuracy": float(opt.hydraulic.accuracy),
        "demand_model": str(opt.hydraulic.demand_model or "DDA").upper(),
        "emitter_exponent": gamma,          # 用户 γ
        "qexp": 1.0 / gamma,                # EPANET 内部 Qexp = 1/γ
        "duration_sec": int(opt.time.duration),
        "hyd_step_sec": int(opt.time.hydraulic_timestep),
        "pat_step_sec": int(opt.time.pattern_timestep),
        "pat_start_sec": int(opt.time.pattern_start),
        "demand_multiplier": float(opt.hydraulic.demand_multiplier),
        # ---- B2 增补标量 ----
        "extra_iter": int(extra_iter),
        "checkfreq": int(checkfreq),
        "maxcheck": int(maxcheck),
        "damp_limit": float(damp_limit),
        "htol": float(htol),
        "qtol": float(qtol),
        "rqtol": float(rqtol),
        "headerror": float(headerror),
        "flowchange": float(flowchange),
        "tstart_sec": int(opt.time.start_clocktime),
        "report_step_sec": int(opt.time.report_timestep),
        "report_start_sec": int(opt.time.report_start),
        # ---- 梯队3 增补 ----
        "spgrav": float(spgrav),          # input1.c:121（可被 SPECIFIC GRAVITY 覆盖）
        "press_units": press,             # Pressflag 词（PSI/KPA/METERS）
        "rule_step_sec": int(rule_step),  # input3.c:1683（0=未设→Hstep/10）
    }

    return Net(
        node_id=node_ids, node_type=node_type, elev_ft=elev_ft, node_ke=node_ke,
        res_head_pat=res_head_pat,
        link_id=link_ids, link_type=link_type, link_n1=link_n1, link_n2=link_n2,
        diam_ft=diam_ft, len_ft=len_ft, roughness=roughness, km_int=km_int,
        r_hw=r_hw, init_status=init_status, valve_setting_user=valve_setting_user,
        dem_node=np.asarray(dem_node, dtype=np.int32),
        dem_base_cfs=np.asarray(dem_base_cfs, dtype=np.float64),
        dem_pat=np.asarray(dem_pat, dtype=np.int32),
        pattern_ids=pattern_ids, patterns=patterns, meta=meta,
        controls_raw=controls_raw, rules_raw=rules_raw,
        tank_node=np.asarray(t_node, dtype=np.int32),
        tank_h0=np.asarray(t_h0, dtype=np.float64),
        tank_hmin=np.asarray(t_hmin, dtype=np.float64),
        tank_hmax=np.asarray(t_hmax, dtype=np.float64),
        tank_area=np.asarray(t_area, dtype=np.float64),
        tank_vmin=np.asarray(t_vmin, dtype=np.float64),
        tank_v0=np.asarray(t_v0, dtype=np.float64),
        tank_vmax=np.asarray(t_vmax, dtype=np.float64),
        tank_overflow=np.asarray(t_ovf, dtype=np.int8),
        pump_link=np.asarray(p_link, dtype=np.int32),
        pump_ptype=np.asarray(p_ptype, dtype=np.int8),
        pump_h0=np.asarray(p_h0, dtype=np.float64),
        pump_r=np.asarray(p_r, dtype=np.float64),
        pump_n=np.asarray(p_n, dtype=np.float64),
        pump_q0=np.asarray(p_q0, dtype=np.float64),
        pump_qmax=np.asarray(p_qmax, dtype=np.float64),
        pump_hmax=np.asarray(p_hmax, dtype=np.float64),
        pump_upat=np.asarray(p_upat, dtype=np.int32),
        pump_curve_q=np.asarray(p_cq, dtype=np.float64),
        pump_curve_h=np.asarray(p_ch, dtype=np.float64),
        pump_curve_ptr=np.asarray(p_cptr, dtype=np.int32),
        ctl_link=np.asarray(c_link, dtype=np.int32),
        ctl_node=np.asarray(c_node, dtype=np.int32),
        ctl_type=np.asarray(c_type, dtype=np.int8),
        ctl_status=np.asarray(c_status, dtype=np.int8),
        ctl_setting=np.asarray(c_setting, dtype=np.float64),
        ctl_grade=np.asarray(c_grade, dtype=np.float64),
        ctl_time=np.asarray(c_time, dtype=np.int64),
    )


# ----------------------------------------------------------------------
def exact_fixed_inputs_from_inp(net, inp_path, fields=("res", "vset")):
    """【新增，非缺省路径】把 wntr 的 m 往返值换回 INP 原文 strtod 值。

    背景（2026-09 模块四实测，data/exempt_diagnosis_wip.txt）：
    solver._apply_exact_props 已从 INP 原文位级重建 diam/len/r_hw/Km/需水/
    emitter，但**没有覆盖**下面三类量，它们仍走 wntr 的 m 表示往返
    （x_m / MperFT），在 US 制网上可差 1 ULP：
      "res"  reservoir 定水头 base（elev_ft[node_type==1]） - 直接是边界条件；
      "vset" PRV/PSV/FCV 的 setting（parse.py:634 已把该往返记为
             "发布审计欠账 d 项"）；
      "elev" 节点高程（DDA 下 junction 高程不进方程，只经 PRV/PSV 的
             hset = El[n2]+setting 与 [CONTROLS] 压力阈值间接进入）。
    实测：只改 "res"（+bwsn_1 的 "vset"）即可让 pub_net3 / pub_bwsn_network_1 /
    pub_bwsn_network_2 / pub_anytown 与官方 DLL 的 EPS 对拍从 2.1e-6 /
    6.1e-6 / 6.836 ft / 6.1e-12 降到**逐位 0.0 ft**。

    **就地修改 net，缺省不被任何现有通路调用**；调用方显式选择。
    返回 dict：各字段实际改动的元素数。
    """
    raw = {"JUNCTIONS": {}, "RESERVOIRS": {}, "TANKS": {}, "VALVES": {}}
    sec = None
    with open(inp_path, "r", encoding="latin-1") as f:
        for line in f:
            s0 = line.strip()
            if not s0:
                continue
            if s0.startswith("["):
                sec = s0[1:s0.find("]")].strip().upper()
                continue
            body = s0.split(";", 1)[0].strip()
            if not body:
                continue
            tok = body.split()
            if sec in raw:
                raw[sec][tok[0]] = tok
    hcf = MperFT if str(net.meta["flow_units"]) in SI_FLOW_UNITS else 1.0
    nt = np.asarray(net.node_type)
    lt = np.asarray(net.link_type)
    n_changed = {"res": 0, "elev": 0, "vset": 0}
    if "res" in fields or "elev" in fields:
        el = np.asarray(net.elev_ft, dtype=np.float64).copy()
        for i in range(net.N):
            nid = net.node_id[i]
            if nt[i] == 1 and nid in raw["RESERVOIRS"]:
                if "res" in fields or "elev" in fields:
                    v = float(raw["RESERVOIRS"][nid][1]) / hcf
                    n_changed["res"] += int(v != el[i])
                    el[i] = v
            elif "elev" in fields and nt[i] == 0 and nid in raw["JUNCTIONS"]:
                v = float(raw["JUNCTIONS"][nid][1]) / hcf
                n_changed["elev"] += int(v != el[i])
                el[i] = v
            elif "elev" in fields and nt[i] == 2 and nid in raw["TANKS"]:
                v = float(raw["TANKS"][nid][1]) / hcf
                n_changed["elev"] += int(v != el[i])
                el[i] = v
        net.elev_ft = el
    if "vset" in fields:
        vs = np.asarray(net.valve_setting_user, dtype=np.float64).copy()
        for k in range(net.L):
            lid = net.link_id[k]
            if lt[k] >= 3 and lid in raw["VALVES"] and len(raw["VALVES"][lid]) > 5:
                v = float(raw["VALVES"][lid][5])
                n_changed["vset"] += int(v != vs[k])
                vs[k] = v
        net.valve_setting_user = vs
    return n_changed


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    import os as _os  # noqa: E402
    ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    CASES = [
        (os.path.join(ROOT, "networks", "realInpData", "city_d.inp"), "city_d"),
        (os.path.join(ROOT, "networks", "InpData", "EXA4.inp"), "EXA4"),
    ]
    TYPE_NAMES = {0: "CVPIPE", 1: "PIPE", 2: "PUMP", 3: "PRV", 4: "PSV",
                  5: "PBV", 6: "FCV", 7: "TCV", 8: "GPV"}

    for inp, stem in CASES:
        print("=" * 70)
        print("解析", inp)
        net = parse_inp(inp)
        nt = np.bincount(net.node_type, minlength=3)
        print(f"N={net.N} L={net.L}  junction={nt[0]} reservoir={nt[1]} tank={nt[2]}")
        lt = {TYPE_NAMES[t]: int(np.sum(net.link_type == t))
              for t in sorted(set(net.link_type.tolist()))}
        print("管段类型计数:", lt)
        print("前 5 个节点 ID:", net.node_id[:5])
        print("前 5 个管段 ID:", net.link_id[:5])
        print("init_status 计数:", dict(zip(*[a.tolist() for a in
                                              np.unique(net.init_status, return_counts=True)])))

        # r_hw 统计（仅管道）
        is_pipe = net.link_type <= 1
        rp = net.r_hw[is_pipe]
        print(f"r_hw(管道) min={rp.min():.6g} median={np.median(rp):.6g} max={rp.max():.6g}")

        if stem == "city_d":
            i162 = net.node_id.index("162")
            d0 = net.demand_cfs_at(0)
            d1 = net.demand_cfs_at(3600)
            print(f"节点162 需水: t=0 -> {d0[i162]:.10f} cfs, t=3600 -> {d1[i162]:.10f} cfs")
            # 手算对拍：162 有 3 类（13.361545598@2439, 0.1622850038934615@2483,
            # 3.056386019832966@2484，单位 LPS），F[0]/F[1] 直接查 pattern
            expect = {}
            for t_i, t in enumerate((0, 3600)):
                s = 0.0
                for b, pid in ((13.361545598025238, "2439"),
                               (0.1622850038934615, "2483"),
                               (3.056386019832966, "2484")):
                    F = net.patterns[net.pattern_ids.index(pid)]
                    s += b * F[t_i]
                expect[t] = s / LPSperCFS
            print(f"  手算期望: t=0 -> {expect[0]:.10f}, t=3600 -> {expect[3600]:.10f}")
            assert abs(d0[i162] - expect[0]) < 1e-12 and abs(d1[i162] - expect[3600]) < 1e-12
            # 888 有 4 类
            n888 = int(np.sum(net.dem_node == net.node_id.index("888")))
            n162 = int(np.sum(net.dem_node == i162))
            print(f"  需水类别数: 162 -> {n162}（应 3）, 888 -> {n888}（应 4）")
            assert n162 == 3 and n888 == 4
            # 4 条 [STATUS] Closed（city_d 里是 4 个 TCV）
            closed = [net.link_id[i] for i in np.where(net.init_status == 0)[0]]
            print("  初始 Closed 管段:", closed, "（应含 66/87/2493/2497）")
            assert set(closed) == {"66", "87", "2493", "2497"}
            # reservoir 水头
            h0 = net.reservoir_head_ft_at(0)
            ir = int(np.where(net.node_type == 1)[0][0])
            print(f"  reservoir {net.node_id[ir]} t=0 水头 = {h0[ir]:.4f} ft "
                  f"(base {net.elev_ft[ir]:.4f} ft × pattern)")
        if stem == "EXA4":
            # EXA4 段序为 PUMPS→VALVES→PIPES，EPANET 索引应以泵开头
            assert net.link_id[0] == "PU2" and net.link_type[0] == EN_PUMP, "EXA4 链序错误"
            print("  EXA4 管段顺序正确以泵开头（EPANET 文件序）")

        # save/load 往返
        with tempfile.TemporaryDirectory() as td:
            net.save(td, stem)
            net2 = Net.load(td, stem)
            assert net2.node_id == net.node_id and net2.link_id == net.link_id
            for a, b in ((net.r_hw, net2.r_hw), (net.elev_ft, net2.elev_ft),
                         (net.dem_base_cfs, net2.dem_base_cfs)):
                assert np.array_equal(a, b)
            assert np.array_equal(net.demand_cfs_at(7200), net2.demand_cfs_at(7200))
            assert np.allclose(net.reservoir_head_ft_at(7200),
                               net2.reservoir_head_ft_at(7200), equal_nan=True)
            npz_kb = os.path.getsize(os.path.join(td, f"{stem}_net.npz")) / 1024
            print(f"  save/load 往返一致 ✓ (net.npz {npz_kb:.1f} KB)")

    print("=" * 70)
    print("parse.py 冒烟测试全部通过")

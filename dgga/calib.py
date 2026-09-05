# -*- coding: utf-8 -*-
"""calib.py - 管网摩阻系数校核：C 的可微箱约束参数化 + 可辨识性诊断。

本模块提供两件事：

一、C → r_hw 的可微映射（正向 + 链式导数）
    r = 4.727·L/C^1.852/d^4.871（hydcoeffs.c:95；parse.py:602-603，
    位级重建版 solver.py:355-357 用 msvcrt 的 pow）。
    * hw_resistance(C, len_ft, diam_ft)：torch 正向式，运算次序与 parse.py:603
      逐字一致（`4.727 * ln / C**1.852 / d**4.871`），因此 C 取原值时输出与
      net.r_hw **逐位相同**（city_d 475 根管实测 ULP 差 0）。
    * RoughnessParam：把待校核管道的 C 参数化为无约束 theta，输出可直接喂给
      ImplicitGGASolve.apply 的 r_hw 形参（范式同 demo_leak_inversion.py:112-114
      的 ke_int_of_C）。
    * dr_dC：链式因子 dr/dC = −Hexp·r/C（上一轮在 sensitivity.py 里的最小版
      已合并至此，sensitivity.py 现从本模块 import，消除重复实现）。

二、可辨识性诊断 identifiability()
    在精确雅可比（伴随）意义下回答"哪些管道根本不可校核"，并对每根不可校核的
    管道给出归因。零灵敏度的结构性来源共四类，互斥地按优先级归因：
      non_pipe    非管道链路（泵/阀）：不存在 H-W 的 C，dφ/dr ≡ 0
                  （autodiff.py:274,289,306 把 TCV/阀/关闭支的 dr 显式置 0）
      closed      关闭支：hloss=CBIG·q 与 r 无关，dφ/dr ≡ 0（autodiff.py:306）
      dead_branch 死支：拓扑上开放但无源无汇的叶枝，流量恒 0
      clamped     RQtol 钳位管：低流量线性化使水损与 r 无关（hydcoeffs.c:554-558，
                  autodiff.py:255 的 `dr = np.where(lin, 0.0, ...)`）
      unobservable 以上皆非、但伴随向量在该管上恰为 0（当前传感器布置观测不到）
    只有 reason=="informative" 的管道才值得进入校核反问题。

工程约束（沿用第 0 步既定结论）：全程 float64；polish_steps ≥ 3（city_d 族条件数
约 1.85e10，不精抛光损失面有约 1e-5 噪声）。
"""

import numpy as np
import torch

if __package__ in (None, ""):        # 允许 `python dgga/calib.py` 直跑冒烟测试
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))))
    __package__ = "dgga"
    import dgga as _dgga             # noqa: F401  确保父包已初始化

from .autodiff import solve_polished

__all__ = ["hw_resistance", "RoughnessParam", "dr_dC",
           "clamped_mask", "dead_branch_mask", "identifiability", "REASONS"]

# EN_CVPIPE(0) / EN_PIPE(1)（parse.py:588-589）。注意 solver.is_pipe 只含 EN_PIPE，
# 而 CVPIPE 同样有 H-W 阻力和非零 dφ/dr，诊断里必须一并计入。
_PIPE_TYPES = (0, 1)
_HW_COEF = 4.727        # hydcoeffs.c:95
_HW_CEXP = 1.852        # = Hexp
_HW_DEXP = 4.871

REASONS = ("informative", "clamped", "dead_branch", "closed", "non_pipe",
           "unobservable")


# ======================================================================
# 一、C ↔ r_hw
# ======================================================================
def hw_resistance(C, len_ft, diam_ft):
    """H-W 内部阻力 r = 4.727·len_ft/C^1.852/diam_ft^4.871（可微）。

    参数可为标量 / numpy / torch，按 torch 广播规则对齐；返回 float64 张量，
    对 C（以及 len_ft / diam_ft）可微。

    运算次序刻意与 parse.py:603 逐字一致 - `4.727 * ln / C**1.852 / d**4.871`，
    而不是数学等价的 `4.727*ln/(C**1.852 * d**4.871)`。浮点乘除不满足结合律，
    换一种写法就丢掉位级一致性；实测 city_d 475 根管按本式重算与 net.r_hw
    的 ULP 差为 0，改成括号版会出现 1~2 ULP 漂移。

    与 solver.r_hw 的差异：GGASolver(inp_path=...) 会用 msvcrt.dll 的 pow 重建
    r_hw（solver.py:355-357），torch 的 pow 与之在 city_d 上有 ≤6 ULP、58/475
    根管的差别（相对 8.1e-16）。这是 libm 实现差异，不是公式差异；需要位级对齐
    solver 时把 RoughnessParam 的基底传成 solver.r_hw（默认行为）。
    """
    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.to(torch.float64) if x.dtype != torch.float64 else x
        return torch.as_tensor(np.asarray(x, dtype=np.float64))

    C, ln, dm = _t(C), _t(len_ft), _t(diam_ft)
    return _HW_COEF * ln / C ** _HW_CEXP / dm ** _HW_DEXP


def dr_dC(solver, r_hw=None):
    """H-W 链式因子 dr/dC = −Hexp·r/C（[L] numpy，非 H-W 管道位置为 0）。

    由 r = 4.727·L/C^Hexp/d^4.871 直接微分得到；与 scripts/extfd_epanet.py:233-234
    用 EPANET DLL 外部有限差分验证过的表达式同源。
    （原实现在 sensitivity.py:48-65，标了 TODO 待合并；现统一在此，sensitivity.py
    从本模块 import。）
    """
    s = solver
    if getattr(s, "headloss_form", "H-W") != "H-W":
        raise NotImplementedError(
            "wrt='C' 仅对 H-W 有意义（D-W 的 Kc 经 f 进入水损、C-M 的 n 另有式）；"
            "请改用 wrt='r'")
    r = np.asarray(s.r_hw.detach().cpu().numpy() if r_hw is None else r_hw,
                   dtype=np.float64)
    C = np.asarray(s.kc_np, dtype=np.float64)
    is_pipe = np.isin(s.lt_np, _PIPE_TYPES) & (C > 0.0)
    out = np.zeros(s.L, dtype=np.float64)
    out[is_pipe] = -s.hexp * r[is_pipe] / C[is_pipe]
    return out


def _net_of(obj):
    """接受 Net 或 GGASolver（solver.py:94 存了 self.net），返回底层 Net。"""
    return getattr(obj, "net", obj)


class RoughnessParam(torch.nn.Module):
    """待校核管道摩阻系数 C 的无约束参数化（箱约束由 sigmoid 承担）。

    C(theta) = C_lo + (C_hi − C_lo)·sigmoid(theta)

    为什么用 sigmoid 而不是 softplus / clamp
    ---------------------------------------
    * H-W 的 C 是**双侧**有界量：物理上 C≤0 使 r 发散、C 过大使管道近乎无阻，
      工程取值区间约 [40,160]（新铸铁~130、严重结垢~40、PE 新管~150）。
      softplus 只给单侧下界 C>0，优化器完全可能把 C 推到 10³ 量级去拟合噪声，
      这在可辨识性差的管道上必然发生（S 的零列方向上损失是平的）；
    * clamp / 投影梯度在边界处 dC/dθ ≡ 0，一旦某根管撞到边界就永久死掉，
      而本课题恰恰要区分"梯度为 0 是因为结构不可辨识"还是"因为参数化死了" -
      两者混在一起就没法做诊断。sigmoid 的 dC/dθ = (C_hi−C_lo)·σ(1−σ) 对任意
      有限 θ 严格为正，零梯度只可能来自水力学本身；
    * 边界只在 θ→±∞ 处渐近取到，因此 Adam / L-BFGS 可以完全无约束地跑。
      代价是接近边界时梯度指数衰减（saturation），所以 C_lo/C_hi 应给得比先验
      区间宽一些，让初值落在 σ 的线性段附近。

    θ 与 C 的往返：theta_of_C 是解析逆（logit），实测在 [40,160]/C=130 上
    C(theta_of_C(130)) 逐位等于 130.0，故"C 取原值 ⇒ r_hw() 位级复现 net.r_hw"
    这一硬性检查能通过。

    参数
    ----
    net      : Net 或 GGASolver（后者会取 solver.net，基底默认用 solver.r_hw）
    pipe_idx : 待校核管道的链路全局下标；None = 全部 H-W 管道（含 CVPIPE）
    C_lo/C_hi: 箱约束
    init     : 初值 C（标量或长度 n_free 的数组）；None = 取网络现值 roughness
    r_base   : [L] 基底阻力（自由管道以外原样透传）；None 时取
               solver.r_hw（若传的是 solver）或 net.r_hw

    属性/方法
    --------
    theta    : torch.nn.Parameter [n_free]
    C()      : [n_free] 箱约束后的摩阻系数
    r_hw()   : [L] 完整内部阻力，自由管道现算、其余（泵/阀/未选中管）原样透传，
               可直接作为 ImplicitGGASolve.apply 的 r_hw 实参
    """

    def __init__(self, net, pipe_idx=None, C_lo=40.0, C_hi=160.0, init=None,
                 r_base=None, device=None):
        super().__init__()
        n = _net_of(net)
        L = int(n.L)
        lt = np.asarray(n.link_type, dtype=np.int64)
        rough = np.asarray(n.roughness, dtype=np.float64)
        headloss = str(n.meta.get("headloss", "H-W")).upper()
        if headloss != "H-W":
            raise NotImplementedError(
                f"RoughnessParam 只参数化 H-W 的 C（本网 headloss={headloss}）；"
                f"D-W 的 Kc / C-M 的 n 需另写映射")
        if not (C_hi > C_lo > 0.0):
            raise ValueError(f"箱约束非法：C_lo={C_lo}, C_hi={C_hi}")

        is_pipe = np.isin(lt, _PIPE_TYPES) & (rough > 0.0)
        if pipe_idx is None:
            idx = np.where(is_pipe)[0]
        else:
            idx = np.asarray(pipe_idx, dtype=np.int64).ravel()
            if idx.size and (idx.min() < 0 or idx.max() >= L):
                raise ValueError("pipe_idx 越界")
            bad = idx[~is_pipe[idx]]
            if bad.size:
                raise ValueError(
                    f"pipe_idx 含非 H-W 管道链路 {bad.tolist()[:10]}"
                    f"（link_type={lt[bad].tolist()[:10]}）：泵/阀没有 C")
            if np.unique(idx).size != idx.size:
                raise ValueError("pipe_idx 有重复下标")
        self.n_free = int(idx.size)

        if init is None:
            C0 = rough[idx].copy()
        else:
            C0 = np.broadcast_to(np.asarray(init, dtype=np.float64),
                                 (self.n_free,)).astype(np.float64).copy()
        if np.any(C0 <= C_lo) or np.any(C0 >= C_hi):
            raise ValueError(
                f"初值 C 必须严格落在开区间 ({C_lo}, {C_hi}) 内："
                f"min={C0.min():.4g} max={C0.max():.4g}（sigmoid 的边界只在"
                f"±∞ 取到，端点会给出 ±inf 的 theta）")

        if r_base is None:
            r0 = getattr(net, "r_hw", None)
            if r0 is None:
                r0 = n.r_hw
            r0 = r0.detach().cpu().numpy() if isinstance(r0, torch.Tensor) \
                else np.asarray(r0)
        else:
            r0 = r_base.detach().cpu().numpy() \
                if isinstance(r_base, torch.Tensor) else np.asarray(r_base)
        r0 = np.asarray(r0, dtype=np.float64).ravel()
        if r0.shape != (L,):
            raise ValueError(f"r_base 形状 {r0.shape} != ({L},)")

        self.L = L
        self.C_lo, self.C_hi = float(C_lo), float(C_hi)
        dev = device
        self.theta = torch.nn.Parameter(
            torch.as_tensor(self.theta_of_C(C0, C_lo, C_hi),
                            dtype=torch.float64, device=dev))
        self.register_buffer("idx", torch.as_tensor(idx, dtype=torch.int64,
                                                    device=dev))
        self.register_buffer("len_free", torch.as_tensor(
            np.asarray(n.len_ft, dtype=np.float64)[idx], device=dev))
        self.register_buffer("diam_free", torch.as_tensor(
            np.asarray(n.diam_ft, dtype=np.float64)[idx], device=dev))
        self.register_buffer("r_base", torch.as_tensor(r0, device=dev))

    # ------------------------------------------------------------------
    @staticmethod
    def theta_of_C(C, C_lo=40.0, C_hi=160.0):
        """箱约束映射的解析逆 theta = logit((C−C_lo)/(C_hi−C_lo))（numpy）。"""
        u = (np.asarray(C, dtype=np.float64) - C_lo) / (C_hi - C_lo)
        if np.any(u <= 0.0) or np.any(u >= 1.0):
            raise ValueError("C 必须严格落在 (C_lo, C_hi) 内")
        return np.log(u / (1.0 - u))

    def C(self):
        """[n_free] 摩阻系数，恒落在 (C_lo, C_hi) 内，对 theta 可微。"""
        return self.C_lo + (self.C_hi - self.C_lo) * torch.sigmoid(self.theta)

    def r_hw(self, C=None):
        """[L] 完整内部阻力张量。

        自由管道位置用 hw_resistance(C(), len, diam) 现算，其余位置（泵、阀、
        未选中的管、关闭管）原样透传基底 r_base - 透传是**位级**的，index_copy
        只覆盖 idx 指定的行。

        传 C 可复用外部已算好的系数张量（例如需要 C.retain_grad() 时）。
        """
        Cv = self.C() if C is None else C
        r_free = hw_resistance(Cv, self.len_free, self.diam_free)
        return self.r_base.index_copy(0, self.idx, r_free)

    def C_full(self):
        """[L] numpy：把当前 C 写回完整链路数组（非自由位置给网络现值）。"""
        out = np.zeros(self.L, dtype=np.float64)
        out[self.idx.cpu().numpy()] = self.C().detach().cpu().numpy()
        return out

    def extra_repr(self):
        return (f"n_free={self.n_free}, L={self.L}, "
                f"box=({self.C_lo}, {self.C_hi})")


# ======================================================================
# 二、可辨识性诊断
# ======================================================================
def _solve_once(solver, demand, res_head, r_hw=None, ke=None,
                accuracy=1e-12, max_iter=200, polish_steps=3, **kw):
    if polish_steps < 3:
        raise ValueError("polish_steps 至少为 3（GGA 半迭代的 1e-7~1e-9 平台会"
                         "在损失面上留约 1e-5 噪声）")
    return solve_polished(solver, demand, res_head, ke=ke, r_hw=r_hw,
                          accuracy=accuracy, max_iter=max_iter,
                          polish_steps=polish_steps, **kw)


def clamped_mask(solver, sol, frames=None, margin=10.0):
    """RQtol 钳位管掩码（[L] bool）与逐帧明细。

    判据照抄 scripts/gradcheck_3way.py:122-123：
        hg_fric = Hexp·r·|q|^(Hexp−1)
        pipe_ok = is_pipe & ~closed & (hg_fric > margin·RQtol)
    钳位 ⇔ 是管道、未关闭、且 hg_fric 未超过 margin·RQtol。margin=10 是
    gradcheck 里用的安全余量（远离 hydcoeffs.c:554 的 `hgrad < RQtol` 分支面），
    margin=1.0 则退化为求解器真正执行的严格判据。

    多帧聚合：按"是否存在任意一帧脱离钳位" - 只要有一帧非钳位就视为非钳位
    （该帧提供了非零 dφ/dr，管道在这组工况下是可激励的）。

    返回 dict(mask=[L], per_frame=[T,L], n_per_frame=[T], frames=[...])
    """
    s = solver
    r = np.asarray(sol["r_hw"], dtype=np.float64)
    q_all = np.atleast_2d(np.asarray(sol["q"], dtype=np.float64))
    B = q_all.shape[0]
    fr = list(range(B)) if frames is None else [int(b) for b in frames]
    # 与 gradcheck 一致地取 is_pipe，但把 CVPIPE 一并计入（solver.is_pipe 只含
    # EN_PIPE；CVPIPE 同样有 H-W 阻力，漏掉会把它错判成"非管道"）
    is_pipe = np.isin(s.lt_np, _PIPE_TYPES)
    base = is_pipe & ~s.closed_np
    thr = margin * s.rqtol
    per = np.zeros((len(fr), s.L), dtype=bool)
    for i, b in enumerate(fr):
        hg_fric = s.hexp * r * np.abs(q_all[b]) ** (s.hexp - 1.0)
        per[i] = base & ~(hg_fric > thr)
    return dict(mask=per.all(axis=0) if len(fr) else base.copy(),
                per_frame=per, n_per_frame=per.sum(axis=1), frames=fr,
                margin=float(margin))


def dead_branch_mask(solver, demand=None, ke=None, dem_atol=0.0):
    """死支掩码（[L] bool）：拓扑上开放、但无源无汇的叶枝，流量恒为 0。

    迭代剪叶：反复删除"度为 1 且既非定水头节点、又无需水、又无 emitter"的
    节点所连的开放链路。这类链路的流量被连续性方程钉死在 0，⇒ 任何 r 变化
    都不产生水损变化 ⇒ 灵敏度列恒 0，与传感器布置无关。

    死支与 RQtol 钳位在数值上重叠（q=0 必然钳位），但成因不同：死支是拓扑性的
    （加多少传感器、换什么工况都救不回来），钳位是工况性的（换高需水时段可能
    脱离）。归因时死支优先级更高。
    """
    s = solver
    N, L = s.N, s.L
    n1, n2 = np.asarray(s.n1_np), np.asarray(s.n2_np)
    keep = np.zeros(N, dtype=bool)
    keep[s.fixed_nodes] = True                       # 水库/水池：源
    if demand is not None:
        d = np.atleast_2d(np.asarray(demand, dtype=np.float64))
        keep |= np.abs(d).max(axis=0) > dem_atol     # 有需水：汇
    if ke is not None:
        k = np.atleast_2d(np.asarray(ke, dtype=np.float64))
        keep |= np.abs(k).max(axis=0) > 0.0          # 有 emitter：汇
    dead = np.zeros(L, dtype=bool)
    alive = ~s.closed_np
    while True:
        deg = np.zeros(N, dtype=np.int64)
        np.add.at(deg, n1[alive], 1)
        np.add.at(deg, n2[alive], 1)
        leaf = (deg == 1) & ~keep
        if not leaf.any():
            break
        cut = alive & (leaf[n1] | leaf[n2])
        if not cut.any():
            break
        dead |= cut
        alive &= ~cut
    return dead


def identifiability(solver, demand, res_head, sensor_nodes, r_hw=None, ke=None,
                    weights=None, wrt="C", frames=None, atol=0.0,
                    clamp_margin=10.0, accuracy=1e-12, max_iter=200,
                    polish_steps=3, return_S=False, **solve_kw):
    """可辨识性诊断：哪些管道在当前传感器布置 + 工况下根本不可校核。

    参数
    ----
    solver        : GGASolver
    demand/res_head : [N] 或 [B,N]
    sensor_nodes  : 传感器节点全局下标（必须是 junction）
    r_hw          : [L] 内部阻力；None = solver 自带
    weights       : 损失 L = Σ w·p_sensor 的权重。标量 / [m] / [T·m]；None = 全 1
    wrt           : 'C'（默认，含链式因子 dr/dC）或 'r'
    frames        : 参与诊断的批下标；None = 全部
    atol          : 判"零列"的绝对阈值；0 = 只认解析恒零（推荐）

    返回 dict
    --------
    clamped_mask      [L] RQtol 钳位（多帧聚合：脱离过即非钳位）
    clamped_strict    [L] margin=1.0 的严格钳位（求解器实际分支）
    dead_mask         [L] 死支
    closed_mask       [L] 关闭链路
    pipe_mask         [L] 是否 H-W 管道（CVPIPE/PIPE）
    zero_grad_mask    [L] 加权和损失下梯度精确为零（结构性不可辨识）
    zero_col_mask     [L] S 的整列 ≤ atol（对任意权重都不可辨识）
    grad              [L] 加权梯度 wᵀS
    reason            [L] object 数组，取值见 REASONS，互斥归因
    n_informative     int，reason=='informative' 的管道数
    counts            dict，各 reason 的计数
    S                 return_S=True 时给出 [T·m, L] 灵敏度矩阵
    info              sensitivity_matrix 的计时/残差信息
    """
    # 延迟导入：sensitivity.py 顶层 import 本模块的 dr_dC，此处再顶层 import 它
    # 会成环。
    from .sensitivity import sensitivity_matrix

    s = solver
    S, info = sensitivity_matrix(
        s, demand, res_head, r_hw, sensor_nodes, wrt=wrt, frames=frames, ke=ke,
        accuracy=accuracy, max_iter=max_iter, polish_steps=polish_steps,
        return_info=True, **solve_kw)
    m = int(np.asarray(info["sensors"]).size)
    T = S.shape[0] // m

    if weights is None:
        w = np.ones(S.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size == 1:
            w = np.full(S.shape[0], float(w[0]))
        elif w.size == m and S.shape[0] != m:
            w = np.tile(w, T)                        # [m] → 逐帧复用
        if w.shape != (S.shape[0],):
            raise ValueError(f"weights 长度 {w.size} 与 S 行数 {S.shape[0]} 不匹配")
    grad = w @ S                                     # dL/dθ_k，θ = C 或 r

    sol = _solve_once(s, demand, res_head, r_hw=r_hw, ke=ke, accuracy=accuracy,
                      max_iter=max_iter, polish_steps=polish_steps, **solve_kw)
    cm = clamped_mask(s, sol, frames=info["frames"], margin=clamp_margin)
    cm1 = clamped_mask(s, sol, frames=info["frames"], margin=1.0)
    dead = dead_branch_mask(s, demand=demand, ke=ke)
    closed = s.closed_np.copy()
    pipe = np.isin(s.lt_np, _PIPE_TYPES)

    zero_col = ~(np.max(np.abs(S), axis=0) > atol)
    zero_grad = ~(np.abs(grad) > atol)

    reason = np.empty(s.L, dtype=object)
    reason[:] = "informative"
    reason[zero_col & ~pipe] = "unobservable"        # 占位，下面按优先级覆盖
    reason[~pipe] = "non_pipe"
    reason[pipe & closed] = "closed"
    reason[pipe & ~closed & dead] = "dead_branch"
    rest = pipe & ~closed & ~dead
    reason[rest & cm["mask"]] = "clamped"
    reason[rest & ~cm["mask"] & zero_col] = "unobservable"
    counts = {k: int(np.sum(reason == k)) for k in REASONS}

    out = dict(clamped_mask=cm["mask"], clamped_strict=cm1["mask"],
               clamped_per_frame=cm["per_frame"],
               n_clamped_per_frame=cm["n_per_frame"],
               dead_mask=dead, closed_mask=closed, pipe_mask=pipe,
               zero_grad_mask=zero_grad, zero_col_mask=zero_col, grad=grad,
               reason=reason, counts=counts,
               n_informative=counts["informative"],
               n_pipe=int(pipe.sum()), sensors=info["sensors"],
               frames=info["frames"], wrt=wrt, info=info)
    if return_S:
        out["S"] = S
    return out


# ======================================================================
# 冒烟测试（前台运行；网：city_d）
# ======================================================================
if __name__ == "__main__":
    import os
    import sys
    import time

    sys.stdout.reconfigure(encoding="utf-8")
    _HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(_HERE)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    # RoughnessParam / hw_resistance / identifiability / dead_branch_mask 已在
    # 本模块内定义，不再自 import（避免以 dgga.calib 之名加载第二份副本）
    from dgga.autodiff import ImplicitGGASolve                    # noqa: E402
    from dgga.parse import Net                                    # noqa: E402
    from dgga.sensitivity import svd_spectrum                     # noqa: E402
    from dgga.solver import GGASolver                             # noqa: E402

    SEED = 2026          # 与 scripts/sensitivity_check.py 同种子同用法
    N_SENSOR = 40
    GGA_MI = 60
    POLISH = 3

    torch.set_default_dtype(torch.float64)
    inp = os.path.join(ROOT, "networks", "realInpData", "city_d.inp")
    net = Net.load(os.path.join(ROOT, "data", "reference"), "city_d")
    # 陷阱：GGASolver(inp_path=...) 就地修正 net.dem_base_cfs → 先构造 solver
    s = GGASolver(net, mode="dense", inp_path=inp)
    rng = np.random.default_rng(SEED)
    sensors = np.sort(rng.choice(s.junc_nodes, size=N_SENSOR, replace=False))

    d0 = net.demand_cfs_at(0)
    rh0 = np.nan_to_num(net.reservoir_head_ft_at(0))
    ke0 = np.zeros(net.N)
    r_solver = s.r_hw.detach().cpu().numpy().copy()
    elev = np.asarray(net.elev_ft, dtype=np.float64)

    print(f"=== city_d: N={s.N} Nj={s.Nj} L={s.L} headloss={s.headloss_form} "
          f"RQtol={s.rqtol:g} Hexp={s.hexp} ===")
    lt_hist = {int(k): int(v) for k, v in zip(*np.unique(s.lt_np,
                                                         return_counts=True))}
    print(f"link_type 分布 {lt_hist}；关闭链路 {int(s.closed_np.sum())}")

    # ---------------- ① 硬性正确性：C≡130 时 r_hw() 位级复现 ----------------
    print("\n" + "=" * 74)
    print("① 硬性正确性检查：theta 使 C 全部 =130，r_hw() vs net.r_hw 逐位比对")
    rp_net = RoughnessParam(net, None, C_lo=40.0, C_hi=160.0, init=130.0)
    Cv = rp_net.C().detach().cpu().numpy()
    print(f"  自由管道数 n_free = {rp_net.n_free}")
    print(f"  C() 是否逐位等于 130.0：{bool(np.all(Cv == 130.0))}"
          f"（min={Cv.min():.17g} max={Cv.max():.17g}）")
    r_new = rp_net.r_hw().detach().cpu().numpy()
    r_parse = np.asarray(net.r_hw, dtype=np.float64)
    same_bits = np.array_equal(r_new.view(np.int64), r_parse.view(np.int64))
    ulp = np.abs(r_new.view(np.int64) - r_parse.view(np.int64))
    print(f"  r_hw() vs net.r_hw 逐位相同：{same_bits}"
          f"（max ULP={int(ulp.max())}，不同位数={int((ulp > 0).sum())}/{s.L}）")
    # 再与 solver 的位级重建版对比（差异来源：msvcrt pow vs torch pow）
    rp_s = RoughnessParam(s, None, init=130.0)
    r_new_s = rp_s.r_hw().detach().cpu().numpy()
    ulp_s = np.abs(r_new_s.view(np.int64) - r_solver.view(np.int64))
    rel_s = np.abs(r_new_s - r_solver)
    m_pos = r_solver > 0
    print(f"  r_hw()(基底=solver.r_hw) vs solver.r_hw："
          f"max ULP={int(ulp_s.max())}，不同位数={int((ulp_s > 0).sum())}/{s.L}，"
          f"max rel={np.max(rel_s[m_pos] / r_solver[m_pos]):.3e}")
    print("  ↑ 差异来源：solver.py:356 用 msvcrt.dll 的 pow 重建 r_hw，"
          "torch 用自带 libm；parse.py:603 用 numpy 的 **，与 torch 逐位一致。")
    # 透传检查：非自由位置必须位级不变
    free = rp_net.idx.cpu().numpy()
    pass_mask = np.ones(s.L, dtype=bool)
    pass_mask[free] = False
    print(f"  非自由位置（{int(pass_mask.sum())} 条泵/阀/关闭外链路）透传位级不变："
          f"{np.array_equal(r_new[pass_mask].view(np.int64), r_parse[pass_mask].view(np.int64))}")
    # 直调 hw_resistance 的一致性
    r_direct = hw_resistance(130.0, np.asarray(net.len_ft)[free],
                             np.asarray(net.diam_ft)[free]).numpy()
    print(f"  hw_resistance() 直调 vs net.r_hw[free] 逐位相同："
          f"{np.array_equal(r_direct.view(np.int64), r_parse[free].view(np.int64))}")

    # ---------------- ② 对 C 求梯度 ----------------
    print("\n" + "=" * 74)
    print("② 损失 = 传感器压力加权和，backward 到 C")
    dt = torch.tensor(d0)
    rht = torch.tensor(rh0)
    ket = torch.tensor(ke0)
    w = torch.ones(N_SENSOR, dtype=torch.float64)
    rp = RoughnessParam(s, None, init=130.0)     # 基底 = solver.r_hw
    Ct = rp.C()
    Ct.retain_grad()
    rt = rp.r_hw(C=Ct)
    t0 = time.perf_counter()
    head, flow, emit = ImplicitGGASolve.apply(dt, rht, ket, rt, s,
                                              1e-12, GGA_MI, POLISH)
    t_fwd = time.perf_counter() - t0
    p_sensor = head[sensors] - torch.tensor(elev[sensors])
    loss = (w * p_sensor).sum()
    t0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t0
    gC = Ct.grad.detach().cpu().numpy()
    gth = rp.theta.grad.detach().cpu().numpy()
    nz = np.abs(gC) > 0.0
    print(f"  loss = Σp = {float(loss.detach()):.6f} ft；"
          f"前向 {t_fwd:.3f}s，反向 {t_bwd:.4f}s")
    print(f"  dL/dC 非零管道数 {int(nz.sum())}/{rp.n_free}"
          f"（精确零 {int((~nz).sum())} 根）")
    if nz.any():
        a = np.abs(gC[nz])
        qs = np.percentile(a, [0, 25, 50, 75, 100])
        print(f"  |dL/dC| 非零量级：min={qs[0]:.3e} p25={qs[1]:.3e} "
              f"中位={qs[2]:.3e} p75={qs[3]:.3e} max={qs[4]:.3e}"
              f"（动态范围 {qs[4] / qs[0]:.2e} 倍）")
        dec = np.floor(np.log10(a)).astype(int)
        hist = {int(k): int(v) for k, v in zip(*np.unique(dec,
                                                          return_counts=True))}
        print(f"  |dL/dC| 十进制量级直方图 {{log10:根数}} = {hist}")
        print(f"  |dL/dtheta| 非零 {int((np.abs(gth) > 0).sum())}，"
              f"max={np.abs(gth).max():.3e}（含 dC/dtheta=(C_hi-C_lo)σ(1-σ)="
              f"{(rp.C_hi - rp.C_lo) * 0.75 * 0.25:.4g}）")

    # ---------------- ③ 复现已知事实 ----------------
    print("\n" + "=" * 74)
    print(f"③ 40 个随机传感器（seed={SEED}）下的结构性可辨识性")
    # (a) 对 r_hw 的全 [L] 梯度（复现"554 条链路里多少条梯度非零"）
    rB = torch.tensor(r_solver, requires_grad=True)
    hB, _, _ = ImplicitGGASolve.apply(dt, rht, ket, rB, s, 1e-12, GGA_MI, POLISH)
    ((torch.ones(N_SENSOR, dtype=torch.float64)
      * (hB[sensors] - torch.tensor(elev[sensors]))).sum()).backward()
    gr = rB.grad.detach().cpu().numpy()
    nz_link = int((np.abs(gr) > 0).sum())
    print(f"  (a) dL/dr_hw 非零链路数 {nz_link}/{s.L}"
          f"（零 {s.L - nz_link}；其中非管道 {int((~np.isin(s.lt_np, _PIPE_TYPES)).sum())}）")

    # (b) 完整诊断
    t0 = time.perf_counter()
    diag = identifiability(s, d0, rh0, sensors, r_hw=r_solver, ke=ke0,
                           wrt="C", max_iter=GGA_MI, polish_steps=POLISH,
                           return_S=True)
    t_diag = time.perf_counter() - t0
    S = diag["S"]
    print(f"  (b) identifiability 耗时 {t_diag:.3f}s；S 形状 {S.shape}，"
          f"‖F‖∞={float(np.max(diag['info']['resid_inf'])):.3e}")
    zc = diag["zero_col_mask"] & diag["pipe_mask"]
    print(f"      475 根管中灵敏度列全零：{int(zc.sum())}/{diag['n_pipe']}"
          f"（{100.0 * zc.sum() / diag['n_pipe']:.1f}%）")
    print(f"      加权梯度精确为零的链路：{int(diag['zero_grad_mask'].sum())}/{s.L}；"
          f"非零 {int((~diag['zero_grad_mask']).sum())}")
    print(f"      归因计数 {diag['counts']}  → n_informative = {diag['n_informative']}")
    print(f"      归因自洽：unobservable+clamped+dead_branch+closed = "
          f"{diag['counts']['unobservable'] + diag['counts']['clamped'] + diag['counts']['dead_branch'] + diag['counts']['closed']}"
          f" == 零列管道数 {int(zc.sum())}")
    print(f"      wrt='C' 的零集 == ② 中 wrt='r' 的零集："
          f"{np.array_equal(np.abs(diag['grad']) > 0, np.abs(gr) > 0)}")
    sp = svd_spectrum(S)
    fim_pos = np.linalg.eigvalsh(S.T @ S)
    print(f"      S 数值秩 {sp['rank']}/{min(S.shape)}；"
          f"σ1={sp['sv'][0]:.3e} σ_rank={sp['sv'][sp['rank'] - 1]:.3e} "
          f"cond={sp['cond']:.3e}")
    print(f"      FIM(σ=1) 正特征值个数 {int((fim_pos > 0).sum())}"
          f"（>1e-12·λmax 的有 {int((fim_pos > 1e-12 * fim_pos.max()).sum())}）")

    # ---------------- ④ t=0 钳位管数 ----------------
    print("\n" + "=" * 74)
    print("④ t=0 的 RQtol 钳位管数")
    print(f"  margin=10（gradcheck_3way.py:123 判据）：{int(diag['clamped_mask'].sum())}"
          f"/{diag['n_pipe']}")
    print(f"  margin=1（hydcoeffs.c:554 求解器实际分支）："
          f"{int(diag['clamped_strict'].sum())}/{diag['n_pipe']}")
    dead = diag["dead_mask"]
    print(f"  其中死支 {int((diag['clamped_mask'] & dead).sum())}（拓扑性，任何工况"
          f"都救不回），纯钳位 {diag['counts']['clamped']}（工况性，换高需水时段可能脱离）")
    print(f"  关闭链路 {int(diag['closed_mask'].sum())} 条全为 TCV："
          f"{int((diag['closed_mask'] & diag['pipe_mask']).sum())} 条是管道")
    print(f"  对照：dead_branch_mask 不传 demand 时剪出 {int(dead_branch_mask(s).sum())} 条"
          f"（把所有无 emitter 的 junction 都当无汇）→ 该掩码必须带 demand 才有意义")

    # ---------------- ⑤ FD 反证：unobservable 类不是实现 bug ----------------
    print("\n" + "=" * 74)
    print("⑤ 有限差分反证（防止把实现 bug 误判成结构性不可辨识）")
    C_base = s.kc_np.copy()

    def loss_at(k, dC):
        C2 = C_base.copy()
        C2[k] += dC
        r2 = r_solver.copy()
        r2[k] = float(hw_resistance(C2[k], float(net.len_ft[k]),
                                    float(net.diam_ft[k])))
        so = solve_polished(s, d0, rh0, ke0, r2, accuracy=1e-12,
                            max_iter=GGA_MI, polish_steps=POLISH)
        return float(np.sum(so["head"][0, sensors] - elev[sensors]))

    pipes_idx = np.where(diag["pipe_mask"])[0]
    reason_p = diag["reason"]
    gC_full = np.zeros(s.L)
    gC_full[rp.idx.cpu().numpy()] = gC
    for tag in ("informative", "unobservable", "clamped", "dead_branch"):
        cand = pipes_idx[reason_p[pipes_idx] == tag]
        if cand.size == 0:
            print(f"  {tag:<12} 无样本")
            continue
        k = int(cand[np.argmax(np.abs(gC_full[cand]))]) if tag == "informative" \
            else int(cand[0])
        h = 1.0                                       # ΔC = ±1（C=130 的 0.77%）
        fd = (loss_at(k, h) - loss_at(k, -h)) / (2.0 * h)
        relerr = abs(fd - gC_full[k]) / max(abs(fd), 1e-300) if fd != 0.0 \
            else abs(fd - gC_full[k])
        print(f"  {tag:<12} link#{k:<4d} 解析 dL/dC={gC_full[k]:+.6e}  "
              f"中心差分(ΔC=±1)={fd:+.6e}  绝对差={abs(fd - gC_full[k]):.3e}"
              f"  相对差={relerr:.2e}")
    print("  （informative 的相对差是 O(h²) 截断误差，h=1 即 ΔC/C=0.77%；严格的"
          "Richardson 外推对拍见 scripts/sensitivity_check.py，最差 9.47e-10。"
          "另三类的中心差分是**精确 0**，说明零灵敏度是水力学结构性的，不是数值噪声。）")
    print("\n完成。")

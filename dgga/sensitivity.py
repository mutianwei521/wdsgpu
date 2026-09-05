# -*- coding: utf-8 -*-
"""sensitivity.py - 传感器压力对管道摩阻系数的灵敏度矩阵 S = ∂p_sensor/∂C。

设计要点（"一次 splu 分解、多 RHS 求解"）
--------------------------------------------------------------------
dgga.autodiff.ImplicitGGASolve.backward 每被调用一次就重新装配并分解一次
雅可比（autodiff.py:628,633）。若对 m 个传感器逐个构造标量损失再 backward，
则同一个 J 被分解 m 次 - 分解是 O(nnz·fill) 的主导开销，而三角回代只是
O(nnz)。本模块在收敛态下：

  1. 用 _build_J（autodiff.py:383-422）装配一次 J；
  2. lu = splu(J) 一次；
  3. 把 m 个传感器的单位向量 e_i（H 分量位）拼成 [M, m] 的多列 RHS，
     一次 lu.solve(V, trans='T') 拿到全部伴随向量 Λ = J^{-T} V；
  4. ∂H_i/∂r_k = −λ_{i,l}[k]·dφ_k/dr_k（符号与 autodiff.py:653 完全一致），
     再乘链式因子 dr/dC = −Hexp·r/C 得 ∂H_i/∂C_k。

压力 p_i = H_i − El_i，El 与 C 无关 ⇒ ∂p_i/∂C ≡ ∂H_i/∂C。

不可辨识性的两个结构性来源（本模块直接暴露为 S 的零列）：
  * RQtol 钳位管（hydcoeffs.c:554-558 的低流量线性化使水损与 r 无关）：
    dφ/dr 解析恒为 0 ⇒ 该管整列为 0，任何传感器布置都无法校核；
  * 非管道链路（泵/阀）与关闭支：不存在 C，列恒 0。

本文件不修改 dgga/ 下任何既有模块，只做只读调用。
"""

from time import perf_counter as _perf

import numpy as np
from scipy.sparse.linalg import splu

from .autodiff import (_adj_cache, _build_J, _emitter_coeffs_np,
                       _link_coeffs_np, _valve_act_masks, solve_polished)
# C↔r_hw 的换算已统一到 calib.py（正向 hw_resistance + 链式因子 dr_dC），
# 本模块曾有的最小版实现（及其 TODO）已删除，此处 import 后原样再导出以保持
# 既有调用点（scripts/sensitivity_check.py）不变。
from .calib import _PIPE_TYPES, dr_dC  # noqa: F401

__all__ = ["sensitivity_matrix", "fisher_information", "svd_spectrum",
           "dr_dC", "identifiability_mask"]


# ======================================================================
# 灵敏度矩阵
# ======================================================================
def _sensor_rows(s, sensor_nodes):
    """节点索引 → J 中 H 分量的行号（相对 junction 编号）。定水头节点无 H 未知量。"""
    c = _adj_cache(s)
    idx = np.asarray(sensor_nodes, dtype=np.int64).ravel()
    rows = c["junc_row"][idx]
    bad = idx[rows < 0]
    if bad.size:
        raise ValueError(f"传感器节点 {bad.tolist()} 不是 junction（定水头节点的"
                         f"水头是输入而非未知量，∂p/∂C ≡ 0）")
    return idx, rows


def sensitivity_matrix(solver, demand, res_head, r_hw, sensor_nodes,
                       wrt="C", frames=None, ke=None,
                       accuracy=1e-12, max_iter=200, polish_steps=3,
                       speed=None, status=None, pump_h0=None, pump_r=None,
                       return_info=False):
    """S[T*m, L] = ∂p_sensor/∂θ，θ = C（摩阻系数）或 r（内部阻力 r_hw）。

    参数
    ----
    solver      : GGASolver（dense 或 epanet 模式均可）
    demand      : [N] 或 [B,N]（cfs）；res_head : [N] 或 [B,N]（ft）
    r_hw        : [L] 内部阻力；None = 用 solver 自带
    sensor_nodes: 传感器节点的全局索引（必须是 junction），长度 m
    wrt         : 'C' 或 'r'
    frames      : 取哪些批下标做工况；None = 全部。返回按帧-主序堆叠
                  [frame0 的 m 行; frame1 的 m 行; ...]，共 T*m 行
    return_info : True 时返回 (S, info)，info 含 resid_inf / 每帧 splu 耗时等

    实现：每帧只做一次 splu(J)，m 个传感器共用同一分解的多列回代。
    """
    s = solver
    if wrt not in ("C", "r"):
        raise ValueError("wrt 必须是 'C' 或 'r'")
    sensors, srows = _sensor_rows(s, sensor_nodes)
    m = sensors.size

    _t0 = _perf()
    sol = solve_polished(s, demand, res_head, ke=ke, r_hw=r_hw,
                         accuracy=accuracy, max_iter=max_iter,
                         polish_steps=polish_steps, speed=speed, status=status,
                         pump_h0=pump_h0, pump_r=pump_r)
    t_forward = _perf() - _t0
    B = sol["q"].shape[0]
    fr = list(range(B)) if frames is None else [int(b) for b in frames]

    L, Nj = s.L, s.Nj
    has_valve = bool((s.is_valve_np & ~s.is_tcv_np).any())
    chain = dr_dC(s, sol["r_hw"]) if wrt == "C" else None

    blocks = []
    info = dict(resid_inf=sol["resid_inf"], frames=fr, sensors=sensors,
                n_factorizations=0, n_rhs=0, t_forward=t_forward,
                t_assemble=0.0, t_factorize=0.0, t_solve=0.0)
    _t_adj0 = _perf()
    for b in fr:
        _ta = _perf()
        q = sol["q"][b]
        e_j = sol["e_j"][b]
        ke_j = sol["ke"][b, s.junc_nodes]
        _, hg, dphi_dr, _ = _link_coeffs_np(
            s, q, sol["r_hw"], speed=sol["speed"], status=sol["status"],
            h0p=sol["h0p"], rp=sol["rp"])
        _, hge, _dke = _emitter_coeffs_np(s, e_j, ke_j)
        em = ke_j > 0.0
        aprv = apsv = None
        if has_valve:
            aprv, apsv, _af = _valve_act_masks(s, sol["speed"], sol["status"])
        J, em_rows = _build_J(s, hg, hge, em, aprv, apsv)
        ne = em_rows.size
        M = L + ne + Nj
        # 多 RHS：第 i 列 = e_i（只在该传感器的 H 行上取 1）
        V = np.zeros((M, m), dtype=np.float64)
        V[L + ne + srows, np.arange(m)] = 1.0
        _tb = _perf()
        lu = splu(J)                                   # ← 每帧仅一次分解
        _tc = _perf()
        lam = lu.solve(V, trans="T")                   # [M, m] 一次多列回代
        _td = _perf()
        info["n_factorizations"] += 1
        info["n_rhs"] += m
        info["t_assemble"] += _tb - _ta
        info["t_factorize"] += _tc - _tb
        info["t_solve"] += _td - _tc
        lam_l = lam[:L, :]                             # [L, m]
        # ∂p_i/∂r_k = −λ_{i,l}[k]·dφ_k/dr_k（同 autodiff.py:653）
        Sb = -(lam_l * dphi_dr[:, None]).T             # [m, L]
        if chain is not None:
            Sb = Sb * chain[None, :]
        blocks.append(Sb)

    info["t_adjoint"] = _perf() - _t_adj0
    S = np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]
    return (S, info) if return_info else S


# ======================================================================
# 可辨识性诊断
# ======================================================================
def fisher_information(S, sigma):
    """FIM = Sᵀ Σ⁻¹ S，Σ = diag(sigma²)。

    sigma 可为标量（各行同方差）或长度 = S.shape[0] 的向量（逐行/逐传感器噪声）。
    返回 [L, L] 对称半正定阵；其零空间即"任何量测都无法区分"的参数方向。
    """
    S = np.asarray(S, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    if sig.ndim == 0:
        sig = np.full(S.shape[0], float(sig))
    if sig.shape != (S.shape[0],):
        raise ValueError(f"sigma 形状 {sig.shape} 与 S 的行数 {S.shape[0]} 不匹配")
    if np.any(sig <= 0):
        raise ValueError("sigma 必须为正")
    W = S / sig[:, None]
    return W.T @ W


def svd_spectrum(S, rtol=None, full=False):
    """S 的奇异谱与数值秩。

    返回 dict:
      sv        : 奇异值（降序，长度 min(T*m, L)）
      rank      : 数值秩（sv > rtol·sv[0]；rtol 默认 max(shape)·eps）
      cond      : sv[0]/sv[rank-1]（有效条件数）
      energy    : 累计能量占比 cumsum(sv²)/Σsv²
      Vt        : full=True 时给出右奇异向量（行 = 参数空间方向）
      null_dirs : full=True 时给出 rank 之后的右奇异向量（不可辨识方向）
    """
    S = np.asarray(S, dtype=np.float64)
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    if rtol is None:
        rtol = max(S.shape) * np.finfo(np.float64).eps
    thr = rtol * (sv[0] if sv.size and sv[0] > 0 else 0.0)
    rank = int(np.sum(sv > thr))
    tot = float(np.sum(sv ** 2))
    out = dict(sv=sv, rank=rank,
               cond=float(sv[0] / sv[rank - 1]) if rank > 0 else np.inf,
               energy=np.cumsum(sv ** 2) / tot if tot > 0 else np.zeros_like(sv))
    if full:
        out["Vt"] = Vt
        out["null_dirs"] = Vt[rank:]
        out["U"] = U
    return out


def identifiability_mask(S, atol=0.0):
    """逐管可辨识性布尔掩码：max_i |S[i,k]| > atol。

    atol=0 时给出"结构性不可校核"集合（RQtol 钳位管、非管道链路、关闭支 -
    这些列解析恒为 0，与数值噪声无关）。
    """
    S = np.asarray(S, dtype=np.float64)
    return np.max(np.abs(S), axis=0) > atol

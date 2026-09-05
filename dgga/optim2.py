# -*- coding: utf-8 -*-
"""dgga.optim2 - 一阶优化器增强（模块三）。**纯新增模块**：不改任何既有
文件，dgga 现有接口逐位不变；本模块的东西不被任何旧代码引用。

三件事：

一、GPU 批量多起点（multistart_heads）
    dense 前向对 r_hw 是"逐场景"的（solver._pipe_PY:2191 的 `self.r_hw * qa`
    对 [B,L] 的 r_hw 一样广播；baselines_calib.BatchObjective.loss_batch 早已
    这么喂无梯度的种群）。本函数把 **B 个起点 × F 个训练帧** 摊成 B·F 个场景
    一次喂给 ImplicitGGASolveGPU：前向是一次批量 dense GGA + 批量 Newton 精
    抛光，反向是一次批量约化伴随（复用前向终态分解，零新增 factorize）。
    B 个起点因此共享同一批 GPU kernel，而不是 B 次串行调用。
    **计账**：B 起点跑 N 步 = B·N 次模型调用（NFE）。批量只省墙钟，不省
    NFE - 本模块所有接口都按这个口径返回计数器，不允许把 B 摊成 1。

二、伴随 Schur 补对角预条件（schur_diag_precond / gn_diag_from_S）
    GGA 的雅可比 J = [[D, −A12],[A21, 0]]（autodiff._build_J 的无发射器形式），
    D = diag(hgrad)。消去链路块得节点 Schur 补
        Ā = A21 D^{-1} A12,  Ā_ii = Σ_{k∋i} 1/hgrad_k,  Ā_ij = −1/hgrad_k
    （即收敛态 GGA 矩阵本身，权 1/hgrad 的图 Laplacian）。
    对第 k 根管的阻力 r_k 求灵敏度：只有链路行 k 有源 ∂F_k/∂r_k = dr_k，
        Ā·δH = (dr_k/D_kk)·a_k,  a_k = A21 e_k = e_{n2(k)} − e_{n1(k)}
    对 H-W 管 dr_k = sgn(q)|q|^{Hexp}、D_kk = Hexp·r_k|q|^{Hexp−1}，
    再乘 C 空间链式因子 ∂r/∂C = −Hexp·r/C，得到**干净的解析式**
        ∂H/∂C_k = −(q_k / C_k) · Ā^{-1}(e_{n2} − e_{n1})            (★)
    高斯-牛顿对角（= 训练损失 Hessian 的 GN 部分的对角）为
        [S^T S]_kk / n = (1/n)·Σ_{帧,传感器 s} (∂p_s/∂C_k)²
    (★) 里唯一贵的是 Ā^{-1}。两种取法，本模块都给：
      · schur_diag_precond（**零新增线性解**）：用 Jacobi 近似 Ā^{-1}≈diag(Ā)^{-1}，
        即把响应就地化到管两端节点，
            M_k = (1/F)Σ_f (q_{k,f}/C_k)²·[ m1/Ā_{n1n1}² + m2/Ā_{n2n2}² ]
        m1/m2 = 端点是否 junction（定水头端响应恒 0）。只用收敛态的 q 和
        hgrad，**不解任何线性方程组**，成本 ~0。
      · gn_diag_from_S（**精确**）：diag(SᵀS)/n，S 由 sensitivity_matrix 给
        （1 次批前向 + 1 次伴随雅可比，按本项目口径 nfe+=1、nbwd+=1）。
        用来量 Jacobi 近似丢了多少 - 这是可发表的对照，不是装饰。
    两者都加 Tikhonov 正则项的精确对角 w_reg = λ/(REG_SCALE·f) 与相对地板
    floor_rel·max(M)（City D 自由集 432 而良态维数 ~41，不设地板时零空间
    方向的 M_k→0 会让 M^{-1} 把步长炸到箱外 - 地板是必需品，且如实报告）。

三、对角预条件 L-BFGS（lbfgs_precond）
    两回路递归里把初始逆 Hessian 由 γI 换成 γM^{-1}：
        q ← g
        for i = k−1..k−m:  α_i = ρ_i s_iᵀq;  q ← q − α_i y_i
        z ← γ·M^{-1}q                                   ← 预条件子在这里进
        for i = k−m..k−1:  β = ρ_i y_iᵀz;  z ← z + s_i(α_i − β)
        d = −z,   ρ_i = 1/(y_iᵀs_i),   γ = (s_{k−1}ᵀy_{k−1})/(y_{k−1}ᵀM^{-1}y_{k−1})
    （M=I 时逐字退化为 torch.optim.LBFGS 的方向；γ 是常用 s·y/y·y 的 M 加权
    推广。等价说法：对 u = M^{1/2}x 跑普通 L-BFGS。这里直接写两回路，
    因为要**逐次函数求值精确计账** - 2000 次调用的对比不许把线搜索的试探
    步漏掉。）线搜索直接复用 torch.optim.lbfgs._strong_wolfe（与 G-C1 的
    L-BFGS 同一实现，避免"换了线搜索所以赢了"的混淆）。
"""

import numpy as np
import torch

if __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__))))
    __package__ = "dgga"
    import dgga as _dgga             # noqa: F401

from .autodiff import (ImplicitGGASolve, ImplicitGGASolveGPU,   # noqa: F401
                       _adj_cache, _link_coeffs_np)

__all__ = ["multistart_heads", "schur_nodal_diag", "schur_diag_precond",
           "gn_diag_from_S", "lbfgs_precond", "lhs_starts"]


# ======================================================================
# 一、GPU 批量多起点
# ======================================================================
def multistart_heads(solver, r_starts, d_frames, rh_frames, ke=None,
                     accuracy=1e-6, max_iter=30, polish_steps=4,
                     adjoint="gpu"):
    """B 起点 × F 帧一次前向（+可微伴随）。

    r_starts [B,L]（可含 grad）；d_frames/rh_frames [F,N]；
    返回 (head [B,F,N], flow [B,F,L])。
    accuracy 是 **GGA 半迭代的停机判据**，不是最终精度：city_d 的 GGA relerr
    有 ~1.5e-7 的平台（baselines_calib.BASE_MI_NET 注释里量过），
    ImplicitGGASolveGPU 的"未收敛即 raise"守卫按 accuracy 判，所以这里默认
    1e-6，随后的 Newton 精抛光（polish_steps）才是把 ‖F‖ 压下去的那一步 -
    与 CPU 侧 solve_polished(accuracy=1e-12, max_iter=20, polish=4) 的实测
    max|ΔH| 见 scripts/optimizer_v2.py --stage verify。

    adjoint="cpu" 回退（无 CUDA 时用；r_starts 必须 B=1，CPU 伴随的 r_hw 是
    批共享的 [L]）。
    """
    F = int(d_frames.shape[0])
    B = int(r_starts.shape[0])
    dev, dt = solver.device, solver.dtype
    d_t = torch.as_tensor(d_frames, dtype=dt, device=dev)
    rh_t = torch.as_tensor(rh_frames, dtype=dt, device=dev)
    ke_t = solver.node_ke_default.to(dev) if ke is None \
        else torch.as_tensor(ke, dtype=dt, device=dev)
    if adjoint == "cpu":
        if B != 1:
            raise ValueError("CPU 伴随的 r_hw 是批共享的，多起点请用 gpu 伴随")
        head, flow, _ = ImplicitGGASolve.apply(d_t, rh_t, ke_t, r_starts[0],
                                               solver, 1e-12, 20, polish_steps)
        return head.reshape(1, F, -1), flow.reshape(1, F, -1)
    rr = r_starts.repeat_interleave(F, dim=0)          # [B*F, L]
    dd = d_t.repeat(B, 1)
    rhh = rh_t.repeat(B, 1)
    head, flow, _e = ImplicitGGASolveGPU.apply(
        dd, rhh, ke_t, rr, solver, accuracy, max_iter, None,
        "dense", "dense", int(polish_steps))
    return head.reshape(B, F, -1), flow.reshape(B, F, -1)


def lhs_starts(n, lo, hi, dim, seed):
    """拉丁超立方起点 [n,dim]（与 calibrate.lhs_inits 同构，独立种子）。"""
    rng = np.random.default_rng(seed)
    u = np.empty((n, dim))
    for j in range(dim):
        u[:, j] = (rng.permutation(n) + rng.uniform(size=n)) / n
    return lo + (hi - lo) * u


# ======================================================================
# 二、Schur 补对角
# ======================================================================
def schur_nodal_diag(solver, q, r_hw):
    """收敛态 GGA 矩阵（Schur 补 Ā = A21 D^{-1} A12）的对角 [Nj]。

    q [L] 收敛流量、r_hw [L]。hgrad 取 autodiff._link_coeffs_np（与伴随、与
    EPANET hydcoeffs.c 的钳位语义同源），Ā_ii = Σ_{k∋i} 1/hgrad_k。
    """
    s = solver
    c = _adj_cache(s)
    _hl, hg, dr, _pd = _link_coeffs_np(s, np.asarray(q, dtype=np.float64),
                                       np.asarray(r_hw, dtype=np.float64))
    w = 1.0 / hg                                       # 支导（GGA 的 1/hgrad）
    Adiag = np.zeros(s.Nj, dtype=np.float64)
    np.add.at(Adiag, c["j1"][c["m1"]], w[c["m1"]])
    np.add.at(Adiag, c["j2"][c["m2"]], w[c["m2"]])
    return Adiag, hg, dr


def schur_diag_precond(solver, q_frames, r_hw, C_free, free_idx,
                       w_reg=0.0, floor_rel=1e-6):
    """(★) + Jacobi(Ā) 的 C 空间对角度量 M [n_free]（零新增线性解）。

    q_frames [F,L] 收敛流量；r_hw [L]（各帧共用）；C_free [n_free]。
    M_k = (1/F)Σ_f (q_{k,f}/C_k)²·[ m1/Ā_{n1n1}² + m2/Ā_{n2n2}² ] + w_reg
    再套相对地板 floor_rel·max(M)。返回 (M, info)。
    """
    s = solver
    c = _adj_cache(s)
    q_frames = np.atleast_2d(np.asarray(q_frames, dtype=np.float64))
    F = q_frames.shape[0]
    f_idx = np.asarray(free_idx)
    m1 = c["m1"][f_idx]
    m2 = c["m2"][f_idx]
    j1 = c["j1"][f_idx]
    j2 = c["j2"][f_idx]
    r_f = np.asarray(r_hw, dtype=np.float64)[f_idx]
    acc = np.zeros(len(f_idx), dtype=np.float64)
    for b in range(F):
        Adiag, hg, dr = schur_nodal_diag(s, q_frames[b], r_hw)
        inv1 = np.where(m1, 1.0 / np.maximum(Adiag[np.where(m1, j1, 0)],
                                             1e-300), 0.0)
        inv2 = np.where(m2, 1.0 / np.maximum(Adiag[np.where(m2, j2, 0)],
                                             1e-300), 0.0)
        # 前置因子 (∂r/∂C)·(dr_k/D_kk)：非钳位 H-W 管化简为 −q_k/C_k；
        # RQtol 钳位支 dr=0 → 恒 0（真灵敏度确实为 0，见 calib.clamped_mask），
        # 所以这里用未化简式，钳位支自动落地板而不是被 q/C 高估。
        prefac = (s.hexp * r_f / C_free) * (dr[f_idx] / hg[f_idx])
        acc += prefac ** 2 * (inv1 ** 2 + inv2 ** 2)
    M = acc / F + w_reg
    mx = float(M.max())
    fl = floor_rel * mx
    n_floored = int((M < fl).sum())
    M = np.maximum(M, fl)
    return M, dict(kind="schur_jacobi", max=mx, min=float(M.min()),
                   floor_rel=floor_rel, n_floored=n_floored,
                   cond=float(M.max() / M.min()), w_reg=w_reg)


def gn_diag_from_S(S, n_obs, w_reg=0.0, floor_rel=1e-6):
    """精确高斯-牛顿对角 diag(SᵀS)/n_obs + w_reg（S = [n_obs, n_free]）。"""
    M = np.einsum("ij,ij->j", S, S) / float(n_obs) + w_reg
    mx = float(M.max())
    fl = floor_rel * mx
    n_floored = int((M < fl).sum())
    M = np.maximum(M, fl)
    return M, dict(kind="gn_exact", max=mx, min=float(M.min()),
                   floor_rel=floor_rel, n_floored=n_floored,
                   cond=float(M.max() / M.min()), w_reg=w_reg)


# ======================================================================
# 三、对角预条件 L-BFGS
# ======================================================================
def _strong_wolfe():
    from torch.optim.lbfgs import _strong_wolfe as sw
    return sw


def lbfgs_precond(fun, x0, minv=None, max_iter=100, max_eval=None,
                  history=30, tol_grad=1e-13, tol_change=1e-18,
                  c1=1e-4, c2=0.9, max_ls=25, callback=None, f0=None,
                  g0=None):
    """对角预条件 L-BFGS（strong-Wolfe 线搜索，逐次函数求值精确计数）。

    fun(x) -> (f, g)（numpy float64，x [n]）；minv = M^{-1} 的对角 [n]
    （None = 单位阵，退化为普通 L-BFGS）。返回 dict(x, f, n_eval, n_iter,
    stop, f_hist)。**n_eval 含线搜索的每一次试探步**。

    停止：达到 max_iter / max_eval，或 ‖g‖∞ ≤ tol_grad，或 ‖d·t‖∞ ≤ tol_change，
    或线搜索无进展。
    """
    sw = _strong_wolfe()
    x = np.asarray(x0, dtype=np.float64).copy()
    n = x.size
    minv = None if minv is None else np.asarray(minv, dtype=np.float64)
    max_eval = max_eval if max_eval is not None else max_iter * 5
    cnt = {"n": 0}

    def ev(xx):
        f, g = fun(np.asarray(xx, dtype=np.float64))
        cnt["n"] += 1
        return float(f), np.asarray(g, dtype=np.float64).copy()

    if f0 is None or g0 is None:
        f, g = ev(x)
    else:                        # 复用外部已算好的同点求值（不重复计 NFE）
        f, g = float(f0), np.asarray(g0, dtype=np.float64).copy()
    f_hist = [f]
    S, Y, RHO = [], [], []
    stop = "max_iter"
    n_iter = 0
    d = None
    t = None
    for it in range(max_iter):
        if np.max(np.abs(g)) <= tol_grad:
            stop = "tol_grad"
            break
        if cnt["n"] + max_ls > max_eval:     # 线搜索最坏情形也不许超预算
            stop = "max_eval"
            break
        # ---- 两回路递归（H0 = γ·M^{-1}） ----
        qv = g.copy()
        m = len(S)
        alpha = [0.0] * m
        for i in range(m - 1, -1, -1):
            alpha[i] = RHO[i] * float(S[i] @ qv)
            qv -= alpha[i] * Y[i]
        if m:
            yl, sl = Y[-1], S[-1]
            My = yl * minv if minv is not None else yl
            denom = float(yl @ My)
            gamma = float(sl @ yl) / denom if denom > 0 else 1.0
        else:
            gamma = 1.0
        z = gamma * (qv * minv if minv is not None else qv)
        for i in range(m):
            beta = RHO[i] * float(Y[i] @ z)
            z += S[i] * (alpha[i] - beta)
        d = -z
        gtd = float(g @ d)
        if gtd > -1e-300:                              # 非下降方向 → 重启
            d = -(g * minv if minv is not None else g)
            S, Y, RHO = [], [], []
            gtd = float(g @ d)
            if gtd > -1e-300:
                stop = "no_descent"
                break
        # 首步步长：无预条件时同 torch.optim.LBFGS 的 min(1, 1/‖g‖₁)；
        # 有预条件时 M^{-1}g 已带尺度，取 1（拟牛顿的自然步长）
        t0 = 1.0 if (it > 0 or minv is not None or m) else \
            min(1.0, 1.0 / max(float(np.abs(g).sum()), 1e-300))

        xt = torch.from_numpy(x)
        dt_ = torch.from_numpy(d)
        gt = torch.from_numpy(g)

        def obj(xx, tt, dd):
            fn, gn = ev((xx + tt * dd).numpy())
            return fn, torch.from_numpy(gn)

        f_new, g_new, t, ls_evals = sw(obj, xt, t0, dt_, f, gt, gtd,
                                       c1=c1, c2=c2,
                                       tolerance_change=1e-12, max_ls=max_ls)
        # torch.optim.lbfgs._strong_wolfe 把步长 t 作为 **0 维 tensor** 返回
        # （f_new 是 float、g_new 是 tensor）。numpy 侧 x + t*d 会走 ndarray
        # 与 tensor 的加法回退路径并抛 TypeError，所以这里逐个落回标量/ndarray。
        f_new = float(f_new)
        g_new = np.asarray(g_new.detach().cpu().numpy(), dtype=np.float64)
        t = float(t)
        x_new = x + t * d
        s_vec = x_new - x
        y_vec = g_new - g
        ys = float(y_vec @ s_vec)
        if ys > 1e-10 * float(np.linalg.norm(s_vec) * np.linalg.norm(y_vec)
                              + 1e-300):
            if len(S) == history:
                S.pop(0)
                Y.pop(0)
                RHO.pop(0)
            S.append(s_vec)
            Y.append(y_vec)
            RHO.append(1.0 / ys)
        moved = float(np.max(np.abs(s_vec)))
        x, f, g = x_new, float(f_new), g_new
        f_hist.append(f)
        n_iter += 1
        if callback is not None:
            callback(it, x, f, cnt["n"])
        if moved <= tol_change:
            stop = "tol_change"
            break
        if cnt["n"] >= max_eval:
            stop = "max_eval"
            break
    return dict(x=x, f=f, n_eval=cnt["n"], n_iter=n_iter, stop=stop,
                f_hist=f_hist)

# -*- coding: utf-8 -*-
"""placement.py - 压力传感器最优布点：贝叶斯 D-最优贪心 + 基线方法。

一、全候选多帧灵敏度 full_sensitivity()
    对 T 个独立稳态帧、全部 Nj 个 junction 候选传感器，构造
    S_full[T, Nj, P]（P = H-W 管道数，θ = C 空间）。内部复用
    sensitivity.sensitivity_matrix：每帧只做一次 splu(J)，Nj 个候选共用
    同一分解的多列回代（绝不逐传感器循环 backward）。

二、贝叶斯 D-最优贪心 bayes_dopt_greedy()
    目标  f(S) = logdet(σ_prior²·M(S)) = logdet(I + (σ_prior²/σ_noise²)·Σ_S Σ_t s s^T)
          M(S) = M0 + Σ_{i∈S} A_i，M0 = (1/σ_prior²)I，A_i = (1/σ_noise²)Σ_t s_{i,t}s_{i,t}^T
    正则来源：M0 是高斯先验 C ~ N(C0=130, σ_prior²I) 的**精度矩阵**，不是随手加的
    ε - f(S) 就是贝叶斯线性模型下先验→后验的信息增益（log 体积压缩比），
    σ_prior 有物理单位（C 的先验标准差），"正则系数怎么选"由先验知识回答：
    C 合理范围约 [70,160] ⇒ 缺省 σ_prior=15；压力计精度 ⇒ 缺省 σ_noise=0.1 ft。
    注意恒等式 f 只依赖比值 ρ = σ_prior/σ_noise（σ_prior²M = I + ρ²Σ_S UU^T），
    因此 σ_prior×4 与 σ_noise÷4 给出**逐位相同**的贪心序列 - 敏感性检查会
    如实呈现这一点。

    实现：
    * 边际增益用矩阵行列式引理  Δ_i = logdet(I_T + (1/σ_noise²)·U_i^T M^{-1} U_i)，
      U_i = [P, T]；维护 M 的 Cholesky，每步重新分解（P=475 很小，数值稳定优先）。
    * lazy greedy（Minoux/CELF）：批量首评 + 堆上惰性重评；
    * 逐实例最优性证书：对任意已访问状态 S_j 有
      f(OPT_k) ≤ f(S_j) + Σ(S_j 处前 k 大边际增益)（单调子模标准上界，
      对任意参考集都成立），取所有快照（含空集首评）的最小值作上界，
      报告 ratio_k = f(S_k)/上界。

三、基线（都在同一 S_full 上评估）
    random_orders / degree_order / norm_order / aopt_greedy / spectral_order。
    A-最优贪心非子模，无 (1-1/e) 保证（如实标注）；spectral_order 是
    Zhou et al. 2024 (Water Research) 思路的**简化代理**（图拉普拉斯低频
    特征子空间上的行选 D-最优贪心），非原文复刻。

四、评估 eval_subset()：f、数值秩、零列数、可辨识子空间 CRLB 迹、
    贝叶斯后验方差迹、λ_min(M) - 全部由一次 SVD 派生
    （M 与 FIM 同特征向量：λ_i(M) = 1/σ_prior² + sv_i²/σ_noise²）。

五、增设模式 bayes_dopt_augment()（新增，不改动上面任何函数的行为）
    水司的问题不是"从零重选 k 个传感器"，而是"现有 S0 一个不动，再加 k 个装哪"。
    增设 = S0 固定为已选集，只在余下候选上贪心加点。两种目标：
      objective="dopt"   贝叶斯 D-最优：f(S0∪S) − f(S0)，从 M(S0) 起 lazy 贪心，
                          证书按同一子模上界改成相对 S0 的增量；
      objective="cover"  可辨识管数：直接最大化"跨过可辨识阈值"的管数
                          （判据与 calib.identifiability 的普查完全一致：
                          列 max|S| > atol，atol=0 即解析恒零），平手用 D-最优
                          增益裁决；覆盖饱和后（再加谁都救不回新管）退回 D-最优。
    单调性是增设的定义，函数内逐步断言：可辨识集只增不减、贝叶斯后验方差
    diag(M^{-1}) 逐管不增（M 只加 PSD 项，Loewner 序保证）；违反即抛 RuntimeError
    （那是实现 bug，不是设计结果）。
    recovery_report() 用普查同判据对指定管集（如 149 根"传感不足"管）逐管判定
    "找回/未找回"，并给出 S0 下可辨识、增设后反而丢失的管数（增设下恒为 0，
    作为断言）。

六、相干驱动增设 coherence_augment()（新增，不改动上面任何函数的行为）
    上面的增设目标（D-最优 / 覆盖）是为**粗糙度可辨识性**设计的：它们看的是
    ∂p/∂C 的 Fisher 信息。漏损搜索的成败却由**漏损签名字典的互相干**决定：
    D[t, i, j] = 候选漏点 j 单位漏损系数在传感位置 i、帧 t 的压力响应，传感集 S
    下字典 A_S = D[:, S, :] 展平成 [T|S|, NC]，列归一化后 μ_ij(S) = |⟨a_i, a_j⟩|/
    (‖a_i‖‖a_j‖)。coherence_augment 让 S0 固定不动，在允许的候选位置池上贪心加
    k 个，压低 μ(S0∪S) 的整体水平。目标（全部真值无关：只用字典、S0 与候选池，
    漏点身份不进入任何一步）：
      objective="logdet2"（缺省）  J(S) = Σ_{i<j} −log(1 − μ_ij(S)² + eps)
              = −Σ_{i<j} logdet(Ĝ_ij)，Ĝ_ij 为列归一化字典的 2×2 Gram 子块：
              每一对单漏假设的"可区分体积"的对数之和；μ→1 的对贡献 −log(1−μ²)
              → ∞（μ=0.9999 记 8.5，0.99 记 3.9，0.5 记 0.29），所以它盯住的是
              近共线的对，同时对所有 NC(NC−1)/2 对求和 - 单独最小化 max μ 会在
              "任何布点都压不下去的拓扑双胞胎对"上停摆（所有候选并列），
              这一项则仍能在其余对上下降。
      objective="max"        字典序最小化 (max_{i<j} μ_ij, J)；
      objective="quantile"   字典序最小化 (q-分位 μ, J)。
    实现：Gram 增量 G(S∪{v}) = G(S) + Σ_t D[t,v,:]ᵀD[t,v,:]，每步对全部候选位置
    一次向量化评估（O(m·NC²)），无需重建字典。每步记录 max / 中位 / 分位 /
    >0.999 对数 / 正交对数 / J，供可读输出（零编号）。

本模块只做只读调用，不修改 dgga/ 既有模块。全程 float64。
"""

import heapq
import warnings
from time import perf_counter as _perf

import numpy as np
from scipy.linalg import solve_triangular

from .calib import _PIPE_TYPES
from .sensitivity import sensitivity_matrix

__all__ = ["pipe_param_idx", "full_sensitivity", "bayes_dopt_greedy",
           "submodularity_check", "random_orders", "degree_order",
           "norm_order", "aopt_greedy", "spectral_order", "eval_subset",
           "zero_col_curve", "bayes_dopt_augment", "recovery_report",
           "posterior_std", "coherence_stats_rows", "coherence_augment"]


# ======================================================================
# 一、全候选多帧灵敏度
# ======================================================================
def pipe_param_idx(solver):
    """参数空间 = H-W 管道（PIPE/CVPIPE 且 C>0）的链路全局下标 [P]。"""
    return np.where(np.isin(solver.lt_np, _PIPE_TYPES)
                    & (solver.kc_np > 0.0))[0]


def full_sensitivity(solver, net, t_secs, sensor_nodes=None, r_hw=None,
                     wrt="C", accuracy=1e-12, max_iter=200, polish_steps=3):
    """S_full[T, m, P] = 逐帧、全候选传感器、H-W 管道 C 空间的灵敏度。

    参数
    ----
    solver       : GGASolver（须先于 net.demand_cfs_at 构造 - inp_path 就地修正
                   net.dem_base_cfs）
    net          : Net（提供 demand_cfs_at / reservoir_head_ft_at）
    t_secs       : 各帧时刻（秒）；独立稳态快照，批式一次求解
    sensor_nodes : 候选传感器节点；None = 全部 junction
    r_hw         : [L] 内部阻力；None = solver 自带。wrt='r' 时返回 ∂p/∂r
                   的管道列（供在别的 C 真值处重算 FIM 用）

    返回 (S_full, meta)：meta 含 pipe_idx / sensors / t_secs / info（计时）/
    t_total。实现：每帧 1 次 splu + m 列回代（见 sensitivity.py）。
    """
    s = solver
    sensors = (np.asarray(s.junc_nodes) if sensor_nodes is None
               else np.asarray(sensor_nodes, dtype=np.int64))
    t_secs = [int(t) for t in t_secs]
    T, m = len(t_secs), sensors.size

    D = np.stack([net.demand_cfs_at(t) for t in t_secs])          # [T, N]
    RH = np.stack([np.nan_to_num(net.reservoir_head_ft_at(t))
                   for t in t_secs])                              # [T, N]
    ke0 = np.zeros(net.N, dtype=np.float64)

    _t0 = _perf()
    S, info = sensitivity_matrix(s, D, RH, r_hw, sensors, wrt=wrt, ke=ke0,
                                 accuracy=accuracy, max_iter=max_iter,
                                 polish_steps=polish_steps, return_info=True)
    t_total = _perf() - _t0
    pidx = pipe_param_idx(s)
    S_full = S.reshape(T, m, s.L)[:, :, pidx].copy()              # [T, m, P]
    meta = dict(pipe_idx=pidx, sensors=sensors, t_secs=t_secs, info=info,
                t_total=t_total, demand=D, res_head=RH,
                resid_inf=float(np.max(info["resid_inf"])))
    return S_full, meta


def _as_U(S_full):
    """[T, m, P] → 候选块 U[m, P, T]（U_i = S_full[:, i, :].T）。"""
    return np.ascontiguousarray(np.transpose(S_full, (1, 2, 0)))


# ======================================================================
# 二、贝叶斯 D-最优贪心（lazy greedy + 证书）
# ======================================================================
def _chol_spd(M):
    """cholesky(M)，失败时给出可诊断的 ValueError（而非裸 LinAlgError）。

    M = I/σ_p² + Σ UU^T/σ_n² 在精确算术下恒正定；float64 下失败只会因
    cond(M) ~ 1 + ρ²·σ_max(S)² 超出机器精度（σ_prior 极大 ⇒ 正则趋零、
    FIM 奇异）。此时任何基于 M 的 logdet 数字都不可信，如实拒绝。
    """
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        raise ValueError(
            "cholesky(M) 数值非正定：ρ=σ_prior/σ_noise 过大使 cond(M) 超出 "
            "float64（正则趋零、FIM 奇异）。f 只依赖比值 ρ，请降低 ρ；"
            "固定子集的 f 可改用 eval_subset（SVD+log1p 路径，任意 ρ 稳定）。"
        ) from None


def _warn_if_ill_conditioned(U, sp2, sn2):
    """ρ²·‖S‖_F² 给出 cond(M) 的上界估计；超过 1e12 时增量 logdet 的
    相对精度按 eps·cond 退化（不再是 ~1e-13 量级）。如实警告而非静默。"""
    cond_est = 1.0 + (sp2 / sn2) * float(np.einsum("ipt,ipt->", U, U))
    if cond_est > 1e12:
        warnings.warn(
            f"cond(M) 上界估计 ~{cond_est:.1e} > 1e12：σ_prior/σ_noise 比值"
            f"过大，增量 logdet 的相对精度按 eps·cond 退化（f 可能只有 "
            f"~{min(cond_est * 2.3e-16, 1.0):.0e} 的相对精度）。",
            RuntimeWarning, stacklevel=3)
    return cond_est


def _batch_gains(L_chol, U, idx, sn2):
    """在当前 Cholesky(M)=L_chol 下，批量算 idx 中各候选的边际增益。

    Δ_i = logdet(I_T + (1/σ_n²)·U_i^T M^{-1} U_i)
        = logdet(I_T + (1/σ_n²)·X_i^T X_i)，X_i = L^{-1} U_i（一次批式三角回代）。
    """
    P, T = U.shape[1], U.shape[2]
    n = len(idx)
    Uf = U[idx].transpose(1, 0, 2).reshape(P, n * T)              # [P, n*T]
    X = solve_triangular(L_chol, Uf, lower=True, check_finite=False)
    Xr = X.reshape(P, n, T)
    G = np.einsum("pnt,pnu->ntu", Xr, Xr) / sn2                   # [n, T, T]
    B = G + np.eye(T)[None, :, :]
    sign, ld = np.linalg.slogdet(B)
    if not np.all(sign > 0):
        raise FloatingPointError("logdet 非正定（不应发生：I+PSD）")
    return ld


def bayes_dopt_greedy(S_full, kmax, sigma_prior=15.0, sigma_noise=0.1,
                      cert_ks=(), verbose=False):
    """贝叶斯 D-最优 lazy 贪心。返回 dict：
       order[kmax] 选中候选（S_full 第二维下标，选择顺序）
       gains[kmax] 每步边际增益；f_curve[kmax] 累积 f(S_k)
       cert{k: dict(f, upper, ratio)} 逐实例最优性证书
       n_evals 惰性重评次数（含首评）；t_total 秒
    """
    U = _as_U(S_full)
    m, P, T = U.shape
    kmax = min(int(kmax), m)
    sp2, sn2 = float(sigma_prior) ** 2, float(sigma_noise) ** 2
    cert_ks = set(int(k) for k in cert_ks)

    _t0 = _perf()
    _warn_if_ill_conditioned(U, sp2, sn2)
    M = np.eye(P) / sp2
    L = _chol_spd(M)
    all_idx = np.arange(m)
    g0 = _batch_gains(L, U, all_idx, sn2)
    n_evals = m
    # 证书快照：(f(S_j), S_j 处按降序排好的边际增益)。对任意 S_j 都有
    # f(OPT_k) ≤ f(S_j) + Σ(前 k 大增益)，上界取全部快照的最小值。
    snapshots = [(0.0, np.sort(g0)[::-1])]
    # 堆元素 (−gain, stamp, i)：stamp == 当前步号 ⇒ gain 新鲜可直接选
    heap = [(-g0[i], 0, int(i)) for i in range(m)]
    heapq.heapify(heap)

    order, gains, f_curve, cert = [], [], [], {}
    f = 0.0
    for k in range(1, kmax + 1):
        while True:
            ng, stamp, i = heapq.heappop(heap)
            if stamp == k - 1:                    # 本步已重评过 → 精确最优
                break
            g = float(_batch_gains(L, U, np.array([i]), sn2)[0])
            n_evals += 1
            heapq.heappush(heap, (-g, k - 1, i))
        gain = -ng
        order.append(int(i))
        gains.append(float(gain))
        f += float(gain)
        f_curve.append(f)
        M += (U[i] @ U[i].T) / sn2
        L = _chol_spd(M)

        if k in cert_ks:
            rem = np.array([j for j in range(m) if j not in set(order)])
            if rem.size:
                gr = _batch_gains(L, U, rem, sn2)
                n_evals += rem.size
                snapshots.append((f, np.sort(gr)[::-1]))
                # 顺手刷新堆（全部 stamp=k，下一步 O(1) 选中）
                heap = [(-gr[j], k, int(rem[j])) for j in range(rem.size)]
                heapq.heapify(heap)
            upper = min(fj + float(np.sum(gj[:k]))
                        for fj, gj in snapshots)
            upper = max(upper, f)                 # 上界不得低于已实现值
            # f 的独立复算（防增量漂移）：logdet(σ_p²M)
            sgn, ld = np.linalg.slogdet(sp2 * M)
            cert[k] = dict(f=f, f_recomputed=float(ld), upper=upper,
                           ratio=f / upper if upper > 0 else 1.0)
            if verbose:
                print(f"  [cert] k={k:3d}  f={f:.4f}  上界={upper:.4f}  "
                      f"ratio={cert[k]['ratio']:.4f}  |f-复算|="
                      f"{abs(f - ld):.2e}")
    return dict(order=order, gains=gains, f_curve=f_curve, cert=cert,
                n_evals=int(n_evals), t_total=_perf() - _t0,
                sigma_prior=float(sigma_prior), sigma_noise=float(sigma_noise))


def submodularity_check(S_full, sigma_prior=15.0, sigma_noise=0.1,
                        n_samples=200, seed=0, max_small=12, max_extra=12,
                        tol=1e-9):
    """经验抽查边际增益递减：随机 (T1 ⊂ T2, i∉T2)，验证 Δ(i|T1) ≥ Δ(i|T2) − tol。

    logdet(带 PSD 增量) 的单调子模性是定理；本检查防的是实现 bug 与数值出格。
    返回 dict(n_samples, n_violations, worst_margin)（worst_margin =
    min[Δ(i|T1) − Δ(i|T2)]，≥ −tol 即通过）。
    """
    U = _as_U(S_full)
    m, P, T = U.shape
    sp2, sn2 = float(sigma_prior) ** 2, float(sigma_noise) ** 2
    rng = np.random.default_rng(seed)
    eyeP = np.eye(P)

    def gain_at(subset, i):
        M = eyeP / sp2
        if len(subset):
            Uf = U[np.asarray(subset)].transpose(1, 0, 2).reshape(P, -1)
            M = M + (Uf @ Uf.T) / sn2
        L = _chol_spd(M)
        return float(_batch_gains(L, U, np.array([i]), sn2)[0])

    n_vio, worst = 0, np.inf
    for _ in range(int(n_samples)):
        n1 = int(rng.integers(0, max_small + 1))
        n2 = n1 + int(rng.integers(1, max_extra + 1))
        pick = rng.choice(m, size=min(n2 + 1, m), replace=False)
        T2, i = list(pick[:n2]), int(pick[-1])
        T1 = list(rng.choice(T2, size=n1, replace=False)) if n1 else []
        margin = gain_at(T1, i) - gain_at(T2, i)
        worst = min(worst, margin)
        if margin < -tol:
            n_vio += 1
    return dict(n_samples=int(n_samples), n_violations=int(n_vio),
                worst_margin=float(worst), tol=float(tol))


# ======================================================================
# 三、基线
# ======================================================================
def random_orders(m, n_seeds=30, seed0=0):
    """30 个种子的随机排列（前缀即随机布点）。返回 [n_seeds, m]。"""
    return np.stack([np.random.default_rng(seed0 + s).permutation(m)
                     for s in range(n_seeds)])


def degree_order(net, junc_nodes):
    """度中心性降序（自建图，net.link_n1/n2 全链路计度；平手按节点号）。"""
    deg = np.zeros(net.N, dtype=np.int64)
    np.add.at(deg, np.asarray(net.link_n1, dtype=np.int64), 1)
    np.add.at(deg, np.asarray(net.link_n2, dtype=np.int64), 1)
    dj = deg[np.asarray(junc_nodes)]
    return np.argsort(-dj, kind="stable")


def norm_order(S_full):
    """最大灵敏度范数排序（Bush & Uber 1998 风格）：候选行 Frobenius 范数降序。"""
    U = _as_U(S_full)
    rn = np.sqrt(np.einsum("ipt,ipt->i", U, U))
    return np.argsort(-rn, kind="stable")


def aopt_greedy(S_full, kmax, sigma_prior=15.0, sigma_noise=0.1):
    """A-最优贪心：每步选使 trace(M^{-1}) 下降最多的候选。

    注意：trace(M^{-1}) 的减量**非子模**，本法无 (1-1/e) 近似保证（如实标注）；
    这里做普通贪心（无 lazy），每步批量精确评估全部剩余候选。
    Woodbury：trace 减量 = trace[(σ_n²I + U^T M^{-1}U)^{-1}·(U^T M^{-2}U)]。
    """
    U = _as_U(S_full)
    m, P, T = U.shape
    kmax = min(int(kmax), m)
    sp2, sn2 = float(sigma_prior) ** 2, float(sigma_noise) ** 2
    _t0 = _perf()
    Minv = np.eye(P) * sp2
    eyeT = np.eye(T)
    remaining = list(range(m))
    order, tr_curve = [], []
    tr = float(np.trace(Minv))
    for _ in range(kmax):
        idx = np.asarray(remaining)
        Up = U[idx].transpose(1, 0, 2)                            # [P, n, T]
        Yf = Minv @ Up.reshape(P, -1)
        Y = Yf.reshape(P, len(idx), T)
        G1 = np.einsum("pnt,pnu->ntu", Up, Y)                     # U^T Minv U
        G2 = np.einsum("pnt,pnu->ntu", Y, Y)                      # U^T Minv² U
        K = G1 + sn2 * eyeT[None]
        red = np.trace(np.linalg.solve(K, G2), axis1=1, axis2=2)
        j = int(np.argmax(red))
        i = int(idx[j])
        order.append(i)
        remaining.remove(i)
        Yi = Minv @ U[i]
        Ki = sn2 * eyeT + U[i].T @ Yi
        Minv = Minv - Yi @ np.linalg.solve(Ki, Yi.T)
        Minv = 0.5 * (Minv + Minv.T)                              # 对称化防漂移
        tr = float(np.trace(Minv))
        tr_curve.append(tr)
    return dict(order=order, trace_curve=tr_curve, t_total=_perf() - _t0)


def spectral_order(solver, net, kmax, n_eig=None, ridge=1e-9):
    """谱近似代理（Zhou et al. 2024 Water Research 的简化版，非原文复刻）。

    图拉普拉斯（开启链路、无权、无向）最低频 r 个非平凡特征向量组成
    V[Nj, r]，在其行空间上做 D-最优行选贪心：每步选
    argmax log(1 + v_i^T B^{-1} v_i)，B = V_S^T V_S + ridge·I。
    直觉：低频图信号近似压力场的平滑分量，选行使采样算子对低频子空间
    最可逆。标注：**简化代理** - 原文的水力加权、聚类约束等均未复刻。
    """
    s = solver
    N = net.N
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    open_l = ~s.closed_np
    A = np.zeros((N, N))
    for a, b in zip(n1[open_l], n2[open_l]):
        A[a, b] += 1.0
        A[b, a] += 1.0
    Lap = np.diag(A.sum(1)) - A
    w, V = np.linalg.eigh(Lap)
    nontriv = np.where(w > 1e-8 * max(w[-1], 1.0))[0]             # 跳过每个连通分量的常向量
    junc = np.asarray(s.junc_nodes)
    r = min(int(n_eig or kmax), nontriv.size)
    Vj = V[np.ix_(junc, nontriv[:r])]                             # [Nj, r]
    m = junc.size
    kmax = min(int(kmax), m)
    B = ridge * np.eye(r)
    Lb = np.linalg.cholesky(B)
    remaining = list(range(m))
    order = []
    for _ in range(kmax):
        idx = np.asarray(remaining)
        Z = solve_triangular(Lb, Vj[idx].T, lower=True, check_finite=False)
        gains = np.log1p(np.sum(Z * Z, axis=0))
        j = int(np.argmax(gains))
        i = int(idx[j])
        order.append(i)
        remaining.remove(i)
        B += np.outer(Vj[i], Vj[i])
        Lb = np.linalg.cholesky(B)
    return dict(order=order, n_eig=r, n_components=int(N - nontriv.size))


# ======================================================================
# 四、评估（全部由一次 SVD 派生）
# ======================================================================
def eval_subset(S_full, sel, sigma_prior=15.0, sigma_noise=0.1, rank_rtol=None):
    """在同一 S_full 上评估候选子集 sel 的全部指标。

    返回 dict：
      f          贝叶斯 D 目标 = Σ log(1 + ρ²·sv²)，ρ = σ_prior/σ_noise
      rank       堆叠 [T·k, P] 的数值秩（sv > rtol·sv₁，rtol 默认 max(shape)·eps）
      n_zero_col 灵敏度列（max|·|）精确为 0 的管道数
      crlb_ident 可辨识子空间 CRLB 迹 = Σ_{i≤rank} σ_n²/sv_i²（无先验，
                 仅在可辨识方向上求逆 - 后验方差的频率学派代理）
      bayes_trace trace(M^{-1})（贝叶斯后验方差迹；M 与 FIM 同特征向量）
      lam_min_M  λ_min(M) = 1/σ_p² + λ_min(FIM)（rank<P 时 = 1/σ_p²）
    """
    sel = np.asarray(sel, dtype=np.int64)
    T, m, P = S_full.shape
    k = sel.size
    Ssub = S_full[:, sel, :].reshape(T * k, P)
    sp2, sn2 = float(sigma_prior) ** 2, float(sigma_noise) ** 2
    rho2 = sp2 / sn2
    if k == 0:                       # 空子集：无量测 ⇒ f=0，全列零，后验=先验
        return dict(f=0.0, rank=0, n_zero_col=int(P), crlb_ident=np.inf,
                    bayes_trace=float(P * sp2), lam_min_M=1.0 / sp2,
                    k=0, sv1=0.0, sv_rank=0.0)
    sv = np.linalg.svd(Ssub, compute_uv=False)
    if rank_rtol is None:
        rank_rtol = max(Ssub.shape) * np.finfo(np.float64).eps
    thr = rank_rtol * (sv[0] if sv.size and sv[0] > 0 else 0.0)
    rank = int(np.sum(sv > thr))
    f = float(np.sum(np.log1p(rho2 * sv ** 2)))
    lam = 1.0 / sp2 + sv ** 2 / sn2                                # M 的特征值
    bayes_trace = float(np.sum(1.0 / lam) + (P - sv.size) * sp2)
    lam_min = float(lam[-1]) if sv.size >= P else 1.0 / sp2
    crlb = float(np.sum(sn2 / sv[:rank] ** 2)) if rank else np.inf
    n_zero = int(np.sum(np.max(np.abs(Ssub), axis=0) == 0.0))
    return dict(f=f, rank=rank, n_zero_col=n_zero, crlb_ident=crlb,
                bayes_trace=bayes_trace, lam_min_M=lam_min, k=int(k),
                sv1=float(sv[0]) if sv.size else 0.0,
                sv_rank=float(sv[rank - 1]) if rank else 0.0)


def zero_col_curve(S_full, order):
    """order 前缀 k=1..len 的零列数曲线（增量 running-max，O(m·P)）。

    返回 (n_zero[k]（k 从 1 起）, nonzero_mask_final[P])。
    """
    T, m, P = S_full.shape
    colmax = np.zeros(P)
    out = np.empty(len(order), dtype=np.int64)
    for j, i in enumerate(order):
        colmax = np.maximum(colmax, np.max(np.abs(S_full[:, i, :]), axis=0))
        out[j] = int(np.sum(colmax == 0.0))
    return out, colmax > 0.0


# ======================================================================
# 五、增设模式（S0 固定，只在余下候选上加点）
# ======================================================================
def _post_var_diag(L_chol):
    """diag(M^{-1})，M = L L^T ⇒ M^{-1} = L^{-T}L^{-1}，第 j 个对角 = ‖L^{-1}e_j‖²。"""
    P = L_chol.shape[0]
    X = solve_triangular(L_chol, np.eye(P), lower=True, check_finite=False)
    return np.einsum("ij,ij->j", X, X)


def posterior_std(S_full, sel, sigma_prior=15.0, sigma_noise=0.1):
    """逐管贝叶斯后验标准差 sqrt(diag(M(S)^{-1}))，[P]，单位同 C。

    与 eval_subset / bayes_dopt_greedy 同一 M = I/σ_p² + Σ_S UU^T/σ_n²；
    空子集退化为 σ_prior（后验=先验）。这是"σ 口径"的逐管可辨识度：
    后验标准差被数据压到 σ_prior 的多少。
    """
    U = _as_U(S_full)
    m, P, T = U.shape
    sp2, sn2 = float(sigma_prior) ** 2, float(sigma_noise) ** 2
    sel = np.asarray(sel, dtype=np.int64)
    M = np.eye(P) / sp2
    if sel.size:
        Uf = U[sel].transpose(1, 0, 2).reshape(P, -1)
        M = M + (Uf @ Uf.T) / sn2
    return np.sqrt(_post_var_diag(_chol_spd(M)))


def _colmax_per_candidate(S_full):
    """[m, P]：候选 i 的每根管跨帧 max|S|（普查判据 max|S| > atol 的原料）。"""
    return np.max(np.abs(S_full), axis=0)


def bayes_dopt_augment(S_full, fixed, kadd, sigma_prior=15.0, sigma_noise=0.1,
                       objective="dopt", struct_mask=None, atol=0.0,
                       cert_ks=(), verbose=False):
    """增设贪心：已有传感器集 fixed（S0）固定不动，在余下候选上加 kadd 个。

    参数
    ----
    S_full     : [T, m, P] 全候选多帧灵敏度（同 bayes_dopt_greedy）
    fixed      : S0 的候选下标（S_full 第二维），一个不动
    kadd       : 增设个数（超出余量时截到余量）
    objective  : "dopt"  最大化 f(S0∪S) − f(S0)（贝叶斯 D-最优，lazy 贪心）
                 "cover" 最大化跨过可辨识阈值的管数（普查判据 max|S|>atol，
                         对非结构管计数；平手按 D-最优增益裁决；覆盖饱和后
                         退回 D-最优，饱和步号记在 cover_saturated_at）
    struct_mask: [P] bool，结构性不可辨识管（死支/全帧钳位），不计入可辨识集
    atol       : 普查的零列阈值（0 = 解析恒零，与 calib.identifiability 一致）
    cert_ks    : 需要证书与逐管后验标准差快照的 k

    单调性断言（增设的定义，违反即 RuntimeError = 实现 bug）：
      每加一个点，可辨识集 (max|S|>atol)&~struct 只增不减；
      diag(M^{-1}) 逐管不增（M 只加 PSD 项）。

    返回 dict：
      fixed, order（增设顺序，候选下标）, gains（D 目标边际增益）,
      f0 = f(S0), f_curve[kadd+1]（f_curve[0]=f0）,
      n_ident0, ident_curve[kadd+1]（可辨识管数）, cover_gains[kadd]（每步新覆盖管数）,
      cover_saturated_at（None 或步号）, cert{k: f, f0, df, f_recomputed, upper_df,
      ratio, n_ident, post_std[P], (cover 模式另有 cover_upper/cover_ratio)},
      ident_final[P], post_std_final[P], n_evals, t_total, objective, atol
    """
    if objective not in ("dopt", "cover"):
        raise ValueError("objective 必须是 'dopt' 或 'cover'")
    U = _as_U(S_full)
    m, P, T = U.shape
    sp2, sn2 = float(sigma_prior) ** 2, float(sigma_noise) ** 2
    fixed = np.unique(np.asarray(fixed, dtype=np.int64).ravel())
    if fixed.size and (fixed.min() < 0 or fixed.max() >= m):
        raise ValueError("fixed 越界（必须是 S_full 第二维下标）")
    fixed_set = set(fixed.tolist())
    cand = np.array([i for i in range(m) if i not in fixed_set], dtype=np.int64)
    kadd = max(0, min(int(kadd), cand.size))
    if struct_mask is None:
        struct = np.zeros(P, dtype=bool)
    else:
        struct = np.asarray(struct_mask, dtype=bool).ravel()
        if struct.shape != (P,):
            raise ValueError(f"struct_mask 形状 {struct.shape} != ({P},)")
    cert_ks = set(int(k) for k in cert_ks)
    atol = float(atol)

    _t0 = _perf()
    _warn_if_ill_conditioned(U, sp2, sn2)
    M = np.eye(P) / sp2
    if fixed.size:
        Uf = U[fixed].transpose(1, 0, 2).reshape(P, -1)
        M += (Uf @ Uf.T) / sn2
    L = _chol_spd(M)
    _, ld0 = np.linalg.slogdet(sp2 * M)
    f0 = float(ld0)
    colmax_c = _colmax_per_candidate(S_full)                      # [m, P]
    colmax = colmax_c[fixed].max(axis=0) if fixed.size else np.zeros(P)
    ident = (colmax > atol) & ~struct
    n_ident0 = int(ident.sum())
    pv = _post_var_diag(L)

    order, gains, cover_gains, cert = [], [], [], {}
    f_curve, ident_curve = [f0], [n_ident0]
    f = f0
    cover_saturated_at = None
    n_evals = 0
    snapshots, cover_snapshots = [], []
    heap = []
    remaining = set(cand.tolist())
    if kadd:
        g0 = _batch_gains(L, U, cand, sn2)
        n_evals = cand.size
        snapshots.append((0.0, np.sort(g0)[::-1]))
        heap = [(-g0[j], 0, int(cand[j])) for j in range(cand.size)]
        heapq.heapify(heap)

    for k in range(1, kadd + 1):
        use_heap = objective == "dopt" or cover_saturated_at is not None
        if not use_heap:
            rem = np.array(sorted(remaining), dtype=np.int64)
            newc = ((colmax_c[rem] > atol) & ~ident[None, :]
                    & ~struct[None, :]).sum(axis=1)
            cover_snapshots.append((int(ident.sum()) - n_ident0,
                                    np.sort(newc)[::-1]))
            best = int(newc.max())
            if best > 0:
                ties = rem[newc == best]
                gt = _batch_gains(L, U, ties, sn2)
                n_evals += ties.size
                j = int(np.argmax(gt))
                i, gain, cg = int(ties[j]), float(gt[j]), best
            else:
                cover_saturated_at = k
                gr = _batch_gains(L, U, rem, sn2)
                n_evals += rem.size
                heap = [(-gr[j], k - 1, int(rem[j])) for j in range(rem.size)]
                heapq.heapify(heap)
                use_heap = True
        if use_heap:
            while True:
                ng, stamp, i = heapq.heappop(heap)
                if i not in remaining:
                    continue
                if stamp == k - 1:
                    break
                g = float(_batch_gains(L, U, np.array([i]), sn2)[0])
                n_evals += 1
                heapq.heappush(heap, (-g, k - 1, i))
            gain = -ng
            cg = int(((colmax_c[i] > atol) & ~ident & ~struct).sum())
        order.append(int(i))
        remaining.discard(int(i))
        gains.append(float(gain))
        cover_gains.append(int(cg))
        f += float(gain)
        f_curve.append(f)
        M += (U[i] @ U[i].T) / sn2
        L = _chol_spd(M)

        # ---- 单调性断言（增设的定义）----
        colmax = np.maximum(colmax, colmax_c[i])
        ident_new = (colmax > atol) & ~struct
        if np.any(ident & ~ident_new):
            raise RuntimeError(
                f"增设第 {k} 步可辨识集缩小（{int((ident & ~ident_new).sum())} "
                f"根管）：running-max 不可能下降，属实现 bug")
        if int(ident_new.sum()) - int(ident.sum()) != cg:
            raise RuntimeError(f"增设第 {k} 步新覆盖计数不自洽：cover_gain={cg} "
                               f"vs 实际 {int(ident_new.sum()) - int(ident.sum())}")
        ident = ident_new
        ident_curve.append(int(ident.sum()))
        pv_new = _post_var_diag(L)
        if np.any(pv_new > pv * (1.0 + 1e-9)):
            worst = float(np.max(pv_new / pv))
            raise RuntimeError(
                f"增设第 {k} 步贝叶斯后验方差上升（最大比 {worst:.3e}）："
                f"M 只加 PSD 项，diag(M^-1) 不可能上升，属实现 bug")
        pv = pv_new

        if k in cert_ks:
            rem = np.array(sorted(remaining), dtype=np.int64)
            if rem.size:
                gr = _batch_gains(L, U, rem, sn2)
                n_evals += rem.size
                snapshots.append((f - f0, np.sort(gr)[::-1]))
                heap = [(-gr[j], k, int(rem[j])) for j in range(rem.size)]
                heapq.heapify(heap)
            upper = min(dfj + float(np.sum(gj[:k])) for dfj, gj in snapshots)
            upper = max(upper, f - f0)
            _, ld = np.linalg.slogdet(sp2 * M)
            c = dict(f=f, f0=f0, df=f - f0, f_recomputed=float(ld),
                     upper_df=upper,
                     ratio=(f - f0) / upper if upper > 0 else 1.0,
                     n_ident=int(ident.sum()), post_std=np.sqrt(pv).copy())
            if cover_snapshots:
                cu = min(cj + float(np.sum(gj[:k])) for cj, gj in cover_snapshots)
                cu = max(cu, float(int(ident.sum()) - n_ident0))
                c["cover_upper"] = cu
                c["cover_ratio"] = ((int(ident.sum()) - n_ident0) / cu
                                    if cu > 0 else 1.0)
            cert[k] = c
            if verbose:
                extra = (f"  覆盖证书 ratio={c['cover_ratio']:.4f}"
                         if "cover_ratio" in c else "")
                print(f"  [augment/{objective}] k=+{k:3d}  Δf={f - f0:.4f}  "
                      f"上界={upper:.4f}  ratio={c['ratio']:.4f}  可辨识 "
                      f"{n_ident0}→{int(ident.sum())}  |f-复算|={abs(f - ld):.2e}"
                      f"{extra}")
    return dict(fixed=fixed.tolist(), order=order, gains=gains, f0=f0,
                f_curve=f_curve, n_ident0=n_ident0, ident_curve=ident_curve,
                cover_gains=cover_gains, cover_saturated_at=cover_saturated_at,
                cert=cert, ident_final=ident, post_std_final=np.sqrt(pv),
                n_evals=int(n_evals), t_total=_perf() - _t0,
                sigma_prior=float(sigma_prior), sigma_noise=float(sigma_noise),
                objective=objective, atol=atol)


def recovery_report(S_full, ref, new, target_mask, struct_mask=None, atol=0.0):
    """普查同判据的逐管"找回"报告：参考布点 ref（S0）vs 评估布点 new。

    判据与 calib.identifiability 完全一致：管 k 可辨识 ⇔ max_{t,i∈S}|S[t,i,k]| > atol
    且非结构性（struct_mask）。target_mask 是要逐管判定的管集（如普查的 149 根
    "传感不足"管）。

    返回 dict：
      n_target, n_recovered（target 中在 new 下可辨识）, n_target_left,
      n_lost（ref 下可辨识、new 下不可辨识；若 ref ⊆ new 则必为 0，否则
      RuntimeError），n_unobservable_before/after（非结构且零列的管数），
      recovered_mask / lost_mask / unobservable_mask / ident_before / ident_after
    """
    T, m, P = S_full.shape
    ref = np.unique(np.asarray(ref, dtype=np.int64).ravel())
    new = np.unique(np.asarray(new, dtype=np.int64).ravel())
    target = np.asarray(target_mask, dtype=bool).ravel()
    struct = (np.zeros(P, dtype=bool) if struct_mask is None
              else np.asarray(struct_mask, dtype=bool).ravel())
    if target.shape != (P,) or struct.shape != (P,):
        raise ValueError("target_mask / struct_mask 形状须为 (P,)")
    colmax_c = _colmax_per_candidate(S_full)

    def _ident(sel):
        cm = colmax_c[sel].max(axis=0) if sel.size else np.zeros(P)
        return (cm > atol) & ~struct

    id0, id1 = _ident(ref), _ident(new)
    lost = id0 & ~id1
    if set(ref.tolist()) <= set(new.tolist()) and lost.any():
        raise RuntimeError(f"ref ⊆ new 却有 {int(lost.sum())} 根管从可辨识变为"
                           f"不可辨识：running-max 不可能下降，属实现 bug")
    return dict(n_target=int(target.sum()),
                n_recovered=int((target & id1).sum()),
                n_target_left=int((target & ~id1).sum()),
                n_lost=int(lost.sum()),
                n_unobservable_before=int((~id0 & ~struct).sum()),
                n_unobservable_after=int((~id1 & ~struct).sum()),
                n_ident_before=int(id0.sum()), n_ident_after=int(id1.sum()),
                recovered_mask=target & id1, lost_mask=lost,
                unobservable_mask=~id1 & ~struct,
                ident_before=id0, ident_after=id1)


# ======================================================================
# 六、相干驱动增设（S0 固定，压低漏损签名字典的互相干）
# ======================================================================
ORTH_TOL = 1e-12          # "精确正交对"判据（与 augment_public.coherence_stats 同）


def _gram_per_row(Dfull):
    """[T, m, NC] → 每个传感位置的 Gram 贡献 G_i = Σ_t D[t,i,:]ᵀ D[t,i,:]，[m, NC, NC]。"""
    D = np.asarray(Dfull, dtype=np.float64)
    if D.ndim == 2:
        D = D[None]
    return np.einsum("tia,tib->iab", D, D)


def _coh_from_gram(G):
    """Gram [..., NC, NC] → 互相干 |μ| [..., NC, NC]（零列的相干记 0，与 coherence_stats 同）。"""
    n = np.sqrt(np.clip(np.diagonal(G, axis1=-2, axis2=-1), 0.0, None))
    return np.abs(G) / np.maximum(n[..., :, None] * n[..., None, :], 1e-300)


def _pair_objective(off, objective, quantile, eps):
    """off [..., n_pairs] → (主目标 [...], J=logdet2 [...])。"""
    J = -np.log(np.clip(1.0 - off ** 2, eps, None)).sum(axis=-1)
    if objective == "logdet2":
        prim = J
    elif objective == "max":
        prim = off.max(axis=-1)
    elif objective == "quantile":
        prim = np.quantile(off, quantile, axis=-1)
    else:
        raise ValueError("objective 必须是 'logdet2' | 'max' | 'quantile'")
    return prim, J


def coherence_stats_rows(Dfull, sel, eps=1e-12, quantile=0.99):
    """传感集 sel（行下标）下签名字典互相干的摘要（真值无关；零编号）。

    返回 dict：n_sensors, n_pairs, coh_max, coh_median_all, coh_median_within
    （非精确正交对）, coh_q90, coh_q99, coh_mean, n_orthogonal_pairs,
    n_pairs_gt_0999, n_pairs_gt_099, n_pairs_gt_09, logdet2（=J）,
    min_col_norm / median_col_norm（列 2-范数 = 单位漏损系数在传感集上的信号幅值，
    可检测性诊断，与相干无关但同样真值无关）。
    """
    D = np.asarray(Dfull, dtype=np.float64)
    if D.ndim == 2:
        D = D[None]
    sel = np.unique(np.asarray(sel, dtype=np.int64).ravel())
    NC = D.shape[2]
    A = D[:, sel, :].reshape(-1, NC)
    G = A.T @ A
    mu = _coh_from_gram(G)
    iu, ju = np.triu_indices(NC, 1)
    off = mu[iu, ju]
    orth = off < ORTH_TOL
    within = off[~orth]
    norms = np.sqrt(np.clip(np.diag(G), 0.0, None))
    _, J = _pair_objective(off, "logdet2", quantile, eps)
    return dict(n_sensors=int(sel.size), n_pairs=int(off.size),
                coh_max=float(off.max()) if off.size else 0.0,
                coh_median_all=float(np.median(off)) if off.size else 0.0,
                coh_median_within=float(np.median(within)) if within.size else None,
                coh_q90=float(np.quantile(off, 0.90)) if off.size else 0.0,
                coh_q99=float(np.quantile(off, 0.99)) if off.size else 0.0,
                coh_mean=float(off.mean()) if off.size else 0.0,
                n_orthogonal_pairs=int(orth.sum()),
                n_pairs_gt_0999=int((off > 0.999).sum()),
                n_pairs_gt_099=int((off > 0.99).sum()),
                n_pairs_gt_09=int((off > 0.9).sum()),
                logdet2=float(J),
                min_col_norm=float(norms.min()), median_col_norm=float(np.median(norms)))


def coherence_augment(Dfull, fixed, kadd, pool=None, objective="logdet2",
                      quantile=0.99, eps=1e-12, verbose=False, log=print):
    """相干驱动增设：S0 = fixed 固定不动，在 pool 上贪心加 kadd 个传感位置。

    参数
    ----
    Dfull     : [T, m, NC] 漏损签名字典（T 帧 × m 个候选传感位置 × NC 个漏损候选）；
                [m, NC] 视为 T=1
    fixed     : S0 的行下标
    kadd      : 增设个数（超出池余量时截到余量）
    pool      : 允许的传感位置行下标（None = 全部 − fixed）。是否剔除某些位置
                （如"公平池"剔除漏点节点）由调用方决定，本函数不知道漏点是谁
    objective : "logdet2"（缺省）最小化 J(S) = Σ_{i<j} −log(1 − μ_ij² + eps)；
                "max" 字典序最小化 (max μ, J)；"quantile" 字典序最小化 (q 分位 μ, J)
    quantile  : objective="quantile" 的分位
    eps       : −log 的下限保护（μ 精确为 1 的对记 −log eps）

    真值无关：只用 Dfull、fixed、pool；每一步对 pool 内全部位置向量化评估
    G(S∪{v}) = G(S) + G_v 后的目标值，取最小者（np.lexsort，并列取最小下标）。

    返回 dict：fixed, pool, order（增设顺序，行下标）, objective, quantile, eps,
      steps[kadd+1]（第 k 步（k=0 为 S0）的 coherence_stats_rows 摘要 + objective_value
      + chosen（本步所选行下标，k=0 为 None）+ gain（主目标下降量）),
      n_evals, t_total
    """
    D = np.asarray(Dfull, dtype=np.float64)
    if D.ndim == 2:
        D = D[None]
    T, m, NC = D.shape
    fixed = np.unique(np.asarray(fixed, dtype=np.int64).ravel())
    if fixed.size and (fixed.min() < 0 or fixed.max() >= m):
        raise ValueError("fixed 越界（必须是 Dfull 第二维下标）")
    fset = set(fixed.tolist())
    if pool is None:
        pool = np.array([i for i in range(m) if i not in fset], dtype=np.int64)
    else:
        pool = np.unique(np.asarray(pool, dtype=np.int64).ravel())
        if pool.size and (pool.min() < 0 or pool.max() >= m):
            raise ValueError("pool 越界")
        pool = np.array([i for i in pool.tolist() if i not in fset], dtype=np.int64)
    kadd = max(0, min(int(kadd), pool.size))
    if objective not in ("logdet2", "max", "quantile"):
        raise ValueError("objective 必须是 'logdet2' | 'max' | 'quantile'")
    _t0 = _perf()
    Gi = _gram_per_row(D)                                       # [m, NC, NC]
    G = Gi[fixed].sum(axis=0) if fixed.size else np.zeros((NC, NC))
    iu, ju = np.triu_indices(NC, 1)

    def _step_record(Gm, sel, chosen, gain):
        rec = coherence_stats_rows(D, sel, eps, quantile)
        off = _coh_from_gram(Gm)[iu, ju]
        prim, _ = _pair_objective(off, objective, quantile, eps)
        rec.update(objective_value=float(prim), chosen=chosen, gain=gain)
        return rec

    sel = fixed.copy()
    steps = [_step_record(G, sel, None, 0.0)]
    order, n_evals = [], 0
    remaining = pool.tolist()
    for k in range(1, kadd + 1):
        rem = np.array(remaining, dtype=np.int64)
        G_all = G[None, :, :] + Gi[rem]                          # [n_rem, NC, NC]
        off = _coh_from_gram(G_all)[:, iu, ju]                   # [n_rem, n_pairs]
        prim, J = _pair_objective(off, objective, quantile, eps)
        n_evals += rem.size
        j = int(np.lexsort((J, prim))[0])
        i = int(rem[j])
        G = G + Gi[i]
        order.append(i)
        remaining.remove(i)
        sel = np.r_[sel, i]
        prev = steps[-1]["objective_value"]
        steps.append(_step_record(G, sel, i, float(prev - prim[j])))
        if verbose:
            s = steps[-1]
            log(f"  [coh/{objective}] k=+{k:3d}  目标 {s['objective_value']:.4f}"
                f"（降 {s['gain']:.4f}）  max μ={s['coh_max']:.8f}  中位 "
                f"{s['coh_median_all']:.4f}  >0.999 {s['n_pairs_gt_0999']}  "
                f">0.99 {s['n_pairs_gt_099']}  评估 {rem.size}")
    return dict(fixed=fixed.tolist(), pool=pool.tolist(), order=order,
                objective=objective, quantile=float(quantile), eps=float(eps),
                steps=steps, n_evals=int(n_evals), t_total=_perf() - _t0)

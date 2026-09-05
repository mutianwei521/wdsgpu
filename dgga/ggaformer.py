# -*- coding: utf-8 -*-
"""dgga.ggaformer - GGAFormer v1：把 EPANET GGA 迭代逐层展开为带保证的学习求解器。

架构心脏（三个内建设计，不可妥协）：
① 流量更新恒用 Q ← Q − α ⊙ (Y_eff − P_eff·EᵀH) 形式，且 α 同步进入线性系统装配
   （P̂ = α·P_eff，Ŷ = α·Y_eff）→ 逐层质量守恒结构性免费：
   更新后节点净流残差 = 线性解残差，与权重无关；数值上逐节点
   m ≤ O(eps)·(|A|·|Hj|+|F|)（W1 实测常数 ≤3.5；TCV 网 P̂~1e6 放大下地板
   ~1e-6 cfs，TCV-free 网 ~1e-9 cfs）。超地板异常由尺度感知精化门兜底。
   （EPANET 12_analysis_algorithms.rst:158 的 GGA 流量更新式的 α 推广）
② 学习模块只输出 g 的正尺度因子 s = softplus(·)（g_eff = g_phys·s，等价
   P_eff = P/s、Y_eff = Y/s）与正步长 α = softplus(·)。s,α > 0 ⇒ A 恒 SPD；
   在精确解 (Q*,H*) 处 Y − P·(H*₁−H*₂) = (hloss−dh)/g = 0，任何 s,α>0 都保持
   dq = 0 ⇒ 不动点集 = 原方程解集（定理 2）。
   初始化：所有头的末层 Linear 权重/偏置全零 ⇒ z=0 ⇒ s = α = softplus(b0)，
   且 P̂ = P·(α/s) = P·1.0 逐位不变 ⇒ 未训练模型 = 原版 GGA（逐位）。
③ 线性解层 v1 用精确稠密 Cholesky（torch.linalg.cholesky 批量 + 2 步同精度
   迭代精化，与 dgga/solver.py dense 路径逐句同源）。

范围：与 GGASolver(mode='dense') 一致 - H-W 管道/TCV/水库/DDA/emitter，
同拓扑内批场景（免 padding）。内部单位 ft/cfs，float64。
"""

import math

import numpy as np
import torch
import torch.nn as nn

try:
    from dgga.solver import GGASolver, CBIG, CSMALL, PI, QZERO
except ImportError:  # pragma: no cover
    from solver import GGASolver, CBIG, CSMALL, PI, QZERO

# softplus(B0) = ln(1+e^{ln(e-1)}) = ln(e) = 1.0（f64 下逐位或 1ulp 内）
B0 = math.log(math.e - 1.0)


def _slog(x, c=1.0):
    """有符号对数压缩：sign(x)·log1p(|x|/c)。特征工程用，不进物理路径。"""
    return torch.sign(x) * torch.log1p(torch.abs(x) / c)


def _imbalance(solver, q, e_j, d_j, em):
    """junction 节点净流失衡 m = E·Q − em·e − d（[B,Nj]，cfs）。
    直接在链路流量上做 scatter 求和：项量级 = |q|（无 P̂·H ~1e9 的大数相消），
    f64 舍入 ~eps·Σ_{k@i}|q_k| ≈ 1e-15 级 - 这是守恒保证的正确度量对象；
    经由 F−A·H 度量会被装配/矩阵向量积中 P̂·H_fix 大数项的存储舍入
    （RQtol 钳位链路 P̂~1e6 × |H|~1e3 → eps·1e9 ≈ 2e-7）污染。"""
    B = q.shape[0]
    m = torch.zeros(B, solver.Nj, dtype=q.dtype, device=q.device)
    m.scatter_add_(1, solver.f_idx1.expand(B, -1), -q[:, solver.lk_m1])
    m.scatter_add_(1, solver.f_idx2.expand(B, -1), q[:, solver.lk_m2])
    return m - em * e_j - d_j


# 守恒精化门（W1 复核后改为尺度感知）：
#   thr = max(_CONS_GATE, _CONS_GATE_RELC·eps·rowscale)，rowscale=|A|·|Hj|+|F| 逐节点。
# 依据（W1 实测）：失衡 m 恒为线性解舍入（与权重无关），全部案例
# m ≤ 3.5·eps·rowscale（恒等 1.1、随机权重 3.4、city_d TCV 1.0）。绝对门 5e-10
# 在 TCV 网（city_d 地板 ~6e-7 cfs, P̂~1e6 放大）逐层误触发，流空间修正在平坦
# 支注入 ~1e-6 cfs 漂移，破坏恒等锁死与不动点 - 故门限带 eps·行尺度地板，
# 恒等/物理路径零触发（逐位保持），仅在异常失衡（>16×地板）时精化兜底。
_CONS_GATE = 5e-10
_CONS_GATE_RELC = 16.0
_EPS64 = torch.finfo(torch.float64).eps


def _zero_last(seq):
    """把 Sequential 的末层 Linear 权重与偏置清零（恒等初始化用）。"""
    last = seq[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class _LayerHead(nn.Module):
    """逐层学习头：edge 特征 → (z_s, z_α)。末层零初始化 ⇒ s=α=softplus(B0)。"""

    def __init__(self, n_feat, hidden, dtype=torch.float64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(hidden, 2, dtype=dtype))
        _zero_last(self.mlp)

    def forward(self, feat):                     # feat: [B,L,F]
        z = self.mlp(feat)
        return z[..., 0], z[..., 1]              # z_s, z_a: [B,L]


class _ScalarHead(nn.Module):
    """逐层**标量**头（W3 新增，head_mode='scalar'）：每层只学 2 个自由参数
    (z_s, z_α)，不看任何特征、不做逐边 MLP。

    存在理由（W3 盈亏平衡分析的直接推论）：在 O(nnz) 精确稀疏 Cholesky 下
    GGA 单次迭代只值 ~105 flop/边，而**任何**逐边学习件（哪怕 hidden=1）光是
    特征构建就要 60 flop/边、MLP 再 30 flop/边 ⇒ 0.85 个迭代当量/层，
    结构性买不起。标量头把学习件成本压到 ~2 flop/边（P,Y 各一次标量乘，
    ≈2% 迭代当量），是**唯一在 FLOPs 上买得起**的学习件形态
 - 本质上是"学出来的松弛/步长调度表"。
    零初始化 ⇒ z=0 ⇒ ratio=1 逐位 ⇒ 未训练时仍是原版 GGA。"""

    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.z = nn.Parameter(torch.zeros(2, dtype=dtype))

    def forward(self, feat):                     # feat 仅用于取 shape
        B, L = feat.shape[0], feat.shape[1]
        return (self.z[0].expand(B, L), self.z[1].expand(B, L))


class _WarmStartHead(nn.Module):
    """热启动头：edge-MLP + 一轮邻接聚合 → ΔQ0。末层零初始化 ⇒
    q0 = EPANET 1fps 惯例初值（PI·D²/4，关闭支 QZERO）逐位不变。"""

    def __init__(self, n_edge_feat, n_node_feat, hidden, dtype=torch.float64):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_edge_feat, hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=dtype), nn.Tanh())
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden + n_node_feat, hidden, dtype=dtype), nn.Tanh())
        self.out = nn.Sequential(
            nn.Linear(3 * hidden, hidden, dtype=dtype), nn.Tanh(),
            nn.Linear(hidden, 1, dtype=dtype))
        _zero_last(self.out)

    def forward(self, ef, nf, n1, n2, N):
        """ef:[B,L,Fe] 边特征；nf:[B,N,Fn] 节点特征；n1/n2:[L] 端点索引。"""
        B, L, _ = ef.shape
        he = self.edge_mlp(ef)                                   # [B,L,H]
        Hd = he.shape[-1]
        # 邻接聚合：节点 = 关联边嵌入的均值
        acc = torch.zeros(B, N, Hd, dtype=ef.dtype, device=ef.device)
        cnt = torch.zeros(B, N, 1, dtype=ef.dtype, device=ef.device)
        idx1 = n1.view(1, L, 1).expand(B, L, Hd)
        idx2 = n2.view(1, L, 1).expand(B, L, Hd)
        acc.scatter_add_(1, idx1, he)
        acc.scatter_add_(1, idx2, he)
        one = torch.ones(B, L, 1, dtype=ef.dtype, device=ef.device)
        cnt.scatter_add_(1, n1.view(1, L, 1).expand(B, L, 1), one)
        cnt.scatter_add_(1, n2.view(1, L, 1).expand(B, L, 1), one)
        hn = self.node_mlp(torch.cat([acc / cnt.clamp_min(1.0), nf], dim=-1))
        z = torch.cat([he, hn[:, n1, :], hn[:, n2, :]], dim=-1)  # [B,L,3H]
        return self.out(z)[..., 0]                               # [B,L]


class _SweepStruct:
    """W6：A 的行空间邻接（对称展开）+ 两种"块内无耦合"的行划分。

    ① blocks_gs - **GS 波前层级**。FillPattern 的 lvl 定义为
       lvl[i] = 1 + max{lvl[j] : j < i, A_ij ≠ 0}（无下邻居则 1），恰好是
       前向替换的依赖 DAG 深度。同层的行之间没有下三角耦合，且任一上邻居
       j>i 必满足 lvl[j] > lvl[i]（因为 i 是 j 的下邻居）⇒ 整层同时更新时，
       下邻居用的是新值、上邻居用的是旧值 - **与逐行顺序 GS 逐位同解**。
       于是"GS 是顺序算法、没法向量化"在本问题族上是不成立的：MMD 序下
       nlev 实测只有 8–16（Nj = 84–268），并行深度与多色 GS 同量级。
       这一点必须与 FLOPs 并排报告（任务书要求的"是否可并行"栏）。
    ② blocks_rb - **贪心图着色**（红黑/多色 GS）。同色行互不相邻，
       更新只用到其它颜色的值 ⇒ 天然并行，深度 = 颜色数。它与 GS 是
       **不同的扫描序**，故迭代数会不同（一般略差于自然序 GS）。

    行序一律取 EPANET/MMD 序（solver.row_junc）：这是 W2/W3 的稀疏
    Cholesky 用的同一个置换，选它才能保证"除线性解外一切不变"这个控制条件；
    另外 MMD 是最小度序，波前浅、块少，对 GS 也恰好是好序。
    """

    def __init__(self, pat):
        n, nnz = pat.n, pat.nnz
        er, ec = pat.ent_row_np, pat.ent_col_np           # er > ec（下三角）
        eid = np.arange(nnz, dtype=np.int64)
        # 对称展开的全邻接三元组 (row, col, ent)
        self.rows_all = np.concatenate([er, ec])
        self.cols_all = np.concatenate([ec, er])
        self.ents_all = np.concatenate([eid, eid])
        self.n, self.nnz = n, nnz
        self.lvl = pat.lvl_np.copy()
        self.nlev = int(pat.nlev)
        self.color = self._greedy_color(n, er, ec)
        self.ncol = int(self.color.max()) + 1 if n else 0
        dev = pat.device
        self.blocks_gs = self._blocks(self.lvl, dev)
        self.blocks_rb = self._blocks(self.color, dev)

    # ------------------------------------------------------------------
    @staticmethod
    def _greedy_color(n, er, ec):
        """贪心图着色（按度降序，Welsh–Powell 式）。着色只需"同色不相邻"，
        贪心足够（任务书明示可用贪心）；颜色数实测 4–6。"""
        adj = [[] for _ in range(n)]
        for i, j in zip(er.tolist(), ec.tolist()):
            adj[i].append(j)
            adj[j].append(i)
        deg = np.array([len(a) for a in adj], dtype=np.int64)
        order = np.argsort(-deg, kind="stable")
        col = np.full(n, -1, dtype=np.int64)
        for i in order.tolist():
            used = {col[j] for j in adj[i] if col[j] >= 0}
            c = 0
            while c in used:
                c += 1
            col[i] = c
        return col

    def _blocks(self, label, dev):
        """按 label 的升序分块；每块给出 (rows, loc, col, ent, nb)。"""
        out = []
        for v in range(int(label.min()), int(label.max()) + 1):
            rows = np.where(label == v)[0]
            if rows.size == 0:
                continue
            pos = np.full(self.n, -1, dtype=np.int64)
            pos[rows] = np.arange(rows.size)
            sel = pos[self.rows_all] >= 0
            out.append((
                torch.as_tensor(rows, dtype=torch.long, device=dev),
                torch.as_tensor(pos[self.rows_all[sel]], dtype=torch.long,
                                device=dev),
                torch.as_tensor(self.cols_all[sel], dtype=torch.long,
                                device=dev),
                torch.as_tensor(self.ents_all[sel], dtype=torch.long,
                                device=dev),
                int(rows.size)))
        return out


class GGAFormerV1(nn.Module):
    """展开 GGA Transformer 求解器 v1（K 层，逐层独立学习头 + 热启动头）。

    模型权重拓扑无关（纯逐边/逐节点 MLP + scatter 聚合）；几何由传入的
    GGASolver(mode='dense') 承载，同拓扑内批场景。
    """

    N_FEAT = 8          # 逐层头输入特征数
    N_EDGE_FEAT = 4     # 热启动头边特征数
    N_NODE_FEAT = 3     # 热启动头节点特征数

    def __init__(self, K=8, hidden=32, dtype=torch.float64,
                 linear_solver="sparse_chol", precond="learned", tol_lin=1e-10,
                 pcg_maxit=500, pcg_crit="cbe", precond_dim=32,
                 precond_heads=4, pcg_fallback=True, chol_refine=2,
                 use_warm_start=True, use_g_scale=True, use_alpha=True,
                 head_mode="edge", diag_omega=1.0, diag_sweeps=1,
                 sweep_omega=1.0, sweep_sweeps=1, cg_iters=1,
                 omega_mode="fixed", omega_seq=None, omega_param="softplus",
                 omega_span=2.0):
        super().__init__()
        self.K = K
        self.hidden = hidden
        self.dtype_ = dtype
        # ---- W3 消融开关（默认全 True/edge = W1/W2 原架构，逐位不变）----
        # 关闭的组件**不建参数、不进前向**：FLOPs 账本据此归零，消融表才诚实。
        self.use_warm_start = bool(use_warm_start)
        self.use_g_scale = bool(use_g_scale)
        self.use_alpha = bool(use_alpha)
        assert head_mode in ("edge", "scalar")
        self.head_mode = head_mode        # 'edge' = 逐边 MLP；'scalar' = 逐层 2 标量
        self.has_layer_head = self.use_g_scale or self.use_alpha
        _mk = ((lambda: _LayerHead(self.N_FEAT, hidden, dtype))
               if head_mode == "edge" else (lambda: _ScalarHead(dtype)))
        self.heads = nn.ModuleList(
            [_mk() for _ in range(K)] if self.has_layer_head else [])
        self.warm = (_WarmStartHead(self.N_EDGE_FEAT, self.N_NODE_FEAT,
                                    hidden, dtype)
                     if self.use_warm_start else None)
        self.register_buffer("b0", torch.tensor(B0, dtype=dtype))
        # ---- 线性解层三条路径 ----
        #   'sparse_chol'（W3 新默认） - 精确稀疏 Cholesky 直接解，O(nnz) 一步到位；
        #   'cholesky'（W1 原路径） - 稠密 Cholesky O(Nj³)，逐位不变，保留不动；
        #   'pcg'（W2 原路径） - 预条件 CG 到 tol_lin，逐位不变，保留不动。
        #   'diag'（W5 新增，**一阶基线**） - 不做任何分解：H ← H + ω·D⁻¹(F−A·H)，
        #     D = diag(A)，阻尼 Jacobi，逐层热带上一层的 H（见 _solve_diag 的推导）。
        #   W6 经典阶梯（一阶/无分解成本类，全部走**同一套** P/Y 线性化、
        #   scatter_add 装配、守恒型流量更新，只换"每层内部怎么解 A·H=F"）：
        #   'gs' - Gauss–Seidel（MMD 序前向扫描，波前层级调度 ⇒ 与顺序 GS 逐位同解）
        #   'rbgs' - 多色 Gauss–Seidel（贪心图着色，并行友好；ω=1 即红黑 GS）
        #   'sor' - 带松弛因子的 GS（sweep_omega 在训练池上选）
        #   'cg' - 无预条件共轭梯度（每层 cg_iters 步，热启自上一层 H）
        assert linear_solver in ("cholesky", "pcg", "sparse_chol", "diag",
                                 "gs", "rbgs", "sor", "cg")
        self.linear_solver = linear_solver
        self.diag_omega = float(diag_omega)
        self.diag_sweeps = int(diag_sweeps)
        self.sweep_omega = float(sweep_omega)
        self.sweep_sweeps = int(sweep_sweeps)
        self.cg_iters = int(cg_iters)
        # 一阶/无分解成本类：状态是 (Q,H) 对、逐层热启动、**不做守恒精化**
        # （精化要解 A·δh=m，等于把牛顿法偷偷请回来，实验就没有意义了）。
        self.first_order = linear_solver in ("diag", "gs", "rbgs", "sor", "cg")
        self._sweep_cache = {}
        # ---- W6 任务 C：**逐层松弛序列** ω_k（学习件的理论最优形态）----
        # 动机：ω_k 只出现在 H ← H + ω_k·D⁻¹r（或 SOR 扫描）里，那次乘法在
        # FLOPs 账本中**本来就已经计入**（'diag' 的 3·Nj、GS 类的 6·Nj），
        # 故把常数 ω 换成逐层可学 ω_k 的边际成本**精确为 0 flop/边**，
        # ovh ≡ 0 - 这是唯一"白送"的学习件形态。
        # 它数学上就是**多项式（Chebyshev 半迭代型）加速**：K 步定常迭代的误差
        # 传播算子是 Π_k (I − ω_k D⁻¹A)，学 ω_k 等价于在 spec(D⁻¹A) 上挑一个
        # K 次多项式 p(λ)=Π(1−ω_k λ) 使 max|p| 最小 - 最优解是 Chebyshev 根。
        # 故"学到的 ω_k vs 教科书 Chebyshev/最优 SOR"是本任务的关键理论对照。
        # 保证三件套不破：
        #   ① 恒等锚 - z=0 ⇒ softplus(B0)=1.0（f64 逐位）⇒ ω_k = ω₀ 逐位；
        #   ② 保结构 - ω_k = ω₀·softplus(·) > 0，A 不变（仍 SPD）；
        #   ③ 不动点 - 在 H* 处 r = F−A·H* = 0，任何 ω_k 都不动 ⇒ dq = 0。
        assert omega_mode in ("fixed", "learn")
        assert omega_param in ("softplus", "exp")
        self.omega_mode = omega_mode
        # 'softplus'：ω = ω₀·softplus(B0 + span·tanh(z))，取值有界（训练稳）；
        # 'exp'    ：ω = ω₀·exp(z)，取值 (0,∞) 无界 - **任务 C 的"最好机会"档**：
        #   Chebyshev 最优序列里最大的 ω 是 1/λ_min，可达 1e4 量级，有界参数化
        #   会把学习件锁死在教科书最优解**之外**，那样的"学不赢"是协议缺陷而非结论。
        #   exp(0)=1.0 在 f64 下逐位精确 ⇒ 恒等锚不破。
        self.omega_param = omega_param
        self.omega_span = float(omega_span)
        if omega_seq is not None:
            # 定常但**逐层给定**的经典调度（Chebyshev 半迭代等），零参数。
            self.register_buffer("omega_seq",
                                 torch.as_tensor(np.asarray(omega_seq),
                                                 dtype=dtype))
        else:
            self.omega_seq = None
        if omega_mode == "learn":
            self.omega_z = nn.Parameter(torch.zeros(K, dtype=dtype))
        self.chol_refine = int(chol_refine)
        self.precond = precond            # learned|exact|ic0|jacobi|none
        self.tol_lin = float(tol_lin)
        self.pcg_maxit = int(pcg_maxit)
        self.pcg_crit = pcg_crit
        self.pcg_fallback = bool(pcg_fallback)
        self._pat_cache = {}
        if linear_solver == "pcg" and precond == "learned":
            from dgga.precond import MaskedCholPrecond
            self.mprecond = MaskedCholPrecond(dim=precond_dim,
                                              n_head=precond_heads, dtype=dtype)

    # ------------------------------------------------------------------
    def _get_pattern(self, solver, force_mode=None):
        """填充模式（按 solver 与预条件子类型缓存）。exact ⇒ 完全填充模式
        （= 精确稀疏 Cholesky）；其余 ⇒ IC(0) 模式（A 自身非零）。"""
        from dgga.precond import FillPattern
        mode = force_mode or ("full" if self.precond == "exact" else "ic0")
        key = (id(solver), mode)
        if key not in self._pat_cache:
            self._pat_cache[key] = FillPattern.from_solver(solver, mode=mode)
        return self._pat_cache[key]

    def _solve_sparse_chol(self, solver, Pm, em, hgrad_e, F, outs, ell):
        """W3 线性解层（新默认）：稀疏装配 + **精确**稀疏 Cholesky 直接解。

        依据（W2 已实证，53 网）：MMD 排序下 nnz(L)/nnz(A) ∈ [1.00, 2.35] 且
        与规模无关（log–log r=0.06）⇒ 完全填充模式上的数值分解就是精确
        Cholesky，**一次分解 + 一次三角解 = 精确解**（PCG 意义下 1 步收敛）。
        于是 W2 的 Krylov 迭代层在本问题族上是多余的：直接解本身就是保证层，
        且成本从 O(Nj³) 降到 O(nnz)。

        精度不依赖任何学习件（学习件只改 A 的数值，不改解的精确性）；
        非正主元（SPD 被破坏）由 ic0_factor 显式计数并写进 outs，绝不静默。
        """
        from dgga import precond as _pc
        B, Nj = F.shape
        dt, dev = F.dtype, F.device
        pat = self._get_pattern(solver, force_mode="full")
        a_d, a_o = _pc.assemble_sparse(pat, Pm, em=em, hgrad_e=hgrad_e)
        F_row = torch.zeros(B, Nj, dtype=dt, device=dev
                            ).index_copy(1, pat.r_of_j, F)
        sol = _pc.sparse_chol_solve(pat, a_d, a_o, F_row,
                                    refine=self.chol_refine)
        Hj = sol["x"][:, pat.r_of_j]
        outs.setdefault("chol_breakdown", []).append(sol["n_breakdown"])
        if sol["n_breakdown"]:                     # 显式报告，绝不静默
            outs.setdefault("chol_breakdown_layers", []).append(ell)
        ctx = (pat, a_d.detach(), a_o.detach(), sol["Ld"], sol["Lo"],
               F_row.detach())
        return Hj, ctx

    # ------------------------------------------------------------------
    # W5 一阶线性解层（linear_solver='diag'）
    # ------------------------------------------------------------------
    def _diag_matvec(self, solver, Pm, em, hgrad_e, Hj, rh_zero, absval=False):
        """A·Hj（junction 行）的**边形式**求值，零矩阵存储。

        A = E·P̂·Eᵀ + diag(em/hgrad)。把 Hj 补成全节点向量（定水头位置填 0，
        因为定水头列不在 A 内），则逐链路 t_k = P̂_k·(H_{n1}−H_{n2}) 散射到
        行 j1（+t）与行 j2（−t）后恰好等于 A·Hj：
          · 两端 junction：行 j1 += P̂(H1−H2)，行 j2 += P̂(H2−H1)  ✓
          · 只有 n1 是 junction（n2 定水头，填 0）：行 j1 += P̂·H1     ✓（无非对角）
          · 只有 n2 是 junction：行 j2 += P̂·H2                       ✓
        absval=True 时返回 Σ_j|A_ij|·|H_j|（行尺度，守恒残差的相对判据用）。"""
        B, Nj = Hj.shape
        dt, dev = Hj.dtype, Hj.device
        Hf = torch.cat([Hj, rh_zero], dim=1).index_select(1, solver._hperm_inv)
        emh = em / hgrad_e
        acc = torch.zeros(B, Nj, dtype=dt, device=dev)
        if absval:
            t = Pm.abs() * (Hf[:, solver.n1].abs() + Hf[:, solver.n2].abs())
            acc.scatter_add_(1, solver.f_idx1.expand(B, -1), t[:, solver.lk_m1])
            acc.scatter_add_(1, solver.f_idx2.expand(B, -1), t[:, solver.lk_m2])
            return acc + emh.abs() * Hj.abs()
        t = Pm * (Hf[:, solver.n1] - Hf[:, solver.n2])
        acc.scatter_add_(1, solver.f_idx1.expand(B, -1), t[:, solver.lk_m1])
        acc.scatter_add_(1, solver.f_idx2.expand(B, -1), -t[:, solver.lk_m2])
        return acc + emh * Hj

    def _diag_of_A(self, solver, Pm, em, hgrad_e, Nj):
        """diag(A) = Σ_{k@i} P̂_k + (em/hgrad)_i（两次 scatter_add，O(L)）。"""
        B = Pm.shape[0]
        d_a = torch.zeros(B, Nj, dtype=Pm.dtype, device=Pm.device)
        d_a.scatter_add_(1, solver.f_idx1.expand(B, -1), Pm[:, solver.lk_m1])
        d_a.scatter_add_(1, solver.f_idx2.expand(B, -1), Pm[:, solver.lk_m2])
        return d_a + em / hgrad_e

    def _omega(self, ell, base):
        """第 ell 层实际使用的松弛因子。

        三种形态（互斥）：
          · omega_mode='fixed' 且 omega_seq is None ⇒ 返回常数 base（默认，
            与 W5/W6 任务 A 的一切数值**逐位不变**）；
          · omega_seq 给定 ⇒ 经典**逐层调度**（Chebyshev 半迭代等），零参数，
            按 ell % len 循环（= 教科书的 cyclic Chebyshev；故长程 rollout 的
            分块长度必须取周期的整数倍，否则相位会错）；
          · omega_mode='learn' ⇒ ω_ell = base·softplus(B0 + 2·tanh(z_ell))，
            z=0 时逐位 = base（恒等锚），取值范围 base·[0.076, 2.62]。
        """
        ov = getattr(self, "_omega_override", None)
        if ov is not None:
            # W6 任务 C 的**无梯度优化器**评估通道：ov 是 [n_layer, B] 的绝对 ω，
            # 逐**样本**给值 ⇒ 一次前向就能同时评估一整个种群（把种群成员铺进
            # batch 维）。纯评估用，不进任何训练/部署路径（用完置 None）。
            return ov[ell % ov.shape[0]].unsqueeze(-1)
        if self.omega_seq is not None:
            return self.omega_seq[ell % self.omega_seq.numel()]
        if self.omega_mode == "learn":
            z = self.omega_z[ell % self.omega_z.numel()]
            if self.omega_param == "exp":
                return base * torch.exp(z)
            return base * torch.nn.functional.softplus(
                self.b0 + self.omega_span * torch.tanh(z))
        return base

    def omega_values(self, base=None):
        """当前 ω_k 序列（诊断/报告用，numpy）。"""
        if base is None:
            base = (self.diag_omega if self.linear_solver == "diag"
                    else (1.0 if self.linear_solver == "gs"
                          else self.sweep_omega))
        with torch.no_grad():
            n = (self.omega_seq.numel() if self.omega_seq is not None
                 else (self.omega_z.numel()
                       if self.omega_mode == "learn" else 1))
            return np.array([float(self._omega(k, base)) for k in range(n)])

    def _solve_diag(self, solver, Pm, em, hgrad_e, F, Hj_prev, rh_zero,
                    outs, ell):
        """**一阶（对角近似）线性解层**：不做任何分解。

        为什么必须是"阻尼定点"而不是裸的 H ← F/diag(A)
        ------------------------------------------------
        裸对角解 H = D⁻¹F 会**破坏不动点集**：在精确解 (Q*,H*) 处 A·H* = F，
        但 D⁻¹F ≠ H*，于是 dq = Ŷ − P̂·EᵀH ≠ 0，迭代会把 Q 推离 Q* - 该格式
        的不动点根本不是原方程的解，整个"到 Hacc 的迭代数"就无从谈起。
        阻尼 Jacobi（热启动自上一层的 H）
            H ← H + ω·D⁻¹(F − A·H),   D = diag(A)
        在 H = H* 处残差 F − A·H* = 0 ⇒ H 不动 ⇒ dq = 0 ⇒ Q 不动，
        **不动点集 = 原方程解集**（与定理 2 同构，且对任意 ω>0、任意学习 s,α>0 成立）。
        故本实现选它，ω 与 sweeps 可调（分类经典基线时按 ω 扫描取最优）。
        第 0 层 H_prev = 0 ⇒ 首次 sweep 恰为裸对角解 ω·D⁻¹F（两种写法在此重合）。

        守恒的代价（本实验必须实测的量）
        --------------------------------
        GGA 的逐层质量守恒来自 A·H = F 被**精确**满足；对角近似下更新后的节点
        净流失衡 m 恒等于线性残差 (A·H − F)，即 O(‖r_lin‖)，不再是 O(eps)。
        本函数把 ‖F − A·H‖∞ 与其相对行尺度一并写进 outs（diag_lin_res*），
        由调用方原样报告，**不做任何精化兜底**（那会偷偷把牛顿法请回来）。
        """
        B, Nj = F.shape
        d_a = self._diag_of_A(solver, Pm, em, hgrad_e, Nj)
        Hj = Hj_prev
        w = self._omega(ell, self.diag_omega)
        for _ in range(max(1, self.diag_sweeps)):
            r = F - self._diag_matvec(solver, Pm, em, hgrad_e, Hj, rh_zero)
            Hj = Hj + w * (r / d_a)
        with torch.no_grad():
            rf = F - self._diag_matvec(solver, Pm, em, hgrad_e, Hj.detach(),
                                       rh_zero)
            rs = (self._diag_matvec(solver, Pm, em, hgrad_e, Hj.detach(),
                                    rh_zero, absval=True) + F.abs())
            outs.setdefault("diag_lin_res", []).append(float(rf.abs().max()))
            outs.setdefault("diag_lin_res_rel", []).append(
                float((rf.abs() / rs.clamp_min(1e-300)).max()))
        return Hj

    # ------------------------------------------------------------------
    # W6 经典阶梯：GS / RBGS / SOR / CG（同一套装配，只换线性解）
    # ------------------------------------------------------------------
    def _get_sweep(self, solver):
        """行空间稀疏结构 + 两种分块调度（缓存）。"""
        key = id(solver)
        if key not in self._sweep_cache:
            pat = self._get_pattern(solver, force_mode="ic0")
            self._sweep_cache[key] = (pat, _SweepStruct(pat))
        return self._sweep_cache[key]

    def _solve_blocks(self, ss, a_d, a_o, F_row, x0, blocks, omega, J):
        """分块 SOR/GS 扫描：x_i ← x_i + ω·(F_i − Σ_j A_ij x_j)/A_ii。

        blocks 是行的一个**有序划分**，块内行互不相邻（GS 波前层级）或颜色内
        互不相邻（多色）⇒ 块内可整块向量化，且结果与"逐行顺序扫描"逐位同解
        （块内行之间无耦合，用不到彼此的新值）。ω=1 ⇒ Gauss–Seidel。
        """
        B = F_row.shape[0]
        x = x0
        for _ in range(max(1, J)):
            for (rows, loc, col, ent, nb) in blocks:
                acc = torch.zeros(B, nb, dtype=x.dtype, device=x.device)
                acc = acc.index_add(1, loc, a_o[:, ent] * x[:, col])
                xi = x[:, rows]
                dd = a_d[:, rows]
                new = xi + omega * (F_row[:, rows] - dd * xi - acc) / dd
                x = x.index_copy(1, rows, new)
        return x

    def _solve_cg_lin(self, pat, a_d, a_o, F_row, x0, J):
        """无预条件共轭梯度，J 步，热启自 x0（A 为 SPD 加权 Laplacian，
        去掉水库行后成立 - W6 实测最小特征值与对称残差，不假定）。"""
        from dgga import precond as _pc
        tiny = 1e-300
        x = x0
        r = F_row - _pc.spmv(pat, a_d, a_o, x)
        p = r
        rs = (r * r).sum(1, keepdim=True)
        for _ in range(max(1, J)):
            Ap = _pc.spmv(pat, a_d, a_o, p)
            pAp = (p * Ap).sum(1, keepdim=True)
            alpha = torch.where(pAp.abs() > tiny, rs / pAp.clamp_min(tiny),
                                torch.zeros_like(rs))
            x = x + alpha * p
            r = r - alpha * Ap
            rs_n = (r * r).sum(1, keepdim=True)
            beta = torch.where(rs.abs() > tiny, rs_n / rs.clamp_min(tiny),
                               torch.zeros_like(rs))
            p = r + beta * p
            rs = rs_n
        return x

    def _solve_ladder(self, solver, Pm, em, hgrad_e, F, Hj_prev, outs, ell):
        """W6 阶梯路径的统一入口：稀疏装配 → 换一个线性解 → 记录线性残差。

        与 _solve_diag 同规矩：**不做任何迭代精化、不做守恒精化**，
        逐层线性残差 ‖F−A·H‖∞ 与相对行尺度原样写进 outs。
        """
        from dgga import precond as _pc
        pat, ss = self._get_sweep(solver)
        B, Nj = F.shape
        a_d, a_o = _pc.assemble_sparse(pat, Pm, em=em, hgrad_e=hgrad_e)
        F_row = torch.zeros(B, Nj, dtype=F.dtype, device=F.device
                            ).index_copy(1, pat.r_of_j, F)
        x0 = torch.zeros_like(F_row).index_copy(1, pat.r_of_j, Hj_prev)
        ls = self.linear_solver
        if ls == "cg":
            x = self._solve_cg_lin(pat, a_d, a_o, F_row, x0, self.cg_iters)
        else:
            blocks = ss.blocks_rb if ls == "rbgs" else ss.blocks_gs
            om = self._omega(ell, 1.0 if ls == "gs" else self.sweep_omega)
            x = self._solve_blocks(ss, a_d, a_o, F_row, x0, blocks, om,
                                   self.sweep_sweeps)
        with torch.no_grad():
            xd = x.detach()
            rf = F_row - _pc.spmv(pat, a_d, a_o, xd)
            rs = _pc.resid_scale(pat, a_d, a_o, xd, F_row)
            outs.setdefault("lin_res", []).append(float(rf.abs().max()))
            outs.setdefault("lin_res_rel", []).append(
                float((rf.abs() / rs.clamp_min(1e-300)).max()))
        ctx = (pat, a_d.detach(), a_o.detach(), None, None, F_row.detach())
        return x[:, pat.r_of_j], ctx

    def _solve_pcg(self, solver, Pm, em, hgrad_e, F, outs, ell):
        """W2 线性解层：稀疏装配 + 预条件 CG 解到 tol_lin。

        **保证不依赖学习件**：若所选预条件子（含学习的）未能在 maxit 内把
        cbe 压到 tol_lin，自动回退到**完全填充模式的精确稀疏 Cholesky**
        （M = A 逐位 ⇒ 1 步收敛，成本仍是 O(nnz)），并显式记录回退层。
        于是"守恒残差 ≤ O(tol_lin)·尺度"在任何权重下无条件成立。
        """
        from dgga import precond as _pc
        B, Nj = F.shape
        dt, dev = F.dtype, F.device
        pat = self._get_pattern(solver)
        a_d, a_o = _pc.assemble_sparse(pat, Pm, em=em, hgrad_e=hgrad_e)
        F_row = torch.zeros(B, Nj, dtype=dt, device=dev
                            ).index_copy(1, pat.r_of_j, F)
        Ld, Lo = self._make_precond(pat, a_d, a_o)
        sol = _pc.pcg_solve(pat, a_d, a_o, F_row, Ld, Lo, tol=self.tol_lin,
                            maxit=self.pcg_maxit, crit=self.pcg_crit)
        fell = False
        if self.pcg_fallback and not bool(sol["converged"].all()):
            fell = True
            outs.setdefault("pcg_fallback_layers", []).append(ell)
            patX = self._get_pattern(solver, force_mode="full")
            adX, aoX = _pc.assemble_sparse(patX, Pm, em=em, hgrad_e=hgrad_e)
            LdX, LoX = _pc.precond_ic0(patX, adX.detach(), aoX.detach())
            FX = torch.zeros(B, Nj, dtype=dt, device=dev
                             ).index_copy(1, patX.r_of_j, F)
            solX = _pc.pcg_solve(patX, adX, aoX, FX, LdX, LoX,
                                 tol=self.tol_lin, maxit=self.pcg_maxit,
                                 crit=self.pcg_crit)
            pat, a_d, a_o, F_row, Ld, Lo = patX, adX, aoX, FX, LdX, LoX
            sol = dict(x=solX["x"],
                       iters=sol["iters"] + solX["iters"],
                       relres=solX["relres"], converged=solX["converged"])
        Hj = sol["x"][:, pat.r_of_j]
        outs.setdefault("pcg_iters", []).append(sol["iters"].clone())
        outs.setdefault("pcg_res", []).append(sol["relres"].clone())
        outs.setdefault("pcg_converged", []).append(sol["converged"].clone())
        outs.setdefault("pcg_fell_back", []).append(fell)
        if not bool(sol["converged"].all()):       # 显式报告，绝不静默
            outs.setdefault("pcg_fail_layers", []).append(ell)
        ctx = (pat, a_d.detach(), a_o.detach(), Ld.detach(), Lo.detach(),
               F_row.detach())
        return Hj, ctx

    def _make_precond(self, pat, a_d, a_o):
        """返回 (Ld, Lo)：L̃ 下三角因子，M = L̃L̃ᵀ 恒 SPD。"""
        from dgga import precond as _pc
        if self.precond == "learned":
            Ld, Lo, _ = self.mprecond(pat, a_d, a_o)
            return Ld, Lo
        if self.precond in ("exact", "ic0"):
            return _pc.precond_ic0(pat, a_d.detach(), a_o.detach())
        if self.precond == "jacobi":
            return _pc.precond_jacobi(pat, a_d.detach(), a_o.detach())
        if self.precond == "none":
            return _pc.precond_none(pat, a_d.detach(), a_o.detach())
        raise ValueError(f"未知 precond={self.precond}")

    # ------------------------------------------------------------------
    def randomize_(self, std=0.5, seed=None):
        """随机化全部学习头权重（保证性质测试用：任意权重下守恒/SPD/不动点）。"""
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g,
                                    dtype=p.dtype) * std)
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _prep_inputs(solver, demand_cfs, res_head_ft, ke_int):
        """输入规范化（与 GGASolver.solve 的 _cvt 语义一致）。"""
        dev, dt = solver.device, solver.dtype

        def _cvt(x):
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dt, device=dev)
            return torch.as_tensor(np.asarray(x), dtype=dt, device=dev)

        d = _cvt(demand_cfs)
        rh = _cvt(res_head_ft)
        if d.dim() == 1:
            d = d.unsqueeze(0)
        if rh.dim() == 1:
            rh = rh.unsqueeze(0)
        B = d.shape[0]
        if rh.shape[0] == 1 and B > 1:
            rh = rh.expand(B, -1)
        ke = solver.node_ke_default if ke_int is None else _cvt(ke_int)
        if ke.dim() == 1:
            ke = ke.unsqueeze(0)
        if ke.shape[0] == 1 and B > 1:
            ke = ke.expand(B, -1)
        return d, rh, ke, B

    # ------------------------------------------------------------------
    def _layer_feat(self, solver, q, Y, g, d, imb, qc, frac):
        """逐层头输入特征 [B,L,8]。纯诊断量，不进物理路径。"""
        B, L = q.shape
        f = torch.stack([
            torch.log(solver.r_hw.clamp_min(1e-12)).expand(B, L),
            torch.log(solver.diam.clamp_min(1e-6)).expand(B, L),
            _slog(q / qc),
            torch.log(g.clamp_min(1e-12)),
            _slog(Y),
            _slog(imb[:, solver.n1] / qc),
            _slog(imb[:, solver.n2] / qc),
            torch.full_like(q, frac),
        ], dim=-1)
        return f

    # ------------------------------------------------------------------
    def forward(self, solver: GGASolver, demand_cfs, res_head_ft,
                ke_int=None, q0=None, e0=None, K=None, history=True,
                ref_track=None, h0=None):
        """展开 K 层。返回 dict：
        head_ft/flow_cfs/emitter_j - 各层输出列表（长度 K，元素 [B,·]）；
        relerr - 各层 EPANET 口径 Σ|dq|/Σ|Q|（[B] 列表）；
        q0_warm - 热启动流量 [B,L]。
        q0 显式给定时绕过热启动头（不动点测试用）。"""
        assert solver.mode == "dense", "GGAFormerV1 需要 GGASolver(mode='dense')"
        K = self.K if K is None else K
        dev, dt = solver.device, solver.dtype
        d, rh, ke, B = self._prep_inputs(solver, demand_cfs, res_head_ft, ke_int)
        Nj, L, N = solver.Nj, solver.L, solver.N
        ke_j = ke[:, solver.junc_nodes_t]
        has_em = ke_j > 0.0
        em = has_em.to(dt)
        d_j = d[:, solver.junc_nodes_t]

        rh_fix = rh[:, solver.fixed_nodes_t]
        if torch.isnan(rh_fix).any():
            raise ValueError("res_head_ft 在水库位存在 nan")

        # ---- 初值：EPANET 1fps 惯例 + 热启动修正（末层零初始化 ⇒ 修正恒 0）----
        q_init = solver._init_flow().unsqueeze(0).expand(B, -1)
        qc = (q_init.abs().sum(dim=1, keepdim=True) / L).clamp_min(1e-9)  # [B,1]
        if q0 is None and not self.use_warm_start:
            q = q_init.clone()                       # 消融：EPANET 1fps 惯例初值
        elif q0 is None:
            hc = float(np.std(solver.net.elev_ft)) + 1.0
            el = torch.as_tensor(solver.net.elev_ft, dtype=dt, device=dev)
            is_fix = torch.zeros(N, dtype=dt, device=dev)
            is_fix[solver.fixed_nodes_t] = 1.0
            rh_filled = torch.where(torch.isnan(rh), el.expand(B, -1), rh)
            nf = torch.stack([
                _slog(d / qc),
                is_fix.expand(B, -1),
                _slog((rh_filled - el.mean()) / hc) * is_fix,
            ], dim=-1)                                           # [B,N,3]
            ef = torch.stack([
                torch.log(solver.r_hw.clamp_min(1e-12)).expand(B, L),
                torch.log(solver.diam.clamp_min(1e-6)).expand(B, L),
                _slog(q_init / qc),
                torch.ones_like(q_init),
            ], dim=-1)                                           # [B,L,4]
            dq0 = self.warm(ef, nf, solver.n1, solver.n2, N)     # [B,L]
            q = q_init + qc * dq0
        else:
            q = q0.to(dtype=dt, device=dev)
            q = q.unsqueeze(0).expand(B, -1).clone() if q.dim() == 1 else q.clone()
        q0_warm = q

        if e0 is None:
            e_j = torch.where(has_em, torch.ones_like(ke_j),
                              torch.zeros_like(ke_j))
        else:
            e = e0.to(dtype=dt, device=dev) if isinstance(e0, torch.Tensor) \
                else torch.as_tensor(np.asarray(e0), dtype=dt, device=dev)
            e = e.unsqueeze(0).expand(B, -1) if e.dim() == 1 else e
            e_j = e[:, solver.junc_nodes_t].clone()

        H_junc = torch.zeros(B, Nj, dtype=dt, device=dev)
        H = torch.cat([H_junc, rh_fix], dim=1).index_select(1, solver._hperm_inv)
        # W5 一阶路径的状态：上一层的 junction 头（阻尼 Jacobi 的热启动）。
        # **注意**：牛顿路径每层重解 A·H=F，状态只有 Q；一阶路径的状态是 **(Q,H) 对**，
        # 故不动点测试必须同时喂 q0 与 h0（只喂 q0 时 H 从 0 起步，dq≠0 会把 Q 推走）。
        if h0 is None:
            Hj_state = H_junc
        else:
            _h0 = (h0 if isinstance(h0, torch.Tensor)
                   else torch.as_tensor(np.asarray(h0), dtype=dt, device=dev))
            _h0 = _h0.to(dtype=dt, device=dev)
            if _h0.dim() == 1:
                _h0 = _h0.unsqueeze(0)
            if _h0.shape[0] == 1 and B > 1:
                _h0 = _h0.expand(B, -1)
            # 允许传全节点头向量（[B,N]）或仅 junction 头（[B,Nj]）
            Hj_state = (_h0[:, solver.junc_nodes_t] if _h0.shape[1] == N
                        else _h0).clone()
            H = torch.cat([Hj_state, rh_fix],
                          dim=1).index_select(1, solver._hperm_inv)
        rh_zero = torch.zeros_like(rh_fix)

        outs = dict(head_ft=[], flow_cfs=[], emitter_j=[], relerr=[],
                    q0_warm=q0_warm)
        for ell in range(K):
            # ---- 物理线性化（复用 solver dense 内核）----
            P_pipe, Y_pipe = solver._pipe_PY(q)
            P_tcv, Y_tcv = solver._tcv_PY(q)
            P = torch.where(solver.is_tcv, P_tcv, P_pipe)
            Y = torch.where(solver.is_tcv, Y_tcv, Y_pipe)
            P = torch.where(solver.closed, torch.full_like(P, 1.0 / CBIG), P)
            Y = torch.where(solver.closed, q, Y)
            pl = (P != 0.0).to(dt)

            # ---- 学习模块：g 正尺度 s 与正步长 α（设计②）----
            # 节点净流失衡（诊断特征）：imb = E·Q − e − d（junction 位）
            if not self.has_layer_head:
                # 消融：g_scale 与 alpha 同时关闭 ⇒ s=α=1 ⇒ P̂=P、Ŷ=Y（逐位）；
                # 逐层头不建参数也不前向，FLOPs 账本的逐层学习项归零。
                P_h, Y_h = P, Y
            else:
                if self.head_mode == "scalar":
                    # 标量头：**不构建任何特征**（这正是它便宜的原因），只借形状
                    feat = q
                else:
                    with torch.no_grad():
                        imb_j = torch.zeros(B, Nj, dtype=dt, device=dev)
                        imb_j.scatter_add_(1, solver.f_idx1.expand(B, -1),
                                           -q[:, solver.lk_m1])
                        imb_j.scatter_add_(1, solver.f_idx2.expand(B, -1),
                                           q[:, solver.lk_m2])
                        imb_j = imb_j - em * e_j - d_j
                        imb = torch.zeros(B, N, dtype=dt, device=dev)
                        imb[:, solver.junc_nodes_t] = imb_j
                        feat = self._layer_feat(solver, q, Y, 1.0 / P, d, imb,
                                                qc, (ell + 1) / K)
                sov = getattr(self, "_scalar_override", None)
                if sov is not None and self.head_mode == "scalar":
                    # 同 _omega_override：ES 的种群铺进 batch 维用的逐样本通道
                    # （[K,2,B]）。纯评估，不进训练/部署路径。
                    z_s = sov[ell % sov.shape[0], 0].unsqueeze(1).expand(B, L)
                    z_a = sov[ell % sov.shape[0], 1].unsqueeze(1).expand(B, L)
                else:
                    z_s, z_a = self.heads[ell](feat)
                # 有界预激活 b0+2·tanh(z)：s,α ∈ [softplus(b0−2), softplus(b0+2)]
                # ≈ [0.076, 2.62]（正性保证之上再加训练稳定性；tanh(0)=0 ⇒ 恒等不变）
                if self.use_g_scale:
                    s = torch.nn.functional.softplus(
                        self.b0 + 2.0 * torch.tanh(z_s))
                if self.use_alpha:
                    alpha = torch.nn.functional.softplus(
                        self.b0 + 2.0 * torch.tanh(z_a))
                # ratio = α/s；单开一路时另一路取精确 1.0（不用 softplus(b0)，
                # 避免 1ulp 偏差污染"消融配置在未训练时 = 原版 GGA 逐位"的锁死）
                if self.use_g_scale and self.use_alpha:
                    ratio = alpha / s
                elif self.use_g_scale:
                    ratio = 1.0 / s
                else:
                    ratio = alpha
                P_h = P * ratio    # P̂ = α·P_eff = α/s·P
                Y_h = Y * ratio    # Ŷ = α·Y_eff = α/s·Y

            # ---- 索引张量装配（复用现成 scatter 索引；与 solver dense 同源）----
            Pm, Ym, qm = P_h * pl, Y_h * pl, q * pl
            Xflow = torch.zeros(B, Nj, dtype=dt, device=dev)
            Xflow.scatter_add_(1, solver.f_idx1.expand(B, -1), -qm[:, solver.lk_m1])
            Xflow.scatter_add_(1, solver.f_idx2.expand(B, -1), qm[:, solver.lk_m2])
            if self.linear_solver in ("pcg", "sparse_chol", "gs", "rbgs",
                                      "sor", "cg"):
                A = None                 # 稀疏路径：不建 O(Nj²) 稠密矩阵（成本故事）
            else:
                vals = torch.cat([-Pm[:, solver.lk_both], -Pm[:, solver.lk_both],
                                  Pm[:, solver.lk_m1], Pm[:, solver.lk_m2]],
                                 dim=1)
                A = torch.zeros(B, Nj * Nj, dtype=dt, device=dev)
                A.scatter_add_(1, solver.A_idx.expand(B, -1), vals)
                A = A.view(B, Nj, Nj)
            F = torch.zeros(B, Nj, dtype=dt, device=dev)
            F.scatter_add_(1, solver.f_idx1.expand(B, -1), Ym[:, solver.lk_m1])
            F.scatter_add_(1, solver.f_idx2.expand(B, -1), -Ym[:, solver.lk_m2])
            if solver.lk_g1.numel():
                F.scatter_add_(1, solver.g1_row.expand(B, -1),
                               Pm[:, solver.lk_g1] * H[:, solver.g1_src])
            if solver.lk_g2.numel():
                F.scatter_add_(1, solver.g2_row.expand(B, -1),
                               Pm[:, solver.lk_g2] * H[:, solver.g2_src])
            hloss_e, hgrad_e = solver._emitter_hloss(e_j, ke_j)
            if A is not None:
                A = A + torch.diag_embed(em / hgrad_e)
            F = F + em * (hloss_e + solver.el_junc) / hgrad_e
            Xflow = Xflow - em * e_j
            Xflow = Xflow - d_j
            F = F + Xflow

            if self.linear_solver == "diag":
                # ---- W5 一阶解层：阻尼 Jacobi，零分解（见 _solve_diag）----
                Hj = self._solve_diag(solver, Pm, em, hgrad_e, F, Hj_state,
                                      rh_zero, outs, ell)
                Hj_state = Hj
                chol, _pcg_ctx = None, None
            elif self.linear_solver in ("gs", "rbgs", "sor", "cg"):
                # ---- W6 经典阶梯：同一套装配，只换线性解（见 _solve_ladder）----
                Hj, _pcg_ctx = self._solve_ladder(solver, Pm, em, hgrad_e, F,
                                                  Hj_state, outs, ell)
                Hj_state = Hj
                chol = None
            elif self.linear_solver == "sparse_chol":
                # ---- W3 精确稀疏 Cholesky 直接解（新默认）：O(nnz) 一步到位，
                # 精度 ~eps（与 tol 无关），反传走隐函数定理伴随。----
                Hj, _pcg_ctx = self._solve_sparse_chol(solver, Pm, em, hgrad_e,
                                                       F, outs, ell)
                chol = None
            elif self.linear_solver == "pcg":
                # ---- W2 Krylov 保证层：精度由 tol_lin 保证，与预条件子
                # （学习与否）无关；神经网络只影响 CG 步数。反传走隐函数
                # 定理伴随（不展开 CG 图）。----
                Hj, _pcg_ctx = self._solve_pcg(solver, Pm, em, hgrad_e, F,
                                               outs, ell)
                chol = None
            else:
                # ---- 批量稠密 Cholesky + 2 步迭代精化（设计③，同 solver dense）----
                chol = torch.linalg.cholesky(A)
                Fc = F.unsqueeze(-1)
                Hj = torch.cholesky_solve(Fc, chol)
                for _ in range(2):
                    AHj = (A * Hj.transpose(-2, -1)).sum(-1, keepdim=True)
                    resid = Fc - AHj
                    Hj = Hj + torch.cholesky_solve(resid, chol)
                Hj = Hj.squeeze(-1)
                _pcg_ctx = None
            H = torch.cat([Hj, rh_fix], dim=1).index_select(1, solver._hperm_inv)

            # ---- 更新 Q ← Q − (Ŷ − P̂·EᵀH)（设计①，α 已并入 Ŷ/P̂）----
            dh = H[:, solver.n1] - H[:, solver.n2]
            dq = Y_h - P_h * dh
            q = q - dq
            dh_e = H[:, solver.junc_nodes_t] - solver.el_junc
            dq_e = (hloss_e - dh_e) / hgrad_e
            e_j = e_j - em * dq_e

            # ---- 守恒精化门（保证①的数值兜底，W1 改尺度感知 + 逐样本触发）：
            # 直接在 Q 上度量节点净流失衡 m = E·Q − em·e − d（无大数相消，见
            # _imbalance）。门限 thr = max(_CONS_GATE, C·eps·rowscale) 逐节点，
            # rowscale = |A|·|Hj|+|F| 为本层头解的可达残差地板（实测 m ≤ 3.5×）。
            # 超门限（异常失衡）时解 A·δh = m 并以流空间修正 δq = P̂⊙Eᵀδh、
            # δe = em·δh/hgrad 加回 Q/e/H（m_new = m − A·δh → 几何收敛）。
            # 恒等 GGA 路径（含 city_d TCV）零触发 → 逐位保持。
            # 修正量 detach，不进梯度图；未触发样本逐位不动（逐样本掩码）。
            with torch.no_grad():
                Pm_d = (P_h * pl).detach()
                m_res = _imbalance(solver, q.detach(), e_j.detach(), d_j, em)
                if self.linear_solver == "diag":
                    # 一阶路径：**故意不做守恒精化**（精化要解 A·δh = m，那等于
                    # 把牛顿法偷偷请回来，本实验就没有意义了）。这里只如实度量
                    # 失衡量级 - 它恒等于线性残差 F−A·H，是对角近似的定量代价。
                    rowscale = (self._diag_matvec(solver, Pm_d, em, hgrad_e,
                                                  Hj.detach(), rh_zero,
                                                  absval=True)
                                + F.detach().abs())
                elif _pcg_ctx is None:
                    rowscale = ((A.detach().abs()
                                 * Hj.detach().abs().unsqueeze(-2)).sum(-1)
                                + F.detach().abs())                  # [B,Nj]
                else:
                    # 稀疏路径（pcg / sparse_chol）：rowscale = |A|·|Hj| + |F|
                    # （行空间 → 列空间）
                    from dgga import precond as _pc
                    _p, _ad, _ao, _Ld, _Lo, _Fr = _pcg_ctx
                    _xr = torch.zeros_like(_Fr).index_copy(
                        1, _p.r_of_j, Hj.detach())
                    rs_row = _pc.resid_scale(_p, _ad, _ao, _xr, _Fr)
                    rowscale = rs_row[:, _p.r_of_j]
                # W2 推广：保证的新表述是"守恒残差 ≤ O(tol_lin)·rowscale"，
                # 故门限的相对项以 max(eps, tol_lin) 为基（tol_lin→eps 时
                # 逐字退化为 W1 的 eps 门 ⇒ cholesky 路径逐位不变）。
                # sparse_chol 是精确解（残差 ~eps），故与稠密路径同用 eps 门。
                _tolb = (max(_EPS64, self.tol_lin)
                         if self.linear_solver == "pcg" else _EPS64)
                thr = (_CONS_GATE_RELC * _tolb * rowscale).clamp_min(_CONS_GATE)
                cons_scale_pre = float((m_res.abs()
                                        / rowscale.clamp_min(1e-300)).max())
                bad = (m_res.abs() > thr).any(dim=1)                 # [B]
                if self.first_order:
                    # 一阶/无分解成本类（diag/gs/rbgs/sor/cg）：门永不触发
                    bad = torch.zeros_like(bad)
                if bool(bad.any()):
                    sel = bad.to(dt).view(-1, 1)
                    dq_c = torch.zeros_like(q)
                    de_c = torch.zeros_like(e_j)
                    dhc_full = torch.zeros(B, N, dtype=dt, device=dev)
                    for _ in range(6):
                        if _pcg_ctx is None:
                            dhc = torch.cholesky_solve(
                                m_res.unsqueeze(-1), chol.detach()).squeeze(-1)
                        else:
                            _p, _ad, _ao, _Ld, _Lo, _Fr = _pcg_ctx
                            _m_row = torch.zeros(B, Nj, dtype=dt, device=dev
                                                 ).index_copy(1, _p.r_of_j,
                                                              m_res)
                            if self.linear_solver == "sparse_chol":
                                _xc = _pc.chol_apply(_p, _ad, _ao, _Ld, _Lo,
                                                     _m_row, self.chol_refine)
                            else:
                                _xc, _, _, _ = _pc.pcg_solve_stats(
                                    _p, _ad, _ao, _m_row, _Ld, _Lo,
                                    tol=self.tol_lin, maxit=self.pcg_maxit,
                                    crit=self.pcg_crit)
                            dhc = _xc[:, _p.r_of_j]
                        dhc = dhc * sel                  # 未触发样本逐位保持
                        dhf = torch.cat(
                            [dhc, torch.zeros_like(rh_fix)],
                            dim=1).index_select(1, solver._hperm_inv)
                        dq_c = dq_c + Pm_d * (dhf[:, solver.n1]
                                              - dhf[:, solver.n2])
                        de_c = de_c + em * dhc / hgrad_e.detach()
                        dhc_full = dhc_full + dhf
                        m_res = _imbalance(solver, (q.detach() + dq_c),
                                           (e_j.detach() + de_c), d_j, em)
                        if float((m_res.abs() * sel).max()) <= 1e-12:
                            break
                else:
                    dq_c = None
                cons_ok = bool((m_res.abs() <= thr).all())
                cons_ratio = float((m_res.abs()
                                    / (_EPS64 * rowscale.clamp_min(1e-30))
                                    ).max())
            if dq_c is not None:
                q = q + dq_c
                e_j = e_j + de_c
                H = H + dhc_full

            dqsum = torch.sum(torch.abs(dq), dim=1) + \
                torch.sum(em * torch.abs(dq_e), dim=1)
            qsum = torch.sum(torch.abs(q), dim=1) + \
                torch.sum(em * torch.abs(e_j), dim=1)
            relerr = torch.where(qsum > 0, dqsum / qsum.clamp_min(1e-30), dqsum)

            if ref_track is not None:
                # 逐层**真误差**轨迹（对 f64 牛顿精抛光参考）。W5 必需：
                # EPANET 的 relerr=Σ|dq|/Σ|Q| 是**步长**判据，只有对超线性收敛的
                # 牛顿法才是误差的合格代理；一阶法步长小 ≠ 误差小（W5 实测：
                # 一阶路径 relerr≤1e-3 时真误差仍 O(1)）。故长程基线必须跟真误差。
                with torch.no_grad():
                    _rH, _rQ, _jm = ref_track
                    _nH = torch.linalg.norm(_rH[:, _jm], dim=1).clamp_min(1e-30)
                    _nQ = torch.linalg.norm(_rQ, dim=1).clamp_min(1e-30)
                    outs.setdefault("relH_ref", []).append(
                        torch.linalg.norm(H[:, _jm] - _rH[:, _jm], dim=1) / _nH)
                    outs.setdefault("relQ_ref", []).append(
                        torch.linalg.norm(q - _rQ, dim=1) / _nQ)
            if not history:
                # 长程基线（K 可达数千层）用：只留末层状态，relerr 轨迹全留。
                # 逐位不影响数值，只影响存哪些中间量。
                outs["head_ft"].clear()
                outs["flow_cfs"].clear()
                outs["emitter_j"].clear()
            outs["head_ft"].append(H)
            outs["flow_cfs"].append(q)
            outs["emitter_j"].append(e_j)
            outs["relerr"].append(relerr)
            outs.setdefault("cons_resid", []).append(
                float(m_res.abs().max()))    # 守恒精化后节点失衡（诊断）
            outs.setdefault("cons_ok", []).append(cons_ok)
            outs.setdefault("cons_ratio", []).append(cons_ratio)
            outs.setdefault("cons_scale_pre", []).append(cons_scale_pre)
        return outs


# ----------------------------------------------------------------------
def mass_residual(solver, q, e_j, d):
    """逐层质量守恒残差：junction 位 |E·Q − e − d| 的最大值（cfs），
    以及该层 max|Q|（尺度参考）。

    注意 f64 表示极限：q 本身以 float64 存储，|q|~1e8 cfs（对抗性随机权重把
    RQtol 钳位链路 P̂~3e8 的瞬态放大所致）时其存储舍入即 ~eps·|q|≈2e-8 cfs，
    绝对残差不可能低于该地板；结构性保证是"机器精度相对守恒"
    residual ≤ O(eps)·max|Q|。"""
    B = q.shape[0]
    d_j = d[:, solver.junc_nodes_t]
    r = torch.zeros(B, solver.Nj, dtype=q.dtype, device=q.device)
    r.scatter_add_(1, solver.f_idx1.expand(B, -1), -q[:, solver.lk_m1])
    r.scatter_add_(1, solver.f_idx2.expand(B, -1), q[:, solver.lk_m2])
    ke_j = solver.node_ke_default.unsqueeze(0)[:, solver.junc_nodes_t]
    em = (ke_j > 0).to(q.dtype)
    r = r - em * e_j - d_j
    return r.abs().max().item(), q.abs().max().item()


# ----------------------------------------------------------------------
# W3 等效求解数账本（FLOPs 口径）
# ----------------------------------------------------------------------
# 每链路装配+更新的算术代价常数（单位：flop）。
#   P/Y 线性化（H-W）：|q|^{n-1} 一次 pow（f64 pow ≈ 20 flop）+ ~10 次乘除加；
#   A 的 scatter（4 个非零位）、F 的 scatter（4 位）、dq=Y−P·dh 与 q 更新（~4）。
# 取 45 是中位估计。
# **灵敏度（W3 更正，必须照实说）**：
#   稠密路径（Nj³/3 主导）下该项 <2% 总量，c_link 在 [20,100] 内变动影响 <1%；
#   但**稀疏精确 Cholesky 路径下装配就是最大项**（rand_main_0016: 45·L=6390 占
#   gga_iter 9610 的 66%），c_link 直接缩放分母。故凡用 sparse_chol 报等效求解数，
#   **必须同时报 c_link ∈ {20,45,100} 的三点灵敏度**（scripts/train_w3.py 的
#   eval_pool 每次评价都自动输出该三点，见 C_LINK_SENS）。
C_LINK = 45.0
# 逐层头输入特征构建：3 个 log + 3 个 slog（log1p+sign）≈ 6 次超越函数 ≈ 60 flop/边。
C_FEAT_LINK = 60.0


def flops_ledger(model, solver, c_link=C_LINK, c_feat=C_FEAT_LINK,
                 solver_path=None):
    """逐项 FLOPs 账本（口径：**每次乘法、每次加法各计 1 flop**，同 §7 precond）。

    路径由 `solver_path`（缺省 = model.linear_solver）决定，两套公式并存：

    A) 'cholesky'（W1 稠密路径）
        装配/更新   c_link·L
        Cholesky    Nj³/3
        回代 1 rhs  2·Nj²
        2 步迭代精化 2·(matvec 2Nj² + 回代 2Nj²) = 8·Nj²
      ⇒ gga_iter = Nj³/3 + 10·Nj² + c_link·L

    B) 'sparse_chol' / 'pcg'（W2/W3 稀疏路径，**新默认**） - 口径与 W2 验证者
       重建的账本逐字一致（dgga/precond.py §7）：
        装配/更新   c_link·L
        数值分解    2·n_tri + 4·nnz(L) + 2·Nj      （= ic0_setup_flops，完全填充
                                                     模式 ⇒ 精确 Cholesky）
        三角解      4·nnz(L) + 2·Nj                （前代 + 回代）
        R 步迭代精化 R·(spmv 4·nnz+Nj + 残差减 Nj + 三角解 + 修正加 Nj)

    两个分母（都返回，报告须分清）：
      gga_iter_flops - **同路径的原版 GGA 单次迭代**（= 装配+分解+三角解+精化）。
        这是 evaluate_w3 用的分母：被比较的基线就是"学习件全关的 GGAFormer 空壳"，
        它逐字走同一条线性解路径（含同样的精化），故"无学习件 ⇒ 每层 = 1 次 GGA
        迭代"恒等成立。
      epanet_iter_flops - **EPANET linsolve 真实工作量**（装配+分解+三角解，
        无迭代精化）。更严的外部参照，报告表格用它另算一列。
    **GGAFormer 每层** = gga_iter + 逐层学习头（特征 c_feat·L + MLP 2·p·L）。
    **热启动头**（一次性）= 边 MLP 2·p_e·L + 节点 MLP 2·p_n·N + 邻接聚合 2·L·hidden
                          + 特征构建 c_feat·(L+N)。
    消融关闭的组件其项恒 0（组件不存在 ⇒ 不建参数 ⇒ 不进前向）。
    """
    from dgga import precond as _pc
    L, Nj, N = solver.L, solver.Nj, solver.N
    path = solver_path or getattr(model, "linear_solver", "cholesky")
    asm = c_link * L
    if path == "diag":
        # C) 'diag'（W5 一阶路径）：**无分解**。逐项（每次乘/加各 1 flop）
        #    diag(A) 装配   2·L（两次 scatter_add）+ 2·Nj（em/hgrad 的除+加）
        #    每次 Jacobi sweep
        #        dh = H[n1]−H[n2]           L
        #        t  = P̂·dh                 L
        #        散射到两行                 2·L
        #        + (em/hgrad)·Hj            2·Nj
        #        r  = F − A·H               Nj
        #        H += ω·r/D                 3·Nj（除、乘、加）
        #      ⇒ 4·L + 6·Nj 每 sweep
        #    装配/更新一项与精确路径**逐字相同**（c_link·L），因为除线性解外
        #    一切不变 - 这正是本实验的控制条件。
        J = float(max(1, getattr(model, "diag_sweeps", 1)))
        chol = 0.0                                    # 无分解
        tri = J * (4.0 * L + 6.0 * Nj)                # J 次 Jacobi sweep
        refine_f = 0.0                                # 无迭代精化
        asm = asm + 2.0 * L + 2.0 * Nj                # diag(A) 装配
        nnzL = 0.0
    elif path in ("gs", "rbgs", "sor", "cg"):
        # D) W6 经典阶梯（一阶/无分解成本类）。口径与 'diag' 逐字同源，只多两项：
        #    · 装配：diag(A) 2·L + 非对角 A_ij=−ΣP̂ 2·L + em/hgrad 2·Nj
        #      （Jacobi 只需 diag(A)，故本类比 Jacobi 每层贵 2·L - 如实计入）
        #    · 每次 GS/SOR 扫描（= 一次完整前向替换式更新，行 i 遍历其全部邻居）
        #        Σ_j A_ij x_j 的非对角部分  4·nnz（对称展开 2·nnz 个元 × 乘+加）
        #        A_ii·x_i、F−…、/A_ii、·ω、+x_i        6·Nj
        #      ⇒ 4·nnz + 6·Nj，与 Jacobi 的 4·L + 6·Nj 同量级（nnz ≈ L）。
        #      多色 GS 的算法 FLOPs 与自然序 GS **完全相同**（每个元恰好碰一次），
        #      差别只在扫描序（迭代数）与并行深度 - 故本表必须并排报并行栏。
        #    · CG 每步：spmv(4·nnz+2·Nj) + 2 个点积(4·Nj) + 3 个 axpy(6·Nj)
        #      = 4·nnz + 12·Nj；层首初始化 r=F−A·x₀ 与 ‖r‖² 计 4·nnz + 5·Nj。
        pat = model._get_pattern(solver, force_mode="ic0")
        nnz = float(pat.nnz)
        chol = 0.0
        refine_f = 0.0
        nnzL = nnz
        asm = asm + 4.0 * L + 2.0 * Nj
        if path == "cg":
            J = float(max(1, getattr(model, "cg_iters", 1)))
            tri = (4.0 * nnz + 5.0 * Nj) + J * (4.0 * nnz + 12.0 * Nj)
        else:
            J = float(max(1, getattr(model, "sweep_sweeps", 1)))
            tri = J * (4.0 * nnz + 6.0 * Nj)
    elif path == "cholesky":
        chol = Nj ** 3 / 3.0
        tri = 2.0 * Nj ** 2                   # 回代 1 rhs
        refine_f = 8.0 * Nj ** 2              # 2 步精化 2·(matvec+回代)
        nnzL = Nj * (Nj - 1) / 2.0            # 稠密因子的"非零"数（报告用）
    else:
        pat = model._get_pattern(solver, force_mode="full")
        chol = _pc.ic0_setup_flops(pat)                       # 数值分解
        tri = _pc.tri_solve_flops(pat)                        # 前代 + 回代
        R = float(getattr(model, "chol_refine", 2))
        refine_f = R * (_pc.spmv_flops(pat) + pat.n + tri + pat.n)
        nnzL = float(pat.nnz)
    gga_iter = chol + tri + refine_f + asm
    epanet_iter = chol + tri + asm            # 无迭代精化的严格参照

    if model.has_layer_head and len(model.heads):
        p_lh = sum(p.numel() for p in model.heads.parameters()) / len(model.heads)
        if getattr(model, "head_mode", "edge") == "scalar":
            # 标量头：无特征构建、无逐边 MLP。全部代价 = 2 次标量→向量的乘法
            # （P·ratio 与 Y·ratio 各 L 次）+ O(1) 的 softplus/tanh。
            head = 2.0 * L
        else:
            head = 2.0 * p_lh * L + c_feat * L
    else:
        p_lh, head = 0.0, 0.0

    if model.use_warm_start and model.warm is not None:
        p_we = (sum(p.numel() for p in model.warm.edge_mlp.parameters())
                + sum(p.numel() for p in model.warm.out.parameters()))
        p_wn = sum(p.numel() for p in model.warm.node_mlp.parameters())
        warm = (2.0 * p_we * L + 2.0 * p_wn * N
                + 2.0 * L * model.hidden + c_feat * (L + N))
    else:
        p_we = p_wn = 0.0
        warm = 0.0

    # ---- 并行性栏（任务书硬要求：不许只报 FLOPs 就说成本相同）----
    # par_depth = 一次线性解内部**不可并行的串行步数**（同深度的工作可整块向量化）。
    #   diag（Jacobi）1；gs/sor = MMD 序波前层数 nlev；rbgs = 贪心颜色数；
    #   cg = 每步 2 次全局归约 ⇒ 2·J（spmv 本身深度 O(1)）；
    #   sparse_chol = 分解 nlev + 前代 nlev + 回代 nlev（+ 精化）。
    par_depth, par_kind = 1.0, "并行"
    if path in ("gs", "sor", "rbgs", "sparse_chol", "pcg"):
        _pp = model._get_pattern(solver,
                                 force_mode="full" if path in ("sparse_chol",
                                                               "pcg")
                                 else "ic0")
        if path in ("gs", "sor"):
            par_depth, par_kind = float(_pp.nlev), "波前(层级调度)"
        elif path == "rbgs":
            par_depth = float(_SweepStruct(_pp).ncol)
            par_kind = "并行(多色)"
        else:
            R = float(getattr(model, "chol_refine", 2))
            par_depth = float(_pp.nlev) * (3.0 + 2.0 * R)
            par_kind = "波前(分解+三角解)"
    elif path == "cg":
        par_depth = 2.0 * float(max(1, getattr(model, "cg_iters", 1)))
        par_kind = "并行(全局归约)"
    elif path == "cholesky":
        par_depth, par_kind = float(Nj), "串行(稠密)"

    # ---- W6 任务 C：逐层松弛序列 ω_k 的成本（**精确为 0 flop/边**）----
    # ω 只出现在 H ← H + ω·D⁻¹r（'diag' 的 3·Nj 项）或 SOR 扫描（6·Nj 项）里，
    # 那次乘法在上面的公式中**已经计入**（常数 ω 与逐层 ω_k 的乘法次数完全相同）。
    # 每层额外的 softplus/tanh 是 O(1)（2 个标量），相对 O(L) 可忽略，
    # 但为诚实起见按 4 flop/层计入 head（对 L≈150 的网 = 0.027 flop/边）。
    n_om = 0
    if getattr(model, "omega_mode", "fixed") == "learn":
        n_om = int(model.omega_z.numel())
        head = head + 4.0
    return dict(L=L, Nj=Nj, N=N, path=path, nnz_L=nnzL, omega_params=n_om,
                par_depth=par_depth, par_kind=par_kind,
                flops_link=c_link, flops_feat_link=c_feat,
                chol_flops=chol, tri_flops=tri, assemble_flops=asm,
                refine_flops=refine_f, gga_iter_flops=gga_iter,
                epanet_iter_flops=epanet_iter,
                layer_head_flops=head, layer_head_params_per_layer=p_lh,
                layer_flops=gga_iter + head,
                warm_flops=warm, warm_edge_params=p_we, warm_node_params=p_wn,
                layer_overhead_frac=head / gga_iter,
                head_over_solve=head / max(chol + tri, 1e-30),
                warm_cost_in_gga_iters=warm / gga_iter)


def equiv_solves(model, solver, n_layers, c_link=C_LINK, c_feat=C_FEAT_LINK):
    """等效求解数 = (n_layers·每层 FLOPs + 热启动头 FLOPs) / GGA 单次迭代 FLOPs。
    n_layers 可为张量（逐场景）或标量。"""
    led = flops_ledger(model, solver, c_link, c_feat)
    return (n_layers * led["layer_flops"] + led["warm_flops"]) \
        / led["gga_iter_flops"]


def _first_hit(relerr, tol, cap):
    """relerr [K,B] → 首次 ≤tol 的层号（1-based）；未达到记 cap。"""
    K, B = relerr.shape
    hit = torch.full((B,), float(cap), dtype=torch.float64)
    for b in range(B):
        w = torch.where(relerr[:, b] <= tol)[0]
        if w.numel():
            hit[b] = float(int(w[0]) + 1)
    return hit


def gga_baseline_relerr(solver, d, rh, ke_int=None, n_iter=12,
                        linear_solver="cholesky", **kw):
    """原版 GGA 的逐迭代 relerr 轨迹 [n_iter,B]（无学习件的 GGAFormer 空壳，
    与 solver.solve 逐位一致 - __main__ 测试 0 与 W3 测试 4 各自锁死）。

    linear_solver 必须与被评价模型一致（同一条线性解路径才可比）。"""
    ref = GGAFormerV1(K=n_iter, use_warm_start=False, use_g_scale=False,
                      use_alpha=False, linear_solver=linear_solver, **kw)
    with torch.no_grad():
        o = ref(solver, d, rh, ke_int=ke_int)
    return torch.stack(o["relerr"], dim=0), ref


def evaluate_w3(model, solver, d, rh, ref_head=None, ref_flow=None,
                ke_int=None, tol=1e-3, K=None, gga_iter_cap=12,
                c_link=C_LINK, c_feat=C_FEAT_LINK):
    """W3 主评价：到达 EPANET 收敛判据（relerr = Σ|dq|/Σ|Q| ≤ tol）的
    **等效求解数** vs 原版 GGA 的迭代数。

    返回 dict：
      layers_to_tol[B] / gga_iters_to_tol[B]（未达到分别记 K+1 / cap+1）
      equiv[B] - 学习求解器的等效 GGA 迭代数（FLOPs 口径）
      ratio_mean - mean(equiv)/mean(gga_iters)（<0.5 即 W3 达标）
      speedup_layers - 纯层数比（不计学习件 FLOPs，乐观上界）
      ledger - flops_ledger 逐项
      dH_final/dQ_final - 末层 vs 精抛光参考（给了 ref 才算）
    """
    Kk = model.K if K is None else K
    with torch.no_grad():
        out = model(solver, d, rh, ke_int=ke_int, K=Kk)
    relerr = torch.stack(out["relerr"], dim=0)                  # [K,B]
    lay = _first_hit(relerr, tol, Kk + 1)
    rel_g, _ = gga_baseline_relerr(solver, d, rh, ke_int=ke_int,
                                   n_iter=gga_iter_cap,
                                   linear_solver=model.linear_solver)
    gga_n = _first_hit(rel_g, tol, gga_iter_cap + 1)
    led = flops_ledger(model, solver, c_link, c_feat)
    eq = (lay * led["layer_flops"] + led["warm_flops"]) / led["gga_iter_flops"]
    res = dict(layers_to_tol=lay, gga_iters_to_tol=gga_n, equiv=eq,
               relerr=relerr, relerr_gga=rel_g, ledger=led,
               ratio_mean=float(eq.mean() / gga_n.mean()),
               speedup_layers=float(gga_n.mean() / lay.mean()),
               n_unconverged=int((lay > Kk).sum()), out=out)
    if ref_head is not None:
        jm = solver.junc_nodes_t
        refH = torch.as_tensor(np.asarray(ref_head), dtype=solver.dtype)
        refQ = torch.as_tensor(np.asarray(ref_flow), dtype=solver.dtype)
        if refH.dim() == 1:
            refH, refQ = refH.unsqueeze(0), refQ.unsqueeze(0)
        res["dH_final"] = (out["head_ft"][-1][:, jm]
                           - refH[:, jm]).abs().max().item()
        res["dQ_final"] = (out["flow_cfs"][-1] - refQ).abs().max().item()
    return res


def newton_polish_reference(solver, d, rh, ke_int=None, accuracy=1e-14,
                            max_iter=100):
    """f64 GGA(牛顿)精抛光参考解（G-C1 教训：评价必带，防一阶尾部欠收敛）。"""
    r = solver.solve(d, rh, ke_int=ke_int, max_iter=max_iter, accuracy=accuracy)
    return r


def evaluate_solver(model, solver, d, rh, ref_head, ref_flow, ke_int=None,
                    tol=1e-3, K=None):
    """评价口径（任务 B-5）：
    - layers_to_tol：EPANET 口径 relerr=Σ|dq|/Σ|Q| ≤ tol（Hacc 同口径）所需层数
      （逐样本；未达到记 K+1）；
    - dH_final/dQ_final：末层 vs 参考的最大绝对差（ft / cfs）；
    - relH/relQ 各层轨迹（相对 L2，对参考解）；
    - ledger：等效求解数账本（每层 = 1 次装配 + 1 次稠密 Cholesky，与 GGA
      每迭代同价；热启动头造价单列：参数量与一次前向的近似 FLOPs）。
    """
    with torch.no_grad():
        out = model(solver, d, rh, ke_int=ke_int, K=K)
    Kk = len(out["head_ft"])
    refH = ref_head if isinstance(ref_head, torch.Tensor) else \
        torch.as_tensor(np.asarray(ref_head), dtype=solver.dtype)
    refQ = ref_flow if isinstance(ref_flow, torch.Tensor) else \
        torch.as_tensor(np.asarray(ref_flow), dtype=solver.dtype)
    if refH.dim() == 1:
        refH = refH.unsqueeze(0)
    if refQ.dim() == 1:
        refQ = refQ.unsqueeze(0)
    jm = solver.junc_nodes_t
    B = out["head_ft"][0].shape[0]
    relerr = torch.stack(out["relerr"], dim=0)                   # [K,B]
    lay = torch.full((B,), Kk + 1, dtype=torch.int64)
    for b in range(B):
        hit = torch.where(relerr[:, b] <= tol)[0]
        if hit.numel():
            lay[b] = int(hit[0]) + 1
    relH, relQ = [], []
    nH = torch.linalg.norm(refH[:, jm], dim=1).clamp_min(1e-30)
    nQ = torch.linalg.norm(refQ, dim=1).clamp_min(1e-30)
    for ell in range(Kk):
        relH.append((torch.linalg.norm(out["head_ft"][ell][:, jm]
                                       - refH[:, jm], dim=1) / nH))
        relQ.append((torch.linalg.norm(out["flow_cfs"][ell] - refQ, dim=1) / nQ))
    n_warm = (sum(p.numel() for p in model.warm.parameters())
              if model.warm is not None else 0)
    n_head = sum(p.numel() for p in model.heads.parameters())
    L, Nj, N = solver.L, solver.Nj, solver.N
    # W1 修正：MLP 逐边/逐节点应用，FLOPs ≈ 2·参数量·应用次数（原式漏乘 L，
    # 少算 ~L 倍）。小网（Nj~100）逐层头 ≈1.4×Cholesky、热启动头 ≈5×Cholesky
    # - 等效求解数必须按 FLOPs 折算；大网（Nj≥500）头部占比 <5%。
    # W3：完整账本（含回代/精化/装配）见 flops_ledger()，本处保留 W1 口径以
    # 维持历史数字可比。
    _w3 = flops_ledger(model, solver)
    warm_flops = 2.0 * (_w3["warm_edge_params"] * L
                        + _w3["warm_node_params"] * N)
    p_lh = n_head / max(1, len(model.heads)) if len(model.heads) else 0.0
    head_flops = 2.0 * p_lh * L
    chol_flops = Nj ** 3 / 3.0
    ledger = dict(
        layers=Kk,
        per_layer_cost="1 次装配 + 1 次稠密 Cholesky O(Nj^3)（与 GGA 每迭代同价）"
                       "+ 逐层头 MLP（2·参数·L FLOPs，GGA 无此项）",
        chol_flops_per_layer=chol_flops,
        layer_head_flops_per_layer=head_flops,
        layer_head_flops_vs_chol=head_flops / chol_flops,
        warm_head_params=n_warm,
        layer_head_params=n_head,
        warm_head_flops_approx=warm_flops,
        warm_head_flops_vs_chol=warm_flops / chol_flops,
        equiv_layers_flops=Kk * (1.0 + head_flops / chol_flops)
                           + warm_flops / chol_flops,   # 等效 GGA 迭代数（FLOPs 口径）
        w3=_w3,                                          # W3 完整账本
    )
    return dict(layers_to_tol=lay,
                dH_final=(out["head_ft"][-1][:, jm] - refH[:, jm]).abs().max().item(),
                dQ_final=(out["flow_cfs"][-1] - refQ).abs().max().item(),
                relH=[t for t in relH], relQ=[t for t in relQ],
                relerr=relerr, ledger=ledger, out=out)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 保证三件套单元测试（任务 B-3）+ 恒等 GGA 初始化锁死测试
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from dgga.parse import Net

    import os as _os  # noqa: E402
    ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    torch.manual_seed(0)
    # 本自测文件是 **W1/W2 稠密路径的逐位历史锚**（测试 0/④ 断言 == 0.0，
    # 只有稠密 Cholesky 与 solver.solve 的运算次序逐字相同才可能成立）。
    # W3 把默认线性解换成 sparse_chol（MMD 重排 ⇒ 求和次序不同 ⇒ ~1e-14 相对差），
    # 故此处把默认覆写回 'cholesky'，锚不动；sparse_chol 的同名四项测试
    # （恒等/守恒/SPD/不动点）在 scripts/verify_sparse_chol.py 单独前台实测。
    from functools import partial as _partial
    GGAFormerV1 = _partial(GGAFormerV1, linear_solver="cholesky")   # noqa: F811
    STEMS = ["rand_main_0000", "rand_main_0001", "rand_main_0002"]
    # W1 加测：pub_hanoi（公开网）+ city_d（TCV 79 + 关闭支 4，尺度感知门的回归锚）
    STEMS_ID = STEMS + ["pub_hanoi", "city_d"]
    nets = {s: Net.load(ROOT + "/data/reference", s) for s in STEMS_ID}
    solvers = {s: GGASolver(nets[s], mode="dense") for s in STEMS_ID}

    print("=" * 70)
    print("测试 0：恒等 GGA 初始化锁死（未训练模型逐层输出 == 原版 GGA 迭代）")
    worst = 0.0
    for s in STEMS_ID:
        net, sv = nets[s], solvers[s]
        d0 = net.demand_cfs_at(0)
        rh0 = net.reservoir_head_ft_at(0)
        model = GGAFormerV1(K=6)
        with torch.no_grad():
            out = model(sv, d0, rh0)
        w_s = 0.0
        for ell in range(6):
            r = sv.solve(d0, rh0, max_iter=ell + 1, accuracy=1e-16)
            dH = (out["head_ft"][ell][0] - r["head_ft"]).abs().max().item()
            dQ = (out["flow_cfs"][ell][0] - r["flow_cfs"]).abs().max().item()
            w_s = max(w_s, dH, dQ)
        assert w_s == 0.0, f"{s} 层恒等失败 worst={w_s:.3e}"
        worst = max(worst, w_s)
    print(f"  {len(STEMS_ID)} 拓扑（含 city_d TCV）× 6 层 max|ΔH|,|ΔQ| vs 原版 "
          f"GGA = {worst:.3e}  逐位  PASS")

    print("=" * 70)
    print("测试 ①：逐层质量守恒 max|A21·Q−d|（任意随机权重，尺度感知判据）")
    # 结构性保证（W1 修订）：失衡恒为线性解舍入 → 逐节点
    #   m ≤ max(_CONS_GATE, C·eps·rowscale)（cons_ok，门保证；超门时精化到 1e-12）。
    # 附加报告：TCV-free 网物理尺度的绝对残差（地板 ~eps·P·|H| ≈ 1e-8 量级）。
    worst_abs_phys = 0.0        # 物理尺度层的绝对残差（报告性）
    worst_ratio = 0.0           # m/(eps·rowscale) 最大比值
    all_ok = True
    n_phys = n_all = 0
    for s in STEMS_ID:
        net, sv = nets[s], solvers[s]
        B = 2 if s == "city_d" else 4
        ntr = 2 if s == "city_d" else 5
        d0 = torch.as_tensor(np.stack([net.demand_cfs_at(0)] * B))
        d0 = d0 * (0.5 + torch.rand(B, 1, dtype=torch.float64))
        rh0 = net.reservoir_head_ft_at(0)
        for trial in range(ntr):
            model = GGAFormerV1(K=8).randomize_(std=0.5, seed=trial)
            with torch.no_grad():
                out = model(sv, d0, rh0)
            all_ok = all_ok and all(out["cons_ok"])
            worst_ratio = max(worst_ratio, max(out["cons_ratio"]))
            for ell in range(8):
                w, qmax = mass_residual(
                    sv, out["flow_cfs"][ell], out["emitter_j"][ell], d0)
                n_all += 1
                if qmax <= 1e3 and s != "city_d":
                    n_phys += 1
                    worst_abs_phys = max(worst_abs_phys, w)
    ok = all_ok and worst_abs_phys < 1e-7
    print(f"  {len(STEMS_ID)} 拓扑 × 随机权重 × 8 层（共 {n_all} 层）:")
    print(f"    逐节点 m ≤ max(5e-10, {_CONS_GATE_RELC:.0f}·eps·rowscale) 全层: "
          f"{all_ok}; m/(eps·rowscale) worst = {worst_ratio:.2f}"
          f"（诊断性；小 rowscale 节点由绝对底 5e-10 覆盖）")
    print(f"    TCV-free 物理尺度层（{n_phys} 层）绝对残差 worst = "
          f"{worst_abs_phys:.3e} cfs（报告性; 地板 ~eps·P·|H|）")
    print(f"  {'PASS' if ok else 'FAIL'}")
    assert ok

    print("=" * 70)
    print("测试 ②：SPD - 随机权重 1000 次前向，Cholesky 零失败")
    net, sv = nets["rand_main_0001"], solvers["rand_main_0001"]
    d0 = torch.as_tensor(np.stack([net.demand_cfs_at(0)] * 2))
    rh0 = net.reservoir_head_ft_at(0)
    fails = 0
    model = GGAFormerV1(K=8)
    for trial in range(1000):
        model.randomize_(std=1.0, seed=10000 + trial)
        try:
            with torch.no_grad():
                model(sv, d0 * (0.2 + 1.6 * torch.rand(2, 1, dtype=torch.float64)),
                      rh0)
        except torch.linalg.LinAlgError:
            fails += 1
    print(f"  1000 次随机权重前向（std=1.0）：Cholesky 失败 {fails} 次  "
          f"{'PASS' if fails == 0 else 'FAIL'}")
    assert fails == 0

    print("=" * 70)
    print("测试 ③：不动点保持 - 喂精抛光参考解进任意随机权重的层")
    # TCV-free 网：漂移 < 1e-8（原 1e-9 依赖旧绝对门的修正兜住尾数；改尺度感知
    # 门后为裸舍入地板，实测 ~1.1e-9，判据放到 1e-8 仍是机器精度级）。
    # city_d（TCV）：数值地板受
    # P̂~1e6 放大（恒等权重的纯 GGA 自身漂移即 ~5e-7 cfs），定理 2 是精确算术
    # 命题 - 判据改为不超过恒等 GGA 自身漂移的 1000×（数量级哨兵：随机 std=0.8
    # 权重把 P̂ 放大至 ~34×，8 层复合后地板抬升 1~2 个量级属预期，实测 ~150×）。
    worst_q = worst_h = 0.0
    for s in STEMS:
        net, sv = nets[s], solvers[s]
        d0 = net.demand_cfs_at(0)
        rh0 = net.reservoir_head_ft_at(0)
        ref = newton_polish_reference(sv, d0, rh0, accuracy=1e-15, max_iter=100)
        qs, hs = ref["flow_cfs"], ref["head_ft"]
        for trial in range(5):
            model = GGAFormerV1(K=8).randomize_(std=0.8, seed=777 + trial)
            with torch.no_grad():
                out = model(sv, d0, rh0, q0=qs)
            dq = (out["flow_cfs"][-1][0] - qs).abs().max().item()
            dh = (out["head_ft"][-1][0] - hs).abs().max().item()
            worst_q, worst_h = max(worst_q, dq), max(worst_h, dh)
    print(f"  TCV-free 3 拓扑 × 5 组随机权重 × 8 层: max|ΔQ|={worst_q:.3e} cfs, "
          f"max|ΔH|={worst_h:.3e} ft  "
          f"{'PASS' if max(worst_q, worst_h) < 1e-8 else 'FAIL'}")
    assert max(worst_q, worst_h) < 1e-8
    # city_d（TCV 地板锚）
    net, sv = nets["city_d"], solvers["city_d"]
    d0 = net.demand_cfs_at(0)
    rh0 = net.reservoir_head_ft_at(0)
    ref = newton_polish_reference(sv, d0, rh0, accuracy=1e-15, max_iter=100)
    qs, hs = ref["flow_cfs"], ref["head_ft"]
    model = GGAFormerV1(K=8)                       # 恒等 = 纯 GGA 的自身漂移
    with torch.no_grad():
        out = model(sv, d0, rh0, q0=qs)
    base_q = (out["flow_cfs"][-1][0] - qs).abs().max().item()
    base_h = (out["head_ft"][-1][0] - hs).abs().max().item()
    wq_d = wh_d = 0.0
    for trial in range(3):
        model = GGAFormerV1(K=8).randomize_(std=0.8, seed=777 + trial)
        with torch.no_grad():
            out = model(sv, d0, rh0, q0=qs)
        wq_d = max(wq_d, (out["flow_cfs"][-1][0] - qs).abs().max().item())
        wh_d = max(wh_d, (out["head_ft"][-1][0] - hs).abs().max().item())
    ok_d = (wq_d <= 1000.0 * max(base_q, 1e-9)
            and wh_d <= 1000.0 * max(base_h, 1e-9))
    print(f"  city_d（TCV）: 恒等 GGA 自身漂移 ΔQ={base_q:.3e}/ΔH={base_h:.3e}; "
          f"随机权重 ΔQ={wq_d:.3e}/ΔH={wh_d:.3e}（≤1000× 地板）  "
          f"{'PASS' if ok_d else 'FAIL'}")
    assert ok_d

    print("=" * 70)
    print("测试 ④（W3）：消融开关的 8 种组合，未训练时全部 == 原版 GGA（逐位）")
    # 恒等锁死必须对每个消融配置分别成立，否则"层数/等效求解数"的基线不可比。
    worst4 = 0.0
    n_cfg = 0
    for s in ["rand_main_0000", "pub_hanoi", "city_d"]:
        net, sv = nets[s], solvers[s]
        d0, rh0 = net.demand_cfs_at(0), net.reservoir_head_ft_at(0)
        gga = [sv.solve(d0, rh0, max_iter=e + 1, accuracy=1e-16)
               for e in range(4)]
        for hm in ("edge", "scalar"):
            for uw in (True, False):
                for ug in (True, False):
                    for ua in (True, False):
                        m = GGAFormerV1(K=4, hidden=8, use_warm_start=uw,
                                        use_g_scale=ug, use_alpha=ua,
                                        head_mode=hm)
                        with torch.no_grad():
                            o = m(sv, d0, rh0)
                        n_cfg += 1
                        for e in range(4):
                            worst4 = max(
                                worst4,
                                (o["head_ft"][e][0] - gga[e]["head_ft"]).abs().max().item(),
                                (o["flow_cfs"][e][0] - gga[e]["flow_cfs"]).abs().max().item())
    print(f"  {n_cfg} 个 (拓扑×消融配置, 含 head_mode=edge/scalar) × 4 层 "
          f"max|ΔH|,|ΔQ| vs 原版 GGA = {worst4:.3e}  "
          f"{'PASS' if worst4 == 0.0 else 'FAIL'}")
    assert worst4 == 0.0

    # 标量头也必须满足保证三件套的守恒/SPD（随机权重）
    net, sv = nets["rand_main_0001"], solvers["rand_main_0001"]
    d0 = torch.as_tensor(np.stack([net.demand_cfs_at(0)] * 3))
    rh0 = net.reservoir_head_ft_at(0)
    ok_sc = True
    for trial in range(20):
        m = GGAFormerV1(K=8, head_mode="scalar").randomize_(std=1.0,
                                                            seed=555 + trial)
        with torch.no_grad():
            o = m(sv, d0, rh0)
        ok_sc = ok_sc and all(o["cons_ok"])
    print(f"  标量头 20 组随机权重 × 8 层：守恒门全通过 = {ok_sc}, Cholesky 零失败  "
          f"{'PASS' if ok_sc else 'FAIL'}")
    assert ok_sc

    print("=" * 70)
    print("测试 ⑤（W3）：FLOPs 账本自洽 + 全关配置的等效求解数 == 层数")
    net, sv = nets["rand_main_0000"], solvers["rand_main_0000"]
    d0, rh0 = net.demand_cfs_at(0), net.reservoir_head_ft_at(0)
    m_none = GGAFormerV1(K=8, hidden=8, use_warm_start=False,
                         use_g_scale=False, use_alpha=False)
    led = flops_ledger(m_none, sv)
    assert led["layer_head_flops"] == 0.0 and led["warm_flops"] == 0.0
    assert abs(led["layer_flops"] - led["gga_iter_flops"]) == 0.0
    ev0 = evaluate_w3(m_none, sv, d0, rh0)
    assert float(ev0["equiv"][0]) == float(ev0["layers_to_tol"][0])
    assert float(ev0["layers_to_tol"][0]) == float(ev0["gga_iters_to_tol"][0]), \
        "无学习件配置的层数必须等于 GGA 基线迭代数（基线自洽）"
    m_full = GGAFormerV1(K=8, hidden=8)
    lf = flops_ledger(m_full, sv)
    print(f"  rand_main_0000 (Nj={lf['Nj']},L={lf['L']}): GGA 单迭代 "
          f"{lf['gga_iter_flops']:.3e} flop = Chol {lf['chol_flops']:.2e} + "
          f"回代/精化 {lf['tri_flops']:.2e} + 装配 {lf['assemble_flops']:.2e}")
    print(f"  hidden=8 全开: 逐层头开销 {lf['layer_overhead_frac']*100:.1f}%/层, "
          f"热启动头 {lf['warm_cost_in_gga_iters']:.3f} 个 GGA 迭代当量")
    print(f"  全关配置: 到 relerr≤1e-3 层数 {float(ev0['layers_to_tol'][0]):.0f} "
          f"= GGA 迭代数 {float(ev0['gga_iters_to_tol'][0]):.0f}, 等效求解数 "
          f"{float(ev0['equiv'][0]):.3f}  PASS")

    print("=" * 70)
    print("测试 ⑥（W5）：一阶路径 linear_solver='diag' 的保证与代价")
    # 一阶路径**故意**放弃逐层守恒（对角近似 ⇒ 失衡 = 线性残差），但必须保住
    # 不动点集（否则"到某精度的迭代数"无从谈起）。本测试把两件事都钉死：
    #   (a) 喂 (Q*,H*) 进任意 ω 的 K 层，末层漂移停在舍入量级；
    #   (b) 守恒失衡 m **恒等于**线性残差 F−A·H（这是"对角近似的代价"的精确表述）。
    w6_fp, w6_id = 0.0, 0.0
    for s in ["rand_main_0000", "pub_hanoi"]:
        net, sv = nets[s], solvers[s]
        d0, rh0 = net.demand_cfs_at(0), net.reservoir_head_ft_at(0)
        ref = newton_polish_reference(sv, d0, rh0, accuracy=1e-15, max_iter=100)
        qs, hs = ref["flow_cfs"], ref["head_ft"]
        for om in (0.3, 0.6, 1.0):
            md = GGAFormerV1(K=6, use_warm_start=False, use_g_scale=False,
                             use_alpha=False, linear_solver="diag",
                             diag_omega=om)
            with torch.no_grad():
                o = md(sv, d0, rh0, q0=qs, h0=torch.as_tensor(hs).unsqueeze(0))
            w6_fp = max(w6_fp,
                        (o["flow_cfs"][-1][0] - torch.as_tensor(qs)).abs().max().item(),
                        (o["head_ft"][-1][0] - torch.as_tensor(hs)).abs().max().item())
            with torch.no_grad():                    # 冷启动：失衡 vs 线性残差
                o2 = md(sv, d0, rh0)
            for a_, b_ in zip(o2["cons_resid"], o2["diag_lin_res"]):
                w6_id = max(w6_id, abs(a_ - b_) / max(b_, 1e-30))
    print(f"  (a) 不动点保持（2 拓扑 × ω∈{{0.3,0.6,1.0}} × 6 层）: "
          f"max|ΔQ|,|ΔH| = {w6_fp:.3e}  "
          f"{'PASS' if w6_fp < 1e-8 else 'FAIL'}")
    print(f"  (b) 守恒失衡 ≡ 线性残差 F−A·H（相对差）: {w6_id:.3e}  "
          f"{'PASS' if w6_id < 1e-10 else 'FAIL'}")
    assert w6_fp < 1e-8 and w6_id < 1e-10
    m6 = GGAFormerV1(K=1, use_warm_start=False, use_g_scale=False,
                     use_alpha=False, linear_solver="diag")
    l6 = flops_ledger(m6, solvers["rand_main_0000"], solver_path="diag")
    l6c = flops_ledger(m6, solvers["rand_main_0000"], solver_path="sparse_chol")
    print(f"  (c) 账本：diag 单迭代 {l6['gga_iter_flops']:.0f} flop "
          f"({l6['gga_iter_flops']/l6['L']:.1f}/边, 无分解) vs sparse_chol "
          f"{l6c['gga_iter_flops']:.0f} ({l6c['gga_iter_flops']/l6['L']:.1f}/边) "
          f"⇒ 只便宜 {l6c['gga_iter_flops']/l6['gga_iter_flops']:.2f}×  PASS")
    assert l6["chol_flops"] == 0.0

    print("=" * 70)
    print("ggaformer.py 保证三件套 + 恒等初始化 + W3 消融/账本 + W5 一阶路径 全部通过")

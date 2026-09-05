# -*- coding: utf-8 -*-
"""dgga.precond - W2：可学习不完全 Cholesky 预条件子 + Krylov 保证层。

架构心脏（不可妥协）
--------------------
注意力**不替换**精确线性解 - 近似解会同时毁掉守恒（A·H=F 不再成立）与不动点。
正确位置是**预条件子**：

    掩码注意力 → 下三角因子 L̃（填充掩码限定的稀疏模式）→ M = L̃·L̃ᵀ ≈ A
    外套预条件共轭梯度（PCG），解 A·H = F 到相对残差 ≤ tol_lin

于是神经网络**只影响 CG 步数（速度），不影响解的正确性（精度由 tol_lin 保证）**。
与 W1 同一哲学：学习被限制在"只能加速、不能算错"的位置。

保证的新表述（论文用）
- 守恒残差 ≤ O(tol_lin)·尺度；tol_lin → 0 时退化为 W1 的精确 Cholesky 层。
- M 恒正定：L̃ 对角 = L0_diag·softplus(·) > 0 严格为正 ⇒ L̃ 非奇异下三角
  ⇒ M = L̃L̃ᵀ SPD **无条件成立**（与学习权重无关，无 breakdown 可能）。
- 恒等初始化：输出头零初始化 ⇒ L̃ ≡ IC(0)（A 自身模式上的精确不完全 Cholesky）。
  选 IC(0) 而非 Jacobi 作恒等起点的理由：本问题的 A 是 **Stieltjes 矩阵**
  （SPD + 非正非对角元 −P̂ + 弱对角占优，接定水头节点的行严格占优），
  由 Meijerink–van der Vorst (1977) 定理，M-矩阵的 IC(0) **必然存在且不 breakdown**。
  即"未训练时 = 该类问题上有存在性保证的最强 O(nnz) 经典预条件子"。

梯度的诚实说明（重要架构结论）
- 因为解 H 由 tol_lin 保证（与 M 无关），**∂H/∂θ_precond ≡ 0**：预条件子拿不到
  来自解误差损失的梯度。它必须用显式"加速损失"训练（见 pcg_reduction_loss）。
- 线性层对上游物理量的反传走**隐函数定理伴随**（不展开 CG 的图）：
  x = A⁻¹b ⇒ λ = A⁻ᵀ(∂L/∂x) = A⁻¹(∂L/∂x)（A 对称，同一个 M 复用），
  ∂L/∂b = λ，∂L/∂A_ii = −λ_i x_i，∂L/∂A_ij(对称存一份) = −(λ_i x_j + λ_j x_i)。

依赖：dgga/smatrix.py（EpanetSmatrix：MMD 重排 Row/Order + 完全填充结构
XLNZ/NZSUB）。FillPattern 是**最小版符号分解** -
TODO(任务A): dgga/symbolic.py 就绪后把 FillPattern 的模式提取合并过去。
"""

import math

import numpy as np
import torch
import torch.nn as nn

B0 = math.log(math.e - 1.0)          # softplus(B0) = 1.0（与 W1 同一常数）
_EPS64 = float(np.finfo(np.float64).eps)


def _zero_last(seq):
    """把 Sequential 的末层 Linear 权重与偏置清零（恒等初始化用）。"""
    last = seq[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


# ======================================================================
#  1. 符号层：填充模式 + 消元树 + 层级调度
# ======================================================================
class FillPattern:
    """稀疏 Cholesky 的符号结构（MMD 行空间，0 基）。

    mode='ic0' - 模式 = A 自身下三角非零（零填充，IC(0)）；
    mode='full' - 模式 = EPANET 符号分解的完全填充（XLNZ/NZSUB）⇒ 精确稀疏 Cholesky。

    产出（全部 torch long 缓冲）：
      ent_row/ent_col [nnz]     下三角非对角元 (i>j) 的行/列
      lvl [n], nlev             列依赖 DAG 的层级（三角解/因子分解的并行调度深度）
      ent_by_rowlvl[l]          row 位于第 l 层的元索引（算 diag 与前代用）
      ent_by_collvl[l]          col 位于第 l 层的元索引（算列外元与回代用）
      tri_t/tri_e1/tri_e2 (按 lvl[col(t)] 分组)  L[i,j] -= L[i,k]·L[j,k] 的三元组
      etree_parent/etree_depth  消元树（位置编码用）
    """

    def __init__(self, n, link_rows, xlnz=None, nzsub=None, mode="ic0",
                 device="cpu", dtype=torch.float64):
        self.n = int(n)
        self.mode = mode
        self.device = device
        self.dtype = dtype
        n_ = self.n

        # ---- A 自身的下三角非零（去平行管、去自环）----
        lr = np.asarray(link_rows, dtype=np.int64)          # [L,2]，−1 = 非 junction
        both = (lr[:, 0] >= 0) & (lr[:, 1] >= 0)
        ri = np.maximum(lr[both, 0], lr[both, 1])
        rj = np.minimum(lr[both, 0], lr[both, 1])
        ok = ri != rj                                       # 自环单独处理
        a_pairs = np.stack([ri[ok], rj[ok]], axis=1)

        # ---- 模式 ----
        if mode == "ic0":
            pat = np.unique(a_pairs, axis=0) if a_pairs.size else \
                np.zeros((0, 2), dtype=np.int64)
        elif mode == "full":
            assert xlnz is not None and nzsub is not None
            ps = []
            for jj in range(1, n_ + 1):
                for p in range(xlnz[jj], xlnz[jj + 1]):
                    ps.append((nzsub[p] - 1, jj - 1))       # (row i, col j), i>j
            pat = np.asarray(sorted(ps), dtype=np.int64) if ps else \
                np.zeros((0, 2), dtype=np.int64)
        else:
            raise ValueError(f"未知 mode={mode}")
        # 按 (col, row) 排序（列压缩顺序，便于分组）
        if pat.size:
            order = np.lexsort((pat[:, 0], pat[:, 1]))
            pat = pat[order]
        self.ent_row_np = pat[:, 0].copy() if pat.size else np.zeros(0, np.int64)
        self.ent_col_np = pat[:, 1].copy() if pat.size else np.zeros(0, np.int64)
        self.nnz = int(pat.shape[0])

        # (i,j) → 元索引
        key = self.ent_row_np * n_ + self.ent_col_np
        self._key = key
        kmap = {int(k): e for e, k in enumerate(key)}

        # ---- 链路 → 元索引（装配用）----
        eidx = np.full(lr.shape[0], -1, dtype=np.int64)
        idx_both = np.where(both)[0]
        for t, k in enumerate(idx_both):
            i, j = int(max(lr[k, 0], lr[k, 1])), int(min(lr[k, 0], lr[k, 1]))
            if i == j:
                continue
            eidx[k] = kmap[i * n_ + j]
        self.link_ent_np = eidx
        self.link_selfloop_np = np.where(both & (lr[:, 0] == lr[:, 1]))[0]

        # ---- 列依赖层级 ----
        by_row = [[] for _ in range(n_)]
        by_col = [[] for _ in range(n_)]
        for e in range(self.nnz):
            by_row[self.ent_row_np[e]].append(e)
            by_col[self.ent_col_np[e]].append(e)
        lvl = np.zeros(n_, dtype=np.int64)
        for i in range(n_):
            if by_row[i]:
                lvl[i] = 1 + max(lvl[self.ent_col_np[e]] for e in by_row[i])
            else:
                lvl[i] = 1
        self.lvl_np = lvl
        self.nlev = int(lvl.max()) if n_ else 0

        # ---- 三元组：L[i,j] -= L[i,k]·L[j,k]（k = 公共列）----
        tri_t, tri_1, tri_2 = [], [], []
        for k in range(n_):
            rows = by_col[k]
            if len(rows) < 2:
                continue
            rows = sorted(rows, key=lambda e: self.ent_row_np[e])
            for a in range(len(rows)):
                ea = rows[a]
                ia = int(self.ent_row_np[ea])
                for b in range(a):
                    eb = rows[b]
                    ib = int(self.ent_row_np[eb])     # ib < ia
                    t = kmap.get(ia * n_ + ib, -1)
                    if t >= 0:                        # 不在模式内 → IC(0) 丢弃
                        tri_t.append(t); tri_1.append(ea); tri_2.append(eb)
        tri_t = np.asarray(tri_t, dtype=np.int64)
        tri_1 = np.asarray(tri_1, dtype=np.int64)
        tri_2 = np.asarray(tri_2, dtype=np.int64)
        self.n_tri = int(tri_t.size)

        # ---- 消元树（parent[j] = 列 j 的最小行号）+ 深度 ----
        parent = np.full(n_, -1, dtype=np.int64)
        for j in range(n_):
            if by_col[j]:
                parent[j] = int(min(self.ent_row_np[e] for e in by_col[j]))
        depth = np.zeros(n_, dtype=np.int64)
        for j in range(n_ - 1, -1, -1):
            p = parent[j]
            if p >= 0:
                depth[j] = depth[p] + 1
        self.etree_parent_np = parent
        self.etree_depth_np = depth

        # ---- 分层索引 ----
        def _ti(a):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long,
                                   device=device)

        self.ent_row = _ti(self.ent_row_np)
        self.ent_col = _ti(self.ent_col_np)
        self.link_ent = _ti(eidx)
        lvl_row = lvl[self.ent_row_np] if self.nnz else np.zeros(0, np.int64)
        lvl_col = lvl[self.ent_col_np] if self.nnz else np.zeros(0, np.int64)
        lvl_tri = lvl[self.ent_col_np[tri_t]] if self.n_tri else np.zeros(0, np.int64)
        self.ent_by_rowlvl = [_ti(np.where(lvl_row == l)[0])
                              for l in range(1, self.nlev + 1)]
        self.ent_by_collvl = [_ti(np.where(lvl_col == l)[0])
                              for l in range(1, self.nlev + 1)]
        self.tri_by_lvl = []
        for l in range(1, self.nlev + 1):
            s = np.where(lvl_tri == l)[0]
            self.tri_by_lvl.append((_ti(tri_t[s]), _ti(tri_1[s]), _ti(tri_2[s])))
        # 每层的列掩码 [n]（bool）
        self.col_mask = [torch.as_tensor(lvl == l, device=device)
                         for l in range(1, self.nlev + 1)]
        self.lvl = _ti(lvl)
        self.etree_depth = _ti(depth)
        deg = np.zeros(n_, dtype=np.int64)
        for e in range(self.nnz):
            deg[self.ent_row_np[e]] += 1
            deg[self.ent_col_np[e]] += 1
        self.deg = _ti(deg)

    # ------------------------------------------------------------------
    @classmethod
    def from_solver(cls, solver, mode="ic0"):
        """由 GGASolver 构造（复用其 EpanetSmatrix：MMD 行号 + 完全填充结构）。"""
        n = solver.Nj
        r_of_j = np.asarray(solver.row_junc, dtype=np.int64) - 1   # 0 基行号
        junc_row = np.full(solver.N, -1, dtype=np.int64)
        junc_row[solver.junc_nodes] = np.arange(n)
        n1 = solver.n1_np if hasattr(solver, "n1_np") else \
            solver.n1.cpu().numpy()
        n2 = solver.n2_np if hasattr(solver, "n2_np") else \
            solver.n2.cpu().numpy()
        lr = np.full((len(n1), 2), -1, dtype=np.int64)
        m1 = junc_row[n1] >= 0
        m2 = junc_row[n2] >= 0
        lr[m1, 0] = r_of_j[junc_row[n1][m1]]
        lr[m2, 1] = r_of_j[junc_row[n2][m2]]
        sm = solver.sm
        obj = cls(n, lr, xlnz=sm.XLNZ, nzsub=sm.NZSUB, mode=mode,
                  device=solver.device, dtype=solver.dtype)
        obj.r_of_j_np = r_of_j
        obj.r_of_j = torch.as_tensor(r_of_j, dtype=torch.long,
                                     device=solver.device)
        # 装配索引：diag 由 junction 端链路累加
        obj.aii_rows = torch.as_tensor(
            np.concatenate([lr[m1, 0], lr[m2, 1]]), dtype=torch.long,
            device=solver.device)
        obj.aii_lnk = torch.as_tensor(
            np.concatenate([np.where(m1)[0], np.where(m2)[0]]),
            dtype=torch.long, device=solver.device)
        obj.link_both = torch.as_tensor(np.where(obj.link_ent_np >= 0)[0],
                                        dtype=torch.long, device=solver.device)
        obj.link_both_ent = torch.as_tensor(
            obj.link_ent_np[obj.link_ent_np >= 0], dtype=torch.long,
            device=solver.device)
        sl = obj.link_selfloop_np
        if sl.size:
            obj.selfloop_lnk = torch.as_tensor(sl, dtype=torch.long,
                                               device=solver.device)
            obj.selfloop_row = torch.as_tensor(lr[sl, 0], dtype=torch.long,
                                               device=solver.device)
        else:
            obj.selfloop_lnk = None
            obj.selfloop_row = None
        return obj


# ======================================================================
#  2. 稀疏装配 / matvec / IC(0) 数值分解 / 三角解（全部批量 + 层级调度）
# ======================================================================
def assemble_sparse(pat, Pm, em=None, hgrad_e=None):
    """由链路电导 P̂（已乘 pl 掩码）装配 A 的 (diag, offdiag)（行空间）。

    A_ii = Σ_{k@i} P̂_k (+ em_i/hgrad_i)，A_ij = −Σ_{k: i~j} P̂_k。
    与 GGAFormer 稠密装配同源（求和次序不同 ⇒ 逐位可差 O(eps)）。
    """
    B = Pm.shape[0]
    dt, dev = Pm.dtype, Pm.device
    a_d = torch.zeros(B, pat.n, dtype=dt, device=dev)
    a_d = a_d.index_add(1, pat.aii_rows, Pm[:, pat.aii_lnk])
    a_o = torch.zeros(B, pat.nnz, dtype=dt, device=dev)
    if pat.link_both.numel():
        a_o = a_o.index_add(1, pat.link_both_ent, -Pm[:, pat.link_both])
    if pat.selfloop_lnk is not None:
        # 自环：稠密路径 +P(n1 端)+P(n2 端)−2P(对角两次装配) = 净 0，此处扣回
        a_d = a_d.index_add(1, pat.selfloop_row,
                            -2.0 * Pm[:, pat.selfloop_lnk])
    if em is not None:
        # emitter 项在 junction 列空间 → 映射到行空间
        a_d = a_d.index_add(1, pat.r_of_j, em / hgrad_e)
    return a_d, a_o


def spmv(pat, a_d, a_o, x):
    """对称稀疏矩阵向量积 A·x（行空间）。O(n + nnz)。"""
    y = a_d * x
    if pat.nnz:
        y = y.index_add(1, pat.ent_row, a_o * x[:, pat.ent_col])
        y = y.index_add(1, pat.ent_col, a_o * x[:, pat.ent_row])
    return y


@torch.no_grad()
def ic0_factor(pat, a_d, a_o, shift0=0.0, max_shift_try=6):
    """批量不完全 Cholesky（模式 = pat；pat.mode='full' 时 = 精确稀疏 Cholesky）。

    层级调度：同层的列互不依赖 ⇒ 整层向量化。返回 (Ld [B,n], Lo [B,nnz],
    n_breakdown)。若出现非正主元（Stieltjes 性被破坏时才可能），按 Manteuffel
    对角移位 A + s·diag(A) 重试，s 逐次 ×4；仍失败则钳位为 sqrt(eps·a_d)。
    """
    B, n = a_d.shape
    dt, dev = a_d.dtype, a_d.device
    shift = shift0
    for _try in range(max_shift_try + 1):
        Ad = a_d * (1.0 + shift)
        Ld = torch.zeros(B, n, dtype=dt, device=dev)
        Lo = torch.zeros(B, pat.nnz, dtype=dt, device=dev)
        bad = 0
        for l in range(pat.nlev):
            er = pat.ent_by_rowlvl[l]
            dsum = torch.zeros(B, n, dtype=dt, device=dev)
            if er.numel():
                dsum = dsum.index_add(1, pat.ent_row[er], Lo[:, er] ** 2)
            dj = Ad - dsum
            mask = pat.col_mask[l]
            neg = (dj <= 0.0) & mask
            nb = int(neg.sum())
            if nb:
                bad += nb
                dj = torch.where(neg, (_EPS64 * Ad.abs()).clamp_min(1e-300), dj)
            Ld = torch.where(mask, torch.sqrt(dj), Ld)
            tt, t1, t2 = pat.tri_by_lvl[l]
            ec = pat.ent_by_collvl[l]
            if ec.numel():
                tsum = torch.zeros(B, pat.nnz, dtype=dt, device=dev)
                if tt.numel():
                    tsum = tsum.index_add(1, tt, Lo[:, t1] * Lo[:, t2])
                v = (a_o[:, ec] - tsum[:, ec]) / Ld[:, pat.ent_col[ec]]
                Lo = Lo.index_copy(1, ec, v)
        if bad == 0 or _try == max_shift_try:
            return Ld, Lo, bad
        shift = 0.01 if shift == 0.0 else shift * 4.0
    return Ld, Lo, bad


def tri_solve_L(pat, Ld, Lo, b):
    """前代 L·y = b（层级调度，批量，可微）。"""
    y = b
    for l in range(pat.nlev):
        er = pat.ent_by_rowlvl[l]
        if er.numel():
            y = y.index_add(1, pat.ent_row[er], -(Lo[:, er] * y[:, pat.ent_col[er]]))
        y = y * torch.where(pat.col_mask[l], 1.0 / Ld, torch.ones_like(Ld))
    return y


def tri_solve_LT(pat, Ld, Lo, b):
    """回代 Lᵀ·x = b（层级逆序，批量，可微）。"""
    x = b
    for l in range(pat.nlev - 1, -1, -1):
        ec = pat.ent_by_collvl[l]
        if ec.numel():
            x = x.index_add(1, pat.ent_col[ec], -(Lo[:, ec] * x[:, pat.ent_row[ec]]))
        x = x * torch.where(pat.col_mask[l], 1.0 / Ld, torch.ones_like(Ld))
    return x


def apply_Minv(pat, Ld, Lo, r):
    """M⁻¹r = L̃⁻ᵀ(L̃⁻¹ r)。M = L̃L̃ᵀ 恒 SPD（Ld > 0 严格）。"""
    return tri_solve_LT(pat, Ld, Lo, tri_solve_L(pat, Ld, Lo, r))


# ======================================================================
#  3. Krylov 保证层：PCG（前向不建图，反传走隐函数定理伴随）
# ======================================================================
def resid_scale(pat, a_d, a_o, x, b):
    """行尺度 s_i = (|A|·|x|)_i + |b_i| - 残差的可达地板参照。

    与 W1 守恒门的 rowscale = |A|·|Hj| + |F| 逐字同义：这是线性残差
    （= 更新后节点净流失衡）在 f64 下的物理量纲参照。
    """
    return spmv(pat, a_d.abs(), a_o.abs(), x.abs()) + b.abs()


def resid_metric(pat, a_d, a_o, x, b, r, crit):
    """残差度量。

    crit='cbe'（默认，逐分量后向误差 / Oettli–Prager）：
        max_i |r_i| / ((|A||x|)_i + |b_i|)
 - **唯一在 f64 下可达 ~eps 的判据**，且逐节点对齐守恒的物理尺度。
    crit='rel'（经典）：‖r‖₂/‖b‖₂ - 在 cond(A)~1e15 的网（city_d：
      ‖A‖‖x‖/‖b‖ ≈ 3e15）上地板 ≈ eps·3e15 ≈ 0.7，**1e-10 根本不可达**，
      不可用作这类系统的容差口径。
    """
    if crit == "rel":
        return torch.linalg.norm(r, dim=1) / \
            torch.linalg.norm(b, dim=1).clamp_min(1e-300)
    if crit == "cbe":
        s = resid_scale(pat, a_d, a_o, x, b)
        return (r.abs() / s.clamp_min(1e-300)).max(dim=1).values
    raise ValueError(f"未知 crit={crit}")


def _pcg_core(pat, a_d, a_o, b, Ld, Lo, tol, maxit, x0=None, crit="cbe",
              replace_every=25, stall_win=0):
    """裸 PCG（无梯度）。返回 (x, iters[B], res[B], converged[B])。

    - 递推残差只进 CG 递推（换成真残差会破坏共轭性、显著拖慢收敛）；
      **停机判据一律由真残差 b−A·x 判定**：递推残差只用来"提名"可能达标的
      样本，提名后立刻做残差替换（van der Vorst–Ye）并用真残差确认，
      **绝不允许未经真残差确认的样本被标记为收敛**（否则报出的步数会偏小）；
    - 未在 maxit 内达到 tol ⇒ converged=False（显式报告，绝不静默返回）。
    """
    B, n = b.shape
    dt, dev = b.dtype, b.device
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - spmv(pat, a_d, a_o, x)
    z = apply_Minv(pat, Ld, Lo, r)
    p = z.clone()
    rz = (r * z).sum(1, keepdim=True)
    iters = torch.zeros(B, dtype=torch.long, device=dev)
    done = resid_metric(pat, a_d, a_o, x, b, r, crit) <= tol
    n_recheck = 0                     # 真残差复核次数（成本统计）
    # breakdown 计数（CG 意义下）：pᵀAp ≤ 0 ⇒ A 非正定；rᵀM⁻¹r < 0 ⇒ M 非正定。
    # 二者恒 >0 是 SPD 保证的可观测判据（与"能否在预算内收敛"是两回事）。
    bd = dict(pAp=0, rz=int((rz < 0).sum()))
    for it in range(1, maxit + 1):
        if bool(done.all()):
            break
        act = (~done).to(dt).unsqueeze(1)
        Ap = spmv(pat, a_d, a_o, p)
        pAp = (p * Ap).sum(1, keepdim=True)
        bd["pAp"] += int(((pAp <= 0.0) & (~done).unsqueeze(1)).sum())
        alpha = torch.where(pAp.abs() > 0, rz / pAp, torch.zeros_like(pAp)) * act
        x = x + alpha * p
        r = r - alpha * Ap
        iters = iters + (~done).to(torch.long)
        m = resid_metric(pat, a_d, a_o, x, b, r, crit)   # 递推残差：仅提名
        nominate = bool(((m <= tol) & (~done)).any())
        if (it % replace_every == 0) or nominate:
            r = b - spmv(pat, a_d, a_o, x)          # 残差替换（真残差）
            m = resid_metric(pat, a_d, a_o, x, b, r, crit)
            done = done | (m <= tol)                # 只认真残差确认过的
            n_recheck += 1
        z = apply_Minv(pat, Ld, Lo, r)
        rz_new = (r * z).sum(1, keepdim=True)
        bd["rz"] += int(((rz_new < 0.0) & (~done).unsqueeze(1)).sum())
        beta = torch.where(rz.abs() > 0, rz_new / rz, torch.zeros_like(rz))
        p = z + beta * p * act
        rz = rz_new
    rt = b - spmv(pat, a_d, a_o, x)
    res = resid_metric(pat, a_d, a_o, x, b, rt, crit)
    converged = res <= tol
    bd["n_recheck"] = n_recheck
    _pcg_core.last_breakdown = bd
    return x, iters, res, converged


class _PCGFunction(torch.autograd.Function):
    """线性解层的隐函数定理反传（不展开 CG 的计算图）。

    x = A⁻¹b ⇒ 给定 ḡ = ∂L/∂x：λ = A⁻¹ḡ（A 对称，同一 M 复用），
    ∂L/∂b = λ，∂L/∂a_d[i] = −λ_i x_i，∂L/∂a_o[e] = −(λ_i x_j + λ_j x_i)。
    """

    @staticmethod
    def forward(ctx, a_d, a_o, b, pat, Ld, Lo, tol, maxit, stats, crit):
        with torch.no_grad():
            x, iters, relres, conv = _pcg_core(pat, a_d, a_o, b, Ld, Lo,
                                               tol, maxit, crit=crit)
        ctx.save_for_backward(a_d, a_o, x, Ld, Lo)
        ctx.pat, ctx.tol, ctx.maxit, ctx.crit = pat, tol, maxit, crit
        if stats is not None:
            stats["iters"] = iters
            stats["relres"] = relres
            stats["converged"] = conv
        return x

    @staticmethod
    def backward(ctx, gx):
        a_d, a_o, x, Ld, Lo = ctx.saved_tensors
        pat = ctx.pat
        with torch.no_grad():
            lam, _, _, _ = _pcg_core(pat, a_d, a_o, gx.contiguous(), Ld, Lo,
                                     ctx.tol, ctx.maxit, crit=ctx.crit)
        g_ad = -lam * x
        if pat.nnz:
            g_ao = -(lam[:, pat.ent_row] * x[:, pat.ent_col]
                     + lam[:, pat.ent_col] * x[:, pat.ent_row])
        else:
            g_ao = torch.zeros_like(a_o)
        return g_ad, g_ao, lam, None, None, None, None, None, None, None


def pcg_solve(pat, a_d, a_o, b, Ld, Lo, tol=1e-10, maxit=500, crit="cbe"):
    """Krylov 保证层（可微，隐函数定理反传）。

    返回 dict(x, iters, relres, converged)。
    **未在 maxit 内达到 tol 时显式报告 converged=False**（绝不静默返回）。
    """
    stats = {}
    x = _PCGFunction.apply(a_d, a_o, b, pat, Ld, Lo, tol, maxit, stats, crit)
    return dict(x=x, iters=stats["iters"], relres=stats["relres"],
                converged=stats["converged"])


def pcg_solve_stats(pat, a_d, a_o, b, Ld, Lo, tol=1e-10, maxit=500, x0=None,
                    crit="cbe"):
    """纯诊断版（无梯度）：返回 (x, iters, res, converged)。"""
    with torch.no_grad():
        return _pcg_core(pat, a_d, a_o, b, Ld, Lo, tol, maxit, x0=x0, crit=crit)


# ======================================================================
#  3b. 精确稀疏 Cholesky 直接解层（W3 任务 A）
# ======================================================================
#  依据（W2 已实证）：53 网 MMD 填充比恒在 1.00–2.35 且与规模无关（r=0.06）
#  ⇒ pat.mode='full' 上的 ic0_factor **就是精确 Cholesky**（无丢弃），
#  一次分解 + 一次三角解即得精确解（PCG 意义下 1 步收敛），成本 O(nnz)。
#  故这里不再需要 Krylov 迭代 - 直接解就是"保证层"本身。
def _fast_level_index(pat):
    """惰性预计算"层级紧凑索引"（缓存在 pat 上；不改 FillPattern.__init__）。

    ic0_factor / tri_solve_* 的层级循环里每层都开 [B,n] 或 [B,nnz] 的整长缓冲，
    总量 O(B·(n+nnz)·nlev) - 在 nlev~76（net6）/114（bwsn2）时纯属浪费：
    第 l 层只碰 lvl==l+1 的列与元。这里预存每层的
      cols   [n_l]  该层的列（= 节点行号）
      rowloc [|er|] ent_row[er] 在 cols 内的位置
      ttloc  [|tt|] 三元组目标 t 在 ec 内的位置
    使每层只做 O(B·(n_l + |ec_l|)) 的工作，总量降回 O(B·(n+nnz))。
    **不改变任何 index_add 的源元素次序 ⇒ 逐位与原实现一致**（由
    scripts/verify_sparse_chol.py 在 53 网上实测锁死）。
    """
    fi = getattr(pat, "_fastidx", None)
    if fi is not None:
        return fi
    n, lvl = pat.n, pat.lvl_np
    dev = pat.device
    out = []
    for l in range(pat.nlev):
        cols = np.where(lvl == l + 1)[0]
        pos = np.full(n, -1, dtype=np.int64)
        pos[cols] = np.arange(cols.size)
        er = pat.ent_by_rowlvl[l].cpu().numpy()
        rowloc = pos[pat.ent_row_np[er]] if er.size else np.zeros(0, np.int64)
        ec = pat.ent_by_collvl[l].cpu().numpy()
        epos = np.full(max(pat.nnz, 1), -1, dtype=np.int64)
        if ec.size:
            epos[ec] = np.arange(ec.size)
        tt = pat.tri_by_lvl[l][0].cpu().numpy()
        ttloc = epos[tt] if tt.size else np.zeros(0, np.int64)
        if (rowloc < 0).any() or (ttloc < 0).any():
            raise AssertionError("层级紧凑索引不自洽（ent_row/tri 目标越层）")

        def _t(a):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long,
                                   device=dev)
        out.append((_t(cols), _t(rowloc), _t(ttloc)))
    pat._fastidx = out
    return out


@torch.no_grad()
def chol_factor(pat, a_d, a_o, shift0=0.0, max_shift_try=6):
    """批量数值 Cholesky（层级**紧凑**调度） - 与 ic0_factor 逐位一致的快路径。

    pat.mode='full' ⇒ 精确稀疏 Cholesky；'ic0' ⇒ 零填充不完全分解。
    返回 (Ld [B,n], Lo [B,nnz], n_breakdown)。
    """
    fi = _fast_level_index(pat)
    B, n = a_d.shape
    dt, dev = a_d.dtype, a_d.device
    shift = shift0
    for _try in range(max_shift_try + 1):
        Ad = a_d * (1.0 + shift)
        Ld = torch.zeros(B, n, dtype=dt, device=dev)
        Lo = torch.zeros(B, pat.nnz, dtype=dt, device=dev)
        bad = 0
        for l in range(pat.nlev):
            cols, rowloc, ttloc = fi[l]
            er = pat.ent_by_rowlvl[l]
            Adc = Ad[:, cols]
            dj = Adc
            if er.numel():
                ds = torch.zeros(B, cols.numel(), dtype=dt, device=dev)
                ds = ds.index_add(1, rowloc, Lo[:, er] ** 2)
                dj = Adc - ds
            neg = dj <= 0.0
            nb = int(neg.sum())
            if nb:
                bad += nb
                dj = torch.where(neg, (_EPS64 * Adc.abs()).clamp_min(1e-300), dj)
            Ld = Ld.index_copy(1, cols, torch.sqrt(dj))
            ec = pat.ent_by_collvl[l]
            if ec.numel():
                _, t1, t2 = pat.tri_by_lvl[l]
                v = a_o[:, ec]
                if t1.numel():
                    ts = torch.zeros(B, ec.numel(), dtype=dt, device=dev)
                    ts = ts.index_add(1, ttloc, Lo[:, t1] * Lo[:, t2])
                    v = v - ts
                Lo = Lo.index_copy(1, ec, v / Ld[:, pat.ent_col[ec]])
        if bad == 0 or _try == max_shift_try:
            return Ld, Lo, bad
        shift = 0.01 if shift == 0.0 else shift * 4.0
    return Ld, Lo, bad


def tri_solve_L_fast(pat, Ld, Lo, b):
    """前代 L·y = b（层级紧凑，批量，可微） - 与 tri_solve_L 逐位一致。"""
    fi = _fast_level_index(pat)
    y = b
    for l in range(pat.nlev):
        cols, rowloc, _ = fi[l]
        er = pat.ent_by_rowlvl[l]
        yc = y[:, cols]
        if er.numel():
            yc = yc.index_add(1, rowloc,
                              -(Lo[:, er] * y[:, pat.ent_col[er]]))
        y = y.index_copy(1, cols, yc * (1.0 / Ld[:, cols]))
    return y


def tri_solve_LT_fast(pat, Ld, Lo, b):
    """回代 Lᵀ·x = b（层级逆序，紧凑） - 与 tri_solve_LT 逐位一致。"""
    fi = _fast_level_index(pat)
    x = b
    for l in range(pat.nlev - 1, -1, -1):
        cols, _, _ = fi[l]
        ec = pat.ent_by_collvl[l]
        xc = x[:, cols]
        if ec.numel():
            # 目标是 ent_col[ec]（= 本层的列），源是 x[ent_row[ec]]（更高层，已定）
            xc = xc.index_add(1, _colloc_for(pat, l),
                              -(Lo[:, ec] * x[:, pat.ent_row[ec]]))
        x = x.index_copy(1, cols, xc * (1.0 / Ld[:, cols]))
    return x


def _colloc_for(pat, l):
    """ent_col[ec_l] 在第 l 层 cols 内的位置（缓存）。"""
    cache = getattr(pat, "_collocs", None)
    if cache is None:
        cache = {}
        pat._collocs = cache
    if l in cache:
        return cache[l]
    cols = np.where(pat.lvl_np == l + 1)[0]
    pos = np.full(pat.n, -1, dtype=np.int64)
    pos[cols] = np.arange(cols.size)
    ec = pat.ent_by_collvl[l].cpu().numpy()
    loc = pos[pat.ent_col_np[ec]] if ec.size else np.zeros(0, np.int64)
    if (loc < 0).any():
        raise AssertionError("回代紧凑索引不自洽")
    cache[l] = torch.as_tensor(np.ascontiguousarray(loc), dtype=torch.long,
                               device=pat.device)
    return cache[l]


def apply_Minv_fast(pat, Ld, Lo, r):
    """M⁻¹r = L̃⁻ᵀ(L̃⁻¹ r)（紧凑层级）。与 apply_Minv 逐位一致。"""
    return tri_solve_LT_fast(pat, Ld, Lo, tri_solve_L_fast(pat, Ld, Lo, r))


def chol_apply(pat, a_d, a_o, Ld, Lo, b, refine=2):
    """x = A⁻¹b：三角解 + `refine` 步同精度迭代精化（与稠密路径逐句同源）。

    分解精确时 L̃L̃ᵀ = A，首次三角解即达 ~eps；精化步用于吃掉三角解自身的
    舍入（与 dgga/solver.py dense 路径的 2 步精化对齐）。
    """
    x = apply_Minv_fast(pat, Ld, Lo, b)
    for _ in range(int(refine)):
        r = b - spmv(pat, a_d, a_o, x)
        x = x + apply_Minv_fast(pat, Ld, Lo, r)
    return x


class _SparseCholFunction(torch.autograd.Function):
    """精确稀疏 Cholesky 解层的隐函数定理反传（不展开分解/三角解的计算图）。

    x = A⁻¹b ⇒ 给定 ḡ = ∂L/∂x：λ = A⁻¹ḡ（A 对称，复用同一个 L），
    ∂L/∂b = λ，∂L/∂a_d[i] = −λ_i x_i，∂L/∂a_o[e] = −(λ_i x_j + λ_j x_i)。
    与 _PCGFunction 的伴随式逐字相同（同一数学对象，只是解法不同）。
    """

    @staticmethod
    def forward(ctx, a_d, a_o, b, pat, refine, stats):
        with torch.no_grad():
            Ld, Lo, bad = chol_factor(pat, a_d, a_o)
            x = chol_apply(pat, a_d, a_o, Ld, Lo, b, refine)
        ctx.save_for_backward(a_d, a_o, x, Ld, Lo)
        ctx.pat, ctx.refine = pat, int(refine)
        if stats is not None:
            stats["Ld"], stats["Lo"] = Ld, Lo
            stats["n_breakdown"] = int(bad)
        return x

    @staticmethod
    def backward(ctx, gx):
        a_d, a_o, x, Ld, Lo = ctx.saved_tensors
        pat = ctx.pat
        with torch.no_grad():
            lam = chol_apply(pat, a_d, a_o, Ld, Lo, gx.contiguous(), ctx.refine)
        g_ad = -lam * x
        if pat.nnz:
            g_ao = -(lam[:, pat.ent_row] * x[:, pat.ent_col]
                     + lam[:, pat.ent_col] * x[:, pat.ent_row])
        else:
            g_ao = torch.zeros_like(a_o)
        return g_ad, g_ao, lam, None, None, None


def sparse_chol_solve(pat, a_d, a_o, b, refine=2):
    """精确稀疏 Cholesky 直接解（批量 [B,n]，f64，可微）。

    要求 pat.mode == 'full'（完全填充模式） - 否则分解是不完全的，解不精确。
    返回 dict(x, Ld, Lo, n_breakdown)；n_breakdown>0 表示出现非正主元
    （SPD 被破坏，显式报告，绝不静默）。
    """
    if pat.mode != "full":
        raise ValueError("sparse_chol_solve 需要 mode='full' 的完全填充模式"
                         f"（当前 {pat.mode}） - 否则不是精确分解")
    stats = {}
    x = _SparseCholFunction.apply(a_d, a_o, b, pat, int(refine), stats)
    return dict(x=x, Ld=stats["Ld"], Lo=stats["Lo"],
                n_breakdown=stats["n_breakdown"])


# ======================================================================
#  4. 掩码注意力预条件子
# ======================================================================
def _seg_softmax(s, index, n):
    """按 index 分组的 softmax。s:[B,M,H]，index:[M]，组数 n。"""
    B, M, H = s.shape
    idx = index.view(1, M, 1).expand(B, M, H)
    mx = torch.full((B, n, H), -1e30, dtype=s.dtype, device=s.device)
    mx = mx.scatter_reduce(1, idx, s, reduce="amax", include_self=True)
    ex = torch.exp(s - mx.gather(1, idx))
    den = torch.zeros(B, n, H, dtype=s.dtype, device=s.device)
    den = den.index_add(1, index, ex)
    return ex / den.gather(1, idx).clamp_min(1e-300)


def _std(x, dim=1):
    """逐样本标准化（对数特征跨网量级差 10 个数量级，必须归一）。"""
    m = x.mean(dim=dim, keepdim=True)
    s = x.std(dim=dim, keepdim=True).clamp_min(1e-8)
    return (x - m) / s


class MaskedCholPrecond(nn.Module):
    """掩码注意力 → 可学习不完全 Cholesky 因子 L̃（M = L̃L̃ᵀ）。

    注意力**只在填充掩码限定的稀疏模式上**进行：以列 j 为分组，query = 节点 j
    的嵌入，key/value = 该列非零行 i 的节点嵌入 ⊕ 元嵌入，组内 softmax。

    参数化（保证的落点）：
      L̃_jj = L0_jj · softplus(B0 + 2·tanh(z_d))        > 0 严格（M 恒 SPD）
      L̃_ij = L0_ij + tanh(z_o) · |a_o_ij| / L0_jj       （自然量纲尺度）
    输出头零初始化 ⇒ z ≡ 0 ⇒ L̃ ≡ L0 = IC(0)（恒等初始化）。
    """

    N_NODE_FEAT = 6
    N_ENT_FEAT = 5

    def __init__(self, dim=32, n_head=4, dtype=torch.float64, gate=1.0):
        super().__init__()
        assert dim % n_head == 0
        self.dim, self.n_head, self.dh = dim, n_head, dim // n_head
        self.gate = float(gate)
        self.node_enc = nn.Sequential(
            nn.Linear(self.N_NODE_FEAT, dim, dtype=dtype), nn.Tanh(),
            nn.Linear(dim, dim, dtype=dtype), nn.Tanh())
        self.ent_enc = nn.Sequential(
            nn.Linear(self.N_ENT_FEAT, dim, dtype=dtype), nn.Tanh(),
            nn.Linear(dim, dim, dtype=dtype), nn.Tanh())
        self.q_proj = nn.Linear(dim, dim, dtype=dtype)
        self.k_proj = nn.Linear(2 * dim, dim, dtype=dtype)
        self.v_proj = nn.Linear(2 * dim, dim, dtype=dtype)
        self.bias_proj = nn.Linear(dim, n_head, dtype=dtype)     # 元偏置
        self.out_diag = nn.Sequential(
            nn.Linear(2 * dim, dim, dtype=dtype), nn.Tanh(),
            nn.Linear(dim, 1, dtype=dtype))
        self.out_off = nn.Sequential(
            nn.Linear(3 * dim, dim, dtype=dtype), nn.Tanh(),
            nn.Linear(dim, 1, dtype=dtype))
        _zero_last(self.out_diag)
        _zero_last(self.out_off)
        self.register_buffer("b0", torch.tensor(B0, dtype=dtype))

    # ------------------------------------------------------------------
    def randomize_(self, std=0.5, seed=None):
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * std)
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _feats(pat, a_d, a_o, Ld0, Lo0):
        """节点特征 [B,n,6] 与元特征 [B,nnz,5]（纯诊断量，全部 detach）。"""
        B, n = a_d.shape
        dt, dev = a_d.dtype, a_d.device
        rs = torch.zeros(B, n, dtype=dt, device=dev)
        if pat.nnz:
            rs = rs.index_add(1, pat.ent_row, a_o.abs())
            rs = rs.index_add(1, pat.ent_col, a_o.abs())
        la = torch.log(a_d.abs().clamp_min(1e-300))
        nf = torch.stack([
            _std(la),
            _std(torch.log((rs / a_d.abs().clamp_min(1e-300)).clamp_min(1e-30))),
            (pat.deg.to(dt) / max(1.0, float(pat.deg.max()))).expand(B, n),
            (pat.lvl.to(dt) / max(1, pat.nlev)).expand(B, n),
            (pat.etree_depth.to(dt)
             / max(1.0, float(pat.etree_depth.max()))).expand(B, n),
            _std(torch.log(Ld0.clamp_min(1e-300))),
        ], dim=-1)
        if pat.nnz:
            ai, aj = a_d[:, pat.ent_row], a_d[:, pat.ent_col]
            ef = torch.stack([
                _std(torch.log(a_o.abs().clamp_min(1e-300))),
                _std(torch.log((a_o.abs()
                                / (ai * aj).abs().clamp_min(1e-300).sqrt()
                                ).clamp_min(1e-30))),
                _std(torch.log(Lo0.abs().clamp_min(1e-300))),
                (pat.lvl[pat.ent_col].to(dt) / max(1, pat.nlev)).expand(B, -1),
                ((pat.lvl[pat.ent_row] - pat.lvl[pat.ent_col]).to(dt)
                 / max(1, pat.nlev)).expand(B, -1),
            ], dim=-1)
        else:
            ef = torch.zeros(B, 0, MaskedCholPrecond.N_ENT_FEAT,
                             dtype=dt, device=dev)
        return nf, ef

    # ------------------------------------------------------------------
    def forward(self, pat, a_d, a_o):
        """返回 (Ld, Lo, info)。a_d/a_o 内部 detach（预条件子对解无数学影响，
        不应把梯度经 M 反灌物理参数；θ_precond 用 pcg_reduction_loss 单独训）。"""
        a_d = a_d.detach()
        a_o = a_o.detach()
        Ld0, Lo0, nbad = ic0_factor(pat, a_d, a_o)
        B, n = a_d.shape
        with torch.no_grad():
            nf, ef = self._feats(pat, a_d, a_o, Ld0, Lo0)
        hn = self.node_enc(nf)                                    # [B,n,D]
        if pat.nnz:
            he = self.ent_enc(ef)                                 # [B,nnz,D]
            q = self.q_proj(hn)[:, pat.ent_col]                   # [B,nnz,D]
            kv_in = torch.cat([hn[:, pat.ent_row], he], dim=-1)
            k = self.k_proj(kv_in)
            v = self.v_proj(kv_in)
            H, dh = self.n_head, self.dh
            sc = (q.view(B, -1, H, dh) * k.view(B, -1, H, dh)).sum(-1) \
                / math.sqrt(dh)
            sc = sc + self.bias_proj(he)                          # [B,nnz,H]
            w = _seg_softmax(sc, pat.ent_col, n)                  # [B,nnz,H]
            ctx = torch.zeros(B, n, H, dh, dtype=a_d.dtype, device=a_d.device)
            ctx = ctx.index_add(1, pat.ent_col,
                                w.unsqueeze(-1) * v.view(B, -1, H, dh))
            ctx = ctx.reshape(B, n, self.dim)
        else:
            he = torch.zeros(B, 0, self.dim, dtype=a_d.dtype, device=a_d.device)
            ctx = torch.zeros(B, n, self.dim, dtype=a_d.dtype, device=a_d.device)

        z_d = self.out_diag(torch.cat([hn, ctx], dim=-1))[..., 0]         # [B,n]
        Ld = Ld0 * torch.nn.functional.softplus(
            self.b0 + 2.0 * self.gate * torch.tanh(z_d))
        if pat.nnz:
            z_o = self.out_off(torch.cat([hn[:, pat.ent_row],
                                          hn[:, pat.ent_col] + ctx[:, pat.ent_col],
                                          he], dim=-1))[..., 0]           # [B,nnz]
            scale = a_o.abs() / Ld0[:, pat.ent_col].clamp_min(1e-300)
            Lo = Lo0 + self.gate * torch.tanh(z_o) * scale
        else:
            Lo = Lo0
        return Ld, Lo, dict(ic0_breakdown=nbad, Ld0=Ld0, Lo0=Lo0)


# ======================================================================
#  5. 经典基线预条件子（对照用）
# ======================================================================
def precond_none(pat, a_d, a_o):
    """无预条件（M = I）：L̃ = I。"""
    return torch.ones_like(a_d), torch.zeros_like(a_o)


def precond_jacobi(pat, a_d, a_o):
    """Jacobi：M = diag(A) ⇒ L̃ = diag(sqrt(a_d))。"""
    return torch.sqrt(a_d.clamp_min(1e-300)), torch.zeros_like(a_o)


def precond_ic0(pat, a_d, a_o):
    """经典 IC(0)（= 本模块恒等初始化的落点）。"""
    Ld, Lo, _ = ic0_factor(pat, a_d, a_o)
    return Ld, Lo


# ======================================================================
#  6. 预条件子的训练损失（∂H/∂θ_M ≡ 0 ⇒ 必须用显式加速损失）
# ======================================================================
def pcg_reduction_loss(pat, a_d, a_o, b, Ld, Lo, k=8):
    """展开 k 步可微 PCG，返回 log10(‖r_k‖/‖r_0‖) 的均值（越小越好）。

    这是"加速"的直接代理：CG 步数 ∝ 达到 tol 所需的残差衰减。
    注意此处**是训练用的展开图**，与推理路径（_PCGFunction 隐函数反传）无关。
    """
    B = b.shape[0]
    x = torch.zeros_like(b)
    r = b - spmv(pat, a_d, a_o, x)
    r0 = torch.linalg.norm(r, dim=1).clamp_min(1e-300)
    z = apply_Minv(pat, Ld, Lo, r)
    p = z
    rz = (r * z).sum(1, keepdim=True)
    for _ in range(k):
        Ap = spmv(pat, a_d, a_o, p)
        pAp = (p * Ap).sum(1, keepdim=True)
        alpha = rz / pAp.clamp_min(1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        z = apply_Minv(pat, Ld, Lo, r)
        rz_new = (r * z).sum(1, keepdim=True)
        p = z + (rz_new / rz.clamp_min(1e-300)) * p
        rz = rz_new
    rk = torch.linalg.norm(r, dim=1).clamp_min(1e-300)
    return torch.log10(rk / r0).mean(), (rk / r0).detach()


# ======================================================================
#  7. FLOPs 成本模型（**每次乘法、每次加法各计 1 FLOP** - LAPACK/BLAS 口径）
# ======================================================================
#  设计原则：账本必须包含**全部**成本，否则会系统性偏袒学习件。
#    总成本 = 预条件子构造（一次性，含注意力前向）+ 步数 × 每步成本
#    每步成本 = spmv + M⁻¹应用 + CG 向量运算 + **停机判据的第二次 spmv**
#  （旧版账本只有 spmv+M⁻¹ 且按 FMA=1 计，实测低估 ~2.9 倍，见 data/w2_audit_wip.txt）
def dense_chol_flops(n):
    """稠密 Cholesky：n³/3。"""
    return n ** 3 / 3.0


def sparse_chol_flops(pat):
    """稀疏 Cholesky（模式 = pat）：Σ_j (c_j² + 3c_j) + n，c_j = 列 j 非对角高度。

    等价于该模式上 2·n_tri + 4·nnz + 2n 的逐元计数（本实现的实际算术），
    两式在完全填充模式下一致（实测偏差 <1%）。
    """
    c = np.bincount(pat.ent_col_np, minlength=pat.n).astype(np.float64)
    return float(np.sum(c ** 2 + 3.0 * c) + pat.n)


def ic0_setup_flops(pat):
    """本实现的 IC(0)/精确稀疏分解逐元计数：三元组更新 + 列缩放 + 开方。"""
    return float(2 * pat.n_tri + 4 * pat.nnz + 2 * pat.n)


def spmv_flops(pat):
    """对称稀疏 matvec（下三角存一份）：对角 n 乘 + 每个非对角元 2 乘 2 加。"""
    return float(4 * pat.nnz + pat.n)


def tri_solve_flops(pat):
    """两次三角解（前代 + 回代）：每个非对角元 2 乘 2 加 + 每次 n 次除。"""
    return float(4 * pat.nnz + 2 * pat.n)


def chol_solve_flops(pat, refine=2):
    """精确稀疏 Cholesky 一次"分解 + 解"的总成本（不含装配）。

    = 数值分解 + 三角解 + refine × (spmv + 残差减 n + 三角解 + 修正加 n)。
    返回 (total, factor, solve, refine_cost)。
    """
    fac = ic0_setup_flops(pat)
    sol = tri_solve_flops(pat)
    ref = float(refine) * (spmv_flops(pat) + pat.n + tri_solve_flops(pat) + pat.n)
    return fac + sol + ref, fac, sol, ref


def attention_flops(n, nnz, dim=32, n_head=4):
    """MaskedCholPrecond.forward 的前向 FLOPs（Linear: 2·in·out per token）。"""
    D = int(dim)
    f = n * (2 * MaskedCholPrecond.N_NODE_FEAT * D + 2 * D * D)   # node_enc
    f += nnz * (2 * MaskedCholPrecond.N_ENT_FEAT * D + 2 * D * D)  # ent_enc
    f += n * 2 * D * D                                             # q_proj
    f += nnz * 2 * (2 * D) * D * 2                                 # k_proj+v_proj
    f += nnz * 2 * D * n_head                                      # bias_proj
    f += nnz * (2 * D + 3 * n_head + 2 * D)                        # 打分/softmax/聚合
    f += n * (2 * (2 * D) * D + 2 * D)                             # out_diag
    f += nnz * (2 * (3 * D) * D + 2 * D)                           # out_off
    return float(f)


def precond_setup_flops(pat, kind, dim=32, n_head=4):
    """预条件子构造的一次性成本。kind ∈ none|jacobi|ic0|exact|learned。"""
    if kind == "none":
        return 0.0
    if kind == "jacobi":
        return float(pat.n)                     # n 次开方
    if kind in ("ic0", "exact"):
        return ic0_setup_flops(pat)
    if kind == "learned":
        # 学习件仍要先算一次 IC(0)（作为 L0），再叠注意力前向
        return ic0_setup_flops(pat) + attention_flops(pat.n, pat.nnz,
                                                      dim, n_head)
    raise ValueError(f"未知 kind={kind}")


def pcg_iter_flops(nnz_a, n, nnz_m, crit="cbe"):
    """PCG 每步成本。

    spmv(对称，下三角存一份)  = 4·nnz_A + n
    M⁻¹ = 2 次三角解            = 4·nnz_M + 2n
    CG 向量运算（2 点积 + 3 axpy）= 10n
    判据 cbe：|A|·|x| 再做一次 spmv + 除 + max = 4·nnz_A + 3n
    （crit='rel' 只要 2 个范数 = 4n）
    """
    f = (4 * nnz_a + n) + (4 * nnz_m + 2 * n) + 10 * n
    f += (4 * nnz_a + 3 * n) if crit == "cbe" else 4 * n
    return float(f)


def pcg_total_flops(pat_a, pat_m, kind, iters, crit="cbe", dim=32, n_head=4,
                    n_recheck=0):
    """总成本 = 构造 + 步数×每步 + 真残差复核次数×spmv。返回 (total, setup, per)。

    n_recheck = _pcg_core 报告的真残差复核（残差替换）次数 - 判据只认真残差，
    每次复核多花一次 spmv，必须入账。
    """
    nnz_m = 0 if kind in ("none", "jacobi") else pat_m.nnz
    per = pcg_iter_flops(pat_a.nnz, pat_a.n, nnz_m, crit)
    setup = precond_setup_flops(pat_m, kind, dim, n_head)
    extra = n_recheck * (4 * pat_a.nnz + pat_a.n)
    return setup + iters * per + extra, setup, per

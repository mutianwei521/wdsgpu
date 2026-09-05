# -*- coding: utf-8 -*-
"""dgga.symbolic - W2 掩码基建：从 EPANET 符号分解导出稀疏注意力的结构先验。

W2 的架构立场：掩码注意力 **不是** 用来替换精确 Cholesky 的，而是学习一个
不完全 Cholesky 预条件子 M ≈ A，外面套 CG 解到 tol_lin。因此本模块只负责
**结构**（谁能看见谁、层级位置编码），不碰数值 - 数值正确性由 CG 的 tol_lin 保证。

三件产物（一次性预计算、跨层跨场景复用）：
  ① 填充掩码 chordal_index [2, nnz_chordal]：pattern(L + Lᵀ) + 对角，
     即消元后的弦图（chordal completion）。这是"允许注意"的最大集合 -
     稀疏注意力在此模式上做，其填充量正是 MMD 排序优化的目标。
  ② 消元树 parent/depth/subtree_size/postorder：parent[j] = 列 j 对角线以下的
     最小行号（标准定义）。深度与子树大小是层级位置编码的原料。
  ③ 原始邻接 a_index [2, nnz_A]：pattern(A)+对角（未填充），消融对照。

坐标系铁律：所有索引都在 **行空间**（MMD 重排后的行号 0..Nj-1，0 基）。
  perm[r]  = 行 r 对应的 junction 槽位（0 基，槽位序 = np.where(node_type==0)[0] 的下标，
             与 GGASolver.junc_nodes / GGAFormer 稠密 A 的行列序一致）
  iperm[j] = junction 槽位 j 的行号
  x_row = x_junc[perm]；x_junc = x_row[iperm]

EPANET 侧对接：EpanetSmatrix 的 Row/Order 是 1 基 ep 编号（junction 为文件序
1..Nj），故 perm[r] = Order[r+1]-1、iperm[j] = Row[j+1]-1。XLNZ/NZSUB 给出列压缩的
下三角 L 结构，LNZ 给出每个非零对应的 Aij 槽位（W2 装配 M 的值时要用）。

排序方案对比（掩码优化器的实证）用的通用符号分解走自写的 O(nnz·α(n)) 算法
（Liu 消元树 + Gilbert–Ng–Peyton 列计数），因此随机序/自然序即使填充灾难
（O(n²)）也只花 O(nnz) 时间 - 只算计数，不物化模式。
"""

import numpy as np

__all__ = [
    "junction_pattern", "etree_from_pattern", "postorder", "col_counts",
    "symbolic_pattern", "permute_pattern", "tree_depth_size",
    "OrderingStats", "ordering_stats", "SymbolicStructure",
    "build_symbolic", "mmd_permutation",
]


# ======================================================================
# 0. 网络 → junction-junction 对称邻接（0 基 CSR，无对角，无重边）
# ======================================================================
def junction_pattern(node_type, link_n1, link_n2):
    """把网络拓扑压成 junction 子图的对称模式。

    node_type: int[N]（0=junction, 1=reservoir, 2=tank）；link_n1/n2: 0 基节点索引。
    只保留两端都是 junction 的链路（定水头节点不进线性系统，与 EPANET 一致）；
    平行管合并、自环丢弃。

    返回 dict(Nj, indptr, indices, junc_nodes, n_edges, n_nodes, n_links)。
    槽位序：junc_nodes = np.where(node_type == 0)[0]，槽位 j 对应节点 junc_nodes[j]。
    """
    nt = np.asarray(node_type)
    n1 = np.asarray(link_n1, dtype=np.int64)
    n2 = np.asarray(link_n2, dtype=np.int64)
    N = int(nt.shape[0])
    junc_nodes = np.where(nt == 0)[0]
    Nj = int(junc_nodes.shape[0])
    slot = np.full(N, -1, dtype=np.int64)
    slot[junc_nodes] = np.arange(Nj, dtype=np.int64)
    s1, s2 = slot[n1], slot[n2]
    keep = (s1 >= 0) & (s2 >= 0) & (s1 != s2)
    a, b = s1[keep], s2[keep]
    if a.size:
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        key = lo * np.int64(Nj) + hi
        _, uix = np.unique(key, return_index=True)
        lo, hi = lo[uix], hi[uix]
    else:
        lo = hi = np.zeros(0, dtype=np.int64)
    n_edges = int(lo.shape[0])
    rows = np.concatenate([lo, hi])
    cols = np.concatenate([hi, lo])
    cnt = np.bincount(rows, minlength=Nj)
    indptr = np.zeros(Nj + 1, dtype=np.int64)
    np.cumsum(cnt, out=indptr[1:])
    order = np.argsort(rows, kind="stable")
    indices = cols[order].astype(np.int64)
    return dict(Nj=Nj, indptr=indptr, indices=indices, junc_nodes=junc_nodes,
                n_edges=n_edges, n_nodes=N, n_links=int(n1.shape[0]))


def permute_pattern(Nj, indptr, indices, iperm):
    """把对称模式按 iperm（旧槽位 → 新行号）重排，返回新的 (indptr, indices)。"""
    iperm = np.asarray(iperm, dtype=np.int64)
    src = np.repeat(np.arange(Nj, dtype=np.int64), np.diff(indptr))
    rows = iperm[src]
    cols = iperm[indices]
    cnt = np.bincount(rows, minlength=Nj)
    ip = np.zeros(Nj + 1, dtype=np.int64)
    np.cumsum(cnt, out=ip[1:])
    order = np.argsort(rows, kind="stable")
    return ip, cols[order].astype(np.int64)


# ======================================================================
# 1. 消元树（Liu 1986，带路径压缩，O(nnz·α(n))）
# ======================================================================
def etree_from_pattern(n, indptr, indices):
    """由 A 的对称模式算消元树 parent（0 基，根的 parent = -1）。

    等价定义：parent[j] = min{ i > j : L[i,j] ≠ 0 }（L 为精确符号 Cholesky 因子）。
    本实现不物化 L，直接在 A 上做带路径压缩的并查集。
    """
    parent = np.full(n, -1, dtype=np.int64)
    ancestor = np.full(n, -1, dtype=np.int64)
    ip = indptr
    ix = indices
    for k in range(n):
        for p in range(ip[k], ip[k + 1]):
            i = int(ix[p])
            if i >= k:
                continue
            while ancestor[i] != -1 and ancestor[i] != k:
                nxt = int(ancestor[i])
                ancestor[i] = k
                i = nxt
            if ancestor[i] == -1:
                ancestor[i] = k
                parent[i] = k
    return parent


def postorder(parent):
    """消元森林的后序（子先于父）。返回 post[k] = 第 k 个被访问的节点。"""
    n = int(parent.shape[0])
    head = np.full(n, -1, dtype=np.int64)     # head[p] = p 的首个子节点
    nxt = np.full(n, -1, dtype=np.int64)      # 兄弟链
    for j in range(n - 1, -1, -1):            # 逆序入链 → 子节点升序
        p = int(parent[j])
        if p != -1:
            nxt[j] = head[p]
            head[p] = j
    post = np.empty(n, dtype=np.int64)
    k = 0
    stack = np.empty(n, dtype=np.int64)
    for r in range(n):
        if parent[r] != -1:
            continue
        top = 0
        stack[0] = r
        while top >= 0:
            j = int(stack[top])
            c = int(head[j])
            if c == -1:
                post[k] = j
                k += 1
                top -= 1
            else:
                head[j] = nxt[c]
                top += 1
                stack[top] = c
    assert k == n, "postorder 未覆盖全部节点（parent 非森林？）"
    return post


def tree_depth_size(parent, post):
    """返回 (depth[n], subtree_size[n], height)。根 depth=0，height = max(depth)+1。"""
    n = int(parent.shape[0])
    depth = np.zeros(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)
    for j in post[::-1]:                      # 父先于子
        p = int(parent[j])
        if p != -1:
            depth[j] = depth[p] + 1
    for j in post:                            # 子先于父
        p = int(parent[j])
        if p != -1:
            size[p] += size[j]
    height = int(depth.max()) + 1 if n else 0
    return depth, size, height


# ======================================================================
# 2. 列计数（Gilbert–Ng–Peyton；CSparse cs_counts 移植） - 不物化 L
# ======================================================================
def col_counts(n, indptr, indices, parent, post):
    """精确列计数 colcount[j] = |L(:,j)|（含对角），O(nnz·α(n))，不物化 L。

    随机序/自然序在大网上填充可达 O(n²)，只有走列计数才跑得动。
    """
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    delta = np.zeros(n, dtype=np.int64)
    ancestor = np.arange(n, dtype=np.int64)
    maxfirst = np.full(n, -1, dtype=np.int64)
    prevleaf = np.full(n, -1, dtype=np.int64)
    first = np.full(n, -1, dtype=np.int64)
    for k in range(n):
        j = int(post[k])
        delta[j] = 1 if first[j] == -1 else 0
        while j != -1 and first[j] == -1:
            first[j] = k
            j = int(parent[j])
    ip, ix = indptr, indices
    for k in range(n):
        j = int(post[k])
        pj = int(parent[j])
        if pj != -1:
            delta[pj] -= 1
        for p in range(ip[j], ip[j + 1]):
            i = int(ix[p])
            # cs_leaf(i, j)
            if i <= j or first[j] <= maxfirst[i]:
                continue                       # j 不是 i 的子树的叶
            maxfirst[i] = first[j]
            jprev = int(prevleaf[i])
            prevleaf[i] = j
            delta[j] += 1                      # A(i,j) 在骨架矩阵中
            if jprev != -1:                    # 非首叶 → 扣掉与前一叶的重叠
                q = jprev
                while q != ancestor[q]:
                    q = int(ancestor[q])
                s = jprev
                while s != q:                  # 路径压缩
                    sp = int(ancestor[s])
                    ancestor[s] = q
                    s = sp
                delta[q] -= 1                  # q = lca(jprev, j)
        if pj != -1:
            ancestor[j] = pj
    colcount = delta.copy()
    for k in range(n):
        j = int(post[k])
        p = int(parent[j])
        if p != -1:
            colcount[p] += colcount[j]
    return colcount


# ======================================================================
# 3. 完整符号分解（行子树法，O(nnz(L))） - 需要物化模式时才用
# ======================================================================
def symbolic_pattern(n, indptr, indices, parent):
    """精确符号 Cholesky：返回严格下三角 L 模式的 COO (rows, cols)，rows > cols。

    行子树法：行 k 的模式 = ∪_{i∈A(k,·), i<k} etree 上 i→k 的路径（遇标记即停）。
    """
    mark = np.full(n, -1, dtype=np.int64)
    rows, cols = [], []
    ip, ix = indptr, indices
    for k in range(n):
        mark[k] = k
        for p in range(ip[k], ip[k + 1]):
            i = int(ix[p])
            if i >= k:
                continue
            j = i
            while j != -1 and mark[j] != k:
                mark[j] = k
                rows.append(k)
                cols.append(j)
                j = int(parent[j])
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    order = np.lexsort((r, c))                 # 列主序、列内行号升序
    return r[order], c[order]


# ======================================================================
# 4. 排序方案统计（掩码优化器的定量证据）
# ======================================================================
class OrderingStats(dict):
    """排序方案的结构指标（dict 子类，便于直接 json 落盘）。"""

    __getattr__ = dict.__getitem__


def ordering_stats(Nj, indptr, indices, iperm, n_edges=None, name=""):
    """给定排序（iperm: 旧槽位→新行号）算 nnz(L)/填充比/树高/FLOPs。"""
    if n_edges is None:
        n_edges = int(indices.shape[0]) // 2
    if Nj == 0:
        return OrderingStats(name=name, Nj=0, nnz_A_lower=0, nnz_L=0,
                             fill_ratio=float("nan"), etree_height=0,
                             flops_sparse=0.0, flops_dense=0.0,
                             flops_ratio=float("nan"), max_col_height=0)
    ip, ix = permute_pattern(Nj, indptr, indices, iperm)
    parent = etree_from_pattern(Nj, ip, ix)
    post = postorder(parent)
    cc = col_counts(Nj, ip, ix, parent, post)
    _, _, height = tree_depth_size(parent, post)
    eta = (cc - 1).astype(np.float64)          # 列高度（对角以下非零数）
    nnz_L = int(cc.sum())
    nnz_A_lower = int(n_edges) + Nj
    flops_sparse = float((eta * eta).sum())
    dj = np.arange(Nj - 1, -1, -1, dtype=np.float64)
    flops_dense = float((dj * dj).sum())       # Σ (Nj-1-j)² = 稠密同口径
    return OrderingStats(
        name=name, Nj=int(Nj), nnz_A_lower=nnz_A_lower, nnz_L=nnz_L,
        fill_ratio=nnz_L / nnz_A_lower, etree_height=int(height),
        flops_sparse=flops_sparse, flops_dense=flops_dense,
        flops_ratio=(flops_dense / flops_sparse) if flops_sparse > 0 else float("inf"),
        max_col_height=int(eta.max()) if Nj else 0,
        nnz_L_offdiag=int(nnz_L - Nj))


def mmd_permutation(node_type, link_n1, link_n2):
    """跑 EPANET 的 genmmd（dgga.smatrix）拿 MMD 排序。

    返回 (perm, iperm, sm)：perm[r] = 行 r 的 junction 槽位；iperm[j] = 槽位 j 的行号；
    sm = EpanetSmatrix（带 XLNZ/NZSUB/LNZ/Ndx/Ncoeffs）。
    """
    try:
        from dgga.smatrix import EpanetSmatrix
    except ImportError:                        # pragma: no cover
        from smatrix import EpanetSmatrix
    nt = np.asarray(node_type)
    n1 = np.asarray(link_n1, dtype=np.int64)
    n2 = np.asarray(link_n2, dtype=np.int64)
    N = int(nt.shape[0])
    junc = np.where(nt == 0)[0]
    fixed = np.where(nt != 0)[0]
    Nj = int(junc.shape[0])
    ep_of = np.zeros(N, dtype=np.int64)
    ep_of[junc] = np.arange(1, Nj + 1)
    ep_of[fixed] = np.arange(Nj + 1, N + 1)
    sm = EpanetSmatrix(N, Nj, list(zip(ep_of[n1].tolist(), ep_of[n2].tolist())))
    Order = np.asarray(sm.Order, dtype=np.int64)
    Row = np.asarray(sm.Row, dtype=np.int64)
    perm = Order[1:Nj + 1] - 1                 # 行 r(0 基) → 槽位
    iperm = Row[1:Nj + 1] - 1                  # 槽位 j → 行号
    return perm, iperm, sm


# ======================================================================
# 5. 主产物容器
# ======================================================================
class SymbolicStructure:
    """一个网（一个排序）的全部结构先验。索引一律 0 基、行空间。

    属性：
      Nj, n_nodes, n_links, n_edges, ordering
      perm[Nj], iperm[Nj]                      行空间 ↔ junction 槽位
      l_rows/l_cols                            严格下三角 L 模式（列主序）
      lnz_slot                                 与 l_rows 对齐的 EPANET Aij 槽位（仅 mmd 排序）
      ndx[n_links]                             链路 → Aij 槽位（仅 mmd 排序）
      chordal_index[2, nnz_chordal]            pattern(L+Lᵀ)+对角（对称、含自环）
      a_index[2, nnz_A]                        pattern(A)+对角（未填充，消融对照）
      parent[Nj], depth[Nj], subtree_size[Nj], post[Nj], colcount[Nj], height
    """

    def __init__(self, Nj, perm, iperm, l_rows, l_cols, a_indptr, a_indices,
                 parent, ordering="mmd", n_nodes=0, n_links=0, n_edges=0,
                 lnz_slot=None, ndx=None):
        self.Nj = int(Nj)
        self.ordering = ordering
        self.n_nodes, self.n_links, self.n_edges = int(n_nodes), int(n_links), int(n_edges)
        self.perm = np.asarray(perm, dtype=np.int64)
        self.iperm = np.asarray(iperm, dtype=np.int64)
        self.l_rows = np.asarray(l_rows, dtype=np.int64)
        self.l_cols = np.asarray(l_cols, dtype=np.int64)
        self.lnz_slot = None if lnz_slot is None else np.asarray(lnz_slot, dtype=np.int64)
        self.ndx = None if ndx is None else np.asarray(ndx, dtype=np.int64)
        self.a_indptr = np.asarray(a_indptr, dtype=np.int64)
        self.a_indices = np.asarray(a_indices, dtype=np.int64)
        self.parent = np.asarray(parent, dtype=np.int64)
        self.post = postorder(self.parent)
        self.depth, self.subtree_size, self.height = tree_depth_size(self.parent, self.post)
        self.colcount = np.bincount(self.l_cols, minlength=self.Nj).astype(np.int64) + 1
        # ---- ① 填充掩码：pattern(L + Lᵀ) + 对角 ----
        self.chordal_index = self._sym_index(self.l_rows, self.l_cols)
        # ---- ③ 原始邻接：pattern(A) + 对角 ----
        src = np.repeat(np.arange(self.Nj, dtype=np.int64), np.diff(self.a_indptr))
        m = src > self.a_indices                # 取严格下三角再对称化，保证与 ① 同口径
        self.a_index = self._sym_index(src[m], self.a_indices[m])

    def _sym_index(self, rows, cols):
        d = np.arange(self.Nj, dtype=np.int64)
        r = np.concatenate([rows, cols, d])
        c = np.concatenate([cols, rows, d])
        order = np.lexsort((c, r))              # 行主序（CSR 友好）
        return np.stack([r[order], c[order]], axis=0)

    # ------------------------------------------------------------------
    @property
    def nnz_L(self):
        """|L|（含对角）。"""
        return int(self.l_rows.shape[0]) + self.Nj

    @property
    def nnz_chordal(self):
        return int(self.chordal_index.shape[1])

    @property
    def nnz_A_sym(self):
        return int(self.a_index.shape[1])

    def stats(self):
        eta = (self.colcount - 1).astype(np.float64)
        dj = np.arange(self.Nj - 1, -1, -1, dtype=np.float64)
        nnz_A_lower = self.n_edges + self.Nj
        fs = float((eta * eta).sum())
        return OrderingStats(
            name=self.ordering, Nj=self.Nj, nnz_A_lower=nnz_A_lower,
            nnz_L=self.nnz_L, fill_ratio=self.nnz_L / max(nnz_A_lower, 1),
            etree_height=int(self.height), flops_sparse=fs,
            flops_dense=float((dj * dj).sum()),
            flops_ratio=(float((dj * dj).sum()) / fs) if fs > 0 else float("inf"),
            max_col_height=int(eta.max()) if self.Nj else 0,
            nnz_L_offdiag=int(self.l_rows.shape[0]))

    def torch(self, device=None):
        """转成 torch 索引张量（int64）。一次性预计算、跨层跨场景复用。"""
        import torch
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long,
                                      device=device)
        out = dict(chordal_index=t(self.chordal_index), a_index=t(self.a_index),
                   l_rows=t(self.l_rows), l_cols=t(self.l_cols),
                   perm=t(self.perm), iperm=t(self.iperm),
                   parent=t(np.where(self.parent < 0, np.arange(self.Nj), self.parent)),
                   parent_raw=t(self.parent), depth=t(self.depth),
                   subtree_size=t(self.subtree_size), post=t(self.post),
                   colcount=t(self.colcount))
        if self.lnz_slot is not None:
            out["lnz_slot"] = t(self.lnz_slot)
        if self.ndx is not None:
            out["ndx"] = t(self.ndx)
        return out

    # ------------------------------------------------------------------
    def check_symmetry(self):
        """④c：掩码必须对称、必须含全部对角。返回 dict(symmetric, has_diag, n_diag)。"""
        r, c = self.chordal_index
        key = r * np.int64(self.Nj) + c
        keyT = c * np.int64(self.Nj) + r
        symmetric = bool(np.array_equal(np.sort(key), np.sort(keyT)))
        n_diag = int((r == c).sum())
        return dict(symmetric=symmetric, has_diag=(n_diag == self.Nj),
                    n_diag=n_diag, n_unique=int(np.unique(key).size),
                    nnz=int(key.size))

    def check_etree(self, max_cols=None):
        """④b：验证 struct(L(:,j))\\{j} ⊆ etree 上 parent[j]→根 的路径（Liu 定理）。

        返回 dict(ok, n_checked, n_violations, first_violation)。
        """
        n = self.Nj
        starts = np.searchsorted(self.l_cols, np.arange(n), side="left")
        ends = np.searchsorted(self.l_cols, np.arange(n), side="right")
        cols = range(n) if max_cols is None else range(min(n, max_cols))
        onpath = np.full(n, -1, dtype=np.int64)
        viol = 0
        first = None
        checked = 0
        for j in cols:
            a, b = int(starts[j]), int(ends[j])
            if b <= a:
                continue
            checked += 1
            p = int(self.parent[j])
            while p != -1:                      # 标记 parent[j]→根 的祖先路径
                onpath[p] = j
                p = int(self.parent[p])
            for k in range(a, b):
                i = int(self.l_rows[k])
                if onpath[i] != j:
                    viol += 1
                    if first is None:
                        first = (i, j)
        # 附带核对 parent[j] == min{i>j: L[i,j]≠0}
        pmin = np.full(n, -1, dtype=np.int64)
        for j in range(n):
            a, b = int(starts[j]), int(ends[j])
            if b > a:
                pmin[j] = int(self.l_rows[a:b].min())
        parent_ok = bool(np.array_equal(pmin, self.parent))
        return dict(ok=(viol == 0 and parent_ok), n_checked=checked,
                    n_violations=viol, first_violation=first,
                    parent_is_min_row=parent_ok)


def build_symbolic(node_type, link_n1, link_n2, ordering="mmd", seed=0,
                   pattern=None):
    """构造 SymbolicStructure。ordering ∈ {mmd, natural, random, rcm}。

    mmd 走 EPANET genmmd + 其 XLNZ/NZSUB 填充模式（含 LNZ 槽位映射）；
    其余排序走自写符号分解（行子树法）。
    """
    pat = pattern if pattern is not None else junction_pattern(node_type, link_n1, link_n2)
    Nj, ip, ix = pat["Nj"], pat["indptr"], pat["indices"]
    lnz_slot = ndx = None
    if ordering == "mmd":
        perm, iperm, sm = mmd_permutation(node_type, link_n1, link_n2)
        XLNZ = np.asarray(sm.XLNZ, dtype=np.int64)
        NZSUB = np.asarray(sm.NZSUB, dtype=np.int64)
        LNZ = np.asarray(sm.LNZ, dtype=np.int64)
        nz = int(XLNZ[Nj + 1] - 1) if Nj else 0
        l_rows = NZSUB[1:nz + 1] - 1
        l_cols = np.repeat(np.arange(Nj, dtype=np.int64), np.diff(XLNZ[1:Nj + 2]))
        lnz_slot = LNZ[1:nz + 1]
        ndx = np.asarray(sm.Ndx[1:], dtype=np.int64)
        pip, pix = permute_pattern(Nj, ip, ix, iperm)
        parent = etree_from_pattern(Nj, pip, pix)
    else:
        if ordering == "natural":
            perm = np.arange(Nj, dtype=np.int64)
        elif ordering == "random":
            perm = np.random.default_rng(seed).permutation(Nj).astype(np.int64)
        elif ordering == "rcm":
            perm = _rcm_perm(Nj, ip, ix)
        else:
            raise ValueError(f"未知排序 {ordering}")
        iperm = np.empty(Nj, dtype=np.int64)
        iperm[perm] = np.arange(Nj, dtype=np.int64)
        pip, pix = permute_pattern(Nj, ip, ix, iperm)
        parent = etree_from_pattern(Nj, pip, pix)
        l_rows, l_cols = symbolic_pattern(Nj, pip, pix, parent)
    return SymbolicStructure(Nj, perm, iperm, l_rows, l_cols, pip, pix, parent,
                             ordering=ordering, n_nodes=pat["n_nodes"],
                             n_links=pat["n_links"], n_edges=pat["n_edges"],
                             lnz_slot=lnz_slot, ndx=ndx)


def _rcm_perm(Nj, indptr, indices):
    """RCM 排序（scipy）。返回 perm[r] = 行 r 的旧槽位。"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    A = csr_matrix((np.ones(indices.shape[0], dtype=np.int8), indices, indptr),
                   shape=(Nj, Nj))
    return np.asarray(reverse_cuthill_mckee(A, symmetric_mode=True), dtype=np.int64)


# ======================================================================
# 6. 自检冒烟（python -X utf8 -m dgga.symbolic）
# ======================================================================
if __name__ == "__main__":                                     # pragma: no cover
    import os
    import sys
    import torch

    sys.stdout.reconfigure(encoding="utf-8")
    from dgga.parse import Net

    ref = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "reference")
    for stem in ("pub_hanoi", "city_d", "pub_net6"):
        net = Net.load(ref, stem)
        S = build_symbolic(net.node_type, net.link_n1, net.link_n2, "mmd")
        td = S.torch()
        # ① 置换往返：x_row = x_junc[perm]；x_junc = x_row[iperm]
        x = np.random.default_rng(0).normal(size=S.Nj)
        rt = bool(np.array_equal(x[S.perm][S.iperm], x))
        # ② 掩码可直接建稀疏张量（W2 的注意力就在这个模式上算）
        vals = torch.ones(S.nnz_chordal, dtype=torch.float64)
        M = torch.sparse_coo_tensor(td["chordal_index"], vals,
                                    (S.Nj, S.Nj)).coalesce()
        dense_ok = bool((M.to_dense() == M.to_dense().T).all())
        # ③ LNZ 槽位有效性（W2 装配 M 的值时用）
        slot_ok = bool(S.lnz_slot.min() >= 1 and
                       S.lnz_slot.size == S.l_rows.size)
        print(f"[{stem:<10s}] Nj={S.Nj:<6d} nnzL={S.nnz_L:<7d} 掩码nnz={S.nnz_chordal:<7d} "
              f"树高={S.height:<4d} 置换往返={rt} 稀疏对称={dense_ok} 槽位={slot_ok} "
              f"检查={S.check_symmetry()['symmetric'] and S.check_etree()['ok']}")

# -*- coding: utf-8 -*-
"""dgga.cluster - 簇级（子区）漏损定位：相干矩阵 × 管网拓扑的受约束聚类 + 组稀疏反演。

本模块只新增函数，不改任何既有行为（dgga 其余模块一行未动）。

问题设定
--------
候选漏点 j = 1..NC，传感集 S（行下标）。签名字典 A(S) ∈ R^{n×NC}，
n = 帧数 × |S|，第 j 列 a_j = 单位漏损系数 C 在 S 上的压力响应（ft / C）。
观测 y = h(真值漏损) − h(基线)，加 N(0, σ²) 传感噪声。

单点定位在高相干时不适定：μ_ij = |a_iᵀa_j| / (‖a_i‖‖a_j‖) → 1 的一对，
任何反演都只能给出"两者之一"。本模块把不可分的近共线候选**合并成拓扑簇**，
把定位目标从"哪个节点"降级为"哪个子区"，并如实报告降级换来的模糊半径。

一、拓扑图（parameter-free）
    管网 Voronoi：以全部候选为源做多源 Dijkstra（边权 = 管长），每个网络节点归属
    最近的候选；若存在一条管 (u,v) 使 owner(u)=i ≠ owner(j)=j，则候选 i、j 拓扑相邻。
    候选间的管网距离 d_ij（米）由逐候选 Dijkstra 给出。无阈值参数。

二、受约束凝聚层次聚类（相干 × 拓扑）
    相似度 = 互相干 μ_ij；只允许**拓扑相邻**的两簇合并（regionalization 的连通约束）。
    linkage="complete"：s(g,h) = min_{i∈g, j∈h} μ_ij（保证簇内**任意**两点相干 ≥ τ）；
    "average" / "single" 亦可。反复合并 argmax s(g,h)，直到 max s < τ 停机。
    输出 labels[NC]、合并轨迹。复杂度 O(NC³)（NC ≤ 100 时可忽略；用堆可降到
    O(NC² log NC)）。τ=1 → 全单点（退化为单点定位）；τ=0 → 每个拓扑连通分量一簇。

三、组稀疏反演（组 = 拓扑簇）
    min_{x ≥ 0} ½‖A x − y‖² + λ Σ_g w_g ‖x_g‖₂ ,  w_g = √|g|
    非负 + 组 L2 的近端算子（本文件 _prox_group_nn 有推导）：
        prox(v)_g = (1 − λw_g/‖v_g⁺‖₂)₊ · v_g⁺ ,  v⁺ = max(v, 0)
    用 FISTA（Nesterov 动量 + 回溯免除：步长 1/L，L = ‖A‖₂²）求解，
    每步 O(n·NC)。groups 取单点集合时退化为**非负 Lasso = 单点定位对照**。

四、判据
    簇级命中：真漏点所在簇按 ‖x_g‖₂ 排名进前 r。
    模糊半径：簇内候选的最大两两欧氏距离（米，坐标为投影米制）；
              另给"子区管长"= 两端都归属该簇的管长之和（km，= 巡检工作量）。
"""
import heapq

import numpy as np

__all__ = ["pipe_graph", "voronoi_owner", "candidate_adjacency",
           "candidate_pipe_distance", "constrained_agglomerative",
           "cluster_geometry", "group_lasso_nn", "lambda_max_nn",
           "group_scores", "detection_noncentrality"]


# ======================================================================
# 一、拓扑
# ======================================================================
def pipe_graph(N, n1, n2, weight, keep=None):
    """无向加权邻接表。keep: 布尔 [L]，只保留为真的链路（缺省全保留）。"""
    n1 = np.asarray(n1, dtype=np.int64)
    n2 = np.asarray(n2, dtype=np.int64)
    w = np.asarray(weight, dtype=np.float64)
    if keep is None:
        keep = np.ones(n1.size, dtype=bool)
    adj = [[] for _ in range(int(N))]
    for a, b, ww, k in zip(n1.tolist(), n2.tolist(), w.tolist(), np.asarray(keep).tolist()):
        if not k or a == b:
            continue
        ww = max(float(ww), 0.0)
        adj[a].append((b, ww))
        adj[b].append((a, ww))
    return adj


def _dijkstra(adj, sources, init=0.0):
    """多源 Dijkstra，返回 (dist[N], owner[N])；owner = 最近源的下标（−1 = 不可达）。"""
    N = len(adj)
    dist = np.full(N, np.inf)
    owner = np.full(N, -1, dtype=np.int64)
    pq = []
    for k, s in enumerate(sources):
        dist[s] = init
        owner[s] = k
        heapq.heappush(pq, (init, int(s), k))
    while pq:
        d, u, o = heapq.heappop(pq)
        if d > dist[u] + 1e-12 or owner[u] != o:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v] - 1e-12:
                dist[v] = nd
                owner[v] = o
                heapq.heappush(pq, (nd, v, o))
    return dist, owner


def voronoi_owner(adj, cand_nodes):
    """管网 Voronoi：每个节点归属最近的候选（下标），返回 (dist[N], owner[N])。"""
    return _dijkstra(adj, np.asarray(cand_nodes, dtype=np.int64).tolist())


def candidate_adjacency(N, n1, n2, weight, cand_nodes, keep=None):
    """候选级拓扑相邻矩阵（无参数）：存在一条管两端分属不同 Voronoi 元 → 两候选相邻。

    返回 dict(adjacency[NC,NC] bool, owner[N], vor_dist[N], n_edges)。
    """
    cand_nodes = np.asarray(cand_nodes, dtype=np.int64)
    NC = cand_nodes.size
    adj = pipe_graph(N, n1, n2, weight, keep)
    vd, owner = voronoi_owner(adj, cand_nodes)
    A = np.zeros((NC, NC), dtype=bool)
    n1 = np.asarray(n1, dtype=np.int64)
    n2 = np.asarray(n2, dtype=np.int64)
    kp = np.ones(n1.size, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    for a, b, k in zip(n1.tolist(), n2.tolist(), kp.tolist()):
        if not k:
            continue
        oa, ob = owner[a], owner[b]
        if oa >= 0 and ob >= 0 and oa != ob:
            A[oa, ob] = A[ob, oa] = True
    return dict(adjacency=A, owner=owner, vor_dist=vd,
                n_edges=int(A.sum() // 2))


def candidate_pipe_distance(N, n1, n2, weight, cand_nodes, keep=None):
    """候选两两的管网最短路（米/与 weight 同单位），[NC, NC]，不可达为 inf。"""
    cand_nodes = np.asarray(cand_nodes, dtype=np.int64)
    adj = pipe_graph(N, n1, n2, weight, keep)
    out = np.zeros((cand_nodes.size, cand_nodes.size))
    for k, s in enumerate(cand_nodes.tolist()):
        d, _ = _dijkstra(adj, [s])
        out[k] = d[cand_nodes]
    return 0.5 * (out + out.T)


# ======================================================================
# 二、受约束凝聚层次聚类
# ======================================================================
def constrained_agglomerative(sim, adjacent, tau, linkage="complete", min_clusters=None):
    """相干相似度 sim[NC,NC] + 拓扑相邻 adjacent[NC,NC] → 阈值 τ 下的簇。

    只合并拓扑相邻的簇（簇相邻 = 成员间存在相邻对）；linkage:
      "complete" s(g,h) = min 交叉相干（簇内任意两点相干 ≥ τ 的充要停机）
      "average"  平均      "single" 最大
    合并到 max s(g,h) < τ 为止。adjacent=None → 无拓扑约束（纯相干聚类，消融用）。
    min_clusters 给定时，簇数降到它就停（配 tau=-inf 可做"给定簇数"的几何对照）。

    返回 dict(labels[NC], n_clusters, merges[...], tau, linkage,
              constrained（是否加了拓扑约束）)。
    """
    S = np.array(sim, dtype=np.float64, copy=True)
    NC = S.shape[0]
    np.fill_diagonal(S, -np.inf)
    if adjacent is None:
        Adj = np.ones((NC, NC), dtype=bool)
        np.fill_diagonal(Adj, False)
        constrained = False
    else:
        Adj = np.array(adjacent, dtype=bool, copy=True)
        np.fill_diagonal(Adj, False)
        constrained = True
    if linkage not in ("complete", "average", "single"):
        raise ValueError("linkage 必须是 'complete' | 'average' | 'single'")
    members = {i: [i] for i in range(NC)}
    alive = np.ones(NC, dtype=bool)
    L = S.copy()                       # 簇间 linkage 值（活簇为准）
    cnt = np.ones((NC, NC))            # average 用的交叉对计数
    merges = []
    while True:
        if min_clusters is not None and int(alive.sum()) <= int(min_clusters):
            break
        M = np.where(Adj & alive[:, None] & alive[None, :], L, -np.inf)
        if not np.isfinite(M).any():
            break
        k = int(np.argmax(M))
        i, j = divmod(k, NC)
        best = M[i, j]
        if best < tau:
            break
        if i > j:
            i, j = j, i
        # 合并 j → i
        if linkage == "complete":
            newL = np.minimum(L[i], L[j])
        elif linkage == "single":
            newL = np.maximum(L[i], L[j])
        else:
            tot = L[i] * cnt[i] + L[j] * cnt[j]
            newc = cnt[i] + cnt[j]
            newL = tot / np.maximum(newc, 1e-300)
            cnt[i] = cnt[:, i] = newc
        L[i] = L[:, i] = newL
        Adj[i] = Adj[i] | Adj[j]
        Adj[:, i] = Adj[i]
        Adj[i, i] = False
        alive[j] = False
        merges.append(dict(step=len(merges) + 1, similarity=float(best),
                           size_a=len(members[i]), size_b=len(members[j])))
        members[i] = members[i] + members[j]
        del members[j]
    labels = np.full(NC, -1, dtype=np.int64)
    for c, (_, mem) in enumerate(sorted(members.items())):
        labels[np.asarray(mem, dtype=np.int64)] = c
    return dict(labels=labels, n_clusters=int(labels.max()) + 1, merges=merges,
                tau=float(tau), linkage=linkage, constrained=bool(constrained))


def cluster_geometry(labels, xy, pipe_dist=None, owner=None, len_by_link=None,
                     link_n1=None, link_n2=None):
    """每簇的几何：成员数、欧氏直径（坐标单位）、质心、管网直径、子区管长。

    xy       : [NC, 2] 候选坐标（投影米制）
    pipe_dist: [NC, NC] 候选两两管网最短路（可选）
    owner/len_by_link/link_n1/link_n2: 给出即算"子区管长"= 两端都归属本簇的管长之和
    """
    labels = np.asarray(labels, dtype=np.int64)
    xy = np.asarray(xy, dtype=np.float64)
    ncl = int(labels.max()) + 1
    sub_len = None
    if owner is not None and len_by_link is not None:
        owner = np.asarray(owner, dtype=np.int64)
        a = np.asarray(link_n1, dtype=np.int64)
        b = np.asarray(link_n2, dtype=np.int64)
        w = np.asarray(len_by_link, dtype=np.float64)
        la = np.where(owner[a] >= 0, labels[np.clip(owner[a], 0, None)], -1)
        lb = np.where(owner[b] >= 0, labels[np.clip(owner[b], 0, None)], -1)
        sub_len = np.zeros(ncl)
        m = (la == lb) & (la >= 0)
        np.add.at(sub_len, la[m], w[m])
        m2 = (la != lb) & (la >= 0) & (lb >= 0)         # 界管：两簇各记一半
        np.add.at(sub_len, la[m2], 0.5 * w[m2])
        np.add.at(sub_len, lb[m2], 0.5 * w[m2])
    out = []
    for c in range(ncl):
        idx = np.where(labels == c)[0]
        p = xy[idx]
        if idx.size > 1:
            dd = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
            diam = float(dd.max())
            rad = float(np.hypot(*(p - p.mean(0)).T).max())
        else:
            diam = rad = 0.0
        e = dict(cluster=c, size=int(idx.size), diameter=diam, radius_centroid=rad)
        if pipe_dist is not None and idx.size > 1:
            sub = pipe_dist[np.ix_(idx, idx)]
            e["pipe_diameter"] = float(np.max(sub[np.isfinite(sub)])) if np.isfinite(sub).any() else float("inf")
        elif pipe_dist is not None:
            e["pipe_diameter"] = 0.0
        if sub_len is not None:
            e["district_pipe_len"] = float(sub_len[c])
        out.append(e)
    return out


# ======================================================================
# 三、非负组 Lasso（FISTA）
# ======================================================================
def _groups_from_labels(labels):
    labels = np.asarray(labels, dtype=np.int64)
    return [np.where(labels == c)[0] for c in range(int(labels.max()) + 1)]


def _prox_group_nn(v, labels, thr, ngroup):
    """prox of  Σ_g thr_g‖x_g‖₂ + ι_{x≥0}  在 v 处（按 labels 向量化）。

    推导：min_{x≥0} ½‖x−v‖² + t‖x‖₂。KKT 给出：x_i>0 的坐标满足
    x_i(1+t/‖x‖) = v_i（故必 v_i>0），x_i=0 的坐标满足 v_i ≤ 0；
    于是 x = (1 − t/‖v⁺‖)₊ v⁺，其中 v⁺ = max(v,0)。
    """
    out = np.maximum(v, 0.0)
    nrm = np.sqrt(np.bincount(labels, weights=out * out, minlength=ngroup))
    sc = np.where(nrm > thr, 1.0 - thr / np.maximum(nrm, 1e-300), 0.0)
    return out * sc[labels]


def lambda_max_nn(A, y, labels, weights=None):
    """使 x=0 成为非负组 Lasso 最优解的最小 λ：max_g ‖(A_gᵀy)₊‖₂ / w_g。"""
    groups = _groups_from_labels(labels)
    w = _weights(groups, weights)
    c = np.maximum(np.asarray(A, dtype=np.float64).T @ np.asarray(y, dtype=np.float64).ravel(), 0.0)
    return float(max(np.linalg.norm(c[g]) / wg for g, wg in zip(groups, w)))


def _weights(groups, weights):
    if weights is None:
        return np.array([np.sqrt(g.size) for g in groups], dtype=np.float64)
    return np.asarray(weights, dtype=np.float64)


def _spec_norm2(A, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        u = A @ v
        nu = np.linalg.norm(u)
        if nu == 0:
            return 0.0
        v = A.T @ (u / nu)
        nv = np.linalg.norm(v)
        if nv == 0:
            return 0.0
        v /= nv
        new = nv * nu
        if abs(new - lam) <= 1e-12 * max(new, 1.0):
            lam = new
            break
        lam = new
    return float(lam)


def group_lasso_nn(A, y, labels, lam, weights=None, max_iter=4000, tol=1e-9,
                   x0=None, lipschitz=None, check_every=20):
    """min_{x≥0} ½‖Ax−y‖² + λ Σ_g w_g‖x_g‖₂（FISTA，步长 1/L）。

    labels 为逐候选的组号（单点组 = 非负 Lasso）。返回
    dict(x, n_iter, obj, rss, active_groups, converged, lipschitz)。
    每步两次矩阵-向量乘（A@z 与 Aᵀr）：O(n·NC)；目标值每 check_every 步查一次。
    """
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64)
    ng = int(labels.max()) + 1
    groups = _groups_from_labels(labels)
    w = _weights(groups, weights)
    L = float(lipschitz) if lipschitz is not None else _spec_norm2(A) ** 2
    L = max(L, 1e-300)
    x = np.zeros(A.shape[1]) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    z, t = x.copy(), 1.0
    thr = lam * w / L
    obj = np.inf
    it = 0
    conv = False
    for it in range(1, int(max_iter) + 1):
        r = A @ z - y
        xn = _prox_group_nn(z - (A.T @ r) / L, labels, thr, ng)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = xn + ((t - 1.0) / tn) * (xn - x)
        dx = np.linalg.norm(xn - x)
        x, t = xn, tn
        if it % check_every == 0 or it == max_iter:
            rr = A @ x - y
            nrm = np.sqrt(np.bincount(labels, weights=x * x, minlength=ng))
            new = 0.5 * float(np.dot(rr, rr)) + lam * float(np.dot(w, nrm))
            if dx <= tol * max(np.linalg.norm(x), 1.0) and \
               abs(obj - new) <= tol * max(abs(new), 1.0):
                obj = new
                conv = True
                break
            obj = new
    rr = A @ x - y
    rss = float(np.dot(rr, rr))
    nrm = np.sqrt(np.bincount(labels, weights=x * x, minlength=ng))
    return dict(x=x, n_iter=int(it), obj=float(0.5 * rss + lam * float(np.dot(w, nrm))),
                rss=rss, active_groups=int((nrm > 0).sum()),
                converged=bool(conv), lipschitz=float(L))


def group_scores(x, labels):
    """每组的 ‖x_g‖₂ 与 Σ x_g（后者 = 簇内漏损总量，物理量）。"""
    x = np.asarray(x, dtype=np.float64)
    groups = _groups_from_labels(labels)
    return (np.array([np.linalg.norm(x[g]) for g in groups]),
            np.array([x[g].sum() for g in groups]))


# ======================================================================
# 四、可检测性（线性高斯模型下的精确非中心度）
# ======================================================================
def detection_noncentrality(A, mean_signal, labels, sigma):
    """y = s + n, n~N(0, σ²I)。组 g 的 GLRT 统计量 ‖P_g y‖²/σ² ~ χ²_{r_g}(λ_g)，
    λ_g = ‖P_g s‖²/σ²，r_g = rank(A_g)。

    返回每组 dict(rank, ncp, z = ncp/sqrt(2 r_g)（对零分布标准差的标准分）,
    E_alt/E_null)。要点：当 s 落在 A_g 的列空间内（真漏点在本簇），λ_g 与
    单点时的 ‖s‖²/σ² **完全相同** - 合并只增加零分布自由度 r_g，不增加信号。
    """
    A = np.asarray(A, dtype=np.float64)
    s = np.asarray(mean_signal, dtype=np.float64).ravel()
    groups = _groups_from_labels(labels)
    out = []
    for c, g in enumerate(groups):
        Ag = A[:, g]
        q, _ = np.linalg.qr(Ag)
        sv = np.linalg.svd(Ag, compute_uv=False)
        r = int((sv > max(Ag.shape) * np.finfo(float).eps * (sv[0] if sv.size else 0.0)).sum())
        proj = q[:, :max(r, 1)] @ (q[:, :max(r, 1)].T @ s)
        ncp = float(np.dot(proj, proj) / sigma ** 2)
        out.append(dict(cluster=int(c), size=int(g.size), rank=r, ncp=ncp,
                        z=float(ncp / np.sqrt(2.0 * max(r, 1))),
                        E_null=float(r), E_alt=float(r + ncp)))
    return out

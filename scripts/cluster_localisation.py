# -*- coding: utf-8 -*-
"""cluster_localisation.py - 模块一：簇级（子区）漏损定位，实现 + 老实测量。

动机：上一轮已测定三个漏点是三种机理（data/coh_controls_wip.txt） - T2 由**相干**驱动
（劲敌间隙决定找回/丢失），T3 与相干无关，T1 **幅值受限**（全网最大压降 0.089 ft <
0.1 ft 噪声）。相干驱动的候选对本质上是"两者之一"，单点定位注定不适定；本模块把定位
目标降级为"哪个子区"，并如实报告降级换来的模糊半径。

算法（全部在 dgga/cluster.py，只新增函数）：
  变量   候选漏点 j=1..NC 的漏损系数 x_j ≥ 0（用户制 C，q = C·p^γ）；传感集 S；
         字典 A(S) ∈ R^{n×NC}，n = 帧数×|S|，a_j = 单位 C 的传感压力响应（ft）。
         观测 y = h(真值) − h(基线)（+ N(0,σ²)）。互相干 μ_ij = |a_iᵀa_j|/(‖a_i‖‖a_j‖)。
  拓扑   管网 Voronoi（多源 Dijkstra，边权 = 管长 m）→ 候选级相邻矩阵，无阈值参数。
  聚类   受拓扑约束的凝聚层次聚类，complete linkage：只合并相邻簇，
         s(g,h)=min 交叉相干，合并到 max s < τ 停机 ⇔ 每簇内任意两点 μ ≥ τ。
  反演   组稀疏：min_{x≥0} ½‖Ax−y‖² + λ Σ_g √|g| ‖x_g‖₂（FISTA，非负组近端）。
         组 = 拓扑簇；组 = 单点集合时退化为非负 Lasso = **单点定位对照**。
  判据   簇级命中（真漏点所在簇按 ‖x_g‖₂ 进前 r）；模糊半径 = 簇内候选最大两两
         欧氏距离（米）；子区管长（km，巡检工作量）。
  复杂度 Voronoi O(L log N)；候选两两管距 O(NC·L log N)；聚类 O(NC³)（NC ≤ 60）；
         FISTA 每步 O(n·NC)。

阶段
  build  --net city_d|ltown   前向观测 + 拓扑 + 坐标 → data/cluster_cache_<net>.npz
  run    --net city_d|ltown   τ 扫描 × 传感配置 × 噪声组 → data/cluster_<net>.json
  report                      汇总 → data/cluster_localisation_wip.txt /
                              data/cluster_localisation.json / data/fig_cluster_localisation.png

运行：python -X utf8 scripts/cluster_localisation.py --stage build --net city_d
可读输出零节点/链路/传感器编号（过 augment_suite.assert_no_ids）。
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.cluster import (candidate_adjacency, candidate_pipe_distance,      # noqa: E402
                          cluster_geometry, constrained_agglomerative,
                          detection_noncentrality, group_lasso_nn,
                          group_scores, lambda_max_nn)

LABELS = ("T1", "T2", "T3")
MperFT_ = 0.3048
TAUS = [0.0, 0.5, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1.01]
TAU_CTRL = [0.90, 0.99, 0.999]          # 做随机-分组对照的 τ
N_CTRL = 20                              # 随机分组对照重复数
LAM_GRID = 14                            # λ 路径长度（λ_max × logspace(-4, 0)）
LT_B_DEFAULT = 256


def jload(fp, default=None):
    if os.path.isfile(fp):
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def jdump(obj, fp):
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, fp)


def paths(net):
    return dict(cache=os.path.join(DATA, f"cluster_cache_{net}.npz"),
                out=os.path.join(DATA, f"cluster_{net}.json"))


def read_coords(inp):
    """[COORDINATES] → {node_id: (x, y)}（投影米制；已在 probe 中用管长核对过）。"""
    out, sec = {}, None
    with open(inp, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.split(";")[0].strip()
            if not s:
                continue
            if s.startswith("["):
                sec = s.strip("[]").upper()
                continue
            if sec == "COORDINATES":
                p = s.split()
                if len(p) >= 3:
                    out[p[0]] = (float(p[1]), float(p[2]))
    return out


def topo_pack(net, cand_idx, inp):
    """坐标 + 候选级拓扑（Voronoi 相邻、两两管距、子区管长所需的 owner/边）。"""
    co = read_coords(inp)
    ids = list(net.node_id)
    miss = [n for n in ids if n not in co]
    if miss:
        raise RuntimeError(f"{len(miss)} 个节点无坐标")
    xy = np.array([co[n] for n in ids], dtype=np.float64)
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    wm = np.maximum(np.asarray(net.len_ft, dtype=np.float64), 0.0) * MperFT_
    # 坐标单位自检：直线距离 / 记录管长（EPANET 里管道多为直线段）
    dd = np.hypot(xy[n1, 0] - xy[n2, 0], xy[n1, 1] - xy[n2, 1])
    ok = (wm > 1.0)
    ratio = float(np.median(dd[ok] / wm[ok])) if ok.any() else float("nan")
    t0 = time.perf_counter()
    ad = candidate_adjacency(net.N, n1, n2, wm, cand_idx)
    pd = candidate_pipe_distance(net.N, n1, n2, wm, cand_idx)
    return dict(xy_cand=xy[cand_idx], adjacency=ad["adjacency"], owner=ad["owner"],
                pipe_dist=pd, link_n1=n1, link_n2=n2, link_len_m=wm,
                coord_ratio=ratio, n_topo_edges=ad["n_edges"],
                topo_sec=time.perf_counter() - t0)


# ======================================================================
# build：City D
# ======================================================================
def build_city_d():
    import torch
    import demo_leak_inversion as dm
    from augment_suite import demo_truth
    from dgga.units import LPSperCFS, MperFT
    torch.set_default_dtype(torch.float64)
    t0 = time.perf_counter()
    pb = dm.Problem()
    junc = np.asarray(pb.s.junc_nodes)
    z = np.load(os.path.join(DATA, "placement_cache_coh_city_d.npz"))
    if not np.array_equal(np.asarray(z["junc"]), junc):
        raise RuntimeError("junction 顺序与相干缓存不一致")
    Dfull = np.asarray(z["Dfull"])                       # [25, Nj, NC]
    s0_pos = np.asarray(z["s0_pos"], dtype=np.int64)
    leak_pos = np.asarray(z["leak_pos"], dtype=np.int64)
    true_col = np.asarray(z["true_col"], dtype=np.int64)
    targets, sens_demo, label = demo_truth(pb, dm)
    # 真值重放（demo_leak_inversion.main 逐位同款）
    sol_base = pb.fsolve(max_iter=dm.OBS_MI)
    C_true, ke_true = {}, np.zeros(pb.net.N)
    for n, q in targets.items():
        i = pb.node_index[n]
        p_m = (sol_base["head"][:, i] - pb.net.elev_ft[i]).mean() * MperFT
        C_true[n] = q / p_m ** pb.gamma
        ke_true[i] = pb.ke_int_of_C(C_true[n])
    sol_true = pb.fsolve(ke=ke_true, max_iter=dm.OBS_MI)
    true_lk = {n: float(sol_true["emitter"][:, pb.node_index[n]].mean() * LPSperCFS)
               for n in targets}
    ref = jload(os.path.join(DATA, "demo_leak_inversion.json"))
    for n in targets:
        if abs(true_lk[n] - ref["config"]["true_nodes"][n]["true_mean_lps"]) > 1e-9:
            raise RuntimeError("真值漏损流量重放与 demo 不一致")
    dh = sol_true["head"] - sol_base["head"]                     # [25, N]
    noise = dm.NOISE_FT * np.random.default_rng(dm.SEED_NOISE).standard_normal((25, pb.net.N))
    cand_idx = np.asarray(pb.cidx, dtype=np.int64)
    tp = topo_pack(pb.net, cand_idx, os.path.join(ROOT, "networks", "realInpData", "city_d.inp"))
    orders = np.load(os.path.join(DATA, "placement_orders_coh_city_d.npz"))
    x_true = np.zeros(pb.nc)
    for n in targets:
        x_true[pb.cand.index(n)] = C_true[n]
    print(f"[city_d] Dfull{Dfull.shape} |S0|={s0_pos.size} 候选 {pb.nc}；"
          f"拓扑边 {tp['n_topo_edges']}（坐标/管长 中位 {tp['coord_ratio']:.4f}，"
          f"{tp['topo_sec']:.1f}s）；总 {time.perf_counter() - t0:.1f}s")
    return dict(net="city_d", Dfull=Dfull, junc=junc, s0_pos=s0_pos, leak_pos=leak_pos,
                true_col=true_col, cand_idx=cand_idx, dh=dh, noise=noise,
                sigma=float(dm.NOISE_FT), x_true=x_true, n_frames=25,
                true_lps=np.array([true_lk[n] for n in targets]),
                aug_order=np.asarray(orders["coh_fair"], dtype=np.int64),
                **{k: tp[k] for k in ("xy_cand", "adjacency", "owner", "pipe_dist",
                                      "link_n1", "link_n2", "link_len_m")},
                coord_ratio=tp["coord_ratio"], n_topo_edges=tp["n_topo_edges"])


# ======================================================================
# build：L-TOWN
# ======================================================================
def build_ltown(B=LT_B_DEFAULT):
    import augment_public as ap
    t0 = time.perf_counter()
    net, se = ap.lt_load()
    jn, cand, truth, sens = ap.lt_setup(se)
    qexp, ucf_e = ap.lt_units(net)
    d0, rh0 = ap.lt_nominal(net)
    z = np.load(os.path.join(DATA, "placement_cache_coh_ltown.npz"))
    if not np.array_equal(np.asarray(z["junc"]), jn):
        raise RuntimeError("junction 顺序与相干缓存不一致")
    Dfull = np.asarray(z["Dfull"])                                # [1, 782, 60]
    s0_pos = np.asarray(z["s0_pos"], dtype=np.int64)
    leak_pos = np.asarray(z["leak_pos"], dtype=np.int64)
    true_col = np.asarray(z["true_col"], dtype=np.int64)
    # 场景批：augment_public.stage_lt_leak 同一 seed / 同一构造，取前 B 个
    g = np.random.default_rng(ap.LT_SEED_BATCH)
    Dn_ = d0[None, :] * g.uniform(0.85, 1.15, (ap.LT_B, d0.size))
    Rn_ = rh0[None, :] + g.uniform(-1.0, 1.0, (ap.LT_B, rh0.size))
    noise_all = ap.NOISE_FT * np.random.default_rng(ap.SEED_NOISE).standard_normal(
        (ap.LT_B, net.N))
    ke_base = np.asarray(net.node_ke, dtype=np.float64)
    ke_true = ke_base.copy()
    ke_true[truth] = ucf_e / ap.LT_C_TRUE ** qexp
    dh = np.zeros((B, net.N))
    for b in range(B):
        a = se.run_gga(Dn_[b], Rn_[b], ke=ke_base, do_status=True,
                       hacc=ap.LT_HACC, max_iter=ap.LT_MI)
        c = se.run_gga(Dn_[b], Rn_[b], ke=ke_true, do_status=True,
                       hacc=ap.LT_HACC, max_iter=ap.LT_MI)
        dh[b] = c["head"] - a["head"]
        if b == 0 or b == B - 1:
            print(f"  场景 {b + 1}/{B}：relerr base {float(a['relerr']):.1e} "
                  f"leak {float(c['relerr']):.1e}（{time.perf_counter() - t0:.0f}s）")
            sys.stdout.flush()
    tp = topo_pack(net, np.asarray(cand, dtype=np.int64), ap.LT_INP)
    orders = np.load(os.path.join(DATA, "placement_orders_coh_ltown.npz"))
    x_true = np.zeros(cand.size)
    x_true[true_col] = ap.LT_C_TRUE
    print(f"[ltown] Dfull{Dfull.shape} |S0|={s0_pos.size} 候选 {cand.size} B={B}；"
          f"拓扑边 {tp['n_topo_edges']}（坐标/管长 中位 {tp['coord_ratio']:.4f}）；"
          f"总 {time.perf_counter() - t0:.1f}s")
    return dict(net="ltown", Dfull=Dfull, junc=jn, s0_pos=s0_pos, leak_pos=leak_pos,
                true_col=true_col, cand_idx=np.asarray(cand, dtype=np.int64),
                dh=dh, noise=noise_all[:B], sigma=float(ap.NOISE_FT), x_true=x_true,
                n_frames=B, true_lps=np.full(3, np.nan),
                aug_order=np.asarray(orders["coh_fair"], dtype=np.int64),
                **{k: tp[k] for k in ("xy_cand", "adjacency", "owner", "pipe_dist",
                                      "link_n1", "link_n2", "link_len_m")},
                coord_ratio=tp["coord_ratio"], n_topo_edges=tp["n_topo_edges"])


BUILDERS = dict(city_d=build_city_d, ltown=build_ltown)


def stage_build(net, B=LT_B_DEFAULT):
    info = BUILDERS[net]() if net != "ltown" else build_ltown(B)
    p = paths(net)
    np.savez_compressed(p["cache"], **{k: np.asarray(v) for k, v in info.items()
                                       if not isinstance(v, str)})
    print(f"写 {os.path.relpath(p['cache'], ROOT)}")


# ======================================================================
# run：聚类 + 组稀疏反演
# ======================================================================
def coherence_at(Dfull, sel):
    """传感集 sel 上的互相干 μ（零列记 0）与列范数。"""
    A = Dfull[:, sel, :].reshape(-1, Dfull.shape[2])
    n = np.linalg.norm(A, axis=0)
    mu = np.abs(A.T @ A) / np.maximum(n[:, None] * n[None, :], 1e-300)
    np.clip(mu, 0.0, 1.0, out=mu)
    return mu, n, A


def lam_path(A, y, labels, n=LAM_GRID):
    lmax = lambda_max_nn(A, y, labels)
    return lmax * np.logspace(-4, 0, n)[::-1], lmax


def noise_floor(sigma, n_obs):
    """Morozov 偏差原则的统计门限：χ²_n 的均值 + 2 个标准差 = σ²(n + 2√(2n))。"""
    return sigma ** 2 * (n_obs + 2.0 * np.sqrt(2.0 * n_obs))


def solve_path(A, y, labels, sigma, n_obs, lipschitz=None, n_lam=LAM_GRID,
               max_iter=3000, rss_offset=0.0):
    """λ 路径（暖启动）+ 真值无关的 λ 选择（偏差原则；达不到噪声地板则取最小 λ）。

    rss_offset：场景折叠留下的常数残差（见 stage_run）。它与 x 无关，但把 RSS 与噪声
    门限比较时必须加回去，否则偏差原则会挑到错误的 λ。
    """
    lams, lmax = lam_path(A, y, labels, n_lam)
    x = None
    recs = []
    for lam in lams:
        r = group_lasso_nn(A, y, labels, lam, x0=x, lipschitz=lipschitz,
                           max_iter=max_iter)
        x = r["x"]
        recs.append(dict(lam=float(lam), rss=r["rss"] + float(rss_offset), x=x.copy(),
                         active=r["active_groups"], n_iter=r["n_iter"]))
    # 偏差原则：σ²(n + 2√(2n))。字典是线性化近似，实测 RSS 常达不到该地板（模型
    # 误差不随 λ 消失），故门限取 max(理论地板, 路径最小 RSS + 2σ²√(2n)) - 后者是
    # "比可达最小 RSS 只超出一个 χ² 涨落尺度"的最大 λ，同样与真值无关。
    floor = noise_floor(sigma, n_obs)
    rmin = min(r["rss"] for r in recs)
    thr = max(floor, rmin + 2.0 * sigma ** 2 * np.sqrt(2.0 * n_obs))
    ok = [i for i, r in enumerate(recs) if r["rss"] <= thr]
    isel = ok[0] if ok else len(recs) - 1      # λ 由大到小：第一个进入门限者
    return recs, isel, float(lmax), float(thr)


def group_ranks(x, labels, A, y):
    """组排名：主键 ‖x_g‖₂；未激活组（‖x_g‖=0）之间按 KKT 筛选量
    ‖(A_gᵀ r)₊‖₂/w_g（r = y − Ax）降序 - 这正是 λ 继续下降时下一个进入支撑的次序，
    比"按下标"打破并列有意义（否则大量零组的名次是任意的）。"""
    labels = np.asarray(labels, dtype=np.int64)
    ng = int(labels.max()) + 1
    sc = np.sqrt(np.bincount(labels, weights=x * x, minlength=ng))
    c = np.maximum(A.T @ (y - A @ x), 0.0)
    w = np.sqrt(np.bincount(labels, minlength=ng))
    s2 = np.sqrt(np.bincount(labels, weights=c * c, minlength=ng)) / np.maximum(w, 1e-300)
    order = np.lexsort((-s2, -sc))
    rank = np.empty(ng, dtype=np.int64)
    rank[order] = np.arange(1, ng + 1)
    return rank, sc, s2, order


def rank_of_truth(x, labels, true_col, A, y):
    """每个真漏点所在簇（或节点）的排名（1 起）与得分。"""
    rank, sc, s2, _ = group_ranks(x, labels, A, y)
    out = []
    for t in np.asarray(true_col).tolist():
        c = int(labels[t])
        out.append(dict(cluster_rank=int(rank[c]), cluster_score=float(sc[c]),
                        screen_score=float(s2[c])))
    return out


def point_reference(A, y, sigma, n_obs, lipschitz, xy, NC, rss_offset=0.0):
    """单点定位对照（组 = 单点集合 ⇒ 非负 Lasso），同一 λ 规则；给出逐节点名次与
    "取前 K 名"的模糊半径（用于与簇级做**等模糊预算**对照）。"""
    labels = np.arange(NC, dtype=np.int64)
    recs, isel, lmax, floor = solve_path(A, y, labels, sigma, n_obs, lipschitz,
                                         rss_offset=rss_offset)
    x = recs[isel]["x"]
    rank, _sc, _s2, order = group_ranks(x, labels, A, y)
    diam_k = np.zeros(NC + 1)
    rad_k = np.zeros(NC + 1)
    for k in range(2, NC + 1):
        p = xy[order[:k]]
        diam_k[k] = float(np.hypot(p[:, None, 0] - p[None, :, 0],
                                   p[:, None, 1] - p[None, :, 1]).max())
        rad_k[k] = float(np.hypot(*(p - p.mean(0)).T).max())
    return dict(x=x, rank=rank, order=order, diam_k=diam_k, rad_k=rad_k,
                lam_sel=float(recs[isel]["lam"]), lam_max=lmax,
                rss=float(recs[isel]["rss"]), floor=floor,
                active=int(recs[isel]["active"]))


def eval_config(A, y, mu, labels, true_col, geo, sigma, n_obs, lipschitz, sig_mean,
                point=None, rss_offset=0.0):
    """一个 (传感配置, 噪声组, τ) 下的完整评价。"""
    recs, isel, lmax, floor = solve_path(A, y, labels, sigma, n_obs, lipschitz,
                                         rss_offset=rss_offset)
    r = recs[isel]
    sc, tot = group_scores(r["x"], labels)
    tr = rank_of_truth(r["x"], labels, true_col, A, y)
    ncl = int(labels.max()) + 1
    sizes = np.array([g["size"] for g in geo])
    diam = np.array([g["diameter"] for g in geo])
    rad = np.array([g["radius_centroid"] for g in geo])
    dlen = np.array([g.get("district_pipe_len", np.nan) for g in geo])
    per = []
    for k, (lab, t) in enumerate(zip(LABELS, np.asarray(true_col).tolist())):
        c = int(labels[t])
        per.append(dict(label=lab, cluster_size=int(sizes[c]),
                        cluster_diameter_m=float(diam[c]),
                        cluster_radius_m=float(rad[c]),
                        district_pipe_len_km=float(dlen[c] / 1000.0),
                        rank=tr[k]["cluster_rank"], score=tr[k]["cluster_score"],
                        hit_top1=bool(tr[k]["cluster_rank"] == 1),
                        hit_top3=bool(tr[k]["cluster_rank"] <= 3),
                        cluster_leak_sum=float(tot[c]),
                        max_rival_coh_outside=float(
                            max([mu[t, j] for j in range(mu.shape[0])
                                 if labels[j] != c] or [0.0])),
                        max_rival_coh_inside=float(
                            max([mu[t, j] for j in range(mu.shape[0])
                                 if labels[j] == c and j != t] or [0.0]))))
        if point is not None:      # 等模糊预算：簇级 top-1 覆盖 K 个候选 vs 单点 top-K
            K = int(sizes[c])
            per[-1].update(point_rank=int(point["rank"][t]),
                           point_hit_at_budget=bool(point["rank"][t] <= K),
                           point_diameter_at_budget_m=float(point["diam_k"][K]),
                           point_radius_at_budget_m=float(point["rad_k"][K]))
    # 命中曲线沿 λ 路径（诊断 λ 选择的敏感性）
    hits_path = []
    for rr in recs:
        tt = rank_of_truth(rr["x"], labels, true_col, A, y)
        hits_path.append(int(sum(1 for q in tt if q["cluster_rank"] <= 3)))
    ncp = detection_noncentrality(A, sig_mean, labels, sigma) if sigma > 0 else None
    out = dict(n_clusters=ncl, size_max=int(sizes.max()), size_mean=float(sizes.mean()),
               diameter_mean_m=float(diam.mean()), diameter_median_m=float(np.median(diam)),
               diameter_max_m=float(diam.max()),
               radius_mean_m=float(rad.mean()), radius_median_m=float(np.median(rad)),
               radius_max_m=float(rad.max()),
               district_km_mean=float(np.nanmean(dlen) / 1000.0),
               lam_sel=float(r["lam"]), lam_max=lmax, rss_sel=float(r["rss"]),
               rss_floor=floor, reached_floor=bool(r["rss"] <= floor),
               active_groups=int(r["active"]), per_leak=per,
               hit_top1=int(sum(p["hit_top1"] for p in per)),
               hit_top3=int(sum(p["hit_top3"] for p in per)),
               hit_rate_top1=float(np.mean([p["hit_top1"] for p in per])),
               hit_rate_top3=float(np.mean([p["hit_top3"] for p in per])),
               hits_top3_along_lambda=hits_path)
    if point is not None:
        out["point_hit_at_budget"] = int(sum(p["point_hit_at_budget"] for p in per))
        out["point_diameter_at_budget_mean_m"] = float(
            np.mean([p["point_diameter_at_budget_m"] for p in per]))
        out["point_radius_at_budget_mean_m"] = float(
            np.mean([p["point_radius_at_budget_m"] for p in per]))
    if ncp is not None:
        out["ncp_true_clusters"] = [dict(label=lab, rank=ncp[int(labels[t])]["rank"],
                                         ncp=ncp[int(labels[t])]["ncp"],
                                         z=ncp[int(labels[t])]["z"])
                                    for lab, t in zip(LABELS, np.asarray(true_col).tolist())]
    return out


def aggregation_probe(A, y, mu, pipe_dist, true_col, sigma, n_obs, lipschitz,
                      sig_mean, xy, kmax=12, rss_offset=0.0):
    """"簇级聚合能不能把某个漏点救回"的直接实验（针对 T1 的幅值受限判据）。

    对每个真漏点 t，强制把它与 k 个邻居并成一组（其余候选保持单点），k = 0..kmax：
      geo  邻居 = 管网最近的 k 个候选（地理聚合）
      coh  邻居 = 与 t 互相干最高的 k 个候选（相干聚合）
    报告每一步的
      ncp λ_g = ‖P_g s‖²/σ²（线性高斯模型下的精确非中心度）、rank r_g、
      z = λ_g/√(2r_g)（对零分布 χ²_{r_g} 标准差的标准分）、组稀疏反演里该组的名次、
      组的欧氏直径。
    结论判据：若 λ_g 随 k 基本不变而 r_g 上升（z 单调下降），则聚合**不可能**提升
    可检测性 - 它只减少了"分辨谁"的负担，不增加信号。
    """
    NC = A.shape[1]
    out = {}
    for lab, t in zip(LABELS, np.asarray(true_col).tolist()):
        rows = {}
        for mode in ("geo", "coh"):
            key = (np.argsort(np.where(np.arange(NC) == t, -np.inf, -mu[t]))
                   if mode == "coh" else
                   np.argsort(np.where(np.arange(NC) == t, np.inf, pipe_dist[t])))
            seq = []
            for k in range(0, kmax + 1):
                mem = np.r_[t, key[:k]].astype(np.int64)
                lb = np.arange(NC, dtype=np.int64)
                lb[mem] = -1
                uniq = np.unique(lb[lb >= 0])
                remap = {int(u): i for i, u in enumerate(uniq)}
                labels = np.array([remap[int(v)] if v >= 0 else len(uniq) for v in lb],
                                  dtype=np.int64)
                nc_ = detection_noncentrality(A, sig_mean, labels, sigma) if sigma > 0 else None
                r = solve_path(A, y, labels, sigma, n_obs, lipschitz,
                               rss_offset=rss_offset)
                recs, isel = r[0], r[1]
                rank = int(group_ranks(recs[isel]["x"], labels, A, y)[0][labels[t]])
                p = xy[mem]
                diam = float(np.hypot(p[:, None, 0] - p[None, :, 0],
                                      p[:, None, 1] - p[None, :, 1]).max()) if mem.size > 1 else 0.0
                e = dict(k=k, size=int(mem.size), rank=rank, diameter_m=diam)
                if nc_ is not None:
                    q = nc_[int(labels[t])]
                    e.update(ncp=q["ncp"], rank_Ag=q["rank"], z=q["z"])
                seq.append(e)
            rows[mode] = seq
        out[lab] = rows
    return out


def random_labels_like(labels, adjacency, rng):
    """尺寸匹配的随机分组对照：保持簇大小分布，把候选随机分配（打散拓扑与相干结构）。"""
    labels = np.asarray(labels)
    sizes = np.bincount(labels)
    perm = rng.permutation(labels.size)
    out = np.empty(labels.size, dtype=np.int64)
    a = 0
    for c, s in enumerate(sizes):
        out[perm[a:a + s]] = c
        a += s
    return out


def stage_run(net, quick=False):
    p = paths(net)
    z = np.load(p["cache"], allow_pickle=False)
    Dfull = z["Dfull"]
    s0_pos = z["s0_pos"].astype(np.int64)
    junc = z["junc"].astype(np.int64)
    cand_idx = z["cand_idx"].astype(np.int64)
    true_col = z["true_col"].astype(np.int64)
    leak_pos = z["leak_pos"].astype(np.int64)
    dh, noise, sigma = z["dh"], z["noise"], float(z["sigma"])
    adjacency, xy, pipe_dist = z["adjacency"], z["xy_cand"], z["pipe_dist"]
    owner, ln1, ln2, lm = z["owner"], z["link_n1"], z["link_n2"], z["link_len_m"]
    aug_order = z["aug_order"].astype(np.int64)
    x_true = z["x_true"]
    NC = Dfull.shape[2]
    taus = TAUS if not quick else [0.9, 0.99, 1.01]

    configs = [("S0", s0_pos)]
    if not quick:
        configs.append(("coh+40", np.r_[s0_pos, aug_order[:40]]))
    groups_noise = [("noisy", True)] + ([("noiseless", False)] if net == "ltown" else [])

    out = dict(config=dict(
        net=net, n_candidates=int(NC), n_frames=int(dh.shape[0]),
        n_sensors_s0=int(s0_pos.size), sigma_ft=sigma,
        n_topo_edges=int(z["n_topo_edges"]), coord_over_pipelen=float(z["coord_ratio"]),
        n_dict_frames=int(Dfull.shape[0]),
        dict_tiling=("字典逐帧、观测逐帧" if int(dh.shape[0]) == int(Dfull.shape[0]) else
                     f"字典为标称单帧、观测为 {int(dh.shape[0])} 个场景；按平铺问题的"
                     f"代数等价形式（√B 缩放 + 场景均值 + 常数 RSS 偏移）求解"),
        taus=taus, lam_grid=LAM_GRID, n_random_controls=N_CTRL,
        objective="min_{x>=0} 0.5||Ax-y||^2 + lam * sum_g sqrt(|g|) ||x_g||_2 (FISTA)",
        cluster_rule="受拓扑约束凝聚层次聚类（complete linkage，只合并 Voronoi 相邻簇，"
                     "阈值 tau：簇内任意两候选互相干 >= tau）",
        lam_rule="偏差原则：λ 由大到小，第一个使 RSS <= n·σ² 的 λ；达不到则取最小 λ",
        host=platform.node()), runs={})

    for cname, sel in configs:
        sel = np.unique(sel)
        mu, colnorm, A = coherence_at(Dfull, sel)
        sens_nodes = junc[sel]
        # City D 的字典逐帧算（Dfull 帧数 = 观测帧数）；L-TOWN 的字典在标称工况下算一次，
        # 观测却是 B 个需水/水库场景（与 augment_public.lt_invert 的 n_obs = B×|S| 同口径）。
        # 后者 = 沿行平铺字典 A_tiled = 1_B ⊗ A。恒等式
        #   ‖A_tiled x − y‖² = Σ_b‖Ax − y_b‖² = ‖(√B A)x − (√B ȳ)‖² + Σ_b‖y_b − ȳ‖²
        # 说明平铺问题与"√B 缩放字典 + 场景均值观测 + 常数 RSS 偏移"同解（组罚项不含 y）。
        # 于是用 |S| 行而非 B×|S| 行求解，结果不变（已数值复验：max|Δx|/‖x‖∞ 7.7e-15、
        # |ΔRSS|/RSS 3.6e-16、非中心度与 rank 逐位相同）。偏差原则用的 n 仍是 B×|S|；
        # 场景间散布落进常数偏移与 model_err_rel，如实报出。
        reps = int(dh.shape[0]) // int(Dfull.shape[0])
        if reps > 1 and int(dh.shape[0]) != reps * int(Dfull.shape[0]):
            raise RuntimeError("观测帧数不是字典帧数的整数倍")
        if reps > 1:
            A = np.sqrt(reps) * A
            colnorm = np.linalg.norm(A, axis=0)
        n_obs = int(dh.shape[0]) * int(sel.size)
        lipschitz = np.linalg.norm(A, 2) ** 2
        sig_mat = dh[:, sens_nodes]                          # [帧, |S|] 无噪真信号
        sig = sig_mat.reshape(-1)
        for gname, add_noise in groups_noise:
            y_mat = sig_mat + (noise[:, sens_nodes] if add_noise else 0.0)
            sg = sigma if add_noise else 0.0
            if reps > 1:
                ybar = y_mat.mean(0)
                y = np.sqrt(reps) * ybar
                rss_off = float(((y_mat - ybar[None, :]) ** 2).sum())
                sbar = sig_mat.mean(0)
                sig_eff = np.sqrt(reps) * sbar        # 供非中心度用（与平铺等价）
                model_num = float(np.sqrt(
                    np.sum((A @ x_true - sig_eff) ** 2)
                    + np.sum((sig_mat - sbar[None, :]) ** 2)))
            else:
                y = y_mat.reshape(-1)
                rss_off = 0.0
                sig_eff = sig
                model_num = float(np.linalg.norm(A @ x_true - sig))
            key = f"{cname}/{gname}"
            print(f"\n===== [{net}] {key}：|S|={sel.size} n={n_obs} "
                  f"max μ={mu[np.triu_indices(NC, 1)].max():.6f} =====")
            pt = point_reference(A, y, sg, n_obs, lipschitz, xy, NC,
                                 rss_offset=rss_off)
            rec = dict(n_sensors=int(sel.size), n_obs=int(n_obs),
                       coh_max=float(mu[np.triu_indices(NC, 1)].max()),
                       coh_median=float(np.median(mu[np.triu_indices(NC, 1)])),
                       n_pairs_gt_099=int((mu[np.triu_indices(NC, 1)] > 0.99).sum()),
                       signal_norm_true={lab: float(np.linalg.norm(x_true[t] * A[:, t]))
                                         for lab, t in zip(LABELS, true_col.tolist())},
                       col_norm_true={lab: float(colnorm[t])
                                      for lab, t in zip(LABELS, true_col.tolist())},
                       noise_norm=float(np.linalg.norm(noise[:, sens_nodes]))
                       if add_noise else 0.0,
                       signal_norm=float(np.linalg.norm(sig)),
                       model_err_rel=float(model_num
                                           / max(np.linalg.norm(sig), 1e-300)),
                       point=dict(rank={lab: int(pt["rank"][t]) for lab, t
                                        in zip(LABELS, true_col.tolist())},
                                  lam_sel=pt["lam_sel"], lam_max=pt["lam_max"],
                                  rss=pt["rss"], floor=pt["floor"],
                                  active=pt["active"]),
                       taus=[])
            for tau in taus:
                cl = constrained_agglomerative(mu, adjacency, tau, "complete")
                geo = cluster_geometry(cl["labels"], xy, pipe_dist, owner, lm, ln1, ln2)
                e = eval_config(A, y, mu, cl["labels"], true_col, geo,
                                sg, n_obs, lipschitz, sig_eff, pt, rss_offset=rss_off)
                e["tau"] = tau
                # 消融：无拓扑约束的纯相干聚类
                cl2 = constrained_agglomerative(mu, None, tau, "complete")
                geo2 = cluster_geometry(cl2["labels"], xy, pipe_dist, owner, lm, ln1, ln2)
                e2 = eval_config(A, y, mu, cl2["labels"], true_col, geo2,
                                 sg, n_obs, lipschitz, sig_eff, rss_offset=rss_off)
                e["no_topology"] = dict(n_clusters=e2["n_clusters"],
                                        diameter_mean_m=e2["diameter_mean_m"],
                                        radius_mean_m=e2["radius_mean_m"],
                                        hit_top1=e2["hit_top1"], hit_top3=e2["hit_top3"])
                # 消融：纯几何聚类（忽略相干，簇数对齐）
                cl3 = constrained_agglomerative(-pipe_dist, adjacency, -np.inf,
                                                "complete", min_clusters=e["n_clusters"])
                geo3 = cluster_geometry(cl3["labels"], xy, pipe_dist, owner, lm, ln1, ln2)
                e3 = eval_config(A, y, mu, cl3["labels"], true_col, geo3,
                                 sg, n_obs, lipschitz, sig_eff, rss_offset=rss_off)
                e["geometry_only"] = dict(n_clusters=e3["n_clusters"],
                                          diameter_mean_m=e3["diameter_mean_m"],
                                          radius_mean_m=e3["radius_mean_m"],
                                          hit_top1=e3["hit_top1"], hit_top3=e3["hit_top3"])
                # 对照：尺寸匹配的随机分组
                if tau in TAU_CTRL and e["n_clusters"] < NC:
                    rng = np.random.default_rng(909 + int(tau * 1e4))
                    h1, h3, dm_, rd_ = [], [], [], []
                    for _ in range(N_CTRL):
                        lb = random_labels_like(cl["labels"], adjacency, rng)
                        g4 = cluster_geometry(lb, xy, pipe_dist, owner, lm, ln1, ln2)
                        e4 = eval_config(A, y, mu, lb, true_col, g4, sg, n_obs,
                                         lipschitz, sig_eff, rss_offset=rss_off)
                        h1.append(e4["hit_top1"])
                        h3.append(e4["hit_top3"])
                        dm_.append(e4["diameter_mean_m"])
                        rd_.append(e4["radius_mean_m"])
                    e["random_group_control"] = dict(
                        n=N_CTRL, hit_top1_mean=float(np.mean(h1)),
                        hit_top3_mean=float(np.mean(h3)),
                        hit_top3_ge_design=int(sum(1 for v in h3 if v >= e["hit_top3"])),
                        p_one_sided=float((1 + sum(1 for v in h3 if v >= e["hit_top3"]))
                                          / (1 + N_CTRL)),
                        diameter_mean_m=float(np.mean(dm_)),
                        radius_mean_m=float(np.mean(rd_)))
                rec["taus"].append(e)
                print(f"  τ={tau:<7g} 簇 {e['n_clusters']:3d}  平均直径 "
                      f"{e['diameter_mean_m']:8.1f} m  top1 {e['hit_top1']}/3  "
                      f"top3 {e['hit_top3']}/3  等预算单点 "
                      f"{e['point_hit_at_budget']}/3  λ*={e['lam_sel']:.3e} "
                      f"活跃组 {e['active_groups']}  (无拓扑 {e2['hit_top3']}/3，"
                      f"纯几何 {e3['hit_top3']}/3)")
                sys.stdout.flush()
            if not quick:
                t0 = time.perf_counter()
                rec["aggregation_probe"] = aggregation_probe(
                    A, y, mu, pipe_dist, true_col, sg, n_obs, lipschitz, sig_eff, xy,
                    rss_offset=rss_off)
                print(f"  聚合救援探针 {time.perf_counter() - t0:.1f}s：" + "；".join(
                    f"{lab} k=0 名次 {v['coh'][0]['rank']} → k=6 名次 {v['coh'][6]['rank']}"
                    + (f"（λ_g {v['coh'][0].get('ncp', float('nan')):.2f}→"
                       f"{v['coh'][6].get('ncp', float('nan')):.2f}，z "
                       f"{v['coh'][0].get('z', float('nan')):.2f}→"
                       f"{v['coh'][6].get('z', float('nan')):.2f}）" if sg > 0 else "")
                    for lab, v in rec["aggregation_probe"].items()))
                sys.stdout.flush()
            out["runs"][key] = rec
    jdump(out, p["out"])
    print(f"\n写 {os.path.relpath(p['out'], ROOT)}")


# ======================================================================
# report
# ======================================================================
def complexity_probe(net="city_d"):
    """实测各步耗时（供"复杂度"一节配数）：拓扑 / 聚类 / FISTA。缓存已在则不重算前向。"""
    p = paths(net)
    if not os.path.isfile(p["cache"]):
        return None
    z = np.load(p["cache"], allow_pickle=False)
    Dfull, s0 = z["Dfull"], z["s0_pos"].astype(np.int64)
    cand_idx = z["cand_idx"].astype(np.int64)
    ln1, ln2, lm = z["link_n1"], z["link_n2"], z["link_len_m"]
    adjacency, mu_src = z["adjacency"], None
    N = int(max(int(ln1.max()), int(ln2.max())) + 1)
    NC = int(Dfull.shape[2])
    t = {}
    t0 = time.perf_counter(); candidate_adjacency(N, ln1, ln2, lm, cand_idx)
    t["voronoi_adjacency_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); candidate_pipe_distance(N, ln1, ln2, lm, cand_idx)
    t["pairwise_pipe_dist_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); mu, _cn, A = coherence_at(Dfull, s0)
    t["coherence_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cl = constrained_agglomerative(mu, adjacency, 0.99, "complete")
    t["clustering_tau0.99_s"] = time.perf_counter() - t0
    y = z["dh"][:, z["junc"].astype(np.int64)[s0]].reshape(-1) +         z["noise"][:, z["junc"].astype(np.int64)[s0]].reshape(-1)
    L2 = np.linalg.norm(A, 2) ** 2
    t0 = time.perf_counter()
    r1 = group_lasso_nn(A, y, cl["labels"], 1e-2 * lambda_max_nn(A, y, cl["labels"]),
                        lipschitz=L2, max_iter=3000)
    t["fista_single_lambda_s"] = time.perf_counter() - t0
    t["fista_single_lambda_iters"] = int(r1["n_iter"])
    t0 = time.perf_counter()
    solve_path(A, y, cl["labels"], float(z["sigma"]), A.shape[0], L2)
    t["fista_full_path_s"] = time.perf_counter() - t0
    t.update(N_nodes=N, L_links=int(ln1.size), NC=NC, n_obs=int(A.shape[0]),
             n_clusters_tau099=int(cl["n_clusters"]), lam_grid=LAM_GRID, net=net,
             host=platform.node())
    return t


def stage_report():
    from augment_suite import assert_no_ids
    res = {n: jload(paths(n)["out"]) for n in ("city_d", "ltown")}
    res = {k: v for k, v in res.items() if v}
    if not res:
        raise RuntimeError("先跑 --stage run")
    lines = []
    W = lines.append
    W("=" * 78)
    W("模块一：簇级（子区）漏损定位 - 相干 × 拓扑受约束聚类 + 组稀疏反演")
    W("=" * 78)
    W(f"机器 {platform.node()}；零节点/链路/传感器编号")
    W("  T1 / T2 / T3 = 本文给三个真漏点的匿名别名（与任何网中同名元件无关）；")
    W("  簇、传感器、候选一律不带编号，只报数量、半径（m）、直径（m）与子区管长（km）。")
    W("")
    W("算法（dgga/cluster.py，只新增函数）")
    W("  变量  x_j ≥ 0 = 候选 j 的漏损系数（用户制 C，q = C·p^γ）；A(S) = 签名字典，")
    W("        n = 帧数×|S| 行；观测 y = h(真值) − h(基线) + N(0,σ²)。")
    W("  相干  μ_ij = |a_iᵀa_j| / (‖a_i‖‖a_j‖)。")
    W("  拓扑  管网 Voronoi（多源 Dijkstra，边权 = 管长 m）→ 候选级相邻矩阵（无阈值参数）。")
    W("  聚类  受约束凝聚层次（complete linkage）：只合并拓扑相邻簇，")
    W("        s(g,h) = min 交叉相干，合并到 max s < τ ⇔ 簇内任意两候选 μ ≥ τ。")
    W("  反演  min_{x≥0} ½‖Ax−y‖² + λ Σ_g √|g|·‖x_g‖₂（FISTA；非负组近端")
    W("        prox(v)_g = (1 − λw_g/‖v_g⁺‖)₊·v_g⁺）。组 = 单点 ⇒ 非负 Lasso = 单点定位对照。")
    W("  选 λ  真值无关：λ 由大到小，第一个使 RSS ≤ max(σ²(n+2√(2n)), min RSS + 2σ²√(2n)) 者。")
    W("        （字典是线性化近似，实测 RSS 常达不到纯噪声地板，故用后一项兜底。）")
    W("  步骤 ① 前向算签名字典 A（冻结状态灵敏度）→ ② 管网 Voronoi 得候选相邻 →")
    W("        ③ 受约束层次聚类得 labels(τ) → ④ 沿 λ 由大到小暖启动跑 FISTA 路径 →")
    W("        ⑤ 偏差原则选 λ* → ⑥ 按 ‖x_g‖₂ 给簇排名，报命中与该簇半径/直径/子区管长。")
    W("  复杂度 Voronoi O(L log N)；候选两两管距 O(NC·L log N)；聚类 O(NC³)（堆可降 O(NC² log NC)）；")
    W("        FISTA 每步两次矩阵-向量乘 O(n·NC)，n = 帧数×|S|；λ 路径 = LAM_GRID × 迭代数。")
    cp = complexity_probe("city_d")
    if cp:
        W(f"  实测（{cp['host']}，单核 numpy；N={cp['N_nodes']} 节点、L={cp['L_links']} 管、"
          f"NC={cp['NC']} 候选、n={cp['n_obs']} 观测）：")
        W(f"        Voronoi+相邻 {cp['voronoi_adjacency_s']:.2f}s；候选两两管距 "
          f"{cp['pairwise_pipe_dist_s']:.2f}s；相干矩阵 {cp['coherence_s']:.3f}s；")
        W(f"        聚类(τ=0.99→{cp['n_clusters_tau099']} 簇) {cp['clustering_tau0.99_s']:.3f}s；"
          f"单 λ FISTA {cp['fista_single_lambda_s']:.2f}s/{cp['fista_single_lambda_iters']} 步；"
          f"整条 λ 路径({cp['lam_grid']} 点，暖启动) {cp['fista_full_path_s']:.2f}s。")
    W("")
    for net, d in res.items():
        c = d["config"]
        W("-" * 78)
        W(f"[{net}] 候选 {c['n_candidates']}，帧/场景 {c['n_frames']}，|S0| = {c['n_sensors_s0']}，"
          f"σ = {c['sigma_ft']} ft，候选级拓扑边 {c['n_topo_edges']}")
        W(f"        坐标单位自检：直线距离 / 管长（已换算成米）中位 "
          f"{c['coord_over_pipelen']:.4f} ⇒ 坐标为投影米制，模糊半径单位是米")
        for key, r in d["runs"].items():
            W("")
            W(f"  == {key} ==  |S| = {r['n_sensors']}，n = {r['n_obs']}，"
              f"max μ = {r['coh_max']:.6f}，中位 μ = {r['coh_median']:.4f}，"
              f">0.99 的候选对 {r['n_pairs_gt_099']}")
            W(f"     真漏点信号 ‖C·a‖ (ft)：" +
              "，".join(f"{k} {v:.4f}" for k, v in r["signal_norm_true"].items()) +
              f"；噪声 ‖n‖ = {r['noise_norm']:.3f}；线性字典模型误差 "
              f"{100 * r['model_err_rel']:.1f}%（相对 ‖信号‖ = {r['signal_norm']:.3f}）")
            W(f"     单点定位对照（非负 Lasso，组 = 单点）真漏点名次 / {c['n_candidates']}："
              + "，".join(f"{k} 第 {v}" for k, v in r["point"]["rank"].items()))
            W("     τ        簇数  平均半径m  平均直径m  中位直径m  子区管长km  簇top1 "
              "簇top3  等预算单点  无拓扑top3 纯几何top3")
            for e in r["taus"]:
                W(f"     {e['tau']:<8g} {e['n_clusters']:4d} "
                  f"{e.get('radius_mean_m', float('nan')):10.1f} "
                  f"{e['diameter_mean_m']:10.1f} "
                  f"{e['diameter_median_m']:10.1f} {e['district_km_mean']:11.3f} "
                  f"{e['hit_top1']:6d} {e['hit_top3']:6d} "
                  f"{e.get('point_hit_at_budget', -1):10d} "
                  f"{e['no_topology']['hit_top3']:10d} {e['geometry_only']['hit_top3']:9d}")
            W("")
            W("     ** 权衡曲线（本模块的核心交付）：簇级命中率 vs 平均簇半径（米） **")
            W("       平均半径m  平均直径m  簇数  命中率top1  命中率top3  纯几何top3  "
              "等预算单点命中率  单点等预算半径m")
            for e in sorted(r["taus"], key=lambda q: q.get("radius_mean_m", 0.0)):
                W(f"       {e.get('radius_mean_m', float('nan')):9.1f} "
                  f"{e['diameter_mean_m']:10.1f} {e['n_clusters']:5d} "
                  f"{e.get('hit_rate_top1', float('nan')):11.3f} "
                  f"{e.get('hit_rate_top3', float('nan')):11.3f} "
                  f"{e['geometry_only']['hit_top3'] / 3.0:11.3f} "
                  f"{e.get('point_hit_at_budget', 0) / 3.0:17.3f} "
                  f"{e.get('point_radius_at_budget_mean_m', float('nan')):16.1f}")
            W("       （半径 = 簇内候选到簇质心的最大距离；簇越大越易命中但巡检范围越无用，"
              "只有同时看这两列才不是自欺。）")
            W("")
            W("     λ 选择敏感性（λ* 由偏差原则定，与真值无关；这里列出整条 λ 路径上"
              "top-3 命中的最大值，看 λ* 是否被挑过）：")
            W("       τ        λ* 处 top3   路径最优 top3   路径上取到最优的 λ 点数 / 总点数")
            for e in r["taus"]:
                hp = e.get("hits_top3_along_lambda") or []
                if not hp:
                    continue
                best = max(hp)
                W(f"       {e['tau']:<8g} {e['hit_top3']:9d} {best:14d} "
                  f"{sum(1 for v in hp if v == best):20d} / {len(hp)}")
            W("       （λ* 处的命中不高于路径最优是正常的；若两者相差很大，说明结论对 λ 敏感，"
              "本模块不据此挑 λ。）")
            W("")
            W("     消融（同 τ、同 λ 规则）：命中更多若靠更大的模糊范围换来，就不算赢")
            W("       τ        设计top3/半径m      无拓扑纯相干top3/半径m   纯几何top3/半径m")
            for e in r["taus"]:
                nt, go = e["no_topology"], e["geometry_only"]
                W(f"       {e['tau']:<8g} {e['hit_top3']}/3 @ "
                  f"{e.get('radius_mean_m', float('nan')):8.1f}      "
                  f"{nt['hit_top3']}/3 @ {nt.get('radius_mean_m', float('nan')):8.1f}"
                  f" ({nt['n_clusters']} 簇)      "
                  f"{go['hit_top3']}/3 @ {go.get('radius_mean_m', float('nan')):8.1f}"
                  f" ({go['n_clusters']} 簇)")
            W("     （等预算单点 = 真漏点是否落在单点排名的前 K 名内，K = 它在该 τ 下所属簇的"
              "候选数；这是与簇级 top-1 完全同等模糊度的对照）")
            W("     逐漏点（τ 扫描：簇内候选数 / R=簇半径 m / D=簇直径 m / #簇名次）：")
            for i, lab in enumerate(LABELS):
                W(f"       {lab}: " + "  ".join(
                    f"τ={e['tau']:g}:{e['per_leak'][i]['cluster_size']}/"
                    f"R{e['per_leak'][i].get('cluster_radius_m', float('nan')):.0f}"
                    f"/D{e['per_leak'][i]['cluster_diameter_m']:.0f}m/#"
                    f"{e['per_leak'][i]['rank']}" for e in r["taus"]))
            for e in r["taus"]:
                if "random_group_control" in e:
                    q = e["random_group_control"]
                    W(f"     随机分组对照（尺寸分布相同、打散拓扑与相干）τ={e['tau']:g}："
                      f"设计 top3 {e['hit_top3']}/3 @半径 "
                      f"{e.get('radius_mean_m', float('nan')):.0f} m，随机均值 "
                      f"{q['hit_top3_mean']:.2f}/3 @半径 "
                      f"{q.get('radius_mean_m', float('nan')):.0f} m"
                      f"（{q['n']} 次，≥设计 {q['hit_top3_ge_design']} 次，"
                      f"交换性单侧 p = {q['p_one_sided']:.3f}）")
            if r["taus"] and "ncp_true_clusters" in r["taus"][0]:
                W("     可检测性（线性高斯**精确**非中心度 λ_g = ‖P_g s‖²/σ²，"
                  "零分布 χ²_{r_g}，z = λ_g/√(2r_g)）：")
                sg_ = c["sigma_ft"]
                if sg_ > 0:
                    own = {k: (v / sg_) ** 2 for k, v in r["signal_norm_true"].items()}
                    W("       先记住 λ_g 的口径：s 是**三个漏点合起来**的信号，P_g 是该组列空间的"
                      "投影。各漏点**自己**的信噪 ‖C·a‖²/σ² = "
                      + "，".join(f"{k} {v:.2f}" for k, v in own.items()) + "。")
                    W("       高相干下任何一列都能吸走别人的大部分信号，所以 λ_g ≫ 自己的信噪"
                      "只说明「网里有漏」，不说明「是它」 - 受限的是分辨，不是检测。")
                for e in r["taus"]:
                    if e["tau"] in (0.0, 0.9, 0.99, 1.01):
                        W(f"       τ={e['tau']:g}: " + "  ".join(
                            f"{q['label']} r_g={q['rank']} λ_g={q['ncp']:.2f} z={q['z']:.2f}"
                            for q in e["ncp_true_clusters"]))
            if "aggregation_probe" in r:
                W("     聚合救援探针（强制把真漏点与 k 个最相干邻居并为一组，其余单点）：")
                W("       标签  k=0            k=2            k=4            k=8            k=12")
                for lab, v in r["aggregation_probe"].items():
                    cells = []
                    for k in (0, 2, 4, 8, 12):
                        q = v["coh"][k]
                        cells.append(f"#{q['rank']}/λ{q.get('ncp', float('nan')):.1f}"
                                     f"/z{q.get('z', float('nan')):.2f}" if "ncp" in q
                                     else f"#{q['rank']}")
                    W(f"       {lab}   " + "  ".join(f"{cc:<14s}" for cc in cells))
                W("       （#名次 / λ_g 非中心度 / z；λ_g 随 k 基本不变而 r_g 上升 ⇒ "
                  "聚合不增加信号，只减轻分辨谁的负担）")
        W("")
    txt = "\n".join(lines)
    with open(os.path.join(DATA, "cluster_localisation_wip.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    summary = dict(config={n: d["config"] for n, d in res.items()},
                   runs={n: d["runs"] for n, d in res.items()})
    for n in res:
        # T1–T3 是本文给三个真漏点的匿名别名。L-TOWN 里恰好存在一个同名元件（公开网，
        # 非脱敏对象），别名与它无关；把别名从 id 集合里剔除后再查，其余编号照查不误。
        ids = net_id_set(n)
        clash = sorted(set(LABELS) & ids)
        if clash:
            print(f"[提示] {n} 网中存在与漏点别名同名的元件 {len(clash)} 个："
                  f"别名按别名论，不计入编号泄漏")
        assert_no_ids(summary["runs"][n], ids - set(LABELS), set(LABELS),
                      f"cluster_localisation.json[{n}]")
    jdump(summary, os.path.join(DATA, "cluster_localisation.json"))
    make_figure(res)
    print(txt)
    print("\n写 data/cluster_localisation_wip.txt / data/cluster_localisation.json / "
          "data/fig_cluster_localisation.png")


def net_id_set(net):
    from dgga.parse import Net, parse_inp
    if net == "city_d":
        n = Net.load(os.path.join(DATA, "reference"), "city_d")
    else:
        import augment_public as ap
        n = parse_inp(ap.LT_INP)
    return set(str(x) for x in n.node_id) | set(str(x) for x in n.link_id)


def make_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    nets = list(res)
    fig, ax = plt.subplots(2, len(nets), figsize=(5.4 * len(nets), 8.0), squeeze=False)
    for c, net in enumerate(nets):
        d = res[net]
        key = next(k for k in d["runs"] if k.startswith("S0"))
        r = d["runs"][key]
        # (上) 权衡曲线：top-3 命中 vs 平均簇直径
        a = ax[0][c]
        ts = sorted(r["taus"], key=lambda q: q.get("radius_mean_m", 0.0))
        rm = [e.get("radius_mean_m", np.nan) for e in ts]
        a.plot(rm, [e["hit_top3"] / 3.0 for e in ts], "o-",
               label="簇级 top-3（相干×拓扑）")
        a.plot(rm, [e["hit_top1"] / 3.0 for e in ts], "o-", lw=1, alpha=.6,
               label="簇级 top-1")
        a.plot(rm, [e["geometry_only"]["hit_top3"] / 3.0 for e in ts], "s--",
               label="纯几何聚类（簇数对齐）")
        a.plot(rm, [e.get("point_hit_at_budget", np.nan) / 3.0 for e in ts], "^:",
               label="等预算单点定位")
        xc = [e.get("radius_mean_m", np.nan) for e in ts if "random_group_control" in e]
        yc = [e["random_group_control"]["hit_top3_mean"] / 3.0 for e in ts
              if "random_group_control" in e]
        if xc:
            a.plot(xc, yc, "x", ms=9, color="0.4", label="随机分组对照（均值）")
        a.set_xscale("symlog", linthresh=10)
        a.set_xlabel("平均簇半径（模糊半径，m）")
        a.set_ylabel("簇级命中率（3 个漏点）")
        a.set_ylim(-0.05, 1.12)
        a.set_title(f"{net}：命中率 vs 平均簇半径 的权衡（{key}）")
        a.grid(alpha=.3)
        a.legend(fontsize=8)
        # (下) 聚合救援：z = λ_g/√(2 r_g) vs 组内候选数
        b = ax[1][c]
        pr = r.get("aggregation_probe")
        if pr and all("z" in pr[lab]["coh"][0] for lab in pr):
            sg = d["config"]["sigma_ft"]
            lo = hi = None
            for lab in pr:
                ks = [q["size"] for q in pr[lab]["coh"]]
                zs = [q["z"] for q in pr[lab]["coh"]]
                ln, = b.plot(ks, zs, "o-", label=f"{lab}")
                vals = [v for v in zs if v > 0]
                own = r["signal_norm_true"].get(lab)
                if own and sg > 0:      # 该漏点**自己**的信号能撑起的可检测性
                    o = (own / sg) ** 2 / np.sqrt(2.0)
                    b.axhline(o, color=ln.get_color(), ls=":", lw=1.2)
                    vals.append(o)
                if vals:
                    lo = min(vals) if lo is None else min(lo, min(vals))
                    hi = max(vals) if hi is None else max(hi, max(vals))
            b.set_yscale("log")         # 全为正数，用纯对数轴（symlog 会被水平线拖出负区）
            if lo and hi:
                b.set_ylim(lo / 3.0, hi * 3.0)
            b.set_xlabel("组内候选数（聚合规模）")
            b.set_ylabel("z = λ_g / √(2 r_g)")
            b.set_title(f"{net}：聚合能否提升可检测性\n"
                        f"（实线 = 组的 z；虚线 = 该漏点自己信号的 z）")
            b.grid(alpha=.3)
            b.legend(fontsize=8)
        else:
            b.text(.5, .5, "无噪声组：非中心度不适用", ha="center", va="center")
            b.set_axis_off()
    for a in ax.ravel():
        for t in ([a.title, a.xaxis.label, a.yaxis.label] + a.get_xticklabels()
                  + a.get_yticklabels()):
            t.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, "fig_cluster_localisation.png"), dpi=160)
    plt.close(fig)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True,
                     choices=["build", "run", "report"])
    ap_.add_argument("--net", default="city_d", choices=["city_d", "ltown"])
    ap_.add_argument("--B", type=int, default=LT_B_DEFAULT)
    ap_.add_argument("--quick", action="store_true")
    a = ap_.parse_args()
    if a.stage == "build":
        stage_build(a.net, a.B)
    elif a.stage == "run":
        stage_run(a.net, a.quick)
    else:
        stage_report()


if __name__ == "__main__":
    main()

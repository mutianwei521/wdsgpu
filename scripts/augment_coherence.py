# -*- coding: utf-8 -*-
"""augment_coherence.py - 相干驱动的传感器增设（S0 固定 + k）：真正为漏损搜索选点。

上一轮（data/audit_augment_wip.txt、data/augment_public_wip.txt）的结论：为粗糙度
可辨识性设计的增设（D-最优 / 覆盖）把传感不足管全部找回，但**不找回漏点** -
决定漏损搜索成败的是漏损签名字典的互相干，而那两套目标根本不看它。本脚本把
dgga.placement.coherence_augment 接到两条漏损链上，如实报告结果是什么。

目标函数（真值无关）：字典 D[t, i, j] = 候选漏点 j 单位漏损系数在传感位置 i、帧 t 的
压力响应；传感集 S 下列归一化互相干 μ_ij(S)。缺省目标
    J(S) = Σ_{i<j} −log(1 − μ_ij(S)² + 1e-12)      （"logdet2"，成对对数体积）
S0 固定，在候选位置池上贪心加 k 个使 J 最小；候选池、S0、字典是它的全部输入，
漏点身份不进入任何一步（详见 dgga/placement.py 第六节）。另跑 "max" 目标
（字典序 (max μ, J)）作变体对照。

公平池（主结果）= junction − S0 − 三个漏点节点：把"传感器恰好装在漏点本节点"这个
trivial 案排除在设计之外（上一轮 City D 六套 1/3 全是它）。原池（junction − S0，
含漏点节点）另报。L-TOWN 上一轮的池已经是公平池。

阶段（--stage）：
  select  --net hanoi|ltown|city_d
          构造字典（City D：demo_leak_inversion.build_dictionary 全 junction 行，25 帧；
          L-TOWN：augment_public.lt_dictionary 名义帧，同 ltown_coherence 构造；
          Hanoi：冒烟，单帧 emitter 扰动，31 junction 既是候选漏点也是候选传感位置），
          S0 = 上一轮同一 S0，贪心到 k=80（Hanoi 20），另算上一轮 D-opt / cover 序列
          与随机 +k（≥5 seeds）在同一相干口径下的曲线。
          → data/placement_orders_coh_<net>.npz（下标）、data/placement_coh_<net>.json
            （可读，零编号：每步 max / 中位 / 分位 / >0.999 对数 / 目标值 / 所选点数）
  leak    --net ltown   论文 §3.5 L-TOWN 反演原样重跑（augment_public.stage_lt_leak，
                        传感器 = S0 ∪ S_k），配置 = S0 / coh+k / cohmax+20 / rand+20×5
                        → data/ltown_coh_leak.json（--out 可改；多卡分跑后 merge）
          --net city_d  工单案例原样重跑（demo_leak_inversion 反演器不动，配方同
                        audit_augment/hv_leak_control.py），每个 --specs 一个 json
                        → data/leak_coh_city_d/<spec>.json（只含 T1..T3 标签、跳数、kind）
  merge   把多卡分跑的 L-TOWN json 合并成 data/ltown_coh_leak.json
  report  汇总 → data/augment_coherence.json（零编号）+ data/augment_coherence_wip.txt

运行：& python -X utf8 scripts/augment_coherence.py --stage select --net hanoi
      & python -X utf8 scripts/augment_coherence.py --stage select --net ltown
      & python -X utf8 scripts/augment_coherence.py --stage select --net city_d
      服务器（scripts/augment_coherence_v100.sh）：
      python -X utf8 scripts/augment_coherence.py --stage leak --net ltown --linear cudss --chunk 256 --only S0,coh+5
      python -X utf8 scripts/augment_coherence.py --stage leak --net city_d --specs coh+20
      & python -X utf8 scripts/augment_coherence.py --stage report
全程"虚拟增设"：在模型上模拟新传感器，是设计与模拟验证，不是现场实装。
"""

import argparse
import glob
import json
import os
import platform
import re
import sys
import time
from collections import deque

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.placement import coherence_augment, coherence_stats_rows, ORTH_TOL   # noqa: E402

DATA = os.path.join(ROOT, "data")
KS = {"hanoi": [5, 10, 20], "ltown": [5, 10, 20, 40, 80], "city_d": [5, 10, 20, 40, 80]}
RAND_SEEDS = [0, 1, 2, 3, 4]
RAND_K = 20
C_PROBE = 0.3
OBJECTIVE_DEF = ("J(S) = sum_{i<j} -log(1 - mu_ij(S)^2 + 1e-12)，mu_ij = 列归一化签名字典在 S0∪S 行上"
                 "的互相干；S0 固定，贪心加点使 J 最小（dgga.placement.coherence_augment，objective="
                 "'logdet2'）。输入只有字典、S0 与候选位置池：真值无关")
LABELS = ["T1", "T2", "T3"]


def paths(net):
    d = dict(orders=os.path.join(DATA, f"placement_orders_coh_{net}.npz"),
             readable=os.path.join(DATA, f"placement_coh_{net}.json"),
             cache=os.path.join(DATA, f"placement_cache_coh_{net}.npz"),
             wip=os.path.join(DATA, "augment_coherence_wip.txt"),
             summary=os.path.join(DATA, "augment_coherence.json"))
    if net == "ltown":
        d["leak"] = os.path.join(DATA, "ltown_coh_leak.json")
    if net == "city_d":
        d["leak_dir"] = os.path.join(DATA, "leak_coh_city_d")
    return d


def jload(fp, default=None):
    if os.path.isfile(fp):
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    return {} if default is None else default


def jdump(obj, fp):
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, fp)


def hop_distances(N, n1, n2, src):
    adj = [[] for _ in range(N)]
    for a, b in zip(n1.tolist(), n2.tolist()):
        adj[a].append(b)
        adj[b].append(a)
    dist = np.full(N, -1, dtype=np.int64)
    dist[src] = 0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


# ======================================================================
# 字典构造（三个网）：返回 dict(Dfull[T, m, NC], junc, s0_pos, leak_pos, true_col, ...)
# ======================================================================
def build_city_d():
    import torch
    import demo_leak_inversion as dm
    from augment_suite import demo_truth
    torch.set_default_dtype(torch.float64)
    t0 = time.perf_counter()
    pb = dm.Problem()
    junc = np.asarray(pb.s.junc_nodes)
    targets, sens_demo, label = demo_truth(pb, dm)
    sol_base = pb.fsolve(max_iter=dm.OBS_MI)
    D, _Dn, _coh = dm.build_dictionary(pb, sol_base, junc)            # [25*Nj, nc]
    Dfull = D.reshape(25, junc.size, pb.nc)
    o = np.load(os.path.join(DATA, "placement_orders_city_d.npz"))
    s0_pos = np.unique(np.asarray(o["augment_fixed"], dtype=np.int64))
    s0_nodes = np.sort(junc[s0_pos])
    if not np.array_equal(s0_nodes, np.sort(np.random.default_rng(dm.SEED).choice(junc, 40, replace=False))):
        raise RuntimeError("S0 不是 seed 2026 的抽样")
    pos_of = {int(n): i for i, n in enumerate(junc)}
    leak_pos = np.array([pos_of[pb.node_index[n]] for n in targets], dtype=np.int64)
    true_col = [pb.cand.index(n) for n in targets]
    # 自检：S0 行上的相干与 demo 同函数（按传感器子集重建字典）逐位一致
    _D0, _Dn0, coh0 = dm.build_dictionary(pb, sol_base, junc[s0_pos])
    st = coherence_stats_rows(Dfull, s0_pos)
    iu, ju = np.triu_indices(pb.nc, 1)
    chk = float(abs(np.abs(coh0)[iu, ju].max() - st["coh_max"]))
    prev = {obj: np.asarray(o["augment_" + obj], dtype=np.int64) for obj in ("dopt", "cover")}
    print(f"[city_d] 字典 D[T=25, m={junc.size}, NC={pb.nc}]，|S0|={s0_pos.size}，"
          f"自检 |max μ(S0) − demo 重建| = {chk:.2e}；{time.perf_counter() - t0:.1f}s")
    return dict(net="city_d", Dfull=Dfull, junc=junc, s0_pos=s0_pos, leak_pos=leak_pos,
                true_col=true_col, prev=prev, N=pb.net.N, self_check=chk,
                dict_note="demo_leak_inversion.build_dictionary（t=0 帧 C=0.3 emitter FD 签名，"
                          "按 sqrt(p_j(t)/p_j(0)) 帧间缩放，25 帧）在全部 junction 行上；"
                          "候选漏点 = 工单记录的 49 个节点",
                s0_note="S0 = 普查用的 40 个现有传感器（seed=2026 直接抽样）",
                pool_note="公平池 = junction − S0 − 三个漏点节点（主结果）；原池 = junction − S0")


def build_ltown():
    import augment_public as ap
    t0 = time.perf_counter()
    net, se = ap.lt_load()
    jn, cand, truth, sens = ap.lt_setup(se)
    qexp, ucf_e = ap.lt_units(net)
    d0, rh0 = ap.lt_nominal(net)
    Dn_ = ap.lt_dictionary(net, se, d0, rh0, cand, qexp, ucf_e)          # [N, 60]
    Dfull = Dn_[jn][None]                                                 # [1, 782, 60]
    pos_of = {int(n): i for i, n in enumerate(jn)}
    s0_pos = np.array(sorted(pos_of[int(n)] for n in sens), dtype=np.int64)
    leak_pos = np.array([pos_of[int(n)] for n in truth], dtype=np.int64)
    true_col = [int(np.where(cand == t)[0][0]) for t in truth]
    cs = ap.coherence_stats(Dn_, sens, cand, truth)
    st = coherence_stats_rows(Dfull, s0_pos)
    chk = max(abs(cs["coh_max"] - st["coh_max"]), abs(cs["coh_median_all"] - st["coh_median_all"]))
    o = np.load(os.path.join(DATA, "placement_orders_ltown.npz"))
    prev = {obj: np.asarray(o["augment_" + obj], dtype=np.int64) for obj in ("dopt", "cover")}
    pool_prev = np.asarray(o["pool"], dtype=np.int64)
    print(f"[ltown] 字典 D[T=1, m={jn.size}, NC={cand.size}]，|S0|={s0_pos.size}，"
          f"自检 |相干(S0) − augment_public.coherence_stats| = {chk:.2e}；"
          f"{time.perf_counter() - t0:.1f}s")
    return dict(net="ltown", Dfull=Dfull, junc=jn, s0_pos=s0_pos, leak_pos=leak_pos,
                true_col=true_col, prev=prev, pool_prev=pool_prev, N=net.N, self_check=chk,
                dict_note="augment_public.lt_dictionary（名义帧，C=0.3 emitter 扰动，状态机 "
                          "hacc=1e-10；与 ltown_coherence.py 同构造）在全部 junction 行上；"
                          "候选漏点 = seed 909 的 60 个",
                s0_note="S0 = 论文 §3.5 L-TOWN 反演所用 33 个传感器（seed 909）",
                pool_note="公平池 = junction − S0 − 三个漏点（上一轮 placement_orders_ltown.npz 的 pool 同）")


def build_hanoi():
    import augment_public as ap
    from calibrate import Problem
    from dgga.autodiff import solve_polished
    t0 = time.perf_counter()
    pb = Problem("hanoi")
    s, net = pb.s, pb.net
    junc = np.asarray(s.junc_nodes)
    qexp, ucf_e = ap.lt_units(net)
    d0, rh0 = pb.d[:1], pb.rh[:1]
    base = solve_polished(s, d0, rh0, accuracy=1e-12, max_iter=60, polish_steps=3)
    h0 = base["head"][0]
    Dn_ = np.zeros((net.N, junc.size))
    for j, c in enumerate(junc):
        ke = np.zeros((1, net.N))
        ke[0, c] = ucf_e / C_PROBE ** qexp
        sol = solve_polished(s, d0, rh0, ke=ke, accuracy=1e-12, max_iter=60, polish_steps=3)
        Dn_[:, j] = (sol["head"][0] - h0) / C_PROBE
    Dfull = Dn_[junc][None]                                               # [1, 31, 31]
    o = np.load(os.path.join(DATA, "placement_orders_pub_hanoi_synth25.npz"))
    s0_pos = np.unique(np.asarray(o["augment_fixed"], dtype=np.int64))
    prev = {obj: np.asarray(o["augment_" + obj], dtype=np.int64) for obj in ("dopt", "cover")}
    print(f"[hanoi] 字典 D[T=1, m={junc.size}, NC={junc.size}]，|S0|={s0_pos.size}；"
          f"{time.perf_counter() - t0:.1f}s")
    return dict(net="hanoi", Dfull=Dfull, junc=junc, s0_pos=s0_pos,
                leak_pos=np.array([], dtype=np.int64), true_col=[], prev=prev, N=net.N,
                self_check=0.0,
                dict_note="冒烟：calibrate.Problem('hanoi') 合成帧 0，C=0.3 emitter 扰动单帧签名，"
                          "31 个 junction 既是候选漏点也是候选传感位置（无漏损案例）",
                s0_note="S0 = random:10:2026（augment_public hanoi-augment 同 S0）",
                pool_note="池 = junction − S0（无漏点可剔）")


BUILDERS = dict(city_d=build_city_d, ltown=build_ltown, hanoi=build_hanoi)


# ======================================================================
# 评估：相干摘要 + 真值劲敌（评估用，不进入选点）
# ======================================================================
def truth_eval(Dfull, sel, true_col, leak_pos):
    """真值相关的**评估**量（只在选点结束后计算，用 T1..T3 标签）：各真漏点列的
    最强劲敌相干、>0.99 劲敌数、列范数（信号幅值），以及漏点本节点是否装了传感器。"""
    if not true_col:
        return {}
    D = np.asarray(Dfull)
    if D.ndim == 2:
        D = D[None]
    sel = np.unique(np.asarray(sel, dtype=np.int64))
    NC = D.shape[2]
    A = D[:, sel, :].reshape(-1, NC)
    n = np.linalg.norm(A, axis=0)
    mu = np.abs(A.T @ A) / np.maximum(n[:, None] * n[None, :], 1e-300)
    np.fill_diagonal(mu, -1.0)
    out = {}
    sset = set(sel.tolist())
    for lab, t, lp in zip(LABELS, true_col, leak_pos.tolist()):
        out[lab] = dict(rival_coh=float(mu[t].max()),
                        n_rivals_gt_099=int((mu[t] > 0.99).sum()),
                        n_rivals_gt_0999=int((mu[t] > 0.999).sum()),
                        col_norm=float(n[t]),
                        leak_node_is_sensor=bool(lp in sset))
    return out


def per_k_entry(Dfull, s0_pos, order, k, true_col, leak_pos):
    sel = np.r_[s0_pos, np.asarray(order[:k], dtype=np.int64)]
    e = coherence_stats_rows(Dfull, sel)
    e["truth"] = truth_eval(Dfull, sel, true_col, leak_pos)
    return e


# ======================================================================
# stage select
# ======================================================================
def stage_select(net):
    p = paths(net)
    info = BUILDERS[net]()
    Dfull, junc, s0_pos, leak_pos = info["Dfull"], info["junc"], info["s0_pos"], info["leak_pos"]
    true_col, prev = info["true_col"], info["prev"]
    T, m, NC = Dfull.shape
    ks = KS[net]
    kmax = max(ks)
    all_pos = np.arange(m)
    pool_full = np.setdiff1d(all_pos, s0_pos)
    pool_fair = np.setdiff1d(pool_full, leak_pos)
    if net == "ltown" and not np.array_equal(np.sort(np.setdiff1d(info["pool_prev"], s0_pos)), pool_fair):
        raise RuntimeError("L-TOWN 公平池与上一轮 pool 不一致")
    for obj, arr in prev.items():
        if np.intersect1d(arr, s0_pos).size:
            raise RuntimeError(f"上一轮 {obj} 序列与 S0 重叠")
    print(f"[{net}] 池：全 {pool_full.size}，公平 {pool_fair.size}（剔除漏点 {leak_pos.size}）；k={ks}")
    np.savez_compressed(p["cache"], Dfull=Dfull, junc=junc, s0_pos=s0_pos, leak_pos=leak_pos,
                        true_col=np.asarray(true_col, dtype=np.int64))

    runs = {}
    variants = [("coh_fair", "logdet2", pool_fair), ("cohmax_fair", "max", pool_fair)]
    if leak_pos.size:
        variants.append(("coh_full", "logdet2", pool_full))
    for name, obj, pool in variants:
        print(f"\n---- [{net}] {name}：objective={obj}，池 {pool.size}，k→{kmax} ----")
        r = coherence_augment(Dfull, s0_pos, kmax, pool=pool, objective=obj, verbose=True)
        if np.intersect1d(r["order"], s0_pos).size:
            raise RuntimeError("增设点与 S0 重叠（bug）")
        if name != "coh_full" and np.intersect1d(r["order"], leak_pos).size:
            raise RuntimeError("公平池选到了漏点节点（bug）")
        runs[name] = r
        print(f"  {name}：{r['t_total']:.1f}s，评估 {r['n_evals']} 次；J {r['steps'][0]['logdet2']:.2f}"
              f" → {r['steps'][-1]['logdet2']:.2f}；max μ {r['steps'][0]['coh_max']:.8f} → "
              f"{r['steps'][-1]['coh_max']:.8f}；中位 {r['steps'][0]['coh_median_all']:.4f} → "
              f"{r['steps'][-1]['coh_median_all']:.4f}")

    # ---- 随机对照：City D 复用上一轮 hv_leak_control 的抽法（原池抽，断言不落漏点，
    #      等价于公平池条件抽样；漏损重跑已在 data/audit_leak_control）；L-TOWN 从公平池抽 ----
    rand = {}
    rand_note = ""
    if net == "city_d":
        pool_nodes = np.array(sorted(set(junc.tolist()) - set(junc[s0_pos].tolist())), dtype=np.int64)
        pos_of = {int(n): i for i, n in enumerate(junc)}
        for seed in RAND_SEEDS:
            nodes = np.random.default_rng(seed).choice(pool_nodes, RAND_K, replace=False)
            rand[f"rand{RAND_K}_s{seed}"] = np.array([pos_of[int(n)] for n in nodes], dtype=np.int64)
        nodes = np.random.default_rng(0).choice(pool_nodes, 40, replace=False)
        rand["rand40_s0"] = np.array([pos_of[int(n)] for n in nodes], dtype=np.int64)
        for nm, arr in rand.items():
            if np.intersect1d(arr, leak_pos).size:
                raise RuntimeError(f"随机对照 {nm} 落在漏点上：不是公平样本")
        rand_note = ("random_rng(seed).choice(junction − S0, k) - 与 audit_augment/hv_leak_control.py "
                     "的 rand:k:seed 逐位同一抽样（漏损重跑复用 data/audit_leak_control/rand-*.json）；"
                     "断言无一落在漏点节点，故等价于公平池的条件抽样")
    else:
        for seed in RAND_SEEDS:
            rand[f"rand{RAND_K}_s{seed}"] = np.sort(np.random.default_rng(seed).choice(
                pool_fair, RAND_K, replace=False)).astype(np.int64)
        rand_note = "random_rng(seed).choice(公平池, 20) × 5 seeds"

    # ---- 每 k 的相干口径对比（同一字典、同一 S0）----
    per_k = {}
    for k in ks:
        row = {}
        for name, r in runs.items():
            row[name] = per_k_entry(Dfull, s0_pos, r["order"], k, true_col, leak_pos)
        for obj in ("dopt", "cover"):
            if prev[obj].size >= k:
                row[obj] = per_k_entry(Dfull, s0_pos, prev[obj], k, true_col, leak_pos)
        rk = [nm for nm, arr in rand.items() if arr.size == k]
        if rk:
            ents = [per_k_entry(Dfull, s0_pos, rand[nm], k, true_col, leak_pos) for nm in rk]
            agg = {}
            for key in ("coh_max", "coh_median_all", "coh_q99", "n_pairs_gt_0999", "n_pairs_gt_099",
                        "logdet2", "min_col_norm"):
                v = np.array([e[key] for e in ents], dtype=np.float64)
                agg[key] = dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
            if true_col:
                agg["truth"] = {lab: dict(rival_coh=dict(
                    median=float(np.median([e["truth"][lab]["rival_coh"] for e in ents])),
                    min=float(min(e["truth"][lab]["rival_coh"] for e in ents)),
                    max=float(max(e["truth"][lab]["rival_coh"] for e in ents))),
                    n_leak_node_is_sensor=int(sum(e["truth"][lab]["leak_node_is_sensor"] for e in ents)))
                    for lab in LABELS}
            row["rand"] = dict(n_seeds=len(ents), seeds=rk, agg=agg, per_seed=ents)
        per_k[f"+{k}"] = row

    s0_entry = coherence_stats_rows(Dfull, s0_pos)
    s0_entry["truth"] = truth_eval(Dfull, s0_pos, true_col, leak_pos)
    readable = dict(
        config=dict(net=net, T=int(T), m_candidate_positions=int(m), NC=int(NC),
                    n_fixed=int(s0_pos.size), n_pool_full=int(pool_full.size),
                    n_pool_fair=int(pool_fair.size), n_leak_nodes_excluded=int(leak_pos.size),
                    ks=ks, kmax=int(kmax), objective_main=OBJECTIVE_DEF,
                    objective_variant="cohmax_fair：字典序最小化 (max μ, J)；coh_full：同 logdet2 但用原池",
                    dictionary=info["dict_note"], s0=info["s0_note"], pool=info["pool_note"],
                    random_note=rand_note, self_check_abs_diff=float(info["self_check"]),
                    truth_independence="选点只用字典 D、S0 与候选位置池（公平池的剔除让方法知道得更少，"
                                       "不是更多）；本文件 truth 字段是选点结束后的评估量，标签 T1..T3",
                    host=platform.node(), generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                    mode="virtual augmentation：模拟新传感器，设计与模拟验证，非现场实装"),
        s0=s0_entry,
        curves={name: dict(objective=r["objective"], pool=("full" if name == "coh_full" else "fair"),
                           n_evals=r["n_evals"], t_total_sec=r["t_total"],
                           steps=[{kk: v for kk, v in s.items() if kk != "chosen"} | {"k": i}
                                  for i, s in enumerate(r["steps"])])
                for name, r in runs.items()},
        per_k=per_k)
    if net == "city_d":
        from augment_suite import assert_no_ids, id_set
        assert_no_ids(readable, id_set("city_d"), set(), "placement_coh_city_d.json")
    jdump(readable, p["readable"])
    orders = dict(augment_fixed=s0_pos, pool_fair=pool_fair, pool_full=pool_full, leak_pos=leak_pos)
    orders.update({name: np.asarray(r["order"], dtype=np.int64) for name, r in runs.items()})
    orders.update(rand)
    np.savez_compressed(p["orders"], **orders)
    print_select_table(net, readable)
    print(f"select[{net}] 完成：{p['readable']}、{p['orders']}")


def print_select_table(net, rd, L=print):
    ks = rd["config"]["ks"]
    cols = [c for c in ("coh_fair", "cohmax_fair", "coh_full", "dopt", "cover", "rand")
            if any(c in rd["per_k"][f"+{k}"] for k in ks)]
    s0 = rd["s0"]
    L(f"\n[{net}] S0 |S0|={rd['config']['n_fixed']}：max μ={s0['coh_max']:.8f} 中位 {s0['coh_median_all']:.4f} "
      f"q99 {s0['coh_q99']:.6f} >0.999 {s0['n_pairs_gt_0999']} >0.99 {s0['n_pairs_gt_099']} "
      f"J={s0['logdet2']:.2f}"
      + ("  真值劲敌 " + " ".join(f"{lab}:{v['rival_coh']:.6f}" for lab, v in s0["truth"].items())
         if s0.get("truth") else ""))
    L(f"  {'k':>4} {'序列':>12} | {'max μ':>12} {'中位':>7} {'q99':>9} {'>.999':>5} {'>.99':>5} "
      f"{'J':>9} {'minnorm':>9}" + (" | " + " ".join(f"{'劲敌' + lab:>9}" for lab in LABELS)
                                    + " 漏点上传感" if s0.get("truth") else ""))
    for k in ks:
        row = rd["per_k"][f"+{k}"]
        for c in cols:
            if c not in row:
                continue
            e = row[c]
            if c == "rand":
                a = e["agg"]
                base = (f"  {k:>4} {'rand×' + str(e['n_seeds']) + '中位':>12} | {a['coh_max']['median']:>12.8f} "
                        f"{a['coh_median_all']['median']:>7.4f} {a['coh_q99']['median']:>9.6f} "
                        f"{a['n_pairs_gt_0999']['median']:>5.0f} {a['n_pairs_gt_099']['median']:>5.0f} "
                        f"{a['logdet2']['median']:>9.2f} {a['min_col_norm']['median']:>9.3g}")
                if "truth" in a:
                    base += " | " + " ".join(f"{a['truth'][lab]['rival_coh']['median']:>9.6f}" for lab in LABELS)
                    base += "  " + "/".join(str(a["truth"][lab]["n_leak_node_is_sensor"]) for lab in LABELS)
            else:
                base = (f"  {k:>4} {c:>12} | {e['coh_max']:>12.8f} {e['coh_median_all']:>7.4f} "
                        f"{e['coh_q99']:>9.6f} {e['n_pairs_gt_0999']:>5} {e['n_pairs_gt_099']:>5} "
                        f"{e['logdet2']:>9.2f} {e['min_col_norm']:>9.3g}")
                if e.get("truth"):
                    base += " | " + " ".join(f"{e['truth'][lab]['rival_coh']:>9.6f}" for lab in LABELS)
                    base += "  " + "/".join("Y" if e["truth"][lab]["leak_node_is_sensor"] else "n"
                                            for lab in LABELS)
            L(base)


# ======================================================================
# stage mechanism：真值相关的**诊断**（不是设计） - 每个真漏点的劲敌相干在公平池上
# 单点最优能压到多低、需要装在几跳内、oracle 贪心 5 步的下限，以及该对在 J 里的份额
# ======================================================================
def _net_topology(net):
    if net == "city_d":
        import demo_leak_inversion as dm
        pb = dm.Problem()
        return pb.net.N, np.asarray(pb.net.link_n1), np.asarray(pb.net.link_n2), np.asarray(pb.cidx)
    import augment_public as ap
    n, se = ap.lt_load()
    _jn, cand, _truth, _sens = ap.lt_setup(se)
    return n.N, np.asarray(n.link_n1), np.asarray(n.link_n2), np.asarray(cand)


def stage_mechanism(net):
    from dgga.placement import _gram_per_row, _coh_from_gram
    p = paths(net)
    z = np.load(p["cache"])
    Dfull, junc, s0_pos, leak_pos = z["Dfull"], z["junc"], z["s0_pos"], z["leak_pos"]
    true_col = [int(x) for x in z["true_col"]]
    if not true_col:
        raise SystemExit("无漏损案例")
    N, n1, n2, cand_nodes = _net_topology(net)
    T, m, NC = Dfull.shape
    pool_full = np.setdiff1d(np.arange(m), s0_pos)
    pool_fair = np.setdiff1d(pool_full, leak_pos)
    Gi = _gram_per_row(Dfull)
    G0 = Gi[s0_pos].sum(axis=0)
    mu0 = _coh_from_gram(G0)
    np.fill_diagonal(mu0, -1.0)
    iu, ju = np.triu_indices(NC, 1)
    J0 = float(-np.log(np.clip(1 - mu0[iu, ju] ** 2, 1e-12, None)).sum())
    pos_of = {int(n): i for i, n in enumerate(junc)}
    dist_leak = {}
    out = {}
    for lab, t, lp in zip(LABELS, true_col, leak_pos.tolist()):
        dist_leak[lab] = hop_distances(N, n1, n2, int(junc[lp]))
        r = int(np.argmax(mu0[t]))
        rival_node = int(cand_nodes[r])
        d_tr = int(dist_leak[lab][rival_node])
        rival_pos = pos_of.get(rival_node)
        # 单点扫描：S0 + {v} 后 t 的劲敌相干（对全池向量化）
        G_all = G0[None] + Gi[pool_full]                                   # [n, NC, NC]
        mu = _coh_from_gram(G_all)[:, t, :]                                 # [n, NC]
        mu[:, t] = -1.0
        riv = mu.max(axis=1)                                                # 劲敌相干
        pair = _coh_from_gram(G_all)[:, t, r]                               # 与 S0 劲敌的相干
        is_fair = np.isin(pool_full, pool_fair)
        hops = dist_leak[lab][junc[pool_full]]
        best_fair = int(np.argmin(np.where(is_fair, riv, np.inf)))
        best_full = int(np.argmin(riv))
        rec = dict(
            rival_coh_S0=float(mu0[t, r]), n_rivals_gt_099_S0=int((mu0[t] > 0.99).sum()),
            hops_leak_to_rival=d_tr, rival_is_candidate_sensor_position=bool(rival_pos is not None),
            rival_in_fair_pool=bool(rival_pos is not None and rival_pos in set(pool_fair.tolist())),
            rival_node_already_in_S0=bool(rival_pos is not None and rival_pos in set(s0_pos.tolist())),
            single_sensor=dict(
                fair_pool_min_rival_coh=float(riv[best_fair]),
                fair_pool_best_hops_from_leak=int(hops[best_fair]),
                fair_pool_n_positions_rival_lt_0999=int(((riv < 0.999) & is_fair).sum()),
                fair_pool_n_positions_rival_lt_099=int(((riv < 0.99) & is_fair).sum()),
                fair_pool_n_positions_rival_lt_09=int(((riv < 0.9) & is_fair).sum()),
                fair_pool_min_hops_among_lt_0999=(int(hops[(riv < 0.999) & is_fair].min())
                                                  if ((riv < 0.999) & is_fair).any() else None),
                full_pool_min_rival_coh=float(riv[best_full]),
                full_pool_best_is_leak_node=bool(hops[best_full] == 0),
                leak_node_sensor_rival_coh=(float(riv[np.where(pool_full == lp)[0][0]])
                                            if lp in set(pool_full.tolist()) else None),
                rival_node_sensor_rival_coh=(float(riv[np.where(pool_full == rival_pos)[0][0]])
                                             if rival_pos is not None and rival_pos in set(pool_full.tolist())
                                             else None),
                s0_rival_pair_coh_after_best_fair=float(pair[best_fair])),
            pair_share_of_J=dict(
                pair_term_S0=float(-np.log(max(1 - mu0[t, r] ** 2, 1e-12))),
                J_S0=J0,
                pair_term_if_pushed_to_0999=float(-np.log(1 - 0.999 ** 2)),
                note="该对在 J 里的份额：把 μ 从 S0 值压到 0.999 最多只减这么多，而公平池 +k 的 J "
                     "总降幅见 curves；真值无关目标没有理由优先这一对"))
        # oracle 贪心（诊断上限，不是设计）：只看 t 的劲敌相干，公平池 5 步
        Gk = G0.copy()
        rem = pool_fair.copy()
        curve, hop_curve = [float(mu0[t].max())], []
        for _ in range(5):
            Ga = Gk[None] + Gi[rem]
            mm = _coh_from_gram(Ga)[:, t, :]
            mm[:, t] = -1.0
            rv = mm.max(axis=1)
            j = int(np.argmin(rv))
            curve.append(float(rv[j]))
            hop_curve.append(int(dist_leak[lab][junc[rem[j]]]))
            Gk = Gk + Gi[rem[j]]
            rem = np.delete(rem, j)
        rec["oracle_greedy_fair_pool"] = dict(rival_coh_curve=curve, hops_from_leak=hop_curve,
                                              note="以 t 的劲敌相干为目标的贪心（用真值身份，仅作可达下限）")
        out[lab] = rec
        print(f"[{net}] {lab}: 劲敌 S0 {rec['rival_coh_S0']:.6f}（{d_tr} 跳），公平池单点最优 "
              f"{rec['single_sensor']['fair_pool_min_rival_coh']:.6f}（距漏点 "
              f"{rec['single_sensor']['fair_pool_best_hops_from_leak']} 跳；<0.999 的位置 "
              f"{rec['single_sensor']['fair_pool_n_positions_rival_lt_0999']}，<0.99 "
              f"{rec['single_sensor']['fair_pool_n_positions_rival_lt_099']}），漏点本节点 "
              f"{rec['single_sensor']['leak_node_sensor_rival_coh']}，劲敌本节点 "
              f"{rec['single_sensor']['rival_node_sensor_rival_coh']}；oracle 5 步 "
              f"{[round(x, 6) for x in curve]} 跳 {hop_curve}；该对 J 份额 "
              f"{rec['pair_share_of_J']['pair_term_S0']:.2f}/{J0:.0f}")
    # 原池 coh_full 前 k 步里落在候选漏点节点上的个数（目标"喜欢"候选节点本身）
    zo = np.load(p["orders"])
    cand_pos = set(pos_of[int(n)] for n in cand_nodes if int(n) in pos_of)
    on_cand = {f"+{k}": int(sum(1 for v in zo["coh_full"][:k] if int(v) in cand_pos)) for k in KS[net]}
    on_cand_fair = {f"+{k}": int(sum(1 for v in zo["coh_fair"][:k] if int(v) in cand_pos)) for k in KS[net]}
    rd = jload(p["readable"])
    rd["mechanism"] = dict(per_truth=out, n_candidate_leak_nodes=int(len(cand_pos)),
                           coh_full_n_added_on_candidate_leak_nodes=on_cand,
                           coh_fair_n_added_on_candidate_leak_nodes=on_cand_fair,
                           note="真值相关诊断（选点结束后计算）：单点扫描 = S0+{v} 对全池向量化；"
                                "oracle 贪心用真值身份，只给可达下限")
    if net == "city_d":
        from augment_suite import assert_no_ids, id_set
        assert_no_ids(rd, id_set("city_d"), set(), "placement_coh_city_d.json")
    jdump(rd, p["readable"])
    print(f"  coh_full 前 k 步落在候选漏点节点上：{on_cand}；coh_fair：{on_cand_fair}（候选漏点 {len(cand_pos)}）")
    print(f"mechanism[{net}] 完成：{p['readable']}")


# ======================================================================
# stage leak（L-TOWN：augment_public.stage_lt_leak 原样反演器）
# ======================================================================
def ltown_configs():
    import augment_public as ap
    p = paths("ltown")
    net, se = ap.lt_load()
    jn, cand, truth, sens = ap.lt_setup(se)
    z = np.load(p["orders"])
    if not np.array_equal(np.sort(jn[z["augment_fixed"]]), sens):
        raise RuntimeError("orders 的 S0 与 lt_setup 不一致")
    cfgs = [("S0", sens, 0, "-")]
    for k in KS["ltown"]:
        cfgs.append((f"coh+{k}", np.sort(np.r_[sens, jn[z["coh_fair"][:k]]]), k, "coh"))
    cfgs.append(("cohmax+20", np.sort(np.r_[sens, jn[z["cohmax_fair"][:20]]]), 20, "cohmax"))
    for seed in RAND_SEEDS:
        cfgs.append((f"rand+20:s{seed}", np.sort(np.r_[sens, jn[z[f"rand20_s{seed}"]]]), 20, "rand"))
    for name, sn, k, _ in cfgs:
        if np.unique(sn).size != sens.size + k or np.intersect1d(sn, truth).size:
            raise RuntimeError(f"配置 {name} 传感器数或漏点重叠不对")
    return cfgs


def stage_leak_ltown(only, chunk, linear, out):
    import augment_public as ap
    cfgs = ltown_configs()
    ap.stage_lt_leak(chunk=chunk, only=only, linear=linear, configs=cfgs, out_path=out,
                     config_note="传感配置 = S0 / coh+k（相干驱动，公平池）/ cohmax+20（max 目标变体）/ "
                                 "rand+20:s0..4（公平池随机对照）；序列在 data/placement_orders_coh_ltown.npz")


def stage_merge_ltown(parts, out):
    merged = None
    for fp in parts:
        d = jload(fp)
        if not d:
            continue
        if merged is None:
            merged = dict(config=dict(d["config"]), runs={})
            merged["config"]["merged_from"] = []
        merged["config"]["merged_from"].append(os.path.relpath(fp, ROOT).replace(os.sep, "/"))
        for name, r in d["runs"].items():
            if name in merged["runs"] and all(g in merged["runs"][name] for g in ("noiseless", "noisy")):
                continue
            merged["runs"][name] = r
    if merged is None:
        raise SystemExit("无可合并文件")
    order = [c[0] for c in ltown_configs()]
    merged["runs"] = {n: merged["runs"][n] for n in order if n in merged["runs"]}
    jdump(merged, out)
    print(f"merge 完成：{out}（{len(merged['runs'])} 个配置）")


# ======================================================================
# stage leak（City D：工单案例，反演器 = demo_leak_inversion 原样）
# ======================================================================
def city_d_rand_positions(k, seed, junc, s0_pos, leak_pos):
    """随机对照 rand<k>_s<seed> 的抽法，与 audit_augment/hv_leak_control.py 的 rand:k:seed
    和 stage_select 逐位同一：从 junction − S0 的**节点**池按 seed 抽 k 个。抽到漏点节点的
    seed 直接作废（拒绝抽样） - 在"不含漏点"这个事件上取条件，等价于公平池上的均匀抽样，
    于是既与 coh_fair 同池，又让上一轮已跑完的 rand20_s0..s4 逐位复用。"""
    pool_nodes = np.array(sorted(set(junc.tolist()) - set(junc[s0_pos].tolist())), dtype=np.int64)
    pos_of = {int(n): i for i, n in enumerate(junc)}
    nodes = np.random.default_rng(int(seed)).choice(pool_nodes, int(k), replace=False)
    pos = np.array([pos_of[int(n)] for n in nodes], dtype=np.int64)
    if np.intersect1d(pos, np.asarray(leak_pos, dtype=np.int64)).size:
        raise ValueError(f"rand{k}_s{seed} 抽到了漏点节点：非公平样本，该 seed 作废")
    return pos


def city_d_sensors(spec, junc, z):
    s0_pos = np.unique(np.asarray(z["augment_fixed"], dtype=np.int64))
    if spec == "S0":
        add = np.array([], dtype=np.int64)
    else:
        m = re.fullmatch(r"(coh|cohfull|cohmax)\+(\d+)", spec)
        if m:
            key = {"coh": "coh_fair", "cohfull": "coh_full", "cohmax": "cohmax_fair"}[m.group(1)]
            add = np.asarray(z[key][:int(m.group(2))], dtype=np.int64)
        else:
            m = re.fullmatch(r"rand(\d+)_s(\d+)", spec)
            if not m:
                raise ValueError(spec)
            # npz 里存了的 seed 原样取（上一轮结果逐位不变）；没存的现抽，抽法同一
            add = (np.asarray(z[spec], dtype=np.int64) if spec in z.files
                   else city_d_rand_positions(m.group(1), m.group(2), junc, s0_pos, z["leak_pos"]))
    return np.sort(junc[s0_pos]), np.sort(junc[add])


def stage_leak_city_d(specs, out_dir):
    import torch
    import demo_leak_inversion as dm
    from augment_suite import assert_no_ids, demo_truth, id_set
    from dgga.units import LPSperCFS, MperFT
    torch.set_default_dtype(torch.float64)
    os.makedirs(out_dir, exist_ok=True)
    t_setup = time.time()
    pb = dm.Problem()
    net, s = pb.net, pb.s
    N = net.N
    junc = np.asarray(s.junc_nodes)
    z = np.load(paths("city_d")["orders"])
    targets, sens_demo, label = demo_truth(pb, dm)
    tnode = {label[n]: pb.node_index[n] for n in targets}
    true_idx = [pb.cand.index(n) for n in targets]
    lab_of = {pb.cand.index(n): label[n] for n in targets}
    sol_base = pb.fsolve(max_iter=dm.OBS_MI)
    ke_true = np.zeros(N)
    for n, q in targets.items():
        i = pb.node_index[n]
        p_m = (sol_base["head"][:, i] - net.elev_ft[i]).mean() * MperFT
        ke_true[i] = pb.ke_int_of_C(q / p_m ** pb.gamma)
    sol_true = pb.fsolve(ke=ke_true, max_iter=dm.OBS_MI)
    true_lk = {label[n]: float(sol_true["emitter"][:, pb.node_index[n]].mean() * LPSperCFS) for n in targets}
    ref = jload(os.path.join(DATA, "leak_augment_city_d.json"))
    for t, v in ref["config"]["true_leak_lps"].items():
        if abs(true_lk[t] - v) > 1e-9:
            raise RuntimeError(f"真值漏损流量 {t} 与记录不一致")
    noise_full = dm.NOISE_FT * np.random.default_rng(dm.SEED_NOISE).standard_normal((25, N))
    dist = {t: hop_distances(N, net.link_n1, net.link_n2, i) for t, i in tnode.items()}
    ids = id_set("city_d")
    print(f"[setup] host={platform.node()} threads={torch.get_num_threads()} nc={pb.nc} "
          f"真值流量 {true_lk}  {time.time() - t_setup:.0f}s")

    for spec in specs:
        tag = spec.replace("+", "_p").replace(":", "-")
        fp = os.path.join(out_dir, f"{tag}.json")
        if os.path.isfile(fp):
            print(f"[{spec}] 已存在，跳过")
            continue
        s0_nodes, added = city_d_sensors(spec, junc, z)
        sens = np.unique(np.r_[s0_nodes, added])
        prox = {}
        for t in LABELS:
            dn = dist[t]
            prox[t] = dict(hop_to_nearest_sensor=int(dn[sens].min()),
                           hop_to_nearest_added=int(dn[added].min()) if added.size else None,
                           n_added_within_1hop=int((dn[added] <= 1).sum()) if added.size else 0,
                           n_added_within_2hops=int((dn[added] <= 2).sum()) if added.size else 0,
                           leak_node_is_sensor=bool(tnode[t] in set(sens.tolist())))
        t0 = time.time()
        n_fwd0 = dm.FWD_COUNT[0]
        elev_s = torch.tensor(net.elev_ft[sens])
        obs = sol_true["head"][:, sens] - net.elev_ft[sens] + noise_full[:, sens]
        pred0 = (sol_base["head"][:, sens] - net.elev_ft[sens]).reshape(-1)
        dm._BASE_CACHE["pred0"] = pred0
        mse0 = float(((obs.reshape(-1) - pred0) ** 2).mean())
        D, Dn, coh = dm.build_dictionary(pb, sol_base, sens)
        A = np.abs(coh)
        iu, ju = np.triu_indices(pb.nc, 1)
        off = A[iu, ju]
        B = A.copy()
        np.fill_diagonal(B, -1.0)
        cs = dict(n_pairs=int(off.size), n_orthogonal_pairs=int((off < ORTH_TOL).sum()),
                  max_offdiag=float(off.max()), median_all=float(np.median(off)),
                  q99_all=float(np.quantile(off, 0.99)),
                  n_pairs_gt_0999=int((off > 0.999).sum()), n_pairs_gt_099=int((off > 0.99).sum()),
                  logdet2=float(-np.log(np.clip(1 - off ** 2, 1e-12, None)).sum()),
                  true_node_max_rival_coh={lab_of[j]: float(B[j].max()) for j in true_idx},
                  true_node_n_rivals_gt_099={lab_of[j]: int((B[j] > 0.99).sum()) for j in true_idx},
                  true_col_norm={lab_of[j]: float(np.linalg.norm(D[:, j])) for j in true_idx})
        print(f"\n[{spec}] 传感器 {sens.size}（增设 {added.size}）prox {prox}\n  相干 max {cs['max_offdiag']:.6f} "
              f"中位 {cs['median_all']:.4f} >.999 {cs['n_pairs_gt_0999']} J {cs['logdet2']:.2f} "
              f"劲敌 {cs['true_node_max_rival_coh']}")
        sys.stdout.flush()
        obs_t = torch.tensor(obs)
        st1 = dm.stage1_adam_l1(pb, obs_t, sens, elev_s, 1e-4)
        st1_top3 = [int(j) for j in np.argsort(-st1["leak_lps"])[:3]]
        support, fit = dm.stage2_support_search(pb, obs_t, sens, elev_s, D, Dn, coh, mse0)
        support = [int(j) for j in support]
        order = [j for j in np.argsort(-fit["leak_lps"]) if j in support]
        top3 = order[:3]
        flow_err = {}
        for n in targets:
            j = pb.cand.index(n)
            est = float(fit["leak_lps"][j]) if j in support else 0.0
            flow_err[label[n]] = dict(true_lps=true_lk[label[n]], est_lps=est,
                                      rel_err=abs(est - true_lk[label[n]]) / true_lk[label[n]])
        sup = []
        for j in support:
            if j in lab_of:
                sup.append(dict(kind=lab_of[j], leak_lps=float(fit["leak_lps"][j])))
            else:
                cw = {lab_of[t]: float(abs(coh[j, t])) for t in true_idx}
                best = max(cw, key=cw.get)
                sup.append(dict(kind="nontrue", closest_true=best, coh=cw[best],
                                leak_lps=float(fit["leak_lps"][j])))
        res = dict(spec=spec, n_sensors=int(sens.size), n_added=int(added.size),
                   n_in_s0=int(np.isin(sens, s0_nodes).sum()), proximity=prox, mse0=mse0, coherence=cs,
                   stage1=dict(mse=st1["mse"], top3_hits=len(set(st1_top3) & set(true_idx)),
                               top3_kinds=[lab_of.get(j, "nontrue") for j in st1_top3]),
                   support_size=len(support), support=sup,
                   top1_hit=bool(top3 and top3[0] in true_idx), top3_hits=len(set(top3) & set(true_idx)),
                   top3_exact=bool(set(top3) == set(true_idx)), flow_err=flow_err,
                   final_mse_ft2=fit["mse"], time_sec=time.time() - t0,
                   n_forward_solves=dm.FWD_COUNT[0] - n_fwd0, host=platform.node(),
                   threads=torch.get_num_threads(), noise_ft=dm.NOISE_FT, seed_noise=dm.SEED_NOISE,
                   lam=1e-4, inverter="demo_leak_inversion（Adam+L1 阶段 1 → 非线性 OMP + 互换抛光）原样",
                   noise_note="噪声场按 [25, N] 全节点生成（seed 909）再按传感器取列，"
                              "与 leak_augment_city_d.json / audit_leak_control 同一实现")
        assert_no_ids(res, ids, set(), f"leak json {spec}")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=float)
        print(f"[{spec}] 支撑 {[d['kind'] for d in sup]} top1 {res['top1_hit']} top3 {res['top3_hits']}/3 "
              f"流量误差 { {k: round(v['rel_err'], 3) for k, v in flow_err.items()} } "
              f"final_mse {fit['mse']:.3e} {res['time_sec']:.0f}s → {fp}")
        sys.stdout.flush()


# ======================================================================
# stage report
# ======================================================================
def _city_d_prev_rows():
    """上一轮 City D 结果（同反演器、同噪声实现）作对照行。"""
    rows = []
    la = jload(os.path.join(DATA, "leak_augment_city_d.json"))
    for gkey, g in la.get("groups", {}).items():
        name, grp = gkey.split(":")
        if grp != "noisy":
            continue
        c = g["coherence"]
        rows.append(dict(spec={"ga40": "S0(上一轮)", "demo40": "demo40"}.get(name, name),
                         source="data/leak_augment_city_d.json", n_sensors=g["n_sensors"],
                         n_added=g["n_sensors"] - 40 if name != "demo40" else 0,
                         coh_max=c["max_offdiag"], coh_median=c["median_all"],
                         n_gt_0999=c["n_pairs_gt_0999"], rival=c["true_node_max_rival_coh"],
                         leak_node_is_sensor=None, support=[d["kind"] for d in g["support"]],
                         top1=g["top1_hit"], top3=g["top3_hits"],
                         flow_err={k: v["rel_err"] for k, v in g["flow_err"].items()},
                         final_mse=g["final_mse_ft2"], time_sec=g["time_sec"], host=g.get("host")))
    for fp in sorted(glob.glob(os.path.join(DATA, "audit_leak_control", "*.json"))):
        g = jload(fp)
        c = g["coherence"]
        rows.append(dict(spec=g["spec"], source="data/audit_leak_control", n_sensors=g["n_sensors"],
                         n_added=g["n_added"], coh_max=c["max_offdiag"], coh_median=c["median_all"],
                         n_gt_0999=c["n_pairs_gt_0999"], rival=c["true_node_max_rival_coh"],
                         leak_node_is_sensor={t: v["leak_node_is_sensor"] for t, v in g["proximity"].items()},
                         support=[d["kind"] for d in g["support"]], top1=g["top1_hit"], top3=g["top3_hits"],
                         flow_err={k: v["rel_err"] for k, v in g["flow_err"].items()},
                         final_mse=g["final_mse_ft2"], time_sec=g["time_sec"], host=g.get("host")))
    return rows


def _city_d_new_rows():
    rows = []
    for fp in sorted(glob.glob(os.path.join(paths("city_d")["leak_dir"], "*.json"))):
        g = jload(fp)
        c = g["coherence"]
        rows.append(dict(spec=g["spec"], source="data/leak_coh_city_d", n_sensors=g["n_sensors"],
                         n_added=g["n_added"], coh_max=c["max_offdiag"], coh_median=c["median_all"],
                         n_gt_0999=c["n_pairs_gt_0999"], logdet2=c.get("logdet2"),
                         rival=c["true_node_max_rival_coh"],
                         leak_node_is_sensor={t: v["leak_node_is_sensor"] for t, v in g["proximity"].items()},
                         hop_to_nearest_added={t: v["hop_to_nearest_added"] for t, v in g["proximity"].items()},
                         support=[d["kind"] for d in g["support"]], top1=g["top1_hit"], top3=g["top3_hits"],
                         flow_err={k: v["rel_err"] for k, v in g["flow_err"].items()},
                         stage1_top3_hits=g["stage1"]["top3_hits"],
                         final_mse=g["final_mse_ft2"], time_sec=g["time_sec"], host=g.get("host"),
                         threads=g.get("threads")))
    return rows


def _spec_sort_key(spec):
    m = re.fullmatch(r"(coh|cohfull|cohmax)\+(\d+)", spec)
    if spec == "S0":
        return (0, 0, "")
    if m:
        return ({"coh": 1, "cohmax": 2, "cohfull": 3}[m.group(1)], int(m.group(2)), "")
    return (4, 0, spec)


def stage_report(machine=""):
    L = []
    A = L.append
    A("=" * 78)
    A("相干驱动增设（S0 固定 + k，为漏损搜索选点） - 数值表（scripts/augment_coherence.py --stage report）")
    A("=" * 78)
    A(f"目标：{OBJECTIVE_DEF}")
    A("对照：cohmax（字典序 (max μ, J)）、coh_full（原池，含漏点节点）、上一轮 D-opt / cover 序列、"
      "随机 +k（≥5 seeds）；公平池 = junction − S0 − 漏点节点。全程虚拟增设（设计与模拟验证，非实装）。")
    if machine:
        A(f"机器：{machine}")
    summary = dict(objective=OBJECTIVE_DEF, generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                   machine=machine, nets={})
    for net in ("hanoi", "ltown", "city_d"):
        rd = jload(paths(net)["readable"])
        if not rd:
            continue
        c = rd["config"]
        A(f"\n### {net} 选点（{os.path.relpath(paths(net)['readable'], ROOT)}）：字典 T={c['T']} × 位置 "
          f"{c['m_candidate_positions']} × 候选漏点 {c['NC']}；|S0|={c['n_fixed']}，公平池 {c['n_pool_fair']}"
          f"（剔除 {c['n_leak_nodes_excluded']}），原池 {c['n_pool_full']}；自检差 {c['self_check_abs_diff']:.1e}")
        print_select_table(net, rd, A)
        if "mechanism" in rd:
            mech = rd["mechanism"]
            A(f"  机理诊断（真值相关，选点后计算；单点扫描对全池向量化，oracle 贪心只给可达下限）：")
            A(f"  {'':>4} {'劲敌S0':>9} {'跳':>3} | {'公平池单点最优':>12} {'跳':>3} {'<.999位置':>8} {'<.99位置':>8} | "
              f"{'漏点本节点':>10} {'劲敌本节点':>10} | {'oracle5步':>10} {'对在J份额':>10}")
            for lab, r in mech["per_truth"].items():
                ss, oc, pj = r["single_sensor"], r["oracle_greedy_fair_pool"], r["pair_share_of_J"]
                A(f"  {lab:>4} {r['rival_coh_S0']:>9.6f} {r['hops_leak_to_rival']:>3} | "
                  f"{ss['fair_pool_min_rival_coh']:>12.6f} {ss['fair_pool_best_hops_from_leak']:>3} "
                  f"{ss['fair_pool_n_positions_rival_lt_0999']:>8} {ss['fair_pool_n_positions_rival_lt_099']:>8} | "
                  f"{(ss['leak_node_sensor_rival_coh'] if ss['leak_node_sensor_rival_coh'] is not None else float('nan')):>10.6f} "
                  f"{(ss['rival_node_sensor_rival_coh'] if ss['rival_node_sensor_rival_coh'] is not None else float('nan')):>10.6f} | "
                  f"{oc['rival_coh_curve'][-1]:>10.6f} {pj['pair_term_S0']:>6.2f}/{pj['J_S0']:.0f}"
                  + ("  （劲敌节点已在 S0 里）" if r.get("rival_node_already_in_S0") else ""))
            A(f"  原池 coh_full 前 k 步落在候选漏点节点上的个数 {mech['coh_full_n_added_on_candidate_leak_nodes']}；"
              f"公平池 coh_fair {mech['coh_fair_n_added_on_candidate_leak_nodes']}（候选漏点 {mech['n_candidate_leak_nodes']}）")
        ent = dict(config={k: v for k, v in c.items() if k not in ("objective_main", "truth_independence")},
                   mechanism=rd.get("mechanism"),
                   s0=rd["s0"], per_k=rd["per_k"],
                   curves={n: dict(objective=v["objective"], pool=v["pool"],
                                   objective_curve=[s["objective_value"] for s in v["steps"]],
                                   coh_max_curve=[s["coh_max"] for s in v["steps"]],
                                   coh_median_curve=[s["coh_median_all"] for s in v["steps"]],
                                   n_gt_0999_curve=[s["n_pairs_gt_0999"] for s in v["steps"]])
                           for n, v in rd["curves"].items()})
        summary["nets"][net] = ent

    # ---- L-TOWN 漏损反演 ----
    lk = jload(paths("ltown").get("leak", ""))
    prev = jload(os.path.join(DATA, "ltown_augment_leak.json"))
    if lk:
        c = lk["config"]
        A(f"\n### L-TOWN 漏损反演原样重跑（{os.path.relpath(paths('ltown')['leak'], ROOT)}）：seed "
          f"{c['seed_setup']}/{c['seed_batch']}，B={c['B']}，候选 {c['n_candidates']}，|S0|={c['n_sensors_s0']}，"
          f"C_true={c['C_true']}，Adam lr={c['lr']} {c['steps']} 步，λ={c['lam']}，{c.get('assemble')}+"
          f"{c.get('linear_solver')} chunk={c['chunk']}，GPU={c['gpu']}（{c.get('host')}，torch {c.get('torch')}）；"
          f"噪声组 σ={c['noise_ft']} ft seed {c['seed_noise']}")
        A(f"  {'配置':>12} {'传感':>4} | {'相干max':>12} {'中位全':>7} {'>.999':>5} {'>.99':>5} | "
          + " ".join(f"{'劲敌' + str(t):>9}" for t in c["truth_nodes"]) + f" | {'组':>9} {'loss末':>10} "
          f"{'top1真':>5} {'top3中':>5} " + " ".join(f"{'C' + str(t) + '(#)':>10}" for t in c["truth_nodes"])
          + f" {'步时s':>5}")
        rows = {}
        for src, d in (("prev", prev), ("coh", lk)):
            for name, r in d.get("runs", {}).items():
                if src == "prev" and name not in ("S0", "dopt+20", "cover+20", "cover+40", "cover+80"):
                    continue
                key = name if src == "coh" else f"{name}(上轮5090)"
                rows[key] = r
        for name, r in rows.items():
            co = r["coherence"]
            base = (f"  {name:>12} {r['n_sensors']:>4} | {co['coh_max']:>12.10f} {co['coh_median_all']:>7.4f} "
                    f"{co['n_pairs_gt_0999']:>5} {co['n_pairs_gt_099']:>5} | "
                    + " ".join(f"{co['rivals'][str(t)][1]:>9.6f}" for t in c["truth_nodes"]) + " | ")
            first = True
            for gp in ("noiseless", "noisy"):
                if gp not in r:
                    continue
                g = r[gp]
                A((base if first else " " * len(base))
                  + f"{gp:>9} {g['loss_end']:>10.4e} {str(g['top1_true']):>5} {g['n_true_in_top3']:>5} "
                  + " ".join(f"{g['truth_C'][str(t)]:>6.3f}(#{g['truth_rank'][str(t)]:>2})"
                             for t in c["truth_nodes"]) + f" {g['t_step_median']:>5.2f}")
                first = False
        summary["ltown_leak"] = dict(
            config={k: v for k, v in c.items() if k != "recorded"},
            runs={name: dict(n_sensors=r["n_sensors"], k=r["k"], objective=r["objective"],
                             coherence={k: v for k, v in r["coherence"].items()
                                        if k not in ("rivals", "n_rivals_gt_099", "top_coh_to_nearest_truth")},
                             rival_coh={f"T{i + 1}": r["coherence"]["rivals"][str(t)][1]
                                        for i, t in enumerate(c["truth_nodes"])},
                             n_rivals_gt_099={f"T{i + 1}": r["coherence"]["n_rivals_gt_099"][str(t)]
                                              for i, t in enumerate(c["truth_nodes"])},
                             groups={gp: dict(loss_end=r[gp]["loss_end"], top1_true=r[gp]["top1_true"],
                                              n_true_in_top3=r[gp]["n_true_in_top3"],
                                              truth_rank={f"T{i + 1}": r[gp]["truth_rank"][str(t)]
                                                          for i, t in enumerate(c["truth_nodes"])},
                                              truth_C={f"T{i + 1}": r[gp]["truth_C"][str(t)]
                                                       for i, t in enumerate(c["truth_nodes"])},
                                              t_step_median=r[gp]["t_step_median"])
                                     for gp in ("noiseless", "noisy") if gp in r})
                  for name, r in lk["runs"].items()})

    # ---- City D 漏损重跑 ----
    new_rows = _city_d_new_rows()
    if new_rows:
        prev_rows = _city_d_prev_rows()
        A(f"\n### City D 工单漏损案例（0.1 ft 噪声，seed 909，反演器原样）：相干驱动增设（data/leak_coh_city_d/）"
          f"与上一轮对照（data/leak_augment_city_d.json、data/audit_leak_control/）")
        A(f"  {'配置':>16} {'传感':>4} {'增设':>4} | {'相干max':>10} {'中位':>7} {'>.999':>5} | "
          f"{'劲敌T1':>8} {'劲敌T2':>8} {'劲敌T3':>8} 漏点上传感 | 支撑 | top1 top3 | T2误差 T3误差 | 秒  主机")
        for r in sorted(new_rows, key=lambda r: _spec_sort_key(r["spec"])) + prev_rows:
            lns = r["leak_node_is_sensor"]
            lns_s = ("/".join("Y" if lns[t] else "n" for t in LABELS) if isinstance(lns, dict) else "  -  ")
            A(f"  {r['spec']:>16} {r['n_sensors']:>4} {r['n_added']:>4} | {r['coh_max']:>10.6f} "
              f"{r['coh_median']:>7.4f} {r['n_gt_0999']:>5} | "
              + " ".join(f"{r['rival'][t]:>8.5f}" for t in LABELS)
              + f" {lns_s:>10} | {'/'.join(r['support']):<22} | {str(r['top1']):>5} {r['top3']:>3}/3 | "
              f"{r['flow_err']['T2']:>6.3f} {r['flow_err']['T3']:>6.3f} | {r['time_sec']:>5.0f} {r['host']}")
        summary["city_d_leak"] = dict(new=new_rows, controls=prev_rows)

    txt = "\n".join(L)
    print(txt)
    # 只重写表格部分：wip 文件里表头之前的手写记录原样保留（augment_public.stage_report 同法）
    wip = paths("city_d")["wip"]
    head = ""
    if os.path.isfile(wip):
        old = open(wip, "r", encoding="utf-8").read()
        mark = "\n".join(L[:2]) + "\n"
        if mark in old:
            head = old[:old.index(mark)]
    with open(wip, "w", encoding="utf-8") as f:
        f.write(head + txt + "\n")
    from augment_suite import assert_no_ids, id_set
    assert_no_ids(summary, id_set("city_d"), set(), "augment_coherence.json")
    jdump(summary, paths("city_d")["summary"])
    print(f"\n已写 {paths('city_d')['wip']}、{paths('city_d')['summary']}")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["select", "mechanism", "leak", "merge", "report"])
    ap.add_argument("--net", default="city_d", choices=["hanoi", "ltown", "city_d"])
    ap.add_argument("--only", default="", help="leak/ltown：只跑这些配置名（逗号分隔）")
    ap.add_argument("--specs", default="", help="leak/city_d：配置名（逗号分隔），如 coh+20,cohfull+20")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--linear", default="dense", choices=["dense", "cudss"])
    ap.add_argument("--out", default="", help="leak/ltown 输出 json（多卡分跑时各给一个）；merge 的输出")
    ap.add_argument("--parts", default="", help="merge：逗号分隔的分跑 json")
    ap.add_argument("--machine", default="", help="report：一句话记录机器")
    a = ap.parse_args()
    if a.stage == "select":
        stage_select(a.net)
    elif a.stage == "mechanism":
        stage_mechanism(a.net)
    elif a.stage == "leak":
        if a.net == "ltown":
            stage_leak_ltown([x for x in a.only.split(",") if x], a.chunk, a.linear,
                             a.out or paths("ltown")["leak"])
        elif a.net == "city_d":
            specs = [x for x in a.specs.split(",") if x]
            if not specs:
                raise SystemExit("--specs 必填")
            stage_leak_city_d(specs, paths("city_d")["leak_dir"])
        else:
            raise SystemExit("Hanoi 无漏损案例")
    elif a.stage == "merge":
        stage_merge_ltown([x for x in a.parts.split(",") if x], a.out or paths("ltown")["leak"])
    elif a.stage == "report":
        stage_report(a.machine)
    return 0


if __name__ == "__main__":
    sys.exit(main())

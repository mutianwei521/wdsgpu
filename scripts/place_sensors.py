# -*- coding: utf-8 -*-
"""place_sensors.py - 传感器最优布点实验 CLI（Phase G-B）。

用法：
  python -X utf8 scripts/place_sensors.py --stem city_d --frames 25 --kmax 80
  可选 --stage full|dopt|baselines|metrics|robust|report|all（默认 all；
  各阶段结果落盘可断点续跑，robust 阶段支持分批）。

流程：
  full      全候选（全部 junction）× T 帧灵敏度 S_full[T,Nj,P]（每帧一次 splu、
            Nj 列回代），并复现 G-A 基线（40 随机传感器 seed=2026, t=0）。
  dopt      贝叶斯 D-最优 lazy 贪心（先验精度正则 M0 = I/σ_prior²）+
            逐实例最优性证书 + 子模性经验抽查 + σ 敏感性检查。
  baselines 随机(30 种子)/度中心性/灵敏度范数/A-最优贪心/谱代理。
  metrics   全方法 × k=5..kmax 步进 5 的指标曲线，写 data/placement_<stem>.json。
  robust    local 设计鲁棒性：20 组随机真值 C_true 处对比 f(C0 处设计) vs
            f(C_true 处重设计)。
  augment   增设模式（处方式）：现有传感器 S0（--s0，缺省 ga40 = 普查用的 40 个
            "现有"传感器）一个不动，再加 k∈--augment-ks 个；两种目标
            （D-最优 / 跨过可辨识阈值的管数）都跑并报差异；找回计数与普查同判据
            同 σ；单调性由 dgga.placement.bayes_dopt_augment 逐步断言；
            另跑"从零重选 |S0|+k 个"作对照（报它丢掉多少 S0 下本可辨识的管）。
            输出 data/placement_<stem>.json["augment"]（含候选下标）与
            data/placement_augment_<stem>.json（不含任何节点/候选编号的可读版）。
            全程"虚拟增设"：在标定后的模型上模拟新传感器，是设计与模拟验证，
            不是现场实装。
  report    从 JSON 打印中文汇总。

中间产物：data/placement_cache_<stem>.npz（S_full 等）、
          data/placement_orders_<stem>.npz（各方法选点序列）、
          data/placement_<stem>.json（全部指标）。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.calib import clamped_mask, dead_branch_mask            # noqa: E402
from dgga.parse import Net                                       # noqa: E402
from dgga.placement import (aopt_greedy, bayes_dopt_augment,     # noqa: E402
                            bayes_dopt_greedy, degree_order, eval_subset,
                            full_sensitivity, norm_order,
                            random_orders, recovery_report,
                            spectral_order, submodularity_check)
from dgga.solver import GGASolver                                # noqa: E402
from dgga.autodiff import solve_polished                         # noqa: E402

DATA = os.path.join(ROOT, "data")
GGA_MI = 60
POLISH = 3
SEED_GA = 2026            # G-A 基线：40 随机传感器
SEED_RANDOM0 = 100        # 随机基线 30 种子 = 100..129
SEED_SUBMOD = 1
SEED_ROBUST = 20260810
SIG_P, SIG_N = 15.0, 0.1  # 缺省先验/噪声（正则来源见 dgga/placement.py 模块注释）
HW_COEF, HW_CEXP, HW_DEXP = 4.727, 1.852, 4.871   # hydcoeffs.c:95


# ----------------------------------------------------------------------
def resolve_inp(stem):
    """同 scripts/align.py:38 的约定。"""
    if stem.startswith("pub_"):
        idx_path = os.path.join(DATA, "public_reference_index.json")
        with open(idx_path, "r", encoding="utf-8") as f:
            ent = json.load(f).get(stem)
        if ent and ent.get("inp_used"):
            return os.path.join(ROOT, *ent["inp_used"].split("/"))
    special = {"city_d": os.path.join(ROOT, "networks", "realInpData"),
               "city_d_emit": os.path.join(ROOT, "networks", "variants")}
    return os.path.join(special.get(stem, os.path.join(ROOT, "networks",
                                                       "InpData")),
                        f"{stem}.inp")


def load_case(stem):
    """Net + GGASolver（陷阱：solver 会就地修正 net.dem_base_cfs → 先构造）。"""
    net = Net.load(os.path.join(DATA, "reference"), stem)
    inp = resolve_inp(stem)
    try:
        s = GGASolver(net, mode="dense", inp_path=inp)
    except NotImplementedError:
        s = GGASolver(net, mode="epanet", inp_path=inp)
    return net, s


def cache_path(stem):
    return os.path.join(DATA, f"placement_cache_{stem}.npz")


def orders_path(stem):
    return os.path.join(DATA, f"placement_orders_{stem}.npz")


def json_path(stem):
    return os.path.join(DATA, f"placement_{stem}.json")


def load_json(stem):
    p = json_path(stem)
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(stem, obj):
    with open(json_path(stem), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def frame_times(net, n_frames):
    step = int(net.meta["hyd_step_sec"])
    n_avail = int(net.meta["duration_sec"]) // step + 1
    T = min(int(n_frames), n_avail)
    return [i * step for i in range(T)]


def curve_counts(S_full, order, struct, mask149):
    """order 前缀 k=1.. 的 (零列数, unobservable 数, 149 集合中已抢回数)。"""
    T, m, P = S_full.shape
    colmax = np.zeros(P)
    zero_k = np.empty(len(order), dtype=np.int64)
    unob_k = np.empty(len(order), dtype=np.int64)
    rec_k = np.empty(len(order), dtype=np.int64)
    for j, i in enumerate(order):
        colmax = np.maximum(colmax, np.max(np.abs(S_full[:, i, :]), axis=0))
        z = colmax == 0.0
        zero_k[j] = int(z.sum())
        unob_k[j] = int((z & ~struct).sum())
        rec_k[j] = int((~z & mask149).sum())
    return zero_k, unob_k, rec_k


# ======================================================================
# stage full
# ======================================================================
def stage_full(stem, n_frames):
    print(f"===== stage full：{stem} 全候选多帧灵敏度 =====")
    net, s = load_case(stem)
    t_secs = frame_times(net, n_frames)
    T = len(t_secs)
    junc = np.asarray(s.junc_nodes)
    print(f"网络 N={s.N} Nj={junc.size} L={s.L}；帧数 T={T}"
          f"（整点，hyd_step={net.meta['hyd_step_sec']}s）")

    t0 = time.perf_counter()
    S_full, meta = full_sensitivity(s, net, t_secs, max_iter=GGA_MI,
                                    polish_steps=POLISH)
    t_full = time.perf_counter() - t0
    pidx = meta["pipe_idx"]
    P = pidx.size
    print(f"S_full 形状 {S_full.shape}（P={P} 根 H-W 管）；"
          f"总耗时 {t_full:.2f}s（前向 {meta['info']['t_forward']:.2f}s + "
          f"伴随 {meta['info']['t_adjoint']:.2f}s；splu 次数="
          f"{meta['info']['n_factorizations']}，RHS 列数={meta['info']['n_rhs']}）；"
          f"‖F‖∞={meta['resid_inf']:.3e}")

    # 结构掩码（dead / clamped，均限制在 P 根管上）
    D, RH = meta["demand"], meta["res_head"]
    ke0 = np.zeros(net.N)
    sol = solve_polished(s, D, RH, ke=ke0, r_hw=None, accuracy=1e-12,
                         max_iter=GGA_MI, polish_steps=POLISH)
    dead = dead_branch_mask(s, demand=D)[pidx]
    cm25 = clamped_mask(s, sol, margin=10.0)["mask"][pidx]
    cm_t0 = clamped_mask(s, sol, frames=[0], margin=10.0)["mask"][pidx]
    print(f"结构掩码：死支 {int(dead.sum())}；t=0 钳位(margin=10) "
          f"{int((cm_t0 & ~dead).sum())}（另含死支 {int((cm_t0 & dead).sum())}）；"
          f"25 帧全钳位 {int((cm25 & ~dead).sum())}")

    # ---- G-A 基线复现：40 随机传感器 seed=2026 ----
    rng = np.random.default_rng(SEED_GA)
    sens40 = np.sort(rng.choice(junc, size=min(40, junc.size), replace=False))
    pos40 = np.searchsorted(junc, sens40)
    ga = {}
    for tag, Sv, cm in (("t0", S_full[:1], cm_t0), ("all", S_full, cm25)):
        ev = eval_subset(Sv, pos40, SIG_P, SIG_N)
        colmax = np.max(np.abs(Sv[:, pos40, :]), axis=(0, 1))
        z = colmax == 0.0
        unob = z & ~dead & ~cm
        ga[tag] = dict(rank=ev["rank"], n_zero_col=int(z.sum()),
                       n_unobservable=int(unob.sum()), f=ev["f"],
                       crlb_ident=ev["crlb_ident"], n_frames=Sv.shape[0])
        if tag == "t0":
            mask149 = unob.copy()
    print(f"G-A 复现（40 随机传感器 seed={SEED_GA}）：")
    print(f"  单帧 t=0 ：数值秩 {ga['t0']['rank']}，零列 {ga['t0']['n_zero_col']}，"
          f"unobservable {ga['t0']['n_unobservable']}（G-A 实测 149）")
    print(f"  {T} 帧    ：数值秩 {ga['all']['rank']}，零列 {ga['all']['n_zero_col']}，"
          f"unobservable {ga['all']['n_unobservable']}")

    # 全候选可观测上限（"地板"）
    colmax_all = np.max(np.abs(S_full), axis=(0, 1))
    z_all = colmax_all == 0.0
    floor_unob = z_all & ~dead & ~cm25
    print(f"全候选 {junc.size} 传感器 × {T} 帧：零列 {int(z_all.sum())}"
          f"（死支 {int((z_all & dead).sum())} + 全帧钳位 "
          f"{int((z_all & cm25 & ~dead).sum())} + 其余 {int(floor_unob.sum())}）"
          f" → unobservable 地板 = {int(floor_unob.sum())}")

    np.savez_compressed(
        cache_path(stem), S_full=S_full, pipe_idx=pidx, junc=junc,
        t_secs=np.asarray(t_secs), dead=dead, cm25=cm25, cm_t0=cm_t0,
        mask149=mask149, pos40=pos40, t_full=t_full,
        resid_inf=meta["resid_inf"])
    res = load_json(stem)
    res["config"] = dict(stem=stem, n_frames=T, n_candidates=int(junc.size),
                         P=int(P), sigma_prior=SIG_P, sigma_noise=SIG_N,
                         gga_max_iter=GGA_MI, polish_steps=POLISH)
    res["full_sensitivity"] = dict(
        t_total_sec=t_full, t_forward_sec=meta["info"]["t_forward"],
        t_adjoint_sec=meta["info"]["t_adjoint"],
        n_factorizations=int(meta["info"]["n_factorizations"]),
        n_rhs=int(meta["info"]["n_rhs"]), resid_inf=meta["resid_inf"])
    res["structural"] = dict(
        n_dead=int(dead.sum()), n_clamped_t0=int((cm_t0 & ~dead).sum()),
        n_clamped_allframes=int((cm25 & ~dead).sum()),
        n_zero_col_all_sensors=int(z_all.sum()),
        unobservable_floor=int(floor_unob.sum()))
    res["baseline_ga"] = ga
    save_json(stem, res)
    print("stage full 完成，缓存已落盘。")


# ======================================================================
# stage dopt
# ======================================================================
def stage_dopt(stem, kmax, k_step=5):
    print(f"===== stage dopt：贝叶斯 D-最优 lazy 贪心（σ_prior={SIG_P}, "
          f"σ_noise={SIG_N} ft）=====")
    z = np.load(cache_path(stem))
    S_full = z["S_full"]
    m = S_full.shape[1]
    kmax = min(kmax, m)
    grid = list(range(k_step, kmax + 1, k_step))
    g = bayes_dopt_greedy(S_full, kmax, SIG_P, SIG_N, cert_ks=grid,
                          verbose=True)
    worst_ratio = min(c["ratio"] for c in g["cert"].values())
    drift = max(abs(c["f"] - c["f_recomputed"]) for c in g["cert"].values())
    print(f"贪心 k={kmax} 完成：耗时 {g['t_total']:.2f}s，惰性评估 "
          f"{g['n_evals']} 次（满评 {m}×{kmax}={m * kmax} 次）")
    print(f"最优性证书最差 ratio = {worst_ratio:.4f}"
          f"（子模贪心理论下界 1-1/e=0.632；ratio 为逐实例上界之比）")
    print(f"f 增量累计 vs 独立复算 logdet 最大漂移 = {drift:.3e}")

    sub = submodularity_check(S_full, SIG_P, SIG_N, n_samples=200,
                              seed=SEED_SUBMOD)
    print(f"子模性经验抽查：{sub['n_samples']} 组，违例 {sub['n_violations']}，"
          f"最小边际差 {sub['worst_margin']:.3e}（容差 -{sub['tol']:.0e}）")

    # σ 敏感性：各变 4 倍，比较前 20 个选点。恒等式：贪心只依赖 ρ=σ_p/σ_n。
    k_sens = min(20, kmax)
    base20 = set(g["order"][:k_sens])
    sens = {}
    for tag, sp, sn in (("sigma_prior x4", SIG_P * 4, SIG_N),
                        ("sigma_prior /4", SIG_P / 4, SIG_N),
                        ("sigma_noise x4", SIG_P, SIG_N * 4),
                        ("sigma_noise /4", SIG_P, SIG_N / 4)):
        gv = bayes_dopt_greedy(S_full, k_sens, sp, sn)
        ov = len(base20 & set(gv["order"]))
        sens[tag] = dict(sigma_prior=sp, sigma_noise=sn, overlap_top20=ov,
                         order=gv["order"])
        print(f"  敏感性 {tag:16s} (ρ={sp / sn:7.1f})：前 {k_sens} 选点与缺省"
              f"重合 {ov}/{k_sens}")
    same_a = sens["sigma_prior x4"]["order"] == sens["sigma_noise /4"]["order"]
    same_b = sens["sigma_prior /4"]["order"] == sens["sigma_noise x4"]["order"]
    print(f"  恒等式验证：σ_p×4 与 σ_n÷4 序列逐位相同 = {same_a}；"
          f"σ_p÷4 与 σ_n×4 = {same_b}（f 只依赖 ρ²）")

    ords = dict(np.load(orders_path(stem))) if os.path.isfile(
        orders_path(stem)) else {}
    ords["dopt"] = np.asarray(g["order"], dtype=np.int64)
    np.savez_compressed(orders_path(stem), **ords)
    res = load_json(stem)
    res["dopt"] = dict(order=g["order"], gains=g["gains"],
                       f_curve=g["f_curve"],
                       cert={str(k): v for k, v in g["cert"].items()},
                       worst_cert_ratio=worst_ratio, f_drift=drift,
                       n_lazy_evals=g["n_evals"], t_total_sec=g["t_total"],
                       submodularity=sub,
                       sensitivity={k: {kk: (vv if kk != "order" else vv)
                                        for kk, vv in v.items()}
                                    for k, v in sens.items()},
                       ratio_identity=dict(prior_x4_eq_noise_div4=bool(same_a),
                                           prior_div4_eq_noise_x4=bool(same_b)))
    save_json(stem, res)
    print("stage dopt 完成。")


# ======================================================================
# stage baselines
# ======================================================================
def stage_baselines(stem, kmax, n_random=30):
    print("===== stage baselines：随机/度中心性/范数/A-最优/谱代理 =====")
    net, s = load_case(stem)
    z = np.load(cache_path(stem))
    S_full = z["S_full"]
    junc = z["junc"]
    m = S_full.shape[1]
    kmax = min(kmax, m)

    t0 = time.perf_counter()
    rnd = random_orders(m, n_seeds=n_random, seed0=SEED_RANDOM0)
    deg = degree_order(net, junc)
    nrm = norm_order(S_full)
    ao = aopt_greedy(S_full, kmax, SIG_P, SIG_N)
    sp = spectral_order(s, net, kmax)
    print(f"A-最优贪心耗时 {ao['t_total']:.2f}s（非子模，无 (1-1/e) 保证，"
          f"如实标注）；谱代理用 {sp['n_eig']} 个低频特征向量"
          f"（图含 {sp['n_components']} 个连通分量；简化代理，非 Zhou et al. "
          f"2024 原文复刻）；基线总耗时 {time.perf_counter() - t0:.2f}s")

    ords = dict(np.load(orders_path(stem))) if os.path.isfile(
        orders_path(stem)) else {}
    ords.update(random=rnd, degree=np.asarray(deg, dtype=np.int64),
                norm=np.asarray(nrm, dtype=np.int64),
                aopt=np.asarray(ao["order"], dtype=np.int64),
                spectral=np.asarray(sp["order"], dtype=np.int64))
    np.savez_compressed(orders_path(stem), **ords)
    res = load_json(stem)
    res["baselines_meta"] = dict(
        n_random_seeds=n_random, random_seed0=SEED_RANDOM0,
        aopt_t_sec=ao["t_total"], aopt_note="A-最优贪心非子模，无 (1-1/e) 保证",
        spectral_n_eig=int(sp["n_eig"]),
        spectral_note="Zhou et al. 2024 Water Research 简化代理（图拉普拉斯低频"
                      "子空间行选贪心），非原文复刻")
    save_json(stem, res)
    print("stage baselines 完成。")


# ======================================================================
# stage metrics
# ======================================================================
def stage_metrics(stem, kmax, k_step=5):
    print("===== stage metrics：全方法 × k 网格指标曲线 =====")
    z = np.load(cache_path(stem))
    S_full, dead, cm25, mask149 = (z["S_full"], z["dead"], z["cm25"],
                                   z["mask149"])
    struct = dead | cm25
    ords = dict(np.load(orders_path(stem)))
    m = S_full.shape[1]
    kmax = min(kmax, m)
    grid = list(range(k_step, kmax + 1, k_step))
    n149 = int(mask149.sum())

    methods = ["dopt", "aopt", "norm", "degree", "spectral"]
    curves = {}
    for name in methods:
        order = ords[name][:kmax].tolist()
        zc, un, rc = curve_counts(S_full, order, struct, mask149)
        cur = dict(k=grid, f=[], rank=[], n_zero_col=[], n_unobservable=[],
                   recovered149=[], crlb_ident=[], bayes_trace=[],
                   lam_min_M=[])
        for k in grid:
            ev = eval_subset(S_full, order[:k], SIG_P, SIG_N)
            cur["f"].append(ev["f"])
            cur["rank"].append(ev["rank"])
            cur["crlb_ident"].append(ev["crlb_ident"])
            cur["bayes_trace"].append(ev["bayes_trace"])
            cur["lam_min_M"].append(ev["lam_min_M"])
            cur["n_zero_col"].append(int(zc[k - 1]))
            cur["n_unobservable"].append(int(un[k - 1]))
            cur["recovered149"].append(int(rc[k - 1]))
        cur["unobservable_every_k"] = un.tolist()
        cur["recovered149_every_k"] = rc.tolist()
        curves[name] = cur
        print(f"  {name:8s} 完成（k={kmax} 时 f={cur['f'][-1]:.2f}，"
              f"秩={cur['rank'][-1]}，unobservable={cur['n_unobservable'][-1]}）")

    # 随机基线：30 种子 → 中位数与 IQR
    rnd = ords["random"]
    agg = {k: {mt: [] for mt in ("f", "rank", "n_unobservable", "crlb_ident",
                                 "bayes_trace", "lam_min_M", "recovered149",
                                 "n_zero_col")} for k in grid}
    t0 = time.perf_counter()
    for srow in rnd:
        order = srow[:kmax].tolist()
        zc, un, rc = curve_counts(S_full, order, struct, mask149)
        for k in grid:
            ev = eval_subset(S_full, order[:k], SIG_P, SIG_N)
            a = agg[k]
            a["f"].append(ev["f"])
            a["rank"].append(ev["rank"])
            a["crlb_ident"].append(ev["crlb_ident"])
            a["bayes_trace"].append(ev["bayes_trace"])
            a["lam_min_M"].append(ev["lam_min_M"])
            a["n_unobservable"].append(int(un[k - 1]))
            a["recovered149"].append(int(rc[k - 1]))
            a["n_zero_col"].append(int(zc[k - 1]))
    rc_cur = dict(k=grid)
    for mt in ("f", "rank", "n_unobservable", "crlb_ident", "bayes_trace",
               "lam_min_M", "recovered149", "n_zero_col"):
        arrs = np.array([agg[k][mt] for k in grid], dtype=np.float64)
        rc_cur[mt + "_median"] = np.median(arrs, axis=1).tolist()
        rc_cur[mt + "_q25"] = np.percentile(arrs, 25, axis=1).tolist()
        rc_cur[mt + "_q75"] = np.percentile(arrs, 75, axis=1).tolist()
    curves["random"] = rc_cur
    print(f"  random   完成（{rnd.shape[0]} 种子，"
          f"{time.perf_counter() - t0:.1f}s）")

    # ---- 关键科学数字 ----
    res = load_json(stem)
    ga = res.get("baseline_ga", {})
    un_d = np.asarray(curves["dopt"]["unobservable_every_k"])
    rc_d = np.asarray(curves["dopt"]["recovered149_every_k"])
    target = 0.8 * n149
    k80 = int(np.argmax(rc_d >= target) + 1) if np.any(rc_d >= target) else -1
    i40 = grid.index(40) if 40 in grid else len(grid) - 1
    key = dict(
        n149=n149, recover_target=float(target), k_recover80=k80,
        dopt_k40=dict(f=curves["dopt"]["f"][i40],
                      rank=curves["dopt"]["rank"][i40],
                      n_unobservable=curves["dopt"]["n_unobservable"][i40],
                      recovered149=curves["dopt"]["recovered149"][i40],
                      crlb_ident=curves["dopt"]["crlb_ident"][i40],
                      bayes_trace=curves["dopt"]["bayes_trace"][i40]),
        random_k40=dict(f=curves["random"]["f_median"][i40],
                        rank=curves["random"]["rank_median"][i40],
                        n_unobservable=curves["random"][
                            "n_unobservable_median"][i40],
                        recovered149=curves["random"][
                            "recovered149_median"][i40],
                        crlb_ident=curves["random"]["crlb_ident_median"][i40],
                        bayes_trace=curves["random"]["bayes_trace_median"][i40]))
    res["curves"] = curves
    res["key_numbers"] = key
    save_json(stem, res)

    # ---- 中文汇总表 ----
    print("\n---- f(S)（同一贝叶斯 D 目标）----")
    hdr = f"{'k':>4} | {'D-opt':>10} {'A-opt':>10} {'范数':>10} {'度':>10} " \
          f"{'谱':>10} {'随机中位':>10}"
    print(hdr)
    for j, k in enumerate(grid):
        print(f"{k:>4} | {curves['dopt']['f'][j]:>10.2f} "
              f"{curves['aopt']['f'][j]:>10.2f} "
              f"{curves['norm']['f'][j]:>10.2f} "
              f"{curves['degree']['f'][j]:>10.2f} "
              f"{curves['spectral']['f'][j]:>10.2f} "
              f"{curves['random']['f_median'][j]:>10.2f}")
    print("\n---- unobservable 管道数（零列且非死支/非全帧钳位）----")
    print(hdr)
    for j, k in enumerate(grid):
        print(f"{k:>4} | {curves['dopt']['n_unobservable'][j]:>10d} "
              f"{curves['aopt']['n_unobservable'][j]:>10d} "
              f"{curves['norm']['n_unobservable'][j]:>10d} "
              f"{curves['degree']['n_unobservable'][j]:>10d} "
              f"{curves['spectral']['n_unobservable'][j]:>10d} "
              f"{curves['random']['n_unobservable_median'][j]:>10.1f}")
    print("\n---- 数值秩（25 帧堆叠灵敏度）----")
    print(hdr)
    for j, k in enumerate(grid):
        print(f"{k:>4} | {curves['dopt']['rank'][j]:>10d} "
              f"{curves['aopt']['rank'][j]:>10d} "
              f"{curves['norm']['rank'][j]:>10d} "
              f"{curves['degree']['rank'][j]:>10d} "
              f"{curves['spectral']['rank'][j]:>10d} "
              f"{curves['random']['rank_median'][j]:>10.1f}")
    if ga:
        print("\n---- 关键科学数字 ----")
        print(f"  单帧 vs 多帧（同 40 随机传感器 seed={SEED_GA}）：秩 "
              f"{ga['t0']['rank']} → {ga['all']['rank']}；unobservable "
              f"{ga['t0']['n_unobservable']} → {ga['all']['n_unobservable']}")
        print(f"  D-opt k=40 vs 随机 40 中位：unobservable "
              f"{key['dopt_k40']['n_unobservable']} vs "
              f"{key['random_k40']['n_unobservable']:.1f}；秩 "
              f"{key['dopt_k40']['rank']} vs {key['random_k40']['rank']:.1f}；"
              f"CRLB(可辨识子空间) {key['dopt_k40']['crlb_ident']:.3e} vs "
              f"{key['random_k40']['crlb_ident']:.3e}")
        floor = res.get("structural", {}).get("unobservable_floor", "?")
        if k80 > 0:
            print(f"  抢回 149 根中 80%（≥{target:.1f} 根）最少需要 D-opt 传感器"
                  f" k = {k80}（unobservable 地板 = {floor}）")
        else:
            mx = int(rc_d.max()) if rc_d.size else 0
            print(f"  k≤{kmax} 内未达 80% 抢回目标：最多抢回 {mx}/{n149}"
                  f"（unobservable 地板 = {floor}）")
    print("stage metrics 完成，JSON 已落盘。")


# ======================================================================
# stage robust
# ======================================================================
def stage_robust(stem, n_robust=20, k_robust=40, budget_sec=480):
    print(f"===== stage robust：local 设计鲁棒性（{n_robust} 组 C_true，"
          f"k={k_robust}）=====")
    net, s = load_case(stem)
    z = np.load(cache_path(stem))
    pidx, t_secs = z["pipe_idx"], z["t_secs"].tolist()
    ords = dict(np.load(orders_path(stem)))
    design_c0 = ords["dopt"][:k_robust].tolist()
    P = pidx.size
    len_p = np.asarray(net.len_ft, dtype=np.float64)[pidx]
    diam_p = np.asarray(net.diam_ft, dtype=np.float64)[pidx]
    r_base = s.r_hw.detach().cpu().numpy().copy()
    hexp = s.hexp

    res = load_json(stem)
    rb = res.get("robust", dict(samples=[], seed=SEED_ROBUST,
                                k_robust=k_robust, n_target=n_robust))
    done = len(rb["samples"])
    rng = np.random.default_rng(SEED_ROBUST)
    draws = [rng.uniform(70.0, 150.0, size=P) for _ in range(n_robust)]

    t_start = time.perf_counter()
    for j in range(done, n_robust):
        if time.perf_counter() - t_start > budget_sec:
            print(f"  预算 {budget_sec}s 用尽，已完成 {j}/{n_robust}；"
                  f"重跑本 stage 续算。")
            break
        C_true = draws[j]
        r_true = r_base.copy()
        r_true[pidx] = HW_COEF * len_p / C_true ** HW_CEXP / diam_p ** HW_DEXP
        t0 = time.perf_counter()
        # wrt='r' 再乘 C_true 处的链式因子（dr_dC 用的是 solver 的 C0，不适用）
        S_r, meta = full_sensitivity(s, net, t_secs, r_hw=r_true, wrt="r",
                                     max_iter=GGA_MI, polish_steps=POLISH)
        chain = -hexp * r_true[pidx] / C_true
        S_true = S_r[:, :, :] * chain[None, None, :]
        g_true = bayes_dopt_greedy(S_true, k_robust, SIG_P, SIG_N)
        f_c0 = eval_subset(S_true, design_c0, SIG_P, SIG_N)["f"]
        f_opt = eval_subset(S_true, g_true["order"], SIG_P, SIG_N)["f"]
        gap = (f_opt - f_c0) / f_opt if f_opt > 0 else 0.0
        ov = len(set(design_c0) & set(g_true["order"]))
        rb["samples"].append(dict(
            f_design_c0=f_c0, f_redesign=f_opt, rel_gap=gap,
            overlap=ov, resid_inf=meta["resid_inf"],
            t_sec=time.perf_counter() - t0))
        print(f"  样本 {j + 1:2d}/{n_robust}：f(C0 设计)={f_c0:.2f}  "
              f"f(重设计)={f_opt:.2f}  相对差={gap * 100:.2f}%  "
              f"选点重合 {ov}/{k_robust}  （{rb['samples'][-1]['t_sec']:.1f}s）")
        res["robust"] = rb
        save_json(stem, res)

    if len(rb["samples"]) == n_robust:
        gaps = np.array([x["rel_gap"] for x in rb["samples"]])
        ovs = np.array([x["overlap"] for x in rb["samples"]])
        rb["summary"] = dict(
            median_rel_gap=float(np.median(gaps)),
            q25_rel_gap=float(np.percentile(gaps, 25)),
            q75_rel_gap=float(np.percentile(gaps, 75)),
            max_rel_gap=float(gaps.max()),
            median_overlap=float(np.median(ovs)))
        print(f"鲁棒性总结：f 相对差中位 {rb['summary']['median_rel_gap'] * 100:.2f}%"
              f"（IQR [{rb['summary']['q25_rel_gap'] * 100:.2f}%, "
              f"{rb['summary']['q75_rel_gap'] * 100:.2f}%]，最大 "
              f"{rb['summary']['max_rel_gap'] * 100:.2f}%）；选点重合中位 "
              f"{rb['summary']['median_overlap']:.0f}/{k_robust}")
        res["robust"] = rb
        save_json(stem, res)
        print("stage robust 完成。")


# ======================================================================
# stage augment（增设模式：S0 固定，再加 k 个）
# ======================================================================
def augment_readable_path(stem):
    return os.path.join(DATA, f"placement_augment_{stem}.json")


def parse_s0(spec, z, ords):
    """--s0 规范 → S0 的候选下标（S_full 第二维）。
    ga40           缓存 pos40：普查用的 40 个"现有"传感器（seed=2026；
                   Hanoi 上 = 全部 31 个 junction，无可增设余量）
    random:<n>:<seed>  n 个随机 junction（冒烟/公开网复现用）
    dopt:<n>       从零 D-opt 前 n 个（需先跑 stage dopt）
    """
    m = z["S_full"].shape[1]
    if spec == "ga40":
        return np.asarray(z["pos40"], dtype=np.int64)
    if spec.startswith("random:"):
        _, n, seed = spec.split(":")
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(m, size=min(int(n), m), replace=False))
    if spec.startswith("dopt:"):
        return np.asarray(ords["dopt"][:int(spec.split(":")[1])], dtype=np.int64)
    raise ValueError(f"--s0 不识别：{spec}")


def _post_summary(post_std, mask, sig_p):
    """逐管后验标准差在 mask 管集上的摘要（σ 口径的可辨识度）。"""
    v = np.asarray(post_std)[np.asarray(mask, dtype=bool)]
    if v.size == 0:
        return dict(n=0, median=None, max=None, n_half_prior=0)
    return dict(n=int(v.size), median=float(np.median(v)), max=float(v.max()),
                n_half_prior=int(np.sum(v <= 0.5 * sig_p)))


def _k_metrics(S_full, fixed, sel, mask149, struct, atol, post_std=None):
    """一个布点 sel 的全部指标（找回/新不可辨识集/CRLB/σ 口径摘要）。"""
    ev = eval_subset(S_full, sel, SIG_P, SIG_N)
    rep = recovery_report(S_full, fixed, sel, mask149, struct, atol)
    out = dict(n_sensors=int(np.unique(sel).size), f=ev["f"], rank=ev["rank"],
               crlb_ident=ev["crlb_ident"], bayes_trace=ev["bayes_trace"],
               n_recovered=rep["n_recovered"], n_target=rep["n_target"],
               n_target_left=rep["n_target_left"], n_lost=rep["n_lost"],
               n_unobservable=rep["n_unobservable_after"],
               n_identifiable=rep["n_ident_after"])
    if post_std is not None:
        nonstruct = ~struct
        out["post_std_all_nonstruct"] = _post_summary(post_std, nonstruct, SIG_P)
        out["post_std_target"] = _post_summary(post_std, mask149, SIG_P)
        out["post_std_recovered"] = _post_summary(post_std, rep["recovered_mask"],
                                                  SIG_P)
    return out


def stage_augment(stem, ks, s0_spec="ga40", atol=0.0):
    print(f"===== stage augment：增设模式（S0={s0_spec} 固定，再加 k∈{ks}；"
          f"σ_prior={SIG_P}, σ_noise={SIG_N} ft，普查判据 max|S|>{atol:g}）=====")
    z = np.load(cache_path(stem))
    S_full, dead, cm25, mask149 = (z["S_full"], z["dead"], z["cm25"],
                                   z["mask149"])
    struct = dead | cm25
    ords = dict(np.load(orders_path(stem))) if os.path.isfile(
        orders_path(stem)) else {}
    T, m, P = S_full.shape
    fixed = parse_s0(s0_spec, z, ords)
    n0 = int(fixed.size)
    room = m - n0
    ks_req = sorted(set(int(k) for k in ks))
    ks = [k for k in ks_req if k <= room]
    if ks != ks_req:
        print(f"  候选余量 {room}（m={m} − |S0|={n0}）：k 网格截为 {ks}")
    if not ks:
        print("  无可增设余量，stage augment 跳过（S0 已含全部候选）。")
        return
    kadd = max(ks)
    # 目标集 = S0 下"传感不足"的管（普查同判据：非结构且零列）。S0=ga40 时
    # 必须与缓存里普查得到的 mask149 逐位相同（断言），其他 S0 则按各自的普查算。
    unob0 = recovery_report(S_full, fixed, fixed, np.zeros(P, bool), struct,
                            atol)["unobservable_mask"]
    if s0_spec == "ga40":
        if not np.array_equal(unob0, mask149):
            raise RuntimeError("S0=ga40 的传感不足集与缓存 mask149 不一致（bug）")
        target_note = "普查 mask149（与缓存逐位一致）"
    else:
        target_note = f"S0={s0_spec} 自身普查所得（缓存 mask149 为 ga40 口径，不用）"
    mask149 = unob0
    n149 = int(mask149.sum())
    print(f"网络 {stem}：T={T} 帧，候选 {m}，P={P}（结构性 {int(struct.sum())}"
          f" = 死支 {int(dead.sum())} + 全帧钳位 {int((cm25 & ~dead).sum())}）；"
          f"S0 = {n0} 个传感器；目标集 = {n149} 根传感不足管（{target_note}）")

    # ---- S0 自身 ----
    from dgga.placement import posterior_std
    ps0 = posterior_std(S_full, fixed, SIG_P, SIG_N)
    base = _k_metrics(S_full, fixed, fixed, mask149, struct, atol, ps0)
    def _g3(x):
        return "n/a" if x is None else f"{x:.3g}"

    print(f"S0 自身：可辨识 {base['n_identifiable']}，unobservable "
          f"{base['n_unobservable']}，秩 {base['rank']}，f={base['f']:.2f}，"
          f"CRLB(可辨识子空间)={base['crlb_ident']:.3e}；σ 口径（后验 std 中位，"
          f"先验 {SIG_P}）：目标集 {_g3(base['post_std_target']['median'])}，"
          f"全部非结构管 {_g3(base['post_std_all_nonstruct']['median'])}")

    colmax_c = np.max(np.abs(S_full), axis=0)                   # [m, P]
    colmax0 = colmax_c[fixed].max(axis=0) if n0 else np.zeros(P)
    target80 = 0.8 * n149

    # ---- 两种目标的增设 ----
    runs = {}
    for obj in ("dopt", "cover"):
        t0 = time.perf_counter()
        a = bayes_dopt_augment(S_full, fixed, kadd, SIG_P, SIG_N, objective=obj,
                               struct_mask=struct, atol=atol, cert_ks=ks,
                               verbose=True)
        # 逐步找回曲线（普查同判据，running-max）
        cm = colmax0.copy()
        rec = np.empty(kadd, dtype=np.int64)
        for j, i in enumerate(a["order"]):
            cm = np.maximum(cm, colmax_c[i])
            rec[j] = int(((cm > atol) & mask149).sum())
        if np.any(np.diff(rec) < 0):
            raise RuntimeError("找回曲线非单调（bug）")
        k80 = (int(np.argmax(rec >= target80)) + 1
               if n149 and np.any(rec >= target80) else -1)
        kall = (int(np.argmax(rec >= n149)) + 1
                if n149 and np.any(rec >= n149) else -1)
        per_k = {}
        for k in ks:
            sel = np.r_[fixed, np.asarray(a["order"][:k], dtype=np.int64)]
            mk = _k_metrics(S_full, fixed, sel, mask149, struct, atol,
                            a["cert"][k]["post_std"])
            c = a["cert"][k]
            mk.update(k=k, df=c["df"], cert_ratio=c["ratio"],
                      upper_df=c["upper_df"], f_drift=abs(c["f"] - c["f_recomputed"]),
                      ident_curve_k=a["ident_curve"][k])
            if "cover_ratio" in c:
                mk.update(cover_ratio=c["cover_ratio"], cover_upper=c["cover_upper"])
            if mk["n_lost"] != 0:
                raise RuntimeError("增设丢管 ≠ 0：违反单调性（bug）")
            if mk["n_identifiable"] != a["ident_curve"][k]:
                raise RuntimeError("recovery_report 与 augment 内部可辨识计数不一致")
            per_k[k] = mk
        runs[obj] = dict(added=a["order"], gains=a["gains"], f0=a["f0"],
                         f_curve=a["f_curve"], ident_curve=a["ident_curve"],
                         recovered_every_k=rec.tolist(),
                         k_recover80=k80, k_recover_all=kall,
                         cover_gains=a["cover_gains"],
                         cover_saturated_at=a["cover_saturated_at"],
                         n_evals=a["n_evals"], t_total_sec=a["t_total"],
                         per_k=per_k)
        print(f"  目标 {obj:5s}：耗时 {time.perf_counter() - t0:.2f}s，评估 "
              f"{a['n_evals']} 次；覆盖饱和步 {a['cover_saturated_at']}；"
              f"找回 80% 最少增设 k={k80}，全部找回 k={kall}（-1=网格内未达）；"
              f"单调性断言逐步通过（可辨识集只增、后验方差只降）")

    # ---- 对照：从零重选 |S0|+k 个（现有设计的做法）----
    g = bayes_dopt_greedy(S_full, n0 + kadd, SIG_P, SIG_N)
    resel = {}
    for k in ks:
        sel = np.asarray(g["order"][:n0 + k], dtype=np.int64)
        mk = _k_metrics(S_full, fixed, sel, mask149, struct, atol)
        mk["k"] = k
        resel[k] = mk
    # 与主线"从零重选 k=|S0|"对照（论文口径：k=40 找回 52、丢 38）
    same_k = _k_metrics(S_full, fixed, np.asarray(g["order"][:n0]), mask149,
                        struct, atol)

    # ---- 汇总表 ----
    print(f"\n---- 增设结果（S0={n0} 固定 + k；目标集 {n149} 根）----")
    hdr = (f"{'k':>4} | {'目标':>6} | {'找回':>5} {'剩余':>5} {'丢失':>4} "
           f"{'unob':>5} {'可辨识':>6} {'秩':>4} | {'Δf':>9} {'证书':>6} "
           f"{'CRLB_ident':>11} {'bayes_tr':>10} | {'目标集后验std中位':>10} "
           f"{'≤σp/2':>6}")
    print(hdr)
    for k in ks:
        for obj in ("dopt", "cover"):
            r = runs[obj]["per_k"][k]
            print(f"{k:>4} | {obj:>6} | {r['n_recovered']:>5} "
                  f"{r['n_target_left']:>5} {r['n_lost']:>4} "
                  f"{r['n_unobservable']:>5} {r['n_identifiable']:>6} "
                  f"{r['rank']:>4} | {r['df']:>9.2f} {r['cert_ratio']:>6.3f} "
                  f"{r['crlb_ident']:>11.3e} {r['bayes_trace']:>10.1f} | "
                  f"{_g3(r['post_std_target']['median']):>10} "
                  f"{r['post_std_target']['n_half_prior']:>6}")
        r = resel[k]
        print(f"{k:>4} | {'重选':>6} | {r['n_recovered']:>5} "
              f"{r['n_target_left']:>5} {r['n_lost']:>4} "
              f"{r['n_unobservable']:>5} {r['n_identifiable']:>6} "
              f"{r['rank']:>4} | {r['f'] - base['f']:>9.2f} {'-':>6} "
              f"{r['crlb_ident']:>11.3e} {r['bayes_trace']:>10.1f} | "
              f"{'(重选 |S0|+k 个，非增设)':>10}")
    print(f"对照（从零重选 |S0|={n0} 个，主线口径）：找回 {same_k['n_recovered']}，"
          f"丢失 {same_k['n_lost']}，unobservable {same_k['n_unobservable']}")
    dk = runs["dopt"]["per_k"][ks[-1]]
    ck = runs["cover"]["per_k"][ks[-1]]
    print(f"两目标差异（k={ks[-1]}）：找回 D-opt {dk['n_recovered']} vs 覆盖 "
          f"{ck['n_recovered']}；Δf {dk['df']:.2f} vs {ck['df']:.2f}；"
          f"CRLB_ident {dk['crlb_ident']:.3e} vs {ck['crlb_ident']:.3e}")

    # ---- 落盘：主 JSON（含候选下标）、orders npz、可读版（零编号）----
    cfg = dict(s0=s0_spec, n_fixed=n0, ks=ks, ks_requested=ks_req, kadd=kadd,
               atol=atol, sigma_prior=SIG_P, sigma_noise=SIG_N, n_frames=T,
               n_candidates=m, P=P, n_structural=int(struct.sum()),
               n_target=n149, target_note=target_note,
               criterion="管可辨识 ⇔ max_{t,i∈S}|∂p_i/∂C_k| > atol 且非结构性"
                         "（与 dgga.calib.identifiability 普查同判据）；"
                         "σ 口径 = 贝叶斯后验 std sqrt(diag(M^-1))，"
                         "M = I/σ_prior² + Σ S Sᵀ/σ_noise²",
               mode="virtual augmentation：在标定后模型上模拟新传感器"
                    "（设计与模拟验证，非现场实装）")
    res = load_json(stem)
    res["augment"] = dict(config=cfg, fixed=fixed.tolist(), s0=base,
                          dopt=runs["dopt"], cover=runs["cover"],
                          reselect=dict(per_k=resel, same_k_as_s0=same_k))
    save_json(stem, res)
    ords["augment_fixed"] = fixed
    ords["augment_dopt"] = np.asarray(runs["dopt"]["added"], dtype=np.int64)
    ords["augment_cover"] = np.asarray(runs["cover"]["added"], dtype=np.int64)
    np.savez_compressed(orders_path(stem), **ords)

    def _strip(d):
        return {k: v for k, v in d.items() if k not in ("added", "fixed")}
    readable = dict(
        stem=stem, config=cfg,
        s0=base,
        augment={obj: dict(per_k={str(k): runs[obj]["per_k"][k] for k in ks},
                           ident_curve=runs[obj]["ident_curve"],
                           recovered_every_k=runs[obj]["recovered_every_k"],
                           k_recover80=runs[obj]["k_recover80"],
                           k_recover_all=runs[obj]["k_recover_all"],
                           cover_gains=runs[obj]["cover_gains"],
                           cover_saturated_at=runs[obj]["cover_saturated_at"],
                           f0=runs[obj]["f0"], f_curve=runs[obj]["f_curve"],
                           n_evals=runs[obj]["n_evals"],
                           t_total_sec=runs[obj]["t_total_sec"])
                 for obj in ("dopt", "cover")},
        reselect=dict(per_k={str(k): resel[k] for k in ks},
                      same_k_as_s0=same_k),
        monotonicity="bayes_dopt_augment 逐步断言：可辨识集只增不减、"
                     "diag(M^-1) 逐管不增；n_lost 恒 0（均通过，否则本文件不会生成）",
        note="不含任何节点/候选编号；选点序列在 data/placement_orders_<stem>.npz"
             "（augment_fixed / augment_dopt / augment_cover）")
    with open(augment_readable_path(stem), "w", encoding="utf-8") as f:
        json.dump(readable, f, ensure_ascii=False, indent=1)
    print(f"stage augment 完成：{json_path(stem)}['augment']、"
          f"{orders_path(stem)}（augment_*）、可读版 {augment_readable_path(stem)}")


# ======================================================================
# stage report
# ======================================================================
def stage_report(stem):
    res = load_json(stem)
    print(f"===== {stem} 布点实验汇总（data/placement_{stem}.json）=====")
    cfg = res.get("config", {})
    print(f"帧数 {cfg.get('n_frames')}，候选 {cfg.get('n_candidates')}，"
          f"P={cfg.get('P')}，σ_prior={cfg.get('sigma_prior')}，"
          f"σ_noise={cfg.get('sigma_noise')} ft")
    fs = res.get("full_sensitivity", {})
    print(f"S_full 总耗时 {fs.get('t_total_sec', 0):.2f}s"
          f"（splu {fs.get('n_factorizations')} 次 / RHS {fs.get('n_rhs')} 列）")
    d = res.get("dopt", {})
    if d:
        print(f"D-opt：证书最差 ratio {d.get('worst_cert_ratio'):.4f}；"
              f"子模抽查违例 {d.get('submodularity', {}).get('n_violations')}"
              f"/200；惰性评估 {d.get('n_lazy_evals')} 次")
    k = res.get("key_numbers", {})
    if k:
        print(f"D-opt k=40 unobservable {k['dopt_k40']['n_unobservable']} vs "
              f"随机中位 {k['random_k40']['n_unobservable']:.1f}；"
              f"80% 抢回所需 k = {k.get('k_recover80')}")
    rb = res.get("robust", {}).get("summary")
    if rb:
        print(f"鲁棒性：f 相对差中位 {rb['median_rel_gap'] * 100:.2f}%")
    au = res.get("augment")
    if au:
        c = au["config"]
        print(f"增设（S0={c['s0']}，|S0|={c['n_fixed']}，目标集 {c['n_target']}）：")
        for obj in ("dopt", "cover"):
            r = au[obj]
            row = "  ".join(f"k={k}:{v['n_recovered']}/{v['n_target']}"
                            f"(unob {v['n_unobservable']})"
                            for k, v in r["per_k"].items())
            print(f"  {obj:5s} {row}；80% 找回 k={r['k_recover80']}，"
                  f"全部 k={r['k_recover_all']}")
        sk = au["reselect"]["same_k_as_s0"]
        print(f"  对照 从零重选 |S0| 个：找回 {sk['n_recovered']}，丢失 "
              f"{sk['n_lost']}")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="city_d")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--kmax", type=int, default=80)
    ap.add_argument("--k-step", type=int, default=5)
    ap.add_argument("--stage", default="all",
                    choices=["all", "full", "dopt", "baselines", "metrics",
                             "robust", "augment", "report"])
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--n-robust", type=int, default=20)
    ap.add_argument("--k-robust", type=int, default=40)
    ap.add_argument("--robust-budget", type=float, default=480.0)
    ap.add_argument("--augment-ks", default="5,10,20,40,80",
                    help="增设个数网格（逗号分隔）")
    ap.add_argument("--s0", default="ga40",
                    help="增设的现有传感器集：ga40 | random:<n>:<seed> | dopt:<n>")
    ap.add_argument("--augment-atol", type=float, default=0.0,
                    help="普查零列阈值（0 = 解析恒零，与 identifiability 一致）")
    a = ap.parse_args()

    stages = ([a.stage] if a.stage != "all"
              else ["full", "dopt", "baselines", "metrics", "robust",
                    "report"])
    for st in stages:
        if st == "full":
            stage_full(a.stem, a.frames)
        elif st == "dopt":
            stage_dopt(a.stem, a.kmax, a.k_step)
        elif st == "baselines":
            stage_baselines(a.stem, a.kmax, a.n_random)
        elif st == "metrics":
            stage_metrics(a.stem, a.kmax, a.k_step)
        elif st == "robust":
            stage_robust(a.stem, a.n_robust, a.k_robust, a.robust_budget)
        elif st == "augment":
            stage_augment(a.stem, [int(x) for x in a.augment_ks.split(",")],
                          a.s0, a.augment_atol)
        elif st == "report":
            stage_report(a.stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())

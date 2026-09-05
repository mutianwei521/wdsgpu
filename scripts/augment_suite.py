# -*- coding: utf-8 -*-
"""augment_suite.py - 增设（augment）全套处方式实验驱动：City D 主线 + Hanoi 可复现版。

通讯作者的问题："既然标不出，能不能想办法解决 - 真正地设置上传感器。"
本脚本把增设选点（place_sensors.py --stage augment / augment_public.py hanoi-augment：
S0 = "现有"传感器一个不动，再加 k 个；两种目标 D-最优 / 覆盖）接到下游两条链上，
并如实报告结果。所有阶段按 --net 选网：

  --net city_d     S0 = 普查用的 40 个现有传感器（seed=2026；149 根传感不足管的口径），
                   k∈{5,10,20,40,80}；标定与工单漏损案例都在 City D 上重跑。
  --net pub_hanoi  S0 = random:10:2026（augment_public.py 同 S0），25 合成帧
                   （calibrate.py 同帧），k∈{5,10,20}；无工单案例（leak 阶段不适用）。

阶段（--stage，可逗号连写按序执行）：
  verify  从灵敏度缓存 + 选点序列独立复算增设曲线（两种目标）：目标集里找回多少、
          新的不可辨识集、CRLB（可辨识子空间 + 良态子空间 σ_i>1e-2σ_1 两种口径）、
          贝叶斯后验 std，并逐步核对单调性（可辨识集只增、后验方差只降、丢失恒 0）；
          与 data/placement_augment_<stem>.json 交叉核对（不一致即报错）。
  calib   用 S0 与 S0∪S_k 重跑 σ 梯标定（σ∈{0.03,0.1,0.3}，calibrate.py 同引擎、
          同真值、同噪声种子，按节点生成的噪声场换布点时同一实现）；
          落到 data/calib_gc1_<stem>.json["runs"]["AUG_*"]。
  leak    （仅 City D）用增设后的传感配置重跑工单漏损案例（demo_leak_inversion.py
          同真值三漏点、同反演器；噪声 0.1 ft），报 top-1/top-3 命中与签名相干
          （双分母口径：全对 / 非精确正交对）变化；demo40 配置 = demo 原 40 个传感器
          + demo 原噪声实现，作为逐位复现对照。落到 data/leak_augment_<net>.json。
  report  汇总为 data/augment_suite_<net>.json（可读版）与 data/augment_<net>_wip.txt
          （--machine 一句话记录跑 calib/leak 的机器与 GPU 使用情况，写入 meta）。
  fig     data/fig_augment_<net>.png。

零编号守卫：可读 JSON 落盘前扫描全部字典键与字符串值，命中任何节点/链路编号即报错
（结构键 k / σ 除外）；选点序列只在 data/placement_orders_<stem>.npz。
全程"虚拟增设"：在标定后的模型上模拟新传感器，是设计与模拟验证，不是现场实装。

运行：& python -X utf8 scripts/augment_suite.py --net city_d --stage verify,report,fig
      & python -X utf8 scripts/augment_suite.py --net city_d --stage calib
      & python -X utf8 scripts/augment_suite.py --net city_d --stage leak
      & python -X utf8 scripts/augment_suite.py --net pub_hanoi --stage verify,calib,report,fig
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
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DATA = os.path.join(ROOT, "data")
SIG_P, SIG_N = 15.0, 0.1          # 与 place_sensors.py / augment_public.py 同
SUB_TOL = 1e-2                    # 与 calibrate.py 的良态子空间截断同
CALIB_SIGMAS = [0.1, 0.03, 0.3]
CALIB_SEED = 100                  # calibrate.py NOISE_SEED0（桥接实验同种子）
ORTH_TOL = 1e-12                  # "精确正交对"判据（与 augment_public.coherence_stats 同）

NETS = {
    "city_d": dict(
        calib_key="city_d", ref_stem="city_d", stem="city_d", s0="ga40",
        s0_spec="ga40", ks=[5, 10, 20, 40, 80],
        cache="placement_cache_city_d.npz", orders="placement_orders_city_d.npz",
        aug_json="placement_augment_city_d.json", calib_json="calib_gc1_city_d.json",
        calib_placements=["ga40", "augdopt20", "augdopt40", "augcover20", "augcover40"],
        leak=True,
        leak_configs=["demo40", "ga40", "augcover20", "augcover40", "augdopt20",
                      "augdopt40", "augcover80", "augdopt80"],
        s0_note="S0 = 普查用的 40 个现有传感器（seed=2026 直接抽样，149 根传感不足管的"
                "口径）；demo 原 40 个 = 同种子多消耗一次抽样所得"),
    "pub_hanoi": dict(
        calib_key="hanoi", ref_stem="pub_hanoi", stem="pub_hanoi_synth25", s0="augS0",
        s0_spec="random:10:2026", ks=[5, 10, 20],
        cache=None, orders="placement_orders_pub_hanoi_synth25.npz",
        aug_json="placement_augment_pub_hanoi_synth25.json",
        calib_json="calib_gc1_pub_hanoi.json",
        calib_placements=["augS0", "augdopt5", "augdopt10", "augdopt20",
                          "augcover5", "augcover10", "augcover20"],
        leak=False, leak_configs=[],
        s0_note="S0 = random:10:2026（augment_public.py hanoi-augment 同 S0）；"
                "帧 = calibrate.py 的 25 个合成工况（SYNTH_SEED=4242）"),
}


# ----------------------------------------------------------------------
def paths(net):
    cfg = NETS[net]
    return dict(
        suite=os.path.join(DATA, f"augment_suite_{net}.json"),
        wip=os.path.join(DATA, f"augment_{net}_wip.txt"),
        leak=os.path.join(DATA, f"leak_augment_{net}.json"),
        fig=os.path.join(DATA, f"fig_augment_{net}.png"),
        orders=os.path.join(DATA, cfg["orders"]),
        cache=os.path.join(DATA, cfg["cache"]) if cfg["cache"] else None,
        aug_json=os.path.join(DATA, cfg["aug_json"]),
        calib_json=os.path.join(DATA, cfg["calib_json"]),
        demo_json=os.path.join(DATA, "demo_leak_inversion.json"))


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


def _stats(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)), mean=float(v.mean()),
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                max=float(v.max()), rmse=float(np.sqrt(np.mean(v ** 2))))


def _med(d):
    return d.get("median", float("nan"))


# ----------------------------------------------------------------------
# 零编号守卫
# ----------------------------------------------------------------------
def id_set(net):
    """该网全部节点 + 链路编号（字符串）。"""
    from dgga.parse import Net
    n = Net.load(os.path.join(DATA, "reference"), NETS[net]["ref_stem"])
    return set(str(x) for x in n.node_id) | set(str(x) for x in n.link_id)


def assert_no_ids(obj, ids, allow, what):
    """可读输出零编号守卫：任何 dict 键（结构键 allow 除外）或字符串值等于某个节点/链路
    编号即抛错。数值不查（计数与编号无法区分，也不构成指纹）。"""
    hits = []

    def walk(x, p):
        if isinstance(x, dict):
            for k, v in x.items():
                ks = str(k)
                if ks not in allow and ks in ids:
                    hits.append(f"{p}.{ks}")
                walk(v, f"{p}.{ks}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, f"{p}[{i}]")
        elif isinstance(x, str) and x in ids:
            hits.append(f"{p}={x!r}")

    walk(obj, "$")
    if hits:
        raise RuntimeError(f"{what} 含节点/链路编号 {len(hits)} 处（如 {hits[:5]}）")


def allow_keys(net):
    cfg = NETS[net]
    return ({str(k) for k in cfg["ks"]} | {f"{s:g}" for s in CALIB_SIGMAS}
            | {str(s) for s in CALIB_SIGMAS})


# ----------------------------------------------------------------------
# 案例装载：S_full / 结构掩码 / 目标集 / S0 / 选点序列
# ----------------------------------------------------------------------
def load_case(net):
    from dgga.placement import recovery_report
    cfg, p = NETS[net], paths(net)
    o = dict(np.load(p["orders"]))
    fixed = np.asarray(o["augment_fixed"], dtype=np.int64)
    for obj in ("dopt", "cover"):
        if len(set(o[f"augment_{obj}"].tolist()) & set(fixed.tolist())):
            raise RuntimeError(f"{obj} 增设序列与 S0 重叠")
    target_ref = None
    if p["cache"]:
        z = np.load(p["cache"])
        S_full, struct = z["S_full"], z["dead"] | z["cm25"]
        junc, pidx = z["junc"], z["pipe_idx"]
        if not np.array_equal(np.sort(fixed), np.sort(z["pos40"])):
            raise RuntimeError("augment_fixed ≠ 普查的现有传感器（pos40）")
        target_ref = z["mask149"]
        frames_note = "pattern_24h_25f"
    else:
        import calibrate as cb
        from dgga.placement import pipe_param_idx
        from dgga.sensitivity import sensitivity_matrix
        pb = cb.Problem(cfg["calib_key"])
        junc = np.asarray(pb.s.junc_nodes)
        pidx = pipe_param_idx(pb.s)
        T = pb.d.shape[0]
        t0 = time.perf_counter()
        S = sensitivity_matrix(pb.s, pb.d, pb.rh, None, junc, wrt="C",
                               accuracy=1e-12, max_iter=200, polish_steps=3)
        S_full = S.reshape(T, junc.size, pb.s.L)[:, :, pidx].copy()
        struct = np.asarray(pb.struct_mask[pidx], dtype=bool)
        print(f"  [{net}] S_full {S_full.shape} 现算 {time.perf_counter() - t0:.1f}s"
              f"（帧 {pb.frames_note}）")
        _, n, seed = cfg["s0_spec"].split(":")
        exp = np.sort(np.random.default_rng(int(seed)).choice(
            junc.size, size=int(n), replace=False))
        if not np.array_equal(np.sort(fixed), exp):
            raise RuntimeError(f"augment_fixed ≠ {cfg['s0_spec']}")
        frames_note = pb.frames_note
    P = S_full.shape[2]
    unob0 = recovery_report(S_full, fixed, fixed, np.zeros(P, bool), struct,
                            0.0)["unobservable_mask"]
    if target_ref is not None and not np.array_equal(unob0, target_ref):
        raise RuntimeError("S0 的传感不足集与缓存 mask149 不一致")
    return dict(net=net, cfg=cfg, S_full=S_full, struct=struct, target=unob0,
                fixed=fixed, junc=np.asarray(junc), pidx=np.asarray(pidx),
                orders=o, frames_note=frames_note, ks=list(cfg["ks"]))


def config_cand(name, case):
    """配置名 → 候选下标（S_full 第二维；与 calibrate.get_sensors 同规则）。"""
    o, fixed, s0 = case["orders"], case["fixed"], case["cfg"]["s0"]
    if name == s0:
        return np.unique(fixed)
    for obj, pre in (("dopt", "augdopt"), ("cover", "augcover")):
        if name.startswith(pre):
            return np.unique(np.r_[fixed, o[f"augment_{obj}"][:int(name[len(pre):])]])
    raise ValueError(name)


# ======================================================================
# stage verify
# ======================================================================
def stage_verify(net):
    from dgga.placement import eval_subset, posterior_std, recovery_report
    print(f"===== verify[{net}]：独立复算增设曲线并核对单调性 =====")
    case = load_case(net)
    S_full, struct, mask_t, fixed, o, ks = (case["S_full"], case["struct"],
                                            case["target"], case["fixed"],
                                            case["orders"], case["ks"])
    T, m, P = S_full.shape
    n_t = int(mask_t.sum())
    colmax_c = np.max(np.abs(S_full), axis=0)                      # [m, P]
    ref = jload(paths(net)["aug_json"])
    if not ref:
        raise RuntimeError(f"缺 {paths(net)['aug_json']}（先跑增设选点）")
    print(f"T={T} 候选 {m} P={P} 结构性 {int(struct.sum())}；S0={fixed.size}，"
          f"目标集（S0 下传感不足）{n_t}")

    def sub_crlb(sel):
        """良态子空间 CRLB：σ_i > SUB_TOL·σ_1 的方向上 Σ σ_n²/σ_i²，及其维数。"""
        Ssub = S_full[:, sel, :].reshape(T * len(sel), P)
        sv = np.linalg.svd(Ssub, compute_uv=False)
        k = int(np.sum(sv > SUB_TOL * sv[0]))
        return dict(k_sub=k, crlb_sub=float(np.sum(SIG_N ** 2 / sv[:k] ** 2)),
                    sv1=float(sv[0]), sv_ksub=float(sv[k - 1]))

    out = dict(n_target=n_t, n_fixed=int(fixed.size), ks=ks, P=int(P),
               n_structural=int(struct.sum()), n_candidates=int(m), T=int(T),
               frames=case["frames_note"], s0=case["cfg"]["s0"],
               sigma_prior=SIG_P, sigma_noise=SIG_N, sub_tol=SUB_TOL)
    ev0 = eval_subset(S_full, fixed, SIG_P, SIG_N)
    ps0 = posterior_std(S_full, fixed, SIG_P, SIG_N)
    rep0 = recovery_report(S_full, fixed, fixed, mask_t, struct, 0.0)
    out["s0_metrics"] = dict(
        n_sensors=int(fixed.size), n_identifiable=rep0["n_ident_after"],
        n_unobservable=rep0["n_unobservable_after"], rank=ev0["rank"], f=ev0["f"],
        crlb_ident=ev0["crlb_ident"], bayes_trace=ev0["bayes_trace"], **sub_crlb(fixed),
        post_std_target=_stats(ps0[mask_t]), post_std_ident=_stats(ps0[rep0["ident_after"]]))
    if rep0["n_unobservable_after"] != ref["s0"]["n_unobservable"]:
        raise RuntimeError("S0 的 unobservable 与 placement_augment json 不一致")

    for obj in ("dopt", "cover"):
        order = np.asarray(o[f"augment_{obj}"], dtype=np.int64)
        colmax = colmax_c[fixed].max(axis=0)
        ident = (colmax > 0.0) & ~struct
        pv = ps0 ** 2
        rec_curve, unob_curve, n_viol = [], [], 0
        for i in order:
            colmax = np.maximum(colmax, colmax_c[i])
            ident_new = (colmax > 0.0) & ~struct
            if np.any(ident & ~ident_new):
                n_viol += 1
            ident = ident_new
            rec_curve.append(int((ident & mask_t).sum()))
            unob_curve.append(int((~ident & ~struct).sum()))
        per_k = {}
        for k in ks:
            sel = np.r_[fixed, order[:k]]
            rep = recovery_report(S_full, fixed, sel, mask_t, struct, 0.0)
            ev = eval_subset(S_full, sel, SIG_P, SIG_N)
            ps = posterior_std(S_full, sel, SIG_P, SIG_N)
            pv_new = ps ** 2
            if np.any(pv_new > pv * (1 + 1e-9)):
                n_viol += 1
            pv = pv_new
            rm = rep["recovered_mask"]
            r = ref["augment"][obj]["per_k"][str(k)]
            if (rep["n_recovered"] != r["n_recovered"]
                    or rep["n_unobservable_after"] != r["n_unobservable"]
                    or rep["n_lost"] != 0 or r["n_lost"] != 0):
                raise RuntimeError(f"{obj} k={k} 与 placement_augment json 不一致")
            sc = sub_crlb(sel)
            per_k[str(k)] = dict(
                k=k, n_sensors=int(np.unique(sel).size),
                n_recovered=rep["n_recovered"], n_target_left=rep["n_target_left"],
                n_lost=rep["n_lost"], n_unobservable=rep["n_unobservable_after"],
                n_identifiable=rep["n_ident_after"], rank=ev["rank"], f=ev["f"],
                df=ev["f"] - ev0["f"], crlb_ident=ev["crlb_ident"],
                bayes_trace=ev["bayes_trace"], **sc,
                post_std_recovered=_stats(ps[rm]), post_std_target=_stats(ps[mask_t]),
                post_std_ident=_stats(ps[rep["ident_after"]]),
                n_recovered_half_prior=int(np.sum(ps[rm] <= 0.5 * SIG_P)),
                n_recovered_std_le5=int(np.sum(ps[rm] <= 5.0)),
                cert_ratio=r.get("cert_ratio"), cover_ratio=r.get("cover_ratio"),
                recovered_every_k_check=rec_curve[k - 1])
            print(f"  {obj:5s} k=+{k:2d} 找回 {rep['n_recovered']:3d}/{n_t} "
                  f"未找回 {rep['n_target_left']:3d} 丢失 {rep['n_lost']} "
                  f"unob {rep['n_unobservable_after']:3d} 秩 {ev['rank']:3d} "
                  f"k_sub {sc['k_sub']:3d} CRLB_ident {ev['crlb_ident']:.2e} "
                  f"CRLB_sub {sc['crlb_sub']:.2e} 后验std(找回)中位 "
                  f"{_med(per_k[str(k)]['post_std_recovered']):.2f} "
                  f"≤σp/2 {per_k[str(k)]['n_recovered_half_prior']}")
        rc = np.asarray(rec_curve)
        t80 = 0.8 * n_t
        out[obj] = dict(per_k=per_k, recovered_every_k=rec_curve,
                        unobservable_every_k=unob_curve,
                        k_recover80=int(np.argmax(rc >= t80) + 1) if n_t and np.any(rc >= t80) else -1,
                        k_recover_all=int(np.argmax(rc >= n_t) + 1) if n_t and np.any(rc >= n_t) else -1,
                        monotonic_violations=int(n_viol),
                        cover_saturated_at=ref["augment"][obj]["cover_saturated_at"])
        if n_viol:
            raise RuntimeError(f"{obj} 单调性违例 {n_viol}")
        print(f"  {obj}: 找回 80% 需 +{out[obj]['k_recover80']}，全部需 "
              f"+{out[obj]['k_recover_all']}（-1=网格内未达）；单调性违例 0")
    out["reselect"] = {k: dict(n_recovered=v["n_recovered"], n_lost=v["n_lost"],
                               n_unobservable=v["n_unobservable"])
                       for k, v in ref["reselect"]["per_k"].items()}
    out["reselect_same_k"] = dict(
        n_recovered=ref["reselect"]["same_k_as_s0"]["n_recovered"],
        n_lost=ref["reselect"]["same_k_as_s0"]["n_lost"])
    st = jload(paths(net)["suite"])
    st["augment_curve"] = out
    jdump(st, paths(net)["suite"])
    print("verify 完成。")


# ======================================================================
# stage calib
# ======================================================================
def stage_calib(net, placements, sigmas, seeds, lam=1e-4):
    import calibrate as cb
    cfg, p = NETS[net], paths(net)
    print(f"===== calib[{net}]：增设配置 σ 梯标定 {placements} × σ{sigmas} × 种子{seeds} =====")
    pb = cb.Problem(cfg["calib_key"])
    case = load_case(net)
    junc = case["junc"]
    aug = jload(p["aug_json"])
    ref_unob = {cfg["s0"]: aug["s0"]["n_unobservable"]}
    for obj in ("dopt", "cover"):
        for k, v in aug["augment"][obj]["per_k"].items():
            ref_unob[f"aug{obj}{k}"] = v["n_unobservable"]
    data = jload(p["calib_json"])
    done = set(data.get("runs", {}).keys())
    for sigma in sigmas:
        for pl in placements:
            for sd in seeds:
                key = f"AUG_{pl}_s{sigma:g}_n{sd}"
                if key in done:
                    print(f"  跳过（已存在）{key}")
                    continue
                sens = cb.get_sensors(pb, pl)
                exp = junc[config_cand(pl, case)]
                if not np.array_equal(np.sort(sens), np.sort(exp)):
                    raise RuntimeError(f"{pl} 传感器集与 orders npz 不一致")
                t0 = time.perf_counter()
                rec = cb.run_config(pb, key, sigma=sigma, noise_seed=sd, lam=lam,
                                    placement=pl,
                                    note=f"虚拟增设：{cfg['s0']} 固定+增设 k；同真值同噪声种子；"
                                         f"host={platform.node()}")
                if pl in ref_unob and rec["n_unobservable"] != ref_unob[pl]:
                    print(f"  [注] {pl} identifiability unobservable={rec['n_unobservable']}"
                          f" ≠ 布点口径 {ref_unob[pl]}（calibrate 的结构掩码含关闭管/"
                          f"margin=1 钳位，与布点缓存的 dead|cm25 口径不同；如实并列）")
                cb.merge_save(pb, [rec])
                print(f"  {key} 完成 {time.perf_counter() - t0:.0f}s")
    print("calib 完成。")


# ======================================================================
# stage leak（仅 City D 工单案例）
# ======================================================================
def coh_stats(coh, nc, true_idx, label_of):
    """签名字典互相干摘要，双分母口径：全部候选对 / 非精确正交对（|coh|≥1e-12）。"""
    A = np.abs(coh)
    iu, ju = np.triu_indices(nc, 1)
    off = A[iu, ju]
    orth = off < ORTH_TOL
    within = off[~orth]
    B = A.copy()
    np.fill_diagonal(B, -1.0)
    return dict(
        n_pairs=int(off.size), n_orthogonal_pairs=int(orth.sum()),
        n_pairs_within=int(within.size),
        max_offdiag=float(off.max()),
        median_all=float(np.median(off)),
        median_within=float(np.median(within)) if within.size else None,
        q90_all=float(np.quantile(off, 0.9)),
        n_pairs_gt_0999=int((off > 0.999).sum()),
        n_pairs_gt_099=int((off > 0.99).sum()),
        n_pairs_gt_09=int((off > 0.9).sum()),
        true_node_max_rival_coh={label_of[j]: float(B[j].max()) for j in true_idx},
        true_node_n_rivals_gt_099={label_of[j]: int((B[j] > 0.99).sum()) for j in true_idx},
        true_pair_coh={f"{label_of[a]}-{label_of[b]}": float(coh[a, b])
                       for i, a in enumerate(true_idx) for b in true_idx[i + 1:]})


def demo_truth(pb, dm):
    """demo_leak_inversion.py 的真值重放：与 demo 逐位同一 rng 序列（第三真值节点、
    demo 原 40 传感器）。返回 (targets{node: LPS}, sens_demo, label{node: T1..T3})。"""
    rng = np.random.default_rng(dm.SEED)
    others = [n for n in pb.cand if n not in ("195", "3083")]
    third = str(rng.choice(others))
    sens_demo = np.sort(rng.choice(pb.s.junc_nodes, dm.N_SENSOR, replace=False))
    targets = {"195": 3.0, "3083": 1.5, third: 2.2}
    label = {n: f"T{i + 1}" for i, n in enumerate(targets)}      # 零编号标签
    return targets, sens_demo, label


def leak_footprint(net, configs):
    """每个真值漏点**单独**存在时的压力足迹：25 帧 × 该配置传感器上 |Δh| 的 RMS 与 max（ft），
    与 0.1 ft 噪声对照；另给全部 junction 上的足迹作上界。解释某漏点为何在某配置下
    不可能被找回（足迹低于噪声）或为何增设后被找回。只报 T1/T2/T3 标签。"""
    import torch
    import demo_leak_inversion as dm
    from dgga.units import MperFT
    torch.set_default_dtype(torch.float64)
    pb = dm.Problem()
    case = load_case(net)
    junc = case["junc"]
    targets, sens_demo, label = demo_truth(pb, dm)
    sol_base = pb.fsolve(max_iter=dm.OBS_MI)
    dh = {}
    for n, q in targets.items():
        i = pb.node_index[n]
        p_m = (sol_base["head"][:, i] - pb.net.elev_ft[i]).mean() * MperFT
        ke = np.zeros(pb.net.N)
        ke[i] = pb.ke_int_of_C(q / p_m ** pb.gamma)
        dh[n] = pb.fsolve(ke=ke, max_iter=dm.OBS_MI)["head"] - sol_base["head"]   # [25, N]

    def fp(sens):
        return {label[n]: dict(rms_ft=float(np.sqrt((dh[n][:, sens] ** 2).mean())),
                               max_ft=float(np.abs(dh[n][:, sens]).max())) for n in targets}

    out = dict(noise_ft=dm.NOISE_FT, frames=25,
               note="足迹 = 该漏点单独存在时传感器压力相对基线的变化（ft）；rms 对 25 帧×传感器，"
                    "max 为逐帧逐传感器最大绝对值；all_junctions = 全部 junction 上的同口径上界",
               per_config={}, all_junctions=fp(np.asarray(pb.s.junc_nodes)))
    for name in configs:
        sens = sens_demo if name == "demo40" else np.sort(junc[config_cand(name, case)])
        out["per_config"][name] = dict(n_sensors=int(len(sens)), **fp(sens))
    return out


def stage_leak(net, configs, groups):
    cfg, p = NETS[net], paths(net)
    if not cfg["leak"]:
        raise SystemExit(f"leak 阶段仅对工单案例所在网（city_d）适用，--net {net} 不适用")
    import torch
    import demo_leak_inversion as dm
    from dgga.units import LPSperCFS, MperFT
    torch.set_default_dtype(torch.float64)
    print(f"===== leak[{net}]：增设配置下的工单漏损案例（噪声 {dm.NOISE_FT} ft）"
          f"{configs} × {groups} =====")
    t_all = time.time()
    pb = dm.Problem()
    case = load_case(net)
    junc, fixed = case["junc"], case["fixed"]
    # ---- 真值：与 demo 逐位同一 rng 序列（第三真值节点、demo 原 40 传感器）----
    targets, sens_demo, label = demo_truth(pb, dm)
    ref = jload(p["demo_json"])
    if not ref or set(targets) != set(ref["config"]["true_nodes"]):
        raise RuntimeError("真值节点重放与 demo 不一致")
    if set(ref["config"]["sensors"]) != set(pb.net.node_id[i] for i in sens_demo):
        raise RuntimeError("demo 原 40 传感器重放与 demo_leak_inversion.json 不一致")
    true_idx = [pb.cand.index(n) for n in targets]
    label_of = {pb.cand.index(n): label[n] for n in targets}
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
    for n in targets:
        if abs(true_lk[n] - ref["config"]["true_nodes"][n]["true_mean_lps"]) > 1e-9:
            raise RuntimeError("真值漏损流量重放与 demo 不一致")
    # 噪声：noisy = 按全节点生成（seed 909）再按传感器取列（各配置同一实现；
    # calibrate.make_obs 同法）；noisy_demo = demo 原样（按 [25,40] 形状抽样，
    # 只对 demo40 有意义，作逐位复现对照）
    noise_full = dm.NOISE_FT * np.random.default_rng(dm.SEED_NOISE).standard_normal(
        (25, pb.net.N))
    noise_demo = dm.NOISE_FT * np.random.default_rng(dm.SEED_NOISE).standard_normal(
        (25, dm.N_SENSOR))
    s0_nodes = set(junc[fixed].tolist())
    n_overlap = len(s0_nodes & set(sens_demo.tolist()))

    def sensors_of(name):
        if name == "demo40":
            return sens_demo
        return np.sort(junc[config_cand(name, case)])

    def kinds_of(node_ids):
        return sorted(label[n] if n in targets else "nontrue" for n in node_ids)

    res = jload(p["leak"])
    res.setdefault("config", dict(
        seed=dm.SEED, seed_noise=dm.SEED_NOISE, noise_ft=dm.NOISE_FT, frames=25,
        n_candidates=pb.nc, lam=1e-4, stage1_iters=dm.STAGE1_ITERS,
        true_leak_lps={label[n]: true_lk[n] for n in targets},
        n_s0=int(fixed.size), n_demo_sensors=int(sens_demo.size),
        n_overlap_demo40_s0=int(n_overlap),
        groups_note="noisy: 噪声场按全节点生成（seed=909）再按传感器取列，各配置同一实现；"
                    "noisy_demo: demo 原样按 [25,40] 抽样（seed=909），仅 demo40，"
                    "用于逐位复现 data/demo_leak_inversion.json 的噪声组；"
                    "noiseless: 无噪声（demo 无噪声组同口径）",
        s0_note=cfg["s0_note"],
        mode="virtual augmentation：模拟新传感器，设计与模拟验证，非现场实装"))
    res.setdefault("groups", {})

    for name in configs:
        for grp in groups:
            if grp == "noisy_demo" and name != "demo40":
                continue
            gkey = f"{name}:{grp}"
            if gkey in res["groups"]:
                print(f"  跳过（已存在）{gkey}")
                continue
            t0 = time.time()
            n_fwd0 = dm.FWD_COUNT[0]
            sens = sensors_of(name)
            elev_s = torch.tensor(pb.net.elev_ft[sens])
            obs = sol_true["head"][:, sens] - pb.net.elev_ft[sens]
            if grp == "noisy":
                obs = obs + noise_full[:, sens]
            elif grp == "noisy_demo":
                obs = obs + noise_demo
            elif grp != "noiseless":
                raise ValueError(grp)
            pred0 = (sol_base["head"][:, sens] - pb.net.elev_ft[sens]).reshape(-1)
            dm._BASE_CACHE["pred0"] = pred0
            mse0 = float(((obs.reshape(-1) - pred0) ** 2).mean())
            D, Dn, coh = dm.build_dictionary(pb, sol_base, sens)
            cs = coh_stats(coh, pb.nc, true_idx, label_of)
            print(f"\n---- [{gkey}] 传感器 {len(sens)} 个；基线-观测 MSE={mse0:.3e}；"
                  f"相干 max={cs['max_offdiag']:.6f} 中位(全/非正交)={cs['median_all']:.4f}/"
                  f"{cs['median_within']:.4f} 正交对 {cs['n_orthogonal_pairs']} "
                  f">0.999 对数={cs['n_pairs_gt_0999']}；真值最强竞争者 coh="
                  f"{ {k: round(v, 6) for k, v in cs['true_node_max_rival_coh'].items()} } ----")
            obs_t = torch.tensor(obs)
            st1 = dm.stage1_adam_l1(pb, obs_t, sens, elev_s, 1e-4)
            st1_top3 = [int(j) for j in np.argsort(-st1["leak_lps"])[:3]]
            print(f"  阶段1 (L1, λ=1e-4): mse={st1['mse']:.3e} top3 命中 "
                  f"{len(set(st1_top3) & set(true_idx))}/3")
            support, fit = dm.stage2_support_search(pb, obs_t, sens, elev_s, D, Dn,
                                                    coh, mse0)
            support = [int(j) for j in support]
            order = [j for j in np.argsort(-fit["leak_lps"]) if j in support]
            top3 = order[:3]
            top1_hit = bool(top3 and top3[0] in true_idx)
            top3_hits = len(set(top3) & set(true_idx))
            exact = set(top3) == set(true_idx)
            flow_err = {}
            for n in targets:
                j = pb.cand.index(n)
                est = float(fit["leak_lps"][j]) if j in support else 0.0
                flow_err[label[n]] = dict(true_lps=true_lk[n], est_lps=est,
                                          rel_err=abs(est - true_lk[n]) / true_lk[n])
            sup_desc = []
            for j in support:
                if j in label_of:
                    sup_desc.append(dict(kind=label_of[j], leak_lps=float(fit["leak_lps"][j])))
                else:
                    cw = {label_of[t]: float(abs(coh[j, t])) for t in true_idx}
                    best = max(cw, key=cw.get)
                    sup_desc.append(dict(kind="nontrue", closest_true=best, coh=cw[best],
                                         leak_lps=float(fit["leak_lps"][j])))
            g = dict(config=name, group=grp, n_sensors=int(len(sens)),
                     n_sensors_in_s0=int(len(set(sens.tolist()) & s0_nodes)),
                     mse0=mse0, coherence=cs,
                     stage1=dict(mse=st1["mse"], top3_hits=len(set(st1_top3) & set(true_idx)),
                                 top1_hit=bool(st1_top3[0] in true_idx),
                                 top3_kinds=[label_of.get(j, "nontrue") for j in st1_top3]),
                     support_size=len(support), support=sup_desc,
                     top1_hit=top1_hit, top3_hits=top3_hits, top3_exact=bool(exact),
                     flow_err=flow_err,
                     max_flow_rel_err=float(max(v["rel_err"] for v in flow_err.values())),
                     resid_nontrue_final_lps=float(max(
                         [fit["leak_lps"][j] for j in support if j not in true_idx],
                         default=0.0)),
                     final_mse_ft2=fit["mse"], time_sec=time.time() - t0,
                     n_forward_solves=dm.FWD_COUNT[0] - n_fwd0, host=platform.node())
            if name == "demo40" and grp in ("noisy_demo", "noiseless"):
                rn = ref["groups"]["noisy" if grp == "noisy_demo" else "noiseless"]
                sup_nodes = set(pb.cand[j] for j in support)
                rel = (abs(fit["mse"] - rn["final_mse_ft2"])
                       / max(abs(rn["final_mse_ft2"]), 1e-300))
                g["reproduction"] = dict(
                    demo_group="noisy" if grp == "noisy_demo" else "noiseless",
                    support_same=bool(sup_nodes == set(rn["support"])),
                    support_kinds_demo=kinds_of(rn["support"]),
                    top3_hits_demo=len(set(rn["top3"]) & set(targets)),
                    top3_exact_demo=bool(rn["top3_exact"]),
                    final_mse_demo=rn["final_mse_ft2"], final_mse_rel_diff=float(rel),
                    matches=bool(sup_nodes == set(rn["support"]) and rel < 1e-6))
                print(f"  [复现对照] 与 demo_leak_inversion.json[{g['reproduction']['demo_group']}]"
                      f"：支撑相同={g['reproduction']['support_same']} "
                      f"final_mse 相对差={rel:.2e} → matches={g['reproduction']['matches']}")
            res["groups"][gkey] = g
            assert_no_ids(res, id_set(net), allow_keys(net), "leak json")
            jdump(res, p["leak"])
            print(f"[{gkey}] 支撑 {len(support)} 个：{[d['kind'] for d in sup_desc]}；"
                  f"top-1 命中={top1_hit} top-3 命中={top3_hits}/3 精确={exact}；"
                  f"最大流量相对误差={g['max_flow_rel_err']:.3f}；final_mse={fit['mse']:.3e}；"
                  f"{g['time_sec']:.0f}s，前向 {g['n_forward_solves']} 次")
    res["total_time_sec"] = res.get("total_time_sec", 0.0) + (time.time() - t_all)
    assert_no_ids(res, id_set(net), allow_keys(net), "leak json")
    jdump(res, p["leak"])
    print("leak 完成。")


# ======================================================================
# stage report
# ======================================================================
def calib_perpipe(net, case=None):
    """从 calib_gc1_<stem>.json 的 AUG_* 记录复算找回管的逐管参数误差。"""
    import calibrate as cb
    from dgga.placement import recovery_report, posterior_std
    cfg, p = NETS[net], paths(net)
    data = jload(p["calib_json"])
    runs = {k: v for k, v in data.get("runs", {}).items() if k.startswith("AUG_")}
    if not runs:
        return {}
    pb = cb.Problem(cfg["calib_key"])
    case = case or load_case(net)
    S_full, struct, mask_t, fixed, pidx = (case["S_full"], case["struct"], case["target"],
                                           case["fixed"], case["pidx"])
    if not np.array_equal(pidx, pb.pidx):
        raise RuntimeError("布点 pipe_idx 与 calibrate 的 pidx 不一致")
    free_pos = {int(g): j for j, g in enumerate(pb.free_idx)}
    C_true, _ = cb.make_truth(pb, "perpipe", cb.TRUTH_SEED["perpipe"])
    ident0 = recovery_report(S_full, fixed, fixed, mask_t, struct, 0.0)["ident_after"]
    out = {}
    for key, r in sorted(runs.items()):
        pl = r["placement"]
        try:
            sel = config_cand(pl, case)
        except ValueError:
            continue                                   # 非本套命名的记录，跳过
        rep = recovery_report(S_full, fixed, sel, mask_t, struct, 0.0)
        ps = posterior_std(S_full, sel, SIG_P, SIG_N)
        C_hat = np.asarray(r["C_hat_free"], dtype=np.float64)

        def errs(mask_P):
            g = pidx[mask_P]
            j = np.array([free_pos[int(x)] for x in g if int(x) in free_pos], dtype=np.int64)
            gg = np.array([int(x) for x in g if int(x) in free_pos], dtype=np.int64)
            return np.abs(C_hat[j] - C_true[gg]), np.abs(cb.C0 - C_true[gg])

        rm = rep["recovered_mask"]
        e_rec, e_rec_prior = errs(rm)
        e_id0, e_id0_prior = errs(ident0)
        e_left, e_left_prior = errs(mask_t & ~rm)
        e_t, e_t_prior = errs(mask_t)
        ps_rec = ps[rm]
        strong = rm.copy()
        strong[rm] = ps_rec <= 0.5 * SIG_P            # 找回管中 σ 口径也强的子集
        e_strong, e_strong_prior = errs(strong)
        e_weak, e_weak_prior = errs(rm & ~strong)
        out[key] = dict(
            placement=pl, sigma=r["sigma"], noise_seed=r["noise_seed"],
            n_sensors=r["n_sensors"], n_informative=r["n_informative"],
            n_unobservable=r["n_unobservable"], sub_rank=r.get("sub_rank"),
            info_rmse=r.get("info_rmse"), info_mae=r.get("info_mae"),
            sub_rmse=r.get("sub_rmse"), sub_rmse_prior=r.get("sub_rmse_prior"),
            val_frame_rmse=r["val_frame_rmse"], val_sensor_rmse=r["val_sensor_rmse"],
            train_rmse=r["train_rmse"], t_total_sec=r["t_total_sec"], nfe=r["nfe"],
            host=(r.get("note", "").split("host=")[-1] if "host=" in r.get("note", "") else ""),
            n_recovered=int(rm.sum()),
            recovered_err=_stats(e_rec), recovered_err_prior=_stats(e_rec_prior),
            recovered_n_le5=int(np.sum(e_rec <= 5.0)),
            recovered_n_le10=int(np.sum(e_rec <= 10.0)),
            recovered_n_better_than_prior=int(np.sum(e_rec < e_rec_prior)),
            recovered_strong_n=int(strong.sum()),
            recovered_strong_err=_stats(e_strong), recovered_strong_err_prior=_stats(e_strong_prior),
            recovered_weak_err=_stats(e_weak), recovered_weak_err_prior=_stats(e_weak_prior),
            target_err=_stats(e_t), target_err_prior=_stats(e_t_prior),
            target_n_le10=int(np.sum(e_t <= 10.0)),
            still_unobs_err=_stats(e_left), still_unobs_err_prior=_stats(e_left_prior),
            ident0_err=_stats(e_id0), ident0_err_prior=_stats(e_id0_prior),
            ident0_n_le10=int(np.sum(e_id0 <= 10.0)),
            post_std_recovered=_stats(ps_rec))
    return out


def stage_report(net, machine=""):
    cfg, p = NETS[net], paths(net)
    if "augment_curve" not in jload(p["suite"]):
        stage_verify(net)
    st = jload(p["suite"])
    ac = st["augment_curve"]
    case = load_case(net)
    cal = calib_perpipe(net, case)
    st["calibration"] = cal
    leak = jload(p["leak"]) if cfg["leak"] else {}
    st["leak"] = leak
    if leak.get("groups"):
        seen = list(dict.fromkeys(g["config"] for g in leak["groups"].values()))
        st["leak_footprint"] = leak_footprint(net, seen)
    # 机器/耗时台账：calib 与 leak 记录自带 host 与墙钟，这里只汇总；--machine 补一句
    # 人读的说明（哪台机器、GPU 是否用到），不写死在代码里
    hosts = sorted(({r.get("host", "") for r in cal.values()}
                    | {g.get("host", "") for g in leak.get("groups", {}).values()}) - {""})
    st["meta"] = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                      machine=machine or st.get("meta", {}).get("machine", ""),
                      run_hosts=hosts,
                      calib_total_sec=float(sum(r["t_total_sec"] for r in cal.values())),
                      leak_total_sec=float(sum(g["time_sec"] for g in leak.get("groups", {}).values())),
                      net=net, s0=cfg["s0"], s0_note=cfg["s0_note"],
                      mode="virtual augmentation：在标定后模型上模拟新传感器，"
                           "是设计与模拟验证，非现场实装",
                      sources=[f"data/{cfg['cache']}" if cfg["cache"] else
                               "S_full 现算（calibrate.Problem + dgga.sensitivity）",
                               f"data/{cfg['orders']}", f"data/{cfg['aug_json']}",
                               f"data/{cfg['calib_json']} (runs AUG_*)"]
                              + ([os.path.relpath(p["leak"], ROOT).replace(os.sep, "/"),
                                  "data/demo_leak_inversion.json (对照)"] if cfg["leak"] else []),
                      note="全部可读输出零节点/链路/传感器编号（落盘前守卫扫描）")
    assert_no_ids(st, id_set(net), allow_keys(net), "suite json")
    jdump(st, p["suite"])

    L = []
    n_t = ac["n_target"]
    ks = ac["ks"]
    L.append(f"[{net}] 增设（augment）全套实验 - 实测数值（虚拟增设：设计与模拟验证，非现场实装）")
    L.append(f"生成 {st['meta']['generated']} @ {st['meta']['host']}；出处见 "
             f"{os.path.relpath(p['suite'], ROOT)}[meta.sources]")
    if st["meta"]["machine"]:
        L.append(f"机器：{st['meta']['machine']}")
    L.append(f"实跑主机 {st['meta']['run_hosts']}；标定墙钟合计 {st['meta']['calib_total_sec']:.0f}s"
             f"（{len(cal)} 条），漏损案例墙钟合计 {st['meta']['leak_total_sec']:.0f}s"
             f"（{len(leak.get('groups', {}))} 组）；逐条耗时见下表末列")
    L.append("")
    L.append(f"一、增设曲线（S0 = {cfg['s0']}（{ac['n_fixed']} 个）固定不动；目标集 = S0 下传感不足的 "
             f"{n_t} 根管；普查判据 max|∂p/∂C|>0 且非结构性；σ_prior={SIG_P}，σ_noise={SIG_N} ft；"
             f"帧 {ac['frames']}）")
    s0 = ac["s0_metrics"]
    L.append(f"  S0：可辨识 {s0['n_identifiable']}，unobservable {s0['n_unobservable']}，"
             f"秩 {s0['rank']}，良态维数(σ_i>1e-2σ_1) {s0['k_sub']}，f={s0['f']:.2f}，"
             f"CRLB_ident={s0['crlb_ident']:.3e}，CRLB_sub={s0['crlb_sub']:.3e}，"
             f"bayes_trace={s0['bayes_trace']:.1f}")
    L.append(f"  {'+k':>3} | {'目标':>5} | {'找回':>4} {'未找回':>5} {'丢失':>4} {'unob':>4} "
             f"{'可辨识':>5} {'秩':>4} {'k_sub':>5} | {'Δf':>8} {'证书':>5} | {'CRLB_ident':>10} "
             f"{'CRLB_sub':>9} {'bayes_tr':>8} | {'找回管后验std中位':>10} {'≤σp/2':>5} {'≤5':>3}")
    for k in ks:
        for obj in ("dopt", "cover"):
            r = ac[obj]["per_k"][str(k)]
            cr = r["cert_ratio"] if r["cert_ratio"] is not None else float("nan")
            L.append(f"  {k:>3} | {obj:>5} | {r['n_recovered']:>4} {r['n_target_left']:>5} "
                     f"{r['n_lost']:>4} {r['n_unobservable']:>4} {r['n_identifiable']:>5} "
                     f"{r['rank']:>4} {r['k_sub']:>5} | {r['df']:>8.2f} {cr:>5.3f} | "
                     f"{r['crlb_ident']:>10.2e} {r['crlb_sub']:>9.2e} {r['bayes_trace']:>8.1f} | "
                     f"{_med(r['post_std_recovered']):>10.2f} "
                     f"{r['n_recovered_half_prior']:>5} {r['n_recovered_std_le5']:>3}")
        rs = ac["reselect"][str(k)]
        L.append(f"  {k:>3} | {'重选':>5} | {rs['n_recovered']:>4} {n_t - rs['n_recovered']:>5} "
                 f"{rs['n_lost']:>4} {rs['n_unobservable']:>4}   （从零重选 |S0|+k 个，非增设；对照）")
    L.append(f"  对照 从零重选 |S0| 个（主线口径）：找回 {ac['reselect_same_k']['n_recovered']}，"
             f"丢失 {ac['reselect_same_k']['n_lost']}")
    for obj in ("dopt", "cover"):
        L.append(f"  {obj}: 找回 80%（≥{0.8 * n_t:.1f}）最少增设 +{ac[obj]['k_recover80']}，"
                 f"全部找回 +{ac[obj]['k_recover_all']}（-1=网格内未达）；覆盖饱和步 "
                 f"{ac[obj]['cover_saturated_at']}；单调性逐步核对违例 {ac[obj]['monotonic_violations']}"
                 f"（可辨识集只增、后验方差只降、丢失恒 0）")
    L.append("  说明：CRLB_ident = 数值秩子空间 Σσ_n²/σ_i²（被 ~1e-13·σ_1 的名义方向主导，量级无意义，"
             "只作同口径对照）；CRLB_sub = 良态子空间（σ_i>1e-2σ_1，与 calibrate.py SUB_TOL 同）。")
    L.append("")

    # ---- 二、标定 ----
    L.append("二、σ 梯标定（calibrate.py 同引擎：perpipe 真值 seed=7、噪声按节点生成、λ=1e-4、"
             "20% 传感器留出；'找回管' = 目标集中在该配置下跨过普查阈值的管；"
             "误差 = |Ĉ−C_true|，先验误差 = |130−C_true|）")
    if cal:
        L.append(f"  {'配置':>11} {'σ':>5} {'种子':>4} | {'传感':>4} {'info':>4} {'unob':>4} {'k_sub':>5} | "
                 f"{'info C-RMSE':>11} {'sub C-RMSE':>10}{'(先验)':>8} {'valF':>7} {'valS':>7} | "
                 f"{'找回':>4} {'找回管|ΔC|中位':>10}{'(先验)':>7} {'≤10':>4} {'优于先验':>6} | "
                 f"{'强(std≤7.5)':>10} {'其|ΔC|中位':>9}{'(先验)':>7} | {'S0可辨识管中位':>8}{'(先验)':>7} | "
                 f"{'NFE':>4} {'秒':>5}")
        order = {pl: i for i, pl in enumerate(cfg["calib_placements"])}
        for key, r in sorted(cal.items(), key=lambda kv: (kv[1]["sigma"], order.get(kv[1]["placement"], 99), kv[1]["noise_seed"])):
            re_, rp_ = r["recovered_err"], r["recovered_err_prior"]
            se, sp = r["recovered_strong_err"], r["recovered_strong_err_prior"]
            L.append(f"  {r['placement']:>11} {r['sigma']:>5g} {r['noise_seed']:>4} | {r['n_sensors']:>4} "
                     f"{r['n_informative']:>4} {r['n_unobservable']:>4} {r['sub_rank']:>5} | "
                     f"{r['info_rmse']:>11.3f} {r['sub_rmse']:>10.3f}{r['sub_rmse_prior']:>8.2f} "
                     f"{r['val_frame_rmse']:>7.4f} {r['val_sensor_rmse']:>7.4f} | {r['n_recovered']:>4} "
                     f"{_med(re_):>10.2f}{_med(rp_):>7.2f} {r['recovered_n_le10']:>4} "
                     f"{r['recovered_n_better_than_prior']:>6} | {r['recovered_strong_n']:>10} "
                     f"{_med(se):>9.2f}{_med(sp):>7.2f} | {_med(r['ident0_err']):>8.2f}"
                     f"{_med(r['ident0_err_prior']):>7.2f} | {r['nfe']:>4} {r['t_total_sec']:>5.0f}")
    else:
        L.append("  （尚无 AUG_* 标定记录）")
    L.append("")

    # ---- 三、漏损 ----
    if cfg["leak"]:
        L.append("三、工单漏损案例（demo_leak_inversion.py 同真值三漏点、同反演器：Adam+L1 → "
                 "非线性 OMP + 互换抛光 + 留一消元；相干为双分母口径：中位(全对)/中位(非精确正交对)）")
        if leak.get("groups"):
            ref = jload(p["demo_json"])
            rn = ref["groups"]["noisy"]
            L.append(f"  对照（demo 原 40 传感器，噪声组，data/demo_leak_inversion.json）：支撑 "
                     f"{len(rn['support'])} 个，top-3 命中 "
                     f"{len(set(rn['top3']) & set(ref['config']['true_nodes']))}/3，"
                     f"精确 {rn['top3_exact']}，final_mse={rn['final_mse_ft2']:.3e}；"
                     f"demo 原 40 与 S0 重合 {leak['config']['n_overlap_demo40_s0']} 个")
            L.append(f"  {'配置':>11} {'组':>10} | {'传感':>4} {'∈S0':>3} | {'相干max':>9} {'中位全':>7} "
                     f"{'中位非正交':>8} {'正交对':>6} {'>0.999对':>8} {'T1竞争':>8} {'T2竞争':>8} {'T3竞争':>8} | "
                     f"{'L1 top3':>7} | {'支撑':>4} {'top-1':>5} {'top-3':>5} {'精确':>4} | "
                     f"{'T1误差':>7} {'T2误差':>7} {'T3误差':>7} | {'final_mse':>9} {'秒':>5}")
            for gkey, g in leak["groups"].items():
                cs = g["coherence"]
                rv = cs["true_node_max_rival_coh"]
                fe = g["flow_err"]
                mw = cs["median_within"] if cs["median_within"] is not None else float("nan")
                L.append(f"  {g['config']:>11} {g['group']:>10} | {g['n_sensors']:>4} {g['n_sensors_in_s0']:>3} | "
                         f"{cs['max_offdiag']:>9.6f} {cs['median_all']:>7.4f} {mw:>8.4f} "
                         f"{cs['n_orthogonal_pairs']:>6} {cs['n_pairs_gt_0999']:>8} "
                         f"{rv['T1']:>8.5f} {rv['T2']:>8.5f} {rv['T3']:>8.5f} | {g['stage1']['top3_hits']:>7} | "
                         f"{g['support_size']:>4} {str(g['top1_hit']):>5} {g['top3_hits']:>5} "
                         f"{str(g['top3_exact']):>4} | {fe['T1']['rel_err']:>7.3f} {fe['T2']['rel_err']:>7.3f} "
                         f"{fe['T3']['rel_err']:>7.3f} | {g['final_mse_ft2']:>9.3e} {g['time_sec']:>5.0f}")
                L.append("      支撑成员：" + "；".join(
                    (f"{d['kind']} {d['leak_lps']:.3f} LPS" if d["kind"] != "nontrue"
                     else f"非真值(与{d['closest_true']} coh={d['coh']:.5f}) {d['leak_lps']:.3f} LPS")
                    for d in g["support"]))
                if "reproduction" in g:
                    rp = g["reproduction"]
                    L.append(f"      复现对照 demo[{rp['demo_group']}]：支撑相同={rp['support_same']}，"
                             f"final_mse 相对差={rp['final_mse_rel_diff']:.2e}，matches={rp['matches']}")
            fpr = st.get("leak_footprint", {})
            if fpr:
                L.append(f"  漏点足迹（各漏点单独存在时 25 帧×传感器压力变化，ft；噪声 {fpr['noise_ft']} ft）：")
                L.append(f"  {'配置':>11} {'传感':>4} | " + " ".join(
                    f"{t + ' rms':>7} {t + ' max':>7}" for t in ("T1", "T2", "T3")))
                for name, r in list(fpr["per_config"].items()) + [("all_junctions", fpr["all_junctions"])]:
                    L.append(f"  {name:>11} {r.get('n_sensors', ''):>4} | " + " ".join(
                        f"{r[t]['rms_ft']:>7.4f} {r[t]['max_ft']:>7.4f}" for t in ("T1", "T2", "T3")))
                L.append("  读法：足迹 rms 低于噪声一个量级的漏点在该配置下不可能被找回（与布点无关时看 all_junctions 行）；"
                         "增设后 rms/max 上升的漏点才有机会被找回；足迹不变而仍未找回的漏点由相干（近双胞胎）解释。")
        else:
            L.append("  （尚无漏损记录）")
        L.append("")
    L.append("四、口径与措辞")
    L.append("  * 全程虚拟增设：新传感器在标定后的模型上模拟；论文须写为'设计与模拟验证'，现场安装留待后续。")
    L.append("  * 找回 = 普查同判据（解析零列→非零列）；σ 口径 = 贝叶斯后验 std（先验 15）。两者分开报，不混。")
    L.append(f"  * 本文件不含任何节点/链路/传感器编号；选点序列只在 data/{cfg['orders']}。")
    with open(p["wip"], "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nreport 完成：{p['suite']}、{p['wip']}")


# ======================================================================
# stage fig
# ======================================================================
def stage_fig(net):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg, p = NETS[net], paths(net)
    st = jload(p["suite"])
    ac = st["augment_curve"]
    cal = st.get("calibration", {})
    leak = st.get("leak", {}).get("groups", {}) if cfg["leak"] else {}
    n_t, ks = ac["n_target"], ac["ks"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    kk = np.arange(1, len(ac["dopt"]["recovered_every_k"]) + 1)
    for obj, c, lab in (("dopt", "C0", "augment, D-optimal"), ("cover", "C1", "augment, coverage")):
        ax.plot(kk, ac[obj]["recovered_every_k"], color=c, label=lab)
    rk = [int(k) for k in ac["reselect"]]
    ax.plot(rk, [ac["reselect"][str(k)]["n_recovered"] for k in rk], "s--", color="C3",
            label="re-select |S0|+k from scratch (recovered)")
    ax.plot(rk, [ac["reselect"][str(k)]["n_lost"] for k in rk], "x:", color="C3",
            label="re-select: previously identifiable pipes lost")
    ax.axhline(n_t, color="k", lw=0.8, ls=":")
    ax.axhline(0.8 * n_t, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(f"added sensors k (existing {ac['n_fixed']} kept)")
    ax.set_ylabel(f"pipes (of {n_t} sensor-starved)")
    ax.set_title("(a) recovered vs. added sensors")
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    for obj, c in (("dopt", "C0"), ("cover", "C1")):
        ax.plot(ks, [_med(ac[obj]["per_k"][str(k)]["post_std_recovered"]) for k in ks],
                "o-", color=c, label=f"{obj}: median posterior std of recovered pipes")
        ax.plot(ks, [ac[obj]["per_k"][str(k)]["n_recovered_half_prior"] for k in ks],
                "^--", color=c, label=f"{obj}: recovered pipes with std <= prior/2")
    ax.axhline(SIG_P, color="k", lw=0.8, ls=":")
    ax.set_xlabel("added sensors k")
    ax.set_title(f"(b) sigma-scale identifiability (prior std {SIG_P:g})")
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    if cal:
        pls = [pl for pl in cfg["calib_placements"] if pl != cfg["s0"]]
        x = np.arange(len(pls))
        w = 0.25
        for i, sg in enumerate(CALIB_SIGMAS):
            vals, pri = [], []
            for pl in pls:
                r = [v for v in cal.values() if v["placement"] == pl and v["sigma"] == sg]
                vals.append(float(np.median([_med(q["recovered_err"]) for q in r])) if r else np.nan)
                pri.append(float(np.median([_med(q["recovered_err_prior"]) for q in r])) if r else np.nan)
            ax.bar(x + (i - 1) * w, vals, w, label=f"sigma={sg:g}: |dC| median after")
            ax.plot(x + (i - 1) * w, pri, "k_", ms=10)
        ax.set_xticks(x)
        ax.set_xticklabels(pls, fontsize=8)
        ax.set_ylabel("|C_hat - C_true| of recovered pipes")
        ax.set_title("(c) calibration error of recovered pipes (tick = prior error)")
        ax.legend(fontsize=7)
    else:
        ax.set_title("(c) calibration: no AUG_* runs yet")
    ax = axes[1, 1]
    if leak:
        gs = [g for g in leak.values() if g["group"] == "noisy"]
        names = [g["config"] for g in gs]
        x = np.arange(len(names))
        ax.bar(x - 0.2, [g["top3_hits"] for g in gs], 0.4, label="top-3 hits (of 3)")
        ax.bar(x + 0.2, [int(g["top1_hit"]) for g in gs], 0.4, label="top-1 hit")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=20)
        ax.set_ylim(0, 3.2)
        ax.set_title("(d) noisy leak search (0.1 ft), work-order case")
        ax.legend(fontsize=7)
    else:
        ax.set_title("(d) leak search: " + ("no runs yet" if cfg["leak"] else "n/a for this network"))
    fig.suptitle(f"{net}: virtual sensor augmentation (design & simulation study, not installed)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(p["fig"], dpi=200)
    print(f"fig 完成：{p['fig']}")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="city_d", choices=sorted(NETS))
    ap.add_argument("--stage", required=True,
                    help="verify|calib|leak|report|fig，可逗号连写按序执行")
    ap.add_argument("--placements", default="", help="calib：布点名（缺省该网全套）")
    ap.add_argument("--sigmas", default=",".join(str(s) for s in CALIB_SIGMAS))
    ap.add_argument("--seeds", default=str(CALIB_SEED))
    ap.add_argument("--configs", default="", help="leak：传感配置名（缺省该网全套）")
    ap.add_argument("--groups", default="noisy,noisy_demo",
                    help="leak：noisy | noisy_demo（仅 demo40）| noiseless")
    ap.add_argument("--machine", default="",
                    help="report：一句话记录 calib/leak 实跑机器与 GPU 使用情况（写入 meta）")
    a = ap.parse_args()
    cfg = NETS[a.net]
    stages = [s for s in a.stage.split(",") if s]
    bad = [s for s in stages if s not in ("verify", "calib", "leak", "report", "fig")]
    if bad:
        raise SystemExit(f"未知阶段 {bad}")
    t0 = time.time()
    for st in stages:
        if st == "verify":
            stage_verify(a.net)
        elif st == "calib":
            stage_calib(a.net,
                        [s for s in a.placements.split(",") if s] or cfg["calib_placements"],
                        [float(s) for s in a.sigmas.split(",") if s],
                        [int(s) for s in a.seeds.split(",") if s])
        elif st == "leak":
            stage_leak(a.net, [s for s in a.configs.split(",") if s] or cfg["leak_configs"],
                       [s for s in a.groups.split(",") if s])
        elif st == "report":
            stage_report(a.net, a.machine)
        elif st == "fig":
            stage_fig(a.net)
    print(f"[{a.net} {a.stage}] 总墙钟 {time.time() - t0:.0f}s @ {platform.node()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

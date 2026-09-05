# -*- coding: utf-8 -*-
"""hv_fisher.py: hostile re-computation of the City D augmentation curve.

Own code throughout (no import of dgga.placement / augment_suite / place_sensors):
reads only the raw sensitivity cache, the selection orders and the readable JSONs,
then recomputes with an independent implementation

  * the census criterion  identifiable(S) <=> max_{t, i in S} |dp_i/dC_k| > 0 and not structural
  * the target set (pipes unobservable under S0), the recovered / left / lost counts
    for three randomly drawn k of the {5,10,20,40,80} grid (seed recorded)
  * monotonicity for every one of the 80 steps of both objectives (identifiable set
    never shrinks, posterior variance diag(M^-1) via explicit inverse never rises,
    lost == 0)
  * f = logdet(sigma_p^2 M), CRLB on the eps-rank subspace (the section 3.6 formula) and
    on the 1e-2 subspace, Bayesian posterior std of the recovered pipes
  * the same-threshold / same-sigma question: t=0 census vs 25-frame census of S0,
    sigma_prior / sigma_noise / atol across placement_city_d.json, placement_augment_city_d.json
    and augment_suite_city_d.json
  * section 3.6 anchors: D-optimal-from-scratch recovered/unobservable at k=10..80 and the
    random median (Table 2), Hanoi CRLB trace of the D-optimal and random-median layouts
  * the sigma-ladder claim: |C_hat - C_true| of the recovered pipes from the AUG_* records,
    with the truth regenerated here and the recovered mask from this script's criterion

Output: data/audit_augment_fisher.json (no node / link / candidate identifiers) and stdout.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
SP, SN = 15.0, 0.1
EPS = np.finfo(np.float64).eps
KGRID = [5, 10, 20, 40, 80]


def jload(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def ident_mask(S, sel, struct):
    if len(sel) == 0:
        return np.zeros(S.shape[2], dtype=bool)
    cm = np.abs(S[:, sel, :]).max(axis=(0, 1))
    return (cm > 0.0) & ~struct


def fisher_M(S, sel):
    P = S.shape[2]
    A = S[:, sel, :].reshape(-1, P)
    return np.eye(P) / SP ** 2 + (A.T @ A) / SN ** 2


def spectrum_metrics(S, sel, sub_tol=1e-2):
    P = S.shape[2]
    A = S[:, sel, :].reshape(-1, P)
    sv = np.linalg.svd(A, compute_uv=False)
    thr = max(A.shape) * EPS * sv[0]
    r = int((sv > thr).sum())
    ks = int((sv > sub_tol * sv[0]).sum())
    lam = 1.0 / SP ** 2 + sv ** 2 / SN ** 2
    return dict(rank=r, crlb_ident=float((SN ** 2 / sv[:r] ** 2).sum()),
                k_sub=ks, crlb_sub=float((SN ** 2 / sv[:ks] ** 2).sum()),
                f=float(np.log1p((SP / SN) ** 2 * sv ** 2).sum()),
                bayes_trace=float((1.0 / lam).sum() + (P - sv.size) * SP ** 2))


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260903, help="seed for drawing the 3 k values")
    ap.add_argument("--no-calib", action="store_true", help="skip the sigma-ladder section (needs the solver)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    z = np.load(os.path.join(DATA, "placement_cache_city_d.npz"))
    S = z["S_full"]
    dead, cm25, cm_t0 = z["dead"], z["cm25"], z["cm_t0"]
    struct = dead | cm25
    o = np.load(os.path.join(DATA, "placement_orders_city_d.npz"))
    fixed = np.asarray(o["augment_fixed"], dtype=np.int64)
    adds = {"dopt": np.asarray(o["augment_dopt"], dtype=np.int64),
            "cover": np.asarray(o["augment_cover"], dtype=np.int64)}
    ref = jload(os.path.join(DATA, "placement_augment_city_d.json"))
    suite = jload(os.path.join(DATA, "augment_suite_city_d.json"))
    ac = suite["augment_curve"]
    pc = jload(os.path.join(DATA, "placement_city_d.json"))
    T, m, P = S.shape
    out = dict(host=platform.node(), seed=a.seed, T=int(T), m=int(m), P=int(P),
               n_struct=int(struct.sum()), n_dead=int(dead.sum()),
               n_clamped=int((cm25 & ~dead).sum()), checks={}, notes=[])
    C = out["checks"]

    # ---------------- same sigma / same threshold ----------------
    C["sigma_prior_same"] = bool(pc["config"]["sigma_prior"] == ref["config"]["sigma_prior"]
                                 == ac["sigma_prior"] == SP)
    C["sigma_noise_same"] = bool(pc["config"]["sigma_noise"] == ref["config"]["sigma_noise"]
                                 == ac["sigma_noise"] == SN)
    C["atol_zero"] = bool(ref["config"]["atol"] == 0.0)
    C["frames_25_same"] = bool(pc["config"]["n_frames"] == ref["config"]["n_frames"] == T == 25)
    for k in ("dopt", "cover"):
        if set(adds[k].tolist()) & set(fixed.tolist()):
            raise RuntimeError(f"{k} order overlaps S0")
        if len(set(adds[k].tolist())) != adds[k].size:
            raise RuntimeError(f"{k} order has duplicates")
    C["orders_disjoint_from_S0_and_unique"] = True

    # S0 census, 25 frames (augment criterion) and t=0 (the census that produced 149)
    id0 = ident_mask(S, fixed, struct)
    target = ~id0 & ~struct
    struct_t0 = dead | cm_t0
    id0_t0 = (np.abs(S[:1, fixed, :]).max(axis=(0, 1)) > 0.0) & ~struct_t0
    unob_t0 = ~id0_t0 & ~struct_t0
    C["n_target_149"] = int(target.sum())
    C["target_equals_cache_mask149"] = bool(np.array_equal(target, z["mask149"]))
    C["t0_census_equals_25frame_census"] = bool(np.array_equal(unob_t0, target))
    C["n_unob_t0"] = int(unob_t0.sum())
    C["census_partition"] = dict(identifiable=int(id0.sum()), unobservable=int(target.sum()),
                                 dead=int(dead.sum()), clamped=int((cm25 & ~dead).sum()),
                                 sum=int(id0.sum() + target.sum() + struct.sum()), P=int(P))
    C["census_partition_matches_paper_279_149_41_6"] = bool(
        id0.sum() == 279 and target.sum() == 149 and dead.sum() == 41 and (cm25 & ~dead).sum() == 6)
    # pos40 replay: 40 random junctions, seed 2026, over the cached junction list
    rng = np.random.default_rng(2026)
    sens40 = np.sort(rng.choice(z["junc"], size=40, replace=False))
    C["S0_is_seed2026_sample_of_junctions"] = bool(np.array_equal(np.sort(z["junc"][fixed]), sens40))

    # ---------------- three random k, both objectives ----------------
    rng = np.random.default_rng(a.seed)
    ks = sorted(int(k) for k in rng.choice(KGRID, size=3, replace=False))
    out["ks_drawn"] = ks
    M0 = fisher_M(S, fixed)
    pv0 = np.diag(np.linalg.inv(M0))
    m0 = spectrum_metrics(S, fixed)
    out["s0"] = dict(n_ident=int(id0.sum()), n_unob=int(target.sum()), **m0,
                     post_std_median_target=float(np.median(np.sqrt(pv0[target]))),
                     ref_rank=ref["s0"]["rank"], ref_f=ref["s0"]["f"], ref_crlb_ident=ref["s0"]["crlb_ident"],
                     suite_crlb_sub=ac["s0_metrics"]["crlb_sub"], suite_k_sub=ac["s0_metrics"]["k_sub"])
    C["s0_rank_f_crlb_match"] = bool(m0["rank"] == ref["s0"]["rank"] and rel(m0["f"], ref["s0"]["f"]) < 1e-9
                                     and rel(m0["crlb_ident"], ref["s0"]["crlb_ident"]) < 1e-6
                                     and rel(m0["crlb_sub"], ac["s0_metrics"]["crlb_sub"]) < 1e-6
                                     and m0["k_sub"] == ac["s0_metrics"]["k_sub"])
    table = []
    mono = {}
    for obj in ("dopt", "cover"):
        add = adds[obj]
        prev_id, prev_pv = id0.copy(), pv0.copy()
        viol_id = viol_pv = 0
        lost_max = 0
        rec_curve, unob_curve = [], []
        for j in range(add.size):
            sel = np.r_[fixed, add[:j + 1]]
            idj = ident_mask(S, sel, struct)
            if np.any(prev_id & ~idj):
                viol_id += 1
            lost_max = max(lost_max, int((id0 & ~idj).sum()))
            pvj = np.diag(np.linalg.inv(fisher_M(S, sel)))
            if np.any(pvj > prev_pv * (1.0 + 1e-9)):
                viol_pv += 1
            prev_id, prev_pv = idj, pvj
            rec_curve.append(int((idj & target).sum()))
            unob_curve.append(int((~idj & ~struct).sum()))
        rc_ref = ref["augment"][obj]["recovered_every_k"]
        mono[obj] = dict(steps=int(add.size), ident_shrink_violations=viol_id,
                         post_var_rise_violations=viol_pv, lost_max=lost_max,
                         recovered_every_k_equals_ref=bool(rec_curve == list(rc_ref)),
                         k_recover80=int(next((i + 1 for i, r in enumerate(rec_curve) if r >= 0.8 * target.sum()), -1)),
                         k_recover_all=int(next((i + 1 for i, r in enumerate(rec_curve) if r >= target.sum()), -1)),
                         ref_k_recover80=ref["augment"][obj]["k_recover80"],
                         ref_k_recover_all=ref["augment"][obj]["k_recover_all"])
        for k in ks:
            sel = np.r_[fixed, add[:k]]
            idk = ident_mask(S, sel, struct)
            rec_m = idk & target
            n_rec, n_left = int(rec_m.sum()), int((target & ~idk).sum())
            n_unob, n_lost = int((~idk & ~struct).sum()), int((id0 & ~idk).sum())
            sm = spectrum_metrics(S, sel)
            pv = np.diag(np.linalg.inv(fisher_M(S, sel)))
            ps = np.sqrt(pv)
            r = ref["augment"][obj]["per_k"][str(k)]
            q = ac[obj]["per_k"][str(k)]
            row = dict(objective=obj, k=k, n_sensors=int(np.unique(sel).size),
                       n_recovered=n_rec, n_left=n_left, n_unobservable=n_unob, n_lost=n_lost,
                       n_identifiable=int(idk.sum()), **sm,
                       post_std_recovered_median=float(np.median(ps[rec_m])) if n_rec else None,
                       n_recovered_half_prior=int((ps[rec_m] <= 0.5 * SP).sum()),
                       ref=dict(n_recovered=r["n_recovered"], n_unobservable=r["n_unobservable"],
                                n_lost=r["n_lost"], n_identifiable=r["n_identifiable"], rank=r["rank"],
                                f=r["f"], crlb_ident=r["crlb_ident"], bayes_trace=r["bayes_trace"],
                                post_std_recovered_median=r["post_std_recovered"]["median"],
                                n_half_prior=r["post_std_recovered"]["n_half_prior"]),
                       suite=dict(crlb_sub=q["crlb_sub"], k_sub=q["k_sub"], rank=q["rank"],
                                  n_recovered=q["n_recovered"], n_unobservable=q["n_unobservable"]))
            row["match_counts"] = bool(n_rec == r["n_recovered"] == q["n_recovered"]
                                       and n_unob == r["n_unobservable"] == q["n_unobservable"]
                                       and n_lost == 0 == r["n_lost"]
                                       and idk.sum() == r["n_identifiable"])
            row["match_spectrum"] = bool(sm["rank"] == r["rank"] == q["rank"]
                                         and rel(sm["f"], r["f"]) < 1e-9
                                         and rel(sm["crlb_ident"], r["crlb_ident"]) < 1e-6
                                         and rel(sm["bayes_trace"], r["bayes_trace"]) < 1e-9
                                         and sm["k_sub"] == q["k_sub"]
                                         and rel(sm["crlb_sub"], q["crlb_sub"]) < 1e-6)
            row["match_post_std"] = bool(n_rec == 0 or (
                abs(row["post_std_recovered_median"] - r["post_std_recovered"]["median"]) < 1e-9
                and row["n_recovered_half_prior"] == r["post_std_recovered"]["n_half_prior"]))
            table.append(row)
    out["monotonicity"] = mono
    out["table"] = table
    C["all_rows_match"] = bool(all(r["match_counts"] and r["match_spectrum"] and r["match_post_std"]
                                   for r in table))
    C["monotonic_all_steps_both_objectives"] = bool(all(
        v["ident_shrink_violations"] == 0 and v["post_var_rise_violations"] == 0 and v["lost_max"] == 0
        for v in mono.values()))

    # ---------------- section 3.6 anchors (from-scratch D-opt and random median) ----------------
    dord = np.asarray(o["dopt"], dtype=np.int64)
    paper = {10: (32, 159), 20: (43, 148), 40: (52, 135), 60: (68, 105), 80: (73, 97)}
    paper_rand = {10: (9, 199), 20: (18.5, 174), 40: (31.5, 140.5), 60: (43, 125.5), 80: (53, 109.5)}
    anchors = {}
    R = np.asarray(o["random"], dtype=np.int64)
    for k, (prec, punob) in paper.items():
        idk = ident_mask(S, dord[:k], struct)
        rec, unob, lost = int((idk & target).sum()), int((~idk & ~struct).sum()), int((id0 & ~idk).sum())
        rr = [int((ident_mask(S, R[s, :k], struct) & target).sum()) for s in range(R.shape[0])]
        ru = [int((~ident_mask(S, R[s, :k], struct) & ~struct).sum()) for s in range(R.shape[0])]
        anchors[str(k)] = dict(dopt_recovered=rec, dopt_unobservable=unob, dopt_lost=lost,
                               paper_dopt=prec, paper_dopt_unob=punob,
                               random_median_recovered=float(np.median(rr)),
                               random_median_unobservable=float(np.median(ru)),
                               paper_random=paper_rand[k][0], paper_random_unob=paper_rand[k][1],
                               match=bool(rec == prec and unob == punob
                                          and float(np.median(rr)) == paper_rand[k][0]
                                          and float(np.median(ru)) == paper_rand[k][1]))
    idk = ident_mask(S, dord[:40], struct)
    anchors["k40_lost"] = int((id0 & ~idk).sum())
    anchors["k40_lost_paper_38"] = bool(anchors["k40_lost"] == 38)
    anchors["k40_crlb_ident_same_formula"] = spectrum_metrics(S, dord[:40])["crlb_ident"]
    anchors["k40_crlb_ident_ref"] = pc["key_numbers"]["dopt_k40"]["crlb_ident"]
    anchors["k40_crlb_match"] = bool(rel(anchors["k40_crlb_ident_same_formula"], anchors["k40_crlb_ident_ref"]) < 1e-6)
    out["section36_anchors"] = anchors
    C["section36_city_d_table_matches"] = bool(all(v["match"] for k, v in anchors.items() if k.isdigit())
                                               and anchors["k40_lost_paper_38"] and anchors["k40_crlb_match"])

    # Hanoi CRLB (Table 2 of the manuscript): same SVD formula, 1 frame, 31 candidates
    zh = np.load(os.path.join(DATA, "placement_cache_pub_hanoi.npz"))
    oh = np.load(os.path.join(DATA, "placement_orders_pub_hanoi.npz"))
    Sh = zh["S_full"]
    tab_d = {2: 0.034, 4: 0.153, 6: 0.454, 8: 0.832, 10: 1.475}
    tab_r = {2: 0.077, 4: 0.696, 6: 2.509, 8: 5.164, 10: 13.642}
    han = {}
    Rh = np.asarray(oh["random"], dtype=np.int64)
    for k in tab_d:
        cd = spectrum_metrics(Sh, np.asarray(oh["dopt"][:k]))["crlb_ident"]
        cr = float(np.median([spectrum_metrics(Sh, Rh[s, :k])["crlb_ident"] for s in range(Rh.shape[0])]))
        han[str(k)] = dict(dopt=cd, random_median=cr, table_dopt=tab_d[k], table_random=tab_r[k],
                           match=bool(round(cd, 3) == tab_d[k] and round(cr, 3) == tab_r[k]))
    han["k10_full_precision"] = dict(dopt=han["10"]["dopt"], text_1_4746=1.4746,
                                     random=han["10"]["random_median"], ratio=han["10"]["random_median"] / han["10"]["dopt"])
    out["hanoi_crlb"] = han
    C["section36_hanoi_crlb_matches"] = bool(all(v["match"] for k, v in han.items() if k.isdigit()))

    # ---------------- sigma ladder: recovered-pipe error from AUG_* records ----------------
    if not a.no_calib:
        sys.path.insert(0, ROOT)
        from dgga.parse import Net
        from dgga.solver import GGASolver
        from dgga.autodiff import solve_polished
        from dgga.calib import dead_branch_mask, clamped_mask
        net = Net.load(os.path.join(DATA, "reference"), "city_d")
        s = GGASolver(net, mode="dense", inp_path=os.path.join(ROOT, "networks", "realInpData", "city_d.inp"))
        TT = [t * 3600 for t in range(25)]
        d = np.stack([net.demand_cfs_at(t) for t in TT])
        rh = np.stack([np.nan_to_num(net.reservoir_head_ft_at(t)) for t in TT])
        pidx = np.where(np.isin(s.lt_np, (0, 1)) & (s.kc_np > 0.0))[0]
        if not np.array_equal(pidx, z["pipe_idx"]):
            raise RuntimeError("pipe index differs from the placement cache")
        sol0 = solve_polished(s, d, rh, accuracy=1e-12, max_iter=60, polish_steps=3)
        cm1 = clamped_mask(s, sol0, margin=1.0)["mask"]
        dd = dead_branch_mask(s, demand=d)
        pipe = np.zeros(s.L, dtype=bool)
        pipe[pidx] = True
        struct_c = pipe & (dd | cm1 | s.closed_np)
        free_idx = pidx[~struct_c[pidx]]
        pos_of = {int(g): j for j, g in enumerate(free_idx)}
        rng = np.random.default_rng(7)
        C_true = np.full(s.L, 130.0)
        C_true[pidx] = rng.uniform(75.0, 145.0, pidx.size)
        cal = jload(os.path.join(DATA, "calib_gc1_city_d.json"))["runs"]
        scal = suite.get("calibration", {})
        rows = []
        for key in sorted(k for k in cal if k.startswith("AUG_")):
            r = cal[key]
            pl = r["placement"]
            if pl == "ga40":
                sel = fixed
            elif pl.startswith("augdopt"):
                sel = np.r_[fixed, adds["dopt"][:int(pl[7:])]]
            elif pl.startswith("augcover"):
                sel = np.r_[fixed, adds["cover"][:int(pl[8:])]]
            else:
                continue
            Ch = np.asarray(r["C_hat_free"], dtype=np.float64)
            if Ch.size != free_idx.size:
                rows.append(dict(key=key, error=f"C_hat_free size {Ch.size} != free {free_idx.size}"))
                continue
            idk = ident_mask(S, sel, struct)
            rec_m = idk & target
            g = pidx[rec_m]
            jj = np.array([pos_of[int(x)] for x in g if int(x) in pos_of], dtype=np.int64)
            gg = np.array([int(x) for x in g if int(x) in pos_of], dtype=np.int64)
            e = np.abs(Ch[jj] - C_true[gg])
            ep = np.abs(130.0 - C_true[gg])
            # unobservable pipes stay at the prior in calibrate: infer C0 from them
            gu = pidx[~idk & ~struct]
            ju = np.array([pos_of[int(x)] for x in gu if int(x) in pos_of], dtype=np.int64)
            c0_inferred = float(np.median(Ch[ju])) if ju.size else None
            q = scal.get(key, {})
            row = dict(key=key, placement=pl, sigma=r["sigma"], seed=r["noise_seed"], n_sensors=r["n_sensors"],
                       n_recovered=int(rec_m.sum()), n_recovered_in_free=int(jj.size),
                       err_median=float(np.median(e)) if e.size else None,
                       err_prior_median=float(np.median(ep)) if ep.size else None,
                       n_better_than_prior=int((e < ep).sum()), n_le10=int((e <= 10).sum()),
                       c0_inferred_from_unobservable=c0_inferred,
                       n_unob_record=r["n_unobservable"], n_unob_here=int((~idk & ~struct).sum()),
                       suite_err_median=q.get("recovered_err", {}).get("median"),
                       suite_err_prior_median=q.get("recovered_err_prior", {}).get("median"),
                       suite_n_better=q.get("recovered_n_better_than_prior"))
            row["match_suite"] = bool(q and e.size and abs(row["err_median"] - q["recovered_err"]["median"]) < 1e-9
                                      and abs(row["err_prior_median"] - q["recovered_err_prior"]["median"]) < 1e-9
                                      and row["n_better_than_prior"] == q["recovered_n_better_than_prior"]) \
                if e.size else bool(q.get("n_recovered", -1) == 0)
            rows.append(row)
        out["sigma_ladder"] = rows
        C["sigma_ladder_all_match"] = bool(rows and all(r.get("match_suite") for r in rows))
        C["n_free_432"] = bool(free_idx.size == 432)

    out["elapsed_sec"] = time.perf_counter() - t0
    fp = os.path.join(DATA, "audit_augment_fisher.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---------------- print ----------------
    print(f"[hv_fisher] host={out['host']}  T={T} m={m} P={P} struct={out['n_struct']} "
          f"(dead {out['n_dead']} + clamped {out['n_clamped']})  ks drawn (seed {a.seed}) = {ks}")
    for k, v in C.items():
        print(f"  check {k:48s} {v}")
    print(f"  S0: ident {out['s0']['n_ident']} unob {out['s0']['n_unob']} rank {m0['rank']} f {m0['f']:.6f} "
          f"CRLB_ident {m0['crlb_ident']:.4e} k_sub {m0['k_sub']} CRLB_sub {m0['crlb_sub']:.4e}")
    print(f"  {'obj':5s} {'k':>3} {'rec':>4} {'left':>4} {'unob':>4} {'lost':>4} {'ident':>5} {'rank':>4} "
          f"{'f':>10} {'CRLB_id':>10} {'k_sub':>5} {'CRLB_sub':>10} {'pstd_med':>8} {'<=7.5':>5} | ref rec/unob | match")
    for r in table:
        print(f"  {r['objective']:5s} {r['k']:>3} {r['n_recovered']:>4} {r['n_left']:>4} {r['n_unobservable']:>4} "
              f"{r['n_lost']:>4} {r['n_identifiable']:>5} {r['rank']:>4} {r['f']:>10.4f} {r['crlb_ident']:>10.3e} "
              f"{r['k_sub']:>5} {r['crlb_sub']:>10.3e} "
              f"{(r['post_std_recovered_median'] or float('nan')):>8.3f} {r['n_recovered_half_prior']:>5} | "
              f"{r['ref']['n_recovered']:>3}/{r['ref']['n_unobservable']:<3} | "
              f"{r['match_counts']} {r['match_spectrum']} {r['match_post_std']}")
    for obj, v in mono.items():
        print(f"  monotonic {obj}: {v}")
    for k, v in anchors.items():
        if k.isdigit():
            print(f"  sec3.6 k={k}: dopt rec/unob {v['dopt_recovered']}/{v['dopt_unobservable']} (paper "
                  f"{v['paper_dopt']}/{v['paper_dopt_unob']}), random median {v['random_median_recovered']}/"
                  f"{v['random_median_unobservable']} (paper {v['paper_random']}/{v['paper_random_unob']}) match={v['match']}")
    print(f"  sec3.6 k=40 lost {anchors['k40_lost']} (paper 38); CRLB_ident {anchors['k40_crlb_ident_same_formula']:.4e} "
          f"vs ref {anchors['k40_crlb_ident_ref']:.4e}")
    for k, v in han.items():
        if k.isdigit():
            print(f"  hanoi k={k}: dopt {v['dopt']:.4f} (table {v['table_dopt']}), random median {v['random_median']:.4f} "
                  f"(table {v['table_random']}) match={v['match']}")
    if "sigma_ladder" in out:
        print(f"  sigma ladder (free {free_idx.size}):")
        for r in out["sigma_ladder"]:
            if "error" in r:
                print(f"    {r['key']}: {r['error']}")
                continue
            print(f"    {r['placement']:>11} s={r['sigma']:<5g} rec {r['n_recovered']:>3} err_med "
                  f"{(r['err_median'] if r['err_median'] is not None else float('nan')):>6.2f} prior "
                  f"{(r['err_prior_median'] if r['err_prior_median'] is not None else float('nan')):>6.2f} "
                  f"better {r['n_better_than_prior']:>3} <=10 {r['n_le10']:>3} unob rec/here {r['n_unob_record']}/{r['n_unob_here']} "
                  f"C0_inferred {r['c0_inferred_from_unobservable']} match_suite={r['match_suite']}")
    print(f"[hv_fisher] wrote {fp}  ({out['elapsed_sec']:.1f}s)")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""hv_leak_fair.py: fairness audit of the augmented work-order leak rerun (City D).

Own code (uses the dgga engine only, none of demo_leak_inversion / augment_suite):
  * replays the recorded case: seed 2026 (third leak node, demo's 40 sensors), noise seed 909,
    0.1 ft, 25 frames, 49 candidates, lambda 1e-4, 100 stage-1 iterations; checks that
    data/leak_augment_city_d.json used exactly those values and the same true leak flows
  * S0 = the 40 census sensors (seed 2026 draw over the junction list); overlap with the demo's 40
  * proximity: hop distance from each true leak node to the nearest S0 sensor and to the
    nearest ADDED sensor for every augmentation set (dopt/cover, k = 5..80), whether the
    leak node itself carries an added sensor, and the step at which the first added sensor
    lands within 0 / 1 / 2 hops of each leak node (the trivial-success test)
  * footprint of each leak alone (25 frames, all junctions): max |dh| and the hop distance
    from the leak to the node where the footprint peaks; rms / max on every sensor set
  * signature-dictionary coherence recomputed from scratch (t=0 finite-difference signatures at
    C = 0.3, frame scaling sqrt(p(t)/p(0)), column-normalised Gram matrix) for every sensor set,
    with the two-denominator convention (all pairs / pairs that are not exactly orthogonal,
    |coh| >= 1e-12); compared with the recorded coherence blocks
  * noise realisation: the recorded "noisy" group draws the field on all N nodes (seed 909)
    and slices sensor columns; the paper's demo drew a [25, 40] field with the same seed.
    Both are regenerated and compared.

Output: data/audit_augment_leak_fair.json (labels T1..T3 and hop counts only; no identifiers).
"""
import json
import os
import platform
import sys
import time
from collections import deque

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
from dgga.parse import Net                                   # noqa: E402
from dgga.solver import GGASolver                            # noqa: E402
from dgga.autodiff import solve_polished                     # noqa: E402
from dgga.units import LPSperCFS, MperFT                     # noqa: E402

SEED, SEED_NOISE, N_SENSOR, NOISE_FT = 2026, 909, 40, 0.1
OBS_MI, GGA_MI, POLISH = 60, 20, 4
C_PROBE = 0.3
ORTH_TOL = 1e-12
KS = [5, 10, 20, 40, 80]


def jload(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


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


def coh_stats(coh, true_idx, labels):
    A = np.abs(coh)
    nc = A.shape[0]
    iu, ju = np.triu_indices(nc, 1)
    off = A[iu, ju]
    orth = off < ORTH_TOL
    within = off[~orth]
    B = A.copy()
    np.fill_diagonal(B, -1.0)
    return dict(n_pairs=int(off.size), n_orthogonal=int(orth.sum()), n_within=int(within.size),
                max_offdiag=float(off.max()), median_all=float(np.median(off)),
                median_within=float(np.median(within)) if within.size else None,
                n_gt_0999=int((off > 0.999).sum()), n_gt_099=int((off > 0.99).sum()),
                rival_max={labels[j]: float(B[j].max()) for j in true_idx},
                n_rivals_gt_099={labels[j]: int((B[j] > 0.99).sum()) for j in true_idx},
                true_pair={f"{labels[a]}-{labels[b]}": float(coh[a, b])
                           for i, a in enumerate(true_idx) for b in true_idx[i + 1:]})


def main():
    t_start = time.perf_counter()
    net = Net.load(os.path.join(DATA, "reference"), "city_d")
    s = GGASolver(net, mode="dense", inp_path=os.path.join(ROOT, "networks", "realInpData", "city_d.inp"))
    N = net.N
    node_index = {nid: i for i, nid in enumerate(net.node_id)}
    TT = [t * 3600 for t in range(25)]
    d = np.stack([net.demand_cfs_at(t) for t in TT])
    rh = np.stack([np.nan_to_num(net.reservoir_head_ft_at(t)) for t in TT])
    rec = jload(os.path.join(ROOT, "networks", "field_records", "city_d_leak_records.json"))
    cand = [n for n in rec["distinct_nodes"] if n in node_index and net.node_type[node_index[n]] == 0]
    cidx = np.array([node_index[n] for n in cand])
    nc = len(cand)
    junc = np.asarray(s.junc_nodes)

    # ---- replay of the recorded case ----
    rng = np.random.default_rng(SEED)
    others = [n for n in cand if n not in ("195", "3083")]
    third = str(rng.choice(others))
    sens_demo = np.sort(rng.choice(junc, N_SENSOR, replace=False))
    targets = {"195": 3.0, "3083": 1.5, third: 2.2}
    labels = {n: f"T{i + 1}" for i, n in enumerate(targets)}
    tidx = {n: node_index[n] for n in targets}
    true_idx = [cand.index(n) for n in targets]
    lab_of = {cand.index(n): labels[n] for n in targets}
    demo = jload(os.path.join(DATA, "demo_leak_inversion.json"))
    leak = jload(os.path.join(DATA, "leak_augment_city_d.json"))
    dc, lc = demo["config"], leak["config"]
    out = dict(host=platform.node(), checks={}, notes=[])
    C = out["checks"]
    C["demo_true_nodes_replayed"] = bool(set(dc["true_nodes"]) == set(targets))
    C["demo_sensors_replayed"] = bool(set(dc["sensors"]) == set(net.node_id[i] for i in sens_demo))
    C["demo_candidates_49"] = bool(dc["candidates"] == cand and nc == 49 == lc["n_candidates"])
    C["recipe_same"] = bool(lc["seed"] == dc["seed"] == SEED and lc["seed_noise"] == dc["seed_noise"] == SEED_NOISE
                            and lc["noise_ft"] == dc["noise_ft"] == NOISE_FT and lc["frames"] == dc["frames"] == 25
                            and lc["stage1_iters"] == dc["stage1_iters"] == 100
                            and lc["lam"] == demo["groups"]["noiseless"]["lambda_star"] == 1e-4)

    # S0 and augmentation orders
    o = np.load(os.path.join(DATA, "placement_orders_city_d.npz"))
    fixed = np.asarray(o["augment_fixed"], dtype=np.int64)
    adds = {"dopt": np.asarray(o["augment_dopt"], dtype=np.int64),
            "cover": np.asarray(o["augment_cover"], dtype=np.int64)}
    s0_nodes = junc[fixed]
    rng2 = np.random.default_rng(SEED)
    C["S0_is_seed2026_draw"] = bool(np.array_equal(np.sort(s0_nodes), np.sort(rng2.choice(junc, 40, replace=False))))
    C["overlap_demo40_S0"] = int(len(set(s0_nodes.tolist()) & set(sens_demo.tolist())))
    C["overlap_matches_record_22"] = bool(C["overlap_demo40_S0"] == lc["n_overlap_demo40_s0"] == 22)
    C["leak_nodes_in_candidate_pool_of_placement"] = {labels[n]: bool(tidx[n] in set(junc.tolist())) for n in targets}
    C["leak_node_in_S0"] = {labels[n]: bool(tidx[n] in set(s0_nodes.tolist())) for n in targets}
    C["leak_node_in_demo40"] = {labels[n]: bool(tidx[n] in set(sens_demo.tolist())) for n in targets}

    # ---- units and truth ----
    gamma, qexp = float(net.meta["emitter_exponent"]), float(net.meta["qexp"])
    ucf_e = LPSperCFS ** qexp / (MperFT * 0.998)

    def ke_of_C(c):
        return ucf_e / c ** qexp

    sol_base = solve_polished(s, d, rh, accuracy=1e-12, max_iter=OBS_MI, polish_steps=POLISH)
    ke_true = np.zeros(N)
    ke_one = {}
    for n, q in targets.items():
        i = tidx[n]
        p_m = (sol_base["head"][:, i] - net.elev_ft[i]).mean() * MperFT
        ke_true[i] = ke_of_C(q / p_m ** gamma)
        k1 = np.zeros(N)
        k1[i] = ke_true[i]
        ke_one[n] = k1
    sol_true = solve_polished(s, d, rh, ke=ke_true, accuracy=1e-12, max_iter=OBS_MI, polish_steps=POLISH)
    true_lk = {labels[n]: float(sol_true["emitter"][:, tidx[n]].mean() * LPSperCFS) for n in targets}
    out["true_leak_lps"] = true_lk
    C["true_leak_flows_match_record"] = bool(all(abs(true_lk[k] - lc["true_leak_lps"][k]) < 1e-9 for k in true_lk))
    C["true_leak_flows_match_demo"] = bool(all(
        abs(true_lk[labels[n]] - dc["true_nodes"][n]["true_mean_lps"]) < 1e-9 for n in targets))

    # ---- hop distances ----
    dist = {n: hop_distances(N, net.link_n1, net.link_n2, tidx[n]) for n in targets}
    is_junc = np.zeros(N, dtype=bool)
    is_junc[junc] = True
    # footprint of each leak alone, all nodes
    foot = {}
    dh = {}
    for n in targets:
        sol_n = solve_polished(s, d, rh, ke=ke_one[n], accuracy=1e-12, max_iter=OBS_MI, polish_steps=POLISH)
        dh[n] = sol_n["head"] - sol_base["head"]                       # [25, N]
        amax = np.abs(dh[n]).max(axis=0)                               # [N]
        amax_j = np.where(is_junc, amax, -1.0)
        peak = int(np.argmax(amax_j))
        # how many junctions see a footprint above the noise, and above 3x noise
        foot[labels[n]] = dict(max_ft_all_junctions=float(amax_j.max()),
                               hop_leak_to_peak=int(dist[n][peak]),
                               peak_is_leak_node=bool(peak == tidx[n]),
                               n_junctions_max_gt_noise=int((amax_j > NOISE_FT).sum()),
                               n_junctions_max_gt_3noise=int((amax_j > 3 * NOISE_FT).sum()),
                               n_junctions_rms_gt_noise=int((np.sqrt((dh[n] ** 2).mean(axis=0))[is_junc] > NOISE_FT).sum()),
                               hops_of_junctions_max_gt_noise=sorted(int(x) for x in dist[n][(amax_j > NOISE_FT)]))
    out["footprint_all_junctions"] = foot

    # ---- signature dictionary from scratch (all nodes once, then sliced per sensor set) ----
    d0, rh0 = d[0], rh[0]
    P0 = sol_base["head"][0]
    sig_all = np.zeros((nc, N))
    for a in range(0, nc, 25):
        b = min(a + 25, nc)
        ke = np.zeros((b - a, N))
        for k, j in enumerate(range(a, b)):
            ke[k, cidx[j]] = ke_of_C(C_PROBE)
        sj = solve_polished(s, np.tile(d0, (b - a, 1)), np.tile(rh0, (b - a, 1)), ke=ke,
                            accuracy=1e-12, max_iter=GGA_MI, polish_steps=POLISH)
        for k, j in enumerate(range(a, b)):
            sig_all[j] = (sj["head"][k] - P0) / C_PROBE
    p_base = sol_base["head"][:, cidx] - net.elev_ft[cidx]
    scale = np.sqrt(np.maximum(p_base, 1e-9) / np.maximum(p_base[0], 1e-9))       # [25, nc]

    def coherence(sens):
        D = (scale[:, None, :] * sig_all[:, sens].T[None, :, :]).reshape(25 * len(sens), nc)
        Dn = D / np.maximum(np.linalg.norm(D, axis=0), 1e-12)
        return Dn.T @ Dn

    # ---- per configuration ----
    def sensors_of(name):
        if name == "demo40":
            return sens_demo
        if name == "ga40":
            return np.sort(s0_nodes)
        for obj, pre in (("dopt", "augdopt"), ("cover", "augcover")):
            if name.startswith(pre):
                k = int(name[len(pre):])
                return np.sort(junc[np.r_[fixed, adds[obj][:k]]]), np.sort(junc[adds[obj][:k]])
        raise ValueError(name)

    names = ["demo40", "ga40"] + [f"aug{obj}{k}" for k in KS for obj in ("cover", "dopt")]
    rows = {}
    for name in names:
        r = sensors_of(name)
        if isinstance(r, tuple):
            sens, added = r
        else:
            sens, added = r, np.array([], dtype=np.int64)
        s0set = np.sort(s0_nodes) if name != "demo40" else sens_demo
        prox = {}
        for n in targets:
            L = labels[n]
            dn = dist[n]
            prox[L] = dict(hop_to_nearest_S0=int(dn[s0set].min()),
                           hop_to_nearest_added=int(dn[added].min()) if added.size else None,
                           hop_to_nearest_sensor=int(dn[sens].min()),
                           leak_node_is_added=bool(tidx[n] in set(added.tolist())),
                           n_added_within_1hop=int((dn[added] <= 1).sum()) if added.size else 0,
                           n_added_within_2hops=int((dn[added] <= 2).sum()) if added.size else 0,
                           n_S0_within_2hops=int((dn[s0set] <= 2).sum()),
                           foot_rms_ft=float(np.sqrt((dh[n][:, sens] ** 2).mean())),
                           foot_max_ft=float(np.abs(dh[n][:, sens]).max()))
        cs = coh_stats(coherence(sens), true_idx, lab_of)
        rows[name] = dict(n_sensors=int(sens.size), n_added=int(added.size), proximity=prox, coherence=cs)
        g = leak["groups"].get(f"{name}:noisy")
        if g:
            gc = g["coherence"]
            rows[name]["coherence_recorded"] = dict(max_offdiag=gc["max_offdiag"], median_all=gc["median_all"],
                                                    median_within=gc["median_within"], n_gt_0999=gc["n_pairs_gt_0999"],
                                                    n_orthogonal=gc["n_orthogonal_pairs"],
                                                    rival_max=gc["true_node_max_rival_coh"])
            rows[name]["coherence_match"] = bool(
                abs(cs["max_offdiag"] - gc["max_offdiag"]) < 1e-9 and abs(cs["median_all"] - gc["median_all"]) < 1e-9
                and cs["n_gt_0999"] == gc["n_pairs_gt_0999"] and cs["n_orthogonal"] == gc["n_orthogonal_pairs"]
                and all(abs(cs["rival_max"][t] - gc["true_node_max_rival_coh"][t]) < 1e-9 for t in cs["rival_max"]))
            rows[name]["result_recorded"] = dict(top1=g["top1_hit"], top3=g["top3_hits"],
                                                 support=[x["kind"] for x in g["support"]],
                                                 flow_rel_err={t: g["flow_err"][t]["rel_err"] for t in g["flow_err"]})
            fp = suite_footprint = None
    out["configs"] = rows
    # first step at which an added sensor lands within h hops of each leak (per objective)
    first = {}
    for obj in ("dopt", "cover"):
        add_nodes = junc[adds[obj]]
        first[obj] = {}
        for n in targets:
            dn = dist[n][add_nodes]
            first[obj][labels[n]] = {f"within_{h}": int(next((j + 1 for j, x in enumerate(dn) if x <= h), -1))
                                     for h in (0, 1, 2, 3)}
    out["first_added_step_within_hops"] = first
    C["coherence_all_match"] = bool(all(v.get("coherence_match", True) for v in rows.values()))
    C["coherence_two_denominators_identical_on_city_d"] = bool(all(
        v["coherence"]["n_orthogonal"] == 0 and v["coherence"]["median_all"] == v["coherence"]["median_within"]
        for v in rows.values()))

    # ---- noise realisations ----
    nf = NOISE_FT * np.random.default_rng(SEED_NOISE).standard_normal((25, N))
    nd = NOISE_FT * np.random.default_rng(SEED_NOISE).standard_normal((25, N_SENSOR))
    out["noise"] = dict(demo_field_shape=[25, N_SENSOR], augment_field_shape=[25, int(N)],
                        max_abs_diff_on_demo40=float(np.abs(nf[:, sens_demo] - nd).max()),
                        same_realisation=bool(np.array_equal(nf[:, sens_demo], nd)),
                        note="the 'noisy' group of the rerun is a different realisation from the paper's "
                             "demo group; the demo group itself was reproduced separately (noisy_demo)")
    out["elapsed_sec"] = time.perf_counter() - t_start
    fp = os.path.join(DATA, "audit_augment_leak_fair.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"[hv_leak_fair] host={out['host']} nc={nc} N={N} junctions={junc.size}")
    for k, v in C.items():
        print(f"  check {k:48s} {v}")
    print(f"  true leak flows (LPS): {true_lk}")
    for L, v in foot.items():
        print(f"  footprint {L}: max over junctions {v['max_ft_all_junctions']:.4f} ft, peak at hop "
              f"{v['hop_leak_to_peak']} (peak is leak node: {v['peak_is_leak_node']}), junctions with max>noise "
              f"{v['n_junctions_max_gt_noise']} (hops {v['hops_of_junctions_max_gt_noise']}), rms>noise {v['n_junctions_rms_gt_noise']}")
    print(f"  {'config':>11} {'n':>3} {'add':>3} | " + " | ".join(
        f"{t}: S0 add lk1 lk2 rms   max" for t in ("T1", "T2", "T3")) + " | coh max  med  >.999 match | rec top3 support")
    for name, v in rows.items():
        p = v["proximity"]
        cells = []
        for t in ("T1", "T2", "T3"):
            q = p[t]
            cells.append(f"{t}: {q['hop_to_nearest_S0']:>2} {str(q['hop_to_nearest_added']):>4} "
                         f"{q['n_added_within_1hop']:>3} {q['n_added_within_2hops']:>3} {q['foot_rms_ft']:.3f} {q['foot_max_ft']:.3f}")
        cs = v["coherence"]
        rr = v.get("result_recorded", {})
        print(f"  {name:>11} {v['n_sensors']:>3} {v['n_added']:>3} | " + " | ".join(cells)
              + f" | {cs['max_offdiag']:.6f} {cs['median_all']:.4f} {cs['n_gt_0999']:>3} {v.get('coherence_match', '-')} | "
              + (f"{rr.get('top3')} {rr.get('support')}" if rr else "-"))
    for obj, v in first.items():
        print(f"  first added step within h hops [{obj}]: {v}")
    print(f"  noise: {out['noise']}")
    print(f"[hv_leak_fair] wrote {fp} ({out['elapsed_sec']:.1f}s)")


if __name__ == "__main__":
    main()

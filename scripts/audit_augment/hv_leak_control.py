# -*- coding: utf-8 -*-
"""hv_leak_control.py: control reruns of the City D work-order leak case with arbitrary sensor sets.

The inverter is the recorded one (demo_leak_inversion: Adam+L1 stage 1, nonlinear OMP + swap
polishing stage 2), untouched. Truth, noise and sensor bookkeeping are rebuilt here from the
recorded recipe (seed 2026 / noise seed 909 / 0.1 ft / lambda 1e-4) and cross-checked against
data/leak_augment_city_d.json before anything runs.

Sensor specs (--spec, several allowed):
  s0                 the 40 census sensors
  demo40             the demo's 40 sensors
  cover:K | dopt:K   S0 + first K of the augmentation order
  rand:K:SEED        S0 + K junctions drawn uniformly from the same 541-junction pool minus S0
  modifiers, appended with ';':
    drop:T:H         remove every non-S0 sensor within H hops of true leak T (T = 1, 2, 3)
    dropall:T:H      remove every sensor (S0 included) within H hops of true leak T
    add:T:H          add every junction within H hops of true leak T (H = 0: the leak node itself)
Each spec writes <out>/<spec>.json (labels T1..T3, hop counts, kinds; no identifiers).
"""
import argparse
import json
import os
import platform
import re
import sys
import time
from collections import deque

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
import demo_leak_inversion as dm  # noqa: E402
from dgga.units import LPSperCFS, MperFT  # noqa: E402

ORTH_TOL = 1e-12


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


def coh_stats(coh, true_idx, lab_of):
    A = np.abs(coh)
    nc = A.shape[0]
    iu, ju = np.triu_indices(nc, 1)
    off = A[iu, ju]
    orth = off < ORTH_TOL
    within = off[~orth]
    B = A.copy()
    np.fill_diagonal(B, -1.0)
    return dict(n_pairs=int(off.size), n_orthogonal_pairs=int(orth.sum()), max_offdiag=float(off.max()),
                median_all=float(np.median(off)), median_within=float(np.median(within)) if within.size else None,
                n_pairs_gt_0999=int((off > 0.999).sum()),
                true_node_max_rival_coh={lab_of[j]: float(B[j].max()) for j in true_idx})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(DATA, "audit_leak_control"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_setup = time.time()
    pb = dm.Problem()
    net, s = pb.net, pb.s
    N = net.N
    junc = np.asarray(s.junc_nodes)
    # ---- recorded recipe replay ----
    rng = np.random.default_rng(dm.SEED)
    others = [n for n in pb.cand if n not in ("195", "3083")]
    third = str(rng.choice(others))
    sens_demo = np.sort(rng.choice(junc, dm.N_SENSOR, replace=False))
    targets = {"195": 3.0, "3083": 1.5, third: 2.2}
    labels = {n: f"T{i + 1}" for i, n in enumerate(targets)}
    tnode = {labels[n]: pb.node_index[n] for n in targets}
    true_idx = [pb.cand.index(n) for n in targets]
    lab_of = {pb.cand.index(n): labels[n] for n in targets}
    o = np.load(os.path.join(DATA, "placement_orders_city_d.npz"))
    fixed = np.asarray(o["augment_fixed"], dtype=np.int64)
    adds = {"dopt": np.asarray(o["augment_dopt"], dtype=np.int64), "cover": np.asarray(o["augment_cover"], dtype=np.int64)}
    s0_nodes = np.sort(junc[fixed])
    rng2 = np.random.default_rng(dm.SEED)
    if not np.array_equal(s0_nodes, np.sort(rng2.choice(junc, 40, replace=False))):
        raise RuntimeError("S0 is not the seed-2026 draw")
    sol_base = pb.fsolve(max_iter=dm.OBS_MI)
    ke_true = np.zeros(N)
    for n, q in targets.items():
        i = pb.node_index[n]
        p_m = (sol_base["head"][:, i] - net.elev_ft[i]).mean() * MperFT
        ke_true[i] = pb.ke_int_of_C(q / p_m ** pb.gamma)
    sol_true = pb.fsolve(ke=ke_true, max_iter=dm.OBS_MI)
    true_lk = {labels[n]: float(sol_true["emitter"][:, pb.node_index[n]].mean() * LPSperCFS) for n in targets}
    with open(os.path.join(DATA, "leak_augment_city_d.json"), "r", encoding="utf-8") as f:
        ref = json.load(f)
    for t, v in ref["config"]["true_leak_lps"].items():
        if abs(true_lk[t] - v) > 1e-9:
            raise RuntimeError(f"true leak flow {t} differs from the record")
    noise_full = dm.NOISE_FT * np.random.default_rng(dm.SEED_NOISE).standard_normal((25, N))
    dist = {t: hop_distances(N, net.link_n1, net.link_n2, i) for t, i in tnode.items()}
    pool = np.array(sorted(set(junc.tolist()) - set(s0_nodes.tolist())), dtype=np.int64)
    print(f"[setup] host={platform.node()} threads={torch.get_num_threads()} nc={pb.nc} "
          f"true flows {true_lk} pool {pool.size}  {time.time() - t_setup:.0f}s")

    for spec in a.spec:
        parts = spec.split(";")
        base = parts[0]
        if base == "s0":
            sens, added = s0_nodes.copy(), np.array([], dtype=np.int64)
        elif base == "demo40":
            sens, added = sens_demo.copy(), np.array([], dtype=np.int64)
        elif base.startswith(("cover:", "dopt:")):
            obj, k = base.split(":")
            added = np.sort(junc[adds[obj][:int(k)]])
            sens = np.sort(np.r_[s0_nodes, added])
        elif base.startswith("rand:"):
            _, k, seed = base.split(":")
            added = np.sort(np.random.default_rng(int(seed)).choice(pool, int(k), replace=False))
            sens = np.sort(np.r_[s0_nodes, added])
        else:
            raise ValueError(base)
        mods = []
        for mod in parts[1:]:
            m = re.fullmatch(r"(drop|dropall|add):([123]):(\d+)", mod)
            if not m:
                raise ValueError(mod)
            what, t, h = m.group(1), f"T{m.group(2)}", int(m.group(3))
            dn = dist[t]
            if what == "drop":
                keep = ~((dn[sens] <= h) & ~np.isin(sens, s0_nodes))
                mods.append(dict(mod=mod, removed=int((~keep).sum())))
                sens, added = sens[keep], added[dn[added] > h]
            elif what == "dropall":
                keep = dn[sens] > h
                mods.append(dict(mod=mod, removed=int((~keep).sum())))
                sens, added = sens[keep], added[dn[added] > h]
            else:
                extra = junc[(dn[junc] <= h)]
                extra = extra[~np.isin(extra, sens)]
                mods.append(dict(mod=mod, added=int(extra.size)))
                sens = np.sort(np.r_[sens, extra])
                added = np.sort(np.r_[added, extra])
        sens = np.unique(sens)
        prox = {}
        for t in ("T1", "T2", "T3"):
            dn = dist[t]
            prox[t] = dict(hop_to_nearest_sensor=int(dn[sens].min()),
                           hop_to_nearest_added=int(dn[added].min()) if added.size else None,
                           n_added_within_1hop=int((dn[added] <= 1).sum()) if added.size else 0,
                           n_added_within_2hops=int((dn[added] <= 2).sum()) if added.size else 0,
                           leak_node_is_sensor=bool(tnode[t] in set(sens.tolist())))
        tag = spec.replace(":", "-").replace(";", "_")
        fp = os.path.join(a.out, f"{tag}.json")
        if os.path.isfile(fp):
            print(f"[{spec}] exists, skip")
            continue
        t0 = time.time()
        n_fwd0 = dm.FWD_COUNT[0]
        elev_s = torch.tensor(net.elev_ft[sens])
        obs = sol_true["head"][:, sens] - net.elev_ft[sens] + noise_full[:, sens]
        pred0 = (sol_base["head"][:, sens] - net.elev_ft[sens]).reshape(-1)
        dm._BASE_CACHE["pred0"] = pred0
        mse0 = float(((obs.reshape(-1) - pred0) ** 2).mean())
        D, Dn, coh = dm.build_dictionary(pb, sol_base, sens)
        cs = coh_stats(coh, true_idx, lab_of)
        print(f"\n[{spec}] sensors {sens.size} (added {added.size}) prox {prox}\n  coh max {cs['max_offdiag']:.6f} "
              f"median {cs['median_all']:.4f} >.999 {cs['n_pairs_gt_0999']} rivals {cs['true_node_max_rival_coh']}")
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
            flow_err[labels[n]] = dict(true_lps=true_lk[labels[n]], est_lps=est,
                                       rel_err=abs(est - true_lk[labels[n]]) / true_lk[labels[n]])
        sup = []
        for j in support:
            if j in lab_of:
                sup.append(dict(kind=lab_of[j], leak_lps=float(fit["leak_lps"][j])))
            else:
                cw = {lab_of[t]: float(abs(coh[j, t])) for t in true_idx}
                best = max(cw, key=cw.get)
                sup.append(dict(kind="nontrue", closest_true=best, coh=cw[best], leak_lps=float(fit["leak_lps"][j])))
        res = dict(spec=spec, modifiers=mods, n_sensors=int(sens.size), n_added=int(added.size),
                   n_in_s0=int(np.isin(sens, s0_nodes).sum()), proximity=prox, mse0=mse0, coherence=cs,
                   stage1=dict(mse=st1["mse"], top3_hits=len(set(st1_top3) & set(true_idx)),
                               top3_kinds=[lab_of.get(j, "nontrue") for j in st1_top3]),
                   support_size=len(support), support=sup,
                   top1_hit=bool(top3 and top3[0] in true_idx), top3_hits=len(set(top3) & set(true_idx)),
                   top3_exact=bool(set(top3) == set(true_idx)), flow_err=flow_err,
                   final_mse_ft2=fit["mse"], time_sec=time.time() - t0,
                   n_forward_solves=dm.FWD_COUNT[0] - n_fwd0, host=platform.node(),
                   threads=torch.get_num_threads(), noise_ft=dm.NOISE_FT, seed_noise=dm.SEED_NOISE, lam=1e-4)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=float)
        print(f"[{spec}] support {[d['kind'] for d in sup]} top1 {res['top1_hit']} top3 {res['top3_hits']}/3 "
              f"T2 rel err {flow_err['T2']['rel_err']:.3f} final_mse {fit['mse']:.3e} {res['time_sec']:.0f}s -> {fp}")


if __name__ == "__main__":
    main()

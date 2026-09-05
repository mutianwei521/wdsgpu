# -*- coding: utf-8 -*-
"""ADVERSARIAL M1 follow-up: what does a "top-3 cluster" claim actually cost?

The report quotes the MEAN cluster radius and the MEAN district pipe length
over ALL clusters.  The operationally meaningful burden of a top-3 claim is the
union of the three clusters you would actually go and inspect.  Recompute it.
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adv_m1 import (my_cluster, geometry, solve_path, ranks)   # noqa: E402

import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
z = np.load(os.path.join(ROOT, "data", "cluster_cache_city_d.npz"))
Dfull, junc, s0 = z["Dfull"], z["junc"], z["s0_pos"].astype(np.int64)
true_col = z["true_col"].astype(np.int64)
dh, noise, sigma = z["dh"], z["noise"], float(z["sigma"])
xy, adj = z["xy_cand"], z["adjacency"]
owner = z["owner"].astype(np.int64)
ln1, ln2, lm = z["link_n1"].astype(np.int64), z["link_n2"].astype(np.int64), z["link_len_m"]
NC = Dfull.shape[2]
sel = np.unique(s0)
A = Dfull[:, sel, :].reshape(-1, NC)
n = np.linalg.norm(A, axis=0)
mu = np.clip(np.abs(A.T @ A) / np.maximum(n[:, None] * n[None, :], 1e-300), 0, 1)
y = (dh[:, junc[sel]] + noise[:, junc[sel]]).reshape(-1)
n_obs = dh.shape[0] * sel.size
L = np.linalg.norm(A, 2) ** 2
total_km = lm.sum() / 1000.0


def district_km(lab, ncl):
    la = np.where(owner[ln1] >= 0, lab[np.clip(owner[ln1], 0, None)], -1)
    lb = np.where(owner[ln2] >= 0, lab[np.clip(owner[ln2], 0, None)], -1)
    s = np.zeros(ncl)
    m = (la == lb) & (la >= 0)
    np.add.at(s, la[m], lm[m])
    m2 = (la != lb) & (la >= 0) & (lb >= 0)
    np.add.at(s, la[m2], 0.5 * lm[m2])
    np.add.at(s, lb[m2], 0.5 * lm[m2])
    return s / 1000.0


rows = []
for tau in [0.9, 0.95, 0.99, 0.995, 0.999]:
    lab, mem = my_cluster(mu, adj, tau)
    ncl = len(mem)
    geo = geometry(mem, xy)
    km = district_km(lab, ncl)
    recs, isel = solve_path(A, y, lab, sigma, n_obs, L)
    rk, order = ranks(recs[isel][2], lab, A, y)
    top3 = list(order[:3])
    covered = int(sum(len(mem[c]) for c in top3))
    km3 = float(km[top3].sum())
    rad3 = [geo[c][2] for c in top3]
    r = dict(tau=tau, n_clusters=ncl, mean_radius_m=float(np.mean([g[2] for g in geo])),
             mean_district_km=float(km.mean()), total_km=float(total_km),
             top3_candidates=covered, top3_district_km=km3,
             top3_share_of_network=km3 / total_km,
             top3_radius_m=[float(v) for v in rad3],
             top1_district_km=float(km[top3[0]]), top1_radius_m=float(rad3[0]))
    rows.append(r)
    print(f"tau={tau:<7g} ncl={ncl:2d} mean_rad={r['mean_radius_m']:7.1f} m "
          f"mean_km={r['mean_district_km']:6.2f}  ||  top-3 clusters cover "
          f"{covered:2d}/{NC} candidates, {km3:6.2f} km = "
          f"{100*km3/total_km:5.1f}% of the {total_km:.1f} km network; "
          f"top-3 radii {[round(v) for v in rad3]} m", flush=True)

json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "adv_m1b.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

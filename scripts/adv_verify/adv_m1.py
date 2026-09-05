# -*- coding: utf-8 -*-
"""ADVERSARIAL M1 - independent re-implementation.

Does NOT import dgga.cluster or scripts/cluster_localisation.py.  Everything
(coherence, constrained complete-linkage clustering, cluster geometry, FISTA
group lasso, deviation-principle lambda) is written from the stated definitions
and checked against the archived numbers.

Extra hostile checks:
  * per-cluster min in-cluster coherence must be >= tau (else tau is cosmetic);
  * cluster radius / diameter recomputed from raw coordinates;
  * the "equal fuzz budget" single-point control is re-done FAIRLY, i.e. the
    budget for a cluster-level top-3 claim is the number of candidates in the
    three top-ranked clusters, not the size of one cluster.
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "data", "cluster_cache_city_d.npz")
ARCH = os.path.join(ROOT, "data", "cluster_localisation.json")
TAUS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1.01]
LABELS = ["T1", "T2", "T3"]


# ---------------------------------------------------------------- clustering
def my_cluster(mu, adj, tau):
    """Constrained complete-linkage agglomeration, written from the definition.

    Deliberately naive: at every step recompute min cross-coherence directly
    from mu over the member sets, so an incremental-update bug upstream cannot
    hide here.
    """
    NC = mu.shape[0]
    members = [[i] for i in range(NC)]
    A = np.array(adj, dtype=bool, copy=True)
    np.fill_diagonal(A, False)
    while True:
        best, bi, bj = -np.inf, -1, -1
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                mi = np.asarray(members[i])
                mj = np.asarray(members[j])
                if not A[np.ix_(mi, mj)].any():
                    continue
                s = float(mu[np.ix_(mi, mj)].min())
                if s > best:
                    best, bi, bj = s, i, j
        if bi < 0 or best < tau:
            break
        members[bi] = members[bi] + members.pop(bj)
    members.sort(key=lambda m: min(m))
    lab = np.empty(NC, dtype=np.int64)
    for c, m in enumerate(members):
        lab[np.asarray(m)] = c
    return lab, members


def geometry(members, xy):
    out = []
    for m in members:
        p = xy[np.asarray(m)]
        if len(m) > 1:
            d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
            diam = float(d.max())
            rad = float(np.hypot(*(p - p.mean(0)).T).max())
        else:
            diam = rad = 0.0
        out.append((len(m), diam, rad))
    return out


# ------------------------------------------------------------------- solver
def fista(A, y, lab, lam, L, x0=None, iters=3000):
    ng = int(lab.max()) + 1
    w = np.sqrt(np.bincount(lab, minlength=ng))
    x = np.zeros(A.shape[1]) if x0 is None else x0.copy()
    z = x.copy()
    t = 1.0
    for _ in range(iters):
        v = z - (A.T @ (A @ z - y)) / L
        vp = np.maximum(v, 0.0)
        nrm = np.sqrt(np.bincount(lab, weights=vp * vp, minlength=ng))
        sc = np.maximum(1.0 - (lam / L) * w / np.maximum(nrm, 1e-300), 0.0)
        xn = sc[lab] * vp
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = xn + ((t - 1.0) / tn) * (xn - x)
        x, t = xn, tn
    return x


def lam_max(A, y, lab):
    ng = int(lab.max()) + 1
    w = np.sqrt(np.bincount(lab, minlength=ng))
    c = np.maximum(A.T @ y, 0.0)
    g = np.sqrt(np.bincount(lab, weights=c * c, minlength=ng))
    return float((g / np.maximum(w, 1e-300)).max())


def solve_path(A, y, lab, sigma, n_obs, L, n_lam=14):
    lm = lam_max(A, y, lab)
    lams = lm * np.logspace(-4, 0, n_lam)[::-1]
    x = None
    recs = []
    for l in lams:
        x = fista(A, y, lab, l, L, x0=x)
        recs.append((float(l), float(((A @ x - y) ** 2).sum()), x.copy()))
    floor = sigma ** 2 * (n_obs + 2.0 * np.sqrt(2.0 * n_obs))
    rmin = min(r[1] for r in recs)
    thr = max(floor, rmin + 2.0 * sigma ** 2 * np.sqrt(2.0 * n_obs))
    ok = [i for i, r in enumerate(recs) if r[1] <= thr]
    return recs, (ok[0] if ok else len(recs) - 1)


def ranks(x, lab, A, y):
    ng = int(lab.max()) + 1
    sc = np.sqrt(np.bincount(lab, weights=x * x, minlength=ng))
    c = np.maximum(A.T @ (y - A @ x), 0.0)
    w = np.sqrt(np.bincount(lab, minlength=ng))
    s2 = np.sqrt(np.bincount(lab, weights=c * c, minlength=ng)) / np.maximum(w, 1e-300)
    order = np.lexsort((-s2, -sc))
    rk = np.empty(ng, dtype=np.int64)
    rk[order] = np.arange(1, ng + 1)
    return rk, order


def main():
    z = np.load(CACHE)
    Dfull, junc, s0 = z["Dfull"], z["junc"], z["s0_pos"].astype(np.int64)
    true_col = z["true_col"].astype(np.int64)
    dh, noise, sigma = z["dh"], z["noise"], float(z["sigma"])
    xy, adj = z["xy_cand"], z["adjacency"]
    NC = Dfull.shape[2]
    sel = np.unique(s0)
    A = Dfull[:, sel, :].reshape(-1, NC)
    n = np.linalg.norm(A, axis=0)
    mu = np.clip(np.abs(A.T @ A) / np.maximum(n[:, None] * n[None, :], 1e-300), 0, 1)
    sens = junc[sel]
    y = (dh[:, sens] + noise[:, sens]).reshape(-1)
    n_obs = dh.shape[0] * sel.size
    L = np.linalg.norm(A, 2) ** 2
    print(f"A {A.shape}  n_obs={n_obs}  sigma={sigma}  maxmu="
          f"{mu[np.triu_indices(NC,1)].max():.6f}  medmu="
          f"{np.median(mu[np.triu_indices(NC,1)]):.4f}")

    arch = json.load(open(ARCH, encoding="utf-8"))["runs"]["city_d"]["S0/noisy"]
    aidx = {e["tau"]: e for e in arch["taus"]}

    # single-point reference (groups = singletons)
    lab1 = np.arange(NC, dtype=np.int64)
    recs1, i1 = solve_path(A, y, lab1, sigma, n_obs, L)
    rk1, ord1 = ranks(recs1[i1][2], lab1, A, y)
    print("single-point ranks of T1/T2/T3:", [int(rk1[t]) for t in true_col])

    rows = []
    for tau in TAUS:
        lab, mem = my_cluster(mu, adj, tau)
        geo = geometry(mem, xy)
        rad = np.array([g[2] for g in geo])
        dia = np.array([g[1] for g in geo])
        recs, isel = solve_path(A, y, lab, sigma, n_obs, L)
        rk, order = ranks(recs[isel][2], lab, A, y)
        tr = [int(rk[lab[t]]) for t in true_col]
        hit3 = sum(r <= 3 for r in tr)
        # fair equal-budget: candidates covered by the top-3 clusters
        budget3 = int(sum(len(mem[c]) for c in order[:3]))
        fair3 = sum(int(rk1[t]) <= budget3 for t in true_col)
        budget1 = int(len(mem[order[0]]))
        a = aidx.get(tau, {})
        rows.append(dict(tau=tau, ncl=len(mem), rad_mean=float(rad.mean()),
                         dia_mean=float(dia.mean()), ranks=tr, hit3=hit3,
                         budget_top3=budget3, point_hit_at_fair_budget=fair3,
                         budget_top1=budget1,
                         arch_ncl=a.get("n_clusters"),
                         arch_rad=a.get("radius_mean_m"),
                         arch_hit3=a.get("hit_top3"),
                         arch_ranks=[p["rank"] for p in a.get("per_leak", [])],
                         arch_point_hit=a.get("point_hit_at_budget")))
        print(f"tau={tau:<8g} ncl={len(mem):3d}(arch {a.get('n_clusters')})  "
              f"rad={rad.mean():8.1f}(arch {a.get('radius_mean_m', float('nan')):8.1f})  "
              f"ranks={tr}(arch {[p['rank'] for p in a.get('per_leak', [])]})  "
              f"top3={hit3}(arch {a.get('hit_top3')})  |  FAIR budget(top3)="
              f"{budget3} -> point hits {fair3}/3   (upstream 'equal budget' "
              f"hits {a.get('point_hit_at_budget')}/3)", flush=True)

    # three clusters inspected in detail at tau = 0.99
    lab, mem = my_cluster(mu, adj, 0.99)
    geo = geometry(mem, xy)
    sizes = [len(m) for m in mem]
    pick = list(np.argsort(sizes)[::-1][:3])
    det = []
    for c in pick:
        m = np.asarray(mem[c])
        sub = mu[np.ix_(m, m)]
        off = sub[~np.eye(len(m), dtype=bool)]
        p = xy[m]
        d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        det.append(dict(size=len(m), min_incluster_mu=float(off.min()),
                        max_incluster_mu=float(off.max()),
                        euclid_diameter_m=float(d.max()),
                        radius_centroid_m=float(np.hypot(*(p - p.mean(0)).T).max()),
                        reported_diameter_m=geo[c][1]))
        print(f"cluster size={len(m):2d}  min in-cluster mu={off.min():.6f} "
              f"(tau=0.99)  diam={d.max():.1f} m  rad={np.hypot(*(p-p.mean(0)).T).max():.1f} m")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adv_m1.json")
    json.dump(dict(rows=rows, detail=det,
                   point_ranks=[int(rk1[t]) for t in true_col]),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written", out)


if __name__ == "__main__":
    main()

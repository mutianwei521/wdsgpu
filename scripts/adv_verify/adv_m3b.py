# -*- coding: utf-8 -*-
"""ADVERSARIAL M3 cross-check: re-rank every arm on SEED-MATCHED pairs, using
module two's own placement-independent metric (RMSE_ident, gamma=2) instead of
the design-dependent sub_rmse the module-three table is built on.

If the ordering of the arms changes when the metric stops depending on the
design, the module-three conclusions are metric-dependent and must say so.
"""
import glob
import json
import os
import sys
from collections import defaultdict
from math import comb

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
z = np.load(os.path.join(ROOT, "data", "placement_cache_city_d.npz"))
S = z["S_full"]
pos_free = np.where(np.abs(S).max(axis=(0, 1)) > 0.0)[0]
A = np.ascontiguousarray(S[:, :, pos_free]).reshape(-1, pos_free.size)
sv, Vt = np.linalg.svd(A, full_matrices=False)[1:]
k_ref = int((sv > 0.10 * np.sqrt(3.0) / 15.0).sum())
C_true = np.random.default_rng(7).uniform(75.0, 145.0, S.shape[2])[pos_free]
print("k_ref =", k_ref, " free =", pos_free.size)


def ident(chat):
    dC = np.asarray(chat, dtype=np.float64) - C_true
    return float(np.linalg.norm(Vt[:k_ref] @ dC) / np.sqrt(k_ref))


arms = defaultdict(dict)          # arm -> seed -> (ident, sub, loss)
for p in glob.glob(os.path.join(ROOT, "data", "optimizer_v2_ov2_*.json")):
    for r in json.load(open(p, encoding="utf-8"))["runs"].values():
        if r.get("budget") == 2000 and r.get("C_hat_free"):
            arms[r["arm"]][int(r["noise_seed"])] = (
                ident(r["C_hat_free"]), r["sub_rmse"], r["loss"])
for p in glob.glob(os.path.join(ROOT, "data", "baselines_v2*.json")) + \
         [os.path.join(ROOT, "data", "baselines_gd.json")]:
    d = json.load(open(p, encoding="utf-8"))
    for k, r in d.get("runs", {}).items():
        if k.startswith("tune") or "city_d" not in k or not r.get("C_hat_free"):
            continue
        tag = f"{k.split('|')[0]}:{r['algo']}"
        arms[tag][int(r["noise_seed"])] = (
            ident(r["C_hat_free"]), np.nan, r.get("best_loss", np.nan))

print(f"\n{'arm':<24}{'n':>3}{'seeds':>26}{'ident(med)':>11}{'sub(med)':>10}{'loss(med)':>12}")
for a in sorted(arms):
    v = arms[a]
    s = sorted(v)
    print(f"{a:<24}{len(s):>3}{str(s)[:25]:>26}"
          f"{np.median([v[i][0] for i in s]):>11.3f}"
          f"{np.nanmedian([v[i][1] for i in s]):>10.3f}"
          f"{np.median([v[i][2] for i in s]):>12.5e}")


def sign_test(a, b, idx=0):
    ca, cb = arms.get(a, {}), arms.get(b, {})
    s = sorted(set(ca) & set(cb))
    if not s:
        return None
    da = np.array([ca[i][idx] for i in s])
    db = np.array([cb[i][idx] for i in s])
    w = int((da < db).sum())
    n = len(s)
    k = min(w, n - w)
    p = min(1.0, 2.0 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)
    return n, w, float(np.median(da)), float(np.median(db)), p, s


print("\n-- seed-matched, on the placement-independent RMSE_ident (gamma=2) --")
for a, b in [("fo_schur", "fo_plain"), ("fo_gnex", "fo_schur"),
             ("fo_ms8_schur", "fo_schur"), ("fo_schur", "gd_ref"),
             ("lm_schur", "gd_ref"), ("lm_ms8_schur", "gd_ref"),
             ("eval:hybrid", "gd_ref"), ("evalold:hybrid", "gd_ref"),
             ("eval2:de", "gd_ref"), ("big:de", "gd_ref"),
             ("eval:hybrid", "fo_schur")]:
    r = sign_test(a, b)
    if r:
        n, w, ma, mb, p, s = r
        print(f"{a:<16} vs {b:<12} n={n:2d} A wins {w:2d}/{n}  "
              f"medA={ma:7.3f} medB={mb:7.3f} p_sign={p:.4f}  seeds={s}")
    else:
        print(f"{a:<16} vs {b:<12} NO COMMON SEEDS")

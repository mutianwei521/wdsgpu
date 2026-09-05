# -*- coding: utf-8 -*-
"""ADVERSARIAL M2 - independent recomputation of RMSE_ident and the exact p.

Nothing from scripts/placement_metric.py is imported.  V_ref is rebuilt by my
own SVD of the pooled sensitivity cache, C_true is regenerated from the stated
RNG recipe, and the randomisation p-value is recomputed by counting.

Hostile checks:
  * do the archived rmse_g2/g3/g10 reproduce from the raw C_hat_free?
  * is the design run comparable to the randoms (same sigma, same noise seed,
    same sensor count, same NFE budget)?
  * exact one-sided p recomputed by counting, for every cell, at every gamma
    and for the unprojected 432-pipe RMSE.
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
SP, SN = 15.0, 0.10

z = np.load(os.path.join(ROOT, "data", "placement_cache_city_d.npz"))
S, pipe_idx = z["S_full"], z["pipe_idx"]
pos_free = np.where(np.abs(S).max(axis=(0, 1)) > 0.0)[0]      # 432 of 475
Sf = np.ascontiguousarray(S[:, :, pos_free])
A = Sf.reshape(-1, Sf.shape[2])
sv, Vt = np.linalg.svd(A, full_matrices=False)[1:]
C_true_all = np.full(len(pipe_idx), np.nan)
C_true_all[:] = np.random.default_rng(7).uniform(75.0, 145.0, len(pipe_idx))
C_true = C_true_all[pos_free]
print(f"pool sensitivity {A.shape}; free pipes {pos_free.size}")
for g in (2.0, 3.0, 10.0):
    thr = SN * np.sqrt(g * g - 1.0) / SP
    print(f"  gamma={g:<4g} thr={thr:.6f}  k_ref={int((sv > thr).sum())}")


def rmse(dC, g):
    k = int((sv > SN * np.sqrt(g * g - 1.0) / SP).sum())
    return float(np.linalg.norm(Vt[:k] @ dC) / np.sqrt(k)), k


d = json.load(open(os.path.join(ROOT, "data", "calib_augrand_city_d.json"),
                   encoding="utf-8"))["runs"]

# ---- 1. does the archived metric reproduce from raw C_hat?
worst = defaultdict(float)
for key, r in d.items():
    dC = np.asarray(r["C_hat_free"]) - C_true
    for g, fld in ((2.0, "rmse_g2"), (3.0, "rmse_g3"), (10.0, "rmse_g10")):
        v, k = rmse(dC, g)
        worst[fld] = max(worst[fld], abs(v - r[fld]) / max(abs(r[fld]), 1e-12))
        if k != r["k_g%g" % g]:
            print("!! k mismatch", key, g, k, r["k_g%g" % g])
    va = float(np.linalg.norm(dC) / np.sqrt(len(dC)))
    worst["rmse_all"] = max(worst["rmse_all"],
                            abs(va - r.get("frozen_mae_vs_truth", va)) * 0.0)
print("max relative deviation, my recompute vs archive:",
      {k: f"{v:.3e}" for k, v in worst.items() if k != "rmse_all"})

# ---- 2. exact one-sided p, recomputed by counting
CELLS = [(20, 0.03), (20, 0.1), (20, 0.3), (40, 0.03), (40, 0.1), (40, 0.3)]
rows = []
for budget, sig in CELLS:
    rnd = {k: v for k, v in d.items()
           if re.match(rf"RAND_augrand{budget}_s{sig}_n100_r\d+$", k)}
    des = {k: v for k, v in d.items()
           if re.match(rf"DES_(augcover|augdopt){budget}_s{sig}_n100$", k)}
    if not rnd or not des:
        print(f"cell +{budget} sigma={sig}: rnd={len(rnd)} des={len(des)}  SKIP")
        continue
    # comparability audit
    fields = ("sigma", "noise_seed", "n_sensors", "lam", "nfe",
              "adam_steps", "n_train_sensors")
    bad = []
    ref = {f: des[list(des)[0]][f] for f in fields}
    for k, v in list(rnd.items()) + list(des.items()):
        for f in fields:
            if v[f] != ref[f]:
                bad.append((k, f, v[f], ref[f]))
    R = len(rnd)
    for dk, dv in sorted(des.items()):
        dCd = np.asarray(dv["C_hat_free"]) - C_true
        line = {"cell": f"+{budget}/sigma={sig}", "design": dk, "R": R,
                "mismatched_fields": bad[:6]}
        for tag, g in (("g2", 2.0), ("g3", 3.0), ("g10", 10.0)):
            dval = rmse(dCd, g)[0]
            rv = [rmse(np.asarray(v["C_hat_free"]) - C_true, g)[0]
                  for v in rnd.values()]
            cnt = int(sum(1 for x in rv if x <= dval))
            line[tag] = dict(design=dval, rand_min=min(rv),
                             rand_med=float(np.median(rv)), better=cnt,
                             p=(1 + cnt) / (R + 1))
        dval = float(np.linalg.norm(dCd) / np.sqrt(len(dCd)))
        rv = [float(np.linalg.norm(np.asarray(v["C_hat_free"]) - C_true)
                    / np.sqrt(len(C_true))) for v in rnd.values()]
        cnt = int(sum(1 for x in rv if x <= dval))
        line["raw432"] = dict(design=dval, rand_min=min(rv), better=cnt,
                              p=(1 + cnt) / (R + 1))
        rows.append(line)
        print(f"+{budget} s={sig} {dk:<28s} R={R:3d} | "
              f"g2 {line['g2']['design']:7.3f} vs rand min {line['g2']['rand_min']:7.3f} "
              f"med {line['g2']['rand_med']:7.3f}  better={line['g2']['better']} "
              f"p={line['g2']['p']:.4f} | g10 p={line['g10']['p']:.4f} | "
              f"raw432 p={line['raw432']['p']:.4f}"
              + ("   [MISMATCH]" if bad else ""), flush=True)
    if bad:
        print("   comparability mismatches:", bad[:6])

json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "adv_m2.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("done")

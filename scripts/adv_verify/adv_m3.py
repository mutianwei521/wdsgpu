# -*- coding: utf-8 -*-
"""ADVERSARIAL M3 - independent audit of the call accounting and the timing.

Checks, none of them using scripts/optimizer_v2_report.py:
  1. does the recorded nfe equal B x steps for the multi-start arms, i.e. is a
     batched start actually charged B calls, as claimed?
  2. are the arms compared on the SAME seed set?  (the report quotes medians
     with n = 4, 5 and 10.)
  3. per-arm wall clock / nfe, to see whether any arm is timed on a different
     device or with a different warmup.
  4. re-derive the headline comparisons on the seed intersection only.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
runs = {}
for p in glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "optimizer_v2_ov2_*.json")):
    runs.update(json.load(open(p, encoding="utf-8"))["runs"])

by = defaultdict(dict)
for k, r in runs.items():
    if r.get("budget") != 2000:
        continue
    by[r["arm"]][int(r["noise_seed"])] = r

print(f"{'arm':<16}{'n':>3} {'seeds':<34}{'nfe':>7}{'nbwd':>7}"
      f"{'loss(med)':>12}{'sub(med)':>9}{'wall(med)':>10}{'s/NFE':>8} dev")
info = {}
for arm in sorted(by):
    rs = by[arm]
    seeds = sorted(rs)
    nfe = sorted({r["nfe"] for r in rs.values()})
    loss = float(np.median([r["loss"] for r in rs.values()]))
    sub = float(np.median([r["sub_rmse"] for r in rs.values()]))
    wall = float(np.median([r["wall_sec"] for r in rs.values()]))
    dev = sorted({str(r.get("device")) for r in rs.values()})
    info[arm] = dict(seeds=seeds, loss=loss, sub=sub, wall=wall)
    print(f"{arm:<16}{len(rs):>3} {str(seeds)[:33]:<34}{nfe[0]:>7}"
          f"{sorted({r['nbwd'] for r in rs.values()})[0]:>7}{loss:>12.5e}"
          f"{sub:>9.3f}{wall:>10.1f}{wall/max(nfe[0],1):>8.4f} {','.join(dev)}")

# ---- 1. B x steps accounting
print("\n-- multi-start call accounting (detail field) --")
for arm in sorted(by):
    r = list(by[arm].values())[0]
    det = r.get("detail") or {}
    keys = {k: det[k] for k in det
            if any(t in k.lower() for t in ("b", "step", "start", "iter", "nfe"))}
    print(f"{arm:<16} nfe={r['nfe']:>5} nbwd={r['nbwd']:>5}  detail={json.dumps(keys, ensure_ascii=False)[:190]}")

# ---- 2/4. common-seed re-derivation of the headline comparisons
def paired(a, b, field="sub_rmse"):
    ca = by.get(a, {})
    cb = by.get(b, {})
    s = sorted(set(ca) & set(cb))
    if not s:
        return None
    da = np.array([ca[i][field] for i in s])
    db = np.array([cb[i][field] for i in s])
    wins = int((da < db).sum())
    # two-sided exact sign test
    from math import comb
    n = len(s)
    k = min(wins, n - wins)
    p = min(1.0, 2.0 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)
    return dict(n=n, seeds=s, wins_a=wins, med_a=float(np.median(da)),
                med_b=float(np.median(db)), med_diff=float(np.median(da - db)),
                p_sign=p)

print("\n-- headline comparisons re-derived on the seed INTERSECTION --")
for a, b in [("fo_schur", "fo_plain"), ("fo_gnex", "fo_plain"),
             ("fo_gnex", "fo_schur"), ("fo_ms8_schur", "fo_schur"),
             ("fo_ms16_schur", "fo_schur"), ("fo_ms8", "fo_plain"),
             ("lm_schur", "lm_plain"), ("lm_ms8_schur", "lm_plain")]:
    r = paired(a, b)
    if r:
        print(f"{a:<15} vs {b:<12} n={r['n']:2d} A wins {r['wins_a']:2d} "
              f"medA={r['med_a']:7.3f} medB={r['med_b']:7.3f} "
              f"medDiff={r['med_diff']:+8.4f} p_sign={r['p_sign']:.4f} "
              f"seeds={r['seeds']}")

print("\n-- best arm vs the frozen 200-call gradient config, per-seed --")
ref = by.get("gd_ref", {})
for arm in sorted(by):
    if arm == "gd_ref":
        continue
    s = sorted(set(ref) & set(by[arm]))
    if not s:
        continue
    da = np.array([by[arm][i]["sub_rmse"] for i in s])
    db = np.array([ref[i]["sub_rmse"] for i in s])
    print(f"{arm:<16} common seeds {len(s)}  arm med {np.median(da):7.3f}  "
          f"gd_ref med {np.median(db):7.3f}  arm wins {(da < db).sum()}/{len(s)}  "
          f"nfe {list(by[arm].values())[0]['nfe']} vs {list(ref.values())[0]['nfe']}")

json.dump({k: {kk: (vv if kk != "seeds" else vv) for kk, vv in v.items()}
           for k, v in info.items()},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "adv_m3.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

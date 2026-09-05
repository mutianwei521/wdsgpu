# -*- coding: utf-8 -*-
"""独立审计 step0：扫描全部参考网，找出 dense 可构造者，并统计拓扑病态结构。"""
import os, sys, glob
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dgga.parse import Net
from dgga.solver import GGASolver

REF = os.path.join(ROOT, "data", "reference")
stems = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(REF, "*_net.npz")))

rows = []
for st in stems:
    net = Net.load(REF, st)
    nt = np.asarray(net.node_type)
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    Nj = int((nt == 0).sum())
    # 并联管：同一无序节点对出现 >1 条链路
    key = np.minimum(n1, n2) * (len(nt) + 1) + np.maximum(n1, n2)
    uk, cnt = np.unique(key, return_counts=True)
    npar = int((cnt > 1).sum())
    nparlinks = int(cnt[cnt > 1].sum())
    nself = int((n1 == n2).sum())
    deg = np.bincount(np.concatenate([n1, n2]), minlength=len(nt))
    iso = int(((nt == 0) & (deg == 0)).sum())
    ok = "-"
    try:
        GGASolver(net, mode="dense")
        ok = "dense"
    except Exception as e:
        try:
            GGASolver(net, mode="dense", dense_tank_bound_check=False)
            ok = "dense*"
        except Exception as e2:
            ok = type(e2).__name__ + ":" + str(e2)[:40]
    rows.append((st, len(nt), Nj, len(n1), npar, nparlinks, nself, iso, ok))

print(f"{'stem':28s} {'N':>6s} {'Nj':>6s} {'L':>6s} {'par对':>5s} {'par链':>5s} {'self':>4s} {'iso':>4s}  dense")
for r in rows:
    print(f"{r[0]:28s} {r[1]:6d} {r[2]:6d} {r[3]:6d} {r[4]:5d} {r[5]:5d} {r[6]:4d} {r[7]:4d}  {r[8]}")

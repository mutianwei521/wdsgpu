# -*- coding: utf-8 -*-
"""count_unlock.py - PRV 轮验收 ④：dense 可跑网数的变化。

口径：
  A. `_cleaned` 版 21 公开网（networks/public，有清洗版用清洗版）；
  B. networks/EXAMPLE 的 82 个拓扑唯一网（p5 盘点的 topo 指纹分组，
     scripts/p5_bignet/dedup_topo.json，每组取字典序第一个可解析文件）。

每网实测：GGASolver(mode="dense", dense_status_machine=True) 构造 + B=1 名义帧
（t=0，水池取上下限中点）真解一发（Nj>12000 只构造不解，survey 同口径；含
CVPIPE/PRV 网自动 status_machine=True）。

"本轮前"数不用猜：PRV 网在旧准入下构造期必 raise（其它网行为逐位不变，影子包
已证），所以 before = after 里剔除"含 PRV 且现在能构造"的网。

用法：python -X utf8 scripts/prv_port/count_unlock.py
"""

import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch                                    # noqa: E402
from dgga.parse import parse_inp                # noqa: E402
from dgga.solver import GGASolver               # noqa: E402

PUB = os.path.join(ROOT, "networks", "public")
CLEAN = os.path.join(PUB, "_cleaned")
NJ_CAP = 12000
LT = ["CVPIPE", "PIPE", "PUMP", "PRV", "PSV", "PBV", "FCV", "TCV", "GPV"]


def probe(path):
    """返回 dict(build, solve, has_prv, blockers, Nj)。"""
    try:
        net = parse_inp(path)
    except Exception as e:
        return dict(build="parseFAIL", solve="-", has_prv=False,
                    why=type(e).__name__, Nj=-1)
    lt = np.asarray(net.link_type)
    has_prv = bool((lt == 3).any())
    nt = np.asarray(net.node_type)
    Nj = int((nt == 0).sum())
    try:
        s = GGASolver(net, mode="dense", inp_path=path,
                      dense_status_machine=True)
    except Exception as e:
        why = str(e)
        short = why[:60]
        return dict(build="FAIL", solve="-", has_prv=has_prv, why=short, Nj=Nj)
    if Nj > NJ_CAP:
        return dict(build="ok", solve="skip(Nj>cap)", has_prv=has_prv,
                    why="", Nj=Nj)
    try:
        d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
        rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                     dtype=np.float64))
        tn = np.asarray(net.tank_node, dtype=np.int64)
        if tn.size:
            rh0[tn] = .5 * (np.asarray(net.tank_hmin)[:tn.size]
                            + np.asarray(net.tank_hmax)[:tn.size])
        sm = bool((lt == 0).any() or has_prv or s.n_pumps or s.n_tanks)
        with torch.no_grad():
            out = s.solve(d0[None, :], rh0[None, :], status_machine=sm)
        conv = bool(out["converged"][0])
        return dict(build="ok", solve="ok" if conv else "NOTCONV",
                    has_prv=has_prv, why="", Nj=Nj)
    except Exception as e:
        return dict(build="ok", solve="FAIL", has_prv=has_prv,
                    why=str(e)[:60], Nj=Nj)


def report(tag, items):
    after_b = sum(1 for _, r in items if r["build"] == "ok")
    after_s = sum(1 for _, r in items
                  if r["solve"] in ("ok", "skip(Nj>cap)"))
    prv_new = [(n, r) for n, r in items
               if r["build"] == "ok" and r["has_prv"]]
    before_b = after_b - len(prv_new)
    before_s = after_s - sum(1 for _, r in prv_new
                             if r["solve"] in ("ok", "skip(Nj>cap)"))
    print("\n【%s】共 %d 网" % (tag, len(items)))
    print("  dense 可构造：%d → %d（+%d，全部因 PRV 准入）" %
          (before_b, after_b, after_b - before_b))
    print("  dense 真解出（含 Nj>cap 只构造）：%d → %d（+%d）" %
          (before_s, after_s, after_s - before_s))
    print("  新解锁明细：")
    for n, r in prv_new:
        print("    %-22s Nj=%-6d solve=%s %s" %
              (n, r["Nj"], r["solve"], r["why"]))
    still = [(n, r) for n, r in items if r["build"] not in ("ok",)]
    print("  仍进不去（前 12）：")
    for n, r in still[:12]:
        print("    %-22s %s: %s" % (n, r["build"], r["why"]))
    return dict(before_b=before_b, after_b=after_b,
                before_s=before_s, after_s=after_s)


def main():
    # ---- A. 21 公开网 ----
    pubs = sorted(f[:-4] for f in os.listdir(PUB) if f.endswith(".inp"))
    items_a = []
    for nm in pubs:
        p = os.path.join(CLEAN, nm + ".inp")
        if not os.path.exists(p):
            p = os.path.join(PUB, nm + ".inp")
        items_a.append((nm, probe(p)))
        print("  [pub] %-22s build=%-6s solve=%s" %
              (nm, items_a[-1][1]["build"], items_a[-1][1]["solve"]),
              flush=True)
    ra = report("A. _cleaned 口径 21 公开网", items_a)

    # ---- B. EXAMPLE 82 拓扑唯一网 ----
    with open(os.path.join(ROOT, "scripts", "p5_bignet", "dedup_topo.json"),
              encoding="utf-8") as f:
        recs = json.load(f)
    groups = {}
    for r in recs:
        if r.get("parse") != "ok":
            continue
        groups.setdefault(r["topo"], []).append(r["path"])
    reps = sorted(min(v) for v in groups.values())
    print("\nEXAMPLE 拓扑唯一网 %d 个（dedup_topo.json，parse=ok 分组取首）"
          % len(reps))
    items_b = []
    for rel in reps:
        p = os.path.join(ROOT, rel)
        nm = os.path.basename(rel)[:-4]
        items_b.append((nm, probe(p)))
        print("  [exa] %-22s Nj=%-6d build=%-6s solve=%s" %
              (nm, items_b[-1][1]["Nj"], items_b[-1][1]["build"],
               items_b[-1][1]["solve"]), flush=True)
    rb = report("B. EXAMPLE 拓扑唯一网", items_b)
    print("\n汇总：A %d→%d 构造 / %d→%d 解；B %d→%d 构造 / %d→%d 解" %
          (ra["before_b"], ra["after_b"], ra["before_s"], ra["after_s"],
           rb["before_b"], rb["after_b"], rb["before_s"], rb["after_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

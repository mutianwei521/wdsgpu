# -*- coding: utf-8 -*-
"""任务一出表：把 survey_inp.json + dedup_topo.json 合成按 Nj 降序的准入总表。"""
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))

S = json.load(open(os.path.join(HERE, "survey_inp.json"), encoding="utf-8"))
D = {r["path"]: r for r in json.load(open(os.path.join(HERE, "dedup_topo.json"),
                                          encoding="utf-8"))}

LN = ["CVPIPE", "PIPE", "PUMP", "PRV", "PSV", "PBV", "FCV", "TCV", "GPV"]


def linkstr(d):
    return "+".join(f"{k}{v}" for k, v in sorted(d.items(),
                                                 key=lambda x: LN.index(x[0])))


ok = [r for r in S if r["parse"] == "ok"]
bad = [r for r in S if r["parse"] != "ok"]

# 拓扑分组：同一拓扑只在表里出一行主记录，其余列为"同网副本"
grp = collections.defaultdict(list)
for r in ok:
    grp[D[r["path"]].get("topo", r["sha256"])].append(r)

print("=" * 128)
print("表 1  networks/EXAMPLE 准入总表（按 Nj 降序；拓扑去重后每网一行）")
print("=" * 128)
print(f"{'Nj':>6} {'L':>6} {'水池':>4} {'水库':>4} {'水损':>4} "
      f"{'epanet构造':>10} {'dense构造':>9} {'守卫':>4} {'解出':>8} {'iter':>4} "
      f"{'ms':>8}  链路构成 / 网名")
print("-" * 128)

rows = []
for topo, rs in grp.items():
    rs = sorted(rs, key=lambda r: r["path"])
    m = rs[0]
    for r in rs:                       # 主记录取"能跑最远"的那份
        if r.get("solve_dense") == "ok" and m.get("solve_dense") != "ok":
            m = r
    rows.append((m, rs))
rows.sort(key=lambda x: (-x[0]["Nj"], x[0]["path"]))

n_reach = n_solve = 0
for m, rs in rows:
    dz = m.get("build_dense")
    dz2 = m.get("build_dense_noguard")
    dense = "ok" if dz == "ok" else ("ok*" if dz2 == "ok" else "-")
    guard = "关" if m["dense_needs_noguard"] else ("开" if dense.startswith("ok") else "-")
    sv = m.get("solve_dense", "-")
    if dense.startswith("ok"):
        n_reach += 1
    if sv == "ok":
        n_solve += 1
    nm = os.path.basename(m["path"])
    extra = ""
    if len(rs) > 1:
        extra = "   [同网副本 %d 份: %s]" % (
            len(rs), ", ".join(os.path.basename(r["path"]) for r in rs if r is not m))
    print(f"{m['Nj']:>6} {m['L']:>6} {m['n_tank']:>4} {m['n_res']:>4} "
          f"{m['headloss']:>4} {m.get('build_epanet','-'):>10} {dense:>9} "
          f"{guard:>4} {sv:>8} {m.get('iters_dense',-1):>4} "
          f"{m.get('solve_ms',float('nan')):>8.1f}  {linkstr(m['links'])} / {nm}{extra}")

print("-" * 128)
print(f"拓扑唯一网 {len(rows)} 个（可解析文件 {len(ok)} / 全部 {len(S)}）；"
      f"dense 可构造 {n_reach} 网，**dense 真的解出来 {n_solve} 网**")
print("注：dense 列 'ok*' = 必须 dense_tank_bound_check=False；守卫列'关'即该网需关水池守卫。")
print("    'iter'/'ms' 是本机 CPU 单帧 B=1 的实测（GGASolver dense，float64）。")

print()
print("=" * 128)
print("表 2  parse 失败的文件（%d 个） - 分因" % len(bad))
print("=" * 128)
byerr = collections.defaultdict(list)
for r in bad:
    byerr[r["parse_err"].split(":")[0]].append(r)
for k in sorted(byerr, key=lambda k: -len(byerr[k])):
    print(f"\n--- {k}（{len(byerr[k])} 个）")
    for r in byerr[k]:
        print(f"    {r['path']}")
        print(f"        {r['parse_err'][:180]}")

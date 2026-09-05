# -*- coding: utf-8 -*-
"""把多节点的 p4r_time_gpu.py / p4r_mem_gpu.py 原始输出合成**区间表**。

审计 R6：凡引用倍数至少两节点，写成区间并标节点名。本脚本只读 data/gpu/ 下的
原始 .out，不重算任何数；输出直接贴进 data/p4_remeasure_wip.txt。

用法：
    python scripts/p4_remeasure/p4r_merge.py time  data/gpu/5090_p4rt_*.out
    python scripts/p4_remeasure/p4r_merge.py mem   data/gpu/5090_p4rm_*.out
"""
import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
KIND = sys.argv[1]
FILES = sys.argv[2:]
NETORD = ["Net1", "Anytown", "Hanoi", "Net2", "Fossolo", "Pescara", "Net3",
          "Modena", "City_D", "ky4"]
BORD = [1, 8, 64, 256, 512, 1024]


def node_of(txt):
    m = re.search(r"node:\s*(\S+)", txt)
    return m.group(1) if m else "?"


def num(x):
    return None if x.strip() in ("OOM", "-", "") else float(x)


def rng(vals, fmt="%.2f", suffix="x"):
    v = [x for x in vals if x is not None]
    if not v:
        return "OOM"
    lo, hi = min(v), max(v)
    if abs(hi - lo) < 5e-3 * max(abs(hi), 1e-30):
        return (fmt + suffix) % hi
    return (fmt + "~" + fmt + suffix) % (lo, hi)


T1, T2, M1 = {}, {}, {}
NODES = []
for f in FILES:
    txt = io.open(f, encoding="utf-8", errors="replace").read()
    nd = node_of(txt)
    NODES.append(nd)
    for ln in txt.splitlines():
        p = [x.strip() for x in ln.split("|")]
        c0 = p[0].split()
        if KIND == "time" and len(p) == 3 and len(c0) == 4 and c0[0] in NETORD:
            stem, Nj, B, K = c0[0], int(c0[1]), int(c0[2]), int(c0[3])
            a = p[1].split()
            b = p[2].split()
            if len(a) >= 3 and len(b) >= 3:
                T1.setdefault((stem, Nj, B, K), []).append(
                    (nd, num(a[0]), num(a[1]), num(a[2].rstrip("x")),
                     num(b[0]), num(b[1]), num(b[2].rstrip("x"))))
        if KIND == "time" and len(p) == 5 and len(c0) == 3 and c0[0] in NETORD:
            stem, Nj, B = c0[0], int(c0[1]), int(c0[2])
            r = p[4].split()
            if len(r) == 3:
                T2.setdefault((stem, Nj, B), []).append(
                    (nd, num(p[1]), num(p[2]), num(p[3]),
                     num(r[0].rstrip("x")), num(r[1].rstrip("x")),
                     num(r[2].rstrip("x"))))
        if KIND == "mem" and len(p) == 4 and len(c0) == 2 and c0[0] in NETORD:
            stem, B = c0[0], int(c0[1])
            d = p[1].split()
            c = p[2].split()
            dv = [num(x) for x in d] if len(d) == 4 else [None] * 4
            cv = [num(x) for x in c] if len(c) == 4 else [None] * 4
            M1.setdefault((stem, B), []).append((nd, dv, cv, p[1].strip(),
                                                 p[2].strip()))

print("节点：", " / ".join(sorted(set(NODES))), " （文件 %d 个）" % len(FILES))
print()
if KIND == "time":
    print("§T1 区间表（ms/场景，多节点 min~max；倍数 = dense/cudss）")
    print("net      Nj    B     K  | fwd dense       fwd cudss      倍数        "
          "| f+b dense       f+b cudss      倍数")
    for stem in NETORD:
        for B in BORD:
            for key in list(T1):
                if key[0] == stem and key[2] == B:
                    rows = T1[key]
                    print("%-8s %-5d %-5d %-2d | %-15s %-14s %-11s | %-15s %-14s %s"
                          % (stem, key[1], B, key[3],
                             rng([r[1] for r in rows], "%.5f", ""),
                             rng([r[2] for r in rows], "%.5f", ""),
                             rng([r[3] for r in rows]),
                             rng([r[4] for r in rows], "%.5f", ""),
                             rng([r[5] for r in rows], "%.5f", ""),
                             rng([r[6] for r in rows])))
    print()
    print("§T2 区间表（一轮线代 f+b，ms/轮，整批）")
    print("net      Nj    B     | A 稠密+autograd   Badj 稠密+手写伴随  C cudss"
          "        | Badj/C 稀疏赢   A/Badj 通用autograd  A/C 仓库现状")
    for stem in NETORD:
        for B in BORD:
            for key in list(T2):
                if key[0] == stem and key[2] == B:
                    rows = T2[key]
                    print("%-8s %-5d %-5d | %-17s %-19s %-14s | %-15s %-20s %s"
                          % (stem, key[1], B,
                             rng([r[1] for r in rows], "%.4f", ""),
                             rng([r[2] for r in rows], "%.4f", ""),
                             rng([r[3] for r in rows], "%.4f", ""),
                             rng([r[4] for r in rows]),
                             rng([r[5] for r in rows]),
                             rng([r[6] for r in rows])))
else:
    print("§M1 区间表（R5 口径 MiB；多节点逐位一致时只写单值）")
    print("net      B     | dense peak/resv/非torch/total          | "
          "cudss peak/resv/非torch/total        | total 倍数")
    for stem in NETORD:
        for B in BORD:
            rows = M1.get((stem, B))
            if not rows:
                continue
            same_d = len(set(r[3] for r in rows)) == 1
            same_c = len(set(r[4] for r in rows)) == 1

            def show(idx, same):
                if same:
                    return rows[0][3 if idx == 0 else 4]
                return " / ".join("[%s]%s" % (r[0], r[3 if idx == 0 else 4])
                                  for r in rows)
            rat = rng([(r[1][3] / r[2][3]) if (r[1][3] and r[2][3]) else None
                       for r in rows])
            print("%-8s %-5d | %-38s | %-36s | %s%s"
                  % (stem, B, show(0, same_d), show(1, same_c), rat,
                     "" if (same_d and same_c) else "   <<跨节点不一致"))

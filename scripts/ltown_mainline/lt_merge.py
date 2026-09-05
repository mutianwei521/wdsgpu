# -*- coding: utf-8 -*-
"""lt_merge.py - 把两节点的 ltt/ltm .out 合成区间表（只读，不重算任何数）。

用法：python -X utf8 scripts/ltown_mainline/lt_merge.py data/gpu/5090_ltt_*.out
                                                       data/gpu/5090_ltm_*.out
"""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

TT, TM = [], []
for p in sys.argv[1:]:
    (TT if "_ltt_" in p else TM).append(p)


def parse_tt(path):
    node = None
    rows = {}
    dec = {}
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"node: (\S+)", ln)
        if m:
            node = m.group(1)
        m = re.match(r"\s+B=(\d+)\s+fwd dense\s+([\d.]+|OOM)\s+cudss\s+([\d.]+|OOM)"
                     r"\s+\| impl\(共享CPU\)\s+([\d.]+|OOM)\s+ms/场景"
                     r" \(状态组 (\d+), 两路状态同 (-?\d+)/(\d+), dense收敛 (-?\d+)/(\d+)\)"
                     r" \| f\+b dense\s+([\d.]+|OOM)\s+cudss\s+([\d.]+|OOM)", ln)
        if m:
            g = m.groups()
            rows[int(g[0])] = dict(
                fd=g[1], fc=g[2], ti=g[3], ngrp=int(g[4]), steq=g[5],
                fbd=g[9], fbc=g[10])
        m = re.match(r"\s+分解 一轮线代 f\+b\(ms/轮\): A\s+([\d.]+|OOM)\s+\| Badj\s+"
                     r"([\d.]+|OOM)\s+\| C\s+([\d.]+)\s+\|.*gA一致 \(B/A\) (\S+) \(C/A\) (\S+)", ln)
        if m:
            dec[max(rows) if rows else -1] = m.groups()
    return node, rows, dec


def parse_tm(path):
    node = None
    cells = {}
    for ln in open(path, encoding="utf-8"):
        m = re.match(r".*node: (\S+)", ln)
        if m and node is None:
            node = m.group(1)
        m = re.match(r"RESULT (\d+) (\w+) (ok|OOM|ERR)", ln)
        if m:
            B, mode, st = int(m.group(1)), m.group(2), m.group(3)
            if st == "ok":
                kv = dict(x.split("=", 1) for x in ln.split() if "=" in x)
                cells[(B, mode)] = kv
            else:
                cells[(B, mode)] = st
    return node, cells


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


if TT:
    P = [parse_tt(p) for p in TT]
    print("== 时间（ms/场景），区间 = %s ==" % " / ".join(n for n, _, _ in P))
    print("B     | fwd dense        | fwd cudss        | 倍数        | impl(CPU伴随) | f+b dense | f+b cudss | f+b 倍数")
    for B in sorted(P[0][1]):
        rs = [p[1].get(B, {}) for p in P]
        def rng(k, fmt="%.5f"):
            vs = [fnum(r.get(k)) for r in rs]
            if any(v is None for v in vs):
                return "OOM/ERR"
            lo, hi = min(vs), max(vs)
            return (fmt % lo) if abs(hi - lo) < 5e-6 else (fmt + "~" + fmt) % (lo, hi)
        def ratio(k1, k2):
            vs = [(fnum(r.get(k1)), fnum(r.get(k2))) for r in rs]
            if any(a is None or b is None for a, b in vs):
                return "-"
            qs = [a / b for a, b in vs]
            lo, hi = min(qs), max(qs)
            return "%.2fx" % lo if abs(hi - lo) < 5e-3 else "%.2fx~%.2fx" % (lo, hi)
        print("%-5d | %-16s | %-16s | %-11s | %-13s | %-9s | %-9s | %s"
              % (B, rng("fd"), rng("fc"), ratio("fd", "fc"), rng("ti"),
                 rng("fbd"), rng("fbc"), ratio("fbd", "fbc")))
    print()
    print("== 分解（ms/轮，A/Badj/C 与两因子），逐节点原样 ==")
    for node, rows, dec in P:
        print("-- %s" % node)
        for B in sorted(dec):
            a, b, c, rab, rac = dec[B]
            fa, fb, fc = fnum(a), fnum(b), fnum(c)
            print("  B=%-5d A %10s Badj %8s C %8s | Badj/C %6s A/Badj %6s A/C %6s | gA(B/A) %s (C/A) %s"
                  % (B, a, b, c,
                     "-" if (fb is None) else "%.2fx" % (fb / fc),
                     "-" if (fa is None or fb is None) else "%.2fx" % (fa / fb),
                     "-" if fa is None else "%.2fx" % (fa / fc), rab, rac))

if TM:
    P = [parse_tm(p) for p in TM]
    print()
    print("== 显存（R5，MiB），节点 = %s ==" % " / ".join(n for n, _ in P))
    keys = sorted({k for _, c in P for k in c if k[1] in ("dense", "cudss")})
    Bs = sorted({k[0] for k in keys})
    print("B     | dense total      | cudss total      | 倍数   | 两节点逐位?")
    for B in Bs:
        vals = {}
        bitsame = True
        for mode in ("dense", "cudss"):
            cs = [c.get((B, mode)) for _, c in P]
            cs = [c for c in cs if c is not None]
            if not cs:
                vals[mode] = None
                continue
            if all(isinstance(c, dict) for c in cs):
                ts = [fnum(c["total"]) for c in cs]
                fours = [(c["torch_peak"], c["torch_resv"], c["nontorch"],
                          c["total"]) for c in cs]
                bitsame &= all(f == fours[0] for f in fours)
                vals[mode] = ts[0] if max(ts) - min(ts) < 1e-6 else \
                    "%.0f~%.0f" % (min(ts), max(ts))
            else:
                vals[mode] = cs[0] if not isinstance(cs[0], dict) else "MIX"
                bitsame = False
        d, c_ = vals.get("dense"), vals.get("cudss")
        r = "-"
        if isinstance(d, float) and isinstance(c_, float):
            r = "%.2fx" % (d / c_)
        print("%-5d | %-16s | %-16s | %-6s | %s"
              % (B, d, c_, r, "是" if bitsame and len(P) > 1 else
                 ("单节点" if len(P) == 1 else "否")))
    print()
    for node, c in P:
        im = {k: v for k, v in c.items() if k[1] == "impl_dense"}
        for (B, _), v in sorted(im.items()):
            print("impl_dense %s B=%-5d %s" % (node, B,
                  v if not isinstance(v, dict) else " ".join(
                      "%s=%s" % kv for kv in v.items())))

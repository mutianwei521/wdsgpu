# -*- coding: utf-8 -*-
"""XF-4（本机前台）：从原始 .out **自己重算** README 的两句数字与分解表两个口径。

不看任何 wip / 不调用 scripts/p4_closeout/* - 只认 data/gpu/*.out 里的
原始数值，而且**只取 ms 原始值自己算比值**（不抄它印出来的 x 倍数），
这样连"比值是不是它自己算错的"也一并查了。

四问：
  Q1 一致性区间（全网格 / README 分解表那 8 格）到底是多少，极值在哪一格；
     旧印的 4.3e-16~7.2e-09 被多少个测量突破。
  Q2 README 印的 "3.3e-16 to 5.3e-08"、"1.9e-13 to 3.9e-08" 对不对。
  Q3 分解表第 4/6/7 列（Badj/C、A/Badj、A/C）能不能从 p4rt 三份 .out 复现；
     第 5 列（等工作量 Badj_r2/C）能不能从 p4cdec 两份 .out 复现。
  Q4 "重测的第 4 列落在现印区间内" 是不是真的（8/8）。
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()
G = os.path.join(ROOT, "data", "gpu")

RT = ["5090_p4rt_1459647.out", "5090_p4rt_1459681.out", "5090_p4rt_1459702.out"]
DEC = ["5090_p4cdec_1459999.out", "5090_p4cdec_1460001.out"]

HEAD = re.compile(r"^###\s+(\S+)\s+Nj=(\d+)")
BROW = re.compile(r"^\s+B=(\d+)\s")
DEC_RE = re.compile(
    r"分解 一轮线代 f\+b\(ms/轮\): A稠密\+autograd\s+(\S+) \| "
    r"Badj稠密\+手写伴随\s+(\S+) \| C cudss\s+(\S+) \| "
    r"A/C\s+(\S+)x\s+Badj/C\s+(\S+)x\s+A/Badj\s+(\S+)x \| "
    r"一致性 gA\(B/A\) (\S+) \(C/A\) (\S+)\s*$")
DECL = re.compile(
    r"^\s+(\S+)\s+Nj=(\d+)\s+B=(\d+)\s+\| Badj\(rb=0\)\s+(\S+)\s+"
    r"Badj_r2\(rb=2\)\s+(\S+)\s+C\s+(\S+) ms/轮 \| Badj/C\s+(\S+)x\s+"
    r"Badj_r2/C\s+(\S+)x")

EIGHT = [("Net3", 256), ("Modena", 256), ("Modena", 1024), ("City_D", 64),
         ("City_D", 256), ("City_D", 1024), ("ky4", 64), ("ky4", 256)]


def parse_rt(path):
    """→ {(net,B): dict(A,Badj,C,agree_BA,agree_CA)}，只取原始 ms 与一致性。"""
    net, B, out = None, None, {}
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        m = HEAD.match(ln)
        if m:
            net, B = m.group(1), None
            continue
        m = BROW.match(ln)
        if m:
            B = int(m.group(1))
            continue
        m = DEC_RE.search(ln)
        if m and net and B:
            f = lambda s: float("nan") if s in ("-", "nan") else float(s)  # noqa
            out[(net, B)] = dict(A=f(m.group(1)), Badj=f(m.group(2)),
                                 C=f(m.group(3)),
                                 pA_C=m.group(4), pB_C=m.group(5),
                                 pA_B=m.group(6),
                                 ag_BA=m.group(7), ag_CA=m.group(8))
    return out


def parse_dec(path):
    out = {}
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        m = DECL.match(ln)
        if m:
            out[(m.group(1), int(m.group(3)))] = dict(
                Badj=float(m.group(4)), Badj_r2=float(m.group(5)),
                C=float(m.group(6)))
    return out


def fnum(s):
    try:
        return float(s)
    except Exception:                                     # noqa: BLE001
        return None


def main():
    print("=" * 104)
    print("XF-4 从原始 .out 独立重算 README 数字（只用 ms 原始值自己算比值）")
    print("=" * 104)
    rts = {p: parse_rt(os.path.join(G, p)) for p in RT}
    for p, d in rts.items():
        print("  %-26s 解析到 %d 格" % (p, len(d)))

    # ---------------- Q1 一致性 ----------------
    meas, cells = [], set()
    for p, d in rts.items():
        for (net, B), v in d.items():
            cells.add((net, B))
            for tag in ("ag_BA", "ag_CA"):
                x = fnum(v[tag])
                if x is not None:
                    meas.append((x, net, B, tag.replace("ag_", ""), p))
    meas.sort()
    print("\n【Q1 一致性】测量数 %d / 格数 %d（三份 .out 合计）"
          % (len(meas), len(cells)))
    print("  全网格最小 %.2e  @ %s B=%d (%s) %s"
          % (meas[0][0], meas[0][1], meas[0][2], meas[0][3], meas[0][4][-12:]))
    print("  全网格最大 %.2e  @ %s B=%d (%s) %s"
          % (meas[-1][0], meas[-1][1], meas[-1][2], meas[-1][3], meas[-1][4][-12:]))
    sub = [m for m in meas if (m[1], m[2]) in EIGHT]
    print("  8 格子集：测量数 %d，最小 %.2e @ %s B=%d (%s)，最大 %.2e @ %s B=%d (%s)"
          % (len(sub), sub[0][0], sub[0][1], sub[0][2], sub[0][3],
             sub[-1][0], sub[-1][1], sub[-1][2], sub[-1][3]))
    n_lo = sum(1 for m in meas if m[0] < 4.3e-16)
    n_hi = sum(1 for m in meas if m[0] > 7.2e-09)
    hi_nets = sorted({m[1] for m in meas if m[0] > 7.2e-09})
    over8 = sorted({m[1] for m in meas if m[0] > 1e-08})
    print("  旧印 4.3e-16~7.2e-09：更小的测量 %d 个，更大的 %d 个（落在 %s）"
          % (n_lo, n_hi, "/".join(hi_nets)))
    print("  超过 1e-08 的测量落在：%s" % "/".join(over8))

    # ---------------- Q2 README 字面 ----------------
    print("\n【Q2 README 字面核对】")
    chk = [("全网格下界 3.3e-16", meas[0][0], 3.25e-16, 3.35e-16),
           ("全网格上界 5.3e-08", meas[-1][0], 5.25e-08, 5.35e-08),
           ("8 格下界 1.9e-13", sub[0][0], 1.85e-13, 1.95e-13),
           ("8 格上界 3.9e-08", sub[-1][0], 3.85e-08, 3.95e-08)]
    q2 = True
    for name, v, lo, hi in chk:
        ok = lo <= v <= hi
        q2 = q2 and ok
        print("  %-18s 实测 %.3e  %s" % (name, v, "对得上" if ok else "**对不上**"))
    print("  最差格：全网格 = %s B=%d（README 说 Pescara B=1）；"
          "8 格 = %s B=%d（README 说 City_D B=64）"
          % (meas[-1][1], meas[-1][2], sub[-1][1], sub[-1][2]))
    print("  测量数/格数：实测 %d/%d（README 说 348/174）" % (len(meas), len(cells)))

    # ---------------- Q3/Q4 分解表 ----------------
    print("\n【Q3 分解表第 4/6/7 列：从三份 p4rt 的 ms 原始值自己算】")
    print("  %-8s %-5s | %-16s | %-16s | %-16s" %
          ("net", "B", "Badj/C(第4列)", "A/Badj(第6列)", "A/C(第7列)"))
    rng = {}
    for net, B in EIGHT:
        b_c, a_b, a_c = [], [], []
        for p, d in rts.items():
            v = d.get((net, B))
            if not v:
                continue
            b_c.append(v["Badj"] / v["C"])
            a_b.append(v["A"] / v["Badj"])
            a_c.append(v["A"] / v["C"])
        rng[(net, B)] = (min(b_c), max(b_c))
        print("  %-8s %-5d | %6.3f–%6.3fx  | %6.2f–%6.2fx  | %6.2f–%6.2fx"
              % (net, B, min(b_c), max(b_c), min(a_b), max(a_b),
                 min(a_c), max(a_c)))

    decs = {p: parse_dec(os.path.join(G, p)) for p in DEC}
    print("\n【Q3b 第 5 列（等工作量 Badj_r2/C）：从两份 p4cdec 的 ms 原始值自己算】")
    print("  net      B     | 第5列 Badj_r2/C  | 重测第4列 Badj/C | 等工作量贵了 | "
          "重测第4列落在 p4rt 区间内?")
    q4ok = 0
    for net, B in EIGHT:
        r5, r4, up = [], [], []
        for p, d in decs.items():
            v = d.get((net, B))
            if not v:
                continue
            r5.append(v["Badj_r2"] / v["C"])
            r4.append(v["Badj"] / v["C"])
            up.append(v["Badj_r2"] / v["Badj"] - 1.0)
        lo, hi = rng[(net, B)]
        # p4rt 三节点区间按 README 的两位小数印法放宽到印刷精度
        plo, phi = round(lo, 2) - 5e-3, round(hi, 2) + 5e-3
        inside = all(plo <= x <= phi for x in r4)
        q4ok += 1 if inside else 0
        print("  %-8s %-5d | %6.3f–%6.3fx    | %6.3f–%6.3fx   | +%4.1f%%~+%4.1f%% | %s"
              % (net, B, min(r5), max(r5), min(r4), max(r4),
                 100 * min(up), 100 * max(up), "是" if inside else "**否**"))
    print("\n【Q4】重测第 4 列落在 p4rt 三节点区间（印刷精度）内：%d/8" % q4ok)
    print("  等工作量让稠密对照贵的幅度：全部 8 格 × 2 节点 = "
          "+%.1f%% ~ +%.1f%%（README 说 21–32%%）"
          % (100 * min(min(decs[p][(n, b)]["Badj_r2"] / decs[p][(n, b)]["Badj"] - 1
                            for p in decs) for n, b in EIGHT),
             100 * max(max(decs[p][(n, b)]["Badj_r2"] / decs[p][(n, b)]["Badj"] - 1
                            for p in decs) for n, b in EIGHT)))
    print("=" * 104)
    return 0 if q2 else 1


if __name__ == "__main__":
    sys.exit(main())

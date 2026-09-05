# -*- coding: utf-8 -*-
"""P4 收尾 P1-b：README 那句 "agreement ... relative across the grid" 重算。

不看任何 wip、不复用任何上游结论，直接解析集群三节点的原始输出
`data/gpu/5090_p4rt_{1459647,1459681,1459702}.out` - 每一格自己打印的
"一致性 gA(B/A) x (C/A) y"（B/A = 手写稠密伴随 vs 通用 autograd；
C/A = cuDSS 伴随 vs 通用 autograd），重算全网格区间与最差格所在。

结论（本机前台跑出来的）：
  全网格 174 格 / 348 个测量 = **3.26e-16 ~ 5.30e-08**，最差在 Pescara B=1 的 C/A；
  只取 README 分解表印的那 8 格（48 个测量）= **1.85e-13 ~ 3.93e-08**，
  最差在 City_D B=64 的 B/A。
  README 原印的 "4.3e-16 to 7.2e-09" 两端都不是极值：有 6 个测量小于 4.3e-16，
  有 57 个测量大于 7.2e-09（全部落在 Pescara 与 City_D 两个网上）。
CPU 前台，几秒。
"""
import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FS = sorted(glob.glob(os.path.join(ROOT, "data", "gpu", "5090_p4rt_*.out")))

HDR = re.compile(r"^### (\S+)\s+Nj=(\d+)")
BLN = re.compile(r"^  B=(\d+)\s")
AGR = re.compile(r"一致性 gA\(B/A\)\s+(\S+)\s+\(C/A\)\s+(\S+)")

rows = []          # (job, net, Nj, B, which, value)
for f in FS:
    job = os.path.basename(f).split("_")[-1][:-4]
    net = Nj = B = None
    for ln in open(f, encoding="utf-8", errors="replace"):
        m = HDR.match(ln)
        if m:
            net, Nj = m.group(1), int(m.group(2))
            continue
        m = BLN.match(ln)
        if m:
            B = int(m.group(1))
            continue
        m = AGR.search(ln)
        if m and m.group(1) != "-" and m.group(2) != "-":   # "-" = 稠密侧 OOM
            rows.append((job, net, Nj, B, "B/A", float(m.group(1))))
            rows.append((job, net, Nj, B, "C/A", float(m.group(2))))

print("原始 .out %d 份: %s" % (len(FS), [os.path.basename(f) for f in FS]))
print("测量点 %d 个（%d 格 × 2 个比值）" % (len(rows), len(rows) // 2))
lo, hi = min(rows, key=lambda r: r[5]), max(rows, key=lambda r: r[5])
print("全网格区间 %.2e ~ %.2e" % (lo[5], hi[5]))
print("  最小 @ %-8s B=%-5d %s (job %s)" % (lo[1], lo[3], lo[4], lo[0]))
print("  最大 @ %-8s B=%-5d %s (job %s)" % (hi[1], hi[3], hi[4], hi[0]))

CELLS = [("Net3", 256), ("Modena", 256), ("Modena", 1024), ("City_D", 64),
         ("City_D", 256), ("City_D", 1024), ("ky4", 64), ("ky4", 256)]
sub = [r for r in rows if (r[1], r[3]) in CELLS]
slo, shi = min(sub, key=lambda r: r[5]), max(sub, key=lambda r: r[5])
print("\nREADME 分解表那 8 格（%d 个测量）%.2e ~ %.2e" % (len(sub), slo[5], shi[5]))
print("  最小 @ %-8s B=%-5d %s (job %s)" % (slo[1], slo[3], slo[4], slo[0]))
print("  最大 @ %-8s B=%-5d %s (job %s)" % (shi[1], shi[3], shi[4], shi[0]))

print("\n每网最差:")
for net in ("Net1", "Anytown", "Hanoi", "Net2", "Fossolo", "Pescara", "Net3",
            "Modena", "City_D", "ky4"):
    r = [x for x in rows if x[1] == net]
    if r:
        w = max(r, key=lambda x: x[5])
        print("  %-8s %.2e @ B=%-5d %s" % (net, w[5], w[3], w[4]))

print("\nREADME 原印的 4.3e-16 / 7.2e-09 是极值吗？")
print("  小于 4.3e-16 的测量 %d 个" % sum(1 for r in rows if r[5] < 4.3e-16))
print("  大于 7.2e-09 的测量 %d 个" % sum(1 for r in rows if r[5] > 7.2e-09))
print("  超出 7.2e-09 的格全在: %s"
      % sorted(set(r[1] for r in rows if r[5] > 7.2e-09)))

# -*- coding: utf-8 -*-
"""AUDIT 6 - README "Choosing a solve path" 一节里的每个数字回到原始 .out。

不看 wip（wip 是人写的中间层），直接解析集群原始输出：
  时间三节点 data/gpu/5090_p4rt_{1459647,1459681,1459702}.out
  显存两节点 data/gpu/5090_p4rm_{1459648,1459700}.out
重算三张表的 min~max 倍数区间，与 README 里印的逐格比。
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G = os.path.join(ROOT, "data", "gpu")
TIME = ["5090_p4rt_1459647.out", "5090_p4rt_1459681.out", "5090_p4rt_1459702.out"]
MEM = ["5090_p4rm_1459648.out", "5090_p4rm_1459700.out"]

NET_RE = re.compile(r"^### (\S+)\s+Nj=(\d+)\s+L=(\d+)\s+N=(\d+)\s+nnz=(\d+)\s+K=(\d+)")
B_RE = re.compile(r"^  B=(\d+)\s+fwd\s+dense\s+(\S+)\s+cudss\s+(\S+)\s+\|\s+"
                  r"fwd\+bwd\s+dense\s+(\S+)\s+cudss\s+(\S+)")
DEC_RE = re.compile(r"A稠密\+autograd\s+(\S+)\s+\|\s+Badj稠密\+手写伴随\s+(\S+)\s+\|"
                    r"\s+C cudss\s+(\S+)\s+\|.*一致性 gA\(B/A\) (\S+) \(C/A\) (\S+)")
MEM_RE = re.compile(r"^RESULT (\S+)\s+(\d+)\s+(\S+)\s+(ok|OOM)"
                    r"(?:.*torch_peak=(\S+) torch_resv=(\S+) nontorch=(\S+) total=(\S+))?")


def f(x):
    return None if x.strip() in ("OOM", "OOM_", "") else float(x)


T = {}          # (net,B) -> dict(fwd_d=[], fwd_c=[], fb_d=[], fb_c=[], dec...)
NETINFO = {}
for fn in TIME:
    net = None
    lastB = None
    for ln in open(os.path.join(G, fn), encoding="utf-8", errors="replace"):
        m = NET_RE.match(ln)
        if m:
            net = m.group(1)
            NETINFO[net] = dict(Nj=int(m.group(2)), nnz=int(m.group(5)),
                                K=int(m.group(6)))
            continue
        m = B_RE.match(ln)
        if m and net:
            lastB = int(m.group(1))
            d = T.setdefault((net, lastB), {})
            for k, v in zip(("fwd_d", "fwd_c", "fb_d", "fb_c"), m.groups()[1:]):
                d.setdefault(k, []).append(f(v))
            continue
        m = DEC_RE.search(ln)
        if m and net and lastB is not None:
            d = T.setdefault((net, lastB), {})
            for k, v in zip(("A", "Badj", "C", "r_ab", "r_ac"), m.groups()):
                d.setdefault(k, []).append(f(v))

M = {}
for fn in MEM:
    for ln in open(os.path.join(G, fn), encoding="utf-8", errors="replace"):
        m = MEM_RE.match(ln)
        if m:
            key = (m.group(1).replace(".inp", ""), int(m.group(2)), m.group(3))
            M.setdefault(key, []).append(
                None if m.group(4) == "OOM" else
                dict(peak=f(m.group(5)), resv=f(m.group(6)),
                     nontorch=f(m.group(7)), total=f(m.group(8))))


def rng(net, B, kd, kc):
    d = T.get((net, B), {})
    a, b = d.get(kd), d.get(kc)
    if not a or not b or any(x is None for x in a):
        return None
    r = [x / y for x, y in zip(a, b)]
    return min(r), max(r)


def fmt(r):
    if r is None:
        return "dense OOM"
    lo, hi = r
    return ("%.2fx" % lo) if round(lo, 2) == round(hi, 2) \
        else "%.2f–%.2fx" % (lo, hi)


# README 里印的三张表（人工誊抄自 README.md，用于逐格对拍）
RM_FWD = {
 "Net1": ["0.74–0.75x", "0.44x", "0.19x", "0.06x"],
 "Anytown": ["0.77x", "0.47x", "0.20x", "0.06x"],
 "Hanoi": ["0.71x", "0.37x", "0.14x", "0.04–0.05x"],
 "Net2": ["0.71–0.72x", "0.38x", "0.15x", "0.05x"],
 "Fossolo": ["0.71–0.72x", "0.37x", "0.14x", "0.05x"],
 "Pescara": ["0.77–0.78x", "0.41x", "0.16x", "0.06x"],
 "Net3": ["0.85x", "0.50x", "0.21x", "0.09x"],
 "Modena": ["1.22–1.24x", "0.71–0.72x", "0.50–0.51x", "0.41–0.43x"],
 "City_D": ["2.12–2.19x", "2.00–2.08x", "1.78–1.83x", "1.76–1.81x"],
 "ky4": ["3.54–3.62x", "5.24–5.40x", "6.00–6.27x", "6.59–6.92x"]}
RM_FB = {
 "Net1": ["0.82–0.84x", "0.49–0.50x", "0.20x", "0.06x"],
 "Anytown": ["0.85–0.86x", "0.52x", "0.24x", "0.08–0.09x"],
 "Hanoi": ["0.81x", "0.44–0.45x", "0.17–0.18x", "0.05x"],
 "Net2": ["0.83x", "0.45–0.46x", "0.18x", "0.06x"],
 "Fossolo": ["0.82–0.84x", "0.46x", "0.18x", "0.06x"],
 "Pescara": ["0.97–1.00x", "0.48x", "0.20–0.21x", "0.11x"],
 "Net3": ["1.00–1.01x", "0.55x", "0.25x", "0.15–0.16x"],
 "Modena": ["1.89–1.95x", "1.35–1.41x", "1.65–1.73x", "1.83–1.91x"],
 "City_D": ["4.27–4.49x", "6.71–7.08x", "9.49–9.81x", "dense OOM"],
 "ky4": ["10.27–10.88x", "28.45–29.86x", "dense OOM", "dense OOM"]}
RM_MEM = {
 "Net1": ["4.23x", "4.23x", "4.17x", "3.79x"],
 "Anytown": ["4.23x", "4.11x", "3.73x", "2.61x"],
 "Hanoi": ["4.23x", "4.22x", "4.38x", "3.82x"],
 "Net2": ["4.23x", "4.28x", "4.49x", "3.67x"],
 "Fossolo": ["4.26x", "4.19x", "4.33x", "3.88x"],
 "Pescara": ["4.29x", "4.50x", "5.11x", "4.65x"],
 "Net3": ["4.37x", "4.83x", "5.45x", "4.51x"],
 "Modena": ["5.25x", "9.33x", "12.83x", "14.41x"],
 "City_D": ["7.70x", "19.35x", "27.03x", "dense OOM"],
 "ky4": ["17.60x", "33.56x", "dense OOM", "dense OOM"]}
BS = [8, 64, 256, 1024]
ALIAS = {"Fossolo": "Fossolo_poly1"}

bad = 0
tot = 0
for title, tab, kd, kc in (("纯前向", RM_FWD, "fwd_d", "fwd_c"),
                           ("前向+反向", RM_FB, "fb_d", "fb_c")):
    print("\n【README %s 表 vs 原始 .out 重算】" % title)
    for net, cells in tab.items():
        row = []
        for B, printed in zip(BS, cells):
            tot += 1
            got = fmt(rng(net, B, kd, kc))
            ok = got.replace("–", "-") == printed.replace("–", "-")
            bad += 0 if ok else 1
            row.append("%s%s" % (got, "" if ok else "(印:%s)<<" % printed))
        print("  %-10s %s" % (net, " | ".join("%-22s" % x for x in row)))

print("\n【README 显存倍数表 vs 原始 .out 重算（total 口径）】")
for net, cells in RM_MEM.items():
    key = ALIAS.get(net, net)
    row = []
    for B, printed in zip(BS, cells):
        tot += 1
        dd = M.get((key, B, "dense"), [])
        cc = M.get((key, B, "cudss"), [])
        if not dd or dd[0] is None:
            got = "dense OOM"
        else:
            r = [a["total"] / b["total"] for a, b in zip(dd, cc)]
            got = ("%.2fx" % min(r)) if round(min(r), 2) == round(max(r), 2) \
                else "%.2f–%.2fx" % (min(r), max(r))
        ok = got.replace("–", "-") == printed.replace("–", "-")
        bad += 0 if ok else 1
        row.append("%s%s" % (got, "" if ok else "(印:%s)<<" % printed))
    print("  %-10s %s" % (net, " | ".join("%-20s" % x for x in row)))

print("\n【README 两因子分解表】")
DEC_RM = {("Net3", 256): ("0.06x", "3.25–3.37x", "0.19–0.20x"),
          ("Modena", 256): ("0.25–0.26x", "8.35–8.59x", "2.10–2.24x"),
          ("Modena", 1024): ("0.21–0.22x", "10.61–10.81x", "2.22–2.32x"),
          ("City_D", 64): ("1.35–1.45x", "8.11–8.19x", "10.91–11.88x"),
          ("City_D", 256): ("0.98–1.03x", "12.88–13.02x", "12.64–13.45x"),
          ("City_D", 1024): ("0.91–0.94x", "15.03–15.22x", "13.67–14.28x"),
          ("ky4", 64): ("4.11–4.36x", "12.26–12.39x", "50.42–53.97x"),
          ("ky4", 256): ("3.47–3.67x", "17.47–17.58x", "60.62–64.56x")}
for (net, B), (p1, p2, p3) in DEC_RM.items():
    d = T.get((net, B), {})
    g1 = fmt(rng(net, B, "Badj", "C"))
    g2 = fmt(rng(net, B, "A", "Badj"))
    g3 = fmt(rng(net, B, "A", "C"))
    for got, printed in ((g1, p1), (g2, p2), (g3, p3)):
        tot += 1
        if got.replace("–", "-") != printed.replace("–", "-"):
            bad += 1
    print("  %-8s B=%-5d Badj/C %-14s(印 %-12s) A/Badj %-14s(印 %-13s) "
          "A/C %-14s(印 %s)" % (net, B, g1, p1, g2, p2, g3, p3))

# 一致性区间（README 印 "4.3e-16 to 7.2e-09"）
ab = [x for k, v in T.items() for x in v.get("r_ab", []) if x is not None]
ac = [x for k, v in T.items() for x in v.get("r_ac", []) if x is not None]
print("\n【一致性 gA 相对差全格区间】B/A: %.2e ~ %.2e   C/A: %.2e ~ %.2e   "
      "合并 %.2e ~ %.2e （README 印 4.3e-16 ~ 7.2e-09）"
      % (min(ab), max(ab), min(ac), max(ac),
         min(min(ab), min(ac)), max(max(ab), max(ac))))

# 非 torch 工作区（README 印 dense B>=8 恒 230，B=1 是 360；cudss 68 -> 740）
nt_d1 = sorted({v[0]["nontorch"] for k, v in M.items()
                if k[2] == "dense" and k[1] == 1 and v[0]})
nt_d8 = sorted({v[0]["nontorch"] for k, v in M.items()
                if k[2] == "dense" and k[1] >= 8 and v[0]})
nt_c = sorted({v[0]["nontorch"] for k, v in M.items()
               if k[2] == "cudss" and v[0]})
print("【非 torch 工作区】dense B=1: %s   dense B>=8: %s   cudss: %.0f ~ %.0f"
      % (nt_d1, nt_d8, min(nt_c), max(nt_c)))
print("【显存两节点是否逐位相同】",
      all(len(set(round(x[k], 6) for x in v if x)) <= 1
          for v in M.values() for k in ("peak", "resv", "nontorch", "total")
          if all(v)),
      "（%d 个配置）" % len(M))
print("\nAUD-README-VERDICT 对拍格数=%d 不符=%d" % (tot, bad))

# -*- coding: utf-8 -*-
"""stats_recheck.py - §3.3 元启发式对比的统计口径复核（WR 大修，2026-10-23）。

不改 compare_calib.py，不覆盖原产物；读同一批数据（data/baselines_gd.json + calib_gc1_*.json）重算：
 1. 检验名称：配对 Wilcoxon signed-rank（scipy 缺省 method，n<=50 无零差无结时为精确分布；双侧；
    零差丢弃）。同时报告非零差个数 n' 与精确双侧 p 的下限 2/2^n'，标出 p 是否正好在下限上
    （= 所有种子同向，"完全分离"，p 本身不再携带强度信息）。
 2. 预算公平：只有基线预算 b 不低于梯度法同种子的实际用量时才算"同预算"。用量两种记账：
      A  只计前向调用 NFE（原稿口径）；
      B  前向与反向各计一次调用（保守口径。原稿称反向约为前向的 1%，
         本次实测 CPU 约 26%、GPU 75%–170%，见 09_第一阶段结果.md）。
 3. 配对效应：基线比梯度法差的种子比例；配对差（基线 − 梯度法，训练损失）中位数及
    bootstrap 95% CI（1e4 次重抽样，种子 0）。
 4. A12 保留作参考，注明为非配对定义（compare_calib.a12）。
输出：data/stats_recheck_report.txt、data/stats_recheck.csv
运行：python -X utf8 scripts/stats_recheck.py
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_calib as cc                                      # noqa: E402

RPT = os.path.join(cc.ROOT, "data", "stats_recheck_report.txt")
CSV = os.path.join(cc.ROOT, "data", "stats_recheck.csv")


def boot_ci(d, n_boot=10000, seed=0):
    d = np.asarray(d, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data = json.load(open(cc.OUT, encoding="utf-8"))
    runs = [r for r in data["runs"].values() if r["tag"] == "eval"]
    lines, rows = [], []

    def out(s=""):
        print(s)
        lines.append(s)
    for net, level in sorted({(r["net"], r["level"]) for r in runs}):
        if level != "L1":
            continue                       # L0 无噪声，梯度法为确定性单值，不做配对检验
        sub = [r for r in runs if r["net"] == net and r["level"] == level]
        gd = cc.load_gradient(net, level, data)
        cost_a = {s: gd[s]["nfe"] for s in gd}
        cost_b = {s: gd[s]["nfe"] + gd[s]["nbwd"] for s in gd}
        out("=" * 118)
        out(f"### {net} {level}：梯度法 {len(gd)} 个种子；用量 A 中位 {np.median(list(cost_a.values())):.0f}"
            f"（最大 {max(cost_a.values())}），用量 B 中位 {np.median(list(cost_b.values())):.0f}（最大 {max(cost_b.values())}）")
        out(f"{'预算':>6} {'算法':<7}{'n':>3} {'同预算A':>7} {'同预算B':>7} {'signed-rank p':>14} {'p下限':>9} "
            f"{'在下限':>5} {'基线更差':>8} {'配对差中位 [95% CI]':>34} {'A12(非配对)':>11}")
        for b in sorted({int(x) for r in sub for x in r["snapshots"]}):
            for algo in sorted({r["algo"] for r in sub}):
                prs = [(r["snapshots"][str(b)].get("loss"), gd[r["noise_seed"]]["loss"], r["noise_seed"])
                       for r in sub if r["algo"] == algo and str(b) in r["snapshots"] and r["noise_seed"] in gd]
                prs = [p for p in prs if p[0] is not None and p[0] == p[0]]
                if len(prs) < 5:
                    continue
                x = np.array([p[0] for p in prs])
                y = np.array([p[1] for p in prs])
                seeds = [p[2] for p in prs]
                d = x - y
                nz = int(np.sum(d != 0))
                p = float(wilcoxon(x, y, zero_method="wilcox").pvalue) if nz else float("nan")
                floor = 2.0 / 2 ** nz if nz else float("nan")
                at_floor = bool(p == p and abs(p - floor) <= 1e-12 * max(1.0, floor))
                eq_a = all(cost_a[s] <= b for s in seeds)
                eq_b = all(cost_b[s] <= b for s in seeds)
                lo, hi = boot_ci(d)
                worse = float(np.mean(d > 0))
                a12 = float(cc.a12(x, y))
                rows.append(dict(net=net, level=level, budget=b, algo=algo, n=len(prs), n_nonzero=nz,
                                 equal_budget_A=eq_a, equal_budget_B=eq_b, signed_rank_p=p, p_floor=floor,
                                 p_at_floor=at_floor, frac_baseline_worse=worse, median_diff=float(np.median(d)),
                                 ci95_lo=lo, ci95_hi=hi, a12_unpaired=a12))
                ps = f"{p:.3g}" if (eq_a and p == p) else "不检验"
                out(f"{b:>6} {algo:<7}{len(prs):>3} {('是' if eq_a else '否'):>7} {('是' if eq_b else '否'):>7} "
                    f"{ps:>14} {floor:>9.3g} {('是' if at_floor else ''):>5} {worse:>8.2f} "
                    f"{np.median(d):>12.4g} [{lo:>9.4g}, {hi:>9.4g}] {a12:>11.2f}")
    out()
    out("说明：'不检验' = 至少一个种子的梯度法用量 A 超过该预算，预算不等，只报描述统计。"
        "'在下限' = 所有非零配对差同号，p 等于精确检验能给出的最小值。")
    open(RPT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open(CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {RPT}, {CSV}")


if __name__ == "__main__":
    main()

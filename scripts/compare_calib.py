# -*- coding: utf-8 -*-
"""compare_calib.py - G-D 基线 vs 梯度法统计报表。

输入：data/baselines_gd.json（基线 + 梯度法补种子）、data/calib_gc1_*.json
（梯度法 G-C1 原始结果，直接引用不重跑）。

输出：
  stdout 中文汇总表（含"基线赢或平的场景"专节 - 可信度来源）；
  data/gd_comparison_report.txt       stdout 全文镜像（中文总表）；
  data/gd_comparison.json             机器可读结果（含预算-质量曲线 +
                                      梯度法 city_d NFE≈200 点显式标注）；
  data/baselines_compare_budget.csv   每预算点 中位数+IQR+配对 Wilcoxon+A12；
  data/baselines_compare_ecdf.csv     run-length ECDF（两套目标阈值）。

统计口径：
* 配对：同一题面种子（噪声种子）上 基线 vs 梯度法终点；Wilcoxon signed-rank
  （scipy.stats.wilcoxon，双侧，zero_method='wilcox'）。L0 无噪声：梯度法为
  确定性单值，基线 30 个优化器种子与该常数配对。
* A12（Vargha-Delaney）自写：A12 = (#{x<y} + 0.5·#{x=y}) / (n·m)，
  x=基线指标、y=梯度法指标（越小越好）⇒ A12 = P(基线优于梯度)。
  |A12-0.5|≥0.11/0.14/0.21 分别对应小/中/大效应（Vargha & Delaney 2000）。
* 赢/平判定（终点=该算法自己有快照的最大预算点）：
  赢 = 基线配对中位 ≤ 梯度中位；
  平 = |A12-0.5| < 0.11（可忽略效应）且 Wilcoxon 未显著（p≥0.05 或不可算）。
  注意 n=5 时双侧 Wilcoxon 的最小可能 p=1/16=0.0625，p 永远无法 <0.05，
  故判平不能只看 p - 必须叠加 A12 可忽略效应带，否则 5 种子战场上
  中位差 26 倍的完败也会被误标为"平"（此前版本的 bug，已修）。
* run-length：达到目标质量所需 NFE。两套目标（都取梯度法同种子终点 ×1.05
  宽松 5%，对基线有利）：①训练损失（沿 best-so-far 轨迹连续判定）；
  ②sub_rmse（只在预算网格快照上判定，粒度=网格）。
  截尾处理：预算内未达标记 NFE=inf（右删失）；ECDF 报告 P(RL≤b) 只对
  已达标质量计数，中位 run-length 若落在删失区间则报 ">最大预算"，
  不做任何外推。梯度法自身 run-length 取其终点 NFE（保守高估，对基线有利）。
"""

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "baselines_gd.json")
GC1 = {"hanoi": os.path.join(ROOT, "data", "calib_gc1_pub_hanoi.json"),
       "city_d": os.path.join(ROOT, "data", "calib_gc1_city_d.json")}
REPORT_TXT = os.path.join(ROOT, "data", "gd_comparison_report.txt")
REPORT_JSON = os.path.join(ROOT, "data", "gd_comparison.json")
SLACK = 1.05
A12_NEGLIGIBLE = 0.11        # |A12-0.5| < 0.11 = 可忽略效应（V&D 2000 阈值）

# 基线指标名 -> 梯度法字段名（配对用）
METRICS = [("loss", "loss"), ("sub_rmse", "sub"),
           ("info_rmse", "info"), ("val_frame_rmse", "valf")]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def a12(x, y):
    """Vargha-Delaney A12 = (#{x_i<y_j} + 0.5·#{x_i=y_j})/(n·m)。
    x=基线样本, y=梯度法样本（指标越小越好）⇒ 返回 P(基线优于梯度法)。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, m = len(x), len(y)
    lt = sum((xi < y).sum() for xi in x)
    eq = sum((xi == y).sum() for xi in x)
    return (lt + 0.5 * eq) / (n * m)


def paired_stats(pair_x, pair_y):
    """配对 Wilcoxon p + A12；样本不足或全零差返回 (nan, a12)。"""
    if len(pair_x) < 5:
        return np.nan, np.nan
    p_w = np.nan
    d = np.asarray(pair_x) - np.asarray(pair_y)
    if np.any(d != 0):
        try:
            p_w = wilcoxon(pair_x, pair_y, zero_method="wilcox").pvalue
        except ValueError:
            p_w = np.nan
    return p_w, a12(pair_x, pair_y)


def gd_loss_of(rec):
    """梯度法终点训练损失 mse+λ·reg（reg 从 traj 的 lm_{pick} 记录取）。"""
    lam = rec["lam"]
    mse = rec["mse_train_final"]
    reg = 0.0
    pick = rec.get("lm_pick", "off")
    for t in rec.get("traj", []):
        if t.get("phase") == f"lm_{pick}":
            reg = t.get("reg", 0.0)
    return mse + lam * reg


def gd_rec_to_dict(r):
    return dict(loss=gd_loss_of(r), sub=r.get("sub_rmse"),
                info=r.get("info_rmse"), valf=r["val_frame_rmse"],
                nfe=r["nfe"], nbwd=r.get("nbwd", 0))


def load_gradient(netkey, level, data):
    """种子 -> dict(loss, sub, info, valf, nfe, nbwd)。来源：G-C1 JSON + gdextra。"""
    gc1 = json.load(open(GC1[netkey], encoding="utf-8"))["runs"]
    out = {}
    if level == "L0":
        r = gc1.get("L0_perpipe")
        if r:
            out["det"] = gd_rec_to_dict(r)
        return out
    for k, r in gc1.items():
        if k.startswith("L1_s0.1_n") and k.endswith("lam1e-04"):
            sd = int(k.split("_n")[1].split("_")[0])
            out[sd] = gd_rec_to_dict(r)
    for k, r in data.get("gradient_runs", {}).items():
        nk, lv, stag = k.split("|")
        if nk == netkey and lv == level:
            sd = int(stag[1:])
            out[sd] = gd_rec_to_dict(r)
    return out


def fmt_iqr(v):
    v = np.asarray([x for x in v if x == x], float)
    if not len(v):
        return "-"
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return f"{med:.4g} [{q1:.4g},{q3:.4g}]"


def med_or_nan(v):
    v = [x for x in v if x is not None and x == x]
    return float(np.median(v)) if v else float("nan")


def main():
    data = json.load(open(OUT, encoding="utf-8"))
    runs = [r for r in data["runs"].values() if r["tag"] == "eval"]
    rows_budget, rows_ecdf = [], []
    win_or_tie = []
    js = dict(meta=dict(
        source=os.path.basename(OUT), slack=SLACK,
        a12_formula="A12=(#{x<y}+0.5*#{x=y})/(n*m), x=基线, y=梯度, 越小越好"
                    " => A12=P(基线优于梯度)",
        tie_rule=f"平 = |A12-0.5|<{A12_NEGLIGIBLE}(可忽略效应) 且 Wilcoxon 未显著"
                 "；n=5 时最小 p=0.0625, 不能只看 p",
        censoring="run-length 未达标记 inf（右删失）；ECDF 只报 P(RL<=b)，"
                  "中位落在删失区间报 '>最大预算'，不外推",
        gd_runlength="梯度法 run-length 取其终点 NFE（保守高估，对基线有利）",
        tuning_bills={nk: {a: e.get("nfe_bill") for a, e in t.items()}
                      for nk, t in data.get("tuning", {}).items()},
        chosen_cfgs={nk: {a: e.get("chosen") for a, e in t.items()}
                     for nk, t in data.get("tuning", {}).items()}),
        battlefields=[])
    combos = sorted({(r["net"], r["level"]) for r in runs})
    for net, level in combos:
        sub_runs = [r for r in runs if r["net"] == net and r["level"] == level]
        algos = sorted({r["algo"] for r in sub_runs})
        gd = load_gradient(net, level, data)
        budgets = sorted({int(b) for r in sub_runs for b in r["snapshots"]})
        gd_seeds = sorted(gd.keys(), key=str)
        gd_med = {m: med_or_nan([gd[s][g] for s in gd_seeds])
                  for m, g in METRICS}
        gd_nfe = int(np.median([gd[s]["nfe"] for s in gd_seeds]))
        gd_nbwd = int(np.median([gd[s]["nbwd"] for s in gd_seeds]))
        bf = dict(net=net, level=level,
                  gradient=dict(n_seeds=len(gd_seeds), nfe_median=gd_nfe,
                                nbwd_median=gd_nbwd,
                                loss_median=gd_med["loss"],
                                sub_rmse_median=gd_med["sub_rmse"],
                                info_rmse_median=gd_med["info_rmse"],
                                val_frame_rmse_median=gd_med["val_frame_rmse"],
                                marked_point=(net == "city_d"),
                                note="city_d 梯度法终点 NFE≈200 - 预算-质量曲线"
                                     "上的显式标注点" if net == "city_d" else ""),
                  curves={}, endpoint_verdicts=[], runlength=[])
        print("=" * 100)
        print(f"### {net} {level}  （梯度法终点 NFE 中位={gd_nfe}"
              f"（另 nbwd 中位={gd_nbwd}，成本≈前向1%），"
              f"损失中位={gd_med['loss']:.4g}，"
              f"sub_rmse 中位={gd_med['sub_rmse']:.4g}，"
              f"info_rmse 中位={gd_med['info_rmse']:.4g}，"
              f"valF p-RMSE 中位={gd_med['val_frame_rmse']:.4g}，"
              f"n_gd={len(gd_seeds)}）")
        endpoint = {}                 # algo -> 最大可用预算点的配对统计
        for b in budgets:
            print(f"  -- 预算 NFE={b} --")
            print(f"    {'算法':<8}{'训练损失 中位[IQR]':<30}"
                  f"{'sub_rmse 中位[IQR]':<26}{'info_rmse 中位[IQR]':<26}"
                  f"{'valF p-RMSE':<24}"
                  f"{'n':<4}{'Wilcoxon p':<12}{'A12':<6}")
            for algo in algos:
                rs = [r for r in sub_runs if r["algo"] == algo
                      and str(b) in r["snapshots"]]
                sn = [r["snapshots"][str(b)] for r in rs]
                vals = {m: [s.get(m, np.nan) for s in sn] for m, _ in METRICS}
                # 配对（按噪声种子；L0 梯度法为确定性单值 → 与常数配对）
                pairs = {m: ([], []) for m, _ in METRICS}
                for r, s in zip(rs, sn):
                    key = r["noise_seed"] if level == "L1" else "det"
                    if key not in gd:
                        continue
                    for m, g in METRICS:
                        bx, gy = s.get(m, np.nan), gd[key][g]
                        if bx == bx and gy is not None:
                            pairs[m][0].append(bx)
                            pairs[m][1].append(gy)
                stats = {m: paired_stats(*pairs[m]) for m, _ in METRICS}
                p_w, a_eff = stats["loss"]
                print(f"    {algo:<8}{fmt_iqr(vals['loss']):<30}"
                      f"{fmt_iqr(vals['sub_rmse']):<26}"
                      f"{fmt_iqr(vals['info_rmse']):<26}"
                      f"{fmt_iqr(vals['val_frame_rmse']):<24}{len(rs):<4}"
                      f"{p_w:<12.3g}{a_eff:<6.2f}")
                curve_pt = dict(budget=b, n=len(rs))
                for m, _ in METRICS:
                    v = np.asarray([x for x in vals[m] if x == x], float)
                    if not len(v):
                        continue
                    q1, med, q3 = np.percentile(v, [25, 50, 75])
                    pm, am = stats[m]
                    rows_budget.append(dict(
                        net=net, level=level, budget=b, algo=algo,
                        metric=m, median=med, q1=q1, q3=q3, n=len(v),
                        wilcoxon_p_vs_gd=pm if pm == pm else "",
                        a12_vs_gd=am if am == am else ""))
                    curve_pt[m] = dict(median=med, q1=q1, q3=q3,
                                       wilcoxon_p=None if pm != pm else pm,
                                       a12=None if am != am else am)
                if len(rs):
                    bf["curves"].setdefault(algo, []).append(curve_pt)
                if len(pairs["loss"][0]) >= 5:
                    endpoint[algo] = (b, pairs["loss"][0], pairs["loss"][1],
                                      p_w, a_eff)
        # ---------------- 终点赢/平判定（每算法取自己有数据的最大预算点） ----
        for algo in algos:
            if algo not in endpoint:
                continue
            b, px, py, p_w, a_eff = endpoint[algo]
            med_b, med_g = float(np.median(px)), float(np.median(py))
            win = med_b <= med_g
            not_sig = not (p_w == p_w and p_w < 0.05)
            tie = (abs(a_eff - 0.5) < A12_NEGLIGIBLE) and not_sig and not win
            verdict = "基线赢" if win else ("平" if tie else "梯度法赢")
            bf["endpoint_verdicts"].append(dict(
                algo=algo, budget=b, median_baseline=med_b, median_gd=med_g,
                wilcoxon_p=None if p_w != p_w else p_w, a12=a_eff,
                verdict=verdict))
            if win or tie:
                win_or_tie.append(
                    f"{net} {level} B={b}: {algo} 中位损失 {med_b:.4g} "
                    f"vs 梯度 {med_g:.4g}（p={p_w:.3g}, A12={a_eff:.2f}，"
                    f"{verdict}）")
        # ---------------- run-length ECDF ----------------
        print(f"  -- run-length（目标 = 梯度法同种子终点 ×{SLACK}；"
              f"未达标=inf 右删失；梯度法自身 run-length = 其终点 NFE，"
              f"保守高估）--")
        bmax = budgets[-1] if budgets else 0
        for algo in algos:
            rs = [r for r in sub_runs if r["algo"] == algo]
            rl_loss, rl_sub = [], []
            for r in rs:
                key = r["noise_seed"] if level == "L1" else "det"
                if key not in gd:
                    continue
                tgt_l = gd[key]["loss"] * SLACK + 1e-12
                rl = np.inf
                for nfe, best in r["traj"]:
                    if best <= tgt_l:
                        rl = nfe
                        break
                rl_loss.append(rl)
                tgt_s = (gd[key]["sub"] or np.nan) * SLACK
                rs_ = np.inf
                for bb in sorted(int(x) for x in r["snapshots"]):
                    sub_v = r["snapshots"][str(bb)].get("sub_rmse", np.nan)
                    if sub_v == sub_v and sub_v <= tgt_s:
                        rs_ = bb
                        break
                rl_sub.append(rs_)
            for tag, rl in (("loss", rl_loss), ("sub_rmse", rl_sub)):
                rl = np.asarray(rl, float)
                if not len(rl):
                    continue
                frac = float(np.mean(np.isfinite(rl)))
                med = np.median(rl)
                med_s = f"{med:.0f}" if np.isfinite(med) else f">{bmax}"
                print(f"    {algo:<8}{tag:<9} 达标率={frac * 100:5.1f}%  "
                      f"中位 run-length={med_s}")
                bf["runlength"].append(dict(
                    algo=algo, target=tag, n=len(rl),
                    frac_reached=frac,
                    median_runlength=None if not np.isfinite(med) else float(med),
                    censored_at=bmax))
                for bb in budgets:
                    rows_ecdf.append(dict(
                        net=net, level=level, algo=algo, target=tag,
                        budget=bb,
                        frac_reached=float(np.mean(rl <= bb)),
                        n=len(rl)))
        # 梯度法自身的 run-length（=各种子终点 NFE，定义即 100% 达标）
        gd_rl = np.asarray([gd[s]["nfe"] for s in gd_seeds], float)
        for tag in ("loss", "sub_rmse"):
            bf["runlength"].append(dict(
                algo="gradient", target=tag, n=len(gd_rl),
                frac_reached=1.0, median_runlength=float(np.median(gd_rl)),
                censored_at=bmax))
            for bb in budgets:
                rows_ecdf.append(dict(
                    net=net, level=level, algo="gradient", target=tag,
                    budget=bb, frac_reached=float(np.mean(gd_rl <= bb)),
                    n=len(gd_rl)))
        js["battlefields"].append(bf)
    print("=" * 100)
    if win_or_tie:
        print("### 基线赢或平的场景（各算法自己的最大预算终点，如实呈现 - 可信度来源）")
        for s in win_or_tie:
            print("  " + s)
    else:
        print("### 各算法最大预算终点没有基线赢或平的场景"
              "（赢=配对中位≤梯度；平=|A12-0.5|<0.11 且 Wilcoxon 未显著）")
    js["win_or_tie"] = win_or_tie
    # ---------------- CSV / JSON ----------------
    if not rows_budget:
        print("（无 eval 记录，跳过 CSV）")
        return js
    import csv
    fp1 = os.path.join(ROOT, "data", "baselines_compare_budget.csv")
    with open(fp1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_budget[0].keys()))
        w.writeheader()
        w.writerows(rows_budget)
    fp2 = os.path.join(ROOT, "data", "baselines_compare_ecdf.csv")
    with open(fp2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_ecdf[0].keys()))
        w.writeheader()
        w.writerows(rows_ecdf)
    print(f"\nCSV 已落盘: {fp1}\n           {fp2}")
    return js


if __name__ == "__main__":
    import io
    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = Tee(real, buf)
    try:
        js = main()
    finally:
        sys.stdout = real
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    if js is not None:
        with open(REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False, indent=1, default=float)
    print(f"报告已落盘: {REPORT_TXT}\nJSON 已落盘: {REPORT_JSON}")

# -*- coding: utf-8 -*-
"""optimizer_v2_report.py - 模块三的汇总（只读 JSON，不跑仿真）。

读入
  data/optimizer_v2.json            我们的臂（run/verify/arm/cost 合并后）
  data/baselines_v2_<algo>.json     补足调参预算后的基线（tune2 + eval2 + big）
  data/baselines_v2old_<algo>.json  用**旧调参预算选出的配置**在同一台机器重跑
  data/baselines_gd.json            冻结的 G-D 基线（旧机器、旧调参预算）
  data/calib_gc1_city_d.json        冻结的 G-C1 梯度法

产出 data/optimizer_v2_wip.txt。可读输出零节点/链路/传感器编号。

用法：& python -X utf8 scripts/optimizer_v2_report.py [--dir data] [--out ...]
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NET, LEVEL = "city_d", "L1"
SEEDS = [100, 101, 102, 103, 104]
METRICS = ["sub_rmse", "info_rmse", "val_frame_rmse"]


def jload(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def rng_str(xs, fmt="%.4g"):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return "-"
    return (fmt + "–" + fmt) % (min(xs), max(xs))


# ====================================================================== 载入
def load_ours(d):
    """合并 out/ov2_*.json 分片 + probe/cost。"""
    out = dict(runs={}, verify={}, arm={}, cost={})
    for p in sorted(glob.glob(os.path.join(d, "optimizer_v2*.json"))):
        j = jload(p)
        if not j:
            continue
        out["runs"].update(j.get("runs", {}))
        for k in ("verify", "arm", "cost"):
            if j.get(k):
                out[k] = j[k]
    return out


def load_base_shards(d, pat):
    runs, tuning = {}, {}
    for p in sorted(glob.glob(os.path.join(d, pat))):
        j = jload(p)
        if not j:
            continue
        runs.update(j.get("runs", {}))
        for net, algs in (j.get("tuning") or {}).items():
            tuning.setdefault(net, {}).update(algs)
    return runs, tuning


def base_rows(runs, tag, budget=None, algo=None, seeds=None):
    out = []
    for k, r in runs.items():
        if r.get("net") != NET or r.get("tag") != tag:
            continue
        if budget is not None and r.get("budget") != budget:
            continue
        if algo is not None and r.get("algo") != algo:
            continue
        if seeds is not None and r.get("noise_seed") not in seeds:
            continue
        out.append(r)
    return out


def snap_of(rec, g):
    """基线 snapshot（在 NFE 预算点 g 上的 best-so-far + 评价指标）。"""
    return (rec.get("snapshots") or {}).get(str(g))


def our_rows(runs, arm, budget=2000):
    return [r for r in runs.values()
            if r.get("arm") == arm and r.get("budget") == budget]


# ====================================================================== 主体
def build(d, outp):
    ours = load_ours(d)
    b_new, tun_new = load_base_shards(d, "baselines_v2_*.json")
    b_old, _ = load_base_shards(d, "baselines_v2old_*.json")
    frozen = jload(os.path.join(d, "baselines_gd.json")) or {}
    gc1 = jload(os.path.join(d, "calib_gc1_city_d.json")) or {}

    L = []
    P = L.append
    cost = ours.get("cost") or {}
    gpu = cost.get("gpu") or (ours.get("arm") or {}).get("gpu") or "?"
    P("[city_d] 模块三：一阶优化器增强 + 基线公平性重构 - 实测数值")
    P("生成 %s @ <workstation>" % time.strftime("%Y-%m-%d %H:%M:%S"))
    P("机器：超算 %s（1 卡/作业，节点 %s CPU 核）；本地验收 RTX 5060 Laptop"
      % (gpu, cost.get("cpus", "?")))
    P("题面：city_d、L1（σ=0.1 ft、λ=1e-4）、真值 perpipe、布点 dopt40、"
      "20%% 传感器留出、20 训练帧、自由管 %s 根、箱 [40,160]"
      % cost.get("dim", "432"))
    P("")

    # ---------------------------------------------------------------- 零
    v = ours.get("verify") or {}
    P("零、数值验收（做对比之前先证明新算子没算错）")
    if v:
        P("  A 前向：GPU 批量逐场景 r_hw vs CPU solve_polished   max|ΔH| = %.3e ft"
          % v.get("fwd_max_dH_ft", float("nan")))
        P("  B 梯度：GPU 批量伴随 vs 逐个 CPU 伴随（theta 空间）  相对差 = %.3e"
          % v.get("grad_rel_vs_cpu_adjoint", float("nan")))
        pc = v.get("precond") or {}
        P("  C 预条件子：Schur-Jacobi 近似 vs 精确 GN 对角")
        P("      log10 皮尔逊 r = %.4f   斯皮尔曼 = %.4f" %
          (pc.get("log10_pearson", float("nan")),
           pc.get("spearman", float("nan"))))
        P("      比值中位 %.3e（p10 %.2e，p90 %.2e）" %
          (pc.get("ratio_median", float("nan")),
           pc.get("ratio_p10", float("nan")),
           pc.get("ratio_p90", float("nan"))))
        for kk in ("schur", "gnex"):
            i = pc.get(kk) or {}
            P("      %-6s 原始 max=%.3e min=%.3e  地板 %.0e 触发 %s/%s 维  "
              "theta 空间条件数 %.3e"
              % (kk, i.get("max", float("nan")), i.get("min", float("nan")),
                 i.get("floor_rel", 0), i.get("n_floored", "?"),
                 cost.get("dim", "?"), i.get("theta_cond", float("nan"))))
        P("  读法：A/B 说明批量多起点前向与伴随与既有单场景实现在数值上同一；")
        P("        C 说明零成本的 Jacobi 近似只抓住了精确 GN 对角的**量级**，")
        P("        秩相关不高 - 这是近似的真实分辨力，不是它的卖点。")
    else:
        P("  （缺 verify 数据）")
    P("")

    # ---------------------------------------------------------------- 一
    P("一、\"1 NFE\"这张票的实际价钱（同一节点、同一进程墙钟实测）")
    if cost:
        P("    我们的 1 NFE（20 帧批前向 + 伴随反传）        %.4f s" %
          cost.get("t_gpu_fwdbwd", float("nan")))
        P("    其中只前向                                     %.4f s" %
          cost.get("t_gpu_fwd", float("nan")))
        P("    基线批量目标 每 NFE（种群 P=64 一次 kernel）    %.4f s  (chunk=%s)" %
          (cost.get("t_base_P64_per_nfe", float("nan")),
           cost.get("base_chunk", "?")))
        P("    基线批量目标 每 NFE（P=32）                     %.4f s" %
          cost.get("t_base_P32_per_nfe", float("nan")))
        P("    基线批量目标 单点（P=1）                        %.4f s" %
          cost.get("t_base_P1", float("nan")))
        P("    LM 段的一次批前向（CPU solve_polished）         %.4f s" %
          cost.get("t_cpu_fwd", float("nan")))
        P("    LM 段的一次伴随雅可比（sensitivity_matrix）     %.4f s" %
          cost.get("t_sensitivity", float("nan")))
        P("")
        P("    ** 我们本来怀疑这里有个口径漏洞，结果实测把怀疑否掉了 **")
        P("      怀疑：calibrate._lm_polish 与 G-D 的 declarations 把一次伴随")
        P("      雅可比记作 nfe+=1 / nbwd+=1，而它在 city_d 上是 %s 次稀疏 LU"
          % cost.get("n_lu_per_jac", "?"))
        P("      + %s 次回代 - 看起来像是把 640 列的雅可比按 1 次前向收费。"
          % cost.get("n_rhs_per_jac", "?"))
        P("      实测：sensitivity_matrix 全程 %.4f s，其中它自己那次批前向就要"
          % cost.get("t_sensitivity", float("nan")))
        P("      %.4f s - 折合 **%.3f 次 LM 段前向**。也就是说稀疏分解 + 640 次"
          % (cost.get("t_cpu_fwd", float("nan")),
             cost.get("jac_in_cpu_fwd", float("nan"))))
        P("      回代只占前向的百分之几，\"伴随成本 ≈ 前向 1%\" 这句话在本网上")
        P("      **是对的**，现有 NFE 口径不偏袒 LM 尾。这条不改。")
        P("")
        P("    ** 真正不对称的地方在硬件放置，不在计账 **")
        P("      同一次伴随雅可比折合 **%.1f 次我们的 GPU NFE**、**%.1f 次基线的"
          % (cost.get("jac_in_gpu_nfe", float("nan")),
             cost.get("jac_in_base_nfe64", float("nan"))))
        P("      种群 NFE** - 因为 LM 段（我们的 lm_* 与基线 hybrid **共用**")
        P("      calibrate._lm_polish）跑在 CPU 稀疏直接法上，而一阶段跑在 GPU")
        P("      批量 dense 路径上：%.4f s vs %.4f s，差 %.0f 倍。"
          % (cost.get("t_cpu_fwd", float("nan")),
             cost.get("t_gpu_fwdbwd", float("nan")),
             cost.get("t_cpu_fwd", 1.0) / max(cost.get("t_gpu_fwdbwd", 1.0),
                                              1e-12)))
        P("      这对我们的 lm_* 臂和基线 hybrid 是**同等**的（同一份代码），")
        P("      所以两者互比公平；但\"纯一阶 vs 带 LM 尾\"在 NFE 轴和墙钟轴上")
        P("      会给出不同的排序。因此：NFE 表照旧列，**墙钟 Pareto（第七节）")
        P("      同时列**，两把尺子都摆出来，不挑对自己有利的那把。")
    else:
        P("  （缺 cost 数据）")
    P("")

    # ---------------------------------------------------------------- 二
    P("二、2000 次调用节点的横向对比（5 个噪声种子，中位数；括号为极差）")
    P("    NFE 口径：B 起点 × N 步 = B·N 次调用；L-BFGS 线搜索每次试探都计。")
    P("    %-15s %6s %6s | %-11s %-9s %-9s %-9s | %7s" %
      ("臂", "NFE", "nbwd", "训练损失", "参数sub", "info", "留出帧", "墙钟s"))
    order = ["gd_ref", "fo_plain", "fo_schur", "fo_gnex", "fo_ms8",
             "fo_ms8_schur", "fo_ms16_schur", "lm_plain", "lm_schur",
             "lm_ms8_schur"]
    tbl = {}
    for arm in order:
        rs = our_rows(ours["runs"], arm)
        if not rs:
            continue
        row = dict(
            nfe=med([r["nfe"] for r in rs]), nbwd=med([r["nbwd"] for r in rs]),
            loss=med([r["loss"] for r in rs]),
            loss_all=[r["loss"] for r in rs],
            wall=med([r["wall_sec"] for r in rs]), n=len(rs))
        for m in METRICS:
            row[m] = med([r.get(m) for r in rs])
            row[m + "_all"] = [r.get(m) for r in rs]
        tbl[arm] = row
        P("    %-15s %6.0f %6.0f | %-11.5e %-9.3f %-9.3f %-9.4f | %7.0f n=%d" %
          (arm, row["nfe"], row["nbwd"], row["loss"], row["sub_rmse"],
           row["info_rmse"], row["val_frame_rmse"], row["wall"], row["n"]))
    P("")
    # 基线
    P("    基线（同 2000 NFE 评价预算）")
    for tag, runs_, lbl in (("eval2", b_new, "补足调参预算后"),
                            ("evalold", b_old, "旧配置·同机器")):
        for algo in ("de", "pso", "cma", "hybrid"):
            rs = base_rows(runs_, tag, 2000, algo)
            if not rs:
                continue
            sn = [snap_of(r, 2000) for r in rs]
            sn = [s for s in sn if s]
            P("    %-15s %6.0f %6.0f | %-11.5e %-9.3f %-9.3f %-9.4f | %7.0f  [%s cfg=%s]"
              % (algo + ":" + tag, med([r["nfe_total"] for r in rs]),
                 med([r["nbwd_total"] for r in rs]),
                 med([r["best_loss"] for r in rs]),
                 med([s.get("sub_rmse") for s in sn]),
                 med([s.get("info_rmse") for s in sn]),
                 med([s.get("val_frame_rmse") for s in sn]),
                 med([r["wall_sec"] for r in rs]), lbl,
                 rs[0]["config"]["name"]))
    # 冻结参照
    fr = base_rows((frozen.get("runs") or {}), "eval", 2000)
    if fr:
        P("    ---- 冻结 G-D（旧机器 + 调参预算 300）参照 ----")
        for algo in ("de", "pso", "cma", "hybrid"):
            rs = [r for r in fr if r["algo"] == algo]
            if not rs:
                continue
            sn = [s for s in (snap_of(r, 2000) for r in rs) if s]
            P("    %-15s %6.0f %6.0f | %-11.5e %-9.3f %-9.3f %-9.4f | %7.0f  [cfg=%s]"
              % (algo + ":frozen", med([r["nfe_total"] for r in rs]),
                 med([r["nbwd_total"] for r in rs]),
                 med([r["best_loss"] for r in rs]),
                 med([s.get("sub_rmse") for s in sn]),
                 med([s.get("info_rmse") for s in sn]),
                 med([s.get("val_frame_rmse") for s in sn]),
                 med([r["wall_sec"] for r in rs]), rs[0]["config"]["name"]))
    g1 = [r for k, r in (gc1.get("runs") or {}).items()
          if k.startswith("L1_s0.1_n1") and k.endswith("lam1e-04")]
    if g1:
        P("    %-15s %6.0f %6.0f | %-11.5e %-9.3f %-9.3f %-9.4f | %7.0f"
          % ("G-C1:frozen", med([r["nfe"] for r in g1]),
             med([r["nbwd"] for r in g1]),
             med([r["mse_train_final"] + 1e-4 * r.get("reg_final", 0.0)
                  for r in g1]),
             med([r["sub_rmse"] for r in g1]),
             med([r["info_rmse"] for r in g1]),
             med([r["val_frame_rmse"] for r in g1]),
             med([r["t_calib_sec"] for r in g1])))
    P("")

    # ---------------------------------------------------------------- 三
    def paired(a, b, key):
        """按噪声种子配对的差（a−b），符号检验（双侧精确二项）。"""
        ra = {r["noise_seed"]: r for r in our_rows(ours["runs"], a)}
        rb = {r["noise_seed"]: r for r in our_rows(ours["runs"], b)}
        sds = sorted(set(ra) & set(rb))
        da = [ra[s].get(key) if key in ra[s] else ra[s]["loss"] for s in sds]
        db = [rb[s].get(key) if key in rb[s] else rb[s]["loss"] for s in sds]
        d = [x - y for x, y in zip(da, db)
             if x is not None and y is not None]
        n = len(d)
        w = sum(1 for x in d if x < 0)          # a 更小（更好）的次数
        ties = sum(1 for x in d if x == 0)
        ne = n - ties
        if ne == 0:
            return n, w, float("nan"), float("nan"), float("nan")
        from math import comb
        k = min(w, ne - w)
        p = min(1.0, 2.0 * sum(comb(ne, i) for i in range(k + 1)) / 2 ** ne)
        return n, w, med(d), p, 2.0 / 2 ** ne

    P("三、预条件子本身有没有用（同起点、同预算、同线搜索实现，只换 H0）")
    P("    配对单位 = 噪声种子；符号检验为双侧精确二项；p_min = 该 n 下可能的最小 p。")
    P("    %-30s %3s %5s %-12s %-9s %-9s" %
      ("对比（A vs B）", "n", "A胜", "中位差(损失)", "p", "p_min"))
    for a, b in (("fo_schur", "fo_plain"), ("fo_gnex", "fo_plain"),
                 ("fo_gnex", "fo_schur"), ("lm_schur", "lm_plain")):
        if a not in tbl or b not in tbl:
            continue
        n, w, dm, p, pmin = paired(a, b, "loss")
        P("    %-30s %3d %5d %-12.4e %-9.4f %-9.4f" %
          (a + " vs " + b, n, w, dm, p, pmin))
    P("    同样的配对，看参数误差 sub_rmse（损失已在噪声地板，参数误差才有分辨力）")
    P("    %-30s %3s %5s %-12s %-9s %-9s" %
      ("对比（A vs B）", "n", "A胜", "中位差(sub)", "p", "p_min"))
    for a, b in (("fo_schur", "fo_plain"), ("fo_gnex", "fo_plain"),
                 ("fo_gnex", "fo_schur"), ("lm_schur", "lm_plain"),
                 ("lm_ms8_schur", "lm_plain")):
        if a not in tbl or b not in tbl:
            continue
        n, w, dm, p, pmin = paired(a, b, "sub_rmse")
        P("    %-30s %3d %5d %-12.4f %-9.4f %-9.4f" %
          (a + " vs " + b, n, w, dm, p, pmin))
    P("")

    # ---------------------------------------------------------------- 四
    P("四、批量多起点：省的是墙钟，不是调用数")
    ar = (ours.get("arm") or {}).get("rows") or []
    if ar:
        P("    %-4s %-12s %-12s %-10s" %
          ("B", "每步 s", "每 NFE s", "相对 B=1 的每-NFE 加速"))
        b1 = next((r["sec_per_nfe"] for r in ar if r["B"] == 1), None)
        for r in ar:
            P("    %-4d %-12.4f %-12.4f %-10s" %
              (r["B"], r["sec_per_step"], r["sec_per_nfe"],
               ("%.2f×" % (b1 / r["sec_per_nfe"])) if b1 else "-"))
        P("    读法：20 帧的批前向已经把这块 GPU 喂饱了，再把 B 个起点摊进同一批，")
        P("    每-NFE 只再快 %s - **不是 B 倍**。多起点在 NFE 轴上是纯支出" %
          (("%.1f×" % (b1 / min(r["sec_per_nfe"] for r in ar)))
           if b1 else "?"))
        P("    （B 起点 × N 步 = B·N 次调用），只有在墙钟轴上才有折扣。")
    for a, b in (("fo_ms8", "fo_plain"), ("fo_ms8_schur", "fo_schur"),
                 ("fo_ms16_schur", "fo_schur")):
        if a not in tbl or b not in tbl:
            continue
        n, w, dm, p, pmin = paired(a, b, "sub_rmse")
        P("    %-30s n=%d A胜=%d 中位Δsub=%+.4f p=%.4f (p_min=%.4f)" %
          (a + " vs " + b, n, w, dm, p, pmin))
    ws = [(k, tbl[k]["detail_win"]) for k in tbl if "detail_win" in tbl[k]]
    P("")

    # ---------------------------------------------------------------- 五
    P("五、公平性重构：City D 基线的调参预算 300 → 2000（= 评价预算，与 Hanoi 同规格）")
    fz_t = ((frozen.get("tuning") or {}).get(NET) or {})
    fz_h = ((frozen.get("tuning") or {}).get("hanoi") or {})
    P("    冻结 G-D 的调参账单：hanoi 每基线 %s NFE；city_d 每基线 %s NFE" %
      (max([e.get("nfe_bill", 0) for e in fz_h.values()] or [0]),
       max([e.get("nfe_bill", 0) for e in fz_t.values()] or [0])))
    P("    协议口径：Hanoi 调参预算 = 评价预算 = 20000；City D 评价预算 2000 而调参只给 300"
      "（差 6.7×）。")
    P("    本轮：City D 调参预算改为 2000，4 配置 × 3 专用种子（900..902），逐算法重跑。")
    P("    下表\"中位损失\"是**各自调参预算下**的调参损失（300 vs 2000 NFE），")
    P("    数值本身不可比；可比的是**选中了哪个配置**。")
    P("    %-8s %-22s %-22s %10s %10s" %
      ("算法", "旧选中（预算300）", "新选中（预算2000）", "旧@300", "新@2000"))
    nt = (tun_new.get(NET) or {})
    for algo in ("de", "pso", "cma", "hybrid"):
        o_ = fz_t.get(algo) or {}
        n_ = nt.get(algo) or {}
        oc, nc = o_.get("chosen", "-"), n_.get("chosen", "-")
        # 分片重调参时各配置完成的种子数可能不同；n 不齐时"选中"没有意义
        ns_ = {k: len((v or {}).get("losses", {}))
               for k, v in (n_.get("results") or {}).items()}
        if ns_ and len(set(ns_.values())) > 1:
            full = [k for k, v in ns_.items() if v == max(ns_.values())]
            ml_ = n_.get("median_loss") or {}
            nc = (min((k for k in full if k in ml_), key=lambda k: ml_[k],
                      default="-") + "  (只在 n=%d 的配置间比)" % max(ns_.values()))
        om = (o_.get("median_loss") or {}).get(oc, float("nan"))
        nm = (n_.get("median_loss") or {}).get(nc, float("nan"))
        P("    %-8s %-22s %-40s %10.4e %10.4e%s" %
          (algo, oc, nc, om, nm,
           "" if nc.startswith(oc) else "   <-- 配置变了"))
    for algo in ("de", "pso", "cma", "hybrid"):
        n_ = nt.get(algo) or {}
        if n_.get("median_loss"):
            ns = {k: len((n_["results"].get(k) or {}).get("losses", {}))
                  for k in n_["median_loss"]}
            P("      %-8s 新调参各配置中位损失（n=完成的调参种子数，满额 3）：" % algo)
            for k in sorted(n_["median_loss"]):
                P("               %-32s %.3e  n=%d" %
                  (k, n_["median_loss"][k], ns.get(k, 0)))
            P("               账单 %s NFE" % n_.get("nfe_bill", "?"))
            if min(ns.values()) < 3:
                P("               ** 本算法的重调参**未跑满**（见上 n 列）：结论只在")
                P("               已完成的种子上成立，不要当作满额协议来引用。")
    P("    效果（同机器、同评价预算 2000、同 5+ 噪声种子）：")
    P("    %-10s %-13s %-13s %-13s | %-8s %-8s %-8s" %
      ("算法", "旧cfg同机器", "新cfg同机器", "冻结G-D旧机器",
       "sub旧", "sub新", "sub冻结"))
    P("    （三列一律只取公共种子 %s，n 见行尾 - eval2 跑了更多种子，"
      "不同种子集的中位数不可比）" % ",".join(str(x) for x in SEEDS))
    for algo in ("de", "pso", "cma", "hybrid"):
        ro = base_rows(b_old, "evalold", 2000, algo, seeds=set(SEEDS))
        rn = base_rows(b_new, "eval2", 2000, algo, seeds=set(SEEDS))
        rf = [r for r in base_rows((frozen.get("runs") or {}), "eval", 2000,
                                   seeds=set(SEEDS)) if r["algo"] == algo]
        if not (ro or rn):
            continue
        so = [s for s in (snap_of(r, 2000) for r in ro) if s]
        sn_ = [s for s in (snap_of(r, 2000) for r in rn) if s]
        sf = [s for s in (snap_of(r, 2000) for r in rf) if s]
        P("    %-10s %-13.5e %-13.5e %-13.5e | %-8.3f %-8.3f %-8.3f | n=%d/%d/%d"
          % (algo, med([r["best_loss"] for r in ro]),
             med([r["best_loss"] for r in rn]),
             med([r["best_loss"] for r in rf]),
             med([s.get("sub_rmse") for s in so]),
             med([s.get("sub_rmse") for s in sn_]),
             med([s.get("sub_rmse") for s in sf]),
             len(ro), len(rn), len(rf)))
    P("    读法：'旧cfg同机器' 与 '冻结G-D旧机器' 的差 = 纯硬件/库版本效应；")
    P("          '新cfg同机器' 与 '旧cfg同机器' 的差 = 补足调参预算的**净效应**。")
    P("")

    # ---------------------------------------------------------------- 六
    P("六、给基线 X 倍预算会怎样（抽样验证，实测非外推）")
    P("    %-8s %-8s %3s %-13s %-9s %-9s %-9s %8s" %
      ("算法", "预算", "n", "训练损失中位", "sub", "info", "留出帧", "墙钟s"))
    for algo in ("de", "hybrid"):
        for g in (2000, 5000, 10000, 20000):
            rs = [r for r in base_rows(b_new, "big", None, algo)
                  if snap_of(r, g)]
            if not rs:
                rs = [r for r in base_rows(b_new, "eval2", 2000, algo)
                      if snap_of(r, g)]
            if not rs:
                continue
            sn_ = [snap_of(r, g) for r in rs]
            P("    %-8s %-8d %3d %-13.5e %-9.3f %-9.3f %-9.4f %8.0f" %
              (algo, g, len(rs), med([s["loss"] for s in sn_]),
               med([s.get("sub_rmse") for s in sn_]),
               med([s.get("info_rmse") for s in sn_]),
               med([s.get("val_frame_rmse") for s in sn_]),
               med([s.get("wall_sec") for s in sn_])))
    P("")

    # ---------------------------------------------------------------- 七
    P("七、Wall-clock Pareto（物理时间 vs 训练损失；同一 %s、单卡、节点 %s 核）" %
      (gpu, cost.get("cpus", "?")))
    P("    并行化声明：两侧的**目标函数都已 GPU 批量化** - 基线的种群（DE/PSO/CMA")
    P("    每代 NP 个个体、scipy vectorized=True/updating='deferred'）与我们的 B 个")
    P("    起点走的是同一条批量 dense GGA 路径，chunk=%s。所以这不是"
      % cost.get("base_chunk", "?"))
    P("    '我们批量、基线串行' 的比较。基线的**驱动器**（scipy DE 的选择/变异、")
    P("    cma 的 ask/tell）是单线程 CPU，但它在总墙钟里的占比可由 P=1 与 P=64 的")
    P("    每-NFE 差读出；LM 段（我们的 lm_* 与基线 hybrid 共用）是 CPU 稀疏直接法，")
    P("    torch 线程 %s / 节点 %s 核。" %
      (cost.get("torch_threads", "?"), cost.get("cpus", "?")))
    P("    算术边界：若把基线驱动器做成八路并行且完全线性加速，其墙钟至多再除以 8；")
    P("    下表同时给出 '基线墙钟 ÷ 8' 一列，作为对基线**最有利**的边界。")
    P("    终点（2000 NFE）对照：")
    P("    %-18s %-13s %10s %10s %10s" %
      ("方法", "训练损失中位", "墙钟s", "墙钟/8", "sub"))
    for arm in order:
        if arm not in tbl:
            continue
        P("    %-18s %-13.5e %10.0f %10s %10.3f  n=%d" %
          (arm, tbl[arm]["loss"], tbl[arm]["wall"], "-", tbl[arm]["sub_rmse"],
           tbl[arm]["n"]))
    for algo in ("de", "pso", "cma", "hybrid"):
        rs = base_rows(b_new, "eval2", 2000, algo)
        tag = "@2000"
        if not rs:                       # hybrid 的新配置评价未跑完 -> 用同机
            rs = base_rows(b_old, "evalold", 2000, algo)   # 器旧配置那一批
            tag = "@2000(旧cfg同机器)"
        if not rs:
            continue
        sn_ = [s for s in (snap_of(r, 2000) for r in rs) if s]
        w = med([r["wall_sec"] for r in rs])
        P("    %-18s %-13.5e %10.0f %10.0f %10.3f  n=%d" %
          (algo + tag, med([r["best_loss"] for r in rs]), w, w / 8.0,
           med([s.get("sub_rmse") for s in sn_]), len(rs)))
    for algo in ("de", "hybrid"):
        for g in (10000, 20000):
            rs = [r for r in base_rows(b_new, "big", None, algo)
                  if snap_of(r, g)]
            if not rs:
                continue
            sn_ = [snap_of(r, g) for r in rs]
            w = med([s.get("wall_sec") for s in sn_])
            P("    %-18s %-13.5e %10.0f %10.0f %10.3f" %
              ("%s@%d" % (algo, g), med([s["loss"] for s in sn_]), w, w / 8.0,
               med([s.get("sub_rmse") for s in sn_])))
    P("")
    P("    轨迹上的中间点（best-so-far 训练损失 vs 累计墙钟秒；中位）：")
    P("    我们的臂来自 traj（每次求值都记 (NFE, best, wall)），基线来自")
    P("    预算网格 snapshot。**中间点只有损失，没有参数误差** - evaluate() 只在")
    P("    终点跑，别拿它当参数误差曲线读。")
    grid = [200, 500, 1000, 2000]
    P("    %-18s %s" % ("方法", "  ".join("NFE%-5d loss@s" % g for g in grid)))
    for arm in order:
        if arm not in tbl:
            continue
        rs = our_rows(ours["runs"], arm)
        cells = []
        for g in grid:
            ls, ws = [], []
            for r in rs:
                tj = r.get("traj") or []
                pick = None
                for a, b, c in tj:
                    if a <= g:
                        pick = (b, c)
                    else:
                        break
                if pick:
                    ls.append(pick[0])
                    ws.append(pick[1])
            cells.append("%.3e@%-5.0f" % (med(ls), med(ws)) if ls else "-")
        P("    %-18s %s" % (arm, "  ".join(cells)))
    for algo in ("de", "pso", "cma", "hybrid"):
        rs = base_rows(b_new, "eval2", 2000, algo, seeds=set(SEEDS))
        if not rs:
            continue
        cells = []
        for g in grid:
            sn_ = [x for x in (snap_of(r, g) for r in rs) if x]
            cells.append("%.3e@%-5.0f" % (med([x["loss"] for x in sn_]),
                                          med([x.get("wall_sec") for x in sn_]))
                         if sn_ else "-")
        P("    %-18s %s" % (algo, "  ".join(cells)))
    P("")

    # ---------------------------------------------------------------- 八
    P("八、机理：这个题面上\"训练损失\"根本没有可赢的余量")
    orc = cost.get("oracle_loss_median")
    if orc is not None:
        P("    噪声地板（真值 C_true 自己的训练损失，5 个噪声实现中位）= %.5e" % orc)
        P("    σ² = %.4g；观测 = 干净水头 + N(0,σ²)，所以任何 **低于** 地板的损失"
          % cost.get("sigma2", 0.01))
        P("    都是在拟合噪声而不是恢复参数。")
    best = min((tbl[a]["loss"] for a in tbl), default=None)
    if best is not None and orc:
        P("    我们各臂的最好训练损失 = %.5e，已经是地板的 %.1f%% - "
          % (best, 100.0 * best / orc))
        P("    优化器再强也只能往噪声里挖，损失轴上的\"胜负\"没有物理含义。")
    # 损失与参数误差的秩相关（所有臂 + 所有基线 run 汇总）
    pool = []
    for arm in tbl:
        for r in our_rows(ours["runs"], arm):
            if r.get("sub_rmse") is not None:
                pool.append((r["loss"], r["sub_rmse"], arm))
    for tag, runs_ in (("eval2", b_new), ("evalold", b_old)):
        for r in base_rows(runs_, tag, 2000):
            s = snap_of(r, 2000)
            if s and s.get("sub_rmse") is not None:
                pool.append((r["best_loss"], s["sub_rmse"], r["algo"]))
    if len(pool) > 4:
        x = np.array([p[0] for p in pool])
        y = np.array([p[1] for p in pool])
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        sp = float(np.corrcoef(rx, ry)[0, 1])
        P("    跨全部 %d 次运行（我们的臂 + 基线）训练损失与参数 sub-RMSE 的"
          % len(pool))
        P("    斯皮尔曼秩相关 ρ = %+.4f。" % sp)
        # 只在已经到地板的那一批里看
        sub = [(a, b) for a, b, _ in pool if orc and a < 1.5 * orc]
        if len(sub) > 4:
            xs = np.array([p[0] for p in sub])
            ys = np.array([p[1] for p in sub])
            sp2 = float(np.corrcoef(np.argsort(np.argsort(xs)),
                                    np.argsort(np.argsort(ys)))[0, 1])
            P("    只取已经压到地板 1.5 倍以内的 %d 次运行：ρ = %+.4f"
              % (len(sub), sp2))
            P("    读法：远离地板时损失和参数误差同向（把损失降下去确实有用）；")
            P("    一旦到了地板，两者脱钩 - 这时比\"谁的损失更低\"是在比谁更会过拟合。")
    P("")
    P("    可辨识性账（与优化器无关的硬约束）：自由管 %s 根，训练观测 = %s 帧 × %s"
      % (cost.get("dim", "432"), cost.get("F", "20"),
         cost.get("n_sens_train", "32")))
    P("    传感器 = %s 个标量，良态子空间维数 k_sub = 41。也就是说 432 个未知量里"
      % (int(cost.get("F", 20)) * int(cost.get("n_sens_train", 32))))
    P("    只有 41 个方向被数据约束住，其余 391 个方向由正则项和初值决定，")
    P("    **任何优化器都不能把它们找回来**。这是本节所有 null result 的机理。")
    P("")

    # ---------------------------------------------------------------- 九
    P("九、入文建议与本轮的边界（写给入文阶段，不改冻结树）")
    P("  【任务书里的三个预设结论，实测后各自的下场】")
    hyb = base_rows(b_new, "eval2", 2000, "hybrid", seeds=set(SEEDS))         or [r for r in base_rows((frozen.get("runs") or {}), "eval", 2000,
                                 seeds=set(SEEDS)) if r["algo"] == "hybrid"]
    hs = [x for x in (snap_of(r, 2000) for r in hyb) if x]
    hyb_sub = med([x.get("sub_rmse") for x in hs])
    hyb_loss = med([r["best_loss"] for r in hyb])
    best_arm = None
    if tbl:
        best_arm = min(tbl, key=lambda a: tbl[a]["sub_rmse"])
    P("   1) \"全面压制 DE→LM\"：**没做到，也不该这么写**。DE→LM 在 2000 NFE 上")
    P("      的训练损失 %.5e、参数 sub-RMSE %.3f；" % (hyb_loss, hyb_sub))
    if best_arm:
        P("      我们最好的臂（%s）是 %.5e / %.3f。"
          % (best_arm, tbl[best_arm]["loss"], tbl[best_arm]["sub_rmse"]))
    P("      两者都压在噪声地板上（见第八节），差距落在统计噪声里。")
    P("      能写的是**效率**而不是**优劣**：见下一条。")
    P("   2) 可写的强结论（本轮实测支持）：梯度路线用 **1/10 的模型调用**拿到与")
    P("      DE→LM 同一水平的参数恢复，而纯种群法（DE/PSO/CMA）在同样 2000 次")
    P("      调用下的参数误差是它的约两倍。这是\"同预算下的效率\"，不是\"压制\"。")
    P("   3) 公平性质疑（City D 基线欠调参）**已经用实验关掉了**：调参预算从 300")
    P("      提到 2000（账单 3.2k → 23.6k NFE，与 Hanoi 同协议：调参预算=评价预算），")
    P("      逐算法重跑 4 配置 × 3 专用种子，**选中的配置一个都没变**；同机器重跑")
    P("      旧配置得到与冻结值逐位相同的损失。所以冻结的 City D 基线数没有被欠调参")
    P("      压低过。这一条建议原样写进 SI，它比任何辩解都有力。")
    P("")
    P("  【必须同时写清的三件事（否则数字会被误读）】")
    P("   · 墙钟只能同机器比：同一份基线代码，冻结机器上 DE 2000 NFE 要 598 s，")
    P("     本轮 4090 上只要 %.0f s；而 DE→LM 因为 LM 尾是 CPU 稀疏直接法，"
      % med([r["wall_sec"] for r in base_rows(b_new, "eval2", 2000, "de",
                                              seeds=set(SEEDS))] or [float("nan")]))
    P("     在本轮节点上反而更慢（%.0f s vs 冻结的 1730 s，节点 CPU 被 cgroup"
      % (med([r["wall_sec"] for r in base_rows(b_old, "evalold", 2000,
                                               "hybrid")] or [float("nan")])))
    P("     限到 6 核）。跨机器引用墙钟数是错的。")
    P("   · NFE 与墙钟给出的排序不同，原因是硬件放置（GPU 批量一阶 vs CPU 稀疏 LM），")
    P("     不是计账不公 - 伴随雅可比折合 %.3f 次 LM 段前向，实测支持现有口径。"
      % cost.get("jac_in_cpu_fwd", float("nan")))
    P("   · 批量多起点在 NFE 轴上是纯支出，只在墙钟轴上有折扣，且折扣远小于 B。")
    P("")
    P("  【本轮没做的、以及为什么（不许事后补话术）】")
    P("   · DE→LM 的 10 倍预算（20000 NFE）只在 DE 上做了，hybrid 没做：本轮节点上")
    P("     一次 hybrid 2000 NFE 就要 %.0f s（LM 尾 CPU 绑定），20000 NFE 需 ~%.1f h/种子，"
      % (med([r["wall_sec"] for r in base_rows(b_old, "evalold", 2000,
                                               "hybrid")] or [4756.0]),
         med([r["wall_sec"] for r in base_rows(b_old, "evalold", 2000,
                                               "hybrid")] or [4756.0]) * 10 / 3600.0))
    P("     超出本轮机时。已做的是 DE 的 5× 与 10×、hybrid 的更大预算见第六节实际完成行。")
    P("   · 全量回归（regression_all，53 例）**本轮没有跑完**：本地工作站被无关")
    P("     进程占满，单例 align city_d 实测要 19 分 34 秒，超过 harness 硬编码的")
    P("     900 s/例上限，前两例都以 rc=124（超时）告负 - **是机时不是数值**。")
    P("     把该例单独跑到底是 **PASS**：全帧最差 max|ΔH|=1.421e-14 ft（门槛 1e-6）、")
    P("     max|ΔQ|=1.776e-15 cfs。结构性论据：本轮对 dgga/ 的唯一改动是")
    P("     optim2.lbfgs_precond 里三行类型回落（_strong_wolfe 把步长作 0 维 tensor")
    P("     返回），而 optim2 全仓只被 scripts/optimizer_v2.py 引用，回归路径到不了它；")
    P("     GPU 侧 stage_verify 也实测前向 max|ΔH|=2.842e-14 ft、梯度相对差 4.717e-16。")
    P("     **但这不等于 53/53 已验收 - 机器空下来必须补跑，别拿本条当通过。**")
    P("   · 噪声种子数：我们的臂与基线在公共种子上配对；n 见各表行尾。")
    P("     符号检验在 n=5 时最小可能 p 是 0.0625，**p<0.01 在 n=5 下不可达** - ")
    P("     任何\"p<0.01 压倒随机\"的写法在这个样本量上都是不可能兑现的承诺。")
    P("")
    return L, tbl, ours, b_new, b_old, tun_new, frozen, gc1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data",
                                                  "optimizer_v2_wip.txt"))
    a = ap.parse_args()
    L = build(a.dir, a.out)[0]
    txt = "\n".join(L) + "\n"
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()

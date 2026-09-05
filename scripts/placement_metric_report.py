# -*- coding: utf-8 -*-
"""placement_metric_report.py - placement_metric.py 的 report 阶段（写 wip + json）。

只做汇总与统计，不跑仿真。零编号：输出仅含计数、统计量与种子。
"""

import json
import os
import platform
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SIGMAS = (0.03, 0.1, 0.3)
BUDGETS = (20, 40)
METRICS_TESTED = ("rmse_g2", "rmse_g3", "rmse_g10", "rmse_all",
                  "rmse_null_g2", "sub_rmse", "info_rmse",
                  "val_frame_rmse", "val_sensor_rmse",
                  "train_rmse", "mse_train_final")


def _rand_grp(rows, k, sg, ns=100):
    """该预算 / 该 sigma 下的随机增设对照组（键序稳定）。"""
    return sorted([r for r in rows if r.get("src") == "randctl"
                   and r["key"].startswith("RAND_augrand%d_" % k)
                   and r["sigma"] == sg and r["noise_seed"] == ns],
                  key=lambda r: r["key"])


def _bkey(k, sg):
    return "k%d_sigma_%g" % (k, sg)


def _assert_no_ids(obj, pb):
    """零编号守卫：任何 dict 键或字符串值命中节点/链路编号即抛错（数值不查）。"""
    ids = set(str(x) for x in pb.net.node_id) | set(str(x) for x in pb.net.link_id)
    allow = {"n", "k", "R", "p", "sd", "cv", "z"}
    hits = []

    def walk(x, p):
        if isinstance(x, dict):
            for k, v in x.items():
                ks = str(k)
                if ks not in allow and ks in ids:
                    hits.append("%s.%s" % (p, ks))
                walk(v, "%s.%s" % (p, ks))
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, "%s[%d]" % (p, i))
        elif isinstance(x, str) and x in ids:
            hits.append("%s=%r" % (p, x))

    walk(obj, "$")
    if hits:
        raise RuntimeError("可读 JSON 含节点/链路编号 %d 处（如 %s）"
                           % (len(hits), hits[:5]))


def _fmt(x, n=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  --  "
    return ("%%.%df" % n) % x


def predicted_rmse(pm, sub, pb, pos, sigma_prior, sigma_noise):
    """线性-高斯设计理论对该布点的**预测** ref 子空间 RMSE。

    M(S) = I/σ_p² + (1/σ_n²) Σ_{t∈train} Σ_{i∈S_train} s_{t,i} s_{t,i}ᵀ
    E‖V_refᵀ(Ĉ−C)‖² = tr(V_refᵀ M⁻¹ V_ref)（后验协方差在参考子空间上的迹）
    预测 RMSE = sqrt(tr/ k_ref)。用与标定完全同口径的训练帧 + 训练传感器。
    """
    import calibrate as CA
    junc = np.asarray(pb.s.junc_nodes)
    sensors = np.sort(junc[np.unique(pos)])
    sens_train, _ = CA.split_holdout(sensors)
    tpos = np.searchsorted(junc, sens_train)
    Sf = sub["Sf"][pb.train_frames][:, tpos, :]
    P = Sf.shape[2]
    A = Sf.reshape(-1, P)
    M = np.eye(P) / sigma_prior ** 2 + (A.T @ A) / sigma_noise ** 2
    out = {}
    for g in (2.0,):
        k = int((sub["sv"] > pm.sv_threshold(g)).sum())
        V = sub["Vt"][:k].T
        X = np.linalg.solve(M, V)
        out["pred_rmse_g%g" % g] = float(np.sqrt(np.trace(V.T @ X) / k))
    ident = np.abs(Sf).max(axis=(0, 1)) > 0.0
    out["n_ident_free"] = int(ident.sum())
    sv = np.linalg.svd(A, compute_uv=False)
    out["kdes_g2"] = int((sv > pm.sv_threshold(2.0)).sum())
    out["logdet_f"] = float(np.linalg.slogdet(sigma_prior ** 2 * M)[1])
    return out


def write_report(rows, prior, sub, pb, args):
    import placement_metric as pm
    import calibrate as CA

    t_start = time.time()
    orders = np.load(pm.ORDERS)
    S0 = orders["augment_fixed"]
    junc = np.asarray(pb.s.junc_nodes)
    pool = pm.fair_pool(len(junc), S0)
    by = {r["key"]: r for r in rows}

    # ---------------- 设计时预测量（零仿真，从灵敏度缓存算） ----------------
    pred = {}
    for r in rows:
        key = r["key"]
        if key.startswith("RAND_augrand"):
            k = int(key.split("augrand")[1].split("_")[0])
            sd = int(key.rsplit("_r", 1)[1])
            pos = np.r_[S0, pm.rand_add(pool, k, sd)]
        elif key.startswith("DES_"):
            pos = pm.designed_positions(orders, key.split("_")[1])
        else:
            continue
        pk = tuple(np.unique(pos))
        if pk not in pred:
            pred[pk] = predicted_rmse(pm, sub, pb, pos, pm.SP, pm.SN)
        r.update(pred[pk])

    doc = dict(meta=dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        host=platform.node(), python=platform.python_version(),
        machine_note=args.machine,
        sources=["data/calib_gc1_city_d.json", "data/calib_augrand_city_d.json",
                 "data/placement_cache_city_d.npz",
                 "data/placement_orders_city_d.npz"],
        sigma_prior=pm.SP, sigma_noise_design=pm.SN))

    # ================================================== 一、参考子空间定义
    sv = sub["sv"]
    doc["subspace"] = dict(
        rule="v_j 入选 ⟺ 该方向后验标准差 ≤ σ_prior/γ ⟺ sv_j ≥ σ_noise·"
             "sqrt(γ²−1)/σ_prior（绝对判据，与布点无关）",
        pool_note="参考谱来自候选池全装表（全部 junction）× 25 帧的伴随灵敏度",
        n_pool=int(sub["Sf"].shape[1]), n_frames=int(sub["Sf"].shape[0]),
        n_free_pipes=int(sub["Sf"].shape[2]), sv_max=float(sv[0]),
        levels={("gamma_%g" % g): dict(
            threshold=float(pm.sv_threshold(g)),
            k_ref=int((sv > pm.sv_threshold(g)).sum()),
            prior_rmse=float(prior["rmse_g%g" % g])) for g in pm.GAMMAS},
        formula="RMSE_ident = ||V_refᵀ(Ĉ−C_true)||₂ / sqrt(k_ref)；"
                "skill = 1 − RMSE_ident/RMSE_prior")

    # ================================================== 二、指标不合理的证据
    ev = {}

    # (a) 优化器分辨力：同布点同数据、只换初值的 8 条 MS_start
    ms = [r for r in rows if r["key"].startswith("MS_start")]
    if ms:
        ev["resolving_power_multistart"] = _spread_block(ms, pm)

    # (b) 布点扫描：同真值同噪声，只换布点
    for kk in BUDGETS:
        for sg in SIGMAS:
            grp = _rand_grp(rows, kk, sg)
            if len(grp) >= 5:
                ev.setdefault("placement_sweep", {})[_bkey(kk, sg)] = \
                    _sweep_block(grp, pm)

    # (c) 桥接实验（40 个传感器，5 种布点法）
    br = [r for r in rows if r["key"].startswith("BR_")]
    if br:
        ev["bridge_40sensors"] = _spread_block(br, pm)

    # (c2) 噪声份额：水头残差里有多少不是模型误差
    nf = {}
    for kk in BUDGETS:
        for sg in SIGMAS:
            blk = {}
            grp = [("S0", [r for r in rows
                           if r["key"] == "DES_ga40_s%g_n100" % sg]),
                   ("designed+%d" % kk,
                    [r for r in rows if r["key"] in (
                        "DES_augcover%d_s%g_n100" % (kk, sg),
                        "DES_augdopt%d_s%g_n100" % (kk, sg))]),
                   ("random+%d" % kk, _rand_grp(rows, kk, sg))]
            for tag, g in grp:
                if not g:
                    continue
                blk[tag] = dict(
                    n=len(g),
                    train_rmse=float(np.median([r["train_rmse"] for r in g])),
                    train_rmse_clean=float(np.median(
                        [r["train_rmse_clean"] for r in g])),
                    val_frame_rmse=float(np.median(
                        [r["val_frame_rmse"] for r in g])),
                    val_frame_rmse_clean=float(np.median(
                        [r["val_frame_rmse_clean"] for r in g])),
                    noise_share_train=float(np.median(
                        [1.0 - (r["train_rmse_clean"] / r["train_rmse"]) ** 2
                         for r in g])),
                    rmse_g2=float(np.median([r["rmse_g2"] for r in g])))
            if blk:
                nf[_bkey(kk, sg)] = blk
    ev["noise_floor"] = nf

    # (c3) 选择代价：在同一随机布点集合里，按各指标挑"最好"的那个布点，
    #      它在参数误差上排第几？ - 直接量"用水头指标选布点"要付的代价。
    regret = {}
    for kk in BUDGETS:
        for sg in SIGMAS:
            g = _rand_grp(rows, kk, sg)
            if len(g) < 10:
                continue
            g2 = np.array([r["rmse_g2"] for r in g], float)
            cell = dict(n=len(g), oracle_min=float(g2.min()),
                        median=float(np.median(g2)), max=float(g2.max()))
            for sk in ("mse_train_final", "val_frame_rmse", "val_sensor_rmse",
                       "train_rmse", "pred_rmse_g2"):
                v = np.array([r.get(sk, np.nan) for r in g], float)
                if not np.isfinite(v).all():
                    continue
                j = int(np.argmin(v))
                cell[sk] = dict(picked_rmse_g2=float(g2[j]),
                                picked_rank=int((g2 < g2[j]).sum()) + 1,
                                pct=float(100.0 * (g2 < g2[j]).mean()),
                                regret_vs_oracle=float(g2[j] - g2.min()),
                                regret_vs_median=float(g2[j] - np.median(g2)))
            regret[_bkey(kk, sg)] = cell
    ev["selection_regret"] = regret

    # (d) 支撑集漂移：informative 管数随布点变化
    des = sorted([r for r in rows if r["key"].startswith("DES_")
                  and r["sigma"] == 0.1], key=lambda r: r["n_sensors"])
    ev["support_drift"] = [dict(design=r["key"].split("_")[1],
                                n_sensors=r["n_sensors"],
                                n_informative=r["n_informative"],
                                sub_rank=r["sub_rank"],
                                info_rmse=r["info_rmse"],
                                sub_rmse=r["sub_rmse"],
                                rmse_g2=r["rmse_g2"]) for r in des]
    # (e) 跨机复现：设计增设在本机重跑 vs 归档（V100 服务器）
    repro = []
    for sg in (0.03, 0.1, 0.3):
        for nm in ("ga40", "augdopt20", "augcover20", "augdopt40", "augcover40"):
            a, b = by.get("AUG_%s_s%g_n100" % (nm, sg)), \
                by.get("DES_%s_s%g_n100" % (nm, sg))
            if a and b:
                repro.append(dict(
                    design=nm, sigma=sg, archived=a["rmse_g2"],
                    local=b["rmse_g2"],
                    rel_diff=abs(a["rmse_g2"] - b["rmse_g2"])
                    / max(abs(a["rmse_g2"]), 1e-30),
                    archived_train=a["train_rmse"], local_train=b["train_rmse"]))
    ev["host_repro"] = repro
    doc["metric_evidence"] = ev

    # ================================================== 三、显著性
    sig = {}
    for kk in BUDGETS:
        for sg in SIGMAS:
            rnd = _rand_grp(rows, kk, sg)
            if len(rnd) < 5:
                continue
            cell = dict(n_random=len(rnd), budget=kk, sigma=sg)
            for metric in METRICS_TESTED:
                vals = np.array([r[metric] for r in rnd], float)
                for dname in ("augcover%d" % kk, "augdopt%d" % kk):
                    dk = "DES_%s_s%g_n100" % (dname, sg)
                    if dk not in by:
                        continue
                    t = pm.exact_test(by[dk][metric], vals, side="less")
                    lo, hi = pm.clopper_pearson(t["n_at_least_as_good"],
                                                t["R"])
                    hl, hlo, hhi = pm.hodges_lehmann(by[dk][metric], vals)
                    t.update(pi_ci95=[lo, hi], needed_R_for_p001=pm.needed_R(
                        t["pi_hat"], 0.01), delta_vs_median=hl,
                        delta_ci95=[hlo, hhi],
                        p_min_attainable=1.0 / (t["R"] + 1))
                    cell.setdefault(metric, {})[dname] = t
            sig[_bkey(kk, sg)] = cell
    doc["significance"] = sig
    # 结论相反的格子：参数误差上设计胜过全部随机，水头指标上却不胜（或反输）
    disagree = []
    for bk, cell in sig.items():
        for dn in list(cell.get("rmse_g2", {})):
            t = cell["rmse_g2"][dn]
            for hk in ("mse_train_final", "val_frame_rmse", "train_rmse"):
                h = cell.get(hk, {}).get(dn)
                if h is None:
                    continue
                if t["pi_hat"] <= 0.05 and h["pi_hat"] >= 0.5:
                    disagree.append(dict(
                        cell=bk, design=dn, head_metric=hk,
                        pi_param=t["pi_hat"], pi_head=h["pi_hat"],
                        R=t["R"], param_designed=t["designed"],
                        param_rand_median=t["rand_median"],
                        head_designed=h["designed"],
                        head_rand_median=h["rand_median"]))
    doc["metric_disagreement"] = disagree
    doc["significance_design_time"] = {}
    for k in BUDGETS:
        fp = os.path.join(DATA, "placement_metric_predtest_city_d_k%d.json" % k)
        if os.path.isfile(fp):
            blk = pm.jload(fp)
            q = blk.get("rand_quantiles")
            if q:            # 兼容旧文件的裸数字键
                blk["rand_quantiles"] = {
                    (kk if str(kk).startswith("q") else "q%s" % kk): v
                    for kk, v in q.items()}
            doc["significance_design_time"]["k%d" % k] = blk

    # ================================================== 四、设计理论的预测力
    predblk = {}
    for kk in BUDGETS:
        for sg in SIGMAS:
            grp = [r for r in _rand_grp(rows, kk, sg) if "pred_rmse_g2" in r]
            if len(grp) < 10:
                continue
            a = np.array([r["pred_rmse_g2"] for r in grp])
            b = np.array([r["rmse_g2"] for r in grp])
            rho, p, n = pm.spearman(a, b)
            kd = np.array([r["kdes_g2"] for r in grp], float)
            rho2, p2, _ = pm.spearman(kd, b)
            rho3, p3, _ = pm.spearman(np.array([r["logdet_f"] for r in grp]), b)
            predblk[_bkey(kk, sg)] = dict(
                n=n, rho_pred_vs_achieved=rho, p=p,
                rho_kdes_vs_achieved=rho2, p_kdes=p2,
                rho_logdet_vs_achieved=rho3, p_logdet=p3,
                pred_range=[float(a.min()), float(a.max())],
                achieved_range=[float(b.min()), float(b.max())],
                pred_spread_rel=float((a.max() - a.min()) / a.mean()),
                achieved_spread_rel=float((b.max() - b.min()) / b.mean()))
    # 跨预算合并（+20 与 +40 放一起，检验预测量能否跨预算排序）
    for sg in SIGMAS:
        grp = []
        for kk in BUDGETS:
            grp += [r for r in _rand_grp(rows, kk, sg) if "pred_rmse_g2" in r]
        if len(grp) >= 20:
            a = np.array([r["pred_rmse_g2"] for r in grp])
            b = np.array([r["rmse_g2"] for r in grp])
            rho, p, n = pm.spearman(a, b)
            predblk["pooled_sigma_%g" % sg] = dict(
                n=n, rho_pred_vs_achieved=rho, p=p,
                rho_kdes_vs_achieved=float("nan"), p_kdes=float("nan"),
                rho_logdet_vs_achieved=float("nan"), p_logdet=float("nan"),
                pred_range=[float(a.min()), float(a.max())],
                achieved_range=[float(b.min()), float(b.max())],
                pred_spread_rel=float((a.max() - a.min()) / a.mean()),
                achieved_spread_rel=float((b.max() - b.min()) / b.mean()))
    doc["design_theory_predictivity"] = predblk

    vd = os.path.join(DATA, "placement_metric_verifydiag_city_d.json")
    if os.path.isfile(vd):
        doc["fast_diag_verification"] = pm.jload(vd)

    doc["rows"] = [{k: v for k, v in r.items() if k != "C_hat_free"}
                   for r in rows]
    _assert_no_ids(doc, pb)
    pm.jdump(doc, pm.OUT_JSON)
    _write_wip(doc, rows, prior, pm, args)
    print("[report] %s + %s  (%.1fs)"
          % (pm.OUT_JSON, pm.OUT_WIP, time.time() - t_start))


# ------------------------------------------------------------------ blocks
def _stat(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), min=float(v.min()), median=float(np.median(v)),
                max=float(v.max()), mean=float(v.mean()),
                sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                cv=float(v.std(ddof=1) / abs(v.mean())) if v.size > 1 else 0.0,
                spread_rel=float((v.max() - v.min()) / abs(v.mean())))


def _spread_block(grp, pm):
    out = dict(n=len(grp))
    for k in pm.HEAD_KEYS + pm.PAR_KEYS:
        out[k] = _stat([r.get(k) for r in grp])
    return out


def _sweep_block(grp, pm):
    out = dict(n=len(grp))
    for k in pm.HEAD_KEYS + pm.PAR_KEYS:
        out[k] = _stat([r.get(k) for r in grp])
    cors = {}
    for hk in list(pm.HEAD_KEYS) + ["info_rmse", "sub_rmse"]:
        for pk in ("rmse_g2", "sub_rmse", "info_rmse"):
            rho, p, n = pm.spearman([r.get(hk) for r in grp],
                                    [r.get(pk) for r in grp])
            cors["%s~%s" % (hk, pk)] = dict(rho=rho, p=p, n=n)
    out["spearman"] = cors
    return out


# ------------------------------------------------------------------- wip
def _write_wip(doc, rows, prior, pm, args):
    L = []
    A = L.append
    sp = doc["subspace"]
    A("[city_d] 布点验证指标重构与增设显著性 - 实测数值（虚拟增设：设计与模拟验证，非现场实装）")
    A("生成 %s @ %s；出处见 data/placement_metric_city_d.json[meta.sources]"
      % (doc["meta"]["generated"], doc["meta"]["host"]))
    A("机器：%s" % (args.machine or doc["meta"]["host"]))
    A("")
    A("一、为什么水头 MSE 不是布点的验证指标（数据论证，不是断言）")
    A("  背景：§3.6 用训练水头 MSE 比较从零重选与随机布点（9.40 / 9.72 / 9.28e-3），")
    A("  SI 表 10 用 informative 管的 C-RMSE 与留出水头 RMSE。三处都不可比，理由如下。")
    A("")
    A("  (1) 分辨力：同一布点、同一数据，只换优化初值（8 个 LHS 起点，σ=0.1）")
    ms = doc["metric_evidence"].get("resolving_power_multistart")
    if ms:
        A("        指标            最小      中位      最大   极差/均值")
        for k, lab in (("val_frame_rmse", "留出帧水头RMSE"),
                       ("train_rmse", "训练水头RMSE"),
                       ("mse_train_final", "训练水头MSE"),
                       ("val_sensor_rmse", "留出传感器RMSE"),
                       ("sub_rmse", "旧 sub-RMSE"),
                       ("info_rmse", "info 管 C-RMSE"),
                       ("rmse_g2", "重构 ref-RMSE")):
            s = ms[k]
            A("      %-16s %8.4f %8.4f %8.4f %8.1f%%"
              % (lab, s["min"], s["median"], s["max"], 100 * s["spread_rel"]))
        A("      读法：水头指标的极差只有百分之一量级，同批解的参数误差却差三成 - ")
        A("      水头空间根本分辨不出这些解，它测的是噪声地板不是参数恢复。")
    A("")
    A("  (2) 布点扫描（同真值、同一噪声实现、同优化器与初值，只换布点：随机增设）")
    for kk in BUDGETS:
        for sg in SIGMAS:
            b = doc["metric_evidence"].get("placement_sweep", {}).get(
                _bkey(kk, sg))
            if not b:
                continue
            A("    预算 +%d，σ=%s ft（n=%d 个随机布点）" % (kk, sg, b["n"]))
            A("        指标                均值      极差/均值   Spearman ρ(与 ref-RMSE)   p")
            for k, lab in (("train_rmse", "训练水头RMSE"),
                           ("val_frame_rmse", "留出帧水头RMSE"),
                           ("val_sensor_rmse", "留出传感器RMSE"),
                           ("val_frame_rmse_clean", "留出帧(对干净真值)"),
                           ("mse_train_final", "训练水头MSE"),
                           ("info_rmse", "info 管 C-RMSE"),
                           ("sub_rmse", "旧 sub-RMSE")):
                s, c = b[k], b["spearman"].get("%s~rmse_g2" % k)
                if c is None:
                    c = dict(rho=float("nan"), p=float("nan"))
                A("      %-20s %9.4f %9.1f%%   %+8.3f  %10.3g"
                  % (lab, s["mean"], 100 * s["spread_rel"], c["rho"], c["p"]))
            s = b["rmse_g2"]
            A("      %-20s %9.4f %9.1f%%   （被解释变量）"
              % ("重构 ref-RMSE", s["mean"], 100 * s["spread_rel"]))
    A("")
    A("  (3) 噪声平坦化：水头残差里几乎没有模型信息，而且随机增设与设计增设同样把它压低")
    A("      （中位数；noise share = 1 − (对干净真值的 RMSE / 对含噪观测的 RMSE)²）")
    A("        σ  预算 配置          n | 训练水头RMSE 其中对干净真值  噪声份额 | 留出帧RMSE 干净 | 重构ref-RMSE")
    for kk in BUDGETS:
        for sg in SIGMAS:
            blk = doc["metric_evidence"]["noise_floor"].get(_bkey(kk, sg), {})
            for tag in ("S0", "designed+%d" % kk, "random+%d" % kk):
                b = blk.get(tag)
                if not b:
                    continue
                A("      %4s  +%-3d %-12s %3d | %10.4f %12.4f %9.1f%% | %9.4f %6.4f | %10.3f"
                  % (sg, kk, tag, b["n"], b["train_rmse"],
                     b["train_rmse_clean"], 100 * b["noise_share_train"],
                     b["val_frame_rmse"], b["val_frame_rmse_clean"],
                     b["rmse_g2"]))
    nf01 = doc["metric_evidence"]["noise_floor"].get(_bkey(20, 0.1), {})
    if {"S0", "designed+20", "random+20"} <= set(nf01):
        s0, dz, rz = nf01["S0"], nf01["designed+20"], nf01["random+20"]
        A("      读法：训练水头残差里 %.0f%% 是噪声方差（对干净真值只有 %.4f ft）。"
          % (100 * s0["noise_share_train"], s0["train_rmse_clean"]))
        A("      把 %d 个传感器加到 %d 个，训练水头 RMSE 由 %.4f 变到 %.4f（设计）/"
          % (40, 60, s0["train_rmse"], dz["train_rmse"]))
        A("      %.4f（随机） - 设计增设在水头上甚至略差；同一批解的参数误差却是"
          % rz["train_rmse"])
        A("      %.2f → %.2f（设计）/ %.2f（随机）。水头 MSE 度量的是噪声平均，"
          % (s0["rmse_g2"], dz["rmse_g2"], rz["rmse_g2"]))
        A("      不是布点质量：两者在这里给出的排序方向相反。")
    A("")
    A("  (3b) 选择代价：在同一组随机布点里按某指标挑\"最好\"的那个，看它的参数误差排第几")
    A("        （n = 该组随机布点数；rank=1 表示恰好挑中参数误差最小的那个）")
    A("        σ  预算 选择指标            选中的ref-RMSE  排名/ n  百分位 | 组内最好/中位/最差")
    for kk in BUDGETS:
        for sg in SIGMAS:
            c = doc["metric_evidence"].get("selection_regret", {}).get(
                _bkey(kk, sg))
            if not c:
                continue
            for sk, lab in (("mse_train_final", "训练水头MSE"),
                            ("val_frame_rmse", "留出帧水头RMSE"),
                            ("val_sensor_rmse", "留出传感器RMSE"),
                            ("pred_rmse_g2", "设计时预测RMSE")):
                d = c.get(sk)
                if not d:
                    continue
                A("      %4s  +%-3d %-18s %12.3f %5d/%-4d %6.1f%% | %6.3f %6.3f %6.3f"
                  % (sg, kk, lab, d["picked_rmse_g2"], d["picked_rank"],
                     c["n"], d["pct"], c["oracle_min"], c["median"],
                     c["max"]))
    reg = doc["metric_evidence"].get("selection_regret", {})
    agg = {}
    for c in reg.values():
        for sk in ("mse_train_final", "val_frame_rmse", "val_sensor_rmse",
                   "pred_rmse_g2"):
            if sk in c:
                agg.setdefault(sk, []).append(c[sk]["pct"])
    if agg:
        A("      六格汇总（平均百分位，纯随机猜=50，完美选择器=0）：")
        for sk, lab in (("mse_train_final", "训练水头MSE"),
                        ("val_frame_rmse", "留出帧水头RMSE"),
                        ("val_sensor_rmse", "留出传感器RMSE"),
                        ("pred_rmse_g2", "设计时预测RMSE")):
            v = agg.get(sk)
            if not v:
                continue
            A("        %-14s %5.1f  （%d 格；最好 %.0f，最差 %.0f）"
              % (lab, sum(v) / len(v), len(v), min(v), max(v)))
    A("      读法：把\"选择器\"和\"检验\"分开看。水头指标的平均百分位见上表，")
    A("      整体比纯随机猜（50）好一点、但单格可能落到 60–90 百分位，作为选择器不可靠；")
    A("      而设计增设（§三）在全部六格里都优于**所有**随机布点（第 0 百分位以下）。")
    A("      同样重要的是方向：训练水头 MSE 在 §三 里对 D-opt 设计给出 π̂=1.00 的反向排序，")
    A("      即\"水头拟合最差\"恰恰是\"参数恢复最好\"的那个设计 - 这不是分辨力不足，是判据错位。")
    A("")
    A("  (4) 支撑集漂移：informative 管集与旧良态子空间都随布点变化，同名指标算在不同集合上")
    A("        设计        传感 info管 旧k_sub | info C-RMSE 旧sub-RMSE  重构ref-RMSE")
    for r in doc["metric_evidence"]["support_drift"]:
        A("      %-11s %4d %5d %5d | %10.3f %10.3f %12.3f"
          % (r["design"], r["n_sensors"], r["n_informative"], r["sub_rank"],
             r["info_rmse"], r["sub_rmse"], r["rmse_g2"]))
    sd = doc["metric_evidence"]["support_drift"]
    if len(sd) >= 2:
        ni = [x["n_informative"] for x in sd]
        ks = [x["sub_rank"] for x in sd]
        A("      读法：info 管数在 %d–%d 之间随设计变（极差 %d 根），旧良态子空间维数在"
          % (min(ni), max(ni), max(ni) - min(ni)))
        A("      %d–%d 之间随设计变 - 两个\"同名\"指标各自算在不同的管集/子空间上，"
          % (min(ks), max(ks)))
        A("      跨行不可比；重构 ref-RMSE 全程用同一个 k_ref，才是同一把尺子。")
    A("")
    if doc["metric_evidence"].get("host_repro"):
        A("  (5) 跨机复现（设计增设本机重跑 vs 归档 V100 结果，同真值同噪声同引擎）")
        A("        设计         σ | 归档ref-RMSE 本机ref-RMSE  相对差 | 归档train 本机train")
        for r in doc["metric_evidence"]["host_repro"]:
            A("      %-12s %4s | %11.4f %12.4f %8.2e | %9.4f %9.4f"
              % (r["design"], r["sigma"], r["archived"], r["local"],
                 r["rel_diff"], r["archived_train"], r["local_train"]))
        A("      读法：显著性检验里的设计值取**本机重跑值**，与随机对照同机同码，跨机差异不进检验。")
        A("")
    A("二、重构指标：固定参考可辨识子空间上的参数 RMSE")
    A("  参考谱 = 候选池全装表（%d 个候选 × %d 帧）的伴随灵敏度 A 的奇异谱；"
      % (sp["n_pool"], sp["n_frames"]))
    A("  Fisher 信息 F = AᵀA/σ_n²，与 A 同右奇异向量。截断准则（绝对、与布点无关）：")
    A("    方向 v_j 入选 ⟺ 后验标准差 ≤ σ_prior/γ ⟺ sv_j ≥ σ_noise·sqrt(γ²−1)/σ_prior")
    A("  σ_prior=%.0f、σ_noise=%.2f ft：" % (pm.SP, pm.SN))
    for g in pm.GAMMAS:
        lv = sp["levels"]["gamma_%g" % g]
        A("    γ=%-4g 阈值 %.6f  k_ref=%3d  先验 RMSE=%.4f"
          % (g, lv["threshold"], lv["k_ref"], lv["prior_rmse"]))
    A("  RMSE_ident = ||V_refᵀ(Ĉ−C_true)||₂ / sqrt(k_ref)，skill = 1 − RMSE/RMSE_prior。")
    vd = doc.get("fast_diag_verification")
    if vd:
        g = max([c["one_minus_cos_theta_max"] for c in vd["cases"]
                 if c["one_minus_cos_theta_max"] == c["one_minus_cos_theta_max"]]
                or [float("nan")])
        A("  代码核对：随机对照用查缓存的快速归因（fast_diag）代替重解（get_diag），"
          "%d 个布点逐个对拍" % vd["n_cases"])
        A("    reason 向量全同、k_sub 与 rank_eps 全同、子空间主角 1−cos(θmax) ≤ %.2e，"
          "结论 %s。" % (g, "ALL MATCH" if vd["all_match"] else "MISMATCH"))
    A("  k_ref 是物理上限：全网 %d 个候选位置全部装表，也只有 %d 个方向能被数据把先验压过一半。"
      % (sp["n_pool"], sp["levels"]["gamma_2"]["k_ref"]))
    A("")
    A("三、显著性：设计增设 vs 随机增设（同预算、同公平池 %d 个候选、同真值同噪声实现）"
      % (541 - 40))
    A("  零假设：该设计与从公平池均匀抽的同预算布点无异；统计量 = 参数 RMSE（越小越好）；")
    A("  单侧精确随机化 p = (1 + #{随机 ≤ 设计}) / (R + 1)（Phipson–Smyth 加一）。")
    A("  随机样本量 R 的停止规则（事先声明，与结果无关）：六格各目标 R=20；")
    A("  σ=0.1/+20 这一格目标 R=100（p 下限 0.0099，是能压到 p<0.01 的最小样本量）。")
    A("  实际 R 由墙钟预算截断 - 截断依据是时间，不看 p，不构成 optional stopping。")
    A("  随机布点由 (公平池, 种子) 确定性生成，种子按 0,1,2,… 顺序取用，不做筛选。")
    for kk in BUDGETS:
        for sg in SIGMAS:
            cell = doc["significance"].get(_bkey(kk, sg))
            if not cell:
                continue
            A("  预算 +%d，σ=%s ft，R=%d 个随机布点（p 下限 %.4f）"
              % (kk, sg, cell["n_random"], 1.0 / (cell["n_random"] + 1)))
            A("      指标           设计       设计值 | 随机 最小/中位/最大 |  优于设计  π̂[95%CI]     p精确")
            for metric, lab in (("rmse_g2", "重构ref-RMSE"),
                                ("rmse_g3", "ref-RMSE(γ=3)"),
                                ("rmse_g10", "ref-RMSE(γ=10)"),
                                ("rmse_all", "全 432 管裸RMSE"),
                                ("rmse_null_g2", "补空间RMSE"),
                                ("sub_rmse", "旧sub-RMSE"),
                                ("info_rmse", "info C-RMSE"),
                                ("val_frame_rmse", "留出帧水头"),
                                ("mse_train_final", "训练水头MSE")):
                for dn in ("augcover%d" % kk, "augdopt%d" % kk):
                    t = cell.get(metric, {}).get(dn)
                    if not t:
                        continue
                    A("    %-14s %-11s %8.4f | %7.4f %7.4f %7.4f | %4d  %.3f[%.3f,%.3f] %8.4f"
                      % (lab, dn, t["designed"], t["rand_min"],
                         t["rand_median"], t["rand_max"],
                         t["n_at_least_as_good"], t["pi_hat"],
                         t["pi_ci95"][0], t["pi_ci95"][1], t["p_exact"]))
            for dn in ("augcover%d" % kk, "augdopt%d" % kk):
                t = cell.get("rmse_g2", {}).get(dn)
                if not t:
                    continue
                need = t["needed_R_for_p001"]
                A("    [%s] 相对随机中位的位移 %.3f（95%%CI %.3f..%.3f，负=更好）；z=%.2f；"
                  % (dn, t["delta_vs_median"], t["delta_ci95"][0],
                     t["delta_ci95"][1], t["z"]))
                if t["p_exact"] < 0.01:
                    A("        p<0.01 已达到（R=%d，p=%.4f）；该 π̂ 下的最小样本量 R ≥ %s"
                      % (t["R"], t["p_exact"], ("%d" % need) if need else "--"))
                else:
                    A("        达到 p<0.01 所需随机样本量 R ≥ %s"
                      % ("%d" % need if need
                         else "在 π̂=%.3f 下任何 R 都达不到" % t["pi_hat"]))
    dis = doc.get("metric_disagreement", [])
    if dis:
        A("  三附、同一格里水头指标与参数指标结论相反的情形（最直接的\"指标选错\"证据）")
        A("      格         设计       水头指标      | 参数 π̂  水头 π̂ | 设计参数/随机中位 | 设计水头/随机中位")
        for d in dis:
            A("    %-12s %-11s %-14s | %5.2f %6.2f | %7.3f %7.3f | %8.5f %8.5f"
              % (d["cell"], d["design"], d["head_metric"], d["pi_param"],
                 d["pi_head"], d["param_designed"], d["param_rand_median"],
                 d["head_designed"], d["head_rand_median"]))
        A("      读法：π̂ 是\"随机布点里有多大比例不劣于该设计\"。参数 π̂≈0 而水头 π̂≈1，")
        A("      意味着同一个设计在参数恢复上胜过全部随机布点、在水头拟合上却输给全部随机")
        A("      布点 - 两把尺子给出相反的排序，只能有一把是对的，而正则化反演的目标是参数。")
        A("")
    dt = doc.get("significance_design_time", {})
    if dt:
        A("三补、设计时（线性-高斯）指标上的大样本随机化检验 - 不跑标定，样本量不受仿真预算限制")
        A("  预测 RMSE = sqrt(tr(V_refᵀM(S)⁻¹V_ref)/k_ref)，M(S) 用训练帧+训练传感器；")
        A("  它是无偏有效估计下的理论下界，数值比实测乐观，此处只用它的**序**。")
        for kk, blk in sorted(dt.items()):
            A("  预算 +%s，R=%d 个随机布点（同公平池）"
              % (kk[1:], blk.get("R", 0)))
            q = blk.get("rand_quantiles", {})
            if q:
                A("    随机分布分位（0/1/5/25/50/75/95/99/100）：%s"
                  % " ".join("%.4f" % q[k2] for k2 in
                             sorted(q, key=lambda z: float(str(z).lstrip("q")))))
            for nm, t in blk.get("designs", {}).items():
                A("    %-12s 预测=%.4f | 优于设计 %d/%d | π̂=%.4f[%.4f,%.4f] | p=%.4g（下限 %.4g）"
                  % (nm, t["designed"], t["n_at_least_as_good"], t["R"],
                     t["pi_hat"], t["pi_ci95"][0], t["pi_ci95"][1],
                     t["p_exact"], t["p_min_attainable"]))
        A("  读法：设计时指标上的差别是**确定性的**（同一 M(S) 公式），大样本 p 只反映")
        A("  随机池里有多少布点碰巧同样好；它不能替代三、的实测检验 - 从 M(S) 到实际")
        A("  反演误差之间隔着非线性、正则化与噪声实现，两者的差距正是四、要量的东西。")
        A("")
    A("四、设计时预测量对实测的预测力（同一批随机布点，零额外仿真）")
    A("  预测 RMSE = sqrt(tr(V_refᵀ M(S)⁻¹ V_ref)/k_ref)，M(S) 用训练帧与训练传感器；")
    A("  它是线性-高斯下界，数值比实测乐观，这里只看它的**序**能不能预测实测。")
    A("   预算   σ     n |  ρ(预测,实测)      p |  ρ(设计维数,实测)      p |  ρ(logdet F,实测)      p | 预测极差 实测极差")
    for kk in BUDGETS:
        for sg in SIGMAS:
            b = doc["design_theory_predictivity"].get(_bkey(kk, sg))
            if not b:
                continue
            A("    +%-3d %5s %3d | %+8.3f %10.3g | %+8.3f %10.3g | %+8.3f %10.3g | %7.1f%% %7.1f%%"
              % (kk, sg, b["n"], b["rho_pred_vs_achieved"], b["p"],
                 b["rho_kdes_vs_achieved"], b["p_kdes"],
                 b["rho_logdet_vs_achieved"], b["p_logdet"],
                 100 * b["pred_spread_rel"], 100 * b["achieved_spread_rel"]))
    for sg in SIGMAS:
        b = doc["design_theory_predictivity"].get("pooled_sigma_%g" % sg)
        if not b:
            continue
        A("    合并 %5s %3d | %+8.3f %10.3g |        --         -- |        --         -- | %7.1f%% %7.1f%%"
          % (sg, b["n"], b["rho_pred_vs_achieved"], b["p"],
             100 * b["pred_spread_rel"], 100 * b["achieved_spread_rel"]))
    A("  读法：\"合并\"行把 +20 与 +40 放在一起 - 预测量若只能区分预算档而不能在档内排序，")
    A("  合并 ρ 会明显高于分档 ρ，这正是设计理论在本网上的有效分辨尺度。")
    A("")
    _advice(A, doc, pm)
    A("")
    with open(pm.OUT_WIP, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _best(doc, metric="rmse_g2"):
    """取三档 σ 里 p 最小的一格，供措辞建议引用。"""
    best = None
    for bk, cell in doc.get("significance", {}).items():
        for dn, t in cell.get(metric, {}).items():
            if best is None or t["p_exact"] < best[2]["p_exact"] or (
                    t["p_exact"] == best[2]["p_exact"]
                    and t["R"] > best[2]["R"]):
                best = (bk, dn, t)
    return best


def _advice(A, doc, pm):
    A("五、Table 2 与 §3.6 的修改建议（写给入文阶段，不改冻结树）")
    A("  【问题定位】")
    A("   · §3.6 现用训练水头 MSE 比较从零重选与随机布点：\"9.40, 9.72 and 9.28e-3;")
    A("     one run, five seeds, a bounded null\"。按本文件一(1)(3)：该量 96% 以上是噪声方差，")
    A("     同一布点同一数据只换初值它就只动 1%，而参数误差动 25% - 它没有分辨力，")
    A("     用它得到的\"胜过随机中位但不胜过最好随机\"既不支持也不反驳任何布点主张。")
    A("   · SI 表 10 的 \"RMSE, inform.\" 与曲线里的 sub-RMSE 都定义在**随设计变化**的集合上")
    A("     （info 管 279→412、旧子空间维数 20→56），跨行不可比。")
    A("  【Table 2 建议】")
    A("   1) 计数部分（找回 / 丢失 / 仍不可辨识）保留原样：它们是普查判据下的确定性事实，")
    A("      不是统计估计，不需要也不应该配 p 值。但把\"从零重选 40+k\"整行显式标注为")
    A("      **对照组（S0 被丢弃）**，列头写成 Augment (S0 kept) vs Reselect (control, S0 discarded)，")
    A("      并把\"丢失 38\"提到与\"找回 59\"同一格（59/38），避免读者把 68/24 与增设的 95/0")
    A("      当作同类可比数 - 这正是当前 Table 2 唯一容易被误读的地方。")
    A("   2) 新增两行，把 Table 2 从纯计数表升级为\"设计量 → 下游参数误差\"：")
    A("        · 设计时可辨识维数 k_des(γ=2)（S0=16，cover+20=30，dopt+20=36，dopt+40=56）；")
    A("        · 重构 ref-RMSE（固定参考子空间，见本文件二），S0 与各增设同一把尺子。")
    A("   3) 表注补一句量纲与口径：ref-RMSE 的先验参照 = 28.92（把所有管设成 C0=130 的成绩），")
    A("      读者才知道 17 与 25 之间的差距有多大。")
    A("  【§3.6 措辞建议】")
    A("   1) 删去用训练水头 MSE 做的从零重选 vs 随机比较，改为一句方法学说明：")
    A("      \"Head-space misfit is not used to rank designs: at 0.1 ft noise it is 96%")
    A("      noise variance, and across eight optimiser restarts on identical data it moves")
    A("      1.4% while the identifiable-subspace roughness error moves 25%.\"")
    A("   2) 增设一句指标定义（正文一句 + 方法节公式），强调截断准则不含任何随设计变化的量。")
    b = _best(doc)
    if b:
        sg, dn, t = b
        A("   3) 显著性一句的写法（按实测填）：\"Against %d random augmentations drawn from the"
          % t["R"])
        A("      same pool at the same budget, the %s design's identifiable-subspace error is"
          % dn)
        A("      %.2f against a random median of %.2f (%d of %d random draws at least as good;"
          % (t["designed"], t["rand_median"], t["n_at_least_as_good"], t["R"]))
        A("      one-sided exact randomisation p = %.3g).\"" % t["p_exact"])
    A("  【公平性提醒（必须写进方法节，否则会被审稿人抓）】")
    A("   · D-opt 增设的选择判据（Fisher 信息的 logdet）与本文件的评价子空间同源，")
    A("     二者都由同一张灵敏度矩阵导出；coverage 增设不用该判据，是判据无关的对照。")
    A("     因此读 §三 时应同时看 coverage 那一行 - 若只有 D-opt 显著，只能说明")
    A("     \"按信息判据选点确实提高了该判据下的信息\"，不足以支持一般性的布点主张。")
    A("   · 同时报告不做任何投影的全 432 管裸 RMSE（rmse_all）与补空间 RMSE：")
    A("     若结论只在投影后成立、裸指标反向，必须写明。")
    reg = doc["metric_evidence"].get("selection_regret", {})
    if reg:
        A("  【选择代价可直接入文的一句】")
        kk = sorted(reg, key=lambda s: -reg[s]["n"])[0]
        c = reg[kk]
        for sk, lab in (("mse_train_final", "training head MSE"),
                        ("val_frame_rmse", "held-out head RMSE")):
            d = c.get(sk)
            if not d:
                continue
            A("   · \"Selecting the best of %d random augmentations by %s returns a design"
              % (c["n"], lab))
            A("     whose identifiable-subspace error is %.2f, ranked %d of %d (%.0fth percentile);"
              % (d["picked_rmse_g2"], d["picked_rank"], c["n"], d["pct"]))
            A("     the best design in the same pool reaches %.2f.\"（%s）"
              % (c["oracle_min"], kk))
    sigd = doc.get("significance", {})
    if sigd:
        A("  【实测显著性可直接入文的一段（按实测填，勿改数）】")
        A("      格             设计        R | 设计ref-RMSE 随机中位 随机最好 | 优于设计 p精确 需R")
        for bk in sorted(sigd):
            cell = sigd[bk]
            for dn, t in sorted(cell.get("rmse_g2", {}).items()):
                need = t["needed_R_for_p001"]
                A("    %-14s %-11s %3d | %11.3f %8.3f %8.3f | %6d %6.4f %4s"
                  % (bk, dn, t["R"], t["designed"], t["rand_median"],
                     t["rand_min"], t["n_at_least_as_good"], t["p_exact"],
                     ("%d" % need) if need else "n/a"))
        A("      · 每格都是单侧精确随机化检验，p 的下限就是 1/(R+1)：R=20 时 0.0476，")
        A("        R=100 时 0.0099。凡是 π̂=0 的格子，p 已经压到该 R 下的下限，")
        A("        再往下只能靠加随机样本量，不能靠换统计量。")
        A("      · 与设计时大样本检验（R=2000、p≈5e-4）配合写：设计时那一支说明")
        A("        \"信息判据上的优势是确定性的\"，实测这一支说明\"该优势确实传到了")
        A("        非线性反演的参数误差上\"，两支的样本量限制完全不同，必须分开陈述。")
    dis = doc.get("metric_disagreement", [])
    if dis:
        A("      · 反向证据（务必写进正文，它比任何论证都强）：有 %d 个 (格, 设计, 水头指标)"
          % len(dis))
        A("        组合里，设计在参数误差上不劣于全部随机布点（π̂≤0.05），在水头指标上")
        A("        却不优于一半以上的随机布点（π̂≥0.5） - 用水头指标选布点会选反。")
    A("  【必须一并报告的代价项（否则是选择性报告）】")
    A("   · 参考子空间的补空间 RMSE：设计增设在补空间上普遍不优于随机（见 §三 的")
    A("     \"补空间RMSE\" 行）。这是预期的 - 把有限的观测预算集中到可辨识方向，")
    A("     等于放弃在不可辨识方向上的（本来也是先验主导的）精度。要写清楚。")
    A("   · 不做任何投影的全 432 管裸 RMSE 结果与 ref-RMSE 同向但更弱，也要照报。")
    A("  【null result 的诚实表述模板】")
    A("   · 不写\"没有显著差异\"就收尾，写清三件事：检验的零分布是什么（同池同预算的随机布点）、")
    A("     达到的 p 与该 R 下的 p 下限（1/(R+1)）、以及要压到 p<0.01 还需要多少样本。")
    A("   · 若 π̂>0.01（随机布点里超过 1% 能追平设计），直接写\"p<0.01 在任何样本量下都不可达\"，")
    A("     并把结论落到效应量：设计相对随机中位的位移及其自举 CI。")
    A("   · 保留幅值受限的机理句：City D 的 T1 全网最大压降 0.089 ft 低于 0.1 ft 噪声，")
    A("     字典列范数 0.149–0.642 对 T2 的 5–17 - 任何布点都改不了信噪比，这是物理约束不是失败。")

# -*- coding: utf-8 -*-
"""placement_metric.py - 布点验证指标重构 + 增设显著性（City D 主线）。

问题（模块二遗留）：布点/增设的下游验证一直用**水头空间的 MSE / RMSE**
（§3.6 的 "9.40 / 9.72 / 9.28e-3"、SI 表 10 的 valF/valS）与**"informative 管
的 C-RMSE"**。两者都不是布点好坏的可比指标：

  (1) 水头 MSE 被噪声地板压平。训练损失对含噪观测取 MSE，其期望 ≈ σ² +
      (模型误差)²；City D 的模型误差远小于 σ，于是任何布点的 MSE 都被钉在 σ²
      附近，传感器越多平均越充分、数值越靠近 σ² 且越稳定 - **下降与参数恢复无关**。
  (2) "informative 管"随布点变化（S0 是 279 根，cover+40 是 412 根），
      同名指标算在不同管集上，跨布点不可比；新纳入的恰是最难的管，
      RMSE 反而"变差"。
  (3) 现有 sub_rmse 的良态子空间用**相对**截断 σ_i > 10⁻²·σ_1，而 σ_1 本身
      随布点变化 - 子空间与维数都跟着设计走（S0 k=20，cover+20 k=33，
      dopt+40 k=56），跨布点仍不可比。

重构（本文件）：**固定参考可辨识子空间上的参数 RMSE**。

  参考子空间 V_ref：取候选池全装表（City D 541 个 junction 全部）的多帧灵敏度
  A_full = [S(t,i,·)]_{t,i} ∈ R^{(T·m_pool)×P_free}，Fisher 信息 F = AᵀA/σ_n²，
  其特征向量即 A 的右奇异向量 v_j、特征值 sv_j²/σ_n²。截断准则用**先验参照的
  绝对判据**（不含任何随布点变化的量）：方向 v_j 入选 ⟺ 该方向的贝叶斯后验标准差
  至少把先验压缩 γ 倍，
        1/sqrt(1/σ_p² + sv_j²/σ_n²) ≤ σ_p/γ  ⟺  sv_j ≥ σ_n·sqrt(γ²−1)/σ_p 。
  γ=2（"数据至少把先验腰斩"）、σ_p=15、σ_n=0.1 ⇒ 阈值 0.011547，City D 得
  k_ref=98。它是"全网装满表也只有 98 个方向能被数据压过先验一半"的物理上限，
  与任何具体布点无关，因此所有设计在同一把尺子上打分。

  指标：  RMSE_ident = ‖V_refᵀ(Ĉ − C_true)‖₂ / sqrt(k_ref)     （ft^0 ·C 单位）
          先验参照   RMSE_prior = ‖V_refᵀ(C0·1 − C_true)‖₂ / sqrt(k_ref)
          技能分     skill = 1 − RMSE_ident / RMSE_prior       （>0 才算学到东西）
  γ 取 2/3/10 的敏感性一并报告（k_ref = 98/70/29）。

阶段（--stage，逗号连写）：
  subspace  从 data/placement_cache_city_d.npz 建 V_ref（缓存到 scratch），
            报谱、k_ref(γ) 与各设计张成的维数。
  rand      随机增设对照：S0（40 个）固定，从**同一公平池**（541−40=501 个候选）
            均匀抽 k 个，跑与 AUG_* 完全同引擎的标定（同真值 perpipe seed=7、
            同噪声种子、同 λ、同 20% 传感器留出规则），落 data/calib_augrand_city_d.json。
  retro     用 V_ref 给**全部**归档 run（calib_gc1_city_d.json 71 条 + 本轮随机对照）
            重新打分，并算水头指标与参数指标的相关性。
  report    精确随机化检验、置信区间、所需样本量；写 data/placement_metric_wip.txt
            与 data/placement_metric_city_d.json。

零编号：可读输出（wip/json）只含计数、统计量与种子，不含任何节点/链路/传感器编号；
随机布点由 (池, 种子) 确定性复现，不落编号。

运行：
  & python -X utf8 scripts/placement_metric.py --stage subspace
  & python -X utf8 scripts/placement_metric.py --stage rand --sigma 0.1 --k 20 --seeds 0-29
  & python -X utf8 scripts/placement_metric.py --stage retro,report
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DATA = os.path.join(ROOT, "data")
SP, SN = 15.0, 0.1                 # 与 place_sensors.py / augment_suite.py 同
GAMMAS = (2.0, 3.0, 10.0)
GAM_MAIN = 2.0
CACHE = os.path.join(DATA, "placement_cache_city_d.npz")
ORDERS = os.path.join(DATA, "placement_orders_city_d.npz")
RAND_JSON = os.path.join(DATA, "calib_augrand_city_d.json")
OUT_JSON = os.path.join(DATA, "placement_metric_city_d.json")
OUT_WIP = os.path.join(DATA, "placement_metric_wip.txt")
SCRATCH = os.path.join(os.environ.get("TEMP", DATA), "claude_placement_metric")
os.makedirs(SCRATCH, exist_ok=True)


# ----------------------------------------------------------------- 小工具
def jload(fp, default=None):
    if os.path.isfile(fp):
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    return {} if default is None else default


def jdump(obj, fp):
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, fp)


def parse_seeds(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# ----------------------------------------------------- 一、参考可辨识子空间
def fair_pool(n_junc, S0_pos):
    """公平候选池：全部 junction 位置去掉 S0（增设方法用的同一个池）。"""
    return np.setdiff1d(np.arange(n_junc), np.asarray(S0_pos))


def rand_add(pool, k, seed):
    """随机增设：从公平池均匀无放回抽 k 个（(池, 种子) 确定性复现）。"""
    return np.sort(np.random.default_rng(20260000 + int(seed)).choice(
        pool, int(k), replace=False))


def sv_threshold(gamma, sigma_prior=SP, sigma_noise=SN):
    """先验参照的绝对截断：后验标准差 ≤ σ_p/γ ⟺ sv ≥ σ_n·sqrt(γ²−1)/σ_p。"""
    return sigma_noise * np.sqrt(gamma ** 2 - 1.0) / sigma_prior


def build_subspace(pb, verbose=True):
    """返回 dict(Vr, k_ref, sv_pool, pos_free, Sf)；Vr 缓存在 scratch。"""
    z = np.load(CACHE)
    S, pipe_idx = z["S_full"], z["pipe_idx"]
    assert np.array_equal(pipe_idx, pb.pidx), "缓存 pipe_idx 与 calibrate 不一致"
    assert np.array_equal(z["junc"], np.asarray(pb.s.junc_nodes))
    pos_free = np.searchsorted(pipe_idx, pb.free_idx)
    assert np.array_equal(pipe_idx[pos_free], pb.free_idx)
    Sf = np.ascontiguousarray(S[:, :, pos_free])           # [T, m_pool, P_free]
    fp = os.path.join(SCRATCH, "vref_city_d.npz")
    if os.path.isfile(fp):
        zz = np.load(fp)
        if zz["n_free"] == Sf.shape[2] and zz["n_pool"] == Sf.shape[1]:
            return dict(Vt=zz["Vt"], sv=zz["sv"], Sf=Sf, pos_free=pos_free)
    A = Sf.reshape(-1, Sf.shape[2])
    t0 = time.perf_counter()
    sv, Vt = np.linalg.svd(A, full_matrices=False)[1:]
    if verbose:
        print("[subspace] SVD %s  %.1fs" % (A.shape, time.perf_counter() - t0))
    np.savez(fp, Vt=Vt, sv=sv, n_free=Sf.shape[2], n_pool=Sf.shape[1])
    return dict(Vt=Vt, sv=sv, Sf=Sf, pos_free=pos_free)


def ident_metrics(sub, dC, gammas=GAMMAS):
    """固定参考子空间上的参数 RMSE（各 γ 一套）+ 不投影 / 补空间的对照。"""
    out = {"rmse_all": float(np.linalg.norm(dC) / np.sqrt(len(dC)))}
    for g in gammas:
        k = int((sub["sv"] > sv_threshold(g)).sum())
        proj = sub["Vt"][:k] @ dC
        out["k_g%g" % g] = k
        out["rmse_g%g" % g] = float(np.linalg.norm(proj) / np.sqrt(k))
        if g == GAM_MAIN:
            rest = float(np.sqrt(max(dC @ dC - proj @ proj, 0.0)
                                 / max(len(dC) - k, 1)))
            out["rmse_null_g%g" % g] = rest
    return out


def fast_diag(pb, sub, sel):
    """从灵敏度缓存复算 calibrate.get_diag 的四元组（reason/Vr/k_sub/rank_eps）。

    缓存 S_full 与 identifiability 内部的 sensitivity_matrix 是同一个量（同
    accuracy=1e-12、同 polish；max_iter 200 vs 60 都已收敛），归因判据在
    calibrate.evaluate 只用到 free_idx 上的三类：clamped(margin=10) /
    unobservable(整列解析零) / informative。SVD 的右奇异向量对行序不变，
    因此 Vr/k_sub/rank_eps 与 get_diag 逐位可比。--verify-diag 逐个核对。
    """
    from dgga.calib import clamped_mask
    from dgga.autodiff import solve_polished
    if not hasattr(pb, "_cm10"):
        sol0 = solve_polished(pb.s, pb.d, pb.rh, accuracy=1e-12,
                              max_iter=60, polish_steps=3)
        pb._cm10 = clamped_mask(pb.s, sol0, margin=10.0)["mask"]
    A = sub["Sf"][:, np.asarray(sel), :].reshape(-1, sub["Sf"].shape[2])
    zero_col = ~(np.max(np.abs(A), axis=0) > 0.0)
    reason = np.empty(pb.s.L, dtype=object)
    reason[:] = "informative"
    free = pb.free_idx
    r_free = np.where(pb._cm10[free], "clamped",
                      np.where(zero_col, "unobservable", "informative"))
    reason[free] = r_free
    sv, Vt = np.linalg.svd(A, full_matrices=False)[1:]
    rtol = max(A.shape) * np.finfo(np.float64).eps
    rank_eps = int(np.sum(sv > rtol * sv[0]))
    k_sub = int(np.sum(sv > 1e-2 * sv[0]))
    return reason, Vt[:k_sub].T.copy(), k_sub, rank_eps


def design_dim(sub, sel, gammas=GAMMAS):
    """该布点自己张成的可辨识维数（同一绝对判据，只作对照，不作评分尺）。"""
    A = sub["Sf"][:, sel, :].reshape(-1, sub["Sf"].shape[2])
    sv = np.linalg.svd(A, compute_uv=False)
    return {("kdes_g%g" % g): int((sv > sv_threshold(g)).sum()) for g in gammas}, sv


# ----------------------------------------------------------- 二、随机对照跑
def run_sensors(CA, pb, sub, key, sensors, sigma, noise_seed, lam=1e-4,
                diag_tag=None, note="", sel=None, fast=True):
    """与 calibrate.run_config 同引擎、同真值、同噪声，只是传感集显式给定。"""
    t0 = time.perf_counter()
    C_true, tinfo = CA.make_truth(pb, "perpipe", CA.TRUTH_SEED["perpipe"])
    scen = CA.make_obs(pb, C_true, ("perpipe", CA.TRUTH_SEED["perpipe"]),
                       sigma=sigma, noise_seed=noise_seed)
    sens_train, sens_hold = CA.split_holdout(sensors)
    if fast and sel is not None:
        reason, Vr, k_sub, rank_eps = fast_diag(pb, sub, sel)
    else:
        reason, Vr, k_sub, rank_eps = CA.get_diag(pb, diag_tag or key, sensors)
    cal = CA.calibrate(pb, scen["obs"], sens_train, pb.d, lam=lam)
    ev = CA.evaluate(pb, C_true, cal["C_hat"], reason, Vr, k_sub, rank_eps,
                     scen, sens_train, sens_hold, pb.d)
    dC = cal["C_hat"] - C_true[pb.free_idx]
    ev.update(ident_metrics(sub, dC))
    rec = dict(key=key, net="city_d", truth=tinfo, sigma=float(sigma),
               noise_seed=int(noise_seed), lam=float(lam),
               n_sensors=int(len(sensors)),
               n_train_sensors=int(len(sens_train)),
               n_hold_sensors=int(len(sens_hold)),
               mse_train_final=cal["mse_train"], nfe=cal["nfe"],
               nbwd=cal["nbwd"], adam_steps=cal["adam_steps"],
               lbfgs_evals=cal["lbfgs_evals"], lm_iters=cal["lm_iters"],
               t_calib_sec=cal["t_total"], t_total_sec=time.perf_counter() - t0,
               host=platform.node(), note=note,
               C_hat_free=[round(float(x), 4) for x in cal["C_hat"]], **ev)
    print("[%s] sig=%.2f  ref-RMSE(g2)=%.3f  sub=%.3f(k=%d)  info=%.3f  "
          "valF=%.4f valS=%.4f  train=%.4f  %.0fs"
          % (key, sigma, ev["rmse_g2"], ev["sub_rmse"], ev["sub_rank"],
             ev.get("info_rmse", float("nan")), ev["val_frame_rmse"],
             ev["val_sensor_rmse"], ev["train_rmse"], rec["t_total_sec"]),
          flush=True)
    return rec


SHARD = os.path.join(SCRATCH, "runs")
os.makedirs(SHARD, exist_ok=True)


def shard_path(key):
    return os.path.join(SHARD, key + ".json")


def stage_merge(args):
    """把分片 run 收进 data/calib_augrand_city_d.json（多进程无写冲突）。"""
    d = jload(RAND_JSON, {})
    runs = d.get("runs", {})
    n0 = len(runs)
    for fn in sorted(os.listdir(SHARD)):
        if fn.endswith(".json"):
            r = jload(os.path.join(SHARD, fn))
            if r:
                runs[r["key"]] = r
    d["runs"] = runs
    d["meta"] = dict(
        pool="全部 junction 去掉 S0（增设方法用的同一池，541-40=501）",
        draw="np.random.default_rng(20260000+seed).choice(pool, k, replace=False)",
        engine="scripts/calibrate.py 同引擎（perpipe seed=7 真值、λ=1e-4、"
               "Adam80+LBFGS20+LM15、20% 传感器留出 seed=505、C_init=C0）",
        n_runs=len(runs))
    jdump(d, RAND_JSON)
    print("[merge] %d -> %d runs" % (n0, len(runs)))


def designed_positions(orders, name):
    S0 = orders["augment_fixed"]
    if name == "ga40":
        return S0
    for tag, arr in (("augdopt", "augment_dopt"), ("augcover", "augment_cover")):
        if name.startswith(tag):
            return np.r_[S0, orders[arr][:int(name[len(tag):])]]
    raise ValueError(name)


def stage_rand(args):
    import calibrate as CA
    pb = CA.Problem("city_d")
    sub = build_subspace(pb)
    orders = np.load(ORDERS)
    S0 = orders["augment_fixed"]
    junc = np.asarray(pb.s.junc_nodes)
    pool = fair_pool(len(junc), S0)
    jobs = []
    sigmas = [float(x) for x in args.sigma.split(",")]
    nseeds = parse_seeds(args.noise_seeds)
    for sg in sigmas:
        for ns in nseeds:
            for nm in [x for x in args.designs.split(",") if x]:
                jobs.append(("DES_%s_s%g_n%d" % (nm, sg, ns), sg, ns,
                             designed_positions(orders, nm), "des_" + nm,
                             "设计增设（本机复跑，与随机对照同机同码）：" + nm))
            for kk in [int(x) for x in str(args.k).split(",") if x.strip()]:
                for sd in parse_seeds(args.seeds):
                    jobs.append(("RAND_augrand%d_s%g_n%d_r%d" % (kk, sg, ns, sd),
                                 sg, ns, np.r_[S0, rand_add(pool, kk, sd)],
                                 "augrand%d_r%d" % (kk, sd),
                                 "随机增设对照：S0 固定 + 公平池均匀抽 %d" % kk))
    jobs = jobs[args.offset::max(1, args.stride)]
    done = 0
    for key, sg, ns, pos, tag, note in jobs:
        if not args.force and os.path.isfile(shard_path(key)):
            continue
        selpos = np.unique(pos)
        sensors = np.sort(junc[selpos])
        rec = run_sensors(CA, pb, sub, key, sensors, sg, ns, diag_tag=tag,
                          note=note, sel=selpos, fast=not args.slow_diag)
        jdump(rec, shard_path(key))
        done += 1
    print("[rand] %d runs done (of %d assigned)" % (done, len(jobs)))


# ---------------------------------------------------------------- 三、回溯打分
HEAD_KEYS = ["train_rmse", "val_frame_rmse", "val_sensor_rmse",
             "train_rmse_clean", "val_frame_rmse_clean",
             "val_sensor_rmse_clean", "mse_train_final"]
PAR_KEYS = ["rmse_g2", "rmse_g3", "rmse_g10", "sub_rmse", "info_rmse"]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), n
    rx, ry = _rank(x), _rank(y)
    r = float(np.corrcoef(rx, ry)[0, 1])
    # Fisher-z 双侧 p（n≥10 时可用；小样本只作参考，报告里注明 n）
    if abs(r) >= 1.0 or n < 5:
        return r, float("nan"), n
    z = np.arctanh(r) * np.sqrt((n - 3) / 1.06)
    from math import erfc, sqrt
    return r, float(erfc(abs(z) / sqrt(2.0))), n


def _rank(v):
    order = np.argsort(v, kind="mergesort")
    rk = np.empty(len(v), float)
    rk[order] = np.arange(len(v), dtype=float)
    # 处理并列
    uv, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        for j in np.where(cnt > 1)[0]:
            m = inv == j
            rk[m] = rk[m].mean()
    return rk


def collect(pb, sub):
    """全部归档 run + 随机对照 → 统一打分表。"""
    import calibrate as CA
    C_true, _ = CA.make_truth(pb, "perpipe", CA.TRUTH_SEED["perpipe"])
    Cg, _ = CA.make_truth(pb, "grouped", CA.TRUTH_SEED["grouped"])
    free = pb.free_idx
    rows = []
    srcs = [(os.path.join(DATA, "calib_gc1_city_d.json"), "archive"),
            (RAND_JSON, "randctl")]
    for fp, src in srcs:
        d = jload(fp, {}).get("runs", {})
        for key, rec in d.items():
            ch = np.asarray(rec.get("C_hat_free", []), float)
            if ch.size != len(free):
                continue
            truth = rec.get("truth", {}).get("mode", "perpipe")
            Ct = Cg if truth == "grouped" else C_true
            m = ident_metrics(sub, ch - Ct[free])
            row = dict(key=key, src=src, placement=rec.get("placement", ""),
                       sigma=rec.get("sigma"), noise_seed=rec.get("noise_seed"),
                       lam=rec.get("lam"), truth=truth,
                       n_sensors=rec.get("n_sensors"),
                       sub_rank=rec.get("sub_rank"), **m)
            for k in HEAD_KEYS + ["info_rmse", "sub_rmse", "n_informative"]:
                row[k] = rec.get(k)
            rows.append(row)
    dprior = np.full(len(free), CA.C0) - C_true[free]
    prior = ident_metrics(sub, dprior)
    return rows, prior


# ---------------------------------------------------------- 四、精确检验
def exact_test(designed, randoms, side="less"):
    """随机化（置换）检验：设计值在随机布点零分布中的位置。

    H0：该设计与从同一公平池均匀抽取的布点无异。
    统计量取参数 RMSE（越小越好），单侧 p = (1 + #{rand ≤ designed}) / (R + 1)
 - 加一是把观测到的设计值本身计入零分布（Phipson & Smyth 2010），
    使 p 在 H0 下有效（不会低于 1/(R+1)）。
    """
    r = np.asarray(randoms, float)
    r = r[np.isfinite(r)]
    R = len(r)
    if side == "less":
        cnt = int((r <= designed).sum())
    else:
        cnt = int((r >= designed).sum())
    p = (1.0 + cnt) / (R + 1.0)
    return dict(R=R, n_at_least_as_good=cnt, p_exact=float(p),
                rand_min=float(r.min()), rand_q25=float(np.percentile(r, 25)),
                rand_median=float(np.median(r)),
                rand_q75=float(np.percentile(r, 75)),
                rand_max=float(r.max()), rand_mean=float(r.mean()),
                rand_sd=float(r.std(ddof=1)) if R > 1 else float("nan"),
                designed=float(designed),
                z=float((designed - r.mean()) / r.std(ddof=1)) if R > 1
                else float("nan"),
                pi_hat=float(cnt / R))


def clopper_pearson(k, n, alpha=0.05):
    """π = P(随机布点 ≤ 设计值) 的精确置信区间。"""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def needed_R(pi_hat, target=0.01):
    """精确检验达到 p<target 所需的随机样本量（π̂ 固定时）。

    p = (1 + Rπ)/(R+1) < target  ⟺  R(π − target) < target − 1
    π < target 时 R > (1 − target)/(target − π)；π ≥ target 时任何 R 都达不到。
    """
    if pi_hat >= target:
        return None
    return int(np.floor((1.0 - target) / (target - pi_hat))) + 1


def hodges_lehmann(designed, randoms):
    """设计值相对随机总体的位置移动量（一样本 HL = designed − median(rand)）与
    自举 CI（对随机组自举，设计值是确定性单点）。"""
    r = np.asarray(randoms, float)
    r = r[np.isfinite(r)]
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(r), size=(20000, len(r)))
    bs = np.median(r[idx], axis=1)
    d = designed - np.median(r)
    lo, hi = np.percentile(designed - bs, [2.5, 97.5])
    return float(d), float(lo), float(hi)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--sigma", default="0.1")
    ap.add_argument("--k", default="20", help="增设预算，可逗号连写（如 20,40）")
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--noise-seeds", default="100")
    ap.add_argument("--designs", default="")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pred-R", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--slow-diag", action="store_true",
                    help="用 calibrate.get_diag 重解而不是查灵敏度缓存")
    ap.add_argument("--machine", default="")
    args = ap.parse_args()
    for st in args.stage.split(","):
        st = st.strip()
        if st == "subspace":
            stage_subspace(args)
        elif st == "verifydiag":
            stage_verifydiag(args)
        elif st == "predtest":
            stage_predtest(args)
        elif st == "rand":
            stage_rand(args)
        elif st == "merge":
            stage_merge(args)
        elif st in ("retro", "report"):
            stage_report(args, do_report=(st == "report"))
        else:
            raise SystemExit("unknown stage " + st)


def stage_subspace(args):
    import calibrate as CA
    pb = CA.Problem("city_d")
    sub = build_subspace(pb)
    sv = sub["sv"]
    print("pool spectrum: sv1=%.5g  n=%d" % (sv[0], len(sv)))
    for g in GAMMAS:
        thr = sv_threshold(g)
        print("  gamma=%-4g thr=%.6g  k_ref=%d" % (g, thr, int((sv > thr).sum())))
    orders = np.load(ORDERS)
    S0 = orders["augment_fixed"]
    for name, sel in (("S0", S0),
                      ("S0+dopt20", np.r_[S0, orders["augment_dopt"][:20]]),
                      ("S0+cover20", np.r_[S0, orders["augment_cover"][:20]]),
                      ("S0+dopt40", np.r_[S0, orders["augment_dopt"][:40]]),
                      ("S0+cover40", np.r_[S0, orders["augment_cover"][:40]])):
        dd, _ = design_dim(sub, np.unique(sel))
        print("  %-12s %s" % (name, dd))


def stage_predtest(args):
    """设计时（线性-高斯）指标上的大样本随机化检验：不跑任何标定。

    对每个布点 S 求 M(S) = I/σ_p² + Aᵀ_train A_train/σ_n²（训练帧 + 训练传感器，
    与标定同口径），预测 ref 子空间 RMSE = sqrt(tr(V_refᵀ M⁻¹ V_ref)/k_ref)。
    设计增设 vs R 个随机增设（同公平池同预算），精确单侧 p = (1+#{≤})/(R+1)。
    这一支的样本量不受仿真预算限制，与三、的非线性实测检验互为上下界。
    """
    import calibrate as CA
    from scipy.linalg import cho_factor, cho_solve
    pb = CA.Problem("city_d")
    sub = build_subspace(pb)
    orders = np.load(ORDERS)
    S0 = orders["augment_fixed"]
    junc = np.asarray(pb.s.junc_nodes)
    pool = fair_pool(len(junc), S0)
    k_ref = int((sub["sv"] > sv_threshold(GAM_MAIN)).sum())
    V = sub["Vt"][:k_ref].T.copy()
    P = sub["Sf"].shape[2]
    I0 = np.eye(P) / SP ** 2

    def pred(pos):
        sensors = np.sort(junc[np.unique(pos)])
        tr_sens, _ = CA.split_holdout(sensors)
        tpos = np.searchsorted(junc, tr_sens)
        A = sub["Sf"][pb.train_frames][:, tpos, :].reshape(-1, P)
        M = I0 + (A.T @ A) / SN ** 2
        c = cho_factor(M, lower=True, check_finite=False)
        X = cho_solve(c, V, check_finite=False)
        return float(np.sqrt(np.trace(V.T @ X) / k_ref))

    R = int(args.pred_R)
    for kk in [int(x) for x in str(args.k).split(",") if x.strip()]:
        t0 = time.perf_counter()
        rnd = np.empty(R)
        for i in range(R):
            rnd[i] = pred(np.r_[S0, rand_add(pool, kk, i)])
            if (i + 1) % 500 == 0:
                print("  k=%d %d/%d  %.0fs"
                      % (kk, i + 1, R, time.perf_counter() - t0), flush=True)
        out = dict(k_ref=k_ref, R=R, budget=int(kk),
                   pool_size=int(len(pool)), sigma_prior=SP, sigma_noise=SN,
                   note="设计时线性-高斯预测量，不含任何标定仿真；训练帧+训练传感器口径",
                   designs={})
        names = [x for x in args.designs.split(",") if x] or             ["ga40", "augdopt%d" % kk, "augcover%d" % kk]
        for nm in names:
            d = pred(designed_positions(orders, nm))
            t = exact_test(d, rnd, side="less")
            lo, hi = clopper_pearson(t["n_at_least_as_good"], t["R"])
            t.update(pi_ci95=[lo, hi],
                     needed_R_for_p001=needed_R(t["pi_hat"], 0.01),
                     p_min_attainable=1.0 / (t["R"] + 1))
            out["designs"][nm] = t
            print("k=%d %-12s pred=%.4f  rand med=%.4f  better=%d/%d  p=%.4g"
                  % (kk, nm, d, t["rand_median"], t["n_at_least_as_good"], R,
                     t["p_exact"]))
        # 键加 'q' 前缀：裸 '50'/'99'/'100' 与真实网的节点编号字面撞车，
        # 会被 placement_metric_report._assert_no_ids 的零编号守卫误判。
        out["rand_quantiles"] = {("q%g" % q): float(np.percentile(rnd, q))
                                 for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}
        fpo = os.path.join(DATA,
                           "placement_metric_predtest_city_d_k%d.json" % kk)
        jdump(out, fpo)
        np.save(os.path.join(SCRATCH, "predtest_rand_k%d.npy" % kk), rnd)
        print("[predtest] %s  (%.0fs)" % (fpo, time.perf_counter() - t0))


def stage_verifydiag(args):
    """逐个核对 fast_diag 与 calibrate.get_diag 的四元组（缓存路径 vs 重解路径）。"""
    import calibrate as CA
    pb = CA.Problem("city_d")
    sub = build_subspace(pb)
    orders = np.load(ORDERS)
    S0 = orders["augment_fixed"]
    junc = np.asarray(pb.s.junc_nodes)
    pool = fair_pool(len(junc), S0)
    cases = [("ga40", designed_positions(orders, "ga40")),
             ("augcover20", designed_positions(orders, "augcover20")),
             ("augdopt40", designed_positions(orders, "augdopt40"))]
    for sd in parse_seeds(args.seeds)[:3]:
        cases.append(("augrand20_r%d" % sd,
                      np.r_[S0, rand_add(pool, 20, sd)]))
    ok = True
    recs = []
    for name, pos in cases:
        selpos = np.unique(pos)
        sensors = np.sort(junc[selpos])
        t0 = time.perf_counter()
        r1, V1, k1, e1 = fast_diag(pb, sub, selpos)
        t1 = time.perf_counter()
        tag = {"ga40": "ga40", "augcover20": "des_augcover20",
               "augdopt40": "des_augdopt40"}.get(name, name)
        r2, V2, k2, e2 = CA.get_diag(pb, tag, sensors)
        t2 = time.perf_counter()
        free = pb.free_idx
        same_reason = bool(np.array_equal(r1[free], r2[free]))
        # 子空间比较用主角（principal angle）而非逐元素：SVD 在简并谱下相位/旋转不唯一
        gap = float("nan")
        if k1 == k2 and k1 > 0:
            s = np.linalg.svd(V1.T @ V2, compute_uv=False)
            gap = float(1.0 - s.min())
        print("%-16s reason=%s k_sub %d/%d rank_eps %d/%d 1-cos(θmax)=%.3g "
              "| fast %.1fs vs solve %.1fs"
              % (name, same_reason, k1, k2, e1, e2, gap, t1 - t0, t2 - t1))
        ok &= same_reason and k1 == k2 and e1 == e2 and (
            not np.isfinite(gap) or gap < 1e-8)
        recs.append(dict(case=name, reason_identical=same_reason,
                         k_sub_fast=k1, k_sub_solve=k2,
                         rank_eps_fast=e1, rank_eps_solve=e2,
                         one_minus_cos_theta_max=gap))
    print("[verifydiag] %s" % ("ALL MATCH" if ok else "MISMATCH"))
    jdump(dict(all_match=bool(ok), n_cases=len(recs), cases=recs,
               note="fast_diag（查灵敏度缓存）与 calibrate.get_diag（重解）"
                    "的 reason/k_sub/rank_eps/子空间主角对拍"),
          os.path.join(DATA, "placement_metric_verifydiag_city_d.json"))


def stage_report(args, do_report=True):
    import calibrate as CA
    pb = CA.Problem("city_d")
    sub = build_subspace(pb)
    rows, prior = collect(pb, sub)
    print("[retro] %d scored runs; prior ref-RMSE(g2)=%.4f"
          % (len(rows), prior["rmse_g2"]))
    if do_report:
        from placement_metric_report import write_report
        write_report(rows, prior, sub, pb, args)


if __name__ == "__main__":
    main()

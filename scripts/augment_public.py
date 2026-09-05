# -*- coding: utf-8 -*-
"""augment_public.py - 公开网增设全套（可复现版）：L-TOWN + Hanoi。

与 scripts/place_sensors.py --stage augment 同一套函数（dgga.placement.
bayes_dopt_augment / recovery_report / posterior_std）、同一 σ（σ_prior=15,
σ_noise=0.1 ft）、同一普查判据（管可辨识 ⇔ max|∂p/∂C|>0 且非结构性）。
全程"虚拟增设"：在模型上模拟新传感器，是设计与模拟验证，不是现场实装。

阶段（--stage）：
  ltown-sfull    L-TOWN（networks/EXAMPLE/L-Town/L-TOWN.inp，BattLeDIM 2020）
                 25 帧（t=0..24h 整点需水；水池水头取 mv2/coherence 同款中位；
                 每帧先跑完整状态机拿 (S*,K*)，再在冻结状态下做精确伴随灵敏度）
                 全候选 782 junction × P=905 根管 → data/placement_cache_ltown.npz
  ltown-augment  S0 = 论文 L-TOWN 漏损反演所用的 33 个传感器（mv2_train_gpu.py，
                 seed 909），候选池 = junction − S0 − 3 个注入漏点（与记录用例
                 "传感器不落在漏点上"的抽样约定一致）；D-opt / cover 两目标，
                 k∈{5,10,20,40,80}；对照"从零重选 |S0|+k"。
                 → data/placement_augment_ltown.json（零节点编号）
                   data/placement_orders_ltown.npz（候选下标）
  ltown-leak     论文 §3.5 的 L-TOWN 60 候选/3 漏点/Adam 60 步反演**原样重跑**
                 （seed 909 / 2026，B=256 场景，lr=0.1，λ=1e-4，C_true=1.5，
                 accuracy=1e-6，dense 状态机 + GPU 约化伴随），传感器换成
                 S0 ∪ S_k；两组观测：无噪声（论文口径）与 σ=0.1 ft 高斯噪声
                 （seed 909，按节点生成 → 各传感配置看到同一噪声实现）。
                 每套传感配置另算签名字典互相干（同 ltown_coherence.py 构造）。
                 → data/ltown_augment_leak.json
  hanoi-augment  Hanoi，帧 = calibrate.py 的 25 个合成工况（SYNTH_SEED=4242，
                 与 σ 梯标定同帧），S0 = random:10:2026（同前一轮）
                 → data/placement_augment_pub_hanoi_synth25.json / _orders_*.npz
  hanoi-calib    调 calibrate.py 引擎（--stage l1hanoi_aug）：σ∈{0.03,0.1,0.3}
                 × 3 噪声种子 × {S0, S0+k}。
  report         汇总各 JSON → data/augment_public_wip.txt（表格部分）。

运行：python -X utf8 scripts/augment_public.py --stage <stage>
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.parse import parse_inp                                  # noqa: E402
from dgga.solver import GGASolver                                 # noqa: E402
from dgga.autodiff import implicit_solve, solve_polished          # noqa: E402
from dgga.calib import clamped_mask, dead_branch_mask             # noqa: E402
from dgga.placement import (bayes_dopt_augment, bayes_dopt_greedy,  # noqa: E402
                            eval_subset, pipe_param_idx,
                            posterior_std, recovery_report)
from dgga.sensitivity import sensitivity_matrix                   # noqa: E402
from dgga.units import FLOW_UCF, MperFT                           # noqa: E402
from place_sensors import SIG_P, SIG_N, _k_metrics                # noqa: E402

torch.set_default_dtype(torch.float64)
DATA = os.path.join(ROOT, "data")
KS = [5, 10, 20, 40, 80]

# ---- L-TOWN：与 scripts/mainline_v2/mv2_train_gpu.py 逐项相同的配方 ----
LT_INP_CLEAN = os.path.join(ROOT, "networks", "public", "_cleaned", "L-TOWN.inp")
# 本机 = EXAMPLE 原件（sha256 0bfeaa80…）；集群副本只有 _cleaned（8e3009b8…，与原件
# 的全部内容差 = [OPTIONS] 少一行 "Pattern 1" + CRLF，parse 层同网）。取第一个存在者。
LT_INP = next((p for p in (
    os.path.join(ROOT, "networks", "EXAMPLE", "L-Town", "L-TOWN.inp"),
    os.path.join(ROOT, "networks_prv", "L-TOWN.inp"), LT_INP_CLEAN)
    if os.path.isfile(p)), LT_INP_CLEAN)
LT_SEED_SETUP = 909          # 候选 / 真值 / 传感器
LT_SEED_BATCH = 2026         # ±15% 需水、±1 ft 水库的 B 个场景
LT_NC, LT_NS = 60, 33
LT_C_TRUE, LT_C_PROBE = 1.5, 0.3
LT_TRUTH_EXPECTED = [651, 137, 720]     # 记录作业 data/gpu/5090_mv2_1465211.out
LT_HACC, LT_MI = 1e-10, 200             # 字典/状态判定的收紧停机（同 ltown_coherence.py）
LT_B, LT_STEPS, LT_LR, LT_LAM, LT_ACC = 256, 60, 0.1, 1e-4, 1e-6
LT_INIT_C = 0.02
NOISE_FT, SEED_NOISE = 0.1, 909         # 噪声组（论文 L-TOWN 用例无噪声；City D 用例同 σ 同种子）
LT_RECORDED = dict(loss_step1=2.438914e-01, loss_step60=2.549019e-03,
                   top5=[720, 752, 330, 308, 296], top1_true=True,
                   top3_exact=False, coh_max=0.9999999981271712,
                   coh_median=0.8258403929811211, n_pairs_gt_0999=42)

HANOI_S0_SPEC = "random:10:2026"


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def softplus_inv(x):
    return float(np.log(np.expm1(x)))


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)


def jload(path, default=None):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


# ======================================================================
# L-TOWN 公共件
# ======================================================================
def lt_load(mode="epanet", device="cpu"):
    net = parse_inp(LT_INP)
    if mode == "epanet":
        s = GGASolver(net, mode="epanet", inp_path=LT_INP)
    else:
        s = GGASolver(net, device=device, dtype=torch.float64, mode="dense",
                      inp_path=LT_INP, dense_status_machine=True)
    return net, s


def lt_nominal(net):
    """名义帧：t=0 需水；水池水头 = clip(中位, 30%~70%)（mv2 / ltown_coherence 同款）。"""
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh0[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d0, rh0


def lt_setup(se):
    """mv2_train_gpu.py:133-140 的抽样原样重放；真值必须等于记录作业打印值。"""
    jn = np.asarray(se.junc_nodes)
    rng = np.random.default_rng(LT_SEED_SETUP)
    cand = np.sort(rng.choice(jn, size=LT_NC, replace=False))
    truth_pos = rng.choice(LT_NC, size=3, replace=False)
    truth = cand[truth_pos]
    sens = np.sort(rng.choice(np.setdiff1d(jn, truth), size=LT_NS, replace=False))
    if truth.tolist() != LT_TRUTH_EXPECTED:
        raise RuntimeError(f"真值重放不一致 {truth.tolist()} != {LT_TRUTH_EXPECTED}")
    return jn, cand, truth, sens


def lt_units(net):
    """input1.c:567-573 同链（mv2_train_gpu.py:120-126）。"""
    qexp = float(net.meta["qexp"])
    spgrav = float(net.meta["spgrav"])
    press = str(net.meta.get("press_units", "") or "METERS")
    KPAperPSI, PSIperFT = 6.895, 0.4333
    pcf = KPAperPSI * PSIperFT * spgrav if press == "KPA" else MperFT * spgrav
    qcf = FLOW_UCF[str(net.meta["flow_units"])]
    return qexp, qcf ** qexp / pcf


def lt_paths():
    return dict(cache=os.path.join(DATA, "placement_cache_ltown.npz"),
                orders=os.path.join(DATA, "placement_orders_ltown.npz"),
                readable=os.path.join(DATA, "placement_augment_ltown.json"),
                leak=os.path.join(DATA, "ltown_augment_leak.json"))


# ======================================================================
# stage ltown-sfull
# ======================================================================
def stage_lt_sfull(n_frames=25):
    print("===== ltown-sfull：L-TOWN 全候选多帧灵敏度（逐帧冻结状态）=====")
    print(f"INP {os.path.relpath(LT_INP, ROOT)} sha256={sha256(LT_INP)}")
    if os.path.isfile(LT_INP_CLEAN):
        print(f"    (_cleaned 主线副本 sha256={sha256(LT_INP_CLEAN)}；parse 层两者同网)")
    net, se = lt_load()
    d0, rh0 = lt_nominal(net)
    jn = np.asarray(se.junc_nodes)
    pidx = pipe_param_idx(se)
    P = pidx.size
    t_secs = [h * 3600 for h in range(n_frames)]
    T = len(t_secs)
    D = np.stack([net.demand_cfs_at(t) for t in t_secs])
    print(f"网络 N={se.N} Nj={jn.size} L={se.L} P={P}（H-W 管）；帧 T={T}（整点）")

    S_full = np.zeros((T, jn.size, P))
    status = np.zeros((T, se.L), dtype=np.int8)
    setting = np.zeros((T, se.L))
    cm_per = np.zeros((T, P), dtype=bool)
    it_sm, resid, nact, t_fr, relerr_sm = [], [], [], [], []
    prv = np.asarray(net.link_type) == 3
    t_all = time.perf_counter()
    for i, t in enumerate(t_secs):
        t0 = time.perf_counter()
        # 状态判定：hacc=1e-10 在 L-TOWN 上停在 ~5e-7 的 GGA 平台（ltown_coherence.py
        # 记录的 base_relerr=5.4e-7 同现象）；状态早已定型，精确解由下面冻结状态的
        # Newton 精抛光给出（‖F‖∞ 逐帧打印），故只记录 relerr 不拒绝。
        base = se.run_gga(D[i], rh0, do_status=True, hacc=LT_HACC, max_iter=LT_MI)
        relerr_sm.append(float(base["relerr"]))
        S_t, K_t = base["status"].copy(), base["setting"].copy()
        S, info = sensitivity_matrix(se, D[i], rh0, None, jn, wrt="C",
                                     speed=K_t, status=S_t, accuracy=1e-12,
                                     max_iter=200, polish_steps=3,
                                     return_info=True)
        S_full[i] = S[:, pidx]
        sol = solve_polished(se, D[i], rh0, accuracy=1e-12, max_iter=200,
                             polish_steps=3, speed=K_t, status=S_t)
        cm_per[i] = clamped_mask(se, sol, margin=10.0)["mask"][pidx]
        status[i], setting[i] = S_t, K_t
        it_sm.append(int(base["iters"]))
        resid.append(float(np.max(info["resid_inf"])))
        nact.append(int((S_t[prv] == 4).sum()))
        t_fr.append(time.perf_counter() - t0)
        print(f"  帧 {i:2d} t={t // 3600:2d}h：状态机 iters={it_sm[-1]} "
              f"relerr={relerr_sm[-1]:.1e} ACTIVE PRV={nact[-1]}/3  "
              f"‖F‖∞={resid[-1]:.2e}  零列 "
              f"{int((np.max(np.abs(S_full[i]), axis=0) == 0).sum())}  "
              f"钳位(margin=10) {int(cm_per[i].sum())}  {t_fr[-1]:.1f}s")
    dead = dead_branch_mask(se, demand=D)[pidx]
    cm25 = cm_per.all(axis=0)
    cm_t0 = cm_per[0]
    colmax_all = np.max(np.abs(S_full), axis=(0, 1))
    z_all = colmax_all == 0.0
    floor = z_all & ~dead & ~cm25
    print(f"结构掩码：死支 {int(dead.sum())}；全帧钳位 {int((cm25 & ~dead).sum())}；"
          f"全候选 {jn.size}×{T} 帧零列 {int(z_all.sum())} → unobservable 地板 "
          f"{int(floor.sum())}；总耗时 {time.perf_counter() - t_all:.1f}s")
    np.savez_compressed(lt_paths()["cache"], S_full=S_full, pipe_idx=pidx,
                        junc=jn, t_secs=np.asarray(t_secs), dead=dead,
                        cm25=cm25, cm_t0=cm_t0, status=status,
                        setting=setting, iters_sm=np.asarray(it_sm),
                        relerr_sm=np.asarray(relerr_sm),
                        resid_inf=np.asarray(resid), n_active_prv=np.asarray(nact),
                        t_frame=np.asarray(t_fr))
    meta = dict(inp=os.path.relpath(LT_INP, ROOT), sha256=sha256(LT_INP),
                sha256_cleaned_copy=(sha256(LT_INP_CLEAN)
                                     if os.path.isfile(LT_INP_CLEAN) else None),
                N=int(se.N), Nj=int(jn.size), L=int(se.L), P=int(P),
                n_frames=T, frame_times_h=[t // 3600 for t in t_secs],
                tank_head_rule="clip(0.5*(hmin+hmax), 30%, 70%)（mv2/coherence 同款）",
                status_rule=f"每帧 run_gga(do_status=True, hacc={LT_HACC:g}, "
                            f"max_iter={LT_MI}) 取 (S*,K*)，灵敏度在冻结状态下算",
                iters_status_machine=it_sm, relerr_status_machine=relerr_sm,
                active_prv_per_frame=nact,
                resid_inf_max=float(max(resid)),
                n_dead=int(dead.sum()), n_clamped_allframes=int((cm25 & ~dead).sum()),
                n_zero_col_all_sensors=int(z_all.sum()),
                unobservable_floor=int(floor.sum()),
                t_total_sec=time.perf_counter() - t_all)
    rd = jload(lt_paths()["readable"], {})
    rd["sfull"] = meta
    jdump(rd, lt_paths()["readable"])
    print("ltown-sfull 完成。")


# ======================================================================
# stage ltown-augment
# ======================================================================
def _augment_core(S_full, fixed, struct, ks, tag, log=print):
    """place_sensors.stage_augment 的核心（同函数、同 σ、同判据），返回可落盘 dict。"""
    T, m, P = S_full.shape
    n0 = int(fixed.size)
    room = m - n0
    ks_req = sorted(set(int(k) for k in ks))
    ks = [k for k in ks_req if k <= room]
    kadd = max(ks)
    atol = 0.0
    unob0 = recovery_report(S_full, fixed, fixed, np.zeros(P, bool), struct,
                            atol)["unobservable_mask"]
    mask_t = unob0
    n_t = int(mask_t.sum())
    ps0 = posterior_std(S_full, fixed, SIG_P, SIG_N)
    base = _k_metrics(S_full, fixed, fixed, mask_t, struct, atol, ps0)
    log(f"[{tag}] T={T} 候选 {m} P={P} 结构性 {int(struct.sum())}；|S0|={n0}；"
        f"目标集（S0 下传感不足）{n_t}；S0 可辨识 {base['n_identifiable']} "
        f"unobservable {base['n_unobservable']} 秩 {base['rank']} f={base['f']:.2f} "
        f"CRLB_ident={base['crlb_ident']:.3e} 后验std中位(全非结构)="
        f"{base['post_std_all_nonstruct']['median']:.3g}")
    colmax_c = np.max(np.abs(S_full), axis=0)
    colmax0 = colmax_c[fixed].max(axis=0)
    runs = {}
    for obj in ("dopt", "cover"):
        t0 = time.perf_counter()
        a = bayes_dopt_augment(S_full, fixed, kadd, SIG_P, SIG_N, objective=obj,
                               struct_mask=struct, atol=atol, cert_ks=ks,
                               verbose=True)
        cm = colmax0.copy()
        rec = np.empty(kadd, dtype=np.int64)
        for j, i in enumerate(a["order"]):
            cm = np.maximum(cm, colmax_c[i])
            rec[j] = int(((cm > atol) & mask_t).sum())
        if np.any(np.diff(rec) < 0):
            raise RuntimeError("找回曲线非单调（bug）")
        k80 = (int(np.argmax(rec >= 0.8 * n_t)) + 1
               if n_t and np.any(rec >= 0.8 * n_t) else -1)
        kall = (int(np.argmax(rec >= n_t)) + 1
                if n_t and np.any(rec >= n_t) else -1)
        per_k = {}
        for k in ks:
            sel = np.r_[fixed, np.asarray(a["order"][:k], dtype=np.int64)]
            mk = _k_metrics(S_full, fixed, sel, mask_t, struct, atol,
                            a["cert"][k]["post_std"])
            c = a["cert"][k]
            mk.update(k=k, df=c["df"], cert_ratio=c["ratio"], upper_df=c["upper_df"],
                      f_drift=abs(c["f"] - c["f_recomputed"]),
                      ident_curve_k=a["ident_curve"][k])
            if "cover_ratio" in c:
                mk.update(cover_ratio=c["cover_ratio"], cover_upper=c["cover_upper"])
            if mk["n_lost"] != 0:
                raise RuntimeError("增设丢管 ≠ 0：违反单调性（bug）")
            if mk["n_identifiable"] != a["ident_curve"][k]:
                raise RuntimeError("recovery_report 与 augment 内部计数不一致")
            per_k[k] = mk
        runs[obj] = dict(added=a["order"], gains=a["gains"], f0=a["f0"],
                         f_curve=a["f_curve"], ident_curve=a["ident_curve"],
                         recovered_every_k=rec.tolist(), k_recover80=k80,
                         k_recover_all=kall, cover_gains=a["cover_gains"],
                         cover_saturated_at=a["cover_saturated_at"],
                         n_evals=a["n_evals"], t_total_sec=a["t_total"],
                         per_k=per_k)
        log(f"  目标 {obj:5s}：{time.perf_counter() - t0:.1f}s，评估 {a['n_evals']} "
            f"次；覆盖饱和步 {a['cover_saturated_at']}；找回 80% k={k80}，全部 k={kall}"
            f"（-1=网格内未达）；单调性断言逐步通过")
    g = bayes_dopt_greedy(S_full, n0 + kadd, SIG_P, SIG_N)
    resel = {}
    for k in ks:
        mk = _k_metrics(S_full, fixed, np.asarray(g["order"][:n0 + k]), mask_t,
                        struct, atol)
        mk["k"] = k
        resel[k] = mk
    same_k = _k_metrics(S_full, fixed, np.asarray(g["order"][:n0]), mask_t,
                        struct, atol)
    hdr = (f"{'k':>4} | {'目标':>6} | {'找回':>5} {'剩余':>5} {'丢失':>4} "
           f"{'unob':>5} {'可辨识':>6} {'秩':>4} | {'Δf':>9} {'证书':>6} "
           f"{'CRLB_ident':>11} {'bayes_tr':>10} | {'目标集后验std中位':>10} "
           f"{'≤σp/2':>6}")
    log(f"\n---- [{tag}] 增设结果（S0={n0} 固定 + k；目标集 {n_t} 根）----")
    log(hdr)
    for k in ks:
        for obj in ("dopt", "cover"):
            r = runs[obj]["per_k"][k]
            med = r["post_std_target"]["median"]
            log(f"{k:>4} | {obj:>6} | {r['n_recovered']:>5} {r['n_target_left']:>5} "
                f"{r['n_lost']:>4} {r['n_unobservable']:>5} {r['n_identifiable']:>6} "
                f"{r['rank']:>4} | {r['df']:>9.2f} {r['cert_ratio']:>6.3f} "
                f"{r['crlb_ident']:>11.3e} {r['bayes_trace']:>10.1f} | "
                f"{'n/a' if med is None else f'{med:.3g}':>10} "
                f"{r['post_std_target']['n_half_prior']:>6}")
        r = resel[k]
        log(f"{k:>4} | {'重选':>6} | {r['n_recovered']:>5} {r['n_target_left']:>5} "
            f"{r['n_lost']:>4} {r['n_unobservable']:>5} {r['n_identifiable']:>6} "
            f"{r['rank']:>4} | {r['f'] - base['f']:>9.2f} {'-':>6} "
            f"{r['crlb_ident']:>11.3e} {r['bayes_trace']:>10.1f} |")
    log(f"对照（从零重选 |S0|={n0} 个）：找回 {same_k['n_recovered']}，丢失 "
        f"{same_k['n_lost']}，unobservable {same_k['n_unobservable']}")
    cfg = dict(n_fixed=n0, ks=ks, ks_requested=ks_req, kadd=kadd, atol=atol,
               sigma_prior=SIG_P, sigma_noise=SIG_N, n_frames=T, n_candidates=m,
               P=P, n_structural=int(struct.sum()), n_target=n_t,
               criterion="管可辨识 ⇔ max_{t,i∈S}|∂p_i/∂C_k| > atol 且非结构性"
                         "（与 dgga.calib.identifiability 普查同判据）；σ 口径 = "
                         "贝叶斯后验 std sqrt(diag(M^-1))，M = I/σ_prior² + Σ S Sᵀ/σ_noise²",
               mode="virtual augmentation：在模型上模拟新传感器（设计与模拟验证，非现场实装）")
    readable = dict(
        config=cfg, s0=base,
        augment={obj: dict(per_k={str(k): runs[obj]["per_k"][k] for k in ks},
                           ident_curve=runs[obj]["ident_curve"],
                           recovered_every_k=runs[obj]["recovered_every_k"],
                           k_recover80=runs[obj]["k_recover80"],
                           k_recover_all=runs[obj]["k_recover_all"],
                           cover_gains=runs[obj]["cover_gains"],
                           cover_saturated_at=runs[obj]["cover_saturated_at"],
                           f0=runs[obj]["f0"], f_curve=runs[obj]["f_curve"],
                           n_evals=runs[obj]["n_evals"],
                           t_total_sec=runs[obj]["t_total_sec"])
                 for obj in ("dopt", "cover")},
        reselect=dict(per_k={str(k): resel[k] for k in ks}, same_k_as_s0=same_k),
        monotonicity="bayes_dopt_augment 逐步断言：可辨识集只增不减、diag(M^-1) "
                     "逐管不增；n_lost 恒 0（均通过，否则本文件不会生成）")
    orders = dict(augment_fixed=fixed,
                  augment_dopt=np.asarray(runs["dopt"]["added"], dtype=np.int64),
                  augment_cover=np.asarray(runs["cover"]["added"], dtype=np.int64),
                  reselect_dopt=np.asarray(g["order"], dtype=np.int64))
    return readable, orders


def stage_lt_augment(ks=KS):
    print("===== ltown-augment：S0 = 论文 L-TOWN 反演的 33 个传感器，增设 k =====")
    p = lt_paths()
    z = np.load(p["cache"])
    S_full, dead, cm25, junc = z["S_full"], z["dead"], z["cm25"], z["junc"]
    struct = dead | cm25
    net, se = lt_load()
    jn, cand, truth, sens = lt_setup(se)
    if not np.array_equal(jn, junc):
        raise RuntimeError("缓存 junc 与 solver.junc_nodes 不一致")
    keep = np.where(~np.isin(junc, truth))[0]          # 候选池剔除 3 个漏点
    S_sub = S_full[:, keep, :]
    pos = {int(junc[k]): i for i, k in enumerate(keep)}
    fixed = np.asarray(sorted(pos[int(n)] for n in sens), dtype=np.int64)
    print(f"候选池 = {junc.size} junction − 3 漏点 = {keep.size}；S0 = {fixed.size}")
    readable, orders = _augment_core(S_sub, fixed, struct, ks, "L-TOWN")
    readable["stem"] = "ltown"
    readable["config"]["s0"] = "mv2_train_gpu.py 33 sensors (seed 909)"
    readable["config"]["pool_note"] = ("候选池剔除 3 个注入漏点节点（记录用例的传感器"
                                       "抽样同样排除漏点；避免把传感器直接装在漏点上）")
    rd = jload(p["readable"], {})
    rd.update(readable)
    jdump(rd, p["readable"])
    # 下标口径：S_sub 的候选下标 → junc 位置下标（keep 映射），与 cache 的 junc 对齐
    np.savez_compressed(p["orders"],
                        augment_fixed=keep[orders["augment_fixed"]],
                        augment_dopt=keep[orders["augment_dopt"]],
                        augment_cover=keep[orders["augment_cover"]],
                        reselect_dopt=keep[orders["reselect_dopt"]],
                        pool=keep)
    print(f"ltown-augment 完成：{p['readable']}、{p['orders']}")


# ======================================================================
# stage ltown-leak
# ======================================================================
def lt_dictionary(net, se, d0, rh0, cand, qexp, ucf_e, log=print):
    """全节点签名字典 Dfull[N, NC]（ltown_coherence.py 同构造；行按传感器集取子集）。"""
    t0 = time.perf_counter()
    ke_base = np.asarray(net.node_ke, dtype=np.float64)
    base = se.run_gga(d0, rh0, do_status=True, hacc=LT_HACC, max_iter=LT_MI)
    h0 = base["head"]
    Dfull = np.zeros((net.N, cand.size))
    for j, c in enumerate(cand):
        ke = ke_base.copy()
        ke[c] = ucf_e / LT_C_PROBE ** qexp
        sj = se.run_gga(d0, rh0, ke=ke, do_status=True, hacc=LT_HACC, max_iter=LT_MI)
        Dfull[:, j] = (sj["head"] - h0) / LT_C_PROBE
    log(f"  签名字典 [N={net.N}, {cand.size}]：base iters={int(base['iters'])} "
        f"relerr={float(base['relerr']):.2e}；{time.perf_counter() - t0:.1f}s")
    return Dfull


def coherence_stats(Dfull, sens, cand, truth, top_nodes=()):
    D = Dfull[sens]
    Dn = D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-300)
    coh = np.abs(Dn.T @ Dn)
    nc = cand.size
    iu = np.triu_indices(nc, 1)
    off = coh[iu]
    orth = off < 1e-12
    within = off[~orth]
    node_of = {int(c): k for k, c in enumerate(cand)}

    def rival(n):
        k = node_of[int(n)]
        row = coh[k].copy()
        row[k] = -1.0
        m = int(np.argmax(row))
        return [int(cand[m]), float(row[m])]

    out = dict(n_sensors=int(sens.size), n_pairs=int(off.size),
               coh_max=float(off.max()), coh_median_all=float(np.median(off)),
               coh_median_within=float(np.median(within)) if within.size else None,
               coh_mean=float(off.mean()), n_orthogonal_pairs=int(orth.sum()),
               n_pairs_gt_0999=int((off > 0.999).sum()),
               n_pairs_gt_099=int((off > 0.99).sum()),
               rivals={str(int(t)): rival(t) for t in truth},
               n_rivals_gt_099={str(int(t)): int((coh[node_of[int(t)]] > 0.99).sum() - 1)
                                for t in truth})
    if top_nodes:
        out["top_coh_to_nearest_truth"] = {
            str(int(n)): max(((int(t), float(coh[node_of[int(n)], node_of[int(t)]]))
                              for t in truth), key=lambda q: q[1])
            for n in top_nodes}
    return out


def lt_invert(s, D, R, H_obs_all, sens_np, cand, truth, qexp, ucf_e, chunk,
              log=print, asm="dense", ls="dense"):
    """mv2_train_gpu.py:184-226 的反演循环原样（分块累加 = 同一目标函数）。"""
    dev, N = s.device, s.N
    cand_t = torch.as_tensor(cand, dtype=torch.long, device=dev)
    sens_t = torch.as_tensor(sens_np, dtype=torch.long, device=dev)
    H_obs = H_obs_all.index_select(1, sens_t)
    n_obs = LT_B * int(sens_np.size)
    theta = torch.full((LT_NC,), softplus_inv(LT_INIT_C), dtype=torch.float64,
                       device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=LT_LR)
    losses, times = [], []
    for it in range(LT_STEPS):
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        mse = 0.0
        for a in range(0, LT_B, chunk):
            b = min(a + chunk, LT_B)
            C = torch.nn.functional.softplus(theta)
            ke = torch.zeros(N, dtype=torch.float64, device=dev) \
                .index_copy(0, cand_t, ucf_e / C ** qexp)
            h, _q, _e = implicit_solve(s, D[a:b], R[a:b], ke=ke, adjoint="gpu",
                                       accuracy=LT_ACC, max_iter=200,
                                       status_machine=True, assemble=asm,
                                       linear_solver=ls)
            lc = ((h.index_select(1, sens_t) - H_obs[a:b]) ** 2).sum() / n_obs
            fn = h.grad_fn                     # 自定义 Function 的 grad_fn 就是 ctx
            lc.backward()
            mse += float(lc.detach())
            # ImplicitGGASolveGPU.forward 把自己的输出（head/flow/emitter）和终态分解
            # kf 直接挂在 ctx 上：输出张量 → grad_fn(C++ Node) → ctx → 输出张量 的
            # 引用环穿过 C++ 侧，Python 循环 GC 看不见（本机实测 gc.collect() 无效：
            # 每步净增 2×626 MB = 每块一个 [B,Nj,Nj] Cholesky 因子，第 10 步 12 GB
            # 溢出到 WDDM 共享内存、步时 8s→76s，最终 OOM）。cudss 路径每个 ctx 只有
            # ~5 MB，记录作业 60 步显存平直，故未曾暴露。这里反向后显式拆环
            # （只动本脚本，dgga 缺省不改；根治留给 dgga 侧 save_for_backward）。
            for nm in ("kf", "head", "flow", "emitter", "ke_t", "rt", "S", "resid"):
                if fn is not None and hasattr(fn, nm):
                    try:
                        delattr(fn, nm)
                    except AttributeError:
                        pass
            del h, _q, _e, lc, ke, C, fn
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        C = torch.nn.functional.softplus(theta)
        l1 = LT_LAM * C.sum()
        l1.backward()
        opt.step()
        torch.cuda.synchronize() if dev.type == "cuda" else None
        times.append(time.perf_counter() - t0)
        losses.append(mse + float(l1.detach()))
        if it == 0 or it == 9:
            mem = (f"，显存 {torch.cuda.memory_allocated() / 2 ** 20:.0f} MiB"
                   if dev.type == "cuda" else "")
            log(f"    第 {it + 1} 步 {times[-1]:.1f}s（loss={losses[-1]:.6e}{mem}）"
                f"→ 预计本组 {np.mean(times) * LT_STEPS / 60:.1f} min")
            sys.stdout.flush()
    with torch.no_grad():
        C_fin = torch.nn.functional.softplus(theta).cpu().numpy()
    order = np.argsort(-C_fin)
    top5 = [dict(node=int(cand[i]), C=float(C_fin[i]), true=bool(cand[i] in truth))
            for i in order[:5]]
    top3 = sorted(int(cand[i]) for i in order[:3])
    rank = {str(int(t)): int(np.where(cand[order] == t)[0][0]) + 1 for t in truth}
    return dict(loss_step1=losses[0], loss_end=losses[-1],
                loss_drop=losses[0] / max(losses[-1], 1e-300),
                loss_curve=[losses[i] for i in (0, 4, 9, 19, 29, 39, 49, 59)],
                top5=top5, top1_true=bool(top5[0]["true"]),
                top3_exact=(top3 == sorted(int(t) for t in truth)),
                n_true_in_top3=int(sum(1 for i in order[:3] if cand[i] in truth)),
                truth_C={str(int(t)): float(C_fin[cand == t][0]) for t in truth},
                truth_rank=rank,
                truth_left_zero={str(int(t)): bool(C_fin[cand == t][0] <= LT_INIT_C)
                                 for t in truth},
                n_C_above_0p1=int((C_fin > 0.1).sum()),
                t_total_sec=float(sum(times)), t_step_median=float(np.median(times)))


def stage_lt_leak(ks=KS, objectives=("dopt", "cover"), groups=("noiseless", "noisy"),
                  chunk=128, only=None, linear="dense", configs=None, out_path=None,
                  config_note=None):
    """configs / out_path / config_note 为新增可选参数（缺省行为逐位不变）：
    configs = [(name, sensor_node_idx[np], k, objective_tag), ...] 直接给定传感配置
    （scripts/augment_coherence.py 用它跑相干驱动增设与随机对照），此时不读
    placement_orders_ltown.npz；out_path 指定输出 json（缺省 lt_paths()['leak']）。"""
    print("===== ltown-leak：论文 L-TOWN 反演原样重跑，传感器 = S0 ∪ S_k =====")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA（dense 状态机 + GPU 约化伴随）")
    asm, ls = ("csr", "cudss") if linear == "cudss" else ("dense", "dense")
    if ls == "cudss":
        import nvmath  # noqa: F401  （缺 nvmath 时在此明确失败，不静默降级）
    p = lt_paths()
    if out_path:
        p["leak"] = out_path
    out = jload(p["leak"], {})
    print(f"INP {os.path.relpath(LT_INP, ROOT)} sha256={sha256(LT_INP)}；"
          f"assemble={asm} linear_solver={ls} chunk={chunk}")
    sys.stdout.flush()
    net, s = lt_load("dense", "cuda")
    se = GGASolver(net, mode="epanet", inp_path=LT_INP)
    jn, cand, truth, sens = lt_setup(se)
    qexp, ucf_e = lt_units(net)
    d0, rh0 = lt_nominal(net)
    dev = s.device
    print(f"L-TOWN Nj={s.Nj}；候选 {LT_NC} / 传感器 {LT_NS} / 真值 {truth.tolist()}"
          f"（C_true={LT_C_TRUE}，Ke_int={ucf_e / LT_C_TRUE ** qexp:.4e}）；"
          f"GPU {torch.cuda.get_device_name(0)}")
    # ---- 场景批（mv2 batchify(256)，seed 2026）----
    g = np.random.default_rng(LT_SEED_BATCH)
    Dn_ = d0[None, :] * g.uniform(0.85, 1.15, (LT_B, d0.size))
    Rn_ = rh0[None, :] + g.uniform(-1.0, 1.0, (LT_B, rh0.size))
    D = torch.as_tensor(Dn_, dtype=torch.float64, device=dev)
    R = torch.as_tensor(Rn_, dtype=torch.float64, device=dev)
    ke_true = np.zeros(net.N)
    ke_true[truth] = ucf_e / LT_C_TRUE ** qexp
    ke_true_t = torch.as_tensor(ke_true, dtype=torch.float64, device=dev)
    # ---- 观测：全节点 head（分块），无噪声 + 噪声场（按节点生成）----
    t0 = time.perf_counter()
    H_all = torch.empty(LT_B, net.N, dtype=torch.float64, device=dev)
    conv, iters, nstat = 0, 0, set()
    with torch.no_grad():
        for a in range(0, LT_B, chunk):
            b = min(a + chunk, LT_B)
            o = s.solve(D[a:b], R[a:b], ke_int=ke_true_t.unsqueeze(0).expand(b - a, -1),
                        status_machine=True, accuracy=LT_ACC, max_iter=200,
                        assemble=asm, linear_solver=ls)
            H_all[a:b] = o["head_ft"]
            conv += int(o["converged"].sum())
            iters = max(iters, int(o["iters"].max()))
            for row in np.unique(o["status"].cpu().numpy(), axis=0):
                nstat.add(row.tobytes())
    print(f"  观测：conv={conv}/{LT_B} iters≤{iters} 状态组={len(nstat)} "
          f"（{time.perf_counter() - t0:.1f}s，chunk={chunk}）")
    sys.stdout.flush()
    if conv != LT_B:
        raise RuntimeError("观测场景未全部收敛")
    noise = torch.as_tensor(NOISE_FT * np.random.default_rng(SEED_NOISE)
                            .standard_normal((LT_B, net.N)), dtype=torch.float64,
                            device=dev)
    H_obs = {"noiseless": H_all, "noisy": H_all + noise}
    # ---- 签名字典（CPU，一次）----
    Dfull = lt_dictionary(net, se, d0, rh0, cand, qexp, ucf_e)
    # ---- 传感配置 ----
    if configs is not None:
        cfgs = []
        for name, sn, k, obj in configs:
            sn = np.asarray(sn, dtype=np.int64)
            if not set(sens.tolist()) <= set(sn.tolist()):
                raise RuntimeError(f"配置 {name} 不含全部 S0（增设的定义是 S0 固定）")
            if np.intersect1d(sn, truth).size:
                raise RuntimeError(f"配置 {name} 的传感器落在漏点上（公平池违规）")
            cfgs.append((name, np.sort(np.unique(sn)), int(k), obj))
    else:
        z = np.load(p["orders"])
        cfgs = [("S0", sens, 0, "-")]
        for obj in objectives:
            for k in ks:
                add = jn[z["augment_" + obj][:k]]
                if np.intersect1d(add, sens).size or np.intersect1d(add, truth).size:
                    raise RuntimeError("增设点与 S0 / 漏点重叠（bug）")
                cfgs.append((f"{obj}+{k}", np.sort(np.r_[sens, add]), k, obj))
    if only:                                   # 按 --only 给定顺序跑（可作优先级）
        cfgs = [c for nm in only for c in cfgs if c[0] == nm]
    out.setdefault("config", dict(
        inp=os.path.relpath(LT_INP, ROOT), sha256=sha256(LT_INP),
        seed_setup=LT_SEED_SETUP, seed_batch=LT_SEED_BATCH, B=LT_B,
        n_candidates=LT_NC, n_sensors_s0=LT_NS, truth_nodes=truth.tolist(),
        C_true=LT_C_TRUE, C_init=LT_INIT_C, steps=LT_STEPS, lr=LT_LR, lam=LT_LAM,
        accuracy=LT_ACC, chunk=chunk, assemble=asm, linear_solver=ls,
        host=__import__("platform").node(), torch=torch.__version__,
        noise_ft=NOISE_FT, seed_noise=SEED_NOISE,
        noise_note="噪声按 [B, N] 全节点生成（seed 909）：各传感配置看到同一噪声实现；"
                   "论文 L-TOWN 用例本身无噪声（noiseless 组即论文口径）",
        obs_note="观测 = dense 状态机前向 accuracy=1e-6（mv2 用 cudss 前向，两路 "
                 "1e-16 级一致）；分块累加 = 同一 MSE 目标",
        recorded=LT_RECORDED, gpu=torch.cuda.get_device_name(0),
        C_probe=LT_C_PROBE, dict_hacc=LT_HACC))
    if config_note:
        out["config"]["config_note"] = config_note
    out.setdefault("runs", {})
    for name, sn, k, obj in cfgs:
        if name in out["runs"] and all(gp in out["runs"][name] for gp in groups):
            print(f"  [{name}] 已有结果，跳过")
            continue
        rec = out["runs"].get(name, {})
        rec.update(n_sensors=int(sn.size), k=k, objective=obj)
        rec["coherence"] = coherence_stats(Dfull, sn, cand, truth)
        c = rec["coherence"]
        print(f"\n  [{name}] 传感器 {sn.size}：相干 max={c['coh_max']:.10f} 中位(全)="
              f"{c['coh_median_all']:.4f} 中位(区内)={c['coh_median_within']:.4f} "
              f">0.999 {c['n_pairs_gt_0999']} >0.99 {c['n_pairs_gt_099']} 正交对 "
              f"{c['n_orthogonal_pairs']}；真值劲敌 "
              + " ".join(f"{t}:{v[0]}@{v[1]:.6f}" for t, v in c["rivals"].items()))
        for gp in groups:
            if gp in rec:
                continue
            t0 = time.perf_counter()
            r = lt_invert(s, D, R, H_obs[gp], sn, cand, truth, qexp, ucf_e, chunk,
                          asm=asm, ls=ls)
            r["coherence_top5"] = coherence_stats(
                Dfull, sn, cand, truth, [d["node"] for d in r["top5"]]
            )["top_coh_to_nearest_truth"]
            rec[gp] = r
            print(f"    {gp:9s}: loss {r['loss_step1']:.6e} → {r['loss_end']:.6e}"
                  f"（降 {r['loss_drop']:.1f}x）top-5 "
                  + "  ".join(f"n{d['node']}={d['C']:.3f}{'(真)' if d['true'] else ''}"
                              for d in r["top5"])
                  + f" | top-1 真={r['top1_true']} top-3 全中={r['top3_exact']} "
                  f"(中 {r['n_true_in_top3']}/3) 真值 C="
                  + " ".join(f"{t}:{v:.3f}(#{r['truth_rank'][t]})"
                             for t, v in r["truth_C"].items())
                  + f"  {time.perf_counter() - t0:.0f}s")
            sys.stdout.flush()
            out["runs"][name] = rec
            jdump(out, p["leak"])
        out["runs"][name] = rec
        jdump(out, p["leak"])
    print(f"ltown-leak 完成：{p['leak']}")


# ======================================================================
# Hanoi
# ======================================================================
def hanoi_paths():
    return dict(readable=os.path.join(DATA, "placement_augment_pub_hanoi_synth25.json"),
                orders=os.path.join(DATA, "placement_orders_pub_hanoi_synth25.npz"))


def stage_hanoi_augment(ks=KS):
    print("===== hanoi-augment：Hanoi 25 合成帧（calibrate.py 同帧），S0=random:10:2026 =====")
    from calibrate import Problem, SYNTH_SEED
    pb = Problem("hanoi")
    s, net = pb.s, pb.net
    junc = np.asarray(s.junc_nodes)
    pidx = pipe_param_idx(s)
    T = pb.d.shape[0]
    t0 = time.perf_counter()
    S, info = sensitivity_matrix(s, pb.d, pb.rh, None, junc, wrt="C",
                                 accuracy=1e-12, max_iter=200, polish_steps=3,
                                 return_info=True)
    S_full = S.reshape(T, junc.size, s.L)[:, :, pidx].copy()
    struct = pb.struct_mask[pidx]
    print(f"S_full {S_full.shape}，‖F‖∞={float(np.max(info['resid_inf'])):.2e}，"
          f"{time.perf_counter() - t0:.2f}s；结构性 {int(struct.sum())}；帧={pb.frames_note}")
    m = junc.size
    _, n, seed = HANOI_S0_SPEC.split(":")
    fixed = np.sort(np.random.default_rng(int(seed)).choice(m, size=int(n), replace=False))
    prev = os.path.join(DATA, "placement_orders_pub_hanoi.npz")
    if os.path.isfile(prev):
        zp = np.load(prev)
        if "augment_fixed" in zp.files and not np.array_equal(zp["augment_fixed"], fixed):
            raise RuntimeError("S0 与前一轮 placement_orders_pub_hanoi.npz 不一致")
    readable, orders = _augment_core(S_full, fixed, struct, ks, "Hanoi/25f")
    readable["stem"] = "pub_hanoi_synth25"
    readable["config"]["s0"] = HANOI_S0_SPEC
    readable["config"]["frames_note"] = pb.frames_note
    readable["config"]["synth_seed"] = SYNTH_SEED
    p = hanoi_paths()
    jdump(readable, p["readable"])
    np.savez_compressed(p["orders"], **orders)
    print(f"hanoi-augment 完成：{p['readable']}、{p['orders']}")


def stage_hanoi_calib(placements):
    import subprocess
    cmd = [sys.executable, "-X", "utf8", os.path.join(ROOT, "scripts", "calibrate.py"),
           "--stage", "l1hanoi_aug", "--placements", ",".join(placements)]
    print("===== hanoi-calib：", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        raise RuntimeError(f"calibrate.py rc={rc}")


# ======================================================================
# stage sigma-ladder：线性化贝叶斯后验的 σ 梯（L-TOWN 与 Hanoi 同算）
# ======================================================================
def stage_sigma_ladder(sigmas=(0.03, 0.1, 0.3)):
    """对 S0 / S0∪S_k 在 σ_noise∈{0.03,0.1,0.3} ft 下算逐管后验 std（同 posterior_std，
    σ_prior=15）。这是线性化高斯模型的期望标定误差（CRLB 型），**不是**非线性标定
    实跑；Hanoi 另有 calibrate.py 实跑可对照，L-TOWN（905 管、PRV 状态机）的非线性
    σ 梯实跑不在本轮预算内，如实标注。"""
    print("===== sigma-ladder：线性化后验 std（σ_prior=15）=====")
    out = {}
    for stem, cache, orders, target_from in (
            ("ltown", lt_paths()["cache"], lt_paths()["orders"], "pool"),
            ("pub_hanoi_synth25", None, hanoi_paths()["orders"], None)):
        if stem == "ltown":
            z = np.load(cache)
            S_full, struct = z["S_full"], z["dead"] | z["cm25"]
            o = np.load(orders)
            pool = o["pool"]
            S_use = S_full[:, pool, :]
            inv = {int(p): i for i, p in enumerate(pool)}
            fixed = np.asarray([inv[int(x)] for x in o["augment_fixed"]])
            added = {obj: np.asarray([inv[int(x)] for x in o["augment_" + obj]])
                     for obj in ("dopt", "cover")}
            ks = KS
        else:
            from calibrate import Problem
            pb = Problem("hanoi")
            junc = np.asarray(pb.s.junc_nodes)
            pidx = pipe_param_idx(pb.s)
            S = sensitivity_matrix(pb.s, pb.d, pb.rh, None, junc, wrt="C",
                                   accuracy=1e-12, max_iter=200, polish_steps=3)
            S_use = S.reshape(pb.d.shape[0], junc.size, pb.s.L)[:, :, pidx].copy()
            struct = pb.struct_mask[pidx]
            o = np.load(orders)
            fixed = o["augment_fixed"]
            added = {obj: o["augment_" + obj] for obj in ("dopt", "cover")}
            ks = [5, 10, 20]
        P = S_use.shape[2]
        nonstruct = ~struct
        unob0 = recovery_report(S_use, fixed, fixed, np.zeros(P, bool), struct,
                                0.0)["unobservable_mask"]
        rows = {}
        cfgs = [("S0", fixed, None)] + [(f"{obj}+{k}", np.r_[fixed, added[obj][:k]], obj)
                                        for obj in ("dopt", "cover") for k in ks]
        print(f"\n[{stem}] 非结构管 {int(nonstruct.sum())}，S0 下传感不足 {int(unob0.sum())}")
        print(f"  {'配置':>9} | " + " | ".join(
            f"σ={sg:<4} 中位  ≤σp/2  ≤σp/10  目标集中位" for sg in sigmas))
        for name, sel, obj in cfgs:
            rec_mask = (recovery_report(S_use, fixed, sel, unob0, struct, 0.0)
                        ["recovered_mask"])
            r = {}
            cells = []
            for sg in sigmas:
                ps = posterior_std(S_use, sel, SIG_P, sg)
                v = ps[nonstruct]
                vt = ps[unob0]
                vr = ps[rec_mask]
                r[str(sg)] = dict(
                    median_nonstruct=float(np.median(v)),
                    mean_nonstruct=float(v.mean()),
                    n_half_prior=int((v <= 0.5 * SIG_P).sum()),
                    n_tenth_prior=int((v <= 0.1 * SIG_P).sum()),
                    median_target=float(np.median(vt)) if vt.size else None,
                    median_recovered=float(np.median(vr)) if vr.size else None,
                    n_recovered=int(rec_mask.sum()))
                cells.append(f"{r[str(sg)]['median_nonstruct']:>10.3g} "
                             f"{r[str(sg)]['n_half_prior']:>6} "
                             f"{r[str(sg)]['n_tenth_prior']:>7} "
                             f"{'n/a' if vt.size == 0 else f'{np.median(vt):.3g}':>9}")
            r["n_sensors"] = int(np.unique(sel).size)
            rows[name] = r
            print(f"  {name:>9} | " + " | ".join(cells))
        out[stem] = dict(sigma_prior=SIG_P, sigmas=list(sigmas),
                         n_nonstruct=int(nonstruct.sum()),
                         n_target=int(unob0.sum()), rows=rows,
                         note="线性化高斯后验 std sqrt(diag(M^-1))，M=I/σ_p²+ΣSSᵀ/σ_n²；"
                              "期望标定误差的 CRLB 型代理，非非线性标定实跑")
    jdump(out, os.path.join(DATA, "augment_public_sigma_ladder.json"))
    print(f"\n已写 {os.path.join(DATA, 'augment_public_sigma_ladder.json')}")


# ======================================================================
# report
# ======================================================================
def stage_report():
    lines = []
    L = lines.append
    L("=" * 78)
    L("公开网增设全套（可复现版） - 数值表（由 scripts/augment_public.py --stage report 生成）")
    L("=" * 78)
    for stem, path in (("L-TOWN", lt_paths()["readable"]),
                       ("Hanoi/25f", hanoi_paths()["readable"])):
        rd = jload(path)
        if not rd:
            continue
        c = rd["config"]
        s0 = rd["s0"]
        L(f"\n### {stem} 增设曲线（{os.path.relpath(path, ROOT)}）")
        if "sfull" in rd:
            sf = rd["sfull"]
            L(f"  S_full：{sf['n_frames']} 帧 × {sf['Nj']} 候选 × P={sf['P']}；"
              f"状态机 iters={min(sf['iters_status_machine'])}~{max(sf['iters_status_machine'])}，"
              f"ACTIVE PRV/帧={sorted(set(sf['active_prv_per_frame']))}，"
              f"‖F‖∞max={sf['resid_inf_max']:.2e}；死支 {sf['n_dead']}，全帧钳位 "
              f"{sf['n_clamped_allframes']}，unobservable 地板 {sf['unobservable_floor']}")
        L(f"  |S0|={c['n_fixed']}（{c.get('s0')}），候选 {c['n_candidates']}，P={c['P']}，"
          f"结构性 {c['n_structural']}，目标集（S0 下传感不足）{c['n_target']}；"
          f"S0：可辨识 {s0['n_identifiable']} unobservable {s0['n_unobservable']} "
          f"秩 {s0['rank']} f={s0['f']:.2f} CRLB_ident={s0['crlb_ident']:.3e} "
          f"后验std中位(非结构)={s0['post_std_all_nonstruct']['median']:.3g} "
          f"≤σp/2 {s0['post_std_all_nonstruct']['n_half_prior']}")
        L(f"  {'k':>4} | {'目标':>6} | {'找回':>5} {'剩余':>5} {'丢失':>4} {'unob':>5} "
          f"{'可辨识':>6} {'秩':>4} | {'Δf':>9} {'证书':>6} {'CRLB_ident':>11} "
          f"{'bayes_tr':>10} | {'非结构后验std中位':>10} {'≤σp/2':>6} | {'目标集后验中位':>8}")
        for k in c["ks"]:
            for obj in ("dopt", "cover"):
                r = rd["augment"][obj]["per_k"][str(k)]
                pa, pt = r["post_std_all_nonstruct"], r["post_std_target"]
                pt_med = "n/a" if pt["median"] is None else "%.3g" % pt["median"]
                L(f"  {k:>4} | {obj:>6} | {r['n_recovered']:>5} {r['n_target_left']:>5} "
                  f"{r['n_lost']:>4} {r['n_unobservable']:>5} {r['n_identifiable']:>6} "
                  f"{r['rank']:>4} | {r['df']:>9.2f} {r['cert_ratio']:>6.3f} "
                  f"{r['crlb_ident']:>11.3e} {r['bayes_trace']:>10.1f} | "
                  f"{pa['median']:>10.3g} {pa['n_half_prior']:>6} | {pt_med:>8}")
            r = rd["reselect"]["per_k"][str(k)]
            L(f"  {k:>4} | {'重选':>6} | {r['n_recovered']:>5} {r['n_target_left']:>5} "
              f"{r['n_lost']:>4} {r['n_unobservable']:>5} {r['n_identifiable']:>6} "
              f"{r['rank']:>4} | {r['f'] - s0['f']:>9.2f} {'-':>6} "
              f"{r['crlb_ident']:>11.3e} {r['bayes_trace']:>10.1f} |")
        sk = rd["reselect"]["same_k_as_s0"]
        L(f"  对照 从零重选 |S0| 个：找回 {sk['n_recovered']} 丢失 {sk['n_lost']} "
          f"unobservable {sk['n_unobservable']}；80% 找回 k：dopt={rd['augment']['dopt']['k_recover80']} "
          f"cover={rd['augment']['cover']['k_recover80']}；全部找回 k：dopt="
          f"{rd['augment']['dopt']['k_recover_all']} cover={rd['augment']['cover']['k_recover_all']}")
    leak_files = [(lt_paths()["leak"], "主结果（集群 5090 作业 1543441，记录作业 mv2 的 csr+cuDSS 配置）"),
                  (os.path.join(DATA, "ltown_augment_leak_v100.json"),
                   "V100 服务器三卡分跑合并交叉核对（scripts/augment_public_compare.py）"),
                  (os.path.join(DATA, "ltown_augment_leak_local_dense.json"),
                   "本机 dense 路径交叉核对")]
    for lk_path, lk_tag in leak_files:
        lk = jload(lk_path)
        if not lk:
            continue
        c = lk["config"]
        L(f"\n### L-TOWN 漏损反演原样重跑 · {lk_tag}（{os.path.relpath(lk_path, ROOT)}）")
        L(f"  配方：seed {c['seed_setup']}/{c['seed_batch']}，B={c['B']}，候选 {c['n_candidates']}，"
          f"|S0|={c['n_sensors_s0']}，真值 {c['truth_nodes']}，C_true={c['C_true']}，"
          f"Adam lr={c['lr']} {c['steps']} 步，λ={c['lam']}，accuracy={c['accuracy']}，"
          f"chunk={c['chunk']}，{c.get('assemble', 'dense')}+{c.get('linear_solver', 'dense')}，"
          f"GPU={c['gpu']}（{c.get('host', '?')}，torch {c.get('torch', '?')}）；"
          f"噪声组 σ={c['noise_ft']} ft seed {c['seed_noise']}")
        rec = c["recorded"]
        L(f"  记录作业（5090，cudss）：loss {rec['loss_step1']:.6e}→{rec['loss_step60']:.6e}，"
          f"top-5 {rec['top5']}，top-1 真={rec['top1_true']}，top-3 全中={rec['top3_exact']}；"
          f"相干 max={rec['coh_max']:.10f} 中位={rec['coh_median']:.4f} >0.999={rec['n_pairs_gt_0999']}")
        L(f"  {'配置':>9} {'传感':>4} | {'相干max':>12} {'中位全':>7} {'中位区内':>7} {'>.999':>5} "
          f"{'>.99':>5} {'正交':>4} | {'劲敌651':>9} {'劲敌137':>9} {'劲敌720':>9} | "
          f"{'组':>9} {'loss末':>10} {'降':>6} {'top1真':>5} {'top3中':>5} "
          f"{'C651(#)':>10} {'C137(#)':>10} {'C720(#)':>10} {'top-5':>28}")
        for name, r in lk["runs"].items():
            co = r["coherence"]
            rv = co["rivals"]
            base = (f"  {name:>9} {r['n_sensors']:>4} | {co['coh_max']:>12.10f} "
                    f"{co['coh_median_all']:>7.4f} {co['coh_median_within']:>7.4f} "
                    f"{co['n_pairs_gt_0999']:>5} {co['n_pairs_gt_099']:>5} "
                    f"{co['n_orthogonal_pairs']:>4} | "
                    + " ".join(f"{rv[t][1]:>9.6f}" for t in ("651", "137", "720")) + " | ")
            first = True
            for gp in ("noiseless", "noisy"):
                if gp not in r:
                    continue
                g = r[gp]
                row = (base if first else " " * len(base)) + (
                    f"{gp:>9} {g['loss_end']:>10.4e} {g['loss_drop']:>6.1f} "
                    f"{str(g['top1_true']):>5} {g['n_true_in_top3']:>5} "
                    + " ".join(f"{g['truth_C'][t]:>6.3f}(#{g['truth_rank'][t]:>2})"
                               for t in ("651", "137", "720"))
                    + " " + ",".join(f"{d['node']}{'*' if d['true'] else ''}"
                                     for d in g["top5"]))
                L(row)
                first = False
    for cal_name, cal_tag in (("calib_gc1_pub_hanoi_v100.json", "V100 服务器"),
                              ("calib_gc1_pub_hanoi.json", "Windows 工作站")):
        cal = jload(os.path.join(DATA, cal_name))
        if not cal or not any(k.startswith("AUG_") for k in cal["runs"]):
            continue
        L(f"\n### Hanoi σ 梯标定：S0 vs S0∪S_k（data/{cal_name} AUG_*，{cal_tag}；"
          "λ=1e-4，3 噪声种子中位 [min,max]）")
        runs = {k: v for k, v in cal["runs"].items() if k.startswith("AUG_")}
        pls = sorted(set(k.split("_")[1] for k in runs),
                     key=lambda x: (0 if x == "augS0" else 1, x))
        L(f"  {'布点':>12} {'传感':>4} {'训练':>4} {'秩sub':>5} | " +
          " | ".join(f"{'σ=' + sg:>6} info-RMSE          sub-RMSE           valF p-RMSE"
                     for sg in ("0.03", "0.1", "0.3")))
        for pl in pls:
            cells = []
            n_s = n_tr = rk = None
            for sg in ("0.03", "0.1", "0.3"):
                rs = [v for k, v in runs.items()
                      if k.split("_")[1] == pl and f"_s{sg}_" in k]
                if not rs:
                    cells.append(" " * 60)
                    continue
                n_s, n_tr, rk = rs[0]["n_sensors"], rs[0]["n_train_sensors"], rs[0]["sub_rank"]

                def agg(key):
                    v = np.array([r[key] for r in rs])
                    return f"{np.median(v):.3f}[{v.min():.3f},{v.max():.3f}]"
                cells.append(f"{agg('info_rmse'):>19} {agg('sub_rmse'):>19} "
                             f"{agg('val_frame_rmse'):>19}")
            L(f"  {pl:>12} {n_s:>4} {n_tr:>4} {rk:>5} | " + " | ".join(cells))
    txt = "\n".join(lines)
    print(txt)
    # 只重写表格部分：wip 文件里表头之前的手写记录原样保留
    wip = os.path.join(DATA, "augment_public_wip.txt")
    head = ""
    if os.path.isfile(wip):
        old = open(wip, "r", encoding="utf-8").read()
        mark = "\n".join(lines[:2]) + "\n"
        if mark in old:
            head = old[:old.index(mark)]
    with open(wip, "w", encoding="utf-8") as f:
        f.write(head + txt + "\n")
    print(f"\n已写 {wip}（表格部分；表头之前的手写记录保留 {len(head)} 字符）")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["ltown-sfull", "ltown-augment", "ltown-leak",
                             "hanoi-augment", "hanoi-calib", "sigma-ladder",
                             "report"])
    ap.add_argument("--ks", default="5,10,20,40,80")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--groups", default="noiseless,noisy")
    ap.add_argument("--objectives", default="dopt,cover")
    ap.add_argument("--only", default="", help="ltown-leak 只跑这些配置名（逗号分隔）")
    ap.add_argument("--placements", default="augS0,augdopt5,augdopt10,augdopt20")
    ap.add_argument("--linear", default="dense", choices=["dense", "cudss"],
                    help="ltown-leak 的线性求解：dense（本机）| cudss（集群，csr 装配，"
                         "= 记录作业 mv2 的配置）")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    if a.stage == "ltown-sfull":
        stage_lt_sfull(a.frames)
    elif a.stage == "ltown-augment":
        stage_lt_augment(ks)
    elif a.stage == "ltown-leak":
        stage_lt_leak(ks, tuple(a.objectives.split(",")), tuple(a.groups.split(",")),
                      a.chunk, [x for x in a.only.split(",") if x], a.linear)
    elif a.stage == "hanoi-augment":
        stage_hanoi_augment(ks)
    elif a.stage == "hanoi-calib":
        stage_hanoi_calib([x for x in a.placements.split(",") if x])
    elif a.stage == "sigma-ladder":
        stage_sigma_ladder()
    elif a.stage == "report":
        stage_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""cpu_compare_hanoi.py - 公平 CPU 对比（Water Research 大修，回应 R2 Major 5 / Major 6）：
原生 EPANET 2.2 DLL + 有限差分  vs  HydroGrad(dgga) + 隐式伴随。
同一优化器、同一初值、同一观测、同一参数范围、同级求解精度，全部单线程 CPU。

固定协议（2026-10-23 定稿；之后只允许换种子/网络，不允许改口径）
  题面     scripts/calibrate.Problem("hanoi")：34 根 H-W 管全部自由（满秩）；真值 perpipe
           seed=7（C~U[75,145]）；25 帧合成正弦工况（SYNTH_SEED=4242）：20 训练帧 / 5 验证帧
           （[2,7,12,17,22]）；31 个 junction 按 HOLDOUT_SEED=505 留出 20%（6 个）只验证
           → 25 个训练传感器；观测 = 压力(ft) + iid 高斯噪声 σ=0.1 ft（noise_seed 两臂配对）。
  损失     L(C) = mean_{训练帧×训练传感器}(p_model − p_obs)² + λ·mean((C−C0)²)/30²，
           λ=1e-4，C0=130（与论文 §3.3 的 G-C1 训练损失逐项相同）。
  变量     C ∈ [40,160]^34，两臂都直接在 C 空间优化（不经 sigmoid 变换）。
  优化器   scipy.optimize.minimize(method="L-BFGS-B", jac=True)，两臂 options 完全相同。
  梯度     A1  EPANET 前向差分，h = h_rel·C（h_rel 由 --stage grad 的步长扫描定，默认 1e-4）
           A2  EPANET 中心差分，h = h_rel·C（默认 1e-3）
           B   HydroGrad ImplicitGGASolve：固定设备状态下收敛残差 F(z,θ)=0 的隐函数导数
               （一次伴随解），再由 torch 自动微分穿过 r(C)=4.727·L/C^1.852/d^4.871。
  求解精度 EPANET: EN_ACCURACY=1e-8（API 路径不受 1e-5 钳位）、EN_TRIALS=200、DAMPLIMIT=1e-6；
           HydroGrad B-tight: GGA accuracy=1e-12, max_iter=60, Newton 精修 3 步（与观测生成同款）；
           HydroGrad B-fast : 论文校核器口径 max_iter=20 + 精修 4 步（如实另报）。
  并行     torch.set_num_threads(1)，OMP/MKL=1；EPANET DLL 单线程。
  计数     1 次"模型调用" = 20 训练帧一次完整水力求解（EPANET = 20 次 EN_solveH，
           HydroGrad = 一次批量前向）。另单列 EN_solveH 总次数、伴随求解次数。

输出  data/cpu_compare_hanoi.json（按 seed 合并）、data/cpu_compare_hanoi_report.txt（追加）
运行  KMP_DUPLICATE_LIB_OK=TRUE python -X utf8 scripts/cpu_compare_hanoi.py --stage all --seeds 100
"""
import argparse
import ctypes
import json
import os
import platform
import shutil
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from scipy.optimize import minimize                           # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import calibrate as cal                                       # noqa: E402
from dgga.autodiff import ImplicitGGASolve, solve_polished    # noqa: E402
from dgga.calib import hw_resistance                          # noqa: E402
from dgga import epanet_ref as er                             # noqa: E402

EN_DAMPLIMIT = 17          # epanet2_enums.h:315
LAM = 1e-4
SIGMA = 0.1
C0, C_LO, C_HI = cal.C0, cal.C_LO, cal.C_HI
REG_SCALE = cal.REG_SCALE
LBFGSB_OPTS = dict(maxiter=400, maxfun=3000, ftol=1e-13, gtol=1e-8, maxcor=30)
EP_ACC, EP_TRIALS, EP_DAMP = 1e-8, 200, 1e-6
JSON_PATH = os.path.join(ROOT, "data", "cpu_compare_hanoi.json")
RPT_PATH = os.path.join(ROOT, "data", "cpu_compare_hanoi_report.txt")


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


# ================================================================ 题面
class Case:
    def __init__(self, noise_seed):
        self.pb = pb = cal.Problem("hanoi")
        self.C_true, _ = cal.make_truth(pb, "perpipe", cal.TRUTH_SEED["perpipe"])
        sensors = cal.get_sensors(pb)
        self.sens_train, self.sens_hold = cal.split_holdout(sensors)
        ob = cal.make_obs(pb, self.C_true, "perpipe7", sigma=SIGMA, noise_seed=noise_seed)
        self.obs, self.p_clean, self.d_used = ob["obs"], ob["p_clean"], ob["d_used"]
        self.tr, self.va = pb.train_frames, pb.val_frames
        self.free = pb.free_idx
        assert len(self.free) == 34, len(self.free)
        self.obs_tr = self.obs[np.ix_(self.tr, self.sens_train)]
        self.elev_s = pb.elev[self.sens_train]
        self.noise_seed = noise_seed

    def reg(self, C):
        return float(np.mean((C - C0) ** 2) / REG_SCALE)

    def evaluate(self, C_free, tag):
        """公共评价器（HydroGrad 精确前向，60 迭代 + 3 步精修）。"""
        pb = self.pb
        C_full = np.full(pb.s.L, C0)
        C_full[self.free] = C_free
        sol = solve_polished(pb.s, self.d_used, pb.rh, r_hw=pb.r_of_C(C_full),
                             accuracy=1e-12, max_iter=60, polish_steps=3)
        p = sol["head"] - pb.elev[None, :]
        junc = pb.s.junc_nodes
        err = p - self.p_clean
        return dict(
            tag=tag,
            C_rmse=float(np.sqrt(np.mean((C_free - self.C_true[self.free]) ** 2))),
            C_maxabs=float(np.max(np.abs(C_free - self.C_true[self.free]))),
            train_rmse_vs_obs=float(np.sqrt(np.mean(
                (p[np.ix_(self.tr, self.sens_train)] - self.obs_tr) ** 2))),
            train_rmse_clean=float(np.sqrt(np.mean(err[np.ix_(self.tr, self.sens_train)] ** 2))),
            holdout_sensor_rmse_clean=float(np.sqrt(np.mean(err[np.ix_(self.tr, self.sens_hold)] ** 2))),
            val_frame_rmse_clean=float(np.sqrt(np.mean(err[np.ix_(self.va, junc)] ** 2))),
        )


# ================================================================ 臂 A：EPANET DLL + FD
class EpanetArm:
    def __init__(self, case: Case):
        self.case = case
        pb = case.pb
        tmpd = tempfile.mkdtemp(prefix="hanoi_cmp_")            # TEMP 为 ASCII 路径
        inp = shutil.copy(os.path.join(ROOT, pb.cfg["inp"]), os.path.join(tmpd, "Hanoi.inp"))
        self.ep = er.Epanet(inp)
        self.ep.set_option(er.EN_ACCURACY, EP_ACC)
        self.ep.set_option(er.EN_TRIALS, EP_TRIALS)
        self.ep.set_option(EN_DAMPLIMIT, EP_DAMP)
        nid = {s: i + 1 for i, s in enumerate(self.ep.node_ids())}
        lid = {s: i + 1 for i, s in enumerate(self.ep.link_ids())}
        self.node_idx = np.array([nid[s] for s in pb.net.node_id])
        self.link_idx = np.array([lid[s] for s in pb.net.link_id])
        self.junc = pb.s.junc_nodes
        self.junc_ep = [int(x) for x in self.node_idx[self.junc]]
        self.sens_ep = [int(x) for x in self.node_idx[case.sens_train]]
        self.free_ep = [int(x) for x in self.link_idx[case.free]]
        self.d = case.d_used
        self.lib, self.ph = self.ep.lib, self.ep._ph
        self.ucf_flow, self.ucf_head = self.ep._ucf_flow, self.ep._ucf_head
        self._val = ctypes.c_double()
        self.n_solve = 0
        self.n_loss = 0
        self.max_iter_seen = 0
        self.max_relerr_seen = 0.0

    def set_C(self, C_free):
        for li, c in zip(self.free_ep, C_free):
            self.ep._check(self.lib.EN_setlinkvalue(self.ph, li, er.EN_ROUGHNESS, float(c)),
                           "EN_setlinkvalue(ROUGHNESS)")

    def _solve_frame(self, f, nodes_ep):
        for j, nj in zip(self.junc, self.junc_ep):
            self.ep._check(self.lib.EN_setnodevalue(self.ph, nj, er.EN_BASEDEMAND,
                                                    float(self.d[f, j]) * self.ucf_flow),
                           "EN_setnodevalue(BASEDEMAND)")
        self.ep._check(self.lib.EN_solveH(self.ph), "EN_solveH")
        self.n_solve += 1
        out = np.empty(len(nodes_ep))
        v = self._val
        for i, ni in enumerate(nodes_ep):
            self.lib.EN_getnodevalue(self.ph, ni, er.EN_HEAD, ctypes.byref(v))
            out[i] = v.value / self.ucf_head
        return out

    def stats_last(self):
        st = ctypes.c_double()
        self.lib.EN_getstatistic(self.ph, er.EN_ITERATIONS, ctypes.byref(st))
        it = int(st.value)
        self.lib.EN_getstatistic(self.ph, er.EN_RELATIVEERROR, ctypes.byref(st))
        return it, float(st.value)

    def pressures_all_junc(self, C_free, frames):
        """验证用：全部 junction 压力 [len(frames), N]（非传感器位 NaN）。"""
        self.set_C(C_free)
        P = np.full((len(frames), self.case.pb.net.N), np.nan)
        for i, f in enumerate(frames):
            h = self._solve_frame(f, self.junc_ep)
            P[i, self.junc] = h - self.case.pb.elev[self.junc]
        return P

    def loss(self, C_free):
        c = self.case
        self.set_C(C_free)
        P = np.stack([self._solve_frame(f, self.sens_ep) for f in c.tr]) - c.elev_s[None, :]
        it, re = self.stats_last()
        self.max_iter_seen = max(self.max_iter_seen, it)
        self.max_relerr_seen = max(self.max_relerr_seen, re)
        self.n_loss += 1
        mse = float(np.mean((P - c.obs_tr) ** 2))
        return mse + LAM * c.reg(C_free)

    def fun_fwd(self, C, h_rel):
        L0 = self.loss(C)
        g = np.empty_like(C)
        for k in range(len(C)):
            Ck = C.copy()
            h = h_rel * C[k]
            Ck[k] += h
            g[k] = (self.loss(Ck) - L0) / h
        return L0, g

    def fun_ctr(self, C, h_rel):
        L0 = self.loss(C)
        g = np.empty_like(C)
        for k in range(len(C)):
            h = h_rel * C[k]
            Cp, Cm = C.copy(), C.copy()
            Cp[k] += h
            Cm[k] -= h
            g[k] = (self.loss(Cp) - self.loss(Cm)) / (2 * h)
        return L0, g


# ================================================================ 臂 B：HydroGrad + 隐式伴随
class HydroArm:
    def __init__(self, case: Case, max_iter, polish):
        self.case, self.mi, self.polish = case, max_iter, polish
        pb = case.pb
        self.dt = torch.tensor(case.d_used[case.tr])
        self.rht = torch.tensor(pb.rh[case.tr])
        self.ket = torch.zeros(pb.net.N)
        self.r_base = torch.tensor(pb.r_base)
        self.free_t = torch.as_tensor(case.free)
        self.ln = torch.tensor(pb.len_ft[case.free])
        self.dm = torch.tensor(pb.diam_ft[case.free])
        self.obs_t = torch.tensor(case.obs_tr)
        self.elev_t = torch.tensor(case.elev_s)
        self.sens_t = torch.as_tensor(case.sens_train)
        self.n_fwd = 0
        self.n_adj = 0

    def _r(self, Ct):
        r = self.r_base.clone()
        r[self.free_t] = hw_resistance(Ct, self.ln, self.dm)
        return r

    def loss_only(self, C):
        with torch.no_grad():
            sol = solve_polished(self.case.pb.s, self.case.d_used[self.case.tr],
                                 self.case.pb.rh[self.case.tr],
                                 r_hw=self._r(torch.tensor(C)).numpy(),
                                 accuracy=1e-12, max_iter=self.mi, polish_steps=self.polish)
        self.n_fwd += 1
        P = sol["head"][:, self.case.sens_train] - self.case.elev_s[None, :]
        return float(np.mean((P - self.case.obs_tr) ** 2)) + LAM * self.case.reg(C)

    def fun(self, C):
        Ct = torch.tensor(np.asarray(C, dtype=np.float64), requires_grad=True)
        head, _, _ = ImplicitGGASolve.apply(self.dt, self.rht, self.ket, self._r(Ct),
                                            self.case.pb.s, 1e-12, self.mi, self.polish)
        pred = head[:, self.sens_t] - self.elev_t
        mse = ((pred - self.obs_t) ** 2).mean()
        reg = ((Ct - C0) ** 2).mean() / REG_SCALE
        loss = mse + LAM * reg
        loss.backward()
        self.n_fwd += 1
        self.n_adj += 1
        return float(loss), Ct.grad.detach().numpy().copy()


# ================================================================ 阶段 1：目标函数与梯度
def stage_grad(case: Case, A: EpanetArm, B: HydroArm, Bf: HydroArm):
    out = {}
    C_init = np.full(len(case.free), C0)
    Ct = case.C_true[case.free]

    # (0) 前向一致性：真值处 EPANET 压力 vs HydroGrad 观测生成器（p_clean），训练帧全 junction
    P_ep = A.pressures_all_junc(Ct, case.tr)
    junc = case.pb.s.junc_nodes
    dP = np.abs(P_ep[:, junc] - case.p_clean[np.ix_(case.tr, junc)])
    it, re = A.stats_last()
    out["forward_agreement"] = dict(max_abs_dp_ft=float(dP.max()), mean_abs_dp_ft=float(dP.mean()),
                                    epanet_iters_last=it, epanet_relerr_last=re,
                                    epanet_accuracy=EP_ACC, epanet_trials=EP_TRIALS)
    print(f"[fwd] EPANET vs HydroGrad 压力一致性（真值，20 训练帧×31 junction）："
          f"max|Δp|={dP.max():.3e} ft, mean={dP.mean():.3e} ft; EPANET iters={it}, relerr={re:.2e}")

    # (1) 目标函数一致性 @C0
    LA = A.loss(C_init)
    LB = B.loss_only(C_init)
    LBf = Bf.loss_only(C_init)
    out["objective_at_C0"] = dict(epanet=LA, hydrograd_tight=LB, hydrograd_fast=LBf,
                                  rel_diff_tight=abs(LA - LB) / abs(LB), rel_diff_fast=abs(LA - LBf) / abs(LBf))
    print(f"[obj] L(C0): EPANET={LA:.12e}  HydroGrad-tight={LB:.12e}  (rel diff {abs(LA-LB)/abs(LB):.2e})"
          f"  HydroGrad-fast={LBf:.12e} (rel diff {abs(LA-LBf)/abs(LBf):.2e})")

    # (2) 单次前向成本（20 帧一次模型调用），各取 5 次中位
    def med_time(fn, n=5):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    t_ep_fwd = med_time(lambda: A.loss(C_init))
    t_hg_fwd = med_time(lambda: B.loss_only(C_init))
    t_hgf_fwd = med_time(lambda: Bf.loss_only(C_init))
    t_hg_grad = med_time(lambda: B.fun(C_init))
    t_hgf_grad = med_time(lambda: Bf.fun(C_init))
    out["cost_single_call_s"] = dict(epanet_loss_20frames=t_ep_fwd,
                                     hydrograd_tight_loss=t_hg_fwd, hydrograd_tight_loss_grad=t_hg_grad,
                                     hydrograd_fast_loss=t_hgf_fwd, hydrograd_fast_loss_grad=t_hgf_grad)
    print(f"[cost] 一次 20 帧模型调用：EPANET {t_ep_fwd*1e3:.1f} ms | HydroGrad-tight 前向 {t_hg_fwd*1e3:.1f} ms,"
          f" 前向+伴随 {t_hg_grad*1e3:.1f} ms | HydroGrad-fast 前向 {t_hgf_fwd*1e3:.1f} ms, 前向+伴随 {t_hgf_grad*1e3:.1f} ms")

    # (3) 梯度对拍 @C0：伴随 vs EPANET FD（步长扫描）
    _, gB = B.fun(C_init)
    _, gBf = Bf.fun(C_init)
    out["grad_adjoint_tight_vs_fast_rel"] = float(np.linalg.norm(gB - gBf) / np.linalg.norm(gB))
    print(f"[grad] 伴随梯度 ‖g‖={np.linalg.norm(gB):.4e}; tight vs fast 相对差 {out['grad_adjoint_tight_vs_fast_rel']:.2e}")
    scan = []
    for scheme, fn in (("fwd", A.fun_fwd), ("ctr", A.fun_ctr)):
        for h_rel in (1e-2, 1e-3, 1e-4, 1e-5):
            n0 = A.n_solve
            t0 = time.perf_counter()
            _, gA = fn(C_init, h_rel)
            dt_ = time.perf_counter() - t0
            rel = float(np.linalg.norm(gA - gB) / np.linalg.norm(gB))
            cmax = float(np.max(np.abs(gA - gB) / np.maximum(np.abs(gB), 1e-3 * np.abs(gB).max())))
            cos = float(gA @ gB / np.linalg.norm(gA) / np.linalg.norm(gB))
            scan.append(dict(scheme=scheme, h_rel=h_rel, rel_l2_err=rel, max_comp_rel_err=cmax,
                             cosine=cos, time_s=dt_, en_solveH=A.n_solve - n0))
            print(f"  FD-{scheme} h_rel={h_rel:.0e}: rel-L2 err={rel:.2e} max-comp={cmax:.2e} cos={cos:.8f}"
                  f"  time={dt_:.2f}s  EN_solveH={A.n_solve-n0}")
    out["fd_scan_at_C0"] = scan
    best = {}
    for scheme in ("fwd", "ctr"):
        rows = [r for r in scan if r["scheme"] == scheme]
        best[scheme] = min(rows, key=lambda r: r["rel_l2_err"])["h_rel"]
    out["fd_best_h_rel"] = best
    # 速度比（同一 C0 处一次完整目标+梯度）
    tA_f = [r for r in scan if r["scheme"] == "fwd" and r["h_rel"] == best["fwd"]][0]["time_s"]
    tA_c = [r for r in scan if r["scheme"] == "ctr" and r["h_rel"] == best["ctr"]][0]["time_s"]
    out["speedup_loss_grad"] = dict(epanet_fwdFD_over_hydrograd_tight=tA_f / t_hg_grad,
                                    epanet_ctrFD_over_hydrograd_tight=tA_c / t_hg_grad,
                                    epanet_fwdFD_over_hydrograd_fast=tA_f / t_hgf_grad,
                                    epanet_ctrFD_over_hydrograd_fast=tA_c / t_hgf_grad)
    print(f"[ratio] 一次目标+梯度：EPANET-fwdFD/HydroGrad-tight = {tA_f/t_hg_grad:.2f}×, "
          f"EPANET-ctrFD/HydroGrad-tight = {tA_c/t_hg_grad:.2f}×; vs fast: {tA_f/t_hgf_grad:.2f}× / {tA_c/t_hgf_grad:.2f}×")
    return out


# ================================================================ 阶段 2：完整校准（配对种子）
def run_calib(case: Case, name, fun, counters_before, counters_after):
    C_init = np.full(len(case.free), C0)
    bounds = [(C_LO, C_HI)] * len(case.free)
    hist = []
    cb = counters_before()
    t0 = time.perf_counter()

    def f(C):
        L, g = fun(C)
        hist.append((time.perf_counter() - t0, float(L)))
        return L, g

    res = minimize(f, C_init, jac=True, method="L-BFGS-B", bounds=bounds, options=LBFGSB_OPTS)
    wall = time.perf_counter() - t0
    ca = counters_after()
    ev = case.evaluate(res.x, name)
    rec = dict(arm=name, wall_s=wall, nit=int(res.nit), nfev=int(res.nfev),
               status=int(res.status), message=str(res.message), final_loss=float(res.fun),
               counters={k: ca[k] - cb.get(k, 0) for k in ca},
               evaluation=ev, C_hat=[float(x) for x in res.x],
               traj=[(round(t, 4), L) for t, L in hist[::max(1, len(hist) // 60)]])
    print(f"[calib] {name:<22} wall={wall:8.2f}s nit={res.nit:4d} nfev={res.nfev:4d} "
          f"loss={res.fun:.6e} C_rmse={ev['C_rmse']:.4f} valF={ev['val_frame_rmse_clean']:.4f} "
          f"counters={rec['counters']}  [{res.message}]")
    return rec


def stage_calib(case: Case, A: EpanetArm, B: HydroArm, Bf: HydroArm, h_fwd, h_ctr, arms):
    recs = []
    cA = lambda: dict(en_solveH=A.n_solve, model_calls=A.n_loss)            # noqa: E731
    cB = lambda: dict(model_calls=B.n_fwd, adjoint_solves=B.n_adj)         # noqa: E731
    cBf = lambda: dict(model_calls=Bf.n_fwd, adjoint_solves=Bf.n_adj)      # noqa: E731
    if "B" in arms:
        recs.append(run_calib(case, "HydroGrad-adjoint-tight", B.fun, cB, cB))
    if "Bf" in arms:
        recs.append(run_calib(case, "HydroGrad-adjoint-fast", Bf.fun, cBf, cBf))
    if "A1" in arms:
        recs.append(run_calib(case, f"EPANET-fwdFD(h={h_fwd:.0e})", lambda C: A.fun_fwd(C, h_fwd), cA, cA))
    if "A2" in arms:
        recs.append(run_calib(case, f"EPANET-ctrFD(h={h_ctr:.0e})", lambda C: A.fun_ctr(C, h_ctr), cA, cA))
    return recs


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["grad", "calib", "all"])
    ap.add_argument("--seeds", default="100")
    ap.add_argument("--arms", default="B,Bf,A1,A2")
    ap.add_argument("--h_fwd", type=float, default=None)
    ap.add_argument("--h_ctr", type=float, default=None)
    a = ap.parse_args()
    sys.stdout = Tee(RPT_PATH)
    print("=" * 100)
    print(f"cpu_compare_hanoi  {time.strftime('%Y-%m-%d %H:%M:%S')}  stage={a.stage} seeds={a.seeds}")
    print(f"CPU={platform.processor()}  torch={torch.__version__} threads={torch.get_num_threads()}  "
          f"numpy={np.__version__}  L-BFGS-B opts={LBFGSB_OPTS}")
    allrec = json.load(open(JSON_PATH, encoding="utf-8")) if os.path.exists(JSON_PATH) else {}
    for sd in [int(x) for x in a.seeds.split(",")]:
        case = Case(sd)
        A = EpanetArm(case)
        B = HydroArm(case, 60, 3)
        Bf = HydroArm(case, 20, 4)
        print(f"--- seed={sd}  EPANET ver={A.ep.version()} flow_units={A.ep.flow_units} "
              f"ucf_flow={A.ucf_flow:.6f} ucf_head={A.ucf_head:.6f}  free pipes={len(case.free)} "
              f"train sensors={len(case.sens_train)} train frames={len(case.tr)}")
        key = str(sd)
        rec = allrec.get(key, {})
        rec["protocol"] = dict(net="Hanoi", truth_seed=7, noise_seed=sd, sigma_ft=SIGMA, lam=LAM, C0=C0,
                               bounds=[C_LO, C_HI], n_free=int(len(case.free)),
                               n_train_sensors=int(len(case.sens_train)), n_train_frames=int(len(case.tr)),
                               optimizer="scipy L-BFGS-B jac=True", lbfgsb=LBFGSB_OPTS,
                               epanet=dict(accuracy=EP_ACC, trials=EP_TRIALS, damplimit=EP_DAMP, version=A.ep.version()),
                               hydrograd_tight=dict(accuracy=1e-12, max_iter=60, polish=3),
                               hydrograd_fast=dict(accuracy=1e-12, max_iter=20, polish=4),
                               threads=1, cpu=platform.processor(), torch=torch.__version__,
                               date=time.strftime("%Y-%m-%d"))
        if a.stage in ("grad", "all"):
            rec["grad_test"] = stage_grad(case, A, B, Bf)
        if a.stage in ("calib", "all"):
            best = rec.get("grad_test", {}).get("fd_best_h_rel", {})
            h_fwd = a.h_fwd or best.get("fwd", 1e-4)
            h_ctr = a.h_ctr or best.get("ctr", 1e-3)
            A.max_iter_seen, A.max_relerr_seen = 0, 0.0
            rec["calib"] = stage_calib(case, A, B, Bf, h_fwd, h_ctr, a.arms.split(","))
            rec["epanet_worst_convergence_during_calib"] = dict(max_iters=A.max_iter_seen,
                                                                max_relerr=A.max_relerr_seen)
            print(f"[epanet] 校准期间最差收敛：iters={A.max_iter_seen} relerr={A.max_relerr_seen:.2e}")
        allrec[key] = rec
        json.dump(allrec, open(JSON_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        A.ep.close()
    print(f"saved -> {JSON_PATH}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""optimizer_v2.py - 模块三：一阶优化器增强 + 基线公平性重构（City D）。

题面与 G-D（scripts/baselines_calib.py）**逐条同一**：city_d、L1 档
（σ=0.1 ft、λ=1e-4）、真值 perpipe(seed=7)、布点 dopt40、20% 传感器留出
（seed=505）、20 训练帧、自由管 432 根、箱 [40,160]。评价一律走
calibrate.evaluate()。**NFE 口径**：1 NFE = 一次 20 训练帧的批量前向；
B 个起点跑 N 步 = B·N 次调用（批量只省墙钟，不省调用数）；伴随反传另计
nbwd（成本 ≈ 前向 1%，如实另列，不折进 NFE）；L-BFGS 线搜索的**每一次**
试探求值都计 NFE。

本文件新增的优化器（数学与实现见 dgga/optim2.py）：
  fo_*    纯一阶族（无 LM 尾） - 隔离"预条件子/多起点"本身的作用
    fo_plain        单起点 Adam → 普通 L-BFGS
    fo_schur        单起点 Adam → **Schur 补对角预条件** L-BFGS（零新增解）
    fo_gnex         单起点 Adam → **精确 GN 对角**预条件（每次刷新 1 NFE+1 nbwd）
    fo_ms<B>        B 起点 GPU 批量 Adam 筛 → 冠军起点普通 L-BFGS
    fo_ms<B>_schur  B 起点批量筛 → 冠军起点预条件 L-BFGS
  lm_*    带 LM 尾的完整管线（与 G-C1 梯度法、与基线混合法 DE→LM 同框比）
    lm_plain / lm_schur / lm_ms8_schur
  gd_ref  G-C1 固化配置原样重跑（CPU 伴随、~200 NFE），作平台锚点

基线（de/pso/cma/hybrid）由 baselines_calib.py 原样驱动，本文件只负责
**补足 City D 的调参预算**（stage tune2：budget 2000 = 评价预算，与 Hanoi
的 20000=评价预算同规格）与**补做 X 倍预算**（stage big：10000 NFE）。

运行：& python -X utf8 scripts/optimizer_v2.py --stage <verify|arm|run|tune2|
      eval2|big|report> [--seeds ...] [--arms ...]
"""

import argparse
import gc
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

import calibrate as cal                                    # noqa: E402
import baselines_calib as bl                               # noqa: E402
from dgga.solver import GGASolver                          # noqa: E402
from dgga.calib import hw_resistance                       # noqa: E402
from dgga import optim2                                    # noqa: E402
from dgga.sensitivity import sensitivity_matrix            # noqa: E402

torch.set_default_dtype(torch.float64)

OUT_JSON = os.path.join(ROOT, "data", "optimizer_v2.json")
NET, LEVEL = "city_d", "L1"
SIGMA, LAM = 0.1, 1e-4
BUDGET = 2000
GPU_ACC, GPU_MI, GPU_POLISH = 1e-6, 30, 4     # 见 optim2.multistart_heads
MS_SEED = 7070                                # 多起点 LHS 种子（新，独立）
EVAL_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

_OUT = {"path": OUT_JSON}


def set_out(p):
    _OUT["path"] = p if os.path.isabs(p) else os.path.join(ROOT, "data", p)


def load_out():
    fp = _OUT["path"]
    if os.path.exists(fp):
        return json.load(open(fp, encoding="utf-8"))
    return dict(meta={}, runs={}, arm={}, verify={})


def save_out(d):
    fp = _OUT["path"]
    os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, fp)


# ======================================================================
# GPU 伴随的显存回收（**本文件内解决，不动 autodiff.py**）
# ======================================================================
_GGA_CTX_FIELDS = ("kf", "head", "flow", "emitter", "ke_t", "rt", "S",
                   "resid", "solver")


def _free_gga_ctx(t, max_nodes=64):
    """把 ImplicitGGASolveGPU 那个节点上挂的大张量就地置空。

    为什么需要（memdiag 实测，city_d Nj=541）：autodiff.ImplicitGGASolveGPU.forward
    把终态 Cholesky 因子等直接写成 **ctx 属性**（ctx.kf 里的 chol 是
    [B·F, Nj, Nj] float64 = 46.8 MB/次），而不是走 ctx.save_for_backward。
    这样 ctx（Python 对象）与 C++ 侧的 autograd Node 构成一个**跨语言引用环**：
    Python 的循环垃圾回收器看不穿 C++ 那一段，所以
      · 显式 del 掉 head/flow/loss 不管用；
      · 强制 gc.collect() 也不管用 - 实测 25 次调用后 1247.2 MB，
        collect 之后仍是 1247.2 MB，25 个 ImplicitGGASolveGPUBackward 全都活着。
    在 200 步 Adam 的长循环里这就是每次 +46.8 MB，24 GB 的 4090 必 OOM
    （实测 22.17 GiB 时报 CUDA OOM）。
    既有代码没踩到：G-C1 走 CPU 伴随，基线批量目标全程 no_grad，
    只有本模块把 GPU 伴随放进了长优化循环。

    做法：backward 跑完后从输出张量沿 grad_fn 回溯找到该节点，把大字段置 None。
    节点对象本身（几十字节）仍可能泄漏，但显存被释放。
    **必须在 backward 之后调用** - 这些字段正是反向要用的。
    """
    if t is None or getattr(t, "grad_fn", None) is None:
        return 0
    seen, stack, n = set(), [t.grad_fn], 0
    while stack and len(seen) < max_nodes:
        nd = stack.pop()
        if nd is None or id(nd) in seen:
            continue
        seen.add(id(nd))
        if type(nd).__name__.startswith("ImplicitGGASolveGPU"):
            for f in _GGA_CTX_FIELDS:
                try:
                    if getattr(nd, f, None) is not None:
                        setattr(nd, f, None)
                except Exception:
                    pass
            n += 1
            continue
        for nxt in (getattr(nd, "next_functions", None) or ()):
            if nxt and nxt[0] is not None:
                stack.append(nxt[0])
    return n


# ======================================================================
# 引擎：与 G-C1 同一题面的批量可微目标（GPU 伴随）
# ======================================================================
class Engine:
    """theta（sigmoid 箱约束）空间的 B 起点批量损失/梯度 + NFE 账本。"""

    def __init__(self, pb, noise_seed, device=None, lam=LAM, sigma=SIGMA):
        self.pb = pb
        self.lam = lam
        self.C_true, _ = cal.make_truth(pb, "perpipe", cal.TRUTH_SEED["perpipe"])
        self.scen = cal.make_obs(pb, self.C_true,
                                 ("perpipe", cal.TRUTH_SEED["perpipe"]),
                                 sigma=sigma, noise_seed=noise_seed)
        self.sensors = cal.get_sensors(pb, "default")
        self.sens_train, self.sens_hold = cal.split_holdout(self.sensors)
        self.d_model = pb.d
        self.tr = pb.train_frames
        self.F = len(self.tr)
        self.free = pb.free_idx
        self.dim = len(self.free)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sg = GGASolver(pb.net, device=dev, mode="dense",
                            inp_path=os.path.join(ROOT, pb.cfg["inp"]))
        self.dev = self.sg.device
        dt = self.sg.dtype
        self.d_tr = torch.as_tensor(pb.d[self.tr], dtype=dt, device=self.dev)
        self.rh_tr = torch.as_tensor(pb.rh[self.tr], dtype=dt, device=self.dev)
        self.ke = self.sg.node_ke_default.to(self.dev)
        self.obs_t = torch.as_tensor(
            self.scen["obs"][np.ix_(self.tr, self.sens_train)],
            dtype=dt, device=self.dev)
        self.elev_t = torch.as_tensor(pb.elev[self.sens_train], dtype=dt,
                                      device=self.dev)
        self.sens_t = torch.as_tensor(np.asarray(self.sens_train),
                                      device=self.dev, dtype=torch.long)
        self.free_t = torch.as_tensor(np.asarray(self.free), device=self.dev,
                                      dtype=torch.long)
        self.len_t = torch.as_tensor(pb.len_ft[self.free], dtype=dt,
                                     device=self.dev)
        self.dia_t = torch.as_tensor(pb.diam_ft[self.free], dtype=dt,
                                     device=self.dev)
        self.rbase_t = torch.as_tensor(pb.r_base, dtype=dt, device=self.dev)
        self.n_obs = self.F * len(self.sens_train)
        self.w_reg = lam / (cal.REG_SCALE * self.dim) if lam > 0 else 0.0
        # ---- 显存看门狗 ----------------------------------------------
        # ImplicitGGASolveGPU.forward 把终态分解 kf（稠密 [B,Nj,Nj]，city_d
        # 上约 110 MB）直接挂在 ctx 上。反向跑完后这些 ctx 只能靠**循环垃圾
        # 回收**释放，而 gc 的触发看的是 Python 对象分配数，不是显存字节数 -
        # 200 步 Adam 连着跑就攒下 ~22 GiB，24 GB 的 4090 直接 OOM。
        # （既有代码没踩到：G-C1 走 CPU 伴随，基线批量目标全程 no_grad，
        # 只有本模块把 GPU 伴随放进了长优化循环。）
        # 修在**本文件**：不动 autodiff.py，缺省行为逐位不变。
        self.mem_gc_thresh = float(os.environ.get("M3_GC_BYTES", 2.0e9))
        self.mem_peak = 0
        self.n_gc = 0
        self.reset()

    # -------------------------------------------------- 账本
    def reset(self):
        self.nfe = 0
        self.nbwd = 0
        self.best = float("inf")
        self.best_C = None
        self.traj = []
        self.t0 = time.perf_counter()
        self.last_q = None

    def _account(self, d_nfe, losses, Cs):
        self.nfe += int(d_nfe)
        i = int(np.argmin(losses))
        if losses[i] < self.best:
            self.best = float(losses[i])
            self.best_C = np.asarray(Cs[i], dtype=np.float64).copy()
        self.traj.append((self.nfe, self.best,
                          time.perf_counter() - self.t0))

    # -------------------------------------------------- theta <-> C
    def C_of_theta(self, TH):
        return cal.C_LO + (cal.C_HI - cal.C_LO) * torch.sigmoid(TH)

    def theta_of_C(self, C):
        u = (np.asarray(C, dtype=np.float64) - cal.C_LO) / (cal.C_HI - cal.C_LO)
        u = np.clip(u, 1e-9, 1 - 1e-9)
        return np.log(u / (1 - u))

    def r_of_C(self, C):
        r = self.rbase_t.unsqueeze(0).expand(C.shape[0], -1)
        return r.index_copy(1, self.free_t,
                            4.727 * self.len_t / C ** 1.852
                            / self.dia_t ** 4.871)

    # -------------------------------------------------- 批量损失/梯度
    def loss_grad(self, TH_np, keep_flow=False):
        """TH_np [B,dim] -> (loss[B], grad[B,dim], C[B,dim])；nfe += B。"""
        TH = torch.as_tensor(np.atleast_2d(TH_np), dtype=self.sg.dtype,
                             device=self.dev).requires_grad_(True)
        C = self.C_of_theta(TH)
        r = self.r_of_C(C)
        head, flow = optim2.multistart_heads(
            self.sg, r, self.d_tr, self.rh_tr, self.ke,
            accuracy=GPU_ACC, max_iter=GPU_MI, polish_steps=GPU_POLISH)
        pred = head.index_select(2, self.sens_t) - self.elev_t
        mse = ((pred - self.obs_t) ** 2).mean(dim=(1, 2))
        reg = ((C - cal.C0) ** 2).mean(dim=1) / cal.REG_SCALE
        loss = mse + self.lam * reg
        loss.sum().backward()
        B = TH.shape[0]
        self.nbwd += B
        Cn = C.detach().cpu().numpy()
        ln = loss.detach().cpu().numpy()
        gn = TH.grad.detach().cpu().numpy()
        self._account(B, ln, Cn)
        if keep_flow:
            self.last_q = flow.detach()[0].cpu().numpy()      # [F,L]
        _free_gga_ctx(head)                    # 见函数注释：必须在 backward 之后
        del TH, C, r, head, flow, pred, mse, reg, loss
        self._sweep()
        return ln, gn, Cn

    def _sweep(self):
        """显存看门狗：越过阈值就强制一次循环回收（见 __init__ 注释）。"""
        if self.dev.type != "cuda":
            return
        a = torch.cuda.memory_allocated()
        if a > self.mem_peak:
            self.mem_peak = a
        if a > self.mem_gc_thresh:
            gc.collect()
            torch.cuda.empty_cache()
            self.n_gc += 1

    def loss_only(self, TH_np):
        with torch.no_grad():
            TH = torch.as_tensor(np.atleast_2d(TH_np), dtype=self.sg.dtype,
                                 device=self.dev)
            C = self.C_of_theta(TH)
            head, _ = optim2.multistart_heads(
                self.sg, self.r_of_C(C), self.d_tr, self.rh_tr, self.ke,
                accuracy=GPU_ACC, max_iter=GPU_MI, polish_steps=GPU_POLISH)
            pred = head.index_select(2, self.sens_t) - self.elev_t
            mse = ((pred - self.obs_t) ** 2).mean(dim=(1, 2))
            reg = ((C - cal.C0) ** 2).mean(dim=1) / cal.REG_SCALE
            ln = (mse + self.lam * reg).cpu().numpy()
        self._account(TH.shape[0], ln, C.cpu().numpy())
        return ln

    # -------------------------------------------------- 预条件子
    def precond(self, theta, C, kind, floor_rel=1e-6):
        """返回 (M_theta, info)。M_theta = (dC/dθ)²·M_C（sigmoid 链式）。"""
        if kind is None:
            return None, dict(kind="identity")
        dCdth = (cal.C_HI - cal.C_LO) * (
            1.0 / (1.0 + np.exp(-theta))) * (1.0 - 1.0 / (1.0 + np.exp(-theta)))
        if kind == "schur":
            if self.last_q is None:
                raise RuntimeError("schur 预条件子需要上一次前向的流量")
            r_full = np.full(self.pb.s.L, 0.0)
            r_full[:] = self.pb.r_base
            r_full[self.free] = hw_resistance(
                C, self.pb.len_ft[self.free], self.pb.diam_ft[self.free]).numpy()
            M_C, info = optim2.schur_diag_precond(
                self.pb.s, self.last_q, r_full, C, self.free,
                w_reg=self.w_reg, floor_rel=floor_rel)
        elif kind == "gnex":
            C_full = np.full(self.pb.s.L, cal.C0)
            C_full[self.free] = C
            S = sensitivity_matrix(self.pb.s, self.d_model[self.tr],
                                   self.pb.rh[self.tr], self.pb.r_of_C(C_full),
                                   self.sens_train, wrt="C", accuracy=1e-12,
                                   max_iter=cal.GGA_MI,
                                   polish_steps=cal.POLISH)[:, self.free]
            self.nfe += 1                    # sensitivity 内部的批前向
            self.nbwd += 1                   # 伴随雅可比
            M_C, info = optim2.gn_diag_from_S(S, self.n_obs, w_reg=self.w_reg,
                                              floor_rel=floor_rel)
        else:
            raise ValueError(kind)
        M = dCdth ** 2 * M_C
        fl = floor_rel * M.max()
        M = np.maximum(M, fl)
        info = dict(info, theta_cond=float(M.max() / M.min()))
        return M, info


# ======================================================================
# 各臂
# ======================================================================
ARMS = {
    "fo_plain":       dict(fam="fo", B=1, adam=200, pre=None),
    "fo_schur":       dict(fam="fo", B=1, adam=200, pre="schur"),
    "fo_gnex":        dict(fam="fo", B=1, adam=200, pre="gnex"),
    "fo_ms8":         dict(fam="fo", B=8, adam=150, pre=None),
    "fo_ms8_schur":   dict(fam="fo", B=8, adam=150, pre="schur"),
    "fo_ms16_schur":  dict(fam="fo", B=16, adam=80, pre="schur"),
    "lm_plain":       dict(fam="lm", B=1, adam=200, pre=None, lm_frac=0.35),
    "lm_schur":       dict(fam="lm", B=1, adam=200, pre="schur", lm_frac=0.35),
    "lm_ms8_schur":   dict(fam="lm", B=8, adam=100, pre="schur", lm_frac=0.35),
    "gd_ref":         dict(fam="gd"),
}
LBFGS_ROUNDS = 4


def _starts(eng, B):
    """起点 [B,dim]：0 号 = C0=130（与梯度法唯一初值一致），其余 LHS[80,145]。"""
    C = np.full((B, eng.dim), cal.C0)
    if B > 1:
        C[1:] = optim2.lhs_starts(B - 1, 80.0, 145.0, eng.dim, MS_SEED)
    return C


def run_arm(eng, arm, budget=BUDGET, lr0=0.3, floor_rel=1e-6, verbose=True):
    cfg = ARMS[arm]
    eng.reset()
    t0 = time.perf_counter()
    if cfg["fam"] == "gd":                       # G-C1 固化配置（CPU 伴随）
        out = cal.calibrate(eng.pb, eng.scen["obs"], eng.sens_train,
                            eng.d_model, lam=eng.lam)
        return dict(arm=arm, C=out["C_hat"], loss=out["mse_train"]
                    + eng.lam * out["reg"], nfe=out["nfe"], nbwd=out["nbwd"],
                    wall=time.perf_counter() - t0, traj=[],
                    detail=dict(adam=out["adam_steps"],
                                lbfgs=out["lbfgs_evals"],
                                lm=out["lm_iters"], pick=out["lm_pick"],
                                engine="cpu_adjoint"))

    B = cfg["B"]
    lm_frac = cfg.get("lm_frac", 0.0)
    b_first = int(round((1.0 - lm_frac) * budget))
    C0s = _starts(eng, B)
    TH = np.stack([eng.theta_of_C(c) for c in C0s])

    # ---- 阶段一：B 起点 GPU 批量 Adam（每步 B 次调用） ----
    n_adam = min(cfg["adam"], max(1, b_first // max(B, 1) - 4))
    TH_t = torch.as_tensor(TH, dtype=eng.sg.dtype, device=eng.dev)
    opt = torch.optim.Adam([TH_t.requires_grad_(True)], lr=lr0)
    for it in range(n_adam):
        if it in (int(0.6 * n_adam), int(0.85 * n_adam)):
            for g in opt.param_groups:
                g["lr"] *= 0.5
        ln, gr, Cn = eng.loss_grad(TH_t.detach().cpu().numpy(),
                                   keep_flow=False)
        opt.zero_grad()
        TH_t.grad = torch.as_tensor(gr, dtype=eng.sg.dtype, device=eng.dev)
        opt.step()
    ln, gr, Cn = eng.loss_grad(TH_t.detach().cpu().numpy(), keep_flow=False)
    win = int(np.argmin(ln))
    th = TH_t.detach().cpu().numpy()[win].copy()
    adam_nfe = eng.nfe
    if verbose:
        print("    [%s] adam %d 步 × B=%d → %d NFE，冠军起点 #%d loss=%.5e "
              "（各起点 %s）" % (arm, n_adam, B, adam_nfe, win, ln[win],
                                " ".join("%.3e" % x for x in np.sort(ln)[:4])),
              flush=True)

    # ---- 阶段二：单起点（预条件）L-BFGS ----
    pre_info = []

    def fun(x):
        l_, g_, _ = eng.loss_grad(x[None, :], keep_flow=False)
        return float(l_[0]), g_[0]

    lb_stop = []
    for rd in range(LBFGS_ROUNDS):
        if b_first - eng.nfe <= 32:
            break
        # 每轮开头在当前点求一次值/梯度（1 NFE，各臂一律计入）：既给
        # Schur 预条件子提供该点的收敛流量，又直接喂给 L-BFGS 当 f0/g0
        f0v, g0v, _C = eng.loss_grad(th[None, :], keep_flow=True)
        C_cur = cal.C_LO + (cal.C_HI - cal.C_LO) / (1.0 + np.exp(-th))
        M, info = eng.precond(th, C_cur, cfg["pre"], floor_rel=floor_rel)
        pre_info.append(info)
        minv = None if M is None else 1.0 / M
        quota = max(32, int(np.ceil((b_first - eng.nfe)
                                    / (LBFGS_ROUNDS - rd))))
        res = optim2.lbfgs_precond(fun, th, minv=minv, max_iter=10 ** 6,
                                   max_eval=quota, history=30,
                                   f0=float(f0v[0]), g0=g0v[0])
        th = res["x"]
        lb_stop.append(res["stop"])
        if res["stop"] in ("tol_grad", "no_descent"):
            break
    fo_nfe = eng.nfe
    fo_loss = eng.best
    C_fo = eng.best_C.copy()
    if verbose:
        print("    [%s] 一阶段终点 NFE=%d loss=%.6e stop=%s" %
              (arm, fo_nfe, fo_loss, ",".join(lb_stop)), flush=True)

    # ---- 阶段三：LM 精抛光（可选；口径与 G-C1/_lm_polish 一致） ----
    lm_iters = 0
    if lm_frac > 0:
        C = C_fo.copy()
        state = {"prev": None}
        stall = 0
        while budget - eng.nfe > 3 and stall < 2:
            cnt = dict(nfe=0, nbwd=0, tcv_flips=0)
            seg = int(min(5, max(1, (budget - eng.nfe) // 3)))
            C, inf = cal._lm_polish(eng.pb, C, eng.scen["obs"], eng.sens_train,
                                    eng.d_model, eng.lam, seg, cnt, state)
            loss = inf["mse"] + eng.lam * inf["reg"]
            eng.nbwd += cnt["nbwd"]
            eng._account(cnt["nfe"], [loss], [C])
            lm_iters += inf["n_iter"]
            stall = stall + 1 if inf["n_iter"] == 0 else 0
    return dict(arm=arm, C=eng.best_C, loss=eng.best, nfe=eng.nfe,
                nbwd=eng.nbwd, wall=time.perf_counter() - t0,
                traj=[[int(a), float(b), float(c)] for a, b, c in eng.traj],
                detail=dict(B=B, adam_steps=n_adam, adam_nfe=adam_nfe,
                            mem_peak_mb=eng.mem_peak / 1e6, n_gc=eng.n_gc,
                            fo_nfe=fo_nfe, fo_loss=fo_loss, lm_iters=lm_iters,
                            win_start=win, lbfgs_stop=lb_stop,
                            precond=pre_info, engine="gpu_adjoint"))


# ======================================================================
# stages
# ======================================================================
def stage_verify(device=None):
    """数值验收：批量逐场景 r_hw、GPU 伴随、预条件子与精确 GN 对角的关系。"""
    from dgga.autodiff import ImplicitGGASolve, solve_polished
    pb = bl.get_pb(NET)
    eng = Engine(pb, 100, device=device)
    out = dict(net=NET, N=int(pb.net.N), Nj=int(pb.s.Nj), L=int(pb.s.L),
               dim=eng.dim, device=str(eng.dev), F=eng.F,
               n_sens_train=int(len(eng.sens_train)))
    tr = eng.tr
    # A 前向：批量逐场景 r_hw vs 逐个 CPU solve_polished
    Cs = np.stack([np.full(eng.dim, cal.C0),
                   np.full(eng.dim, cal.C0) + 7.0,
                   np.full(eng.dim, cal.C0) - 12.0])
    Ct = torch.as_tensor(Cs, dtype=eng.sg.dtype, device=eng.dev)
    with torch.no_grad():
        hb, qb = optim2.multistart_heads(eng.sg, eng.r_of_C(Ct), eng.d_tr,
                                         eng.rh_tr, eng.ke, accuracy=GPU_ACC,
                                         max_iter=GPU_MI,
                                         polish_steps=GPU_POLISH)
    worst = 0.0
    for b in range(3):
        Cfull = np.full(pb.s.L, cal.C0)
        Cfull[eng.free] = Cs[b]
        sol = solve_polished(pb.s, pb.d[tr], pb.rh[tr], r_hw=pb.r_of_C(Cfull),
                             accuracy=1e-12, max_iter=cal.GGA_MI,
                             polish_steps=cal.POLISH)
        worst = max(worst, float(np.abs(hb[b].cpu().numpy()
                                        - sol["head"]).max()))
    out["fwd_max_dH_ft"] = worst
    print("A 前向：GPU 批量逐场景 r_hw vs CPU solve_polished  max|ΔH| = %.3e ft"
          % worst)

    # B 梯度：批量 GPU 伴随 vs 逐个 CPU 伴随
    TH = np.stack([eng.theta_of_C(c) for c in Cs])
    _l, g_gpu, _C = eng.loss_grad(TH)
    g_cpu = np.empty_like(g_gpu)
    for b in range(3):
        Cfull = np.full(pb.s.L, cal.C0)
        Cfull[eng.free] = Cs[b]
        rt = torch.tensor(pb.r_of_C(Cfull), requires_grad=True)
        head, _, _ = ImplicitGGASolve.apply(
            torch.tensor(pb.d[tr]), torch.tensor(pb.rh[tr]),
            torch.zeros(pb.net.N), rt, pb.s, 1e-12, cal.GGA_MI, cal.POLISH)
        pred = head[:, eng.sens_train] - torch.tensor(pb.elev[eng.sens_train])
        Ct_b = torch.tensor(Cs[b], requires_grad=True)
        reg = ((Ct_b - cal.C0) ** 2).mean() / cal.REG_SCALE
        (((pred - torch.tensor(eng.scen["obs"][np.ix_(tr, eng.sens_train)]))
          ** 2).mean() + eng.lam * reg).backward()
        dC = -1.852 * pb.r_of_C(Cfull)[eng.free] / Cs[b]
        gC = rt.grad.numpy()[eng.free] * dC + Ct_b.grad.numpy()
        dCdth = (cal.C_HI - cal.C_LO) / (2 + np.exp(TH[b]) + np.exp(-TH[b]))
        g_cpu[b] = gC * dCdth
    rel = float(np.linalg.norm(g_gpu - g_cpu) / np.linalg.norm(g_cpu))
    out["grad_rel_vs_cpu_adjoint"] = rel
    print("B 梯度：GPU 批量伴随 vs CPU 伴随（theta 空间）  相对差 = %.3e" % rel)

    # C 预条件子：Schur-Jacobi 近似 vs 精确 GN 对角
    eng.loss_grad(TH[:1], keep_flow=True)
    C_cur = Cs[0]
    M_s, i_s = eng.precond(TH[0], C_cur, "schur")
    M_x, i_x = eng.precond(TH[0], C_cur, "gnex")
    lo = np.log10(np.maximum(M_s, 1e-300))
    lx = np.log10(np.maximum(M_x, 1e-300))
    pear = float(np.corrcoef(lo, lx)[0, 1])
    spear = float(np.corrcoef(np.argsort(np.argsort(M_s)),
                              np.argsort(np.argsort(M_x)))[0, 1])
    ratio = M_s / M_x
    out["precond"] = dict(schur=i_s, gnex=i_x, log10_pearson=pear,
                          spearman=spear,
                          ratio_median=float(np.median(ratio)),
                          ratio_p10=float(np.percentile(ratio, 10)),
                          ratio_p90=float(np.percentile(ratio, 90)))
    print("C 预条件子：Schur-Jacobi vs 精确 GN 对角  log10 皮尔逊 r=%.4f  "
          "斯皮尔曼=%.4f  比值中位=%.3e [p10 %.2e, p90 %.2e]"
          % (pear, spear, np.median(ratio), np.percentile(ratio, 10),
             np.percentile(ratio, 90)))
    print("   条件数（theta 空间，地板 1e-6 后）：schur=%.3e  gnex=%.3e"
          % (i_s["theta_cond"], i_x["theta_cond"]))
    d = load_out()
    d["verify"] = out
    save_out(d)
    return out


def stage_arm(device=None, budget=200):
    """臂长：B 与每起点步耗、以及 2000 NFE 的墙钟外推。"""
    pb = bl.get_pb(NET)
    eng = Engine(pb, 100, device=device)
    rows = []
    for B in (1, 2, 4, 8, 16, 32):
        try:
            TH = np.stack([eng.theta_of_C(c) for c in _starts(eng, B)])
            eng.reset()
            eng.loss_grad(TH)                 # 预热
            if eng.dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(3):
                eng.loss_grad(TH)
            if eng.dev.type == "cuda":
                torch.cuda.synchronize()
            el = (time.perf_counter() - t) / 3.0
            rows.append(dict(B=B, sec_per_step=el, sec_per_nfe=el / B))
            print("  B=%2d  每步 %.4f s  → 每 NFE %.4f s（2000 NFE 外推 "
                  "%.0f s）" % (B, el, el / B, el / B * 2000), flush=True)
        except Exception as e:
            print("  B=%d 失败：%s" % (B, type(e).__name__), flush=True)
            break
    d = load_out()
    d["arm"] = dict(rows=rows, device=str(eng.dev),
                    gpu=torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "cpu")
    save_out(d)
    return rows


def stage_run(arms, seeds, budget=BUDGET, device=None, floor_rel=1e-6):
    pb = bl.get_pb(NET)
    reason, Vr, k_sub, rank_eps = None, None, None, None
    data = load_out()
    for sd in seeds:
        eng = Engine(pb, sd, device=device)
        if reason is None:
            reason, Vr, k_sub, rank_eps = cal.get_diag(pb, "default",
                                                       eng.sensors)
        for arm in arms:
            key = "%s|%s|n%d|B%d" % (arm, NET, sd, budget)
            if key in data["runs"]:
                print("  跳过已有 %s" % key, flush=True)
                continue
            print("== %s 种子 %d ==" % (arm, sd), flush=True)
            r = run_arm(eng, arm, budget=budget, floor_rel=floor_rel)
            ev = cal.evaluate(pb, eng.C_true, r["C"], reason, Vr, k_sub,
                              rank_eps, eng.scen, eng.sens_train,
                              eng.sens_hold, eng.d_model)
            rec = dict(arm=arm, net=NET, level=LEVEL, noise_seed=sd,
                       budget=budget, loss=float(r["loss"]), nfe=int(r["nfe"]),
                       nbwd=int(r["nbwd"]), wall_sec=float(r["wall"]),
                       device=str(eng.dev), detail=r["detail"],
                       traj=r["traj"][-4000:],
                       C_hat_free=[round(float(x), 3) for x in r["C"]], **ev)
            data["runs"][key] = rec
            save_out(data)
            print("  -> loss=%.6e NFE=%d nbwd=%d sub=%.3f info=%.3f "
                  "valF=%.4f  %.0fs" %
                  (r["loss"], r["nfe"], r["nbwd"], ev["sub_rmse"],
                   ev.get("info_rmse", float("nan")), ev["val_frame_rmse"],
                   r["wall"]), flush=True)


def stage_cost(device=None, reps=3):
    """把"1 NFE"这张票的**实际价钱**量出来（同一节点、同一进程）。

    项目既有口径把 LM 的一次伴随雅可比记作 nfe+=1 / nbwd+=1
    （calibrate._lm_polish 的注释与 G-D 的 declarations 都这么写）。
    但 sensitivity_matrix 在 city_d 上是 **20 次稀疏 LU + 20×32 次回代**，
    不是一次前向。这个便宜同时被**我们的 lm_* 臂和基线 hybrid（DE→LM）**
    享用，所以两者互比不受影响；但"纯一阶 vs 带 LM 尾"的 NFE 曲线会被扭曲。
    本 stage 不改口径（改了就与冻结的 G-C1/G-D 不可比），而是把折算比测出来
    如实登记，并让墙钟 Pareto 去当裁判。
    """
    from dgga.autodiff import solve_polished
    pb = bl.get_pb(NET)
    eng = Engine(pb, 100, device=device)
    tr, free = eng.tr, eng.free
    C0f = np.full(eng.dim, cal.C0)
    Cfull = np.full(pb.s.L, cal.C0)
    Cfull[free] = C0f
    r_full = pb.r_of_C(Cfull)
    th = eng.theta_of_C(C0f)[None, :]

    def tm(fn, n):
        fn()
        if eng.dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        if eng.dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / n

    o = {}
    o["t_gpu_fwdbwd"] = tm(lambda: eng.loss_grad(th), reps)      # 我们的 1 NFE
    o["t_gpu_fwd"] = tm(lambda: eng.loss_only(th), reps)
    o["t_cpu_fwd"] = tm(lambda: solve_polished(
        pb.s, pb.d[tr], pb.rh[tr], r_hw=r_full, accuracy=1e-12,
        max_iter=cal.GGA_MI, polish_steps=cal.POLISH), reps)     # _lm_polish 的 fwd
    o["t_sensitivity"] = tm(lambda: sensitivity_matrix(
        pb.s, eng.d_model[tr], pb.rh[tr], r_full, eng.sens_train, wrt="C",
        accuracy=1e-12, max_iter=cal.GGA_MI, polish_steps=cal.POLISH), reps)
    # 基线的种群批（同一 GPU 批量目标，population 一次 kernel）
    for P in (1, 32, 64):
        ob = bl.BatchObjective(pb, LEVEL, 100, str(eng.dev), 10 ** 9,
                               bl.GRID[NET])
        X = np.tile(C0f, (P, 1))
        o["t_base_P%d" % P] = tm(lambda: ob.loss_batch(X), reps)
        o["t_base_P%d_per_nfe" % P] = o["t_base_P%d" % P] / P
        o["base_chunk"] = int(ob.chunk)
    # 折算：一次伴随雅可比 = 多少次"名义 1 NFE"
    o["jac_in_cpu_fwd"] = o["t_sensitivity"] / o["t_cpu_fwd"]
    o["jac_in_gpu_nfe"] = o["t_sensitivity"] / o["t_gpu_fwdbwd"]
    o["jac_in_base_nfe64"] = o["t_sensitivity"] / o["t_base_P64_per_nfe"]
    o["n_lu_per_jac"] = int(eng.F)
    o["n_rhs_per_jac"] = int(eng.F * len(eng.sens_train))
    # ---- 噪声地板：真值 C_true 自己的训练损失（任何优化器的下界参照） ----
    # 观测 = 干净水头 + N(0,σ²)，所以 C_true 的训练 MSE 就是该噪声实现的样本
    # 方差，期望 σ²=0.01。低于它 = 在拟合噪声，不是在恢复参数。
    orc = {}
    for sd in (100, 101, 102, 103, 104):
        e = Engine(pb, sd, device=device)
        th_t = e.theta_of_C(e.C_true[e.free])[None, :]
        orc[str(sd)] = float(e.loss_only(th_t)[0])
    o["oracle_loss_by_seed"] = orc
    o["oracle_loss_median"] = float(np.median(list(orc.values())))
    o["sigma2"] = SIGMA ** 2
    print("  噪声地板：真值 C_true 的训练损失 中位 %.5e（σ²=%.4g）" %
          (o["oracle_loss_median"], o["sigma2"]), flush=True)
    o["cpus"] = int(os.cpu_count() or 0)
    o["torch_threads"] = int(torch.get_num_threads())
    o["gpu"] = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu")
    o["dim"] = eng.dim
    o["n_sens_train"] = int(len(eng.sens_train))
    o["F"] = int(eng.F)
    for k in sorted(o):
        v = o[k]
        print("  %-22s %s" % (k, ("%.6g" % v) if isinstance(v, float) else v),
              flush=True)
    d = load_out()
    d["cost"] = o
    save_out(d)
    return o


OLD_CFG = {"de": "de1_NP32", "pso": "pso1_NP32", "cma": "cma1_auto_s18",
           "hybrid": "hy3_NP32_f0.3"}


def stage_evalold(algos, seeds, device, budget=BUDGET,
                  out="baselines_v2.json", tag="evalold"):
    """机器对照：用**旧调参预算(300)选出的配置**在本机跑同样的评价预算。

    这样"调参预算 300→2000"的效果与"换了机器"的效果就分得开了 - 不然
    新老结果差多少都可以赖到硬件上。
    """
    bl.set_outfile(out)
    data = bl.load_out()
    for algo in algos:
        cfg = next(c for c in bl.HYPER[algo] if c["name"] == OLD_CFG[algo])
        for sd in seeds:
            k = bl.run_key(NET, LEVEL, algo, cfg["name"], sd, 1000 + sd,
                           budget, tag)
            if k in data["runs"]:
                print("  跳过已有 %s" % k, flush=True)
                continue
            rec = bl.one_run(NET, LEVEL, algo, cfg, sd, 1000 + sd, budget,
                             device, do_eval=True, tag=tag)
            data["runs"][k] = rec
            bl.save_out(data)


def stage_tune2(algos, device, budget=BUDGET, out="baselines_v2.json",
                cfgs=None):
    """公平性重构：City D 基线调参预算补到 = 评价预算（与 Hanoi 同规格）。

    cfgs 非空时只跑这些配置（分片用；分片各写各的文件，随后 tunemerge 合并
    并在**全部配置**上重新选优 - 分片内选优会选错）。
    """
    bl.set_outfile(out)
    if cfgs:
        keep = set(cfgs)
        saved = {a: bl.HYPER[a] for a in algos}
        try:
            for a in algos:
                bl.HYPER[a] = [c for c in saved[a] if c["name"] in keep]
                if not bl.HYPER[a]:
                    raise SystemExit("算法 %s 没有匹配的配置：%s" % (a, cfgs))
            bl.stage_tune(NET, algos, budget, device)
        finally:
            for a in algos:
                bl.HYPER[a] = saved[a]
    else:
        bl.stage_tune(NET, algos, budget, device)


def stage_tunemerge(algos, shards, out="baselines_v2.json"):
    """把 tune 分片合并进 out，并在全部配置上重新算 median_loss / chosen。"""
    bl.set_outfile(out)
    data = bl.load_out()
    tun = data.setdefault("tuning", {}).setdefault(NET, {})
    # 幂等：每次都从分片重建，不在既有条目上累加（否则重复 merge 会把
    # nfe_bill 翻倍 - 第一次跑就踩到了）
    for algo in algos:
        tun.pop(algo, None)
    for sp in shards:
        fp = sp if os.path.isabs(sp) else os.path.join(ROOT, "data", sp)
        if not os.path.exists(fp):
            print("  缺分片 %s（跳过）" % fp, flush=True)
            continue
        j = json.load(open(fp, encoding="utf-8"))
        data["runs"].update(j.get("runs", {}))
        for algo, e in ((j.get("tuning") or {}).get(NET) or {}).items():
            t = tun.setdefault(algo, dict(budget=e.get("budget"),
                                          seeds=e.get("seeds"), results={},
                                          nfe_bill=0))
            for cname, r in e.get("results", {}).items():
                cur = t["results"].setdefault(cname, dict(cfg=r["cfg"],
                                                          losses={}))
                cur["losses"].update(r.get("losses", {}))
            t["nfe_bill"] = t.get("nfe_bill", 0) + e.get("nfe_bill", 0)
    for algo in algos:
        t = tun.get(algo)
        if not t:
            continue
        med = {n: float(np.median(list(r["losses"].values())))
               for n, r in t["results"].items() if r["losses"]}
        t["median_loss"] = med
        t["chosen"] = min(med, key=med.get)
        print("[tunemerge %s] 配置数=%d 中位损失=%s 选中=%s 账单=%s NFE" %
              (algo, len(med),
               {k: "%.3e" % v for k, v in sorted(med.items())},
               t["chosen"], t["nfe_bill"]), flush=True)
    bl.save_out(data)


def stage_eval2(algos, seeds, device, budget=BUDGET, out="baselines_v2.json",
                tag="eval2"):
    bl.set_outfile(out)
    bl.stage_evalrun(NET, LEVEL, algos, seeds, budget, device, tag=tag)


def stage_big(algos, seeds, device, budget=10000, out="baselines_v2.json"):
    """X 倍预算：基线在 5× 评价预算下再跑一遍（外推的实测版）。"""
    bl.set_outfile(out)
    bl.GRID[NET] = [200, 500, 1000, 2000, 5000, 10000, 20000]
    bl.stage_evalrun(NET, LEVEL, algos, seeds, budget, device, tag="big")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--seeds", default="100,101,102,103,104")
    ap.add_argument("--algos", default="de,pso,cma,hybrid")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bout", default="baselines_v2.json")
    ap.add_argument("--floor", type=float, default=1e-6)
    ap.add_argument("--cfgs", default="")
    ap.add_argument("--shards", default="")
    a = ap.parse_args()
    if a.out:
        set_out(a.out)
    seeds = [int(x) for x in a.seeds.split(",") if x]
    arms = list(ARMS) if a.arms == "all" else a.arms.split(",")
    algos = [x for x in a.algos.split(",") if x]
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device=%s torch=%s cuda=%s" %
          (dev, torch.__version__,
           torch.cuda.get_device_name(0) if torch.cuda.is_available()
           else "-"), flush=True)
    if a.stage == "verify":
        stage_verify(a.device)
    elif a.stage == "arm":
        stage_arm(a.device)
    elif a.stage == "run":
        stage_run(arms, seeds, a.budget, a.device, floor_rel=a.floor)
    elif a.stage == "tune2":
        stage_tune2(algos, dev, a.budget, a.bout,
                    cfgs=[c for c in a.cfgs.split(",") if c] or None)
    elif a.stage == "tunemerge":
        stage_tunemerge(algos, [x for x in a.shards.split(",") if x], a.bout)
    elif a.stage == "eval2":
        stage_eval2(algos, seeds, dev, a.budget, a.bout)
    elif a.stage == "big":
        stage_big(algos, seeds, dev, a.budget, a.bout)
    elif a.stage == "cost":
        stage_cost(a.device)
    elif a.stage == "evalold":
        stage_evalold(algos, seeds, dev, a.budget, a.bout)
    else:
        raise SystemExit("未知 stage: %s" % a.stage)


if __name__ == "__main__":
    main()

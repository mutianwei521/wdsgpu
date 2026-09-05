# -*- coding: utf-8 -*-
"""baselines_calib.py - G-D：全局优化基线 vs 梯度法（G-C1）配对对比矩阵。

题面与口径（与 scripts/calibrate.py 的 G-C1 完全一致，逐条声明）：
* 题面：同一 make_truth(perpipe, seed=7) 真值、同一 make_obs 噪声实现（按节点
  生成、种子配对）、同一布点（city_d dopt40 / hanoi 全 junction）、同一 20%
  传感器留出（HOLDOUT_SEED=505）、同一 20 训练帧。
* 决策空间：与 G-C1 相同的自由管集合（city_d 432，冻结 43 根结构性不可辨识管
  在 C0=130；hanoi 34 满秩），基线在箱 [C_LO,C_HI]=[40,160] 内直接搜 C
  （不经 sigmoid 变换，对基线更自然）。
* 损失：与 G-C1 同一训练损失 mse + λ·mean((C−C0)²)/30²（L0: λ=0，L1: λ=1e-4）。
* 评价：同一 calibrate.evaluate()（sub_rmse / info_rmse / 验证压力全指标）。
* NFE 口径：1 NFE = 一次完整 20 训练帧批量前向；种群第 i 个个体的前向各计
  1 NFE（初始化种群也计入）。混合法 LM 段的伴随雅可比另计 nbwd（与梯度法
  口径一致：反传成本约前向 1%，如实另列）。

对基线有利的设计决策（逐条声明，防 straw man 指控）：
1. 种群/邻域评估整体喂批维 B=pop×20帧 给 dense 求解器（GPU float64 若可用），
   基线拿到与梯度法同级的批量前向引擎；
2. 基线前向用 max_iter=60 的完整收敛（梯度法反演内环是 20 迭代+4 步 Newton
   抛光的省预算版） - 同记 1 NFE，基线每 NFE 拿到的前向更收敛；
3. 初始种群 LHS 覆盖真值先验支撑 [75,145]（真值 C~U[75,145]），并额外注入
   C0=130 个体（梯度法的唯一初值）；SA/CMA-ES 起点 x0=C0=130 与梯度法相同；
4. 预算网格快照取"首次跨越预算点的批次末" - 基线在预算 b 处可多用至多
   pop−1 次评估（宽松方向）；
5. 超参协议：每基线默认 + 3 组备选，在 3 个专用调参种子（900..902，不在评价
   种子集内）上按中位最终训练损失选优；调参 NFE 单独列账。梯度法不做任何
   新调参（沿用 G-C1 固化配置，其调参历史属 G-C1 阶段）。

基线实现：
  DE      scipy.optimize.differential_evolution（vectorized=True, deferred；
          init 显式传入 [NP,dim] LHS 种群 → 绝对种群规模，popsize 参数失效；
          polish=False 关闭额外局部搜索，tol=atol=0 关闭提前停）。
  SA      scipy.optimize.dual_annealing（maxfun=预算；本质串行，无法吃批维，
          臂长实测后如实缩放其预算/种子数）。
  PSO     自写全局最优版惯性权重 PSO：w=0.7298, c1=c2=1.49618
          （Clerc-Kennedy 收缩系数的等价惯性形式，Eberhart & Shi 2000,
          "Comparing inertia weights and constriction factors in PSO"），
          速度上限 0.5·箱宽，位置越界钳位+速度清零。
  CMA-ES  pip 包 cma 4.4.4（Hansen 官方实现），ask/tell 批量评估，
          bounds=[40,160]（内部 BoundTransform）。
  混合法  DE 粗搜（预算 60%）→ LM 精修（余下预算，直接复用
          calibrate._lm_polish：精确伴随雅可比 + 阻尼正规方程，分段调用以
          记录轨迹；每段重置 mu=1e-6，如实声明）。

运行：& python -X utf8 scripts/baselines_calib.py --stage <arm|tune|evalrun|gdextra|status> ...
"""

import argparse
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

import calibrate as cal                                   # noqa: E402
from dgga.solver import GGASolver                         # noqa: E402
from dgga.calib import hw_resistance                      # noqa: E402

torch.set_default_dtype(torch.float64)

OUT_JSON = os.path.join(ROOT, "data", "baselines_gd.json")
# 基线前向迭代数（臂长实测定）：hanoi 60（提前收敛 relerr<1e-12，实际 <30 迭代）；
# city_d 30（relerr 平台 ~3e-7 与 mi=60 相同，损失与 G-C1 前向相对差 ~1e-7，
# 而墙钟 267 vs 562 ms/NFE - GPU 8GB 与他人实验共享，显存受限 chunk=12）
BASE_MI_NET = {"hanoi": 60, "city_d": 30}
BASE_MI = 60                    # 兼容旧引用（实际取 BASE_MI_NET）
TUNE_SEEDS = [900, 901, 902]    # 专用调参噪声种子（不与评价种子 100.. 相交）
GRID = {"hanoi": [200, 600, 1000, 2000, 5000, 10000, 20000],
        "city_d": [200, 500, 1000, 2000, 5000]}
LEVEL_CFG = {"L0": dict(sigma=0.0, lam=0.0),
             "L1": dict(sigma=0.1, lam=1e-4)}
INIT_LO, INIT_HI = 75.0, 145.0  # 初始种群 = 真值先验支撑（对基线有利，见文件头）

_PB_CACHE = {}
_SOLVER_CACHE = {}


def get_pb(netkey):
    if netkey not in _PB_CACHE:
        _PB_CACHE[netkey] = cal.Problem(netkey)
    return _PB_CACHE[netkey]


def get_batch_solver(pb, device):
    key = (pb.key, device)
    if key not in _SOLVER_CACHE:
        _SOLVER_CACHE[key] = GGASolver(
            pb.net, device=device, mode="dense",
            inp_path=os.path.join(ROOT, pb.cfg["inp"]))
    return _SOLVER_CACHE[key]


def pick_device(force=None):
    if force:
        return force
    return "cuda" if torch.cuda.is_available() else "cpu"


def lhs(n, d, rng, lo=INIT_LO, hi=INIT_HI):
    u = np.empty((n, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.uniform(size=n)) / n
    return lo + (hi - lo) * u


# ================================================================ 批量目标
class BudgetExhausted(Exception):
    """硬预算闸：dual_annealing 的 maxfun 只在外层迭代间软检查，其 L-BFGS-B
    局部搜索（数值雅可比 2n+1 evals/步）可严重超支（smoke 实测 800 预算被
    冲到 8259）。目标函数入口硬检查并抛此异常，由算法包装层捕获。"""


class BatchObjective:
    """与 G-C1 同一训练损失的批量评估器 + NFE 账本 + best-so-far 轨迹/快照。"""

    def __init__(self, pb, level, noise_seed, device, budget, grid):
        lc = LEVEL_CFG[level]
        self.pb, self.level = pb, level
        self.lam = lc["lam"]
        self.noise_seed = noise_seed
        self.C_true, _ = cal.make_truth(pb, "perpipe", cal.TRUTH_SEED["perpipe"])
        self.scen = cal.make_obs(pb, self.C_true,
                                 ("perpipe", cal.TRUTH_SEED["perpipe"]),
                                 sigma=lc["sigma"], noise_seed=noise_seed)
        self.sensors = cal.get_sensors(pb, "default")
        self.sens_train, self.sens_hold = cal.split_holdout(self.sensors)
        self.d_model = pb.d
        tr = pb.train_frames
        self.F = len(tr)
        self.d_tr, self.rh_tr = pb.d[tr], pb.rh[tr]
        self.obs_tr = self.scen["obs"][np.ix_(tr, self.sens_train)]
        self.free = pb.free_idx
        self.dim = len(self.free)
        self.s2 = get_batch_solver(pb, device)
        dev = self.s2.device
        self.sens_t = torch.as_tensor(self.sens_train, device=dev)
        self.elev_t = torch.as_tensor(pb.elev[self.sens_train], device=dev)
        self.obs_t = torch.as_tensor(self.obs_tr, device=dev)
        Nj = self.s2.Nj
        cap = 6e8 if dev.type == "cuda" else 2e8      # A[B,Nj,Nj] 内存上限
        self.chunk = int(np.clip(cap / (self.F * Nj * Nj * 8), 2, 1024))
        self.mi = BASE_MI_NET[pb.key]
        self.budget = budget
        self.grid = [g for g in grid if g <= budget]
        self.nfe = 0
        self.nbwd = 0
        self.best = float("inf")
        self.best_C = None
        self.traj = []                   # (nfe, best_loss)
        self.snap = {}                   # budget -> dict(nfe, loss, C, wall)
        self.relerr_max = 0.0
        self.t0 = time.perf_counter()

    # -------------------------------------------------- 核心批量损失
    def loss_batch(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.nfe + X.shape[0] > self.budget + max(64, X.shape[0]):
            raise BudgetExhausted(f"nfe={self.nfe} budget={self.budget}")
        Xc = np.clip(X, cal.C_LO, cal.C_HI)
        P = Xc.shape[0]
        mse = np.empty(P)
        for a in range(0, P, self.chunk):
            b = min(a + self.chunk, P)
            pc = b - a
            Rc = np.tile(self.pb.r_base, (pc, 1))
            Rc[:, self.free] = hw_resistance(
                Xc[a:b], self.pb.len_ft[self.free][None, :],
                self.pb.diam_ft[self.free][None, :]).numpy()
            r_big = np.repeat(Rc, self.F, axis=0)
            d_big = np.tile(self.d_tr, (pc, 1))
            rh_big = np.tile(self.rh_tr, (pc, 1))
            rt = torch.as_tensor(r_big, device=self.s2.device)
            old_r, old_rnp = self.s2.r_hw, self.s2.r_np
            try:
                self.s2.r_hw = rt
                with torch.no_grad():
                    out = self.s2.solve(d_big, rh_big, max_iter=self.mi,
                                        accuracy=1e-12)
            finally:
                self.s2.r_hw, self.s2.r_np = old_r, old_rnp
            self.relerr_max = max(self.relerr_max,
                                  float(out["relerr"].max()))
            head = out["head_ft"].reshape(pc, self.F, -1)
            pred = head[:, :, self.sens_t] - self.elev_t
            mse[a:b] = ((pred - self.obs_t) ** 2).mean(dim=(1, 2)).cpu().numpy()
        reg = ((Xc - cal.C0) ** 2).mean(axis=1) / cal.REG_SCALE
        loss = mse + self.lam * reg
        i = int(np.argmin(loss))
        self._account(P, float(loss[i]), Xc[i])
        return loss

    def loss_scalar(self, x):
        return float(self.loss_batch(np.asarray(x)[None, :])[0])

    # -------------------------------------------------- 外部账目（LM 段）
    def note_external(self, d_nfe, d_nbwd, C, loss):
        self.nbwd += d_nbwd
        self._account(d_nfe, float(loss), np.asarray(C, dtype=np.float64))

    def _account(self, d_nfe, cand_loss, cand_C):
        self.nfe += int(d_nfe)
        if cand_loss < self.best:
            self.best = cand_loss
            self.best_C = cand_C.copy()
        self.traj.append((self.nfe, self.best))
        for g in self.grid:
            if g not in self.snap and self.nfe >= g:
                self.snap[g] = dict(nfe=self.nfe, loss=self.best,
                                    C=None if self.best_C is None
                                    else self.best_C.copy(),
                                    wall=time.perf_counter() - self.t0)

    def remaining(self):
        return self.budget - self.nfe


# ================================================================ 基线算法
def _mk_rng_kw(seed):
    """scipy SPEC7 兼容：优先 rng=Generator，旧版回落 seed=。"""
    return dict(rng=np.random.default_rng(seed))


def run_de(obj, cfg, seed, budget=None):
    from scipy.optimize import differential_evolution
    budget = budget if budget is not None else obj.budget
    NP = cfg["NP"]
    rng = np.random.default_rng(seed)
    pop = lhs(NP, obj.dim, rng)
    pop[0] = cal.C0                                   # 注入梯度法初值个体
    maxiter = max(1, budget // NP - 1)
    func = lambda x: obj.loss_batch(x.T)              # scipy vectorized: [dim,S]
    kw = dict(strategy=cfg.get("strategy", "best1bin"),
              mutation=cfg.get("mutation", (0.5, 1.0)),
              recombination=cfg.get("recombination", 0.7),
              init=pop, maxiter=maxiter, tol=0.0, atol=0.0,
              polish=False, vectorized=True, updating="deferred")
    bounds = [(cal.C_LO, cal.C_HI)] * obj.dim
    try:
        differential_evolution(func, bounds, **kw, **_mk_rng_kw(seed))
    except TypeError:
        differential_evolution(func, bounds, **kw, seed=seed)


def run_sa(obj, cfg, seed, budget=None):
    from scipy.optimize import dual_annealing
    budget = budget if budget is not None else obj.budget
    kw = dict(maxfun=budget, maxiter=10 ** 8,
              x0=np.full(obj.dim, cal.C0),
              no_local_search=cfg.get("no_local_search", False))
    for k in ("initial_temp", "visit", "accept", "restart_temp_ratio"):
        if k in cfg:
            kw[k] = cfg[k]
    bounds = list(zip([cal.C_LO] * obj.dim, [cal.C_HI] * obj.dim))
    try:
        dual_annealing(obj.loss_scalar, bounds, **kw, **_mk_rng_kw(seed))
    except TypeError:
        dual_annealing(obj.loss_scalar, bounds, **kw, seed=seed)


def run_pso(obj, cfg, seed, budget=None):
    """全局最优（gbest）惯性权重 PSO。
    参数出处：w=0.7298, c1=c2=1.49618 - Clerc-Kennedy 收缩因子
    χ=0.7298, χ·φ_i=1.49618 的等价惯性形式（Eberhart & Shi 2000）。"""
    budget = budget if budget is not None else obj.budget
    NP = cfg["NP"]
    w, c1, c2 = cfg.get("w", 0.7298), cfg.get("c1", 1.49618), cfg.get("c2", 1.49618)
    rng = np.random.default_rng(seed)
    lo, hi = cal.C_LO, cal.C_HI
    X = lhs(NP, obj.dim, rng)
    X[0] = cal.C0
    V = np.zeros_like(X)
    vmax = 0.5 * (hi - lo)
    f = obj.loss_batch(X)
    pbest_x, pbest_f = X.copy(), f.copy()
    g = int(np.argmin(f))
    gbest_x, gbest_f = X[g].copy(), f[g]
    while obj.nfe + NP <= budget:
        r1 = rng.uniform(size=X.shape)
        r2 = rng.uniform(size=X.shape)
        V = w * V + c1 * r1 * (pbest_x - X) + c2 * r2 * (gbest_x - X)
        V = np.clip(V, -vmax, vmax)
        X = X + V
        outb = (X < lo) | (X > hi)
        X = np.clip(X, lo, hi)
        V[outb] = 0.0
        f = obj.loss_batch(X)
        imp = f < pbest_f
        pbest_x[imp], pbest_f[imp] = X[imp], f[imp]
        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest_x, gbest_f = pbest_x[g].copy(), pbest_f[g]


def run_cma(obj, cfg, seed, budget=None):
    import cma
    budget = budget if budget is not None else obj.budget
    opts = {"bounds": [cal.C_LO, cal.C_HI], "seed": int(seed),
            "verbose": -9, "maxfevals": budget}
    if cfg.get("popsize_mult"):
        base = 4 + int(3 * np.log(obj.dim))
        opts["popsize"] = int(cfg["popsize_mult"] * base)
    es = cma.CMAEvolutionStrategy(np.full(obj.dim, cal.C0),
                                  cfg.get("sigma0", 36.0), opts)
    while not es.stop() and obj.nfe + es.popsize <= budget:
        xs = es.ask()
        L = obj.loss_batch(np.asarray(xs))
        es.tell(xs, list(L))


def run_hybrid(obj, cfg, seed, budget=None):
    """DE 粗搜（de_frac 预算）→ calibrate._lm_polish 精修（余下预算）。
    LM 计数口径与梯度法一致：每次批前向 nfe+=1（含 sensitivity 内部 solve
    与试探步），每次伴随雅可比 nbwd+=1（成本≈前向 1%，另列）。分段调用
    _lm_polish（每段 ≤5 次接受迭代）以记录轨迹；每段重置 LM 阻尼 mu=1e-6。"""
    budget = budget if budget is not None else obj.budget
    b_de = int(cfg.get("de_frac", 0.6) * budget)
    run_de(obj, cfg, seed, budget=b_de)
    pb = obj.pb
    C = obj.best_C.copy()
    state = {"prev": None}
    stall = 0
    while obj.remaining() > 3 and stall < 2:
        cnt = dict(nfe=0, nbwd=0, tcv_flips=0)
        seg = int(min(5, max(1, obj.remaining() // 3)))
        C, info = cal._lm_polish(pb, C, obj.scen["obs"], obj.sens_train,
                                 obj.d_model, obj.lam, seg, cnt, state)
        loss = info["mse"] + obj.lam * info["reg"]
        obj.note_external(cnt["nfe"], cnt["nbwd"], C, loss)
        stall = stall + 1 if info["n_iter"] == 0 else 0


ALGOS = dict(de=run_de, sa=run_sa, pso=run_pso, cma=run_cma, hybrid=run_hybrid)

# 超参配置：cfg0 = 默认，cfg1..3 = 备选（调参协议见文件头第 5 条）
HYPER = {
    "de": [dict(name="de0_NP64", NP=64),
           dict(name="de1_NP32", NP=32),
           dict(name="de2_NP128", NP=128),
           dict(name="de3_NP64_rand1bin_m0.7_cr0.9", NP=64,
                strategy="rand1bin", mutation=0.7, recombination=0.9)],
    "sa": [dict(name="sa0_default"),
           dict(name="sa1_nolocal", no_local_search=True),
           dict(name="sa2_T5e4_v2.9", initial_temp=5e4, visit=2.9),
           dict(name="sa3_T500_v2.2", initial_temp=500.0, visit=2.2)],
    "pso": [dict(name="pso0_NP64", NP=64),
            dict(name="pso1_NP32", NP=32),
            dict(name="pso2_NP128", NP=128),
            dict(name="pso3_NP64_w0.6_c1.7", NP=64, w=0.6, c1=1.7, c2=1.7)],
    "cma": [dict(name="cma0_auto_s36", sigma0=36.0),
            dict(name="cma1_auto_s18", sigma0=18.0),
            dict(name="cma2_pop2x_s36", sigma0=36.0, popsize_mult=2),
            dict(name="cma3_pop4x_s36", sigma0=36.0, popsize_mult=4)],
    "hybrid": [dict(name="hy0_NP64_f0.6", NP=64, de_frac=0.6),
               dict(name="hy1_NP32_f0.6", NP=32, de_frac=0.6),
               dict(name="hy2_NP64_f0.4", NP=64, de_frac=0.4),
               dict(name="hy3_NP32_f0.3", NP=32, de_frac=0.3)],
}


# ================================================================ 落盘
# 并行车道（GPU 批量车道 / CPU 串行 SA 车道）各写自己的分片文件，最后 merge
# 汇入 OUT_JSON - 避免两个进程对同一 JSON 的读-改-写竞态。
_OUTFILE = {"path": OUT_JSON}


def set_outfile(p):
    _OUTFILE["path"] = p if os.path.isabs(p) else os.path.join(ROOT, "data", p)


def load_out():
    fp = _OUTFILE["path"]
    if os.path.exists(fp):
        return json.load(open(fp, encoding="utf-8"))
    return dict(meta={}, tuning={}, runs={}, gradient_runs={})


def save_out(data):
    fp = _OUTFILE["path"]
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, fp)


def downsample_traj(traj, n=120):
    if len(traj) <= n:
        return [[int(a), float(b)] for a, b in traj]
    idx = np.unique(np.geomspace(1, len(traj), n).astype(int) - 1)
    return [[int(traj[i][0]), float(traj[i][1])] for i in idx]


# ================================================================ 单次运行
def one_run(netkey, level, algo, cfg, noise_seed, opt_seed, budget,
            device, do_eval=True, tag="eval"):
    pb = get_pb(netkey)
    obj = BatchObjective(pb, level, noise_seed, device, budget, GRID[netkey])
    t0 = time.perf_counter()
    try:
        ALGOS[algo](obj, cfg, opt_seed)
    except BudgetExhausted:
        pass                                  # 硬预算闸触发（见类文档）
    wall = time.perf_counter() - t0
    # 终点也强制快照（预算未用尽时 best-so-far 顺延到更大预算点）
    for g in obj.grid:
        if g not in obj.snap:
            obj.snap[g] = dict(nfe=obj.nfe, loss=obj.best,
                               C=obj.best_C.copy(), wall=wall)
    snaps = {}
    if do_eval:
        reason, Vr, k_sub, rank_eps = cal.get_diag(pb, "default", obj.sensors)
        for g, sn in sorted(obj.snap.items()):
            ev = cal.evaluate(pb, obj.C_true, sn["C"], reason, Vr, k_sub,
                              rank_eps, obj.scen, obj.sens_train,
                              obj.sens_hold, obj.d_model)
            snaps[str(g)] = dict(nfe_at=sn["nfe"], loss=sn["loss"],
                                 wall_sec=sn["wall"], **ev)
    else:
        for g, sn in sorted(obj.snap.items()):
            snaps[str(g)] = dict(nfe_at=sn["nfe"], loss=sn["loss"],
                                 wall_sec=sn["wall"])
    rec = dict(net=netkey, level=level, algo=algo, config=cfg,
               noise_seed=noise_seed, opt_seed=opt_seed, budget=budget,
               nfe_total=obj.nfe, nbwd_total=obj.nbwd,
               best_loss=obj.best, wall_sec=wall,
               relerr_max=obj.relerr_max, device=str(obj.s2.device),
               chunk=obj.chunk, base_mi=obj.mi,
               traj=downsample_traj(obj.traj), snapshots=snaps, tag=tag)
    if do_eval and obj.best_C is not None:
        rec["C_hat_free"] = [round(float(x), 3) for x in obj.best_C]
    fin = snaps.get(str(obj.grid[-1]), {}) if obj.grid else {}
    print(f"[{netkey}:{level}:{algo}:{cfg['name']}:n{noise_seed}:o{opt_seed}] "
          f"NFE={obj.nfe} loss={obj.best:.4e} "
          f"sub={fin.get('sub_rmse', float('nan')):.3f} "
          f"info={fin.get('info_rmse', float('nan')):.3f} "
          f"valF={fin.get('val_frame_rmse', float('nan')):.4f} "
          f"{wall:.0f}s", flush=True)
    return rec


def run_key(netkey, level, algo, cfgname, noise_seed, opt_seed, budget, tag):
    return f"{tag}|{netkey}|{level}|{algo}|{cfgname}|n{noise_seed}|o{opt_seed}|B{budget}"


# ================================================================ stages
def stage_arm(device):
    """臂长测试 + 目标一致性验证（批量 dense-60 vs G-C1 前向 20+4）。"""
    for netkey in ("hanoi", "city_d"):
        pb = get_pb(netkey)
        obj = BatchObjective(pb, "L1", 100, device, 10 ** 9, [])
        print(f"== {netkey}: dim={obj.dim} Nj={obj.s2.Nj} 传感器={len(obj.sens_train)}"
              f"+{len(obj.sens_hold)}留出 chunk={obj.chunk} dev={obj.s2.device}")
        rng = np.random.default_rng(1)
        for P in (1, 16, 64, 128):
            X = lhs(P, obj.dim, rng)
            t0 = time.perf_counter()
            L = obj.loss_batch(X)
            dt = time.perf_counter() - t0
            print(f"   P={P:4d} t={dt:7.3f}s ({dt / P * 1000:7.2f} ms/NFE) "
                  f"loss中位={np.median(L):.4e} relerr_max={obj.relerr_max:.2e}",
                  flush=True)
        # 一致性：同一 C 用 G-C1 前向（solve_polished 20+4）算损失
        X = lhs(4, obj.dim, np.random.default_rng(2))
        L_batch = obj.loss_batch(X)
        tr = pb.train_frames
        for i in range(4):
            Cf = np.full(pb.s.L, cal.C0)
            Cf[obj.free] = X[i]
            sol = cal.solve_polished(pb.s, pb.d[tr], pb.rh[tr],
                                     r_hw=pb.r_of_C(Cf), accuracy=1e-12,
                                     max_iter=cal.GGA_MI,
                                     polish_steps=cal.POLISH)
            res = (sol["head"][:, obj.sens_train] - pb.elev[None, obj.sens_train]
                   - obj.obs_tr)
            mse = float(np.mean(res ** 2))
            loss_ref = mse + obj.lam * float(
                np.mean((X[i] - cal.C0) ** 2)) / cal.REG_SCALE
            print(f"   一致性[{i}] batch={L_batch[i]:.10e} "
                  f"gc1={loss_ref:.10e} 相对差="
                  f"{abs(L_batch[i] - loss_ref) / loss_ref:.2e}", flush=True)


def stage_tune(netkey, algos, budget, device, sa_budget=None):
    """调参：L1 档 × 3 专用种子 × 4 配置，按中位最终训练损失选优。"""
    data = load_out()
    tun = data["tuning"].setdefault(netkey, {})
    for algo in algos:
        b = sa_budget if (algo == "sa" and sa_budget) else budget
        entry = tun.setdefault(algo, dict(budget=b, seeds=TUNE_SEEDS,
                                          results={}, nfe_bill=0))
        for cfg in HYPER[algo]:
            res = entry["results"].setdefault(cfg["name"],
                                              dict(cfg=cfg, losses={}))
            for sd in TUNE_SEEDS:
                if str(sd) in res["losses"]:
                    continue
                rec = one_run(netkey, "L1", algo, cfg, sd, 10000 + sd, b,
                              device, do_eval=False, tag="tune")
                res["losses"][str(sd)] = rec["best_loss"]
                entry["nfe_bill"] += rec["nfe_total"]
                data["runs"][run_key(netkey, "L1", algo, cfg["name"], sd,
                                     10000 + sd, b, "tune")] = rec
                save_out(data)
        med = {n: float(np.median(list(r["losses"].values())))
               for n, r in entry["results"].items()}
        entry["median_loss"] = med
        entry["chosen"] = min(med, key=med.get)
        save_out(data)
        print(f"[tune {netkey}:{algo}] 中位损失={ {k: f'{v:.3e}' for k, v in med.items()} } "
              f"选中={entry['chosen']} 调参 NFE 账单={entry['nfe_bill']}",
              flush=True)


def chosen_cfg(data, netkey, algo):
    name = data["tuning"][netkey][algo]["chosen"]
    return next(c for c in HYPER[algo] if c["name"] == name)


def stage_evalrun(netkey, level, algos, seeds, budget, device, tag="eval"):
    data = load_out()
    for algo in algos:
        cfg = chosen_cfg(data, netkey, algo)
        for sd in seeds:
            if level == "L0":                 # L0 无噪声：种子只喂优化器
                nz, opt = 0, sd
            else:
                nz, opt = sd, 1000 + sd
            k = run_key(netkey, level, algo, cfg["name"], nz, opt, budget,
                        tag)
            if k in data["runs"]:
                continue
            rec = one_run(netkey, level, algo, cfg, nz, opt, budget, device,
                          do_eval=True, tag=tag)
            data["runs"][k] = rec
            save_out(data)


def stage_merge(parts):
    """把分片文件汇入主 OUT_JSON（runs/gradient_runs/tuning 逐键并集）。"""
    set_outfile(OUT_JSON)
    data = load_out()
    for p in parts:
        fp = p if os.path.isabs(p) else os.path.join(ROOT, "data", p)
        if not os.path.exists(fp):
            print(f"  跳过缺失分片 {fp}")
            continue
        part = json.load(open(fp, encoding="utf-8"))
        data["runs"].update(part.get("runs", {}))
        data["gradient_runs"].update(part.get("gradient_runs", {}))
        for nk, algs in part.get("tuning", {}).items():
            data["tuning"].setdefault(nk, {}).update(algs)
        print(f"  已并入 {os.path.basename(fp)}: runs+{len(part.get('runs', {}))}")
    save_out(data)
    print(f"合并完成：runs={len(data['runs'])} "
          f"gradient_runs={len(data['gradient_runs'])}")


def stage_gdextra(netkey, level, seeds):
    """梯度法补种子：G-C1 固化配置原样跑（无任何新调参），落盘 gradient_runs。"""
    data = load_out()
    pb = get_pb(netkey)
    lc = LEVEL_CFG[level]
    for sd in seeds:
        k = f"{netkey}|{level}|n{sd}"
        if k in data["gradient_runs"]:
            continue
        rec = cal.run_config(pb, f"GDX_{level}_n{sd}", mode="perpipe",
                             sigma=lc["sigma"], noise_seed=sd, lam=lc["lam"],
                             note="G-D 配对补种子（G-C1 固化配置）")
        rec.pop("C_hat_free", None)
        data["gradient_runs"][k] = rec
        save_out(data)


def stage_meta(note, device):
    import scipy
    import cma
    data = load_out()
    data["meta"] = dict(
        date=time.strftime("%Y-%m-%d %H:%M"),
        scipy=scipy.__version__, cma=cma.__version__,
        torch=torch.__version__, cuda=torch.cuda.is_available(),
        device=device, base_mi=BASE_MI_NET,
        init_prior=[INIT_LO, INIT_HI], tune_seeds=TUNE_SEEDS,
        grid=GRID, note=note,
        declarations=[
            "1 NFE = 一次 20 训练帧批量前向；种群逐个体计数，初始化计入",
            "基线前向 hanoi max_iter=60（提前收敛判停）；city_d max_iter=30"
            "（relerr 平台 ~3e-7 与 mi=60 相同，损失与 G-C1 前向相对差 ~1e-7；"
            "臂长实测 267 vs 562 ms/NFE 后的省时决策） - 同记 1 NFE",
            "硬预算闸：目标函数入口检查 nfe≤budget+max(64,batch)，"
            "dual_annealing 局部搜索超支被截断（超出部分如实计入 nfe_total）",
            "决策空间 = G-C1 同一自由管集合 + 箱 [40,160]，基线直接搜 C（无 sigmoid）",
            "初始种群 LHS 覆盖真值先验 [75,145] 并注入 C0=130 个体（对基线有利）",
            "预算网格快照 = 首次跨越预算点的批次末（基线可多用 ≤pop−1 次评估，宽松方向）",
            "调参：每基线 4 配置 × 3 专用种子（900..902）按中位训练损失选优，"
            "NFE 账单单独列；梯度法沿用 G-C1 固化配置，无新调参",
            "混合法 LM 段 nbwd（伴随雅可比）另计，成本≈前向 1%（与梯度法口径一致）",
            "梯度法补种子 = G-C1 固化配置原样重跑（无调参），保证 30/10 种子配对",
        ])
    save_out(data)
    print("meta 已写入", flush=True)


def stage_status():
    data = load_out()
    from collections import Counter
    c = Counter()
    for k, r in data["runs"].items():
        c[(r["tag"], r["net"], r["level"], r["algo"])] += 1
    for k in sorted(c):
        print(k, c[k])
    print("gradient_runs:", len(data["gradient_runs"]))
    for nk, t in data.get("tuning", {}).items():
        for a, e in t.items():
            print(f"tuning {nk}:{a} chosen={e.get('chosen')} "
                  f"bill={e.get('nfe_bill')}")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--net", default="hanoi")
    ap.add_argument("--level", default="L1")
    ap.add_argument("--algos", default="de,sa,pso,cma,hybrid")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--sa-budget", type=int, default=0)
    ap.add_argument("--device", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--outfile", default="")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--parts", default="")
    a = ap.parse_args()
    if a.outfile:
        set_outfile(a.outfile)
    dev = pick_device(a.device or None)
    algos = [x for x in a.algos.split(",") if x]
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    t0 = time.perf_counter()
    if a.stage == "arm":
        stage_arm(dev)
    elif a.stage == "tune":
        stage_tune(a.net, algos, a.budget, dev, sa_budget=a.sa_budget or None)
    elif a.stage == "evalrun":
        stage_evalrun(a.net, a.level, algos, seeds, a.budget, dev, tag=a.tag)
    elif a.stage == "gdextra":
        stage_gdextra(a.net, a.level, seeds)
    elif a.stage == "merge":
        stage_merge([p for p in a.parts.split(",") if p])
    elif a.stage == "meta":
        stage_meta(a.note, dev)
    elif a.stage == "status":
        stage_status()
    else:
        raise SystemExit(f"未知 stage: {a.stage}")
    print(f"[stage {a.stage}] 总墙钟 {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

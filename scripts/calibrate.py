# -*- coding: utf-8 -*-
"""calibrate.py - G-C1 摩阻系数校核引擎 + L0/L1/L2/桥接实验编排。

引擎（本文件一节）：
  make_truth   逐管（perpipe，C~U[75,145]）/ 按管径档分组（grouped，组值同分布
               + 组内 ±3 抖动）两种真值。
  make_obs     真值正演 25 帧（accuracy=1e-12, max_iter=60, polish=3），全 junction
               压力 + 噪声场（iid 高斯 σ / 每传感器常值偏置 σ_b，种子固定；噪声按
               节点生成，换布点时同一种子给出同一实现 - 桥接实验的可比性所在）。
  calibrate    RoughnessParam（sigmoid 箱约束 [40,160]，自由集 = 非结构性不可辨识
               管，其余冻结在 C0=130）；损失 = 训练帧×训练传感器压力 MSE
               + λ·mean((C−C0)²)/30²（30 ≈ 真值先验半宽，λ 因此无量纲）；
               Adam(lr=0.3, 60%/85% 处减半) 粗搜 → L-BFGS(strong_wolfe) 精修。
               记录 NFE（每次 = 训练帧批量前向）、反向次数、墙钟、TCV 流向翻转数
               （dense 模式 TCV 无状态机，状态恒定；此处监控的是优化轨迹中开启
               TCV 的流向符号翻转，作为状态稳定性的代理）、收敛轨迹。
  evaluate     C 空间：informative / unobservable（按当前布点的 identifiability
               归因）分开报 MAE/RMSE/max|ΔC|；冻结管误差如实另报。
               压力空间：留出验证 - 训练 20 帧 / 验证 5 帧（固定 [2,7,12,17,22]），
               另留 20% 传感器只验证不训练；报训练/帧留出/传感器留出三套 RMSE
               （对含噪观测；对干净真值的同名指标以 *_clean 一并落盘）。
  多起点        8 个 LHS 初值（C∈[80,145] → theta），中位数/最好/最差。

实验矩阵（data/calib_gc1_<stem>.json）：
  L0  无噪声 sanity（city_d D-opt40 + Hanoi dopt10 + Modena random40；
      perpipe/grouped）。注意：这是 inverse crime 设定（观测与反演用同一位级
      求解器），只证明优化器收敛，不证明方法实用。
  L1  量测噪声（city_d 为主）：σ∈{0.03,0.1,0.3} ft ×5 种子（λ=1e-4）；
      λ∈{0,1e-2} 补扫（σ=0.1）；带偏置档 σ=0.1+σ_b=0.05。Hanoi 快速版。
  L2  需水不确定性（city_d）：真值正演需水乘 U[0.85,1.15] 逐节点乘子（反演不知），
      σ=0.1；对照组 = 需水完全已知（同一观测）。
  桥接 同一真值+噪声种子只换布点：dopt40 / random40×5 / spectral50 / degree40。

帧数与步数预算：city_d 20 训练帧批前向 ≈1.2 s（GGA max_iter=20 + Newton 抛光 4 步，
与 demo_leak_inversion.py 同款、已验证与 60 迭代结果一致）；Adam 步数按预算取
ADAM_STEPS（<300，如实报告）。

运行：& python -X utf8 scripts/calibrate.py --stage <stage> [选项]
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

from dgga.parse import Net                                    # noqa: E402
from dgga.solver import GGASolver                             # noqa: E402
from dgga.autodiff import ImplicitGGASolve, solve_polished    # noqa: E402
from dgga.calib import (RoughnessParam, hw_resistance,        # noqa: E402
                        identifiability, clamped_mask, dead_branch_mask)

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------- 常量
GGA_MI, POLISH = 20, 4          # 反演前向（demo_leak_inversion.py 同款）
OBS_MI, OBS_POLISH = 60, 3      # 观测生成 / 评价正演（保守设置）
C_LO, C_HI, C0 = 40.0, 160.0, 130.0
REG_SCALE = 30.0 ** 2           # λ 无量纲化：mean((C-C0)/30)²，30≈先验半宽
VAL_FRAMES = [2, 7, 12, 17, 22]  # 固定验证帧
TRUTH_SEED = {"perpipe": 7, "grouped": 8}
HOLDOUT_SEED = 505              # 20% 传感器留出
NOISE_SEED0 = 100               # 噪声种子 100..104
DEM_SEED0 = 300                 # L2 需水乘子种子 300..304
LHS_SEED = 606                  # 多起点 LHS
SYNTH_SEED = 4242               # Hanoi/Modena 合成多工况帧
SUB_TOL = 1e-2                  # 良态子空间奇异值截断（σ_i > SUB_TOL·σ_1）

NETS = {
    "city_d": dict(stem="city_d", inp="networks/realInpData/city_d.inp",
                   adam=80, lbfgs=20, rounds=2, lm=15),
    "hanoi": dict(stem="pub_hanoi", inp="networks/public/Hanoi.inp",
                  adam=300, lbfgs=50, rounds=4, lm=40),
    "modena": dict(stem="pub_modena", inp="networks/public/Modena.inp",
                   adam=200, lbfgs=40, rounds=3, lm=25),
}

SCRATCH = os.path.join(os.environ.get("TEMP", os.path.join(ROOT, "data")),
                       "claude_calib_gc1_cache")
os.makedirs(SCRATCH, exist_ok=True)


# ================================================================ 场景容器
class Problem:
    """网络 + 帧 + 结构性可辨识性（进程内构造一次，跨 run 复用）。"""

    def __init__(self, netkey):
        cfg = NETS[netkey]
        self.key, self.cfg = netkey, cfg
        self.net = Net.load(os.path.join(ROOT, "data", "reference"), cfg["stem"])
        # 陷阱：GGASolver(inp_path=...) 就地修正 net.dem_base_cfs → 先构造 solver
        self.s = GGASolver(self.net, mode="dense",
                           inp_path=os.path.join(ROOT, cfg["inp"]))
        N = self.net.N
        if netkey == "city_d":                    # 真实 24h 模式 ×25 帧
            T = [t * 3600 for t in range(25)]
            self.d = np.stack([self.net.demand_cfs_at(t) for t in T])
            self.rh = np.stack([np.nan_to_num(self.net.reservoir_head_ft_at(t))
                                for t in T])
            self.frames_note = "pattern_24h_25f"
        else:                                     # 无模式公开网：合成多工况激励
            d0 = self.net.demand_cfs_at(0)        # （反演侧完全已知，见 make_obs）
            rh0 = np.nan_to_num(self.net.reservoir_head_ft_at(0))
            rng = np.random.default_rng(SYNTH_SEED)
            phi = rng.uniform(0.0, 2 * np.pi, N)
            amp = rng.uniform(0.10, 0.30, N)
            t = np.arange(25)[:, None]
            mult = 1.0 + amp[None, :] * np.sin(2 * np.pi * t / 24.0 + phi[None, :])
            self.d = d0[None, :] * mult
            self.rh = np.tile(rh0, (25, 1))
            self.frames_note = "synthetic_sine_25f(seed=%d)" % SYNTH_SEED
        self.elev = np.asarray(self.net.elev_ft, dtype=np.float64)
        self.len_ft = np.asarray(self.net.len_ft, dtype=np.float64)
        self.diam_ft = np.asarray(self.net.diam_ft, dtype=np.float64)
        self.pidx = np.where(np.isin(self.s.lt_np, (0, 1))
                             & (self.s.kc_np > 0.0))[0]
        self.r_base = self.s.r_hw.detach().cpu().numpy().copy()
        # ---- 结构性不可辨识（与布点无关）：死支 | 全帧钳位 | 关闭 ----
        sol0 = solve_polished(self.s, self.d, self.rh, accuracy=1e-12,
                              max_iter=OBS_MI, polish_steps=OBS_POLISH)
        # 结构性冻结判据用 margin=1.0（hydcoeffs.c:554 求解器实际钳位分支；
        # margin=10 是保守诊断余量，city_d 会多冻 4 根仍有弱灵敏度的管，
        # 与 G-B 的 43 根地板（41 死支+2 全帧严格钳位）不符）
        cm = clamped_mask(self.s, sol0, margin=1.0)      # mask=全帧皆严格钳位
        dead = dead_branch_mask(self.s, demand=self.d)
        pipe = np.zeros(self.s.L, dtype=bool)
        pipe[self.pidx] = True
        self.dead_mask = dead & pipe
        self.clamp_all_mask = cm["mask"] & pipe & ~dead
        self.struct_mask = pipe & (dead | cm["mask"] | self.s.closed_np)
        self.free_idx = self.pidx[~self.struct_mask[self.pidx]]
        self.tcv_open = self.s.is_tcv_np & ~self.s.closed_np
        tr = [f for f in range(25) if f not in VAL_FRAMES]
        self.train_frames = np.asarray(tr)
        self.val_frames = np.asarray(VAL_FRAMES)
        self._obs_cache = {}
        self._diag_cache = {}
        self._truth_cache = {}

    def r_of_C(self, C_full):
        """[L] C 数组 → 内部阻力（管道位现算，其余透传 solver 基底）。"""
        r = self.r_base.copy()
        r[self.pidx] = hw_resistance(C_full[self.pidx], self.len_ft[self.pidx],
                                     self.diam_ft[self.pidx]).numpy()
        return r


# ================================================================ 布点
def get_sensors(pb, placement="default"):
    junc = np.asarray(pb.s.junc_nodes)
    if pb.key == "city_d":
        z = np.load(os.path.join(ROOT, "data", "placement_orders_city_d.npz"))
        if placement in ("default", "dopt40"):
            cand = z["dopt"][:40]
        elif placement.startswith("dopt"):       # doptN 通用（验证/剂量响应用）
            cand = z["dopt"][:int(placement[4:])]
        elif placement == "spectral50":
            cand = z["spectral"][:50]
        elif placement == "degree40":
            cand = z["degree"][:40]
        elif placement.startswith("random40_s"):
            cand = z["random"][int(placement.split("_s")[1])][:40]
        # ---- 增设模式（place_sensors.py --stage augment 的输出；新增分支，
        # 缺省 default/dopt40 行为不变）：ga40 = 普查用的 40 个"现有"传感器 S0
        # （seed=2026，149 根传感不足管的口径）；augdopt<k>/augcover<k> =
        # S0 ∪ 增设 k 个（D-最优 / 覆盖目标）。全程虚拟增设（模拟，非实装）。
        elif placement == "ga40":
            cand = z["augment_fixed"]
        elif placement.startswith("augdopt"):
            k = int(placement[7:])
            cand = np.r_[z["augment_fixed"], z["augment_dopt"][:k]]
        elif placement.startswith("augcover"):
            k = int(placement[8:])
            cand = np.r_[z["augment_fixed"], z["augment_cover"][:k]]
        else:
            raise ValueError(placement)
        return np.sort(junc[np.unique(cand)])
    if pb.key == "hanoi":
        if placement.startswith("aug"):          # 增设实验（scripts/augment_public.py）：
            # augS0 = 10 个"现有"传感器；aug<obj><k> = S0 ∪ 增设前 k 个（obj=dopt/cover）
            z = np.load(os.path.join(ROOT, "data",
                                     "placement_orders_pub_hanoi_synth25.npz"))
            pos = z["augment_fixed"]
            if placement != "augS0":
                import re
                m = re.fullmatch(r"aug(dopt|cover)(\d+)", placement)
                if not m:
                    raise ValueError(placement)
                pos = np.r_[pos, z["augment_" + m.group(1)][:int(m.group(2))]]
            return np.sort(junc[pos])
        return np.sort(junc)                     # 全 junction（31；D-opt10 布点
                                                 # cond(S)=2.7e6，全测点 1.2e3）
    rng = np.random.default_rng(2026)            # modena：固定随机 80
    return np.sort(rng.choice(junc, 80, replace=False))


def split_holdout(sensors):
    """20% 传感器只验证不训练（固定种子）。"""
    m = len(sensors)
    n_hold = max(2, int(round(0.2 * m)))
    perm = np.random.default_rng(HOLDOUT_SEED).permutation(m)
    hold = np.sort(perm[:n_hold])
    train = np.sort(perm[n_hold:])
    return sensors[train], sensors[hold]


def get_diag(pb, placement, sensors):
    """当前布点下的 identifiability 归因 + 可辨识子空间（scratchpad npz 缓存）。

    返回 (reason[L], Vr[n_free,k], k, rank_eps)：Vr = 自由集灵敏度矩阵 S_free 的
    良态右奇异向量（σ_i > SUB_TOL·σ_1，city_d dopt40 下 k=41；eps 数值秩 122
    含大量 σ~1e-13·σ1 的名义方向）。‖Vrᵀ·ΔC‖/√k 是"数据可实际约束方向上"的
    C 误差 - city_d 自由集 432 而良态维数 ~41，裸 C-RMSE 被零空间位移淹没，
    噪声退化曲线只能在该子空间里看（实测 L0 无噪声：sub 先验 30.4 → 13.1，
    tol 收到 1e-1 时 49.7 → 9.5；eps 秩全投影则完全被弱方向噪声淹没）。"""
    from dgga.sensitivity import svd_spectrum
    key = f"{pb.cfg['stem']}_{placement}"
    if key in pb._diag_cache:
        return pb._diag_cache[key]
    fp = os.path.join(SCRATCH, f"diag2_{key}.npz")
    if os.path.exists(fp):
        z = np.load(fp, allow_pickle=False)
        if np.array_equal(z["sensors"], sensors):
            out = (z["reason"].astype(object), z["Vr"], int(z["k"]),
                   int(z["rank_eps"]))
            pb._diag_cache[key] = out
            return out
    diag = identifiability(pb.s, pb.d, pb.rh, sensors, ke=np.zeros(pb.net.N),
                           wrt="C", max_iter=OBS_MI, polish_steps=OBS_POLISH,
                           return_S=True)
    reason = diag["reason"]
    sp = svd_spectrum(diag["S"][:, pb.free_idx], full=True)
    k = int(np.sum(sp["sv"] > SUB_TOL * sp["sv"][0]))
    Vr = sp["Vt"][:k].T.copy()
    out = (reason, Vr, k, int(sp["rank"]))
    np.savez(fp, reason=reason.astype("U12"), sensors=sensors, Vr=Vr, k=k,
             rank_eps=sp["rank"])
    pb._diag_cache[key] = out
    return out


# ================================================================ 真值 / 观测
def make_truth(pb, mode, seed):
    """[L] C_true（管道位赋值，其余占位 C0）。grouped 按管径档（city_d 16 档）。"""
    ck = (mode, seed)
    if ck in pb._truth_cache:
        return pb._truth_cache[ck]
    rng = np.random.default_rng(seed)
    C = np.full(pb.s.L, C0, dtype=np.float64)
    if mode == "perpipe":
        C[pb.pidx] = rng.uniform(75.0, 145.0, len(pb.pidx))
        out = (C, dict(mode=mode, seed=seed))
    else:
        dkey = np.round(pb.diam_ft * 12.0, 3)
        gvals = {k: rng.uniform(75.0, 145.0) for k in np.unique(dkey[pb.pidx])}
        base = np.array([gvals[dkey[i]] for i in pb.pidx])
        C[pb.pidx] = np.clip(base + rng.uniform(-3.0, 3.0, len(pb.pidx)),
                             75.0, 145.0)
        out = (C, dict(mode=mode, seed=seed, n_groups=len(gvals)))
    pb._truth_cache[ck] = out
    return out


def make_obs(pb, C_true, tkey, sigma=0.0, sigma_b=0.0, noise_seed=0,
             dem_seed=None):
    """真值正演 + 噪声场（按节点生成 → 桥接实验换布点时噪声实现一致）。"""
    key = (tkey, sigma, sigma_b, noise_seed, dem_seed)
    if key in pb._obs_cache:
        return pb._obs_cache[key]
    mult = None
    if dem_seed is not None:
        mult = np.random.default_rng(dem_seed).uniform(0.85, 1.15, pb.net.N)
    d_used = pb.d if mult is None else pb.d * mult[None, :]
    sol = solve_polished(pb.s, d_used, pb.rh, r_hw=pb.r_of_C(C_true),
                         accuracy=1e-12, max_iter=OBS_MI,
                         polish_steps=OBS_POLISH)
    p = sol["head"] - pb.elev[None, :]                     # [25, N] 压力 ft
    rng = np.random.default_rng(noise_seed)
    noise = np.zeros_like(p)
    if sigma > 0:
        noise += sigma * rng.standard_normal(p.shape)
    if sigma_b > 0:
        noise += (sigma_b * rng.standard_normal(p.shape[1]))[None, :]
    out = dict(p_clean=p, obs=p + noise, d_used=d_used,
               resid_inf=float(sol["resid_inf"].max()),
               n_obs_solve=1)
    pb._obs_cache[key] = out
    return out


# ================================================================ 校核
def _lm_polish(pb, C_free, obs, sens_train, d_model, lam, max_iter, cnt,
               state, rel_tol=1e-4, mu0=1e-6):
    """Levenberg–Marquardt 精抛光（C 空间，精确伴随雅可比 + 阻尼正规方程）。

    为什么需要：Adam/L-BFGS 在 theta(sigmoid) 空间是纯一阶法，对本问题的
    最小二乘结构尾部收敛极慢 - 独立验证（G-C1 复核）实测 Hanoi L0 满秩
    inverse crime 下一阶法停在 info C-RMSE≈5.5（train p-RMSE 3.4e-2 ft），
    而 LM 30 步即到 C-RMSE 2e-4；city_d L0 train p-RMSE 3.6e-2 → <2e-3 ft。
    未加此阶段时"L0 地板 13 由条件数决定"的说法把优化器欠收敛错算进了
    可辨识性账里。

    目标函数与 torch 侧完全一致：mean(res²) + λ·mean((C−C0)²)/REG_SCALE。
    正规方程 (SᵀS/n + λ/(f·REG_SCALE)·I + μI)·dC = −(Sᵀres/n
    + λ/(f·REG_SCALE)·(C−C0))；步长裁剪 ±20，箱约束以裁剪投影维持
    （[C_LO+1e-3, C_HI−1e-3]，仅抛光阶段，diagnosability 由前段 sigmoid 保证）。
    计数口径：每次批前向 nfe+=1（含 sensitivity 内部 solve 与试探步），
    每次伴随雅可比 nbwd+=1；TCV 流向符号沿用 state 连续监控。
    """
    from dgga.sensitivity import sensitivity_matrix
    s = pb.s
    tr = pb.train_frames
    free = pb.free_idx
    f = len(free)
    n_ = len(tr) * len(sens_train)
    w_reg = lam / (REG_SCALE * f) if lam > 0 else 0.0
    obs_tr = obs[np.ix_(tr, sens_train)]

    def r_of_free(Cf):
        r = pb.r_base.copy()
        r[free] = hw_resistance(Cf, pb.len_ft[free], pb.diam_ft[free]).numpy()
        return r

    def fwd(Cf):
        cnt["nfe"] += 1
        sol = solve_polished(s, d_model[tr], pb.rh[tr], r_hw=r_of_free(Cf),
                             accuracy=1e-12, max_iter=GGA_MI,
                             polish_steps=POLISH)
        if pb.tcv_open.any():
            sg = np.sign(sol["q"][:, pb.tcv_open])
            if state["prev"] is not None:
                cnt["tcv_flips"] += int((sg != state["prev"]).sum())
            state["prev"] = sg
        res = (sol["head"][:, sens_train] - pb.elev[None, sens_train]
               - obs_tr).ravel()
        mse = float(res @ res) / n_
        reg = float(np.mean((Cf - C0) ** 2)) / REG_SCALE
        return res, mse, reg

    C = np.asarray(C_free, dtype=np.float64).copy()
    res, mse, reg = fwd(C)
    loss = mse + lam * reg
    mu = mu0
    n_iter = n_reject = 0
    for _ in range(max_iter):
        cnt["nbwd"] += 1
        S = sensitivity_matrix(s, d_model[tr], pb.rh[tr], r_of_free(C),
                               sens_train, wrt="C", accuracy=1e-12,
                               max_iter=GGA_MI, polish_steps=POLISH)[:, free]
        cnt["nfe"] += 1                          # sensitivity 内部的批前向
        A0 = S.T @ S / n_
        if w_reg > 0:
            A0[np.diag_indices_from(A0)] += w_reg
        g = S.T @ res / n_ + w_reg * (C - C0)
        accepted = False
        for _try in range(6):
            dC = np.linalg.solve(
                A0 + mu * np.eye(f), -g)
            Cn = np.clip(C + np.clip(dC, -20.0, 20.0),
                         C_LO + 1e-3, C_HI - 1e-3)
            res_n, mse_n, reg_n = fwd(Cn)
            loss_n = mse_n + lam * reg_n
            if loss_n < loss:
                gain = (loss - loss_n) / max(loss, 1e-300)
                C, res, mse, reg, loss = Cn, res_n, mse_n, reg_n, loss_n
                mu = max(mu * 0.3, 1e-12)
                accepted = True
                n_iter += 1
                break
            mu *= 10.0
            n_reject += 1
        if not accepted or gain < rel_tol:
            break
    return C, dict(n_iter=n_iter, n_reject=n_reject, mse=mse, reg=reg)


def calibrate(pb, obs, sens_train, d_model, lam=1e-4, C_init=None,
              adam_steps=None, lbfgs_iter=None, lm_iter=None, lr0=0.3,
              verbose=False):
    """Adam 粗搜 → L-BFGS(strong_wolfe) 精修 → LM 精抛光。返回 C_hat 与计数器。"""
    s = pb.s
    adam_steps = adam_steps or pb.cfg["adam"]
    lbfgs_iter = lbfgs_iter or pb.cfg["lbfgs"]
    tr = pb.train_frames
    dt = torch.tensor(d_model[tr])
    rht = torch.tensor(pb.rh[tr])
    ket = torch.zeros(pb.net.N)
    obs_t = torch.tensor(obs[np.ix_(tr, sens_train)])
    elev_t = torch.tensor(pb.elev[sens_train])
    rp = RoughnessParam(s, pipe_idx=pb.free_idx, C_lo=C_LO, C_hi=C_HI,
                        init=(C0 if C_init is None else C_init))
    cnt = dict(nfe=0, nbwd=0, tcv_flips=0)
    state = {"prev": None}

    def fwd():
        cnt["nfe"] += 1
        head, flow, _ = ImplicitGGASolve.apply(dt, rht, ket, rp.r_hw(), s,
                                               1e-12, GGA_MI, POLISH)
        if pb.tcv_open.any():
            sg = np.sign(flow.detach().cpu().numpy()[:, pb.tcv_open])
            if state["prev"] is not None:
                cnt["tcv_flips"] += int((sg != state["prev"]).sum())
            state["prev"] = sg
        return head

    def losses():
        head = fwd()
        pred = head[:, sens_train] - elev_t
        mse = ((pred - obs_t) ** 2).mean()
        reg = ((rp.C() - C0) ** 2).mean() / REG_SCALE
        return mse, reg

    traj = []
    t0 = time.perf_counter()
    opt = torch.optim.Adam([rp.theta], lr=lr0)
    for it in range(adam_steps):
        if it in (int(0.6 * adam_steps), int(0.85 * adam_steps)):
            for g in opt.param_groups:
                g["lr"] *= 0.5
        opt.zero_grad()
        mse, reg = losses()
        (mse + lam * reg).backward()
        cnt["nbwd"] += 1
        opt.step()
        if it % 10 == 0 or it == adam_steps - 1:
            traj.append(dict(phase="adam", it=it, mse=float(mse),
                             reg=float(reg)))
        if verbose and it % 20 == 0:
            print(f"    adam[{it}] mse={float(mse):.3e}")
    t_adam = time.perf_counter() - t0

    lb = {"n": 0, "last": None}
    t1 = time.perf_counter()
    rounds = pb.cfg.get("rounds", 1)
    for _ in range(rounds):                      # 多轮重启：重置曲率记忆与线搜索
        opt2 = torch.optim.LBFGS([rp.theta], lr=1.0, max_iter=lbfgs_iter,
                                 max_eval=int(lbfgs_iter * 1.6),
                                 history_size=30,
                                 line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-13, tolerance_change=1e-18)

        def closure():
            opt2.zero_grad()
            mse, reg = losses()
            loss = mse + lam * reg
            loss.backward()
            cnt["nbwd"] += 1
            lb["n"] += 1
            lb["last"] = (float(mse), float(reg))
            return loss

        opt2.step(closure)
    t_lbfgs = time.perf_counter() - t1
    mse_f, reg_f = lb["last"] if lb["last"] else (float("nan"), float("nan"))
    traj.append(dict(phase="lbfgs", it=lb["n"], mse=mse_f, reg=reg_f))

    # ---- LM 精抛光（见 _lm_polish 文档字符串；lm_iter=0 关闭） ----
    # 双起点：①L-BFGS 终点（延续一阶轨迹）②本次校核自己的初值（一阶阶段沿
    # 弱奇异方向的漂移会让 LM 从①出发爬行 - 独立复核实测 Hanoi L0 从①40 步
    # 停在 train 6.6e-3 ft，从②30 步到 2e-4 - 取总损失更低者，如实记录）。
    lm_iter = pb.cfg.get("lm", 0) if lm_iter is None else lm_iter
    C_hat = rp.C().detach().cpu().numpy()
    t2 = time.perf_counter()
    lm_info = dict(n_iter=0, n_reject=0, mse=mse_f, reg=reg_f)
    if lm_iter:
        C_start0 = np.broadcast_to(
            np.asarray(C0 if C_init is None else C_init, dtype=np.float64),
            (len(pb.free_idx),)).copy()
        best = None
        n_it = n_rej = 0
        for tag, Cs in (("warm", C_hat), ("init", C_start0)):
            C_cand, inf = _lm_polish(pb, Cs, obs, sens_train, d_model,
                                     lam, lm_iter, cnt, state)
            n_it += inf["n_iter"]
            n_rej += inf["n_reject"]
            tot = inf["mse"] + lam * inf["reg"]
            traj.append(dict(phase=f"lm_{tag}", it=inf["n_iter"],
                             mse=inf["mse"], reg=inf["reg"]))
        # 逐起点比较（保持两条记录，选优）
            if best is None or tot < best[0]:
                best = (tot, C_cand, inf, tag)
        _, C_hat, lm_info, lm_pick = best
        lm_info = dict(lm_info, n_iter=n_it, n_reject=n_rej, pick=lm_pick)
        mse_f, reg_f = lm_info["mse"], lm_info["reg"]
    t_lm = time.perf_counter() - t2
    return dict(C_hat=C_hat,
                mse_train=mse_f, reg=reg_f, traj=traj,
                nfe=cnt["nfe"], nbwd=cnt["nbwd"], tcv_flips=cnt["tcv_flips"],
                n_tcv_open=int(pb.tcv_open.sum()),
                t_adam=t_adam, t_lbfgs=t_lbfgs, t_lm=t_lm,
                t_total=t_adam + t_lbfgs + t_lm,
                adam_steps=adam_steps, lbfgs_evals=lb["n"],
                lm_iters=lm_info["n_iter"], lm_rejects=lm_info["n_reject"],
                lm_pick=lm_info.get("pick", "off"))


# ================================================================ 评价
def evaluate(pb, C_true, C_hat_free, reason, Vr, k_sub, rank_eps, scen,
             sens_train, sens_hold, d_model):
    C_hat_full = np.full(pb.s.L, C0, dtype=np.float64)
    C_hat_full[pb.free_idx] = C_hat_free
    sol = solve_polished(pb.s, d_model, pb.rh, r_hw=pb.r_of_C(C_hat_full),
                         accuracy=1e-12, max_iter=OBS_MI,
                         polish_steps=OBS_POLISH)
    p_model = sol["head"] - pb.elev[None, :]
    obs, p_clean = scen["obs"], scen["p_clean"]
    tr, va = pb.train_frames, pb.val_frames
    rmse = lambda a: float(np.sqrt(np.mean(np.asarray(a) ** 2)))

    pm = {}
    for tag, ref in (("", obs), ("_clean", p_clean)):
        e = p_model - ref
        pm["train_rmse" + tag] = rmse(e[np.ix_(tr, sens_train)])
        pm["val_frame_rmse" + tag] = rmse(e[np.ix_(va, sens_train)])
        pm["val_sensor_rmse" + tag] = rmse(e[:, sens_hold])

    free = pb.free_idx
    r_free = reason[free]
    info_m = r_free == "informative"
    unob_m = r_free == "unobservable"
    dC = C_hat_free - C_true[free]
    cm = dict(n_free=int(len(free)), n_informative=int(info_m.sum()),
              n_unobservable=int(unob_m.sum()))
    if info_m.any():
        cm.update(info_mae=float(np.mean(np.abs(dC[info_m]))),
                  info_rmse=rmse(dC[info_m]),
                  info_maxabs=float(np.max(np.abs(dC[info_m]))))
    if unob_m.any():
        cm.update(unob_mae=float(np.mean(np.abs(dC[unob_m]))),
                  unob_dist_prior=float(np.mean(
                      np.abs(C_hat_free[unob_m] - C0))))
    frz = pb.pidx[pb.struct_mask[pb.pidx]]
    if len(frz):
        cm["frozen_mae_vs_truth"] = float(np.mean(np.abs(C0 - C_true[frz])))
    # 良态子空间投影误差（详见 get_diag 注释）；参照值 = 先验 C0 的投影误差
    proj = Vr.T @ dC
    proj0 = Vr.T @ (np.full(len(free), C0) - C_true[free])
    cm.update(sub_rank=int(k_sub), rank_eps=int(rank_eps),
              sub_tol=SUB_TOL,
              sub_rmse=float(np.linalg.norm(proj) / np.sqrt(k_sub)),
              sub_rmse_prior=float(np.linalg.norm(proj0) / np.sqrt(k_sub)))
    return dict(**pm, **cm)


# ================================================================ run 编排
def lhs_inits(n, d, seed):
    rng = np.random.default_rng(seed)
    u = np.empty((n, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.uniform(size=n)) / n
    return 80.0 + 65.0 * u          # C ∈ [80,145]


def run_config(pb, key, mode="perpipe", sigma=0.0, sigma_b=0.0,
               noise_seed=0, lam=1e-4, placement="default", dem_seed=None,
               demand_known=False, C_init=None, adam_steps=None,
               lbfgs_iter=None, note=""):
    t0 = time.perf_counter()
    C_true, tinfo = make_truth(pb, mode, TRUTH_SEED[mode])
    scen = make_obs(pb, C_true, (mode, TRUTH_SEED[mode]), sigma=sigma,
                    sigma_b=sigma_b, noise_seed=noise_seed, dem_seed=dem_seed)
    sensors = get_sensors(pb, placement)
    sens_train, sens_hold = split_holdout(sensors)
    reason, Vr, k_sub, rank_eps = get_diag(pb, placement, sensors)
    d_model = scen["d_used"] if demand_known else pb.d
    cal = calibrate(pb, scen["obs"], sens_train, d_model, lam=lam,
                    C_init=C_init, adam_steps=adam_steps,
                    lbfgs_iter=lbfgs_iter)
    ev = evaluate(pb, C_true, cal["C_hat"], reason, Vr, k_sub, rank_eps, scen,
                  sens_train, sens_hold, d_model)
    rec = dict(key=key, net=pb.key, truth=tinfo, sigma=sigma, sigma_b=sigma_b,
               noise_seed=noise_seed, lam=lam, placement=placement,
               dem_seed=dem_seed, demand_known=bool(demand_known),
               n_sensors=int(len(sensors)), n_train_sensors=int(len(sens_train)),
               n_hold_sensors=int(len(sens_hold)),
               train_frames=int(len(pb.train_frames)),
               val_frames=int(len(pb.val_frames)),
               obs_resid_inf=scen["resid_inf"], note=note,
               mse_train_final=cal["mse_train"],
               nfe=cal["nfe"], nbwd=cal["nbwd"], tcv_flips=cal["tcv_flips"],
               n_tcv_open=cal["n_tcv_open"], adam_steps=cal["adam_steps"],
               lbfgs_evals=cal["lbfgs_evals"], lm_iters=cal["lm_iters"],
               lm_rejects=cal["lm_rejects"], lm_pick=cal["lm_pick"],
               t_calib_sec=cal["t_total"],
               t_total_sec=time.perf_counter() - t0,
               traj=cal["traj"],
               C_hat_free=[round(float(x), 4) for x in cal["C_hat"]], **ev)
    print(f"[{pb.key}:{key}] info C-RMSE={ev.get('info_rmse', float('nan')):.3f} "
          f"MAE={ev.get('info_mae', float('nan')):.3f} "
          f"max={ev.get('info_maxabs', float('nan')):.2f} "
          f"sub[r={ev['sub_rank']}]={ev['sub_rmse']:.3f}"
          f"(先验 {ev['sub_rmse_prior']:.3f}) | "
          f"train p-RMSE={ev['train_rmse']:.4f} "
          f"valF={ev['val_frame_rmse']:.4f} valS={ev['val_sensor_rmse']:.4f} ft | "
          f"unobs→先验 {ev.get('unob_dist_prior', float('nan')):.2f} | "
          f"NFE={cal['nfe']} bwd={cal['nbwd']} TCV翻转={cal['tcv_flips']} "
          f"{cal['t_total']:.0f}s")
    return rec


# ================================================================ 落盘
def json_path(pb):
    return os.path.join(ROOT, "data", f"calib_gc1_{pb.cfg['stem']}.json")


def merge_save(pb, recs):
    fp = json_path(pb)
    data = {}
    if os.path.exists(fp):
        data = json.load(open(fp, encoding="utf-8"))
    data.setdefault("config", dict(
        stem=pb.cfg["stem"], frames=pb.frames_note, n_frames=25,
        val_frames=VAL_FRAMES, gga_mi=GGA_MI, polish=POLISH,
        obs_mi=OBS_MI, obs_polish=OBS_POLISH,
        C_box=[C_LO, C_HI], C0=C0, reg_scale=REG_SCALE,
        truth_seed=TRUTH_SEED, holdout_seed=HOLDOUT_SEED,
        adam_default=pb.cfg["adam"], lbfgs_default=pb.cfg["lbfgs"],
        lm_default=pb.cfg.get("lm", 0),
        budget_note="Adam 步数按预算削减至默认值（任务书允许减半并如实报告），"
                    "L-BFGS strong_wolfe 精修 + LM(伴随雅可比) 精抛光托底；"
                    "LM 阶段为 G-C1 独立复核后补：一阶法尾部欠收敛曾把 L0 "
                    "地板误算进可辨识性账（Hanoi 满秩 L0 一阶法停在 info "
                    "C-RMSE≈5.5，LM 后 <1e-2）"))
    data.setdefault("structural", dict(
        n_pipe=int(len(pb.pidx)), n_dead=int(pb.dead_mask.sum()),
        n_clamped_allframes=int(pb.clamp_all_mask.sum()),
        n_structural=int(pb.struct_mask.sum()),
        n_free=int(len(pb.free_idx)),
        note="结构性不可辨识管冻结在 C0=130，不进入决策变量"))
    data.setdefault("runs", {})
    for r in recs:
        data["runs"][r["key"]] = r
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, fp)
    print(f"  已落盘 {fp}（runs 共 {len(data['runs'])} 条）")


# ================================================================ stages
def stage_l0(netkey):
    pb = Problem(netkey)
    print(f"== L0 无噪声 sanity（{netkey}；inverse crime 设定：观测与反演用同一"
          f"位级求解器，位级一致使其比一般情况更严重，只证明优化器收敛，"
          f"不证明方法实用）==")
    print(f"  结构性不可辨识 {int(pb.struct_mask.sum())}"
          f"（死支 {int(pb.dead_mask.sum())} + 全帧钳位 "
          f"{int(pb.clamp_all_mask.sum())}），自由集 {len(pb.free_idx)}"
          f"/{len(pb.pidx)}")
    recs = [run_config(pb, f"L0_{m}", mode=m, sigma=0.0, noise_seed=0,
                       lam=0.0, note="inverse_crime")
            for m in ("perpipe", "grouped")]
    merge_save(pb, recs)


def stage_l1sigma(sigma, seeds):
    pb = Problem("city_d")
    recs = [run_config(pb, f"L1_s{sigma:g}_n{sd}_lam1e-04", sigma=sigma,
                       noise_seed=sd, lam=1e-4) for sd in seeds]
    merge_save(pb, recs)


def stage_l1lam(lam, seeds):
    pb = Problem("city_d")
    tag = f"{lam:.0e}" if lam > 0 else "0"
    recs = [run_config(pb, f"L1_s0.1_n{sd}_lam{tag}", sigma=0.1,
                       noise_seed=sd, lam=lam) for sd in seeds]
    merge_save(pb, recs)


def stage_l1bias(seeds):
    pb = Problem("city_d")
    recs = [run_config(pb, f"L1bias_n{sd}", sigma=0.1, sigma_b=0.05,
                       noise_seed=sd, lam=1e-4) for sd in seeds]
    merge_save(pb, recs)


def stage_l1hanoi():
    pb = Problem("hanoi")
    recs = []
    for sg in (0.03, 0.1, 0.3):
        for sd in range(NOISE_SEED0, NOISE_SEED0 + 3):
            recs.append(run_config(pb, f"L1_s{sg:g}_n{sd}_lam1e-04",
                                   sigma=sg, noise_seed=sd, lam=1e-4))
    merge_save(pb, recs)


def stage_l1hanoi_aug(placements):
    """Hanoi σ 梯 × 3 种子 × 增设布点（S0 / S0∪S_k）：同真值、同噪声实现（按节点生成），
    只换布点。键 AUG_<placement>_s<σ>_n<seed>；缺省 stage 不受影响。"""
    pb = Problem("hanoi")
    recs = []
    for pl in placements:
        for sg in (0.03, 0.1, 0.3):
            for sd in range(NOISE_SEED0, NOISE_SEED0 + 3):
                recs.append(run_config(pb, f"AUG_{pl}_s{sg:g}_n{sd}", sigma=sg,
                                       noise_seed=sd, lam=1e-4, placement=pl,
                                       note="增设实验：同真值+噪声种子，只换布点"))
        merge_save(pb, recs)


def stage_l2(seeds):
    pb = Problem("city_d")
    recs = []
    for i in seeds:
        dem_seed = DEM_SEED0 + i
        nz = NOISE_SEED0 + i
        recs.append(run_config(pb, f"L2_unknown_r{i}", sigma=0.1,
                               noise_seed=nz, lam=1e-4, dem_seed=dem_seed,
                               demand_known=False,
                               note="真值需水×U[0.85,1.15]，反演用名义需水"))
        recs.append(run_config(pb, f"L2_known_r{i}", sigma=0.1,
                               noise_seed=nz, lam=1e-4, dem_seed=dem_seed,
                               demand_known=True,
                               note="同一观测，反演已知扰动后需水（对照）"))
    merge_save(pb, recs)


def stage_bridge(placements):
    pb = Problem("city_d")
    recs = [run_config(pb, f"BR_{pl}", sigma=0.1, noise_seed=NOISE_SEED0,
                       lam=1e-4, placement=pl,
                       note="同一真值+噪声种子，只换布点")
            for pl in placements]
    merge_save(pb, recs)


def stage_multistart(starts):
    pb = Problem("city_d")
    inits = lhs_inits(8, len(pb.free_idx), LHS_SEED)
    recs = [run_config(pb, f"MS_start{i}", sigma=0.1, noise_seed=NOISE_SEED0,
                       lam=1e-4, C_init=inits[i], adam_steps=60,
                       lbfgs_iter=15,
                       note="8 起点 LHS C∈[80,145]（预算减半：Adam60+LBFGS15）")
            for i in starts]
    merge_save(pb, recs)


# ================================================================ 汇总
def _agg(vals):
    v = np.asarray([x for x in vals if x == x], dtype=np.float64)
    if not len(v):
        return "-"
    if len(v) == 1:
        return f"{v[0]:.3f}"
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return f"{med:.3f} [IQR {q1:.3f},{q3:.3f}]"


def stage_summary():
    print("=" * 78)
    print("G-C1 校核实验汇总（详细数值见 data/calib_gc1_*.json）")
    print("=" * 78)
    for netkey in ("city_d", "hanoi", "modena"):
        fp = os.path.join(ROOT, "data",
                          f"calib_gc1_{NETS[netkey]['stem']}.json")
        if not os.path.exists(fp):
            continue
        data = json.load(open(fp, encoding="utf-8"))
        runs = data["runs"]
        st = data["structural"]
        print(f"\n### 网络 {netkey}（管 {st['n_pipe']}，结构性不可辨识 "
              f"{st['n_structural']}=死支{st['n_dead']}+全帧钳位"
              f"{st['n_clamped_allframes']}，自由集 {st['n_free']}）")
        l0 = {k: r for k, r in runs.items() if k.startswith("L0")}
        if l0:
            print("  [L0 无噪声 | inverse crime 设定，只证优化器收敛，不证实用性]")
            for k, r in sorted(l0.items()):
                print(f"    {k:<14} info C-RMSE={r.get('info_rmse', np.nan):.4f} "
                      f"max|ΔC|={r.get('info_maxabs', np.nan):.3f} "
                      f"训练p-RMSE={r['train_rmse']:.2e} "
                      f"验证帧={r['val_frame_rmse']:.2e} "
                      f"验证传感器={r['val_sensor_rmse']:.2e} ft "
                      f"NFE={r['nfe']}")
        if netkey == "city_d":
            print("  [L1 噪声退化曲线（λ=1e-4，5 种子中位数；sub=可辨识子空间"
                  "投影 C 误差，裸 C-RMSE 被零空间淹没时看它）]")
            for sg in ("0.03", "0.1", "0.3"):
                rs = [r for k, r in runs.items()
                      if k.startswith(f"L1_s{sg}_") and k.endswith("lam1e-04")]
                if rs:
                    print(f"    σ={sg:<5} info C-RMSE={_agg([r['info_rmse'] for r in rs]):<24} "
                          f"sub C-RMSE={_agg([r.get('sub_rmse') for r in rs]):<24} "
                          f"valF p-RMSE={_agg([r['val_frame_rmse'] for r in rs]):<24} "
                          f"valS={_agg([r['val_sensor_rmse'] for r in rs])}")
            print("  [L1 Tikhonov λ 的作用（σ=0.1，5 种子中位数）]")
            for tag in ("0", "1e-04", "1e-02"):
                rs = [r for k, r in runs.items()
                      if k.startswith("L1_s0.1_") and k.endswith(f"lam{tag}")]
                if rs:
                    print(f"    λ={tag:<6} info C-RMSE={_agg([r['info_rmse'] for r in rs]):<28} "
                          f"unob→先验={_agg([r.get('unob_dist_prior') for r in rs]):<24} "
                          f"valF={_agg([r['val_frame_rmse'] for r in rs])}")
            rs = [r for k, r in runs.items() if k.startswith("L1bias")]
            if rs:
                print(f"    偏置档 σ=0.1+σ_b=0.05: info C-RMSE="
                      f"{_agg([r['info_rmse'] for r in rs])} "
                      f"valF={_agg([r['val_frame_rmse'] for r in rs])}")
            un = [r for k, r in runs.items() if k.startswith("L2_unknown")]
            kn = [r for k, r in runs.items() if k.startswith("L2_known")]
            if un and kn:
                print("  [L2 需水不确定性（±15%，σ=0.1，5 副本中位数） - WDN 头号误差源量化]")
                print(f"    需水未知: info C-RMSE={_agg([r['info_rmse'] for r in un]):<28} "
                      f"valF={_agg([r['val_frame_rmse'] for r in un])}")
                print(f"    需水已知: info C-RMSE={_agg([r['info_rmse'] for r in kn]):<28} "
                      f"valF={_agg([r['val_frame_rmse'] for r in kn])}")
            br = {k: r for k, r in runs.items() if k.startswith("BR_")}
            if br:
                print("  [桥接：同一真值+噪声，只换布点（σ=0.1, λ=1e-4）]")
                rnd = [r for k, r in br.items() if "random" in k]
                for k, r in sorted(br.items()):
                    if "random" in k:
                        continue
                    print(f"    {k:<16} informative={r['n_informative']:<4} "
                          f"秩={r.get('sub_rank', '-'):<4} "
                          f"info C-RMSE={r['info_rmse']:.3f}  "
                          f"sub={r.get('sub_rmse', np.nan):.3f}"
                          f"(先验 {r.get('sub_rmse_prior', np.nan):.3f})  "
                          f"valF p-RMSE={r['val_frame_rmse']:.4f}  "
                          f"valS={r['val_sensor_rmse']:.4f}")
                if rnd:
                    print(f"    random40×{len(rnd)}     informative="
                          f"{_agg([r['n_informative'] for r in rnd]):<10} "
                          f"秩={_agg([r.get('sub_rank') for r in rnd]):<10} "
                          f"info C-RMSE={_agg([r['info_rmse'] for r in rnd]):<22} "
                          f"sub={_agg([r.get('sub_rmse') for r in rnd]):<22} "
                          f"valF={_agg([r['val_frame_rmse'] for r in rnd])}")
            ms = [r for k, r in runs.items() if k.startswith("MS_")]
            if ms:
                v = sorted(r["info_rmse"] for r in ms)
                pv = sorted(r["val_frame_rmse"] for r in ms)
                print(f"  [多起点 8×LHS（σ=0.1 档，预算减半）] info C-RMSE "
                      f"最好={v[0]:.3f} 中位={np.median(v):.3f} 最差={v[-1]:.3f}；"
                      f"valF p-RMSE 最好={pv[0]:.4f} 中位={np.median(pv):.4f} "
                      f"最差={pv[-1]:.4f}")
        if netkey == "hanoi":
            rs = {k: r for k, r in runs.items() if k.startswith("L1")}
            if rs:
                print("  [L1 快速版（3 种子中位数）]")
                for sg in ("0.03", "0.1", "0.3"):
                    g = [r for k, r in rs.items() if k.startswith(f"L1_s{sg}_")]
                    if g:
                        print(f"    σ={sg:<5} info C-RMSE={_agg([r['info_rmse'] for r in g]):<28} "
                              f"valF={_agg([r['val_frame_rmse'] for r in g])}")
        nfe = [r["nfe"] for r in runs.values()]
        tt = sum(r["t_total_sec"] for r in runs.values())
        fl = [r["tcv_flips"] for r in runs.values()]
        print(f"  [口径] 共 {len(runs)} 次校核；NFE 合计 {sum(nfe)}"
              f"（单次中位 {int(np.median(nfe))}）；TCV 流向翻转合计 {sum(fl)}；"
              f"墙钟合计 {tt / 60:.1f} min")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--net", default="city_d")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--placements", default="")
    ap.add_argument("--starts", default="")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    t0 = time.perf_counter()
    if a.stage == "l0":
        stage_l0(a.net)
    elif a.stage == "l1sigma":
        stage_l1sigma(a.sigma, seeds)
    elif a.stage == "l1lam":
        stage_l1lam(a.lam, seeds)
    elif a.stage == "l1bias":
        stage_l1bias(seeds)
    elif a.stage == "l1hanoi":
        stage_l1hanoi()
    elif a.stage == "l1hanoi_aug":
        stage_l1hanoi_aug([p for p in a.placements.split(",") if p])
    elif a.stage == "l2":
        stage_l2(seeds)
    elif a.stage == "bridge":
        stage_bridge([p for p in a.placements.split(",") if p])
    elif a.stage == "multistart":
        stage_multistart([int(x) for x in a.starts.split(",") if x != ""])
    elif a.stage == "summary":
        stage_summary()
    else:
        raise SystemExit(f"未知 stage: {a.stage}")
    print(f"[stage {a.stage}] 总墙钟 {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()

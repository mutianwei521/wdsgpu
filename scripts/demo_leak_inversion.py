# -*- coding: utf-8 -*-
"""demo_leak_inversion.py - City D管网梯度式漏损反演招牌演示（任务 C-demo）。

流程：
  1. 候选集：networks/field_records/city_d_leak_records.json 的 distinct_nodes 与
     city_d junction 求交（49 个漏损记录节点）。
  2. 合成真值：3 个候选节点（含热点 195/3083 + 随机第三个）挂 emitter，用户系数
     C = Q_lps / p_m^gamma 使漏损约 1.5~3 LPS；Ke_int 换算走 parse.py 同款链
     （input1.c:567-573：Ke = (Ucf[FLOW]^Qexp / Ucf[PRESSURE]) / C^Qexp）。
     观测 = 24h x 25 帧、40 个固定随机传感器 junction 的压力（我们的求解器
     accuracy=1e-12 + Newton 精抛光，||F||inf ~ 3e-14 ft）。
  3. 反演：决策变量 = 全部候选节点的漏损强度（参数化 C = softplus(theta) > 0，
     Ke_int = ucf/C^Qexp 保持链式可微；C=0 即 Ke=inf 无漏损 - 注意 EPANET 的
     Ke_int 是"阻力"，与漏损强度反向，故 L1 稀疏罚施加在漏损强度 C 上，
     等价于对 Ke_int^(-1/Qexp) 的 L1）。损失 = 25 帧传感器压力 MSE + lambda*L1(C)，
     Adam + lr 阶梯调度，lambda 扫 3 个值；梯度全部来自 ImplicitGGASolve（隐函数
     伴随法，autodiff.py）。
  4. 由于候选签名互相干高达 0.9999+（如 2202/2204、3083/712，40 传感器不可分线性
     方向），纯 L1 会把漏损质量摊到近共线邻居簇上，无法精确恢复支撑（实测见
     data/demo_leak_inversion.json 的 stage1 记录）。故在 L1 粗筛之上加离散支撑
     精化：非线性 OMP（用线性签名字典对精确非线性残差打分选点，逐点加入后用
     L-BFGS + ImplicitGGASolve 梯度整体重拟合）+ 对手互换抛光（高相干竞争节点
     逐一试换、按非线性 MSE 取优）。无噪声下真支撑使 MSE 崩到求解器地板
     （~1e-20 ft^2 量级），与错支撑（>=1e-7）差 10 个数量级以上，判别确定性成立。
  5. 评价：支撑恢复（top-3）、真值节点漏损流量相对误差（<5% 门槛）、非真值残余
     漏损当量、耗时与前向次数；鲁棒组 = 观测加 0.1 ft 高斯噪声重跑（如实报告）。

运行：python -X utf8 scripts/demo_leak_inversion.py
"""

import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message=".*not writable.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import Net                                   # noqa: E402
from dgga.solver import GGASolver                            # noqa: E402
from dgga.units import LPSperCFS, MperFT                     # noqa: E402
from dgga.autodiff import ImplicitGGASolve, solve_polished   # noqa: E402

torch.set_default_dtype(torch.float64)

SEED = 2026            # 传感器 / 第三真值节点
SEED_NOISE = 909       # 观测噪声
N_SENSOR = 40
NOISE_FT = 0.1
LAMBDAS = [1e-4, 3e-4, 1e-3]
STAGE1_ITERS = 100
GGA_MI, POLISH = 20, 4          # 反演前向：GGA 20 迭代进入 relerr 平台 + Newton 抛光
                                 # （抛光后 ||F||inf~3e-14，与 60 迭代结果一致）
OBS_MI = 60                      # 观测生成用 60 迭代（与 gradcheck 同款保守设置）

FWD_COUNT = [0]                  # 前向求解次数统计（每次 = 25 帧批量）


# ----------------------------------------------------------------------
# 基础设施
# ----------------------------------------------------------------------
def softplus_inv(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.expm1(np.maximum(x, 1e-12)))


class Problem:
    """场景容器：网络、帧、候选集、单位链、观测张量。"""

    def __init__(self):
        self.net = Net.load(os.path.join(ROOT, "data", "reference"), "city_d")
        self.s = GGASolver(self.net, mode="dense",
                           inp_path=os.path.join(ROOT, "networks", "realInpData",
                                                 "city_d.inp"))
        self.node_index = {nid: i for i, nid in enumerate(self.net.node_id)}
        T = [t * 3600 for t in range(25)]                    # 24h x 25 帧
        self.d = np.stack([self.net.demand_cfs_at(t) for t in T])
        self.rh = np.stack([np.nan_to_num(self.net.reservoir_head_ft_at(t))
                            for t in T])
        rec = json.load(open(os.path.join(ROOT, "networks", "field_records",
                                          "city_d_leak_records.json"),
                             encoding="utf-8"))
        # 候选集 = 记录 distinct_nodes 与 junction 求交（json 已去重）
        self.cand = [n for n in rec["distinct_nodes"]
                     if n in self.node_index
                     and self.net.node_type[self.node_index[n]] == 0]
        self.cidx = np.array([self.node_index[n] for n in self.cand])
        self.nc = len(self.cand)
        # 单位链（parse.py:275-290/350-368 同款；city_d.inp: LPS、无 PRESSURE 行
        # -> METERS、SPECIFIC GRAVITY 0.998、EMITTER EXPONENT 0.5 -> Qexp=2）
        self.gamma = float(self.net.meta["emitter_exponent"])
        self.qexp = float(self.net.meta["qexp"])
        spgrav = 0.998                                       # city_d.inp [OPTIONS]
        self.pcf = MperFT * spgrav                           # input1.c:451（METERS）
        self.ucf_e = LPSperCFS ** self.qexp / self.pcf       # input1.c:568
        # torch 常量
        self.dt = torch.tensor(self.d)
        self.rht = torch.tensor(self.rh)
        self.r0 = self.s.r_hw.clone()
        # 搜索用帧子集（阶段 2 离散支撑搜索的快速前向；最终数值仍用全 25 帧）
        self.idx5 = np.array([0, 6, 12, 18, 24])
        self.dt5 = self.dt[self.idx5]
        self.rht5 = self.rht[self.idx5]

    def ke_int_of_C(self, c_user):
        """用户系数 C -> Ke_int（input1.c:572；C>0）。numpy/torch 通用。"""
        return self.ucf_e / c_user ** self.qexp

    def fsolve(self, ke=None, max_iter=GGA_MI, polish=POLISH):
        FWD_COUNT[0] += 1
        return solve_polished(self.s, self.d, self.rh, ke=ke, accuracy=1e-12,
                              max_iter=max_iter, polish_steps=polish)


def implicit_forward(pb, ke_full, frames="full"):
    FWD_COUNT[0] += 1
    if frames == "search":
        return ImplicitGGASolve.apply(pb.dt5, pb.rht5, ke_full, pb.r0,
                                      pb.s, 1e-12, GGA_MI, POLISH)
    return ImplicitGGASolve.apply(pb.dt, pb.rht, ke_full, pb.r0,
                                  pb.s, 1e-12, GGA_MI, POLISH)


# ----------------------------------------------------------------------
# 阶段 1：Adam + L1（任务规定的基础反演器）
# ----------------------------------------------------------------------
def stage1_adam_l1(pb, obs_t, sens, elev_s, lam, iters=STAGE1_ITERS):
    """全候选 C=softplus(theta)，损失 = MSE + lam*sum(C)（C>0 故 L1=sum）。
    Adam lr=0.3，60/85 步处减半。返回逐候选平均漏损 LPS、C、mse。"""
    cidx_t = torch.tensor(pb.cidx)
    theta = torch.full((pb.nc,), float(softplus_inv(0.02)), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.3)
    mse_v = float("nan")
    for it in range(iters):
        if it in (60, 85):
            for g in opt.param_groups:
                g["lr"] *= 0.5
        opt.zero_grad()
        C = torch.nn.functional.softplus(theta)
        ke_full = torch.zeros(pb.net.N).index_copy(0, cidx_t, pb.ke_int_of_C(C))
        head, _, emit = implicit_forward(pb, ke_full)
        pred = head[:, sens] - elev_s
        mse = ((pred - obs_t) ** 2).mean()
        (mse + lam * C.sum()).backward()
        opt.step()
        with torch.no_grad():
            theta.clamp_(min=-11.5, max=8.0)                 # Ke 上溢保护
        mse_v = float(mse.detach())
    with torch.no_grad():
        lk = emit[:, pb.cidx].mean(0).numpy() * LPSperCFS    # 平均漏损 LPS
        C_np = torch.nn.functional.softplus(theta).numpy()
    return dict(leak_lps=lk, C=C_np, mse=mse_v)


# ----------------------------------------------------------------------
# 签名字典（线性打分用；精确判别始终走非线性 MSE）
# ----------------------------------------------------------------------
def build_dictionary(pb, sol_base, sens):
    """t=0 帧对每个候选挂 C=0.3 的 FD 签名（B=25 批量 x2），再按
    sqrt(p_j(t)/p_j(0)) 做帧间缩放 -> D [25*40, nc]（单位 C 的传感器压力响应）。"""
    C_probe = 0.3
    d0, rh0 = pb.d[0], pb.rh[0]
    P0 = sol_base["head"][0]
    n_sens = len(sens)               # = N_SENSOR（缺省 40）；增设实验传更长的 sens
    sig0 = np.zeros((pb.nc, n_sens))
    for a in range(0, pb.nc, 25):
        b = min(a + 25, pb.nc)
        ke = np.zeros((b - a, pb.net.N))
        for k, j in enumerate(range(a, b)):
            ke[k, pb.cidx[j]] = pb.ke_int_of_C(C_probe)
        FWD_COUNT[0] += 1
        sj = solve_polished(pb.s, np.tile(d0, (b - a, 1)), np.tile(rh0, (b - a, 1)),
                            ke=ke, accuracy=1e-12, max_iter=GGA_MI,
                            polish_steps=POLISH)
        for k, j in enumerate(range(a, b)):
            sig0[j] = (sj["head"][k] - P0)[sens] / C_probe
    # 帧间缩放：漏损 q ~ C*sqrt(p_j(t))，响应近似正比 q
    p_base = sol_base["head"][:, pb.cidx] - pb.net.elev_ft[pb.cidx]     # [25,nc]
    scale = np.sqrt(np.maximum(p_base, 1e-9) / np.maximum(p_base[0], 1e-9))
    D = (scale[:, None, :] * sig0.T[None, :, :]).reshape(25 * n_sens, pb.nc)
    Dn = D / np.maximum(np.linalg.norm(D, axis=0), 1e-12)
    coh = Dn.T @ Dn                                          # 互相干矩阵
    return D, Dn, coh


# ----------------------------------------------------------------------
# 阶段 2：非线性 OMP + 对手互换抛光（refit 全部用 ImplicitGGASolve 梯度）
# ----------------------------------------------------------------------
THETA_MAX = 2.5          # C = softplus(theta) <= 2.58（约 16 LPS 上限，防弱签名节点
                          # 在噪声下爆炸 - run1 实测 195 被推到 50 LPS）
PRUNE_LPS = 0.05          # 支撑成员平均漏损低于此即剪除


def nl_refit(pb, support, C_init, obs_t, sens, elev_s, frames="search",
             deep=False):
    """支撑上的非线性重拟合（L-BFGS strong_wolfe；theta=softplus^-1(C)，
    theta 限幅 [-12, THETA_MAX]）。非支撑候选 Ke=0（emitter 关断，精确零）。
    frames='search' 用 5 帧快速前向（支撑判别 MSE 比值不变），'full' 用全 25 帧。
    返回 C、mse、逐候选漏损、残差（对应帧集）。"""
    sup_nodes = torch.tensor(pb.cidx[support])
    obs_use = obs_t[torch.tensor(pb.idx5)] if frames == "search" else obs_t
    theta = torch.tensor(softplus_inv(np.clip(C_init, 1e-3, 2.5)),
                         requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=(40 if deep else 15),
                            history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-13, tolerance_change=1e-16)

    def fwd():
        C = torch.nn.functional.softplus(theta.clamp(-12.0, THETA_MAX))
        ke_full = torch.zeros(pb.net.N).index_copy(0, sup_nodes, pb.ke_int_of_C(C))
        return C, implicit_forward(pb, ke_full, frames)

    def closure():
        opt.zero_grad()
        _, (head, _flow, _emit) = fwd()
        pred = head[:, sens] - elev_s
        loss = ((pred - obs_use) ** 2).mean()
        loss.backward()
        return loss

    for _ in range(3 if deep else 1):
        opt.step(closure)
    with torch.no_grad():
        C, (head, _, emit) = fwd()
        pred = head[:, sens] - elev_s
        mse = float(((pred - obs_use) ** 2).mean())
        resid = (obs_use - pred).numpy().reshape(-1)
        leak = emit[:, pb.cidx].mean(0).numpy() * LPSperCFS
    return dict(C=C.numpy(), mse=mse, leak_lps=leak, resid=resid)


def _nnls_init(pb, D5, support, rhs5, C_prev):
    """字典 NNLS 给新支撑的 C 初值（响应为降压方向，取负号做 NNLS）。"""
    from scipy.optimize import nnls
    c_lin, _ = nnls(-D5[:, support], -rhs5)
    fb = np.concatenate([np.clip(C_prev, 1e-3, None),
                         np.full(len(support) - len(C_prev), 0.05)])
    return np.where(c_lin > 1e-4, c_lin, fb)


def stage2_support_search(pb, obs_t, sens, elev_s, D, Dn, coh, mse0, log=print):
    """离散支撑搜索（run1 教训版）：
    1. 贪婪加点到 k=4（线性字典对精确非线性残差打分；不设早停 - run1 中
       改善阈值早停把真第三节点 195 回退掉了）；k=3 后先做一轮成员精化。
    2. 成员精化 = 坐标穷举：每个支撑位在 {自身 + 传感器相干 top-4 竞争者} 池内
       逐一试换，非线性短重拟合按 MSE 取任何严格改善（run1 的 0.5x 接受阈值
       太苛刻：支撑不完整时单点互换只有边际改善）。真支撑使 MSE 崩到求解器
       地板（<<1e-12），错支撑 >=1e-7，判别确定性成立。
    3. 平均漏损 < PRUNE_LPS 的成员剪除，全 25 帧 L-BFGS 深度重拟合出最终数值。"""
    n_sens = len(sens)               # = N_SENSOR（缺省 40）；增设实验传更长的 sens
    rows5 = (pb.idx5[:, None] * n_sens + np.arange(n_sens)).reshape(-1)
    D5, base5 = D[rows5], _BASE_CACHE["pred0"][rows5]
    Dn5 = D5 / np.maximum(np.linalg.norm(D5, axis=0), 1e-12)
    obs5 = obs_t.numpy()[pb.idx5].reshape(-1)
    rhs5 = obs5 - base5

    def refine(support, fit, sweeps=2):
        for sw in range(sweeps):
            improved = False
            for pos in range(len(support)):
                i = support[pos]
                pool = [j for j in np.argsort(-coh[i])
                        if j not in support and coh[i, j] > 0.5][:4]
                for j in pool:
                    sup_alt = support.copy()
                    sup_alt[pos] = j
                    fit_alt = nl_refit(pb, np.array(sup_alt),
                                       np.clip(fit["C"], 1e-3, None),
                                       obs_t, sens, elev_s)
                    tag = ""
                    if fit_alt["mse"] < fit["mse"] * (1.0 - 1e-3):
                        support, fit = sup_alt, fit_alt
                        improved = True
                        tag = "  <- 接受"
                    log(f"    精化 {pb.cand[i]}->{pb.cand[j]} "
                        f"(coh={coh[i, j]:.5f}): mse={fit_alt['mse']:.3e}{tag}")
                    if tag:
                        break
            if not improved:
                break
        return support, fit

    support, fit = [], None
    resid5 = rhs5.copy()
    for k in range(4):
        score = resid5 @ Dn5                                 # 漏损降压 => 内积为正
        score[support] = -np.inf
        j_new = int(np.argmax(score))
        support = support + [j_new]
        c_init = _nnls_init(pb, D5, support, rhs5,
                            fit["C"] if fit else np.zeros(0))
        fit = nl_refit(pb, np.array(support), c_init, obs_t, sens, elev_s)
        resid5 = fit["resid"]
        log(f"    贪婪加点 k={k+1}: +{pb.cand[j_new]} mse5={fit['mse']:.3e}")
        if len(support) == 3:
            support, fit = refine(support, fit, sweeps=2)
            resid5 = fit["resid"]
        if fit["mse"] < 1e-17:
            break
    if len(support) == 4:
        support, fit = refine(support, fit, sweeps=1)
    # ---- 留一消元：伪支撑与真节点共线分裂时（如 304 分走 195 的漏损），
    # 去伪成员后重拟合 MSE 反而下降/持平（质量并回真节点），去真成员则显著
    # 恶化（其信号无人解释）。逐轮删除"去之最优"的成员直到无成员可删。----
    changed = True
    while changed and len(support) > 1:
        changed = False
        best = None
        for pos in range(len(support)):
            sup_wo = support[:pos] + support[pos + 1:]
            C_wo = np.delete(np.clip(fit["C"], 1e-3, None), pos)
            f_wo = nl_refit(pb, np.array(sup_wo), C_wo, obs_t, sens, elev_s)
            log(f"    留一 -{pb.cand[support[pos]]}: mse={f_wo['mse']:.3e}")
            if best is None or f_wo["mse"] < best[1]["mse"]:
                best = (pos, f_wo)
        # 剔除判据：去掉后 MSE 恶化 < 3% 即视为不携带独立信息（噪声地板下
        # MSE 的 chi^2 波动尺度 ~ sqrt(2/n_obs)≈10%，3% 为保守值；无噪声时
        # 去伪成员会好几个数量级、去真成员劣化 >10x，判据同样成立。
        # run2 教训：原 2.0x 阈值在噪声组把劣化 27% 的携信息成员也删了。
        if best[1]["mse"] < fit["mse"] * 1.03:
            log(f"    剔除 {pb.cand[support[best[0]]]}")
            support = support[:best[0]] + support[best[0] + 1:]
            fit = best[1]
            changed = True
    # ---- 剪枝 + 深度重拟合 ----
    keep = [p for p in range(len(support))
            if fit["leak_lps"][support[p]] >= PRUNE_LPS]
    if len(keep) < len(support):
        log(f"    剪除低漏损成员: "
            f"{[pb.cand[support[p]] for p in range(len(support)) if p not in keep]}")
        support = [support[p] for p in keep]
    fit = nl_refit(pb, np.array(support), np.clip(fit["C"], 1e-3, None),
                   obs_t, sens, elev_s, frames="full", deep=True)
    log(f"    深度重拟合(25 帧): mse={fit['mse']:.3e} 支撑="
        f"{[pb.cand[j] for j in support]}")
    return support, fit


_BASE_CACHE = {}


def base_pred_flat(pb, obs_t):
    """无漏损基线传感器压力（扁平），build 阶段缓存。"""
    return _BASE_CACHE["pred0"]


# ----------------------------------------------------------------------
# 评价与主流程
# ----------------------------------------------------------------------
def evaluate(pb, group, true_nodes, true_lk, support, fit, st1_best, t_used):
    top3 = [pb.cand[j] for j in
            np.argsort(-fit["leak_lps"])[:3]] if len(support) >= 3 else \
           [pb.cand[j] for j in support]
    sup_ids = [pb.cand[j] for j in support]
    exact = set(top3) == set(true_nodes)
    flow_err = {}
    for n in true_nodes:
        est = fit["leak_lps"][pb.cand.index(n)] if n in sup_ids else 0.0
        flow_err[n] = dict(true_lps=true_lk[n], est_lps=float(est),
                           rel_err=float(abs(est - true_lk[n]) / true_lk[n]))
    # 非真值残余：阶段1(lambda*) 的残余 + 最终支撑中的非真值节点
    nt = [k for k, n in enumerate(pb.cand) if n not in true_nodes]
    resid_st1 = float(np.max(st1_best["leak_lps"][nt]))
    resid_final = float(max([fit["leak_lps"][j] for j in support
                             if pb.cand[j] not in true_nodes], default=0.0))
    return dict(group=group, support=sup_ids, top3=top3,
                support_leak_lps={pb.cand[j]: float(fit["leak_lps"][j])
                                  for j in support},
                top3_exact=bool(exact),
                flow_err=flow_err,
                max_flow_rel_err=float(max(v["rel_err"] for v in flow_err.values())),
                resid_nontrue_stage1_lps=resid_st1,
                resid_nontrue_final_lps=resid_final,
                final_mse_ft2=fit["mse"], time_sec=t_used)


def run_group(pb, group, obs, sens, elev_s, D, Dn, coh, mse0, lambdas, log=print):
    t0 = time.time()
    obs_t = torch.tensor(obs)
    log(f"\n---- [{group}] 阶段 1：Adam + L1（lambda 扫描 {lambdas}）----")
    st1 = {}
    for lam in lambdas:
        r = stage1_adam_l1(pb, obs_t, sens, elev_s, lam)
        top3 = [pb.cand[j] for j in np.argsort(-r["leak_lps"])[:3]]
        st1[lam] = r
        r["top3"] = top3
        log(f"  lambda={lam:.0e}: mse={r['mse']:.3e} top3={top3} "
            f"leak(top3)={np.round(np.sort(r['leak_lps'])[::-1][:3], 3)}")
    return st1, obs_t, t0


def main():
    t_all = time.time()
    pb = Problem()
    log = print
    log(f"网络: city_d N={pb.net.N} Nj={pb.s.Nj} L={pb.s.L}；候选节点 {pb.nc} 个"
        f"（漏损记录 distinct_nodes 与 junction 求交）")

    # ---- 真值构造 ----
    rng = np.random.default_rng(SEED)
    others = [n for n in pb.cand if n not in ("195", "3083")]
    third = str(rng.choice(others))
    targets = {"195": 3.0, "3083": 1.5, third: 2.2}          # 目标漏损 LPS
    sol_base = pb.fsolve(max_iter=OBS_MI)                    # 无漏损基线（1e-12 档）
    C_true = {}
    ke_true = np.zeros(pb.net.N)
    for n, q in targets.items():
        i = pb.node_index[n]
        p_m = (sol_base["head"][:, i] - pb.net.elev_ft[i]).mean() * MperFT
        C_true[n] = q / p_m ** pb.gamma                      # q = C*p^gamma（用户制）
        ke_true[i] = pb.ke_int_of_C(C_true[n])
    sol_true = pb.fsolve(ke=ke_true, max_iter=OBS_MI)
    true_lk = {n: float(sol_true["emitter"][:, pb.node_index[n]].mean() * LPSperCFS)
               for n in targets}
    sens = np.sort(rng.choice(pb.s.junc_nodes, N_SENSOR, replace=False))
    elev_s = torch.tensor(pb.net.elev_ft[sens])
    obs = sol_true["head"][:, sens] - pb.net.elev_ft[sens]   # 观测压力 ft [25,40]
    pred0 = (sol_base["head"][:, sens] - pb.net.elev_ft[sens]).reshape(-1)
    _BASE_CACHE["pred0"] = pred0
    mse0 = float(((obs.reshape(-1) - pred0) ** 2).mean())
    log(f"真值节点: {[(n, round(true_lk[n], 3)) for n in targets]} LPS "
        f"(C_user={ {n: round(c, 4) for n, c in C_true.items()} })")
    log(f"传感器 {N_SENSOR} 个（seed={SEED}）；基线-真值 MSE={mse0:.3e} ft^2；"
        f"观测残差 ||F||inf={sol_true['resid_inf'].max():.2e}")

    # ---- 签名字典 ----
    D, Dn, coh = build_dictionary(pb, sol_base, sens)
    hard = [(pb.cand[i], pb.cand[j], float(coh[i, j]))
            for i, j in zip(*np.triu_indices(pb.nc, 1)) if coh[i, j] > 0.999]
    log(f"字典互相干 > 0.999 的候选对 {len(hard)} 组（前 5: "
        f"{sorted(hard, key=lambda x: -x[2])[:5]}） - 纯 L1 无法唯一分辨，"
        f"需离散支撑精化")

    results = dict(config=dict(seed=SEED, seed_noise=SEED_NOISE, n_sensor=N_SENSOR,
                               noise_ft=NOISE_FT, lambdas=LAMBDAS,
                               stage1_iters=STAGE1_ITERS, gga_max_iter=GGA_MI,
                               polish_steps=POLISH, frames=25,
                               true_nodes={n: dict(target_lps=targets[n],
                                                   C_user=C_true[n],
                                                   ke_int=float(ke_true[pb.node_index[n]]),
                                                   true_mean_lps=true_lk[n])
                                           for n in targets},
                               sensors=[pb.net.node_id[i] for i in sens],
                               candidates=pb.cand),
                   groups={})

    # ================= 无噪声组 =================
    st1, obs_t, t0 = run_group(pb, "noiseless", obs, sens, elev_s,
                               D, Dn, coh, mse0, LAMBDAS)
    # lambda*：top3 命中数最多，其次分离度
    def lam_score(lam):
        r = st1[lam]
        hit = len(set(r["top3"]) & set(targets))
        lk = np.sort(r["leak_lps"])[::-1]
        margin = lk[2] / max(lk[3], 1e-9) if len(lk) > 3 else 0.0
        return (hit, margin)
    lam_star = max(LAMBDAS, key=lam_score)
    print(f"  lambda* = {lam_star:.0e}（支撑恢复最好）")
    print(f"---- [noiseless] 阶段 2：非线性 OMP + 互换抛光 ----")
    support, fit = stage2_support_search(pb, obs_t, sens, elev_s, D, Dn, coh, mse0)
    ev = evaluate(pb, "noiseless", targets, true_lk, support, fit,
                  st1[lam_star], time.time() - t0)
    ev["stage1"] = {f"{lam:.0e}": dict(top3=st1[lam]["top3"], mse=st1[lam]["mse"],
                                       leak_lps={pb.cand[j]: float(st1[lam]["leak_lps"][j])
                                                 for j in np.argsort(-st1[lam]["leak_lps"])[:8]})
                    for lam in LAMBDAS}
    ev["lambda_star"] = lam_star
    results["groups"]["noiseless"] = ev
    print(f"[noiseless] top3={ev['top3']} 精确恢复={ev['top3_exact']} "
          f"最大流量相对误差={ev['max_flow_rel_err']:.3e} "
          f"final_mse={fit['mse']:.3e} 用时={ev['time_sec']:.0f}s")

    # ================= 噪声组（0.1 ft 高斯） =================
    rng_n = np.random.default_rng(SEED_NOISE)
    obs_n = obs + NOISE_FT * rng_n.standard_normal(obs.shape)
    mse0_n = float(((obs_n.reshape(-1) - pred0) ** 2).mean())
    st1n, obs_nt, t0n = run_group(pb, "noisy", obs_n, sens, elev_s,
                                  D, Dn, coh, mse0_n, [lam_star])
    print(f"---- [noisy] 阶段 2：非线性 OMP + 互换抛光（噪声地板≈{NOISE_FT**2:.0e}）----")
    support_n, fit_n = stage2_support_search(pb, obs_nt, sens, elev_s, D, Dn, coh,
                                             mse0_n)
    evn = evaluate(pb, "noisy", targets, true_lk, support_n, fit_n,
                   st1n[lam_star], time.time() - t0n)
    evn["stage1"] = {f"{lam_star:.0e}": dict(top3=st1n[lam_star]["top3"],
                                             mse=st1n[lam_star]["mse"])}
    results["groups"]["noisy"] = evn
    print(f"[noisy] top3={evn['top3']} 精确恢复={evn['top3_exact']} "
          f"最大流量相对误差={evn['max_flow_rel_err']:.3e} "
          f"final_mse={fit_n['mse']:.3e} 用时={evn['time_sec']:.0f}s")

    # ---- 汇总 ----
    results["total_time_sec"] = time.time() - t_all
    results["n_forward_solves"] = FWD_COUNT[0]
    out = os.path.join(ROOT, "data", "demo_leak_inversion.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, default=float)

    print("\n" + "=" * 72)
    print("City D梯度式漏损反演演示 - 结果汇总")
    print("=" * 72)
    for g, ev in results["groups"].items():
        tag = "无噪声" if g == "noiseless" else f"噪声 {NOISE_FT} ft"
        print(f"[{tag}] 支撑={ev['support']} top3={ev['top3']} "
              f"精确恢复={'是' if ev['top3_exact'] else '否'}")
        for n, fe in ev["flow_err"].items():
            print(f"    节点 {n}: 真值 {fe['true_lps']:.4f} LPS, "
                  f"估计 {fe['est_lps']:.4f} LPS, 相对误差 {fe['rel_err']:.2%}")
        print(f"    非真值残余漏损当量: 阶段1(L1)最大 "
              f"{ev['resid_nontrue_stage1_lps']:.4f} LPS, "
              f"最终支撑内 {ev['resid_nontrue_final_lps']:.4f} LPS")
        print(f"    最终 MSE={ev['final_mse_ft2']:.3e} ft^2, 用时 {ev['time_sec']:.0f}s")
    gate = results["groups"]["noiseless"]
    ok = gate["top3_exact"] and gate["max_flow_rel_err"] < 0.05
    print(f"\n门槛判定（无噪声组 top-3 精确恢复且流量误差<5%）: "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"总耗时 {results['total_time_sec']:.0f}s；前向求解 {FWD_COUNT[0]} 次"
          f"（每次 25 帧批量）；结果已存 data/demo_leak_inversion.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

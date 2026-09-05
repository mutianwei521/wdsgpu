# -*- coding: utf-8 -*-
"""dgga.mlds - GGAFormer 训练数据管道（任务 A）。

职责：
1. 拓扑注册表（训练池 22 网 / 留出测试 6 网，拓扑级零泄漏）；
2. 需水场景采样（逐节点乘性 U[0.3,1.7] × 全局缩放 U[0.6,1.4]，固定种子可复现；
   水库水头 v1 不动 = t=0 水头）；
3. 标签生成：dgga dense f64 批量正演
   - GGA 原版迭代数 + relerr 轨迹：按 EPANET 同口径判据（hacc=solver.hacc_default
     即 INP ACCURACY 经 input3.c:2014-2019 钳位后的值；max_iter=TRIALS）逐步展开，
     与单发 solve() 逐位一致（有断言）；
   - 标签权威 Q*/H*：GGA(accuracy=1e-12) + 整装 Newton 精抛光（autodiff.solve_polished，
     ‖F‖∞ ~ 1e-13 量级） - G-C1 教训：一阶尾部欠收敛会污染结论，标签必须精抛光。
4. 特征规范：边特征 [log r_hw, log diam, log len, is_tcv]，节点特征
   [demand_norm(逐场景), elev_rel(相对水库均值水头), degree]；
   标准化统计只在训练池上算（norm_stats.json）。

单位铁律：内部 ft/cfs（Ucf）。构造顺序铁律：先 GGASolver(net, inp_path=...)
（_apply_exact_props 就地修正 net.dem_base_cfs 位级基值），再调 net.demand_cfs_at()。
"""

import json
import os
import zlib

import numpy as np
import torch

try:
    from dgga.parse import Net
    from dgga.solver import GGASolver
    from dgga.autodiff import solve_polished
except ImportError:  # pragma: no cover
    from parse import Net
    from solver import GGASolver
    from autodiff import solve_polished

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(ROOT, "data", "reference")
MLDS_DIR = os.path.join(ROOT, "data", "mlds")

GLOBAL_SEED = 20260811          # 任务 A 固定全局种子
N_SCENARIOS_DEFAULT = 256

# 场景采样分布（任务书规定）
NODE_MULT_LO, NODE_MULT_HI = 0.3, 1.7
GLOB_SCALE_LO, GLOB_SCALE_HI = 0.6, 1.4


# ======================================================================
# 1. 拓扑注册表
# ======================================================================
def topo_registry():
    """返回 list[dict(stem, split, inp)]。训练池 22 网 + 留出测试 6 网。
    拓扑级零泄漏：split='test' 的拓扑任何场景不得进训练（含 norm_stats）。
    INP 路径与 data/reference 参考解构建时的 inp_used 一致
    （data/public_reference_index.json）。"""
    reg = []
    for i in range(16):
        reg.append(dict(stem=f"rand_main_{i:04d}", split="train",
                        inp=f"networks/random_main/rand_{i:04d}.inp"))
    for i in range(3):
        reg.append(dict(stem=f"rand_small_{i:04d}", split="train",
                        inp=f"networks/random_small/rand_{i:04d}.inp"))
    reg.append(dict(stem="pub_hanoi", split="train",
                    inp="networks/public/Hanoi.inp"))
    reg.append(dict(stem="pub_fossolo_poly1", split="train",
                    inp="networks/public/_cleaned/Fossolo_poly1.inp"))
    reg.append(dict(stem="pub_pescara", split="train",
                    inp="networks/public/_cleaned/Pescara.inp"))
    # ---- 留出测试（拓扑级零泄漏）----
    for i in range(16, 20):
        reg.append(dict(stem=f"rand_main_{i:04d}", split="test",
                        inp=f"networks/random_main/rand_{i:04d}.inp"))
    reg.append(dict(stem="pub_modena", split="test",
                    inp="networks/public/_cleaned/Modena.inp"))
    reg.append(dict(stem="city_d", split="test",
                    inp="networks/realInpData/city_d.inp"))
    return reg


def load_topo(stem, inp, mode="dense"):
    """Net.load + GGASolver(inp_path=...) 的规约顺序封装。
    返回 (net, solver)。注意：solver 构造会就地修正 net.dem_base_cfs / node_ke
    为 INP 位级基值 - demand_cfs_at() 必须在此之后调用。"""
    net = Net.load(REF_DIR, stem)
    solver = GGASolver(net, mode=mode, inp_path=os.path.join(ROOT, inp))
    return net, solver


def topo_seed(stem):
    """拓扑级确定性种子（跨平台稳定：crc32）。"""
    return [GLOBAL_SEED, zlib.crc32(stem.encode("utf-8"))]


# ======================================================================
# 2. 场景采样
# ======================================================================
def sample_scenarios(net, n_scenarios=N_SCENARIOS_DEFAULT, stem=None, seed=None):
    """需水场景采样。基准 = t=0 名义需水（demand_cfs_at(0)，含 pattern 首帧因子与
    Demand Multiplier）；逐节点乘性扰动 U[0.3,1.7] × 全局缩放 U[0.6,1.4]。
    零基值节点保持 0（乘性）。水库水头 v1 不动 = t=0 水头。
    返回 dict(demand[S,N], res_head[S,N], node_mult[S,N], glob_scale[S],
              base_demand[N])。"""
    if seed is None:
        seed = topo_seed(stem)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    base = net.demand_cfs_at(0)                       # [N] cfs（solver 构造后的位级基值）
    rh0 = net.reservoir_head_ft_at(0)                 # [N] ft（非水库位 nan）
    S, N = int(n_scenarios), net.N
    node_mult = rng.uniform(NODE_MULT_LO, NODE_MULT_HI, size=(S, N))
    glob = rng.uniform(GLOB_SCALE_LO, GLOB_SCALE_HI, size=(S, 1))
    demand = base[None, :] * node_mult * glob         # [S,N]
    res_head = np.broadcast_to(rh0, (S, N)).copy()
    return dict(demand=demand, res_head=res_head, node_mult=node_mult,
                glob_scale=glob[:, 0], base_demand=base)


# ======================================================================
# 3. 标签生成
# ======================================================================
def _chunk_size(Nj, budget_bytes=3.0e8, cap=256):
    """按稠密 A[B,Nj,Nj] f64 的内存预算选批块大小。"""
    return int(max(1, min(cap, budget_bytes / (Nj * Nj * 8))))


def gga_rollout(solver, demand, res_head, max_iter=None, hacc=None):
    """GGA 原版逐步展开（dense f64 批量）：逐迭代记录 relerr 轨迹与迭代数。

    与单发 solver.solve(max_iter=trials) 逐位一致（利用 dense 路径迭代状态仅为
    (q, e_j)、H 每轮由 q 重解、批不变性已在 regression ⑤ 验证的事实）：
    第 k 步以上一步的 q 热启动跑 max_iter=1；已收敛样本冻结（其 q 不再更新，
    与 solve() 内部 active 掩码语义一致）。本期网均无 emitter（e_j 恒 0 通路）。

    返回 dict(head_ft[S,N], flow_cfs[S,L], iters[S], relerr_traj[S,T](nan 填充),
              relerr[S], converged[S], hacc, max_iter)。"""
    dt, dev = solver.dtype, solver.device
    max_iter = solver.max_iter_default if max_iter is None else int(max_iter)
    hacc = solver.hacc_default if hacc is None else float(hacc)
    d = torch.as_tensor(np.ascontiguousarray(demand), dtype=dt, device=dev)
    rh = torch.as_tensor(np.ascontiguousarray(res_head), dtype=dt, device=dev)
    S = d.shape[0]

    q = solver._init_flow().unsqueeze(0).expand(S, -1).clone()      # [S,L]
    H_fin = torch.zeros(S, solver.N, dtype=dt, device=dev)
    q_fin = torch.zeros(S, solver.L, dtype=dt, device=dev)
    active = torch.ones(S, dtype=torch.bool, device=dev)
    iters = np.zeros(S, dtype=np.int32)
    relerr_fin = np.zeros(S, dtype=np.float64)
    traj = np.full((S, max_iter), np.nan, dtype=np.float64)

    for it in range(1, max_iter + 1):
        out = solver.solve(d, rh, q0=q, max_iter=1, accuracy=hacc)
        rel = out["relerr"]                                          # [S] 本步 relerr
        am = active
        traj[am.cpu().numpy(), it - 1] = rel[am].cpu().numpy()
        # solve() 语义：本轮开始时 active 的样本更新 q/H 后再判收敛
        q = torch.where(am.view(-1, 1), out["flow_cfs"], q)
        H_fin = torch.where(am.view(-1, 1), out["head_ft"], H_fin)
        q_fin = torch.where(am.view(-1, 1), out["flow_cfs"], q_fin)
        newly = am & (rel <= hacc)
        nn = newly.cpu().numpy()
        iters[nn] = it
        relerr_fin[nn] = rel[newly].cpu().numpy()
        active = am & (rel > hacc)
        if not bool(active.any()):
            break
    # 未收敛样本（理论上不应出现）：记满迭代数与末步 relerr
    an = active.cpu().numpy()
    if an.any():
        iters[an] = max_iter
        last = traj[an]
        relerr_fin[an] = np.array([row[~np.isnan(row)][-1] for row in last])
    T = int(iters.max())
    return dict(head_ft=H_fin.cpu().numpy(), flow_cfs=q_fin.cpu().numpy(),
                iters=iters, relerr_traj=traj[:, :T], relerr=relerr_fin,
                converged=~an, hacc=hacc, max_iter=max_iter)


def generate_labels(solver, demand, res_head, chunk=None, verify_first_chunk=True):
    """全量标签生成（分块）：GGA 原版迭代数/轨迹（EPANET 同口径） + 精抛光标签。

    返回 dict(head_ft, flow_cfs, gga_iters, gga_relerr_traj, gga_relerr,
              gga_converged, polish_resid_inf, gga_head_ft_err, hacc, max_iter,
              rollout_bitexact)。
    head_ft/flow_cfs = 精抛光权威标签；gga_head_ft_err = GGA 原版终态与精抛光
    标签的 junction 头最大偏差（EPANET 口径欠收敛量的实测记录）。"""
    S = demand.shape[0]
    if chunk is None:
        chunk = _chunk_size(solver.Nj)
    heads, flows, resid = [], [], []
    g_iters, g_rel, g_conv, g_traj, g_herr = [], [], [], [], []
    bitexact = True
    jm = solver.junc_nodes
    for a in range(0, S, chunk):
        b = min(S, a + chunk)
        d_c, rh_c = demand[a:b], res_head[a:b]
        ro = gga_rollout(solver, d_c, rh_c)
        if verify_first_chunk and a == 0:
            # 逐步展开 vs 单发 solve 逐位一致断言（轨迹口径的正确性证据）
            one = solver.solve(d_c, rh_c)
            ok = (np.array_equal(one["head_ft"].cpu().numpy(), ro["head_ft"])
                  and np.array_equal(one["flow_cfs"].cpu().numpy(), ro["flow_cfs"])
                  and np.array_equal(one["iters"].cpu().numpy().astype(np.int32),
                                     ro["iters"])
                  and np.array_equal(one["relerr"].cpu().numpy(), ro["relerr"]))
            bitexact = bitexact and ok
            if not ok:
                raise AssertionError("gga_rollout 与单发 solve() 不逐位一致")
        # 权威标签：GGA(1e-12) + Newton 精抛光
        pol = solve_polished(solver, d_c, rh_c,
                             accuracy=1e-12, max_iter=200, polish_steps=3)
        heads.append(pol["head"])
        flows.append(pol["q"])
        resid.append(pol["resid_inf"])
        g_iters.append(ro["iters"])
        g_rel.append(ro["relerr"])
        g_conv.append(ro["converged"])
        g_traj.append(ro["relerr_traj"])
        g_herr.append(np.abs(ro["head_ft"][:, jm]
                             - pol["head"][:, jm]).max(axis=1))
    Tm = max(t.shape[1] for t in g_traj)
    traj = np.full((S, Tm), np.nan, dtype=np.float64)
    o = 0
    for t in g_traj:
        traj[o:o + t.shape[0], :t.shape[1]] = t
        o += t.shape[0]
    return dict(head_ft=np.concatenate(heads), flow_cfs=np.concatenate(flows),
                polish_resid_inf=np.concatenate(resid),
                gga_iters=np.concatenate(g_iters),
                gga_relerr=np.concatenate(g_rel),
                gga_converged=np.concatenate(g_conv),
                gga_relerr_traj=traj,
                gga_head_ft_err=np.concatenate(g_herr),
                hacc=solver.hacc_default, max_iter=solver.max_iter_default,
                rollout_bitexact=bitexact)


# ======================================================================
# 4. 特征规范
# ======================================================================
EDGE_FEAT_NAMES = ["log_r_hw", "log_diam", "log_len", "is_tcv"]
NODE_FEAT_NAMES = ["demand_norm", "elev_rel", "degree"]
_LOG_CLAMP = 1e-12

_CVPIPE, _PIPE, _TCV = 0, 1, 7


def edge_features(net):
    """[L,4]：log r_hw / log diam / log len（clip 1e-12 防非管道 0 值）/ is_tcv。"""
    lt = np.asarray(net.link_type)
    f = np.stack([
        np.log(np.clip(np.asarray(net.r_hw, dtype=np.float64), _LOG_CLAMP, None)),
        np.log(np.clip(np.asarray(net.diam_ft, dtype=np.float64), _LOG_CLAMP, None)),
        np.log(np.clip(np.asarray(net.len_ft, dtype=np.float64), _LOG_CLAMP, None)),
        (lt == _TCV).astype(np.float64),
    ], axis=1)
    return f


def node_static_features(net, res_head0):
    """[N,2]：elev_rel = elev - mean(水库水头)（v1 水头不动 → 拓扑常量）；degree。"""
    rh = np.asarray(res_head0, dtype=np.float64)
    res_mean = float(np.nanmean(rh)) if np.isfinite(rh).any() else 0.0
    elev_rel = np.asarray(net.elev_ft, dtype=np.float64) - res_mean
    deg = np.zeros(net.N, dtype=np.float64)
    np.add.at(deg, np.asarray(net.link_n1, dtype=np.int64), 1.0)
    np.add.at(deg, np.asarray(net.link_n2, dtype=np.int64), 1.0)
    return np.stack([elev_rel, deg], axis=1)


class _RunningStats:
    """在线 mean/std 累加器（跨拓扑/场景聚合，数值上用 sum/sumsq 即可 -
    特征量级 O(1)~O(10)，f64 无消cancel风险）。"""

    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0

    def add(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n += x.size
        self.s += float(x.sum())
        self.s2 += float((x * x).sum())

    def result(self):
        mean = self.s / max(1, self.n)
        var = max(0.0, self.s2 / max(1, self.n) - mean * mean)
        return dict(mean=mean, std=float(np.sqrt(var)), count=int(self.n))


def compute_norm_stats(items):
    """标准化统计（只喂训练池！）。items = 迭代器，元素为
    dict(edge_feat[L,4], node_static[N,2], demand[S,N], junc_mask[N])。
    demand 统计只在 junction 位。is_tcv 二值特征不标准化（standardize=False）。"""
    acc_e = [_RunningStats() for _ in EDGE_FEAT_NAMES]
    acc_n = [_RunningStats() for _ in NODE_FEAT_NAMES]
    for it in items:
        ef, ns = it["edge_feat"], it["node_static"]
        jm = it["junc_mask"]
        for j in range(len(EDGE_FEAT_NAMES)):
            acc_e[j].add(ef[:, j])
        acc_n[0].add(it["demand"][:, jm])          # demand_norm：逐场景 junction 需水
        acc_n[1].add(ns[jm, 0])                    # elev_rel
        acc_n[2].add(ns[jm, 1])                    # degree
    out = {"edge": {}, "node": {}, "seed": GLOBAL_SEED,
           "note": "stats computed on TRAIN pool only; demand/elev_rel/degree "
                   "over junction nodes; log-features clipped at 1e-12"}
    for j, nm in enumerate(EDGE_FEAT_NAMES):
        r = acc_e[j].result()
        r["standardize"] = (nm != "is_tcv")
        out["edge"][nm] = r
    for j, nm in enumerate(NODE_FEAT_NAMES):
        r = acc_n[j].result()
        r["standardize"] = True
        out["node"][nm] = r
    return out

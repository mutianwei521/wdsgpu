# -*- coding: utf-8 -*-
"""probe_math.py - 隐式伴随约化路线的数值证明（任务一，CPU，L-TOWN）。

命题：ImplicitGGASolve 的伴随系统 J^T λ = v（J 见 autodiff._build_J，含 ACTIVE
PRV 约束行）可约化为节点方程 M·λ_m = r，其中除 ACTIVE PRV 下游行外
M ≡ 收敛态 GGA 矩阵 Ā（P=1/hgrad 逐支同装配值 + emitter 对角），而 ACTIVE PRV
下游行是约束行 λ_m[j2] − λ_m[j1] = gQ[k_prv]。GGA 前向分解的是
big-M 的 Â = Ā + CBIG·Σe_{j2}e_{j2}^T ⇒ 精确解用 Woodbury 行替换修正。

本脚本在 L-TOWN 名义帧（冻结状态 S*，ACTIVE PRV 3/3）上量三个东西：
  ① naive（直接 Â^{-1}r，忽略约束行）与 splu(J^T) 参考的 λ_m / 四类 θ 梯度差
 - 预计在 PRV 下游坐标处差 O(1)（CBIG=1e8 的罚函数把 λ_m[j2] 压到 ~1e-8）；
  ② Woodbury 行替换修正后的差 - 预计 ~1e-9 以下（f64 + 精化）；
  ③ 逐坐标：3 个 PRV 下游节点的 demand 梯度单列。
判定门槛：② 全类 rel < 1e-8 才放行实现；①如实记录（不设门槛，只证明坑真在）。

用法：python -X utf8 scripts/adjoint_gpu/probe_math.py
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch                                            # noqa: E402
from dgga.parse import parse_inp                        # noqa: E402
from dgga.solver import GGASolver, CBIG                 # noqa: E402
from dgga.autodiff import (implicit_solve, solve_polished,   # noqa: E402
                           _link_coeffs_np, _emitter_coeffs_np,
                           _valve_act_masks, _adj_cache, _build_J)

INP = os.path.join(ROOT, "networks", "public", "_cleaned", "L-TOWN.inp")


def build_Abar(s, hg, hge, em, act_prv, big_m=True):
    """收敛态 GGA 节点矩阵：Ā（P 排除 ACTIVE PRV）＋ em/hge 对角
    （+ big_m 时 ACTIVE PRV 下游对角 CBIG）。csc 返回。"""
    c = _adj_cache(s)
    P = 1.0 / hg
    P = np.where(act_prv, 0.0, P)          # ACTIVE PRV：linkcoeffs P=0 跳过
    rows, cols, vals = [], [], []
    both = c["m1"] & c["m2"]
    j1b, j2b = c["j1"][both], c["j2"][both]
    Pb = P[both]
    rows += [j1b, j2b, j1b, j2b]
    cols += [j2b, j1b, j1b, j2b]
    vals += [-Pb, -Pb, Pb, Pb]
    only1 = c["m1"] & ~c["m2"]
    rows.append(c["j1"][only1]); cols.append(c["j1"][only1]); vals.append(P[only1])
    only2 = c["m2"] & ~c["m1"]
    rows.append(c["j2"][only2]); cols.append(c["j2"][only2]); vals.append(P[only2])
    ar = np.arange(s.Nj)
    rows.append(ar); cols.append(ar)
    vals.append(np.where(em, 1.0 / hge, 0.0))
    if big_m and act_prv.any():
        j2a = c["j2"][act_prv]
        rows.append(j2a); cols.append(j2a)
        vals.append(np.full(j2a.size, CBIG))
    A = sp.coo_matrix((np.concatenate([np.asarray(v, dtype=np.float64) for v in vals]),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(s.Nj, s.Nj)).tocsc()
    return A, P


def reduced_grads(s, sol, gQ, gE_j, gH_j, mode="woodbury"):
    """约化伴随：返回 (gd, grh, gke, gr, lam_m)。mode ∈ {naive, woodbury}。"""
    c = _adj_cache(s)
    q, e_j = sol["q"][0], sol["e_j"][0]
    ke_j = sol["ke"][0, s.junc_nodes]
    _, hg, dr, _pd = _link_coeffs_np(s, q, sol["r_hw"], speed=sol["speed"],
                                     status=sol["status"], h0p=sol["h0p"],
                                     rp=sol["rp"])
    _, hge, dke = _emitter_coeffs_np(s, e_j, ke_j)
    em = ke_j > 0.0
    aprv, apsv, _af = _valve_act_masks(s, sol["speed"], sol["status"])
    assert not apsv.any()
    # ACTIVE PRV 行 hg=0（约束行）：P 置 0（build_Abar 内），此处防 1/0
    hg_safe = np.where(aprv, 1.0, hg)
    Ahat, P = build_Abar(s, hg_safe, hge, em, aprv, big_m=True)
    lu = splu(Ahat)

    def solve_ref(rhs, n_ref=2):
        x = lu.solve(rhs)
        for _ in range(n_ref):
            x = x + lu.solve(rhs - Ahat @ x)
        return x

    # 标准右端：B^T P gQ − em·gE/hge − gH
    r = np.zeros(s.Nj)
    PgQ = P * gQ
    np.add.at(r, c["j2"][c["m2"]], PgQ[c["m2"]])
    np.add.at(r, c["j1"][c["m1"]], -PgQ[c["m1"]])
    r -= np.where(em, gE_j / np.where(em, hge, 1.0), 0.0)
    r -= gH_j
    j2a = c["j2"][aprv]
    j1a = c["j1"][aprv]          # L-TOWN PRV 两端均 junction（dense 守卫同口径）
    if mode == "naive":
        lam_m = solve_ref(r)
    else:
        # 约束行替换：r[j2a] = gQ[prv]；Woodbury 修正 rank-p 行替换
        r2 = r.copy()
        r2[j2a] = gQ[np.where(aprv)[0]]
        x0 = solve_ref(r2)
        S = np.stack([solve_ref(np.eye(s.Nj)[j]) for j in j2a], axis=1)  # [Nj,p]
        # C_ab = δ_ab + (V^T S)_ab，解析消去 δ_ab（Â s_b = e_{j2b} 精确时）：
        # C_ab = (s_b)_{j2a} − (s_b)_{j1a}
        C = S[j2a, :] - S[j1a, :]
        # V^T x0 = (x0_{j2}−x0_{j1}) − (Â x0)_{j2}，解析代入 Â x0 = r2
        wty = (x0[j2a] - x0[j1a]) - r2[j2a]
        y = np.linalg.solve(C, wty)
        lam_m = x0 - S @ y
        # 外层精化（对真 M 的残差；M 行 j2 = e_{j2}−e_{j1}）
        for _ in range(2):
            res = r2 - Ahat @ lam_m
            res[j2a] = r2[j2a] - (lam_m[j2a] - lam_m[j1a])
            dx0 = solve_ref(res)
            dwty = (dx0[j2a] - dx0[j1a]) - res[j2a]
            dy = np.linalg.solve(C, dwty)
            lam_m = lam_m + (dx0 - S @ dy)
    # 回代 λ_l（ACTIVE PRV 位 P=0 ⇒ λ_l=0，恰好不进任何 θ 梯度）
    Blam = np.zeros(s.L)
    Blam[c["m2"]] += lam_m[c["j2"][c["m2"]]]
    Blam[c["m1"]] -= lam_m[c["j1"][c["m1"]]]
    lam_l = P * (gQ - Blam)
    lam_e = np.where(em, (gE_j + lam_m) / np.where(em, hge, 1.0), 0.0)
    gd = np.zeros(s.N); gd[s.junc_nodes] = lam_m
    gke = np.zeros(s.N)
    gke[s.junc_nodes] = -np.where(em, lam_e * dke, 0.0)
    grh = np.zeros(s.N)
    np.add.at(grh, s.n1_np[c["f1"]], lam_l[c["f1"]])
    np.add.at(grh, s.n2_np[c["f2"]], -lam_l[c["f2"]])
    gr = -lam_l * dr
    return gd, grh, gke, gr, lam_m


def main():
    net = parse_inp(INP)
    s = GGASolver(net, mode="epanet", inp_path=INP)
    rng = np.random.default_rng(20260824)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5 * (np.asarray(net.tank_hmin)[:tn.size]
                        + np.asarray(net.tank_hmax)[:tn.size])
    ke0 = np.asarray(net.node_ke, dtype=np.float64).copy()
    jn = np.asarray(s.junc_nodes)
    ke0[jn[rng.integers(0, jn.size, 3)]] = 1e-3
    r0 = s.r_hw.cpu().numpy().copy()

    base = s.run_gga(d0, rh0, ke=ke0, do_status=True)
    S0, K0 = base["status"].copy(), base["setting"].copy()
    prv = np.where(np.asarray(net.link_type) == 3)[0]
    n_act = int((S0[prv] == 4).sum())
    print("L-TOWN 名义帧：iters=%d ACTIVE PRV=%d/3" % (int(base["iters"]), n_act))

    # ---- 参考：CPU ImplicitGGASolve（splu(J^T)）----
    dt = torch.float64
    D = torch.as_tensor(d0, dtype=dt).requires_grad_(True)
    R = torch.as_tensor(rh0, dtype=dt).requires_grad_(True)
    KE = torch.as_tensor(ke0, dtype=dt).requires_grad_(True)
    RW = torch.as_tensor(r0, dtype=dt).requires_grad_(True)
    W = torch.as_tensor(rng.normal(0, 1, s.Nj), dtype=dt)
    Wq = torch.as_tensor(rng.normal(0, 1, s.L), dtype=dt)
    We = torch.as_tensor(rng.normal(0, 1, s.N), dtype=dt)
    head, flow, emit = implicit_solve(s, D, R, ke=KE, r_hw=RW,
                                      speed=K0, status=S0)
    ((head[s.junc_nodes_t] * W).sum() + (flow * Wq).sum()
     + (emit * We).sum()).backward()
    ref = dict(demand=D.grad.numpy(), res_head=R.grad.numpy(),
               ke=KE.grad.numpy(), r_hw=RW.grad.numpy())

    # ---- 同一收敛态（solve_polished 冻结状态）----
    sol = solve_polished(s, d0, rh0, ke=ke0, r_hw=r0, speed=K0, status=S0)
    print("polish 后 ‖F‖inf = %.3e" % sol["resid_inf"][0])
    gQ = Wq.numpy().copy()
    gH_j = W.numpy().copy()
    gE_j = We.numpy()[s.junc_nodes].copy()

    # 参考 λ_m（splu(J^T) 直接解，用于逐坐标印证）
    q, e_j = sol["q"][0], sol["e_j"][0]
    ke_j = sol["ke"][0, s.junc_nodes]
    _, hgJ, _drJ, _pdJ = _link_coeffs_np(s, q, sol["r_hw"], speed=sol["speed"],
                                         status=sol["status"], h0p=sol["h0p"],
                                         rp=sol["rp"])
    _, hgeJ, _dkeJ = _emitter_coeffs_np(s, e_j, ke_j)
    emJ = ke_j > 0.0
    aprvJ, apsvJ, _ = _valve_act_masks(s, sol["speed"], sol["status"])
    J, em_rows = _build_J(s, hgJ, hgeJ, emJ, aprvJ, apsvJ)
    ne = em_rows.size
    v = np.concatenate([gQ, gE_j[em_rows], gH_j])
    lam_ref = splu(J).solve(v, trans="T")
    lam_m_ref = lam_ref[s.L + ne:]

    for mode in ("naive", "woodbury"):
        gd, grh, gke, gr, lam_m = reduced_grads(s, sol, gQ, gE_j, gH_j, mode)
        grh[s.fixed_nodes] += 0.0    # 本 loss 不含定水头头输出，直通项=0
        rd = lambda a, b: float(np.max(np.abs(a - b))
                                / max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300))
        rl = rd(lam_m, lam_m_ref)
        print("\n[%s] λ_m 最大相对差 = %.3e" % (mode, rl))
        c = _adj_cache(s)
        j2a = c["j2"][aprvJ]
        for a, j in enumerate(j2a):
            print("  PRV%d 下游 λ_m[j2=%d]: 参考=% .6e  约化=% .6e" %
                  (a, j, lam_m_ref[j], lam_m[j]))
        worst = 0.0
        for name, g in (("demand", gd), ("res_head", grh), ("ke", gke),
                        ("r_hw", gr)):
            r_ = rd(g, ref[name])
            worst = max(worst, r_)
            print("  θ=%-8s max|Δg|/max|g| = %.3e" % (name, r_))
        dn = np.asarray([int(net.link_n2[k]) for k in prv])
        for j in dn:
            print("  PRV 下游节点 %d demand: 参考=% .6e 约化=% .6e" %
                  (j, ref["demand"][j], gd[j]))
        if mode == "woodbury":
            ok = worst < 1e-8
            print("\n总判定（woodbury 全类 rel<1e-8）: %s（worst=%.3e）"
                  % ("PASS" if ok else "FAIL", worst))
            return 0 if ok else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())

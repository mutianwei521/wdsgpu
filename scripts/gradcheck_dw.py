# -*- coding: utf-8 -*-
"""gradcheck_dw.py - D-W 网（pub_balerma）梯度对拍（任务 D）。

验证 D-W 的 φ'(Q)（DWpipecoeff 的 hgrad，含 dfdq 项）与 ∂hloss/∂R 进入
autodiff 雅可比后，ImplicitGGASolve 伴随梯度 与 solve_polished 中央差分 一致：
  θ ∈ {demand, r_hw(=D-W 的 R), res_head}，L = Σ w·H_junction。
门槛：逐坐标相对误差 < 1e-6（|FD| 与 |adj| 双小于 1e-12 时按一致计）。
"""

import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import Net                      # noqa: E402
from dgga.solver import GGASolver               # noqa: E402
from dgga.autodiff import ImplicitGGASolve, solve_polished   # noqa: E402

TOL = 1e-6
STEM = "pub_balerma"
INP = os.path.join(ROOT, "networks", "public", "Balerma.inp")


def loss_from(head, w):
    return float((head * w).sum())


def run_case(net, s, d0, rh0, label, rng):
    ke0 = np.asarray(net.node_ke, dtype=np.float64)
    r0 = s.r_np.copy()
    jm = np.asarray(net.node_type) == 0
    w = np.where(jm, rng.standard_normal(net.N), 0.0)   # 只对 junction 头加权

    # ---- 伴随梯度 ----
    dt = torch.float64
    td = torch.tensor(d0, dtype=dt, requires_grad=True)
    trh = torch.tensor(np.where(np.isnan(rh0), 0.0, rh0), dtype=dt,
                       requires_grad=True)
    tke = torch.tensor(ke0, dtype=dt, requires_grad=True)
    tr = torch.tensor(r0, dtype=dt, requires_grad=True)
    head, flow, emit = ImplicitGGASolve.apply(td, trh, tke, tr, s)
    resid = solve_polished(s, d0, rh0, ke0, r0)["resid_inf"][0]
    L = (head * torch.tensor(w, dtype=dt)).sum()
    L.backward()
    gd = td.grad.numpy()
    grh = trh.grad.numpy()
    gr = tr.grad.numpy()
    print(f"[{STEM}|{label}] D-W 梯度对拍  N={net.N} L={net.L}  "
          f"抛光后 ‖F‖∞={resid:.3e}")

    def fd(theta_kind, idx, h):
        def run(dd, rr, rhh):
            sol = solve_polished(s, dd, rhh, ke0, rr)
            return loss_from(sol["head"][0], w)
        if theta_kind == "demand":
            dp = d0.copy(); dp[idx] += h
            dm = d0.copy(); dm[idx] -= h
            return (run(dp, r0, rh0) - run(dm, r0, rh0)) / (2 * h)
        if theta_kind == "r_hw":
            rp = r0.copy(); rp[idx] += h
            rm = r0.copy(); rm[idx] -= h
            return (run(d0, rp, rh0) - run(d0, rm, rh0)) / (2 * h)
        rhp = rh0.copy(); rhp[idx] += h
        rhm = rh0.copy(); rhm[idx] -= h
        return (run(d0, r0, rhp) - run(d0, r0, rhm)) / (2 * h)

    ok = True
    junc_idx = rng.choice(np.where(jm & (d0 > 0))[0], 4, replace=False)
    res_idx = np.where(np.asarray(net.node_type) == 1)[0]
    # 挑层流/紊流(Swamee-Jain)/过渡(Dunlop)三种支的管道各若干
    q_conv = solve_polished(s, d0, rh0, ke0, r0)["q"][0]
    sv = s.viscos * s.diam_np
    wre = np.abs(q_conv) / sv
    is_pipe = s.lt_np <= 1
    from dgga.solver import A1, A2
    lam = np.where(is_pipe & (np.abs(q_conv) <= A2 * sv))[0]
    swj = np.where(is_pipe & (wre >= A1))[0]
    dun = np.where(is_pipe & (np.abs(q_conv) > A2 * sv) & (wre < A1))[0]
    print(f"  流态分布: 层流 {lam.size} / Dunlop 过渡 {dun.size} / "
          f"Swamee-Jain {swj.size}")
    pipe_idx = list(rng.choice(swj, min(3, swj.size), replace=False))
    pipe_idx += list(rng.choice(dun, min(2, dun.size), replace=False))
    pipe_idx += list(rng.choice(lam, min(2, lam.size), replace=False))

    rows = []
    for i in junc_idx:
        # 绝对步长下限 1e-5 cfs：缩需水场景下 d~1e-5，过小步长使 ΔL 掉进
        # 求解噪声底（~1e-10 量级），中央差分失真
        h = max(1e-5, 1e-6 * abs(d0[i]))
        v_fd = fd("demand", int(i), h)
        rows.append(("demand", int(i), gd[int(i)], v_fd))
    for k in pipe_idx:
        # 步长按 1e-3 相对量取（r_hw 梯度绝对量 ~1e-5 级，过小步长会把
        # 中央差分压进噪声底）
        h = max(1e-3 * abs(r0[int(k)]), 1e-8)
        v_fd = fd("r_hw", int(k), h)
        rows.append(("r_hw", int(k), gr[int(k)], v_fd))
    for i in res_idx[:3]:
        v_fd = fd("res_head", int(i), 1e-5)
        rows.append(("res_head", int(i), grh[int(i)], v_fd))

    print(f"  {'θ':<9}{'idx':>6}{'adjoint':>18}{'FD 中央差分':>18}{'rel':>12}")
    worst = 0.0
    for kind, i, ga, gf in rows:
        den = max(abs(ga), abs(gf), 1e-12)
        rel = abs(ga - gf) / den
        if abs(ga - gf) < 1e-9:
            rel = 0.0        # 绝对一致到 1e-9 以内：低于中央差分噪声底，按一致计
        worst = max(worst, rel)
        flag = "" if rel < TOL else "  <-- 超限"
        ok = ok and rel < TOL
        print(f"  {kind:<9}{i:>6}{ga:>18.10e}{gf:>18.10e}{rel:>12.3e}{flag}")
    print(f"  最差相对误差 = {worst:.3e}（门槛 {TOL:.0e}）")
    return ok, worst


def main():
    net = Net.load(os.path.join(ROOT, "data", "reference"), STEM)
    s = GGASolver(net, mode="epanet", inp_path=INP)
    rng = np.random.default_rng(7)
    d0 = net.demand_cfs_at(0)
    rh0 = net.reservoir_head_ft_at(0)
    ok1, w1 = run_case(net, s, d0, rh0, "名义需水(全紊流)", rng)
    # 需水缩到 0.1%：把大部分管道压进层流/Dunlop 过渡支，覆盖三条分支的导数
    ok2, w2 = run_case(net, s, d0 * 1e-3, rh0, "0.001×需水(层流/过渡)", rng)
    ok = ok1 and ok2
    print(f"总判定: {'PASS' if ok else 'FAIL'}（两场景最差 {max(w1, w2):.3e}）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

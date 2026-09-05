# -*- coding: utf-8 -*-
"""check_prv_grad.py - PRV 轮验收 ③：L-TOWN 梯度对拍（冻结状态）。

ImplicitGGASolve（隐式伴随；ACTIVE PRV = 约束行，autodiff.py _valve_act_masks /
_residual_np / _build_J）在 L-TOWN 名义帧上，对 demand / res_head / ke / r_hw
四类参数与**中央差分**对拍。B1 审计的做法：
  · 状态冻结：先 run_gga(do_status=True) 拿收敛状态 S*/设定 K*，梯度与 FD 都在
    (S*,K*) 冻结下算（隐函数定理在状态不切换的邻域内成立，论文 lim:frozen-status）；
  · 避开切换点：每个 ±h 点重跑完整状态机，若终态 != S* 则该坐标弃用并另选
    （报弃用数）；
  · FD 基线用 solve_polished（整装 Newton 精抛光到 ‖F‖∞~1e-12，FD 噪声地板
    远低于 GGA 停机噪声）。
门槛：rel<1e-6 或 |g|<1e-9 绝对一致（同 ⑧ gradcheck_dw 口径）。
solve_unrolled 的 PRV 守卫维持 raise（本轮不给 unrolled 开 PRV），此处顺带断言。

用法：python -X utf8 scripts/prv_port/check_prv_grad.py
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch                                       # noqa: E402
from dgga.parse import parse_inp                   # noqa: E402
from dgga.solver import GGASolver                  # noqa: E402
from dgga.autodiff import (implicit_solve, solve_polished,   # noqa: E402
                           solve_unrolled)

PUB = os.path.join(ROOT, "networks", "public")
CLEAN = os.path.join(PUB, "_cleaned")
TOL_REL, TOL_ABS = 1e-6, 1e-9


def main():
    inp = os.path.join(CLEAN, "L-TOWN.inp")
    net = parse_inp(inp)
    s = GGASolver(net, mode="epanet", inp_path=inp)
    rng = np.random.default_rng(20260824)

    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5 * (np.asarray(net.tank_hmin)[:tn.size]
                        + np.asarray(net.tank_hmax)[:tn.size])
    ke0 = np.asarray(net.node_ke, dtype=np.float64).copy()
    # 给 3 个 junction 注入 emitter，让 ke 梯度非平凡
    jn = np.asarray(s.junc_nodes)
    ke0[jn[rng.integers(0, jn.size, 3)]] = 1e-3
    r0 = s.r_hw.cpu().numpy().copy()

    # ---- 冻结状态基准（完整状态机收敛）----
    base = s.run_gga(d0, rh0, ke=ke0, do_status=True)
    S0 = base["status"].copy()
    K0 = base["setting"].copy()
    n_act = int((S0[np.asarray(net.link_type) == 3] == 4).sum())
    print("L-TOWN 名义帧：iters=%d converged=%s ACTIVE PRV=%d/3" %
          (int(base["iters"]), bool(base["converged"]), n_act))

    # ---- 隐式伴随梯度 ----
    dt = torch.float64
    D = torch.as_tensor(d0, dtype=dt).requires_grad_(True)
    R = torch.as_tensor(rh0, dtype=dt).requires_grad_(True)
    KE = torch.as_tensor(ke0, dtype=dt).requires_grad_(True)
    RW = torch.as_tensor(r0, dtype=dt).requires_grad_(True)
    W = torch.as_tensor(rng.normal(0, 1, s.Nj), dtype=dt)
    head, _flow, _em = implicit_solve(s, D, R, ke=KE, r_hw=RW,
                                      speed=K0, status=S0)
    (head[s.junc_nodes_t] * W).sum().backward()
    grads = dict(demand=D.grad.numpy(), res_head=R.grad.numpy(),
                 ke=KE.grad.numpy(), r_hw=RW.grad.numpy())

    # ---- FD 坐标：每类 6 个（demand 含 3 个 PRV 下游节点；res_head 取水库/池）----
    prv = np.where(np.asarray(net.link_type) == 3)[0]
    coords = {
        "demand": ([int(net.link_n2[k]) for k in prv]
                   + [int(jn[i]) for i in rng.integers(0, jn.size, 3)]),
        "res_head": [int(n) for n in np.where(np.asarray(net.node_type)
                                              != 0)[0]],
        # ke 只取 ke>0 的节点：ke=0 处 emitter 开/关是 kink（em 掩码 ke_j>0），
        # 双侧 FD 跨 kink 无意义（与 PRV 无关的既有语义）
        "ke": [int(n) for n in np.where(ke0 > 0)[0]],
        "r_hw": [int(x) for x in rng.integers(0, s.L, 4)
                 if np.asarray(net.link_type)[x] <= 1],
    }

    def loss_of(dv, rhv, kev, rwv):
        out = solve_polished(s, dv, rhv, ke=kev, r_hw=rwv,
                             speed=K0, status=S0)
        Hj = out["head"][0, s.junc_nodes]
        return float((Hj * W.numpy()).sum())

    def status_same(dv, rhv, kev):
        r = s.run_gga(dv, rhv, ke=kev, do_status=True)
        return bool(np.array_equal(r["status"], S0))

    worst = 0.0
    n_skip = 0
    ok_all = True
    for kind, idxs in coords.items():
        wk = 0.0
        used = 0
        for i in idxs:
            dv, rhv, kev, rwv = d0.copy(), rh0.copy(), ke0.copy(), r0.copy()
            vec = dict(demand=dv, res_head=rhv, ke=kev, r_hw=rwv)[kind]
            base_v = vec[i]
            h = max(1e-6, 1e-4 * abs(base_v))
            vec[i] = base_v + h
            same_p = status_same(dv, rhv, kev)
            Lp = loss_of(dv, rhv, kev, rwv)
            vec[i] = base_v - h
            same_m = status_same(dv, rhv, kev)
            Lm = loss_of(dv, rhv, kev, rwv)
            vec[i] = base_v
            if not (same_p and same_m):
                n_skip += 1
                continue
            used += 1
            g_fd = (Lp - Lm) / (2 * h)
            g_ad = float(grads[kind][i])
            rel = abs(g_ad - g_fd) / max(abs(g_fd), abs(g_ad), 1e-300)
            good = rel < TOL_REL or (abs(g_ad) < TOL_ABS
                                     and abs(g_fd) < TOL_ABS)
            ok_all &= good
            wk = max(wk, rel if not (abs(g_ad) < TOL_ABS
                                     and abs(g_fd) < TOL_ABS) else 0.0)
        worst = max(worst, wk)
        print("  θ=%-8s 坐标 %d（弃 %d 切换点） 最差 rel=%.3e  %s" %
              (kind, used, len(idxs) - used, wk,
               "PASS" if wk < TOL_REL else "FAIL"))

    # ---- solve_unrolled PRV 守卫维持 raise ----
    try:
        sd = GGASolver(net, mode="dense", inp_path=inp,
                       dense_status_machine=True)
        solve_unrolled(sd, d0, rh0, K=3)
        guard = "未 raise <-- FAIL"
        ok_all = False
    except NotImplementedError:
        guard = "维持 raise（PASS）"
    print("  solve_unrolled PRV 守卫：%s" % guard)
    print("总判定: %s（最差 rel=%.3e，弃用切换点坐标 %d）" %
          ("PASS" if ok_all else "FAIL", worst, n_skip))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

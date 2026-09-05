# -*- coding: utf-8 -*-
"""gate_b1_grad.py - 门 B1 验收④：新特性网（CVPIPE）上的梯度对拍。

流程与论文 lim:frozen-status 的口径一致：**先用批量状态机把状态解出来并冻结，
再在冻结状态上求梯度**（状态是零测度阶跃，收敛后冻结、隐函数定理仍成立）。

  1. dense 批量状态机（mode="dense", dense_status_machine=True）跑极端场景批，
     取一个"批内 CVPIPE 确有关闭、且无 TEMPCLOSED 孤岛"的场景作为基准点；
     并核对该场景的状态向量与 epanet 串行状态机逐元素相等。
  2. 冻结该状态，在 mode="epanet" 的 solver 上做三方梯度对拍：
       A = solve_unrolled 展开 autograd（K = 收敛迭代数 + 5）
       B = ImplicitGGASolve 隐函数伴随
       C = 中心差分（solve_polished 精抛光 + Richardson 外推）
     四类输入 θ ∈ {demand, ke, res_head, r_hw}。
门槛沿用 gradcheck_3way.py：B-C < 1e-6，A-C < 1e-4。

用法： python -X utf8 scripts/gate_b1_grad.py [网名]
"""

import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.parse import parse_inp                              # noqa: E402
from dgga.solver import GGASolver                             # noqa: E402
from dgga.autodiff import (ImplicitGGASolve, solve_polished,   # noqa: E402
                           solve_unrolled)
from gate_b1_batch_sm import make_scenarios, inp_of            # noqa: E402

TOL_BC, TOL_AC = 1e-6, 1e-4
N_COORD = 20
SEED = 2026


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def pick_scenario(net, inp, B=128):
    """挑一个"CVPIPE 真有关闭、但没有 TEMPCLOSED（避免孤岛）"的场景。"""
    D, RH = make_scenarios(net, B, 20260822)
    s_dn = GGASolver(net, mode="dense", inp_path=inp, dense_status_machine=True)
    r = s_dn.solve(D, RH, status_machine=True)
    S = r["status"].numpy().astype(np.int8)
    conv = r["converged"].numpy()
    lt = np.asarray(net.link_type)
    cv = np.where(lt == 0)[0]
    best = None
    for b in range(B):
        if not conv[b] or (S[b] == 1).any() or (S[b] == 0).any():
            continue                     # 排除 TEMPCLOSED/XHEAD（会切出孤岛）
        ncl = int((S[b, cv] <= 2).sum()) if cv.size else 0
        if ncl and (best is None or ncl > best[1]):
            best = (b, ncl)
    if best is None:
        raise RuntimeError("没找到「有 CV 关闭且无 TEMPCLOSED」的场景")
    b = best[0]
    return D[b], RH[b], S[b], best[1], s_dn


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Richmond_skeleton"
    inp = inp_of(name)
    net = parse_inp(inp)
    d0, rh0, S_star, ncl, s_dn = pick_scenario(net, inp)
    s = GGASolver(net, mode="epanet", inp_path=inp)
    # 核对：同一场景，串行状态机与批量状态机的最终状态逐元素相等
    r_ep = s.solve(d0, rh0, status_machine=True)
    same = bool(np.array_equal(r_ep["status"].numpy().astype(np.int8), S_star))
    lt = np.asarray(net.link_type)
    print("=" * 78)
    print(f"=== {name}: Nj={s.Nj} L={s.L}  CVPIPE={int((lt == 0).sum())} "
          f"PUMP={int((lt == 2).sum())} tank={s.n_tanks} ===")
    print(f"  基准场景：CVPIPE 关闭 {ncl} 条，无 TEMPCLOSED/XHEAD；"
          f"批量 vs 串行状态机状态{'相等' if same else '不等'}")
    if not same:
        return 1

    rng = np.random.default_rng(SEED)
    ke0 = np.zeros(net.N)
    em = rng.choice(s.junc_nodes, size=min(40, s.Nj), replace=False)
    ke0[em] = 0.5
    r0 = s.r_hw.detach().cpu().numpy().copy()
    w = rng.normal(size=s.Nj)
    w_t = torch.tensor(w, dtype=torch.float64)
    St = torch.as_tensor(S_star)

    ref = solve_polished(s, d0, rh0, ke0, r0, accuracy=1e-12, max_iter=200,
                         polish_steps=3, status=S_star)
    K = int(np.atleast_1d(ref["iters"])[0]) + 5
    print(f"  冻结状态收敛：iters={int(np.atleast_1d(ref['iters'])[0])} -> K={K}"
          f"  抛光残差 ‖F‖∞={float(ref['resid_inf'][0]):.3e}")

    t = lambda x: torch.tensor(x, dtype=torch.float64, requires_grad=True)
    # ---- B：隐函数伴随 ----
    dB, rhB, keB, rB = t(d0), t(rh0), t(ke0), t(r0)
    head, _, _ = ImplicitGGASolve.apply(dB, rhB, keB, rB, s, 1e-12, 200, 3,
                                        None, None, None, St)
    (w_t * head[s.junc_nodes]).sum().backward()
    gB = dict(demand=dB.grad.numpy(), rh=rhB.grad.numpy(),
              ke=keB.grad.numpy(), r=rB.grad.numpy())
    # ---- A：展开 autograd ----
    dA, rhA, keA, rA = t(d0), t(rh0), t(ke0), t(r0)
    outA = solve_unrolled(s, dA, rhA, ke=keA, r_hw=rA, K=K, status=S_star)
    (w_t * outA["head_ft"][s.junc_nodes]).sum().backward()
    gA = dict(demand=dA.grad.numpy(), rh=rhA.grad.numpy(),
              ke=keA.grad.numpy(), r=rA.grad.numpy())

    def loss_of(d, rh, ke, r):
        sol = solve_polished(s, d, rh, ke, r, accuracy=1e-12, max_iter=200,
                             polish_steps=3, status=S_star)
        return float(w @ sol["head"][0, s.junc_nodes])

    q_base = ref["q"][0]
    hg = s.hexp * r0 * np.abs(q_base) ** (s.hexp - 1.0)
    closed = S_star <= 2
    pipe_ok = np.isin(lt, (0, 1)) & ~closed & (hg > 10.0 * s.rqtol)

    def pool(cand, gvec, n=N_COORD):
        cand = np.asarray(cand)
        ga = np.abs(gvec[cand])
        keep = cand[ga >= np.percentile(ga, 40.0)]
        if keep.size < n:
            keep = cand[np.argsort(-ga)[:n]]
        return [int(i) for i in rng.choice(keep, size=min(n, keep.size),
                                           replace=False)]

    coords = dict(
        demand=pool(s.junc_nodes[d0[s.junc_nodes] > 0], gB["demand"]),
        ke=pool(np.sort(em), gB["ke"]),
        rh=[int(i) for i in s.fixed_nodes],
        r=pool(np.where(pipe_ok)[0], gB["r"]),
    )
    base = dict(demand=d0.copy(), rh=rh0.copy(), ke=ke0, r=r0)
    ok = True
    print(f"{'θ':>8} {'坐标':>8} {'gC(中心差分)':>16} {'relBC':>10} {'relAC':>10}")
    for kind, clist in coords.items():
        wBC, wAC = (-1.0, None, 0.0), (-1.0, None)
        for idx in clist:
            x = base[kind][idx]

            def central(h):
                ap = {k: v.copy() for k, v in base.items()}
                am = {k: v.copy() for k, v in base.items()}
                ap[kind][idx] = x + h
                am[kind][idx] = x - h
                return (loss_of(ap["demand"], ap["rh"], ap["ke"], ap["r"])
                        - loss_of(am["demand"], am["rh"], am["ke"], am["r"])) / (2.0 * h)

            h1 = max(1e-4 * abs(x), 1e-5) if kind == "rh" \
                else max(1e-3 * abs(x), 1e-5)
            gC = (4.0 * central(h1 / 2.0) - central(h1)) / 3.0
            rBC, rAC = rel(gB[kind][idx], gC), rel(gA[kind][idx], gC)
            if rBC > wBC[0]:
                wBC = (rBC, idx, gC)
            if rAC > wAC[0]:
                wAC = (rAC, idx)
            ok = ok and rBC < TOL_BC and rAC < TOL_AC
        print(f"{kind:>8} {wBC[1]:>8} {wBC[2]:>16.8e} {wBC[0]:>10.2e} {wAC[0]:>10.2e}"
              f"{'' if wBC[0] < TOL_BC and wAC[0] < TOL_AC else '  <-- 超限'}")
    print(f"[{name}] 冻结状态三方梯度对拍: {'PASS' if ok else 'FAIL'} "
          f"(门槛 B-C<{TOL_BC:.0e}, A-C<{TOL_AC:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

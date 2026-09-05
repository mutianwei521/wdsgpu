# -*- coding: utf-8 -*-
"""AUDIT-6 (local, CPU): are the defaults bit-identical to the pre-P3 commit?

Runs a fixed battery through whichever dgga package root is handed on argv and
prints a sha256 per output tensor, taken over the RAW BYTES of the float64
buffer.  Run it once against the shadow copy of 4006df6 and once against the
working tree at c827987; every digest must match.

Battery: dense solve, epanet solve, epanet + status machine, assemble='csr'
with the dense linear solver, solve_unrolled gradients, ImplicitGGASolve
gradients, solve_polished.
"""
import hashlib
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
PKG = sys.argv[1]
sys.path.insert(0, PKG)
import torch                                              # noqa: E402
from dgga.parse import parse_inp                          # noqa: E402
from dgga.solver import GGASolver                         # noqa: E402
from dgga.autodiff import (solve_unrolled, ImplicitGGASolve,
                           solve_polished)                # noqa: E402

import dgga.solver as _s
print("# pkg =", os.path.abspath(os.path.dirname(_s.__file__)))
print("# solver.py md5 =", hashlib.md5(open(_s.__file__, "rb").read()).hexdigest())
NETD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "networks", "public")
DT = torch.float64


def h(name, x):
    if x is None:
        print("%-46s NONE" % name)
        return
    a = np.ascontiguousarray(
        x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x))
    print("%-46s %s %-12s %s" % (name, hashlib.sha256(a.tobytes()).hexdigest()[:32],
                                 a.dtype, a.shape))


def boundary(net, s, B, seed):
    g = np.random.default_rng(seed)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + .3 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - .3 * (net.tank_hmax - net.tank_hmin)
        rh0[tn] = np.clip(.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    rh0 = np.nan_to_num(rh0)
    D = d0[None, :] * g.uniform(.85, 1.15, (B, d0.size))
    R = rh0[None, :] + g.uniform(-1., 1., (B, rh0.size))
    ke = np.asarray(net.node_ke, dtype=np.float64).copy()
    jn = np.asarray(s.junc_nodes)
    ke[jn[g.random(jn.size) < .25]] = 1e-3
    KE = np.repeat(ke[None, :], B, 0)
    W = g.normal(0, 1, (B, d0.size))
    return D, R, KE, W


NETS = [("Hanoi", "Hanoi.inp"), ("Net1", "Net1.inp"), ("Net2", "Net2.inp"),
        ("Net3", "Net3.inp"), ("Modena", "Modena.inp"),
        ("Pescara", "Pescara.inp"), ("ky4", "ky4.inp"),
        ("Fossolo_poly1", "Fossolo_poly1.inp"), ("Anytown", "Anytown.inp")]

def one_net(stem, fn):
    p = os.path.join(NETD, fn)
    net = parse_inp(p)
    # ---- epanet mode (the bit-for-bit replica path) ----
    for sm in (False, True):
        try:
            se = GGASolver(net, dtype=DT, mode="epanet", inp_path=p)
            D, R, KE, W = boundary(net, se, 3, 11)
            o = se.solve(D, R, ke_int=KE, status_machine=sm)
            for k in ("head_ft", "flow_cfs", "emitter_cfs", "iters", "relerr"):
                h("epanet/%s/sm=%d/%s" % (stem, sm, k), o[k])
        except Exception as e:
            print("%-46s RAISE %s" % ("epanet/%s/sm=%d" % (stem, sm),
                                      type(e).__name__))
    # ---- dense mode: default, and assemble='csr' + dense solver ----
    for asm in ("dense", "csr"):
        try:
            sd = GGASolver(net, dtype=DT, mode="dense", inp_path=p,
                           dense_tank_bound_check=False)
            D, R, KE, W = boundary(net, sd, 3, 11)
            o = sd.solve(D, R, ke_int=KE, assemble=asm, linear_solver="dense")
            for k in ("head_ft", "flow_cfs", "emitter_cfs", "iters", "relerr"):
                h("dense/%s/asm=%s/%s" % (stem, asm, k), o[k])
        except Exception as e:
            print("%-46s RAISE %s" % ("dense/%s/asm=%s" % (stem, asm),
                                      type(e).__name__))
    # ---- solve_unrolled gradients (default dense/dense) ----
    try:
        sd = GGASolver(net, dtype=DT, mode="dense", inp_path=p,
                       dense_tank_bound_check=False)
        D, R, KE, W = boundary(net, sd, 3, 11)
        for pname in ("d", "rh", "ke", "r"):
            th = dict(d=torch.as_tensor(D, dtype=DT),
                      rh=torch.as_tensor(R, dtype=DT),
                      ke=torch.as_tensor(KE, dtype=DT),
                      r=sd.r_hw.clone())
            th[pname].requires_grad_(True)
            o = solve_unrolled(sd, th["d"], th["rh"], ke=th["ke"],
                               r_hw=th["r"], K=12)
            L = (torch.as_tensor(W, dtype=DT) * o["head_ft"]).sum()
            g, = torch.autograd.grad(L, th[pname])
            h("unrolled/%s/grad_%s" % (stem, pname), g)
        h("unrolled/%s/head" % stem, o["head_ft"])
        h("unrolled/%s/flow" % stem, o["flow_cfs"])
    except Exception as e:
        print("%-46s RAISE %s" % ("unrolled/%s" % stem, type(e).__name__))
    # ---- ImplicitGGASolve gradients ----
    try:
        si = GGASolver(net, dtype=DT, mode="epanet", inp_path=p)
        D, R, KE, W = boundary(net, si, 2, 11)
        for pname in range(4):
            th = [torch.as_tensor(D, dtype=DT), torch.as_tensor(R, dtype=DT),
                  torch.as_tensor(KE, dtype=DT), si.r_hw.clone()]
            th[pname].requires_grad_(True)
            H, Q, E = ImplicitGGASolve.apply(th[0], th[1], th[2], th[3], si,
                                             1e-12, 200, 3, None, None, None,
                                             None)
            L = (torch.as_tensor(W, dtype=DT) * H).sum() + Q.sum()
            g, = torch.autograd.grad(L, th[pname])
            h("implicit/%s/grad_%d" % (stem, pname), g)
        h("implicit/%s/H" % stem, H)
        h("implicit/%s/Q" % stem, Q)
        h("implicit/%s/E" % stem, E)
    except Exception as e:
        print("%-46s RAISE %s" % ("implicit/%s" % stem, type(e).__name__))
    # ---- solve_polished ----
    try:
        sp = GGASolver(net, dtype=DT, mode="epanet", inp_path=p)
        D, R, KE, W = boundary(net, sp, 2, 11)
        r = solve_polished(sp, D, R, ke=KE)
        for k in ("q", "e_j", "Hj", "head", "flow", "emitter", "resid_inf"):
            if k in r:
                h("polished/%s/%s" % (stem, k), r[k])
    except Exception as e:
        print("%-46s RAISE %s" % ("polished/%s" % stem, type(e).__name__))


for _stem, _fn in NETS:
    try:
        one_net(_stem, _fn)
    except Exception as _e:
        print("%-46s NET-LEVEL RAISE %s %s"
              % ("net/%s" % _stem, type(_e).__name__, str(_e)[:60]))

print("SHADOW BATTERY DONE")

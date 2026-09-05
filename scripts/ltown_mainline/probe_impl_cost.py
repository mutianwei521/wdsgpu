# -*- coding: utf-8 -*-
"""探针：L-TOWN 上 ImplicitGGASolve f+b 的 CPU 单样本成本（为集群作业定预算）。"""
import os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import torch
from dgga.parse import parse_inp
from dgga.solver import GGASolver
from dgga.autodiff import implicit_solve

inp = os.path.join(ROOT, "networks", "public", "_cleaned", "L-TOWN.inp")
net = parse_inp(inp)
se = GGASolver(net, mode="epanet", inp_path=inp)
d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
tn = np.asarray(net.tank_node, dtype=np.int64)
if tn.size:
    rh0[tn] = .5*(np.asarray(net.tank_hmin)[:tn.size]+np.asarray(net.tank_hmax)[:tn.size])
base = se.run_gga(d0, rh0, do_status=True)
S0, K0 = base["status"].copy(), base["setting"].copy()
print("base iters=%d conv=%s" % (int(base["iters"]), bool(base["converged"])))

g = np.random.default_rng(2026)
for B in (1, 8):
    D = torch.as_tensor(d0[None,:]*g.uniform(.85,1.15,(B,d0.size)), dtype=torch.float64).requires_grad_(True)
    R = torch.as_tensor(rh0[None,:]+g.uniform(-1.,1.,(B,rh0.size)), dtype=torch.float64)
    W = torch.as_tensor(np.random.default_rng(7).normal(0,1,(B,net.N)), dtype=torch.float64)
    t0 = time.perf_counter()
    h, q, e = implicit_solve(se, D, R, speed=K0, status=S0)
    tf = time.perf_counter()-t0
    t0 = time.perf_counter()
    (h*W).sum().backward()
    tb = time.perf_counter()-t0
    print("B=%d  impl fwd %.3fs (%.1f ms/样本)  bwd %.3fs (%.1f ms/样本)  |g|=%.4e"
          % (B, tf, tf*1e3/B, tb, tb*1e3/B, float(D.grad.norm())))

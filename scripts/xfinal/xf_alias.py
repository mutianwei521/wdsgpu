# -*- coding: utf-8 -*-
"""XF-7：_cudss_load 的两条分支（alias / 逐行 copy）在真机上到底走哪一条。
T4 量的是传给 _cudss_load 的实参，看不见它内部；所以至少要知道内部走的是哪条路。"""
import os, sys
import numpy as np, torch
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dgga.parse import parse_inp
from dgga.solver import GGASolver
DT = torch.float64
ND = os.path.join(HERE, "p2nets")
print("=" * 90)
print("XF-7 _cudss_load 分支 | 节点 %s" % os.popen("hostname").read().strip())
print("=" * 90)
for stem, fn in (("Hanoi","Hanoi.inp"),("Net3","Net3.inp"),("Modena","Modena.inp"),
                 ("City_D","City_D.inp"),("ky4","ky4.inp")):
    for B in (1, 8, 64):
        p = os.path.join(ND, fn)
        net = parse_inp(p)
        s = GGASolver(net, device="cuda", dtype=DT, mode="dense", inp_path=p,
                      dense_tank_bound_check=False)
        g = np.random.default_rng(20260822)
        d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
        rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
        tn = np.asarray(net.tank_node, dtype=np.int64)
        if tn.size:
            rh0[tn] = .5*(np.asarray(net.tank_hmin)[:tn.size]+np.asarray(net.tank_hmax)[:tn.size])
        rh0 = np.nan_to_num(rh0)
        D = torch.as_tensor(d0[None,:]*g.uniform(.9,1.1,(B,d0.size)), dtype=DT, device="cuda")
        R = torch.as_tensor(np.repeat(rh0[None,:],B,0)+g.uniform(-1.,1.,(B,rh0.size))
                            *(np.asarray(net.node_type)!=0)[None,:], dtype=DT, device="cuda")
        with torch.no_grad():
            s.solve(D, R, assemble="csr", linear_solver="cudss")
        al = [(k[0], v["alias"]) for k, v in s._cudss_cache.items()]
        print("  %-8s B=%-4d | state 数=%d | alias=%s"
              % (stem, B, len(al), al))
        s.cudss_free(); del s
print("=" * 90)

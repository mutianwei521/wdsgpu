# -*- coding: utf-8 -*-
"""a5_worker.py <tree_dir> <nets_dir> - 在指定 dgga 树上跑缺省通路电池，
打印 key=sha256（stdout 机读）。缺省逐位审计用：mode='epanet'、
assemble='dense'、linear_solver='dense'、ImplicitGGASolve CPU 伴随缺省参数。
"""
import hashlib
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
TREE, NETS = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.abspath(TREE))
import dgga                                       # noqa: E402
assert os.path.dirname(os.path.abspath(dgga.__file__)) == \
    os.path.join(os.path.abspath(TREE), "dgga"), dgga.__file__
import torch                                      # noqa: E402
from dgga.parse import parse_inp                  # noqa: E402
from dgga.solver import GGASolver                 # noqa: E402
from dgga.autodiff import implicit_solve, solve_polished  # noqa: E402

torch.use_deterministic_algorithms(True)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:24]


CASES = [
    ("Hanoi", "public/Hanoi.inp", False),
    ("Modena", "public/Modena.inp", False),
    ("city_d", "realInpData/city_d.inp", False),
    ("L-TOWN", "public/_cleaned/L-TOWN.inp", True),
    ("BWSN_1", "public/_cleaned/BWSN_Network_1.inp", True),
]

for name, relp, sm in CASES:
    inp = os.path.join(NETS, *relp.split("/"))
    if not os.path.exists(inp):
        print("%s|MISSING" % name)
        continue
    net = parse_inp(inp)
    se = GGASolver(net, mode="epanet", inp_path=inp)
    rng = np.random.default_rng(20260824)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                 dtype=np.float64))
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5 * (np.asarray(net.tank_hmin)[:tn.size]
                        + np.asarray(net.tank_hmax)[:tn.size])
    ke0 = np.asarray(net.node_ke, dtype=np.float64).copy()
    jn = np.asarray(se.junc_nodes)
    ke0[jn[rng.integers(0, jn.size, 3)]] = 1e-3
    r0 = se.r_hw.detach().cpu().numpy().copy()
    kw = {}
    if sm:
        base = se.run_gga(d0, rh0, ke=ke0, do_status=True)
        kw = dict(speed=base["setting"].copy(), status=base["status"].copy())
        print("%s|frozen_status=%s" % (name, sha(base["status"])))
    # ① epanet 前向（solve_polished 缺省）
    o = solve_polished(se, d0, rh0, ke=ke0, r_hw=r0, **kw)
    for k in ("head", "q", "emitter"):
        print("%s|ep_fwd_%s=%s" % (name, k, sha(o[k])))
    # ② CPU 伴随缺省（B=1 与 B=4）
    dt = torch.float64
    for B in (1, 4):
        if B == 1:
            Db = d0.copy()
            Rb, KEb = rh0.copy(), ke0.copy()
        else:
            Db = d0[None, :] * rng.uniform(0.9, 1.1, (B, d0.size))
            Rb = np.repeat(rh0[None, :], B, 0)
            KEb = np.repeat(ke0[None, :], B, 0)
        D = torch.as_tensor(Db, dtype=dt).requires_grad_(True)
        R = torch.as_tensor(Rb, dtype=dt).requires_grad_(True)
        KE = torch.as_tensor(KEb, dtype=dt).requires_grad_(True)
        RW = torch.as_tensor(r0, dtype=dt).requires_grad_(True)
        h, f, e = implicit_solve(se, D, R, ke=KE, r_hw=RW, **kw)
        WH = torch.as_tensor(rng.normal(0, 1, h.shape[-1]), dtype=dt)
        WQ = torch.as_tensor(rng.normal(0, 1, f.shape[-1]), dtype=dt)
        WE = torch.as_tensor(rng.normal(0, 1, e.shape[-1]), dtype=dt)
        ((h * WH).sum() + (f * WQ).sum() + (e * WE).sum()).backward()
        for k, t in (("gd", D), ("grh", R), ("gke", KE), ("grw", RW)):
            print("%s|grad_B%d_%s=%s" % (name, B, k, sha(t.grad.numpy())))
        for k, t in (("h", h), ("q", f), ("e", e)):
            print("%s|out_B%d_%s=%s" % (name, B, k,
                                        sha(t.detach().numpy())))
    # ③ dense 前向缺省（assemble='dense', linear_solver='dense'）
    sd = GGASolver(net, mode="dense", inp_path=inp,
                   **(dict(dense_status_machine=True) if sm else {}))
    with torch.no_grad():
        od = sd.solve(np.atleast_2d(d0), np.atleast_2d(rh0),
                      ke_int=np.atleast_2d(ke0), accuracy=3e-8, max_iter=200,
                      status_machine=sm)
    for k in ("head_ft", "flow_cfs", "emitter_cfs"):
        print("%s|dn_fwd_%s=%s" % (name, k, sha(od[k].cpu().numpy())))
    print("%s|dn_iters=%s" % (name, int(od["iters"].max())))
print("WORKER DONE")

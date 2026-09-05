# -*- coding: utf-8 -*-
"""aud_grad_spot.py - 敌意抽查（自写）：L-TOWN ImplicitGGASolve 梯度 vs 中央差分。

J = Σ_j w_j · head_j（全 junction，确定性权重）。
θ=demand 抽 4 坐标（3 个 PRV 下游节点 + 1 个大需水节点）。
FD 基线 = mode="epanet" run_gga(do_status=True, hacc=1e-12)（完整串行状态机，
非冻结）；±h 点终态状态向量 != 基态即弃该坐标（切换点上 FD 无意义）。
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp               # noqa: E402
from dgga.solver import GGASolver              # noqa: E402
from dgga.autodiff import implicit_solve, solve_polished   # noqa: E402

net = parse_inp(os.path.join(ROOT, "networks", "public", "_cleaned",
                             "L-TOWN.inp"))
d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
tn = np.asarray(net.tank_node, dtype=np.int64)
rh[tn] = 0.5 * (np.asarray(net.tank_hmin) + np.asarray(net.tank_hmax))

se = GGASolver(net, mode="epanet")
lt = np.asarray(net.link_type)
prv = np.where(lt == 3)[0]
coords = [int(net.link_n2[k]) for k in prv]
junc = np.where(np.asarray(net.node_type) == 0)[0]
coords.append(int(junc[np.argmax(d[junc])]))

w = np.linspace(0.5, 1.5, se.N)
base = se.run_gga(d, rh, do_status=True)      # 状态机跑到 INP ACCURACY 停机
S0 = base["status"].copy()
print("基态 iters=%d converged=%s ACTIVE PRV=%d/3"
      % (base["iters"], base["converged"], int((S0[prv] == 4).sum())))

dg = torch.as_tensor(d)[None, :].clone().requires_grad_(True)
head, flow, em = implicit_solve(se, dg, torch.as_tensor(rh)[None, :],
                                status=S0)     # 冻结 S* 的隐式伴随
J = (head[0] * torch.as_tensor(w)).sum()
J.backward()
g = dg.grad[0].numpy()

worst = 0.0
n_skip = 0
for j in coords:
    h = max(1e-3 * abs(d[j]), 1e-4)   # J~1e5 ft：h 太小时抛光残差噪声盖过 FD
    gs = []
    ok = True
    for sgn in (+1, -1):
        dd = d.copy()
        dd[j] += sgn * h
        o = se.run_gga(dd, rh, do_status=True)     # 状态机复跑：终态必须仍是 S*
        if not np.array_equal(o["status"], S0):
            ok = False
            break
        pol = solve_polished(se, dd, rh, status=S0)   # 冻结 S* 精抛光作 FD 前向
        gs.append(float((pol["head"][0] * w).sum()))
    if not ok:
        print("  coord %d 切换点，弃用" % j)
        n_skip += 1
        continue
    fd = (gs[0] - gs[1]) / (2 * h)
    rel = abs(g[j] - fd) / max(abs(fd), 1e-12)
    worst = max(worst, rel)
    tag = "PRV下游" if j in coords[:3] else "最大需水"
    print("  coord %-5d (%s) impl=%+.10e fd=%+.10e rel=%.3e" % (
        j, tag, g[j], fd, rel))

ok = worst < 1e-6 and n_skip < len(coords)
print("AUD_GRAD_SPOT worst_rel=%.3e skip=%d %s"
      % (worst, n_skip, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)

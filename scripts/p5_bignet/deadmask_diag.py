# -*- coding: utf-8 -*-
"""对拍残差的归因：max|ΔH| 落在**活节点**还是**死支**上。

发现（本轮）：ky13 上 dense~epanet 的 max|ΔH| 高达 1.53 ft，但 max|ΔQ| 只有
1.2e-7 cfs - 说明差异不在方程解上，而落在流量被连续性钉死为 0 的死支：
那里的水头由奇异行的零空间决定，两种线性求解器给的答案本就不同，**不是通路
错误**。本脚本把死支节点掩掉后重报 max|ΔH|，把这件事量化。

用法：python -X utf8 scripts/p5_bignet/deadmask_diag.py --inp <path> [--no-guard]
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from p5lib import base_case  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dgga.calib import dead_branch_mask  # noqa: E402
from dgga.parse import parse_inp  # noqa: E402
from dgga.solver import GGASolver  # noqa: E402


def np_(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def diag(inp, no_guard=False, acc=None, trials=None):
    net = parse_inp(inp)
    nt = np.asarray(net.node_type)
    s_d = GGASolver(net, mode="dense", inp_path=inp,
                    dense_tank_bound_check=not no_guard)
    d, rh = base_case(net)
    kw = {}
    if acc is not None:
        kw["accuracy"] = acc
    if trials is not None:
        kw["max_iter"] = trials
    Hd = np_(s_d.solve(d, rh, **kw)["head_ft"])
    s_e = GGASolver(net, mode="epanet", inp_path=inp)
    He = np_(s_e.solve(d, rh, **kw)["head_ft"])

    dead_l = dead_branch_mask(s_d, demand=d[None, :], ke=None)
    n1, n2 = np.asarray(s_d.n1_np), np.asarray(s_d.n2_np)
    # 死节点：只被死支触及、且自身无需水、非定水头
    touched_alive = np.zeros(net.N, dtype=bool)
    live_l = (~dead_l) & (~np.asarray(s_d.closed_np))
    touched_alive[n1[live_l]] = True
    touched_alive[n2[live_l]] = True
    dead_n = (~touched_alive) & (nt == 0)

    jm = nt == 0
    dH = np.abs(Hd - He)
    live_j = jm & ~dead_n
    name = os.path.basename(inp)
    print(f"{name:<16} Nj={int(jm.sum()):>6} 死支链路={int(dead_l.sum()):>5} "
          f"死节点={int(dead_n.sum()):>5}")
    print(f"    max|ΔH| 全 junction  = {dH[jm].max():.6e} ft "
          f"(节点 {net.node_id[int(np.argmax(np.where(jm, dH, -1)))]})")
    if live_j.any():
        print(f"    max|ΔH| 仅活 junction = {dH[live_j].max():.6e} ft "
              f"(节点 {net.node_id[int(np.argmax(np.where(live_j, dH, -1)))]})")
    if dead_n.any():
        print(f"    max|ΔH| 仅死 junction = {dH[dead_n].max():.6e} ft")
    return dH[live_j].max() if live_j.any() else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", action="append", required=True)
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--accuracy", type=float, default=None)
    ap.add_argument("--trials", type=int, default=None)
    a = ap.parse_args()
    for p in a.inp:
        try:
            diag(os.path.join(ROOT, p) if not os.path.isabs(p) else p,
                 no_guard=a.no_guard, acc=a.accuracy, trials=a.trials)
        except Exception as e:
            print(f"{os.path.basename(p)}: {type(e).__name__}: {str(e)[:160]}")

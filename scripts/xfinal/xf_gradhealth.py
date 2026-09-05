# -*- coding: utf-8 -*-
"""XF-6（本机前台，CPU）：unrolled demand 梯度的三网限制，我自己量一遍。

README/CONTRACT 说 City_D / ky4 / Net3 上 `solve_unrolled` 的截断-K demand 梯度
不随 K 收敛。这里直接调 `unrolled_grad_health`（它是被测代码，不是我重实现的
判据），在能在本机 CPU 建起来的网上各量一次，看那个"不建议"是不是真会亮。
"""
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isdir(os.path.join(ROOT, "dgga")):
    ROOT = os.getcwd()
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                              # noqa: E402
from dgga.solver import GGASolver                             # noqa: E402
from dgga.autodiff import unrolled_grad_health                # noqa: E402

NETS = [("Net1", "networks/public/Net1.inp"),
        ("Hanoi", "networks/public/Hanoi.inp"),
        ("Net2", "networks/public/Net2.inp"),
        ("Net3", "networks/public/Net3.inp"),
        ("Modena", "networks/public/_cleaned/Modena.inp"),
        ("Pescara", "networks/public/_cleaned/Pescara.inp"),
        ("City_D", "datasets/city_d.inp"),
        ("ky4", "networks/public/ky4.inp")]


def main():
    print("=" * 96)
    print("XF-6 unrolled demand 梯度可用性（被测代码自己的 unrolled_grad_health）")
    print("=" * 96)
    for stem, rel in NETS:
        p = os.path.join(ROOT, *rel.split("/"))
        if not os.path.isfile(p):
            print("  %-8s 文件不在：%s" % (stem, rel))
            continue
        t0 = time.time()
        try:
            net = parse_inp(p)
            s = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                          inp_path=p, dense_tank_bound_check=False)
            d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
            rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
            tn = np.asarray(net.tank_node, dtype=np.int64)
            if tn.size:
                rh0[tn] = 0.5 * (np.asarray(net.tank_hmin)[:tn.size]
                                 + np.asarray(net.tank_hmax)[:tn.size])
            rh0 = np.nan_to_num(rh0)
            h = unrolled_grad_health(s, d0, rh0, param="demand", dK=5, seed=0)
            print("  %-8s Nj=%-4d K=%-3d rel_K=%.3e  判定=%s   |g|max=%.3e  (%.0fs)"
                  % (stem, s.Nj, h["K"], h["rel_K"], h["verdict"],
                     h.get("gmax", float("nan")), time.time() - t0))
        except Exception as e:                                # noqa: BLE001
            print("  %-8s 建不起来/跑不动：%s" % (stem, repr(e)[:90]))
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())

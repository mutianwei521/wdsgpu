# -*- coding: utf-8 -*-
"""显存独立复核：**每个配置一个全新进程**，进程里只做这一件事。

用法: python3 adv_mem.py <net> <B> <dense|cudss>
输出一行：torch 峰值(首解/次解) + 设备级峰值（后台线程 2ms 采样 cudaMemGetInfo）
"""
import os
import sys
import threading
import time
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                       # noqa: E402
from dgga.solver import GGASolver                      # noqa: E402

FILES = {"Net1": "Net1.inp", "Anytown": "Anytown.inp", "Hanoi": "Hanoi.inp",
         "Net2": "Net2.inp", "Fossolo": "Fossolo_poly1.inp",
         "Pescara": "Pescara.inp", "Net3": "Net3.inp", "Modena": "Modena.inp",
         "City_D": "City_D.inp", "ky4": "ky4.inp"}
DT = torch.float64
stem, B, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
f = os.path.join(ROOT, "p2nets", FILES[stem])


def dev_used():
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


class Poll(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.pk = 0.0

    def run(self):
        while not self.stop:
            self.pk = max(self.pk, dev_used())
            time.sleep(0.002)


try:
    net = parse_inp(f)
    s = GGASolver(net, device="cuda", dtype=DT, mode="dense", inp_path=f,
                  dense_tank_bound_check=False)
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    g = np.random.default_rng(20260822 + B)
    D = torch.as_tensor(d[None, :] * g.lognormal(0.0, 0.22, (B, d.size)),
                        dtype=DT, device="cuda")
    R = rh[None, :] + g.normal(0.0, 1.5, (B, rh.size))
    R = torch.as_tensor(np.where(np.isnan(rh)[None, :], np.nan, R),
                        dtype=DT, device="cuda")
    kw = {} if path == "dense" else dict(assemble="csr", linear_solver="cudss")
    torch.cuda.synchronize()
    base_dev = dev_used()
    base_alloc = torch.cuda.memory_allocated() / 2 ** 20
    p = Poll()
    p.start()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    o = s.solve(D, R, **kw)                       # 首解（含 plan / 首次分配）
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    pk1 = torch.cuda.max_memory_allocated() / 2 ** 20
    torch.cuda.reset_peak_memory_stats()
    s.solve(D, R, **kw)                           # 次解（稳态）
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    pk2 = torch.cuda.max_memory_allocated() / 2 ** 20
    p.stop = True
    p.join()
    print("MEM %-8s B=%-5d %-6s | torch峰值 首解=%9.2f 次解=%9.2f MiB | "
          "基线alloc=%7.2f | 设备级 起=%8.1f 峰=%8.1f 终=%8.1f MiB | "
          "首解=%8.1f ms 次解=%8.1f ms | iters=%d"
          % (stem, B, path, pk1, pk2, base_alloc, base_dev, p.pk, dev_used(),
             (t1 - t0) * 1e3, (t2 - t1) * 1e3, int(o["iters"].max())))
except Exception:                                  # noqa: BLE001
    print("MEM %-8s B=%-5d %-6s | %s" %
          (stem, B, path, traceback.format_exc().strip().split("\n")[-1][:120]))

# -*- coding: utf-8 -*-
"""P2c：可复现性。两次 P2 全量跑（1459212 / 1459222）的 §1 数字对不上
（例：Pescara B=1 3.176e-06 vs 7.813e-07），而稠密通路在 determinism 下自重跑
恒为 0。到底是 cuDSS 本身跨调用/跨 plan 不可复现，还是我们的 state 有残留？
本脚本把三种"重跑"分开量：
  (a) 同一个 DirectSolver 实例、同一 plan，连解两次
  (b) 同一进程内换新 solver 实例（重新 plan 一次）
  (c) 稠密通路自重跑（对照，应恒为 0）
"""
import os
import sys
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                      # noqa: E402
from dgga.solver import GGASolver                     # noqa: E402

DT = torch.float64
NETDIR = os.path.join(ROOT, "p2nets")
NETS = [("Net1", "Net1.inp"), ("Anytown", "Anytown.inp"), ("Hanoi", "Hanoi.inp"),
        ("Net2", "Net2.inp"), ("Fossolo", "Fossolo_poly1.inp"),
        ("Pescara", "Pescara.inp"), ("Net3", "Net3.inp"), ("Modena", "Modena.inp"),
        ("City_D", "City_D.inp"), ("ky4", "ky4.inp")]
NMAP = dict(NETS)


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, rh


def batchify(d, rh, B, seed, dev):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.6, 1.4, (B, 1)) * g.uniform(0.75, 1.25, (B, d.size))
    R = rh[None, :] + g.uniform(-2.0, 2.0, (B, rh.size))
    R = np.where(np.isnan(rh)[None, :], np.nan, R)
    return (torch.as_tensor(D, dtype=DT, device=dev),
            torch.as_tensor(R, dtype=DT, device=dev))


def mk(stem, dev):
    f = os.path.join(NETDIR, NMAP[stem])
    net = parse_inp(f)
    return net, GGASolver(net, device=dev, dtype=DT, mode="dense", inp_path=f,
                          dense_tank_bound_check=False)


print(torch.cuda.get_device_name(0), "| torch", torch.__version__)
torch.use_deterministic_algorithms(True)
print("use_deterministic_algorithms(True)\n")
print("=" * 100)
print("%-9s %6s %4s | %13s %13s | %13s | %13s" %
      ("net", "Nj", "B", "cudss 同实例", "cudss 换实例", "dense 自重跑",
       "|cudss-dense|"))
for stem, _ in NETS:
    for B in (8, 64):
        try:
            net, s = mk(stem, "cuda")
            d, rh = boundary(net)
            D, R = batchify(d, rh, B, 1234 + B, "cuda")     # 与 p2.py §1 同种子
            kw = dict(assemble="csr", linear_solver="cudss")
            h1 = s.solve(D, R, **kw)["head_ft"]
            h2 = s.solve(D, R, **kw)["head_ft"]
            _, s2 = mk(stem, "cuda")
            h3 = s2.solve(D, R, **kw)["head_ft"]
            g1 = s.solve(D, R)["head_ft"]
            g2 = s.solve(D, R)["head_ft"]
            print("%-9s %6d %4d | %13.3e %13.3e | %13.3e | %13.3e" %
                  (stem, s.Nj, B, float((h1 - h2).abs().max()),
                   float((h1 - h3).abs().max()), float((g1 - g2).abs().max()),
                   float((h1 - g1).abs().max())))
            del s, s2, net
            torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            print("%-9s %4d | %s" % (stem, B,
                  traceback.format_exc().strip().split("\n")[-1][:90]))
            torch.cuda.empty_cache()
print("\nP2c 结束")

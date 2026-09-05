# -*- coding: utf-8 -*-
"""AUDIT 3b - 显存抽查（R5 口径，我自己写的一份，每配置全新进程）。

口径（照抄审计 R5 的**定义**，代码是我自己的）：
  ctx      = CUDA 初始化后、任何张量之前的设备常驻
  peak     = torch.cuda.max_memory_allocated
  resv     = torch.cuda.max_memory_reserved
  nontorch = 收尾时设备已用 − torch reserved − ctx
  total    = resv + nontorch
量的是 fwd+bwd：solve_unrolled(K) + backward(demand)。
用法：python aud_mem_gpu.py <inp> <B> <dense|cudss>
"""
import os
import sys
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NETD = os.path.join(HERE, "p2nets")
DT = torch.float64
SEED = 2026


def mib_used():
    torch.cuda.synchronize()
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


def main(fn, B, mode):
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    from dgga.autodiff import solve_unrolled
    p = os.path.join(NETD, fn)
    net = parse_inp(p)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + .30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - .30 * (net.tank_hmax - net.tank_hmin)
        rh0[tn] = np.clip(.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    rh0 = np.nan_to_num(rh0)
    g0 = np.random.default_rng(SEED)
    Dp = d0[None, :] * g0.uniform(.85, 1.15, (8, d0.size))
    Rp = rh0[None, :] + g0.uniform(-1., 1., (8, rh0.size))
    scpu = GGASolver(net, device="cpu", dtype=DT, mode="dense", inp_path=p,
                     dense_tank_bound_check=False)
    with torch.no_grad():
        K = int(scpu.solve(torch.as_tensor(Dp, dtype=DT),
                           torch.as_tensor(Rp, dtype=DT))["iters"].max())
    del scpu
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    ctx = mib_used() - torch.cuda.memory_reserved() / 2 ** 20
    s = GGASolver(net, device="cuda", dtype=DT, mode="dense", inp_path=p,
                  dense_tank_bound_check=False)
    g = np.random.default_rng(SEED)
    D = torch.as_tensor(d0[None, :] * g.uniform(.85, 1.15, (B, d0.size)),
                        dtype=DT, device="cuda")
    R = torch.as_tensor(rh0[None, :] + g.uniform(-1., 1., (B, rh0.size)),
                        dtype=DT, device="cuda")
    W = torch.as_tensor(np.random.default_rng(7).normal(0, 1, (B, net.N)),
                        dtype=DT, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        dv = D.clone().requires_grad_(True)
        asm = "csr" if mode == "cudss" else "dense"
        o = solve_unrolled(s, dv, R, K=K, assemble=asm, linear_solver=mode)
        (W * o["head_ft"]).sum().backward()
        torch.cuda.synchronize()
        pk = torch.cuda.max_memory_allocated() / 2 ** 20
        rv = torch.cuda.max_memory_reserved() / 2 ** 20
        nt = mib_used() - torch.cuda.memory_reserved() / 2 ** 20 - ctx
        print("AUDMEM %s B=%d %s ok K=%d ctx=%.2f peak=%.2f resv=%.2f "
              "nontorch=%.2f total=%.2f gnorm=%.10e"
              % (fn, B, mode, K, ctx, pk, rv, nt, rv + nt, float(dv.grad.norm())))
    except torch.OutOfMemoryError:
        print("AUDMEM %s B=%d %s OOM K=%d ctx=%.2f" % (fn, B, mode, K, ctx))
    except Exception:                                             # noqa: BLE001
        print("AUDMEM %s B=%d %s ERR %s"
              % (fn, B, mode, traceback.format_exc().strip()
                 .split("\n")[-1][:120]))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])

# -*- coding: utf-8 -*-
"""P3-C 批量扫描：前向+反向的倍数曲线与 dense 的 OOM 边界。

P3-B §1 只取了 B=64/256，而 ky4 B=256 的 dense 侧直接 OOM - 光看那一行读不出
"倍数随 B 怎么走"，也读不出"dense 到哪一档就跑不动了"。这里把 B 扫开。
另附：同一节点上的**纯前向**倍数（与 P2 的口径对齐）与每轮预算复核。
"""
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                     # noqa: E402
from dgga.solver import GGASolver                    # noqa: E402
from dgga.autodiff import solve_unrolled             # noqa: E402

DEV = "cuda"
DT = torch.float64
NETDIR = os.path.join(ROOT, "p2nets")
NETS = [("Net3", "Net3.inp"), ("Modena", "Modena.inp"),
        ("City_D", "City_D.inp"), ("ky4", "ky4.inp")]
NMAP = dict(NETS)
SEED = 2026


def last():
    return traceback.format_exc().strip().split("\n")[-1][:110]


def mk(stem):
    f = os.path.join(NETDIR, NMAP[stem])
    net = parse_inp(f)
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=f,
                  dense_tank_bound_check=False)
    return net, s


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, rh


def batchify(d, rh, B, seed=SEED):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.8, 1.2, (B, 1)) * g.uniform(0.9, 1.1, (B, d.size))
    R = np.nan_to_num(rh)[None, :] + g.uniform(-1.0, 1.0, (B, rh.size))
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


def tg(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def make_run(s, D, R, KE, w, K, ls, grad, slots=None):
    kw = dict(assemble="csr", linear_solver="cudss") if ls == "cudss" else {}
    if ls == "cudss" and slots:
        s.cudss_cache_max = max(8, slots)
        s.cudss_grad_slots = slots

    def run():
        if ls == "cudss":
            s._cudss_slot_rr = 0
        if not grad:
            with torch.no_grad():
                solve_unrolled(s, D, R, ke=KE, K=K, **kw)
            return
        d = D.clone().requires_grad_(True)
        rh = R.clone().requires_grad_(True)
        out = solve_unrolled(s, d, rh, ke=KE, K=K, **kw)
        (w * out["head_ft"]).sum().backward()
    return run


print("=" * 78)
print("node:", os.popen("hostname").read().strip(), "| torch", torch.__version__,
      "|", torch.cuda.get_device_name(0), "| cpus", os.cpu_count())
import nvmath                                        # noqa: E402
print("nvmath", nvmath.__version__)

print()
print("=" * 78)
print("批量扫描：solve_unrolled 前向+反向 ms/场景（best of 5，warm 2，slots=K+1）")
print("   OOM = dense 侧 torch.OutOfMemoryError（cudss 侧同配置照常出梯度）")
print("net      Nj   K  B     | dense f+b   cudss f+b  倍数  | dense fwd  cudss fwd 倍数")
for stem, fn in NETS:
    try:
        net, s = mk(stem)
    except Exception:                                # noqa: BLE001
        print("[skip] %s: %s" % (stem, last()))
        continue
    d0, rh0 = boundary(net)
    ke = np.zeros(net.N)
    ke[np.random.default_rng(1).choice(s.junc_nodes, size=min(40, s.Nj),
                                       replace=False)] = 0.5
    with torch.no_grad():
        D1, R1 = batchify(d0, rh0, 4)
        o = s.solve(D1, R1, ke_int=torch.as_tensor(
            np.broadcast_to(ke[None, :], (4, net.N)).copy(), dtype=DT, device=DEV))
    K = int(o["iters"].max())
    for B in (1, 8, 32, 64, 128, 256, 512, 1024):
        try:
            D, R = batchify(d0, rh0, B)
            KE = torch.as_tensor(np.broadcast_to(ke[None, :], (B, net.N)).copy(),
                                 dtype=DT, device=DEV)
            w = torch.as_tensor(np.random.default_rng(3).normal(size=(B, net.N)),
                                dtype=DT, device=DEV)
            s.cudss_free(empty_cache=True)
            tc = tg(make_run(s, D, R, KE, w, K, "cudss", True,
                             slots=min(K + 1, 16))) * 1e3 / B
            tcf = tg(make_run(s, D, R, KE, w, K, "cudss", False)) * 1e3 / B
            try:
                td = tg(make_run(s, D, R, KE, w, K, "dense", True)) * 1e3 / B
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                td = float("nan")
            try:
                tdf = tg(make_run(s, D, R, KE, w, K, "dense", False)) * 1e3 / B
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                tdf = float("nan")
            print("%-8s %-4d %-2d %-5d | %10s %11.5f %6s | %10s %9.5f %s"
                  % (stem, s.Nj, K, B,
                     ("OOM" if td != td else "%.5f" % td), tc,
                     ("-" if td != td else "%.2fx" % (td / tc)),
                     ("OOM" if tdf != tdf else "%.5f" % tdf), tcf,
                     ("-" if tdf != tdf else "%.2fx" % (tdf / tcf))))
            s.cudss_free(empty_cache=True)
            del D, R, KE, w
            torch.cuda.empty_cache()
        except Exception:                            # noqa: BLE001
            print("[skip] %s B=%d: %s" % (stem, B, last()))
            torch.cuda.empty_cache()
    s.cudss_free(empty_cache=True)
    del s, net
    torch.cuda.empty_cache()

print("P3C DONE")

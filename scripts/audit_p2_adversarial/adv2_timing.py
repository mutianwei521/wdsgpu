# -*- coding: utf-8 -*-
"""复核加速数字的分歧：我测 ky4 B=256 = 6.2x，上游报 8.96x。
同一进程、同一节点内做 A/B：场景发生器 / 计时协议 / 计数器包装 / 顺序 / 预热长度，
再把 cuDSS 每次调用的地板量出来（对照 p2b §C）。
"""
import gc
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                       # noqa: E402
from dgga.solver import GGASolver                      # noqa: E402

DT = torch.float64
NETDIR = os.path.join(ROOT, "p2nets")
FILES = {"Modena": "Modena.inp", "City_D": "City_D.inp", "ky4": "ky4.inp",
         "Net3": "Net3.inp"}

print("node:", os.uname().nodename, "| nproc:",
      subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip(),
      "| os.cpu_count:", os.cpu_count(),
      "| affinity:", len(os.sched_getaffinity(0)))
print("gpu:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
try:
    mdl = [l for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    print("cpu:", mdl.split(":", 1)[1].strip())
except Exception:                                       # noqa: BLE001
    pass
print("clocks:", subprocess.run(
    ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,"
     "power.draw,power.limit", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip())


def mk(stem):
    f = os.path.join(NETDIR, FILES[stem])
    net = parse_inp(f)
    s = GGASolver(net, device="cuda", dtype=DT, mode="dense", inp_path=f,
                  dense_tank_bound_check=False)
    return net, s


def bc(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, rh


def scen_mine(d, rh, B, seed):
    g = np.random.default_rng(20260822 + seed)
    D = d[None, :] * g.lognormal(0.0, 0.22, (B, d.size))
    R = rh[None, :] + g.normal(0.0, 1.5, (B, rh.size))
    R = np.where(np.isnan(rh)[None, :], np.nan, R)
    return (torch.as_tensor(D, dtype=DT, device="cuda"),
            torch.as_tensor(R, dtype=DT, device="cuda"))


def scen_upstream(d, rh, B, seed):
    """逐字复刻上游 p2.py 的 batchify（只为定位分歧，不是复用其结论）。"""
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.6, 1.4, (B, 1)) * g.uniform(0.75, 1.25, (B, d.size))
    R = rh[None, :] + g.uniform(-2.0, 2.0, (B, rh.size))
    R = np.where(np.isnan(rh)[None, :], np.nan, R)
    return (torch.as_tensor(D, dtype=DT, device="cuda"),
            torch.as_tensor(R, dtype=DT, device="cuda"))


def tg_upstream(fn, reps=5, warm=2):
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


def tg_mine(fn, reps=9, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


print("\n" + "=" * 110)
print("[1] 场景发生器 × 计时协议 × 顺序 的 A/B（ms/场景；每格重复 3 轮看漂移）")
print("%-8s %5s %-9s %-9s | %-28s | %-28s" %
      ("net", "B", "场景", "顺序", "dense 三轮", "cudss 三轮"))
for stem in ("Modena", "City_D", "ky4"):
    net, s = mk(stem)
    d, rh = bc(net)
    for B in (64, 256):
        for sname, sfn, seed in (("mine", scen_mine, 900 + B),
                                 ("upstream", scen_upstream, 77 + B)):
            D, R = sfn(d, rh, B, seed)
            s.cudss_plan(B)
            it = int(s.solve(D, R)["iters"].max())
            for order in ("d先", "c先"):
                dd, cc = [], []
                for _ in range(3):
                    if order == "d先":
                        dd.append(tg_upstream(lambda: s.solve(D, R)) / B * 1e3)
                        cc.append(tg_upstream(lambda: s.solve(
                            D, R, assemble="csr", linear_solver="cudss")) / B * 1e3)
                    else:
                        cc.append(tg_upstream(lambda: s.solve(
                            D, R, assemble="csr", linear_solver="cudss")) / B * 1e3)
                        dd.append(tg_upstream(lambda: s.solve(D, R)) / B * 1e3)
                print("%-8s %5d %-9s %-9s | %-28s | %-28s  → %.2fx (iters=%d)" %
                      (stem, B, sname, order,
                       " ".join("%.4f" % x for x in dd),
                       " ".join("%.4f" % x for x in cc),
                       np.median(dd) / np.median(cc), it))
    del s, net
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "=" * 110)
print("[2] 预热长度 / 重复次数 敏感性（ky4 B=256，cudss）")
net, s = mk("ky4")
d, rh = bc(net)
D, R = scen_mine(d, rh, 256, 1156)
s.cudss_plan(256)


def cud():
    s.solve(D, R, assemble="csr", linear_solver="cudss")


for warm, reps in ((0, 5), (2, 5), (3, 9), (10, 25)):
    med, best = tg_mine(cud, reps=reps, warm=warm)
    print("   warm=%2d reps=%2d → 中位 %.5f  最好 %.5f ms/场景"
          % (warm, reps, med / 256 * 1e3, best / 256 * 1e3))

print("\n[3] cuDSS 每次调用的地板（ky4 B=256，直接打 DirectSolver）")
key = [k for k in s._cudss_cache if k[0] == 256][0]
st = s._cudss_cache[key]
ds = st["solver"]


def t1(fn, n=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


print("   factorize      %.4f ms" % t1(ds.factorize))
print("   solve          %.4f ms" % t1(ds.solve))
print("   stack(solve)   %.4f ms" % t1(lambda: torch.stack(ds.solve())))
data = torch.rand(256, s.A_csr_nnz, dtype=DT, device="cuda")
Hj = torch.rand(256, s.Nj, dtype=DT, device="cuda")
print("   csr SpMV       %.4f ms" % t1(lambda: s._csr_spmv(data, Hj, 256)))
print("   vals.copy_     %.4f ms" % t1(lambda: st["vals"].copy_(data)))
print("   每轮 cudss ≈ fact + 3×(solve+stack) = %.3f ms（×9 轮 /256 = %.4f ms/场景）"
      % (t1(ds.factorize) + 3 * t1(lambda: torch.stack(ds.solve())),
         (t1(ds.factorize) + 3 * t1(lambda: torch.stack(ds.solve()))) * 9 / 256))

print("\n[4] 计数器包装的税（在 DirectSolver.solve 外面套一层 python lambda）")
_o = type(ds).solve
CNT = [0]
type(ds).solve = lambda self, **k: (CNT.__setitem__(0, CNT[0] + 1), _o(self, **k))[1]
med_w, _ = tg_mine(cud, reps=9, warm=3)
type(ds).solve = _o
med_n, _ = tg_mine(cud, reps=9, warm=3)
print("   带包装 %.5f vs 不带 %.5f ms/场景（差 %.1f%%）"
      % (med_w / 256 * 1e3, med_n / 256 * 1e3, (med_w / med_n - 1) * 100))

print("\n[5] 时钟/功耗（跑完之后）:", subprocess.run(
    ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
     "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
print("ADV2 DONE")

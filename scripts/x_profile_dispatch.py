# -*- coding: utf-8 -*-
"""单帧稳态求解的时间去向：Python/框架分派 vs 实际算子执行。

论文报单帧 542 节点 6.371 ms 对参考引擎 0.304 ms（21x）。该网一次求解的算术量
约 1.5e4 FLOPs，所以差距不可能来自算术。本脚本用 torch profiler 把时间拆开，
用以判定 LibTorch（C++ 前端，消除 Python 分派）最多能拿走多少。
"""
import os, sys, time, json
import numpy as np, torch
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dgga.parse import Net
from dgga.solver import GGASolver
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bench_batch import build_scenarios, STEM, INP, REF_DIR   # noqa: E402

net = Net.load(REF_DIR, STEM)
s = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense", inp_path=INP)
d0, rh0, D, _ = build_scenarios(net, s)
D1 = D[0]

def once():
    return s.solve(D1, rh0)

for _ in range(5):
    once()
N = 40
t0 = time.perf_counter()
for _ in range(N):
    once()
wall = (time.perf_counter() - t0) / N * 1e3

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as pr:
    for _ in range(N):
        once()
ev = pr.key_averages()
self_ms = sum(e.self_cpu_time_total for e in ev) / N / 1e3
n_ops = sum(e.count for e in ev) / N
top = sorted(ev, key=lambda e: -e.self_cpu_time_total)[:10]

print("=" * 66)
print("network        : %s   Nj=%d  L=%d" % (STEM, s.Nj, net.L))
print("torch threads  : %d" % torch.get_num_threads())
print("wall / solve   : %.3f ms   (%d repeats)" % (wall, N))
print("aten ops/solve : %.0f" % n_ops)
print("sum self CPU   : %.3f ms  (%.1f%% of wall)" % (self_ms, 100 * self_ms / wall))
print("unaccounted    : %.3f ms  (%.1f%% of wall)  <- Python-side dispatch/glue"
      % (wall - self_ms, 100 * (wall - self_ms) / wall))
print("-" * 66)
print("%-34s %9s %8s %7s" % ("op", "self ms", "n/solve", "% wall"))
for e in top:
    ms = e.self_cpu_time_total / N / 1e3
    print("%-34s %9.4f %8.0f %6.1f%%" % (e.key[:34], ms, e.count / N, 100 * ms / wall))
json.dump({"stem": STEM, "Nj": int(s.Nj), "L": int(net.L), "threads": torch.get_num_threads(),
           "wall_ms": wall, "aten_ops_per_solve": n_ops, "self_cpu_ms": self_ms,
           "unaccounted_ms": wall - self_ms,
           "top": [{"op": e.key, "self_ms": e.self_cpu_time_total / N / 1e3,
                    "count": e.count / N} for e in top]},
          open(os.path.join(ROOT, "data", "x_dispatch_profile.json"), "w"), indent=1)
print("-> data/x_dispatch_profile.json")

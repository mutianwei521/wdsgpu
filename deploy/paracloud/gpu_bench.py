# -*- coding: utf-8 -*-
"""HydroGrad dense-path CPU vs GPU benchmark, self-contained.

No EPANET DLL and no reference cache: the network is parsed from .inp and the
CPU float64 result is its own reference. Reports per-scenario wall time at
several batch sizes and the GPU-vs-CPU head deviation, which is the
reduction-order effect the paper argues makes bit-exactness CPU-only.
"""
import json, os, platform, sys, time
import numpy as np, torch
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp
from dgga.solver import GGASolver

INP = os.path.join(ROOT, "city_d.inp")
SEED, B_LIST = 20260808, (1, 64, 256)

def timeit(fn, repeats, warmup=2, sync=None):
    for _ in range(warmup):
        fn()
    if sync: sync()
    best = float("inf")
    for _ in range(repeats):
        if sync: sync()
        t = time.perf_counter()
        fn()
        if sync: sync()
        best = min(best, time.perf_counter() - t)
    return best

def run(net, D, rh0, device, dtype, label):
    s = GGASolver(net, device=device, dtype=dtype, mode="dense", inp_path=INP)
    sync = torch.cuda.synchronize if device.startswith("cuda") else None
    out, H = {}, None
    for B in B_LIST:
        Din = D[0] if B == 1 else D[:B]
        reps = {1: 20, 64: 5, 256: 3}[B]
        sec = timeit(lambda: s.solve(Din, rh0), reps, 2, sync)
        out[B] = sec / B * 1e3
        if B == 64:
            r = s.solve(D[:64], rh0)
            H = r["head_ft"].to(torch.float64).cpu().numpy()
            it = r["iters"].cpu().numpy()
            out["iters_min"], out["iters_max"] = int(it.min()), int(it.max())
            out["unconverged"] = int((~r["converged"]).sum())
    print("[%s] " % label + "  ".join("B=%d: %.3f ms/scen" % (B, out[B]) for B in B_LIST)
          + "   iters %d-%d" % (out["iters_min"], out["iters_max"]))
    return out, H

net = parse_inp(INP)
s0 = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense", inp_path=INP)
d0 = net.demand_cfs_at(0); rh0 = net.reservoir_head_ft_at(0)
rng = np.random.default_rng(SEED)
D = d0[None, :] * rng.uniform(0.8, 1.2, size=(max(B_LIST), net.N))

info = {"host": platform.node(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "threads": torch.get_num_threads(),
        "N": int(net.N), "Nj": int(s0.Nj), "L": int(net.L)}
if torch.cuda.is_available():
    info["gpu"] = torch.cuda.get_device_name(0)
    info["gpu_count"] = torch.cuda.device_count()
    cap = torch.cuda.get_device_capability(0)
    info["compute_capability"] = "%d.%d" % cap
print("=" * 72)
print("host %s | torch %s cu%s | %d CPU threads" % (info["host"], info["torch"], info["cuda"], info["threads"]))
print("GPU  %s x%d (sm_%s)" % (info.get("gpu", "none"), info.get("gpu_count", 0), info.get("compute_capability", "-")))
print("net  %s  N=%d Nj=%d L=%d" % (os.path.basename(INP), net.N, s0.Nj, net.L))
print("=" * 72)

res = {}
res["cpu_f64"], Hc = run(net, D, rh0, "cpu", torch.float64, "CPU float64")
if torch.cuda.is_available():
    res["gpu_f64"], Hg = run(net, D, rh0, "cuda", torch.float64, "GPU float64")
    dH = float(np.abs(Hg - Hc).max())
    print("  GPU f64 vs CPU f64 (B=64): max|dH| = %.3e ft   <- reduction order" % dH)
    res["dH_gpu_vs_cpu_ft"] = dH
    try:
        res["gpu_f32"], Hf = run(net, D, rh0, "cuda", torch.float32, "GPU float32")
        res["dH_f32_vs_f64_ft"] = float(np.abs(Hf - Hc).max())
        print("  GPU f32 vs CPU f64 (B=64): max|dH| = %.3e ft" % res["dH_f32_vs_f64_ft"])
    except Exception as e:
        print("  GPU float32 failed:", e)
    for B in B_LIST:
        print("  speedup GPU/CPU at B=%-3d : %.2fx" % (B, res["cpu_f64"][B] / res["gpu_f64"][B]))
json.dump({"info": info, "res": res}, open(os.path.join(ROOT, "gpu_bench.json"), "w"), indent=1, default=str)
print("-> gpu_bench.json")

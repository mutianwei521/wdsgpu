# -*- coding: utf-8 -*-
"""审计③④补：CUDA 上的对照实验（干净进程、带预热）。
1. 装配核：dense-vs-csr 的 max|ΔA|  与  dense-vs-dense 重跑的 max|ΔA|（自不定性）
 - 若两者同量级，则 GPU 偏差是既有的 atomicAdd 不定性，不是 CSR 引入的。
2. 整条 solve()：dense-vs-csr 的 max|ΔH|  与  dense-vs-dense 重跑的 max|ΔH|。
3. solve() 的 CUDA 峰值（预热后再测，排除 caching allocator 首调用假象）。
4. torch.use_deterministic_algorithms(True) 下 scatter_add_ 在 CUDA 上是否可用。
"""
import gc
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audlib import REF, Net, GGASolver, boundary, batchify    # noqa: E402

DEV = "cuda"


def vals_of(sv, Pm):
    return torch.cat([-Pm[:, sv.lk_both], -Pm[:, sv.lk_both],
                      Pm[:, sv.lk_m1], Pm[:, sv.lk_m2]], dim=1)


def dense_asm(sv, vals, B):
    A = torch.zeros(B, sv.Nj * sv.Nj, dtype=torch.float64, device=DEV)
    A.scatter_add_(1, sv.A_idx.expand(B, -1), vals)
    return A


def peak_of(fn, warm=1):
    for _ in range(warm):
        r = fn(); del r
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    r = fn()
    torch.cuda.synchronize()
    return r, torch.cuda.max_memory_allocated() - base


def main():
    print("=== 1  装配核 @CUDA：CSR 偏差 vs 稠密自身的重跑偏差（B=256）===")
    print(f"{'网':10s} {'Nj':>6s} {'max|A_csr-A_den|':>18s} {'max|A_den-A_den2|':>19s} "
          f"{'max|A_csr-A_csr2|':>19s} {'结论':>12s}")
    for name, stem in [("Net3", "pub_net3"), ("Modena", "pub_modena"),
                       ("City_D", "city_d"), ("ky4", "pub_ky4")]:
        net = Net.load(REF, stem)
        sv = GGASolver(net, mode="dense", device=DEV, dense_tank_bound_check=False)
        B = 256
        g = torch.Generator(device=DEV).manual_seed(5)
        Pm = 0.1 + torch.rand(B, sv.L, dtype=torch.float64, device=DEV, generator=g)
        v = vals_of(sv, Pm)
        A1 = dense_asm(sv, v, B)
        A2 = dense_asm(sv, v, B)
        C1 = sv._csr_to_dense(sv._assemble_csr(v, B), B).reshape(B, -1)
        C2 = sv._csr_to_dense(sv._assemble_csr(v, B), B).reshape(B, -1)
        d_cd = float((C1 - A1).abs().max())
        d_dd = float((A2 - A1).abs().max())
        d_cc = float((C2 - C1).abs().max())
        verdict = "同量级" if d_cd <= max(d_dd, d_cc) * 4 + 1e-300 else "**CSR更差**"
        print(f"{name:10s} {sv.Nj:6d} {d_cd:18.3e} {d_dd:19.3e} {d_cc:19.3e} "
              f"{verdict:>12s}")
        del A1, A2, C1, C2, v, Pm, sv
        gc.collect(); torch.cuda.empty_cache()

    print("\n=== 2  整条 solve() @CUDA：CSR 偏差 vs 稠密自身重跑偏差 ===")
    print(f"{'网':10s} {'B':>4s} {'max|H_csr-H_den|':>18s} {'max|H_den-H_den2|':>19s} "
          f"{'iters同':>8s} {'结论':>12s}")
    for name, stem, Bs in [("Net3", "pub_net3", [8, 64, 256]),
                           ("Modena", "pub_modena", [64, 256]),
                           ("ky4", "pub_ky4", [8, 32])]:
        net = Net.load(REF, stem)
        sv = GGASolver(net, mode="dense", device=DEV, dense_tank_bound_check=False)
        d, rh = boundary(net, stem, 0)
        for B in Bs:
            D, R = batchify(d, rh, B, 4242)
            D, R = D.to(DEV), R.to(DEV)
            o1 = sv.solve(D, R, assemble="dense")
            o1b = sv.solve(D, R, assemble="dense")
            o2 = sv.solve(D, R, assemble="csr")
            e_cd = float((o2["head_ft"] - o1["head_ft"]).abs().max())
            e_dd = float((o1b["head_ft"] - o1["head_ft"]).abs().max())
            same_it = bool(torch.equal(o1["iters"], o2["iters"]))
            verdict = "同量级" if e_cd <= max(e_dd, 1e-300) * 20 else "**CSR更差**"
            print(f"{name:10s} {B:4d} {e_cd:18.3e} {e_dd:19.3e} "
                  f"{str(same_it):>8s} {verdict:>12s}")
            del o1, o1b, o2
            gc.collect(); torch.cuda.empty_cache()
        del sv
        gc.collect(); torch.cuda.empty_cache()

    print("\n=== 3  solve() CUDA 峰值（预热后实测）===")
    print(f"{'网':10s} {'Nj':>5s} {'B':>4s} {'dense MiB':>10s} {'csr MiB':>10s} "
          f"{'csr/dense':>10s} {'[B,Nj²] MiB':>12s} {'[B,nnz] MiB':>12s}")
    for name, stem, Bs in [("Net3", "pub_net3", [64, 256]),
                           ("Modena", "pub_modena", [64, 256]),
                           ("ky4", "pub_ky4", [8, 32])]:
        net = Net.load(REF, stem)
        sv = GGASolver(net, mode="dense", device=DEV, dense_tank_bound_check=False)
        d, rh = boundary(net, stem, 0)
        for B in Bs:
            D, R = batchify(d, rh, B, 4242)
            D, R = D.to(DEV), R.to(DEV)
            _, p1 = peak_of(lambda: sv.solve(D, R, assemble="dense"))
            _, p2 = peak_of(lambda: sv.solve(D, R, assemble="csr"))
            print(f"{name:10s} {sv.Nj:5d} {B:4d} {p1/2**20:10.1f} {p2/2**20:10.1f} "
                  f"{p2/p1:9.3f}x {B*sv.Nj*sv.Nj*8/2**20:12.1f} "
                  f"{B*sv.A_csr_nnz*8/2**20:12.3f}")
            gc.collect(); torch.cuda.empty_cache()
        del sv
        gc.collect(); torch.cuda.empty_cache()

    print("\n=== 4  确定性算法开关下 CUDA scatter_add_ 是否可用 ===")
    net = Net.load(REF, "pub_net3")
    sv = GGASolver(net, mode="dense", device=DEV, dense_tank_bound_check=False)
    B = 16
    g = torch.Generator(device=DEV).manual_seed(1)
    v = vals_of(sv, 0.1 + torch.rand(B, sv.L, dtype=torch.float64, device=DEV,
                                     generator=g))
    torch.use_deterministic_algorithms(True, warn_only=False)
    for nm, fn in (("dense scatter_add_", lambda: dense_asm(sv, v, B)),
                   ("csr   scatter_add_", lambda: sv._assemble_csr(v, B))):
        try:
            fn()
            print(f"    {nm}: 允许（确定性）")
        except Exception as e:                                # noqa: BLE001
            print(f"    {nm}: **禁止** {type(e).__name__}: {str(e)[:100]}")
    torch.use_deterministic_algorithms(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

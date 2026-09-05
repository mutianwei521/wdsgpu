# -*- coding: utf-8 -*-
"""审计③：显存**实测**（不是估算）。
A. 装配核单独：CUDA 峰值分配 [B,nnz] vs [B,Nj²]（torch.cuda.max_memory_allocated）
B. **整条 solve()**：assemble='dense' vs 'csr' 的 CUDA 峰值 - 本轮 CSR 通路
   末尾还要 _csr_to_dense 回稠密，故 solve 级别的峰值不降反升，需实测坐实。
C. BWSN_2 的 299 GiB 口径：实际尝试分配，OOM 与否即证据。
D. CUDA 上两条通路是否仍逐位相同（上游只对 CPU 下了结论）。
"""
import gc
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audlib import REF, Net, GGASolver, make_solver, boundary, batchify   # noqa: E402


def any_solver(net, device):
    for md in ("dense", "epanet"):
        for kw in ({}, dict(dense_tank_bound_check=False)):
            try:
                return GGASolver(net, mode=md, device=device, **kw), md
            except Exception:                                  # noqa: BLE001
                continue
    return None, None


def vals_of(sv, Pm):
    return torch.cat([-Pm[:, sv.lk_both], -Pm[:, sv.lk_both],
                      Pm[:, sv.lk_m1], Pm[:, sv.lk_m2]], dim=1)


def peak(fn):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    pk = torch.cuda.max_memory_allocated() - base
    return r, pk, dt


def main():
    dev = "cuda"
    print("=== A/D  装配核：CUDA 峰值 + 逐位对拍（B=256）===")
    print(f"{'网':10s} {'Nj':>6s} {'nnz':>6s} {'CSR峰值MiB':>11s} {'稠密峰值MiB':>12s} "
          f"{'降幅':>7s} {'CSRms':>7s} {'denms':>7s} {'max|ΔA|':>11s} {'自不定':>10s}")
    for name, stem in [("Net3", "pub_net3"), ("Modena", "pub_modena"),
                       ("City_D", "city_d"), ("ky4", "pub_ky4"),
                       ("Net6", "pub_net6"), ("BWSN_2", "pub_bwsn_network_2")]:
        net = Net.load(REF, stem)
        sv, md = any_solver(net, dev)
        B = 256
        g = torch.Generator(device=dev).manual_seed(5)
        Pm = 0.1 + torch.rand(B, sv.L, dtype=torch.float64, device=dev, generator=g)
        vals = vals_of(sv, Pm)
        data, pk_csr, t_csr = peak(lambda: sv._assemble_csr(vals, B))

        def dense_asm():
            A = torch.zeros(B, sv.Nj * sv.Nj, dtype=torch.float64, device=dev)
            A.scatter_add_(1, sv.A_idx.expand(B, -1), vals)
            return A
        try:
            A, pk_den, t_den = peak(dense_asm)
            dA = float((sv._csr_to_dense(data, B).reshape(B, -1) - A).abs().max())
            A2 = dense_asm()
            self_nd = float((A2 - A).abs().max())   # 同一通路重跑的自不定性
            del A, A2
            s_den = f"{pk_den/2**20:.1f}"
            s_t = f"{t_den*1e3:.1f}"
            s_dA = f"{dA:.3e}"
            s_nd = f"{self_nd:.3e}"
        except torch.OutOfMemoryError:
            s_den = f"OOM({B*sv.Nj*sv.Nj*8/2**30:.1f}GiB需)"
            s_t = "-"; s_dA = "-"; s_nd = "-"
        ratio = sv.Nj * sv.Nj / sv.A_csr_nnz
        print(f"{name:10s} {sv.Nj:6d} {sv.A_csr_nnz:6d} {pk_csr/2**20:11.2f} "
              f"{s_den:>12s} {ratio:6.0f}x {t_csr*1e3:7.2f} {s_t:>7s} "
              f"{s_dA:>11s} {s_nd:>10s}")
        del data, vals, Pm, sv
        gc.collect(); torch.cuda.empty_cache()

    print("\n=== B  整条 solve() 的 CUDA 峰值（同一输入，两种 assemble）===")
    print("    注：本轮 csr 通路末尾 _csr_to_dense 仍开 [B,Nj²]，故峰值不降反升")
    print(f"{'网':10s} {'Nj':>5s} {'B':>4s} {'dense峰值MiB':>13s} {'csr峰值MiB':>11s} "
          f"{'比值':>7s} {'dense s':>8s} {'csr s':>8s} {'max|ΔH|':>10s}")
    for name, stem, Bs in [("Net3", "pub_net3", [8, 64, 256]),
                           ("Modena", "pub_modena", [8, 64, 256]),
                           ("ky4", "pub_ky4", [8, 32])]:
        net = Net.load(REF, stem)
        sv = GGASolver(net, mode="dense", device=dev,
                       dense_tank_bound_check=False)
        d, rh = boundary(net, stem, 0)
        for B in Bs:
            D, R = batchify(d, rh, B, 4242)
            D, R = D.to(dev), R.to(dev)
            try:
                o1, p1, t1 = peak(lambda: sv.solve(D, R, assemble="dense"))
                o2, p2, t2 = peak(lambda: sv.solve(D, R, assemble="csr"))
                dH = float((o1["head_ft"] - o2["head_ft"]).abs().max())
                print(f"{name:10s} {sv.Nj:5d} {B:4d} {p1/2**20:13.1f} "
                      f"{p2/2**20:11.1f} {p2/p1:6.3f}x {t1:8.3f} {t2:8.3f} "
                      f"{dH:10.3e}")
                del o1, o2
            except torch.OutOfMemoryError as e:                  # noqa: BLE001
                print(f"{name:10s} {sv.Nj:5d} {B:4d}  OOM: {str(e)[:60]}")
            gc.collect(); torch.cuda.empty_cache()
        del sv
        gc.collect(); torch.cuda.empty_cache()

    print("\n=== C  BWSN_2 的 299 GiB：实际分配试验（B=256, f64）===")
    net = Net.load(REF, "pub_bwsn_network_2")
    sv, md = any_solver(net, dev)
    B, Nj, nnz = 256, sv.Nj, sv.A_csr_nnz
    need = B * Nj * Nj * 8
    print(f"    Nj={Nj} nnz={nnz}  稠密 [B,Nj²] 需 {need/2**30:.2f} GiB, "
          f"CSR [B,nnz] 需 {B*nnz*8/2**20:.2f} MiB, 卡上共 "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB")
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.zeros(B, nnz, dtype=torch.float64, device=dev)
    print(f"    CSR 值张量分配成功：实测 {x.numel()*x.element_size()/2**20:.2f} MiB, "
          f"CUDA 峰值 {torch.cuda.max_memory_allocated()/2**20:.2f} MiB")
    del x; gc.collect(); torch.cuda.empty_cache()
    try:
        y = torch.zeros(B, Nj * Nj, dtype=torch.float64, device=dev)
        print("    稠密张量竟然分配成功（?!）", y.shape)
        del y
    except torch.OutOfMemoryError as e:
        print(f"    稠密张量分配：**OOM**（预期） - {str(e).splitlines()[0][:110]}")
    # CPU 侧也试一次，坐实"本机开不出来"
    try:
        y = torch.zeros(B, Nj * Nj, dtype=torch.float64)
        print("    CPU 稠密分配成功（?!）")
        del y
    except Exception as e:                                       # noqa: BLE001
        print(f"    CPU 稠密分配：**失败**（预期） - {type(e).__name__}: "
              f"{str(e).splitlines()[0][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

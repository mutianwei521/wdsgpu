# -*- coding: utf-8 -*-
"""审计补：float32 通路、参数守卫、"对角必须入位型"这一理由是否真的成立、
CPU 侧 solve 的实测内存（RSS）。"""
import gc
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audlib import REF, Net, GGASolver, boundary, batchify, synth, bitfp   # noqa: E402


def main():
    torch.set_num_threads(1)
    print("=== A  float32 通路（线性解走 LU 均衡分支）dense vs csr ===")
    for st in ["pub_net1", "pub_hanoi", "pub_net3", "pub_modena"]:
        net = Net.load(REF, st)
        sv = GGASolver(net, mode="dense", dtype=torch.float32,
                       dense_tank_bound_check=False)
        d, rh = boundary(net, st, 0)
        for B in (1, 8):
            if B == 1:
                D = torch.as_tensor(d, dtype=torch.float32)
                R = torch.as_tensor(rh, dtype=torch.float32)
            else:
                D, R = batchify(d, rh, B, 77, dtype=torch.float32)
            o1 = sv.solve(D, R)
            o2 = sv.solve(D, R, assemble="csr")
            dH = float((o1["head_ft"] - o2["head_ft"]).abs().max())
            bit = bitfp(o1["head_ft"]) == bitfp(o2["head_ft"])
            print(f"    {st:12s} B={B:<3d} max|ΔH|={dH:.3e}  位指纹{'相同' if bit else '不同'}"
                  f"  {'OK' if (dH == 0 and bit) else '**FAIL**'}")

    print("\n=== B  参数守卫 ===")
    net = Net.load(REF, "pub_net1")
    d, rh = boundary(net, "pub_net1", 0)
    sd = GGASolver(net, mode="dense")
    se = GGASolver(net, mode="epanet")
    for nm, fn in [
        ("dense + assemble='xyz'", lambda: sd.solve(d, rh, assemble="xyz")),
        ("dense + assemble=None", lambda: sd.solve(d, rh, assemble=None)),
        ("epanet + assemble='csr'", lambda: se.solve(d, rh, assemble="csr")),
        ("epanet + assemble='dense'（应放行）",
         lambda: se.solve(d, rh, assemble="dense")),
    ]:
        try:
            fn()
            print(f"    {nm:34s} -> 放行")
        except Exception as e:                                 # noqa: BLE001
            print(f"    {nm:34s} -> {type(e).__name__}: {str(e)[:70]}")

    print("\n=== C  “对角必须入位型才能逐位复现”这一理由是否真的成立（本轮）===")
    print("    做法：把位型里的多余对角剔掉（只留链路贡献），重算 scatter/dense_pos，")
    print("    再走同一个 _csr_to_dense，看是否仍与稠密逐位相同。")
    for nm, netobj in [("pub_net1", Net.load(REF, "pub_net1")),
                       ("SYN net1 iso2", synth("pub_net1", iso=2)[0]),
                       ("SYN net1 iso2+ke", synth("pub_net1", iso=2, iso_ke=0.5)[0])]:
        sv = GGASolver(netobj, mode="dense", dense_tank_bound_check=False)
        Nj = sv.Nj
        a_flat = sv.A_idx.numpy()
        pos_nodiag = np.unique(a_flat)                 # 不并入全部对角
        n_lost = sv.A_csr_nnz - pos_nodiag.size
        sc = torch.as_tensor(np.searchsorted(pos_nodiag, a_flat))
        dp = torch.as_tensor(pos_nodiag)
        B = 4
        g = torch.Generator().manual_seed(2)
        Pm = 0.1 + torch.rand(B, sv.L, dtype=torch.float64, generator=g)
        vals = torch.cat([-Pm[:, sv.lk_both], -Pm[:, sv.lk_both],
                          Pm[:, sv.lk_m1], Pm[:, sv.lk_m2]], dim=1)
        ref = torch.zeros(B, Nj * Nj, dtype=torch.float64)
        ref.scatter_add_(1, sv.A_idx.expand(B, -1), vals)
        data = torch.zeros(B, pos_nodiag.size, dtype=torch.float64)
        data.scatter_add_(1, sc.expand(B, -1), vals)
        got = torch.zeros(B, Nj * Nj, dtype=torch.float64).scatter(
            1, dp.expand(B, -1), data)
        print(f"    {nm:16s} nnz(全对角)={sv.A_csr_nnz:5d} nnz(无多余对角)="
              f"{pos_nodiag.size:5d} 少存 {n_lost} 个结构零   "
              f"max|ΔA| = {float((got-ref).abs().max()):.3e}")
    print("    ⇒ 若上行仍为 0，则“对角入位型”对本轮的逐位等价并非必要条件"
          "（_csr_to_dense 从 zeros 起散射，未触及位天然为 0）；")
    print("      其真正必要性在 §1b：cuDSS 需要显式对角槽承接 emitter 项与主元。")

    print("\n=== D  CPU 侧 solve 实测内存（RSS 增量，Modena B=256）===")
    try:
        import psutil
        p = psutil.Process()
        net = Net.load(REF, "pub_modena")
        sv = GGASolver(net, mode="dense", dense_tank_bound_check=False)
        d, rh = boundary(net, "pub_modena", 0)
        D, R = batchify(d, rh, 256, 11)
        sv.solve(D, R)                          # 预热
        for asm in ("dense", "csr"):
            gc.collect()
            b0 = p.memory_info().rss
            hi = [0]
            import threading
            stop = threading.Event()

            def mon():
                while not stop.is_set():
                    hi[0] = max(hi[0], p.memory_info().rss)
            th = threading.Thread(target=mon); th.start()
            sv.solve(D, R, assemble=asm)
            stop.set(); th.join()
            print(f"    assemble={asm:6s} RSS 峰值增量 = {(hi[0]-b0)/2**20:8.1f} MiB")
    except ImportError:
        print("    （无 psutil，跳过）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

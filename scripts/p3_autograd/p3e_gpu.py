# -*- coding: utf-8 -*-
"""P3-E 显存的公平口径：稠密如果也用手写伴随，OOM 边界会往右挪多少？

P3-B §4 量的"稠密 OOM"是**仓库现状**（torch 通用 autograd 穿 linalg.cholesky，
每轮要留 A 和 chol 两份 [B,Nj,Nj]）。手写伴随只留 chol 一份，理论上省一半。
省一半够不够翻盘，得实测，不能靠推。

这里搭一条与 solve_unrolled 同构的 K 轮链（每轮的 A 依赖上一轮的解，
所以整条图必须留住），三个变体各测峰值显存与耗时：
  A 稠密 + torch autograd      B 稠密 + 手写伴随      C cudss + _CudssSolveFn
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

DEV = "cuda"
DT = torch.float64
NETDIR = os.path.join(ROOT, "p2nets")
NETS = [("Modena", "Modena.inp"), ("City_D", "City_D.inp"), ("ky4", "ky4.inp")]


def last():
    return traceback.format_exc().strip().split("\n")[-1][:110]


class DenseAdjoint(torch.autograd.Function):
    """稠密版手写伴随（与 _CudssSolveFn 同构）：反向复用同一个 Cholesky 因子。"""

    @staticmethod
    def forward(ctx, A, F, refine):
        with torch.no_grad():
            chol = torch.linalg.cholesky(A)
            Fc = F.unsqueeze(-1)
            x = torch.cholesky_solve(Fc, chol)
            for _ in range(int(refine)):
                AH = (A * x.transpose(-2, -1)).sum(-1, keepdim=True)
                x = x + torch.cholesky_solve(Fc - AH, chol)
        ctx.save_for_backward(chol, x)
        return x.squeeze(-1)

    @staticmethod
    def backward(ctx, g):
        chol, x = ctx.saved_tensors
        with torch.no_grad():
            lam = torch.cholesky_solve(g.unsqueeze(-1), chol)
            gA = -lam @ x.transpose(-2, -1)
        return gA, lam.squeeze(-1), None


def mk(fn):
    f = os.path.join(NETDIR, fn)
    net = parse_inp(f)
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=f,
                  dense_tank_bound_check=False)
    return net, s


print("=" * 78)
print("node:", os.popen("hostname").read().strip(), "|", torch.cuda.get_device_name(0))
import nvmath                                        # noqa: E402
print("nvmath", nvmath.__version__)
print()
print("K 轮链式图（每轮 A 依赖上一轮的解）的 前向+反向：峰值显存 MiB / 耗时 ms")
print("net      Nj   B    K | A 稠密+autograd | B 稠密+手写伴随 | C cudss  | A/C  B/C（显存）")
K = 8
for stem, fn in NETS:
    try:
        net, s = mk(fn)
    except Exception:                                # noqa: BLE001
        print("[skip] %s: %s" % (stem, last()))
        continue
    Nj, nnz = s.Nj, s.A_csr_nnz
    g = np.random.default_rng(7)
    # 逐链路的 P>0，按 solver.solve 的原式装配 ⇒ A 严格对称（Cholesky 才成立）
    baseP = torch.as_tensor(g.uniform(0.5, 2.0, (1, s.L)), dtype=DT, device=DEV)
    for B in (64, 128, 256, 512):
        res = {}
        for tag in ("A", "B", "C"):
            try:
                torch.cuda.empty_cache()
                s.cudss_free(empty_cache=True)
                if tag == "C":
                    s.cudss_grad_slots = 1
                theta = torch.zeros(B, s.L, dtype=DT, device=DEV,
                                    requires_grad=True)
                F0 = torch.as_tensor(g.normal(size=(B, Nj)), dtype=DT, device=DEV)
                gw = torch.as_tensor(g.normal(size=(B, Nj)), dtype=DT, device=DEV)

                def once():
                    x = None
                    for _ in range(K):
                        # 每轮的 P 依赖上一轮的解 ⇒ 图必须整条留住
                        P = baseP.expand(B, -1) + theta
                        if x is not None:
                            P = P * (1.0 + 1e-6 * x[:, :1])
                        v = torch.cat([-P[:, s.lk_both], -P[:, s.lk_both],
                                       P[:, s.lk_m1], P[:, s.lk_m2]], dim=1)
                        data = s._assemble_csr(v, B)
                        data = data.index_add(
                            1, s.A_csr_diag,
                            torch.full((B, Nj), 1.0, dtype=DT, device=DEV))
                        if tag == "C":
                            x = s._cudss_solve(data, F0, B)
                        else:
                            A = s._csr_to_dense(data, B)
                            x = (DenseAdjoint.apply(A, F0, 2) if tag == "B"
                                 else _dense_autograd(A, F0))
                    (gw * x).sum().backward()

                def _dense_autograd(A, F):
                    L_ = torch.linalg.cholesky(A)
                    Fc = F.unsqueeze(-1)
                    y = torch.cholesky_solve(Fc, L_)
                    for _ in range(2):
                        AH = (A * y.transpose(-2, -1)).sum(-1, keepdim=True)
                        y = y + torch.cholesky_solve(Fc - AH, L_)
                    return y.squeeze(-1)

                once()                                # warm（含 plan）
                theta.grad = None
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                once()
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1e3
                res[tag] = (torch.cuda.max_memory_allocated() / 2 ** 20, dt)
                del theta, F0, gw
                torch.cuda.empty_cache()
            except torch.OutOfMemoryError:
                res[tag] = (float("nan"), float("nan"))
                torch.cuda.empty_cache()
            except Exception:                        # noqa: BLE001
                print("[skip] %s B=%d %s: %s" % (stem, B, tag, last()))
                res[tag] = (float("nan"), float("nan"))
                torch.cuda.empty_cache()
        f = lambda t: ("OOM" if res[t][0] != res[t][0]
                       else "%.1f/%.1f" % res[t])
        rat = lambda t: ("-" if res[t][0] != res[t][0] or res["C"][0] != res["C"][0]
                         else "%.1fx" % (res[t][0] / res["C"][0]))
        print("%-8s %-4d %-4d %-2d| %15s | %15s | %8s | %s %s"
              % (stem, Nj, B, K, f("A"), f("B"), f("C"), rat("A"), rat("B")))
    s.cudss_free(empty_cache=True)
    del s, net
    torch.cuda.empty_cache()
print("P3E DONE")

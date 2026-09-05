# -*- coding: utf-8 -*-
"""P3-D 公平性自查：那 30x 里有多少是"稀疏拿了自定义反向、稠密没拿"？

P3-B §1 量到 ky4 B=64 前向+反向 29.65x（B=256 稠密直接 OOM）。但稠密那边的反向
是 **torch 通用 autograd 穿过 linalg.cholesky**（Cholesky 的 VJP 要 O(B·Nj^3) 的
三角解 + matmul），而稀疏这边是我们手写的伴随（复用分解、只多一次回代）。
把同样的手写伴随也给稠密一份，才知道倍数里哪部分是"稀疏赢"、哪部分是
"通用 autograd 输"。三条通路逐轮计时：
  A 稠密 + torch autograd（**仓库现状**）
  B 稠密 + 手写伴随 autograd.Function（cholesky_solve 复用同一个因子）
  C cudss + P3 的 _CudssSolveFn（grad_refine=0/2）
并先验 B 与 A 的梯度一致（否则这个对照不成立）。
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
NETS = [("Net3", "Net3.inp"), ("Modena", "Modena.inp"),
        ("City_D", "City_D.inp"), ("ky4", "ky4.inp")]
SEED = 2026


def last():
    return traceback.format_exc().strip().split("\n")[-1][:110]


class DenseAdjoint(torch.autograd.Function):
    """稠密版的"手写伴随"：前向 chol + (1+2) 次回代，反向复用同一个因子。

    与 _CudssSolveFn 完全同构：grad_F = A^{-1} g（A 对称），
    grad_A = -λ x^T（稠密全阵，不做位型限制）。只用于本对照，不进 dgga。
    """

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
        ctx.refine = int(refine)
        return x.squeeze(-1)

    @staticmethod
    def backward(ctx, g):
        chol, x = ctx.saved_tensors
        with torch.no_grad():
            lam = torch.cholesky_solve(g.unsqueeze(-1), chol)   # 复用同一个因子
            for _ in range(ctx.refine):
                pass
            gA = -lam @ x.transpose(-2, -1)
        return gA, lam.squeeze(-1), None


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


def mk(fn):
    f = os.path.join(NETDIR, fn)
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


print("=" * 78)
print("node:", os.popen("hostname").read().strip(), "| torch", torch.__version__,
      "|", torch.cuda.get_device_name(0))
import nvmath                                        # noqa: E402
print("nvmath", nvmath.__version__)
print()
print("=" * 78)
print("一轮线性代数的 前向+反向（ms/轮，整批；A/F 取真前向最后一轮装出来的那一对）")
print("net      B    | A 稠密+autograd | B 稠密+手写伴随 | C cudss(gr=2) | C cudss(gr=0)"
      " | A/C2   B/C2   B/A")
for stem, fn in NETS:
    for B in (64, 256):
        try:
            net, s = mk(fn)
            d0, rh0 = boundary(net)
            D, R = batchify(d0, rh0, B)
            ke = np.zeros(net.N)
            ke[np.random.default_rng(1).choice(s.junc_nodes, size=min(40, s.Nj),
                                               replace=False)] = 0.5
            KE = torch.as_tensor(np.broadcast_to(ke[None, :], (B, net.N)).copy(),
                                 dtype=DT, device=DEV)
            cap = {}
            _o = GGASolver._cudss_forward

            def _cap(self, data_, F_, B_, refine=None, slot=0):
                r = _o(self, data_, F_, B_, refine, slot)
                cap["d"], cap["F"] = data_.detach().clone(), F_.detach().clone()
                return r
            GGASolver._cudss_forward = _cap
            try:
                with torch.no_grad():
                    s.solve(D, R, ke_int=KE, assemble="csr", linear_solver="cudss")
            finally:
                GGASolver._cudss_forward = _o
            data0, F0 = cap["d"], cap["F"]
            A0 = s._csr_to_dense(data0, B)
            Nj = s.Nj
            gv = torch.as_tensor(np.random.default_rng(5).normal(size=(B, Nj)),
                                 dtype=DT, device=DEV)

            def path_a():
                Av = A0.clone().requires_grad_(True)
                Fv = F0.clone().requires_grad_(True)
                L_ = torch.linalg.cholesky(Av)
                Fc = Fv.unsqueeze(-1)
                x = torch.cholesky_solve(Fc, L_)
                for _ in range(2):
                    AH = (Av * x.transpose(-2, -1)).sum(-1, keepdim=True)
                    x = x + torch.cholesky_solve(Fc - AH, L_)
                (gv * x.squeeze(-1)).sum().backward()
                return Av.grad, Fv.grad

            def path_b():
                Av = A0.clone().requires_grad_(True)
                Fv = F0.clone().requires_grad_(True)
                x = DenseAdjoint.apply(Av, Fv, 2)
                (gv * x).sum().backward()
                return Av.grad, Fv.grad

            def path_c(gr):
                s.cudss_grad_refine = gr
                s.cudss_grad_slots = 1
                dv = data0.clone().requires_grad_(True)
                Fv = F0.clone().requires_grad_(True)
                x = s._cudss_solve(dv, Fv, B)
                (gv * x).sum().backward()
                return dv.grad, Fv.grad

            # 一致性先验：B 的梯度必须与 A 一致，否则这个对照不成立
            ga, gfa = path_a()
            gb, gfb = path_b()
            gc, gfc = path_c(2)
            r_ab = float((ga - gb).abs().max() / ga.abs().max())
            r_fab = float((gfa - gfb).abs().max() / gfa.abs().max())
            # cudss 的 grad 只在 nnz 位置，取出 A 侧同位置比
            ga_nnz = ga.reshape(B, -1).index_select(1, s.A_csr_dense_pos)
            r_ac = float((ga_nnz - gc).abs().max() / ga_nnz.abs().max())
            ta = tg(path_a, reps=3, warm=1) * 1e3
            tb = tg(path_b, reps=3, warm=1) * 1e3
            tc2 = tg(lambda: path_c(2), reps=3, warm=1) * 1e3
            tc0 = tg(lambda: path_c(0), reps=3, warm=1) * 1e3
            print("%-8s %-4d | %15.4f | %15.4f | %13.4f | %13.4f | %6.2fx %6.2fx %5.3f"
                  % (stem, B, ta, tb, tc2, tc0, ta / tc2, tb / tc2, tb / ta))
            print("         %-4s   一致性：B vs A  gA %.3e / gF %.3e ；C vs A（nnz 位）gA %.3e"
                  % ("", r_ab, r_fab, r_ac))
            s.cudss_free(empty_cache=True)
            del A0, data0, F0, s, net
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print("%-8s %-4d | 稠密侧 torch.OutOfMemoryError" % (stem, B))
            torch.cuda.empty_cache()
        except Exception:                            # noqa: BLE001
            print("[skip] %s B=%d: %s" % (stem, B, last()))
            torch.cuda.empty_cache()

print("P3D DONE")

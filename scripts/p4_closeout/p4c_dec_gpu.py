# -*- coding: utf-8 -*-
"""P4 收尾 · 分解表的**等工作量口径**（终审 P2-a）。

背景：README "Read the speed-up as two factors" 那张表的
"sparse vs dense (hand-written adjoint both sides)" 一列，稠密对照 Badj 的**反向
只做 1 次回代**，而 cuDSS 那侧 C 的反向是 1+2 次（`cudss_grad_refine=2`）。
两侧工作量不等，方向对稀疏有利 - 所以那一列是"稀疏赢多少"的**下界**。

本脚本用与 `scripts/p4_remeasure/p4r_time_gpu.py` **逐字相同的口径**
（SEED=2026、d0×U(0.85,1.15)、rh0+U(-1,1)、水池水头夹在 30%~70%、
best-of 计时 budget=3.0/reps_max=3/warm=1、同一处 `_cudss_forward` 抓最后一轮的
(data, F)）在 README 印的那 **8 个格**上同时量两个稠密对照：

    Badj     反向 1 次回代 - README 现印那一列的口径
    Badj_r2  反向 1+2 次回代（=C） - 等工作量口径

同一次运行里两列都出，所以 Badj/C 能直接与 README 现印的列对照（验口径没跑偏），
Badj_r2/C 才是"等工作量下稀疏到底赢多少"。

只读脚本：不改 dgga，缺省三条通路一个字节没碰。
用法：`sbatch p4cdec.sh`（或 `python3 -X utf8 p4c_dec_gpu.py`）。
"""
import hashlib
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                      # noqa: E402
from dgga.solver import GGASolver                     # noqa: E402

DEV = "cuda"
DT = torch.float64
NETDIR = os.path.join(ROOT, "p2nets")
SEED = 2026
NODE = os.popen("hostname").read().strip()

# README 分解表印的 8 个格，顺序照抄
CELLS = [("Net3", "Net3.inp", 256), ("Modena", "Modena.inp", 256),
         ("Modena", "Modena.inp", 1024), ("City_D", "City_D.inp", 64),
         ("City_D", "City_D.inp", 256), ("City_D", "City_D.inp", 1024),
         ("ky4", "ky4.inp", 64), ("ky4", "ky4.inp", 256)]


def last():
    return traceback.format_exc().strip().split("\n")[-1][:130]


class DenseAdjoint(torch.autograd.Function):
    """稠密"手写伴随"对照，与 _CudssSolveFn 同构。

    `refine_b` = 反向的迭代精化步数：0 = README 现印那一列的口径，
    2 = 与 C 的 `cudss_grad_refine=2` 等工作量。**只作对照，不进 dgga**
    （放进去会改缺省通路的梯度值，缺省是冻结的）。
    """

    @staticmethod
    def forward(ctx, A, F, refine, refine_b):
        with torch.no_grad():
            chol = torch.linalg.cholesky(A)
            Fc = F.unsqueeze(-1)
            x = torch.cholesky_solve(Fc, chol)
            for _ in range(int(refine)):
                AH = (A * x.transpose(-2, -1)).sum(-1, keepdim=True)
                x = x + torch.cholesky_solve(Fc - AH, chol)
        ctx.save_for_backward(chol, x, A)
        ctx.rb = int(refine_b)
        return x.squeeze(-1)

    @staticmethod
    def backward(ctx, g):
        chol, x, A = ctx.saved_tensors
        with torch.no_grad():
            gc = g.unsqueeze(-1)
            lam = torch.cholesky_solve(gc, chol)
            for _ in range(ctx.rb):
                AL = (A * lam.transpose(-2, -1)).sum(-1, keepdim=True)
                lam = lam + torch.cholesky_solve(gc - AL, chol)
            gA = -lam @ x.transpose(-2, -1)
        return gA, lam.squeeze(-1), None, None


def tg(fn, budget=3.0, reps_max=3, warm=1):
    """与 p4r_time_gpu.py 逐字相同的 best-of 计时（秒）。"""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = time.perf_counter() - t0
    reps = max(2, min(reps_max, int(budget / max(one, 1e-6))))
    best = one
    for _ in range(reps - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, np.nan_to_num(rh)


def batchify(d, rh, B, seed=SEED):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.85, 1.15, (B, d.size))
    R = rh[None, :] + g.uniform(-1.0, 1.0, (B, rh.size))
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


print("=" * 104)
print("P4 收尾 · 分解表等工作量口径 | node:", NODE, "| torch", torch.__version__,
      "|", torch.cuda.get_device_name(0))
import nvmath                                                     # noqa: E402
print("nvmath", nvmath.__version__)
print("md5 dgga/solver.py",
      hashlib.md5(open(os.path.join(ROOT, "dgga/solver.py"), "rb").read())
      .hexdigest())
print("=" * 104)

ROWS = []
for stem, fnm, B in CELLS:
    A0 = None
    try:
        f = os.path.join(NETDIR, fnm)
        net = parse_inp(f)
        s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=f,
                      dense_tank_bound_check=False)
        d0, rh0 = boundary(net)
        D, R = batchify(d0, rh0, B)

        cap = {}
        _o = GGASolver._cudss_forward

        def _capf(self, data_, F_, B_, refine=None, slot=0):
            r = _o(self, data_, F_, B_, refine, slot)
            cap["d"], cap["F"] = data_.detach().clone(), F_.detach().clone()
            return r

        GGASolver._cudss_forward = _capf
        try:
            with torch.no_grad():
                s.solve(D, R, assemble="csr", linear_solver="cudss")
        finally:
            GGASolver._cudss_forward = _o
        data0, F0 = cap["d"], cap["F"]
        gv = torch.as_tensor(np.random.default_rng(5).normal(size=(B, s.Nj)),
                             dtype=DT, device=DEV)

        def path_c():
            s.cudss_grad_refine = 2
            s.cudss_grad_slots = 1
            dv = data0.clone().requires_grad_(True)
            Fv = F0.clone().requires_grad_(True)
            x = s._cudss_solve(dv, Fv, B)
            (gv * x).sum().backward()
            return dv.grad, Fv.grad

        def mk_b(rb):
            def path_b():
                Av = A0.clone().requires_grad_(True)
                Fv = F0.clone().requires_grad_(True)
                x = DenseAdjoint.apply(Av, Fv, 2, rb)
                (gv * x).sum().backward()
                return Av.grad, Fv.grad
            return path_b

        tc = tg(path_c) * 1e3
        gc, _ = path_c()
        A0 = s._csr_to_dense(data0, B)
        # 一致性：两个稠密对照与 cuDSS 算的是不是同一个梯度
        gb0, _ = mk_b(0)()
        gb2, _ = mk_b(2)()
        r_b0b2 = float((gb0 - gb2).abs().max() / gb0.abs().max())
        gb2_nnz = gb2.reshape(B, -1).index_select(1, s.A_csr_dense_pos)
        r_c = float((gb2_nnz - gc).abs().max() / gb2_nnz.abs().max())
        del gb0, gb2, gb2_nnz
        tb0 = tg(mk_b(0)) * 1e3
        tb2 = tg(mk_b(2)) * 1e3
        ROWS.append((stem, s.Nj, B, tb0, tb2, tc, r_b0b2, r_c))
        print("  %-7s Nj=%-4d B=%-5d | Badj(rb=0) %8.4f  Badj_r2(rb=2) %8.4f  "
              "C %8.4f ms/轮 | Badj/C %5.2fx  Badj_r2/C %5.2fx  "
              "(等工作量高 %+5.1f%%) | 一致性 rb0-vs-rb2 %.2e  C-vs-rb2 %.2e"
              % (stem, s.Nj, B, tb0, tb2, tc, tb0 / tc, tb2 / tc,
                 100.0 * (tb2 / tb0 - 1.0), r_b0b2, r_c))
        sys.stdout.flush()
        del data0, F0, gv, gc, D, R
    except Exception:                                   # noqa: BLE001
        print("  %-7s B=%-5d ERR %s" % (stem, B, last()))
    finally:
        A0 = None
        torch.cuda.empty_cache()

print("\n" + "=" * 104)
print("net      Nj    B     | Badj/C (README 现印口径) | Badj_r2/C (等工作量)")
for (stem, Nj, B, tb0, tb2, tc, _a, _b) in ROWS:
    print("  %-7s %-5d %-5d |        %5.2fx           |     %5.2fx"
          % (stem, Nj, B, tb0 / tc, tb2 / tc))
print("=" * 104)

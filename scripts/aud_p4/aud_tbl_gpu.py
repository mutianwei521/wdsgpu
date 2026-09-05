# -*- coding: utf-8 -*-
"""AUDIT 3 - 全表抽查 + 计时公平性全查 + 倍数分解的独立复核。

敌意口径：**自己写的计时/测量代码**，不 import 上游 p4_remeasure 的任何东西。
只共用 dgga 与 .inp（那是被测对象本身）。

抽查格（含一个 OOM 边界格）：
  ky4  B=8/64/256   Net3 B=8   Modena B=8   City_D B=512/1024   Net1 B=1024
量四件事：
  ① 纯前向 ms/场景（dense vs csr+cuDSS）
  ② 前向+反向 ms/场景（solve_unrolled(K)+backward(demand)）
  ③ 一轮线代的分解：A(稠密+autograd) / Badj(稠密+手写伴随) / C(cuDSS)
 - 外加两个公平性变体：Badj_nc（把 A.clone() 挪出计时区）、
        Badj_r2（反向也做 2 步精化，与 C 的 grad_refine=2 等工作量）
  ④ OOM 边界（前向+反向 dense）
公平性（每格都记，不是抽样）：
  · plan（cuDSS 符号分解）在计时区内发生几次 - 必须 0
  · 每次重复前后都 synchronize；warmup 次数；重复次数；min/median/全样本
  · dense 与 cudss 的**逐样本迭代数**是否相等；K 是否同一个；精化步数各是几
"""
import hashlib
import json
import os
import statistics
import sys
import time
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dgga.parse import parse_inp                                  # noqa: E402
from dgga.solver import GGASolver                                 # noqa: E402
from dgga.autodiff import solve_unrolled                          # noqa: E402

NETD = os.path.join(HERE, "p2nets")
DEV, DT = "cuda", torch.float64
NODE = os.popen("hostname").read().strip()
SEED = 2026

# 与上游 p4_remeasure 同一场景口径（否则格子不可比）：d0×U(0.85,1.15)、
# rh0+U(-1,1)、seed 2026、水池水头取上下限中点（夹在 30%~70% 之间）。
CELLS = [("ky4", "ky4.inp", 8), ("ky4", "ky4.inp", 64), ("ky4", "ky4.inp", 256),
         ("Net3", "Net3.inp", 8), ("Modena", "Modena.inp", 8),
         ("City_D", "City_D.inp", 512), ("City_D", "City_D.inp", 1024),
         ("Net1", "Net1.inp", 1024)]
DEC_CELLS = {("ky4", 8), ("ky4", 64), ("ky4", 256),
             ("City_D", 512), ("Modena", 8)}


def last():
    return traceback.format_exc().strip().split("\n")[-1][:130]


class PlanCounter:
    """数计时区内 cuDSS 的**符号分解(plan)**发生了几次：靠缓存未命中来数。"""

    def __init__(self):
        self.raw = GGASolver._cudss_state
        self.n = 0

    def __enter__(self):
        pc = self

        def st(self_, B, dtype, device, slot=0):
            key = (int(B), dtype, str(device), str(self_.cudss_matrix_type),
                   int(slot))
            miss = key not in self_._cudss_cache
            r = pc.raw(self_, B, dtype, device, slot)
            if miss:
                pc.n += 1
            return r

        GGASolver._cudss_state = st
        return self

    def __exit__(self, *e):
        GGASolver._cudss_state = self.raw
        return False


def bench(fn, warm=2, reps=5, budget=4.0):
    """我自己的计时：warm 次热身 -> 每次重复前后 synchronize -> 返回全样本。"""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = time.perf_counter() - t0
    n = max(3, min(reps, int(budget / max(one, 1e-6)) + 1))
    xs = [one]
    for _ in range(n - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append(time.perf_counter() - t0)
    return xs


class DenseAdjoint(torch.autograd.Function):
    """我自己写的稠密手写伴随（与 _CudssSolveFn 同构）。refine_b 步反向精化。"""

    @staticmethod
    def forward(ctx, A, F, refine, refine_b):
        with torch.no_grad():
            L = torch.linalg.cholesky(A)
            Fc = F.unsqueeze(-1)
            x = torch.cholesky_solve(Fc, L)
            for _ in range(int(refine)):
                AH = (A * x.transpose(-2, -1)).sum(-1, keepdim=True)
                x = x + torch.cholesky_solve(Fc - AH, L)
        ctx.save_for_backward(L, x, A)
        ctx.rb = int(refine_b)
        return x.squeeze(-1)

    @staticmethod
    def backward(ctx, g):
        L, x, A = ctx.saved_tensors
        with torch.no_grad():
            gc = g.unsqueeze(-1)
            lam = torch.cholesky_solve(gc, L)
            for _ in range(ctx.rb):        # 与 cuDSS 的 grad_refine 等工作量
                AL = (A * lam.transpose(-2, -1)).sum(-1, keepdim=True)
                lam = lam + torch.cholesky_solve(gc - AL, L)
            gA = -lam @ x.transpose(-2, -1)
        return gA, lam.squeeze(-1), None, None


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + .30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - .30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, np.nan_to_num(rh)


def batchify(d, rh, B, seed=SEED):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.85, 1.15, (B, d.size))
    R = rh[None, :] + g.uniform(-1.0, 1.0, (B, rh.size))
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


print("=" * 108)
print("AUDIT 3 - 全表抽查 / 计时公平性 / 分解复核 | node:", NODE)
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
import nvmath                                                     # noqa: E402
print("nvmath", nvmath.__version__)
for f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", f, hashlib.md5(open(os.path.join(HERE, f), "rb").read())
          .hexdigest())
print("=" * 108)

OUT = []
for stem, fnm, B in CELLS:
    rec = dict(node=NODE, net=stem, B=B)
    try:
        p = os.path.join(NETD, fnm)
        net = parse_inp(p)
        d0, rh0 = boundary(net)
        # K：在 CPU 上用 B=8 的同一批场景探（与上游同口径，避免 GPU 工作区污染）
        Dk, Rk = batchify(d0, rh0, 8)
        scpu = GGASolver(net, device="cpu", dtype=DT, mode="dense", inp_path=p,
                         dense_tank_bound_check=False)
        with torch.no_grad():
            K = int(scpu.solve(Dk.cpu(), Rk.cpu())["iters"].max())
        del scpu
        s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=p,
                      dense_tank_bound_check=False)
        D, R = batchify(d0, rh0, B)
        W = torch.as_tensor(np.random.default_rng(7).normal(size=(B, net.N)),
                            dtype=DT, device=DEV)
        rec.update(Nj=s.Nj, nnz=int(s.A_csr_nnz), K=K,
                   cudss_refine=int(s.cudss_refine),
                   cudss_grad_refine=int(s.cudss_grad_refine),
                   cudss_grad_slots=int(s.cudss_grad_slots),
                   cudss_cache_max=s.cudss_cache_max, dense_refine=2)
        print("\n### %-7s Nj=%-4d nnz=%-5d B=%-5d K=%d  "
              "(dense 精化=2, cudss_refine=%d, grad_refine=%d, slots=%d, cap=%s)"
              % (stem, s.Nj, s.A_csr_nnz, B, K, s.cudss_refine,
                 s.cudss_grad_refine, s.cudss_grad_slots, s.cudss_cache_max))

        # ---- ⑤ 迭代数逐样本核对（同一批场景，两条通路） ----
        try:
            with torch.no_grad():
                od = s.solve(D, R)
                oc = s.solve(D, R, assemble="csr", linear_solver="cudss")
            it_d = od["iters"].reshape(-1).tolist()
            it_c = oc["iters"].reshape(-1).tolist()
            dH = float((od["head_ft"] - oc["head_ft"]).abs().max())
            rec.update(iters_same=(it_d == it_c),
                       iters_d=sorted(set(it_d)), iters_c=sorted(set(it_c)),
                       maxdH=dH)
            print("    迭代数 dense=%s cudss=%s 逐样本相等=%s | max|dH|=%.3e ft"
                  % (sorted(set(it_d)), sorted(set(it_c)),
                     it_d == it_c, dH))
            del od, oc
        except torch.OutOfMemoryError:
            rec["iters_same"] = None
            print("    迭代数核对 OOM")
        torch.cuda.empty_cache()

        # ---- ① 纯前向 ----
        for tag, kw in (("fwd_d", dict()),
                        ("fwd_c", dict(assemble="csr", linear_solver="cudss"))):
            try:
                def f_(kw=kw):
                    with torch.no_grad():
                        s.solve(D, R, **kw)
                f_()                       # 建 plan / 缓存（计时区外）
                with PlanCounter() as pc:
                    xs = bench(f_)
                    nplan = pc.n
                rec[tag] = [x * 1e3 / B for x in xs]
                rec[tag + "_plan"] = nplan
                print("    %s  min=%.5f med=%.5f ms/场景  n=%d  计时区内plan=%d"
                      % (tag, min(xs) * 1e3 / B,
                         statistics.median(xs) * 1e3 / B, len(xs), nplan))
            except torch.OutOfMemoryError:
                rec[tag] = None
                print("    %s  OOM" % tag)
                torch.cuda.empty_cache()
            except Exception:                                     # noqa: BLE001
                rec[tag] = None
                print("    %s  ERR %s" % (tag, last()))
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        # ---- ② 前向+反向 ----
        for tag, kw in (("fb_d", dict()),
                        ("fb_c", dict(assemble="csr", linear_solver="cudss"))):
            try:
                def g_(kw=kw):
                    dv = D.clone().requires_grad_(True)
                    o = solve_unrolled(s, dv, R, K=K, **kw)
                    (W * o["head_ft"]).sum().backward()
                    return dv.grad
                gref = g_()
                with PlanCounter() as pc:
                    xs = bench(g_, warm=1, reps=4, budget=5.0)
                    nplan = pc.n
                rec[tag] = [x * 1e3 / B for x in xs]
                rec[tag + "_plan"] = nplan
                rec[tag + "_gnorm"] = float(gref.norm())
                print("    %s   min=%.5f med=%.5f ms/场景  n=%d  计时区内plan=%d"
                      "  |g|=%.10e"
                      % (tag, min(xs) * 1e3 / B,
                         statistics.median(xs) * 1e3 / B, len(xs), nplan,
                         float(gref.norm())))
                del gref
            except torch.OutOfMemoryError:
                rec[tag] = "OOM"
                print("    %s   **OOM**" % tag)
                torch.cuda.empty_cache()
            except Exception:                                     # noqa: BLE001
                rec[tag] = None
                print("    %s   ERR %s" % (tag, last()))
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        # ---- ③ 一轮线代的分解 ----
        if (stem, B) in DEC_CELLS:
            try:
                cap = {}
                raw = GGASolver._cudss_forward

                def capf(self_, data_, F_, B_, refine=None, slot=0):
                    r = raw(self_, data_, F_, B_, refine, slot)
                    cap["d"], cap["F"] = data_.detach().clone(), F_.detach().clone()
                    return r
                GGASolver._cudss_forward = capf
                try:
                    with torch.no_grad():
                        s.solve(D, R, assemble="csr", linear_solver="cudss")
                finally:
                    GGASolver._cudss_forward = raw
                data0, F0 = cap["d"], cap["F"]
                gv = torch.as_tensor(
                    np.random.default_rng(5).normal(size=(B, s.Nj)),
                    dtype=DT, device=DEV)
                A0 = None

                def path_c():
                    s.cudss_grad_refine = 2
                    s.cudss_grad_slots = 1
                    dv = data0.clone().requires_grad_(True)
                    Fv = F0.clone().requires_grad_(True)
                    x = s._cudss_solve(dv, Fv, B)
                    (gv * x).sum().backward()
                    return dv.grad, Fv.grad

                def path_a():
                    Av = A0.clone().requires_grad_(True)
                    Fv = F0.clone().requires_grad_(True)
                    L = torch.linalg.cholesky(Av)
                    Fc = Fv.unsqueeze(-1)
                    x = torch.cholesky_solve(Fc, L)
                    for _ in range(2):
                        AH = (Av * x.transpose(-2, -1)).sum(-1, keepdim=True)
                        x = x + torch.cholesky_solve(Fc - AH, L)
                    (gv * x.squeeze(-1)).sum().backward()
                    return Av.grad, Fv.grad

                def mk_b(rb, clone_in=True, pre=None):
                    def path_b():
                        Av = (A0.clone() if clone_in else pre)
                        Av = Av.requires_grad_(True)
                        Fv = F0.clone().requires_grad_(True)
                        x = DenseAdjoint.apply(Av, Fv, 2, rb)
                        (gv * x).sum().backward()
                        if not clone_in:
                            Av.grad = None
                            Av.requires_grad_(False)
                        return None
                    return path_b

                path_c()
                tc = bench(path_c, warm=1, reps=4, budget=3.0)
                gc, _ = path_c()
                dec = dict(C=[x * 1e3 for x in tc])
                try:
                    A0 = s._csr_to_dense(data0, B)
                    ga, _ = path_a()
                    bA = mk_b(0)
                    b2 = mk_b(2)
                    gb_run = DenseAdjoint.apply(A0.clone().requires_grad_(True),
                                                F0.clone(), 2, 0)
                    del gb_run
                    ta = bench(path_a, warm=1, reps=4, budget=3.0)
                    tb = bench(bA, warm=1, reps=4, budget=3.0)
                    tb2 = bench(b2, warm=1, reps=4, budget=3.0)
                    pre = A0.clone()
                    tbnc = bench(mk_b(0, clone_in=False, pre=pre),
                                 warm=1, reps=4, budget=3.0)
                    del pre
                    # 一致性：三条通路算的是不是同一个梯度
                    Av = A0.clone().requires_grad_(True)
                    Fv = F0.clone().requires_grad_(True)
                    x = DenseAdjoint.apply(Av, Fv, 2, 0)
                    (gv * x).sum().backward()
                    gb = Av.grad
                    r_ab = float((ga - gb).abs().max() / ga.abs().max())
                    ga_nnz = ga.reshape(B, -1).index_select(
                        1, s.A_csr_dense_pos)
                    r_ac = float((ga_nnz - gc).abs().max()
                                 / ga_nnz.abs().max())
                    dec.update(A=[x * 1e3 for x in ta],
                               Badj=[x * 1e3 for x in tb],
                               Badj_r2=[x * 1e3 for x in tb2],
                               Badj_nc=[x * 1e3 for x in tbnc],
                               r_ab=r_ab, r_ac=r_ac)
                    print("    分解(ms/轮) A=%.4f Badj=%.4f Badj_r2=%.4f "
                          "Badj_nc=%.4f C=%.4f | Badj/C=%.2fx Badj_r2/C=%.2fx "
                          "Badj_nc/C=%.2fx A/Badj=%.2fx A/C=%.2fx | "
                          "一致性 gA(B/A)=%.2e (C/A)=%.2e"
                          % (min(ta) * 1e3, min(tb) * 1e3, min(tb2) * 1e3,
                             min(tbnc) * 1e3, min(tc) * 1e3,
                             min(tb) / min(tc), min(tb2) / min(tc),
                             min(tbnc) / min(tc), min(ta) / min(tb),
                             min(ta) / min(tc), r_ab, r_ac))
                    del ga, gb, ga_nnz, Av, Fv, x
                except torch.OutOfMemoryError:
                    dec["A"] = dec["Badj"] = "OOM"
                    print("    分解 稠密侧 OOM（C=%.4f ms/轮）" % (min(tc) * 1e3))
                finally:
                    A0 = None
                    torch.cuda.empty_cache()
                rec["dec"] = dec
                del data0, F0, gv, gc
            except Exception:                                     # noqa: BLE001
                print("    分解 ERR %s" % last())
            torch.cuda.empty_cache()

        del s, D, R, W
        torch.cuda.empty_cache()
    except Exception:                                             # noqa: BLE001
        rec["err"] = last()
        print("### %s B=%d 整格失败 %s" % (stem, B, last()))
    OUT.append(rec)
    sys.stdout.flush()

print("\nAUDIT-TBL-JSON " + json.dumps(OUT))
print("=" * 108)

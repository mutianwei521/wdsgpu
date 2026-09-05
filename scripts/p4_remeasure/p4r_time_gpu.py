# -*- coding: utf-8 -*-
"""P4 全面重测 - 统一口径的时间全表（任务一 ①②、任务二的分解列）。

一张表取代 p2/p3 各 wip 里散落的零碎数字。行=网（全部 dense 可跑的公开网
+ City_D + ky4），列=B∈{1,8,64,256,512,1024}，每格报四组量：

  §A 纯前向 ms/场景        GGASolver.solve(no_grad)，dense vs csr+cuDSS
  §B 前向+反向 ms/场景     solve_unrolled(K) + backward(d)，dense vs csr+cuDSS
  §C **倍数的分解**（任务二）：一轮线性代数的前向+反向，三条通路
        A 稠密 + torch 通用 autograd（穿 linalg.cholesky，**仓库现状**）
        Badj 稠密 + 手写伴随（cholesky_solve 复用同一个因子，只作对照，不进 dgga）
        C cudss + _CudssSolveFn（grad_refine=2，仓库现状）
      ⇒ Badj/C 才是"稀疏 vs 稠密"本身；A/Badj 是"稠密走通用 autograd"的欠账。
      两者相乘 ≈ §B 报的整体倍数。**表里分两列，不合成一个数。**
  §D OOM 边界：每条通路在每个网上最大能跑通的 B。

缺省三条通路（mode="epanet" / assemble="dense" / linear_solver="dense"）未被
触碰；本脚本只读，不改 dgga。
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
from dgga.parse import parse_inp                      # noqa: E402
from dgga.solver import GGASolver                     # noqa: E402
from dgga.autodiff import solve_unrolled              # noqa: E402

DEV = "cuda"
DT = torch.float64
NETDIR = os.path.join(ROOT, "p2nets")
SEED = 2026
NODE = os.popen("hostname").read().strip()

NETS = [("Net1", "Net1.inp"), ("Anytown", "Anytown.inp"), ("Hanoi", "Hanoi.inp"),
        ("Net2", "Net2.inp"), ("Fossolo", "Fossolo_poly1.inp"),
        ("Pescara", "Pescara.inp"), ("Net3", "Net3.inp"),
        ("Modena", "Modena.inp"), ("City_D", "City_D.inp"), ("ky4", "ky4.inp")]
BS = [int(x) for x in os.environ.get("P4R_BS", "1,8,64,256,512,1024").split(",")]
ONLY = os.environ.get("P4R_NETS", "").strip()
if ONLY:
    keep = set(ONLY.split(","))
    NETS = [n for n in NETS if n[0] in keep]


def last():
    return traceback.format_exc().strip().split("\n")[-1][:120]


class DenseAdjoint(torch.autograd.Function):
    """稠密版"手写伴随"，与 _CudssSolveFn 同构：前向 chol + (1+refine) 次回代，
    反向 grad_F = A^{-1} g（复用同一个因子）、grad_A = -λ x^T。只作对照。"""

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


def tg(fn, budget=3.0, reps_max=5, warm=1):
    """best-of 计时，秒。先跑 warm 次热身，再按预算定重复数（>=2）。"""
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
    return d, np.nan_to_num(rh)


def batchify(d, rh, B, seed=SEED):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.85, 1.15, (B, d.size))
    R = rh[None, :] + g.uniform(-1.0, 1.0, (B, rh.size))
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


print("=" * 96)
print("P4 全面重测 · 时间全表 | node:", NODE, "| torch", torch.__version__,
      "|", torch.cuda.get_device_name(0))
import nvmath                                          # noqa: E402
print("nvmath", nvmath.__version__, "| B 列:", BS)
import hashlib                                         # noqa: E402
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f,
          hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print()

ROWS = []          # (stem, Nj, B, K, fwd_d, fwd_c, fb_d, fb_c)
DEC = []           # (stem, Nj, B, A, Badj, C, consistency)

for stem, fnm in NETS:
    try:
        net, s = mk(fnm)
    except Exception:                                  # noqa: BLE001
        print("[skip net] %s: %s" % (stem, last()))
        continue
    d0, rh0 = boundary(net)
    # K = 该网 dense 模式实测收敛迭代数（同一批 B=8 场景上取 max）。**在 CPU 上探**：
    # 与 p4r_mem_gpu.py 口径一致（那边必须避开 GPU 探测，否则 cuBLAS/cuSOLVER 的
    # 230 MiB 工作区会被算进 cudss 那一栏的 nontorch），两张表的 K 因此逐网相同。
    Dk, Rk = batchify(d0, rh0, 8)
    scpu = GGASolver(net, device="cpu", dtype=DT, mode="dense", inp_path=
                     os.path.join(NETDIR, fnm), dense_tank_bound_check=False)
    with torch.no_grad():
        K = int(scpu.solve(Dk.cpu(), Rk.cpu())["iters"].max())
    del scpu
    print("### %-8s Nj=%-4d L=%-5d N=%-5d nnz=%-6d K=%d" %
          (stem, s.Nj, s.L, net.N, int(s.A_csr_dense_pos.numel()), K))
    sys.stdout.flush()
    for B in BS:
        try:
            D, R = batchify(d0, rh0, B)
        except Exception:                              # noqa: BLE001
            print("  B=%-5d 造数失败 %s" % (B, last()))
            continue
        W = torch.as_tensor(np.random.default_rng(7).normal(size=(B, net.N)),
                            dtype=DT, device=DEV)
        res = {}

        # ---- ① 纯前向 ----
        for tag, kw in (("fwd_c", dict(assemble="csr", linear_solver="cudss")),
                        ("fwd_d", dict())):
            try:
                def f_(kw=kw):
                    with torch.no_grad():
                        s.solve(D, R, **kw)
                res[tag] = tg(f_) * 1e3 / B
            except torch.OutOfMemoryError:
                res[tag] = None
                torch.cuda.empty_cache()
            except Exception:                          # noqa: BLE001
                res[tag] = None
                print("  [err %s B=%d] %s" % (tag, B, last()))
                torch.cuda.empty_cache()

        # ---- ② 前向+反向 ----
        for tag, kw in (("fb_c", dict(assemble="csr", linear_solver="cudss")),
                        ("fb_d", dict())):
            try:
                def g_(kw=kw):
                    dv = D.clone().requires_grad_(True)
                    o = solve_unrolled(s, dv, R, K=K, **kw)
                    (W * o["head_ft"]).sum().backward()
                    return dv.grad
                res[tag] = tg(g_, budget=4.0, reps_max=3) * 1e3 / B
            except torch.OutOfMemoryError:
                res[tag] = None
                torch.cuda.empty_cache()
            except Exception:                          # noqa: BLE001
                res[tag] = None
                print("  [err %s B=%d] %s" % (tag, B, last()))
                torch.cuda.empty_cache()
        ROWS.append((stem, s.Nj, B, K, res["fwd_d"], res["fwd_c"],
                     res["fb_d"], res["fb_c"]))
        print("  B=%-5d fwd  dense %s  cudss %s  |  fwd+bwd  dense %s  cudss %s"
              % (B,
                 "OOM     " if res["fwd_d"] is None else "%8.5f" % res["fwd_d"],
                 "OOM     " if res["fwd_c"] is None else "%8.5f" % res["fwd_c"],
                 "OOM      " if res["fb_d"] is None else "%9.5f" % res["fb_d"],
                 "OOM      " if res["fb_c"] is None else "%9.5f" % res["fb_c"]))
        sys.stdout.flush()

        # ---- ③ 倍数分解：一轮线性代数的前向+反向 ----
        A0 = None
        try:
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

            def path_a():
                Av = A0.clone().requires_grad_(True)
                Fv = F0.clone().requires_grad_(True)
                Lc = torch.linalg.cholesky(Av)
                Fc = Fv.unsqueeze(-1)
                x = torch.cholesky_solve(Fc, Lc)
                for _ in range(2):
                    AH = (Av * x.transpose(-2, -1)).sum(-1, keepdim=True)
                    x = x + torch.cholesky_solve(Fc - AH, Lc)
                (gv * x.squeeze(-1)).sum().backward()
                return Av.grad, Fv.grad

            def path_b():
                Av = A0.clone().requires_grad_(True)
                Fv = F0.clone().requires_grad_(True)
                x = DenseAdjoint.apply(Av, Fv, 2)
                (gv * x).sum().backward()
                return Av.grad, Fv.grad

            def path_c():
                s.cudss_grad_refine = 2
                s.cudss_grad_slots = 1
                dv = data0.clone().requires_grad_(True)
                Fv = F0.clone().requires_grad_(True)
                x = s._cudss_solve(dv, Fv, B)
                (gv * x).sum().backward()
                return dv.grad, Fv.grad

            tc = tg(path_c, budget=3.0, reps_max=3) * 1e3
            gc, _ = path_c()
            try:
                A0 = s._csr_to_dense(data0, B)
                ga, _ = path_a()
                gb, _ = path_b()
                r_ab = float((ga - gb).abs().max() / ga.abs().max())
                ga_nnz = ga.reshape(B, -1).index_select(1, s.A_csr_dense_pos)
                r_ac = float((ga_nnz - gc).abs().max() / ga_nnz.abs().max())
                ta = tg(path_a, budget=3.0, reps_max=3) * 1e3
                tb = tg(path_b, budget=3.0, reps_max=3) * 1e3
                del ga, gb, ga_nnz
            except torch.OutOfMemoryError:
                ta = tb = r_ab = r_ac = None
            finally:
                A0 = None
                torch.cuda.empty_cache()
            DEC.append((stem, s.Nj, B, ta, tb, tc, r_ab, r_ac))
            print("     分解 一轮线代 f+b(ms/轮): A稠密+autograd %s | Badj稠密+手写伴随 %s"
                  " | C cudss %9.4f | A/C %s  Badj/C %s  A/Badj %s | 一致性 gA(B/A) %s (C/A) %s"
                  % ("OOM      " if ta is None else "%9.4f" % ta,
                     "OOM      " if tb is None else "%9.4f" % tb, tc,
                     "  -   " if ta is None else "%6.2fx" % (ta / tc),
                     "  -   " if tb is None else "%6.2fx" % (tb / tc),
                     "  -   " if ta is None else "%6.2fx" % (ta / tb),
                     "  -    " if r_ab is None else "%.2e" % r_ab,
                     "  -    " if r_ac is None else "%.2e" % r_ac))
            del data0, F0, gv, gc
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print("     分解 B=%d cudss 侧 OOM" % B)
            torch.cuda.empty_cache()
        except Exception:                              # noqa: BLE001
            print("     [err 分解 B=%d] %s" % (B, last()))
            torch.cuda.empty_cache()
        finally:
            A0 = None
            torch.cuda.empty_cache()
        del D, R, W
        torch.cuda.empty_cache()
        sys.stdout.flush()
    try:
        s.cudss_free(empty_cache=True)
    except Exception:                                  # noqa: BLE001
        pass
    del s, net
    torch.cuda.empty_cache()
    print()

print("=" * 96)
print("§T1 汇总（ms/场景；倍数 = dense/cudss）  node=%s" % NODE)
print("net      Nj    B     K  | fwd dense  fwd cudss  倍数   | f+b dense  f+b cudss  倍数")
for (stem, Nj, B, K, fd, fc, bd, bc) in ROWS:
    def c(x, w=9):
        return ("OOM").rjust(w) if x is None else ("%*.5f" % (w, x))
    rf = "   -   " if (fd is None or fc is None) else "%6.2fx" % (fd / fc)
    rb = "   -   " if (bd is None or bc is None) else "%6.2fx" % (bd / bc)
    print("%-8s %-5d %-5d %-2d | %s  %s %s | %s  %s %s"
          % (stem, Nj, B, K, c(fd), c(fc), rf, c(bd), c(bc), rb))

print()
print("§T2 倍数分解（一轮线性代数 f+b，ms/轮，整批）  node=%s" % NODE)
print("net      Nj    B     | A 稠密+autograd | Badj 稠密+手写伴随 | C cudss(gr=2)"
      " | Badj/C(稀疏赢)  A/Badj(通用autograd欠账)  A/C(仓库现状)")
for (stem, Nj, B, ta, tb, tc, r_ab, r_ac) in DEC:
    print("%-8s %-5d %-5d | %15s | %18s | %13.4f | %14s %23s %14s"
          % (stem, Nj, B,
             "OOM" if ta is None else "%.4f" % ta,
             "OOM" if tb is None else "%.4f" % tb, tc,
             "-" if tb is None else "%.2fx" % (tb / tc),
             "-" if ta is None else "%.2fx" % (ta / tb),
             "-" if ta is None else "%.2fx" % (ta / tc)))

print()
print("§T3 OOM 边界（每条通路最大跑通的 B / 最小 OOM 的 B）  node=%s" % NODE)
byn = {}
for (stem, Nj, B, K, fd, fc, bd, bc) in ROWS:
    e = byn.setdefault(stem, dict(Nj=Nj, fd=[], fc=[], bd=[], bc=[], oom={}))
    for k, v in (("fd", fd), ("fc", fc), ("bd", bd), ("bc", bc)):
        (e[k].append(B) if v is not None else e["oom"].setdefault(k, []).append(B))
print("net      Nj    | fwd dense      | fwd cudss      | f+b dense      | f+b cudss")
for stem in [n[0] for n in NETS if n[0] in byn]:
    e = byn[stem]

    def cell(k):
        ok = e[k]
        oom = e["oom"].get(k, [])
        return "max ok %-5s OOM %-5s" % (max(ok) if ok else "-",
                                         min(oom) if oom else "-")
    print("%-8s %-5d | %s | %s | %s | %s"
          % (stem, e["Nj"], cell("fd"), cell("fc"), cell("bd"), cell("bc")))
print("P4R TIME DONE")

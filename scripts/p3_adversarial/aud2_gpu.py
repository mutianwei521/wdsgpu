# -*- coding: utf-8 -*-
"""AUDIT-2 (independent): forward+backward timing, with the fairness checks
that P2 was audited on, redone from scratch.

  T0  harness self-checks: plan() is outside the timed region, sync is complete,
      warmup is enough (per-rep spread printed), both paths run the same K and
      the same number of triangular solves, and the two paths agree on H.
  T1  end-to-end solve_unrolled forward-only and forward+backward,
      dense vs cudss, ms/scenario, several nets x several B.
  T2  linear-algebra microbenchmark on the REAL A/F of the last Newton round:
      (A) dense + torch generic autograd through linalg.cholesky
      (B) dense + hand-written adjoint (same math cuDSS gets)
      (C) cudss, cudss_grad_refine = 2 and 0
      all three cross-checked to produce the same gA/gF.
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
import dgga.solver as S                               # noqa: E402
from dgga.autodiff import solve_unrolled              # noqa: E402

DEV, DT = "cuda", torch.float64
NETD = os.path.join(ROOT, "p2nets")
NODE = os.popen("hostname").read().strip()
NETS = [("Net3", "Net3.inp"), ("Modena", "Modena.inp"),
        ("City_D", "City_D.inp"), ("ky4", "ky4.inp")]


def err():
    return traceback.format_exc().strip().split("\n")[-1][:130]


def build(fn, B, seed=4242):
    p = os.path.join(NETD, fn)
    net = parse_inp(p)
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=p,
                  dense_tank_bound_check=False)
    g = np.random.default_rng(seed)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + .3 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - .3 * (net.tank_hmax - net.tank_hmin)
        rh0[tn] = np.clip(.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    rh0 = np.nan_to_num(rh0)
    D = torch.as_tensor(d0[None, :] * g.uniform(.85, 1.15, (B, d0.size)),
                        dtype=DT, device=DEV)
    R = torch.as_tensor(rh0[None, :] + g.uniform(-1., 1., (B, rh0.size)),
                        dtype=DT, device=DEV)
    W = torch.as_tensor(g.normal(0, 1, (B, d0.size)), dtype=DT, device=DEV)
    with torch.no_grad():
        pr = s.solve(D, R, assemble="dense", linear_solver="dense")
    K = int(pr["iters"].max())
    return net, s, D, R, W, K


def timeit(fn, target_s=3.0, warm=3, min_rep=3, max_rep=40):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = time.perf_counter() - t0
    rep = int(max(min_rep, min(max_rep, target_s / max(one, 1e-6))))
    ts = []
    for _ in range(rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts) * 1e3
    return float(ts.mean()), float(ts.std()), float(ts.min()), rep


# ---------------------------------------------------------------- T0
def sec_T0():
    print("\n" + "=" * 78)
    print("T0  harness self-checks (node %s)" % NODE)
    net, s, D, R, W, K = build("ky4.inp", 64)
    print("   ky4: Nj=%d nnz=%d converged iters K=%d" % (s.Nj, s.A_csr_nnz, K))

    from nvmath.sparse.advanced import DirectSolver
    rf, rs = DirectSolver.factorize, DirectSolver.solve
    C = dict(f=0, s=0)
    DirectSolver.factorize = lambda self, *a, **k: (C.__setitem__("f", C["f"] + 1),
                                                    rf(self, *a, **k))[1]
    DirectSolver.solve = lambda self, *a, **k: (C.__setitem__("s", C["s"] + 1),
                                                rs(self, *a, **k))[1]
    ncho = dict(n=0)
    rc = torch.linalg.cholesky
    rcs = torch.cholesky_solve
    ncs = dict(n=0)
    torch.linalg.cholesky = lambda *a, **k: (ncho.__setitem__("n", ncho["n"] + 1),
                                             rc(*a, **k))[1]
    torch.cholesky_solve = lambda *a, **k: (ncs.__setitem__("n", ncs["n"] + 1),
                                            rcs(*a, **k))[1]

    d1 = D.clone().requires_grad_(True)
    o1 = solve_unrolled(s, d1, R, K=K, assemble="csr", linear_solver="cudss")
    (W * o1["head_ft"]).sum().backward()
    fwd_bwd_cudss = dict(C)
    d2 = D.clone().requires_grad_(True)
    o2 = solve_unrolled(s, d2, R, K=K, assemble="dense", linear_solver="dense")
    (W * o2["head_ft"]).sum().backward()
    torch.linalg.cholesky, torch.cholesky_solve = rc, rcs
    DirectSolver.factorize, DirectSolver.solve = rf, rs

    print("   work done per fwd+bwd, K=%d, slots=%d (default):" % (K, s.cudss_grad_slots))
    print("     cudss : DirectSolver.factorize=%d  DirectSolver.solve=%d"
          % (fwd_bwd_cudss["f"], fwd_bwd_cudss["s"]))
    print("             expected fwd 1 fac + 3 solve per round = %d fac, %d solve;"
          % (K, 3 * K))
    print("             bwd adds 3 solve/round and (K-1) refac  ->  %d fac, %d solve"
          % (2 * K - 1, 6 * K))
    print("     dense : linalg.cholesky=%d  cholesky_solve=%d (fwd 1+3 per round)"
          % (ncho["n"], ncs["n"]))
    dH = float((o1["head_ft"] - o2["head_ft"]).abs().max())
    dQ = float((o1["flow_cfs"] - o2["flow_cfs"]).abs().max())
    dg = float((d1.grad - d2.grad).abs().max()
               / d2.grad.abs().max().clamp_min(1e-300))
    print("   same answer? max|dH|=%.3e ft  max|dQ|=%.3e cfs  rel grad diff=%.3e"
          % (dH, dQ, dg))

    # plan cost, and proof it is not inside the timed region
    s.cudss_free()
    t0 = time.perf_counter()
    pm = s.cudss_plan(64)
    torch.cuda.synchronize()
    print("   cudss_plan(B=64) = %.2f ms (reported) / %.2f ms (wall)"
          % (pm, (time.perf_counter() - t0) * 1e3))

    def run():
        d = D.clone().requires_grad_(True)
        o = solve_unrolled(s, d, R, K=K, assemble="csr", linear_solver="cudss")
        (W * o["head_ft"]).sum().backward()

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(12):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    print("   warmup adequacy, 12 timed reps after 3 warmups (ms/round):")
    print("     " + " ".join("%.2f" % x for x in ts))
    print("     first/median = %.3f ; spread(max-min)/median = %.3f"
          % (ts[0] / float(np.median(ts)),
             (max(ts) - min(ts)) / float(np.median(ts))))
    s.cudss_free()
    del s
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- T1
def sec_T1():
    print("\n" + "=" * 78)
    print("T1  end-to-end solve_unrolled, ms/scenario (node %s)" % NODE)
    print("   plan() pre-warmed and outside the timed region; 3 warmups;")
    print("   torch.cuda.synchronize() on both sides of every rep.")
    print("   net    Nj   K B    | dense fwd  cudss fwd  x   | dense f+b   "
          "cudss f+b   x     | reps d/c | max|dH| ft")
    for stem, fn in NETS:
        for B in (8, 64, 128, 256, 512):
            try:
                net, s, D, R, W, K = build(fn, B)
            except Exception:
                print("   %-6s B=%-4d build ERR %s" % (stem, B, err()))
                continue
            row = dict()
            dH = float("nan")
            try:
                s.cudss_plan(B)
            except Exception:
                pass

            def cf():
                with torch.no_grad():
                    solve_unrolled(s, D, R, K=K, assemble="csr",
                                   linear_solver="cudss")

            def df():
                with torch.no_grad():
                    solve_unrolled(s, D, R, K=K, assemble="dense",
                                   linear_solver="dense")

            def cb():
                d = D.clone().requires_grad_(True)
                o = solve_unrolled(s, d, R, K=K, assemble="csr",
                                   linear_solver="cudss")
                (W * o["head_ft"]).sum().backward()

            def db():
                d = D.clone().requires_grad_(True)
                o = solve_unrolled(s, d, R, K=K, assemble="dense",
                                   linear_solver="dense")
                (W * o["head_ft"]).sum().backward()

            for tag, f in (("cf", cf), ("df", df), ("cb", cb), ("db", db)):
                try:
                    m, sd, mn, rp = timeit(f)
                    row[tag] = (m / B, rp)
                except torch.cuda.OutOfMemoryError:
                    row[tag] = None
                    torch.cuda.empty_cache()
                except Exception:
                    row[tag] = None
            try:
                with torch.no_grad():
                    a = solve_unrolled(s, D, R, K=K, assemble="csr",
                                       linear_solver="cudss")["head_ft"]
                    b = solve_unrolled(s, D, R, K=K, assemble="dense",
                                       linear_solver="dense")["head_ft"]
                    dH = float((a - b).abs().max())
            except Exception:
                pass

            def fm(x):
                return "  OOM    " if x is None else "%9.5f" % x[0]

            def rat(a, b):
                if a is None or b is None:
                    return "  -   "
                return "%6.2f" % (a[0] / b[0])
            print("   %-6s %-4d %-1d %-4d | %s %s %s | %s %s %s | %s/%s | %.2e"
                  % (stem, s.Nj, K, B, fm(row["df"]), fm(row["cf"]),
                     rat(row["df"], row["cf"]), fm(row["db"]), fm(row["cb"]),
                     rat(row["db"], row["cb"]),
                     "-" if row["db"] is None else row["db"][1],
                     "-" if row["cb"] is None else row["cb"][1], dH))
            s.cudss_free()
            del s
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- T2
class DenseAdjoint(torch.autograd.Function):
    """Dense A x = b with the SAME hand-written adjoint cuDSS gets:
    forward = 1 cholesky + (1+refine) triangular solves,
    backward = (1+refine) triangular solves reusing the same factor and
    grad_A = -lambda x^T.  A is kept only so the refinement residual can be
    formed; the [B,Nj,Nj] Cholesky factor is the extra activation."""

    @staticmethod
    def forward(ctx, A, F, refine):
        with torch.no_grad():
            chol = torch.linalg.cholesky(A)
            Fc = F.unsqueeze(-1)
            x = torch.cholesky_solve(Fc, chol)
            for _ in range(refine):
                x = x + torch.cholesky_solve(Fc - torch.bmm(A, x), chol)
        ctx.save_for_backward(A, chol, x)
        ctx.refine = refine
        return x.squeeze(-1)

    @staticmethod
    def backward(ctx, g):
        A, chol, x = ctx.saved_tensors
        gc = g.unsqueeze(-1).contiguous()
        lam = torch.cholesky_solve(gc, chol)
        for _ in range(ctx.refine):
            lam = lam + torch.cholesky_solve(gc - torch.bmm(A, lam), chol)
        return -torch.bmm(lam, x.transpose(-2, -1)), lam.squeeze(-1), None


def sec_T2():
    print("\n" + "=" * 78)
    print("T2  linear-algebra microbenchmark on the REAL last-round A/F "
          "(node %s)" % NODE)
    print("   A = dense + torch generic autograd (repo status quo)")
    print("   B = dense + hand-written adjoint (the fair baseline)")
    print("   C = cudss (grad_refine 2 / 0)")
    print("   net    B    | A ms/round   B ms/round   C(gr2)     C(gr0)    | "
          "A/C2   B/C2   B/A  | gA cross-check")
    for stem, fn in NETS:
        for B in (64, 256):
            try:
                net, s, D, R, W, K = build(fn, B)
                cap = dict()
                raw = S.GGASolver._cudss_solve

                def cap_solve(self, data, F, Bq, refine=None):
                    cap["data"] = data.detach().clone()
                    cap["F"] = F.detach().clone()
                    return raw(self, data, F, Bq, refine)

                S.GGASolver._cudss_solve = cap_solve
                with torch.no_grad():
                    solve_unrolled(s, D, R, K=K, assemble="csr",
                                   linear_solver="cudss")
                S.GGASolver._cudss_solve = raw
                data0, F0 = cap["data"], cap["F"]
                A0 = s._csr_to_dense(data0, B)
                gW = torch.as_tensor(
                    np.random.default_rng(9).normal(0, 1, (B, s.Nj)),
                    dtype=DT, device=DEV)

                def mkA():
                    A = A0.clone().requires_grad_(True)
                    F = F0.clone().requires_grad_(True)
                    chol = torch.linalg.cholesky(A)
                    Fc = F.unsqueeze(-1)
                    x = torch.cholesky_solve(Fc, chol)
                    for _ in range(2):
                        AH = (A * x.transpose(-2, -1)).sum(-1, keepdim=True)
                        x = x + torch.cholesky_solve(Fc - AH, chol)
                    (gW * x.squeeze(-1)).sum().backward()
                    return A.grad, F.grad

                def mkB():
                    A = A0.clone().requires_grad_(True)
                    F = F0.clone().requires_grad_(True)
                    x = DenseAdjoint.apply(A, F, 2)
                    (gW * x).sum().backward()
                    return A.grad, F.grad

                def mkC(gr):
                    s.cudss_grad_refine = gr
                    d = data0.clone().requires_grad_(True)
                    F = F0.clone().requires_grad_(True)
                    x = s._cudss_solve(d, F, B)
                    (gW * x).sum().backward()
                    return d.grad, F.grad

                gA_a, gF_a = mkA()
                gA_b, gF_b = mkB()
                gA_c, gF_c = mkC(2)
                den = gA_a.abs().max().clamp_min(1e-300)
                eb = float((gA_b - gA_a).abs().max() / den)
                # C only has nnz slots: compare on the CSR pattern
                gA_a_nnz = gA_a.reshape(B, -1).index_select(
                    1, s.A_csr_row * s.Nj + s.A_csr_col)
                ec = float((gA_c - gA_a_nnz).abs().max() / den)
                efb = float((gF_b - gF_a).abs().max()
                            / gF_a.abs().max().clamp_min(1e-300))
                efc = float((gF_c - gF_a).abs().max()
                            / gF_a.abs().max().clamp_min(1e-300))

                s.cudss_plan(B)
                ma = timeit(mkA)[0]
                mb = timeit(mkB)[0]
                mc2 = timeit(lambda: mkC(2))[0]
                mc0 = timeit(lambda: mkC(0))[0]
                s.cudss_grad_refine = 2
                print("   %-6s %-4d | %10.4f  %10.4f  %9.4f  %9.4f | "
                      "%6.2f %6.2f %6.3f | B %.1e C %.1e (gF %.1e/%.1e)"
                      % (stem, B, ma, mb, mc2, mc0, ma / mc2, mb / mc2, mb / ma,
                         eb, ec, efb, efc))
                s.cudss_free()
                del s
            except torch.cuda.OutOfMemoryError:
                print("   %-6s %-4d | dense OOM" % (stem, B))
            except Exception:
                print("   %-6s %-4d | ERR %s" % (stem, B, err()))
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import hashlib
    import nvmath
    print("node:", NODE, "| torch", torch.__version__, "| dev",
          torch.cuda.get_device_name(0), "| nvmath", nvmath.__version__)
    print("solver.py md5:", hashlib.md5(
        open(os.path.join(ROOT, "dgga", "solver.py"), "rb").read()).hexdigest())
    for name, fn in (("T0", sec_T0), ("T1", sec_T1), ("T2", sec_T2)):
        try:
            fn()
        except Exception:
            print("SECTION %s CRASHED: %s" % (name, err()))
            traceback.print_exc()
        sys.stdout.flush()
    print("\nAUD2 DONE")

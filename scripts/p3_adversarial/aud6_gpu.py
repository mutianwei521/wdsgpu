# -*- coding: utf-8 -*-
"""AUDIT-6: what the counters do NOT say.

bwd_refactorize counts "the backward had to factorize again".  But when the
state it wanted has been evicted (rather than merely overwritten), the fallback
is not a refactorization -- _cudss_adjoint calls _cudss_state, which misses the
cache and builds a whole new DirectSolver, including plan().  plan() is the
expensive symbolic step (P2 measured 660-950 ms on ky4).  The counter reports
one refactorization either way.

  P1  count DirectSolver.plan() calls across a warm forward+backward, and
      across a forward+backward whose slots got evicted in between.
  P2  price it: wall time of a backward after eviction vs a clean backward.
  P3  does cudss_plan(B) warm the slots the differentiable path will use?
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

DEV, DT = "cuda", torch.float64
NETD = os.path.join(ROOT, "p2nets")
NODE = os.popen("hostname").read().strip()


def err():
    return traceback.format_exc().strip().split("\n")[-1][:130]


def build(fn, B, seed=4242):
    p = os.path.join(NETD, fn)
    net = parse_inp(p)
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=p,
                  dense_tank_bound_check=False)
    g = np.random.default_rng(seed)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
    D = torch.as_tensor(d0[None, :] * g.uniform(.85, 1.15, (B, d0.size)),
                        dtype=DT, device=DEV)
    R = torch.as_tensor(np.repeat(rh0[None, :], B, 0), dtype=DT, device=DEV)
    W = torch.as_tensor(g.normal(0, 1, (B, d0.size)), dtype=DT, device=DEV)
    return net, s, D, R, W


def main():
    from nvmath.sparse.advanced import DirectSolver
    rp, rf = DirectSolver.plan, DirectSolver.factorize
    C = dict(plan=0, fac=0)

    def pp(self, *a, **k):
        C["plan"] += 1
        return rp(self, *a, **k)

    def pf(self, *a, **k):
        C["fac"] += 1
        return rf(self, *a, **k)

    DirectSolver.plan, DirectSolver.factorize = pp, pf

    print("=" * 78)
    print("P1/P2  plan() accounting and the price of an evicted slot"
          " (node %s)" % NODE)
    for stem, fn, B in (("Modena", "Modena.inp", 128), ("ky4", "ky4.inp", 128)):
        try:
            net, s, D, R, W = build(fn, B)
            K = 8
            s.cudss_cache_max = 16
            s.cudss_grad_slots = K

            def fwdbwd():
                d = D.clone().requires_grad_(True)
                o = solve_unrolled(s, d, R, K=K, assemble="csr",
                                   linear_solver="cudss")
                (W * o["head_ft"]).sum().backward()
                return d.grad

            fwdbwd()                       # warm every slot
            s.cudss_counters(reset=True)
            C["plan"] = C["fac"] = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g_clean = fwdbwd()
            torch.cuda.synchronize()
            t_clean = (time.perf_counter() - t0) * 1e3
            lib = s.cudss_counters()
            print("   %-7s B=%-4d K=%d slots=%d  WARM  : plan=%d factorize=%d"
                  "  bwd_refactorize=%d  wall=%.1f ms"
                  % (stem, B, K, K, C["plan"], C["fac"],
                     lib["bwd_refactorize"], t_clean))

            # now: forward, evict every slot with foreign traffic, then backward
            d = D.clone().requires_grad_(True)
            o = solve_unrolled(s, d, R, K=K, assemble="csr",
                               linear_solver="cudss")
            L = (W * o["head_ft"]).sum()
            with torch.no_grad():
                for bb in range(3, 3 + 16):
                    solve_unrolled(s, D[:bb], R[:bb], K=2, assemble="csr",
                                   linear_solver="cudss")
            s.cudss_counters(reset=True)
            C["plan"] = C["fac"] = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            L.backward()
            torch.cuda.synchronize()
            t_ev = (time.perf_counter() - t0) * 1e3
            lib = s.cudss_counters()
            print("   %-7s B=%-4d K=%d slots=%d  EVICTED backward only:"
                  " plan=%d factorize=%d  bwd_refactorize=%d  wall=%.1f ms"
                  % (stem, B, K, K, C["plan"], C["fac"],
                     lib["bwd_refactorize"], t_ev))
            den = g_clean.abs().max().clamp_min(1e-300)
            print("            gradient after eviction vs clean: %.3e"
                  % float((d.grad - g_clean).abs().max() / den))
            print("            -> the counter reports %d 'refactorize' but %d"
                  " of them were full rebuild+plan()"
                  % (lib["bwd_refactorize"], C["plan"]))
            s.cudss_free()
            del s
        except Exception:
            print("   %-7s ERR %s" % (stem, err()))
        torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("P3  does cudss_plan(B) warm the slots the differentiable path uses?")
    net, s, D, R, W = build("Modena.inp", 128)
    s.cudss_cache_max = 16
    s.cudss_grad_slots = 6
    s.cudss_free()
    C["plan"] = 0
    pm = s.cudss_plan(128)
    print("   after cudss_plan(128): plan calls=%d  cache=%s"
          % (C["plan"], [(k[0], k[-1]) for k in
                         [(i[0], i[-1]) for i in s.cudss_cache_info()]]))
    C["plan"] = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    d = D.clone().requires_grad_(True)
    o = solve_unrolled(s, d, R, K=6, assemble="csr", linear_solver="cudss")
    (W * o["head_ft"]).sum().backward()
    torch.cuda.synchronize()
    t1 = (time.perf_counter() - t0) * 1e3
    print("   first differentiable fwd+bwd after that warmup: plan calls=%d,"
          " wall=%.1f ms (each plan cost ~%.1f ms)" % (C["plan"], t1, pm))
    C["plan"] = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    d = D.clone().requires_grad_(True)
    o = solve_unrolled(s, d, R, K=6, assemble="csr", linear_solver="cudss")
    (W * o["head_ft"]).sum().backward()
    torch.cuda.synchronize()
    t2 = (time.perf_counter() - t0) * 1e3
    print("   second one: plan calls=%d, wall=%.1f ms  -> warmup shortfall"
          " = %.1f ms on the first differentiable step" % (C["plan"], t2, t1 - t2))
    s.cudss_free()
    DirectSolver.plan, DirectSolver.factorize = rp, rf


if __name__ == "__main__":
    import hashlib
    import nvmath
    print("node:", NODE, "| torch", torch.__version__, "| nvmath",
          nvmath.__version__)
    print("solver.py md5:", hashlib.md5(
        open(os.path.join(ROOT, "dgga", "solver.py"), "rb").read()).hexdigest())
    try:
        main()
    except Exception:
        traceback.print_exc()
    print("\nAUD6 DONE")

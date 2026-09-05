# -*- coding: utf-8 -*-
"""P4-C: second-node confirmation of D2 (evicted slot = rebuild + replan) and
D3 (cudss_plan warms slot 0 only), plus two configurations the audit did not
cover (City_D, and cudss_plan on ky4 where plan() is most expensive).

D2  a slot evicted by the LRU is not "refactorized": _cudss_adjoint falls back
    to _cudss_state, misses, and builds a fresh DirectSolver -- plan() and all.
    The counter still says bwd_refactorize.  Priced here in wall time.
D3  cudss_plan(B) builds slot 0 only.  With cudss_grad_slots=k the first
    differentiable step still has to plan the other k-1 slots.
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
    print("E1  D2: the price of an evicted grad slot (node %s)" % NODE)
    print("    WARM = every slot still cached; EVICTED = 16 foreign batches")
    print("    pushed through between the forward and the backward.")
    for stem, fn, B in (("Modena", "Modena.inp", 128),
                        ("City_D", "City_D.inp", 128),
                        ("ky4", "ky4.inp", 128),
                        ("ky4", "ky4.inp", 256)):
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

            fwdbwd()
            s.cudss_counters(reset=True)
            C["plan"] = C["fac"] = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g_clean = fwdbwd()
            torch.cuda.synchronize()
            t_clean = (time.perf_counter() - t0) * 1e3
            lib = s.cudss_counters()
            print("    %-7s B=%-4d K=%d slots=%d  WARM full fwd+bwd : plan=%d"
                  " factorize=%d bwd_refactorize=%d  wall=%.1f ms"
                  % (stem, B, K, K, C["plan"], C["fac"],
                     lib["bwd_refactorize"], t_clean))

            # clean backward alone, for an apples-to-apples backward comparison
            d = D.clone().requires_grad_(True)
            o = solve_unrolled(s, d, R, K=K, assemble="csr",
                               linear_solver="cudss")
            L = (W * o["head_ft"]).sum()
            s.cudss_counters(reset=True)
            C["plan"] = C["fac"] = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            L.backward()
            torch.cuda.synchronize()
            t_bw_clean = (time.perf_counter() - t0) * 1e3
            lib = s.cudss_counters()
            print("    %-7s B=%-4d                WARM backward only: plan=%d"
                  " factorize=%d bwd_refactorize=%d  wall=%.1f ms"
                  % (stem, B, C["plan"], C["fac"], lib["bwd_refactorize"],
                     t_bw_clean))

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
            den = g_clean.abs().max().clamp_min(1e-300)
            print("    %-7s B=%-4d            EVICTED backward only: plan=%d"
                  " factorize=%d bwd_refactorize=%d  wall=%.1f ms"
                  "  -> %.1fx the clean backward, %.1fx the clean fwd+bwd"
                  % (stem, B, C["plan"], C["fac"], lib["bwd_refactorize"],
                     t_ev, t_ev / t_bw_clean, t_ev / t_clean))
            print("            grad after eviction vs clean: %.3e (rel)"
                  % float((d.grad - g_clean).abs().max() / den))
            s.cudss_free()
            del s
        except torch.cuda.OutOfMemoryError:
            print("    %-7s B=%-4d OOM" % (stem, B))
        except Exception:
            print("    %-7s B=%-4d ERR %s" % (stem, B, err()))
        torch.cuda.empty_cache()
        sys.stdout.flush()

    print("\n" + "=" * 78)
    print("E2  D3: does cudss_plan(B) warm the slots the differentiable path"
          " will use?  (node %s)" % NODE)
    for stem, fn, B, slots in (("Modena", "Modena.inp", 128, 6),
                               ("ky4", "ky4.inp", 128, 6)):
        try:
            net, s, D, R, W = build(fn, B)
            s.cudss_cache_max = 16
            s.cudss_grad_slots = slots
            s.cudss_free()
            C["plan"] = 0
            pm = s.cudss_plan(B)
            n1 = C["plan"]
            info = [(i[0], i[-1]) for i in s.cudss_cache_info()]
            C["plan"] = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            d = D.clone().requires_grad_(True)
            o = solve_unrolled(s, d, R, K=6, assemble="csr",
                               linear_solver="cudss")
            (W * o["head_ft"]).sum().backward()
            torch.cuda.synchronize()
            t1 = (time.perf_counter() - t0) * 1e3
            n2 = C["plan"]
            C["plan"] = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            d = D.clone().requires_grad_(True)
            o = solve_unrolled(s, d, R, K=6, assemble="csr",
                               linear_solver="cudss")
            (W * o["head_ft"]).sum().backward()
            torch.cuda.synchronize()
            t2 = (time.perf_counter() - t0) * 1e3
            print("    %-7s B=%-4d slots=%d | cudss_plan(%d): plan calls=%d,"
                  " reported %.1f ms, cache=%s"
                  % (stem, B, slots, B, n1, pm, info))
            print("            1st differentiable fwd+bwd: plan calls=%d,"
                  " wall=%.1f ms | 2nd: plan calls=%d, wall=%.1f ms"
                  "  -> first-step surcharge %.1f ms"
                  % (n2, t1, C["plan"], t2, t1 - t2))
            s.cudss_free()
            del s
        except Exception:
            print("    %-7s B=%-4d ERR %s" % (stem, B, err()))
        torch.cuda.empty_cache()
        sys.stdout.flush()

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
    print("\nP4C DONE")

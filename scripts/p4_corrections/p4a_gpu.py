# -*- coding: utf-8 -*-
"""P4-A: the self-rerun jitter table (audit R3).

P2 and P3 both concluded things with "it lands inside the jitter", but the
jitter floor itself was never tabulated -- and the acceptance criteria kept
using a hard-coded 1e-8.  This builds the missing table:

    for every (net, B, quantity, path):  run the SAME code REP times on the
    SAME inputs and report  self = max_{i<j} max|x_i - x_j|,
    both absolute and relative to max|x|.

    cross = max_{i,j} |cudss_i - dense_j|  is then reported against
    floor = max(self_cudss, self_dense).  Any future acceptance criterion is
    written as  cross <= 3 * floor  instead of  cross <= 1e-8.

Quantities: the forward heads out of GGASolver.solve (that is what acceptance
1 of sparse_gpu_plan.md §4/§7.2 compares) and the four unrolled gradients
d / rh / ke / r_hw (that is what acceptance 3 compares).

Nothing here changes any default: every cudss run is explicit
assemble="csr", linear_solver="cudss"; every dense run is the shipped default.
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
REP = int(os.environ.get("P4_REP", "5"))

NETS = [("Hanoi", "Hanoi.inp", 32), ("Net3", "Net3.inp", 92),
        ("Fossolo", "Fossolo_poly1.inp", 36), ("Modena", "Modena.inp", 268),
        ("City_D", "City_D.inp", 541), ("ky4", "ky4.inp", 959),
        ("Pescara", "Pescara.inp", 68)]
BS = tuple(int(x) for x in os.environ.get("P4_BS", "8,64,256").split(","))


def err():
    return traceback.format_exc().strip().split("\n")[-1][:130]


def build(fn, B, seed=4242, emit=True):
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
    ke = np.asarray(net.node_ke, dtype=np.float64).copy()
    if emit:
        jn = np.asarray(s.junc_nodes)
        ke[jn[g.random(jn.size) < .25]] = 1e-3
    KE = torch.as_tensor(np.repeat(ke[None, :], B, 0), dtype=DT, device=DEV)
    W = torch.as_tensor(g.normal(0, 1, (B, d0.size)), dtype=DT, device=DEV)
    return net, s, D, R, KE, W


def spread(xs):
    """max_{i<j} max|x_i - x_j| over a list of tensors."""
    w = 0.0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            w = max(w, float((xs[i] - xs[j]).abs().max()))
    return w


def cross_spread(a, b):
    w = 0.0
    for x in a:
        for y in b:
            w = max(w, float((x - y).abs().max()))
    return w


def verdict(cross, floor):
    if floor <= 0.0:
        return "floor=0" if cross == 0.0 else "ABOVE(floor 0)"
    r = cross / floor
    return "%6.2fx %s" % (r, "OK" if r <= 3.0 else "ABOVE")


def sec_forward():
    print("=" * 78)
    print("J1  forward heads: self-rerun jitter of GGASolver.solve  (node %s,"
          " REP=%d)" % (NODE, REP))
    print("    self  = max over the %d pairs of the %d reruns, max|dH| in ft"
          % (REP * (REP - 1) // 2, REP))
    print("    cross = max over the %d cudss-vs-dense pairs" % (REP * REP))
    print("    ratio = cross / max(self_cudss, self_dense)   <=3 means 'inside"
          " the jitter'")
    print("    net      Nj   B    K  | self(cudss) ft  self(dense) ft  "
          "cross ft     | maxH ft   | ratio")
    for stem, fn, Nj in NETS:
        for B in BS:
            try:
                net, s, D, R, KE, W = build(fn, B)
                with torch.no_grad():
                    o0 = s.solve(D, R, ke_int=KE)
                    K = int(o0["iters"].max())
                    cu, de = [], []
                    for _ in range(REP):
                        cu.append(s.solve(D, R, ke_int=KE, assemble="csr",
                                          linear_solver="cudss")["head_ft"]
                                  .clone())
                    for _ in range(REP):
                        de.append(s.solve(D, R, ke_int=KE)["head_ft"].clone())
                sc, sd = spread(cu), spread(de)
                cx = cross_spread(cu, de)
                mh = float(de[0].abs().max())
                print("    %-8s %-4d %-4d %-2d | %.6e   %.6e   %.6e | %.3e |"
                      " %s" % (stem, Nj, B, K, sc, sd, cx, mh,
                               verdict(cx, max(sc, sd))))
                print("        rel: self(cudss)=%.3e self(dense)=%.3e"
                      " cross=%.3e" % (sc / mh, sd / mh, cx / mh))
                s.cudss_free()
                del s, cu, de
            except torch.cuda.OutOfMemoryError:
                print("    %-8s %-4d %-4d    | OOM" % (stem, Nj, B))
            except Exception:
                print("    %-8s %-4d %-4d    | ERR %s" % (stem, Nj, B, err()))
            torch.cuda.empty_cache()
            sys.stdout.flush()


def sec_grad():
    print("\n" + "=" * 78)
    print("J2  unrolled gradients: self-rerun jitter  (node %s, REP=%d)"
          % (NODE, REP))
    print("    L = sum(W * head_ft), K = converged iteration count.")
    print("    all figures RELATIVE to max|g| of the dense run.")
    print("    net      B    K  par | self(cudss)  self(dense)  cross      "
          "| max|g_dense| | ratio")
    for stem, fn, Nj in NETS:
        for B in BS:
            try:
                net, s, D, R, KE, W = build(fn, B)
                with torch.no_grad():
                    K = int(s.solve(D, R, ke_int=KE)["iters"].max())

                def gr(ls, p):
                    th = dict(d=D.clone(), rh=R.clone(), ke=KE.clone(),
                              r=s.r_hw.clone())
                    th[p].requires_grad_(True)
                    o = solve_unrolled(
                        s, th["d"], th["rh"], ke=th["ke"], r_hw=th["r"], K=K,
                        assemble="csr" if ls == "cudss" else "dense",
                        linear_solver=ls)
                    g, = torch.autograd.grad((W * o["head_ft"]).sum(), th[p])
                    return g.detach().clone()

                for p in ("d", "rh", "ke", "r"):
                    cu, de, dense_oom = [], [], False
                    for _ in range(REP):
                        cu.append(gr("cudss", p))
                    try:
                        for _ in range(REP):
                            de.append(gr("dense", p))
                    except torch.cuda.OutOfMemoryError:
                        dense_oom = True
                        de = []
                        torch.cuda.empty_cache()
                    sc = spread(cu)
                    if dense_oom:
                        den = float(cu[0].abs().max()) or 1.0
                        print("    %-8s %-4d %-2d %-3s | %.4e   dense-OOM"
                              "     dense-OOM   | %.4e |   -"
                              % (stem, B, K, p, sc / den, den))
                        continue
                    den = float(de[0].abs().max()) or 1.0
                    sd = spread(de)
                    cx = cross_spread(cu, de)
                    print("    %-8s %-4d %-2d %-3s | %.4e   %.4e   %.4e"
                          " | %.4e | %s"
                          % (stem, B, K, p, sc / den, sd / den, cx / den, den,
                             verdict(cx, max(sc, sd))))
                    del cu, de
                s.cudss_free()
                del s
            except torch.cuda.OutOfMemoryError:
                print("    %-8s %-4d      | OOM" % (stem, B))
            except Exception:
                print("    %-8s %-4d      | ERR %s" % (stem, B, err()))
            torch.cuda.empty_cache()
            sys.stdout.flush()


if __name__ == "__main__":
    import hashlib
    import nvmath
    t0 = time.perf_counter()
    print("node:", NODE, "| torch", torch.__version__, "| nvmath",
          nvmath.__version__)
    print("solver.py md5:", hashlib.md5(
        open(os.path.join(ROOT, "dgga", "solver.py"), "rb").read()).hexdigest())
    print("REP=%d  BS=%s" % (REP, BS))
    for fn in (sec_forward, sec_grad):
        try:
            fn()
        except Exception:
            traceback.print_exc()
        sys.stdout.flush()
    print("\nP4A DONE in %.1f s" % (time.perf_counter() - t0))

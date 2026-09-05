# -*- coding: utf-8 -*-
"""AUDIT-5: the two claims the correctness argument actually rests on.

S1  Is A really symmetric?  The whole adjoint reuse rests on A^T = A, so that
    A.lambda = g can be solved with the factorization of A.  Measured on the
    real assembled CSR values of every net, at every Newton round: max over the
    nnz pattern of |A[i,j] - A[j,i]|.  Anything but exactly 0 would make the
    reuse an approximation rather than an identity.

S2  Where cudss and dense disagree (City_D / ky4 grad d), is that a defect of
    the sparse path or is the unrolled gradient simply not reproducible there?
    Decided by rerunning EACH path against ITSELF: if |cudss_1 - cudss_2| and
    |dense_1 - dense_2| are the same size as |cudss - dense|, nobody is wrong,
    the quantity is chaotic.  If dense reruns bit-identically while cudss does
    not, the sparse path owns the spread.
"""
import os
import sys
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
NETS = [("Hanoi", "Hanoi.inp"), ("Net3", "Net3.inp"), ("Modena", "Modena.inp"),
        ("City_D", "City_D.inp"), ("ky4", "ky4.inp"),
        ("Pescara", "Pescara.inp")]


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


def sec_S1():
    print("=" * 78)
    print("S1  is A exactly symmetric?  (node %s)" % NODE)
    print("   For each Newton round: t[k] = index of the CSR slot transposing")
    print("   slot k; report max_k |data[k] - data[t[k]]| over all rounds,")
    print("   both in absolute terms and relative to max|data|.")
    for stem, fn in NETS:
        try:
            net, s, D, R, KE, W = build(fn, 8)
            Nj = s.Nj
            row = s.A_csr_row.cpu().numpy()
            col = s.A_csr_col.cpu().numpy()
            key = row.astype(np.int64) * Nj + col
            tkey = col.astype(np.int64) * Nj + row
            order = np.argsort(key)
            pos = np.searchsorted(key[order], tkey)
            ok = (pos < key.size) & (key[order][np.clip(pos, 0, key.size - 1)]
                                     == tkey)
            if not ok.all():
                print("   %-8s PATTERN NOT SYMMETRIC: %d of %d slots have no "
                      "transpose" % (stem, int((~ok).sum()), key.size))
                continue
            tidx = torch.as_tensor(order[pos], dtype=torch.int64, device=DEV)
            worst = 0.0
            worst_rel = 0.0
            raw = S.GGASolver._cudss_solve
            seen = []

            def cap(self, data, F, B, refine=None):
                seen.append(data.detach())
                return raw(self, data, F, B, refine)

            S.GGASolver._cudss_solve = cap
            with torch.no_grad():
                solve_unrolled(s, D, R, ke=KE, K=8, assemble="csr",
                               linear_solver="cudss")
            S.GGASolver._cudss_solve = raw
            for dd in seen:
                e = (dd - dd.index_select(1, tidx)).abs().max()
                worst = max(worst, float(e))
                worst_rel = max(worst_rel,
                                float(e / dd.abs().max().clamp_min(1e-300)))
            print("   %-8s Nj=%-4d nnz=%-6d rounds=%d | pattern symmetric: yes"
                  " | max|A-A^T| = %.3e (rel %.3e)  %s"
                  % (stem, Nj, s.A_csr_nnz, len(seen), worst, worst_rel,
                     "EXACT" if worst == 0.0 else "<-- NOT exact"))
            s.cudss_free()
            del s
        except Exception:
            print("   %-8s ERR %s" % (stem, err()))
        torch.cuda.empty_cache()


def sec_S2():
    print("\n" + "=" * 78)
    print("S2  run-to-run reproducibility of the gradient, per path"
          " (node %s)" % NODE)
    print("   self = max|g(run1) - g(run2)| / max|g| for the SAME path;")
    print("   cross = the same between cudss and dense.")
    print("   If self(cudss) ~ self(dense) ~ cross, the disagreement is the")
    print("   unrolled route being chaotic, not the sparse path being wrong.")
    print("   net      B    K  param | self(cudss)  self(dense)  cross     "
          "| max|g_dense|")
    for stem, fn in NETS:
        for B in (8, 64):
            try:
                net, s, D, R, KE, W = build(fn, B)
                with torch.no_grad():
                    K = int(s.solve(D, R, ke_int=KE)["iters"].max())

                def gr(ls, p):
                    th = dict(d=D.clone(), rh=R.clone(), ke=KE.clone(),
                              r=s.r_hw.clone())
                    th[p].requires_grad_(True)
                    o = solve_unrolled(s, th["d"], th["rh"], ke=th["ke"],
                                       r_hw=th["r"], K=K,
                                       assemble="csr" if ls == "cudss" else "dense",
                                       linear_solver=ls)
                    L = (W * o["head_ft"]).sum()
                    g, = torch.autograd.grad(L, th[p])
                    return g.detach()

                for p in ("d", "r"):
                    c1, c2 = gr("cudss", p), gr("cudss", p)
                    d1, d2 = gr("dense", p), gr("dense", p)
                    den = d1.abs().max().clamp_min(1e-300)
                    print("   %-8s %-4d %-2d %-5s | %.4e   %.4e   %.4e | %.4e"
                          % (stem, B, K, p,
                             float((c1 - c2).abs().max() / den),
                             float((d1 - d2).abs().max() / den),
                             float((c1 - d1).abs().max() / den), float(den)))
                s.cudss_free()
                del s
            except torch.cuda.OutOfMemoryError:
                print("   %-8s %-4d  dense OOM" % (stem, B))
            except Exception:
                print("   %-8s %-4d  ERR %s" % (stem, B, err()))
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import hashlib
    import nvmath
    print("node:", NODE, "| torch", torch.__version__, "| nvmath",
          nvmath.__version__)
    print("solver.py md5:", hashlib.md5(
        open(os.path.join(ROOT, "dgga", "solver.py"), "rb").read()).hexdigest())
    for fn in (sec_S1, sec_S2):
        try:
            fn()
        except Exception:
            traceback.print_exc()
        sys.stdout.flush()
    print("\nAUD5 DONE")

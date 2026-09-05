# -*- coding: utf-8 -*-
"""AUDIT-1 (independent): gradient correctness + real factorization reuse + guards.

Written from scratch by the auditor. Does not import or reuse any script under
scripts/p3_autograd. Only the library API is used.

  A  directional finite difference (random unit direction, Richardson) on the
     DENSE forward -- one scalar per (net,param), no percentile filtering.
  B  exhaustive per-coordinate FD on Hanoi (every coordinate, no filter).
  C  cudss analytic vs dense analytic per tensor, B in {1,8,64,256}.
  D  vs ImplicitGGASolve.
  E  factorization accounting with an INDEPENDENT counter monkeypatched onto
     nvmath DirectSolver + a decisive staleness probe.
  F  guards / failure surface.
"""
import os
import sys
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                              # noqa: E402
from dgga.solver import GGASolver                             # noqa: E402
import dgga.solver as S                                       # noqa: E402
from dgga.autodiff import solve_unrolled, ImplicitGGASolve    # noqa: E402

DEV, DT = "cuda", torch.float64
NETD = os.path.join(ROOT, "p2nets")
SEED = 7717
torch.use_deterministic_algorithms(True)

NETS = [("Hanoi", "Hanoi.inp", 20), ("Net3", "Net3.inp", 20),
        ("Modena", "Modena.inp", 20), ("City_D", "City_D.inp", 24),
        ("ky4", "ky4.inp", 24), ("Pescara", "Pescara.inp", 20)]


def err():
    return traceback.format_exc().strip().split("\n")[-1][:130]


class Case:
    def __init__(self, stem, fn, B, K, emit_frac=0.25, seed=SEED):
        p = os.path.join(NETD, fn)
        self.net = parse_inp(p)
        self.s = GGASolver(self.net, device=DEV, dtype=DT, mode="dense",
                           inp_path=p, dense_tank_bound_check=False)
        self.stem, self.B, self.K = stem, B, K
        g = np.random.default_rng(seed)
        d0 = np.asarray(self.net.demand_cfs_at(0), dtype=np.float64)
        rh0 = np.array(self.net.reservoir_head_ft_at(0), dtype=np.float64)
        tn = np.asarray(self.net.tank_node, dtype=np.int64)
        if tn.size:
            lo = self.net.tank_hmin + .3 * (self.net.tank_hmax - self.net.tank_hmin)
            hi = self.net.tank_hmax - .3 * (self.net.tank_hmax - self.net.tank_hmin)
            rh0[tn] = np.clip(.5 * (self.net.tank_hmin + self.net.tank_hmax), lo, hi)
        rh0 = np.nan_to_num(rh0)
        D = d0[None, :] * g.uniform(.85, 1.15, (B, d0.size))
        R = rh0[None, :] + g.uniform(-1., 1., (B, rh0.size))
        ke = np.asarray(self.net.node_ke, dtype=np.float64).copy()
        jn = np.asarray(self.s.junc_nodes)
        pick = jn[g.random(jn.size) < emit_frac]
        ke[pick] = np.maximum(ke[pick], 1e-3)
        KE = np.repeat(ke[None, :], B, 0)
        self.n_emit = int((ke[jn] > 0).sum())
        t = lambda a: torch.as_tensor(a, dtype=DT, device=DEV)
        self.D, self.R, self.KE = t(D), t(R), t(KE)
        self.r0 = self.s.r_hw.clone()
        self.wh = t(g.normal(0, 1, (B, d0.size)))
        self.wq = t(g.normal(0, 1, (B, len(self.net.link_id))))

    def theta(self, req=()):
        out = dict(d=self.D.clone(), rh=self.R.clone(), ke=self.KE.clone(),
                   r=self.r0.clone())
        for k in req:
            out[k].requires_grad_(True)
        return out

    def loss(self, th, ls, K=None):
        asm = "csr" if ls == "cudss" else "dense"
        o = solve_unrolled(self.s, th["d"], th["rh"], ke=th["ke"], r_hw=th["r"],
                           K=self.K if K is None else K, assemble=asm,
                           linear_solver=ls)
        return (self.wh * o["head_ft"]).sum() + (self.wq * o["flow_cfs"]).sum()

    def grad(self, ls, params):
        th = self.theta(params)
        L = self.loss(th, ls)
        gs = torch.autograd.grad(L, [th[p] for p in params])
        return {p: g.detach() for p, g in zip(params, gs)}, float(L)

    def val(self, th):
        with torch.no_grad():
            return float(self.loss(th, "dense"))


def reldiff(a, b):
    den = b.abs().max().clamp_min(1e-300)
    e = (a - b).abs()
    i = int(e.view(-1).argmax())
    return float(e.max() / den), i, float(a.view(-1)[i]), float(b.view(-1)[i])


# ---------------------------------------------------------------- A
def sec_A():
    print("\n" + "=" * 78)
    print("A  directional FD on the DENSE forward, random unit direction v.")
    print("   fd = [L(t+h v) - L(t-h v)] / 2h, Richardson from h and h/2.")
    print("   NO coordinate filtering: one scalar per (net,param).")
    print("   net       B  param | g.v (cudss)      fd(Richardson)    relerr"
          "    | g.v(dense)       relerr")
    rng = np.random.default_rng(20260822)
    for stem, fn, K in NETS:
        for B in (1, 8):
            try:
                c = Case(stem, fn, B, K)
            except Exception:
                print("   %-9s %-2d  build ERR %s" % (stem, B, err()))
                continue
            for p in ("d", "rh", "ke", "r"):
                try:
                    gc, _ = c.grad("cudss", (p,))
                    gd, _ = c.grad("dense", (p,))
                    th = c.theta()
                    base = th[p]
                    v = torch.as_tensor(rng.normal(0, 1, tuple(base.shape)),
                                        dtype=DT, device=DEV)
                    v = v / v.norm()
                    scale = float(base.abs().max().clamp_min(1e-12))
                    fds = []
                    for h in (1e-5 * scale, 5e-6 * scale):
                        tp = dict(th)
                        tp[p] = base + h * v
                        tm = dict(th)
                        tm[p] = base - h * v
                        fds.append((c.val(tp) - c.val(tm)) / (2 * h))
                    f1, f2 = fds
                    fd = (4 * f2 - f1) / 3.0
                    dc = float((gc[p] * v).sum())
                    dd = float((gd[p] * v).sum())
                    den = max(abs(fd), 1e-300)
                    print("   %-9s %-2d %-5s | %+.10e  %+.10e  %.3e | "
                          "%+.10e  %.3e"
                          % (stem, B, p, dc, fd, abs(dc - fd) / den, dd,
                             abs(dd - fd) / den))
                except torch.cuda.OutOfMemoryError:
                    print("   %-9s %-2d %-5s | OOM" % (stem, B, p))
                    torch.cuda.empty_cache()
                except Exception:
                    print("   %-9s %-2d %-5s | ERR %s" % (stem, B, p, err()))
            c.s.cudss_free()
            del c
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- B
def sec_B():
    print("\n" + "=" * 78)
    print("B  EXHAUSTIVE per-coordinate FD on Hanoi, B=1, every coordinate,")
    print("   no percentile filter. max_i |g_i - fd_i| / max_i|fd_i|.")
    c = Case("Hanoi", "Hanoi.inp", 1, 20)
    for p in ("d", "rh", "ke", "r"):
        gc, _ = c.grad("cudss", (p,))
        gd, _ = c.grad("dense", (p,))
        th = c.theta()
        base = th[p]
        flat = base.reshape(-1).clone()
        n = int(flat.numel())
        scale = float(base.abs().max().clamp_min(1e-12))
        h = 1e-6 * scale
        fd = torch.zeros(n, dtype=DT, device=DEV)
        for i in range(n):
            e = torch.zeros(n, dtype=DT, device=DEV)
            e[i] = h
            tp = dict(th)
            tp[p] = (flat + e).reshape(base.shape)
            tm = dict(th)
            tm[p] = (flat - e).reshape(base.shape)
            fd[i] = (c.val(tp) - c.val(tm)) / (2 * h)
        den = fd.abs().max().clamp_min(1e-300)
        ec = (gc[p].reshape(-1) - fd).abs()
        ed = (gd[p].reshape(-1) - fd).abs()
        ic = int(ec.argmax())
        print("   %-4s n=%-4d | cudss %.3e (worst i=%d g=%+.8e fd=%+.8e) | "
              "dense %.3e | cudss-vs-dense %.3e"
              % (p, n, float(ec.max() / den), ic,
                 float(gc[p].reshape(-1)[ic]), float(fd[ic]),
                 float(ed.max() / den),
                 float((gc[p] - gd[p]).abs().max()
                       / gd[p].abs().max().clamp_min(1e-300))))
    c.s.cudss_free()


# ---------------------------------------------------------------- C
def sec_C():
    print("\n" + "=" * 78)
    print("C  cudss analytic vs dense analytic, per tensor, B in {1,8,64,256}.")
    print("   net      B    | param | relerr     argmax  gc             gd")
    for stem, fn, K in NETS:
        for B in (1, 8, 64, 256):
            c = None
            try:
                c = Case(stem, fn, B, K)
                for p in ("d", "rh", "ke", "r"):
                    gc, _ = c.grad("cudss", (p,))
                    gd, _ = c.grad("dense", (p,))
                    e, i, a, b = reldiff(gc[p], gd[p])
                    print("   %-8s %-4d | %-5s | %.4e  %-7d %+.6e %+.6e"
                          % (stem, B, p, e, i, a, b))
            except torch.cuda.OutOfMemoryError:
                print("   %-8s %-4d | dense side OOM (cudss side fine)"
                      % (stem, B))
            except Exception:
                print("   %-8s %-4d | ERR %s" % (stem, B, err()))
            if c is not None:
                c.s.cudss_free()
                del c
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- D
def sec_D():
    print("\n" + "=" * 78)
    print("D  vs ImplicitGGASolve. K = converged iters + 5, B=4.")
    print("   net      | param | cudss-vs-imp  dense-vs-imp  max|g_imp|")
    for stem, fn, _K in NETS:
        c = None
        try:
            c = Case(stem, fn, 4, 8)
            with torch.no_grad():
                probe = c.s.solve(c.D, c.R, ke_int=c.KE, assemble="dense",
                                  linear_solver="dense")
            c.K = int(probe["iters"].max()) + 5
            for p in ("d", "rh", "ke", "r"):
                gc, _ = c.grad("cudss", (p,))
                gd, _ = c.grad("dense", (p,))
                th = c.theta((p,))
                H, Q, E = ImplicitGGASolve.apply(th["d"], th["rh"], th["ke"],
                                                 th["r"], c.s, 1e-12, 200, 3,
                                                 None, None, None, None)
                L = (c.wh * H).sum() + (c.wq * Q).sum()
                gi, = torch.autograd.grad(L, th[p])
                den = gi.abs().max().clamp_min(1e-300)
                print("   %-8s | %-5s | %.4e    %.4e    %.4e"
                      % (stem, p, float((gc[p] - gi).abs().max() / den),
                         float((gd[p] - gi).abs().max() / den), float(den)))
        except Exception:
            print("   %-8s | ERR %s" % (stem, err()))
        if c is not None:
            c.s.cudss_free()
            del c
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- E
def sec_E():
    print("\n" + "=" * 78)
    print("E  INDEPENDENT factorization accounting + staleness probe.")
    print("   rawFac/rawSolve come from monkeypatching nvmath DirectSolver,")
    print("   NOT from solver.cudss_counters().")
    print("   stale = backward reuses whose cuDSS-held buffer was NOT bitwise")
    print("   equal to the csr_data forward factorized. Must be 0.")
    from nvmath.sparse.advanced import DirectSolver
    raw_f, raw_s = DirectSolver.factorize, DirectSolver.solve
    CNT = dict(fac=0, sol=0)

    def pf(self, *a, **k):
        CNT["fac"] += 1
        return raw_f(self, *a, **k)

    def ps(self, *a, **k):
        CNT["sol"] += 1
        return raw_s(self, *a, **k)

    DirectSolver.factorize, DirectSolver.solve = pf, ps

    raw_adj = S.GGASolver._cudss_adjoint
    PROBE = dict(reuse=0, refac=0, stale=0, worst=0.0)

    def adj(self, data, g, B, st, gen, slot):
        will_reuse = (st.get("solver") is not None and st.get("gen") == gen
                      and int(st.get("B", -1)) == int(B))
        if will_reuse:
            PROBE["reuse"] += 1
            d = float((st["vals"] - data).abs().max())
            PROBE["worst"] = max(PROBE["worst"], d)
            if d != 0.0:
                PROBE["stale"] += 1
        else:
            PROBE["refac"] += 1
        return raw_adj(self, data, g, B, st, gen, slot)

    S.GGASolver._cudss_adjoint = adj

    print("   net    K  slots | rawFac rawSolve | reuse refac stale worstDelta"
          " | lib(fac,sol,bsol,breuse,brefac) | grad vs slots=K+2")
    for stem, fn in (("Modena", "Modena.inp"), ("ky4", "ky4.inp"),
                     ("Net3", "Net3.inp")):
        K = 12
        ref = None
        for slots in (K + 2, 1, 2, 4, K):
            c = Case(stem, fn, 8, K)
            c.s.cudss_cache_max = K + 4
            c.s.cudss_grad_slots = slots
            c.s.cudss_counters(reset=True)
            CNT["fac"] = CNT["sol"] = 0
            PROBE["reuse"] = PROBE["refac"] = PROBE["stale"] = 0
            PROBE["worst"] = 0.0
            g, _ = c.grad("cudss", ("d", "rh", "ke", "r"))
            lib = c.s.cudss_counters()
            if ref is None:
                ref = g
                cmp = 0.0
            else:
                cmp = max(float((g[p] - ref[p]).abs().max()
                                / ref[p].abs().max().clamp_min(1e-300))
                          for p in g)
            print("   %-6s %-2d %-5d | %-6d %-8d | %-5d %-5d %-5d %.3e | "
                  "(%d,%d,%d,%d,%d) | %.3e"
                  % (stem, K, slots, CNT["fac"], CNT["sol"], PROBE["reuse"],
                     PROBE["refac"], PROBE["stale"], PROBE["worst"],
                     lib["factorize"], lib["solve"], lib["bwd_solve"],
                     lib["bwd_reuse"], lib["bwd_refactorize"], cmp))
            c.s.cudss_free()
            del c
            torch.cuda.empty_cache()

    print("\n   E2  eviction hazard: cache_max smaller than the live slots is")
    print("       refused up front, but a *foreign* solve between forward and")
    print("       backward can still evict. Probe: run a second solve at a")
    print("       different B in between and check the gradient still matches.")
    c = Case("Modena", "Modena.inp", 8, 10)
    c.s.cudss_cache_max = 12
    c.s.cudss_grad_slots = 12
    th = c.theta(("d",))
    L = c.loss(th, "cudss")
    # foreign traffic between forward and backward, at other batch sizes
    c2 = Case("Modena", "Modena.inp", 3, 4)
    c2.s = c.s
    with torch.no_grad():
        for bb in (3, 5, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37):
            g2 = np.random.default_rng(bb)
            D2 = torch.as_tensor(
                np.asarray(c.net.demand_cfs_at(0))[None, :]
                * g2.uniform(.9, 1.1, (bb, 1)), dtype=DT, device=DEV)
            R2 = c.R[:1].expand(bb, -1)
            solve_unrolled(c.s, D2, R2, K=3, assemble="csr",
                           linear_solver="cudss")
    PROBE["reuse"] = PROBE["refac"] = PROBE["stale"] = 0
    PROBE["worst"] = 0.0
    gpo, = torch.autograd.grad(L, th["d"])
    print("       after foreign traffic: reuse=%d refac=%d stale=%d worst=%.3e"
          % (PROBE["reuse"], PROBE["refac"], PROBE["stale"], PROBE["worst"]))
    c.s.cudss_grad_slots = 1
    c.s.cudss_cache_max = 8
    c.s.cudss_free()
    gcl, _ = c.grad("cudss", ("d",))
    gdn, _ = c.grad("dense", ("d",))
    den = gdn["d"].abs().max().clamp_min(1e-300)
    print("       grad(with eviction) vs dense: %.3e ; grad(clean) vs dense:"
          " %.3e" % (float((gpo - gdn["d"]).abs().max() / den),
                     float((gcl["d"] - gdn["d"]).abs().max() / den)))
    c.s.cudss_free()
    del c, c2
    torch.cuda.empty_cache()

    S.GGASolver._cudss_adjoint = raw_adj
    DirectSolver.factorize, DirectSolver.solve = raw_f, raw_s


# ---------------------------------------------------------------- F
def sec_F():
    print("\n" + "=" * 78)
    print("F  failure surface.")
    c = Case("Hanoi", "Hanoi.inp", 2, 8)

    def probe(name, fn):
        try:
            fn()
            print("   %-46s -> NO RAISE  <-- inspect" % name)
        except Exception as e:
            print("   %-46s -> %s: %s"
                  % (name, type(e).__name__, str(e).replace("\n", " ")[:80]))

    def second_order():
        th = c.theta(("d",))
        L = c.loss(th, "cudss")
        g, = torch.autograd.grad(L, th["d"], create_graph=True)
        torch.autograd.grad(g.sum(), th["d"])

    probe("2nd derivative (create_graph=True)", second_order)

    def asm_dense():
        th = c.theta(("d",))
        solve_unrolled(c.s, th["d"], th["rh"], ke=th["ke"], r_hw=th["r"],
                       K=4, assemble="dense", linear_solver="cudss")

    probe("cudss + assemble='dense'", asm_dense)

    def f32():
        p = os.path.join(NETD, "Hanoi.inp")
        s32 = GGASolver(parse_inp(p), device=DEV, dtype=torch.float32,
                        mode="dense", inp_path=p, dense_tank_bound_check=False)
        d = c.D.to(torch.float32).requires_grad_(True)
        o = solve_unrolled(s32, d, c.R.to(torch.float32), K=4,
                           assemble="csr", linear_solver="cudss")
        o["head_ft"].sum().backward()

    probe("cudss + float32 (differentiable path)", f32)

    def oncpu():
        p = os.path.join(NETD, "Hanoi.inp")
        sc = GGASolver(parse_inp(p), device="cpu", dtype=DT, mode="dense",
                       inp_path=p, dense_tank_bound_check=False)
        d = c.D.cpu().clone().requires_grad_(True)
        o = solve_unrolled(sc, d, c.R.cpu(), K=4, assemble="csr",
                           linear_solver="cudss")
        o["head_ft"].sum().backward()

    probe("cudss + device='cpu' (differentiable path)", oncpu)

    def slots_gt_cap():
        c.s.cudss_cache_max = 2
        c.s.cudss_grad_slots = 8
        try:
            th = c.theta(("d",))
            c.loss(th, "cudss")
        finally:
            c.s.cudss_cache_max = 8
            c.s.cudss_grad_slots = 1

    probe("cudss_grad_slots > cudss_cache_max", slots_gt_cap)

    def epanet_mode():
        p = os.path.join(NETD, "Net3.inp")
        n3 = parse_inp(p)
        se = GGASolver(n3, device=DEV, dtype=DT, mode="epanet", inp_path=p)
        se.solve(np.asarray(n3.demand_cfs_at(0)),
                 np.nan_to_num(np.asarray(n3.reservoir_head_ft_at(0))),
                 assemble="csr", linear_solver="cudss")

    probe("cudss + mode='epanet'", epanet_mode)

    def prv_net():
        p = os.path.join(NETD, "Net2.inp")
        n2 = parse_inp(p)
        s2 = GGASolver(n2, device=DEV, dtype=DT, mode="dense", inp_path=p,
                       dense_tank_bound_check=False)
        d = torch.as_tensor(np.asarray(n2.demand_cfs_at(0))[None, :], dtype=DT,
                            device=DEV).requires_grad_(True)
        rh = torch.as_tensor(
            np.nan_to_num(np.asarray(n2.reservoir_head_ft_at(0)))[None, :],
            dtype=DT, device=DEV)
        o = solve_unrolled(s2, d, rh, K=6, assemble="csr",
                           linear_solver="cudss")
        o["head_ft"].sum().backward()

    probe("cudss on a net with unsupported link types", prv_net)

    def no_grad_ok():
        with torch.no_grad():
            th = c.theta()
            c.loss(th, "cudss")

    probe("cudss forward under no_grad (must NOT raise)", no_grad_ok)

    def detached_only():
        # only F requires grad, csr_data does not -> needs_input_grad[0] False
        th = c.theta()
        o = solve_unrolled(c.s, th["d"].requires_grad_(True), th["rh"],
                           ke=th["ke"], r_hw=th["r"], K=4, assemble="csr",
                           linear_solver="cudss")
        o["head_ft"].sum().backward()

    probe("cudss backward with only demand requiring grad", detached_only)
    c.s.cudss_free()


if __name__ == "__main__":
    import hashlib
    import nvmath
    print("node:", os.popen("hostname").read().strip(),
          "| torch", torch.__version__, "| dev", torch.cuda.get_device_name(0),
          "| nvmath", nvmath.__version__)
    print("solver.py md5:", hashlib.md5(
        open(os.path.join(ROOT, "dgga", "solver.py"), "rb").read()).hexdigest())
    for name, fn in (("A", sec_A), ("B", sec_B), ("C", sec_C), ("D", sec_D),
                     ("E", sec_E), ("F", sec_F)):
        try:
            fn()
        except Exception:
            print("SECTION %s CRASHED: %s" % (name, err()))
            traceback.print_exc()
        sys.stdout.flush()
    print("\nAUD1 DONE")

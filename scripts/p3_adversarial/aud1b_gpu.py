# -*- coding: utf-8 -*-
"""AUDIT-1b: the finite-difference study done properly.

Round one exposed that a naive FD on `ke` is meaningless: the emitter branch is
gated on ke > 0, so perturbing a node whose ke is exactly 0 flips the emitter on
for the + step and off for the - step.  The forward is genuinely discontinuous
there and the difference quotient blows up (1e10 vs an analytic 1e3).  Here the
ke direction is restricted to nodes that already carry ke > 0, and the step is
relative so it can never cross zero.

Also: instead of a single step size, sweep h over four decades and print the
whole curve, so the reader can see the FD converge (or fail to) rather than
taking one number on faith.
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
from dgga.autodiff import solve_unrolled                      # noqa: E402

DEV, DT = "cuda", torch.float64
NETD = os.path.join(ROOT, "p2nets")
torch.use_deterministic_algorithms(True)
NODE = os.popen("hostname").read().strip()


def err():
    return traceback.format_exc().strip().split("\n")[-1][:130]


class Case:
    def __init__(self, stem, fn, B, K, emit_frac=0.25, seed=7717):
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
            rh0[tn] = np.clip(.5 * (self.net.tank_hmin + self.net.tank_hmax),
                              lo, hi)
        rh0 = np.nan_to_num(rh0)
        D = d0[None, :] * g.uniform(.85, 1.15, (B, d0.size))
        R = rh0[None, :] + g.uniform(-1., 1., (B, rh0.size))
        ke = np.asarray(self.net.node_ke, dtype=np.float64).copy()
        jn = np.asarray(self.s.junc_nodes)
        pick = jn[g.random(jn.size) < emit_frac]
        ke[pick] = np.maximum(ke[pick], 1e-3)
        KE = np.repeat(ke[None, :], B, 0)
        t = lambda a: torch.as_tensor(a, dtype=DT, device=DEV)
        self.D, self.R, self.KE = t(D), t(R), t(KE)
        self.r0 = self.s.r_hw.clone()
        self.wh = t(g.normal(0, 1, (B, d0.size)))
        self.wq = t(g.normal(0, 1, (B, len(self.net.link_id))))
        self.n_emit = int((ke[jn] > 0).sum())

    def theta(self, req=()):
        out = dict(d=self.D.clone(), rh=self.R.clone(), ke=self.KE.clone(),
                   r=self.r0.clone())
        for k in req:
            out[k].requires_grad_(True)
        return out

    def loss(self, th, ls):
        asm = "csr" if ls == "cudss" else "dense"
        o = solve_unrolled(self.s, th["d"], th["rh"], ke=th["ke"],
                           r_hw=th["r"], K=self.K, assemble=asm,
                           linear_solver=ls)
        return (self.wh * o["head_ft"]).sum() + (self.wq * o["flow_cfs"]).sum()

    def grad(self, ls, p):
        th = self.theta((p,))
        g, = torch.autograd.grad(self.loss(th, ls), th[p])
        return g.detach()

    def val(self, th):
        with torch.no_grad():
            return float(self.loss(th, "dense"))


def direction(base, rng, active=None):
    """Relative direction: v_i = base_i * u_i, so a step of h moves every
    coordinate by a relative h*u_i and never crosses zero."""
    u = torch.as_tensor(rng.normal(0, 1, tuple(base.shape)), dtype=DT,
                        device=DEV)
    if active is not None:
        u = u * active
    v = base * u
    n = v.norm()
    return v / n.clamp_min(1e-300)


def sweep(c, p, gc, gd, active=None, rng=None):
    th = c.theta()
    base = th[p]
    v = direction(base, rng, active)
    dc = float((gc * v).sum())
    dd = float((gd * v).sum())
    print("      analytic  cudss %+.12e | dense %+.12e | rel %.3e"
          % (dc, dd, abs(dc - dd) / max(abs(dd), 1e-300)))
    print("      %-10s %-22s %-11s %-11s" % ("h", "fd(h)", "rel(cudss)",
                                             "rel(dense)"))
    prev = None
    for h in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6):
        tp = dict(th)
        tp[p] = base + h * v
        tm = dict(th)
        tm[p] = base - h * v
        fd = (c.val(tp) - c.val(tm)) / (2 * h)
        den = max(abs(fd), 1e-300)
        mark = ""
        if prev is not None:
            mark = "  (rich %+.12e)" % ((4 * fd - prev) / 3.0)
        print("      %-10.0e %+.12e  %.3e   %.3e%s"
              % (h, fd, abs(dc - fd) / den, abs(dd - fd) / den, mark))
        prev = fd


def main():
    rng = np.random.default_rng(20260822)
    for stem, fn, K in (("Hanoi", "Hanoi.inp", 20), ("Net3", "Net3.inp", 20),
                        ("Modena", "Modena.inp", 20),
                        ("Pescara", "Pescara.inp", 20)):
        for B in (1, 8):
            try:
                c = Case(stem, fn, B, K)
            except Exception:
                print("%s B=%d build ERR %s" % (stem, B, err()))
                continue
            act = (c.KE > 0).to(DT)
            print("\n" + "-" * 70)
            print("%s  B=%d  K=%d  Nj=%d  nnz=%d  active emitters=%d"
                  % (stem, B, K, c.s.Nj, c.s.A_csr_nnz, c.n_emit))
            for p in ("d", "rh", "ke", "r"):
                try:
                    gc = c.grad("cudss", p)
                    gd = c.grad("dense", p)
                    print("   param %s" % p)
                    sweep(c, p, gc, gd,
                          active=(act if p == "ke" else None), rng=rng)
                except Exception:
                    print("   param %s ERR %s" % (p, err()))
            c.s.cudss_free()
            del c
            torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("B2  exhaustive per-coordinate FD on Hanoi, B=1, relative step 1e-6,")
    print("    ke restricted to nodes that already carry ke>0.")
    c = Case("Hanoi", "Hanoi.inp", 1, 20)
    for p in ("d", "rh", "ke", "r"):
        gc = c.grad("cudss", p)
        gd = c.grad("dense", p)
        th = c.theta()
        base = th[p]
        flat = base.reshape(-1).clone()
        n = int(flat.numel())
        keep = (flat.abs() > 0) if p == "ke" else torch.ones_like(flat).bool()
        fd = torch.full((n,), float("nan"), dtype=DT, device=DEV)
        for i in range(n):
            if not bool(keep[i]):
                continue
            step = 1e-6 * max(abs(float(flat[i])), 1e-12)
            e = torch.zeros(n, dtype=DT, device=DEV)
            e[i] = step
            tp = dict(th)
            tp[p] = (flat + e).reshape(base.shape)
            tm = dict(th)
            tm[p] = (flat - e).reshape(base.shape)
            fd[i] = (c.val(tp) - c.val(tm)) / (2 * step)
        m = torch.isfinite(fd)
        den = fd[m].abs().max().clamp_min(1e-300)
        ec = (gc.reshape(-1)[m] - fd[m]).abs()
        ed = (gd.reshape(-1)[m] - fd[m]).abs()
        ic = int(ec.argmax())
        print("   %-4s tested %d/%d coords | cudss %.3e | dense %.3e | "
              "cudss-vs-dense %.3e | worst coord g=%+.8e fd=%+.8e"
              % (p, int(m.sum()), n, float(ec.max() / den),
                 float(ed.max() / den),
                 float((gc - gd).abs().max()
                       / gd.abs().max().clamp_min(1e-300)),
                 float(gc.reshape(-1)[m][ic]), float(fd[m][ic])))
    c.s.cudss_free()


if __name__ == "__main__":
    import hashlib
    import nvmath
    print("node:", NODE, "| torch", torch.__version__, "| dev",
          torch.cuda.get_device_name(0), "| nvmath", nvmath.__version__)
    print("solver.py md5:", hashlib.md5(
        open(os.path.join(ROOT, "dgga", "solver.py"), "rb").read()).hexdigest())
    try:
        main()
    except Exception:
        traceback.print_exc()
    print("\nAUD1B DONE")

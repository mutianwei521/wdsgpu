# -*- coding: utf-8 -*-
"""P4-D: what the default cudss_cache_max=8 actually costs (audit D5).

F4 reported "the bounded cache costs no time (75.4 vs 74.9 s)".  The audit
found that number was measured on a PURE FORWARD loop whose batch sizes lived
in [177, 256] -- almost no eviction pressure.  With backward and B in [8, 600)
the same loop paid 10.4%.  Both loops are run here, in FRESH PROCESSES so the
cap=None run cannot warm the pool for the cap=8 run (the audit ran them back to
back in one process, which biases whichever goes second), and each is repeated
twice so the 10.4% can be read against its own run-to-run noise.

  kind=fwd  : pure forward s.solve, B ~ U[177, 256]     (the F4 caliber)
  kind=fb   : solve_unrolled + backward, B ~ U[8, 600)  (the audit caliber)
"""
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
NETD = os.path.join(ROOT, "p2nets")
DEV, DT = "cuda", torch.float64
NODE = os.popen("hostname").read().strip()
NB = int(os.environ.get("P4D_NB", "80"))


def dev_used():
    torch.cuda.synchronize()
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


def build(fn, B, seed=4242):
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
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
    return net, s, D, R, W


def worker(kind, cap, rep):
    from dgga.autodiff import solve_unrolled
    lo, hi = (177, 257) if kind == "fwd" else (8, 600)
    rng = np.random.default_rng(31337)
    sizes = [int(x) for x in rng.integers(lo, hi, NB)]
    net, s, D, R, W = build("ky4.inp", max(sizes))
    s.cudss_cache_max = None if cap == "none" else int(cap)
    # one warm-up batch so plan/allocator start-up is not charged to the loop
    with torch.no_grad():
        s.solve(D[:16], R[:16], assemble="csr", linear_solver="cudss")
    base = dev_used()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trace = []
    for i, B in enumerate(sizes):
        if kind == "fwd":
            with torch.no_grad():
                s.solve(D[:B], R[:B], assemble="csr", linear_solver="cudss")
        else:
            d = D[:B].clone().requires_grad_(True)
            o = solve_unrolled(s, d, R[:B], K=4, assemble="csr",
                               linear_solver="cudss")
            (W[:B] * o["head_ft"]).sum().backward()
        if i % 16 == 15 or i == len(sizes) - 1:
            trace.append((i + 1, dev_used()))
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    print("RESULT kind=%s cap=%s rep=%d elapsed=%.2f base=%.1f states=%d "
          "final=%.1f growth=%.1f trace=%s"
          % (kind, cap, rep, el, base, len(s._cudss_cache), trace[-1][1],
             trace[-1][1] - base,
             ",".join("%d:%.0f" % (n, m) for n, m in trace)))


def pick(s, key):
    for tok in s.split():
        if tok.startswith(key + "="):
            return tok.split("=", 1)[1]
    return None


def main():
    print("=" * 78)
    print("G1  ragged-batch loop, %d batches, ky4, FRESH PROCESS per cell"
          " (node %s)" % (NB, NODE))
    print("    kind=fwd: pure forward, B~U[177,257)  (the F4 caliber)")
    print("    kind=fb : fwd+bwd,      B~U[8,600)    (the audit caliber)")
    got = {}
    for kind in ("fwd", "fb"):
        for cap in ("none", "8"):
            for rep in (1, 2):
                r = subprocess.run(
                    [sys.executable, "-X", "utf8", os.path.abspath(__file__),
                     "worker", kind, cap, str(rep)],
                    capture_output=True, text=True, cwd=ROOT, timeout=3600)
                ln = [x for x in r.stdout.splitlines()
                      if x.startswith("RESULT")]
                if not ln:
                    print("    %s cap=%s rep=%d CRASH rc=%d %s"
                          % (kind, cap, rep, r.returncode,
                             r.stderr.strip().splitlines()[-1][:110]
                             if r.stderr.strip() else ""))
                    continue
                print("    " + ln[0])
                got[(kind, cap, rep)] = float(pick(ln[0], "elapsed"))
                sys.stdout.flush()
    print("\n    summary: cost of the default cache_max=8 vs unbounded")
    for kind in ("fwd", "fb"):
        a = [got.get((kind, "none", r)) for r in (1, 2)]
        b = [got.get((kind, "8", r)) for r in (1, 2)]
        if None in a or None in b:
            continue
        print("      kind=%-3s  none: %.2f / %.2f s   cap8: %.2f / %.2f s"
              "  -> cap8 costs %+.1f%% (rep1) / %+.1f%% (rep2);"
              "  own rerun noise none %+.1f%%, cap8 %+.1f%%"
              % (kind, a[0], a[1], b[0], b[1],
                 100 * (b[0] / a[0] - 1), 100 * (b[1] / a[1] - 1),
                 100 * (a[1] / a[0] - 1), 100 * (b[1] / b[0] - 1)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        try:
            worker(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        except Exception:
            traceback.print_exc()
        sys.exit(0)
    import hashlib
    print("node:", NODE, "| torch", torch.__version__)
    print("solver.py md5:", hashlib.md5(
        open(os.path.join(ROOT, "dgga", "solver.py"), "rb").read()).hexdigest())
    try:
        main()
    except Exception:
        traceback.print_exc()
    print("\nP4D DONE")

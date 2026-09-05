# -*- coding: utf-8 -*-
"""P4-E: the control run D5 needs.

p4d found that in a PURE FORWARD ragged loop over B ~ U[177,257) the default
cudss_cache_max=8 costs +40%, while F4 reported "same time" for what it called
the same caliber.  The suspected reason is not forward-vs-backward at all:
F4's loop was `range(177, 257)` -- 80 CONSECUTIVE batch sizes, each visited
exactly once.  With no repeat there is nothing for a cache to hit, so a cap
cannot cost anything.  The moment batch sizes recur (which is what a bucketed
loader does) the cap starts discarding plans that would have been reused.

Both loops are run here, same net, same path, fresh process per cell:
    kind=uniq  B = 177,178,...,256          (F4's exact list, 0 repeats)
    kind=rep   B ~ U[177,257) with replacement (50 distinct out of 80)
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


def dev_used():
    torch.cuda.synchronize()
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


def worker(kind, cap, rep):
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    if kind == "uniq":
        sizes = list(range(177, 257))
    else:
        sizes = [int(x) for x in
                 np.random.default_rng(31337).integers(177, 257, 80)]
    p = os.path.join(NETD, "ky4.inp")
    net = parse_inp(p)
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=p,
                  dense_tank_bound_check=False)
    s.cudss_cache_max = None if cap == "none" else int(cap)
    g = np.random.default_rng(4242)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + .3 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - .3 * (net.tank_hmax - net.tank_hmin)
        rh0[tn] = np.clip(.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    rh0 = np.nan_to_num(rh0)
    M = max(sizes)
    D = torch.as_tensor(d0[None, :] * g.uniform(.85, 1.15, (M, d0.size)),
                        dtype=DT, device=DEV)
    R = torch.as_tensor(rh0[None, :] + g.uniform(-1., 1., (M, rh0.size)),
                        dtype=DT, device=DEV)
    with torch.no_grad():
        s.solve(D[:16], R[:16], assemble="csr", linear_solver="cudss")
    base = dev_used()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for B in sizes:
            s.solve(D[:B], R[:B], assemble="csr", linear_solver="cudss")
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    print("RESULT kind=%s cap=%s rep=%d nB=%d distinct=%d elapsed=%.2f "
          "base=%.1f states=%d final=%.1f"
          % (kind, cap, rep, len(sizes), len(set(sizes)), el, base,
             len(s._cudss_cache), dev_used()))


def pick(s, key):
    for tok in s.split():
        if tok.startswith(key + "="):
            return tok.split("=", 1)[1]
    return None


def main():
    print("=" * 78)
    print("H1  does the cap cost anything when no batch size ever repeats?"
          "  (ky4, pure forward, FRESH PROCESS per cell, node %s)" % NODE)
    print("    kind=uniq: B = 177..256, each once      (F4's exact loop)")
    print("    kind=rep : B ~ U[177,257) with replacement")
    got = {}
    for kind in ("uniq", "rep"):
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
    print("\n    summary")
    for kind in ("uniq", "rep"):
        a = [got.get((kind, "none", r)) for r in (1, 2)]
        b = [got.get((kind, "8", r)) for r in (1, 2)]
        if None in a or None in b:
            continue
        print("      kind=%-4s none: %.2f / %.2f s   cap8: %.2f / %.2f s"
              "  -> cap8 costs %+.1f%% / %+.1f%%  (own rerun noise"
              " none %+.1f%%, cap8 %+.1f%%)"
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
    print("\nP4E DONE")

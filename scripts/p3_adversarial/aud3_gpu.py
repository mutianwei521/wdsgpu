# -*- coding: utf-8 -*-
"""AUDIT-3 (independent): memory, and whether the F4 cache fix really returns
device memory.

  M0  worker mode: one config, fresh process, prints peak memory for
      forward+backward.  Driven by M1 through subprocess so every number comes
      from a clean CUDA context.
  M1  driver: dense vs cudss peak, fresh process per config.  Reports BOTH the
      torch allocator peak AND the device-resident footprint (nvidia driver
      view), because cuDSS's factor buffers are invisible to torch.
  M2  ragged batch-size loop: does device usage climb without bound when the
      LRU cap is removed, and stay flat with the default cap of 8?
  M3  does cudss_free() actually hand memory back?  Device-used before/after,
      plus a 20-round free/rebuild loop checked for monotone drift.
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


def err():
    return traceback.format_exc().strip().split("\n")[-1][:130]


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


# ---------------------------------------------------------------- M0
def worker(fn, B, mode, K):
    from dgga.autodiff import solve_unrolled
    base = dev_used()
    net, s, D, R, W = build(fn, B)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pre = dev_used()
    try:
        d = D.clone().requires_grad_(True)
        asm = "csr" if mode == "cudss" else "dense"
        o = solve_unrolled(s, d, R, K=K, assemble=asm, linear_solver=mode)
        (W * o["head_ft"]).sum().backward()
        torch.cuda.synchronize()
        peak_t = torch.cuda.max_memory_allocated() / 2 ** 20
        resv = torch.cuda.max_memory_reserved() / 2 ** 20
        post = dev_used()
        nontorch = post - torch.cuda.memory_reserved() / 2 ** 20
        print("RESULT %s %d %s ok torchpeak=%.2f torchreserved=%.2f "
              "devbase=%.2f devpre=%.2f devpost=%.2f nontorch=%.2f gnorm=%.6e"
              % (fn, B, mode, peak_t, resv, base, pre, post, nontorch,
                 float(d.grad.norm())))
    except torch.cuda.OutOfMemoryError:
        print("RESULT %s %d %s OOM devbase=%.2f" % (fn, B, mode, base))
    except Exception:
        print("RESULT %s %d %s ERR %s" % (fn, B, mode, err()))


# ---------------------------------------------------------------- M1
def sec_M1():
    print("\n" + "=" * 78)
    print("M1  forward+backward peak memory, FRESH PROCESS per config "
          "(node %s)" % NODE)
    print("   torchpeak  = torch.cuda.max_memory_allocated")
    print("   nontorch   = device-resident bytes NOT in torch's allocator")
    print("                (cuDSS factor/work buffers live here)")
    print("   net    Nj   B    K | dense torchpeak | cudss torchpeak  nontorch"
          "  cudss total | ratio(torch) ratio(total)")
    cfg = [("Net3.inp", "Net3", 92, 7), ("Modena.inp", "Modena", 268, 7),
           ("City_D.inp", "City_D", 541, 7), ("ky4.inp", "ky4", 959, 8)]
    for fn, stem, Nj, K in cfg:
        for B in (64, 128, 256, 512):
            got = {}
            for mode in ("dense", "cudss"):
                cmd = [sys.executable, "-X", "utf8", os.path.abspath(__file__),
                       "worker", fn, str(B), mode, str(K)]
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   cwd=ROOT, timeout=1800)
                line = [x for x in r.stdout.splitlines()
                        if x.startswith("RESULT")]
                got[mode] = line[0] if line else ("RESULT crash rc=%d %s"
                                                  % (r.returncode,
                                                     r.stderr[-160:]))

            def pick(s, key):
                for tok in s.split():
                    if tok.startswith(key + "="):
                        return float(tok.split("=")[1])
                return None
            dp = pick(got["dense"], "torchpeak")
            cp = pick(got["cudss"], "torchpeak")
            nt = pick(got["cudss"], "nontorch")
            if cp is None:
                print("   %-6s %-4d %-4d %d | %s" % (stem, Nj, B, K, got["cudss"]))
                continue
            tot = cp + (nt or 0.0)
            ds = "     OOM     " if dp is None else "%13.2f" % dp
            rt = "   -  " if dp is None else "%6.2f" % (dp / cp)
            rr = "   -  " if dp is None else "%6.2f" % (dp / tot)
            print("   %-6s %-4d %-4d %d | %s | %9.2f %9.2f %9.2f | %s %s"
                  % (stem, Nj, B, K, ds, cp, nt or float("nan"), tot, rt, rr))


# ---------------------------------------------------------------- M2
def sec_M2():
    print("\n" + "=" * 78)
    print("M2  ragged batch sizes: bounded LRU vs unbounded (node %s)" % NODE)
    from dgga.autodiff import solve_unrolled
    rng = np.random.default_rng(31337)
    sizes = [int(x) for x in rng.integers(8, 600, 80)]
    for cap in (None, 8):
        net, s, D, R, W = build("ky4.inp", 600)
        s.cudss_cache_max = cap
        t0 = time.perf_counter()
        base = dev_used()
        trace = []
        for i, B in enumerate(sizes):
            d = D[:B].clone().requires_grad_(True)
            o = solve_unrolled(s, d, R[:B], K=4, assemble="csr",
                               linear_solver="cudss")
            (W[:B] * o["head_ft"]).sum().backward()
            if i % 8 == 7 or i == len(sizes) - 1:
                trace.append((i + 1, dev_used()))
        el = time.perf_counter() - t0
        print("   cache_max=%-4s  base=%.1f MiB  elapsed=%.1f s  states=%d"
              % (str(cap), base, el, len(s._cudss_cache)))
        print("      device used MiB after n batches: "
              + " ".join("%d:%.0f" % (n, m) for n, m in trace))
        peak = max(m for _, m in trace)
        print("      max=%.1f  final=%.1f  growth over run=%.1f MiB"
              % (peak, trace[-1][1], trace[-1][1] - trace[0][1]))
        s.cudss_free()
        del s
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- M3
def sec_M3():
    print("\n" + "=" * 78)
    print("M3  does cudss_free() hand device memory back? (node %s)" % NODE)
    from dgga.autodiff import solve_unrolled
    net, s, D, R, W = build("ky4.inp", 256)
    s.cudss_cache_max = None
    a = dev_used()
    with torch.no_grad():
        solve_unrolled(s, D, R, K=4, assemble="csr", linear_solver="cudss")
    b = dev_used()
    n = s.cudss_free()
    c = dev_used()
    torch.cuda.empty_cache()
    d = dev_used()
    print("   ky4 B=256: before=%.1f  after 1 solve=%.1f  after cudss_free(%d)"
          "=%.1f  after empty_cache=%.1f MiB" % (a, b, n, c, d))
    print("   -> one state costs %.1f MiB device; free() returned %.1f MiB to"
          " the driver, %.1f MiB stayed in a reusable pool"
          % (b - a, b - c, c - a))
    # does empty_cache alone recover it?  (the F4 claim)
    with torch.no_grad():
        solve_unrolled(s, D, R, K=4, assemble="csr", linear_solver="cudss")
    e = dev_used()
    torch.cuda.empty_cache()
    f = dev_used()
    print("   rebuild=%.1f ; torch.cuda.empty_cache() alone -> %.1f MiB "
          "(recovered %.1f)" % (e, f, e - f))
    s.cudss_free()
    # 20 free/rebuild rounds: monotone drift?
    seq = []
    for i in range(20):
        with torch.no_grad():
            solve_unrolled(s, D, R, K=3, assemble="csr", linear_solver="cudss")
        seq.append(dev_used())
        s.cudss_free()
    print("   20 x (build ky4 B=256 -> free): device used at each build:")
    print("      " + " ".join("%.0f" % x for x in seq))
    print("      monotone increasing? %s ; first=%.1f last=%.1f max=%.1f"
          % (all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1)),
             seq[0], seq[-1], max(seq)))
    # cross-B reuse
    s.cudss_free()
    with torch.no_grad():
        solve_unrolled(s, D[:64], R[:64], K=3, assemble="csr",
                       linear_solver="cudss")
    g1 = dev_used()
    s.cudss_free()
    with torch.no_grad():
        solve_unrolled(s, D, R, K=3, assemble="csr", linear_solver="cudss")
    g2 = dev_used()
    print("   after free(B=64) build B=256 -> %.1f MiB (B=64 build was %.1f)"
          % (g2, g1))
    del s
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
        sys.exit(0)
    import hashlib
    import nvmath
    print("node:", NODE, "| torch", torch.__version__, "| dev",
          torch.cuda.get_device_name(0), "| nvmath", nvmath.__version__)
    print("solver.py md5:", hashlib.md5(
        open(os.path.join(ROOT, "dgga", "solver.py"), "rb").read()).hexdigest())
    for name, fn in (("M1", sec_M1), ("M2", sec_M2), ("M3", sec_M3)):
        try:
            fn()
        except Exception:
            print("SECTION %s CRASHED: %s" % (name, err()))
            traceback.print_exc()
        sys.stdout.flush()
    print("\nAUD3 DONE")

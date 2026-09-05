# -*- coding: utf-8 -*-
"""AUDIT-4: honest device-memory accounting for forward+backward.

Round three showed that the P3 memory table reports torch.cuda.max_memory_
allocated only, which is blind to everything cuDSS allocates itself.  Here the
CUDA context is measured first and subtracted, so the number reported for each
path is what that path actually costs the device on top of an empty context:

    torch_peak     torch.cuda.max_memory_allocated  (what P3 §4 reports)
    torch_resv     torch.cuda.max_memory_reserved   (what torch holds of the card)
    ctx            device-resident bytes right after CUDA init, before anything
    nontorch       device-resident bytes at the end that are neither torch's
                   reservation nor the context  ->  this is cuDSS's own memory
    total          torch_resv + nontorch          (the OOM-relevant footprint)

One fresh process per (net, B, path).
"""
import os
import subprocess
import sys
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


def used():
    torch.cuda.synchronize()
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


def resv():
    return torch.cuda.memory_reserved() / 2 ** 20


def worker(fn, B, mode, K):
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    from dgga.autodiff import solve_unrolled
    torch.cuda.init()
    torch.zeros(1, device=DEV)          # force context + primary allocator
    torch.cuda.synchronize()
    ctx = used() - resv()
    p = os.path.join(NETD, fn)
    net = parse_inp(p)
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=p,
                  dense_tank_bound_check=False)
    g = np.random.default_rng(4242)
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
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        d = D.clone().requires_grad_(True)
        asm = "csr" if mode == "cudss" else "dense"
        o = solve_unrolled(s, d, R, K=K, assemble=asm, linear_solver=mode)
        (W * o["head_ft"]).sum().backward()
        torch.cuda.synchronize()
        tp = torch.cuda.max_memory_allocated() / 2 ** 20
        tr = torch.cuda.max_memory_reserved() / 2 ** 20
        nt = used() - resv() - ctx
        print("RESULT %s %d %s ok ctx=%.2f torch_peak=%.2f torch_resv=%.2f "
              "nontorch=%.2f total=%.2f gnorm=%.8e"
              % (fn, B, mode, ctx, tp, tr, nt, tr + nt, float(d.grad.norm())))
    except torch.cuda.OutOfMemoryError:
        print("RESULT %s %d %s OOM ctx=%.2f" % (fn, B, mode, ctx))
    except Exception:
        print("RESULT %s %d %s ERR %s" % (fn, B, mode, err()))


def pick(s, key):
    for tok in s.split():
        if tok.startswith(key + "="):
            return float(tok.split("=")[1])
    return None


def main():
    print("=" * 78)
    print("N1  forward+backward device footprint, FRESH PROCESS per config"
          " (node %s)" % NODE)
    print("   'P3 table' column = torch_peak, i.e. the number data/"
          "p3_autograd_wip.txt §4 reports for cudss.")
    print("   'total' = torch reserved + cuDSS's own device memory, context"
          " already subtracted.")
    print("   net    Nj   B    K | dense: peak  resv  nontorch total | "
          "cudss: peak  resv  nontorch total | P3ratio  honest")
    for fn, stem, Nj, K in (("Net3.inp", "Net3", 92, 7),
                            ("Modena.inp", "Modena", 268, 7),
                            ("City_D.inp", "City_D", 541, 7),
                            ("ky4.inp", "ky4", 959, 8)):
        for B in (64, 128, 256, 512, 1024):
            got = {}
            for mode in ("dense", "cudss"):
                r = subprocess.run(
                    [sys.executable, "-X", "utf8", os.path.abspath(__file__),
                     "worker", fn, str(B), mode, str(K)],
                    capture_output=True, text=True, cwd=ROOT, timeout=2400)
                ln = [x for x in r.stdout.splitlines() if x.startswith("RESULT")]
                got[mode] = ln[0] if ln else "RESULT crash rc=%d %s" % (
                    r.returncode, r.stderr.strip().splitlines()[-1][:90]
                    if r.stderr.strip() else "")
            dp, dr = pick(got["dense"], "torch_peak"), pick(got["dense"], "torch_resv")
            dn, dt_ = pick(got["dense"], "nontorch"), pick(got["dense"], "total")
            cp, cr = pick(got["cudss"], "torch_peak"), pick(got["cudss"], "torch_resv")
            cn, ct = pick(got["cudss"], "nontorch"), pick(got["cudss"], "total")
            if cp is None:
                print("   %-6s %-4d %-4d %d | cudss: %s" % (stem, Nj, B, K,
                                                            got["cudss"][:90]))
                continue

            def f(x):
                return "  OOM  " if x is None else "%7.1f" % x
            r1 = "   -  " if dp is None else "%6.2f" % (dp / cp)
            r2 = "   -  " if dt_ is None else "%6.2f" % (dt_ / ct)
            print("   %-6s %-4d %-4d %d | %s %s %s %s | %s %s %s %s | %s  %s"
                  % (stem, Nj, B, K, f(dp), f(dr), f(dn), f(dt_),
                     f(cp), f(cr), f(cn), f(ct), r1, r2))
            print("        raw dense: %s" % got["dense"])
            print("        raw cudss: %s" % got["cudss"])

    print("\n" + "=" * 78)
    print("N2  what does ONE cuDSS state cost, measured in a fresh process")
    print("    (P3 F4 claims 228 MiB for ky4 B=256 and 546 MiB for B=1024)")
    for B in (64, 256, 1024):
        r = subprocess.run(
            [sys.executable, "-X", "utf8", os.path.abspath(__file__),
             "state", "ky4.inp", str(B)],
            capture_output=True, text=True, cwd=ROOT, timeout=1200)
        for x in r.stdout.splitlines():
            if x.startswith("STATE"):
                print("   " + x)


def state_worker(fn, B):
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    from dgga.autodiff import solve_unrolled
    torch.zeros(1, device=DEV)
    torch.cuda.synchronize()
    p = os.path.join(NETD, fn)
    net = parse_inp(p)
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=p,
                  dense_tank_bound_check=False)
    g = np.random.default_rng(1)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
    D = torch.as_tensor(d0[None, :] * g.uniform(.9, 1.1, (B, d0.size)),
                        dtype=DT, device=DEV)
    R = torch.as_tensor(np.repeat(rh0[None, :], B, 0), dtype=DT, device=DEV)
    a, ar = used(), resv()
    with torch.no_grad():
        solve_unrolled(s, D, R, K=3, assemble="csr", linear_solver="cudss")
    b, br = used(), resv()
    n = s.cudss_free()
    c, cr = used(), resv()
    print("STATE ky4 B=%-5d before dev=%.1f/torch=%.1f  after dev=%.1f/torch=%.1f"
          "  -> cuDSS-only=%.1f MiB ; after free(%d) dev=%.1f (returned %.1f)"
          % (B, a, ar, b, br, (b - br) - (a - ar), n, c, b - c))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "state":
        state_worker(sys.argv[2], int(sys.argv[3]))
        sys.exit(0)
    import hashlib
    print("node:", NODE, "| torch", torch.__version__)
    print("solver.py md5:", hashlib.md5(
        open(os.path.join(ROOT, "dgga", "solver.py"), "rb").read()).hexdigest())
    try:
        main()
    except Exception:
        traceback.print_exc()
    print("\nAUD4 DONE")

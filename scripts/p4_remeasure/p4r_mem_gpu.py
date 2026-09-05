# -*- coding: utf-8 -*-
"""P4 全面重测 - 统一口径的显存全表（任务一 ③④）。

口径就是审计 R5 的那一条，一个字不改（与 data/sparse_gpu_plan.md §10.2 同源）：

    torch_peak  torch.cuda.max_memory_allocated
    torch_resv  torch.cuda.max_memory_reserved
    ctx         CUDA 初始化后、任何张量之前的设备常驻（两节点实测都是 588.19 MiB）
    nontorch    收尾时设备已用 − torch reserved − ctx   ← cuDSS 自己那块
    total       torch_resv + nontorch                   ← 决定 OOM 的量

**每个配置一个全新进程**（父进程只派发，不碰 CUDA）。量的是 fwd+bwd
（solve_unrolled(K) + backward(demand)），与时间表 §B 同一件事。
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
SEED = 2026

NETS = [("Net1", "Net1.inp"), ("Anytown", "Anytown.inp"), ("Hanoi", "Hanoi.inp"),
        ("Net2", "Net2.inp"), ("Fossolo", "Fossolo_poly1.inp"),
        ("Pescara", "Pescara.inp"), ("Net3", "Net3.inp"),
        ("Modena", "Modena.inp"), ("City_D", "City_D.inp"), ("ky4", "ky4.inp")]
BS = [int(x) for x in os.environ.get("P4R_BS", "1,8,64,256,512,1024").split(",")]


def err():
    return traceback.format_exc().strip().split("\n")[-1][:130]


def used():
    torch.cuda.synchronize()
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


def resv():
    return torch.cuda.memory_reserved() / 2 ** 20


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + .3 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - .3 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, np.nan_to_num(rh)


def worker(fn, B, mode):
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    from dgga.autodiff import solve_unrolled
    p = os.path.join(NETD, fn)
    net = parse_inp(p)
    d0, rh0 = boundary(net)
    g0 = np.random.default_rng(SEED)
    Dp = d0[None, :] * g0.uniform(.85, 1.15, (8, d0.size))
    Rp = rh0[None, :] + g0.uniform(-1., 1., (8, rh0.size))
    # K 在 **CPU** 上探（B=8 的前 8 个场景）：GPU 上探会把 cuBLAS/cuSOLVER 的
    # 230 MiB 工作区算进 cudss 那一栏的 nontorch，两栏就不可比了。
    scpu = GGASolver(net, device="cpu", dtype=DT, mode="dense", inp_path=p,
                     dense_tank_bound_check=False)
    with torch.no_grad():
        K = int(scpu.solve(torch.as_tensor(Dp, dtype=DT),
                           torch.as_tensor(Rp, dtype=DT))["iters"].max())
    del scpu
    torch.cuda.init()
    torch.zeros(1, device=DEV)
    torch.cuda.synchronize()
    ctx = used() - resv()
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=p,
                  dense_tank_bound_check=False)
    g = np.random.default_rng(SEED)
    D = torch.as_tensor(d0[None, :] * g.uniform(.85, 1.15, (B, d0.size)),
                        dtype=DT, device=DEV)
    R = torch.as_tensor(rh0[None, :] + g.uniform(-1., 1., (B, rh0.size)),
                        dtype=DT, device=DEV)
    W = torch.as_tensor(np.random.default_rng(7).normal(0, 1, (B, net.N)),
                        dtype=DT, device=DEV)
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
        print("RESULT %s %d %s ok K=%d ctx=%.2f torch_peak=%.2f torch_resv=%.2f "
              "nontorch=%.2f total=%.2f gnorm=%.8e"
              % (fn, B, mode, K, ctx, tp, tr, nt, tr + nt, float(d.grad.norm())))
    except torch.OutOfMemoryError:
        print("RESULT %s %d %s OOM K=%d ctx=%.2f" % (fn, B, mode, K, ctx))
    except Exception:                                   # noqa: BLE001
        print("RESULT %s %d %s ERR K=%d %s" % (fn, B, mode, K, err()))


if __name__ == "__main__" and len(sys.argv) > 1:
    worker(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    raise SystemExit(0)

print("=" * 100)
print("P4 全面重测 · 显存全表（R5 统一口径，MiB，每配置全新进程） | node:", NODE)
import hashlib                                          # noqa: E402
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f,
          hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print("B 列:", BS)
print()
CELL = {}
for stem, fnm in NETS:
    for B in BS:
        for mode in ("dense", "cudss"):
            cp = subprocess.run([sys.executable, "-X", "utf8", __file__, fnm,
                                 str(B), mode], capture_output=True, text=True,
                                encoding="utf-8", errors="replace",
                                cwd=ROOT, timeout=1800)
            line = [x for x in (cp.stdout or "").splitlines()
                    if x.startswith("RESULT")]
            if not line:
                tail = ((cp.stdout or "") + (cp.stderr or "")).strip()
                tail = tail.splitlines()[-1][:120] if tail else "(no output)"
                print("RESULT %s %d %s CRASH rc=%d %s" % (fnm, B, mode,
                                                          cp.returncode, tail))
                CELL[(stem, B, mode)] = None
                sys.stdout.flush()
                continue
            print(line[-1])
            sys.stdout.flush()
            tok = line[-1].split()
            status = tok[4] if len(tok) > 4 else "ERR"
            if status == "ok":
                CELL[(stem, B, mode)] = dict(
                    x.split("=", 1) for x in tok if "=" in x)
            elif status == "OOM":
                CELL[(stem, B, mode)] = "OOM"
            else:
                CELL[(stem, B, mode)] = None

print()
print("=" * 100)
print("§M1 显存全表（R5 口径；dense/cudss 各四列；total = torch_resv + nontorch）"
      "  node=%s" % NODE)
print("net      B     | dense peak     resv     非torch   total    | "
      "cudss peak    resv     非torch   total    | total 倍数")
for stem, _ in NETS:
    for B in BS:
        a, b = CELL.get((stem, B, "dense")), CELL.get((stem, B, "cudss"))

        def four(x):
            if x is None:
                return "   CRASH                                 "
            if x == "OOM":
                return "   OOM                                   "
            return "%9.1f %9.1f %8.1f %9.1f" % (
                float(x["torch_peak"]), float(x["torch_resv"]),
                float(x["nontorch"]), float(x["total"]))
        r = "   -   "
        if isinstance(a, dict) and isinstance(b, dict):
            r = "%6.2fx" % (float(a["total"]) / float(b["total"]))
        elif a == "OOM":
            r = "OOM/ok "
        print("%-8s %-5d | %s | %s | %s" % (stem, B, four(a), four(b), r))

print()
print("§M2 OOM 边界（fwd+bwd，R5 口径）  node=%s" % NODE)
print("net      | dense: 最大跑通 B / 最小 OOM B | cudss: 最大跑通 B / 最小 OOM B")
for stem, _ in NETS:
    line = []
    for mode in ("dense", "cudss"):
        ok = [B for B in BS if isinstance(CELL.get((stem, B, mode)), dict)]
        oom = [B for B in BS if CELL.get((stem, B, mode)) == "OOM"]
        line.append("%-6s / %-6s" % (max(ok) if ok else "-",
                                     min(oom) if oom else "-"))
    print("%-8s | %-30s | %s" % (stem, line[0], line[1]))
print("P4R MEM DONE")

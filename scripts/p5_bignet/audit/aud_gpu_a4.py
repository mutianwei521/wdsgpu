# -*- coding: utf-8 -*-
"""审阅项 2/3 收尾：**去掉两处自缚之后，稠密通路真正的天花板在哪。**

前一轮的主线句子是"NW_Model 上稠密最多 B=1"。aud_gpu_a2.py 已经证明：
B>=2 挂的是 torch.cholesky_solve（同一个 (Nj,B) 上 solve_triangular / linalg.solve
都好好的），而且本机 torch 2.13.0+cu132 上 cholesky_solve 根本不挂 - 所以
那是**一个 torch 版本的 API 缺陷**，不是稠密算法的墙。

本脚本再拆掉第二处：仓库的迭代精化残差写成 (A * x^T).sum(-1)（为批不变性
刻意避开 bmm），它会**额外开一整块 [B,Nj,Nj]**，直接压低稠密能跑的 B。
换成 matmul 后重扫最大 B，给出稠密真正的显存天花板与那一格的时间/倍数。

三个变体：
  ship    仓库原样
  tri     只换 cholesky_solve -> 两次 solve_triangular
  tribmm  再把精化残差换成 matmul
"""
import os
import re
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
NETDIR = os.path.join(ROOT, "audnets")
BMM = os.environ.get("AUD_BMM") or os.path.join(
    ROOT, "_bmmtree_%s" % os.environ.get("SLURM_JOB_ID", str(os.getpid())))
OLD = "                    AHj = (A * Hj.transpose(-2, -1)).sum(-1, keepdim=True)"
NEW = "                    AHj = torch.matmul(A, Hj)"


def build_bmm_tree():
    if os.path.isdir(BMM):
        shutil.rmtree(BMM)
    os.makedirs(BMM)
    shutil.copytree(os.path.join(ROOT, "dgga"), os.path.join(BMM, "dgga"))
    p = os.path.join(BMM, "dgga", "solver.py")
    src = open(p, encoding="utf-8").read()
    if OLD not in src:
        raise RuntimeError("没找到要替换的残差行 - solver.py 变了，先看清楚再跑")
    src = src.replace(OLD, NEW)
    open(p, "w", encoding="utf-8").write(src)
    return src.count(NEW)


def child(variant, stem, B, path):
    import time

    import numpy as np
    import torch
    if variant == "tribmm":
        sys.path.insert(0, BMM)
    if variant in ("tri", "tribmm"):
        def _p(b, L, upper=False, out=None):
            y = torch.linalg.solve_triangular(L, b, upper=False)
            return torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
        torch.cholesky_solve = _p
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    import dgga.solver as _sv
    net = parse_inp(path)
    s = GGASolver(net, device="cuda", dtype=torch.float64, mode="dense",
                  inp_path=path, dense_tank_bound_check=False)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    rh = np.nan_to_num(rh)
    g = np.random.default_rng(2026)
    D = torch.as_tensor(d0[None, :] * g.uniform(0.85, 1.15, (B, d0.size)),
                        dtype=torch.float64, device="cuda")
    R = torch.as_tensor(rh[None, :] + g.uniform(-1.0, 1.0, (B, rh.size)),
                        dtype=torch.float64, device="cuda")
    tag = "%s/%s/B=%d" % (variant, stem, B)
    try:
        if variant == "cudss":
            kw = dict(assemble="csr", linear_solver="cudss")
        else:
            kw = {}
        with torch.no_grad():
            o = s.solve(D, R, **kw)
        torch.cuda.synchronize()
        it = int(o["iters"].max())
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.no_grad():
                s.solve(D, R, **kw)
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) * 1e3)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        print("R OK %-22s iters=%-3d %10.2f ms/批 %9.4f ms/场景  peak=%7.0f MiB "
              "(源码 %s)" % (tag, it, best, best / B, peak,
                             os.path.dirname(os.path.abspath(_sv.__file__))))
    except torch.OutOfMemoryError as e:                  # noqa: BLE001
        print("R OOM %-22s | %s" % (tag, str(e).splitlines()[0][:100]))
    except Exception as e:                               # noqa: BLE001
        print("R FAIL %-22s %s | %s" % (tag, type(e).__name__,
                                        str(e).splitlines()[0][:100]))


if len(sys.argv) > 4 and sys.argv[1] == "child":
    child(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    raise SystemExit(0)

import hashlib                                           # noqa: E402
import torch                                             # noqa: E402
NODE = os.popen("hostname").read().strip()
print("=" * 100)
print("审阅项 2/3 收尾 · 拆掉自缚后稠密的真天花板 | node:", NODE, "| torch",
      torch.__version__, "| cuda", torch.version.cuda, "|",
      torch.cuda.get_device_name(0), "| %.0f MiB"
      % (torch.cuda.get_device_properties(0).total_memory / 2 ** 20))
for _f in ("dgga/solver.py",):
    print("md5", _f, hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
n = build_bmm_tree()
print("已生成 bmm 变体树 %s（替换 %d 处残差写法）" % (BMM, n))
print("=" * 100)


def probe(variant, stem, B, path):
    cp = subprocess.run([sys.executable, "-X", "utf8", __file__, "child",
                         variant, stem, str(B), path], capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        cwd=ROOT, timeout=7200,
                        env=dict(os.environ, AUD_BMM=BMM))
    ls = [x for x in (cp.stdout or "").splitlines() if x.startswith("R ")]
    if ls:
        return ls[-1]
    tail = ((cp.stderr or "").strip().splitlines() or [""])[-1][:120]
    return "R CRASH rc=%d %s" % (cp.returncode, tail)


BSMAP = {"NW_Model": [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32],
         "ky8": [64, 256, 384, 448, 464, 480, 512, 640, 768, 896, 1024]}
for stem in ("NW_Model", "ky8"):
    path = os.path.join(NETDIR, stem + ".inp")
    print("\n########## %s ##########" % stem)
    for variant in ("ship", "tri", "tribmm"):
        print("  --- 变体 %s ---" % variant)
        for B in BSMAP[stem]:
            r = probe(variant, stem, B, path)
            print("    %s" % r)
            sys.stdout.flush()
            if r.startswith("R OOM") or r.startswith("R CRASH"):
                break
    print("  --- 对照 cudss（同样的 B 列）---")
    for B in (1, 8, 16, 32, 64, 256, 1024):
        print("    %s" % probe("cudss", stem, B, path))
        sys.stdout.flush()
print("\nAUD A4 DONE")

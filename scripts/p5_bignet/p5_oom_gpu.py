# -*- coding: utf-8 -*-
"""P5 大网 - OOM 边界细扫（主线那句话的直接证据）。

主网格 B∈{1,8,64,256,512,1024} 在 8→64 之间跨度太大，钉不住"**稠密从哪个 B
开始 OOM**"。这里对四条通路各做一次「倍增 + 二分」，每个候选 B **开全新进程**
（父进程绝不碰 CUDA，避免上一格的碎片影响下一格的判定）：

    fwd_dense   / fwd_cudss    纯前向（no_grad）
    fb_dense    / fb_cudss     前向+反向（solve_unrolled(K) + backward）

同时打印稠密 A 的**理论**需求 B·Nj²·8 与 CSR 值的 B·nnz·8，两者对照就是
"物理上不可能 vs 几十 MiB"那句话的算术依据。

用法（父）：python3 -X utf8 p5_oom_gpu.py
用法（子）：python3 -X utf8 p5_oom_gpu.py <inp> <B> <path>
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
NETD = os.path.join(ROOT, "p5nets")
DEV, DT = "cuda", torch.float64
SEED = 2026
NODE = os.popen("hostname").read().strip()

NETS = [("NW_Model", "NW_Model.inp"), ("ky8", "ky8.inp"), ("ky4", "ky4.inp"),
        ("KL", "KL.inp")]
PATHS = ["fwd_dense", "fwd_cudss", "fb_dense", "fb_cudss"]
BMAX = int(os.environ.get("P5_BMAX", "2048"))


def err():
    return traceback.format_exc().strip().split("\n")[-1][:130]


def boundary(net):
    """与 p5_time_gpu / p5_mem_gpu 逐字同一套边界条件。"""
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + .3 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - .3 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, np.nan_to_num(rh)


def worker(fn, B, path):
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    from dgga.autodiff import solve_unrolled
    p = os.path.join(NETD, fn)
    net = parse_inp(p)
    d0, rh0 = boundary(net)
    scpu = GGASolver(net, device="cpu", dtype=DT, mode="dense", inp_path=p,
                     dense_tank_bound_check=False)
    g0 = np.random.default_rng(SEED)
    Dp = torch.as_tensor(d0[None, :] * g0.uniform(.85, 1.15, (8, d0.size)), dtype=DT)
    Rp = torch.as_tensor(rh0[None, :] + g0.uniform(-1., 1., (8, rh0.size)), dtype=DT)
    with torch.no_grad():
        K = int(scpu.solve(Dp, Rp)["iters"].max())
    Nj, nnz = scpu.Nj, int(scpu.A_csr_dense_pos.numel())
    del scpu

    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=p,
                  dense_tank_bound_check=False)
    g = np.random.default_rng(SEED)
    D = torch.as_tensor(d0[None, :] * g.uniform(.85, 1.15, (B, d0.size)),
                        dtype=DT, device=DEV)
    R = torch.as_tensor(rh0[None, :] + g.uniform(-1., 1., (B, rh0.size)),
                        dtype=DT, device=DEV)
    kw = (dict(assemble="csr", linear_solver="cudss") if path.endswith("cudss")
          else dict())
    try:
        if path.startswith("fwd"):
            with torch.no_grad():
                s.solve(D, R, **kw)
        else:
            W = torch.as_tensor(np.random.default_rng(7).normal(0, 1, (B, net.N)),
                                dtype=DT, device=DEV)
            dv = D.clone().requires_grad_(True)
            o = solve_unrolled(s, dv, R, K=K, **kw)
            (W * o["head_ft"]).sum().backward()
            assert dv.grad is not None
        torch.cuda.synchronize()
        print("PROBE %s %d %s OK Nj=%d nnz=%d K=%d resv=%.1f"
              % (fn, B, path, Nj, nnz, K,
                 torch.cuda.max_memory_reserved() / 2 ** 20))
    except torch.OutOfMemoryError:
        print("PROBE %s %d %s OOM Nj=%d nnz=%d K=%d" % (fn, B, path, Nj, nnz, K))
    except Exception:                                   # noqa: BLE001
        print("PROBE %s %d %s ERR Nj=%d nnz=%d K=%d %s"
              % (fn, B, path, Nj, nnz, K, err()))


def run(fn, B, path):
    """返回 'OK' / 'OOM' / 'ERR:...'，外加 (Nj, nnz)。"""
    cp = subprocess.run([sys.executable, "-X", "utf8", __file__, fn, str(B), path],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", cwd=ROOT, timeout=3600)
    ls = [x for x in (cp.stdout or "").splitlines() if x.startswith("PROBE")]
    if not ls:
        tail = ((cp.stdout or "") + (cp.stderr or "")).strip()
        tail = tail.splitlines()[-1][:110] if tail else "(no output)"
        # 进程被 OOM-killer / CUDA 直接打死也算 OOM 边界外，但要分开记
        return ("CRASH rc=%d %s" % (cp.returncode, tail), None, None)
    tok = ls[-1].split()
    kv = dict(x.split("=", 1) for x in tok if "=" in x)
    return (tok[4], int(kv.get("Nj", 0)), int(kv.get("nnz", 0)))


if __name__ == "__main__" and len(sys.argv) > 1:
    worker(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    raise SystemExit(0)

print("=" * 92)
print("P5 大网 · OOM 边界细扫（每个候选 B 全新进程） | node:", NODE)
import hashlib                                          # noqa: E402
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f,
          hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print()

SUMM = []
for stem, fnm in NETS:
    for path in PATHS:
        # 倍增找到第一个失败点
        lo, hi, meta = 0, None, (None, None)
        B = 1
        while B <= BMAX:
            st, Nj, nnz = run(fnm, B, path)
            if Nj:
                meta = (Nj, nnz)
            print("  probe %-9s %-9s B=%-5d -> %s" % (stem, path, B, st))
            sys.stdout.flush()
            if st == "OK":
                lo = B
                B *= 2
            else:
                hi = B
                break
        if hi is None:
            SUMM.append((stem, path, lo, None, meta))
            print("  == %-9s %-9s 最大可跑 B >= %d（未在 %d 内触到边界）"
                  % (stem, path, lo, BMAX))
            continue
        # 二分 (lo, hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            st, Nj, nnz = run(fnm, mid, path)
            print("    bisect %-9s %-9s B=%-5d -> %s" % (stem, path, mid, st))
            sys.stdout.flush()
            if st == "OK":
                lo = mid
            else:
                hi = mid
        SUMM.append((stem, path, lo, hi, meta))
        print("  == %-9s %-9s 最大可跑 B = %d，B = %d 起 OOM"
              % (stem, path, lo, hi))
        sys.stdout.flush()

print()
print("=" * 92)
print("§O1 OOM 边界汇总  node=%s" % NODE)
print("net        通路        最大可跑B   首个OOM的B   Nj     nnz    "
      "稠密A@首OOM(GiB)  CSR值@首OOM(MiB)")
for stem, path, lo, hi, (Nj, nnz) in SUMM:
    if Nj:
        bb = hi if hi else lo
        dg = bb * Nj * Nj * 8 / 2 ** 30
        cm = bb * nnz * 8 / 2 ** 20
        print("%-10s %-10s %8d %11s %7d %7d %14.2f %16.2f"
              % (stem, path, lo, hi if hi else ">BMAX", Nj, nnz, dg, cm))
    else:
        print("%-10s %-10s %8d %11s   (元数据缺失)" % (stem, path, lo, hi))

print()
print("§O2 稠密 A 的理论需求 B·Nj²·8 vs CSR 值 B·nnz·8（GiB / MiB）")
seen = {}
for stem, path, lo, hi, (Nj, nnz) in SUMM:
    if Nj and stem not in seen:
        seen[stem] = (Nj, nnz)
for stem, (Nj, nnz) in seen.items():
    print("  %-10s Nj=%-6d nnz=%-7d Nj²/nnz=%.0f" % (stem, Nj, nnz, Nj * Nj / nnz))
    for B in (1, 8, 64, 256, 512, 1024):
        print("      B=%-5d 稠密A %10.2f GiB   CSR值 %9.2f MiB"
              % (B, B * Nj * Nj * 8 / 2 ** 30, B * nnz * 8 / 2 ** 20))
print("P5 OOM SCAN END")

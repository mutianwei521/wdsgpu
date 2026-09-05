# -*- coding: utf-8 -*-
"""审阅项 2（集群侧）：那堵"墙"到底是什么。

前一轮结论：NW_Model 稠密 B>=2 报 CUDA invalid argument，二分出
Nj<=6016 可用 / >=6017 失败，与批量无关，5090/4090 双卡复现。

本脚本要问三件前一轮没问的事：
  Q1 门槛是**单调**的吗？（前一轮用二分，二分预设单调） - 逐点扫，不二分。
  Q2 只有 torch.cholesky_solve 挂，还是三角回代本身就挂？
     测 cholesky_solve / solve_triangular×2 / linalg.solve 三条，同一个 (Nj,B)。
  Q3 如果只有 cholesky_solve 挂 - 那这堵墙是**API 选择**，不是稠密算法的墙。
     把 torch.cholesky_solve 换成两次 solve_triangular，重测 dgga 稠密通路
     在 NW_Model 上**真正的**最大 B（那才是显存墙），并报时间。

另：本机（RTX 5060 Laptop, torch 2.13.0+cu132）上 Nj=8566 的 cholesky_solve
B=1/2/4 **全部正常**，所以这堵墙高度怀疑是 torch 2.11.0+cu128 这一版的缺陷。
本脚本打印 torch/cuda 版本，好把结论钉在版本上。

用法：python3 -X utf8 aud_gpu_a2.py            （父：全套）
      python3 -X utf8 aud_gpu_a2.py raw Nj B op
      python3 -X utf8 aud_gpu_a2.py dgga NET B patch
"""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
NETDIR = os.path.join(ROOT, "audnets")


def _mem():
    import torch
    free, tot = torch.cuda.mem_get_info()
    return free / 2 ** 20, tot / 2 ** 20


def _hostpeak():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmHWM"):
                return ln.split()[1] + " kB"
    except Exception:                                   # noqa: BLE001
        pass
    return "n/a"


# --------------------------------------------------------------- 裸 torch 探针
def raw(Nj, B, op):
    import torch
    dev, dt = "cuda", torch.float64
    free0, tot = _mem()
    need = B * Nj * Nj * 8 / 2 ** 20
    try:
        if op.endswith("_diag"):        # 前一轮 p5_thresh.py 的构造（纯对角）
            A = torch.zeros(B, Nj, Nj, dtype=dt, device=dev)
            A.diagonal(dim1=-2, dim2=-1).fill_(4.0)
        else:                           # 三对角 SPD
            A = torch.zeros(B, Nj, Nj, dtype=dt, device=dev)
            A.diagonal(dim1=-2, dim2=-1).fill_(4.0)
            i = torch.arange(Nj - 1, device=dev)
            A[:, i, i + 1] = -1.0
            A[:, i + 1, i] = -1.0
        L = torch.linalg.cholesky(A)
        torch.cuda.synchronize()
        base = op.replace("_diag", "")
        if base == "chol":
            print("R OK chol %.6f free0=%.0f need=%.0f hostpeak=%s"
                  % (float(L[0, 0, 0]), free0, need, _hostpeak()))
            return
        F = torch.ones(B, Nj, 1, dtype=dt, device=dev)
        if base == "cholsolve":
            x = torch.cholesky_solve(F, L)
        elif base == "trisolve":
            y = torch.linalg.solve_triangular(L, F, upper=False)
            x = torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
        elif base == "lusolve":
            x = torch.linalg.solve(A, F)
        else:
            raise ValueError(op)
        torch.cuda.synchronize()
        print("R OK %s %.6e free0=%.0f need=%.0f hostpeak=%s"
              % (base, float(x[0, 0, 0]), free0, need, _hostpeak()))
    except torch.OutOfMemoryError as e:                 # noqa: BLE001
        print("R OOM %s free0=%.0f need=%.0f | %s"
              % (op, free0, need, str(e).splitlines()[0][:110]))
    except Exception as e:                              # noqa: BLE001
        print("R FAIL %s %s free0=%.0f need=%.0f | %s"
              % (op, type(e).__name__, free0, need, str(e).splitlines()[0][:110]))


# ------------------------------------------------------- dgga 稠密通路（可打补丁）
def dgga_run(stem, B, patch):
    import time

    import numpy as np
    import torch
    if patch == "tri":
        _o = torch.cholesky_solve

        def _p(b, L, upper=False, out=None):
            y = torch.linalg.solve_triangular(L, b, upper=False)
            return torch.linalg.solve_triangular(L.transpose(-2, -1), y,
                                                 upper=True)
        torch.cholesky_solve = _p
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    f = os.path.join(NETDIR, stem + ".inp")
    net = parse_inp(f)
    s = GGASolver(net, device="cuda", dtype=torch.float64, mode="dense",
                  inp_path=f, dense_tank_bound_check=False)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh[tn] = np.asarray(net.tank_h0, dtype=np.float64)
    rh = np.nan_to_num(rh)
    g = np.random.default_rng(2026)
    D = torch.as_tensor(d0[None, :] * g.uniform(0.85, 1.15, (B, d0.size)),
                        dtype=torch.float64, device="cuda")
    R = torch.as_tensor(rh[None, :] + g.uniform(-1.0, 1.0, (B, rh.size)),
                        dtype=torch.float64, device="cuda")
    free0, tot = _mem()
    try:
        with torch.no_grad():
            o = s.solve(D, R)                       # 预热 + 正确性
        torch.cuda.synchronize()
        it = int(o["iters"].max())
        t0 = time.perf_counter()
        with torch.no_grad():
            s.solve(D, R)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        free1, _ = _mem()
        print("R OK dgga %s B=%d patch=%s iters=%d  %.3f ms/批 = %.5f ms/场景"
              "  peak_alloc=%.0f MiB  free %0.f->%0.f  hostpeak=%s"
              % (stem, B, patch, it, ms, ms / B, peak, free0, free1, _hostpeak()))
    except torch.OutOfMemoryError as e:                 # noqa: BLE001
        print("R OOM dgga %s B=%d patch=%s free0=%.0f | %s"
              % (stem, B, patch, free0, str(e).splitlines()[0][:110]))
    except Exception as e:                              # noqa: BLE001
        print("R FAIL dgga %s B=%d patch=%s %s free0=%.0f | %s"
              % (stem, B, patch, type(e).__name__, free0,
                 str(e).splitlines()[0][:110]))


if len(sys.argv) > 2 and sys.argv[1] == "raw":
    raw(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    raise SystemExit(0)
if len(sys.argv) > 2 and sys.argv[1] == "dgga":
    dgga_run(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    raise SystemExit(0)

# ---------------------------------------------------------------------- 父进程
import hashlib                                          # noqa: E402
import torch                                            # noqa: E402
NODE = os.popen("hostname").read().strip()
print("=" * 100)
print("审阅项 2 · 那堵墙是什么 | node:", NODE, "| torch", torch.__version__,
      "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      "| %.0f MiB" % (torch.cuda.get_device_properties(0).total_memory / 2 ** 20))
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f, hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print("=" * 100)
env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")


def probe(*a):
    cp = subprocess.run([sys.executable, "-X", "utf8", __file__] + [str(x) for x in a],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", cwd=ROOT, env=env, timeout=7200)
    ls = [x for x in (cp.stdout or "").splitlines() if x.startswith("R ")]
    if ls:
        return ls[-1]
    tail = ((cp.stderr or "").strip().splitlines() or [""])[-1][:120]
    sig = " (被信号杀 %d)" % (-cp.returncode) if cp.returncode < 0 else ""
    return "R CRASH rc=%d%s %s" % (cp.returncode, sig, tail)


print("\n########## Q1 逐点扫（不二分）：cholesky_solve，三对角 SPD ##########")
NJS = [4096, 5000, 6000, 6015, 6016, 6017, 6018, 6100, 6500, 7000, 8192, 8566]
for B in (1, 2, 3, 4, 8):
    print("  --- B=%d ---" % B)
    for Nj in NJS:
        print("    Nj=%-6d %s" % (Nj, probe("raw", Nj, B, "cholsolve")))
        sys.stdout.flush()

print("\n########## Q1b 纯对角构造（前一轮 p5_thresh.py 的原样） ##########")
for B in (2, 4):
    for Nj in (6016, 6017, 8566):
        print("  B=%d Nj=%-6d %s" % (B, Nj, probe("raw", Nj, B, "cholsolve_diag")))
        sys.stdout.flush()

print("\n########## Q2 换算子：同一个 (Nj,B) 上三条通路 ##########")
for Nj in (6016, 6017, 8192, 8566):
    for B in (1, 2, 4, 8):
        for op in ("chol", "cholsolve", "trisolve", "lusolve"):
            print("  Nj=%-6d B=%-2d %-10s %s" % (Nj, B, op, probe("raw", Nj, B, op)))
            sys.stdout.flush()

print("\n########## Q3 dgga 稠密通路：原样 vs 把 cholesky_solve 换成两次三角回代 ##########")
for patch in ("none", "tri"):
    for B in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        r = probe("dgga", "NW_Model", B, patch)
        print("  patch=%-4s B=%-3d %s" % (patch, B, r))
        sys.stdout.flush()
        if r.startswith("R OOM") or r.startswith("R CRASH"):
            print("    (该 patch 到此为止：已撞墙)")
            break
print("\nAUD A2 DONE")

# -*- coding: utf-8 -*-
"""审阅项 2：独立复现 torch.cholesky_solve 的批量门槛，并**证伪单调性假设**。

前一轮用二分（p5_thresh.py）得出 "Nj<=6016 可用 / >=6017 失败，与 B 无关"。
二分**预设了单调性**；本脚本改为**逐点扫描**（不二分），并且：
  · 分开 cholesky / cholesky_solve / triangular_solve / linalg.solve，指认真凶
  · 区分 torch.OutOfMemoryError（显存）与 AcceleratorError/RuntimeError（非显存）
  · 每个候选**全新子进程**（CUDA 错误粘性），并打印当时空闲显存与本次张量需求
  · B=1,2,3,4,8 都扫，看门槛到底与 B 有没有关系
用法：python -X utf8 a2_chol_thresh.py [scan|point Nj B op]
"""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))


def child(Nj, B, op):
    import torch
    dev, dt = "cuda", torch.float64
    free0, tot = torch.cuda.mem_get_info()
    need = B * Nj * Nj * 8 / 2 ** 20
    try:
        A = torch.zeros(B, Nj, Nj, dtype=dt, device=dev)
        A.diagonal(dim1=-2, dim2=-1).fill_(4.0)
        i = torch.arange(Nj - 1, device=dev)
        A[:, i, i + 1] = -1.0
        A[:, i + 1, i] = -1.0
        L = torch.linalg.cholesky(A)
        torch.cuda.synchronize()
        if op == "chol":
            print("R OK chol L=%.6f free=%.0fMiB need=%.0fMiB"
                  % (float(L[0, 0, 0]), free0 / 2 ** 20, need))
            return
        F = torch.ones(B, Nj, 1, dtype=dt, device=dev)
        if op == "cholsolve":
            x = torch.cholesky_solve(F, L)
        elif op == "trisolve":
            y = torch.linalg.solve_triangular(L, F, upper=False)
            x = torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
        elif op == "lusolve":
            x = torch.linalg.solve(A, F)
        else:
            raise ValueError(op)
        torch.cuda.synchronize()
        print("R OK %s x=%.6e free=%.0fMiB need=%.0fMiB"
              % (op, float(x[0, 0, 0]), free0 / 2 ** 20, need))
    except torch.OutOfMemoryError as e:                    # noqa: BLE001
        print("R OOM %s | free=%.0fMiB need=%.0fMiB | %s"
              % (op, free0 / 2 ** 20, need, str(e).splitlines()[0][:110]))
    except Exception as e:                                 # noqa: BLE001
        print("R FAIL %s %s | free=%.0fMiB need=%.0fMiB | %s"
              % (op, type(e).__name__, free0 / 2 ** 20, need,
                 str(e).splitlines()[0][:110]))


if len(sys.argv) > 2 and sys.argv[1] == "point":
    child(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    raise SystemExit(0)

import torch                                               # noqa: E402
print("=" * 96)
print("审阅项 2 · cholesky_solve 门槛逐点扫描（**不二分**）")
print("node:", os.popen("hostname").read().strip(), "| torch", torch.__version__,
      "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      "| %.0f MiB" % (torch.cuda.get_device_properties(0).total_memory / 2 ** 20))
print("=" * 96)
env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")


def probe(Nj, B, op):
    cp = subprocess.run([sys.executable, "-X", "utf8", __file__, "point",
                         str(Nj), str(B), op], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", cwd=HERE, env=env,
                        timeout=3600)
    ls = [x for x in (cp.stdout or "").splitlines() if x.startswith("R ")]
    if ls:
        return ls[-1]
    tail = ((cp.stderr or "").strip().splitlines() or [""])[-1][:110]
    return "R CRASH rc=%d %s" % (cp.returncode, tail)


NJS = [int(x) for x in os.environ.get(
    "AUD_NJS", "4096,5000,6000,6010,6015,6016,6017,6018,6020,6100,6500,"
               "7000,8192,8566").split(",")]
BSS = [int(x) for x in os.environ.get("AUD_BS", "1,2,3,4,8").split(",")]
OPS = os.environ.get("AUD_OPS", "cholsolve").split(",")

for op in OPS:
    for B in BSS:
        print("\n#### op=%s  B=%d" % (op, B))
        for Nj in NJS:
            r = probe(Nj, B, op)
            print("   Nj=%-6d %s" % (Nj, r))
            sys.stdout.flush()
print("\nA2 SCAN DONE")

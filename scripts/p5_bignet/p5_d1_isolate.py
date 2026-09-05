# -*- coding: utf-8 -*-
"""P5·D1 定位：NW_Model 上稠密通路 B>=2 报 `CUDA error: invalid argument`
（cudaErrorInvalidValue），**不是显存不足**（31.5 GiB 空闲，只要 1.1 GiB）。

把嫌疑逐个隔开，全部**不经 dgga**，只用裸 torch：
  T1  分配 [B, Nj*Nj] f64 - 装配用的展平缓冲
  T2  scatter_add_ 到 [B, Nj*Nj] - dgga 的稠密散射
  T3  reshape 成 [B, Nj, Nj] 后 torch.linalg.cholesky - 批量 Cholesky
  T4  torch.cholesky_solve
每项对 B=1,2 各跑一次，CUDA_LAUNCH_BLOCKING=1（子进程里设），
这样报错点就是真实的失败算子，不是异步错报。

用法（父）：python3 -X utf8 p5_d1_isolate.py
用法（子）：python3 -X utf8 p5_d1_isolate.py <Nj> <B> <test>
"""
import os
import subprocess
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
NODE = os.popen("hostname").read().strip()
DT = "float64"


def child(Nj, B, test):
    import torch
    dev = "cuda"
    dt = torch.float64
    free, tot = torch.cuda.mem_get_info()
    print("  [%s] Nj=%d B=%d  空闲 %.0f MiB" % (test, Nj, B, free / 2 ** 20))
    sys.stdout.flush()
    try:
        if test == "T1":
            x = torch.zeros(B, Nj * Nj, dtype=dt, device=dev)
            torch.cuda.synchronize()
            print("  OK  numel=%d" % x.numel())
        elif test == "T2":
            x = torch.zeros(B, Nj * Nj, dtype=dt, device=dev)
            idx = torch.arange(0, Nj * Nj, 7, device=dev).unsqueeze(0).expand(B, -1)
            src = torch.ones(B, idx.shape[1], dtype=dt, device=dev)
            x.scatter_add_(1, idx, src)
            torch.cuda.synchronize()
            print("  OK  scatter_add_ 到 numel=%d, 索引 %d" % (x.numel(), idx.shape[1]))
        elif test in ("T3", "T4"):
            # 构造一个稳妥的 SPD 批：对角占优
            A = torch.zeros(B, Nj, Nj, dtype=dt, device=dev)
            A.diagonal(dim1=-2, dim2=-1).fill_(4.0)
            i = torch.arange(Nj - 1, device=dev)
            A[:, i, i + 1] = -1.0
            A[:, i + 1, i] = -1.0
            torch.cuda.synchronize()
            L = torch.linalg.cholesky(A)
            torch.cuda.synchronize()
            if test == "T3":
                print("  OK  cholesky  L[0,0,0]=%.6f" % float(L[0, 0, 0]))
            else:
                F = torch.ones(B, Nj, 1, dtype=dt, device=dev)
                x = torch.cholesky_solve(F, L)
                torch.cuda.synchronize()
                print("  OK  cholesky_solve  x[0,0,0]=%.6e" % float(x[0, 0, 0]))
        else:
            print("  ?? unknown test")
    except Exception as e:                              # noqa: BLE001
        print("  FAIL %s" % type(e).__name__)
        print("  " + str(e)[:600].replace("\n", "\n  "))
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__" and len(sys.argv) > 3:
    child(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
    raise SystemExit(0)

import torch                                            # noqa: E402
print("=" * 88)
print("P5·D1 定位 | node:", NODE, "| torch", torch.__version__,
      "|", torch.cuda.get_device_name(0))
print("=" * 88)
env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
for Nj in (8566, 4096, 6000, 8192):
    print("\n########## Nj=%d ##########" % Nj)
    for test in ("T1", "T2", "T3", "T4"):
        for B in (1, 2):
            cp = subprocess.run(
                [sys.executable, "-X", "utf8", __file__, str(Nj), str(B), test],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", cwd=ROOT, env=env, timeout=1800)
            out = (cp.stdout or "").strip()
            print(out if out else "  [%s] B=%d 无输出 rc=%d" % (test, B, cp.returncode))
            if cp.returncode != 0 and (cp.stderr or "").strip():
                print("   stderr尾:", (cp.stderr or "").strip()[-300:])
            sys.stdout.flush()
print("\nP5 D1 ISOLATE END")

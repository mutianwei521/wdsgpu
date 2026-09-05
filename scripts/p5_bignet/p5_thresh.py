# -*- coding: utf-8 -*-
"""P5·D1 续：把 torch.cholesky_solve 批量>=2 的失效门槛二分到个位。
纯裸 torch，不经 dgga。每个候选 Nj 全新进程（CUDA 错误是粘性的）。"""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))


def child(Nj, B):
    import torch
    dev, dt = "cuda", torch.float64
    A = torch.zeros(B, Nj, Nj, dtype=dt, device=dev)
    A.diagonal(dim1=-2, dim2=-1).fill_(4.0)
    try:
        L = torch.linalg.cholesky(A)
        x = torch.cholesky_solve(torch.ones(B, Nj, 1, dtype=dt, device=dev), L)
        torch.cuda.synchronize()
        print("R OK %.6f" % float(x[0, 0, 0]))
    except Exception as e:                              # noqa: BLE001
        print("R FAIL %s %s" % (type(e).__name__, str(e).splitlines()[0][:90]))


if __name__ == "__main__" and len(sys.argv) > 2:
    child(int(sys.argv[1]), int(sys.argv[2]))
    raise SystemExit(0)

import torch                                            # noqa: E402
print("P5 门槛二分 | node:", os.popen("hostname").read().strip(),
      "| torch", torch.__version__, "|", torch.cuda.get_device_name(0))


def probe(Nj, B):
    cp = subprocess.run([sys.executable, "-X", "utf8", __file__, str(Nj), str(B)],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", cwd=ROOT, timeout=1800)
    ls = [x for x in (cp.stdout or "").splitlines() if x.startswith("R ")]
    return ls[-1] if ls else ("R CRASH rc=%d" % cp.returncode)


for B in (2, 4):
    lo, hi = 4096, 8192
    print("\n#### batch=%d  先确认两端" % B)
    print("  Nj=%-6d %s" % (lo, probe(lo, B)))
    print("  Nj=%-6d %s" % (hi, probe(hi, B)))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = probe(mid, B)
        print("  bisect Nj=%-6d %s" % (mid, r))
        sys.stdout.flush()
        if r.startswith("R OK"):
            lo = mid
        else:
            hi = mid
    print("  ==> batch=%d: cholesky_solve 最大可用 Nj = %d，Nj = %d 起失败"
          % (B, lo, hi))
    print("      (lo²=%d  hi²=%d  B*hi²=%d  B*hi²*8B=%.2f GiB)"
          % (lo * lo, hi * hi, B * hi * hi, B * hi * hi * 8 / 2 ** 30))
print("P5 THRESH END")

# -*- coding: utf-8 -*-
"""P5 诊断：两件在主表里冒出来、**必须先解释清楚**的事。

D1  NW_Model 上稠密通路在 B=2 就挂了 - 但 B=2 的稠密 A 只要 1.17 GiB，
    32 GiB 的卡上**不可能是显存不够**。主表把它记成 OOM 是不诚实的，
    这里把**完整异常**抓出来看是什么。

D2  显存表里 NW_Model B=1 的 gnorm，dense=8.85e3 而 cudss=2.98e4（差 3.4 倍），
    可同一次运行里"一轮线性代数"的梯度一致性却是 1.56e-08。
    怀疑是 **solve_unrolled 截断 K 步的梯度本身不可用**（autodiff.
    unrolled_grad_health 的"不建议"档，ky4 实测 rel_K=1.08e-03），
    而不是 cudss 反向错。这里跑现成的自检接口，并把 dense/cudss 的
    展开梯度逐坐标比一遍、再随 K 加长看两者是否一起收敛。

用法：python3 -X utf8 p5_diag_gpu.py
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


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + .3 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - .3 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, np.nan_to_num(rh)


def load(fn, B):
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    p = os.path.join(NETD, fn)
    net = parse_inp(p)
    d0, rh0 = boundary(net)
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=p,
                  dense_tank_bound_check=False)
    g = np.random.default_rng(SEED)
    D = torch.as_tensor(d0[None, :] * g.uniform(.85, 1.15, (B, d0.size)),
                        dtype=DT, device=DEV)
    R = torch.as_tensor(rh0[None, :] + g.uniform(-1., 1., (B, rh0.size)),
                        dtype=DT, device=DEV)
    return net, s, D, R


# ---------------------------------------------------------------- D1 子进程
def d1_worker(fn, B):
    net, s, D, R = load(fn, B)
    free, tot = torch.cuda.mem_get_info()
    print("D1 %s B=%d Nj=%d  卡上空闲 %.0f MiB / 共 %.0f MiB；"
          "稠密A理论 %.0f MiB" % (fn, B, s.Nj, free / 2 ** 20, tot / 2 ** 20,
                                 B * s.Nj * s.Nj * 8 / 2 ** 20))
    sys.stdout.flush()
    try:
        with torch.no_grad():
            s.solve(D, R)
        print("D1 RESULT OK")
    except torch.OutOfMemoryError as e:
        print("D1 RESULT torch.OutOfMemoryError（真·分配器 OOM）")
        print("   ", str(e)[:400].replace("\n", " "))
    except Exception as e:                              # noqa: BLE001
        print("D1 RESULT %s（**不是** torch 分配器 OOM）" % type(e).__name__)
        print("---- 完整异常 ----")
        traceback.print_exc()
        print("---- 异常文本 ----")
        print(str(e)[:1500])


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "d1":
    d1_worker(sys.argv[2], int(sys.argv[3]))
    raise SystemExit(0)


print("=" * 92)
print("P5 诊断 | node:", NODE, "| torch", torch.__version__)
import hashlib                                          # noqa: E402
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f, hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())

print()
print("#" * 92)
print("D1  稠密通路在 NW_Model 上到底是怎么挂的（每个 B 全新进程）")
print("#" * 92)
for B in (1, 2, 3, 4, 8):
    cp = subprocess.run([sys.executable, "-X", "utf8", __file__, "d1",
                         "NW_Model.inp", str(B)], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", cwd=ROOT, timeout=3600)
    print("\n===== B=%d  (rc=%d) =====" % (B, cp.returncode))
    print((cp.stdout or "").strip()[:4000])
    if cp.returncode != 0:
        print("--- stderr 尾 ---")
        print((cp.stderr or "").strip()[-1500:])
    sys.stdout.flush()

print()
print("#" * 92)
print("D2  展开梯度：dense vs cudss，以及展开梯度本身可不可用")
print("#" * 92)
from dgga.autodiff import solve_unrolled, unrolled_grad_health   # noqa: E402

for stem, fn in (("NW_Model", "NW_Model.inp"), ("ky4", "ky4.inp"),
                 ("KL", "KL.inp")):
    try:
        net, s, D, R = load(fn, 1)
        with torch.no_grad():
            K = int(s.solve(D, R)["iters"].max())
        W = torch.as_tensor(np.random.default_rng(7).normal(0, 1, (1, net.N)),
                            dtype=DT, device=DEV)

        def grad(kk, kw):
            dv = D.clone().requires_grad_(True)
            o = solve_unrolled(s, dv, R, K=kk, **kw)
            (W * o["head_ft"]).sum().backward()
            return dv.grad.detach().clone()

        print("\n### %-9s Nj=%-5d K(实测收敛迭代数)=%d" % (stem, s.Nj, K))
        for kk in (K, K + 5, K + 10, K + 20):
            gd = grad(kk, dict())
            gc = grad(kk, dict(assemble="csr", linear_solver="cudss"))
            den = float(torch.max(gd.abs().max(), gc.abs().max()).clamp_min(1e-300))
            rel = float((gd - gc).abs().max()) / den
            print("   K=%-4d |g|dense=%.6e  |g|cudss=%.6e   逐坐标 rel(dense~cudss)=%.3e"
                  % (kk, float(gd.norm()), float(gc.norm()), rel))
            sys.stdout.flush()
        h = unrolled_grad_health(s, D, R, K=K, dK=5, param="demand")
        print("   unrolled_grad_health(param=demand): verdict=%s  rel_K=%.3e  gmax=%.3e"
              % (h["verdict"], h["rel_K"], h["gmax"]))
        del s, D, R
        torch.cuda.empty_cache()
    except Exception:                                   # noqa: BLE001
        print("### %s 诊断失败" % stem)
        traceback.print_exc()
    sys.stdout.flush()

print()
print("P5 DIAG END")

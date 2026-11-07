# -*- coding: utf-8 -*-
"""gpu_smoke_hanoi.py - 5090 服务器 GPU 环境冒烟测试（2026-10-23）。

Hanoi，B 个随机需水工况（±30%），标量损失 L = Σ w·H（junction）。
对比 CPU ImplicitGGASolve 与 GPU ImplicitGGASolveGPU（dense 分解 / csr+cuDSS）：
前向水头 max|ΔH|、∂L/∂d 与 ∂L/∂r 的相对 L2 差，以及前向+反向墙钟（中位，首轮预热不计）。
只用于确认环境可用；不作论文数字（论文 GPU 口径另按 R2-M6 拆分报告）。
运行：python3 scripts/gpu_smoke_hanoi.py [B]
"""
import os
import sys
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dgga.parse import Net                                         # noqa: E402
from dgga.solver import GGASolver                                  # noqa: E402
from dgga.autodiff import ImplicitGGASolve, ImplicitGGASolveGPU    # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 256
INP = os.path.join(ROOT, "networks", "public", "Hanoi.inp")
REF = os.path.join(ROOT, "data", "reference")


def build(dev):
    net = Net.load(REF, "pub_hanoi")          # 每个 solver 各用一份 Net（inp_path 会就地修正 net）
    return net, GGASolver(net, device=dev, mode="dense", inp_path=INP)


net, s_cpu = build("cpu")
_, s_gpu = build("cuda")
N = net.N
rng = np.random.default_rng(2026)
D = net.demand_cfs_at(0)[None, :] * (1.0 + rng.uniform(-0.3, 0.3, (B, N)))
RH = np.tile(np.nan_to_num(net.reservoir_head_ft_at(0)), (B, 1))
W = rng.standard_normal((B, N)) * (np.asarray(net.node_type) == 0)[None, :]
R0 = s_cpu.r_hw.detach().cpu().numpy()


def run(fn, dev, reps=3):
    ts, out = [], None
    for k in range(reps + 1):
        d = torch.tensor(D, device=dev, requires_grad=True)
        r = torch.tensor(R0, device=dev, requires_grad=True)
        rh = torch.tensor(RH, device=dev)
        ke = torch.zeros(B, N, device=dev)
        w = torch.tensor(W, device=dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        H = fn(d, rh, ke, r)
        (H * w).sum().backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        if k > 0:
            ts.append(time.perf_counter() - t)
        out = (H.detach().cpu().numpy(), d.grad.cpu().numpy(), r.grad.cpu().numpy())
    return out, float(np.median(ts))


print(f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda} B={B} N={N} L={s_cpu.L}")
ref, t_cpu = run(lambda d, rh, ke, r: ImplicitGGASolve.apply(d, rh, ke, r, s_cpu, 1e-12, 60, 3)[0], "cpu")
print(f"[cpu ImplicitGGASolve]  fwd+bwd {t_cpu*1e3:9.1f} ms")
junc = np.asarray(net.node_type) == 0
for tag, asm, lin in (("gpu dense", "dense", "dense"), ("gpu csr+cudss", "csr", "cudss")):
    try:
        (H, gd, gr), t = run(lambda d, rh, ke, r: ImplicitGGASolveGPU.apply(
            d, rh, ke, r, s_gpu, 1e-12, 60, None, asm, lin, 3)[0], "cuda")
        dH = np.abs(H[:, junc] - ref[0][:, junc]).max()
        e_d = np.linalg.norm(gd - ref[1]) / np.linalg.norm(ref[1])
        e_r = np.linalg.norm(gr - ref[2]) / np.linalg.norm(ref[2])
        print(f"[{tag:<14}] fwd+bwd {t*1e3:9.1f} ms | max|dH| vs cpu {dH:.2e} ft | "
              f"rel err dL/dd {e_d:.2e}, dL/dr {e_r:.2e}")
    except Exception as e:                                     # 如实报告失败原因
        print(f"[{tag:<14}] FAILED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
print("SMOKE_DONE")

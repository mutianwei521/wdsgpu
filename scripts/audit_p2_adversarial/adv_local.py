# -*- coding: utf-8 -*-
"""本机（有 CUDA、无 nvmath）对抗性检查：回退路径 / 守卫 / 状态污染 / 签名兼容。
判据：请求 cudss 时**必须**明确抛错，绝不允许静默降级算出一个"看起来对"的结果。
"""
import importlib.util
import inspect
import sys
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
import os as _os  # noqa: E402
R = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, R)
from dgga.parse import parse_inp                        # noqa: E402
from dgga.solver import GGASolver                       # noqa: E402
from dgga.autodiff import solve_unrolled, solve_polished, implicit_solve  # noqa: E402

print("torch", torch.__version__, "| cuda_avail", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
      "| nvmath spec", importlib.util.find_spec("nvmath"))
DT = torch.float64
F = R + "/networks/public/Net3.inp"
net = parse_inp(F)
d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
tn = np.asarray(net.tank_node, dtype=np.int64)
if tn.size:
    lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
    hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
    rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
g = np.random.default_rng(5150)
D = d[None, :] * g.uniform(0.7, 1.3, (4, d.size))
Rh = rh[None, :] + g.uniform(-1.0, 1.0, (4, rh.size))
Rh = np.where(np.isnan(rh)[None, :], np.nan, Rh)

s_gpu = GGASolver(net, mode="dense", inp_path=F, dtype=DT, device="cuda",
                  dense_tank_bound_check=False)
s_cpu = GGASolver(net, mode="dense", inp_path=F, dtype=DT,
                  dense_tank_bound_check=False)
s_ep = GGASolver(net, mode="epanet", inp_path=F, dtype=DT)
s_f32 = GGASolver(net, mode="dense", inp_path=F, dtype=torch.float32,
                  device="cuda", dense_tank_bound_check=False)
Dg = torch.as_tensor(D, dtype=DT, device="cuda")
Rg = torch.as_tensor(Rh, dtype=DT, device="cuda")

print("\n[G] 守卫逐条（要求：抛错，且错误类型/文本可辨认）")
CASES = [
    ("G1 linear_solver 拼错", lambda: s_gpu.solve(Dg, Rg, linear_solver="cuDSS")),
    ("G2 assemble 拼错", lambda: s_gpu.solve(Dg, Rg, assemble="CSR")),
    ("G3 cudss + assemble=dense", lambda: s_gpu.solve(Dg, Rg, linear_solver="cudss")),
    ("G4 cudss + mode=epanet",
     lambda: s_ep.solve(D, Rh, assemble="csr", linear_solver="cudss")),
    ("G5 csr + mode=epanet", lambda: s_ep.solve(D, Rh, assemble="csr")),
    ("G6 cudss + device=cpu",
     lambda: s_cpu.solve(D, Rh, assemble="csr", linear_solver="cudss")),
    ("G7 cudss + CUDA + 无 nvmath",
     lambda: s_gpu.solve(Dg, Rg, assemble="csr", linear_solver="cudss")),
    ("G8 cudss + float32",
     lambda: s_f32.solve(Dg.float(), Rg.float(), assemble="csr",
                         linear_solver="cudss")),
    ("G9 cudss + requires_grad",
     lambda: s_gpu.solve(Dg.clone().requires_grad_(True), Rg, assemble="csr",
                         linear_solver="cudss")),
    ("G10 unrolled cudss + assemble=dense",
     lambda: solve_unrolled(s_gpu, Dg, Rg, K=5, linear_solver="cudss")),
    ("G11 unrolled cudss（无 nvmath）",
     lambda: solve_unrolled(s_gpu, Dg, Rg, K=5, assemble="csr",
                            linear_solver="cudss")),
    ("G12 polished cudss（无 nvmath）",
     lambda: solve_polished(s_gpu, D, Rh, assemble="csr", linear_solver="cudss")),
    ("G13 cudss_plan 直接调（无 nvmath）", lambda: s_gpu.cudss_plan(4)),
    ("G14 cudss_plan 在 CPU solver 上", lambda: s_cpu.cudss_plan(4)),
]
for name, fn in CASES:
    try:
        out = fn()
        print("  [FAIL] %-34s 没抛错 -> 返回 %s" % (name, type(out)))
    except Exception as e:                               # noqa: BLE001
        txt = str(e).replace("\n", " ")
        print("  [OK]   %-34s %s: %s" % (name, type(e).__name__, txt[:95]))

print("\n[S] 状态污染：撞完守卫后缺省通路必须逐位不变")
base = s_gpu.solve(Dg, Rg)["head_ft"].cpu().numpy().copy()
for _ in range(3):
    for _n, fn in CASES:
        try:
            fn()
        except Exception:                                # noqa: BLE001
            pass
after = s_gpu.solve(Dg, Rg)["head_ft"].cpu().numpy()
print("  撞 %d 次守卫后 max|ΔH| = %.3e（要求 0.000e+00），逐位相同=%s"
      % (3 * len(CASES), np.abs(after - base).max(),
         after.tobytes() == base.tobytes()))
print("  _cudss_cache 大小 =", len(s_gpu._cudss_cache), "（要求 0：失败不应留脏 state）")

print("\n[C] assemble='csr' + linear_solver='dense' 逐位等价（CPU/GPU，多网多批量）")
for stem in ("Net1", "Net2", "Net3", "Hanoi", "Anytown", "Modena"):
    try:
        f = R + "/networks/public/%s.inp" % stem
        n2 = parse_inp(f)
        d2 = np.asarray(n2.demand_cfs_at(0), dtype=np.float64)
        r2 = np.array(n2.reservoir_head_ft_at(0), dtype=np.float64)
        t2 = np.asarray(n2.tank_node, dtype=np.int64)
        if t2.size:
            r2[t2] = np.clip(0.5 * (n2.tank_hmin + n2.tank_hmax),
                             n2.tank_hmin + 0.3 * (n2.tank_hmax - n2.tank_hmin),
                             n2.tank_hmax - 0.3 * (n2.tank_hmax - n2.tank_hmin))
        gg = np.random.default_rng(808)
        for dev in ("cpu", "cuda"):
            ss = GGASolver(n2, mode="dense", inp_path=f, dtype=DT, device=dev,
                           dense_tank_bound_check=False)
            worst, ok = 0.0, True
            for B in (1, 3, 16):
                DD = torch.as_tensor(d2[None, :] * gg.uniform(0.7, 1.3, (B, d2.size)),
                                     dtype=DT, device=dev)
                RR = np.where(np.isnan(r2)[None, :], np.nan,
                              r2[None, :] + gg.uniform(-1, 1, (B, r2.size)))
                RR = torch.as_tensor(RR, dtype=DT, device=dev)
                a = ss.solve(DD, RR)
                b = ss.solve(DD, RR, assemble="csr")
                worst = max(worst, float((a["head_ft"] - b["head_ft"]).abs().max()))
                ok &= (a["head_ft"].cpu().numpy().tobytes() ==
                       b["head_ft"].cpu().numpy().tobytes())
                ok &= bool((a["iters"] == b["iters"]).all())
            print("  %-8s %-4s max|ΔH|=%.3e 逐位相同=%s" % (stem, dev, worst, ok))
    except Exception:                                    # noqa: BLE001
        print("  %-8s %s" % (stem, traceback.format_exc().strip().split("\n")[-1][:90]))

print("\n[I] 签名兼容：新参数只能追加在末尾（老式位置调用不能错位）")
sig = inspect.signature(GGASolver.solve)
print("  solve:", list(sig.parameters))
print("  solve_unrolled:", list(inspect.signature(solve_unrolled).parameters))
print("  solve_polished:", list(inspect.signature(solve_polished).parameters))
print("  ImplicitGGASolve.forward 是否透传 assemble/linear_solver：",
      "assemble" in inspect.signature(implicit_solve).parameters)

print("\nADV_LOCAL 结束")

# -*- coding: utf-8 -*-
"""交付判断用：NW_Model 上到底**拿不拿得到可用梯度**。

上一轮自报："solve_unrolled 的截断梯度在这个网上发散（verdict=不建议，
rel_K=1.0, gmax=5e20），要梯度得走 ImplicitGGASolve"。
限制自报不必替他证伪，但"主线能不能用"取决于**另一条路走不走得通**，
所以这里直接验 ImplicitGGASolve：伴随梯度 vs 中心差分，逐坐标对拍。
CPU、float64、B=1。
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                        # noqa: E402
import aud_lib as AL                                      # noqa: E402
import torch                                              # noqa: E402
from dgga.autodiff import implicit_solve, unrolled_grad_health   # noqa: E402
from dgga.parse import parse_inp                          # noqa: E402
from dgga.solver import GGASolver                         # noqa: E402

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "networks", "EXAMPLE",
        "epanet-example-networks/epanet-tests/large/NW_Model.inp")
for k, v in AL.prov():
    print("   %-18s %s" % (k, v))
net = parse_inp(PATH)
s = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
              inp_path=PATH, dense_tank_bound_check=False)
d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
rh0 = np.nan_to_num(AL.fixed_head(net, 0))
Nj = s.Nj
rng = np.random.default_rng(11)
w = rng.normal(size=Nj)
wt = torch.as_tensor(w, dtype=torch.float64)


def loss(dvec):
    H, Q, E = implicit_solve(s, dvec, rh0, accuracy=1e-14, max_iter=30)
    return (wt * H[..., s.junc_nodes_t]).sum()


print("\n=== ImplicitGGASolve 伴随 vs 中心差分（demand 坐标）===")
dv = torch.as_tensor(d0, dtype=torch.float64).requires_grad_(True)
t0 = time.perf_counter()
L = loss(dv)
L.backward()
g = dv.grad.numpy().copy()
print("  伴随一次 f+b 用时 %.1f s   |g|_max=%.6e  |g|_2=%.6e  有限=%s"
      % (time.perf_counter() - t0, float(np.abs(g).max()),
         float(np.linalg.norm(g)), bool(np.isfinite(g).all())))

# 挑 6 个坐标：|g| 最大的 3 个 + 需水最大的 3 个
junc = np.asarray(s.junc_nodes_t)
cand = list(np.argsort(-np.abs(g[junc]))[:2]) + list(np.argsort(-d0[junc])[:2])
seen, coords = set(), []
for c in cand:
    n = int(junc[c])
    if n not in seen:
        seen.add(n)
        coords.append(n)
print("  %-8s %-14s %-14s %-12s %s" % ("node", "adjoint g", "FD g", "rel", "h"))
worst = 0.0
for n in coords:
    h = max(1e-6, abs(d0[n]) * 1e-4)
    gs = []
    for sgn in (+1, -1):
        dd = d0.copy()
        dd[n] += sgn * h
        with torch.no_grad():
            H, Q, E = implicit_solve(s, dd, rh0, accuracy=1e-14, max_iter=30)
        gs.append(float((wt * H[..., s.junc_nodes_t]).sum()))
    fd = (gs[0] - gs[1]) / (2 * h)
    rel = abs(g[n] - fd) / max(abs(fd), 1e-12)
    worst = max(worst, rel)
    print("  %-8s %-14.6e %-14.6e %-12.3e %.3e"
          % (net.node_id[n], g[n], fd, rel, h))
print("  ==> 最差相对误差 %.3e  （门槛 1e-6 => %s）"
      % (worst, "PASS" if worst < 1e-6 else "FAIL"))

print("\n=== 复核上一轮自报的 unrolled 体检 ===")
try:
    K = 5
    hh = unrolled_grad_health(s, d0, rh0, K=K)
    print("  ", hh)
except Exception as e:                                    # noqa: BLE001
    print("   unrolled_grad_health 抛错：%s: %s"
          % (type(e).__name__, str(e).splitlines()[0][:120]))
print("\nA6 DONE")

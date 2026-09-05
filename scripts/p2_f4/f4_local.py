# -*- coding: utf-8 -*-
"""F4 本机冒烟（本机 有 CUDA、无 nvmath）：新接口的存在性/幂等/守卫/缺省不变。

真正的显存证据在集群（f4_mem.py）；这里只保证接口在没有 nvmath 的机器上也
不会把缺省通路带偏，且释放接口是幂等的。
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                       # noqa: E402
from dgga.solver import GGASolver                      # noqa: E402

f = os.path.join(ROOT, "networks", "public", "Hanoi.inp")
net = parse_inp(f)
s = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense", inp_path=f)
print("cache 类型 =", type(s._cudss_cache).__name__, " 缺省上限 =", s.cudss_cache_max)
print("空缓存 cudss_free() ->", s.cudss_free(), " 再来一次 ->", s.cudss_free())
with s.cudss_session():
    pass
print("cudss_session 正常退出，cache len =", len(s._cudss_cache))

d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
D = torch.as_tensor(np.repeat(d[None, :], 2, 0), dtype=torch.float64)
R = torch.as_tensor(np.repeat(rh[None, :], 2, 0), dtype=torch.float64)

for kw, want in [(dict(assemble="csr", linear_solver="cudss"), NotImplementedError),
                 (dict(assemble="dense", linear_solver="cudss"), ValueError)]:
    try:
        s.solve(D, R, **kw)
        print("!! 没有 raise:", kw)
    except want as e:
        print("守卫 ok:", type(e).__name__, "|", str(e)[:56].replace("\n", " "))
print("撞完守卫后 cache len =", len(s._cudss_cache), "（应为 0）")

o1 = s.solve(D, R)
o2 = s.solve(D, R, assemble="csr")
print("csr vs 缺省 max|ΔH| = %.3e（应 0.000e+00）"
      % float((o1["head_ft"] - o2["head_ft"]).abs().max()))

s.cudss_cache_max = 0
try:
    s._cudss_evict()
    print("!! cap=0 没有 raise")
except ValueError as e:
    print("cap 校验 ok:", str(e)[:48])
s.cudss_cache_max = None
print("cap=None（不限）_cudss_evict() ->", s._cudss_evict())
print("F4 LOCAL DONE")

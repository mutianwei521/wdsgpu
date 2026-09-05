# -*- coding: utf-8 -*-
"""F4 第二轮：**每个策略一个全新进程**，避免上一策略把 cuDSS 的可复用池带高。

用法: python3 f4b_gpu.py <cycle|tail|ragged|big|pool> <cap: none|整数>

  cycle   B∈{1,8,64,256} 轮转 6 轮 - 逐出上限太小时 LRU 次次 miss 的代价
  tail    每 epoch 4×B=256 + 1×B=37，4 个 epoch - 最常见的"整批+尾批"
  ragged  ky4 B=177..256 共 80 个互不相同的批量 - 修复前单调爬升
  big     ky4 B=1024..1072 共 49 个互不相同的批量 - 修复前撞 OOM
  pool    释放-重建 20 轮 + 跨 B 复用 - 证明 free() 掉的字节进了可复用池
"""
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                     # noqa: E402
from dgga.solver import GGASolver                    # noqa: E402
from nvmath.sparse.advanced import DirectSolver      # noqa: E402

MODE, CAPARG = sys.argv[1], sys.argv[2]
CAP = None if CAPARG == "none" else int(CAPARG)
DEV, DT = "cuda", torch.float64
KW = dict(assemble="csr", linear_solver="cudss")
_CNT = {"plan": 0, "free": 0}
_op, _ofr = DirectSolver.plan, DirectSolver.free
DirectSolver.plan = lambda self, **k: (_CNT.__setitem__("plan", _CNT["plan"] + 1),
                                       _op(self, **k))[1]
DirectSolver.free = lambda self: (_CNT.__setitem__("free", _CNT["free"] + 1),
                                  _ofr(self))[1]


def dev_used():
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


def torch_res():
    return torch.cuda.memory_reserved() / 2 ** 20


def nontorch():
    return dev_used() - torch_res()


f = os.path.join(ROOT, "p2nets", "ky4.inp")
net = parse_inp(f)
s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=f,
              dense_tank_bound_check=False)
s.cudss_cache_max = CAP
d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
tn = np.asarray(net.tank_node, dtype=np.int64)
if tn.size:
    lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
    hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
    rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)


def batchify(B, seed):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.6, 1.4, (B, 1)) * g.uniform(0.75, 1.25, (B, d.size))
    R = rh[None, :] + g.uniform(-2.0, 2.0, (B, rh.size))
    R = np.where(np.isnan(rh)[None, :], np.nan, R)
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


HEAD = "[%s cap=%s node=%s]" % (MODE, CAPARG, os.uname().nodename)
print("\n" + "-" * 78)
print(HEAD, "起始 非torch = %.1f MiB" % nontorch())

if MODE == "cycle":
    BS = [1, 8, 64, 256]
    DATA = {B: batchify(B, 900 + B) for B in BS}
    for B in BS:                                   # 预热：把 plan 全做掉
        s.solve(*DATA[B], **KW)
    s.cudss_free()
    torch.cuda.empty_cache()
    _CNT.update(plan=0, free=0)
    tots = []
    for r in range(6):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for B in BS:
            s.solve(*DATA[B], **KW)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        tots.append(dt)
        print("%s 轮%d 非torch=%8.1f torch=%7.1f plan累计=%3d 本轮=%9.1f ms"
              % (HEAD, r, nontorch(), torch_res(), _CNT["plan"], dt))
    print("%s 汇总 中位轮=%.1f ms plan=%d free=%d 末非torch=%.1f MiB"
          % (HEAD, float(np.median(tots)), _CNT["plan"], _CNT["free"], nontorch()))

elif MODE == "tail":
    D256, R256 = batchify(256, 1156)
    D37, R37 = batchify(37, 37)
    s.solve(D256, R256, **KW)
    s.solve(D37, R37, **KW)
    s.cudss_free()
    torch.cuda.empty_cache()
    _CNT.update(plan=0, free=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(4):
        for _ in range(4):
            s.solve(D256, R256, **KW)
        s.solve(D37, R37, **KW)
    torch.cuda.synchronize()
    print("%s 4 epoch 总耗时=%.1f ms plan=%d free=%d 末非torch=%.1f MiB"
          % (HEAD, (time.perf_counter() - t0) * 1e3, _CNT["plan"], _CNT["free"],
             nontorch()))

elif MODE in ("ragged", "big"):
    BLIST = list(range(177, 257)) if MODE == "ragged" else list(range(1024, 1073))
    step = 8 if MODE == "ragged" else 4
    print("%s %5s %6s %10s %9s %10s %6s" % (HEAD, "第i个", "B", "非torch",
                                            "torch", "设备已用", "cache"))
    died = None
    t0 = time.perf_counter()
    for i, B in enumerate(BLIST):
        try:
            Db, Rb = batchify(B, 5000 + B)
            s.solve(Db, Rb, **KW)
            del Db, Rb
        except Exception:                          # noqa: BLE001
            died = (i, B, traceback.format_exc().strip().split("\n")[-1][:90])
            break
        if i % step == 0 or i == len(BLIST) - 1:
            print("%s %5d %6d %10.1f %9.1f %10.1f %6d"
                  % (HEAD, i, B, nontorch(), torch_res(), dev_used(),
                     len(s._cudss_cache)))
    if died is not None:
        print("%s **第 %d 个批量（B=%d）挂了**：%s" % ((HEAD,) + died))
    else:
        print("%s 全部 %d 个批量跑完，末非torch=%.1f MiB，cache %d 份，耗时 %.1f s"
              % (HEAD, len(BLIST), nontorch(), len(s._cudss_cache),
                 time.perf_counter() - t0))
    print("%s 收尾 cudss_free() 释放 %d 份 → 非torch=%.1f MiB"
          % (HEAD, s.cudss_free(empty_cache=True), nontorch()))

elif MODE == "pool":
    Db, Rb = batchify(256, 256)
    print("%s 释放-重建 20 轮（同一 B=256），看非torch 是否单调" % HEAD)
    for r in range(20):
        s.solve(Db, Rb, **KW)
        n = nontorch()
        s.cudss_free()
        if r % 4 == 0 or r == 19:
            print("%s   轮%2d 建后非torch=%8.1f 释放后=%8.1f MiB"
                  % (HEAD, r, n, nontorch()))
    del Db, Rb
    torch.cuda.empty_cache()
    base = nontorch()
    print("%s 20 轮后非torch=%.1f MiB（起始 %.1f）" % (HEAD, base, base))
    # 跨 B 复用：先建 B=256 再释放，然后建 B=1024，看占用是否叠加
    D1, R1 = batchify(256, 1)
    s.solve(D1, R1, **KW)
    a = nontorch()
    s.cudss_free()
    b = nontorch()
    del D1, R1
    torch.cuda.empty_cache()
    D2, R2 = batchify(1024, 2)
    s.solve(D2, R2, **KW)
    c = nontorch()
    print("%s 跨 B 复用：建 B=256 → %.1f，free → %.1f，再建 B=1024 → %.1f MiB"
          % (HEAD, a, b, c))
    print("%s cudss_cache_info(): %s" % (HEAD, s.cudss_cache_info()))
    del D2, R2
    s.cudss_free(empty_cache=True)
    print("%s 收尾非torch=%.1f MiB（free 计数 %d）" % (HEAD, nontorch(), _CNT["free"]))

print("%s DONE" % HEAD)

# -*- coding: utf-8 -*-
"""P2 审计 F4 的修复验收：cuDSS state 的逐出与显式释放。

  §A  释放接口真的把设备内存还了（ky4 B=256/1024，拆 torch 侧 / 非 torch 侧）
  §B  变批量循环 B∈{1,8,64,256}：cap=None（=修复前）vs cap=2（缺省）vs cap=4
  §C  ragged 变批量（很多互不相同的 B）：修复前单调爬升到 OOM，修复后持平
  §D  现实形态「整批 + 尾批」：cap=1 vs 2 vs None 的重 plan 次数与耗时 → 定缺省
  §E  逐出后重进的代价（重 plan），即缺省选 2 而不是 1 的定价依据

cap=None 逐字复现修复前的行为（无界 dict、从不 free），故 "修复前/后" 的对照
在同一份代码同一次作业里做，排除跨节点差异（P2 审计 §1 量到 1.45x 的节点方差）。
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

DEV, DT = "cuda", torch.float64
NETDIR = os.path.join(ROOT, "p2nets")
NMAP = {"Net3": "Net3.inp", "Modena": "Modena.inp", "City_D": "City_D.inp",
        "ky4": "ky4.inp"}

from nvmath.sparse.advanced import DirectSolver      # noqa: E402
_CNT = {"plan": 0, "fact": 0, "solve": 0, "free": 0}
_op, _of, _os_, _ofr = (DirectSolver.plan, DirectSolver.factorize,
                        DirectSolver.solve, DirectSolver.free)
DirectSolver.plan = lambda self, **k: (_CNT.__setitem__("plan", _CNT["plan"] + 1),
                                       _op(self, **k))[1]
DirectSolver.factorize = lambda self, **k: (_CNT.__setitem__("fact", _CNT["fact"] + 1),
                                            _of(self, **k))[1]
DirectSolver.solve = lambda self, **k: (_CNT.__setitem__("solve", _CNT["solve"] + 1),
                                        _os_(self, **k))[1]
DirectSolver.free = lambda self: (_CNT.__setitem__("free", _CNT["free"] + 1),
                                  _ofr(self))[1]
KW = dict(assemble="csr", linear_solver="cudss")


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, rh


def batchify(d, rh, B, seed):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.6, 1.4, (B, 1)) * g.uniform(0.75, 1.25, (B, d.size))
    R = rh[None, :] + g.uniform(-2.0, 2.0, (B, rh.size))
    R = np.where(np.isnan(rh)[None, :], np.nan, R)
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


def mk(stem):
    f = os.path.join(NETDIR, NMAP[stem])
    net = parse_inp(f)
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=f,
                  dense_tank_bound_check=False)
    return net, s


def dev_used():
    """设备级已用 MiB（含 CUDA context、torch 缓存分配器、cuDSS 内部缓冲）。"""
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


def torch_res():
    return torch.cuda.memory_reserved() / 2 ** 20


def nontorch():
    """非 torch 占用 = 设备已用 − torch 缓存分配器保留。cuDSS 那块就在这里。"""
    return dev_used() - torch_res()


def last():
    return traceback.format_exc().strip().split("\n")[-1][:110]


print(torch.cuda.get_device_name(0), "| torch", torch.__version__,
      "| cuda", torch.version.cuda, "| host", os.uname().nodename,
      "| cpus", os.cpu_count())
import nvmath                                        # noqa: E402
print("nvmath", nvmath.__version__)
print("缺省 cudss_cache_max =", GGASolver.__init__.__doc__ is not None and "见实例")

# ================================================================= §A 释放接口
print("\n" + "=" * 78)
print("§A 释放接口：cudss_free() 到底还没还设备内存（ky4，MiB）")
print("   列：设备已用 / torch 保留 / 非torch（cuDSS 那块就在这一列）")
net, s = mk("ky4")
d, rh = boundary(net)
print("%-4s %-26s %10s %10s %10s" % ("B", "阶段", "设备已用", "torch保留", "非torch"))
for B in (256, 1024):
    D, R = batchify(d, rh, B, 4242 + B)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    a0 = (dev_used(), torch_res(), nontorch())
    print("%-4d %-26s %10.1f %10.1f %10.1f" % (B, "0 建 state 前", *a0))
    pl = s.cudss_plan(B)
    torch.cuda.synchronize()
    print("%-4d %-26s %10.1f %10.1f %10.1f  (plan %.1f ms)"
          % (B, "1 plan() 后", dev_used(), torch_res(), nontorch(), pl))
    out = s.solve(D, R, **KW)
    torch.cuda.synchronize()
    a2 = (dev_used(), torch_res(), nontorch())
    print("%-4d %-26s %10.1f %10.1f %10.1f  (iters %d)"
          % (B, "2 首解后（含数值分解）", *a2, int(out["iters"].max())))
    del out, D, R
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    a3 = (dev_used(), torch_res(), nontorch())
    print("%-4d %-26s %10.1f %10.1f %10.1f  ← 只 empty_cache() 收不回"
          % (B, "3 丢输入 + empty_cache()", *a3))
    nfreed = s.cudss_free()
    torch.cuda.synchronize()
    a4 = (dev_used(), torch_res(), nontorch())
    print("%-4d %-26s %10.1f %10.1f %10.1f  (释放 %d 份)"
          % (B, "4 cudss_free()", *a4, nfreed))
    torch.cuda.empty_cache()
    a5 = (dev_used(), torch_res(), nontorch())
    print("%-4d %-26s %10.1f %10.1f %10.1f" % (B, "5 再 empty_cache()", *a5))
    print("     ⇒ 一份 state 的非torch(cuDSS) 占用 = %.1f MiB；"
          "cudss_free() 还回 %.1f MiB；单靠 empty_cache() 只还 %.1f MiB"
          % (a3[2] - a0[2], a3[2] - a4[2], a0[2] - a0[2]))
print("free() 调用计数 =", _CNT["free"])

# 上下文管理器
_CNT["free"] = 0
before = nontorch()
with s.cudss_session():
    D, R = batchify(d, rh, 128, 7)
    s.solve(D, R, **KW)
    inside = nontorch()
del D, R
torch.cuda.synchronize()
after = nontorch()
print("cudss_session：进入前 %.1f → 内部 %.1f → 退出后 %.1f MiB（free 计数 %d，"
      "cache len %d）" % (before, inside, after, _CNT["free"], len(s._cudss_cache)))

# ============================================================ §B 变批量循环
print("\n" + "=" * 78)
print("§B 变批量循环 ky4 B∈{1,8,64,256}，每策略 6 轮（cap=None ≡ 修复前）")
BS = [1, 8, 64, 256]
DATA = {B: batchify(d, rh, B, 900 + B) for B in BS}
print("%-8s %-6s %10s %10s %10s %8s %9s"
      % ("策略", "轮", "设备已用", "torch保留", "非torch", "plan数", "本轮ms"))
SUM = {}
for cap in (None, 2, 4):
    s.cudss_free()
    torch.cuda.empty_cache()
    s.cudss_cache_max = cap
    tag = "None" if cap is None else str(cap)
    _CNT.update(plan=0, free=0)
    t_all = 0.0
    for r in range(6):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for B in BS:
            s.solve(*DATA[B], **KW)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        t_all += dt
        print("cap=%-6s %-6d %10.1f %10.1f %10.1f %8d %9.1f"
              % (tag, r, dev_used(), torch_res(), nontorch(), _CNT["plan"], dt))
    SUM[tag] = (nontorch(), _CNT["plan"], _CNT["free"], t_all / 6)
print("\n%-8s %12s %8s %8s %12s" % ("策略", "末轮非torch", "plan总数", "free总数",
                                    "均轮ms"))
for tag, v in SUM.items():
    print("cap=%-6s %12.1f %8d %8d %12.1f" % (tag, v[0], v[1], v[2], v[3]))
s.cudss_free()
torch.cuda.empty_cache()

# ==================================================== §D 现实形态：整批 + 尾批
print("\n" + "=" * 78)
print("§D 现实形态：每 epoch = 4×B=256 + 1×尾批 B=37，跑 4 个 epoch")
D256, R256 = DATA[256]
D37, R37 = batchify(d, rh, 37, 37)
print("%-8s %8s %8s %10s %12s" % ("策略", "plan数", "free数", "总耗时ms", "末轮非torch"))
for cap in (1, 2, None):
    s.cudss_free()
    torch.cuda.empty_cache()
    s.cudss_cache_max = cap
    s.solve(D256, R256, **KW)          # 预热，不计
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
    dt = (time.perf_counter() - t0) * 1e3
    print("cap=%-6s %8d %8d %10.1f %12.1f"
          % ("None" if cap is None else cap, _CNT["plan"], _CNT["free"], dt,
             nontorch()))
s.cudss_free()
torch.cuda.empty_cache()

# ==================================================== §E 逐出后重进的代价
print("\n" + "=" * 78)
print("§E 逐出→重进的代价（ky4，ms）：缓存命中的一次 solve vs 重 plan+solve")
s.cudss_cache_max = 4
for B in (64, 256):
    Db, Rb = DATA[B]
    s.cudss_free()
    torch.cuda.empty_cache()
    s.solve(Db, Rb, **KW)                        # 建好并预热
    torch.cuda.synchronize()
    hit = []
    for _ in range(5):
        t0 = time.perf_counter()
        s.solve(Db, Rb, **KW)
        torch.cuda.synchronize()
        hit.append((time.perf_counter() - t0) * 1e3)
    miss = []
    for _ in range(3):
        s.cudss_free()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s.solve(Db, Rb, **KW)
        torch.cuda.synchronize()
        miss.append((time.perf_counter() - t0) * 1e3)
    hm, mm = float(np.median(hit)), float(np.median(miss))
    print("B=%-5d 命中 %8.2f | 未命中(含 plan) %9.2f | 重建代价 %8.2f ms "
          "= %.3f ms/矩阵" % (B, hm, mm, mm - hm, (mm - hm) / B))
s.cudss_free()
torch.cuda.empty_cache()
del DATA, D256, R256, D37, R37
torch.cuda.empty_cache()

# ============================================ §C ragged 变批量：单调爬升 vs 持平
print("\n" + "=" * 78)
print("§C ragged 变批量（80 个互不相同的 B，ky4 B=177..256）：")
print("   cap=None（修复前）应单调爬升直至 OOM；cap=2（缺省）应持平")
BLIST = list(range(177, 257))
for cap in (None, 2):
    s.cudss_free()
    torch.cuda.empty_cache()
    s.cudss_cache_max = cap
    tag = "None" if cap is None else str(cap)
    print("  --- cap=%s ---" % tag)
    print("  %5s %6s %12s %12s %12s %8s"
          % ("第i个", "B", "设备已用", "torch保留", "非torch", "cache"))
    died = None
    for i, B in enumerate(BLIST):
        try:
            Db, Rb = batchify(d, rh, B, 5000 + B)
            s.solve(Db, Rb, **KW)
            del Db, Rb
        except Exception:                        # noqa: BLE001
            died = (i, B, last())
            break
        if i % 8 == 0 or i == len(BLIST) - 1:
            print("  %5d %6d %12.1f %12.1f %12.1f %8d"
                  % (i, B, dev_used(), torch_res(), nontorch(),
                     len(s._cudss_cache)))
    if died is not None:
        print("  **第 %d 个 B（B=%d）挂了**：%s" % died)
    else:
        print("  80 个批量全部跑完，末态非torch = %.1f MiB，cache %d 份"
              % (nontorch(), len(s._cudss_cache)))
    s.cudss_free()
    torch.cuda.empty_cache()
    print("  收尾 cudss_free() 后非torch = %.1f MiB" % nontorch())

print("\nF4 GPU JOB DONE")

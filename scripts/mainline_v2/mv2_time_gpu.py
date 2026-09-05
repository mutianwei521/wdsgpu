# -*- coding: utf-8 -*-
"""mv2_time_gpu.py - 主线证据包 v2 · L-TOWN f+b 全表（三列真数字）。

三列（任务书口径）：
  旧   = dense 前向(no_grad) + ImplicitGGASolve CPU 伴随（epanet 求解器 +
         scipy splu 逐场景串行；同节点实测，非引用旧表）；
  新d  = implicit_solve(adjoint='gpu') + dense 前向/dense Cholesky 终态分解；
  新c  = implicit_solve(adjoint='gpu') + csr 装配 + cuDSS。
倍数拆两因子（sparse_gpu_plan.md §11.2 同口径，作用在新三列上）：
  F1 = 旧d/新d - “伴随搬上 GPU”本身值多少（线代同为稠密）；
  F2 = 新d/新c - “稀疏 vs 稠密”本身值多少（伴随同在 GPU）；
  总 = 旧d/新c = F1×F2；另给 旧c/新c（上轮 211.9x 的口径，旧c=cudss 前向+CPU 伴随）。
相序（每个 B）：cudss_free+empty_cache → dense 段（fwd_d→fb_gd）→ implCPU
（GPU 闲，状态帧取自 dense 前向）→ cudss 段（fwd_c→fb_gc）。dense 段放最前
是为了让 B=1024 的 fwd_d 在无 cuDSS 常驻、无碎片的状态下起跑（上轮 1464991/
1465032 两种相序 fwd_d@1024 都被前序占用顶出 OOM；本轮若仍 OOM 则如实记
“作业内 OOM”，与 ltm 全新进程实测（B=1024 可过）并列引用）。
布局：扁平部署（ROOT=脚本目录，dgga/ 与 networks_prv/ 同级）；本机冒烟
MV2_DEV=cpu MV2_BS=2 可跑（cudss 段自动跳过）。
"""
import hashlib
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(ROOT, "dgga")):     # 本机冒烟：仓库根布局
    ROOT = os.path.dirname(os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                      # noqa: E402
from dgga.solver import GGASolver                     # noqa: E402
from dgga.autodiff import implicit_solve              # noqa: E402

DEV = os.environ.get("MV2_DEV", "cuda")
DT = torch.float64
INP = os.path.join(ROOT, "networks_prv", "L-TOWN.inp")
if not os.path.exists(INP):
    INP = os.path.join(ROOT, "networks", "public", "_cleaned", "L-TOWN.inp")
SEED = 2026
NODE = os.popen("hostname").read().strip()
BS = [int(x) for x in os.environ.get("MV2_BS", "1,8,64,256,512,1024").split(",")]
ACC = float(os.environ.get("MV2_ACC", "1e-6"))
HAS_CUDSS = DEV == "cuda"
try:
    import nvmath                                      # noqa: F401
except ImportError:
    HAS_CUDSS = False
FAILS = []


def last():
    return traceback.format_exc().strip().split("\n")[-1][:160]


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def tg(fn, budget=6.0, reps_max=3, warm=1):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    one = time.perf_counter() - t0
    reps = max(2, min(reps_max, int(budget / max(one, 1e-6))))
    best = one
    for _ in range(reps - 1):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - t0)
    return best


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, np.nan_to_num(rh)


def batchify(d, rh, B, seed=SEED):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.85, 1.15, (B, d.size))
    R = rh[None, :] + g.uniform(-1.0, 1.0, (B, rh.size))
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


print("=" * 96)
print("主线 v2 · L-TOWN f+b 三列全表（旧 dense+CPU 伴随 / 新 dense+GPU / 新 cudss+GPU）")
print("node:", NODE, "| torch", torch.__version__, "| dev:", DEV,
      "" if DEV != "cuda" else "| " + torch.cuda.get_device_name(0))
print("cudss:", HAS_CUDSS, "| B 列:", BS, "| ACC:", ACC)
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f,
          hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print("sha256 INP", hashlib.sha256(open(INP, "rb").read()).hexdigest())
print()

net = parse_inp(INP)
s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=INP,
              dense_status_machine=True)
se = GGASolver(net, mode="epanet", inp_path=INP)
d0, rh0 = boundary(net)
base = se.run_gga(d0, rh0, do_status=True)
K0set = base["setting"].copy()
jn = torch.as_tensor(np.asarray(se.junc_nodes), dtype=torch.long)
print("### L-TOWN Nj=%d L=%d N=%d nnz=%d" %
      (s.Nj, s.L, net.N, int(s.A_csr_dense_pos.numel())))
sys.stdout.flush()


def gpu_grad(D, R, W, ls):
    Dv = D.detach().clone().requires_grad_(True)
    h, _q, _e = implicit_solve(
        s, Dv, R, adjoint="gpu", accuracy=ACC, max_iter=200,
        status_machine=True, assemble=("csr" if ls == "cudss" else "dense"),
        linear_solver=ls)
    (h.index_select(1, jn.to(DEV)) * W).sum().backward()
    return Dv.grad


def cpu_grad(D, R, W, S_all):
    """旧训练形态：GPU 前向拿状态 → 按状态分组冻结 → CPU 隐式伴随。"""
    uniq, inv = np.unique(S_all, axis=0, return_inverse=True)
    Dc = D.detach().cpu()
    Rc = R.detach().cpu()
    Wc = W.detach().cpu()
    Dv = Dc.clone().requires_grad_(True)
    loss = 0.0
    for gidx in range(uniq.shape[0]):
        ii = torch.as_tensor(np.where(inv == gidx)[0])
        h, _q, _e = implicit_solve(se, Dv.index_select(0, ii),
                                   Rc.index_select(0, ii),
                                   speed=K0set, status=uniq[gidx])
        loss = loss + (h.index_select(1, jn)
                       * Wc.index_select(0, ii).index_select(1, jn)).sum()
    loss.backward()
    return Dv.grad, uniq.shape[0]


# ======================================================================
print("§1 快速正确性（B=8）：三路 demand 梯度对拍")
B1 = 8
D1, R1 = batchify(d0, rh0, B1)
W1 = torch.as_tensor(np.random.default_rng(7).normal(size=(B1, s.Nj)),
                     dtype=DT, device=DEV)
try:
    with torch.no_grad():
        S1 = s.solve(D1, R1, status_machine=True)["status"].cpu().numpy()
    g_ref, ng1 = cpu_grad(D1, R1, W1, S1)
    g_ref = g_ref.to(DEV)
    g_d = gpu_grad(D1, R1, W1, "dense")
    rel = lambda a, b: float((a - b).abs().max()
                             / max(float(a.abs().max()),
                                   float(b.abs().max()), 1e-300))
    r_d = rel(g_d, g_ref)
    msg = "  dense vs CPU=%.3e" % r_d
    ok1 = r_d < 1e-6
    if HAS_CUDSS:
        g_c = gpu_grad(D1, R1, W1, "cudss")
        r_c = rel(g_c, g_ref)
        msg += "  cudss vs CPU=%.3e  cudss vs dense=%.3e" % (r_c, rel(g_c, g_d))
        ok1 &= r_c < 1e-6
    print(msg + "（状态组=%d）门槛 <1e-6：%s" % (ng1, "PASS" if ok1 else "FAIL"))
    if not ok1:
        FAILS.append("§1")
except Exception:                                      # noqa: BLE001
    FAILS.append("§1")
    print("  [err §1] %s" % last())
del D1, R1, W1
if DEV == "cuda":
    torch.cuda.empty_cache()
sys.stdout.flush()

# ======================================================================
print()
print("§2 时间全表（ms/场景）")
ROWS = []
for B in BS:
    D, R = batchify(d0, rh0, B)
    W = torch.as_tensor(np.random.default_rng(7).normal(size=(B, s.Nj)),
                        dtype=DT, device=DEV)
    res = {}
    gn = {}
    S_all = None
    if HAS_CUDSS:
        s.cudss_free(empty_cache=True)
    elif DEV == "cuda":
        torch.cuda.empty_cache()
    # ---- dense 段（先跑：无 cuDSS 常驻、无碎片）----
    try:
        def fd_():
            with torch.no_grad():
                s.solve(D, R, status_machine=True)
        res["fwd_d"] = tg(fd_) * 1e3 / B
        with torch.no_grad():
            S_all = s.solve(D, R,
                            status_machine=True)["status"].cpu().numpy()
    except torch.OutOfMemoryError:
        res["fwd_d"] = None
        if DEV == "cuda":
            torch.cuda.empty_cache()
    except Exception:                                  # noqa: BLE001
        res["fwd_d"] = None
        print("  [err fwd_d B=%d] %s" % (B, last()))
        if DEV == "cuda":
            torch.cuda.empty_cache()
    try:
        def f2d_():
            gn["fb_d"] = float(gpu_grad(D, R, W, "dense").norm())
        res["fb_d"] = tg(f2d_) * 1e3 / B
    except torch.OutOfMemoryError:
        res["fb_d"] = None
        if DEV == "cuda":
            torch.cuda.empty_cache()
    except Exception:                                  # noqa: BLE001
        res["fb_d"] = None
        print("  [err fb_d B=%d] %s" % (B, last()))
        if DEV == "cuda":
            torch.cuda.empty_cache()
    # ---- 旧形态 CPU 伴随段（GPU 闲；一次 + 便宜再来一次取最小）----
    if S_all is None and HAS_CUDSS:
        try:
            with torch.no_grad():
                S_all = s.solve(D, R, status_machine=True, assemble="csr",
                                linear_solver="cudss")["status"].cpu().numpy()
        except Exception:                              # noqa: BLE001
            print("  [err S_all(cudss) B=%d] %s" % (B, last()))
    timp = None
    ngroup = -1
    g_cpu_norm = float("nan")
    if S_all is not None:
        try:
            t0 = time.perf_counter()
            g1, ngroup = cpu_grad(D, R, W, S_all)
            timp = time.perf_counter() - t0
            g_cpu_norm = float(g1.norm())
            if timp < 30.0:
                t0 = time.perf_counter()
                cpu_grad(D, R, W, S_all)
                timp = min(timp, time.perf_counter() - t0)
        except Exception:                              # noqa: BLE001
            print("  [err impl B=%d] %s" % (B, last()))
    timp_per = None if timp is None else timp * 1e3 / B
    # ---- cudss 段 ----
    if HAS_CUDSS:
        try:
            def fc_():
                with torch.no_grad():
                    s.solve(D, R, status_machine=True,
                            assemble="csr", linear_solver="cudss")
            res["fwd_c"] = tg(fc_) * 1e3 / B
        except Exception:                              # noqa: BLE001
            res["fwd_c"] = None
            print("  [err fwd_c B=%d] %s" % (B, last()))
            torch.cuda.empty_cache()
        try:
            def f2c_():
                gn["fb_c"] = float(gpu_grad(D, R, W, "cudss").norm())
            res["fb_c"] = tg(f2c_) * 1e3 / B
        except Exception:                              # noqa: BLE001
            res["fb_c"] = None
            print("  [err fb_c B=%d] %s" % (B, last()))
            torch.cuda.empty_cache()
    else:
        res["fwd_c"] = res["fb_c"] = None
    old_d = None if (res.get("fwd_d") is None or timp_per is None) \
        else res["fwd_d"] + timp_per
    old_c = None if (res.get("fwd_c") is None or timp_per is None) \
        else res["fwd_c"] + timp_per
    ROWS.append((B, res.get("fwd_d"), res.get("fwd_c"), timp_per,
                 old_d, old_c, res.get("fb_d"), res.get("fb_c"),
                 ngroup, g_cpu_norm, gn.get("fb_d"), gn.get("fb_c")))
    cell = lambda x, w=9: ("OOM").rjust(w) if x is None else ("%*.5f" % (w, x))
    print("  B=%-5d fwd d %s c %s | implCPU %s (组%d) | f+b(gpu) d %s c %s"
          " | |g| cpu=%.6e gpu_d=%s gpu_c=%s"
          % (B, cell(res.get("fwd_d")), cell(res.get("fwd_c")),
             cell(timp_per, 10), ngroup, cell(res.get("fb_d")),
             cell(res.get("fb_c")), g_cpu_norm,
             "-" if gn.get("fb_d") is None else "%.6e" % gn["fb_d"],
             "-" if gn.get("fb_c") is None else "%.6e" % gn["fb_c"]))
    sys.stdout.flush()
    del D, R, W
    if DEV == "cuda":
        torch.cuda.empty_cache()

print()
print("=" * 96)
print("§T 三列汇总（ms/场景）node=%s" % NODE)
print("B     | 旧=dense+CPU伴随 | 新d=dense+GPU伴随 | 新c=cudss+GPU伴随 | "
      "F1=旧/新d  F2=新d/新c  总=旧/新c | 旧c/新c(上轮口径)")
for (B, fd, fc, ti, od, oc, nd, nc, ng, gcn, gnd, gnc) in ROWS:
    c = lambda x, w=10: ("OOM").rjust(w) if x is None else ("%*.5f" % (w, x))
    f1 = "-" if (od is None or nd is None) else "%.2fx" % (od / nd)
    f2 = "-" if (nd is None or nc is None) else "%.2fx" % (nd / nc)
    tt = "-" if (od is None or nc is None) else "%.2fx" % (od / nc)
    t2 = "-" if (oc is None or nc is None) else "%.2fx" % (oc / nc)
    print("%-5d | %s | %s | %s | %8s %8s %8s | %8s"
          % (B, c(od, 16), c(nd, 17), c(nc, 17), f1, f2, tt, t2))
if FAILS:
    print("FAILS:", FAILS)
print("MV2 TIME DONE rc=%d" % (1 if FAILS else 0))
sys.exit(1 if FAILS else 0)

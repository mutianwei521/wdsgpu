# -*- coding: utf-8 -*-
"""adjgpu_gpu.py - 隐式伴随 GPU 批量化：集群面验收 + L-TOWN f+b 端到端全表。

与 ltmain/lt_time_gpu.py 同口径（batchify/tg/装配路径/CPU 伴随基线），新增：
  §1 正确性（B=8，deterministic）：
     · cuDSS 计数器坐实"伴随零新增 factorize"（backward 期间 factorize 增量
       必须为 0、bwd_refactorize==0、bwd_reuse>=1）；
     · 梯度对拍：adjoint='gpu' 的 cudss / dense 两条 vs 既有 CPU
       ImplicitGGASolve（按收敛状态分组冻结，splu 伴随）。
  §2 时间全表（B∈LT_BS）：
     fwd_d / fwd_c（no_grad 纯前向，与主线同缺省精度）；
     impl(CPU)（旧训练形态的 CPU 伴随段，同节点重测）；
     f+b(gpu,dense) / f+b(gpu,cudss)（implicit_solve(adjoint='gpu') 一把梭：
       批量 GGA + GPU 约化精抛光 + GPU 约化伴随）；
     加速比 = 旧形态 (fwd + impl(CPU)) / 新形态 f+b(gpu)。
布局：与 lt_time_gpu.py 相同的扁平部署（ROOT=脚本目录，dgga/ 与 networks_prv/
在同级）。用法：sbatch aj.sh（LT_BS / LT_ACC 可覆盖）。
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

DEV = "cuda"
DT = torch.float64
INP = os.path.join(ROOT, "networks_prv", "L-TOWN.inp")
if not os.path.exists(INP):
    INP = os.path.join(ROOT, "networks", "public", "_cleaned", "L-TOWN.inp")
SEED = 2026
NODE = os.popen("hostname").read().strip()
BS = [int(x) for x in os.environ.get("LT_BS", "8,64,256,1024").split(",")]
ACC = float(os.environ.get("LT_ACC", "1e-6"))
FAILS = []


def last():
    return traceback.format_exc().strip().split("\n")[-1][:160]


def tg(fn, budget=6.0, reps_max=3, warm=1):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = time.perf_counter() - t0
    reps = max(2, min(reps_max, int(budget / max(one, 1e-6))))
    best = one
    for _ in range(reps - 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
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
print("隐式伴随 GPU 批量化 · 集群验收 + L-TOWN f+b 端到端全表")
print("node:", NODE, "| torch", torch.__version__, "|",
      torch.cuda.get_device_name(0))
try:
    import nvmath                                      # noqa: E402
    print("nvmath", nvmath.__version__, "| B 列:", BS, "| ACC:", ACC)
except ImportError:
    print("nvmath MISSING | B 列:", BS)
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


def gpu_grad(D, R, W, ls, ret_out=False):
    Dv = D.detach().clone().requires_grad_(True)
    h, _q, _e = implicit_solve(
        s, Dv, R, adjoint="gpu", accuracy=ACC, max_iter=200,
        status_machine=True, assemble=("csr" if ls == "cudss" else "dense"),
        linear_solver=ls)
    (h.index_select(1, jn.to(DEV)) * W).sum().backward()
    return (Dv.grad, h.detach()) if ret_out else Dv.grad


def cpu_grad(D, R, W, S_all=None):
    """旧训练形态：GPU 前向拿状态 → 按状态分组冻结 → CPU 隐式伴随。"""
    if S_all is None:
        with torch.no_grad():
            S_all = s.solve(D, R, status_machine=True)["status"].cpu().numpy()
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
print("§1 正确性（B=8，torch.use_deterministic_algorithms(True)）")
torch.use_deterministic_algorithms(True)
B1 = 8
D1, R1 = batchify(d0, rh0, B1)
W1 = torch.as_tensor(np.random.default_rng(7).normal(size=(B1, s.Nj)),
                     dtype=DT, device=DEV)
ok1 = True
try:
    # --- cudss 计数器：backward 期间 factorize 增量必须为 0 ---
    s.cudss_counters(reset=True)
    Dv = D1.detach().clone().requires_grad_(True)
    h, _q, _e = implicit_solve(s, Dv, R1, adjoint="gpu", accuracy=ACC,
                               max_iter=200, status_machine=True,
                               assemble="csr", linear_solver="cudss")
    c_fwd = s.cudss_counters()
    (h.index_select(1, jn.to(DEV)) * W1).sum().backward()
    c_all = s.cudss_counters()
    d_fact = c_all["factorize"] - c_fwd["factorize"]
    print("  cudss 计数：前向 factorize=%d solve=%d | backward 增量 "
          "factorize=%d（须 0） bwd_solve=%d bwd_reuse=%d bwd_refactorize=%d"
          % (c_fwd["factorize"], c_fwd["solve"], d_fact,
             c_all["bwd_solve"], c_all["bwd_reuse"], c_all["bwd_refactorize"]))
    okc = (d_fact == 0 and c_all["bwd_refactorize"] == 0
           and c_all["bwd_reuse"] >= 1)
    print("  伴随零新增 factorize：%s" % ("PASS" if okc else "FAIL"))
    ok1 &= okc
    g_c = Dv.grad
    # --- dense 与 CPU 伴随对拍 ---
    g_d = gpu_grad(D1, R1, W1, "dense")
    g_ref, ng = cpu_grad(D1, R1, W1)
    g_ref = g_ref.to(DEV)
    rel = lambda a, b: float((a - b).abs().max()
                             / max(float(a.abs().max()),
                                   float(b.abs().max()), 1e-300))
    r_cd = rel(g_c, g_d)
    r_c = rel(g_c, g_ref)
    r_d = rel(g_d, g_ref)
    print("  梯度对拍（demand，B=8，状态组=%d）：cudss vs CPU=%.3e  "
          "dense vs CPU=%.3e  cudss vs dense=%.3e" % (ng, r_c, r_d, r_cd))
    okg = r_c < 1e-6 and r_d < 1e-6
    print("  门槛 <1e-6：%s" % ("PASS" if okg else "FAIL"))
    ok1 &= okg
except Exception:                                      # noqa: BLE001
    ok1 = False
    print("  [err §1] %s" % last())
torch.use_deterministic_algorithms(False)
if not ok1:
    FAILS.append("§1")
del D1, R1, W1
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
    # 相序：cudss 段（fwd_c → fb_c → 状态帧）→ implCPU（GPU 闲）→
    # cudss_free+empty_cache → dense 段（fwd_d → fb_d）。
    # 大 B 的 dense 峰值 ~29 GiB（主线 ltm 实测），必须先清 cuDSS 常驻/torch
    # 缓存，否则 fwd_d/fb_d 在 B=1024 被前序占用顶出 OOM（1464991 实测）。
    S_all = None
    try:
        def fc_():
            with torch.no_grad():
                s.solve(D, R, status_machine=True,
                        assemble="csr", linear_solver="cudss")
        res["fwd_c"] = tg(fc_) * 1e3 / B
        with torch.no_grad():
            S_all = s.solve(D, R, status_machine=True, assemble="csr",
                            linear_solver="cudss")["status"].cpu().numpy()
    except Exception:                                  # noqa: BLE001
        res["fwd_c"] = None
        print("  [err fwd_c B=%d] %s" % (B, last()))
        torch.cuda.empty_cache()
    try:
        def f2c_():
            gn["fb_c"] = float(gpu_grad(D, R, W, "cudss").norm())
        res["fb_c"] = tg(f2c_) * 1e3 / B
    except Exception:                                  # noqa: BLE001
        res["fb_c"] = None
        print("  [err fb_c B=%d] %s" % (B, last()))
        torch.cuda.empty_cache()
    # ---- 旧形态 CPU 伴随段（同节点重测；一次 + 便宜再来一次取最小）----
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
    # ---- dense 段（先清 cuDSS 常驻与 torch 缓存）----
    s.cudss_free(empty_cache=True)
    for tag, run in (("fwd_d", None), ("fb_d", None)):
        try:
            if tag == "fwd_d":
                def fd_():
                    with torch.no_grad():
                        s.solve(D, R, status_machine=True)
                res[tag] = tg(fd_) * 1e3 / B
            else:
                def f2d_():
                    gn["fb_d"] = float(gpu_grad(D, R, W, "dense").norm())
                res[tag] = tg(f2d_) * 1e3 / B
        except torch.OutOfMemoryError:
            res[tag] = None
            torch.cuda.empty_cache()
        except Exception:                              # noqa: BLE001
            res[tag] = None
            print("  [err %s B=%d] %s" % (tag, B, last()))
            torch.cuda.empty_cache()
    fb_cpu_d = None if (res.get("fwd_d") is None or timp_per is None) \
        else res["fwd_d"] + timp_per
    fb_cpu_c = None if (res.get("fwd_c") is None or timp_per is None) \
        else res["fwd_c"] + timp_per
    ROWS.append((B, res.get("fwd_d"), res.get("fwd_c"), timp_per,
                 fb_cpu_d, fb_cpu_c, res.get("fb_d"), res.get("fb_c"),
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
    torch.cuda.empty_cache()

print()
print("=" * 96)
print("§T 汇总（ms/场景）node=%s" % NODE)
print("B     | fwd_d     fwd_c    | implCPU     | 旧f+b_d   旧f+b_c  | "
      "新f+b_d   新f+b_c  | 加速(旧d/新d) 加速(旧c/新c) 加速(旧c/新c最优)")
for (B, fd, fc, ti, od, oc, nd, nc, ng, gcn, gnd, gnc) in ROWS:
    c = lambda x, w=9: ("OOM").rjust(w) if x is None else ("%*.5f" % (w, x))
    r1 = "-" if (od is None or nd is None) else "%.2fx" % (od / nd)
    r2 = "-" if (oc is None or nc is None) else "%.2fx" % (oc / nc)
    best_old = min([x for x in (od, oc) if x is not None], default=None)
    best_new = min([x for x in (nd, nc) if x is not None], default=None)
    r3 = "-" if (best_old is None or best_new is None) \
        else "%.2fx" % (best_old / best_new)
    print("%-5d | %s %s | %s | %s %s | %s %s | %8s %8s %8s"
          % (B, c(fd), c(fc), c(ti, 11), c(od), c(oc), c(nd), c(nc),
             r1, r2, r3))
if FAILS:
    print("FAILS:", FAILS)
print("ADJGPU DONE rc=%d" % (1 if FAILS else 0))
sys.exit(1 if FAILS else 0)

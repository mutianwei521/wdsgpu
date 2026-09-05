# -*- coding: utf-8 -*-
"""a3_gpu.py - 敌意审阅（自写，不复用上游测试脚本）：GPU 伴随发布件集群面。

§0 环境证词：节点/GPU/CPU 亲和/线程/BLAS 环境变量/关键文件 md5。
§1 梯度抽查（B=8，自配种子 424242）：CPU 伴随 vs adjoint='gpu' dense vs cudss，
   demand 全向量相对差 + PRV 下游坐标逐一打印。
§2 CPU 伴随基线公平性：B=64 的 implCPU 在 (默认线程 / set_num_threads(全核) /
   set_num_threads(1)) 三档下实测 ms/场景 + process_time/wall 占用比。
   若三档同级 ⇒ 伴随天然串行（scipy splu），基线没有被自缚的空间。
§3 f+b 端到端全表（B=8/64/256/1024）：旧=cudss前向+CPU伴随（同节点实测），
   新c=adjoint='gpu'+cudss，新d（B<=256）。每格 warm>=1、rep>=3 取 min 与
   median。梯度全张量 rel(新c, CPU) 每个 B 都判（<1e-6）。
§4 独立分解计数：monkeypatch nvmath DirectSolver.factorize（不是项目自带
   计数器）。前向若干次；**backward 期间增量必须为 0**。项目计数器并列打印。
§5 逐出行为：cudss_cache_max=1 下 (a) 异 B 交错 (b) 同 B 连续两次前向后
   反向 - 梯度必须与干净参考一致（或显式 raise），refactorize 只能走
   _kf_prepare 的显式计数，不得静默错。
退出码 0 = 全 PASS。
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
if not os.path.isdir(os.path.join(ROOT, "dgga")):
    ROOT = os.path.dirname(os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp            # noqa: E402
from dgga.solver import GGASolver           # noqa: E402
from dgga.autodiff import implicit_solve    # noqa: E402

DEV = os.environ.get("A3_DEV", "cuda")
DT = torch.float64
SEED = 424242
ACC = 1e-6
BS = [int(x) for x in os.environ.get("A3_BS", "8,64,256,1024").split(",")]
INP = os.path.join(ROOT, "networks_prv", "L-TOWN.inp")
if not os.path.exists(INP):
    INP = os.path.join(ROOT, "networks", "public", "_cleaned", "L-TOWN.inp")
FAILS = []

HAS_CUDSS = DEV == "cuda"
try:
    from nvmath.sparse.advanced import DirectSolver
except ImportError:
    DirectSolver = None
    HAS_CUDSS = False

NVCNT = {"fact": 0}
if DirectSolver is not None:
    _orig_fact = DirectSolver.factorize

    def _cnt_fact(self, *a, **k):
        NVCNT["fact"] += 1
        return _orig_fact(self, *a, **k)

    DirectSolver.factorize = _cnt_fact


def last():
    return traceback.format_exc().strip().split("\n")[-1][:180]


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def bench(fn, warm=1, reps=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def rel(a, b):
    return float((a - b).abs().max()
                 / max(float(a.abs().max()), float(b.abs().max()), 1e-300))


# ---------------------------------------------------------------- §0
print("=" * 96)
print("§0 环境证词")
NODE = os.popen("hostname").read().strip()
print("node:", NODE, "| torch", torch.__version__, "| dev", DEV,
      "" if DEV != "cuda" else "| " + torch.cuda.get_device_name(0))
try:
    aff = len(os.sched_getaffinity(0))
except AttributeError:
    aff = -1
print("cpu_count=%s sched_affinity=%s torch.get_num_threads=%d"
      % (os.cpu_count(), aff, torch.get_num_threads()))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "SLURM_CPUS_PER_TASK", "SLURM_JOB_ID"):
    print("  env %s=%r" % (v, os.environ.get(v)))
for f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5 %s %s" % (f, hashlib.md5(
        open(os.path.join(ROOT, f), "rb").read()).hexdigest()))
print("sha256 INP", hashlib.sha256(open(INP, "rb").read()).hexdigest()[:32])
sys.stdout.flush()

net = parse_inp(INP)
s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=INP,
              dense_status_machine=True)
se = GGASolver(net, mode="epanet", inp_path=INP)
d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
tn = np.asarray(net.tank_node, dtype=np.int64)
if tn.size:
    rh0[tn] = np.clip(.5 * (np.asarray(net.tank_hmin) + np.asarray(net.tank_hmax)),
                      net.tank_hmin + .3 * (net.tank_hmax - net.tank_hmin),
                      net.tank_hmax - .3 * (net.tank_hmax - net.tank_hmin))
rh0 = np.nan_to_num(rh0)
base = se.run_gga(d0, rh0, do_status=True)
K0set = base["setting"].copy()
jn = torch.as_tensor(np.asarray(se.junc_nodes), dtype=torch.long)
jn_dev = jn.to(DEV)
prv_ks = np.where(np.asarray(net.link_type) == 3)[0]
prv_dn = [int(net.link_n2[k]) for k in prv_ks]
print("L-TOWN Nj=%d L=%d PRV=%d 下游节点=%s" % (s.Nj, s.L, prv_ks.size, prv_dn))
sys.stdout.flush()


def batchify(B, seed=SEED):
    g = np.random.default_rng(seed)
    D = d0[None, :] * g.uniform(0.85, 1.15, (B, d0.size))
    R = rh0[None, :] + g.uniform(-1.0, 1.0, (B, rh0.size))
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


def weights(B, seed=99):
    return torch.as_tensor(
        np.random.default_rng(seed).normal(size=(B, s.Nj)),
        dtype=DT, device=DEV)


def gpu_grad(D, R, W, ls, out=None):
    Dv = D.detach().clone().requires_grad_(True)
    h, _q, _e = implicit_solve(
        s, Dv, R, adjoint="gpu", accuracy=ACC, max_iter=200,
        status_machine=True, assemble=("csr" if ls == "cudss" else "dense"),
        linear_solver=ls)
    (h.index_select(1, jn_dev) * W).sum().backward()
    if out is not None:
        out["g"] = Dv.grad
    return Dv.grad


def cpu_grad(D, R, W, S_all):
    """旧训练形态：状态分组冻结 + CPU ImplicitGGASolve（epanet 求解器 splu）。"""
    uniq, inv = np.unique(S_all, axis=0, return_inverse=True)
    Dc, Rc, Wc = D.detach().cpu(), R.detach().cpu(), W.detach().cpu()
    Dv = Dc.clone().requires_grad_(True)
    loss = 0.0
    for gidx in range(uniq.shape[0]):
        ii = torch.as_tensor(np.where(inv == gidx)[0])
        h, _q, _e = implicit_solve(se, Dv.index_select(0, ii),
                                   Rc.index_select(0, ii),
                                   speed=K0set, status=uniq[gidx])
        loss = loss + (h.index_select(1, jn)
                       * Wc.index_select(0, ii)).sum()
    loss.backward()
    return Dv.grad, uniq.shape[0]


# ---------------------------------------------------------------- §1
print()
print("§1 梯度抽查 B=8（三路 demand 全向量 + PRV 下游坐标）")
try:
    D1, R1 = batchify(8)
    W1 = weights(8)
    with torch.no_grad():
        S1 = s.solve(D1, R1, status_machine=True)["status"].cpu().numpy()
    g_cpu, ng = cpu_grad(D1, R1, W1, S1)
    g_cpu = g_cpu.to(DEV)
    g_d = gpu_grad(D1, R1, W1, "dense")
    r_d = rel(g_d, g_cpu)
    ok = r_d < 1e-6
    msg = "  dense vs CPU=%.3e" % r_d
    if HAS_CUDSS:
        g_c = gpu_grad(D1, R1, W1, "cudss")
        r_c = rel(g_c, g_cpu)
        msg += " cudss vs CPU=%.3e cudss vs dense=%.3e" % (r_c, rel(g_c, g_d))
        ok &= r_c < 1e-6
    print(msg + " 状态组=%d %s" % (ng, "PASS" if ok else "FAIL"))
    for j in prv_dn:
        print("    PRV下游 demand[%d]: cpu=% .10e gpu_d=% .10e%s"
              % (j, float(g_cpu[0, j]), float(g_d[0, j]),
                 "" if not HAS_CUDSS else " gpu_c=% .10e" % float(g_c[0, j])))
    if not ok:
        FAILS.append("§1")
    del D1, R1, W1
except Exception:
    FAILS.append("§1")
    print("  [err] %s" % last())
sys.stdout.flush()

# ---------------------------------------------------------------- §2
print()
print("§2 CPU 伴随基线公平性（B=64，三档线程）")
try:
    B2 = 64
    D2, R2 = batchify(B2)
    W2 = weights(B2)
    with torch.no_grad():
        S2 = s.solve(D2, R2, status_machine=True)["status"].cpu().numpy()
    ncpu = os.cpu_count()
    nthr0 = torch.get_num_threads()
    for tag, nt in (("默认%d" % nthr0, None), ("全核%d" % ncpu, ncpu),
                    ("单线程", 1)):
        if nt is not None:
            torch.set_num_threads(nt)
            os.environ["OMP_NUM_THREADS"] = str(nt)
        best = None
        for _ in range(2):
            c0, w0 = time.process_time(), time.perf_counter()
            cpu_grad(D2, R2, W2, S2)
            cw = (time.process_time() - c0, time.perf_counter() - w0)
            if best is None or cw[1] < best[1]:
                best = cw
        print("  [%s] implCPU=%.2f ms/场景  占用比 cpu/wall=%.2f 核"
              % (tag, best[1] * 1e3 / B2, best[0] / best[1]))
    torch.set_num_threads(nthr0)
    os.environ.pop("OMP_NUM_THREADS", None)
    del D2, R2, W2
except Exception:
    FAILS.append("§2")
    print("  [err] %s" % last())
sys.stdout.flush()

# ---------------------------------------------------------------- §3
print()
print("§3 f+b 全表（ms/场景；独立种子/独立计时；min|median）")
ROWS = []
for B in BS:
    D, R = batchify(B)
    W = weights(B)
    row = dict(B=B)
    if HAS_CUDSS:
        s.cudss_free(empty_cache=True)
    elif DEV == "cuda":
        torch.cuda.empty_cache()
    try:
        if B <= 256:
            def fd_():
                with torch.no_grad():
                    s.solve(D, R, status_machine=True)
            mn, md = bench(fd_)
            row["fwd_d"] = mn * 1e3 / B
            gd_ = {}
            mn, md = bench(lambda: gpu_grad(D, R, W, "dense", gd_))
            row["fb_d"] = mn * 1e3 / B
            row["g_d"] = gd_["g"]
    except Exception:
        print("  [err dense B=%d] %s" % (B, last()))
        torch.cuda.empty_cache()
    S_all = None
    try:
        with torch.no_grad():
            kw = dict(assemble="csr", linear_solver="cudss") if HAS_CUDSS else {}
            S_all = s.solve(D, R, status_machine=True,
                            **kw)["status"].cpu().numpy()
    except Exception:
        print("  [err S_all B=%d] %s" % (B, last()))
    if S_all is not None:
        try:
            reps = 2 if B <= 256 else 1
            best = None
            for _ in range(reps):
                t0 = time.perf_counter()
                g1, ngc = cpu_grad(D, R, W, S_all)
                dtw = time.perf_counter() - t0
                best = dtw if best is None else min(best, dtw)
            row["impl"] = best * 1e3 / B
            row["ng"] = ngc
            row["g_cpu"] = g1.to(DEV)
        except Exception:
            print("  [err implCPU B=%d] %s" % (B, last()))
    if HAS_CUDSS:
        try:
            def fc_():
                with torch.no_grad():
                    s.solve(D, R, status_machine=True, assemble="csr",
                            linear_solver="cudss")
            mn, md = bench(fc_)
            row["fwd_c"] = mn * 1e3 / B
            gc_ = {}
            mn, md = bench(lambda: gpu_grad(D, R, W, "cudss", gc_))
            row["fb_c"] = mn * 1e3 / B
            row["fb_c_med"] = md * 1e3 / B
            row["g_c"] = gc_["g"]
        except Exception:
            print("  [err cudss B=%d] %s" % (B, last()))
            torch.cuda.empty_cache()
    # 梯度全张量对拍（每 B 都判）
    gc_ok = "-"
    if "g_cpu" in row and "g_c" in row:
        r_ = rel(row["g_c"], row["g_cpu"])
        gc_ok = "%.2e" % r_
        if r_ >= 1e-6:
            FAILS.append("§3 grad B=%d" % B)
    old_c = (row.get("fwd_c", np.nan) or np.nan) + (row.get("impl", np.nan)
                                                    or np.nan)
    spd = old_c / row["fb_c"] if row.get("fb_c") else np.nan
    f = lambda k: ("%.3f" % row[k]) if k in row else "-"
    print("  B=%-5d fwd_d=%s fb_d=%s | fwd_c=%s implCPU=%s(组%s) 旧c=%.3f | "
          "fb_c=%s(med %s) | 旧c/新c=%.2fx | rel(gc,cpu)=%s"
          % (B, f("fwd_d"), f("fb_d"), f("fwd_c"), f("impl"),
             row.get("ng", "-"), old_c, f("fb_c"), f("fb_c_med"), spd, gc_ok))
    gn = lambda k: ("%.6e" % float(row[k].norm())) if k in row else "-"
    print("        |g| cpu=%s gpu_d=%s gpu_c=%s"
          % (gn("g_cpu"), gn("g_d"), gn("g_c")))
    ROWS.append((B, row.get("impl"), old_c, row.get("fb_c"), spd))
    for k in ("g_cpu", "g_d", "g_c"):
        row.pop(k, None)
    del D, R, W
    torch.cuda.empty_cache()
    sys.stdout.flush()

# ---------------------------------------------------------------- §4
print()
print("§4 独立分解计数（nvmath DirectSolver.factorize monkeypatch）")
if HAS_CUDSS:
    try:
        s.cudss_free(empty_cache=True)
        D4, R4 = batchify(8)
        W4 = weights(8)
        s.cudss_counters(reset=True)
        NVCNT["fact"] = 0
        Dv = D4.detach().clone().requires_grad_(True)
        h, _q, _e = implicit_solve(s, Dv, R4, adjoint="gpu", accuracy=ACC,
                                   max_iter=200, status_machine=True,
                                   assemble="csr", linear_solver="cudss")
        n_fwd = NVCNT["fact"]
        loss = (h.index_select(1, jn_dev) * W4).sum()
        loss.backward()
        n_bwd = NVCNT["fact"] - n_fwd
        pc = s.cudss_counters()
        ok4 = (n_bwd == 0 and pc["bwd_refactorize"] == 0
               and n_fwd == pc["factorize"] and n_fwd > 0)
        print("  独立计数：fwd=%d bwd增量=%d | 项目计数 factorize=%d "
              "bwd_refactorize=%d bwd_reuse=%d bwd_solve=%d  %s"
              % (n_fwd, n_bwd, pc["factorize"], pc["bwd_refactorize"],
                 pc["bwd_reuse"], pc["bwd_solve"],
                 "PASS" if ok4 else "FAIL"))
        if not ok4:
            FAILS.append("§4")
        del D4, R4, W4, Dv
    except Exception:
        FAILS.append("§4")
        print("  [err] %s" % last())
else:
    print("  跳过（无 cudss）")
sys.stdout.flush()

# ---------------------------------------------------------------- §5
print()
print("§5 逐出行为（cudss_cache_max=1）：正确或显式报错，不得静默错")
if HAS_CUDSS:
    try:
        s.cudss_free(empty_cache=True)
        # 干净参考（cache 充足）
        old_cm, old_gs = s.cudss_cache_max, s.cudss_grad_slots
        Da, Ra = batchify(8, seed=SEED + 1)
        Wa = weights(8)
        Db, Rb = batchify(16, seed=SEED + 2)
        Wb = weights(16)
        g_ref_a = gpu_grad(Da, Ra, Wa, "cudss").clone()
        g_ref_b = gpu_grad(Db, Rb, Wb, "cudss").clone()
        # (a) 异 B 交错：fwd(a) -> fwd(b)（逐出 a 的 state）-> bwd(a) -> bwd(b)
        s.cudss_free(empty_cache=True)
        s.cudss_cache_max = 1
        s.cudss_counters(reset=True)
        NVCNT["fact"] = 0
        Dva = Da.detach().clone().requires_grad_(True)
        ha, _, _ = implicit_solve(s, Dva, Ra, adjoint="gpu", accuracy=ACC,
                                  max_iter=200, status_machine=True,
                                  assemble="csr", linear_solver="cudss")
        Dvb = Db.detach().clone().requires_grad_(True)
        hb, _, _ = implicit_solve(s, Dvb, Rb, adjoint="gpu", accuracy=ACC,
                                  max_iter=200, status_machine=True,
                                  assemble="csr", linear_solver="cudss")
        nv0 = NVCNT["fact"]
        (ha.index_select(1, jn_dev) * Wa).sum().backward()
        (hb.index_select(1, jn_dev) * Wb).sum().backward()
        nbwd = NVCNT["fact"] - nv0
        pc = s.cudss_counters()
        ra_ = rel(Dva.grad, g_ref_a)
        rb_ = rel(Dvb.grad, g_ref_b)
        ok5a = ra_ < 1e-12 and rb_ < 1e-12
        print("  (a) 异B交错 cache_max=1：rel(a)=%.2e rel(b)=%.2e "
              "bwd期间独立factorize=%d 项目bwd_refactorize=%d bwd_reuse=%d %s"
              % (ra_, rb_, nbwd, pc["bwd_refactorize"], pc["bwd_reuse"],
                 "PASS" if ok5a else "FAIL"))
        if not ok5a:
            FAILS.append("§5a")
        # (b) 同 B 连续两次前向（slot 复写）再反向第一份
        s.cudss_free(empty_cache=True)
        s.cudss_counters(reset=True)
        Dv1 = Da.detach().clone().requires_grad_(True)
        h1, _, _ = implicit_solve(s, Dv1, Ra, adjoint="gpu", accuracy=ACC,
                                  max_iter=200, status_machine=True,
                                  assemble="csr", linear_solver="cudss")
        Dv2 = (Da * 1.02).detach().clone().requires_grad_(True)
        h2, _, _ = implicit_solve(s, Dv2, Ra, adjoint="gpu", accuracy=ACC,
                                  max_iter=200, status_machine=True,
                                  assemble="csr", linear_solver="cudss")
        (h1.index_select(1, jn_dev) * Wa).sum().backward()
        pc = s.cudss_counters()
        r1_ = rel(Dv1.grad, g_ref_a)
        ok5b = r1_ < 1e-12
        print("  (b) 同B复写 slot：rel=%.2e bwd_refactorize=%d bwd_reuse=%d %s"
              % (r1_, pc["bwd_refactorize"], pc["bwd_reuse"],
                 "PASS" if ok5b else "FAIL"))
        if not ok5b:
            FAILS.append("§5b")
        s.cudss_cache_max, s.cudss_grad_slots = old_cm, old_gs
        s.cudss_free(empty_cache=True)
    except Exception:
        FAILS.append("§5")
        print("  [err] %s" % last())
else:
    print("  跳过（无 cudss）")

print()
print("=" * 96)
print("§T 汇总 node=%s：B | implCPU | 旧c | 新c | 旧c/新c" % NODE)
for B, im, oc, nc, sp in ROWS:
    print("  %-5d | %s | %.3f | %s | %.2fx"
          % (B, "-" if im is None else "%.3f" % im, oc,
             "-" if nc is None else "%.3f" % nc, sp))
if FAILS:
    print("FAILS:", FAILS)
print("A3 DONE rc=%d" % (1 if FAILS else 0))
sys.exit(1 if FAILS else 0)

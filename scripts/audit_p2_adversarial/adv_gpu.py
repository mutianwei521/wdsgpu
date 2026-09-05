# -*- coding: utf-8 -*-
"""P2 对抗性验收（独立脚本，不复用 scripts/p2_cudss/ 任何代码）。

[A] 等价性 + **调用计数**：cudss 是不是少做了精化 / 少迭代了一轮
[B] 计时审计：中位数 + 最好值，plan 计数在计时区必须为 0，plan 摊销门槛
[C] 稠密基线公不公平：逐元素残差 vs bmm 残差的微基准（还原"若基线不自缚"）
[D] CPU f64 参考：谁离真值近
[E] 梯度守卫 / autodiff 透传
"""
import gc
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                       # noqa: E402
from dgga.solver import GGASolver                      # noqa: E402

DT = torch.float64
NETDIR = os.path.join(ROOT, "p2nets")
NETS = [("Net1", "Net1.inp"), ("Anytown", "Anytown.inp"), ("Hanoi", "Hanoi.inp"),
        ("Net2", "Net2.inp"), ("Fossolo", "Fossolo_poly1.inp"),
        ("Pescara", "Pescara.inp"), ("Net3", "Net3.inp"), ("Modena", "Modena.inp"),
        ("City_D", "City_D.inp"), ("ky4", "ky4.inp")]

# ---------------- 计数器（可开关；计时区必须关掉，免得给任一方加税）----------
CNT = dict(plan=0, fact=0, dsolve=0, chol=0, cholsolve=0)
ON = [False]
from nvmath.sparse.advanced import DirectSolver        # noqa: E402
_op, _of, _od = DirectSolver.plan, DirectSolver.factorize, DirectSolver.solve
_oc, _ocs = torch.linalg.cholesky, torch.cholesky_solve


def _bump(k):
    if ON[0]:
        CNT[k] += 1


DirectSolver.plan = lambda self, **k: (_bump("plan"), _op(self, **k))[1]
DirectSolver.factorize = lambda self, **k: (_bump("fact"), _of(self, **k))[1]
DirectSolver.solve = lambda self, **k: (_bump("dsolve"), _od(self, **k))[1]
torch.linalg.cholesky = lambda *a, **k: (_bump("chol"), _oc(*a, **k))[1]
torch.cholesky_solve = lambda *a, **k: (_bump("cholsolve"), _ocs(*a, **k))[1]
# plan 专用哨兵：永远开着（开销 = 每 plan 一次自增，与计时无关）
PLAN_SEEN = [0]
_op2 = DirectSolver.plan
DirectSolver.plan = lambda self, **k: (PLAN_SEEN.__setitem__(0, PLAN_SEEN[0] + 1),
                                       _op2(self, **k))[1]

# A 捕获钩子（[C] 用）
GRAB = {"A": None, "on": False}
_oc2 = torch.linalg.cholesky


def _chol_grab(*a, **k):
    if GRAB["on"]:
        GRAB["A"] = a[0].detach().clone()
    return _oc2(*a, **k)


torch.linalg.cholesky = _chol_grab


def last():
    return traceback.format_exc().strip().split("\n")[-1][:130]


def mk(stem, fn, dev="cuda"):
    f = os.path.join(NETDIR, fn)
    net = parse_inp(f)
    s = GGASolver(net, device=dev, dtype=DT, mode="dense", inp_path=f,
                  dense_tank_bound_check=False)
    return net, s


def bc(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:                       # 水池头取区间中点（避开贴边守卫）
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, rh


def scen(d, rh, B, seed, dev="cuda"):
    """我自己的场景发生器（与上游 batchify 无关：另一套分布 + 另一套种子）。"""
    g = np.random.default_rng(20260822 + seed)
    D = d[None, :] * g.lognormal(0.0, 0.22, (B, d.size))
    R = rh[None, :] + g.normal(0.0, 1.5, (B, rh.size))
    R = np.where(np.isnan(rh)[None, :], np.nan, R)
    return (torch.as_tensor(D, dtype=DT, device=dev),
            torch.as_tensor(R, dtype=DT, device=dev))


def emitter_ke(net, d, frac=0.15, share=0.03, seed=7):
    """在 frac 比例的 junction 上放 emitter，使总喷射流量 ≈ share×总需水。"""
    g = np.random.default_rng(seed)
    jm = np.where(np.asarray(net.node_type) == 0)[0]
    if jm.size == 0:
        return None
    pick = g.choice(jm, size=max(1, int(round(frac * jm.size))), replace=False)
    tot = float(np.abs(d).sum())
    if tot <= 0:
        return None
    ke = np.zeros(net.N)
    ke[pick] = share * tot / pick.size / np.sqrt(40.0)   # p≈40ft, qexp=0.5
    return ke


def free_states(s):
    for st in list(s._cudss_cache.values()):
        try:
            st["solver"].free()
        except Exception:                                # noqa: BLE001
            pass
    s._cudss_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()


print(torch.cuda.get_device_name(0), "| torch", torch.__version__,
      "| cuda", torch.version.cuda, "| cpus", os.cpu_count())
import nvmath                                            # noqa: E402
print("nvmath", nvmath.__version__, "| CUBLAS_WORKSPACE_CONFIG =",
      os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
print("dgga/solver.py mtime", time.ctime(os.path.getmtime(
    os.path.join(ROOT, "dgga", "solver.py"))))

# =====================================================================
# [A] 等价性 + 调用计数
# =====================================================================
print("\n" + "=" * 118)
print("[A] 等价性 + 线性解调用计数（deterministic=True）。判据：iters 逐项相等；"
      "dense chol==itmax & cholesky_solve==3*itmax；cudss fact==itmax & solve==3*itmax")
torch.use_deterministic_algorithms(True)
print("%-8s %5s %3s %5s | %6s %6s | %10s %10s %10s | %s" %
      ("net", "B", "em", "itmax", "d_ch/d_cs", "c_fa/c_so", "max|dH|", "max|dQ|",
       "max|dE|", "iters逐项相等 / 备注"))
for stem, fn in NETS:
    try:
        net, s = mk(stem, fn)
    except Exception:                                    # noqa: BLE001
        print("[skip] %s %s" % (stem, last()))
        continue
    d, rh = bc(net)
    ke = emitter_ke(net, d)
    for B in (1, 8, 64, 256):
        for emflag in (False, True):
            try:
                D, R = scen(d, rh, B, B * 7 + int(emflag))
                kt = None if not emflag else torch.as_tensor(
                    ke, dtype=DT, device="cuda")
                if emflag and kt is None:
                    continue
                s.cudss_plan(B)                     # 预热（不计入等价性）
                ON[0] = True
                CNT.update(plan=0, fact=0, dsolve=0, chol=0, cholsolve=0)
                a = s.solve(D, R, ke_int=kt)
                dch, dcs = CNT["chol"], CNT["cholsolve"]
                CNT.update(fact=0, dsolve=0)
                b = s.solve(D, R, ke_int=kt, assemble="csr", linear_solver="cudss")
                cfa, cso = CNT["fact"], CNT["dsolve"]
                ON[0] = False
                itmax = int(a["iters"].max())
                same = bool((a["iters"] == b["iters"]).all())
                emfrac = float((b["emitter_cfs"].abs().sum(1) /
                                D.abs().sum(1)).mean()) if emflag else 0.0
                note = "OK" if same else "**ITERS 不等** d=%s c=%s" % (
                    a["iters"].tolist()[:8], b["iters"].tolist()[:8])
                if emflag:
                    note += " | emitter占比%.3f" % emfrac
                if not (dch == itmax and dcs == 3 * itmax
                        and cfa == itmax and cso == 3 * itmax):
                    note += " | **调用计数异常**"
                print("%-8s %5d %3d %5d | %6s %6s | %10.3e %10.3e %10.3e | %s" %
                      (stem, B, int(emflag), itmax, "%d/%d" % (dch, dcs),
                       "%d/%d" % (cfa, cso),
                       float((a["head_ft"] - b["head_ft"]).abs().max()),
                       float((a["flow_cfs"] - b["flow_cfs"]).abs().max()),
                       float((a["emitter_cfs"] - b["emitter_cfs"]).abs().max()),
                       note))
            except Exception:                            # noqa: BLE001
                ON[0] = False
                print("%-8s %5d %3d | %s" % (stem, B, int(emflag), last()))
        free_states(s)
    del s, net
    gc.collect()
    torch.cuda.empty_cache()
torch.use_deterministic_algorithms(False)

# =====================================================================
# [B] 计时审计
# =====================================================================
print("\n" + "=" * 118)
print("[B] 计时（determinism 关；warm=3，reps=9，报中位数与最好值；"
      "计时区内 plan 计数必须 0）")
print("%-8s %5s %5s | %9s %9s | %9s %9s | %6s %6s | %8s %7s | %s" %
      ("net", "B", "itmax", "dense中位", "dense最好", "cudss中位", "cudss最好",
       "中位x", "最好x", "plan ms", "摊销解", "计时区plan"))
SPEED = {}
for stem, fn in NETS:
    try:
        net, s = mk(stem, fn)
    except Exception:                                    # noqa: BLE001
        continue
    d, rh = bc(net)
    for B in (1, 8, 64, 256):
        try:
            D, R = scen(d, rh, B, 900 + B)
            itmax = int(s.solve(D, R)["iters"].max())
            plan_ms = s.cudss_plan(B)

            def run_d():
                s.solve(D, R)

            def run_c():
                s.solve(D, R, assemble="csr", linear_solver="cudss")

            def timeit(fn_):
                for _ in range(3):
                    fn_()
                torch.cuda.synchronize()
                ts = []
                for _ in range(9):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    fn_()
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                ts.sort()
                return ts[len(ts) // 2] * 1e3 / B, ts[0] * 1e3 / B

            p0 = PLAN_SEEN[0]
            dmed, dbest = timeit(run_d)
            cmed, cbest = timeit(run_c)
            nplan = PLAN_SEEN[0] - p0
            # plan 摊销：需要多少次同批量求解才能把 plan 赚回来
            save = (dmed - cmed) * B                     # ms/次求解
            amort = plan_ms / save if save > 0 else float("inf")
            SPEED[(stem, B)] = (dmed, cmed, itmax)
            print("%-8s %5d %5d | %9.5f %9.5f | %9.5f %9.5f | %5.2fx %5.2fx | "
                  "%8.1f %7s | %d" %
                  (stem, B, itmax, dmed, dbest, cmed, cbest, dmed / cmed,
                   dbest / cbest, plan_ms,
                   ("%.1f" % amort) if np.isfinite(amort) else "never", nplan))
        except Exception:                                # noqa: BLE001
            print("%-8s %5d | %s" % (stem, B, last()))
        free_states(s)
    del s, net
    gc.collect()
    torch.cuda.empty_cache()

# =====================================================================
# [C] 稠密基线是否被自缚（逐元素残差 vs bmm 残差）
# =====================================================================
print("\n" + "=" * 118)
print("[C] 稠密基线公平性微基准（B=256，取最后一次 cholesky 见到的真实 A）：")
print("    dense 每轮线代 = chol + 3×cholesky_solve + 2×残差；残差现用"
      "(A*H^T).sum(-1)（为批不变性），bmm 更快但破位级一致")
print("%-8s %5s | %8s %8s %8s %8s | %9s %9s | %s" %
      ("net", "Nj", "chol", "cs×1", "resElem", "resBmm", "轮elem", "轮bmm",
       "若基线用bmm的整解倍数（按[B]中位数折算）"))
for stem, fn in NETS:
    if stem not in ("Net3", "Modena", "City_D", "ky4"):
        continue
    try:
        net, s = mk(stem, fn)
        d, rh = bc(net)
        B = 256
        D, R = scen(d, rh, B, 55)
        GRAB["on"] = True
        s.solve(D, R)
        GRAB["on"] = False
        A = GRAB["A"]
        Nj = A.shape[-1]
        Fc = torch.randn(B, Nj, 1, dtype=DT, device="cuda")
        ch = _oc(A)
        Hj = _ocs(Fc, ch)

        def tt(f, n=7):
            for _ in range(3):
                f()
            torch.cuda.synchronize()
            ts = []
            for _ in range(n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                f()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            ts.sort()
            return ts[len(ts) // 2] * 1e3

        t_ch = tt(lambda: _oc(A))
        t_cs = tt(lambda: _ocs(Fc, ch))
        t_re = tt(lambda: (A * Hj.transpose(-2, -1)).sum(-1, keepdim=True))
        t_rb = tt(lambda: torch.bmm(A, Hj))
        rnd_e = t_ch + 3 * t_cs + 2 * t_re
        rnd_b = t_ch + 3 * t_cs + 2 * t_rb
        key = (stem, 256)
        txt = "-"
        if key in SPEED:
            dmed, cmed, itmax = SPEED[key]
            # 整解(ms/场景) = iters×(装配 + 线代)/B；把线代那块换成 bmm 版
            dense_total = dmed * B / itmax               # ms/轮
            asm = dense_total - rnd_e                    # 反推装配+其余
            newd = (asm + rnd_b) * itmax / B
            txt = "dense_bmm=%.5f ms/场景 → %.2fx（原 %.2fx）" % (
                newd, newd / cmed, dmed / cmed)
        print("%-8s %5d | %8.4f %8.4f %8.4f %8.4f | %9.4f %9.4f | %s" %
              (stem, Nj, t_ch, t_cs, t_re, t_rb, rnd_e, rnd_b, txt))
        del A, ch, Hj, Fc, s, net
        GRAB["A"] = None
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:                                    # noqa: BLE001
        GRAB["on"] = False
        print("%-8s | %s" % (stem, last()))
        gc.collect()
        torch.cuda.empty_cache()

# =====================================================================
# [D] CPU f64 参考（谁离"真值"近）
# =====================================================================
print("\n" + "=" * 118)
print("[D] 与 CPU f64 dense 基准比（B=8，同一批场景）：|cudss-cpu| vs |gpu_dense-cpu|")
print("%-8s | %11s %11s | %s" % ("net", "|cudss-cpu|", "|gpuden-cpu|", "判定"))
for stem, fn in NETS:
    try:
        net, s = mk(stem, fn)
        _, sc = mk(stem, fn, dev="cpu")
        d, rh = bc(net)
        B = 8
        D, R = scen(d, rh, B, 4242)
        ref = sc.solve(D.cpu(), R.cpu())["head_ft"]
        a = s.solve(D, R)["head_ft"].cpu()
        b = s.solve(D, R, assemble="csr", linear_solver="cudss")["head_ft"].cpu()
        ea = float((b - ref).abs().max())
        eb = float((a - ref).abs().max())
        print("%-8s | %11.3e %11.3e | %s" %
              (stem, ea, eb, "cudss更准" if ea < eb * 0.9 else
               ("dense更准" if eb < ea * 0.9 else "同量级")))
        free_states(s)
        del s, sc, net
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:                                    # noqa: BLE001
        print("%-8s | %s" % (stem, last()))
        gc.collect()
        torch.cuda.empty_cache()

# =====================================================================
# [E] 梯度守卫 + autodiff 透传
# =====================================================================
print("\n" + "=" * 118)
print("[E] 梯度守卫（GPU）")
from dgga.autodiff import solve_unrolled, solve_polished    # noqa: E402
try:
    net, s = mk("Modena", "Modena.inp")
    d, rh = bc(net)
    D, R = scen(d, rh, 4, 11)
    # E1 solve() 直接要梯度
    try:
        s.solve(D.clone().requires_grad_(True), R, assemble="csr",
                linear_solver="cudss")
        print("  [FAIL] solve() requires_grad + cudss 未 raise")
    except NotImplementedError as e:
        print("  [OK] solve() requires_grad → NotImplementedError: %s" % str(e)[:70])
    # E2 只有 res_head 需要梯度（只进 F、不进 A）
    try:
        s.solve(D, R.clone().requires_grad_(True), assemble="csr",
                linear_solver="cudss")
        print("  [FAIL] 只有 res_head requires_grad 时未 raise（F 需梯度也算断图）")
    except NotImplementedError:
        print("  [OK] 只有 res_head requires_grad → 也 raise")
    # E3 no_grad 下不 raise（合法纯前向）
    with torch.no_grad():
        s.solve(D.clone().requires_grad_(True), R, assemble="csr",
                linear_solver="cudss")
    print("  [OK] no_grad 下 requires_grad 输入不 raise（纯前向合法）")
    # E4 unrolled 要梯度
    try:
        solve_unrolled(s, D.clone().requires_grad_(True), R, K=6,
                       assemble="csr", linear_solver="cudss")
        print("  [FAIL] solve_unrolled requires_grad + cudss 未 raise")
    except NotImplementedError:
        print("  [OK] solve_unrolled requires_grad → raise")
    # E5 输出是否真的断了图（no_grad 外、输入不要梯度时结果 requires_grad 应为 False）
    o = s.solve(D, R, assemble="csr", linear_solver="cudss")
    print("  [info] cudss 输出 head.requires_grad =", bool(o["head_ft"].requires_grad))
    # E6 polished：cudss 前向 + numpy 抛光，梯度应与 dense 一致
    pa = solve_polished(s, d, rh, max_iter=60)
    pc = solve_polished(s, d, rh, max_iter=60, assemble="csr",
                        linear_solver="cudss")
    print("  [OK] polished cudss vs dense: max|ΔH|=%.3e ft max|Δq|=%.3e cfs" %
          (float(np.abs(pa["head"] - pc["head"]).max()),
           float(np.abs(pa["q"] - pc["q"]).max())))
    free_states(s)
except Exception:                                        # noqa: BLE001
    print("  [E] %s" % last())

print("\nADV_GPU 结束")

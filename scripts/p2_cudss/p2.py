# -*- coding: utf-8 -*-
"""P2 验收：cuDSS 前向接入（sparse_gpu_plan.md §1b）。

  §1 正确性  cudss vs dense（torch.use_deterministic_algorithms(True)）
  §2 加速    B=1/64/256 整解 ms/场景 + 相对 dense 的倍数
  §3 显存    整条 solve() 的 CUDA 峰值（torch 分配器）+ 设备级占用
  §4 plan    plan 只做一次的证据（plan/factorize 调用计数）
  §5 autodiff 入口透传
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
from dgga.autodiff import solve_unrolled             # noqa: E402

DEV = "cuda"
DT = torch.float64
NETDIR = os.path.join(ROOT, "p2nets")

from nvmath.sparse.advanced import DirectSolver      # noqa: E402
_CNT = {"plan": 0, "fact": 0, "solve": 0}
_op, _of, _os_ = DirectSolver.plan, DirectSolver.factorize, DirectSolver.solve
DirectSolver.plan = lambda self, **k: (_CNT.__setitem__("plan", _CNT["plan"] + 1),
                                       _op(self, **k))[1]
DirectSolver.factorize = lambda self, **k: (_CNT.__setitem__("fact", _CNT["fact"] + 1),
                                            _of(self, **k))[1]
DirectSolver.solve = lambda self, **k: (_CNT.__setitem__("solve", _CNT["solve"] + 1),
                                        _os_(self, **k))[1]

NETS = [("Net1", "Net1.inp"), ("Anytown", "Anytown.inp"), ("Hanoi", "Hanoi.inp"),
        ("Net2", "Net2.inp"), ("Fossolo", "Fossolo_poly1.inp"),
        ("Pescara", "Pescara.inp"), ("Net3", "Net3.inp"), ("Modena", "Modena.inp"),
        ("City_D", "City_D.inp"), ("ky4", "ky4.inp")]
NMAP = dict(NETS)


def boundary(net):
    """(d, rh)。水池头取 [hmin,hmax] 中点并夹到 30%~70% 区间，避开贴边守卫。"""
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


def tg(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def dev_used_mib():
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


def last(exc=None):
    return traceback.format_exc().strip().split("\n")[-1][:100]


print(torch.cuda.get_device_name(0), "| torch", torch.__version__,
      "| cuda", torch.version.cuda)
import nvmath                                        # noqa: E402
print("nvmath", nvmath.__version__, "| cpus", os.cpu_count(),
      "| CUBLAS_WORKSPACE_CONFIG =", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

CACHE = {}
for stem, fn in NETS:
    try:
        CACHE[stem] = mk(stem)
    except Exception:                                # noqa: BLE001
        print(f"[skip] {stem}: {last()}")

# ---- 确定性探针（P1 审计列的 P2 前置）----
DET = False
try:
    torch.use_deterministic_algorithms(True)
    net, s = CACHE["Net3"]
    d, rh = boundary(net)
    D, R = batchify(d, rh, 4, 1)
    s.solve(D, R)
    s.solve(D, R, assemble="csr", linear_solver="cudss")
    DET = True
    print("确定性探针：use_deterministic_algorithms(True) 下两条通路都跑通 ✓")
except Exception:                                    # noqa: BLE001
    torch.use_deterministic_algorithms(False)
    print("确定性探针：**严格确定性模式下抛错** ->", last())
    print("   改用 warn_only=True 继续（会在报告中如实标注）")
    torch.use_deterministic_algorithms(True, warn_only=True)
    DET = "warn_only"

# =====================================================================
print("\n" + "=" * 104)
print(f"§1 正确性：cudss vs dense（deterministic={DET}）")
print("%-9s %6s %7s %5s | %11s %11s %11s | %-7s %s" %
      ("net", "Nj", "nnz", "B", "max|ΔH| ft", "max|ΔQ| cfs", "max|ΔE| cfs",
       "iters=", "备注"))
for stem, _fn in NETS:
    if stem not in CACHE:
        continue
    net, s = CACHE[stem]
    d, rh = boundary(net)
    for B in (1, 8, 64):
        try:
            D, R = batchify(d, rh, B, 1234 + B)
            a = s.solve(D, R)
            b = s.solve(D, R, assemble="csr", linear_solver="cudss")
            # dense 通路自身重跑的抖动，作为判定 cudss 偏差归属的对照
            a2 = s.solve(D, R)
            dH = float((a["head_ft"] - b["head_ft"]).abs().max())
            dQ = float((a["flow_cfs"] - b["flow_cfs"]).abs().max())
            dE = float((a["emitter_cfs"] - b["emitter_cfs"]).abs().max())
            rep = float((a["head_ft"] - a2["head_ft"]).abs().max())
            same = bool((a["iters"] == b["iters"]).all())
            note = "dense自重跑 %.1e" % rep
            if not same:
                note += " | dense=%s cudss=%s" % (a["iters"].tolist()[:6],
                                                  b["iters"].tolist()[:6])
            print("%-9s %6d %7d %5d | %11.3e %11.3e %11.3e | %-7s %s" %
                  (stem, s.Nj, s.A_csr_nnz, B, dH, dQ, dE, str(same), note))
        except Exception:                            # noqa: BLE001
            print("%-9s %6d %7d %5d | %s" % (stem, s.Nj, s.A_csr_nnz, B, last()))
    torch.cuda.empty_cache()

print("\n§1b 精化步数敏感性（refine=0/1/2 时 cudss 与 dense 的 max|ΔH|）")
for stem in ("Modena", "City_D", "ky4"):
    if stem not in CACHE:
        continue
    net, s = CACHE[stem]
    d, rh = boundary(net)
    D, R = batchify(d, rh, 8, 99)
    ref = s.solve(D, R)
    orig = GGASolver._cudss_solve
    for r in (0, 1, 2):
        try:
            GGASolver._cudss_solve = (lambda self, data, F, B, _r=r, _o=orig:
                                      _o(self, data, F, B, refine=_r))
            o = s.solve(D, R, assemble="csr", linear_solver="cudss")
            print("   %-8s refine=%d  max|ΔH|=%.3e ft  iters相等=%s" %
                  (stem, r, float((ref["head_ft"] - o["head_ft"]).abs().max()),
                   bool((ref["iters"] == o["iters"]).all())))
        except Exception:                            # noqa: BLE001
            print("   %-8s refine=%d  %s" % (stem, r, last()))
    GGASolver._cudss_solve = orig

# =====================================================================
torch.use_deterministic_algorithms(False)
print("\n" + "=" * 104)
print("§2 加速：整条 solve() 的 ms/场景（best of 5，含装配+线代+newflows；"
      "determinism 关，两条通路同条件）")
print("%-9s %6s %5s | %6s | %12s %12s | %8s | %10s" %
      ("net", "Nj", "B", "iters", "dense ms/sc", "cudss ms/sc", "倍数", "plan ms"))
for stem, _fn in NETS:
    if stem not in CACHE:
        continue
    net, s = CACHE[stem]
    d, rh = boundary(net)
    for B in (1, 64, 256):
        try:
            D, R = batchify(d, rh, B, 77 + B)
            it = int(s.solve(D, R)["iters"].max())
            plan_ms = s.cudss_plan(B)
            td = tg(lambda: s.solve(D, R)) / B * 1e3
            tc = tg(lambda: s.solve(D, R, assemble="csr",
                                    linear_solver="cudss")) / B * 1e3
            print("%-9s %6d %5d | %6d | %12.5f %12.5f | %7.2fx | %10.2f" %
                  (stem, s.Nj, B, it, td, tc, td / tc, plan_ms))
        except Exception:                            # noqa: BLE001
            print("%-9s %6d %5d | %s" % (stem, s.Nj, B, last()))
        torch.cuda.empty_cache()

# =====================================================================
print("\n" + "=" * 104)
print("§3 显存：整条 solve() 的 CUDA 峰值（torch 分配器，预热后重测）")
print("   cuDSS 的内部分解缓冲不走 torch 分配器 ⇒ 另列该进程的设备级占用（总-可用）")
print("%-9s %6s %5s | %12s %12s %8s | %12s %12s" %
      ("net", "Nj", "B", "dense MiB", "cudss MiB", "降幅", "dev dense", "dev cudss"))
for stem, _fn in NETS:
    if stem not in CACHE:
        continue
    for B in (64, 256):
        pk_d = dev_d = pk_c = dev_c = float("nan")
        try:
            net, s = mk(stem)
            d, rh = boundary(net)
            D, R = batchify(d, rh, B, 5 + B)
            torch.cuda.empty_cache()
            s.solve(D, R)
            torch.cuda.reset_peak_memory_stats()
            s.solve(D, R)
            pk_d = torch.cuda.max_memory_allocated() / 2 ** 20
            dev_d = dev_used_mib()
            del s, net, D, R
            torch.cuda.empty_cache()
        except Exception:                            # noqa: BLE001
            print("%-9s %5d dense | %s" % (stem, B, last()))
            torch.cuda.empty_cache()
        try:
            net, s = mk(stem)
            d, rh = boundary(net)
            D, R = batchify(d, rh, B, 5 + B)
            torch.cuda.empty_cache()
            s.solve(D, R, assemble="csr", linear_solver="cudss")
            torch.cuda.reset_peak_memory_stats()
            s.solve(D, R, assemble="csr", linear_solver="cudss")
            pk_c = torch.cuda.max_memory_allocated() / 2 ** 20
            dev_c = dev_used_mib()
            del s, net, D, R
            torch.cuda.empty_cache()
        except Exception:                            # noqa: BLE001
            print("%-9s %5d cudss | %s" % (stem, B, last()))
            torch.cuda.empty_cache()
        print("%-9s %6s %5d | %12.2f %12.2f %7.2fx | %12.1f %12.1f" %
              (stem, "-", B, pk_d, pk_c, pk_d / max(pk_c, 1e-9), dev_d, dev_c))

# =====================================================================
print("\n" + "=" * 104)
print("§4 plan 一次性：首解后 plan 计数应停在 1，factorize 计数 == 牛顿迭代数，"
      "第二次 solve 不再 plan")
for stem in ("Modena", "City_D", "ky4"):
    if stem not in CACHE:
        continue
    try:
        net, s = mk(stem)
        d, rh = boundary(net)
        for B in (1, 64, 256):
            D, R = batchify(d, rh, B, 3)
            _CNT.update(plan=0, fact=0, solve=0)
            pl = s.cudss_plan(B)
            n1 = _CNT["plan"]
            o1 = s.solve(D, R, assemble="csr", linear_solver="cudss")
            it1 = int(o1["iters"].max())
            n2, f2, v2 = _CNT["plan"], _CNT["fact"], _CNT["solve"]
            s.solve(D, R, assemble="csr", linear_solver="cudss")
            n3, f3 = _CNT["plan"], _CNT["fact"]
            stt = [v for k, v in s._cudss_cache.items() if k[0] == B][0]
            print("   %-8s B=%-4d plan_ms=%9.2f | plan 计数 预热后=%d 首解后=%d "
                  "二解后=%d | fact 首解=%d(iters=%d) 二解累计=%d | solve 首解=%d "
                  "| values 别名=%s"
                  % (stem, B, pl, n1, n2, n3, f2, it1, f3, v2, stt["alias"]))
        del s, net
        torch.cuda.empty_cache()
    except Exception:                                # noqa: BLE001
        print("   %-8s %s" % (stem, last()))
        torch.cuda.empty_cache()

# =====================================================================
print("\n" + "=" * 104)
print("§5 autodiff 入口透传（solve_unrolled / solve_polished）")
from dgga.autodiff import solve_polished             # noqa: E402
for stem in ("Modena", "ky4"):
    if stem not in CACHE:
        continue
    try:
        net, s = CACHE[stem]
        d, rh = boundary(net)
        D, R = batchify(d, rh, 8, 21)
        K = 14
        a = solve_unrolled(s, D, R, K=K)
        b = solve_unrolled(s, D, R, K=K, assemble="csr")
        print("   %-8s unrolled csr  vs dense: max|ΔH|=%.3e max|ΔQ|=%.3e" %
              (stem, float((a["head_ft"] - b["head_ft"]).abs().max()),
               float((a["flow_cfs"] - b["flow_cfs"]).abs().max())))
        with torch.no_grad():
            c = solve_unrolled(s, D, R, K=K, assemble="csr", linear_solver="cudss")
        print("   %-8s unrolled cudss vs dense: max|ΔH|=%.3e max|ΔQ|=%.3e" %
              (stem, float((a["head_ft"] - c["head_ft"]).abs().max()),
               float((a["flow_cfs"] - c["flow_cfs"]).abs().max())))
        try:
            solve_unrolled(s, D.clone().requires_grad_(True), R, K=K,
                           assemble="csr", linear_solver="cudss")
            print("   [FAIL] 需要梯度时 cudss 未抛错")
        except NotImplementedError as e:
            print("   [OK] 需要梯度时 cudss 明确 raise: %s" % str(e)[:60])
        pa = solve_polished(s, d, rh, max_iter=60)
        pc = solve_polished(s, d, rh, max_iter=60, assemble="csr",
                            linear_solver="cudss")
        print("   %-8s polished cudss vs dense: max|ΔH|=%.3e max|Δq|=%.3e" %
              (stem, float(np.abs(pa["head"] - pc["head"]).max()),
               float(np.abs(pa["q"] - pc["q"]).max())))
    except Exception:                                # noqa: BLE001
        print("   %-8s %s" % (stem, last()))

print("\nP2 GPU 验收脚本结束")

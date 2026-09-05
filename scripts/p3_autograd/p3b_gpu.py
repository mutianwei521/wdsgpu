# -*- coding: utf-8 -*-
"""P3-B 收益与代价（前向+反向）。集群 GPU + 真 cuDSS。

  §1 整体倍数：solve_unrolled 的 **前向+反向** ms/场景，dense vs cudss；
     同一次作业里再测一遍**纯前向**倍数，保证与 P2 的数字同节点可比。
  §2 逐项线代耗时 → 按"每轮预算"重算倍数（对齐审计的 7.7x 估计）
  §3 cudss_grad_slots 的代价/收益（时间 + 显存 + 计数器）
  §4 显存：前向+反向峰值，dense 的 OOM 边界
  §5 cudss_grad_refine = 0/1/2 的时间代价
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
NETS = [("Net1", "Net1.inp"), ("Net3", "Net3.inp"), ("Modena", "Modena.inp"),
        ("City_D", "City_D.inp"), ("ky4", "ky4.inp")]
NMAP = dict(NETS)
SEED = 2026

from nvmath.sparse.advanced import DirectSolver      # noqa: E402
_CNT = {"plan": 0, "fact": 0, "solve": 0}
_op, _of, _os_ = DirectSolver.plan, DirectSolver.factorize, DirectSolver.solve
DirectSolver.plan = lambda self, **k: (_CNT.__setitem__("plan", _CNT["plan"] + 1),
                                       _op(self, **k))[1]
DirectSolver.factorize = lambda self, **k: (_CNT.__setitem__("fact", _CNT["fact"] + 1),
                                            _of(self, **k))[1]
DirectSolver.solve = lambda self, **k: (_CNT.__setitem__("solve", _CNT["solve"] + 1),
                                        _os_(self, **k))[1]


def last():
    return traceback.format_exc().strip().split("\n")[-1][:110]


def mk(stem):
    f = os.path.join(NETDIR, NMAP[stem])
    net = parse_inp(f)
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=f,
                  dense_tank_bound_check=False)
    return net, s


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, rh


def batchify(d, rh, B, seed=SEED):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.8, 1.2, (B, 1)) * g.uniform(0.9, 1.1, (B, d.size))
    R = np.nan_to_num(rh)[None, :] + g.uniform(-1.0, 1.0, (B, rh.size))
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


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


def make_run(s, D, R, KE, w, K, ls, grad, slots=None):
    kw = dict(assemble="csr", linear_solver="cudss") if ls == "cudss" else {}
    if ls == "cudss" and slots:
        s.cudss_cache_max = max(8, slots)
        s.cudss_grad_slots = slots

    def run():
        if ls == "cudss":
            s._cudss_slot_rr = 0
        if not grad:
            with torch.no_grad():
                solve_unrolled(s, D, R, ke=KE, K=K, **kw)
            return
        d = D.clone().requires_grad_(True)
        rh = R.clone().requires_grad_(True)
        out = solve_unrolled(s, d, rh, ke=KE, K=K, **kw)
        (w * out["head_ft"]).sum().backward()
    return run


print("=" * 78)
print("node:", os.popen("hostname").read().strip(), "| torch", torch.__version__,
      "|", torch.cuda.get_device_name(0), "| cpus", os.cpu_count())
import nvmath                                        # noqa: E402
print("nvmath", nvmath.__version__, "| CUBLAS_WORKSPACE_CONFIG =",
      os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

CASES = {}
for stem, fn in NETS:
    try:
        net, s = mk(stem)
        d0, rh0 = boundary(net)
        ke = np.zeros(net.N)
        ke[np.random.default_rng(1).choice(s.junc_nodes,
                                           size=min(40, s.Nj),
                                           replace=False)] = 0.5
        with torch.no_grad():
            D1, R1 = batchify(d0, rh0, 4)
            o = s.solve(D1, R1, ke_int=torch.as_tensor(
                np.broadcast_to(ke[None, :], (4, net.N)).copy(),
                dtype=DT, device=DEV))
        K = int(o["iters"].max())
        CASES[stem] = (net, s, d0, rh0, ke, K)
        print("  %-8s Nj=%-4d nnz=%-5d 收敛迭代数 K=%d" % (stem, s.Nj, s.A_csr_nnz, K))
    except Exception:                                # noqa: BLE001
        print("[skip] %s: %s" % (stem, last()))

# ---------------------------------------------------------------- §1
print()
print("=" * 78)
print("§1 前向+反向 ms/场景（best of 5，warm 2；K=收敛迭代数；slots=K 全复用）")
print("   末两列 = 同一节点、同一次作业里测的**纯前向**，用来和 P2 的倍数对齐")
print("net      B    K  | fwd+bwd dense  cudss   倍数 | 纯前向 dense   cudss   倍数")
for stem in ("Net1", "Net3", "Modena", "City_D", "ky4"):
    if stem not in CASES:
        continue
    net, s, d0, rh0, ke, K = CASES[stem]
    for B in (64, 256):
        try:
            D, R = batchify(d0, rh0, B)
            KE = torch.as_tensor(np.broadcast_to(ke[None, :], (B, net.N)).copy(),
                                 dtype=DT, device=DEV)
            w = torch.as_tensor(np.random.default_rng(3).normal(size=(B, net.N)),
                                dtype=DT, device=DEV)
            s.cudss_free()
            tc = tg(make_run(s, D, R, KE, w, K, "cudss", True, slots=K + 1)) * 1e3 / B
            _CNT["plan"] = 0
            tcf = tg(make_run(s, D, R, KE, w, K, "cudss", False)) * 1e3 / B
            plan_in_timing = _CNT["plan"]
            try:
                td = tg(make_run(s, D, R, KE, w, K, "dense", True)) * 1e3 / B
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                td = float("nan")
            tdf = tg(make_run(s, D, R, KE, w, K, "dense", False)) * 1e3 / B
            print("%-8s %-4d %-3d| %13.5f %8.5f %6s | %12.5f %8.5f %6.2fx  (计时区内 plan=%d)"
                  % (stem, B, K, td, tc,
                     ("OOM" if td != td else "%.2fx" % (td / tc)),
                     tdf, tcf, tdf / tcf, plan_in_timing))
            s.cudss_free()
            torch.cuda.empty_cache()
        except Exception:                            # noqa: BLE001
            print("[skip] %s B=%d: %s" % (stem, B, last()))
            torch.cuda.empty_cache()

# ---------------------------------------------------------------- §2
print()
print("=" * 78)
print("§2 逐项线代耗时（B=256，ms/轮，整批）→ 每轮预算与倍数")
print("net      | csr装配 factorize solve×1 | 稠密装配 chol  cholsolve×1 残差elem | dense反向/轮")
for stem in ("Net3", "Modena", "City_D", "ky4"):
    if stem not in CASES:
        continue
    net, s, d0, rh0, ke, K = CASES[stem]
    B = 256
    try:
        D, R = batchify(d0, rh0, B)
        Nj = s.Nj
        g = np.random.default_rng(11)
        vals = torch.as_tensor(g.normal(size=(B, s.A_idx.numel())), dtype=DT,
                               device=DEV)
        t_csr = tg(lambda: s._assemble_csr(vals, B), reps=9, warm=3) * 1e3

        def dense_asm():
            A = torch.zeros(B, Nj * Nj, dtype=DT, device=DEV)
            A.scatter_add_(1, s.A_idx.expand(B, -1), vals)
            return A.view(B, Nj, Nj)
        t_dasm = tg(dense_asm, reps=9, warm=3) * 1e3
        # A 与 F 取**真前向最后一轮**装出来的那一对（真实条件数、真实位型），
        # 不自造矩阵 - 分解/回代的耗时对数值分布敏感。
        cap = {}
        _o = GGASolver._cudss_forward

        def _cap(self, data_, F_, B_, refine=None, slot=0):
            r = _o(self, data_, F_, B_, refine, slot)
            cap["d"], cap["F"] = data_.detach().clone(), F_.detach().clone()
            return r
        GGASolver._cudss_forward = _cap
        try:
            with torch.no_grad():
                s.solve(D, R, ke_int=torch.as_tensor(
                    np.broadcast_to(ke[None, :], (B, net.N)).copy(),
                    dtype=DT, device=DEV),
                    assemble="csr", linear_solver="cudss")
        finally:
            GGASolver._cudss_forward = _o
        data, F = cap["d"], cap["F"]
        A = s._csr_to_dense(data, B)
        st = s._cudss_state(B, DT, torch.device(DEV), 0)
        s._cudss_load(st, data)
        st["rhs"].copy_(F)
        ds = st["solver"]
        ds.factorize()
        t_fact = tg(lambda: ds.factorize(), reps=9, warm=3) * 1e3
        t_sol = tg(lambda: torch.stack(ds.solve()), reps=9, warm=3) * 1e3
        chol = torch.linalg.cholesky(A)
        Fc = F.unsqueeze(-1)
        t_chol = tg(lambda: torch.linalg.cholesky(A), reps=9, warm=3) * 1e3
        t_cs = tg(lambda: torch.cholesky_solve(Fc, chol), reps=9, warm=3) * 1e3
        Hj = torch.cholesky_solve(Fc, chol)
        t_res = tg(lambda: (A * Hj.transpose(-2, -1)).sum(-1, keepdim=True),
                   reps=9, warm=3) * 1e3
        # dense 侧反向一轮的真实代价：对 A、F 求 (chol → cholesky_solve) 的 vjp
        Av = A.clone().requires_grad_(True)
        Fv = F.clone().requires_grad_(True)
        gv = torch.as_tensor(g.normal(size=(B, Nj)), dtype=DT, device=DEV)

        def dense_bwd():
            L_ = torch.linalg.cholesky(Av)
            x = torch.cholesky_solve(Fv.unsqueeze(-1), L_)
            for _ in range(2):
                AH = (Av * x.transpose(-2, -1)).sum(-1, keepdim=True)
                x = x + torch.cholesky_solve(Fv.unsqueeze(-1) - AH, L_)
            (gv * x.squeeze(-1)).sum().backward()
            Av.grad = None
            Fv.grad = None
        try:
            t_dbwd = tg(dense_bwd, reps=3, warm=1) * 1e3
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            t_dbwd = float("nan")
        print("%-8s | %7.4f %9.4f %7.4f | %8.4f %6.3f %11.4f %8.4f | %.3f"
              % (stem, t_csr, t_fact, t_sol, t_dasm, t_chol, t_cs, t_res, t_dbwd))
        s.cudss_free()
        del A, chol, data, Av, Fv
        torch.cuda.empty_cache()
    except Exception:                                # noqa: BLE001
        print("[skip] %s: %s" % (stem, last()))
        torch.cuda.empty_cache()

# ---------------------------------------------------------------- §3
print()
print("=" * 78)
print("§3 cudss_grad_slots 的代价/收益（ky4 与 Modena，B=256，K=收敛迭代数）")
print("net      slots | fwd+bwd ms/场景  factorize bwd_refact | torch峰值MiB 设备已用MiB")
for stem in ("Modena", "ky4"):
    if stem not in CASES:
        continue
    net, s, d0, rh0, ke, K = CASES[stem]
    B = 256
    D, R = batchify(d0, rh0, B)
    KE = torch.as_tensor(np.broadcast_to(ke[None, :], (B, net.N)).copy(),
                         dtype=DT, device=DEV)
    w = torch.as_tensor(np.random.default_rng(3).normal(size=(B, net.N)),
                        dtype=DT, device=DEV)
    for slots in (1, 2, 4, K, K + 4):
        try:
            s.cudss_free(empty_cache=True)
            fn = make_run(s, D, R, KE, w, K, "cudss", True, slots=slots)
            t = tg(fn) * 1e3 / B
            s.cudss_counters(reset=True)
            torch.cuda.reset_peak_memory_stats()
            fn()
            torch.cuda.synchronize()
            cn = s.cudss_counters()
            pk = torch.cuda.max_memory_allocated() / 2 ** 20
            free, tot = torch.cuda.mem_get_info()
            print("%-8s %-5d | %15.5f  %-9d %-10d | %11.1f %12.1f"
                  % (stem, slots, t, cn["factorize"], cn["bwd_refactorize"],
                     pk, (tot - free) / 2 ** 20))
        except Exception:                            # noqa: BLE001
            print("[skip] %s slots=%d: %s" % (stem, slots, last()))
            torch.cuda.empty_cache()
    s.cudss_free(empty_cache=True)
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- §4
print()
print("=" * 78)
print("§4 显存：前向+反向峰值（每配置前先 free + empty_cache）")
print("net      B    K  | dense torch峰值  cudss torch峰值  降幅 | cudss 设备已用")
for stem in ("Net3", "Modena", "City_D", "ky4"):
    if stem not in CASES:
        continue
    net, s, d0, rh0, ke, K = CASES[stem]
    for B in (64, 256, 512):
        try:
            D, R = batchify(d0, rh0, B)
            KE = torch.as_tensor(np.broadcast_to(ke[None, :], (B, net.N)).copy(),
                                 dtype=DT, device=DEV)
            w = torch.as_tensor(np.random.default_rng(3).normal(size=(B, net.N)),
                                dtype=DT, device=DEV)
            s.cudss_free(empty_cache=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                make_run(s, D, R, KE, w, K, "dense", True)()
                torch.cuda.synchronize()
                pkd = torch.cuda.max_memory_allocated() / 2 ** 20
            except torch.OutOfMemoryError:
                pkd = float("nan")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            make_run(s, D, R, KE, w, K, "cudss", True, slots=K + 1)()
            torch.cuda.synchronize()
            pkc = torch.cuda.max_memory_allocated() / 2 ** 20
            free, tot = torch.cuda.mem_get_info()
            print("%-8s %-4d %-3d| %15s %16.2f %6s | %14.1f"
                  % (stem, B, K, ("OOM" if pkd != pkd else "%.2f" % pkd), pkc,
                     ("-" if pkd != pkd else "%.1fx" % (pkd / pkc)),
                     (tot - free) / 2 ** 20))
            s.cudss_free(empty_cache=True)
            torch.cuda.empty_cache()
        except Exception:                            # noqa: BLE001
            print("[skip] %s B=%d: %s" % (stem, B, last()))
            torch.cuda.empty_cache()

# ---------------------------------------------------------------- §5
print()
print("=" * 78)
print("§5 cudss_grad_refine 的时间代价（B=256，slots=K，ms/场景）")
print("net      | refine=0   refine=1   refine=2 | dense fwd+bwd | refine=0 的倍数")
for stem in ("Net3", "Modena", "City_D", "ky4"):
    if stem not in CASES:
        continue
    net, s, d0, rh0, ke, K = CASES[stem]
    B = 256
    try:
        D, R = batchify(d0, rh0, B)
        KE = torch.as_tensor(np.broadcast_to(ke[None, :], (B, net.N)).copy(),
                             dtype=DT, device=DEV)
        w = torch.as_tensor(np.random.default_rng(3).normal(size=(B, net.N)),
                            dtype=DT, device=DEV)
        ts = []
        for gr in (0, 1, 2):
            s.cudss_free()
            s.cudss_grad_refine = gr
            ts.append(tg(make_run(s, D, R, KE, w, K, "cudss", True,
                                  slots=K + 1)) * 1e3 / B)
        s.cudss_grad_refine = 2
        try:
            td = tg(make_run(s, D, R, KE, w, K, "dense", True)) * 1e3 / B
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            td = float("nan")
        print("%-8s | %9.5f %10.5f %10.5f | %13s | %s"
              % (stem, ts[0], ts[1], ts[2],
                 ("OOM" if td != td else "%.5f" % td),
                 ("-" if td != td else "%.2fx" % (td / ts[0]))))
        s.cudss_free()
        torch.cuda.empty_cache()
    except Exception:                                # noqa: BLE001
        print("[skip] %s: %s" % (stem, last()))
        torch.cuda.empty_cache()

print("P3B DONE")

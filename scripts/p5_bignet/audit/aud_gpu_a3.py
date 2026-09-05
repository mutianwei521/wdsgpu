# -*- coding: utf-8 -*-
"""审阅项 3（集群侧）：倍数分解诚实吗 - 按 P4 老规矩逐条查。

查的六件事：
  C1 plan 在计时区外吗？ - 用 cudss_cache_info()/cudss_counters() 直接看，
     并单报 "第一次调用（含 plan）" 与 "稳态" 的差。
  C2 sync 齐全吗？ - 同一格用 perf_counter+synchronize 与 CUDA event 两把尺量。
  C3 warmup 够吗？ - 连打 10 次，逐次打印，看有没有还在下行。
  C4 两路的**迭代数**逐格相等吗？ - 前一轮的时间表**从没核对过这一条**。
  C5 精化步数相等吗？ - cudss 计数器 factorize/solve/bwd_solve 逐格打印。
  C6 ★ 稠密对照有没有自缚？ - 仓库的稠密精化残差写成
       (A * x^T).sum(-1)（为批不变性刻意避开 bmm），这会开 [B,Nj,Nj] 临时张量。
       本脚本把 Badj 的残差换成 bmm 再测一遍，并把反向也补到 1+2 次回代
       （与 cudss 侧等工作量），给出**修正后的 Badj/C**。

用法：python3 -X utf8 aud_gpu_a3.py
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
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                        # noqa: E402
from dgga.solver import GGASolver                       # noqa: E402

DEV, DT = "cuda", torch.float64
NETDIR = os.path.join(ROOT, "audnets")
NODE = os.popen("hostname").read().strip()
CELLS = [("NW_Model", 1), ("ky8", 1), ("ky8", 64), ("ky4", 1), ("ky4", 64),
         ("KL", 1)]
_ONLY = os.environ.get("AUD_CELLS", "").strip()
if _ONLY:
    CELLS = [(c.split(":")[0], int(c.split(":")[1])) for c in _ONLY.split(",")]


def last():
    return traceback.format_exc().strip().split("\n")[-1][:130]


class Adj:
    """稠密手写伴随的四个变体。resid: 'elem'（仓库写法）/'bmm'；
    bwd_refine: 反向精化步数（仓库对照是 0，cudss 侧是 2）。"""

    @staticmethod
    def make(resid, bwd_refine):
        class F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, A, Fv, refine):
                with torch.no_grad():
                    chol = torch.linalg.cholesky(A)
                    Fc = Fv.unsqueeze(-1)
                    x = torch.cholesky_solve(Fc, chol)
                    for _ in range(int(refine)):
                        AH = (A @ x) if resid == "bmm" else \
                             (A * x.transpose(-2, -1)).sum(-1, keepdim=True)
                        x = x + torch.cholesky_solve(Fc - AH, chol)
                ctx.save_for_backward(chol, x, A)
                return x.squeeze(-1)

            @staticmethod
            def backward(ctx, g):
                chol, x, A = ctx.saved_tensors
                with torch.no_grad():
                    gc = g.unsqueeze(-1)
                    lam = torch.cholesky_solve(gc, chol)
                    for _ in range(int(bwd_refine)):
                        AL_ = (A @ lam) if resid == "bmm" else \
                              (A * lam.transpose(-2, -1)).sum(-1, keepdim=True)
                        lam = lam + torch.cholesky_solve(gc - AL_, chol)
                    gA = -lam @ x.transpose(-2, -1)
                return gA, lam.squeeze(-1), None
        return F


def tg(fn, warm=1, reps=6, budget=6.0):
    """best-of + 逐次序列（查 warmup 够不够）。返回 (best_ms, [每次 ms])。"""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    seq = []
    t_all = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        seq.append(dt)
        t_all += dt / 1e3
        if t_all > budget and len(seq) >= 2:
            break
    return min(seq), seq


def tg_event(fn, warm=1, reps=3):
    """CUDA event 计时（与 perf_counter 互证）。"""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1))
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


def batchify(d, rh, B, seed=2026):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.85, 1.15, (B, d.size))
    R = rh[None, :] + g.uniform(-1.0, 1.0, (B, rh.size))
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


print("=" * 108)
print("审阅项 3 · 倍数分解诚实吗 | node:", NODE, "| torch", torch.__version__,
      "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
import nvmath                                            # noqa: E402
print("nvmath", nvmath.__version__)
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f, hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print("=" * 108)

for stem, B in CELLS:
    print("\n" + "#" * 100)
    f = os.path.join(NETDIR, stem + ".inp")
    net = parse_inp(f)
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=f,
                  dense_tank_bound_check=False)
    d0, rh0 = boundary(net)
    D, R = batchify(d0, rh0, B)
    nnz = int(s.A_csr_nnz)
    print("### %s Nj=%d L=%d nnz=%d  B=%d   Nj^2/nnz=%.0f"
          % (stem, s.Nj, s.L, nnz, B, s.Nj ** 2 / nnz))

    # ---------------- C4/C5 迭代数与精化步数 ----------------
    try:
        with torch.no_grad():
            od = s.solve(D, R)
        s.cudss_counters(reset=True)
        with torch.no_grad():
            oc = s.solve(D, R, assemble="csr", linear_solver="cudss")
        cnt = s.cudss_counters()
        itd = od["iters"].cpu().numpy()
        itc = oc["iters"].cpu().numpy()
        dH = float((od["head_ft"] - oc["head_ft"]).abs().max())
        print("  C4 迭代数  dense=%s..%s  cudss=%s..%s  逐样本全等=%s | max|dH|=%.3e ft"
              % (itd.min(), itd.max(), itc.min(), itc.max(),
                 bool((itd == itc).all()), dH))
        print("  C5 cudss 计数器（一次完整前向）: %s   期望 factorize=迭代数, "
              "solve=(1+refine)*迭代数=%d" % (cnt, 3 * int(itc.max())))
    except Exception:                                    # noqa: BLE001
        print("  [err C4/C5] %s" % last())

    # ---------------- C1 plan 在计时区外吗 ----------------
    try:
        s.cudss_free(empty_cache=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            s.solve(D, R, assemble="csr", linear_solver="cudss")
        torch.cuda.synchronize()
        cold = (time.perf_counter() - t0) * 1e3
        info = s.cudss_cache_info()
        t0 = time.perf_counter()
        with torch.no_grad():
            s.solve(D, R, assemble="csr", linear_solver="cudss")
        torch.cuda.synchronize()
        warm = (time.perf_counter() - t0) * 1e3
        print("  C1 cudss 冷启（含 plan）%.3f ms  vs 热 %.3f ms  差 %.3f ms | "
              "cache_info=%s" % (cold, warm, cold - warm, info))
    except Exception:                                    # noqa: BLE001
        print("  [err C1] %s" % last())

    # ---------------- 纯前向：两把尺 + 逐次序列 ----------------
    fwd = {}
    for tag, kw in (("cudss", dict(assemble="csr", linear_solver="cudss")),
                    ("dense", dict())):
        try:
            def f_(kw=kw):
                with torch.no_grad():
                    s.solve(D, R, **kw)
            best, seq = tg(f_, warm=3, reps=10)
            ev = tg_event(f_, warm=1, reps=3)
            fwd[tag] = best
            print("  fwd %-5s best=%.4f ms/批 (%.5f ms/场景) | event=%.4f ms | "
                  "10 次序列 %s" % (tag, best, best / B, ev,
                                    " ".join("%.2f" % x for x in seq)))
        except Exception:                                # noqa: BLE001
            fwd[tag] = None
            print("  fwd %-5s FAIL %s" % (tag, last()))
            torch.cuda.empty_cache()
    if fwd.get("dense") and fwd.get("cudss"):
        print("  ==> 纯前向倍数 dense/cudss = %.2fx" % (fwd["dense"] / fwd["cudss"]))

    # ---------------- C6 一轮线性代数的分解（clone 提到计时区外） ----------------
    try:
        cap = {}
        _o = GGASolver._cudss_forward

        def _capf(self, data_, F_, B_, refine=None, slot=0):
            r = _o(self, data_, F_, B_, refine, slot)
            cap["d"], cap["F"] = data_.detach().clone(), F_.detach().clone()
            return r
        GGASolver._cudss_forward = _capf
        try:
            with torch.no_grad():
                s.solve(D, R, assemble="csr", linear_solver="cudss")
        finally:
            GGASolver._cudss_forward = _o
        data0, F0 = cap["d"], cap["F"]
        gv = torch.as_tensor(np.random.default_rng(5).normal(size=(B, s.Nj)),
                             dtype=DT, device=DEV)
        A0 = s._csr_to_dense(data0, B)
        s.cudss_grad_refine = 2
        s.cudss_grad_slots = 1

        # 计时区外先备好可求导输入（前一轮把 clone 放在计时区内）
        Av = A0.clone().requires_grad_(True)
        Fv = F0.clone().requires_grad_(True)
        dv = data0.clone().requires_grad_(True)
        Fv2 = F0.clone().requires_grad_(True)

        def mk_dense(resid):
            def run():
                if Av.grad is not None:
                    Av.grad = None
                    Fv.grad = None
                Lc = torch.linalg.cholesky(Av)
                Fc = Fv.unsqueeze(-1)
                x = torch.cholesky_solve(Fc, Lc)
                for _ in range(2):
                    AH = (Av @ x) if resid == "bmm" else \
                         (Av * x.transpose(-2, -1)).sum(-1, keepdim=True)
                    x = x + torch.cholesky_solve(Fc - AH, Lc)
                (gv * x.squeeze(-1)).sum().backward()
                return Av.grad, Fv.grad
            return run

        def mk_adj(resid, bwdref):
            Fn = Adj.make(resid, bwdref)

            def run():
                if Av.grad is not None:
                    Av.grad = None
                    Fv.grad = None
                x = Fn.apply(Av, Fv, 2)
                (gv * x).sum().backward()
                return Av.grad, Fv.grad
            return run

        def path_c():
            if dv.grad is not None:
                dv.grad = None
                Fv2.grad = None
            x = s._cudss_solve(dv, Fv2, B)
            (gv * x).sum().backward()
            return dv.grad, Fv2.grad

        variants = [("A  稠密+autograd(elem, 仓库)", mk_dense("elem")),
                    ("A' 稠密+autograd(bmm)", mk_dense("bmm")),
                    ("B  稠密+手写伴随(elem,bwd0, 上一轮)", mk_adj("elem", 0)),
                    ("B' 稠密+手写伴随(bmm ,bwd0)", mk_adj("bmm", 0)),
                    ("B''稠密+手写伴随(bmm ,bwd2=等工作量)", mk_adj("bmm", 2))]
        tms, grads = {}, {}
        tc, seqc = tg(path_c, warm=2, reps=6)
        gc_ = path_c()[0].detach().clone()
        print("  C6 C cudss(gr=2) = %.4f ms/轮   序列 %s"
              % (tc, " ".join("%.3f" % x for x in seqc)))
        for name, fn in variants:
            try:
                t, seq = tg(fn, warm=2, reps=6)
                tms[name] = t
                grads[name] = fn()[0].detach().clone()
                print("  C6 %-38s = %10.4f ms/轮  /C = %8.2fx  序列 %s"
                      % (name, t, t / tc, " ".join("%.2f" % x for x in seq)))
            except Exception:                            # noqa: BLE001
                print("  C6 %-38s FAIL %s" % (name, last()))
                torch.cuda.empty_cache()
        # 梯度一致性（都与 A 比）
        base = grads.get("A  稠密+autograd(elem, 仓库)")
        if base is not None:
            bn = base.abs().max()
            for name, g in grads.items():
                if name.startswith("A  "):
                    continue
                print("     一致性 gA %-36s rel=%.3e" %
                      (name, float((g - base).abs().max() / bn)))
            gnz = base.reshape(B, -1).index_select(1, s.A_csr_dense_pos)
            print("     一致性 gA %-36s rel=%.3e" %
                  ("C cudss", float((gnz - gc_).abs().max() / gnz.abs().max())))
        print("  ==> 结论行：Badj/C 上一轮口径=%s ；换 bmm 后=%s ；等工作量后=%s"
              % tuple("%.2fx" % (tms[k] / tc) if k in tms else "-"
                      for k in ("B  稠密+手写伴随(elem,bwd0, 上一轮)",
                                "B' 稠密+手写伴随(bmm ,bwd0)",
                                "B''稠密+手写伴随(bmm ,bwd2=等工作量)")))
        del Av, Fv, dv, Fv2, A0, data0, F0, gv, gc_, grads
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        print("  [C6] 稠密侧 OOM")
        torch.cuda.empty_cache()
    except Exception:                                    # noqa: BLE001
        print("  [err C6] %s" % last())
        torch.cuda.empty_cache()
    try:
        s.cudss_free(empty_cache=True)
    except Exception:                                    # noqa: BLE001
        pass
    del s, net, D, R
    torch.cuda.empty_cache()

print("\nAUD A3 DONE")

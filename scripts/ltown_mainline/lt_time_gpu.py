# -*- coding: utf-8 -*-
"""L-TOWN 主线证据包 · 任务二（GPU 全表，时间半）。P4/P5 同口径，仅换网 + 换反向形态。

L-TOWN 含 3 个 PRV：dense 批量路径必须 dense_status_machine=True + solve(...,
status_machine=True)；solve_unrolled 对 PRV 维持 raise（不是本网的训练形态）。
因此本表的"前向+反向"= **实际训练形态**：
    ① GPU 批量前向（dense 或 csr+cuDSS，状态机，no_grad）拿逐场景收敛状态 S*；
    ② 梯度走 ImplicitGGASolve（隐式伴随）：按 S* 的唯一状态分组、冻结状态，
       epanet 模式 CPU 求解器上 forward(solve_polished)+backward(splu)。
    ②与线性求解器无关（两列共享同一 CPU 伴随），表里分列打出两段耗时，
    不合成一个数 - L-TOWN 的性能故事在①的显存与批量，不在②。
§C 倍数分解（P4 §11.2 口径）：一轮线性代数 f+b 的三条通路 A/Badj/C，
    A 的装配值取自收敛帧（含 ACTIVE PRV 的 CBIG 罚函数行）。
只读，不改 dgga。
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
from dgga.parse import parse_inp                      # noqa: E402
from dgga.solver import GGASolver                     # noqa: E402
from dgga.autodiff import implicit_solve              # noqa: E402

DEV = "cuda"
DT = torch.float64
INP = os.path.join(ROOT, "networks_prv", "L-TOWN.inp")
SEED = 2026
NODE = os.popen("hostname").read().strip()
BS = [int(x) for x in os.environ.get("LT_BS", "1,8,64,256,512,1024").split(",")]


def last():
    return traceback.format_exc().strip().split("\n")[-1][:140]


class DenseAdjoint(torch.autograd.Function):
    """稠密版"手写伴随"，与 _CudssSolveFn 同构（P4 §11.2 的 Badj 对照，不进 dgga）。"""

    @staticmethod
    def forward(ctx, A, F, refine):
        with torch.no_grad():
            chol = torch.linalg.cholesky(A)
            Fc = F.unsqueeze(-1)
            x = torch.cholesky_solve(Fc, chol)
            for _ in range(int(refine)):
                AH = (A * x.transpose(-2, -1)).sum(-1, keepdim=True)
                x = x + torch.cholesky_solve(Fc - AH, chol)
        ctx.save_for_backward(chol, x)
        return x.squeeze(-1)

    @staticmethod
    def backward(ctx, g):
        chol, x = ctx.saved_tensors
        with torch.no_grad():
            lam = torch.cholesky_solve(g.unsqueeze(-1), chol)
            gA = -lam @ x.transpose(-2, -1)
        return gA, lam.squeeze(-1), None


def tg(fn, budget=3.0, reps_max=5, warm=1):
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
print("L-TOWN 主线 · 时间全表（P4/P5 同口径；f+b=实际训练形态 GPU前向+隐式伴随）")
print("node:", NODE, "| torch", torch.__version__, "|",
      torch.cuda.get_device_name(0))
try:
    import nvmath                                      # noqa: E402
    print("nvmath", nvmath.__version__, "| B 列:", BS)
except ImportError:                                    # 本机冒烟（无 nvmath）：cudss 列会如实报 err
    print("nvmath MISSING（本机冒烟模式，cudss 列不可用）| B 列:", BS)
import hashlib                                         # noqa: E402
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f,
          hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print("sha256 INP",
      hashlib.sha256(open(INP, "rb").read()).hexdigest())
print()

net = parse_inp(INP)
s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=INP,
              dense_status_machine=True)
se = GGASolver(net, mode="epanet", inp_path=INP)     # 隐式伴随用（CPU）
d0, rh0 = boundary(net)
base = se.run_gga(d0, rh0, do_status=True)
K0set = base["setting"].copy()

# K：CPU dense(SM) B=8 实测收敛迭代数（与显存脚本同口径）
Dk, Rk = batchify(d0, rh0, 8)
scpu = GGASolver(net, device="cpu", dtype=DT, mode="dense", inp_path=INP,
                 dense_status_machine=True)
with torch.no_grad():
    K = int(scpu.solve(Dk.cpu(), Rk.cpu(),
                       status_machine=True)["iters"].max())
del scpu
print("### L-TOWN Nj=%d L=%d N=%d nnz=%d K=%d（CPU B=8 max iters）"
      % (s.Nj, s.L, net.N, int(s.A_csr_dense_pos.numel()), K))
sys.stdout.flush()

ROWS = []   # (B, K, fwd_d, fwd_c, timp_ms_per, fb_d, fb_c, ngroup, st_eq)
DEC = []    # (B, ta, tb, tc, r_ab, r_ac)

for B in BS:
    D, R = batchify(d0, rh0, B)
    W = np.random.default_rng(7).normal(size=(B, net.N))
    res = {}
    outs = {}
    # ---- ① 纯前向（状态机） ----
    for tag, kw in (("fwd_c", dict(assemble="csr", linear_solver="cudss")),
                    ("fwd_d", dict())):
        try:
            def f_(kw=kw):
                with torch.no_grad():
                    outs[tag] = s.solve(D, R, status_machine=True, **kw)
            res[tag] = tg(f_) * 1e3 / B
        except torch.OutOfMemoryError:
            res[tag] = None
            torch.cuda.empty_cache()
        except Exception:                              # noqa: BLE001
            res[tag] = None
            print("  [err %s B=%d] %s" % (tag, B, last()))
            torch.cuda.empty_cache()
    st_eq = -1
    conv_d = -1
    if outs.get("fwd_d") is not None:
        conv_d = int(outs["fwd_d"]["converged"].sum())
    if outs.get("fwd_d") is not None and outs.get("fwd_c") is not None:
        Sd = outs["fwd_d"]["status"].cpu().numpy()
        Sc = outs["fwd_c"]["status"].cpu().numpy()
        st_eq = int(sum(np.array_equal(Sd[b], Sc[b]) for b in range(B)))

    # ---- ② 隐式伴随 f+b（实际训练形态的反向段；与线性求解器无关）----
    timp = None
    ngroup = -1
    gnorm = float("nan")
    src = outs.get("fwd_d") or outs.get("fwd_c")
    if src is not None:
        try:
            S_all = src["status"].cpu().numpy()
            uniq, inv = np.unique(S_all, axis=0, return_inverse=True)
            ngroup = uniq.shape[0]
            Dc = D.detach().cpu()
            Rc = R.detach().cpu()
            Wc = torch.as_tensor(W, dtype=DT)
            jn = torch.as_tensor(np.asarray(se.junc_nodes), dtype=torch.long)

            def impl_():
                Dv = Dc.clone().requires_grad_(True)
                loss = 0.0
                for gidx in range(uniq.shape[0]):
                    ii = torch.as_tensor(np.where(inv == gidx)[0])
                    h, _q, _e = implicit_solve(
                        se, Dv.index_select(0, ii), Rc.index_select(0, ii),
                        speed=K0set, status=uniq[gidx])
                    loss = loss + (h.index_select(1, jn) *
                                   Wc.index_select(0, ii).index_select(1, jn)
                                   ).sum()
                loss.backward()
                return Dv.grad
            t0 = time.perf_counter()
            g1 = impl_()
            timp = time.perf_counter() - t0
            gnorm = float(g1.norm())
            if timp < 30.0:                      # 便宜就再来一次取最小
                t0 = time.perf_counter()
                impl_()
                timp = min(timp, time.perf_counter() - t0)
        except Exception:                              # noqa: BLE001
            timp = None
            print("  [err impl B=%d] %s" % (B, last()))
    timp_per = None if timp is None else timp * 1e3 / B
    fb_d = None if (res.get("fwd_d") is None or timp_per is None) \
        else res["fwd_d"] + timp_per
    fb_c = None if (res.get("fwd_c") is None or timp_per is None) \
        else res["fwd_c"] + timp_per
    ROWS.append((B, K, res.get("fwd_d"), res.get("fwd_c"), timp_per,
                 fb_d, fb_c, ngroup, st_eq))
    cell = lambda x, w=9: ("OOM").rjust(w) if x is None else ("%*.5f" % (w, x))
    print("  B=%-5d fwd dense %s cudss %s | impl(共享CPU) %s ms/场景"
          " (状态组 %d, 两路状态同 %s/%d, dense收敛 %s/%d) | f+b dense %s cudss %s"
          " |g|=%.4e"
          % (B, cell(res.get("fwd_d")), cell(res.get("fwd_c")),
             cell(timp_per), ngroup, st_eq, B, conv_d, B,
             cell(fb_d), cell(fb_c), gnorm))
    sys.stdout.flush()

    # ---- ③ 倍数分解（P4 §11.2；A 取收敛帧装配值，含 ACTIVE PRV CBIG 行）----
    A0 = None
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
                s.solve(D, R, status_machine=True,
                        assemble="csr", linear_solver="cudss")
        finally:
            GGASolver._cudss_forward = _o
        data0, F0 = cap["d"], cap["F"]
        gv = torch.as_tensor(np.random.default_rng(5).normal(size=(B, s.Nj)),
                             dtype=DT, device=DEV)

        def path_a():
            Av = A0.clone().requires_grad_(True)
            Fv = F0.clone().requires_grad_(True)
            Lc = torch.linalg.cholesky(Av)
            Fc = Fv.unsqueeze(-1)
            x = torch.cholesky_solve(Fc, Lc)
            for _ in range(2):
                AH = (Av * x.transpose(-2, -1)).sum(-1, keepdim=True)
                x = x + torch.cholesky_solve(Fc - AH, Lc)
            (gv * x.squeeze(-1)).sum().backward()
            return Av.grad, Fv.grad

        def path_b():
            Av = A0.clone().requires_grad_(True)
            Fv = F0.clone().requires_grad_(True)
            x = DenseAdjoint.apply(Av, Fv, 2)
            (gv * x).sum().backward()
            return Av.grad, Fv.grad

        def path_c():
            s.cudss_grad_refine = 2
            s.cudss_grad_slots = 1
            dv = data0.clone().requires_grad_(True)
            Fv = F0.clone().requires_grad_(True)
            x = s._cudss_solve(dv, Fv, B)
            (gv * x).sum().backward()
            return dv.grad, Fv.grad

        tc = tg(path_c, budget=3.0, reps_max=3) * 1e3
        gc, _ = path_c()
        try:
            A0 = s._csr_to_dense(data0, B)
            ga, _ = path_a()
            gb, _ = path_b()
            r_ab = float((ga - gb).abs().max() / ga.abs().max())
            ga_nnz = ga.reshape(B, -1).index_select(1, s.A_csr_dense_pos)
            r_ac = float((ga_nnz - gc).abs().max() / ga_nnz.abs().max())
            ta = tg(path_a, budget=3.0, reps_max=3) * 1e3
            tb = tg(path_b, budget=3.0, reps_max=3) * 1e3
            del ga, gb, ga_nnz
        except torch.OutOfMemoryError:
            ta = tb = r_ab = r_ac = None
        finally:
            A0 = None
            torch.cuda.empty_cache()
        DEC.append((B, ta, tb, tc, r_ab, r_ac))
        print("     分解 一轮线代 f+b(ms/轮): A %s | Badj %s | C %9.4f"
              " | Badj/C %s  A/Badj %s  A/C %s | gA一致 (B/A) %s (C/A) %s"
              % ("OOM      " if ta is None else "%9.4f" % ta,
                 "OOM      " if tb is None else "%9.4f" % tb, tc,
                 "  -   " if tb is None else "%6.2fx" % (tb / tc),
                 "  -   " if ta is None else "%6.2fx" % (ta / tb),
                 "  -   " if ta is None else "%6.2fx" % (ta / tc),
                 "  -    " if r_ab is None else "%.2e" % r_ab,
                 "  -    " if r_ac is None else "%.2e" % r_ac))
        del data0, F0, gv, gc
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError:
        print("     分解 B=%d cudss 侧 OOM" % B)
        torch.cuda.empty_cache()
    except Exception:                                  # noqa: BLE001
        print("     [err 分解 B=%d] %s" % (B, last()))
        torch.cuda.empty_cache()
    finally:
        A0 = None
        torch.cuda.empty_cache()
    del D, R, outs
    torch.cuda.empty_cache()
    sys.stdout.flush()

try:
    s.cudss_free(empty_cache=True)
except Exception:                                      # noqa: BLE001
    pass

print()
print("=" * 96)
print("§T1 汇总（ms/场景）  node=%s" % NODE)
print("B     K  | fwd dense  fwd cudss  倍数   | impl(CPU伴随/场景) | "
      "f+b dense  f+b cudss  倍数   | 状态组/两路状态同")
for (B, K_, fd, fc, ti, bd, bc, ng, seq) in ROWS:
    c = lambda x, w=9: ("OOM").rjust(w) if x is None else ("%*.5f" % (w, x))
    rf = "   -   " if (fd is None or fc is None) else "%6.2fx" % (fd / fc)
    rb = "   -   " if (bd is None or bc is None) else "%6.2fx" % (bd / bc)
    print("%-5d %-2d | %s  %s %s | %s | %s  %s %s | %d / %s"
          % (B, K_, c(fd), c(fc), rf, c(ti, 12), c(bd), c(bc), rb, ng, seq))

print()
print("§T2 倍数分解（一轮线性代数 f+b，ms/轮，整批；A 含 ACTIVE PRV CBIG 行）"
      "  node=%s" % NODE)
print("B     | A 稠密+autograd | Badj 稠密+手写伴随 | C cudss(gr=2) | "
      "Badj/C(稀疏赢)  A/Badj(通用autograd欠账)  A/C(仓库现状)")
for (B, ta, tb, tc, r_ab, r_ac) in DEC:
    print("%-5d | %15s | %18s | %13.4f | %14s %23s %14s"
          % (B,
             "OOM" if ta is None else "%.4f" % ta,
             "OOM" if tb is None else "%.4f" % tb, tc,
             "-" if tb is None else "%.2fx" % (tb / tc),
             "-" if ta is None else "%.2fx" % (ta / tb),
             "-" if ta is None else "%.2fx" % (ta / tc)))
print("LT TIME DONE")

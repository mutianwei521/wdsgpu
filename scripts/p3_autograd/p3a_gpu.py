# -*- coding: utf-8 -*-
"""P3-A 梯度正确性（sparse_gpu_plan.md §1c）。集群 GPU + 真 cuDSS。

判据链（审计前置：**有限差分一律拿 dense 前向做** - cuDSS 前向自身在病态网上
有 1e-5 ft 级的运行间抖动，拿它做中心差分会被噪声整个吃掉）：
  §A cudss 解析梯度 vs **dense 前向**的中心差分（Richardson 外推）
     同时给出 dense 解析梯度 vs 同一份 FD 作为**对照**（FD 自身的分辨率）
  §B cudss 解析梯度 vs dense 解析梯度（逐参数、B∈{1,8,64,256}）
  §C cudss 解析梯度 vs ImplicitGGASolve（隐函数定理伴随，本项目梯度主通路）
  §D 计数器：factorize 有没有翻倍（slots 扫描）
  §E 守卫：二阶导 / epanet / cpu / f32 / assemble 组合
"""
import os
import sys
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                     # noqa: E402
from dgga.solver import GGASolver                    # noqa: E402
from dgga.autodiff import ImplicitGGASolve, solve_unrolled    # noqa: E402

DEV = "cuda"
DT = torch.float64
NETDIR = os.path.join(ROOT, "p2nets")
NETS = [("Hanoi", "Hanoi.inp", 20), ("Net3", "Net3.inp", 20),
        ("Modena", "Modena.inp", 20), ("City_D", "City_D.inp", 24),
        ("ky4", "ky4.inp", 24)]
SEED = 2026


def SLOTS(K, B):
    """反向零重分解要 K 份 state；B 大时按份收显存，故 B>=256 只开 4 份
    （§D 已量出 slots 只影响计数器与 <1% 的时间，不影响梯度值本身）。"""
    return K + 2 if B <= 64 else 4


def last():
    return traceback.format_exc().strip().split("\n")[-1][:110]


def mk(fn):
    f = os.path.join(NETDIR, fn)
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
    return D, R


def relmax(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


class Case:
    """一个基准点：网 + (D,R,KE,r0) + 损失权重 w。"""

    def __init__(self, stem, fn, B, K, seed=SEED):
        self.stem, self.B, self.K = stem, B, K
        self.net, self.s = mk(fn)
        d0, rh0 = boundary(self.net)
        self.D, self.R = batchify(d0, rh0, B, seed)
        g = np.random.default_rng(seed + 1)
        ke = np.zeros(self.net.N)
        em = g.choice(self.s.junc_nodes, size=min(40, self.s.Nj), replace=False)
        ke[em] = 0.5
        self.KE = np.broadcast_to(ke[None, :], (B, self.net.N)).copy()
        self.r0 = self.s.r_hw.detach().cpu().numpy().copy()
        self.w = torch.as_tensor(g.normal(size=(B, self.net.N)), dtype=DT, device=DEV)
        self.em_nodes = np.sort(em)

    ARG = {"d": "D", "rh": "R", "ke": "KE", "r": "r"}

    def loss(self, D=None, R=None, KE=None, r=None, ls="dense", grad=False,
             slots=None):
        s = self.s
        kw = dict(assemble="csr", linear_solver="cudss") if ls == "cudss" else {}
        if ls == "cudss" and slots:
            s.cudss_cache_max = max(8, slots)
            s.cudss_grad_slots = slots
            s._cudss_slot_rr = 0
        d = torch.tensor(self.D if D is None else D, dtype=DT, device=DEV,
                         requires_grad=grad)
        rh = torch.tensor(self.R if R is None else R, dtype=DT, device=DEV,
                          requires_grad=grad)
        ke = torch.tensor(self.KE if KE is None else KE, dtype=DT, device=DEV,
                          requires_grad=grad)
        rr = torch.tensor(self.r0 if r is None else r, dtype=DT, device=DEV,
                          requires_grad=grad)
        out = solve_unrolled(s, d, rh, ke=ke, r_hw=rr, K=self.K, **kw)
        L = (self.w * out["head_ft"]).sum()
        if not grad:
            return float(L.detach())
        L.backward()
        return dict(d=d.grad.detach().clone(), rh=rh.grad.detach().clone(),
                    ke=ke.grad.detach().clone(), r=rr.grad.detach().clone())


def fd_coords(c, kind, n, rng, gde=None):
    """抽 FD 坐标（沿用 gradcheck_3way 的退化过滤精神）。返回 [(b,i), ...]。

    退化过滤：FD 对 |g| 近 0 的坐标没有分辨率（f64 损失噪声 / 2h 会把相对误差
    顶到 O(1)），所以只在 |解析梯度| >= 该类候选 60 分位的坐标里抽。"""
    B, s = c.B, c.s
    if kind == "d":
        pool = [(b, int(i)) for b in range(B) for i in s.junc_nodes
                if c.D[b, i] > 1e-8]
    elif kind == "rh":
        pool = [(b, int(i)) for b in range(B) for i in s.fixed_nodes]
    elif kind == "ke":
        pool = [(b, int(i)) for b in range(B) for i in c.em_nodes]
    else:                                            # r_hw：批共享，用 b=-1
        tcv = np.asarray(s.is_tcv.detach().cpu()).reshape(-1)
        pool = [(-1, int(k)) for k in range(s.L) if not bool(tcv[k])]
    if not pool:
        return []
    if gde is not None:
        t = gde[kind].detach().cpu().numpy()
        mag = np.asarray([abs(t[i] if b < 0 else t[b, i]) for (b, i) in pool])
        thr = np.percentile(mag, 60.0)
        keep = [p_ for p_, m in zip(pool, mag) if m >= thr and m > 0.0]
        if len(keep) >= n:
            pool = keep
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[int(i)] for i in idx]


def fd_grad(c, kind, coords):
    """dense 前向的中心差分 + Richardson（h2=h1/2，(4*D(h2)-D(h1))/3）。"""
    base = {"d": c.D, "rh": c.R, "ke": c.KE, "r": c.r0}[kind]
    out = []
    for (b, i) in coords:
        x0 = base[i] if b < 0 else base[b, i]
        h1 = max(1e-4 * abs(x0), 1e-5) if kind == "rh" \
            else max(1e-3 * abs(x0), 1e-5)
        ds = []
        for h in (h1, h1 / 2.0):
            vals = []
            for sgn in (+1.0, -1.0):
                pert = base.copy()
                if b < 0:
                    pert[i] = x0 + sgn * h
                else:
                    pert[b, i] = x0 + sgn * h
                vals.append(c.loss(**{Case.ARG[kind]: pert}, ls="dense"))
            ds.append((vals[0] - vals[1]) / (2.0 * h))
        out.append((4.0 * ds[1] - ds[0]) / 3.0)
    return np.asarray(out)


def pick(g, kind, coords):
    t = g[kind].detach().cpu().numpy()
    return np.asarray([(t[i] if b < 0 else t[b, i]) for (b, i) in coords])


print("=" * 78)
print("node:", os.popen("hostname").read().strip(), "| torch", torch.__version__,
      "|", torch.cuda.get_device_name(0))
import nvmath                                        # noqa: E402
print("nvmath", nvmath.__version__, "| CUBLAS_WORKSPACE_CONFIG =",
      os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
torch.use_deterministic_algorithms(True)
print("torch.use_deterministic_algorithms(True) 已开（FD 与解析梯度都要位级可复现）")

# ---------------------------------------------------------------- §A
print()
print("=" * 78)
print("§A cudss 解析梯度 vs **dense 前向**中心差分（Richardson）")
print("   同表给出 dense 解析梯度 vs 同一份 FD 作对照 - 它是 FD 自身的分辨率下限")
print("   每格 = 抽样坐标上的 max 相对误差，分母 max(|FD|, 1e-12)")
print("net       B  K  | param  n | cudss vs FD | dense vs FD | cudss vs dense")
NA = 8
for stem, fn, K in NETS:
    for B in (1, 8):
        c = None
        try:
            c = Case(stem, fn, B, K)
            gcu = c.loss(ls="cudss", grad=True, slots=SLOTS(K, B))
            gde = c.loss(ls="dense", grad=True)
            rng = np.random.default_rng(SEED + 3)
            for kind in ("d", "rh", "r", "ke"):
                co = fd_coords(c, kind, NA, rng, gde)
                if not co:
                    print("%-9s %-2d %-3d| %-6s  - | (无可抽坐标)" % (stem, B, K, kind))
                    continue
                f = fd_grad(c, kind, co)
                a_cu, a_de = pick(gcu, kind, co), pick(gde, kind, co)
                den = np.maximum(np.abs(f), 1e-12)
                print("%-9s %-2d %-3d| %-6s %2d | %11.3e | %11.3e | %.3e"
                      % (stem, B, K, kind, len(co),
                         float(np.max(np.abs(a_cu - f) / den)),
                         float(np.max(np.abs(a_de - f) / den)),
                         float(np.max(np.abs(a_cu - a_de) / den))))
        except Exception:                            # noqa: BLE001
            print("[skip] %s B=%d: %s" % (stem, B, last()))
        finally:
            if c is not None:
                c.s.cudss_free()
                del c
            torch.cuda.empty_cache()

# ---------------------------------------------------------------- §B
print()
print("=" * 78)
print("§B cudss vs dense 解析梯度（max 相对差，分母 = 该张量的 max|dense|）")
print("net       B    K  | grad d     grad rh    grad r     grad ke  | max|g_d|   max|g_r|   ← |g| 爆掉时相对差没有意义")
for stem, fn, K in NETS:
    for B in (1, 8, 64, 256):
        c = None
        try:
            c = Case(stem, fn, B, K)
            gcu = c.loss(ls="cudss", grad=True, slots=SLOTS(K, B))
            try:
                gde = c.loss(ls="dense", grad=True)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("%-9s %-4d %-3d| dense 侧 torch.OutOfMemoryError"
                      "（cudss 侧正常出梯度，|g_d|max=%.4e）"
                      % (stem, B, K, float(gcu["d"].abs().max())))
                continue
            print("%-9s %-4d %-3d| %.3e %.3e %.3e %.3e | %.4e %.4e"
                  % (stem, B, K,
                     relmax(gcu["d"], gde["d"]), relmax(gcu["rh"], gde["rh"]),
                     relmax(gcu["r"], gde["r"]), relmax(gcu["ke"], gde["ke"]),
                     float(gde["d"].abs().max()), float(gde["r"].abs().max())))
            del gcu, gde
        except Exception:                            # noqa: BLE001
            print("[skip] %s B=%d: %s" % (stem, B, last()))
        finally:
            if c is not None:
                c.s.cudss_free()
                del c
            torch.cuda.empty_cache()

# ---------------------------------------------------------------- §C
print()
print("=" * 78)
print("§C cudss vs ImplicitGGASolve（隐函数定理伴随；K 取收敛迭代数 +5）")
print("   同表给出 dense 展开 vs 隐式 作对照（展开 vs 不动点的固有差）")
print("net       B  K  | param | cudss vs implicit | dense vs implicit | max|g_implicit|")
for stem, fn, K in NETS:
    B = 4
    c = None
    try:
        c = Case(stem, fn, B, K)
        with torch.no_grad():
            o = c.s.solve(torch.as_tensor(c.D, dtype=DT, device=DEV),
                          torch.as_tensor(c.R, dtype=DT, device=DEV),
                          ke_int=torch.as_tensor(c.KE, dtype=DT, device=DEV))
        c.K = int(o["iters"].max()) + 5
        gcu = c.loss(ls="cudss", grad=True, slots=c.K + 2)
        gde = c.loss(ls="dense", grad=True)
        t = lambda x: torch.tensor(x, dtype=DT, device=DEV, requires_grad=True)
        dd, rr, kk, r_t = t(c.D), t(c.R), t(c.KE), t(c.r0)
        head, flow, emit = ImplicitGGASolve.apply(dd, rr, kk, r_t, c.s,
                                                  1e-12, 200, 3)
        (c.w * head).sum().backward()
        gim = dict(d=dd.grad, rh=rr.grad, ke=kk.grad, r=r_t.grad)
        for kind in ("d", "rh", "r", "ke"):
            den = gim[kind].abs().max().clamp_min(1e-300)
            print("%-9s %-2d %-3d| %-5s | %17.3e | %9.3e | %.4e"
                  % (stem, B, c.K, kind,
                     float((gcu[kind] - gim[kind]).abs().max() / den),
                     float((gde[kind] - gim[kind]).abs().max() / den),
                     float(den)))
    except Exception:                                # noqa: BLE001
        print("[skip] %s: %s" % (stem, last()))
    finally:
        try:
            c.s.cudss_free()
            del c
        except Exception:                            # noqa: BLE001
            pass
        torch.cuda.empty_cache()

# ---------------------------------------------------------------- §D
print()
print("=" * 78)
print("§D 计数器：反向有没有让 factorize 翻倍（B=8，K=12，前向 12 次线性解）")
print("net     slots | factorize solve bwd_solve bwd_reuse bwd_refact | 梯度 vs slots=K+2")
for stem, fn in (("Modena", "Modena.inp"), ("ky4", "ky4.inp")):
    K = 12
    c = None
    try:
        c = Case(stem, fn, 8, K)
        ref = None
        for slots in (K + 2, 1, 2, 4, K):
            c.s.cudss_free()
            c.s.cudss_cache_max = max(8, slots)
            c.s.cudss_grad_slots = slots
            c.s._cudss_slot_rr = 0
            c.s.cudss_counters(reset=True)
            g = c.loss(ls="cudss", grad=True, slots=slots)
            cn = c.s.cudss_counters()
            if ref is None:
                ref = g
            print("%-7s %-5d | %-9d %-5d %-9d %-9d %-10d | %.3e"
                  % (stem, slots, cn["factorize"], cn["solve"], cn["bwd_solve"],
                     cn["bwd_reuse"], cn["bwd_refactorize"],
                     max(relmax(g[k], ref[k]) for k in g)))
    except Exception:                                # noqa: BLE001
        print("[skip] %s: %s" % (stem, last()))
    finally:
        try:
            c.s.cudss_free()
            del c
        except Exception:                            # noqa: BLE001
            pass
        torch.cuda.empty_cache()

# ---------------------------------------------------------------- §F
print()
print("=" * 78)
print("§F cudss_grad_refine（反向那次解的迭代精化步数）对**梯度精度**的影响")
print("   0 步能省两次 solve 地板（时间见 P3-B §5），这里看它值不值")
print("net       B  K  | refine | vs dense 解析梯度 (d / rh / r / ke)")
for stem, fn, K in NETS:
    B = 8
    c = None
    try:
        c = Case(stem, fn, B, K)
        gde = c.loss(ls="dense", grad=True)
        for gr in (0, 1, 2):
            c.s.cudss_free()
            c.s.cudss_grad_refine = gr
            g = c.loss(ls="cudss", grad=True, slots=SLOTS(K, B))
            print("%-9s %-2d %-3d| %-6d | %.3e %.3e %.3e %.3e"
                  % (stem, B, K, gr, relmax(g["d"], gde["d"]),
                     relmax(g["rh"], gde["rh"]), relmax(g["r"], gde["r"]),
                     relmax(g["ke"], gde["ke"])))
        c.s.cudss_grad_refine = 2
    except Exception:                                # noqa: BLE001
        print("[skip] %s: %s" % (stem, last()))
    finally:
        try:
            c.s.cudss_free()
            del c
        except Exception:                            # noqa: BLE001
            pass
        torch.cuda.empty_cache()

# ---------------------------------------------------------------- §E
print()
print("=" * 78)
print("§E 守卫（全部必须明确 raise，零静默降级）")
net, s = mk("Hanoi.inp")
d0, rh0 = boundary(net)
D, R = batchify(d0, rh0, 2)
w = torch.as_tensor(np.random.default_rng(0).normal(size=(2, net.N)), dtype=DT,
                    device=DEV)


def probe(name, fn):
    try:
        fn()
        print("  %-42s → **没有 raise** ← 不合格" % name)
    except Exception as e:                           # noqa: BLE001
        print("  %-42s → %s: %s"
              % (name, type(e).__name__, str(e)[:72].replace(chr(10), " ")))


def _second_order():
    d = torch.tensor(D, dtype=DT, device=DEV, requires_grad=True)
    o = solve_unrolled(s, d, R, K=4, assemble="csr", linear_solver="cudss")
    torch.autograd.grad((w * o["head_ft"]).sum(), d, create_graph=True)


probe("二阶导 create_graph=True", _second_order)
probe("cudss + assemble='dense'",
      lambda: solve_unrolled(s, torch.tensor(D, dtype=DT, device=DEV,
                                             requires_grad=True), R, K=2,
                             assemble="dense", linear_solver="cudss"))
f_h = os.path.join(NETDIR, "Hanoi.inp")
s32 = GGASolver(parse_inp(f_h), device=DEV, dtype=torch.float32, mode="dense",
                inp_path=f_h, dense_tank_bound_check=False)
probe("cudss + float32（可微通路）",
      lambda: solve_unrolled(s32, torch.tensor(D, dtype=torch.float32, device=DEV,
                                               requires_grad=True), R, K=2,
                             assemble="csr", linear_solver="cudss"))
scpu = GGASolver(parse_inp(f_h), device="cpu", dtype=DT, mode="dense",
                 inp_path=f_h, dense_tank_bound_check=False)
probe("cudss + device='cpu'（可微通路）",
      lambda: solve_unrolled(scpu, torch.tensor(D, dtype=DT, requires_grad=True),
                             R, K=2, assemble="csr", linear_solver="cudss"))


def _slots_gt_cache():
    s.cudss_free()
    s.cudss_cache_max = 4
    s.cudss_grad_slots = 8
    s._cudss_slot_rr = 0
    solve_unrolled(s, torch.tensor(D, dtype=DT, device=DEV, requires_grad=True),
                   R, K=3, assemble="csr", linear_solver="cudss")


probe("cudss_grad_slots > cudss_cache_max", _slots_gt_cache)
s.cudss_cache_max = 8
s.cudss_grad_slots = 1
print("P3A DONE")

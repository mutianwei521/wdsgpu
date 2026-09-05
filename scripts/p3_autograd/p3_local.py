# -*- coding: utf-8 -*-
"""P3 本机自检（cuDSS 用 fake_nvmath 替身，跑在本机 CUDA 上）。

查的是**逻辑与梯度公式**，不查性能（替身是 dense LU）：
  §1 梯度：cudss 通路 vs dense 通路（solve_unrolled，同一 K，逐参数）
  §2 计数器：slots=1 与 slots>=K 下的 factorize / bwd_reuse / bwd_refactorize
  §3 复用判据的杀伤力：slots 不够时结果仍正确（走重分解），
     以及"强行复用一次陈旧分解"会明显错（证明 §2 的复用不是走过场）
  §4 二阶导守卫：create_graph=True 明确 raise
  §5 solve()（带收敛判定的主入口）也可微，且与 dense 一致
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import fake_nvmath                                   # noqa: E402
fake_nvmath.install()
from dgga.parse import parse_inp                     # noqa: E402
from dgga.solver import GGASolver                    # noqa: E402
from dgga.autodiff import solve_unrolled             # noqa: E402

DEV = "cuda"
DT = torch.float64
NETDIR = os.path.join(ROOT, "networks", "public")
NETS = [("Hanoi", "Hanoi.inp"), ("Net3", "Net3.inp"), ("Modena", "Modena.inp")]


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


def batchify(d, rh, B, seed=0):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.8, 1.2, (B, 1)) * g.uniform(0.9, 1.1, (B, d.size))
    R = rh[None, :] + g.uniform(-1.0, 1.0, (B, rh.size))
    R = np.where(np.isnan(rh)[None, :], np.nan, R)
    return D, R


def run(s, D, R, KE, K, ls, w, need=("d", "rh", "r", "ke")):
    d = torch.tensor(D, dtype=DT, device=DEV, requires_grad="d" in need)
    rh = torch.tensor(np.nan_to_num(R), dtype=DT, device=DEV,
                      requires_grad="rh" in need)
    ke = torch.tensor(KE, dtype=DT, device=DEV, requires_grad="ke" in need)
    r = s.r_hw.clone().requires_grad_("r" in need)
    kw = dict(assemble="csr", linear_solver="cudss") if ls == "cudss" else {}
    out = solve_unrolled(s, d, rh, ke=ke, r_hw=r, K=K, **kw)
    L = (w * out["head_ft"]).sum() + (w[:s.L] * out["flow_cfs"]).sum() \
        if False else (w * out["head_ft"]).sum()
    L.backward()
    g = {"d": d.grad, "rh": rh.grad, "ke": ke.grad, "r": r.grad}
    return {k: (v.detach().clone() if v is not None else None) for k, v in g.items()}, \
        out["head_ft"].detach().clone()


def relmax(a, b):
    den = b.abs().max().clamp_min(1e-300)
    return float((a - b).abs().max() / den)


print("torch", torch.__version__, "| dev", torch.cuda.get_device_name(0))
print("=" * 78)
print("§1 梯度：cudss(替身) vs dense - solve_unrolled，同 K，max 相对差")
print("net       B  K | head     | grad d    grad rh   grad r    grad ke")
for stem, fn in NETS:
    net, s = mk(fn)
    d0, rh0 = boundary(net)
    KE = np.zeros(net.N)
    KE[s.junc_nodes[:max(1, s.Nj // 5)]] = 0.35
    for B in (1, 8):
        D, R = batchify(d0, rh0, B)
        K = 12
        g = np.random.default_rng(7)
        w = torch.tensor(g.normal(size=(B, net.N)), dtype=DT, device=DEV)
        gd, hd = run(s, D, R, KE, K, "dense", w)
        gc, hc = run(s, D, R, KE, K, "cudss", w)
        print("%-9s %-2d %-2d | %.3e | %.3e %.3e %.3e %.3e"
              % (stem, B, K, relmax(hc, hd), relmax(gc["d"], gd["d"]),
                 relmax(gc["rh"], gd["rh"]), relmax(gc["r"], gd["r"]),
                 relmax(gc["ke"], gd["ke"])))

print()
print("=" * 78)
print("§2 计数器（Modena B=4，K=10，前向 10 次线性解）")
net, s = mk("Modena.inp")
d0, rh0 = boundary(net)
KE = np.zeros(net.N)
KE[s.junc_nodes[:20]] = 0.35
D, R = batchify(d0, rh0, 4)
w = torch.tensor(np.random.default_rng(1).normal(size=(4, net.N)), dtype=DT, device=DEV)
K = 10
ref = None
print("slots cache_max | factorize solve bwd_solve bwd_reuse bwd_refact | 与 slots=K 的 grad 差")
for slots in (1, 2, 5, 10, 12):
    s.cudss_free()
    s.cudss_cache_max = max(8, slots)
    s.cudss_grad_slots = slots
    s._cudss_slot_rr = 0
    s.cudss_counters(reset=True)
    gc, hc = run(s, D, R, KE, K, "cudss", w)
    c = s.cudss_counters()
    if slots >= K and ref is None:
        ref = gc
    dif = "-" if ref is None else "%.3e" % max(relmax(gc[k], ref[k]) for k in gc)
    print("%-5d %-9d | %-9d %-5d %-9d %-9d %-10d | %s"
          % (slots, s.cudss_cache_max, c["factorize"], c["solve"], c["bwd_solve"],
             c["bwd_reuse"], c["bwd_refactorize"], dif))
# 补一次 slots=1 与 slots=K 的梯度对拍（ref 已是 slots=12 的）
s.cudss_free()
s.cudss_grad_slots = 1
s._cudss_slot_rr = 0
g1, _ = run(s, D, R, KE, K, "cudss", w)
print("slots=1 vs slots=12 的梯度 max 相对差 = %.3e"
      % max(relmax(g1[k], ref[k]) for k in g1))

print()
print("=" * 78)
print("§3 替身的杀伤力检验：若反向复用一次**陈旧**分解，梯度会不会明显错？")
# K 要取在"牛顿还没收敛"的段上：收敛后 A_k≈A_K，陈旧分解与正确分解本就一样，
# 那样的探针查不出任何东西（K=10 时实测差 9e-11 = GPU scatter_add 的抖动底）。
import dgga.solver as _sv                            # noqa: E402
_orig = GGASolver._cudss_adjoint


def _bad_adjoint(self, data, g, B, st, gen, slot):
    return _orig(self, data, g, B, st, st.get("gen"), slot)   # 永远"复用"


for Kp in (3, 4, 6, 10):
    s.cudss_free()
    s.cudss_cache_max = 16
    s.cudss_grad_slots = Kp
    s._cudss_slot_rr = 0
    gok, _ = run(s, D, R, KE, Kp, "cudss", w)      # slots=K：全复用，无重分解
    s.cudss_free()
    s.cudss_grad_slots = 1
    s._cudss_slot_rr = 0
    gre, _ = run(s, D, R, KE, Kp, "cudss", w)      # slots=1：陈旧的那些重分解
    s.cudss_free()
    s.cudss_grad_slots = 1
    s._cudss_slot_rr = 0
    GGASolver._cudss_adjoint = _bad_adjoint
    gbad, _ = run(s, D, R, KE, Kp, "cudss", w)     # 强行复用陈旧分解
    GGASolver._cudss_adjoint = _orig
    print("K=%-3d 正确重分解 vs 全复用 = %.3e   |   强行用陈旧分解 vs 全复用 = %.3e"
          % (Kp, max(relmax(gre[k], gok[k]) for k in gok),
             max(relmax(gbad[k], gok[k]) for k in gok)))

print()
print("=" * 78)
print("§4 二阶导守卫（backward 入口显式查 torch.is_grad_enabled）")
s.cudss_free()
s.cudss_grad_slots = 1
s._cudss_slot_rr = 0
d = torch.tensor(D, dtype=DT, device=DEV, requires_grad=True)
out = solve_unrolled(s, d, np.nan_to_num(R), ke=KE, K=3,
                     assemble="csr", linear_solver="cudss")
L = (w * out["head_ft"]).sum()
try:
    torch.autograd.grad(L, d, create_graph=True)
    print("create_graph=True：**没有 raise** ← 不合格")
except Exception as e:                               # noqa: BLE001
    print("create_graph=True →", type(e).__name__, "|",
          str(e)[:150].replace(chr(10), " "))
d.grad = None
out = solve_unrolled(s, d, np.nan_to_num(R), ke=KE, K=3,
                     assemble="csr", linear_solver="cudss")
(w * out["head_ft"]).sum().backward()
print("同一处一阶反向（create_graph=False）正常：|g|max = %.6e"
      % float(d.grad.abs().max()))

print()
print("=" * 78)
print("§5 solve()（带收敛判定的主入口）也可微，且与 dense 一致")
for stem, fn in (("Hanoi", "Hanoi.inp"), ("Modena", "Modena.inp")):
    net2, s2 = mk(fn)
    d2, r2 = boundary(net2)
    D2, R2 = batchify(d2, r2, 4, seed=3)
    KE2 = np.zeros(net2.N)
    KE2[s2.junc_nodes[:10]] = 0.35
    w2 = torch.tensor(np.random.default_rng(5).normal(size=(4, net2.N)),
                      dtype=DT, device=DEV)
    gg = {}
    for ls in ("dense", "cudss"):
        s2.cudss_free()
        s2.cudss_cache_max = 32
        s2.cudss_grad_slots = 20
        s2._cudss_slot_rr = 0
        s2.cudss_counters(reset=True)
        dd = torch.tensor(D2, dtype=DT, device=DEV, requires_grad=True)
        rr = torch.tensor(np.nan_to_num(R2), dtype=DT, device=DEV, requires_grad=True)
        kk = torch.tensor(KE2, dtype=DT, device=DEV, requires_grad=True)
        kw = dict(assemble="csr", linear_solver="cudss") if ls == "cudss" else {}
        o = s2.solve(dd, rr, ke_int=kk, **kw)
        (w2 * o["head_ft"]).sum().backward()
        gg[ls] = (dd.grad.clone(), rr.grad.clone(), kk.grad.clone(),
                  int(o["iters"].max()), s2.cudss_counters())
    c = gg["cudss"][4]
    print("%-8s iters dense=%d cudss=%d | grad d %.3e rh %.3e ke %.3e | fact=%d bwd_refact=%d"
          % (stem, gg["dense"][3], gg["cudss"][3],
             relmax(gg["cudss"][0], gg["dense"][0]),
             relmax(gg["cudss"][1], gg["dense"][1]),
             relmax(gg["cudss"][2], gg["dense"][2]),
             c["factorize"], c["bwd_refactorize"]))
print("P3 LOCAL DONE")

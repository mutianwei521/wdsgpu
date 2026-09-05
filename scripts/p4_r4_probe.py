# -*- coding: utf-8 -*-
"""p4_r4_probe.py - 展开路径 demand 梯度的可用性判据实测（P4 / 审计 R4）。

审计 R4 的原话：κ≥7e8 的网（City_D/ky4）上 unrolled 的 demand 梯度"重跑自己都差
1e-3~1e-2、未收敛 K 下两条通路范数差 16 倍"，要求给一个**明确结论**，别让调用方
拿着 6e-4 的相对差自己猜。审计那批数是在 GPU 上量的（自重跑抖动来自 atomicAdd 的
不定归约），本脚本在 **CPU** 上量三件与硬件无关、且是根因的东西：

  M1  κ₂(A)：收敛点上装配出来的那个 A 的 2-范数条件数（batch 第 0 个场景）。
      A 的捕获点是 torch.linalg.cholesky 的入参（与 check_symmetry.py 同一手法）。
  M2  K-稳定性：grad_d 在 K = Kc / Kc+5 / Kc+10 三档之间的相对变化
      （Kc = 该场景收敛迭代数）。展开路径的梯度是"截断 K 步的梯度"，它随 K 变，
      这是与线代通路无关的固有性质；κ 大时这个变化不收敛。
  M3  与 ImplicitGGASolve（隐式伴随、不动点梯度、独立的 scipy 稀疏 LU 通路）
      的相对差。这是"展开梯度到底逼近不逼近真梯度"的唯一硬参照。

判据结论写进 dgga/autodiff.py 的 solve_unrolled docstring 与 data/p4_guards_wip.txt。
用法：`python -X utf8 scripts/p4_r4_probe.py`（CPU，几分钟）。
"""

import os
import sys
import time
import warnings

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import parse_inp                                  # noqa: E402
from dgga.solver import GGASolver                                 # noqa: E402
from dgga.autodiff import ImplicitGGASolve, solve_unrolled        # noqa: E402

B = 8
SEED = 20260822

NETS = [("Hanoi", "public/Hanoi.inp"),
        ("Net3", "public/Net3.inp"),
        ("Modena", "public/Modena.inp"),
        ("rand_main_0009", "random_main/rand_0009.inp"),
        ("EXA6", "InpData/EXA6.inp"),
        ("ky3", "InpData/ky3.inp"),
        ("ky5", "InpData/ky5.inp"),
        ("ky4", "public/ky4.inp"),
        ("City_D", "realInpData/city_d.inp"),
        ("city_h", "InpData/city_h.inp")]


def build(inp_rel, b=B, seed=SEED):
    p = os.path.join(ROOT, "networks", inp_rel)
    net = parse_inp(p)
    s = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                  inp_path=p, dense_tank_bound_check=False)
    g = np.random.default_rng(seed)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = 0.5 * (np.asarray(net.tank_hmin)[:tn.size]
                         + np.asarray(net.tank_hmax)[:tn.size])
    rh0 = np.nan_to_num(rh0)
    D = torch.as_tensor(d0[None, :] * g.uniform(.9, 1.1, (b, d0.size)),
                        dtype=torch.float64)
    R = torch.as_tensor(np.repeat(rh0[None, :], b, 0)
                        + g.uniform(-1., 1., (b, rh0.size))
                        * (np.asarray(net.node_type) != 0)[None, :],
                        dtype=torch.float64)
    KE = torch.as_tensor(np.repeat(
        np.asarray(net.node_ke, dtype=np.float64)[None, :], b, 0),
        dtype=torch.float64)
    W = torch.as_tensor(g.normal(0, 1, (b, net.N)), dtype=torch.float64)
    return net, s, D, R, KE, W


def kappa_at_solution(s, D, R, KE):
    """收敛点上 A 的 κ₂（场景 0）。捕获点 = torch.linalg.cholesky 的入参。"""
    grab = []
    raw = torch.linalg.cholesky

    def chol(A, *a, **kw):
        grab.append(A.detach().clone())
        return raw(A, *a, **kw)

    torch.linalg.cholesky = chol
    try:
        with torch.no_grad():
            out = s.solve(D, R, ke_int=KE)
    finally:
        torch.linalg.cholesky = raw
    A = grab[-1][0]
    sv = torch.linalg.svdvals(A)
    return float(sv[0] / sv[-1]), int(out["iters"].max())


def grad_d_unrolled(s, D, R, KE, W, K):
    Dg = D.clone().requires_grad_(True)
    o = solve_unrolled(s, Dg, R, ke=KE, K=K)
    (o["head_ft"] * W).sum().backward()
    return Dg.grad.detach().clone()


def grad_d_implicit(s, D, R, KE, W):
    Dg = D.clone().requires_grad_(True)
    h, _q, _e = ImplicitGGASolve.apply(Dg, R, KE, s.r_hw, s,
                                       1e-12, 200, 3)
    (h * W).sum().backward()
    return Dg.grad.detach().clone()


def rel(a, b_):
    den = float(torch.max(a.abs().max(), b_.abs().max()).clamp_min(1e-300))
    return float((a - b_).abs().max()) / den


def main():
    print("=" * 118)
    print("P4/R4 - 展开路径 demand 梯度：κ、K-稳定性、与隐式伴随的差（CPU, B=%d）"
          % B)
    print("M2 的两列是 rel(K=Kc, Kc+5) 与 rel(Kc+5, Kc+10)；M3 是 rel(Kc+5, 隐式伴随)。")
    print("=" * 118)
    hdr = ("网", "Nj", "Kc", "κ₂(A)", "max|g_d|", "M2 Kc→+5", "M2 +5→+10",
           "M3 vs 隐式", "结论")
    print("%-16s %-5s %-4s %-11s %-11s %-11s %-11s %-11s %s" % hdr)
    rows = []
    for stem, rel_p in NETS:
        t0 = time.time()
        try:
            net, s, D, R, KE, W = build(rel_p)
            kap, Kc = kappa_at_solution(s, D, R, KE)
            g0 = grad_d_unrolled(s, D, R, KE, W, Kc)
            g5 = grad_d_unrolled(s, D, R, KE, W, Kc + 5)
            g10 = grad_d_unrolled(s, D, R, KE, W, Kc + 10)
            gi = grad_d_implicit(s, D, R, KE, W)
            m2a, m2b = rel(g0, g5), rel(g5, g10)
            m3 = rel(g5, gi)
            gmax = float(g5.abs().max())
            verdict = ("可用" if (m3 < 1e-6 and m2b < 1e-8) else
                       ("有条件" if (m3 < 1e-3 and m2b < 1e-5) else "不建议"))
            print("%-16s %-5d %-4d %-11.3e %-11.3e %-11.3e %-11.3e %-11.3e %s "
                  "(%.0fs)" % (stem, s.Nj, Kc, kap, gmax, m2a, m2b, m3,
                               verdict, time.time() - t0), flush=True)
            rows.append((stem, s.Nj, Kc, kap, gmax, m2a, m2b, m3, verdict))
            del s
        except Exception as e:                                   # noqa: BLE001
            print("%-16s ERR %s: %s" % (stem, type(e).__name__, str(e)[:80]),
                  flush=True)
    print("=" * 118)
    ok = [r for r in rows if r[8] == "可用"]
    no = [r for r in rows if r[8] == "不建议"]
    if ok:
        print("κ 上界（判为可用的网）: %.3e" % max(r[3] for r in ok))
    if no:
        print("κ 下界（判为不建议的网）: %.3e" % min(r[3] for r in no))
    print("不建议的网:", ", ".join(r[0] for r in no) or "无")

    # ---- §2 根因：RQtol 钳位支（q≈0 的死支）而不是 κ ----
    print("\n" + "=" * 118)
    print("§2 根因定位：不稳定坐标是不是贴着 RQtol 钳位支（q≈0 死支）的 junction")
    print("   钳位 = hgrad<RQtol 被抬到 RQtol（hydcoeffs.c:554-558），"
          "P=1/RQtol=1e7；解析导数在该支是常数，展开反传会把它逐迭代放大。")
    print("=" * 118)
    print("%-16s %-6s %-8s %-9s %-9s %-11s %-11s %s"
          % ("网", "L", "钳位支", "钳位占比", "不稳坐标", "其中邻钳位",
             "邻钳位占比", "rel(+5→+10)"))
    for stem, rel_p in NETS:
        try:
            net, s, D, R, KE, W = build(rel_p)
            with torch.no_grad():
                out = s.solve(D, R, ke_int=KE)
                q = out["flow_cfs"]
                Kc = int(out["iters"].max())
                P_pipe, _ = s._pipe_PY(q)
                P_tcv, _ = s._tcv_PY(q)
                P = torch.where(s.is_tcv, P_tcv, P_pipe)
                clamp = (P >= (1.0 / s.rqtol) * (1 - 1e-12)) & ~s.closed_dense
            g5 = grad_d_unrolled(s, D, R, KE, W, Kc + 5)
            g10 = grad_d_unrolled(s, D, R, KE, W, Kc + 10)
            den = torch.max(g5.abs(), g10.abs()).clamp_min(1e-30)
            unst = ((g5 - g10).abs() / den > 1e-6)          # [B,N] 逐坐标
            # 邻接钳位支的节点集合
            cl = clamp.any(0).cpu().numpy()
            adj = np.zeros(net.N, dtype=bool)
            adj[np.asarray(net.link_n1)[cl]] = True
            adj[np.asarray(net.link_n2)[cl]] = True
            u = unst.any(0).cpu().numpy() & (np.asarray(net.node_type) == 0)
            nu = int(u.sum())
            na = int((u & adj).sum())
            print("%-16s %-6d %-8d %-9.3f %-9d %-11d %-11s %.3e"
                  % (stem, net.L, int(cl.sum()), cl.sum() / max(net.L, 1),
                     nu, na, ("%.3f" % (na / nu)) if nu else "-",
                     rel(g5, g10)), flush=True)
            del s
        except Exception as e:                               # noqa: BLE001
            print("%-16s ERR %s: %s" % (stem, type(e).__name__, str(e)[:70]),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

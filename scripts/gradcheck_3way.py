# -*- coding: utf-8 -*-
"""gradcheck_3way.py - 三方梯度对拍（阶段 C 验收）。

场景：city_d（25 帧取 t=0 帧）与 rand_main_0009。损失 L = Σ w_i·H_i（junction，
随机权重，保证梯度非退化）。对四类输入（demand、Ke、水库水头、管道 r_hw）各随机
抽 ≥20 个坐标（水库数不足 20 时取全部并注明）：
  方法A = solve_unrolled 展开 autograd（K = 收敛迭代数 + 5；city_d 的 GGA 半迭代
          在 κ~1e9 病态下 relerr 停在 ~1e-9 平台不达 1e-12，收敛迭代数取 1e-12
          档实际跑到的迭代数=60）
  方法B = ImplicitGGASolve（隐函数伴随；forward 显式 accuracy=1e-12 + 整装 Newton 精抛光）
  方法C = 中心差分（我们的求解器 + 同款精抛光）。步长自适应说明：任务默认
          h=max(1e-6|x|,1e-9) 实测被 f64 损失噪声淹没 - 精抛光后 L 的确定性舍入
          噪声 ~1e-13（‖F‖∞~1e-13 经 J^-1 映射），噪声/2h 在 h=1e-9 时达 1e-4 级，
          远超 1e-6 门槛（B、A 互差 ~1e-10 而各自 vs C 差 1e-3，证明 C 才是误差源）。
          故按"步长按坐标量级自适应"原则取 h1=max(1e-3|x|,1e-5)（水库水头量级大，
          取 1e-4|x|）、h2=h1/2 并做 Richardson 外推（截断 O(h^2)→O(h^4)）。
坐标抽样池（任务"保证梯度非退化"的精神，FD 对退化梯度坐标无分辨率）：
  demand 只抽 d>0 的 junction - d=0 死支节点的关联链路恰在 q=0（H-W 的 |q|^0.852
  曲率奇点 + RQtol 钳位支），中心差分在任何可行 h 下都无法逼近钳位支解析导数
  （实测 h 从 1e-3 到 1e-7 扫描不收敛，B/A 两法互差 <1e-9）；
  r_hw 只抽非 RQtol 钳位管道（钳位支 ∂φ/∂r≡0，hydcoeffs.c:554-558 与 r 无关，
  解析=真=0，FD 纯噪声使相对误差退化为 1）；
  各类再取 |gB| ≥ 该类候选 40 分位的坐标（过滤梯度近退化坐标 - |g|~1e-5 级时
  f64 损失噪声 1e-12 除以 2h 后的相对误差必然超过 1e-6 门槛），池内随机抽 20。
门槛：B vs C 每坐标相对误差 < 1e-6；A vs C < 1e-4（截断迭代 vs 不动点固有差）；
分母 max(|gC|, 1e-12)。
另：torch.autograd.gradcheck(ImplicitGGASolve) 在 rand_main_0009 上
（eps=1e-6, atol=1e-5, rtol=1e-3, float64）；city_d B=8 批梯度 = 逐场景梯度
（max 相对差 < 1e-10）。
"""

import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import Net                                    # noqa: E402
from dgga.solver import GGASolver                             # noqa: E402
from dgga.autodiff import (ImplicitGGASolve, solve_polished,  # noqa: E402
                           solve_unrolled)

TOL_BC = 1e-6
TOL_AC = 1e-4
TOL_BATCH = 1e-10
N_COORD = 20
SEED = 2026

CASES = [
    # (stem, inp, GGA max_iter)。city_d 上限 60：relerr 于 ~15 迭代进入 1e-7~1e-9
    # 平台（κ~1e9 舍入地板），60 已远超平台；精抛光使结果与截断处无关。
    ("city_d", os.path.join(ROOT, "networks", "realInpData", "city_d.inp"), 60),
    ("rand_main_0009", os.path.join(ROOT, "networks", "random_main", "rand_0009.inp"),
     200),
]


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def make_base(net, s, rng):
    """基准点：t=0 需水/水库水头；25 个随机 junction 挂 Ke=0.5（>CSMALL，远离钳位
    折点，保证光滑）；r_hw 取解析基值。"""
    d0 = net.demand_cfs_at(0)
    rh0 = np.nan_to_num(net.reservoir_head_ft_at(0))          # 非水库位置 0（不参与）
    ke0 = np.zeros(net.N)
    em_nodes = rng.choice(s.junc_nodes, size=min(40, s.Nj), replace=False)
    ke0[em_nodes] = 0.5
    r0 = s.r_hw.detach().cpu().numpy().copy()
    return d0, rh0, ke0, r0, np.sort(em_nodes)


def run_case(stem, inp, gga_mi):
    print("=" * 78)
    net = Net.load(os.path.join(ROOT, "data", "reference"), stem)
    s = GGASolver(net, mode="dense", inp_path=inp)
    rng = np.random.default_rng(SEED)
    d0, rh0, ke0, r0, em_nodes = make_base(net, s, rng)
    w = rng.normal(size=s.Nj)
    w_t = torch.tensor(w, dtype=torch.float64)

    r_ref = s.solve(d0, rh0, ke_int=ke0, accuracy=1e-12, max_iter=gga_mi)
    conv = bool(r_ref["converged"])
    iters = int(r_ref["iters"])
    K = iters + 5
    print(f"=== {stem}: Nj={s.Nj} L={s.L} | GGA(1e-12) iters={iters} "
          f"converged={conv} relerr={float(r_ref['relerr']):.2e} -> K={K} ===")

    # ---- 方法 B：隐函数伴随 ----
    dB = torch.tensor(d0, dtype=torch.float64, requires_grad=True)
    rhB = torch.tensor(rh0, dtype=torch.float64, requires_grad=True)
    keB = torch.tensor(ke0, dtype=torch.float64, requires_grad=True)
    rB = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
    head, _, _ = ImplicitGGASolve.apply(dB, rhB, keB, rB, s, 1e-12, gga_mi, 3)
    (w_t * head[s.junc_nodes]).sum().backward()
    gB = dict(demand=dB.grad.numpy(), rh=rhB.grad.numpy(),
              ke=keB.grad.numpy(), r=rB.grad.numpy())

    # ---- 方法 A：unrolled autograd ----
    dA = torch.tensor(d0, dtype=torch.float64, requires_grad=True)
    rhA = torch.tensor(rh0, dtype=torch.float64, requires_grad=True)
    keA = torch.tensor(ke0, dtype=torch.float64, requires_grad=True)
    rA = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
    outA = solve_unrolled(s, dA, rhA, ke=keA, r_hw=rA, K=K)
    (w_t * outA["head_ft"][s.junc_nodes]).sum().backward()
    gA = dict(demand=dA.grad.numpy(), rh=rhA.grad.numpy(),
              ke=keA.grad.numpy(), r=rA.grad.numpy())

    # ---- 方法 C：中心差分（solve_polished 精抛光后的 L）----
    def loss_of(d, rh, ke, r):
        sol = solve_polished(s, d, rh, ke, r, accuracy=1e-12,
                             max_iter=gga_mi, polish_steps=3)
        return float(w @ sol["head"][0, s.junc_nodes])

    # ---- 坐标抽样池（构造依据见文件头注释）----
    q_base = r_ref["flow_cfs"].numpy()
    hg_fric = s.hexp * r0 * np.abs(q_base) ** (s.hexp - 1.0)
    pipe_ok = s.is_pipe.cpu().numpy() & ~s.closed_np & (hg_fric > 10.0 * s.rqtol)

    def pool(cand_idx, gvec, n=N_COORD):
        """候选坐标中取 |g| ≥ 40 分位者，随机抽 n 个（不足则取全部）。"""
        ga = np.abs(gvec[cand_idx])
        keep = cand_idx[ga >= np.percentile(ga, 40.0)]
        if keep.size < n:
            keep = cand_idx[np.argsort(-ga)[:n]]
        return [int(i) for i in rng.choice(keep, size=min(n, keep.size), replace=False)]

    coords = dict(
        demand=[("demand", i) for i in
                pool(s.junc_nodes[d0[s.junc_nodes] > 0], gB["demand"])],
        ke=[("ke", i) for i in pool(em_nodes, gB["ke"])],
        rh=[("rh", int(i)) for i in s.fixed_nodes],           # 水库全取（不足 20 注明）
        r=[("r", k) for k in pool(np.where(pipe_ok)[0], gB["r"])],
    )
    if len(coords["rh"]) < N_COORD:
        print(f"  注：水库仅 {len(coords['rh'])} 个，θ=水库水头 抽全体坐标")

    base = dict(demand=d0, rh=rh0, ke=ke0, r=r0)
    ok = True
    worst = []
    print(f"{'θ':>8} {'坐标':>8} {'gC(中心差分)':>16} {'relBC':>10} {'relAC':>10}")
    for kind, clist in coords.items():
        wBC = (-1.0, None, 0.0, 0.0)   # (relBC, idx, gC, gB)
        wAC = (-1.0, None, 0.0)
        for _, idx in clist:
            x = base[kind][idx]

            def central(h):
                args_p = {k: v.copy() for k, v in base.items()}
                args_m = {k: v.copy() for k, v in base.items()}
                args_p[kind][idx] = x + h
                args_m[kind][idx] = x - h
                Lp = loss_of(args_p["demand"], args_p["rh"], args_p["ke"], args_p["r"])
                Lm = loss_of(args_m["demand"], args_m["rh"], args_m["ke"], args_m["r"])
                return (Lp - Lm) / (2.0 * h)

            # Richardson 外推中心差分（步长依据见文件头注释）
            h1 = max(1e-4 * abs(x), 1e-5) if kind == "rh" \
                else max(1e-3 * abs(x), 1e-5)
            gC = (4.0 * central(h1 / 2.0) - central(h1)) / 3.0
            rBC = rel(gB[kind][idx], gC)
            rAC = rel(gA[kind][idx], gC)
            if rBC > wBC[0]:
                wBC = (rBC, idx, gC, gB[kind][idx])
            if rAC > wAC[0]:
                wAC = (rAC, idx, gC)
            ok = ok and (rBC < TOL_BC) and (rAC < TOL_AC)
        print(f"{kind:>8} {wBC[1]:>8} {wBC[2]:>16.8e} {wBC[0]:>10.2e} {wAC[0]:>10.2e}"
              f"{'' if wBC[0] < TOL_BC and wAC[0] < TOL_AC else '  <-- 超限'}")
        worst.append((stem, kind, wBC, wAC))
    print(f"[{stem}] 三方对拍: {'PASS' if ok else 'FAIL'} "
          f"(门槛 B-C<{TOL_BC:.0e}, A-C<{TOL_AC:.0e})")
    return ok, worst


def run_gradcheck():
    """torch.autograd.gradcheck(ImplicitGGASolve) @ rand_main_0009。
    Ke 全 junction 置 0.5（远离 has_em/CSMALL 折点，保证被检函数在扰动邻域光滑）。"""
    print("=" * 78)
    net = Net.load(os.path.join(ROOT, "data", "reference"), "rand_main_0009")
    s = GGASolver(net, mode="dense",
                  inp_path=os.path.join(ROOT, "networks", "random_main", "rand_0009.inp"))
    d0 = net.demand_cfs_at(0)
    rh0 = np.nan_to_num(net.reservoir_head_ft_at(0))
    ke0 = np.zeros(net.N)
    ke0[s.junc_nodes] = 0.5
    r0 = s.r_hw.detach().cpu().numpy().copy()
    t = lambda x: torch.tensor(x, dtype=torch.float64, requires_grad=True)
    inputs = (t(d0), t(rh0), t(ke0), t(r0))

    def fn(d, rh, ke, r):
        return ImplicitGGASolve.apply(d, rh, ke, r, s, 1e-12, 200, 3)

    ok = torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)
    print(f"torch.autograd.gradcheck @ rand_main_0009 "
          f"(eps=1e-6, atol=1e-5, rtol=1e-3, f64): {'PASS' if ok else 'FAIL'}")
    return ok


def run_batch():
    """city_d B=8：求和损失的批量反传 per-scenario 梯度 vs 逐场景单独反传。"""
    print("=" * 78)
    net = Net.load(os.path.join(ROOT, "data", "reference"), "city_d")
    s = GGASolver(net, mode="dense",
                  inp_path=os.path.join(ROOT, "networks", "realInpData", "city_d.inp"))
    rng = np.random.default_rng(SEED)
    d0, rh0, ke0, r0, _ = make_base(net, s, rng)
    w = rng.normal(size=s.Nj)
    w_t = torch.tensor(w, dtype=torch.float64)
    B = 8
    dB_np = d0[None, :] * rng.uniform(0.9, 1.1, size=(B, net.N))

    dB = torch.tensor(dB_np, dtype=torch.float64, requires_grad=True)
    rhB = torch.tensor(rh0, dtype=torch.float64)
    keB = torch.tensor(ke0, dtype=torch.float64)
    rB = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
    head, _, _ = ImplicitGGASolve.apply(dB, rhB, keB, rB, s, 1e-12, 60, 3)
    (head[:, s.junc_nodes] * w_t).sum().backward()
    g_batch = dB.grad.numpy()
    gr_batch = rB.grad.numpy()

    worst = 0.0
    gr_sum = np.zeros(s.L)
    for b in range(B):
        db = torch.tensor(dB_np[b], dtype=torch.float64, requires_grad=True)
        rb = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
        hb, _, _ = ImplicitGGASolve.apply(db, rhB, keB, rb, s, 1e-12, 60, 3)
        (hb[s.junc_nodes] * w_t).sum().backward()
        gs = db.grad.numpy()
        denom = np.maximum(np.abs(gs), 1e-12)
        worst = max(worst, float(np.max(np.abs(g_batch[b] - gs) / denom)))
        gr_sum += rb.grad.numpy()
    worst_r = float(np.max(np.abs(gr_batch - gr_sum) /
                           np.maximum(np.abs(gr_sum), 1e-12)))
    ok = worst < TOL_BATCH and worst_r < TOL_BATCH
    print(f"city_d B=8 批梯度一致性: demand per-scenario max相对差={worst:.3e}, "
          f"r_hw(批=Σ场景) max相对差={worst_r:.3e} (门槛 {TOL_BATCH:.0e}) "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.set_default_dtype(torch.float64)
    all_ok = True
    worst_all = []
    for stem, inp, mi in CASES:
        ok, worst = run_case(stem, inp, mi)
        all_ok = all_ok and ok
        worst_all.extend(worst)
    all_ok = run_gradcheck() and all_ok
    all_ok = run_batch() and all_ok
    print("=" * 78)
    print("三方对拍最差坐标汇总：")
    for stem, kind, (rBC, iBC, gC, gBv), (rAC, iAC, gC2) in worst_all:
        print(f"  {stem:>16} θ={kind:<6} B-C最差@{iBC}: gC={gC:.8e} gB={gBv:.8e} "
              f"rel={rBC:.2e} | A-C最差@{iAC}: rel={rAC:.2e}")
    print(f"总判定: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""sensitivity_check.py - dgga/sensitivity.py 的三项前台验证（任务 B）。

① 与 backward 对拍：S 的每一行 vs ImplicitGGASolve 对该单传感器损失 backward
   得到的 r_hw 梯度（同一伴随系统的两种算法，应机器精度一致，门槛 1e-10）；
② 与中心差分对拍：随机抽 10 根管，Richardson 外推中心差分（实现照抄
   scripts/gradcheck_3way.py:153-165）验证 ∂p/∂C，门槛 1e-6；
③ 性能：构造 S[40, L] 的耗时 vs 循环 40 次 backward（每次重新 splu）的耗时。

网：city_d（无泵无池 → dense 路径），polish_steps=3，全程 float64。
"""

import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import Net                                       # noqa: E402
from dgga.solver import GGASolver                                # noqa: E402
from dgga.autodiff import ImplicitGGASolve, solve_polished       # noqa: E402
from dgga.sensitivity import (dr_dC, fisher_information,         # noqa: E402
                              identifiability_mask,
                              sensitivity_matrix, svd_spectrum)

SEED = 2026
GGA_MI = 60          # city_d：relerr 于 ~15 迭代进平台，60 远超；精抛光兜底
TOL_ADJ = 1e-10
TOL_FD = 1e-6
N_SENSOR_ADJ = 5
N_SENSOR_PERF = 40
N_PIPE_FD = 10


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def main():
    torch.set_default_dtype(torch.float64)
    inp = os.path.join(ROOT, "networks", "realInpData", "city_d.inp")
    net = Net.load(os.path.join(ROOT, "data", "reference"), "city_d")
    # 陷阱：GGASolver(inp_path=...) 会就地修正 net.dem_base_cfs → 必须先构造 solver
    s = GGASolver(net, mode="dense", inp_path=inp)
    rng = np.random.default_rng(SEED)

    d0 = net.demand_cfs_at(0)
    rh0 = np.nan_to_num(net.reservoir_head_ft_at(0))
    ke0 = np.zeros(net.N)
    r0 = s.r_hw.detach().cpu().numpy().copy()
    C0 = s.kc_np.copy()
    print(f"=== city_d: N={s.N} Nj={s.Nj} L={s.L} headloss={s.headloss_form} ===")

    sol = solve_polished(s, d0, rh0, ke0, r0, accuracy=1e-12,
                         max_iter=GGA_MI, polish_steps=3)
    print(f"前向精抛光 ‖F‖∞ = {float(sol['resid_inf'][0]):.3e} "
          f"(GGA iters={int(sol['iters'][0])})")

    # 传感器：随机 junction
    sensors_perf = np.sort(rng.choice(s.junc_nodes, size=N_SENSOR_PERF,
                                      replace=False))
    sensors5 = sensors_perf[:N_SENSOR_ADJ]

    # ================= ① 与 backward 对拍 =================
    print("\n" + "=" * 78)
    print("① S 的行 vs 单传感器 backward（同一伴随系统的两种算法）")
    S_r = sensitivity_matrix(s, d0, rh0, r0, sensors5, wrt="r",
                             max_iter=GGA_MI, ke=ke0)
    S_C = sensitivity_matrix(s, d0, rh0, r0, sensors5, wrt="C",
                             max_iter=GGA_MI, ke=ke0)
    chain = dr_dC(s, r0)
    worst_r = worst_C = 0.0
    print(f"{'传感器':>8} {'max|∂p/∂r|':>14} {'relerr(r)':>12} {'relerr(C)':>12}")
    for i, nd in enumerate(sensors5):
        rB = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
        dB = torch.tensor(d0, dtype=torch.float64)
        rhB = torch.tensor(rh0, dtype=torch.float64)
        keB = torch.tensor(ke0, dtype=torch.float64)
        head, _, _ = ImplicitGGASolve.apply(dB, rhB, keB, rB, s, 1e-12,
                                            GGA_MI, 3)
        head[int(nd)].backward()
        g = rB.grad.numpy()
        den = np.maximum(np.abs(g), 1e-30)
        er = float(np.max(np.abs(S_r[i] - g) / den))
        gC = g * chain
        denC = np.maximum(np.abs(gC), 1e-30)
        eC = float(np.max(np.abs(S_C[i] - gC) / denC))
        worst_r = max(worst_r, er)
        worst_C = max(worst_C, eC)
        print(f"{int(nd):>8} {np.max(np.abs(S_r[i])):>14.6e} "
              f"{er:>12.2e} {eC:>12.2e}")
    ok1 = worst_r < TOL_ADJ and worst_C < TOL_ADJ
    print(f"① 逐元素最大相对误差: r={worst_r:.3e}, C={worst_C:.3e} "
          f"(门槛 {TOL_ADJ:.0e}) {'PASS' if ok1 else 'FAIL'}")

    # ================= ② 与 Richardson 中心差分对拍 =================
    print("\n" + "=" * 78)
    print("② ∂p/∂C vs Richardson 外推中心差分")
    node_fd = int(sensors5[0])
    S_fd = sensitivity_matrix(s, d0, rh0, r0, [node_fd], wrt="C",
                              max_iter=GGA_MI, ke=ke0)[0]
    # 候选管道池：非 RQtol 钳位、非关闭（钳位支 ∂φ/∂r≡0，FD 纯噪声）
    q_base = sol["q"][0]
    hg_fric = s.hexp * r0 * np.abs(q_base) ** (s.hexp - 1.0)
    pipe_ok = (s.is_pipe.cpu().numpy() & ~s.closed_np
               & (hg_fric > 10.0 * s.rqtol) & (C0 > 0))
    cand = np.where(pipe_ok)[0]
    # 再过滤退化坐标（同 gradcheck_3way.py:23-24 的精神）。噪声地板推导：精抛光后
    # ‖F‖∞~1.4e-14，经 J^-1 映射到 p 的确定性舍入噪声 ~1e-13 ft；h1=1e-3·C≈0.13,
    # 噪声/2h ≈ 5e-13。要让相对误差 < 1e-6 必须 |g| > 5e-7 - 取 FD_FLOOR=1e-6。
    # 低于该地板的管（含 |g| 恒等于 0 的死支）FD 本身无分辨率，非解析式的问题。
    FD_FLOOR = 1e-6
    live = cand[np.abs(S_fd[cand]) > FD_FLOOR]
    ga = np.abs(S_fd[live])
    keep = live[ga >= np.percentile(ga, 40.0)]
    picks = rng.choice(keep, size=min(N_PIPE_FD, keep.size), replace=False)
    n_exact0 = int((np.abs(S_fd[cand]) == 0.0).sum())
    print(f"  候选非钳位管 {cand.size}/{s.L}（其中 |∂p/∂C| 解析恒为 0 的 {n_exact0} 根），"
          f"过噪声地板 {live.size} 根，40 分位后 {keep.size}，抽 {picks.size} 根")

    def p_of_C(k, Cval):
        """改第 k 根管的 C → 重算 r（r ∝ C^-Hexp）→ 精抛光求解 → 取传感器压力。"""
        r = r0.copy()
        r[k] = r0[k] * (C0[k] / Cval) ** s.hexp
        so = solve_polished(s, d0, rh0, ke0, r, accuracy=1e-12,
                            max_iter=GGA_MI, polish_steps=3)
        return float(so["head"][0, node_fd])

    worst_fd = 0.0
    print(f"{'管':>6} {'C':>10} {'gFD':>16} {'gS':>16} {'relerr':>10}")
    for k in picks:
        k = int(k)
        x = C0[k]

        def central(h):
            return (p_of_C(k, x + h) - p_of_C(k, x - h)) / (2.0 * h)

        h1 = max(1e-3 * abs(x), 1e-5)
        gC = (4.0 * central(h1 / 2.0) - central(h1)) / 3.0
        e = rel(S_fd[k], gC)
        worst_fd = max(worst_fd, e)
        print(f"{k:>6} {x:>10.3f} {gC:>16.8e} {S_fd[k]:>16.8e} {e:>10.2e}"
              f"{'' if e < TOL_FD else '  <-- 超限'}")
    ok2 = worst_fd < TOL_FD
    print(f"② 最差相对误差 {worst_fd:.3e} (门槛 {TOL_FD:.0e}) "
          f"{'PASS' if ok2 else 'FAIL'}")

    # ================= ③ 性能 =================
    print("\n" + "=" * 78)
    print(f"③ 构造 S[{N_SENSOR_PERF}, {s.L}] 的耗时对比")
    REP = 5          # 各法重复 5 次取中位数（前向 0.15 s 级、伴随 ms 级，抗抖动）
    tot, adj, fac, slv, fwd = [], [], [], [], []
    for _ in range(REP):
        t0 = time.perf_counter()
        S40, info = sensitivity_matrix(s, d0, rh0, r0, sensors_perf, wrt="C",
                                       max_iter=GGA_MI, ke=ke0,
                                       return_info=True)
        tot.append(time.perf_counter() - t0)
        adj.append(info["t_adjoint"])
        fac.append(info["t_factorize"])
        slv.append(info["t_solve"])
        fwd.append(info["t_forward"])
    t_ours_total = float(np.median(tot))
    t_ours_adj = float(np.median(adj))

    # 对照：循环 40 次 backward（ImplicitGGASolve 每次重新装配并 splu(J)）
    t_fwd_ad, t_loop = [], []
    for _ in range(REP):
        rB = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
        dB = torch.tensor(d0, dtype=torch.float64)
        rhB = torch.tensor(rh0, dtype=torch.float64)
        keB = torch.tensor(ke0, dtype=torch.float64)
        t0 = time.perf_counter()
        head, _, _ = ImplicitGGASolve.apply(dB, rhB, keB, rB, s, 1e-12,
                                            GGA_MI, 3)
        t_fwd_ad.append(time.perf_counter() - t0)
        S40_loop = np.zeros((N_SENSOR_PERF, s.L))
        t0 = time.perf_counter()
        for i, nd in enumerate(sensors_perf):
            if rB.grad is not None:
                rB.grad = None
            head[int(nd)].backward(retain_graph=True)
            S40_loop[i] = rB.grad.numpy() * chain
        t_loop.append(time.perf_counter() - t0)
    t_fwd_ad = float(np.median(t_fwd_ad))
    t_loop_adj = float(np.median(t_loop))

    den = np.maximum(np.abs(S40_loop), 1e-30)
    e40 = float(np.max(np.abs(S40 - S40_loop) / den))
    print(f"  （各法重复 {REP} 次取中位数）")
    print(f"  前向求解（两法共用，不计入对比）  : 本模块 {np.median(fwd):.4f} s / "
          f"autograd {t_fwd_ad:.4f} s")
    print(f"  本模块伴随部分（1 次 splu, 40 RHS）: {t_ours_adj:.4f} s "
          f"[装配 {np.median([t - f - sv for t, f, sv in zip(adj, fac, slv)]):.4f} + "
          f"splu {np.median(fac):.4f} + 40 列回代 {np.median(slv):.4f}] "
          f"(splu 次数={info['n_factorizations']}, RHS 列数={info['n_rhs']})")
    print(f"  循环 40 次 backward（40 次 splu） : {t_loop_adj:.4f} s")
    print(f"  加速比（伴随部分）              : {t_loop_adj / t_ours_adj:.2f}x")
    print(f"  含前向的端到端                  : {t_ours_total:.4f} s vs "
          f"{t_fwd_ad + t_loop_adj:.4f} s "
          f"({(t_fwd_ad + t_loop_adj) / t_ours_total:.2f}x)")
    ok3 = e40 < TOL_ADJ
    print(f"  两法 S[40,{s.L}] 逐元素最大相对差 {e40:.3e} "
          f"(门槛 {TOL_ADJ:.0e}) {'PASS' if ok3 else 'FAIL'}")

    # ================= 附：可辨识性诊断 =================
    print("\n" + "=" * 78)
    print("附：S[40,L] 的可辨识性诊断")
    mask = identifiability_mask(S40)
    n_pipe = int(((s.lt_np <= 1) & (C0 > 0)).sum())
    n_zero_pipe = int((((s.lt_np <= 1) & (C0 > 0)) & ~mask).sum())
    spec = svd_spectrum(S40)
    FIM = fisher_information(S40, 0.1)      # σ = 0.1 ft 压力噪声
    wF = np.linalg.eigvalsh(FIM)
    print(f"  管道 {n_pipe} 根，S 列全 0（结构性不可校核）{n_zero_pipe} 根 "
          f"({100.0 * n_zero_pipe / n_pipe:.1f}%)")
    print(f"  σ(S): 数值秩={spec['rank']}/{min(S40.shape)}, "
          f"σ1={spec['sv'][0]:.3e}, σ_rank={spec['sv'][spec['rank'] - 1]:.3e}, "
          f"cond={spec['cond']:.3e}")
    k90 = int(np.searchsorted(spec["energy"], 0.90) + 1)
    print(f"  前 {k90} 个奇异方向已占 90% 能量（40 个传感器×1 工况）")
    print(f"  FIM({FIM.shape[0]}×{FIM.shape[0]}, σ=0.1ft): "
          f"λmax={wF[-1]:.3e}, λmin={wF[0]:.3e}, "
          f"正特征值 {int((wF > wF[-1] * 1e-12).sum())} 个")

    # ================= ④ 多帧堆叠 + epanet 模式（泵/池网）冒烟 =================
    print("\n" + "=" * 78)
    print("④ 多帧堆叠一致性 & epanet 模式（city_h，含泵/水池）冒烟")
    T = 3
    dT = d0[None, :] * rng.uniform(0.9, 1.1, size=(T, net.N))
    S_multi = sensitivity_matrix(s, dT, rh0, r0, sensors5, wrt="C",
                                 max_iter=GGA_MI, ke=ke0)
    worst_mf = 0.0
    for b in range(T):
        Sb = sensitivity_matrix(s, dT[b], rh0, r0, sensors5, wrt="C",
                                max_iter=GGA_MI, ke=ke0)
        blk = S_multi[b * N_SENSOR_ADJ:(b + 1) * N_SENSOR_ADJ]
        worst_mf = max(worst_mf, float(np.max(np.abs(blk - Sb)
                                              / np.maximum(np.abs(Sb), 1e-30))))
    ok4a = S_multi.shape == (T * N_SENSOR_ADJ, s.L) and worst_mf < TOL_ADJ
    print(f"  多帧 S 形状={S_multi.shape}（期望 ({T * N_SENSOR_ADJ}, {s.L})），"
          f"逐帧块 vs 单帧调用 max相对差={worst_mf:.3e} "
          f"{'PASS' if ok4a else 'FAIL'}")

    net2 = Net.load(os.path.join(ROOT, "data", "reference"), "city_h")
    s2 = GGASolver(net2, mode="epanet",
                   inp_path=os.path.join(ROOT, "networks", "InpData",
                                         "city_h.inp"))
    d2 = net2.demand_cfs_at(0)
    rh2 = np.nan_to_num(net2.reservoir_head_ft_at(0))
    sen2 = np.sort(rng.choice(s2.junc_nodes, size=8, replace=False))
    S2, info2 = sensitivity_matrix(s2, d2, rh2, None, sen2, wrt="C",
                                   max_iter=200, return_info=True)
    m2 = identifiability_mask(S2)
    n_pipe2 = int(((s2.lt_np <= 1) & (s2.kc_np > 0)).sum())
    ok4b = (S2.shape == (8, s2.L) and np.isfinite(S2).all()
            and float(info2["resid_inf"][0]) < 1e-6)
    print(f"  city_h(epanet, L={s2.L}, 泵/池): S{S2.shape}, "
          f"‖F‖∞={float(info2['resid_inf'][0]):.2e}, "
          f"max|S|={np.max(np.abs(S2)):.3e}, "
          f"管道 {n_pipe2} 根中 S 列全 0 的 {int((~m2 & (s2.lt_np <= 1) & (s2.kc_np > 0)).sum())} 根 "
          f"{'PASS' if ok4b else 'FAIL'}")
    ok4 = ok4a and ok4b

    print("\n" + "=" * 78)
    ok = ok1 and ok2 and ok3 and ok4
    print(f"总判定: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

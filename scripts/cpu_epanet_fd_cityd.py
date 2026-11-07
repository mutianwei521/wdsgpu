# -*- coding: utf-8 -*-
"""cpu_epanet_fd_cityd.py - City D 上原生 EPANET 2.2 有限差分的计时（WR 大修，City D 保留判据，2026-10-23）。

协议（沿用 cpu_fair_scaling.py v2 的公平原则）：
  题面    City D，B=64 个随机需水工况（需水 ×U[0.7,1.3]，t=0 水库水头），dopt40 的 40 个传感器，
          perpipe 真值 seed=7 + σ=0.1 ft 噪声的压力观测；损失 = 压力残差 MSE + 1e-4·正则，参数为 H-W 系数 C。
  参数    n=432：全部自由管（校准题面）；n=60：随机 60 根自由管，代表 GPU 反演循环里 60 个漏损候选的
          参数规模（每次扰动都是一次完整求解，成本与参数种类无关）。
  EPANET  最强常规用法：EN_openH 一次，基解 EN_initH(10) 冷启动，扰动解 EN_initH(0) 从同工况基解热启动；
          另单独累计纯 C 求解时间（EN_initH+EN_runH）。精度 1e-6（与 GPU 流程的 GGA 容差相同），
          另测 INP 自带的 1e-3（只测 n=60）。TRIALS=200。差分步长 h = 1e-3·C。
  参考    HydroGrad CPU 隐式伴随（ImplicitGGASolve，dense 路径，精修到残差 ~1e-12）的梯度与耗时。
  环境    本机单线程，关闭后台降频、高优先级、绑定最快逻辑核（cpu_fair_scaling.fast_cpu）。
输出：data/cpu_epanet_fd_cityd.json，data/cpu_epanet_fd_cityd_report.txt（追加）
运行：python -X utf8 scripts/cpu_epanet_fd_cityd.py
"""
import ctypes
import json
import os
import platform
import sys
import time

import cpu_fair_scaling as fs           # 复用 EpanetFast / HydroFast / rand_set / fast_cpu
import numpy as np

cal, er, v1 = fs.cal, fs.er, fs.v1
B = 64
H_REL = 1e-3
JSON = os.path.join(fs.ROOT, "data", "cpu_epanet_fd_cityd.json")
RPT = os.path.join(fs.ROOT, "data", "cpu_epanet_fd_cityd_report.txt")


def fd_subset(ep, S, C, idx, central, h_rel=H_REL):
    """只对 idx 内的参数做差分；场景外层、参数内层，扰动解从同工况基解热启动。"""
    C = np.asarray(C, float)
    hs = h_rel * C
    ep.set_C(C)
    g = np.zeros(len(idx))
    tot = 0.0
    for j in range(S.B):
        ep.set_dem(S.D[j])
        ep.solve(True)
        r0 = ep.heads() - S.elev_s - S.obs[j]
        f0 = r0 @ r0
        tot += f0
        for a, k in enumerate(idx):
            ep.set_Ck(k, C[k] + hs[k])
            ep.solve(False)
            rp = ep.heads() - S.elev_s - S.obs[j]
            if central:
                ep.set_Ck(k, C[k] - hs[k])
                ep.solve(False)
                rm = ep.heads() - S.elev_s - S.obs[j]
                g[a] += (rp @ rp - rm @ rm) / (2 * hs[k])
            else:
                g[a] += (rp @ rp - f0) / hs[k]
            ep.set_Ck(k, C[k])
    m = S.B * len(S.sens)
    return tot / m, g / m + fs.LAM * S.dreg(C)[idx]


def iter_sample(ep, S, C, k):
    """冷启动与热启动各一次求解的 EPANET 迭代数与相对误差（场景 0，扰动参数 k）。"""
    st = ctypes.c_double()

    def stat():
        ep.lib.EN_getstatistic(ep.ph, er.EN_ITERATIONS, ctypes.byref(st))
        it = int(st.value)
        ep.lib.EN_getstatistic(ep.ph, er.EN_RELATIVEERROR, ctypes.byref(st))
        return it, float(st.value)
    ep.set_C(C)
    ep.set_dem(S.D[0])
    ep.solve(True)
    cold = stat()
    ep.set_Ck(k, C[k] * (1 + H_REL))
    ep.solve(False)
    warm = stat()
    ep.set_Ck(k, C[k])
    return cold, warm


def main():
    env = fs.fast_cpu()
    sys.stdout = v1.Tee(RPT)
    print("=" * 110)
    print(f"cpu_epanet_fd_cityd  {time.strftime('%Y-%m-%d %H:%M:%S')}  B={B}  h_rel={H_REL}  CPU={platform.processor()}  "
          f"绑定逻辑核={env.get('core')}  关闭降频={env.get('throttle_off')}")
    pb = cal.Problem("city_d")
    C_true, _ = cal.make_truth(pb, "perpipe", cal.TRUTH_SEED["perpipe"])
    sens = cal.get_sensors(pb, "dopt40")
    S = fs.rand_set(pb, B, sens, C_true)
    Cv = np.full(len(pb.free_idx), cal.C0)
    idx_all = np.arange(len(Cv))
    idx60 = np.sort(np.random.default_rng(0).choice(len(Cv), 60, replace=False))
    hg = fs.HydroFast(S)
    hg.loss_grad(Cv)                                            # 预热
    t0 = time.perf_counter()
    L_hg, g_hg = hg.loss_grad(Cv)
    t_hg = time.perf_counter() - t0
    print(f"[HydroGrad CPU 隐式伴随] 一次目标+梯度 {t_hg:.3f} s，每工况 {t_hg / B * 1e3:.2f} ms（全部 432 个参数一次得到）")
    rows = []
    for acc in (1e-6, 1e-3):
        ep = fs.EpanetFast(pb, sens)
        ep.ep.set_option(er.EN_ACCURACY, acc)
        ep.ep.set_option(er.EN_TRIALS, 200)
        cold, warm = iter_sample(ep, S, Cv, int(idx60[0]))
        ep.set_dem(S.D[0])
        ep.solve(True)
        d_ep = np.array([ep.lib.EN_getnodevalue(ep.ph, nj, er.EN_DEMAND, ctypes.byref(ep.v)) or ep.v.value
                         for nj in ep.junc_ep]) / ep.uf
        print(f"  需水核对：EPANET 实际施加需水与 HydroGrad 的最大相对差 "
              f"{np.abs(d_ep - S.D[0][ep.junc]).max() / np.abs(S.D[0][ep.junc]).max():.1e}")
        print(f"[EPANET 精度 {acc:g}] 迭代数/相对误差：冷启动 {cold[0]} / {cold[1]:.1e}，热启动 {warm[0]} / {warm[1]:.1e}")
        for name, idx, central in (("n60 前向差分", idx60, False), ("n60 中心差分", idx60, True),
                                   ("n432 前向差分", idx_all, False), ("n432 中心差分", idx_all, True)):
            if acc != 1e-6 and len(idx) > 60:
                continue
            tc0, n0, t0 = ep.t_c, ep.n_solve, time.perf_counter()
            L_fd, g_fd = fd_subset(ep, S, Cv, idx, central)
            wall = time.perf_counter() - t0
            err = float(np.linalg.norm(g_fd - g_hg[idx]) / np.linalg.norm(g_hg[idx]))
            rec = dict(accuracy=acc, case=name, n_params=int(len(idx)), central=central, B=B, wall_s=wall,
                       c_only_s=ep.t_c - tc0, solves=ep.n_solve - n0, per_scenario_ms=wall / B * 1e3,
                       per_scenario_c_only_ms=(ep.t_c - tc0) / B * 1e3, grad_rel_err_vs_adjoint=err,
                       obj_rel_diff=abs(L_fd - (L_hg - fs.LAM * S.reg(Cv))) / abs(L_hg),
                       iters_cold=cold[0], iters_warm=warm[0])
            rows.append(rec)
            print(f"  {name:<10} 总 {wall:8.2f} s（纯 C {rec['c_only_s']:7.2f} s），求解 {rec['solves']:>6} 次，"
                  f"每工况 {rec['per_scenario_ms']:9.1f} ms（纯 C {rec['per_scenario_c_only_ms']:8.1f} ms）| "
                  f"梯度对伴随的相对误差 {err:.1e} | 目标函数相对差 {rec['obj_rel_diff']:.1e}", flush=True)
        ep.close()
    out = dict(B=B, h_rel=H_REL, env=env, cpu=platform.processor(),
               hydrograd_cpu=dict(wall_s=t_hg, per_scenario_ms=t_hg / B * 1e3), epanet_fd=rows)
    json.dump(out, open(JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(f"saved -> {JSON}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""bwd_cost_ratio.py - 核查原稿第 57 行"反向按实测约为前向的 1% 计费"（WR 大修，2026-10-23）。

出处：data/optimizer_v2_wip.txt 第一节（2026-09-04，City D，RTX 4090 超算节点）：GPU 批量 20 帧
"前向+伴随 0.1144 s / 只前向 0.1115 s"，LM 段"伴随雅可比 3.0209 s / 批前向 3.0196 s"。
Hanoi 从未单独测过。本脚本在校准对比实际走的代码路径上重测两个网络：
  一阶段  scripts/calibrate.py 的 fwd()：CPU，ImplicitGGASolve，20 训练帧，GGA_MI=20，POLISH=4；
          分别计时一次前向（含建图）与随后的一次 backward（隐式伴随）。每个 NFE 配一个 nbwd。
  LM 段   calibrate._lm_polish：一次批前向 solve_polished 与一次伴随雅可比 sensitivity_matrix
          （后者内部含一次批前向，记 nfe+=1、nbwd+=1）。
Hanoi 用默认布点（25 个训练传感器），City D 用 dopt40（与 G-C1/G-D 题面一致）。参数取 C0，观测置零（不影响耗时）。
单线程；关闭后台降频、高优先级、绑定最快逻辑核（cpu_fair_scaling.fast_cpu）。预热 1 次后取中位数。
输出 data/bwd_cost_ratio.json。运行：python -X utf8 scripts/bwd_cost_ratio.py
"""
import json
import os
import platform
import time

import cpu_fair_scaling as fs           # 单线程、float64 环境与 fast_cpu
import numpy as np
import torch

cal = fs.cal
from dgga.autodiff import ImplicitGGASolve, solve_polished   # noqa: E402
from dgga.calib import RoughnessParam                         # noqa: E402
from dgga.sensitivity import sensitivity_matrix               # noqa: E402

OUT = os.path.join(fs.ROOT, "data", "bwd_cost_ratio.json")
NETS = {"hanoi": ("default", 15), "city_d": ("dopt40", 5)}


def tmed(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def run(net, placement, reps):
    pb = cal.Problem(net)
    s, tr = pb.s, pb.train_frames
    sens_train, _ = cal.split_holdout(cal.get_sensors(pb, placement))
    dt, rht, ket = torch.tensor(pb.d[tr]), torch.tensor(pb.rh[tr]), torch.zeros(pb.net.N)
    elev_t = torch.tensor(pb.elev[sens_train])
    obs_t = torch.zeros(len(tr), len(sens_train), dtype=torch.float64)
    rp = RoughnessParam(s, pipe_idx=pb.free_idx, C_lo=cal.C_LO, C_hi=cal.C_HI, init=cal.C0)
    tf, tb = [], []
    for i in range(reps + 1):                 # 第 0 次预热
        t0 = time.perf_counter()
        head, _, _ = ImplicitGGASolve.apply(dt, rht, ket, rp.r_hw(), s, 1e-12, cal.GGA_MI, cal.POLISH)
        loss = (((head[:, sens_train] - elev_t) - obs_t) ** 2).mean() \
            + 1e-4 * ((rp.C() - cal.C0) ** 2).mean() / cal.REG_SCALE
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        rp.theta.grad = None
        if i:
            tf.append(t1 - t0)
            tb.append(t2 - t1)
    r_full = pb.r_of_C(np.full(s.L, cal.C0))
    t_lmf = tmed(lambda: solve_polished(s, pb.d[tr], pb.rh[tr], r_hw=r_full, accuracy=1e-12,
                                        max_iter=cal.GGA_MI, polish_steps=cal.POLISH), reps)
    t_jac = tmed(lambda: sensitivity_matrix(s, pb.d[tr], pb.rh[tr], r_full, sens_train, wrt="C", accuracy=1e-12,
                                            max_iter=cal.GGA_MI, polish_steps=cal.POLISH), reps)
    f, b = float(np.median(tf)), float(np.median(tb))
    rec = dict(placement=placement, n_train_frames=len(tr), n_train_sensors=len(sens_train),
               n_free=len(pb.free_idx), reps=reps, fwd_s=f, bwd_s=b, bwd_over_fwd=b / f,
               bwd_over_fwd_min_max=[float(min(tb) / max(tf)), float(max(tb) / min(tf))],
               lm_fwd_s=t_lmf, lm_jac_s=t_jac, lm_jac_increment_over_fwd=(t_jac - t_lmf) / t_lmf)
    print(f"[{net}] 训练帧 {len(tr)}，训练传感器 {len(sens_train)}，自由管 {len(pb.free_idx)}，中位数 {reps} 次 | "
          f"一阶段：前向 {f*1e3:.2f} ms，反向 {b*1e3:.2f} ms，反向/前向 = {b/f*100:.1f}% | "
          f"LM 段：批前向 {t_lmf*1e3:.1f} ms，伴随雅可比 {t_jac*1e3:.1f} ms，雅可比比前向多 {(t_jac-t_lmf)/t_lmf*100:.1f}%",
          flush=True)
    return rec


def main():
    env = fs.fast_cpu()
    print(f"bwd_cost_ratio  {time.strftime('%Y-%m-%d %H:%M:%S')}  CPU={platform.processor()}  threads={torch.get_num_threads()} "
          f"核={env.get('core')}  关闭降频={env.get('throttle_off')}")
    out = dict(date=time.strftime("%Y-%m-%d"), env=env, cpu=platform.processor(),
               nets={k: run(k, *v) for k, v in NETS.items()})
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()

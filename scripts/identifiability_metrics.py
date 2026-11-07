# -*- coding: utf-8 -*-
"""identifiability_metrics.py - 辨识性指标（WR 大修 R2-M8、R1-Mod23/24，2026-10-23）。

把三个层次分开报告（灵敏度矩阵 S = ∂p/∂C，行 = 20 训练帧 × 传感器，列 = H-W 管道）：
  结构性不灵敏   S 的整列为 0（对任何传感器、任何帧都无影响）
  局部有灵敏度   列非零
  良态子空间内可辨识   奇异值 σ_i > 1e-2·σ_1（与 calibrate.SUB_TOL 相同）的方向个数 k_sub
另给：数值秩（σ_i > 1e-10·σ_1）、条件数 σ_1/σ_rank；在高斯噪声 σ=0.1 ft、
弱先验（C 的先验标准差 30，与校准正则项的尺度相同）下的后验协方差
Cov = (SᵀS/σ² + I/30²)^-1：后验标准差、参数相关系数、CRLB 迹。
输出 data/identifiability_metrics.json
运行：python -X utf8 scripts/identifiability_metrics.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate as cal                                          # noqa: E402
from dgga.calib import identifiability                           # noqa: E402

SIG, PRIOR_SD = 0.1, 30.0
LAYOUTS = (("hanoi", ("default", "augS0", "augdopt10", "augcover10")),
           ("city_d", ("ga40", "augdopt20", "augcover20", "dopt40", "random40_s0")))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {}
    print(f"{'网络':<7}{'布置':<12}{'m':>4}{'管道':>6}{'零列':>6}{'非零':>6}{'秩':>6}{'条件数':>10}{'k_sub':>7}"
          f"{'后验sd中位':>11}{'sd<5占比':>9}{'sd>20占比':>10}{'最大|相关|':>11}{'|相关|>0.9对占比':>15}")
    for net, layouts in LAYOUTS:
        pb = cal.Problem(net)
        C_true, _ = cal.make_truth(pb, "perpipe", cal.TRUTH_SEED["perpipe"])
        tr = pb.train_frames
        for pl in layouts:
            sens = cal.get_sensors(pb, pl)
            dg = identifiability(pb.s, pb.d[tr], pb.rh[tr], sens, r_hw=pb.r_of_C(C_true), wrt="C", return_S=True)
            S = np.asarray(dg["S"])[:, pb.pidx]
            zero = np.all(S == 0.0, axis=0)
            Sn = S[:, ~zero]
            sv = np.linalg.svd(Sn, compute_uv=False)
            rank = int((sv > 1e-10 * sv[0]).sum())
            k_sub = int((sv > 1e-2 * sv[0]).sum())
            F = Sn.T @ Sn / SIG ** 2 + np.eye(Sn.shape[1]) / PRIOR_SD ** 2
            Cov = np.linalg.inv(F)
            sd = np.sqrt(np.diag(Cov))
            R = Cov / np.outer(sd, sd)
            iu = np.triu_indices_from(R, 1)
            rec = dict(m=int(len(sens)), T=int(len(tr)), n_pipes=int(len(pb.pidx)), n_zero=int(zero.sum()),
                       n_nonzero=int((~zero).sum()), rank=rank, cond=float(sv[0] / sv[rank - 1]), k_sub=k_sub,
                       post_sd_median=float(np.median(sd)), frac_sd_lt_5=float(np.mean(sd < 5)),
                       frac_sd_gt_20=float(np.mean(sd > 20)), max_abs_corr=float(np.abs(R[iu]).max()),
                       frac_pairs_abs_corr_gt_0p9=float(np.mean(np.abs(R[iu]) > 0.9)),
                       crlb_trace=float(np.trace(Cov)), sigma_ft=SIG, prior_sd=PRIOR_SD)
            out[f"{net}|{pl}"] = rec
            print(f"{net:<7}{pl:<12}{rec['m']:>4}{rec['n_pipes']:>6}{rec['n_zero']:>6}{rec['n_nonzero']:>6}{rank:>6}"
                  f"{rec['cond']:>10.2e}{k_sub:>7}{rec['post_sd_median']:>11.2f}{rec['frac_sd_lt_5']:>9.2f}"
                  f"{rec['frac_sd_gt_20']:>10.2f}{rec['max_abs_corr']:>11.3f}{rec['frac_pairs_abs_corr_gt_0p9']:>15.4f}",
                  flush=True)
    path = os.path.join(cal.ROOT, "data", "identifiability_metrics.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()

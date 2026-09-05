# -*- coding: utf-8 -*-
"""probe_conditioning.py - 测量 GGA Schur 补 A 的 2-范数条件数。

论文 3.2 节用 kappa(A) 做后向误差论证。此前该数值只在若干脚本注释里写作
"κ~1e9"，没有任何实测记录；本脚本补上实测，结果写入
data/conditioning_report.txt，由 make_paper_figs.py 解析进 claims.json。

做法：对 dense 路径的每一次 Cholesky 调用截获装配好的 A（[B,Nj,Nj]），取
最后一次（收敛迭代）计算 cond2 = smax/smin。再报告两个子矩阵：
  (a) 去掉被关闭链路解耦的行（对角恰为 1/CBIG = 1e-8 量级）；
  (b) 在 (a) 基础上再去掉 1/CSMALL = 1e6 量级的阀门行。
三个数一起给出，因为全阵 cond2 由一个与主网解耦的孤立行支配，直接用它做
式(7) 的后向误差上界会大幅高估。

运行：python -X utf8 \
        scripts/probe_conditioning.py
"""
import os
import sys
import datetime

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

from dgga.parse import parse_inp                            # noqa: E402
from dgga.solver import GGASolver                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "conditioning_report.txt")

# 只列 dense 路径能装配的网（PBV/GPV 未实现，City H 含该类链路会被拒）。
NETS = [("city_d", os.path.join(ROOT, "datasets", "city_d.inp"))]

DECOUPLED = 1e-6        # 对角 <= 该值 => 被 1/CBIG=1e-8 关闭链路解耦的行
VALVE_ROW = 1e5         # 对角 >= 该值 => 含 1/CSMALL=1e6 量级阀门项的行


def capture_A(inp):
    """跑一次 dense 稳态求解，截获每次 Cholesky 的系数矩阵。"""
    net = parse_inp(inp)
    s = GGASolver(net, mode="dense")
    d = torch.as_tensor(net.demand_cfs_at(0), dtype=torch.float64)
    rh = torch.as_tensor(net.reservoir_head_ft_at(0), dtype=torch.float64)
    cap = []
    orig = torch.linalg.cholesky

    def spy(A, *a, **k):
        cap.append(A.detach().clone())
        return orig(A, *a, **k)

    torch.linalg.cholesky = spy
    try:
        s.solve(d, rh)
    finally:
        torch.linalg.cholesky = orig
    return net, cap


def main():
    lines = [
        "Schur 补条件数探针（probe_conditioning.py）",
        f"生成时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   "
        f"解释器: {sys.executable}",
        "口径: dense 路径最后一次 Cholesky 的 A（收敛迭代，t=0 帧），"
        "cond2 = smax/smin（numpy.linalg.cond, 2-范数）",
        "=" * 86,
    ]
    for name, inp in NETS:
        net, cap = capture_A(inp)
        A = cap[-1][0].numpy()
        diag = np.diag(A)
        keep = diag > DECOUPLED
        keep2 = keep & (diag < VALVE_ROW)
        c_full = float(np.linalg.cond(A))
        c_dec = float(np.linalg.cond(A[np.ix_(keep, keep)]))
        c_val = float(np.linalg.cond(A[np.ix_(keep2, keep2)]))
        lines += [
            f"[{name}] INP={os.path.relpath(inp, ROOT)}  N={net.N}  "
            f"Nj={A.shape[0]}  GGA 迭代(截获次数)={len(cap)}",
            f"  对角范围: {diag.min():.4e} .. {diag.max():.4e}",
            f"  cond2(A) 全阵                        = {c_full:.4e}",
            f"  cond2 去 {int((~keep).sum()):3d} 个关闭链路解耦行 "
            f"(Nj={int(keep.sum())})   = {c_dec:.4e}",
            f"  cond2 再去 {int((keep & ~keep2).sum()):3d} 个阀门行 "
            f"(Nj={int(keep2.sum())})       = {c_val:.4e}",
            "",
        ]
    lines.append("=" * 86)
    lines.append("注: 全阵 cond2 由与主网解耦的孤立行支配；式(7) 的后向误差上界"
                 "用去解耦行后的 cond2 才有意义。")
    txt = "\n".join(lines) + "\n"
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    print(txt)
    print(f"写出 {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()

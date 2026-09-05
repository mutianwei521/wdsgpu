# -*- coding: utf-8 -*-
"""bench_partial_eps.py <stem> <n_frames> - 前 N 帧自主 EPS 对拍（大网预算内部分覆盖）。

背景：pub_net6（N=3356, 609 帧）完整自主 EPS 约 6.5 s/帧 ≈ 67 min，超出单次
会话前台预算。本脚本按 dgga.eps.EpsDriver.run() 的同一循环（hydraul.c runhyd
节律：demands → controls → hydsolve → nexthyd）自主推进前 N 帧，与参考解逐帧
对拍后停止。复用 EpsDriver 私有方法的先例见 align.py replay_b2。
判定门槛与 align_eps.py 一致（H<1e-6 ft、开启链路 Q<1e-6 cfs、关闭管 1e-5、
帧时刻 int 相等、状态一致）；覆盖范围如实注明。"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dgga.parse import Net       # noqa: E402
from dgga.eps import EpsDriver   # noqa: E402
from dgga.solver import MISSING  # noqa: E402
from align import resolve_inp    # noqa: E402

REF_DIR = os.path.join(ROOT, "data", "reference")
TOL_H, TOL_Q, TOL_QC = 1e-6, 1e-6, 1e-5


def main(stem, n_frames):
    net = Net.load(REF_DIR, stem)
    ref = np.load(os.path.join(REF_DIR, f"{stem}_ref.npz"))
    inp = resolve_inp(stem)
    drv = EpsDriver(net, inp_path=inp if os.path.isfile(inp) else None)
    s = drv.solver
    drv._inithyd()
    T_ep = len(ref["t_sec"])
    F = min(n_frames, T_ep)
    print(f"=== {stem}: 部分自主 EPS 前 {F}/{T_ep} 帧, N={net.N}, L={net.L} ===")
    worst_h = worst_q = worst_qc = 0.0
    ok = True
    stat_all = True
    it_match = True
    for f in range(F):
        t = drv.Htime
        t_ok = int(t) == int(ref["t_sec"][f])
        drv._demands()
        drv._controls()
        r = s.run_gga(drv.d, drv.H, q0=drv.q, e0=drv.e,
                      status0=drv.S, setting0=drv.K, do_status=True)
        drv.q = r["flow"]
        drv.e = r["emitter"]
        drv.S = r["status"]
        drv.K = r["setting"]
        drv.H = r["head"]
        drv.fixed_dem = r["fixed_demand"]
        # node_dem 与 run() 同口径（规则引擎读；本脚本网无规则时亦保持一致）
        dem_out = np.where(s.is_fixed_node, drv.fixed_dem, drv.d + drv.e)
        drv.node_dem = dem_out

        open_api = drv.S > s.ST_CLOSED
        flow_api = np.where(open_api, drv.q, 0.0)
        opened = ref["status"][f] > 0
        dH = np.abs(drv.H - ref["head_ft"][f]).max()
        dQl = np.abs(flow_api - ref["flow_cfs"][f])
        dQ = dQl[opened].max() if opened.any() else 0.0
        dQc = dQl[~opened].max() if (~opened).any() else 0.0
        st_ok = bool(np.array_equal(open_api.astype(np.int8), ref["status"][f]))
        it_my, it_ep = int(r["iters"]), int(ref["iterations"][f])
        it_match = it_match and it_my == it_ep
        stat_all = stat_all and st_ok
        line_ok = t_ok and dH < TOL_H and dQ < TOL_Q and dQc < TOL_QC and st_ok
        ok = ok and line_ok
        worst_h = max(worst_h, dH)
        worst_q = max(worst_q, dQ)
        worst_qc = max(worst_qc, dQc)
        print(f"{f:>4d} {int(t):>7d} {dH:>12.3e} {dQ:>12.3e} {dQc:>12.3e} "
              f"{'Y' if st_ok else 'N'} {it_my:>4d}/{it_ep:<4d}"
              f"{'' if line_ok else '  <-- 超限'}", flush=True)
        drv._nexthyd(float(r["relerr"]))
    print("-" * 78)
    print(f"前 {F} 帧最差: max|ΔH|={worst_h:.3e} ft, max|ΔQ|={worst_q:.3e} cfs, "
          f"关闭管 max|ΔQ|={worst_qc:.3e} cfs")
    print(f"状态逐帧一致: {'PASS' if stat_all else 'FAIL'}; "
          f"迭代数逐帧相等: {'是' if it_match else '否'}")
    print(f"总判定(前 {F} 帧): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60))

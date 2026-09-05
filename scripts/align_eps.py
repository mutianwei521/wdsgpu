# -*- coding: utf-8 -*-
"""align_eps.py [stem ...] - 完整自主 EPS 对拍（阶段 B2 里程碑 3 验收）。

对每个网从 t=0 由 dgga.eps.EpsDriver 完整自主推进（demands/controls/hydsolve
状态机/nexthyd/tanklevels 全部自算，不看 ref 的任何中间量），与 EPANET 参考解
逐帧对比：
- 帧时刻序列必须逐一 int 相等（硬门槛）；
- 每帧 max|ΔH| < 1e-6 ft、开启链路 max|ΔQ| < 1e-6 cfs（硬门槛；实际值如实报）；
- 关闭链路与 ref 的 API 置零值比较，容差 1e-5 cfs（同旧约定）；
- status/setting（泵）与 EN_* 输出逐帧比对，iterations 对照打印。
ky5 已知豁免（EPANET 固有 Accuracy=1e-4 死端支管陈旧流量）：若某帧超限仅发生在
关闭隔离支管（超限开启链路 ref 流量 |Q|<0.01 cfs 且超限节点只邻接此类链路），
按帧豁免并注明；实测 4 网均位级对齐，未触发豁免。
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dgga.parse import Net       # noqa: E402
from dgga.eps import EpsDriver   # noqa: E402
from align import resolve_inp    # noqa: E402  （rand_*/pub_*/主力网统一 INP 解析）

REF_DIR = os.path.join(ROOT, "data", "reference")
INP_DIR = os.path.join(ROOT, "networks", "InpData")

TOL_H = 1e-6         # ft
TOL_Q = 1e-6         # cfs（开启链路）
TOL_Q_CLOSED = 1e-5  # cfs（关闭链路，ref 被 API 置零）

DEFAULT_STEMS = ["EXA6", "city_h", "ky3", "ky5"]


def run_one(stem):
    net = Net.load(REF_DIR, stem)
    ref = np.load(os.path.join(REF_DIR, f"{stem}_ref.npz"))
    inp = resolve_inp(stem)
    out = EpsDriver(net, inp_path=inp if os.path.isfile(inp) else None).run()

    T_my, T_ep = len(out["t_sec"]), len(ref["t_sec"])
    print(f"=== {stem}: EPS 帧数 {T_my}（ref {T_ep}），N={net.N}, L={net.L}, "
          f"泵={len(net.pump_link)}, 水池={len(net.tank_node)}, "
          f"控制={len(net.ctl_link)} ===")
    ok = True
    # ---- 帧时刻序列逐一 int 相等 ----
    t_ok = T_my == T_ep and bool(np.array_equal(
        out["t_sec"].astype(np.int64), ref["t_sec"].astype(np.int64)))
    print(f"帧时刻序列 int 相等: {'PASS' if t_ok else 'FAIL'}")
    ok = ok and t_ok
    if not t_ok:
        print("  我方:", out["t_sec"].tolist())
        print("  ref :", ref["t_sec"].tolist())
        return 1

    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    pl = np.asarray(net.pump_link, dtype=np.int64)
    # 节点 → 关联链路邻接表（豁免判据扫描用；等价替代逐节点 np.where 全链路
    # 扫描，纯性能优化 - bwsn2 N=1.2 万级逐帧扫描原实现需分钟级）
    inc = [[] for _ in range(net.N)]
    for k in range(net.L):
        inc[int(n1[k])].append(k)
        inc[int(n2[k])].append(k)
    print(f"{'帧':>3} {'t(s)':>7} {'max|ΔH|ft':>12} {'max|ΔQ|cfs':>12} "
          f"{'关闭管|ΔQ|':>12} {'状态一致':>6} {'泵设定Δ':>10} {'iters(我/EP)':>12}")
    worst_h = worst_q = worst_qc = worst_set = 0.0
    n_exempt = 0
    stat_all = True
    it_match = True
    for f in range(T_my):
        dH_n = np.abs(out["head_ft"][f] - ref["head_ft"][f])
        dQ_l = np.abs(out["flow_cfs"][f] - ref["flow_cfs"][f])
        opened = ref["status"][f] > 0
        dH = dH_n.max()
        dQ = dQ_l[opened].max() if opened.any() else 0.0
        dQc = dQ_l[~opened].max() if (~opened).any() else 0.0
        st_ok = bool(np.array_equal(out["status"][f], ref["status"][f]))
        dset = np.abs(out["setting"][f] - ref["setting"][f])[pl].max() if pl.size else 0.0
        it_my, it_ep = int(out["iterations"][f]), int(ref["iterations"][f])
        it_match = it_match and it_my == it_ep
        line_ok = dH < TOL_H and dQ < TOL_Q and dQc < TOL_Q_CLOSED and st_ok
        note = ""
        if not line_ok:
            # ky5 死端豁免判据（见模块 docstring）
            bad_l = np.where(opened & (dQ_l >= TOL_Q))[0]
            dead = all(abs(float(ref["flow_cfs"][f][k])) < 1e-2 for k in bad_l)
            bad_n = np.where(dH_n >= TOL_H)[0]
            for nn in bad_n:
                for k in inc[int(nn)]:
                    if opened[k] and abs(float(ref["flow_cfs"][f][k])) >= 1e-2:
                        dead = False
                        break
                if not dead:
                    break
            if dead and st_ok and dQc < TOL_Q_CLOSED:
                note = "  [豁免: 关闭隔离支管陈旧流量]"
                n_exempt += 1
                line_ok = True
            else:
                note = "  <-- 超限"
        if not note:
            worst_h = max(worst_h, dH)
            worst_q = max(worst_q, dQ)
        worst_qc = max(worst_qc, dQc)
        worst_set = max(worst_set, dset)
        stat_all = stat_all and st_ok
        ok = ok and line_ok
        print(f"{f:>3} {int(out['t_sec'][f]):>7d} {dH:>12.3e} {dQ:>12.3e} "
              f"{dQc:>12.3e} {'Y' if st_ok else 'N':>6} {dset:>10.3e} "
              f"{it_my:>5d}/{it_ep:<5d}{note}")

    print("-" * 90)
    print(f"非豁免帧最差: max|ΔH|={worst_h:.3e} ft (门槛 {TOL_H:.0e}), "
          f"max|ΔQ|={worst_q:.3e} cfs (门槛 {TOL_Q:.0e}), "
          f"关闭管 max|ΔQ|={worst_qc:.3e} cfs (门槛 {TOL_Q_CLOSED:.0e})")
    print(f"状态逐帧一致: {'PASS' if stat_all else 'FAIL'}; "
          f"泵设定 max|Δ|={worst_set:.3e}; "
          f"迭代数逐帧相等: {'是' if it_match else '否'}; 豁免 {n_exempt} 帧")
    print(f"[{stem}] 总判定: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(stems):
    rc = 0
    for stem in stems:
        rc |= run_one(stem)
        print()
    print("全部网络 EPS 对拍:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_STEMS))

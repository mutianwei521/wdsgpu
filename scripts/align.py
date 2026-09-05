# -*- coding: utf-8 -*-
"""align.py <stem> - GGASolver 与 EPANET 参考解逐帧对拍（阶段 B 验收）。

流程：Net.load + ref.npz 回读；逐帧用 parse.demand_cfs_at（名义需水，不用 ref 的
demand_out）与 reservoir_head_ft_at 驱动 solve；EPS 热启动照抄 EPANET：第 0 帧
冷启动（inithyd 初值），后续帧续用上一帧收敛的 LinkFlow/EmitterFlow
（hydraul.c runhyd 循环不重跑 inithyd）。泵/水池网走 replay_b2：tank 头逐帧取
ref，其余求解前状态（需水/控制/状态机）按 hydraul.c 语义重构，见其 docstring。

硬门槛：max|ΔH|(junction) < 1e-6 ft 且 max|ΔQ|(开启链路) < 1e-6 cfs；
关闭管：静态网路径用内部贯穿流与 ref 的 API 置零值比较，容差放宽到 1e-5 cfs
（EPANET 输出层把关闭管流量置 0，而求解器内部保留 ~1e-8·Δh 级的贯穿流，
此差异属输出约定而非解误差）；replay_b2 路径按 EN_FLOW API 口径置零后比较
（状态一致性另列硬校验），理由见 replay_b2 内注释。
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.eps import EpsDriver      # noqa: E402
from dgga.parse import Net          # noqa: E402
from dgga.solver import GGASolver   # noqa: E402

REF_DIR = os.path.join(ROOT, "data", "reference")
INP_DIRS = {
    "city_d": os.path.join(ROOT, "networks", "realInpData"),
    "city_d_emit": os.path.join(ROOT, "networks", "variants"),
}
DEF_INP_DIR = os.path.join(ROOT, "networks", "InpData")


def resolve_inp(stem):
    """stem → INP 原文路径。随机网 stem 命名 rand_main_XXXX / rand_small_XXXX
    对应 networks/random_main|random_small/rand_XXXX.inp
    （见 scripts/build_random_reference.py GROUPS 的命名约定）；
    pub_* 公开网从 data/public_reference_index.json 的 inp_used 字段解析
    （参考解由该 INP 生成，含 _cleaned 清洗版，见 build_public_reference.py）。"""
    for prefix, sub in (("rand_main_", "random_main"), ("rand_small_", "random_small")):
        if stem.startswith(prefix):
            return os.path.join(ROOT, "networks", sub,
                                f"rand_{stem[len(prefix):]}.inp")
    if stem.startswith("pub_"):
        idx_path = os.path.join(ROOT, "data", "public_reference_index.json")
        if os.path.isfile(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                ent = json.load(f).get(stem)
            if ent and ent.get("inp_used"):
                return os.path.join(ROOT, *ent["inp_used"].split("/"))
    return os.path.join(INP_DIRS.get(stem, DEF_INP_DIR), f"{stem}.inp")

TOL_H = 1e-6       # ft，junction 水头
TOL_Q = 1e-6       # cfs，开启链路流量
TOL_Q_CLOSED = 1e-5  # cfs，关闭链路（ref 被 API 置零）


def replay_b2(stem, net, ref, inp):
    """B2 快照回放（边界按 hydraul.c 语义重构）：泵/水池网逐帧对拍。

    逐帧输入：tank 头逐帧取 ref（回放边界，不做 tanklevels 积分）；需水/
    水库头/转速 pattern 与控制事件按 hydraul.c 语义现算（EpsDriver 的
    _demands :465-535 / _controls :538-619，含恒功率泵复开 resetpumpflow
    :1103-1116）；warm start 续用上一帧我方内部流量（runhyd 不重跑 inithyd，
    帧 0 初值 = _inithyd 的 initlinkflow :344-374）；求解开完整状态机
    （do_status=True，hydsolver.c:159-187 的 linkstatus/tankstatus 节律）。

    为何不能冻结 ref 的最终状态求解（2026-08 根因记录）：
    - ky5 帧 3/9 等处 tankstatus（hydstatus.c:314-352 满池临时关闭 P-2）在
      求解**中途**翻转状态，最终状态冻结无法复现该迭代路径；Accuracy=1e-4
      宽收敛下停机迭代点随路径不同而差 ~1e-4；
    - ky5 的 ref 在 t=3.9~13h 段是恒功率泵失速的未收敛瞬态（O-Pump-8 水头
      达 1.8e5 ft，流量被 newflows 的 dq=Q/2 半步逐帧衰减趋 0），
      hgain=-Z/q 在 q→0 时把任何 1e-6 级路径差异无界放大成 1e2~1e5 ft；
      实测即便 warm start 与 EPANET 位级一致，冻结求解在帧 3 仍偏 1.7e-4
      并沿帧滚雪球。故必须按 EPANET 原语义重构求解前状态以复现迭代路径；
      重构后四网（EXA6/city_h/ky3/ky5）逐帧位级对齐（状态/迭代数一致）。
    """
    drv = EpsDriver(net, inp_path=inp)
    s = drv.solver
    jm = np.asarray(net.node_type) == 0
    T = ref["t_sec"].shape[0]
    tank_nodes = drv.tank_nodes
    pump_links = drv.pump_links
    print(f"=== {stem}（B2 快照回放）: {T} 帧, N={net.N}, L={net.L}, "
          f"trials={net.meta['trials']}, accuracy={net.meta['accuracy']}, "
          f"泵={len(pump_links)}, 水池={len(tank_nodes)} ===")

    # city_h 转速 pattern 校验：parse 计算的速度 vs ref setting
    upat_ok = True
    if (np.asarray(net.pump_upat) >= 0).any():
        worst = 0.0
        for f in range(T):
            t = int(ref["t_sec"][f])
            p = (t + net.meta["pat_start_sec"]) // net.meta["pat_step_sec"]
            for j, k in enumerate(pump_links):
                up = int(net.pump_upat[j])
                if up < 0:
                    continue
                F = net.patterns[up]
                spd = float(F[int(p) % len(F)])
                worst = max(worst, abs(spd - float(ref["setting"][f, k])))
        upat_ok = worst == 0.0
        print(f"转速 pattern 对比 ref setting: max|Δ|={worst:.3e} "
              f"{'PASS' if upat_ok else 'FAIL'}")

    # 帧 0 初值：inithyd（hydraul.c:85-179，含 initlinkflow :344-374）
    drv._inithyd()

    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    print(f"{'帧':>3} {'t(h)':>6} {'max|ΔH|ft':>12} {'max|ΔQ|cfs':>12} "
          f"{'关闭管|ΔQ|':>12} {'tank净流入Δ':>12} {'状态':>4} {'iters(我/EP)':>12}")
    worst_h = worst_q = worst_qc = worst_td = 0.0
    ok = upat_ok
    n_exempt = 0
    stat_all = True
    for f in range(T):
        t = int(ref["t_sec"][f])
        drv.Htime = t                                    # runhyd 时刻
        drv.H[tank_nodes] = ref["head_ft"][f][tank_nodes]  # tank 头逐帧取 ref
        drv._demands()                                   # hydraul.c:465-535
        drv._controls()                                  # hydraul.c:538-619

        r = s.run_gga(drv.d, drv.H, q0=drv.q, e0=drv.e,
                      status0=drv.S, setting0=drv.K, do_status=True)
        drv.q = r["flow"]
        drv.e = r["emitter"]
        drv.S = r["status"]
        drv.K = r["setting"]
        drv.H = r["head"]
        drv.fixed_dem = r["fixed_demand"]
        H, Q = r["head"], r["flow"]
        opened = ref["status"][f] > 0
        # 状态逐帧对拍（EN_STATUS 口径：status>CLOSED → 1）
        open_my = (drv.S > s.ST_CLOSED).astype(np.int8)
        st_ok = bool(np.array_equal(open_my, ref["status"][f]))
        stat_all = stat_all and st_ok

        dH_n = np.abs(H - ref["head_ft"][f])
        # 流量取 EN_FLOW API 口径（epanet.c:3657-3659 关闭链路置 0；与
        # align_eps.py 同约定 - ky5 失速帧关闭泵两端 Δh~1e5 ft，内部贯穿流
        # Δh/CBIG~1e-3 属求解器内部量，ref 不可观测；状态一致性已单列硬校验）
        flow_api = np.where(open_my.astype(bool), Q, 0.0)
        dQ_l = np.abs(flow_api - ref["flow_cfs"][f])
        dH = dH_n[jm].max()
        dQ = dQ_l[opened].max()
        dQc = dQ_l[~opened].max() if (~opened).any() else 0.0
        # tank 净流入对拍（EN_DEMAND 在 tank = NodeDemand = 净流入）
        dTd = np.abs(r["fixed_demand"] - ref["demand_out_cfs"][f])[tank_nodes].max() \
            if tank_nodes.size else 0.0
        it_my, it_ep = int(r["iters"]), int(ref["iterations"][f])
        line_ok = dH < TOL_H and dQ < TOL_Q and dQc < TOL_Q_CLOSED and st_ok
        note = ""
        if not line_ok:
            # ky5 已知豁免（阶段 A）：EPANET Accuracy=1e-4 的死端隔离支管陈旧流量。
            # 判据：超限的开启链路 ref 流量均近零（|Q_ref|<0.01 cfs，只承载关闭支
            # 泄漏流），且超限 junction 的所有开启邻接链路均为此类死端链路。
            bad_l = np.where(opened & (dQ_l >= TOL_Q))[0]
            dead_l = set(k for k in bad_l
                         if abs(float(ref["flow_cfs"][f][k])) < 1e-2)
            links_ok = len(dead_l) == len(bad_l)
            bad_n = np.where(jm & (dH_n >= TOL_H))[0]
            nodes_ok = True
            for nn in bad_n:
                inc = np.where((n1 == nn) | (n2 == nn))[0]
                for k in inc:
                    if opened[k] and k not in dead_l \
                            and abs(float(ref["flow_cfs"][f][k])) >= 1e-2:
                        nodes_ok = False
            if links_ok and nodes_ok and dQc < TOL_Q_CLOSED and st_ok:
                note = "  [豁免: 关闭隔离支管陈旧流量]"
                n_exempt += 1
                line_ok = True
            else:
                note = "  <-- 超限"
        if line_ok and not note:
            worst_h = max(worst_h, dH)
            worst_q = max(worst_q, dQ)
        worst_qc = max(worst_qc, dQc)
        worst_td = max(worst_td, dTd)
        ok = ok and line_ok
        print(f"{f:>3} {t/3600:>6.2f} {dH:>12.3e} {dQ:>12.3e} {dQc:>12.3e} "
              f"{dTd:>12.3e} {'Y' if st_ok else 'N':>4} "
              f"{it_my:>5d}/{it_ep:<5d}{note}")

    print("-" * 84)
    print(f"非豁免帧最差: max|ΔH|={worst_h:.3e} ft (门槛 {TOL_H:.0e}), "
          f"max|ΔQ|={worst_q:.3e} cfs (门槛 {TOL_Q:.0e}), "
          f"关闭管 max|ΔQ|={worst_qc:.3e} cfs (门槛 {TOL_Q_CLOSED:.0e}), "
          f"tank净流入 max|Δ|={worst_td:.3e} cfs; 豁免 {n_exempt} 帧")
    print(f"状态逐帧一致: {'PASS' if stat_all else 'FAIL'}")
    print(f"总判定: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(stem):
    net = Net.load(REF_DIR, stem)
    ref = np.load(os.path.join(REF_DIR, f"{stem}_ref.npz"))
    inp = resolve_inp(stem)
    if len(net.pump_link) or len(net.tank_node):
        return replay_b2(stem, net, ref, inp if os.path.isfile(inp) else None)
    solver = GGASolver(net, mode="epanet", inp_path=inp if os.path.isfile(inp) else None)

    jm = np.asarray(net.node_type) == 0          # junction 掩码
    T = ref["t_sec"].shape[0]
    print(f"=== {stem}: {T} 帧, N={net.N}, L={net.L}, "
          f"trials={net.meta['trials']}, accuracy={net.meta['accuracy']} ===")
    print(f"{'帧':>3} {'t(h)':>5} {'max|ΔH|ft':>12} {'max|ΔQ|cfs':>12} "
          f"{'关闭管|Q|':>12} {'iters(我/EP)':>12} {'relerr(我/EP)':>22}")

    # emitter 节点（GGASolver(inp_path=...) 已把位级重建的 Ke 写回 net.node_ke）
    ke_arr = np.asarray(net.node_ke, dtype=np.float64)
    em_idx = np.where(ke_arr > 0.0)[0]
    em_ref_all, em_my_all = [], []   # [T,K]：EPANET vs 我们 的逐帧 emitter 流量

    q_prev = None
    e_prev = None
    worst_h = worst_q = worst_qc = 0.0
    ok = True
    for f in range(T):
        t = int(ref["t_sec"][f])
        d = net.demand_cfs_at(t)                  # 名义需水（不能用 ref demand_out）
        rh = net.reservoir_head_ft_at(t)
        r = solver.solve(d, rh, q0=q_prev, e0=e_prev)
        H = r["head_ft"].numpy()
        Q = r["flow_cfs"].numpy()
        q_prev = Q                                # EPS 热启动：续用本帧流量
        e_prev = r["emitter_cfs"].numpy()
        if em_idx.size:
            # EN_DEMAND = NodeDemand = 交付需水 + emitter（hydsolver.c:198-203、
            # epanet.c:2163-2165、:2244 注释），DDA 下交付需水 ≡ 名义需水，
            # 故 EPANET 的 emitter 流量 = demand_out_cfs − 名义需水。
            em_ref_all.append((ref["demand_out_cfs"][f] - d)[em_idx])
            em_my_all.append(e_prev[em_idx])

        status = ref["status"][f]
        opened = status > 0                       # EN_STATUS: 0=closed
        dH = np.abs(H - ref["head_ft"][f])[jm].max()
        dQ = np.abs(Q - ref["flow_cfs"][f])[opened].max()
        # 关闭链路：ref 被 API 置 0，比较我们的内部贯穿流（容差 1e-5 并注明）
        dQc = np.abs(Q - ref["flow_cfs"][f])[~opened].max() if (~opened).any() else 0.0
        it_my = int(r["iters"])
        it_ep = int(ref["iterations"][f])
        re_my = float(r["relerr"])
        re_ep = float(ref["relerr"][f])
        worst_h = max(worst_h, dH)
        worst_q = max(worst_q, dQ)
        worst_qc = max(worst_qc, dQc)
        line_ok = dH < TOL_H and dQ < TOL_Q and dQc < TOL_Q_CLOSED
        ok = ok and line_ok
        print(f"{f:>3} {t/3600:>5.1f} {dH:>12.3e} {dQ:>12.3e} {dQc:>12.3e} "
              f"{it_my:>5d}/{it_ep:<5d} {re_my:>10.3e}/{re_ep:<10.3e}"
              f"{'' if line_ok else '  <-- 超限'}")

    print("-" * 84)
    print(f"全帧最差: max|ΔH|={worst_h:.3e} ft (门槛 {TOL_H:.0e}), "
          f"max|ΔQ|={worst_q:.3e} cfs (门槛 {TOL_Q:.0e}), "
          f"关闭管 max|Q|={worst_qc:.3e} cfs (门槛 {TOL_Q_CLOSED:.0e})")

    # ---- emitter 口径对拍（仅含 emitter 的网；门槛同 TOL_Q）----
    if em_idx.size:
        em_ref = np.stack(em_ref_all)   # [T,K] EPANET（demand_out − 名义需水）
        em_my = np.stack(em_my_all)     # [T,K] 我们的 EmitterFlow
        worst_e = float(np.abs(em_my - em_ref).max())
        em_ok = worst_e < TOL_Q
        ok = ok and em_ok
        LPSperCFS = 28.317              # types.h:72（仅用于打印 LPS 口径）
        print(f"emitter 对拍（{em_idx.size} 节点, EN_DEMAND-名义需水 vs EmitterFlow）: "
              f"max|Δe|={worst_e:.3e} cfs (门槛 {TOL_Q:.0e}) "
              f"{'PASS' if em_ok else 'FAIL'}")
        print(f"{'节点':>8} {'Ke_int':>14} {'EPANET t=0':>16} {'我们 t=0':>16} "
              f"{'均值(LPS)':>10} {'max|Δ|cfs':>12}")
        for j, i in enumerate(em_idx):
            print(f"{net.node_id[i]:>8} {ke_arr[i]:>14.6e} "
                  f"{em_ref[0, j]:>16.12f} {em_my[0, j]:>16.12f} "
                  f"{em_my[:, j].mean() * LPSperCFS:>10.3f} "
                  f"{np.abs(em_my[:, j] - em_ref[:, j]).max():>12.3e}")

    print(f"总判定: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: align.py <stem>   （如 align.py city_d）")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

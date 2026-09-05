# -*- coding: utf-8 -*-
"""check_schedule.py - F2：状态机**调度级**对拍器（PRV 轮次的前置验收工具）。

**为什么终态判据不够**：门 B1 独立审计（data/gate_b1_audit_wip.txt §3b）用 7 个
源码级变异体实测：M1（收敛即退出、忽略 stat_change）给出的**最终状态向量与正确
实现逐位相同**，却在 Richmond_skeleton 15/64 场景上把连通节点的头算错 1.483e+05
ft - "最终状态逐元素相等"这条硬判据对 M1/M5/M6 三个变异体**零检出力**。
病根：状态机是个不动点迭代，终态对了不代表**路径**对了，而头/流量是跟着路径走
的。唯一能钉死路径的是**调度轨迹**：每次 linkstatus/valvestatus 被调用的迭代号
+ 每次调用的 change 集合。

本脚本做两件事：
  §A 真品对拍：同一批极端场景（gate_b1 配方：需水 log-U + 单点漏损 + 水池水位
     贴边），dense 批量状态机（solve(..., status_machine=True,
     record_schedule=True)）vs mode="epanet" 串行（run_gga(do_status=True,
     record_schedule=True)），逐场景比：
       ①（硬）检查时刻序列：每次 linkstatus 被调用的**迭代号**逐项相等；
       ②（硬）每次检查的 change 集合（链路号集合）逐项相等；
       ③（硬）终态状态向量逐元素相等、iters 逐场景相等（F3 后未收敛口径统一）。
       ④（软，只报不判）收敛支/周期支的**支标签**：dense 与 epanet 的线性
          求解舍入不同，relerr 恰在 Hacc 边界时同一迭代可一支判"已收敛"一支判
          "未收敛" - linkstatus 照样在同一迭代被调、change 相同、后续轨迹与
          终态/iters 全同，只有标签翻转。这不是状态机缺陷（门 B1 审计 [1]
          的调度对拍在 Net2 116/128、ky4 47/48 同现象），实测 Richmond_skeleton
          9/64 场景翻转，全部"仅标签"。标签翻转叠加 (it,chg) 不等才是硬 FAIL。
     串行侧的 valvestatus/pswitch 事件在 dense 可比域上必须是**空 change**
     （dense 无对应物；pswitch 的 junction 控制已被 F1 构造期守卫拒绝），
     出现非空即记硬 mismatch。
  §B 检出力自证（--selftest 也随 main 默认跑）：把 dgga 复制一份、源码级注入
     门 B1 审计的 M1 变异体（`done_now = active & conv`，收敛即退出），子进程
     跑同一套对拍 - **调度判据必须变红，而终态判据必须仍是绿的**（这正是
     M1 的危险之处，也是本工具存在的理由）。

  §C PRV 网调度对拍（洞 C 修复，data/prv_release_audit_wip.txt）：原网表全部
     无 PRV，valvestatus 节律/prvstatus 转移类变异体（MB/MD/ME）在这里测不到。
     对 PRV_NETS（缺省 L-TOWN）用 aud_prv_states 的极端配方（含每阀逼
     OPEN/CLOSED），dense 批量状态机 vs epanet 串行，硬判据：
       ① 终态状态向量逐元素相等；
       ② **转移子序列**（非空 change 事件的 (迭代号, 支, change 集) 序列）
          逐项相等 - 调度硬不同但转移子序列相同且终态相同的场景是"仅停时"
          （Hacc 刀刃，两路各自自洽），只报不判；
       ③ 每阀三态（ACTIVE/OPEN/CLOSED）批内全出（配方失效即红，防空转）；
       ④ 调度同+双收敛场景连通域 max|dH| ≤ 0.5 ft（κ~1.6e11 的线代噪声实测
          <1e-2 ft，0.5 只有实现错误才碰得到）。
  §D PRV 检出力自证：MB（valvestatus 降频）/ MD（prvstatus OPEN case 丢
     AC 转移）/ ME（ACTIVE 的 Y 读 nodecoeffs 前 Xflow）三个源码级变异体
     （锚点同 scripts/aud_release/aud_mutate.py）各自子进程跑 §C 判据，
     **必须全部变红**（发布审计实测三者对旧网表/旧判据全绿逃逸）。

recording 缺省关：solver.solve/run_gga 不传 record_schedule 时零快照零开销，
数值路径逐位不变（影子包 sha256 对拍见 data/prv_pre_wip.txt）。

用法：python -X utf8 scripts/check_schedule.py [--B 64] [网名 ...]
退出码 0 = §A 全部一致 且 §B 变异体如期变红 且 §C 全过 且 §D 三变异体全红。
"""

import os
import shutil
import subprocess
import sys
import tempfile
import zlib

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB = os.path.join(ROOT, "networks", "public")
CLEAN = os.path.join(PUB, "_cleaned")

# 缺省网表：可切换元件真实动作的 B1 网（CVPIPE / 泵截断 / tankstatus / 不收敛）
DEFAULT_NETS = ["Richmond_skeleton", "Net1", "Net3", "ky4", "Anytown_wntr",
                "Net2"]
# M1 自证用的子集（审计实测 M1 在 Richmond_skeleton 上 15/64 场景漏检查）
SELFTEST_NETS = ["Richmond_skeleton", "Net1", "Net3"]

# M1 变异体（门 B1 审计 §3b）：收敛即退出，忽略 stat_change
M1_OLD = ("            done_now = active & conv & ((it > max_iter) | "
          "(~stat_change))")
M1_NEW = "            done_now = active & conv"

# ---- §C/§D：PRV 网（洞 C）----
PRV_NETS = ["L-TOWN"]
PRV_B = 48            # §C 正品场景数（另加每阀 2 条逼阀）
PRV_MUT_B = 24        # §D 变异体场景数
PRV_SEED = 909        # 与 scripts/aud_release/aud_prv_states.py 同配方
DH_GATE = 0.5         # ft；κ~1.6e11 线代噪声实测 <1e-2
# PRV 变异体（锚点与 scripts/aud_release/aud_mutate.py 逐字相同）
MUTS_PRV = {
    "MB_valve_freq": (
        "            if self._dense_prv_np:\n"
        "                S_v, vchg = self._prvstatus_batch(S, H, q)",
        "            if self._dense_prv_np and (it % 2 == 0):\n"
        "                S_v, vchg = self._prvstatus_batch(S, H, q)"),
    "MD_open_case": (
        "        c_open = torch.where(neg, CL,\n"
        "                             torch.where(h2 >= hset + htol, AC, OP))"
        "      # :279-282",
        "        c_open = torch.where(neg, CL, OP)"),
    "ME_y_predemand": (
        "                    P, Y, F, A, _ = self._prvcoeffs_batch(\n"
        "                        P, Y, F, Xflow, q, S, A=A)",
        "                    P, Y, F, A, _ = self._prvcoeffs_batch(\n"
        "                        P, Y, F, Xflow + d_j, q, S, A=A)"),
}


def inp_of(name):
    p = os.path.join(CLEAN, name + ".inp")
    return p if os.path.exists(p) else os.path.join(PUB, name + ".inp")


def make_scenarios(net, B, seed):
    """gate_b1_batch_sm.py 的极端场景配方（确定性种子）。"""
    rng = np.random.default_rng(seed)
    N = net.N
    junc = np.where(np.asarray(net.node_type) == 0)[0]
    tanks = np.asarray(net.tank_node, dtype=np.int64)
    hmin = np.asarray(net.tank_hmin, dtype=np.float64)
    hmax = np.asarray(net.tank_hmax, dtype=np.float64)
    plen = max([len(p) for p in net.patterns], default=1)
    pstep = int(net.meta.get("pat_step_sec", 3600) or 3600)
    D = np.zeros((B, N), dtype=np.float64)
    RH = np.zeros((B, N), dtype=np.float64)
    for b in range(B):
        t = int(rng.integers(0, max(plen, 1))) * pstep
        corner = (b % 4 == 0)
        if corner:
            mult = float(np.exp(rng.uniform(np.log(0.01), np.log(0.10))))
        else:
            mult = float(np.exp(rng.uniform(np.log(0.05), np.log(4.0))))
        d = net.demand_cfs_at(t) * mult
        tot = float(np.abs(d[junc]).sum())
        if not corner:
            j = int(junc[rng.integers(0, junc.size)])
            d[j] += float(rng.uniform(0.02, 0.60)) * max(tot, 1e-6)
        D[b] = d
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(t),
                                    dtype=np.float64))
        if tanks.size:
            lo, hi = hmin[: tanks.size], hmax[: tanks.size]
            u = rng.uniform(0.0, 1.0, tanks.size)
            lev = lo + u * (hi - lo)
            if corner:
                lev = hi - 0.01 * (hi - lo)          # 全池满角
            elif b % 3 == 1:
                lev = np.where(u < 0.5, lo + 0.02 * (hi - lo),
                               hi - 0.02 * (hi - lo))
            rh[tanks] = lev
        RH[b] = rh
    return D, RH


def dense_events(out, b):
    """dense schedule → 场景 b 的 [(it, 支, change 集), ...]。"""
    ev = []
    for e in out["schedule"]:
        if bool(e["conv"][b]):
            ev.append((int(e["it"]), "conv",
                       tuple(np.where(e["chg"][b])[0].tolist())))
        elif bool(e["per"][b]):
            ev.append((int(e["it"]), "per",
                       tuple(np.where(e["chg"][b])[0].tolist())))
    return ev


def serial_events(sched):
    """run_gga schedule → ([(it, 支, change 集), ...], 串行独有的非空事件数)。"""
    ev, extra = [], 0
    for it, kind, changed in sched:
        if kind == "ls_conv":
            ev.append((int(it), "conv", tuple(changed)))
        elif kind == "ls_per":
            ev.append((int(it), "per", tuple(changed)))
        elif changed:                    # valve/pswitch 在 dense 可比域必须空
            extra += 1
    return ev, extra


def compare_net(mods, name, B, verbose=True):
    """一个网的 §A 对拍。返回 dict(计数)。"""
    parse_inp, GGASolver = mods["parse_inp"], mods["GGASolver"]
    torch = mods["torch"]
    p = inp_of(name)
    net = parse_inp(p)
    seed = zlib.crc32(name.encode())
    D, RH = make_scenarios(net, B, seed)
    ke = np.asarray(net.node_ke, dtype=np.float64)

    sd = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                   inp_path=p, dense_status_machine=True)
    with torch.no_grad():
        od = sd.solve(D, RH, status_machine=True, record_schedule=True)
    Sd = od["status"].cpu().numpy()
    itd = od["iters"].cpu().numpy()

    se = GGASolver(net, device="cpu", dtype=torch.float64, mode="epanet",
                   inp_path=p)
    n_sched = n_state = n_iter = n_extra = n_flip = 0
    first_bad = None
    for b in range(B):
        oe = se.run_gga(D[b], RH[b], do_status=True, record_schedule=True)
        ev_d = dense_events(od, b)
        ev_s, extra = serial_events(oe["schedule"])
        strip = lambda ev: [(it, chg) for it, _tag, chg in ev]
        ok_sched = (strip(ev_d) == strip(ev_s)) and extra == 0   # 硬判据
        tag_flip = ok_sched and (ev_d != ev_s)                   # 仅支标签翻转
        ok_state = bool(np.array_equal(Sd[b], oe["status"]))
        ok_iter = int(itd[b]) == int(oe["iters"])
        n_sched += ok_sched
        n_flip += tag_flip
        n_state += ok_state
        n_iter += ok_iter
        n_extra += extra
        if not ok_sched and first_bad is None:
            first_bad = (b, ev_d, ev_s)
    if verbose:
        print("  %-18s B=%-3d 调度(硬)相同 %3d/%d  仅支标签翻转 %2d  "
              "终态相同 %3d/%d  iters相同 %3d/%d  串行独有非空事件 %d"
              % (name, B, n_sched, B, n_flip, n_state, B,
                 n_iter, B, n_extra), flush=True)
        if first_bad is not None:
            b, ev_d, ev_s = first_bad
            print(f"      首个调度硬不一致场景 b={b}:")
            print(f"        dense : {ev_d}")
            print(f"        serial: {ev_s}")
    return dict(B=B, sched=n_sched, state=n_state, iters=n_iter,
                extra=n_extra, flip=n_flip)


def load_mods(pkg_root):
    sys.path.insert(0, pkg_root if pkg_root else ROOT)
    import torch
    import dgga.solver as smod
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    assert os.path.abspath(smod.__file__).startswith(
        os.path.abspath(pkg_root if pkg_root else ROOT)), smod.__file__
    return dict(torch=torch, parse_inp=parse_inp, GGASolver=GGASolver)


def child_m1(tmp_root, B):
    """子进程：在 M1 变异体上跑同一套对拍，打印机器可读结果。"""
    mods = load_mods(tmp_root)
    tot_sched_bad = tot_state_bad = 0
    for name in SELFTEST_NETS:
        r = compare_net(mods, name, B, verbose=True)
        tot_sched_bad += r["B"] - r["sched"]
        tot_state_bad += r["B"] - r["state"]
    print(f"M1-RESULT sched_bad={tot_sched_bad} state_bad={tot_state_bad}")
    return 0


# ----------------------------------------------------------------------
# §C：PRV 网（转移子序列判据，aud_prv_states 口径）
# ----------------------------------------------------------------------
def prv_scenarios(net, B, seed, prv):
    """aud_prv_states 的自配配方：极端需水/漏损/水池贴边 + 每阀逼 OPEN/CLOSED。"""
    rng = np.random.default_rng(seed)
    nt = np.asarray(net.node_type)
    junc = np.where(nt == 0)[0]
    tanks = np.asarray(net.tank_node, dtype=np.int64)
    hmin = np.asarray(net.tank_hmin, dtype=np.float64)
    hmax = np.asarray(net.tank_hmax, dtype=np.float64)
    plen = max([len(p) for p in net.patterns], default=1)
    pstep = int(net.meta.get("pat_step_sec", 3600) or 3600)
    rows = []
    for b in range(B):
        t = int(rng.integers(0, max(plen, 1))) * pstep
        mult = float(np.exp(rng.uniform(np.log(0.05), np.log(5.0))))
        d = net.demand_cfs_at(t) * mult
        tot = float(np.abs(d[junc]).sum())
        if b % 3 != 2:
            j = int(junc[rng.integers(0, junc.size)])
            d[j] += max(tot, 1e-6) * float(rng.uniform(0.05, 1.0))
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(t),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            u = rng.random()
            if u < 0.25:
                rh[int(n)] = hmin[i]
            elif u < 0.50:
                rh[int(n)] = hmax[i]
            else:
                rh[int(n)] = hmin[i] + (hmax[i] - hmin[i]) * rng.random()
        rows.append((d, rh))
    d0 = net.demand_cfs_at(0)
    tot0 = float(np.abs(d0[junc]).sum())
    for k in prv:
        n2 = int(net.link_n2[k])
        d = d0 * 20.0                                   # 逼 OPEN
        if nt[n2] == 0:
            d[n2] += 12.0 * max(tot0, 1e-6)
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            rh[int(n)] = hmin[i]
        rows.append((d, rh))
        d = d0 * 0.1                                    # 逼 CLOSED（下游注入）
        if nt[n2] == 0:
            d[n2] -= 2.0 * max(tot0, 1e-6)
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            rh[int(n)] = hmax[i]
        rows.append((d, rh))
    return (np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]))


def prv_ev_serial(sched):
    ev = []
    for it, kind, changed in sched:
        if kind == "valve":
            ev.append((int(it), "valve", tuple(sorted(changed))))
        elif kind in ("ls_conv", "ls_per"):
            ev.append((int(it), "ls", tuple(sorted(changed))))
        elif changed:
            ev.append((int(it), "extra", tuple(sorted(changed))))
    return ev


def prv_ev_dense(sched, b):
    ev = []
    for e in sched:
        if e.get("kind") == "valve":
            if bool(e["act"][b]):
                ev.append((int(e["it"]), "valve",
                           tuple(sorted(np.where(e["chg"][b])[0].tolist()))))
        elif bool(e["conv"][b]) or bool(e["per"][b]):
            ev.append((int(e["it"]), "ls",
                       tuple(sorted(np.where(e["chg"][b])[0].tolist()))))
    return ev


def prv_conn_mask(net, opened):
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    reach = np.asarray(net.node_type) != 0
    e1, e2 = n1[opened], n2[opened]
    while True:
        new = reach.copy()
        np.logical_or.at(new, e2, reach[e1])
        np.logical_or.at(new, e1, reach[e2])
        if np.array_equal(new, reach):
            break
        reach = new
    return reach


def compare_prv(mods, name, B, verbose=True):
    """§C 一个 PRV 网的对拍。返回 dict(end_bad/transfer/cover_bad/dmax/...)。"""
    parse_inp, GGASolver = mods["parse_inp"], mods["GGASolver"]
    torch = mods["torch"]
    p = inp_of(name)
    net = parse_inp(p)
    lt = np.asarray(net.link_type)
    prv = np.where(lt == 3)[0].tolist()
    if not prv:
        raise RuntimeError(f"{name}: §C 网表必须含 PRV")
    D, RH = prv_scenarios(net, B, PRV_SEED, prv)
    sd = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                   inp_path=p, dense_status_machine=True)
    se = GGASolver(net, device="cpu", dtype=torch.float64, mode="epanet",
                   inp_path=p)
    # badvalve（批 Cholesky 非正定 raise 报样本号）：剔除重跑并如实报
    excluded = []
    keep = np.arange(D.shape[0])
    while True:
        try:
            with torch.no_grad():
                od = sd.solve(D[keep], RH[keep], status_machine=True,
                              record_schedule=True)
            break
        except RuntimeError as e:
            import re as _re
            m = _re.search(r"样本 \[([0-9, ]+)\]", str(e))
            if not m:
                raise
            bad = [int(x) for x in m.group(1).split(",")]
            excluded += [int(keep[i]) for i in bad]
            keep = np.delete(keep, bad)
    D, RH = D[keep], RH[keep]
    Ball = D.shape[0]
    Sd = od["status"].cpu().numpy()
    Hd = od["head_ft"].cpu().numpy()
    itd = od["iters"].cpu().numpy()
    n_sched = n_end = n_it = n_transfer = 0
    dhs = []
    first_bad = None
    for b in range(Ball):
        oe = se.run_gga(D[b], RH[b], do_status=True, record_schedule=True)
        es = prv_ev_serial(oe["schedule"])
        ed = prv_ev_dense(od["schedule"], b)
        same_h = es == ed
        e_end = bool(np.array_equal(oe["status"], Sd[b]))
        n_sched += int(same_h)
        n_end += int(e_end)
        n_it += int(int(oe["iters"]) == int(itd[b]))
        if not same_h:
            ne_s = [e for e in es if e[2]]
            ne_d = [e for e in ed if e[2]]
            td = (ne_s != ne_d) or (not e_end)
            n_transfer += int(td)
            if td and first_bad is None:
                i0 = next((i for i in range(min(len(ne_s), len(ne_d)))
                           if ne_s[i] != ne_d[i]), min(len(ne_s), len(ne_d)))
                first_bad = (b, i0,
                             ne_s[i0] if i0 < len(ne_s) else None,
                             ne_d[i0] if i0 < len(ne_d) else None)
        if same_h and e_end and bool(oe["converged"]) \
                and bool(od["converged"][b]):
            opened = (oe["status"] > 2) & (Sd[b] > 2)
            m = prv_conn_mask(net, opened) & (np.asarray(net.node_type) == 0)
            if m.any():
                dhs.append(float(np.abs(oe["head"] - Hd[b])[m].max()))
    cover_bad = 0
    hist = []
    for k in prv:
        cd = {v: int((Sd[:, k] == v).sum()) for v in (2, 3, 4)}
        lack = [v for v in (2, 3, 4) if cd[v] == 0]
        cover_bad += bool(lack)
        hist.append("PRV#%d CL=%d OP=%d AC=%d%s"
                    % (k, cd[2], cd[3], cd[4],
                       (" 缺态<--" if lack else "")))
    dmax = max(dhs) if dhs else 0.0
    if verbose:
        print("  %-14s B=%-3d(剔%d) 终态同 %3d/%d 转移分岔 %d 停时分岔 %d "
              "iters同 %3d/%d 三态缺阀 %d 调度同+双收敛连通域 max|dH|=%.3e "
              "ft（门 %.1f）" % (name, Ball, len(excluded), n_end, Ball,
                                n_transfer, (Ball - n_sched) - n_transfer,
                                n_it, Ball, cover_bad, dmax, DH_GATE),
              flush=True)
        print("      " + "  ".join(hist), flush=True)
        if first_bad is not None:
            b, i0, a, c = first_bad
            print(f"      首个转移分岔 b={b} 非空事件#{i0} serial={a} "
                  f"dense={c}", flush=True)
    return dict(B=Ball, end_bad=Ball - n_end, transfer=n_transfer,
                cover_bad=cover_bad, dmax=dmax, excluded=len(excluded))


def child_prv(tmp_root, B):
    """子进程：在 PRV 变异体上跑 §C 判据，打印机器可读结果。"""
    mods = load_mods(tmp_root)
    eb = tr = cb = 0
    for name in PRV_NETS:
        r = compare_prv(mods, name, B, verbose=True)
        eb += r["end_bad"]
        tr += r["transfer"]
        cb += r["cover_bad"]
    print(f"PRV-RESULT end_bad={eb} transfer={tr} cover_bad={cb}")
    return 0


def mutate_pkg(tag):
    """把 dgga 复制到临时目录并打 §D 变异（锚点必须恰中 1 次）。返回目录。"""
    old, new = MUTS_PRV[tag]
    tmp = tempfile.mkdtemp(prefix="prvmut_%s_" % tag[:2])
    shutil.copytree(os.path.join(ROOT, "dgga"), os.path.join(tmp, "dgga"))
    tgt = os.path.join(tmp, "dgga", "solver.py")
    with open(tgt, encoding="utf-8") as f:
        src = f.read()
    if src.count(old) != 1:
        raise RuntimeError("%s 锚点命中 %d 次（应为 1）" % (tag, src.count(old)))
    with open(tgt, "w", encoding="utf-8") as f:
        f.write(src.replace(old, new))
    shutil.rmtree(os.path.join(tmp, "dgga", "__pycache__"), ignore_errors=True)
    return tmp


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--child-m1":
        return child_m1(sys.argv[2], int(sys.argv[3]))
    if len(sys.argv) >= 3 and sys.argv[1] == "--child-prv":
        return child_prv(sys.argv[2], int(sys.argv[3]))

    B = 64
    nets = list(DEFAULT_NETS)
    args = sys.argv[1:]
    if "--B" in args:
        i = args.index("--B")
        B = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    if args:
        nets = args

    print("=" * 100)
    print("check_schedule.py - F2 调度级对拍：dense 批量状态机 vs epanet 串行")
    print("硬判据：逐场景 (迭代号, change 链路集) 序列逐项相等 + 终态逐元素相等"
          " + iters 相等；支标签(conv/per)翻转只报不判（Hacc 边界舍入）")
    print("=" * 100)
    mods = load_mods(None)
    print(f"\n【§A 真品对拍：{len(nets)} 网 × B={B} 极端场景】")
    bad = 0
    for name in nets:
        r = compare_net(mods, name, B)
        bad += (r["B"] - r["sched"]) + (r["B"] - r["state"]) \
            + (r["B"] - r["iters"]) + r["extra"]
    print(f"  小计：{'全部一致' if bad == 0 else f'{bad} 处不一致 <-- FAIL'}")

    print("\n【§B 检出力自证：M1 变异体（收敛即退出，门 B1 审计 §3b）】")
    tmp = tempfile.mkdtemp(prefix="schedmut_")
    shutil.copytree(os.path.join(ROOT, "dgga"), os.path.join(tmp, "dgga"))
    tgt = os.path.join(tmp, "dgga", "solver.py")
    with open(tgt, encoding="utf-8") as f:
        src = f.read()
    n_hit = src.count(M1_OLD)
    if n_hit != 1:
        print(f"  锚点命中 {n_hit} 次（应为 1）<-- FAIL")
        return 1
    with open(tgt, "w", encoding="utf-8") as f:
        f.write(src.replace(M1_OLD, M1_NEW))
    shutil.rmtree(os.path.join(tmp, "dgga", "__pycache__"), ignore_errors=True)
    pr = subprocess.run([sys.executable, "-X", "utf8",
                         os.path.abspath(__file__), "--child-m1", tmp, str(B)],
                        cwd=ROOT, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=3600)
    out = (pr.stdout or "") + (pr.stderr or "")
    line = [ln for ln in out.splitlines() if ln.startswith("M1-RESULT")]
    for ln in out.splitlines():
        if ln.strip().startswith(("Richmond", "Net", "ky", "Anytown", "M1-RESULT")):
            print("  " + ln.strip())
    if not line:
        print("  子进程无 M1-RESULT 行 <-- FAIL\n" + out[-800:])
        return 1
    kv = dict(t.split("=") for t in line[-1].split()[1:])
    sched_bad, state_bad = int(kv["sched_bad"]), int(kv["state_bad"])
    caught = sched_bad > 0
    blind = state_bad == 0
    print(f"  调度判据检出 M1：{'是（%d 场景）' % sched_bad if caught else '否 <-- FAIL'}"
          f"；终态判据对 M1 {'零检出（复现审计结论：只有调度判据抓得住）' if blind else f'也检出 {state_bad} 场景（注意：与审计结论不同，如实报）'}")

    # ---- §C：PRV 网转移子序列对拍（正品） ----
    print(f"\n【§C PRV 网调度对拍：{PRV_NETS} × B={PRV_B}+逼阀"
          f"（终态逐元素 + 转移子序列逐项 + 三态覆盖 + dH≤{DH_GATE}ft）】")
    prv_bad = 0
    for name in PRV_NETS:
        r = compare_prv(mods, name, PRV_B)
        prv_bad += (r["end_bad"] + r["transfer"] + r["cover_bad"]
                    + int(r["dmax"] > DH_GATE))
    print(f"  小计：{'全部一致' if prv_bad == 0 else f'{prv_bad} 处不一致 <-- FAIL'}")

    # ---- §D：PRV 检出力自证（MB/MD/ME 必须全红） ----
    print(f"\n【§D PRV 检出力自证：MB/MD/ME 变异体 × B={PRV_MUT_B}+逼阀"
          f"（§C 判据下必须全部变红）】")
    muts_red = 0
    mut_lines = []
    for tag in ("MB_valve_freq", "MD_open_case", "ME_y_predemand"):
        tmp2 = mutate_pkg(tag)
        pr2 = subprocess.run([sys.executable, "-X", "utf8",
                              os.path.abspath(__file__), "--child-prv", tmp2,
                              str(PRV_MUT_B)],
                             cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=3600)
        out2 = (pr2.stdout or "") + (pr2.stderr or "")
        ln2 = [ln for ln in out2.splitlines() if ln.startswith("PRV-RESULT")]
        if not ln2:
            # 变异体把批 Cholesky/求解打崩也算红（如实报）
            red = pr2.returncode != 0
            mut_lines.append(f"  {tag:<16} 子进程 rc={pr2.returncode} 无 "
                             f"PRV-RESULT（崩溃即红）：{'红' if red else '否 <-- FAIL'}")
        else:
            kv2 = dict(t.split("=") for t in ln2[-1].split()[1:])
            red = (int(kv2["end_bad"]) > 0 or int(kv2["transfer"]) > 0
                   or int(kv2["cover_bad"]) > 0)
            mut_lines.append(f"  {tag:<16} {ln2[-1]}  -> "
                             f"{'红（如期检出）' if red else '绿 <-- FAIL（判据没长牙）'}")
        muts_red += int(red)
        shutil.rmtree(tmp2, ignore_errors=True)
    for ln in mut_lines:
        print(ln)

    ok = (bad == 0) and caught and prv_bad == 0 and muts_red == 3
    print("\n" + "=" * 100)
    print(f"SCHED_SUMMARY a_bad={bad} m1_caught={int(caught)} "
          f"prv_bad={prv_bad} muts_red={muts_red}/3")
    print(f"总判定: {'PASS' if ok else 'FAIL'}  "
          f"（§A {'一致' if bad == 0 else f'{bad} 处不一致'}；"
          f"§B M1 {'被调度判据检出' if caught else '未检出'}；"
          f"§C PRV {'一致' if prv_bad == 0 else f'{prv_bad} 处不一致'}；"
          f"§D 变异体 {muts_red}/3 红）")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

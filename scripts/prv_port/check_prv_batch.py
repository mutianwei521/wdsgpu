# -*- coding: utf-8 -*-
"""check_prv_batch.py - PRV 轮验收 ①：含 PRV 网的批量极端场景对拍。

对每个含 PRV 网（缺省 L-TOWN / BWSN_Network_1 / D-Town / Richmond_standard /
ky10）跑 B=128 极端场景（gate_b1 配方：需水 ×0.05~4.0 + 随机时刻 + 单点大漏损
2%~60% + 水池水位贴边），再补一批"逼阀"场景（每阀两条：下游大漏损逼 OPEN、
近零需水+满池逼 CLOSED），然后：

  ① 硬判据：dense 批量（status_machine=True）vs epanet 串行（run_gga
     do_status=True） - **终态状态向量逐元素相等 + iters 相等 + 调度轨迹
     （valvestatus 每迭代 + linkstatus 收敛/周期，含每次调用的 change 链路集）
     逐项相等**。
  ② 头差：连通掩码口径 + 双收敛子集（gate_b1 口径），只报量级不设逐位门槛
     （两路线性求解不同，PRV 的 CBIG 行把 κ 抬到 ~1e10 会放大差异）。
  ③ 逐阀批内状态直方图（ACTIVE/OPEN/CLOSED 是否三态都出现 - 没出全会在
     总结里点名，如实报）。
  ④ 对**每个**不一致场景做首次分岔归因（判定级收紧，洞 A 修复；不抽样，
     判据数值全部印出，可审计）：
       BUG(节律) - 两路 valve 检查时刻（迭代号序列）前缀不一致（正品两路每轮
                  迭代都跑 valvestatus，MB 类降频变异在此必红）；
       stoptime - 非空 change 转移子序列逐项相同 + 终态逐元素相同，差异只在
                  空 change 的检查时刻（何时判收敛/何时停，Hacc 刀刃）；
       knife - 首分岔事件上任一路某阈值的 |边际| ≤ KNIFE_ABS（死支/钳位
                  放大的线代噪声上的判定翻转，EPANET 换平台也会翻）；
       drift - 两路边际都大（> KNIFE_ABS），跨路判据差 diff_c ≥ 较小的
                  一路边际（数值分开才解释得了翻转），**且有头场作证**：
                  分岔事件上两路 max|ΔH| ≥ diff_c/100（真漂移的头场差与判据
                  差同源同量级；M4 实测 CBIG 瞬态 |ΔH|~1e8 时 diff_c~1e2，
                  余量巨大。判据分开而头场没分开 = 判据计算被改 = BUG）。
                  drift 只能解释"两路各自自洽但被退化瞬态散开" - 它对主线
                  质量不构成豁免：主线 L-TOWN 走硬判（见下），根本不给
                  drift 通道，ME 类（Y 读错 Xflow）在 L-TOWN 必红；
       BUG - 其余一律 BUG（含：调度逐项相同而终态不同；两路判据一致
                  （diff_c<m_min）且边际都大却转移不同；边际不可比；头场
                  作证不了判据分开）。
     **退出码收紧**：任一 BUG、或串行独有非空事件即 rc 非 0；HARD_NETS
     （L-TOWN 主线）另加一票否决：终态逐元素全同 + 分岔全为停时 + 三态全出
     + 状态同双收敛子集连通域头差 ≤ DH_GATE=0.5 ft。旧版只看 BUG 类计数且
     只归因前 40 个，MD/ME 类实现错误（终态 6/18 不同）会被归成 drift 放行
     （data/prv_release_audit_wip.txt 洞 A）。
  ⑤ diag_ratio（κ 代理）分布 + L-TOWN 收敛帧真 κ（np.linalg.cond，B=1 名义帧）。

用法：python -X utf8 scripts/prv_port/check_prv_batch.py [--B 128]
      [--pkg DIR] [网名 ...]      （--pkg 指向含 dgga/ 的目录，变异体自证用）
退出码 0 = 每个不一致场景都被归因为 stoptime/knife/drift 之一（判据数据已印出）
且无 BUG、无 extra；硬判网另过一票否决（终态/停时/三态/头差门）。
"""

import os
import sys
import time
import zlib

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PKG = ROOT
if "--pkg" in sys.argv:                        # 变异体自证：dgga 从别处导入
    _i = sys.argv.index("--pkg")
    _PKG = os.path.abspath(sys.argv[_i + 1])
    del sys.argv[_i:_i + 2]
sys.path.insert(0, _PKG)

import dgga                                    # noqa: E402
import torch                                   # noqa: E402
from dgga.parse import parse_inp               # noqa: E402
from dgga.solver import GGASolver              # noqa: E402

assert os.path.abspath(dgga.__file__).startswith(_PKG), dgga.__file__

PUB = os.path.join(ROOT, "networks", "public")
CLEAN = os.path.join(PUB, "_cleaned")
DEFAULT_NETS = ["L-TOWN", "BWSN_Network_1", "D-Town", "Richmond_standard",
                "ky10"]
ST = {0: "XHEAD", 1: "TEMPCLOSED", 2: "CLOSED", 3: "OPEN", 4: "ACTIVE",
      5: "XFLOW", 6: "XFCV", 7: "XPRESSURE"}
LT = ["CVPIPE", "PIPE", "PUMP", "PRV", "PSV", "PBV", "FCV", "TCV", "GPV"]
# 刀刃判定：首次分岔事件上，两路对同一判据的边际都落在该量级以内（或异号）
# 即判为"线代噪声驱动的死支/钳位分岔"。这个数不是容差门槛，是**归因阈**：
# 死支的 dh 噪声 ~1e-10 ft 经 RQtol 钳位 P=1e7 放大即 ~1e-3 cfs 量级。
KNIFE_ABS = 1.0e-2
# drift 类的**头场漂移作证**：跨路判据差 diff_c 只有当分岔事件上两路头场
# 已按同一量级分开（drift_all ≥ diff_c / DRIFT_WITNESS）才可归为漂移 -
# 判据是 H/q 的函数，判据分开而头场没分开的"漂移"无物理来源，是判据计算
# 本身被改（实现错误）。真漂移实测（M4/BWSN）：diff_c~1e2 时事件处
# |ΔH|~1e8（CBIG 瞬态），余量巨大。
DRIFT_WITNESS = 100.0
# 主线（硬判）网：终态逐元素全同 + 转移子序列全同（仅停时可豁免）+ 三态
# 全出 + 双收敛头差 ≤ DH_GATE 一票否决；这些网的正品实测就该干净
# （aud_prv_states：L-TOWN 134/134）。
HARD_NETS = {"L-TOWN"}
# 状态同+双收敛子集的连通域头差硬门（ft，仅 HARD_NETS）：与 aud_prv_states
# 同口径（κ~1.6e11 实测 9.8e-05 ft；0.5 只有实现错误才碰得到）。
DH_GATE = 0.5


def inp_of(name):
    p = os.path.join(CLEAN, name + ".inp")
    return p if os.path.exists(p) else os.path.join(PUB, name + ".inp")


def make_scenarios(net, B, seed, prv_links):
    """gate_b1 极端配方 + 逼阀场景（每 PRV 两条：逼 OPEN / 逼 CLOSED）。"""
    rng = np.random.default_rng(seed)
    N = net.N
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
        corner = (b % 4 == 0)
        if corner:
            mult = float(np.exp(rng.uniform(np.log(0.01), np.log(0.10))))
        else:
            mult = float(np.exp(rng.uniform(np.log(0.05), np.log(4.0))))
        d = net.demand_cfs_at(t) * mult
        tot = float(np.abs(d[junc]).sum())
        if not corner:
            j = int(junc[rng.integers(0, junc.size)])
            d[j] += tot * float(rng.uniform(0.02, 0.60))
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(t),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            n = int(n)
            u = rng.random()
            if corner:
                rh[n] = hmax[i]
            elif u < 0.15:
                rh[n] = hmin[i]
            elif u < 0.30:
                rh[n] = hmax[i]
            else:
                lo, hi = float(hmin[i]), float(hmax[i])
                rh[n] = lo + (hi - lo) * float(rng.random()) if hi > lo \
                    else float(np.asarray(net.tank_h0)[i])
        rows.append((d, rh))
    # 逼阀场景（每阀两条）：
    # (a) 逼 OPEN：需水 ×8 + 下游节点大漏损 2×全网需水 + 空池 - 上游供不上
    #     设定压力（prvstatus:275 h1-hml < hset-htol）；
    # (b) 逼 CLOSED：需水 ×0.2 + **下游节点注入**（负需水，EPANET 合法语义）
    #     + 满池 - 下游压力被顶过设定（prvstatus:281 h2 >= hset+htol → ACTIVE，
    #     继而 :274 反流 q<-qtol → CLOSED）。
    d0 = net.demand_cfs_at(0)
    tot0 = float(np.abs(d0[junc]).sum())
    for k in prv_links:
        n2 = int(net.link_n2[k])
        d = d0 * 25.0
        if nt[n2] == 0:
            d[n2] += 15.0 * max(tot0, 1e-6)
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            rh[int(n)] = hmin[i]
        rows.append((d, rh))
        d = d0 * 0.2
        if nt[n2] == 0:
            d[n2] -= 1.0 * max(tot0, 1e-6)
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            rh[int(n)] = hmax[i]
        rows.append((d, rh))
    D = np.stack([r[0] for r in rows])
    RH = np.stack([r[1] for r in rows])
    return D, RH


def connected_mask(net, opened):
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    fixed = np.asarray(net.node_type) != 0
    out = np.zeros((opened.shape[0], net.N), dtype=bool)
    for b in range(opened.shape[0]):
        reach = fixed.copy()
        e1, e2 = n1[opened[b]], n2[opened[b]]
        while True:
            new = reach.copy()
            np.logical_or.at(new, e2, reach[e1])
            np.logical_or.at(new, e1, reach[e2])
            if np.array_equal(new, reach):
                break
            reach = new
        out[b] = reach
    return out


def serial_events(sched):
    """run_gga schedule → [(it, kind, changeset)]；kind ∈ valve/conv/per。
    空 pswitch 丢弃（dense 构造期守卫已排除 junction 压力控制），非空计 extra。"""
    ev, extra = [], 0
    for it, kind, changed in sched:
        if kind == "valve":
            ev.append((int(it), "valve", tuple(changed)))
        elif kind == "ls_conv":
            ev.append((int(it), "conv", tuple(changed)))
        elif kind == "ls_per":
            ev.append((int(it), "per", tuple(changed)))
        elif changed:
            extra += 1
    return ev, extra


def dense_events(out_sched, b):
    ev = []
    for e in out_sched:
        if e.get("kind") == "valve":
            if bool(e["act"][b]):
                ev.append((int(e["it"]), "valve",
                           tuple(np.where(e["chg"][b])[0].tolist())))
        elif bool(e["conv"][b]):
            ev.append((int(e["it"]), "conv",
                       tuple(np.where(e["chg"][b])[0].tolist())))
        elif bool(e["per"][b]):
            ev.append((int(e["it"]), "per",
                       tuple(np.where(e["chg"][b])[0].tolist())))
    return ev


def strip_tag(ev):
    """支标签 conv/per 软化（Hacc 边界舍入，check_schedule 口径），valve 保留。"""
    return [(it, ("valve" if k == "valve" else "ls"), chg) for it, k, chg in ev]


# ----------------------------------------------------------------------
# 首次分岔归因：两路重跑该场景并抓 (S,H,q) 快照 → 差异链路的判据边际
# ----------------------------------------------------------------------
def _snap_serial(se, d, rh):
    snaps = []
    ov, ol = se._valvestatus, se._linkstatus

    def vs(S, K, H, q):
        snaps.append(("valve", S.copy(), H.copy(), q.copy(), K.copy()))
        return ov(S, K, H, q)

    def ls(S, K, H, q):
        snaps.append(("ls", S.copy(), H.copy(), q.copy(), K.copy()))
        return ol(S, K, H, q)

    se._valvestatus, se._linkstatus = vs, ls
    try:
        se.run_gga(d, rh, do_status=True, record_schedule=True)
    finally:
        se._valvestatus, se._linkstatus = ov, ol
    return snaps


def _snap_dense(sd, d, rh):
    snaps = []
    ov, ol = sd._prvstatus_batch, sd._linkstatus_batch

    def vs(S, H, q):
        snaps.append(("valve", S[0].numpy().copy(), H[0].numpy().copy(),
                      q[0].numpy().copy(), None))
        return ov(S, H, q)

    def ls(S, H, q):
        snaps.append(("ls", S[0].numpy().copy(), H[0].numpy().copy(),
                      q[0].numpy().copy(), None))
        return ol(S, H, q)

    sd._prvstatus_batch, sd._linkstatus_batch = vs, ls
    try:
        with torch.no_grad():
            sd.solve(d[None, :], rh[None, :], status_machine=True)
    finally:
        sd._prvstatus_batch, sd._linkstatus_batch = ov, ol
    return snaps


def _margins(sol, net, kind, k, S, H, q):
    """该链路本次状态判定的**各阈值有符号边际**向量（np.array）。

    约定：向量各分量的含义只依赖 (链路类型, 当前状态)，两路可逐分量比。"""
    lt = int(np.asarray(net.link_type)[k])
    n1 = int(net.link_n1[k])
    n2 = int(net.link_n2[k])
    htol, qtol = sol.htol, sol.qtol
    if kind == "valve" and lt == 3:                       # PRV（prvstatus）
        kk = float(sol.km_valve_ml_np[k])
        hset = float(sol.elev_np[n2] + sol.init_setting[k])
        h1, h2, qk = float(H[n1]), float(H[n2]), float(q[k])
        hml = kk * qk * qk
        s = int(S[k])
        cand = [qk + qtol]
        if s == 4:
            cand.append(h1 - hml - (hset - htol))
        elif s == 3:
            cand.append(h2 - (hset + htol))
        elif s == 2:
            cand += [h1 - (hset + htol), h2 - (hset - htol),
                     h1 - (hset - htol), (h1 - h2) - htol]
        return np.asarray(cand)
    if lt == 0:                                           # CVPIPE（cvstatus）
        dh = float(H[n1] - H[n2])
        qk = float(q[k])
        return np.asarray([abs(dh) - htol, dh + htol, qk + qtol])
    if lt == 2:                                           # PUMP（pumpstatus）
        gain = float(H[n2] - H[n1])
        sp = float(sol.init_setting[k])
        hmx = np.inf if sol.is_chp_np[k] else sp * sp * sol.pl_hmax[k]
        return np.asarray([gain - (hmx + htol)] if np.isfinite(hmx) else [])
    if lt == 1:                                           # PIPE（tankstatus+cv）
        cand = []
        for n in (n1, n2):
            if sol.is_tank_node[n]:
                cand += [float(H[n]) - (sol.tn_hmax[n] - htol),
                         float(H[n]) - (sol.tn_hmin[n] + htol)]
        dh = float(H[n1] - H[n2])
        qk = float(q[k])
        cand += [abs(dh) - htol, dh + htol, qk + qtol]
        return np.asarray(cand)
    return np.asarray([])


def attribute_divergence(net, se, sd, d, rh, ev_s, ev_d, state_same, dr_b):
    """定位首个不同事件并分类。返回 (描述串, 类别)。

    类别与判据（洞 A 收紧版；每条判据的数值都写进描述串，可审计）：
      BUG(节律) - 两路 valve 检查时刻（迭代号序列）前缀不一致：正品两路
                  都每轮迭代跑 valvestatus（hydsolver.c:161），前缀必须逐项
                  相同（MB 类降频变异在此必红，含无转移场景）。
      stoptime - 非空 change 转移子序列逐项相同 且 终态逐元素相同：
                  差异只在空 change 的检查时刻（何时判收敛/何时停）。
      knife - 首分岔事件的差异链路上，任一路最小 |边际| ≤ KNIFE_ABS。
      drift - m_min > KNIFE_ABS 且 diff_c = max|边际_s − 边际_d| ≥ m_min
                  （两路判据数值真分开了，翻转被数值差解释）**且有头场作证**：
                  分岔事件上两路 max|ΔH| ≥ diff_c/DRIFT_WITNESS（判据是 H/q
                  的函数，判据分开而头场没分开的"漂移"无物理来源）。
      BUG - 其余一律 BUG：调度逐项相同而终态不同（同轨迹必同终态）、
                  两路判据一致（diff_c < m_min）且边际都大却转移不同、
                  差异链路取不到可比边际、头场作证不了判据分开。
    """
    hs, hd = strip_tag(ev_s), strip_tag(ev_d)
    ne_s = [e for e in hs if e[2]]
    ne_d = [e for e in hd if e[2]]
    vs_i = [it for it, k, _c in hs if k == "valve"]
    vd_i = [it for it, k, _c in hd if k == "valve"]
    mv = min(len(vs_i), len(vd_i))
    if vs_i[:mv] != vd_i[:mv]:
        j = next(i for i in range(mv) if vs_i[i] != vd_i[i])
        return ("valve 检查节律前缀不一致@第%d次检查：serial 迭代号=%s "
                "dense=%s（正品两路每轮迭代都跑 valvestatus，前缀必须逐项"
                "相同）⇒ 实现错误" % (j, vs_i[:j + 2], vd_i[:j + 2]), "BUG")
    if hs == hd:
        return ("调度逐项相同但终态不同：同一转移轨迹必须给出同一终态"
                " ⇒ 实现错误", "BUG")
    n = min(len(hs), len(hd))
    idx = next((i for i in range(n) if hs[i] != hd[i]), n)
    if ne_s == ne_d and state_same:
        return ("仅停时/检查节律：非空转移子序列逐项相同、终态逐元素相同；"
                "首差事件#%d serial=%s dense=%s（差异均为空 change）"
                % (idx, ev_s[idx] if idx < len(ev_s) else None,
                   ev_d[idx] if idx < len(ev_d) else None), "stoptime")
    ss = _snap_serial(se, d, rh)
    dd = _snap_dense(sd, d, rh)
    it_s, k_s, ch_s = ev_s[idx][:3] if idx < len(ev_s) else (None, None, ())
    it_d, k_d, ch_d = ev_d[idx] if idx < len(ev_d) else (None, None, ())
    links = sorted(set(ch_s) ^ set(ch_d))
    if not links:
        # 首差事件本身空 change（节律差）⇒ 用首个不同的**非空转移**定位链路
        m = min(len(ne_s), len(ne_d))
        j = next((i for i in range(m) if ne_s[i] != ne_d[i]), m)
        cs = ne_s[j][2] if j < len(ne_s) else ()
        cd = ne_d[j][2] if j < len(ne_d) else ()
        links = sorted(set(cs) ^ set(cd)) or sorted(set(cs) | set(cd))
    # 漂移上下文：分岔事件（含其前一事件）两路头差
    drift_conn = drift_all = 0.0
    for j in range(max(0, idx - 1), min(idx + 1, len(ss), len(dd))):
        _, Ssj, Hsj, _, _ = ss[j]
        _, Sdj, Hdj, _, _ = dd[j]
        opened = (Ssj > 2) & (Sdj > 2)
        conn = connected_mask(net, opened[None, :])[0] \
            & (np.asarray(net.node_type) == 0)
        dH = np.abs(Hsj - Hdj)
        drift_all = max(drift_all, float(dH.max()))
        if conn.any():
            drift_conn = max(drift_conn, float(dH[conn].max()))
    cls = None
    det = []
    for k in links[:4]:
        ms = md = None
        if idx < len(ss):
            kind, S, H, q, _ = ss[idx]
            ms = _margins(se, net, kind, k, S, H, q)
        if idx < len(dd):
            kind, S, H, q, _ = dd[idx]
            md = _margins(sd, net, kind, k, S, H, q)
        if (ms is None or md is None or ms.size == 0 or md.size == 0
                or ms.size != md.size):
            k_cls = "BUG"
            det.append("%s#%d(%s) 边际不可比(s=%s d=%s) ⇒ BUG"
                       % (LT[int(np.asarray(net.link_type)[k])], k,
                          net.link_id[k],
                          None if ms is None else ms.size,
                          None if md is None else md.size))
        else:
            m_min = float(min(np.abs(ms).min(), np.abs(md).min()))
            diff_c = float(np.abs(ms - md).max())
            if m_min <= KNIFE_ABS:
                k_cls = "knife"
                why = "m_min=%.2e ≤ KNIFE=%.0e" % (m_min, KNIFE_ABS)
            elif diff_c >= m_min and drift_all >= diff_c / DRIFT_WITNESS:
                k_cls = "drift"
                why = ("m_min=%.2e > KNIFE 且 diff_c=%.2e ≥ m_min 且 头场"
                       "作证 |ΔH|all=%.1e ≥ diff_c/%.0f"
                       % (m_min, diff_c, drift_all, DRIFT_WITNESS))
            else:
                k_cls = "BUG"
                why = ("m_min=%.2e > KNIFE=%.0e 且 (diff_c=%.2e < m_min 判据"
                       "两路一致仍转移不同 或 |ΔH|all=%.1e < diff_c/%.0f "
                       "头场作证不了判据分开)"
                       % (m_min, KNIFE_ABS, diff_c, drift_all, DRIFT_WITNESS))
            det.append("%s#%d(%s) %s ⇒ %s"
                       % (LT[int(np.asarray(net.link_type)[k])], k,
                          net.link_id[k], why, k_cls))
        order = {"BUG": 3, "drift": 2, "knife": 1}
        if cls is None or order[k_cls] > order[cls]:
            cls = k_cls
    if cls is None:
        cls = "BUG"
        det.append("差异链路集为空且转移子序列不可对位 ⇒ BUG")
    desc = ("首分岔@事件%d serial=(%s,%s,%s) dense=(%s,%s,%s) "
            "漂移|ΔH| conn=%.2e all=%.2e diag_ratio=%.1e；%s"
            % (idx, it_s, k_s, list(ch_s), it_d, k_d, list(ch_d),
               drift_conn, drift_all, dr_b, "; ".join(det)))
    return desc, cls


# ----------------------------------------------------------------------
def check_net(name, B, seed):
    inp = inp_of(name)
    net = parse_inp(inp)
    lt = np.asarray(net.link_type)
    prv = np.where(lt == 3)[0].tolist()
    D, RH = make_scenarios(net, B, seed, prv)
    Ball = D.shape[0]

    se = GGASolver(net, mode="epanet", inp_path=inp)
    sd = GGASolver(net, mode="dense", inp_path=inp, dense_status_machine=True)

    # badvalve 情形（批 Cholesky 非正定，solver 显式 raise 报样本号）：按
    # dense_gap_plan §5 的对策 - 剔出失败样本重跑，其余照常；被剔样本单走
    # epanet 并如实报告（EPANET 对它们走 badvalve 改判 XPRESSURE 重试）。
    excluded = []
    keep = np.arange(Ball)
    t0 = time.time()
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
    t_dn = time.time() - t0
    if excluded:
        # 被剔样本的 epanet 侧行为（badvalve 是否触发 → XPRESSURE 状态）
        xp = []
        for b in excluded:
            ob = se.run_gga(D[b], RH[b], do_status=True)
            xp.append(int((np.asarray(ob["status"]) == 7).sum()))
        print("  [badvalve] 批 Cholesky 剔除样本 %s（epanet 侧 XPRESSURE 阀数 %s）"
              % (excluded, xp))
        D, RH = D[keep], RH[keep]
        Ball = D.shape[0]
    S_dn = od["status"].numpy().astype(np.int8)
    it_dn = od["iters"].numpy()
    cv_dn = od["converged"].numpy()
    H_dn = od["head_ft"].numpy()
    Q_dn = od["flow_cfs"].numpy()
    dr = od["diag_ratio"].numpy()

    t0 = time.time()
    S_ep = np.zeros_like(S_dn)
    it_ep = np.zeros(Ball, dtype=np.int64)
    cv_ep = np.zeros(Ball, dtype=bool)
    H_ep = np.zeros_like(H_dn)
    Q_ep = np.zeros_like(Q_dn)
    ev_all_s = []
    n_extra = 0
    for b in range(Ball):
        oe = se.run_gga(D[b], RH[b], do_status=True, record_schedule=True)
        S_ep[b] = oe["status"]
        it_ep[b] = int(oe["iters"])
        cv_ep[b] = bool(oe["converged"])
        H_ep[b] = oe["head"]
        Q_ep[b] = oe["flow"]
        ev, extra = serial_events(oe["schedule"])
        ev_all_s.append(ev)
        n_extra += extra
    t_ep = time.time() - t0

    # ① 硬判据
    n_sched = n_flip = n_state = n_iter = 0
    div = []
    for b in range(Ball):
        ev_d = dense_events(od["schedule"], b)
        ev_s = ev_all_s[b]
        ok_sched = strip_tag(ev_d) == strip_tag(ev_s)
        n_sched += ok_sched
        n_flip += ok_sched and (ev_d != ev_s)
        ok_state = bool(np.array_equal(S_dn[b], S_ep[b]))
        n_state += ok_state
        n_iter += int(it_dn[b]) == int(it_ep[b])
        if not (ok_sched and ok_state):
            div.append(b)

    # ② 头差（连通掩码 + 双收敛子集）
    jm = np.asarray(net.node_type) == 0
    opened = (S_ep > 2) & (S_dn > 2)
    conn = connected_mask(net, opened) & jm[None, :]
    AD = np.abs(H_ep - H_dn)
    per = np.where(conn, AD, 0.0).max(axis=1)
    match = np.array([bool(np.array_equal(S_dn[b], S_ep[b]))
                      for b in range(Ball)])
    bc = cv_ep & cv_dn & match
    dH_bc = float(per[bc].max()) if bc.any() else float("nan")
    dQ_bc = float(np.abs(Q_ep - Q_dn)[bc][opened[bc]].max()) \
        if bc.any() and opened[bc].any() else float("nan")

    # ③ 逐阀状态直方图
    hist_lines = []
    missing = 0
    for k in prv:
        u, c = np.unique(S_dn[:, k], return_counts=True)
        seen = {int(a) for a in u}
        lack = {2, 3, 4} - seen
        missing += bool(lack)
        hist_lines.append("PRV %s: %s%s" % (
            net.link_id[k],
            " / ".join("%s:%d" % (ST[int(a)], int(cc))
                       for a, cc in zip(u, c)),
            ("   <-- 缺 " + ",".join(ST[x] for x in sorted(lack))
             if lack else "")))

    # ④ 分岔归因（**全部**不一致场景逐个做快照级归因，不抽样、不设上限；
    #    stoptime/knife/drift 可接受，出现任一 BUG 即 FAIL - 洞 A 收紧）
    attr_lines = []
    n_knife = n_drift = n_stop = n_bug = 0
    n_attr = len(div)
    for b in div:
        desc, cls = attribute_divergence(net, se, sd, D[b], RH[b],
                                         ev_all_s[b],
                                         dense_events(od["schedule"], b),
                                         bool(np.array_equal(S_dn[b],
                                                             S_ep[b])),
                                         float(dr[b]))
        n_knife += cls == "knife"
        n_drift += cls == "drift"
        n_stop += cls == "stoptime"
        n_bug += cls == "BUG"
        if len(attr_lines) < 8 or cls == "BUG":
            attr_lines.append("b=%d [%s] %s" % (b, cls, desc))
    knife_ok = (n_bug == 0)

    print("\n=== %s (Nj=%d, L=%d, PRV=%d, B=%d+%d逼阀) ===" %
          (name, sd.Nj, net.L, len(prv), B, Ball - B))
    print("  ① 调度(硬)相同 %d/%d（仅支标签翻转 %d） 终态相同 %d/%d  "
          "iters相同 %d/%d  串行独有非空事件 %d" %
          (n_sched, Ball, n_flip, n_state, Ball, n_iter, Ball, n_extra))
    print("  ② 头差（连通掩码，状态相同且双收敛的 %d/%d 场景）："
          "max|dH|=%.3e ft  max|dQ|=%.3e cfs" %
          (int(bc.sum()), Ball, dH_bc, dQ_bc))
    print("     收敛：epanet %d/%d  dense %d/%d   迭代数逐场景相等 %d/%d" %
          (int(cv_ep.sum()), Ball, int(cv_dn.sum()), Ball, n_iter, Ball))
    print("     diag_ratio（κ 代理）：中位 %.2e  最大 %.2e" %
          (float(np.median(dr)), float(dr.max())))
    print("     耗时：epanet 串行 %.1fs  dense 批 %.1fs" % (t_ep, t_dn))
    print("  ③ 逐阀批内状态直方图：")
    for ln in hist_lines:
        print("       " + ln)
    if div:
        print("  ④ 不一致场景 %d 个（全部归因，无抽样）："
              "停时 %d / 刀刃 %d / 漂移 %d / BUG %d（判据：valve 节律前缀必须"
              "逐项同；knife=|边际|≤%.0e；drift=diff_c≥m_min 且事件处 "
              "max|ΔH|≥diff_c/%.0f（头场作证）；其余 BUG）："
              % (len(div), n_stop, n_knife, n_drift,
                 n_bug, KNIFE_ABS, DRIFT_WITNESS))
        for ln in attr_lines:
            print("       " + ln)
    else:
        print("  ④ 无不一致场景")
    # ④' 主线（硬判）网一票否决：终态全同 + 仅停时分岔 + 三态全出 + 头差门
    hard = name in HARD_NETS
    hard_ok = True
    if hard:
        dh_ok = (not np.isfinite(dH_bc)) or dH_bc <= DH_GATE
        hard_ok = (n_state == Ball and n_stop == len(div) and missing == 0
                   and dh_ok)
        print("  ④' 硬判网（%s）：终态 %d/%d 全同=%s；分岔全为停时=%s；"
              "三态全出=%s；双收敛头差 %.3e ≤ %.1f ft=%s  =>  %s"
              % (name, n_state, Ball, n_state == Ball,
                 n_stop == len(div), missing == 0, dH_bc, DH_GATE, dh_ok,
                 "PASS" if hard_ok else "FAIL"))
    return dict(name=name, B=Ball, sched=n_sched, state=n_state,
                iters=n_iter, extra=n_extra, div=len(div),
                knife=n_knife, drift=n_drift, stop=n_stop, bug=n_bug,
                excluded=len(excluded),
                knife_ok=knife_ok and hard_ok, dH_bc=dH_bc, dQ_bc=dQ_bc,
                n_bc=int(bc.sum()), missing=missing,
                dr_max=float(dr.max()))


def ltown_kappa():
    """⑤ L-TOWN 收敛帧真 κ（cond2 of A，B=1 名义帧，最后一轮迭代的 A）。"""
    inp = inp_of("L-TOWN")
    net = parse_inp(inp)
    sd = GGASolver(net, mode="dense", inp_path=inp, dense_status_machine=True)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5 * (np.asarray(net.tank_hmin)[:tn.size]
                        + np.asarray(net.tank_hmax)[:tn.size])
    cap = {}
    orig = torch.linalg.cholesky_ex

    def spy(A, *a, **kw):
        cap["A"] = A.detach().clone()
        return orig(A, *a, **kw)

    torch.linalg.cholesky_ex = spy
    try:
        with torch.no_grad():
            out = sd.solve(d0[None, :], rh0[None, :], status_machine=True)
    finally:
        torch.linalg.cholesky_ex = orig
    A = cap["A"].numpy()[0]
    k2 = float(np.linalg.cond(A, 2))
    print("\n⑤ L-TOWN 名义帧（iters=%d, converged=%s）收敛轮 A：cond2=%.3e，"
          "diag_ratio=%.3e，ACTIVE PRV=%d" %
          (int(out["iters"][0]), bool(out["converged"][0]), k2,
           float(out["diag_ratio"][0]),
           int((out["status"].numpy()[0][np.asarray(net.link_type) == 3] == 4)
               .sum())))
    return k2


def main():
    B = 128
    nets = list(DEFAULT_NETS)
    args = sys.argv[1:]
    if "--B" in args:
        i = args.index("--B")
        B = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    if args:
        nets = args
    print("=" * 100)
    print("check_prv_batch.py - PRV 批量极端场景对拍（dense+SM vs epanet 串行）")
    print("=" * 100)
    res = []
    for name in nets:
        res.append(check_net(name, B, zlib.crc32(name.encode())))
    k2 = ltown_kappa() if "L-TOWN" in nets else float("nan")
    print("\n" + "=" * 100)
    tot = sum(r["B"] for r in res)
    ts = sum(r["sched"] for r in res)
    tst = sum(r["state"] for r in res)
    ok = all(r["knife_ok"] for r in res) and all(r["extra"] == 0 for r in res)
    print("判定规则（洞 A 收紧）：不一致场景**全部**归因，任一 BUG（含 valve "
          "节律前缀不同、同调度不同终态、判据一致仍转移不同、头场作证不了"
          "判据分开、边际不可比）即 FAIL；串行独有非空事件>0 即 FAIL；"
          "硬判网（%s）另加一票否决：终态全同+仅停时+三态全出+双收敛头差"
          "≤%.1f ft。" % (sorted(HARD_NETS), DH_GATE))
    print("总计：调度相同 %d/%d，终态相同 %d/%d，判定：%s"
          % (ts, tot, tst, tot, "是" if ok else "否 <-- FAIL"))
    for r in res:
        print("  %-18s 调度 %3d/%3d 终态 %3d/%3d iters %3d/%3d "
              "分岔 %d(停时%d/刀刃%d/漂移%d/BUG%d) badvalve剔 %d "
              "双收敛头差 %.2e ft（%d 场景） 三态缺阀 %d  κ代理max %.1e" %
              (r["name"], r["sched"], r["B"], r["state"], r["B"], r["iters"],
               r["B"], r["div"], r["stop"], r["knife"], r["drift"], r["bug"],
               r["excluded"], r["dH_bc"], r["n_bc"],
               r["missing"], r["dr_max"]))
    print("总判定: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

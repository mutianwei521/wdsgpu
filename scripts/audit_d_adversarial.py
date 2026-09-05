# -*- coding: utf-8 -*-
"""audit_d_adversarial.py - 阶段 D 补做审计的动态对抗实测（5 项）。

① PRV/PSV 三态边界：prv_adv_{active,open,closed} + psv_adv_active 四网
   GGASolver(status_machine=True) vs DLL（EN_solveH），门槛 dH<1e-6 ft、
   dQ<1e-6 cfs、二值状态一致、迭代数相等、内部态=预期；ACTIVE 网另做
   上下游（J1/J2/J3）demand 梯度 implicit 伴随 vs solve_polished 中央差分。
② D-W 边界数值连续性（pub_balerma 管道逐点扫 Re=2000/4000 两侧：hloss 连续、
   hgrad 有限且正、跨界局部 Lipschitz 一致）+ pub_balerma implicit vs FD
   ≥6 坐标（名义需水与 1e-3×需水两场景覆盖层流/Dunlop/Swamee-Jain 三支）。
③ 规则引擎 priority 仲裁：EXA5 生成三个变体 INP（加对抗规则 RULE 900 与
   PRIORITY：A=1/2 CLOSED 胜、B=2/1 OPEN 胜、C=1/1 平手首规则胜），
   EpsDriver 完整自主 EPS vs DLL solve_eps 逐帧对拍（t 序列/头/流量/状态/
   迭代数），并断言 A 与 B 的 PMP-2 状态轨迹不同、B 与 C 相同（仲裁可观测）。
④ FCV ACTIVE 梯度冒烟（fcv_smoke）：冻结 ACTIVE 态，implicit vs FD
   （demand/res_head/r_hw 共 5 坐标）。
⑤ CUSTOM 泵段选择（pub_richmond_standard 帧 6 替换回放冻结）：泵 p3/p5
   流量落段中、p4 距段点 0.015（用户单位）近边界，speed(ω) 与 demand 坐标
   implicit vs FD，并核 FD 扰动解不跨段。

门槛：前向 dH<1e-6 ft、dQ<1e-6 cfs；梯度逐坐标 rel<1e-6（|Δ|<1e-9 按一致计）。
"""

import bisect
import ctypes
import os
import re
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import Net, parse_inp                        # noqa: E402
from dgga.solver import GGASolver, A1, A2                    # noqa: E402
from dgga.eps import EpsDriver                               # noqa: E402
from dgga.autodiff import ImplicitGGASolve, solve_polished   # noqa: E402
from dgga.epanet_ref import Epanet, EN_STATUS                # noqa: E402

VAR = os.path.join(ROOT, "networks", "variants")
REF = os.path.join(ROOT, "data", "reference")
TMP = os.path.join(ROOT, "data", "audit_d_tmp")
TOL_H, TOL_Q, TOL_QC, TOL_G = 1e-6, 1e-6, 1e-5, 1e-6


# ---------------------------------------------------------------- 公共工具
def dll_single(inp):
    """DLL 单帧 EN_solveH + 二值状态。"""
    with Epanet(inp) as en:
        r = en.solve_single()
        val = ctypes.c_double()
        n_links = en.counts()["links"]
        st = np.empty(n_links, dtype=np.int8)
        for i in range(1, n_links + 1):
            en._check(en.lib.EN_getlinkvalue(en._ph, i, EN_STATUS,
                                             ctypes.byref(val)), "EN_STATUS")
            st[i - 1] = int(val.value)
        r["status"] = st
        return r


def grad_rows_check(rows, label):
    """逐坐标 implicit vs FD 相对误差表；返回 (ok, worst)。"""
    ok = True
    worst = 0.0
    for kind, i, ga, gf in rows:
        rel = abs(ga - gf) / max(abs(ga), abs(gf), 1e-12)
        if abs(ga - gf) < 1e-9:
            rel = 0.0                    # 低于中央差分噪声底按一致计
        worst = max(worst, rel)
        flag = "" if rel < TOL_G else "  <-- 超限"
        ok = ok and rel < TOL_G
        print(f"    {label} {kind:<9}[{i:>4}] adj={ga:>16.8e} fd={gf:>16.8e} "
              f"rel={rel:.3e}{flag}")
    return ok, worst


def implicit_grads(s, d0, rh0, ke0, r0, wH, wQ, S=None, K=None):
    """L=Σ wH·H+Σ wQ·Q 的伴随梯度 (gd, grh, gr, gsp)。S/K=None 时不冻结。"""
    dt = torch.float64
    td = torch.tensor(d0, dtype=dt, requires_grad=True)
    trh = torch.tensor(np.nan_to_num(rh0), dtype=dt, requires_grad=True)
    tke = torch.tensor(ke0, dtype=dt, requires_grad=True)
    tr = torch.tensor(r0, dtype=dt, requires_grad=True)
    tsp = None if K is None else torch.tensor(K, dtype=dt, requires_grad=True)
    head, flow, _ = ImplicitGGASolve.apply(td, trh, tke, tr, s, 1e-12, 200, 3,
                                           tsp, None, None, S)
    L = (head * torch.tensor(wH, dtype=dt)).sum() \
        + (flow * torch.tensor(wQ, dtype=dt)).sum()
    L.backward()
    gsp = None if tsp is None else tsp.grad.numpy()
    return td.grad.numpy(), trh.grad.numpy(), tr.grad.numpy(), gsp


def fd_loss(s, d0, rh0, ke0, r0, wH, wQ, S=None, K=None):
    """返回 lossf(dd, rr, rhh, sp)：solve_polished 前向的标量损失。"""
    def lossf(dd=None, rr=None, rhh=None, sp=None):
        sol = solve_polished(s, d0 if dd is None else dd,
                             rh0 if rhh is None else rhh, ke0,
                             r0 if rr is None else rr,
                             speed=(K if sp is None else sp), status=S)
        return float((sol["head"][0] * wH).sum() + (sol["q"][0] * wQ).sum())
    return lossf


# ---------------------------------------------------------------- ① PRV 三态
CASES1 = [("prv_adv_active", "V1", "ACTIVE"),
          ("prv_adv_open", "V1", "OPEN"),
          ("prv_adv_closed", "V1", "CLOSED"),
          ("psv_adv_active", "V1", "ACTIVE")]


def item1():
    print("① PRV/PSV 三态 vs DLL + ACTIVE 上下游 demand 梯度")
    ok = True
    worst_f = worst_g = 0.0
    for stem, vid, expect in CASES1:
        inp = os.path.join(VAR, f"{stem}.inp")
        net = parse_inp(inp)
        s = GGASolver(net, mode="epanet", inp_path=inp)
        ref = dll_single(inp)
        d0 = net.demand_cfs_at(0)
        rh0 = net.reservoir_head_ft_at(0)
        r = s.solve(d0, rh0, status_machine=True)
        jm = np.asarray(net.node_type) == 0
        open_my = r["status"].numpy() > 2
        dH = np.abs(r["head_ft"].numpy() - ref["head_ft"])[jm].max()
        q_api = np.where(open_my, r["flow_cfs"].numpy(), 0.0)
        opened = ref["status"] > 0
        dQ = np.abs(q_api - ref["flow_cfs"])[opened].max() if opened.any() else 0.0
        st_ok = bool(np.array_equal(open_my.astype(np.int8), ref["status"]))
        it_ok = int(r["iters"]) == int(ref["iterations"])
        kv = net.link_id.index(vid)
        st_name = {2: "CLOSED", 3: "OPEN", 4: "ACTIVE"}.get(
            int(r["status"].numpy()[kv]), "?")
        good = dH < TOL_H and dQ < TOL_Q and st_ok and it_ok and st_name == expect
        ok = ok and good
        worst_f = max(worst_f, dH, dQ)
        print(f"    {stem:<16} dH={dH:.3e} dQ={dQ:.3e} "
              f"状态{'一致' if st_ok else '不一致'} 迭代{'相等' if it_ok else '不等'} "
              f"阀={st_name}(期望 {expect}) {'PASS' if good else 'FAIL  <-- 超限'}")
        if expect != "ACTIVE":
            continue
        # ---- ACTIVE 态：上下游 demand 梯度 implicit vs FD ----
        S = r["status"].numpy().astype(np.int8)
        K = r["setting"].numpy().astype(np.float64)
        ke0 = np.zeros(net.N)
        r0 = s.r_np.copy()
        rng = np.random.default_rng(23)
        wH = np.where(jm, rng.standard_normal(net.N), 0.0)
        wQ = rng.standard_normal(net.L)
        gd, _, _, _ = implicit_grads(s, d0, rh0, ke0, r0, wH, wQ, S=S, K=K)
        lossf = fd_loss(s, d0, rh0, ke0, r0, wH, wQ, S=S, K=K)
        rows = []
        nids = [nid for nid in ("J1", "J2", "J3") if nid in net.node_id]
        for nid in nids:                    # J1=阀上游, J2=阀下游(, J3=末端)
            i = net.node_id.index(nid)
            h = 1e-5
            dp = d0.copy(); dp[i] += h
            dm = d0.copy(); dm[i] -= h
            rows.append((f"d[{nid}]", i, gd[i],
                         (lossf(dd=dp) - lossf(dd=dm)) / (2 * h)))
        g_ok, g_w = grad_rows_check(rows, stem)
        ok = ok and g_ok
        worst_g = max(worst_g, g_w)
    print(f"  ① 前向 worst={worst_f:.3e}  ACTIVE 梯度 worst rel={worst_g:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- ② D-W
def item2():
    print("② D-W Re=2000/4000 连续性 + pub_balerma 梯度(≥6 坐标)")
    stem = "pub_balerma"
    inp = os.path.join(ROOT, "networks", "public", "Balerma.inp")
    net = Net.load(REF, stem)
    s = GGASolver(net, mode="epanet", inp_path=inp)
    k = int(np.where(s.lt_np <= 1)[0][0])

    def hl_hg(qv):
        q = np.zeros(s.L)
        q[k] = qv
        P, Y = s._dw_PY_np(q)
        return Y[k] / P[k], 1.0 / P[k]

    sv = s.viscos * float(s.diam_np[k])
    q1, q2 = A2 * sv, A1 * sv            # Re=2000 / Re=4000 边界流量
    ok = True
    # ---- 边界两侧连续性 + hgrad 有限 ----
    for name, qb in (("Re=2000", q1), ("Re=4000", q2)):
        hlm, hgm = hl_hg(qb * (1 - 1e-12))
        hlp, hgp = hl_hg(qb * (1 + 1e-12))
        j_hl = abs(hlp - hlm) / max(abs(hlm), 1e-300)
        j_hg = abs(hgp - hgm) / max(abs(hgm), 1e-300)
        fin = np.isfinite(hgm) and np.isfinite(hgp) and hgm > 0 and hgp > 0
        good = j_hl < 1e-8 and j_hg < 1e-8 and fin
        ok = ok and good
        print(f"    {name} 两侧: Δhloss/hloss={j_hl:.2e} Δhgrad/hgrad={j_hg:.2e} "
              f"hgrad 有限且正={fin} {'PASS' if good else 'FAIL  <-- 超限'}")
    # ---- 跨边界扫描：40 点几何网格，相邻点 Lipschitz 一致 ----
    qs = np.geomspace(0.5 * q1, 2.0 * q2, 40)
    hls, hgs = np.array([hl_hg(qv) for qv in qs]).T
    mono = bool(np.all(np.diff(hls) > 0))
    lip = True
    for i in range(len(qs) - 1):
        dq = qs[i + 1] - qs[i]
        bound = 2.0 * max(hgs[i], hgs[i + 1]) * dq
        if abs(hls[i + 1] - hls[i]) > bound:
            lip = False
    fin_all = bool(np.all(np.isfinite(hgs)) and np.all(hgs > 0))
    good = mono and lip and fin_all
    ok = ok and good
    print(f"    扫描 40 点 [{qs[0]:.2e},{qs[-1]:.2e}] cfs: hloss 单调递增={mono} "
          f"局部Lipschitz一致={lip} hgrad 全有限={fin_all} "
          f"{'PASS' if good else 'FAIL  <-- 超限'}")
    # ---- implicit vs FD（两场景 8 坐标）----
    rng = np.random.default_rng(31)
    ke0 = np.asarray(net.node_ke, dtype=np.float64)
    r0 = s.r_np.copy()
    jm = np.asarray(net.node_type) == 0
    worst_g = 0.0
    for label, scale, n_d in (("名义需水", 1.0, 2), ("1e-3×需水", 1e-3, 1)):
        d0 = net.demand_cfs_at(0) * scale
        rh0 = net.reservoir_head_ft_at(0)
        # 稀疏 wH（20 junction）：全稠密权重时 L≈Σw·H ~1e3-1e4 ft 级，float64
        # 求和噪声 ~1e-12，而缩需水下 |∂L/∂d| 可低到 1e-2，2h·|g|≈2e-7 逼近
        # 噪声底使中央差分失真（实测 demand[196] 随 h 减小不收敛）；稀疏权重
        # 把损失规模压一个量级、被测梯度相对增大，FD 分辨率恢复 <2e-7
        wH = np.zeros(net.N)
        pick_w = rng.choice(np.where(jm)[0], 20, replace=False)
        wH[pick_w] = rng.standard_normal(20)
        wQ = np.zeros(net.L)
        gd, grh, gr, _ = implicit_grads(s, d0, rh0, ke0, r0, wH, wQ)
        lossf = fd_loss(s, d0, rh0, ke0, r0, wH, wQ)
        # 按收敛流态挑管道坐标
        q_c = solve_polished(s, d0, rh0, ke0, r0)["q"][0]
        svv = s.viscos * s.diam_np
        is_pipe = s.lt_np <= 1
        lam = np.where(is_pipe & (np.abs(q_c) <= A2 * svv))[0]
        swj = np.where(is_pipe & (np.abs(q_c) / svv >= A1))[0]
        dun = np.where(is_pipe & (np.abs(q_c) > A2 * svv)
                       & (np.abs(q_c) / svv < A1))[0]
        print(f"    [{label}] 流态: 层流 {lam.size}/Dunlop {dun.size}/S-J {swj.size}")
        rows = []
        for i in rng.choice(np.where(jm & (d0 > 0))[0], n_d, replace=False):
            # 绝对步长下限 1e-5 cfs（同 gradcheck_dw）：缩需水场景过小步长会把
            # ΔL 压进求解噪声底（~1e-10），中央差分失真
            h = max(1e-5, 1e-6 * abs(d0[int(i)]))
            dp = d0.copy(); dp[int(i)] += h
            dm = d0.copy(); dm[int(i)] -= h
            rows.append(("demand", int(i), gd[int(i)],
                         (lossf(dd=dp) - lossf(dd=dm)) / (2 * h)))
        picks = []
        for arr, n_p in ((swj, 1), (dun, 1), (lam, 1)):
            if arr.size:
                picks += list(rng.choice(arr, min(n_p, arr.size), replace=False))
        for kk in picks:
            h = max(1e-3 * abs(r0[int(kk)]), 1e-8)
            rp = r0.copy(); rp[int(kk)] += h
            rm = r0.copy(); rm[int(kk)] -= h
            rows.append(("r_hw", int(kk), gr[int(kk)],
                         (lossf(rr=rp) - lossf(rr=rm)) / (2 * h)))
        i_res = int(np.where(np.asarray(net.node_type) == 1)[0][0])
        rp = rh0.copy(); rp[i_res] += 1e-5
        rm = rh0.copy(); rm[i_res] -= 1e-5
        rows.append(("res_head", i_res, grh[i_res],
                     (lossf(rhh=rp) - lossf(rhh=rm)) / (2 * 1e-5)))
        g_ok, g_w = grad_rows_check(rows, label)
        ok = ok and g_ok
        worst_g = max(worst_g, g_w)
    print(f"  ② D-W 连续性+梯度 worst rel={worst_g:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- ③ 规则 priority
def make_exa5_variant(text, p1, p2, tag):
    """在 EXA5 [RULES] 上加 PRIORITY p1 与对抗规则 RULE 900（PRIORITY p2）。"""
    eol = "\r\n" if "\r\n" in text else "\n"
    block = (f"THEN PUMP PMP-2 STATUS IS OPEN {eol}PRIORITY {p1}{eol}{eol}"
             f"RULE 900{eol}IF SYSTEM DEMAND > 63.0901969631786{eol}"
             f"AND TANK T-1 HEAD < 175.199040000701{eol}"
             f"THEN PUMP PMP-2 STATUS IS CLOSED{eol}PRIORITY {p2}{eol}")
    out, n = re.subn(r"THEN PUMP PMP-2 STATUS IS OPEN[ \t]*\r?\n",
                     block.replace("\\", "\\\\"), text)
    assert n == 1, f"变体 {tag} 规则替换失败（命中 {n} 处）"
    path = os.path.join(TMP, f"EXA5_prio_{tag}.inp")
    with open(path, "w", encoding="latin-1", newline="") as f:
        f.write(out)
    return path


def eps_compare(out, ref):
    """完整 EPS 逐帧对拍（align_eps 同门槛）。返回 (ok, 摘要串)。"""
    t_ok = len(out["t_sec"]) == len(ref["t_sec"]) and bool(
        np.array_equal(np.asarray(out["t_sec"], dtype=np.int64),
                       np.asarray(ref["t_sec"], dtype=np.int64)))
    if not t_ok:
        return False, (f"t 序列不等: 我 {len(out['t_sec'])} 帧 vs DLL "
                       f"{len(ref['t_sec'])} 帧")
    wh = wq = wqc = 0.0
    st_all = it_all = True
    for f in range(len(out["t_sec"])):
        dH = np.abs(out["head_ft"][f] - ref["head_ft"][f]).max()
        dQ_l = np.abs(out["flow_cfs"][f] - ref["flow_cfs"][f])
        opened = ref["status"][f] > 0
        wh = max(wh, dH)
        if opened.any():
            wq = max(wq, dQ_l[opened].max())
        if (~opened).any():
            wqc = max(wqc, dQ_l[~opened].max())
        st_all = st_all and bool(np.array_equal(out["status"][f],
                                                ref["status"][f]))
        it_all = it_all and int(out["iterations"][f]) == int(ref["iterations"][f])
    ok = wh < TOL_H and wq < TOL_Q and wqc < TOL_QC and st_all and it_all
    return ok, (f"{len(out['t_sec'])} 帧 dH={wh:.3e} dQ={wq:.3e} "
                f"dQc={wqc:.3e} 状态{'一致' if st_all else '不一致'} "
                f"迭代{'全等' if it_all else '不等'}")


def item3():
    print("③ EXA5 规则 priority 仲裁变体 vs DLL（完整 EPS）")
    os.makedirs(TMP, exist_ok=True)
    with open(os.path.join(ROOT, "networks", "InpData", "EXA5.inp"),
              "r", encoding="latin-1", newline="") as f:
        text = f.read()
    variants = [("A", 1, 2, "CLOSED 胜(900 优先)"),
                ("B", 2, 1, "OPEN 胜(589 优先)"),
                ("C", 1, 1, "平手→首规则 589 胜")]
    ok = True
    traj = {}
    for tag, p1, p2, desc in variants:
        inp_v = make_exa5_variant(text, p1, p2, tag)
        net_v = parse_inp(inp_v)
        out = EpsDriver(net_v, inp_path=inp_v).run()
        with Epanet(inp_v) as en:
            ref = en.solve_eps()
        good, msg = eps_compare(out, ref)
        ok = ok and good
        kp = net_v.link_id.index("PMP-2")
        traj[tag] = np.asarray(out["status"])[:, kp].copy()
        n_open = int(traj[tag].sum())
        print(f"    变体{tag}({desc}): {msg} PMP-2 开启帧数={n_open} "
              f"{'PASS' if good else 'FAIL  <-- 超限'}")
    diff_ab = not (traj["A"].shape == traj["B"].shape
                   and bool(np.array_equal(traj["A"], traj["B"])))
    same_bc = traj["B"].shape == traj["C"].shape \
        and bool(np.array_equal(traj["B"], traj["C"]))
    ok = ok and diff_ab and same_bc
    print(f"    仲裁可观测: A≠B(优先级反转改变轨迹)={diff_ab} "
          f"B=C(平手取先到规则)={same_bc}")
    print(f"  ③ priority 仲裁  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- ④ FCV 梯度
def item4():
    print("④ FCV ACTIVE 梯度冒烟（fcv_smoke 冻结态）")
    inp = os.path.join(VAR, "fcv_smoke.inp")
    net = parse_inp(inp)
    s = GGASolver(net, mode="epanet", inp_path=inp)
    d0 = net.demand_cfs_at(0)
    rh0 = net.reservoir_head_ft_at(0)
    r = s.solve(d0, rh0, status_machine=True)
    kv = net.link_id.index("V1")
    S = r["status"].numpy().astype(np.int8)
    K = r["setting"].numpy().astype(np.float64)
    st_name = {2: "CLOSED", 3: "OPEN", 4: "ACTIVE"}.get(int(S[kv]), "?")
    act_ok = st_name == "ACTIVE"
    qv = float(r["flow_cfs"].numpy()[kv])
    print(f"    V1 内部态={st_name} q={qv * s.ucf_flow:.4f} LPS "
          f"(设定 30，|q-set|={abs(qv - K[kv]) * s.ucf_flow:.2e} LPS)")
    ke0 = np.zeros(net.N)
    r0 = s.r_np.copy()
    rng = np.random.default_rng(41)
    jm = np.asarray(net.node_type) == 0
    wH = np.where(jm, rng.standard_normal(net.N), 0.0)
    wQ = rng.standard_normal(net.L)
    gd, grh, gr, _ = implicit_grads(s, d0, rh0, ke0, r0, wH, wQ, S=S, K=K)
    lossf = fd_loss(s, d0, rh0, ke0, r0, wH, wQ, S=S, K=K)
    rows = []
    i3 = net.node_id.index("J3")
    h = 1e-5
    dp = d0.copy(); dp[i3] += h
    dm = d0.copy(); dm[i3] -= h
    rows.append(("d[J3]", i3, gd[i3], (lossf(dd=dp) - lossf(dd=dm)) / (2 * h)))
    ir = net.node_id.index("R1")
    rp = rh0.copy(); rp[ir] += h
    rm = rh0.copy(); rm[ir] -= h
    rows.append(("rh[R1]", ir, grh[ir], (lossf(rhh=rp) - lossf(rhh=rm)) / (2 * h)))
    for pid in ("P1", "P2", "P3"):
        kk = net.link_id.index(pid)
        hh = max(1e-4 * abs(r0[kk]), 1e-8)
        rp2 = r0.copy(); rp2[kk] += hh
        rm2 = r0.copy(); rm2[kk] -= hh
        rows.append((f"r[{pid}]", kk, gr[kk],
                     (lossf(rr=rp2) - lossf(rr=rm2)) / (2 * hh)))
    g_ok, g_w = grad_rows_check(rows, "fcv")
    ok = act_ok and g_ok
    print(f"  ④ FCV 梯度 worst rel={g_w:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- ⑤ CUSTOM 泵
def item5():
    print("⑤ CUSTOM 泵段选择梯度（pub_richmond_standard 帧 6 冻结）")
    stem = "pub_richmond_standard"
    net = Net.load(REF, stem)
    ref = np.load(os.path.join(REF, f"{stem}_ref.npz"))
    inp = os.path.join(ROOT, "networks", "public", "Richmond_standard.inp")
    drv = EpsDriver(net, inp_path=inp)
    s = drv.solver
    drv._inithyd()
    tank_nodes = drv.tank_nodes
    for f in range(7):                      # 替换回放到帧 6（t=10036，5 泵开）
        drv.Htime = int(ref["t_sec"][f])
        drv.H[tank_nodes] = ref["head_ft"][f][tank_nodes]
        drv._demands()
        drv._controls()
        r = s.run_gga(drv.d, drv.H, q0=drv.q, e0=drv.e,
                      status0=drv.S, setting0=drv.K, do_status=True)
        drv.q = r["flow"]; drv.e = r["emitter"]; drv.S = r["status"]
        drv.K = r["setting"]; drv.H = r["head"]; drv.fixed_dem = r["fixed_demand"]
    S6 = drv.S.copy().astype(np.int8)
    K6 = drv.K.copy()
    d6 = drv.d.copy()
    rh6 = drv.H.copy()
    ke0 = np.zeros(net.N)
    r0 = s.r_np.copy()
    pl = np.asarray(net.pump_link, dtype=np.int64)
    cptr = np.asarray(net.pump_curve_ptr)
    cq = np.asarray(net.pump_curve_q)

    def seg_of(j, q_over_w_user):
        """curvecoeff 段号（hydcoeffs.c:818-822 语义）。"""
        x = list(cq[cptr[j]:cptr[j + 1]])
        k2 = bisect.bisect_left(x, q_over_w_user)   # 首个 x[k2]>=q
        while k2 < len(x) and x[k2] < q_over_w_user:
            k2 += 1
        if k2 == 0:
            k2 += 1
        elif k2 == len(x):
            k2 -= 1
        return k2

    sol0 = solve_polished(s, d6, rh6, ke0, r0, speed=K6, status=S6)
    print(f"    冻结帧抛光 ‖F‖∞={sol0['resid_inf'][0]:.2e}")
    pump_pick = []                          # (泵号, 描述)
    for j, k in enumerate(pl):
        k = int(k)
        if S6[k] <= s.ST_CLOSED:
            continue
        qu = abs(float(sol0["q"][0][k])) / K6[k] * s.ucf_flow
        x = cq[cptr[j]:cptr[j + 1]]
        dmin = float(np.abs(x - qu).min())
        pump_pick.append((j, k, qu, dmin))
    pump_pick.sort(key=lambda t: t[3])
    near = pump_pick[0]                     # 距段点最近（近段边界）
    mids = pump_pick[-2:]                   # 距段点最远的两台（段中）
    rng = np.random.default_rng(53)
    jm = np.asarray(net.node_type) == 0
    wH = np.where(jm, rng.standard_normal(net.N), 0.0)
    wQ = rng.standard_normal(net.L)
    gd, _, _, gsp = implicit_grads(s, d6, rh6, ke0, r0, wH, wQ, S=S6, K=K6)
    lossf = fd_loss(s, d6, rh6, ke0, r0, wH, wQ, S=S6, K=K6)
    ok = True
    rows = []
    seg_guard = True
    for j, k, qu, dmin in [near] + mids:
        tagp = "近段边界" if (j, k) == (near[0], near[1]) else "段中"
        seg0 = seg_of(j, qu)
        h = 1e-6                            # ω 步长：扰动后 q/ω 移动 ~1e-5 用户单位
        sp_p = K6.copy(); sp_p[k] += h
        sp_m = K6.copy(); sp_m[k] -= h
        Lp = lossf(sp=sp_p)
        Lm = lossf(sp=sp_m)
        # FD 扰动解不跨段核验
        for sp_v in (sp_p, sp_m):
            solv = solve_polished(s, d6, rh6, ke0, r0, speed=sp_v, status=S6)
            qu_v = abs(float(solv["q"][0][k])) / sp_v[k] * s.ucf_flow
            if seg_of(j, qu_v) != seg0:
                seg_guard = False
        rows.append((f"ω p{j}({tagp},q/ω={qu:.3f},距段点{dmin:.3f})", k,
                     gsp[k], (Lp - Lm) / (2 * h)))
    for i in rng.choice(np.where(jm & (d6 > 0))[0], 2, replace=False):
        h = max(1e-5, 1e-6 * abs(d6[int(i)]))
        dp = d6.copy(); dp[int(i)] += h
        dm = d6.copy(); dm[int(i)] -= h
        rows.append(("demand", int(i), gd[int(i)],
                     (lossf(dd=dp) - lossf(dd=dm)) / (2 * h)))
    g_ok, g_w = grad_rows_check(rows, "richmond")
    ok = ok and g_ok and seg_guard
    print(f"    FD 扰动解不跨段: {seg_guard}")
    print(f"  ⑤ CUSTOM 泵梯度 worst rel={g_w:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("阶段 D 补做审计 动态对抗实测（5 项）")
    r1 = item1()
    r2 = item2()
    r3 = item3()
    r4 = item4()
    r5 = item5()
    ok = r1 and r2 and r3 and r4 and r5
    print(f"总判定: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

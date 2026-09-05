# -*- coding: utf-8 -*-
"""audit_prv_dw_adversarial.py - 任务 D-审计 对抗实测（梯队3 审计）。

对抗小网（networks/variants/{prv_adv_active,prv_adv_open,prv_adv_closed,
psv_adv_active}.inp，GPM/H-W）与 pub_balerma（LPS/D-W）上四组硬校验：

① PRV/PSV 三态前向对拍：4 网 GGASolver(status_machine=True) vs DLL
   （EN_runH 单帧），门槛 H<1e-6 ft、Q<1e-6 cfs、二值状态一致、迭代数相等，
   并核对我方内部状态 = 预期（ACTIVE/OPEN/CLOSED）。
② ACTIVE↔OPEN 边界扫掠：prv_adv_active 网把 R1 水头压到边界
   rh* = hset + ploss(P1) + hml(V1)（prvstatus hydstatus.c:275 的临界），
   ±{1,0.1,1e-2,1e-3,±htol/2} ft 逐点 vs DLL（改 EN_ELEVATION 重解），
   门槛同①（状态为 EN_STATUS 二值口径）。
③ 三态冻结梯度对拍：各网取①收敛的 (S*,K*) 冻结，ImplicitGGASolve 伴随
   vs solve_polished 中央差分，θ∈{demand(全 junction), res_head(全水库),
   r_hw(全管道)}，L=Σw·H_junc+Σw·Q；门槛 rel<1e-6（|Δ|<1e-9 按一致计）。
   附加解耦断言：ACTIVE PRV 下游节点头对上游水库水头的梯度 |g|<1e-9
   （罚函数把 H[n2] 钉在 hset，上游扰动不透传）。
④ D-W C¹ 连续性（pub_balerma 管道，_dw_PY_np 逐点）：Re=2000（层流/Dunlop）
   与 Re=4000（Dunlop/Swamee-Jain）边界两侧 hloss 相对跳变 <1e-8、
   hgrad 相对跳变 <1e-8（解析 C¹，见 frictionFactor 系数推导），跨界中央
   差分 vs 两侧 hgrad 均值 rel<1e-4，支内中央差分 vs hgrad rel<1e-6。
"""

import os
import sys
import ctypes

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import Net, parse_inp                       # noqa: E402
from dgga.solver import GGASolver, A1, A2, MISSING          # noqa: E402
from dgga.autodiff import ImplicitGGASolve, solve_polished  # noqa: E402
from dgga.epanet_ref import (Epanet, EN_HEAD, EN_FLOW, EN_STATUS,  # noqa: E402
                             EN_ELEVATION)

VAR = os.path.join(ROOT, "networks", "variants")
TOL_H, TOL_Q, TOL_G = 1e-6, 1e-6, 1e-6

CASES = [  # (stem, 阀链路 id, 预期内部状态名)
    ("prv_adv_active", "V1", "ACTIVE"),
    ("prv_adv_open", "V1", "OPEN"),
    ("prv_adv_closed", "V1", "CLOSED"),
    ("psv_adv_active", "V1", "ACTIVE"),
]


def dll_single(inp, res_heads=None):
    """DLL 单帧：可选覆盖水库水头（内部 ft）后 EN_solveH，读头/流量/二值状态。"""
    with Epanet(inp) as en:
        if res_heads:
            ids = en.node_ids()
            for nid, h in res_heads.items():
                en.set_node_value(ids.index(nid) + 1, EN_ELEVATION, h)
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


def solve_mine(net, s, rh_override=None):
    d0 = net.demand_cfs_at(0)
    rh0 = net.reservoir_head_ft_at(0)
    if rh_override:
        for i, h in rh_override.items():
            rh0[i] = h
    r = s.solve(d0, rh0, status_machine=True)
    return d0, rh0, r


def cmp_frame(r, ref, jm):
    open_my = r["status"].numpy() > 2
    dH = np.abs(r["head_ft"].numpy() - ref["head_ft"])[jm].max()
    q_api = np.where(open_my, r["flow_cfs"].numpy(), 0.0)
    opened = ref["status"] > 0
    dQ = np.abs(q_api - ref["flow_cfs"])[opened].max() if opened.any() else 0.0
    st_ok = bool(np.array_equal(open_my.astype(np.int8), ref["status"]))
    it_ok = int(r["iters"]) == int(ref["iterations"])
    return dH, dQ, st_ok, it_ok


def item1():
    print("① PRV/PSV 三态前向对拍（vs DLL，位级门槛）")
    worst = 0.0
    ok = True
    ctx = {}
    for stem, vid, expect in CASES:
        inp = os.path.join(VAR, f"{stem}.inp")
        net = parse_inp(inp)
        s = GGASolver(net, mode="epanet", inp_path=inp)
        ref = dll_single(inp)
        d0, rh0, r = solve_mine(net, s)
        jm = np.asarray(net.node_type) == 0
        dH, dQ, st_ok, it_ok = cmp_frame(r, ref, jm)
        kv = net.link_id.index(vid)
        st_name = {2: "CLOSED", 3: "OPEN", 4: "ACTIVE"}.get(
            int(r["status"].numpy()[kv]), "?")
        exp_ok = st_name == expect
        good = dH < TOL_H and dQ < TOL_Q and st_ok and it_ok and exp_ok
        ok = ok and good
        worst = max(worst, dH, dQ)
        print(f"    {stem:<16} dH={dH:.3e} dQ={dQ:.3e} 状态{'一致' if st_ok else '不一致'}"
              f" 迭代{'相等' if it_ok else '不等'} 阀内部态={st_name}(期望 {expect})"
              f" {'PASS' if good else 'FAIL  <-- 超限'}")
        ctx[stem] = (net, s, d0, rh0, r)
    print(f"  ① 三态前向对拍(4网)  worst={worst:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok, ctx


def item2(ctx):
    print("② ACTIVE↔OPEN 边界扫掠（prv_adv_active，R1 水头逼近 rh*）")
    stem = "prv_adv_active"
    inp = os.path.join(VAR, f"{stem}.inp")
    net, s, d0, rh0, r0 = ctx[stem]
    kv = net.link_id.index("V1")
    n2 = int(net.link_n2[kv])
    hset = s.elev_np[n2] + float(r0["setting"].numpy()[kv])   # El+内部设定
    kp1 = net.link_id.index("P1")
    qv = float(r0["flow_cfs"].numpy()[kv])
    ploss = float(s.r_np[kp1]) * abs(qv) ** s.hexp            # H-W 摩阻
    hml = float(s.km_valve_ml_np[kv]) * qv * qv               # prvstatus:267
    rh_star = hset + ploss + hml                              # h1-hml==hset 临界
    i_r1 = net.node_id.index("R1")
    htol = s.htol
    ok = True
    worst = 0.0
    jm = np.asarray(net.node_type) == 0
    for dlt in (1.0, 0.1, 1e-2, 1e-3, htol / 2,
                -htol / 2, -1e-3, -1e-2, -0.1, -1.0):
        rh_v = rh_star + dlt
        ref = dll_single(inp, {"R1": rh_v})
        _, _, r = solve_mine(net, s, {i_r1: rh_v})
        dH, dQ, st_ok, it_ok = cmp_frame(r, ref, jm)
        st_name = {2: "CLOSED", 3: "OPEN", 4: "ACTIVE"}.get(
            int(r["status"].numpy()[kv]), "?")
        good = dH < TOL_H and dQ < TOL_Q and st_ok and it_ok
        ok = ok and good
        worst = max(worst, dH, dQ)
        print(f"    rh*={rh_star:.4f}{dlt:+9.5f} ft  dH={dH:.3e} dQ={dQ:.3e} "
              f"阀={st_name:<6} 状态{'一致' if st_ok else '不一致'} "
              f"迭代{'相等' if it_ok else '不等'} {'PASS' if good else 'FAIL  <-- 超限'}")
    print(f"  ② 边界扫掠(10点)  worst={worst:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def item3(ctx):
    print("③ 三态冻结梯度对拍（implicit vs 中央差分）+ ACTIVE 解耦断言")
    rng = np.random.default_rng(11)
    ok = True
    worst = 0.0
    for stem, vid, expect in CASES:
        net, s, d0, rh0, r0 = ctx[stem]
        S = r0["status"].numpy().astype(np.int8)
        K = r0["setting"].numpy().astype(np.float64)
        ke0 = np.zeros(net.N)
        r_hw0 = s.r_np.copy()
        jm = np.asarray(net.node_type) == 0
        res = np.where(~jm)[0]
        wH = np.where(jm, rng.standard_normal(net.N), 0.0)
        wQ = rng.standard_normal(net.L)
        dt = torch.float64
        td = torch.tensor(d0, dtype=dt, requires_grad=True)
        trh = torch.tensor(np.nan_to_num(rh0), dtype=dt, requires_grad=True)
        tke = torch.tensor(ke0, dtype=dt, requires_grad=True)
        tr = torch.tensor(r_hw0, dtype=dt, requires_grad=True)
        head, flow, _ = ImplicitGGASolve.apply(
            td, trh, tke, tr, s, 1e-12, 200, 3,
            torch.tensor(K, dtype=dt), None, None, S)
        L = (head * torch.tensor(wH, dtype=dt)).sum() \
            + (flow * torch.tensor(wQ, dtype=dt)).sum()
        L.backward()
        gd, grh, gr = td.grad.numpy(), trh.grad.numpy(), tr.grad.numpy()
        pol = solve_polished(s, d0, rh0, ke0, r_hw0, speed=K, status=S)
        resid = pol["resid_inf"][0]

        def lossf(dd, rr, rhh):
            sol = solve_polished(s, dd, rhh, ke0, rr, speed=K, status=S)
            return float((sol["head"][0] * wH).sum() + (sol["q"][0] * wQ).sum())

        rows = []
        for i in np.where(jm)[0]:
            h = 1e-5
            dp = d0.copy(); dp[i] += h
            dm = d0.copy(); dm[i] -= h
            rows.append(("demand", int(i), gd[i],
                         (lossf(dp, r_hw0, rh0) - lossf(dm, r_hw0, rh0)) / (2 * h)))
        for i in res:
            h = 1e-5
            rp = rh0.copy(); rp[i] += h
            rm = rh0.copy(); rm[i] -= h
            rows.append(("res_head", int(i), grh[i],
                         (lossf(d0, r_hw0, rp) - lossf(d0, r_hw0, rm)) / (2 * h)))
        for k in np.where(s.lt_np <= 1)[0]:
            h = max(1e-4 * abs(r_hw0[k]), 1e-8)
            rp = r_hw0.copy(); rp[k] += h
            rm = r_hw0.copy(); rm[k] -= h
            rows.append(("r_hw", int(k), gr[k],
                         (lossf(d0, rp, rh0) - lossf(d0, rm, rh0)) / (2 * h)))
        w_net = 0.0
        for kind, i, ga, gf in rows:
            rel = abs(ga - gf) / max(abs(ga), abs(gf), 1e-12)
            if abs(ga - gf) < 1e-9:
                rel = 0.0
            w_net = max(w_net, rel)
            if rel >= TOL_G:
                ok = False
                print(f"    [{stem}] {kind}[{i}] adj={ga:.6e} fd={gf:.6e} "
                      f"rel={rel:.3e}  <-- 超限")
        worst = max(worst, w_net)
        extra = ""
        if stem == "prv_adv_active":
            # 解耦断言：dH[J2]/d rh[R1]（单独反传）
            td2 = torch.tensor(d0, dtype=dt)
            trh2 = torch.tensor(np.nan_to_num(rh0), dtype=dt, requires_grad=True)
            h2, _, _ = ImplicitGGASolve.apply(
                td2, trh2, torch.tensor(ke0, dtype=dt),
                torch.tensor(r_hw0, dtype=dt), s, 1e-12, 200, 3,
                torch.tensor(K, dtype=dt), None, None, S)
            j2 = net.node_id.index("J2")
            h2[j2].backward()
            g_dec = abs(float(trh2.grad.numpy()[net.node_id.index("R1")]))
            dec_ok = g_dec < 1e-9
            ok = ok and dec_ok
            extra = f" 解耦|dH(J2)/drh(R1)|={g_dec:.1e}{'' if dec_ok else ' <-- 超限'}"
        print(f"    {stem:<16} 坐标数={len(rows)} ‖F‖∞={resid:.1e} "
              f"最差rel={w_net:.3e}{extra}")
    print(f"  ③ 三态冻结梯度(4网)  worst={worst:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def item4():
    print("④ D-W C¹ 连续性（pub_balerma，Re=2000/4000 边界）")
    stem = "pub_balerma"
    inp = os.path.join(ROOT, "networks", "public", "Balerma.inp")
    net = Net.load(os.path.join(ROOT, "data", "reference"), stem)
    s = GGASolver(net, mode="epanet", inp_path=inp)
    k = int(np.where(s.lt_np <= 1)[0][0])          # 第一条管道

    def hl_hg(qv):
        q = np.zeros(s.L)
        q[k] = qv
        P, Y = s._dw_PY_np(q)
        return Y[k] / P[k], 1.0 / P[k]             # (hloss, hgrad)

    sv = s.viscos * float(s.diam_np[k])
    ok = True
    worst = 0.0
    for name, qb in (("Re=2000(层流/Dunlop)", A2 * sv),
                     ("Re=4000(Dunlop/S-J)", A1 * sv)):
        e = 1e-12
        hlm, hgm = hl_hg(qb * (1 - e))
        hlp, hgp = hl_hg(qb * (1 + e))
        j_hl = abs(hlp - hlm) / max(abs(hlm), 1e-300)
        j_hg = abs(hgp - hgm) / max(abs(hgm), 1e-300)
        # 跨界中央差分 vs 两侧 hgrad 均值（C¹）
        d = 1e-6 * qb
        hl_a, hg_a = hl_hg(qb - d)
        hl_b, hg_b = hl_hg(qb + d)
        fd_x = (hl_b - hl_a) / (2 * d)
        rel_x = abs(fd_x - 0.5 * (hg_a + hg_b)) / abs(fd_x)
        # 支内中央差分 vs hgrad
        rels_in = []
        for q0 in (qb * 0.9, qb * 1.1):
            d2 = 1e-7 * q0
            hla, _ = hl_hg(q0 - d2)
            hlb, _ = hl_hg(q0 + d2)
            _, hg0 = hl_hg(q0)
            rels_in.append(abs((hlb - hla) / (2 * d2) - hg0) / hg0)
        good = j_hl < 1e-8 and j_hg < 1e-8 and rel_x < 1e-4 \
            and max(rels_in) < 1e-6
        ok = ok and good
        worst = max(worst, j_hl, j_hg)
        print(f"    {name:<22} Δhloss/hloss={j_hl:.2e} Δhgrad/hgrad={j_hg:.2e} "
              f"跨界FD rel={rel_x:.2e} 支内FD rel={max(rels_in):.2e} "
              f"{'PASS' if good else 'FAIL  <-- 超限'}")
    print(f"  ④ D-W C¹ 连续性(2边界)  worst={worst:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("任务 D-审计 对抗实测（PRV/PSV 三态 + D-W 连续性）")
    ok1, ctx = item1()
    ok2 = item2(ctx)
    ok3 = item3(ctx)
    ok4 = item4()
    ok = ok1 and ok2 and ok3 and ok4
    print(f"总判定: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

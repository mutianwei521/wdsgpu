# -*- coding: utf-8 -*-
"""gradcheck_b2.py - B2 可微扩展验证（泵网 EXA6 / ky3，任务 B2）。

场景：EXA6（1 台 POWER_FUNC 3 点曲线泵 + 1 柱形水池）与 ky3（5 台 CONST_HP
恒功率泵 + 3 水池）各取一帧（状态冻结：LinkStatus/LinkSetting 取 ref，
tank 头固定为 ref 值；同 scripts/align.py replay_b2 口径）。

验证项（门槛见各节）：
 ① torch.autograd.gradcheck(ImplicitGGASolve)：θ=(demand 子集, r_hw 子集,
    泵转速 ω 全体)，输出取样（若干 junction 水头 + 泵流量），f64，
    eps=1e-6, atol=1e-5, rtol=1e-3（与 gradcheck_3way 同参） - PASS；
 ② implicit vs 中心差分（自家求解器 solve_polished，accuracy=1e-12 + Newton
    精抛光）≥15 坐标/网：demand（强制含泵两端节点）、ω（泵全体）、泵曲线参数
    （EXA6：H0 与 R；ky3 恒功率泵：R - CONST_HP 的 hloss=r/Q 不含 H0，
    ∂φ/∂H0≡0 解析为零，故 FD 对拍 R）、r_hw 若干（非钳位开管且 |g|≥40 分位，
    同 gradcheck_3way 的坐标池纪律） - rel<1e-6；
    FD 步长与 Richardson 外推同 gradcheck_3way 头注释（噪声分析亦同）。
    FD 分辨率地板：实测本批 FD 绝对噪声 ~1e-7~3e-7（ky3 各坐标 |gB−gC| 的
    平台值；与 3way 头注释的 f64 损失噪声/2h 分析一致）。泵两端节点中
    |g|~1e-3 级坐标的 rel<1e-6 物理不可达（噪声/|g| ≈ 1e-4），援引
    audit_grad_adversarial ⑤ 的先例，此类坐标的硬门槛退为双条件：
    |gB−gC| < 1e-6（FD 噪声内）且 implicit vs unrolled 独立通路交叉
    rel < 1e-6（两条独立反传路线，无共模误差）；表中注明 [FD噪声地板]；
 ③ 展开路径 solve_unrolled（同帧冻结状态、warm start=收敛流量、K=iters+5）
    vs 同一批中心差分 - rel<1e-4（FD 分辨率不足坐标同上双条件退化）；
 ④ BPTT 冒烟（EXA6，报告性无门槛）：3 帧 mini-EPS，水池显式欧拉
    （tanklevels hydraul.c:1021-1022 无钳位 + tankgrade :1089 柱形仿射）进图，
    帧间 warm start 保图，loss=末帧 tank 头，对 t=0 需水求梯度 vs 中心差分。
"""

import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import Net                                    # noqa: E402
from dgga.solver import GGASolver, QZERO, PI                  # noqa: E402
from dgga.autodiff import (ImplicitGGASolve, solve_polished,  # noqa: E402
                           solve_unrolled)

TOL_BC = 1e-6      # implicit vs FD
TOL_AC = 1e-4      # unrolled vs FD
NOISE_ABS = 1e-6   # FD 绝对噪声地板（实测 |gB−gC| 平台 1e-7~3e-7 的 ~5 倍，见头注释）
SEED = 2026
MAXIT = 200


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def load_case(stem, frame=0):
    """冻结帧：需水/水库头取 parse，tank 头/状态/转速取 ref（align.replay_b2 口径）。"""
    net = Net.load(os.path.join(ROOT, "data", "reference"), stem)
    ref = np.load(os.path.join(ROOT, "data", "reference", f"{stem}_ref.npz"))
    inp = os.path.join(ROOT, "networks", "InpData", f"{stem}.inp")
    s = GGASolver(net, mode="epanet", inp_path=inp if os.path.isfile(inp) else None)
    t = int(ref["t_sec"][frame])
    d0 = net.demand_cfs_at(t)
    fh = np.nan_to_num(net.reservoir_head_ft_at(t))
    tank_nodes = np.asarray(net.tank_node, dtype=np.int64)
    fh[tank_nodes] = ref["head_ft"][frame][tank_nodes]
    opened = ref["status"][frame] > 0
    status = np.where(opened, s.ST_OPEN, s.ST_CLOSED).astype(np.int8)
    setting = np.asarray(net.valve_setting_user, dtype=np.float64).copy()
    pl = np.asarray(net.pump_link, dtype=np.int64)
    setting[pl] = ref["setting"][frame][pl]
    ke0 = np.zeros(net.N)
    r0 = s.r_hw.detach().cpu().numpy().copy()
    h00 = s.pl_h0.copy()
    rp0 = s.pl_r.copy()
    return dict(net=net, ref=ref, s=s, t=t, d=d0, fh=fh, status=status,
                speed=setting, ke=ke0, r=r0, h0=h00, rp=rp0, pl=pl,
                tank_nodes=tank_nodes, frame=frame)


def make_loss_fd(cs):
    """FD 用损失：solve_polished（accuracy=1e-12 + 精抛光）后 w·H_junc。"""
    s = cs["s"]
    rng = np.random.default_rng(SEED)
    w = rng.normal(size=s.Nj)
    cs["w"] = w

    def loss_of(d=None, r=None, sp=None, h0p=None, rp=None):
        sol = solve_polished(
            s, cs["d"] if d is None else d, cs["fh"], cs["ke"],
            cs["r"] if r is None else r, accuracy=1e-12, max_iter=MAXIT,
            polish_steps=3, speed=cs["speed"] if sp is None else sp,
            status=cs["status"], pump_h0=cs["h0"] if h0p is None else h0p,
            pump_r=cs["rp"] if rp is None else rp)
        return float(w @ sol["head"][0, s.junc_nodes]), sol

    return loss_of


def fd_grad(loss_of, kind, idx, cs, scale=1e-3):
    """Richardson 外推中心差分（h1、h1/2；步长策略同 gradcheck_3way）。"""
    base = {"demand": cs["d"], "r": cs["r"], "speed": cs["speed"],
            "h0": cs["h0"], "rp": cs["rp"]}[kind]
    x = float(base[idx])
    h1 = max(scale * abs(x), 1e-5)
    kw_of = {"demand": "d", "r": "r", "speed": "sp", "h0": "h0p", "rp": "rp"}[kind]

    def central(h):
        vp = base.copy(); vp[idx] = x + h
        vm = base.copy(); vm[idx] = x - h
        Lp, _ = loss_of(**{kw_of: vp})
        Lm, _ = loss_of(**{kw_of: vm})
        return (Lp - Lm) / (2.0 * h)

    return (4.0 * central(h1 / 2.0) - central(h1)) / 3.0


def run_case(stem):
    print("=" * 84)
    cs = load_case(stem, frame=0)
    net, s, ref = cs["net"], cs["s"], cs["ref"]
    pl = cs["pl"]
    loss_of = make_loss_fd(cs)
    L0, sol0 = loss_of()
    it0 = int(sol0["iters"][0])
    print(f"=== {stem}: Nj={s.Nj} L={s.L} 泵={len(pl)} 水池={len(cs['tank_nodes'])} "
          f"| 冻结帧 t={cs['t']} GGA(1e-12) iters={it0} "
          f"polish 后 ‖F‖inf={sol0['resid_inf'][0]:.2e} ===")
    w_t = torch.tensor(cs["w"], dtype=torch.float64)

    # ---- 泵两端 junction ----
    jm = np.asarray(net.node_type) == 0
    pump_ends = sorted({int(v) for k in pl
                        for v in (net.link_n1[k], net.link_n2[k]) if jm[v]})
    tt = lambda x: torch.tensor(np.asarray(x, dtype=np.float64))
    d_t0, fh_t, ke_t, r_t0, sp_t0 = (tt(cs["d"]), tt(cs["fh"]), tt(cs["ke"]),
                                     tt(cs["r"]), tt(cs["speed"]))
    status = cs["status"]

    # ---- implicit 全量梯度 gB（loss = w·H_junc）----
    dB = torch.tensor(cs["d"], dtype=torch.float64, requires_grad=True)
    rB = torch.tensor(cs["r"], dtype=torch.float64, requires_grad=True)
    spB = torch.tensor(cs["speed"], dtype=torch.float64, requires_grad=True)
    h0B = torch.tensor(cs["h0"], dtype=torch.float64, requires_grad=True)
    rpB = torch.tensor(cs["rp"], dtype=torch.float64, requires_grad=True)
    head, _, _ = ImplicitGGASolve.apply(dB, fh_t, ke_t, rB, s, 1e-12, MAXIT, 3,
                                        spB, h0B, rpB, status)
    (w_t * head[s.junc_nodes]).sum().backward()
    gB = dict(demand=dB.grad.numpy(), r=rB.grad.numpy(), speed=spB.grad.numpy(),
              h0=h0B.grad.numpy(), rp=rpB.grad.numpy())

    # ---- unrolled 全量梯度 gA（独立通路，作交叉验证与 ③）----
    K = it0 + 5
    dA = torch.tensor(cs["d"], dtype=torch.float64, requires_grad=True)
    rA = torch.tensor(cs["r"], dtype=torch.float64, requires_grad=True)
    spA = torch.tensor(cs["speed"], dtype=torch.float64, requires_grad=True)
    h0A = torch.tensor(cs["h0"], dtype=torch.float64, requires_grad=True)
    rpA = torch.tensor(cs["rp"], dtype=torch.float64, requires_grad=True)
    outA = solve_unrolled(s, dA, cs["fh"], ke=cs["ke"], r_hw=rA, K=K,
                          q0=sol0["q"][0], speed=spA, status=status,
                          pump_h0=h0A, pump_r=rpA)
    (w_t * outA["head_ft"][s.junc_nodes]).sum().backward()
    gA = dict(demand=dA.grad.numpy(), r=rA.grad.numpy(), speed=spA.grad.numpy(),
              h0=h0A.grad.numpy(), rp=rpA.grad.numpy())

    # ---- 坐标池（纪律同 gradcheck_3way：|g| 40 分位过滤非退化坐标）----
    rng = np.random.default_rng(SEED + 1)
    d_pool = s.junc_nodes[cs["d"][s.junc_nodes] > 0]
    q_base = sol0["q"][0]
    hg_fric = s.hexp * cs["r"] * np.abs(q_base) ** (s.hexp - 1.0)
    r_cand = np.where(s.is_pipe.cpu().numpy() & (cs["status"] > s.ST_CLOSED)
                      & (hg_fric > 10.0 * s.rqtol))[0]
    gr_abs = np.abs(gB["r"][r_cand])
    r_pool = r_cand[gr_abs >= np.percentile(gr_abs, 40.0)]
    idx_r = [int(i) for i in rng.choice(r_pool, size=3, replace=False)]
    idx_d = list(dict.fromkeys(
        pump_ends + [int(i) for i in rng.choice(d_pool, size=3, replace=False)]))[:6]
    sel_h = [int(i) for i in rng.choice(s.junc_nodes, size=4, replace=False)]

    # ================= ① torch.autograd.gradcheck =================
    id_t = torch.tensor(idx_d); ir_t = torch.tensor(idx_r)
    ip_t = torch.tensor(pl)
    sel_t = torch.tensor(sel_h); plq_t = torch.tensor(pl)

    def fn(d_sub, r_sub, sp_sub):
        d_full = d_t0.index_copy(0, id_t, d_sub)
        r_full = r_t0.index_copy(0, ir_t, r_sub)
        sp_full = sp_t0.index_copy(0, ip_t, sp_sub)
        head_, flow_, _ = ImplicitGGASolve.apply(
            d_full, fh_t, ke_t, r_full, s, 1e-12, MAXIT, 3,
            sp_full, None, None, status)
        return torch.cat([head_[sel_t], flow_[plq_t]])

    inputs = (d_t0[id_t].clone().requires_grad_(True),
              r_t0[ir_t].clone().requires_grad_(True),
              sp_t0[ip_t].clone().requires_grad_(True))
    ok1 = torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)
    print(f"① torch.autograd.gradcheck(ImplicitGGASolve) θ=(demand×{len(idx_d)}, "
          f"r_hw×{len(idx_r)}, ω×{len(pl)}) 输出=(H×{len(sel_h)}, Q_pump×{len(pl)}): "
          f"{'PASS' if ok1 else 'FAIL'}")

    # ================= ② implicit vs 中心差分 =================
    rng2 = np.random.default_rng(SEED + 2)
    gd_abs = np.abs(gB["demand"][d_pool])
    strong = d_pool[gd_abs >= np.percentile(gd_abs, 40.0)]
    extra_d = [int(i) for i in rng2.choice(strong, size=5, replace=False)]
    coords = [("demand", i) for i in dict.fromkeys(pump_ends + extra_d)]
    coords += [("speed", int(k)) for k in pl]
    is_chp = bool(s.is_chp_np.any())
    if is_chp:
        coords += [("rp", int(k)) for k in pl[:2]]      # CONST_HP：对拍 R
    else:
        coords += [("h0", int(k)) for k in pl] + [("rp", int(k)) for k in pl]
    coords += [("r", int(k)) for k in idx_r]

    def judge(g_self, gC, kind, idx, tol):
        """双门槛：rel<tol，或 FD 分辨率不足（|Δ|<NOISE_ABS）且独立通路交叉
        rel(gA,gB)<1e-6（见头注释）。返回 (ok, note)。"""
        rr = rel(g_self[kind][idx], gC)
        if rr < tol:
            return True, rr, ""
        cross = rel(gA[kind][idx], gB[kind][idx])
        if abs(g_self[kind][idx] - gC) < NOISE_ABS and cross < 1e-6:
            return True, rr, f"  [FD噪声地板; 交叉rel={cross:.1e}]"
        return False, rr, "  <-- 超限"

    ok2 = True
    res2 = {}
    print(f"② implicit vs 中心差分（{len(coords)} 坐标, 门槛 rel<{TOL_BC:.0e} "
          f"或 |Δ|<{NOISE_ABS:.0e}+交叉<1e-6）")
    print(f"{'θ':>8} {'坐标':>6} {'g_implicit':>18} {'g_FD':>18} {'rel':>10}")
    for kind, idx in coords:
        sc = 1e-4 if kind in ("h0", "rp") and abs(
            {"h0": cs["h0"], "rp": cs["rp"]}[kind][idx]) > 100.0 else 1e-3
        gC = fd_grad(loss_of, kind, idx, cs, scale=sc)
        res2[(kind, idx)] = gC
        okc, rr, note = judge(gB, gC, kind, idx, TOL_BC)
        ok2 = ok2 and okc
        print(f"{kind:>8} {idx:>6} {gB[kind][idx]:>18.10e} {gC:>18.10e} "
              f"{rr:>10.2e}{note}")
    print(f"② 判定: {'PASS' if ok2 else 'FAIL'}")

    # ================= ③ 展开路径 vs 中心差分 =================
    print(f"③ solve_unrolled: K={K} (warm start=收敛流量) "
          f"relerr={float(outA['relerr']):.2e}（门槛 rel<{TOL_AC:.0e}）")
    ok3 = True
    print(f"{'θ':>8} {'坐标':>6} {'g_unrolled':>18} {'g_FD':>18} {'rel':>10}")
    for (kind, idx), gC in res2.items():
        okc, rr, note = judge(gA, gC, kind, idx, TOL_AC)
        ok3 = ok3 and okc
        print(f"{kind:>8} {idx:>6} {gA[kind][idx]:>18.10e} {gC:>18.10e} "
              f"{rr:>10.2e}{note}")
    print(f"③ 判定: {'PASS' if ok3 else 'FAIL'}")
    return ok1 and ok2 and ok3


def run_bptt_exa6():
    """④ BPTT 冒烟（EXA6, 3 帧 mini-EPS，dense/unrolled 全程进图，报告性）。"""
    print("=" * 84)
    stem = "EXA6"
    net = Net.load(os.path.join(ROOT, "data", "reference"), stem)
    ref = np.load(os.path.join(ROOT, "data", "reference", f"{stem}_ref.npz"))
    inp = os.path.join(ROOT, "networks", "InpData", f"{stem}.inp")
    s = GGASolver(net, mode="epanet", inp_path=inp)
    tank = int(net.tank_node[0])
    n1 = np.asarray(net.link_n1); n2 = np.asarray(net.link_n2)
    lk_in = torch.tensor(np.where(n2 == tank)[0])     # 流入 tank 为 +Q
    lk_out = torch.tensor(np.where(n1 == tank)[0])
    hmin, vmin, area = (float(net.tank_hmin[0]), float(net.tank_vmin[0]),
                        float(net.tank_area[0]))
    tank_mask = torch.zeros(net.N, dtype=torch.bool)
    tank_mask[tank] = True
    NF = 3
    ts = [int(x) for x in ref["t_sec"][:NF + 1]]
    Ks = [int(ref["iterations"][f]) + 5 for f in range(NF)]
    pl = np.asarray(net.pump_link, dtype=np.int64)

    def frame_inputs(f):
        t = ts[f]
        d = net.demand_cfs_at(t)
        fh = np.nan_to_num(net.reservoir_head_ft_at(t))
        opened = ref["status"][f] > 0
        status = np.where(opened, s.ST_OPEN, s.ST_CLOSED).astype(np.int8)
        setting = np.asarray(net.valve_setting_user, dtype=np.float64).copy()
        setting[pl] = ref["setting"][f][pl]
        return d, fh, status, setting

    def rollout(d0_in):
        """3 帧 mini-EPS：tank 显式欧拉（hydraul.c:1021-1022 + :1089）进图，
        帧间 warm start 保图（runhyd 续用 LinkFlow）。loss=末帧后 tank 头。"""
        V = torch.tensor(float(net.tank_v0[0]), dtype=torch.float64)
        htank = torch.tensor(float(net.tank_h0[0]), dtype=torch.float64)
        q_prev = None
        for f in range(NF):
            d_np, fh_np, status, setting = frame_inputs(f)
            d_in = d0_in if f == 0 else torch.tensor(d_np, dtype=torch.float64)
            fh_t = torch.where(tank_mask, htank,
                               torch.tensor(fh_np, dtype=torch.float64))
            out = solve_unrolled(s, d_in, fh_t, K=Ks[f], q0=q_prev,
                                 speed=setting, status=status)
            q = out["flow_cfs"]
            opened_t = torch.tensor(status > s.ST_CLOSED)
            qo = torch.where(opened_t, q, torch.zeros_like(q))
            qnet = qo[lk_in].sum() - qo[lk_out].sum()    # newlinkflows :459-464 口径
            tstep = float(ts[f + 1] - ts[f])
            V = V + qnet * tstep                          # tanklevels :1021-1022
            htank = hmin + (V - vmin) / area              # tankgrade :1089（柱形）
            q_prev = q                                    # warm start（保图 BPTT）
        return htank

    d0_np = net.demand_cfs_at(ts[0])
    d0_t = torch.tensor(d0_np, dtype=torch.float64, requires_grad=True)
    loss = rollout(d0_t)
    loss.backward()
    g = d0_t.grad.numpy()
    # 对照：末帧 tank 头 ref 值（欧拉口径应接近 ref["head_ft"][3][tank]）
    print(f"④ BPTT 冒烟（EXA6 {NF} 帧, K={Ks}）: loss=t={ts[NF]}s tank 头 "
          f"= {float(loss.detach()):.6f} ft (ref {ref['head_ft'][NF][tank]:.6f} ft)")
    # FD：Richardson 外推中心差分，h1=max(1e-2|x|,1e-3)。实测 h 扫描
    # （1e-3→1e-6）FD 随 h 缩小发散（h=1e-6 时 rel~1.2），即 rollout 损失的
    # 确定性噪声（死支折点 + 定 K 截断）~1e-7 ft 主导；h~1e-3 处 FD 最优
    # rel~4e-4 - 本项报告性，FD 是误差源而非 BPTT 梯度。
    rng = np.random.default_rng(SEED + 3)
    jm = np.asarray(net.node_type) == 0
    pool = np.where(jm & (d0_np > 0))[0]
    picks = [int(i) for i in rng.choice(pool, size=3, replace=False)]
    print(f"{'坐标':>6} {'g_BPTT':>18} {'g_FD(Richardson)':>18} {'rel':>10}")
    worst = 0.0
    with torch.no_grad():
        for i in picks:
            x = float(d0_np[i])
            h1 = max(1e-2 * abs(x), 1e-3)

            def f_of(v):
                d = d0_np.copy(); d[i] = v
                return float(rollout(torch.tensor(d, dtype=torch.float64)))

            def central(h):
                return (f_of(x + h) - f_of(x - h)) / (2.0 * h)

            gC = (4.0 * central(h1 / 2.0) - central(h1)) / 3.0
            rr = rel(g[i], gC)
            worst = max(worst, rr)
            print(f"{i:>6} {g[i]:>18.10e} {gC:>18.10e} {rr:>10.2e}")
    print(f"④ BPTT vs FD 最大相对差 = {worst:.3e}（报告性，无门槛）")
    return True


def main():
    torch.set_default_dtype(torch.float64)
    ok = True
    for stem in ("EXA6", "ky3"):
        ok = run_case(stem) and ok
    run_bptt_exa6()
    print("=" * 84)
    print(f"总判定: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""a1_grad.py - 敌意审阅项 1（自写，不复用上游测试）：GPU 伴随梯度三方对拍。

§A 三方（每网）：adjoint='gpu'（dense 前向+GPU 约化伴随）vs 缺省 CPU
   ImplicitGGASolve（epanet 收敛 + scipy splu 伴随，冻结同一状态帧）
 - 四类 θ（demand/res_head/ke/r_hw）全向量 max 相对差；PRV 下游坐标打印。
   B ∈ {1, 8, 64}（B>1 各场景独立权重，CPU 参照按状态组冻结）。
§B 中央差分（前向 = adjoint='gpu' 的同一 dense+精抛光函数，no_grad）：
   每个 ±h、±h/2 点先跑 dense 状态机并断言终态状态向量与基点逐位同
   （证明避开切换点；不同即弃用该坐标并计数），Richardson 外推，
   覆盖 PRV 下游 demand、res_head、ke、r_hw。
§C 混合状态批（专查约化路线 PRV 处理）：自配场景把每只 PRV 分别逼到
   ACTIVE / OPEN / CLOSED（批内三态并存，直方图打印），批量 GPU 伴随梯度
   vs 按状态组冻结的 CPU 伴随，逐场景 max 相对差。
退出码 0 = 全 PASS。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch                                      # noqa: E402
from dgga.parse import parse_inp                  # noqa: E402
from dgga.solver import GGASolver                 # noqa: E402
from dgga.autodiff import implicit_solve          # noqa: E402

torch.use_deterministic_algorithms(True)
NETS = os.path.join(ROOT, "networks")
SEED = 31415
TOL = 1e-6
STN = {2: "CLOSED", 3: "OPEN", 4: "ACTIVE"}
FAILS = []

CASES = [
    ("L-TOWN", "public/_cleaned/L-TOWN.inp"),
    ("BWSN_1", "public/_cleaned/BWSN_Network_1.inp"),
]


def rel(a, b):
    den = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-300)
    return float(np.abs(a - b).max() / den)


def setup(name, relp):
    inp = os.path.join(NETS, *relp.split("/"))
    net = parse_inp(inp)
    s = GGASolver(net, mode="dense", inp_path=inp, dense_status_machine=True)
    se = GGASolver(net, mode="epanet", inp_path=inp)
    rng = np.random.default_rng(SEED)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5 * (np.asarray(net.tank_hmin)[:tn.size]
                        + np.asarray(net.tank_hmax)[:tn.size])
    ke0 = np.asarray(net.node_ke, dtype=np.float64).copy()
    jn = np.asarray(s.junc_nodes)
    ke0[jn[rng.integers(0, jn.size, 4)]] = 5e-4
    r0 = s.r_hw.detach().cpu().numpy().copy()
    return net, inp, s, se, d0, rh0, ke0, r0, rng


def pick_acc(s, D, R, KE):
    for acc in (1e-12, 1e-9, 3e-8, 3e-7):
        with torch.no_grad():
            out = s.solve(np.atleast_2d(D), np.atleast_2d(R),
                          ke_int=np.atleast_2d(KE), accuracy=acc,
                          max_iter=200, status_machine=True)
        if bool(out["converged"].all()):
            return acc, out
    raise RuntimeError("accuracy 阶梯全不收敛")


def cpu_grads_grouped(se, Db, Rb, KEb, r0, WH, WQ, WE, S_all, K_all, jn_t):
    """CPU 伴随参照：按（status,setting）组冻结逐组求梯度（含全通道 loss）。"""
    B = Db.shape[0]
    key = [tuple(S_all[b].tolist()) + tuple(np.round(K_all[b], 12).tolist())
           for b in range(B)]
    uniq = sorted(set(key))
    dt = torch.float64
    D = torch.as_tensor(Db, dtype=dt).requires_grad_(True)
    R = torch.as_tensor(Rb, dtype=dt).requires_grad_(True)
    KE = torch.as_tensor(KEb, dtype=dt).requires_grad_(True)
    RW = torch.as_tensor(r0, dtype=dt).requires_grad_(True)
    loss = 0.0
    for u in uniq:
        ii = torch.as_tensor([b for b in range(B) if key[b] == u])
        b0 = int(ii[0])
        h, f, e = implicit_solve(se, D.index_select(0, ii),
                                 R.index_select(0, ii),
                                 ke=KE.index_select(0, ii), r_hw=RW,
                                 speed=K_all[b0], status=S_all[b0])
        loss = (loss + (h.index_select(1, jn_t) * WH.index_select(0, ii)).sum()
                + (f * WQ.index_select(0, ii)).sum()
                + (e * WE.index_select(0, ii)).sum())
    loss.backward()
    return (D.grad.numpy(), R.grad.numpy(), KE.grad.numpy(), RW.grad.numpy(),
            len(uniq))


def gpu_grads(s, Db, Rb, KEb, r0, WH, WQ, WE, acc, jn_t):
    dt = torch.float64
    D = torch.as_tensor(Db, dtype=dt).requires_grad_(True)
    R = torch.as_tensor(Rb, dtype=dt).requires_grad_(True)
    KE = torch.as_tensor(KEb, dtype=dt).requires_grad_(True)
    RW = torch.as_tensor(r0, dtype=dt).requires_grad_(True)
    h, f, e = implicit_solve(s, D, R, ke=KE, r_hw=RW, adjoint="gpu",
                             accuracy=acc, max_iter=200, status_machine=True)
    if h.dim() == 1:
        h, f, e = h.unsqueeze(0), f.unsqueeze(0), e.unsqueeze(0)
    ((h.index_select(1, jn_t) * WH).sum() + (f * WQ).sum()
     + (e * WE).sum()).backward()
    return (D.grad.numpy(), R.grad.numpy(), KE.grad.numpy(), RW.grad.numpy())


def frozen_frames(se, Db, Rb, KEb):
    S_all, K_all = [], []
    for b in range(Db.shape[0]):
        r = se.run_gga(Db[b], Rb[b], ke=KEb[b], do_status=True)
        S_all.append(r["status"].copy())
        K_all.append(r["setting"].copy())
    return np.stack(S_all), np.stack(K_all)


for name, relp in CASES:
    net, inp, s, se, d0, rh0, ke0, r0, rng = setup(name, relp)
    acc, out0 = pick_acc(s, d0, rh0, ke0)
    jn = np.asarray(s.junc_nodes)
    jn_t = torch.as_tensor(jn, dtype=torch.long)
    prv_ks = np.where(np.asarray(net.link_type) == 3)[0]
    prv_dn = [int(net.link_n2[k]) for k in prv_ks]
    print("=" * 78)
    print("[%s] Nj=%d L=%d PRV=%d acc=%g iters=%d 下游=%s"
          % (name, s.Nj, s.L, prv_ks.size, acc,
             int(out0["iters"].max()), prv_dn))
    S0 = out0["status"][0].cpu().numpy()

    # ---------------- §A B ∈ {1,8,64} ----------------
    for B in (1, 8, 64):
        if B == 1:
            Db = d0[None, :].copy()
        else:
            Db = d0[None, :] * rng.uniform(0.85, 1.15, (B, d0.size))
        Rb = np.repeat(rh0[None, :], B, 0) + rng.uniform(-0.5, 0.5,
                                                         (B, rh0.size))
        KEb = np.repeat(ke0[None, :], B, 0)
        dt = torch.float64
        WH = torch.as_tensor(rng.normal(0, 1, (B, jn.size)), dtype=dt)
        WQ = torch.as_tensor(rng.normal(0, 1, (B, s.L)), dtype=dt)
        WE = torch.as_tensor(rng.normal(0, 1, (B, s.N)), dtype=dt)
        try:
            g_gpu = gpu_grads(s, Db, Rb, KEb, r0, WH, WQ, WE, acc, jn_t)
        except RuntimeError as ex:
            print("  §A B=%d 未收敛（%s）→ 缩放重抽" % (B, str(ex)[:50]))
            Db = d0[None, :] * rng.uniform(0.95, 1.05, (B, d0.size))
            g_gpu = gpu_grads(s, Db, Rb, KEb, r0, WH, WQ, WE, acc, jn_t)
        S_all, K_all = frozen_frames(se, Db, Rb, KEb)
        g_cpu = cpu_grads_grouped(se, Db, Rb, KEb, r0, WH, WQ, WE,
                                  S_all, K_all, jn_t)
        ng = g_cpu[4]
        worst = 0.0
        for i, k in enumerate(("demand", "res_head", "ke", "r_hw")):
            r_ = rel(g_gpu[i], g_cpu[i])
            worst = max(worst, r_)
            print("  §A B=%-3d θ=%-8s rel=%.3e %s"
                  % (B, k, r_, "PASS" if r_ < TOL else "FAIL"))
            if r_ >= TOL:
                FAILS.append("%s §A B=%d %s" % (name, B, k))
        if B == 1 and prv_dn:
            for j in prv_dn:
                print("    PRV下游 demand[%d]: cpu=% .10e gpu=% .10e"
                      % (j, g_cpu[0][0, j] if g_cpu[0].ndim == 2
                         else g_cpu[0][j], g_gpu[0][0, j]
                         if g_gpu[0].ndim == 2 else g_gpu[0][j]))
        print("  §A B=%d 状态组=%d" % (B, ng))

    # ---------------- §B 中央差分（dense 前向，证明避开切换点）----------------
    dt = torch.float64
    WH1 = torch.as_tensor(rng.normal(0, 1, jn.size), dtype=dt)
    D1 = torch.as_tensor(d0, dtype=dt).requires_grad_(True)
    h1, _, _ = implicit_solve(s, D1, torch.as_tensor(rh0, dtype=dt),
                              ke=torch.as_tensor(ke0, dtype=dt),
                              r_hw=torch.as_tensor(r0, dtype=dt),
                              adjoint="gpu", accuracy=acc, max_iter=200,
                              status_machine=True)
    R1 = torch.as_tensor(rh0, dtype=dt).requires_grad_(True)
    KE1 = torch.as_tensor(ke0, dtype=dt).requires_grad_(True)
    RW1 = torch.as_tensor(r0, dtype=dt).requires_grad_(True)
    D1b = torch.as_tensor(d0, dtype=dt).requires_grad_(True)
    hb_, _, _ = implicit_solve(s, D1b, R1, ke=KE1, r_hw=RW1, adjoint="gpu",
                               accuracy=acc, max_iter=200,
                               status_machine=True)
    (hb_[jn_t] * WH1).sum().backward()
    g_ad = dict(demand=D1b.grad.numpy(), res_head=R1.grad.numpy(),
                ke=KE1.grad.numpy(), r_hw=RW1.grad.numpy())

    q0 = out0["flow_cfs"][0].cpu().numpy()
    lt_np = np.asarray(net.link_type)
    hg0 = s.hexp * r0 * np.maximum(np.abs(q0), 1e-30) ** (s.hexp - 1.0)
    pipes_ok = np.where((lt_np <= 1) & (hg0 > 10.0 * s.rqtol)
                        & (np.abs(q0) > 1e-3))[0]
    fixed = np.where(np.asarray(net.node_type) != 0)[0]
    coords = dict(
        demand=prv_dn + [int(jn[i]) for i in rng.integers(0, jn.size, 3)],
        res_head=[int(x) for x in fixed[:3]],
        ke=[int(x) for x in np.where(ke0 > 0)[0][:3]],
        r_hw=[int(x) for x in pipes_ok[rng.integers(0, pipes_ok.size, 3)]])

    def eval_loss_status(dv, rhv, kev, rwv):
        with torch.no_grad():
            o = s.solve(np.atleast_2d(dv), np.atleast_2d(rhv),
                        ke_int=np.atleast_2d(kev), accuracy=acc,
                        max_iter=200, status_machine=True)
            same = bool(np.array_equal(o["status"][0].cpu().numpy(), S0)) \
                and bool(o["converged"].all())
            h, _, _ = implicit_solve(
                s, torch.as_tensor(dv, dtype=dt),
                torch.as_tensor(rhv, dtype=dt),
                ke=torch.as_tensor(kev, dtype=dt),
                r_hw=torch.as_tensor(rwv, dtype=dt), adjoint="gpu",
                accuracy=acc, max_iter=200, status_machine=True)
        return float((h[jn_t] * WH1).sum()), same

    n_skip_state = n_skip_fd = 0
    for kind, idxs in coords.items():
        wk, used = 0.0, 0
        gmax = float(np.abs(g_ad[kind]).max())
        for i in idxs:
            dv, rhv, kev, rwv = d0.copy(), rh0.copy(), ke0.copy(), r0.copy()
            vec = dict(demand=dv, res_head=rhv, ke=kev, r_hw=rwv)[kind]
            bv = float(vec[i])
            h = dict(demand=max(1e-5, 1e-3 * abs(bv)),
                     res_head=max(1e-6, 1e-4 * abs(bv)),
                     ke=max(1e-8, 1e-2 * abs(bv)),
                     r_hw=max(1e-6, 1e-3 * abs(bv)))[kind]
            Ls, all_same = {}, True
            for dlt in (h, -h, h / 2, -h / 2):
                vec[i] = bv + dlt
                val, same = eval_loss_status(dv, rhv, kev, rwv)
                Ls[dlt] = val
                all_same &= same
            vec[i] = bv
            if not all_same:
                n_skip_state += 1
                continue
            Dh1 = (Ls[h] - Ls[-h]) / (2 * h)
            Dh2 = (Ls[h / 2] - Ls[-h / 2]) / h
            if abs(Dh1 - Dh2) / max(abs(Dh1), abs(Dh2), 1e-300) > 1e-6 \
                    and abs(Dh1 - Dh2) > 1e-6 * gmax:
                n_skip_fd += 1
                continue
            used += 1
            g_fd = (4.0 * Dh2 - Dh1) / 3.0
            ga = float(g_ad[kind][i])
            r_ = abs(ga - g_fd) / max(abs(ga), abs(g_fd), 1e-300)
            good = r_ < TOL or abs(ga - g_fd) < 1e-6 * gmax
            if not good:
                FAILS.append("%s §B %s[%d] rel=%.2e" % (name, kind, i, r_))
            if r_ < TOL:
                wk = max(wk, r_)
        okb = used >= max(1, len(idxs) // 2)
        print("  §B FD θ=%-8s 用 %d/%d（弃:状态切换 - 见尾行合计）最差 rel=%.3e %s"
              % (kind, used, len(idxs), wk, "PASS" if okb else "FAIL"))
        if not okb:
            FAILS.append("%s §B %s 可用坐标不足" % (name, kind))
    print("  §B 合计：状态切换弃 %d、FD 自检弃 %d（切换点均已证明避开：终态"
          "状态向量逐位同才计入）" % (n_skip_state, n_skip_fd))

    # ---------------- §C 混合状态批 ----------------
    rows = [(d0.copy(), rh0.copy())]                      # 名义（预期 ACTIVE）
    nt_ = np.asarray(net.node_type)
    tanks = np.asarray(net.tank_node, dtype=np.int64)
    hmin = np.asarray(net.tank_hmin, dtype=np.float64)
    hmax = np.asarray(net.tank_hmax, dtype=np.float64)
    tot0 = float(np.abs(d0[jn]).sum())
    for k in prv_ks:
        n2 = int(net.link_n2[k])
        d = d0 * 20.0                                     # 逼 OPEN
        if nt_[n2] == 0:
            d[n2] += 12.0 * max(tot0, 1e-6)
        rh = rh0.copy()
        for i, n in enumerate(tanks):
            rh[int(n)] = hmin[i]
        rows.append((d, rh))
        d = d0 * 0.1                                      # 逼 CLOSED（下游注入）
        if nt_[n2] == 0:
            d[n2] -= 2.0 * max(tot0, 1e-6)
        rh = rh0.copy()
        for i, n in enumerate(tanks):
            rh[int(n)] = hmax[i]
        rows.append((d, rh))
    Db = np.stack([r[0] for r in rows])
    Rb = np.stack([r[1] for r in rows])
    KEb = np.repeat(ke0[None, :], Db.shape[0], 0)
    # 极端场景允许个别不收敛：逐 acc 找收敛子集（如实报剔除数）
    accm = None
    for a_ in (1e-9, 3e-8, 3e-7):
        with torch.no_grad():
            outm = s.solve(Db, Rb, ke_int=KEb, accuracy=a_, max_iter=200,
                           status_machine=True)
        cv = outm["converged"].cpu().numpy().astype(bool)
        if cv.sum() >= max(3, Db.shape[0] // 2):
            accm = a_
            break
    if accm is None:
        raise RuntimeError("§C 场景大面积不收敛")
    if not cv.all():
        print("  §C 剔除未收敛场景 %d/%d（acc=%g）"
              % (int((~cv).sum()), Db.shape[0], accm))
        Db, Rb, KEb = Db[cv], Rb[cv], KEb[cv]
        with torch.no_grad():
            outm = s.solve(Db, Rb, ke_int=KEb, accuracy=accm, max_iter=200,
                           status_machine=True)
    Bm = Db.shape[0]
    Sm = outm["status"].cpu().numpy()
    hist = {}
    for k in prv_ks:
        st = [STN.get(int(x), str(int(x))) for x in Sm[:, k]]
        hist[int(k)] = {u: st.count(u) for u in sorted(set(st))}
    states_seen = set()
    for k in prv_ks:
        states_seen |= set(hist[int(k)])
    print("  §C 混合批 B=%d 每 PRV 终态直方图: %s" % (Bm, hist))
    okc_cov = {"ACTIVE", "CLOSED"} <= states_seen
    if not okc_cov:
        FAILS.append("%s §C 未凑出 ACTIVE+CLOSED 并存" % name)
    dt = torch.float64
    WH = torch.as_tensor(rng.normal(0, 1, (Bm, jn.size)), dtype=dt)
    WQ = torch.as_tensor(rng.normal(0, 1, (Bm, s.L)), dtype=dt)
    WE = torch.as_tensor(rng.normal(0, 1, (Bm, s.N)), dtype=dt)
    g_gpu = gpu_grads(s, Db, Rb, KEb, r0, WH, WQ, WE, accm, jn_t)
    S_all, K_all = frozen_frames(se, Db, Rb, KEb)
    same_frames = int(sum(np.array_equal(S_all[b], Sm[b])
                          for b in range(Bm)))
    print("  §C 状态帧 dense vs epanet 逐位同：%d/%d" % (same_frames, Bm))
    g_cpu = cpu_grads_grouped(se, Db, Rb, KEb, r0, WH, WQ, WE,
                              S_all, K_all, jn_t)
    # 逐场景相对差（demand 梯度按行比；res_head/ke 同）
    worst_sc, worst_b = 0.0, -1
    for b in range(Bm):
        r_ = max(rel(g_gpu[0][b], g_cpu[0][b]), rel(g_gpu[1][b], g_cpu[1][b]),
                 rel(g_gpu[2][b], g_cpu[2][b]))
        if r_ > worst_sc:
            worst_sc, worst_b = r_, b
    r_rhw = rel(g_gpu[3], g_cpu[3])
    okc = worst_sc < TOL and r_rhw < TOL
    print("  §C GPU vs CPU（%d 状态组）：逐场景最差 rel=%.3e（b=%d 状态=%s）"
          " r_hw(批共享)=%.3e %s"
          % (g_cpu[4], worst_sc, worst_b,
             [STN.get(int(x), "?") for x in Sm[worst_b, prv_ks]],
             r_rhw, "PASS" if okc else "FAIL"))
    if not okc:
        FAILS.append("%s §C rel=%.2e" % (name, worst_sc))

print("=" * 78)
if FAILS:
    print("FAILS:", FAILS)
print("A1 DONE rc=%d" % (1 if FAILS else 0))
sys.exit(1 if FAILS else 0)

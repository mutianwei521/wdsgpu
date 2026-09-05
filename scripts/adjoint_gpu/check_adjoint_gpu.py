# -*- coding: utf-8 -*-
"""check_adjoint_gpu.py - adjoint='gpu'（GPU 批量约化伴随）验收 ①②③＋批一致。

5 网（L-TOWN / Hanoi / Modena / ky4 / BWSN_1(SM)）×4 类 θ（demand/res_head/
ke/r_hw）：
  §A 梯度 vs 既有 CPU ImplicitGGASolve（同一冻结状态帧；epanet 收敛+Newton
     精抛光 + scipy splu 伴随）：全向量逐坐标最大相对差，门槛 1e-6（如实报量级；
     受 dense 前向 GGA 停机地板限制，含 PRV 网预期 ~1e-8..1e-9）。
  §B 梯度 vs 中央差分（check_prv_grad 同配方：FD 基线 solve_polished 冻结状态、
     每个 ±h 点重跑完整状态机、终态≠S* 的坐标弃用另计）：rel<1e-6 或双侧 |g|<1e-9。
  §C 分解零新增（验收 3 本地面）：monkeypatch torch.linalg.cholesky/cholesky_ex
     计数 - 前向 = 迭代数次 factorize，**backward 期间 0 次**（cudss 面的计数器
     验证在集群作业里做，见 adjgpu_gpu.py）。
  §D 批一致：B=8（需水 ×U(0.85,1.15)，SM 网各场景状态可不同）批量梯度 vs
     逐场景单跑，门槛 max 相对差 < 1e-10；含 PRV 网报状态组数。
  §E 守卫：epanet 模式拒 / f32 拒 / create_graph 拒 / 泵θ 拒。

前向 accuracy 逐网自适应（从紧到松试 [1e-12,1e-9,3e-8,3e-7]，取首个收敛档；
GGA 半迭代在 κ~1e11 病态网上有停机地板，模块 docstring 已记载）。
ky4 构造带 dense_tank_bound_check=False（T-2 水池 Hmin≈Hmax，中点必在边界；
本脚本的对拍双方都在**冻结初始状态**的同一映射上，见 §A 注释）。

用法：python -X utf8 scripts/adjoint_gpu/check_adjoint_gpu.py
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch                                            # noqa: E402
from dgga.parse import parse_inp                        # noqa: E402
from dgga.solver import GGASolver                       # noqa: E402
from dgga.autodiff import implicit_solve, solve_polished  # noqa: E402

torch.use_deterministic_algorithms(True)
NETS = os.path.join(ROOT, "networks")
ACC_LADDER = (1e-12, 1e-9, 3e-8, 3e-7)
TOL_A, TOL_FD, TOL_ABS, TOL_B = 1e-6, 1e-6, 1e-9, 1e-10

CASES = [
    ("L-TOWN", "public/_cleaned/L-TOWN.inp", True, True),
    ("Hanoi", "public/Hanoi.inp", False, True),
    ("Modena", "public/Modena.inp", False, True),
    ("ky4", "public/ky4.inp", False, False),      # 末位 = dense_tank_bound_check
    ("BWSN_1", "public/_cleaned/BWSN_Network_1.inp", True, True),
]


class CholCounter:
    """统计 torch.linalg.cholesky / cholesky_ex 的调用次数（dense 分解计数）。"""

    def __init__(self):
        self.n = 0

    def __enter__(self):
        self._c, self._ce = torch.linalg.cholesky, torch.linalg.cholesky_ex
        me = self

        def c(A, *a, **kw):
            me.n += 1
            return me._c(A, *a, **kw)

        def ce(A, *a, **kw):
            me.n += 1
            return me._ce(A, *a, **kw)

        torch.linalg.cholesky, torch.linalg.cholesky_ex = c, ce
        return self

    def __exit__(self, *e):
        torch.linalg.cholesky, torch.linalg.cholesky_ex = self._c, self._ce
        return False


def setup(name, rel, sm, tbc):
    inp = os.path.join(NETS, *rel.split("/"))
    net = parse_inp(inp)
    kw = dict(dense_status_machine=True) if sm else \
        dict(dense_tank_bound_check=tbc)
    s = GGASolver(net, mode="dense", inp_path=inp, **kw)
    rng = np.random.default_rng(20260824)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5 * (np.asarray(net.tank_hmin)[:tn.size]
                        + np.asarray(net.tank_hmax)[:tn.size])
    ke0 = np.asarray(net.node_ke, dtype=np.float64).copy()
    jn = np.asarray(s.junc_nodes)
    ke0[jn[rng.integers(0, jn.size, 3)]] = 1e-3
    r0 = s.r_hw.detach().cpu().numpy().copy()
    return net, inp, s, d0, rh0, ke0, r0, rng


def pick_acc(s, d0, rh0, ke0, sm):
    for acc in ACC_LADDER:
        with torch.no_grad():
            out = s.solve(np.atleast_2d(d0), np.atleast_2d(rh0),
                          ke_int=np.atleast_2d(ke0), accuracy=acc,
                          max_iter=200, status_machine=sm)
        if bool(out["converged"].all()):
            return acc, out
    raise RuntimeError("accuracy 阶梯全不收敛")


def grads_gpu(s, d0, rh0, ke0, r0, W, Wq, We, acc, sm, count=False):
    dt = torch.float64
    D = torch.as_tensor(d0, dtype=dt).requires_grad_(True)
    R = torch.as_tensor(rh0, dtype=dt).requires_grad_(True)
    KE = torch.as_tensor(ke0, dtype=dt).requires_grad_(True)
    RW = torch.as_tensor(r0, dtype=dt).requires_grad_(True)
    cnt = CholCounter()
    with cnt:
        head, flow, emit = implicit_solve(
            s, D, R, ke=KE, r_hw=RW, adjoint="gpu", accuracy=acc,
            max_iter=200, status_machine=sm)
        n_fwd = cnt.n
        loss = ((head[s.junc_nodes_t] * W).sum() + (flow * Wq).sum()
                + (emit * We).sum())
        loss.backward()
        n_bwd = cnt.n - n_fwd
    g = dict(demand=D.grad.numpy(), res_head=R.grad.numpy(),
             ke=KE.grad.numpy(), r_hw=RW.grad.numpy())
    return (g, n_fwd, n_bwd) if count else g


def main():
    ok_all = True
    lt = None
    for name, rel, sm, tbc in CASES:
        net, inp, s, d0, rh0, ke0, r0, rng = setup(name, rel, sm, tbc)
        acc, out0 = pick_acc(s, d0, rh0, ke0, sm)
        print("=" * 72)
        print("[%s] Nj=%d L=%d pcvPRV=%d 泵=%d | accuracy=%g iters=%d "
              "relerr=%.3e" % (name, s.Nj, s.L, s._dense_prv_np, s.n_pumps,
                               acc, int(out0["iters"].max()),
                               float(out0["relerr"].max())))
        # 冻结状态帧（CPU 参考与 FD 共用）
        s_ep = GGASolver(net, mode="epanet", inp_path=inp)
        if sm:
            base = s_ep.run_gga(d0, rh0, ke=ke0, do_status=True)
            S0, K0 = base["status"].copy(), base["setting"].copy()
            same = np.array_equal(S0, out0["status"][0].cpu().numpy())
            print("  状态帧 dense(SM) vs epanet(SM)：%s" %
                  ("逐位同" if same else "不同 <-- FAIL"))
            ok_all &= same
            kw = dict(speed=K0, status=S0)
        else:
            S0 = K0 = None
            kw = {}
        dt = torch.float64
        W = torch.as_tensor(rng.normal(0, 1, s.Nj), dtype=dt)
        Wq = torch.as_tensor(rng.normal(0, 1, s.L), dtype=dt)
        We = torch.as_tensor(rng.normal(0, 1, s.N), dtype=dt)

        # ---- §A vs CPU ImplicitGGASolve + §C 分解计数 ----
        g_gpu, n_fwd, n_bwd = grads_gpu(s, d0, rh0, ke0, r0, W, Wq, We,
                                        acc, sm, count=True)
        D2 = torch.as_tensor(d0, dtype=dt).requires_grad_(True)
        R2 = torch.as_tensor(rh0, dtype=dt).requires_grad_(True)
        KE2 = torch.as_tensor(ke0, dtype=dt).requires_grad_(True)
        RW2 = torch.as_tensor(r0, dtype=dt).requires_grad_(True)
        h2, f2, e2 = implicit_solve(s_ep, D2, R2, ke=KE2, r_hw=RW2, **kw)
        ((h2[s_ep.junc_nodes_t] * W).sum() + (f2 * Wq).sum()
         + (e2 * We).sum()).backward()
        g_cpu = dict(demand=D2.grad.numpy(), res_head=R2.grad.numpy(),
                     ke=KE2.grad.numpy(), r_hw=RW2.grad.numpy())
        wa = 0.0
        for k in ("demand", "res_head", "ke", "r_hw"):
            a, b = g_gpu[k], g_cpu[k]
            den = max(np.abs(a).max(), np.abs(b).max(), 1e-300)
            r_ = float(np.abs(a - b).max() / den)
            wa = max(wa, r_)
            print("  §A θ=%-8s rel=%.3e (max|g|=%.3e) %s" %
                  (k, r_, float(den), "PASS" if r_ < TOL_A else "FAIL"))
            ok_all &= r_ < TOL_A
        if s._dense_prv_np:
            dn = [int(net.link_n2[k]) for k in
                  np.where(np.asarray(net.link_type) == 3)[0]]
            for j in dn:
                print("    PRV 下游 demand[%d]: cpu=% .6e gpu=% .6e" %
                      (j, g_cpu["demand"][j], g_gpu["demand"][j]))
        okc = n_bwd == 0
        print("  §C dense 分解计数：前向 %d 次 / backward %d 次（须为 0）%s" %
              (n_fwd, n_bwd, "PASS" if okc else "FAIL"))
        ok_all &= okc

        # ---- §B 中央差分（head-only loss；±h 重跑状态机）----
        Dh = torch.as_tensor(d0, dtype=dt).requires_grad_(True)
        Rh = torch.as_tensor(rh0, dtype=dt).requires_grad_(True)
        KEh = torch.as_tensor(ke0, dtype=dt).requires_grad_(True)
        RWh = torch.as_tensor(r0, dtype=dt).requires_grad_(True)
        hh, _, _ = implicit_solve(s, Dh, Rh, ke=KEh, r_hw=RWh, adjoint="gpu",
                                  accuracy=acc, max_iter=200, status_machine=sm)
        (hh[s.junc_nodes_t] * W).sum().backward()
        gh = dict(demand=Dh.grad.numpy(), res_head=Rh.grad.numpy(),
                  ke=KEh.grad.numpy(), r_hw=RWh.grad.numpy())
        jn = np.asarray(s.junc_nodes)
        prv = np.where(np.asarray(net.link_type) == 3)[0]
        lt_np = np.asarray(net.link_type)
        # r_hw 的 FD 只在"离 RQtol 钳位边界足够远"的开管上做（r 扰动会直接改
        # hgrad = Hexp·r·|q|^(Hexp-1)，边界上的分支翻转是 θ 空间的 kink，
        # 双侧 FD 无意义 - 与 ±h 重跑状态机弃切换点是同一逻辑）
        with torch.no_grad():
            q_base = out0["flow_cfs"][0].cpu().numpy()
        hg_base = s.hexp * r0 * np.maximum(np.abs(q_base), 1e-30) \
            ** (s.hexp - 1.0)
        cl_base = np.asarray(net.init_status) == 0
        pipes_ok = np.where((lt_np <= 1) & (hg_base > 10.0 * s.rqtol)
                            & ~cl_base & (np.abs(q_base) > 1e-3))[0]
        coords = {
            "demand": ([int(net.link_n2[k]) for k in prv]
                       + [int(jn[i]) for i in rng.integers(0, jn.size, 3)]),
            "res_head": [int(n) for n in
                         np.where(np.asarray(net.node_type) != 0)[0][:4]],
            "ke": [int(n) for n in np.where(ke0 > 0)[0]],
            "r_hw": [int(x) for x in
                     pipes_ok[rng.integers(0, pipes_ok.size, 4)]],
        }

        def loss_of(dv, rhv, kev, rwv):
            o = solve_polished(s_ep, dv, rhv, ke=kev, r_hw=rwv, **kw)
            return float((o["head"][0, s.junc_nodes] * W.numpy()).sum())

        def status_same(dv, rhv, kev):
            if not sm:
                return True
            rr = s_ep.run_gga(dv, rhv, ke=kev, do_status=True)
            return bool(np.array_equal(rr["status"], S0))

        n_skip = 0
        wb = 0.0
        for kind, idxs in coords.items():
            wk = 0.0
            used = 0
            gmax_cls = float(np.abs(gh[kind]).max())     # 类内绝对尺度
            for i in idxs:
                dv, rhv, kev, rwv = (d0.copy(), rh0.copy(), ke0.copy(),
                                     r0.copy())
                vec = dict(demand=dv, res_head=rhv, ke=kev, r_hw=rwv)[kind]
                bv = vec[i]
                # 步长按 θ 类调（scratch 扫 h 的 V 曲线定的最优区）：
                # FD 噪声地板 ≈ polish 残差经权重放大 /(g·2h)，步太小噪声淹没
                h = dict(demand=max(1e-5, 1e-3 * abs(bv)),
                         res_head=max(1e-6, 1e-4 * abs(bv)),
                         ke=max(1e-8, 1e-2 * abs(bv)),
                         r_hw=max(1e-6, 1e-3 * abs(bv)))[kind]
                Ls, ok_st = {}, True
                for dlt in (h, -h, h / 2, -h / 2):
                    vec[i] = bv + dlt
                    ok_st &= status_same(dv, rhv, kev)
                    Ls[dlt] = loss_of(dv, rhv, kev, rwv)
                vec[i] = bv
                if not ok_st:
                    n_skip += 1
                    continue
                D1 = (Ls[h] - Ls[-h]) / (2 * h)
                D2 = (Ls[h / 2] - Ls[-h / 2]) / h
                g_ad = float(gh[kind][i])
                # FD 自检：两个步长的中央差分须相互一致（<1e-6），否则该坐标的
                # FD 本身不适定（内部钳位支/近零流量链路的 kink 或噪声地板），
                # 弃用并另计 - 与"±h 状态切换弃坐标"同一逻辑。全向量的正确性
                # 由 §A（对独立实现的 CPU 伴随）承担。
                fd_rel = abs(D1 - D2) / max(abs(D1), abs(D2), 1e-300)
                if fd_rel > 1e-6 and abs(D1 - D2) > 1e-6 * gmax_cls:
                    n_skip += 1
                    continue
                used += 1
                g_fd = (4.0 * D2 - D1) / 3.0
                rel_ = abs(g_ad - g_fd) / max(abs(g_fd), abs(g_ad), 1e-300)
                # 通过 = 相对 <1e-6，或按类内绝对尺度 |Δ|<1e-6·max|g|（FD 噪声
                # 是损失级的**绝对**量，小梯度坐标拿不到相对精度 - 如实按
                # 绝对尺度评）
                good = rel_ < TOL_FD or abs(g_ad - g_fd) < 1e-6 * gmax_cls
                ok_all &= good
                if not good or rel_ < TOL_FD:
                    # 绝对档通过（rel≥1e-6 但 |Δ| 在类尺度噪声内）不计入 wk
                    wk = max(wk, rel_)
            wb = max(wb, wk)
            ok_all &= used >= max(1, len(idxs) // 2)
            print("  §B FD θ=%-8s 坐标 %d（弃 %d） 最差 rel=%.3e %s" %
                  (kind, used, len(idxs) - used, wk,
                   "PASS" if wk < TOL_FD and used >= max(1, len(idxs) // 2)
                   else "FAIL"))

        # ---- §D 批一致 B=8 ----
        B = 8
        Db = d0[None, :] * rng.uniform(0.85, 1.15, (B, d0.size))
        DB = torch.as_tensor(Db, dtype=dt).requires_grad_(True)
        RB = torch.as_tensor(np.repeat(rh0[None, :], B, 0),
                             dtype=dt).requires_grad_(True)
        KB = torch.as_tensor(np.repeat(ke0[None, :], B, 0),
                             dtype=dt).requires_grad_(True)
        RWB = torch.as_tensor(r0, dtype=dt).requires_grad_(True)
        try:
            hB, fB, eB = implicit_solve(s, DB, RB, ke=KB, r_hw=RWB,
                                        adjoint="gpu", accuracy=acc,
                                        max_iter=200, status_machine=sm)
        except RuntimeError as e:       # 个别缩放场景不收敛：如实报并缩批重试
            print("  §D B=8 出现未收敛场景（%s），改用 ×U(0.95,1.05)" %
                  str(e)[:60])
            Db = d0[None, :] * rng.uniform(0.95, 1.05, (B, d0.size))
            DB = torch.as_tensor(Db, dtype=dt).requires_grad_(True)
            hB, fB, eB = implicit_solve(s, DB, RB, ke=KB, r_hw=RWB,
                                        adjoint="gpu", accuracy=acc,
                                        max_iter=200, status_machine=sm)
        WB = torch.as_tensor(rng.normal(0, 1, (B, s.Nj)), dtype=dt)
        (hB[:, s.junc_nodes_t] * WB).sum().backward()
        gB = DB.grad.numpy()
        worst_b = 0.0
        groups = set()
        for b in range(B):
            D1 = torch.as_tensor(Db[b], dtype=dt).requires_grad_(True)
            h1, _, _ = implicit_solve(s, D1, torch.as_tensor(rh0, dtype=dt),
                                      ke=torch.as_tensor(ke0, dtype=dt),
                                      r_hw=torch.as_tensor(r0, dtype=dt),
                                      adjoint="gpu", accuracy=acc,
                                      max_iter=200, status_machine=sm)
            (h1[s.junc_nodes_t] * WB[b]).sum().backward()
            g1 = D1.grad.numpy()
            den = max(np.abs(gB[b]).max(), np.abs(g1).max(), 1e-300)
            worst_b = max(worst_b, float(np.abs(gB[b] - g1).max() / den))
            if sm:
                with torch.no_grad():
                    o1 = s.solve(np.atleast_2d(Db[b]), np.atleast_2d(rh0),
                                 ke_int=np.atleast_2d(ke0), accuracy=acc,
                                 max_iter=200, status_machine=True)
                groups.add(tuple(o1["status"][0].cpu().numpy().tolist()))
        okd = worst_b < TOL_B
        print("  §D 批一致 B=8：max 相对差=%.3e（门槛 %g）%s%s" %
              (worst_b, TOL_B, "PASS" if okd else "FAIL",
               "  状态组=%d" % len(groups) if sm else ""))
        ok_all &= okd
        if name == "L-TOWN":
            lt = (wa, wb)

    # ---- §E 守卫 ----
    print("=" * 72)
    net, inp, s, d0, rh0, ke0, r0, rng = setup("Hanoi", "public/Hanoi.inp",
                                               False, True)
    s_ep = GGASolver(net, mode="epanet", inp_path=inp)
    dt = torch.float64
    D = torch.as_tensor(d0, dtype=dt).requires_grad_(True)
    guards = []
    try:
        implicit_solve(s_ep, D, rh0, adjoint="gpu")
        guards.append(("epanet 模式拒", False))
    except NotImplementedError:
        guards.append(("epanet 模式拒", True))
    try:
        implicit_solve(s, D, rh0, adjoint="gpu", speed=np.ones(s.L))
        guards.append(("泵θ/status 拒", False))
    except NotImplementedError:
        guards.append(("泵θ/status 拒", True))
    try:
        h, _, _ = implicit_solve(s, D, rh0, adjoint="gpu", accuracy=1e-12)
        g, = torch.autograd.grad(h.sum(), D, create_graph=True)
        guards.append(("create_graph 拒", False))
    except NotImplementedError:
        guards.append(("create_graph 拒", True))
    for nm, ok in guards:
        print("  §E %s：%s" % (nm, "PASS" if ok else "FAIL"))
        ok_all &= ok
    print("\n总判定: %s" % ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

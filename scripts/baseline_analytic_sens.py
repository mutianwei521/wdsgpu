# -*- coding: utf-8 -*-
"""baseline_analytic_sens.py - 经典解析灵敏度基线（WR 大修 R2-M2 / R2-M5，2026-10-23）。

三方对比（单帧 t=0 稳态；设备状态冻结在收敛态）：
  HG     HydroGrad 隐式伴随：发布版 ImplicitGGASolve（epanet 模式求解器，完整雅可比 J，
         ACTIVE PRV/PSV 用约束行 H[n2]-hset / hset-H[n1]）。
  NAIVE  经典解析灵敏度（Piller et al. 2017 一类，复用 GGA 分解）：在收敛态用 EPANET 自己
         组装的 GGA 矩阵 Â = Bᵀ diag(P) B + CBIG·e eᵀ（P=1/hgrad；ACTIVE PRV：P=0、下游节点对角
         +CBIG，hydcoeffs.c prvcoeff；PSV 同理在上游）解约化系统。伴随形式一次回代，直接形式每个
         参数一次回代，二者数学等价（同一矩阵）。无活动 PRV/PSV 时 Â 就是精确雅可比的 Schur 补，
         NAIVE 应与 HG 逐位接近（本脚本的自检）。
  FD     原生 EPANET 2.2 DLL 中心差分：EN_openH 一次，EN_initH(10)+EN_runH 取 t=0；
         ACCURACY=1e-8（API 下限）；正负扰动后任一链路 EN_STATUS 或 PRV/PSV 是否处于调节状态与基准不同，
         该坐标剔除并计数（状态跨越判据，回应 R2 次要意见 4）。
标量目标 L = Σ_j w_j·H_j（全部 junction，内部 ft），w = default_rng(2026).normal。
参数类：junction 需水 d（内部 cfs；FD 分母取 EN_DEMAND 实测施加值之差）、H-W 管道系数 C、
定水头节点水头（水库、水池；FD 分母取 EN_HEAD 实测值之差，内部 ft）。
需水与定水头取 EPANET 的 t=0 解；设备状态与设定取 EPS 驱动首帧（执行 t=0 控制并运行状态机），
冻结后用于精修前向、隐式伴随与经典法（10-23 修正：此前误用 INP 初始状态）。
输出：data/baseline_analytic_sens.json，data/baseline_analytic_sens_report.txt（追加）
运行：python -X utf8 scripts/baseline_analytic_sens.py --nets hanoi,city_d,l_town,d_town,ky10,bwsn1,richmond
"""
import argparse
import ctypes
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np                                             # noqa: E402
import scipy.sparse as sp                                      # noqa: E402
import torch                                                   # noqa: E402
from scipy.sparse.linalg import splu                           # noqa: E402

torch.set_default_dtype(torch.float64)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dgga import autodiff as ad                                # noqa: E402
from dgga import epanet_ref as er                              # noqa: E402
from dgga.autodiff import ImplicitGGASolve, solve_polished     # noqa: E402
from dgga.calib import dr_dC                                   # noqa: E402
from dgga.parse import parse_inp                               # noqa: E402
from dgga.eps import EpsDriver                                 # noqa: E402
from dgga.solver import CBIG                                   # noqa: E402

EN_STATUS, EN_TANKLEVEL = 11, 8
NETS = {
    "hanoi": "networks/public/Hanoi.inp",
    "city_d": "networks/realInpData/city_d.inp",
    "l_town": "networks/public/_cleaned/L-TOWN.inp",
    "d_town": "networks/public/D-Town.inp",
    "ky10": "networks/public/ky10.inp",
    "bwsn1": "networks/public/_cleaned/BWSN_Network_1.inp",
    "richmond": "networks/public/Richmond_standard.inp",
}
JSON = os.path.join(ROOT, "data", "baseline_analytic_sens.json")
RPT = os.path.join(ROOT, "data", "baseline_analytic_sens_report.txt")


class Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "a", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


class EP:
    """原生 EPANET 2.2 DLL：求解器只打开一次，每次 EN_initH(10)（冷启动）+ EN_runH 取 t=0。"""
    def __init__(self, inp):
        tmpd = tempfile.mkdtemp(prefix="bas_")
        e = self.e = er.Epanet(shutil.copy(os.path.join(ROOT, inp), os.path.join(tmpd, "net.inp")))
        e.set_option(er.EN_ACCURACY, 1e-8)          # API 下限（epanet.c:1241-1243）
        e.set_option(er.EN_TRIALS, 1000)
        lib, ph = self.lib, self.ph = e.lib, e._ph
        lib.EN_openH.argtypes = [ctypes.c_void_p]
        lib.EN_closeH.argtypes = [ctypes.c_void_p]
        lib.EN_initH.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.EN_runH.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
        e._check(lib.EN_openH(ph), "EN_openH")
        self.uh, self.uf = e._ucf_head, e._ucf_flow
        self.v, self.t = ctypes.c_double(), ctypes.c_long()
        self.nid = {s: i + 1 for i, s in enumerate(e.node_ids())}
        self.lid = {s: i + 1 for i, s in enumerate(e.link_ids())}

    def solve(self):
        e1 = self.lib.EN_initH(self.ph, 10)
        e2 = self.lib.EN_runH(self.ph, ctypes.byref(self.t))
        if e1 > 100 or e2 > 100:
            raise RuntimeError(f"EPANET error {e1}/{e2}")
        st = ctypes.c_double()
        self.lib.EN_getstatistic(self.ph, er.EN_ITERATIONS, ctypes.byref(st))
        it = int(st.value)
        self.lib.EN_getstatistic(self.ph, er.EN_RELATIVEERROR, ctypes.byref(st))
        return it, float(st.value), int(e2)

    def nget(self, idx, prop):
        out = np.empty(len(idx))
        for i, n in enumerate(idx):
            self.lib.EN_getnodevalue(self.ph, int(n), prop, ctypes.byref(self.v))
            out[i] = self.v.value
        return out

    def lget(self, idx, prop):
        out = np.empty(len(idx))
        for i, k in enumerate(idx):
            self.lib.EN_getlinkvalue(self.ph, int(k), prop, ctypes.byref(self.v))
            out[i] = self.v.value
        return out

    def nset(self, n, prop, val):
        self.e._check(self.lib.EN_setnodevalue(self.ph, int(n), prop, float(val)), "EN_setnodevalue")

    def lset(self, k, prop, val):
        self.e._check(self.lib.EN_setlinkvalue(self.ph, int(k), prop, float(val)), "EN_setlinkvalue")

    def close(self):
        self.lib.EN_closeH(self.ph)
        self.e.close()


def tmed(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def internal_fd(se, d, rh, r0, w, junc, fixed, coords, S0, K0, q0w, e0w, pst):
    """内部差分：每次扰动先从基准状态出发重跑完整状态机（run_gga do_status=True），状态与设定都与
    基准 S0/K0 相同才采信（状态跨越判据）；再在冻结的 S0/K0 下精修（残差 ~1e-12）取目标值。
    步长：需水 1e-3（相对）、C 1e-4（相对，r ∝ C^-Hexp）、定水头 1e-3 ft；Richardson g = (9g(h)-g(3h))/8。"""
    H1 = dict(d=1e-3, C=1e-4, H=1e-3)

    def at(cls, i, dl):
        dd, rr, hh = d.copy(), r0.copy(), rh.copy()
        if cls == "d":
            n = junc[i]
            dd[n] = d[n] * (1 + dl)
            x = dd[n]
        elif cls == "C":
            rr[i] = r0[i] * (1 + dl) ** (-se.hexp)
            x = se.kc_np[i] * (1 + dl)
        else:
            n = fixed[i]
            hh[n] = rh[n] + dl
            x = hh[n]
        with ad._override_r_hw(se, torch.as_tensor(rr, dtype=se.dtype)):
            g = se.run_gga(dd, hh, q0=q0w.copy(), e0=e0w.copy(), status0=S0.copy(), setting0=K0.copy(),
                           do_status=True)
        same = bool(np.array_equal(np.asarray(g["status"]), S0)
                    and np.array_equal(np.asarray(g["setting"]), K0))
        sl = solve_polished(se, dd, hh, r_hw=rr, accuracy=1e-12, max_iter=200, polish_steps=pst,
                            status=S0, speed=K0)
        return float(w @ sl["head"][0, junc]), x, same
    rows, excl = [], 0
    for cls, i in coords:
        g, ok = [], True
        for h in (H1[cls], 3 * H1[cls]):
            (Lp, xp, sp), (Lm, xm, sm) = at(cls, i, +h), at(cls, i, -h)
            ok = ok and sp and sm
            g.append((Lp - Lm) / (xp - xm))
        if not ok:
            excl += 1
            continue
        gfd = (9 * g[0] - g[1]) / 8
        rows.append(dict(cls=cls, i=i, fd=gfd, fd_unc_rel=abs(g[1] - g[0]) / 8 / max(abs(gfd), 1e-300)))
    return rows, excl


def summarize_fd(out, rows, prefix, g_hg, g_nv):
    for cls in ("d", "C", "H"):
        rs = [r for r in rows if r["cls"] == cls]
        if not rs:
            continue
        fl = 1e-8 * max(abs(r["fd"]) for r in rs)
        e_hg = [abs(float(g_hg[cls][r["i"]]) - r["fd"]) / max(abs(r["fd"]), fl) for r in rs]
        e_nv = [abs(float(g_nv[cls][r["i"]]) - r["fd"]) / max(abs(r["fd"]), fl) for r in rs]
        out[f"{prefix}_{cls}"] = dict(n=len(rs), fd_unc_med=float(np.median([r["fd_unc_rel"] for r in rs])),
                                      hg_max=float(max(e_hg)), hg_med=float(np.median(e_hg)),
                                      naive_max=float(max(e_nv)), naive_med=float(np.median(e_nv)))


def run_net(key, n_pick=4, seed=7):
    inp = NETS[key]
    net = parse_inp(os.path.join(ROOT, inp))
    drv = EpsDriver(net, inp_path=os.path.join(ROOT, inp))      # 与 52 网 EPS 验证同一驱动
    se = drv.solver
    drv._inithyd()
    drv._demands()
    drv._controls()
    r0g = se.run_gga(drv.d, drv.H, q0=drv.q, e0=drv.e, status0=drv.S, setting0=drv.K, do_status=True)
    S0 = np.asarray(r0g["status"], dtype=np.int8).copy()
    K0 = np.asarray(r0g["setting"], dtype=np.float64).copy()
    q0w, e0w = np.asarray(r0g["flow"]).copy(), np.asarray(r0g["emitter"]).copy()
    n_st_diff = int((S0 != np.asarray(se.init_status_int)).sum())
    N, L, Nj = net.N, se.L, se.Nj
    junc, fixed = np.asarray(se.junc_nodes), np.asarray(se.fixed_nodes)
    out = dict(net=key, inp=inp, N=int(N), L=int(L), Nj=int(Nj), n_fixed=int(len(fixed)))
    if np.any(np.asarray(net.node_ke) > 0):
        out["skipped"] = "含 emitter（EN_DEMAND 含 emitter 出流，本脚本不处理）"
        print(f"[{key}] 跳过：{out['skipped']}")
        return out
    ep = EP(inp)
    nmap = np.array([ep.nid[s] for s in net.node_id])
    lmap = np.array([ep.lid[s] for s in net.link_id])
    it, rel, warn = ep.solve()
    d = np.zeros(N)
    d[junc] = ep.nget(nmap[junc], er.EN_DEMAND) / ep.uf
    rh = np.zeros(N)
    rh[fixed] = ep.nget(nmap[fixed], er.EN_HEAD) / ep.uh
    H_ep = ep.nget(nmap[junc], er.EN_HEAD) / ep.uh
    PST = 3
    sol = solve_polished(se, d, rh, accuracy=1e-12, max_iter=200, polish_steps=PST, status=S0, speed=K0)
    if float(np.max(sol["resid_inf"])) > 1e-10:                 # 起点偏远时加精修步数
        PST = 8
        sol = solve_polished(se, d, rh, accuracy=1e-12, max_iter=200, polish_steps=PST, status=S0, speed=K0)
    dH = float(np.abs(sol["head"][0, junc] - H_ep).max())
    out.update(eps_t0_converged=bool(r0g["converged"]), n_status_diff_vs_inp=n_st_diff, polish_steps=PST,
               max_abs_d_eps_vs_epanet=float(np.abs(drv.d[junc] - d[junc]).max()),
               max_abs_H_eps_vs_epanet=float(np.abs(np.asarray(r0g["head"])[junc] - H_ep).max()))
    out.update(ep_iters=it, ep_relerr=rel, ep_warn=warn, fwd_max_abs_dH_ft=dH,
               resid_inf=float(np.max(sol["resid_inf"])))
    # ---------------- HG：发布版隐式伴随 ----------------
    w = np.random.default_rng(2026).normal(size=Nj)
    d_t = torch.tensor(d, requires_grad=True)
    rh_t = torch.tensor(rh, requires_grad=True)
    r_t = torch.tensor(np.asarray(sol["r_hw"], float), requires_grad=True)
    head, _, _ = ImplicitGGASolve.apply(d_t, rh_t, torch.zeros(N), r_t, se, 1e-12, 200, PST, K0, None, None, S0)
    (torch.tensor(w) * head[torch.as_tensor(junc)]).sum().backward()
    dC = dr_dC(se, r_hw=sol["r_hw"])
    g_hg = dict(d=d_t.grad.numpy()[junc].copy(), C=(r_t.grad.numpy() * dC).copy(),
                H=rh_t.grad.numpy()[fixed].copy())
    # ---------------- NAIVE：复用 EPANET 的 GGA 罚项矩阵 ----------------
    q = np.asarray(sol["q"])[0]
    _, hg, dr, _ = ad._link_coeffs_np(se, q, sol["r_hw"], speed=sol["speed"], status=sol["status"],
                                      h0p=sol["h0p"], rp=sol["rp"])
    aprv = apsv = np.zeros(L, dtype=bool)
    if bool((se.is_valve_np & ~se.is_tcv_np).any()):
        aprv, apsv, _ = ad._valve_act_masks(se, sol["speed"], sol["status"])
    c = ad._adj_cache(se)
    P = np.zeros(L)
    nz = hg != 0
    P[nz] = 1.0 / hg[nz]
    P[aprv | apsv] = 0.0
    rows = np.r_[np.where(c["m2"])[0], np.where(c["m1"])[0]]
    cols = np.r_[c["j2"][c["m2"]], c["j1"][c["m1"]]]
    vals = np.r_[np.ones(int(c["m2"].sum())), -np.ones(int(c["m1"].sum()))]
    Bm = sp.csr_matrix((vals, (rows, cols)), shape=(L, Nj))
    pen = np.zeros(Nj)
    np.add.at(pen, c["j2"][aprv & c["m2"]], CBIG)
    np.add.at(pen, c["j1"][apsv & c["m1"]], CBIG)
    A_hat = (Bm.T @ sp.diags(P) @ Bm + sp.diags(pen)).tocsc()
    lu = splu(A_hat)
    lam_m = lu.solve(-w)
    lam_l = P * (-(Bm @ lam_m))
    grh = np.zeros(N)
    np.add.at(grh, se.n1_np[c["f1"]], lam_l[c["f1"]])
    np.add.at(grh, se.n2_np[c["f2"]], -lam_l[c["f2"]])
    g_nv = dict(d=lam_m, C=-lam_l * dr * dC, H=grh[fixed])
    out.update(n_active_prv=int(aprv.sum()), n_active_psv=int(apsv.sum()),
               n_prv=int((se.lt_np == ad._PRV).sum()), n_psv=int((se.lt_np == ad._PSV).sum()))
    for k in ("d", "C", "H"):
        den = np.abs(g_hg[k]).max() if g_hg[k].size else 0.0
        out[f"naive_vs_hg_maxrel_{k}"] = float(np.abs(g_nv[k] - g_hg[k]).max() / den) if den > 0 else 0.0
    # ---------------- FD：原生 EPANET 中心差分 ----------------
    rng = np.random.default_rng(seed)
    is_prv, is_psv = np.where(se.lt_np == ad._PRV)[0], np.where(se.lt_np == ad._PSV)[0]
    sp_ = np.asarray(sol["speed"], float)                       # PRV:n2 / PSV:n1 的设定水头（内部 ft）
    hs_prv = se.elev_np[se.n2_np[is_prv]] + sp_[is_prv]
    hs_psv = se.elev_np[se.n1_np[is_psv]] + sp_[is_psv]

    def lval():
        return float(w @ (ep.nget(nmap[junc], er.EN_HEAD) / ep.uh))

    def state():
        st = ep.lget(lmap, EN_STATUS).astype(int)
        Hall = ep.nget(nmap, er.EN_HEAD) / ep.uh
        a1 = np.abs(Hall[se.n2_np[is_prv]] - hs_prv) < 1e-6 * np.maximum(1.0, np.abs(hs_prv))
        a2 = np.abs(Hall[se.n1_np[is_psv]] - hs_psv) < 1e-6 * np.maximum(1.0, np.abs(hs_psv))
        return np.r_[st, a1.astype(int), a2.astype(int)]
    ep.solve()
    s0 = state()

    def pick(cands, diff):
        cands = np.asarray(cands)
        if cands.size == 0:
            return cands
        top = cands[np.argsort(-diff)[:n_pick]] if diff.max() > 1e-6 else np.array([], int)
        rest = np.setdiff1d(cands, top)
        rnd = rng.choice(rest, min(n_pick, rest.size), replace=False) if rest.size else np.array([], int)
        return np.r_[top, rnd].astype(int)

    dj = np.where(np.abs(d[junc]) > 0)[0]
    sc_d = np.abs(g_hg["d"]).max() + 1e-300
    pipes = np.where(np.isin(se.lt_np, (0, 1)) & (se.kc_np > 0) & ~se.closed_np & (np.abs(q) > 1e-4))[0]
    sc_C = np.abs(g_hg["C"]).max() + 1e-300
    coords = ([("d", int(i)) for i in pick(dj, np.abs(g_nv["d"] - g_hg["d"])[dj] / sc_d)]
              + [("C", int(k)) for k in pick(pipes, np.abs(g_nv["C"] - g_hg["C"])[pipes] / sc_C)]
              + [("H", int(i)) for i in range(min(len(fixed), 6))])
    r0 = np.asarray(sol["r_hw"], float)
    irows, iexcl = internal_fd(se, d, rh, r0, w, junc, fixed, coords, S0, K0, q0w, e0w, PST)
    summarize_fd(out, irows, "ifd", g_hg, g_nv)
    out["ifd_rows"], out["ifd_excluded_status_change"] = irows, iexcl
    ext_ok = (dH <= 1e-3) and out["resid_inf"] <= 1e-8   # EPANET 停在极限环（返回码 1）但与精修解一致时也做，
    out["ep_converged"] = bool(warn == 0)                  # 差分自身误差由两步长之差单独报告
    out["external_fd_done"] = bool(ext_ok)
    rows_fd, n_excl = [], 0
    H1 = dict(d=1e-2, C=1e-3, H=0.01)      # 第二步长 = 3×第一步长；Richardson：g = (9 g(h) − g(3h)) / 8

    def fd_pair(setv, getx, h):
        vals_, xs, ss = [], [], []
        for sgn in (+1, -1):
            setv(sgn * h)
            ep.solve()
            vals_.append(lval())
            xs.append(getx())
            ss.append(state())
        setv(0.0)
        return (vals_[0] - vals_[1]) / (xs[0] - xs[1]), all(np.array_equal(x, s0) for x in ss)
    for cls, i in (coords if ext_ok else []):
        if cls == "d":
            n = nmap[junc[i]]
            b0 = ep.nget([n], er.EN_BASEDEMAND)[0]
            if b0 == 0:
                continue
            setv = lambda dl, n=n, b0=b0: ep.nset(n, er.EN_BASEDEMAND, b0 * (1 + dl))      # noqa: E731
            getx = lambda n=n: ep.nget([n], er.EN_DEMAND)[0] / ep.uf                         # noqa: E731
        elif cls == "C":
            k = lmap[i]
            c0 = ep.lget([k], er.EN_ROUGHNESS)[0]
            setv = lambda dl, k=k, c0=c0: ep.lset(k, er.EN_ROUGHNESS, c0 * (1 + dl))         # noqa: E731
            getx = lambda k=k: ep.lget([k], er.EN_ROUGHNESS)[0]                              # noqa: E731
        else:
            nn = fixed[i]
            n = nmap[nn]
            prop = er.EN_ELEVATION if int(net.node_type[nn]) == 1 else EN_TANKLEVEL
            b0 = ep.nget([n], prop)[0]
            setv = lambda dl, n=n, prop=prop, b0=b0: ep.nset(n, prop, b0 + dl)               # noqa: E731
            getx = lambda n=n: ep.nget([n], er.EN_HEAD)[0] / ep.uh                           # noqa: E731
        g1, ok1 = fd_pair(setv, getx, H1[cls])
        g2, ok2 = fd_pair(setv, getx, 3 * H1[cls])
        if not (ok1 and ok2):
            n_excl += 1
            continue
        g_fd = (9 * g1 - g2) / 8
        unc = abs(g2 - g1) / 8 / max(abs(g_fd), 1e-300)
        rows_fd.append(dict(cls=cls, i=i, fd=g_fd, fd_unc_rel=unc, g_h=g1, g_3h=g2,
                            hg=float(g_hg[cls][i]), naive=float(g_nv[cls][i])))
    ep.solve()
    summarize_fd(out, rows_fd, "fd", g_hg, g_nv)
    out["fd_rows"], out["fd_excluded_status_change"] = rows_fd, n_excl
    # ---------------- 成本：直接法 n 次回代 vs 伴随 1 次（同一矩阵），以及 HG 的完整 J ----------------
    n_par = int(len(pipes))
    rng2 = np.random.default_rng(1)
    t_fact = tmed(lambda: splu(A_hat))
    t_adj = tmed(lambda: lu.solve(rng2.standard_normal(Nj)))
    Rn = rng2.standard_normal((Nj, n_par))
    t_dir = tmed(lambda: lu.solve(Rn))
    Rm = rng2.standard_normal((Nj, Nj))
    t_jac_adj = tmed(lambda: lu.solve(Rm), reps=1)
    J, _ = ad._build_J(se, hg, np.zeros(Nj), np.zeros(Nj, bool), aprv if aprv.any() else None,
                       apsv if apsv.any() else None)
    t_fullJ = tmed(lambda: splu(J).solve(rng2.standard_normal(J.shape[0])))
    out["cost_cpu_s"] = dict(n_params_pipes=n_par, factor_reduced=t_fact, adjoint_1rhs=t_adj,
                             direct_n_rhs=t_dir, jacobian_all_heads_m_rhs=t_jac_adj, m_heads=int(Nj),
                             hg_full_J_factor_plus_solve=t_fullJ)
    ep.close()
    f = out
    print(f"[{key}] EPS 首帧状态与 INP 初始状态不同 {f['n_status_diff_vs_inp']} 条，精修 {f['polish_steps']} 步，"
          f"EPS 首帧与 EPANET 水头差 {f['max_abs_H_eps_vs_epanet']:.1e} ft")
    print(f"[{key}] N={N} L={L} | EPANET iters={it} relerr={rel:.1e} 返回码={warn} | 前向 max|ΔH|={dH:.2e} ft | "
          f"活动 PRV {f['n_active_prv']}/{f['n_prv']} PSV {f['n_active_psv']}/{f['n_psv']}")
    print(f"      NAIVE vs HG 最大相对差：需水 {f['naive_vs_hg_maxrel_d']:.1e}  粗糙系数 {f['naive_vs_hg_maxrel_C']:.1e}  "
          f"定水头 {f['naive_vs_hg_maxrel_H']:.1e}")
    if not f["external_fd_done"]:
        print("      外部差分未做：EPANET 前向与 HydroGrad 不一致（>1e-3 ft，设备配置不同或 EPANET 远未收敛）或 HydroGrad 未收敛（见 JSON）")
    if f["resid_inf"] > 1e-8:
        print(f"      注意：HydroGrad 前向残差 {f['resid_inf']:.1e}，未收敛，本网结果不可用")
    for cls, nm in (("d", "需水"), ("C", "粗糙系数"), ("H", "定水头")):
        if f"ifd_{cls}" in f:
            v = f[f"ifd_{cls}"]
            print(f"      对内部差分（{nm}，{v['n']} 个坐标）：HG 最大 {v['hg_max']:.1e} 中位 {v['hg_med']:.1e} | "
                  f"NAIVE 最大 {v['naive_max']:.1e} 中位 {v['naive_med']:.1e} | 差分自身误差中位 {v['fd_unc_med']:.1e}")
        if f"fd_{cls}" in f:
            v = f[f"fd_{cls}"]
            print(f"      对 EPANET 差分（{nm}，{v['n']} 个坐标）：HG 最大 {v['hg_max']:.1e} 中位 {v['hg_med']:.1e} | "
                  f"NAIVE 最大 {v['naive_max']:.1e} 中位 {v['naive_med']:.1e} | 差分自身误差中位 {v['fd_unc_med']:.1e}")
    cc = f["cost_cpu_s"]
    print(f"      状态跨越剔除：外部 {n_excl} 个、内部 {iexcl} 个 | 成本（CPU，s）：约化矩阵分解 {cc['factor_reduced']:.2e}，伴随 1 次回代 {cc['adjoint_1rhs']:.2e}，"
          f"直接法 {n_par} 次回代 {cc['direct_n_rhs']:.2e}，全部 {Nj} 个水头的雅可比 {cc['jacobian_all_heads_m_rhs']:.2e}，"
          f"HG 完整 J 分解+求解 {cc['hg_full_J_factor_plus_solve']:.2e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="hanoi,city_d,l_town,d_town,ky10,bwsn1,richmond")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout = Tee(RPT)
    print("=" * 110)
    print(f"baseline_analytic_sens  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    allr = json.load(open(JSON, encoding="utf-8")) if os.path.exists(JSON) else {}
    for key in a.nets.split(","):
        try:
            allr[key] = run_net(key)
        except Exception as e:                                          # 如实记录失败
            allr[key] = dict(net=key, error=f"{type(e).__name__}: {str(e)[:300]}")
            print(f"[{key}] FAILED: {allr[key]['error']}", flush=True)
        json.dump(allr, open(JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(f"saved -> {JSON}")


if __name__ == "__main__":
    main()

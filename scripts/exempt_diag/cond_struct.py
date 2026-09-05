# -*- coding: utf-8 -*-
"""cond_struct.py - 条件数 / Wilkinson 前向界 / 结构特征诊断（模块四任务 2b-2c）。

三件事，全部实测：
 (b1) 截获 EPANET 通路 linsolve 的系数矩阵 A（对称正定 Schur 补，行空间），
      估计 kappa。A 由 (Aii, Aij) + 符号结构 (XLNZ/NZSUB/LNZ) 精确重建：
      对称阵有 kappa2 <= kappa1，故用 Higham-Tisseur 的 onenormest（对
      A 与 splu 给的 A^-1）算 kappa1 作为 kappa2 的**上界**；另用 eigsh
      直接取 lmax/lmin 得 kappa2 本身（失败则只报上界，如实注明）。
      分两个口径：全阵；去掉被 1/CBIG=1e-8 关闭链路解耦的行（与
      data/conditioning_report.txt 同一约定）。
 (b2) Wilkinson/Higham Cholesky 前向误差界：
      ||dH||_2/||H||_2 <~ gamma_{3n} * kappa2(A) / (1 - gamma_{3n} kappa2)，
      gamma_k = k*u/(1-k*u)，u = 2^-53。乘 ||H||_2 得到 ft 量纲的界，
      与实测偏差比对。
 (b3) 未流动盲端 / 振荡诊断：逐帧统计"所有相邻链路都关闭"的孤立 junction、
      零流量链路；并用官方 DLL 在 ACCURACY=1e-8 / TRIALS=1000 下重跑，
      看参考引擎自己收不收敛。
 (c)  四个"豁免"网与对照网的共同结构特征表。

用法: python scripts/exempt_diag/cond_struct.py [stem ...]
"""

import ctypes
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.parse import Net                       # noqa: E402
from dgga.solver import GGASolver                # noqa: E402
from dgga.eps import EpsDriver                   # noqa: E402
from dgga import epanet_ref as ER                # noqa: E402
from align import resolve_inp                    # noqa: E402

REF_DIR = os.path.join(ROOT, "data", "reference")
OUT_DIR = os.path.join(ROOT, "data", "exempt")
U = 2.0 ** -53                    # 双精度单位舍入
DECOUPLED = 1e-6                  # 对角 <= 该值 => 1/CBIG 解耦行

EXEMPT = ["pub_net3", "pub_bwsn_network_1", "pub_bwsn_network_2", "pub_net6"]
CONTROL = ["pub_c_town_batadal", "pub_d_town", "pub_richmond_standard",
           "ky5", "pub_net2", "pub_l_town"]


# --------------------------------------------------------------------------
def capture_A(stem, frame=0):
    """自主推进到第 frame 帧，截获该帧最后一次 linsolve 的 (Aii, Aij)，
    重建稀疏 A（njuncs x njuncs，行空间，对称），并返回该帧的 H。"""
    net = Net.load(REF_DIR, stem)
    inp = resolve_inp(stem)
    drv = EpsDriver(net, inp_path=inp if os.path.isfile(inp) else None)
    s = drv.solver
    drv._inithyd()
    cap = []
    sm = s.sm
    orig = sm.linsolve

    def spy(Aii, Aij, B):
        cap.append((list(Aii), list(Aij)))
        return orig(Aii, Aij, B)

    sm.linsolve = spy
    try:
        for f in range(frame + 1):
            drv._demands()
            drv._controls()
            r = s.run_gga(drv.d, drv.H, q0=drv.q, e0=drv.e,
                          status0=drv.S, setting0=drv.K, do_status=True)
            drv.q, drv.e, drv.S = r["flow"], r["emitter"], r["status"]
            drv.K, drv.H = r["setting"], r["head"]
            drv.fixed_dem = r["fixed_demand"]
            drv.node_dem = np.where(s.is_fixed_node, drv.fixed_dem, drv.d + drv.e)
            if f < frame:
                drv._nexthyd(float(r["relerr"]))
    finally:
        sm.linsolve = orig
    Aii, Aij = cap[-1]
    n = sm.njuncs
    rows, cols, vals = [], [], []
    for i in range(1, n + 1):
        rows.append(i - 1)
        cols.append(i - 1)
        vals.append(Aii[i])
    for i in range(1, n + 1):
        for k in range(sm.XLNZ[i], sm.XLNZ[i + 1]):
            j = sm.NZSUB[k]          # 行号 > i
            v = Aij[sm.LNZ[k]]
            if v != 0.0:
                rows += [j - 1, i - 1]
                cols += [i - 1, j - 1]
                vals += [v, v]
    A = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    return net, s, drv, A, np.asarray(r["head"]), int(r["iters"]), len(cap)


def kappa1_upper(A):
    """kappa1 = ||A||_1 * ||A^-1||_1（onenormest + splu）。对称阵下 >= kappa2。"""
    n1 = float(abs(A).sum(axis=0).max())
    lu = spla.splu(A.tocsc())
    op = spla.LinearOperator((A.shape[0],) * 2, matvec=lu.solve,
                             rmatvec=lambda x: lu.solve(x, "T"),
                             matmat=lambda X: lu.solve(X),
                             dtype=np.float64)
    ninv = float(spla.onenormest(op))
    return n1, ninv, n1 * ninv


def kappa2_exact(A, tol=1e-8, maxiter=20000):
    """kappa2 = lmax/lmin（SPD）。小阵走 dense SVD，大阵走 eigsh。"""
    n = A.shape[0]
    if n <= 1200:
        s = np.linalg.svd(A.toarray(), compute_uv=False)
        return float(s[0]), float(s[-1]), float(s[0] / s[-1]), "dense-svd"
    try:
        lmax = float(spla.eigsh(A, k=1, which="LA", return_eigenvectors=False,
                                tol=tol, maxiter=maxiter)[0])
        lu = spla.splu(A.tocsc())
        OP = spla.LinearOperator((n, n), matvec=lu.solve, dtype=np.float64)
        lmin = float(spla.eigsh(A, k=1, sigma=0.0, which="LM", OPinv=OP,
                                return_eigenvectors=False, tol=tol,
                                maxiter=maxiter)[0])
        return lmax, lmin, abs(lmax / lmin), "eigsh(shift-invert)"
    except Exception as e:                                    # noqa: BLE001
        return float("nan"), float("nan"), float("nan"), f"failed: {type(e).__name__}"


def wilkinson_bound(n, kappa2, H_norm):
    """Higham (ASNA 2ed, Thm 10.4) Cholesky 前向界的常用形式。"""
    g = 3 * n * U / (1 - 3 * n * U) if 3 * n * U < 1 else float("inf")
    rel = g * kappa2 / (1 - g * kappa2) if g * kappa2 < 1 else float("inf")
    return g, rel, rel * H_norm


def deadend_stats(net, drv_S, s):
    """盲端 / 孤立口袋统计（当帧状态下）。"""
    n1 = np.asarray(net.link_n1)
    n2 = np.asarray(net.link_n2)
    nt = np.asarray(net.node_type)
    N = net.N
    deg = np.zeros(N, dtype=np.int64)
    np.add.at(deg, n1, 1)
    np.add.at(deg, n2, 1)
    open_m = np.asarray(drv_S) > s.ST_CLOSED
    deg_open = np.zeros(N, dtype=np.int64)
    np.add.at(deg_open, n1[open_m], 1)
    np.add.at(deg_open, n2[open_m], 1)
    junc = nt == 0
    return {"junctions": int(junc.sum()),
            "deadend_deg1": int(((deg == 1) & junc).sum()),
            "isolated_all_closed": int(((deg_open == 0) & junc).sum()),
            "n_links_closed": int((~open_m).sum()),
            "frac_links_closed": float((~open_m).mean())}


def ref_convergence(stem, net):
    """参考解自身的收敛状况：逐帧 relerr 与 INP 的 ACCURACY 比较。"""
    ref = np.load(os.path.join(REF_DIR, f"{stem}_ref.npz"))
    acc = float(net.meta.get("accuracy", np.nan))
    trials = int(net.meta.get("trials", 0))
    re = np.asarray(ref["relerr"], dtype=float)
    it = np.asarray(ref["iterations"], dtype=int)
    return {"accuracy": acc, "trials": trials, "T": int(len(re)),
            "max_relerr": float(re.max()), "median_relerr": float(np.median(re)),
            "n_frames_relerr_gt_acc": int((re > acc).sum()),
            "max_iters": int(it.max()),
            "n_frames_iters_ge_trials": int((it >= trials).sum())}


def tight_probe(inp, acc=1e-8, trials=1000, max_frames=None):
    """用官方 DLL 在收紧精度下重跑：参考引擎自己收不收敛？"""
    with ER.Epanet(inp) as en:
        en.set_option(ER.EN_ACCURACY, acc)
        en.set_option(ER.EN_TRIALS, trials)
        lib, ph = en.lib, en._ph
        t = ctypes.c_long()
        tstep = ctypes.c_long()
        stat = ctypes.c_double()
        its, res, warn = [], [], 0
        if lib.EN_openH(ph) > 100:
            raise RuntimeError("EN_openH")
        try:
            lib.EN_initH(ph, ER.EN_NOSAVE)
            f = 0
            while True:
                rc = lib.EN_runH(ph, ctypes.byref(t))
                if rc > 100:
                    raise RuntimeError(f"EN_runH rc={rc}")
                if rc > 0:
                    warn += 1
                lib.EN_getstatistic(ph, ER.EN_ITERATIONS, ctypes.byref(stat))
                its.append(int(stat.value))
                lib.EN_getstatistic(ph, ER.EN_RELATIVEERROR, ctypes.byref(stat))
                res.append(float(stat.value))
                f += 1
                if lib.EN_nextH(ph, ctypes.byref(tstep)) > 100:
                    raise RuntimeError("EN_nextH")
                if tstep.value == 0 or (max_frames and f >= max_frames):
                    break
        finally:
            lib.EN_closeH(ph)
    its = np.asarray(its)
    res = np.asarray(res)
    return {"acc": acc, "trials": trials, "T": int(len(its)),
            "max_iters": int(its.max()), "n_hit_trials": int((its >= trials).sum()),
            "max_relerr": float(res.max()), "median_relerr": float(np.median(res)),
            "n_frames_not_converged": int((res > acc).sum()),
            "n_runH_warnings": int(warn)}


def main(stems):
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "python": sys.executable, "u": U, "rows": []}
    for stem in stems:
        print(f"\n=== {stem} ===", flush=True)
        t0 = time.perf_counter()
        rec = {"stem": stem}
        try:
            net, s, drv, A, H, iters, n_lin = capture_A(stem, frame=0)
        except Exception as e:                                # noqa: BLE001
            print(f"  截获 A 失败: {type(e).__name__}: {e}", flush=True)
            out["rows"].append({"stem": stem, "error": f"{type(e).__name__}: {e}"})
            continue
        n = A.shape[0]
        d = A.diagonal()
        keep = d > DECOUPLED
        rec["n_juncs"] = int(n)
        rec["nnz"] = int(A.nnz)
        rec["diag_min"] = float(d.min())
        rec["diag_max"] = float(d.max())
        rec["n_decoupled_rows"] = int((~keep).sum())
        rec["gga_iters_frame0"] = int(iters)
        print(f"  Nj={n} nnz={A.nnz} 对角 [{d.min():.3e}, {d.max():.3e}] "
              f"解耦行={int((~keep).sum())} 帧0迭代={iters}", flush=True)

        for tag, M in (("full", A),
                       ("coupled", A[keep][:, keep] if (~keep).any() else A)):
            n1, ninv, k1 = kappa1_upper(M.tocsc())
            lmax, lmin, k2, how = kappa2_exact(M.tocsc())
            rec[f"kappa1_{tag}"] = k1
            rec[f"kappa2_{tag}"] = k2
            rec[f"kappa2_how_{tag}"] = how
            rec[f"lmax_{tag}"] = lmax
            rec[f"lmin_{tag}"] = lmin
            rec[f"n_{tag}"] = int(M.shape[0])
            kk = k2 if np.isfinite(k2) else k1
            g, rel, absft = wilkinson_bound(M.shape[0], kk,
                                            float(np.linalg.norm(H)))
            rec[f"wilk_gamma_{tag}"] = g
            rec[f"wilk_rel_{tag}"] = rel
            rec[f"wilk_ft_{tag}"] = absft
            print(f"  [{tag:8s}] n={M.shape[0]:6d} kappa1<={k1:.3e}  "
                  f"kappa2={k2:.3e} ({how})  Wilkinson 界: 相对 {rel:.3e} "
                  f"=> {absft:.3e} ft", flush=True)

        rec["struct"] = deadend_stats(net, drv.S, s)
        rec["ref_conv"] = ref_convergence(stem, net)
        print(f"  结构: {rec['struct']}", flush=True)
        print(f"  参考解收敛: {rec['ref_conv']}", flush=True)
        rec["sec"] = time.perf_counter() - t0
        out["rows"].append(rec)

    p = os.path.join(OUT_DIR, "cond_struct.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n写出:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else EXEMPT + CONTROL))

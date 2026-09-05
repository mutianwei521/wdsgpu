# -*- coding: utf-8 -*-
"""审阅项 1：NW_Model 解得对吗 - 独立与 EPANET DLL 对拍 + 条件数 + 物理残差。

**不复用 scripts/p5_bignet/nw_correct.py**，也不用 dgga.epanet_ref：
DLL 走本目录 aud_lib.MyEpanet（自写 ctypes 绑定），另用 wntr 自己的
ENepanet 再对一遍（两条独立绑定互证）。

产出（每网一段）：
  · 三方 max|ΔH| / max|ΔQ| / 迭代数 / relerr
  · A 的条件数：全阵 / 去死支后；对角范围；lambda_min/max
  · **物理残差**（完全不信 dgga）：节点连续性 max|Σq - d|（cfs）、
    管段能量 max|h_loss(q) - (H1-H2)|（ft）
  · 收敛停滞探针：max_iter=K 扫，看 H 还动不动
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                        # noqa: E402
import aud_lib as AL                                      # noqa: E402
import torch                                              # noqa: E402
import scipy.sparse as sp                                 # noqa: E402
import scipy.sparse.linalg as spl                         # noqa: E402
from dgga.parse import parse_inp                          # noqa: E402
from dgga.solver import GGASolver                         # noqa: E402

torch.set_num_threads(os.cpu_count())
DT = torch.float64
NETS = sys.argv[1:] or ["NW_Model"]
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "networks", "EXAMPLE")
PATHS = {
    "NW_Model": BASE + "/epanet-example-networks/epanet-tests/large/NW_Model.inp",
    "ky4": BASE + "/asce-tf-wdst/ky4/ky4.inp",
    "KL": BASE + "/asce-tf-wdst/KL/KL.inp",
    "ky13": BASE + "/asce-tf-wdst/ky13/ky13.inp",
}

print("=" * 100)
print("审阅项 1  NW_Model 正确性 + 条件数   独立复核")
for k, v in AL.prov():
    print("   %-18s %s" % (k, v))
print("=" * 100)


def hw_headloss(net, q):
    """独立实现的 H-W 管段水头损失（内部单位 ft）；仅 PIPE(1)/CVPIPE(0)。
    h = r_hw*|q|^1.852*sgn(q) + km*q*|q|   （EPANET hydcoeffs.c pipecoeff）"""
    r = np.asarray(net.r_hw, dtype=np.float64)
    km = np.asarray(net.km_int, dtype=np.float64)
    qa = np.abs(q)
    sgn = np.where(q < 0, -1.0, 1.0)
    return (r * qa ** 1.852 + km * qa * qa) * sgn


def phys_resid(net, H, q, d):
    """物理残差：连续性（cfs）与能量（ft）。返回 (cont_max, cont_where,
    energy_max_pipe, energy_where)。只对 PIPE/CVPIPE 段算能量残差。"""
    N = net.N
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    inflow = np.zeros(N)
    np.add.at(inflow, n1, -q)
    np.add.at(inflow, n2, +q)
    isj = np.asarray(net.node_type, dtype=np.int64) == 0
    cont = inflow - d                      # junction 上应为 0
    cont[~isj] = 0.0
    lt = np.asarray(net.link_type, dtype=np.int64)
    pipe = (lt == 1) | (lt == 0)
    hl = hw_headloss(net, q)
    dh = H[n1] - H[n2]
    en = np.where(pipe, hl - dh, 0.0)
    return (float(np.abs(cont).max()), int(np.abs(cont).argmax()),
            float(np.abs(en).max()), int(np.abs(en).argmax()), pipe.sum())


def cond_report(A, net, q, tag):
    """A: [Nj,Nj] numpy。报对角范围、lambda 端点、cond2、去死支后的 cond2。"""
    Nj = A.shape[0]
    dg = np.diag(A)
    print("   [%s] Nj=%d  对角 min=%.4e max=%.4e  非零元 %d  对称性 max|A-A^T|=%.3e"
          % (tag, Nj, dg.min(), dg.max(), int((A != 0).sum()),
             float(np.abs(A - A.T).max())))
    S = sp.csr_matrix(A)

    def k2(M, lbl):
        n = M.shape[0]
        try:
            lmax = float(spl.eigsh(M, k=1, which="LA",
                                   return_eigenvectors=False, tol=1e-10)[0])
        except Exception as e:                    # noqa: BLE001
            print("      [%s] lmax 失败 %s" % (lbl, e))
            return
        try:
            lmin = float(spl.eigsh(M, k=1, sigma=0.0, which="LM",
                                   return_eigenvectors=False, tol=1e-10)[0])
        except Exception:                         # noqa: BLE001
            try:
                lu = spl.splu(sp.csc_matrix(M))
                op = spl.LinearOperator((n, n), matvec=lu.solve, dtype=np.float64)
                mu = float(spl.eigsh(op, k=1, which="LA",
                                     return_eigenvectors=False, tol=1e-10)[0])
                lmin = 1.0 / mu
            except Exception as e:                # noqa: BLE001
                print("      [%s] lmin 失败 %s" % (lbl, e))
                return
        print("      [%s] n=%d  lambda_min=%.6e  lambda_max=%.6e  cond2=%.4e"
              % (lbl, n, lmin, lmax, abs(lmax / lmin)))

    k2(S, "全阵")
    # 死支：所有相邻链路 |q|<1e-9 的 junction
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    live = np.zeros(net.N, dtype=bool)
    big = np.abs(q) >= 1e-9
    live[n1[big]] = True
    live[n2[big]] = True
    junc = np.where(np.asarray(net.node_type) == 0)[0]
    keep = live[junc]
    print("      死 junction 数 = %d / %d" % (int((~keep).sum()), Nj))
    if (~keep).any() and keep.sum() > 2:
        idx = np.where(keep)[0]
        k2(sp.csr_matrix(A[np.ix_(idx, idx)]), "去死支")
    return dg


for stem in NETS:
    path = PATHS[stem]
    print("\n" + "#" * 100)
    print("### %s   %s" % (stem, path))
    net = parse_inp(path)
    lt = np.asarray(net.link_type)
    hist = {int(t): int((lt == t).sum()) for t in np.unique(lt)}
    print("   N=%d  Nj=%d  L=%d  link_type=%s  水池=%d 水库=%d"
          % (net.N, int((np.asarray(net.node_type) == 0).sum()), net.L, hist,
             int((np.asarray(net.node_type) == 2).sum()),
             int((np.asarray(net.node_type) == 1).sum())))

    # ---------- 1) EPANET DLL（自写 ctypes 绑定） ----------
    t0 = time.perf_counter()
    en = AL.MyEpanet(path)
    fu = en.flowunits()
    uh, uq = AL.ucf_head_flow(fu)
    dll = en.run_first_step()
    ids_dll = en.node_ids()
    en.close()
    H_dll = dll["head_user"] / uh
    Q_dll = dll["flow_user"] / uq
    print("   [DLL 自写绑定] flowunits=%d  ucf_head=%.6f ucf_flow=%.4f  "
          "iters=%d relerr=%.4e  t=%ds  用时 %.1fs"
          % (fu, uh, uq, dll["iters"], dll["relerr"], dll["t"],
             time.perf_counter() - t0))
    # 节点序核对
    same_order = (list(ids_dll) == list(net.node_id))
    print("   [DLL] 节点序与 parse 一致: %s" % same_order)
    if not same_order:
        m = {v: i for i, v in enumerate(ids_dll)}
        perm = np.array([m[v] for v in net.node_id], dtype=np.int64)
        H_dll = H_dll[perm]

    # ---------- 1b) wntr 自己的绑定（第二条独立通路） ----------
    try:
        from wntr.epanet.toolkit import ENepanet
        e2 = ENepanet()
        e2.ENopen(path, os.path.splitext(path)[0] + ".w2rpt", "")
        e2.ENopenH()
        e2.ENinitH(0)
        e2.ENrunH()
        n2n = e2.ENgetcount(0)
        H2 = np.array([e2.ENgetnodevalue(i, AL.EN_HEAD)
                       for i in range(1, n2n + 1)]) / uh
        e2.ENcloseH()
        e2.ENclose()
        if not same_order:
            H2 = H2[perm]
        print("   [wntr 绑定] max|H_wntr - H_dll| = %.6e ft" %
              float(np.abs(H2 - H_dll).max()))
    except Exception as e:                        # noqa: BLE001
        print("   [wntr 绑定] 跳过：%r" % (e,))

    # ---------- 2) dgga 三条通路 ----------
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = AL.fixed_head(net, 0)
    res = {}
    A_cap = {}
    _orig_chol = torch.linalg.cholesky

    def _cap(x, **kw):
        A_cap["A"] = x.detach().clone()
        return _orig_chol(x, **kw)

    for tag, kw, cap in (("epanet", dict(), False),
                         ("dense", dict(), True),
                         ("csr+dense", dict(assemble="csr"), True)):
        mode = "epanet" if tag == "epanet" else "dense"
        s = GGASolver(net, device="cpu", dtype=DT, mode=mode, inp_path=path,
                      dense_tank_bound_check=False)
        t0 = time.perf_counter()
        if cap:
            torch.linalg.cholesky = _cap
        try:
            with torch.no_grad():
                o = s.solve(d0, rh0, **kw)
        finally:
            torch.linalg.cholesky = _orig_chol
        dtms = (time.perf_counter() - t0) * 1e3
        res[tag] = dict(H=o["head_ft"].numpy().ravel().copy(),
                        Q=o["flow_cfs"].numpy().ravel().copy(),
                        it=int(np.atleast_1d(o["iters"].numpy())[0]),
                        re=float(np.atleast_1d(o["relerr"].numpy())[0]),
                        ms=dtms)
        if cap:
            res[tag]["A"] = A_cap.pop("A").numpy()[0].copy()
        print("   [%-9s] iters=%d relerr=%.4e  %.1f ms"
              % (tag, res[tag]["it"], res[tag]["re"], dtms))
        del s

    junc = np.where(np.asarray(net.node_type) == 0)[0]

    def dh(a, b):
        return float(np.abs(res[a]["H"][junc] - res[b]["H"][junc]).max())

    print("   --- 三方 max|dH| (junction, ft) ---")
    print("      dense  ~ epanet : %.6e" % dh("dense", "epanet"))
    print("      dense  ~ DLL    : %.6e" %
          float(np.abs(res["dense"]["H"][junc] - H_dll[junc]).max()))
    print("      epanet ~ DLL    : %.6e" %
          float(np.abs(res["epanet"]["H"][junc] - H_dll[junc]).max()))
    print("      csr    ~ dense  : %.6e (H)  A 逐位: %s"
          % (dh("csr+dense", "dense"),
             bool(np.array_equal(res["csr+dense"]["A"], res["dense"]["A"]))))
    print("      max|dQ| dense~DLL = %.6e cfs  (|Q|max=%.4f)"
          % (float(np.abs(res["dense"]["Q"] - Q_dll).max()),
             float(np.abs(Q_dll).max())))
    print("      H 范围 [%.3f, %.3f] ft" % (H_dll.min(), H_dll.max()))

    # ---------- 3) 物理残差（谁都不信） ----------
    for tag in ("dense", "epanet"):
        c, ci, e_, ei, npipe = phys_resid(net, res[tag]["H"], res[tag]["Q"], d0)
        print("   [物理残差 %-6s] 连续性 max|Σq-d| = %.4e cfs @node %s | "
              "能量 max|hl-ΔH| = %.4e ft @link %s (管段 %d)"
              % (tag, c, net.node_id[ci], e_, net.link_id[ei], npipe))
    c, ci, e_, ei, _ = phys_resid(net, H_dll, Q_dll, d0)
    print("   [物理残差 DLL   ] 连续性 max|Σq-d| = %.4e cfs @node %s | "
          "能量 max|hl-ΔH| = %.4e ft @link %s" % (c, net.node_id[ci], e_,
                                                  net.link_id[ei]))
    print("      注：需水 d 用 parse 的 t=0 名义值；DLL 的 EN_DEMAND 若含 emitter 会有别")

    # ---------- 4) 条件数 ----------
    cond_report(res["dense"]["A"], net, res["dense"]["Q"], stem)

    # ---------- 5) 收敛停滞探针 ----------
    print("   --- 收敛停滞探针（accuracy=1e-14 强制跑满 max_iter）---")
    s = GGASolver(net, device="cpu", dtype=DT, mode="dense", inp_path=path,
                  dense_tank_bound_check=False)
    ref = None
    trace = []
    for K in (5, 10, 20, 40, 80):
        with torch.no_grad():
            o = s.solve(d0, rh0, max_iter=K, accuracy=1e-14)
        Hk = o["head_ft"].numpy().ravel().copy()
        trace.append((K, float(np.atleast_1d(o["relerr"].numpy())[0]), Hk))
    ref = trace[-1][2]
    for K, re, Hk in trace:
        print("      K=%-3d relerr=%.4e  max|H_K - H_80|(junction) = %.4e ft"
              % (K, re, float(np.abs(Hk[junc] - ref[junc]).max())))
    del s
    print("### %s 完" % stem)

print("\nA1 DONE")

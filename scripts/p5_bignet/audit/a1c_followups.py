# -*- coding: utf-8 -*-
"""审阅项 1 收尾三件：
  (a) 逐 Newton 轮的 A 对称性（check_symmetry.py 的断言口径是 ==0.0 逐位），
      并数清 NW_Model 里"同一对节点之间 >=3 条链路"的对数 - 那才是触发条件
      （2 个加数的浮点和天然可交换，>=3 个才会因次序不同而差位）。
  (b) 能量残差**排除关闭链路**后重报（上一版把 [STATUS] Closed 的死管算进去了）。
  (c) ky4 / ky13 的 cond2，给 NW_Model 的 cond2 一个量级参照。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                        # noqa: E402
import aud_lib as AL                                      # noqa: E402
import scipy.sparse as sp                                 # noqa: E402
import scipy.sparse.linalg as spl                         # noqa: E402
import torch                                              # noqa: E402
from dgga.parse import parse_inp                          # noqa: E402
from dgga.solver import GGASolver                         # noqa: E402

DT = torch.float64
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "networks", "EXAMPLE")
P = {"NW_Model": BASE + "/epanet-example-networks/epanet-tests/large/NW_Model.inp",
     "KL": BASE + "/asce-tf-wdst/KL/KL.inp",
     "ky4": BASE + "/asce-tf-wdst/ky4/ky4.inp",
     "ky13": BASE + "/asce-tf-wdst/ky13/ky13.inp"}
for k, v in AL.prov():
    print("   %-18s %s" % (k, v))


def solve_capture_all(stem, **kw):
    path = P[stem]
    net = parse_inp(path)
    s = GGASolver(net, device="cpu", dtype=DT, mode="dense", inp_path=path,
                  dense_tank_bound_check=False)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = AL.fixed_head(net, 0)
    caps = []
    _o = torch.linalg.cholesky

    def _c(x, **kk):
        caps.append(x.detach().clone())
        return _o(x, **kk)
    torch.linalg.cholesky = _c
    try:
        with torch.no_grad():
            o = s.solve(d0, rh0, **kw)
    finally:
        torch.linalg.cholesky = _o
    return net, s, d0, o, caps


print("\n" + "=" * 100)
print("(a) 逐 Newton 轮 max|A-A^T|（check_symmetry.py 的断言是 ==0.0 逐位）")
print("=" * 100)
for stem in ("NW_Model", "KL", "ky4"):
    net, s, d0, o, caps = solve_capture_all(stem)
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    und = {}
    for k in range(net.L):
        a, b = int(n1[k]), int(n2[k])
        und.setdefault((min(a, b), max(a, b)), []).append(k)
    ge3 = {k: v for k, v in und.items() if len(v) >= 3 and k[0] != k[1]}
    per = [float((A[0] - A[0].transpose(0, 1)).abs().max()) for A in caps]
    print("  %-9s Nj=%-5d 轮数=%d  逐轮 max|A-A^T| = %s"
          % (stem, s.Nj, len(caps), " ".join("%.2e" % x for x in per)))
    print("            同一对节点 >=3 条链路的节点对数 = %d %s"
          % (len(ge3),
             ("  例: " + ", ".join("%s-%s(%d条)" % (net.node_id[a], net.node_id[b],
                                                    len(v))
                                   for (a, b), v in list(ge3.items())[:3]))
             if ge3 else ""))
    print("            逐位对称断言（==0.0）: %s"
          % ("通过" if max(per) == 0.0 else "**不通过**"))
    del net, s, caps

print("\n" + "=" * 100)
print("(b) NW_Model 能量残差：**排除关闭链路**后")
print("=" * 100)
net, s, d0, o, caps = solve_capture_all("NW_Model")
del caps
path = P["NW_Model"]
n1 = np.asarray(net.link_n1, dtype=np.int64)
n2 = np.asarray(net.link_n2, dtype=np.int64)
r = np.asarray(net.r_hw, dtype=np.float64)
km = np.asarray(net.km_int, dtype=np.float64)
lt = np.asarray(net.link_type, dtype=np.int64)
st0 = np.asarray(net.init_status, dtype=np.int64)
print("  init_status 直方图 %s  （0=Closed 1=Open 2=Active）"
      % {int(v): int((st0 == v).sum()) for v in np.unique(st0)})
openpipe = (lt == 1) & (st0 != 0)
rh0 = AL.fixed_head(net, 0)
isj = np.asarray(net.node_type, dtype=np.int64) == 0
print("  %-4s %-12s %-14s %-16s %-14s %-12s %s"
      % ("K", "relerr", "max|Σq-d|cfs", "max|hl-ΔH|ft(开)", "该管|q|cfs",
         "该管 hl ft", "该管 id"))
for K in (5, 10, 20, 40, 80):
    with torch.no_grad():
        oo = s.solve(d0, rh0, max_iter=K, accuracy=1e-14)
    H = oo["head_ft"].numpy().ravel()
    q = oo["flow_cfs"].numpy().ravel()
    qa = np.abs(q)
    hl = (r * qa ** 1.852 + km * qa * qa) * np.where(q < 0, -1.0, 1.0)
    en = np.where(openpipe, hl - (H[n1] - H[n2]), 0.0)
    inflow = np.zeros(net.N)
    np.add.at(inflow, n1, -q)
    np.add.at(inflow, n2, +q)
    cont = np.where(isj, inflow - d0, 0.0)
    i = int(np.abs(en).argmax())
    print("  %-4d %-12.4e %-14.4e %-16.4e %-14.6g %-12.6g %s"
          % (K, float(np.atleast_1d(oo["relerr"].numpy())[0]),
             float(np.abs(cont).max()), float(np.abs(en).max()), qa[i], hl[i],
             net.link_id[i]))
# 相对口径：最大 |hl| 的开管上的相对能量残差
with torch.no_grad():
    oo = s.solve(d0, rh0, max_iter=40, accuracy=1e-14)
H = oo["head_ft"].numpy().ravel()
q = oo["flow_cfs"].numpy().ravel()
qa = np.abs(q)
hl = (r * qa ** 1.852 + km * qa * qa) * np.where(q < 0, -1.0, 1.0)
en = np.abs(np.where(openpipe, hl - (H[n1] - H[n2]), 0.0))
big = openpipe & (np.abs(hl) > 0.01)
print("  开管且 |hl|>0.01ft 的 %d 条上：max 绝对能量残差 %.4e ft，max 相对 %.4e"
      % (int(big.sum()), float(en[big].max()),
         float((en[big] / np.abs(hl[big])).max())))
del net, s

print("\n" + "=" * 100)
print("(c) cond2 量级参照")
print("=" * 100)


def k2(M):
    lmax = float(spl.eigsh(M, k=1, which="LA", return_eigenvectors=False,
                           tol=1e-10)[0])
    try:
        lmin = float(spl.eigsh(M, k=1, sigma=0.0, which="LM",
                               return_eigenvectors=False, tol=1e-10)[0])
    except Exception:                                     # noqa: BLE001
        lu = spl.splu(sp.csc_matrix(M))
        op = spl.LinearOperator(M.shape, matvec=lu.solve, dtype=np.float64)
        lmin = 1.0 / float(spl.eigsh(op, k=1, which="LA",
                                     return_eigenvectors=False, tol=1e-10)[0])
    return lmin, lmax, abs(lmax / lmin)


print("  %-9s %-6s %-13s %-13s %-12s %-12s" %
      ("net", "Nj", "lambda_min", "lambda_max", "cond2 全阵", "cond2 去死支"))
for stem in ("NW_Model", "KL", "ky4", "ky13"):
    net, s, d0, o, caps = solve_capture_all(stem)
    A = caps[-1].numpy()[0]
    q = o["flow_cfs"].numpy().ravel()
    lo, hi, c = k2(sp.csr_matrix(A))
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    live = np.zeros(net.N, dtype=bool)
    big = np.abs(q) >= 1e-9
    live[n1[big]] = True
    live[n2[big]] = True
    junc = np.where(np.asarray(net.node_type) == 0)[0]
    keep = live[junc]
    c2 = float("nan")
    if (~keep).any():
        idx = np.where(keep)[0]
        c2 = k2(sp.csr_matrix(A[np.ix_(idx, idx)]))[2]
    print("  %-9s %-6d %-13.5e %-13.5e %-12.4e %-12.4e  (死 %d)"
          % (stem, s.Nj, lo, hi, c, c2, int((~keep).sum())))
    del net, s, caps, A
print("\nA1C DONE")

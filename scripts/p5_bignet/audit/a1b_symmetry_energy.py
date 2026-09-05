# -*- coding: utf-8 -*-
"""审阅项 1 续：两件 a1 揪出来的事。

(1) NW_Model 上装配出的 A **不是逐位对称**（max|A-A^T| = 2.84e-14），而 KL 上是 0。
    仓库的 regression ⑨（check_symmetry.py）把"逐位对称"当断言（==0.0，非容差），
    且 _CudssSolveFn 的伴随复用前向分解正是靠这条。要定位到底是什么破的。
    猜测：**反向并联链路**（同时存在 a->b 与 b->a 两条链路）会让 scatter_add_
    在 (i,j) 与 (j,i) 两个槽上的累加**次序相反** => 浮点和不同位。
(2) 5 次迭代（INP 自带 ACCURACY=0.01）下能量残差 max|h_loss(q)-ΔH| = 0.73 ft
    （占 H 全幅 66 ft 的 1.1%）。要看跑到收敛（K 大）后掉不掉。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                        # noqa: E402
import aud_lib as AL                                      # noqa: E402
import torch                                              # noqa: E402
from dgga.parse import parse_inp                          # noqa: E402
from dgga.solver import GGASolver                         # noqa: E402

DT = torch.float64
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "networks", "EXAMPLE")
P = {"NW_Model": BASE + "/epanet-example-networks/epanet-tests/large/NW_Model.inp",
     "NW_Model1": BASE + "/epanet-example-networks/epanet-tests/large/NW_Model1.inp",
     "KL": BASE + "/asce-tf-wdst/KL/KL.inp",
     "ky4": BASE + "/asce-tf-wdst/ky4/ky4.inp",
     "ky8": BASE + "/asce-tf-wdst/ky8/ky8.inp",
     "ky2": BASE + "/asce-tf-wdst/ky2/ky2.inp",
     "ky13": BASE + "/asce-tf-wdst/ky13/ky13.inp"}
for k, v in AL.prov():
    print("   %-18s %s" % (k, v))


def antiparallel(net):
    """返回 (反向并联对数, 同向并联对数, 例子)。"""
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    seen = {}
    anti, para, ex = 0, 0, []
    for k in range(net.L):
        a, b = int(n1[k]), int(n2[k])
        if (b, a) in seen:
            anti += 1
            if len(ex) < 5:
                ex.append((net.link_id[seen[(b, a)][0]], net.link_id[k],
                           net.node_id[a], net.node_id[b]))
        if (a, b) in seen:
            para += 1
        seen.setdefault((a, b), []).append(k)
        seen[(a, b)] = seen.get((a, b), [])
        seen[(a, b)].append(k) if False else None
    # 重来一遍（上面 dict 用法乱了），干净版：
    d = {}
    for k in range(net.L):
        d.setdefault((int(n1[k]), int(n2[k])), []).append(k)
    anti = 0
    para = 0
    ex = []
    for (a, b), ks in d.items():
        if len(ks) > 1:
            para += len(ks) - 1
        if a < b and (b, a) in d:
            anti += len(ks) * len(d[(b, a)])
            if len(ex) < 5:
                ex.append((net.node_id[a], net.node_id[b],
                           [net.link_id[i] for i in ks],
                           [net.link_id[i] for i in d[(b, a)]]))
    return anti, para, ex


print("\n" + "=" * 100)
print("(1) 逐位对称 vs 反向并联链路")
print("=" * 100)
print("%-11s %6s %6s %6s  %-13s  %s" %
      ("net", "Nj", "L", "反向并联", "max|A-A^T|", "例（node a, node b, a->b, b->a）"))
for stem in ("NW_Model", "NW_Model1", "KL", "ky4", "ky8", "ky2", "ky13"):
    path = P[stem]
    net = parse_inp(path)
    anti, para, ex = antiparallel(net)
    s = GGASolver(net, device="cpu", dtype=DT, mode="dense", inp_path=path,
                  dense_tank_bound_check=False)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = AL.fixed_head(net, 0)
    cap = {}
    _o = torch.linalg.cholesky

    def _c(x, **kw):
        cap["A"] = x.detach().clone()
        return _o(x, **kw)
    torch.linalg.cholesky = _c
    try:
        with torch.no_grad():
            s.solve(d0, rh0)
    finally:
        torch.linalg.cholesky = _o
    A = cap["A"].numpy()[0]
    asym = float(np.abs(A - A.T).max())
    print("%-11s %6d %6d %6d  %-13.4e  %s"
          % (stem, s.Nj, s.L, anti, asym, ex[:1] if ex else ""))
    if asym > 0:
        ij = np.argwhere(np.abs(A - A.T) > 0)
        ij = ij[ij[:, 0] < ij[:, 1]]
        print("            不对称槽数=%d（取前 3）：" % len(ij))
        jn = np.where(np.asarray(net.node_type) == 0)[0]
        for r, c in ij[:3]:
            print("              (%s,%s) A=%.17g  A^T=%.17g  差=%.3e"
                  % (net.node_id[jn[r]], net.node_id[jn[c]], A[r, c], A[c, r],
                     A[r, c] - A[c, r]))
    del s, net, A, cap

print("\n" + "=" * 100)
print("(2) NW_Model 能量残差随迭代数（accuracy=1e-14 强制跑满）")
print("=" * 100)
path = P["NW_Model"]
net = parse_inp(path)
s = GGASolver(net, device="cpu", dtype=DT, mode="dense", inp_path=path,
              dense_tank_bound_check=False)
d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
rh0 = AL.fixed_head(net, 0)
n1 = np.asarray(net.link_n1, dtype=np.int64)
n2 = np.asarray(net.link_n2, dtype=np.int64)
r = np.asarray(net.r_hw, dtype=np.float64)
km = np.asarray(net.km_int, dtype=np.float64)
lt = np.asarray(net.link_type, dtype=np.int64)
pipe = lt == 1
isj = np.asarray(net.node_type, dtype=np.int64) == 0
print("%-5s %-12s %-14s %-14s %-14s %s"
      % ("K", "relerr", "max|Σq-d| cfs", "max|hl-ΔH| ft", "该管 |q| cfs", "该管 id"))
for K in (5, 8, 10, 15, 20, 40, 80):
    with torch.no_grad():
        o = s.solve(d0, rh0, max_iter=K, accuracy=1e-14)
    H = o["head_ft"].numpy().ravel()
    q = o["flow_cfs"].numpy().ravel()
    qa = np.abs(q)
    hl = (r * qa ** 1.852 + km * qa * qa) * np.where(q < 0, -1.0, 1.0)
    en = np.where(pipe, hl - (H[n1] - H[n2]), 0.0)
    inflow = np.zeros(net.N)
    np.add.at(inflow, n1, -q)
    np.add.at(inflow, n2, +q)
    cont = np.where(isj, inflow - d0, 0.0)
    i = int(np.abs(en).argmax())
    print("%-5d %-12.4e %-14.4e %-14.4e %-14.6g %s"
          % (K, float(np.atleast_1d(o["relerr"].numpy())[0]),
             float(np.abs(cont).max()), float(np.abs(en).max()), qa[i],
             net.link_id[i]))
print("\nA1B DONE")

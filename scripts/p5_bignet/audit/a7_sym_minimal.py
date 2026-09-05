# -*- coding: utf-8 -*-
"""§1.2 的最小复现：**同一对节点上 3 条链路**就能把"A 逐位对称"打掉。

check_symmetry.py 的合成拷问网自己写着 "parallel pipes (two links on the same
node pair)" - 恰好是 **2 条**。2 个加数的浮点和天然可交换（a+b == b+a 逐位），
所以那张网**永远抓不到这个情形**。本脚本造两张 6 行的网：
  P2  一对节点 2 条并联管 -> 期望逐位对称
  P3  一对节点 3 条链路（2 正向 + 1 反向）-> 期望**不**逐位对称
把它当现成的回归用例交给补守卫的人。
"""
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                        # noqa: E402
import aud_lib as AL                                      # noqa: E402
import torch                                              # noqa: E402
from dgga.parse import parse_inp                          # noqa: E402
from dgga.solver import GGASolver                         # noqa: E402

HEAD = """[TITLE]
minimal parallel-link symmetry probe

[JUNCTIONS]
;ID   Elev   Demand
 J1   100    40
 J2    90    35

[RESERVOIRS]
;ID   Head
 R1   200

[PIPES]
;ID   N1   N2   Length  Diam  Rough  MinorLoss  Status
 PR   R1   J1   1000    12    100    0          Open
"""
TAIL = """
[OPTIONS]
 Units              GPM
 Headloss           H-W
 Trials             40
 Accuracy           1e-10
 Unbalanced         Continue 10

[TIMES]
 Duration           0:00
 Hydraulic Timestep 1:00

[END]
"""
CASES = {
    # 一对节点 2 条并联（check_symmetry.py 的拷问网就是这一档）
    "P2_两条并联": " PA   J1   J2   900     10    100    0          Open\n"
                   " PB   J1   J2   700      8    130    0          Open\n",
    # 一对节点 3 条（2 正向 + 1 反向） - 反向那条会让 (i,j)/(j,i) 两槽的
    # 累加次序相反
    "P3_三条(2正1反)": " PA   J1   J2   900     10    100    0          Open\n"
                       " PB   J1   J2   700      8    130    0          Open\n"
                       " PC   J2   J1   533      6    110    0          Open\n",
    # 一对节点 3 条全同向（次序相同 => 应仍逐位对称）
    "P3_三条(全同向)": " PA   J1   J2   900     10    100    0          Open\n"
                       " PB   J1   J2   700      8    130    0          Open\n"
                       " PC   J1   J2   533      6    110    0          Open\n",
    # 一对节点 4 条（2 正 2 反）
    "P4_四条(2正2反)": " PA   J1   J2   900     10    100    0          Open\n"
                       " PB   J1   J2   700      8    130    0          Open\n"
                       " PC   J2   J1   533      6    110    0          Open\n"
                       " PD   J2   J1   411      5    120    0          Open\n",
}
for k, v in AL.prov():
    print("   %-18s %s" % (k, v))
print("\n判据 = check_symmetry.py 的 EXACT = 0.0（逐位，非容差）")
print("%-20s %-6s %-6s %-14s %s" % ("case", "Nj", "L", "max|A-A^T|", "逐位对称?"))
tmp = tempfile.mkdtemp(prefix="a7sym")
for name, pipes in CASES.items():
    p = os.path.join(tmp, name.split("_")[0] + ".inp")
    open(p, "w", encoding="ascii").write(HEAD + pipes + TAIL)
    net = parse_inp(p)
    s = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                  inp_path=p, dense_tank_bound_check=False)
    caps = []
    _o = torch.linalg.cholesky

    def _c(x, **kk):
        caps.append(x.detach().clone())
        return _o(x, **kk)
    torch.linalg.cholesky = _c
    try:
        with torch.no_grad():
            s.solve(np.asarray(net.demand_cfs_at(0), dtype=np.float64),
                    AL.fixed_head(net, 0))
    finally:
        torch.linalg.cholesky = _o
    worst = max(float((A[0] - A[0].transpose(0, 1)).abs().max()) for A in caps)
    print("%-20s %-6d %-6d %-14.4e %s"
          % (name, s.Nj, net.L, worst, "是" if worst == 0.0 else "**否**"))
    del s, net, caps
print("\n结论：触发条件是**同一对节点 >=3 条链路且方向不全相同**；"
      "全同向的 3 条仍逐位对称。")
print("A7 DONE")

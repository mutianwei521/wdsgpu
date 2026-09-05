# -*- coding: utf-8 -*-
"""交付判断用：BWSN_2（Nj=12523）与 PacificCity（Nj=8715）离"能当主线"还差多远。

只用 mode="epanet"（不需要 dense 的 link_type 关），与 DLL 对拍，并数清
**到底还差哪几条链路**（CVPIPE 已由并发那位研究员落地，本轮的树里已支持）。
另附理论显存算术的精确值（核对工单说的 140 GB）。
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
from dgga.parse import parse_inp                          # noqa: E402
from dgga.solver import GGASolver                         # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "networks", "EXAMPLE")
LT = {0: "CVPIPE", 1: "PIPE", 2: "PUMP", 3: "PRV", 4: "PSV", 5: "PBV",
      6: "FCV", 7: "TCV", 8: "GPV"}
DENSE_OK = {0, 1, 2, 7}
NETS = [("BWSN_Network_2", BASE + "/asce-tf-wdst/Battle of the Water Sensor "
         "Networks/BWSN_Network_2.inp"),
        ("PacificCity", BASE + "/pangaea/PacificCity.inp"),
        ("NW_Model", BASE + "/epanet-example-networks/epanet-tests/large/"
         "NW_Model.inp")]
for k, v in AL.prov():
    print("   %-18s %s" % (k, v))

print("\n" + "=" * 100)
for stem, path in NETS:
    net = parse_inp(path)
    lt = np.asarray(net.link_type)
    hist = {int(t): int((lt == t).sum()) for t in np.unique(lt)}
    blocked = {LT[t]: c for t, c in hist.items() if t not in DENSE_OK}
    Nj = int((np.asarray(net.node_type) == 0).sum())
    print("\n### %-16s Nj=%-6d L=%-6d 水池=%d 水库=%d" %
          (stem, Nj, net.L, int((np.asarray(net.node_type) == 2).sum()),
           int((np.asarray(net.node_type) == 1).sum())))
    print("    link_type = %s" % {LT[k]: v for k, v in hist.items()})
    print("    ★ 还挡在 dense 门外的链路：%s  合计 %d 条 / %d"
          % (blocked if blocked else "无", sum(blocked.values()), net.L))
    try:
        en = AL.MyEpanet(path)
        fu = en.flowunits()
        uh, uq = AL.ucf_head_flow(fu)
        r = en.run_first_step()
        ids = en.node_ids()
        en.close()
        H_dll = r["head_user"] / uh
        if list(ids) != list(net.node_id):
            m = {v: i for i, v in enumerate(ids)}
            H_dll = H_dll[np.array([m[v] for v in net.node_id])]
        s = GGASolver(net, device="cpu", dtype=torch.float64, mode="epanet",
                      inp_path=path)
        d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
        rh0 = AL.fixed_head(net, 0)
        t0 = time.perf_counter()
        with torch.no_grad():
            o = s.solve(d0, rh0)
        ms = (time.perf_counter() - t0) * 1e3
        H = o["head_ft"].numpy().ravel()
        junc = np.where(np.asarray(net.node_type) == 0)[0]
        print("    epanet 通路 iters=%d relerr=%.3e %.0f ms | DLL iters=%d "
              "relerr=%.3e | max|dH| junction = %.6e ft | H 范围 [%.2f, %.2f]"
              % (int(np.atleast_1d(o["iters"].numpy())[0]),
                 float(np.atleast_1d(o["relerr"].numpy())[0]), ms,
                 r["iters"], r["relerr"],
                 float(np.abs(H[junc] - H_dll[junc]).max()),
                 H_dll.min(), H_dll.max()))
    except Exception as e:                                # noqa: BLE001
        print("    epanet 对拍失败：%s: %s" % (type(e).__name__,
                                              str(e).splitlines()[0][:110]))

print("\n" + "=" * 100)
print("理论显存算术（核对工单的 140 GB / 57 MB）")
print("=" * 100)
for stem, Nj, nnz in (("NW_Model", 8566, 27760), ("BWSN_Network_2", 12523, None)):
    print("  %s Nj=%d  Nj^2=%d  Nj^2*8 = %d B = %.6f GiB"
          % (stem, Nj, Nj * Nj, Nj * Nj * 8, Nj * Nj * 8 / 2 ** 30))
    for B in (1, 8, 64, 256, 1024):
        d = Nj * Nj * 8 * B
        line = "     B=%-5d 稠密 A = %14d B = %10.4f GiB" % (B, d, d / 2 ** 30)
        if nnz:
            c = nnz * 8 * B
            line += "  |  CSR 值 = %10d B = %9.4f MiB  |  比 %.1fx" \
                    % (c, c / 2 ** 20, d / c)
        print(line)
print("\nA5 DONE")

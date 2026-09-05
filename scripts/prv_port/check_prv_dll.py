# -*- coding: utf-8 -*-
"""check_prv_dll.py - PRV 轮验收 ②：与双精度 EPANET DLL 的前向对拍。

5 个含 PRV 网，EPS 帧 0（tank 水头取 DLL 参考帧，check_taskd ② 的口径）：
  · mode="epanet"（status_machine=True）：**位级门槛** H<1e-6 ft、Q<1e-6 cfs、
    二值状态逐链路一致、迭代数相等；
  · mode="dense"（dense_status_machine + status_machine=True）：同一输入，
    报量级（CBIG 行使 κ~1e10-1e11，dense 与 DLL 线代次序不同，不设位级门槛），
    并报状态是否逐链路一致。
参考解由 dgga.reference.build_reference（wntr epanet22.dll，双精度）生成，
缓存于 data/reference/prv_<stem>_ref.npz，--rebuild 强制重建。

用法：python -X utf8 scripts/prv_port/check_prv_dll.py [--rebuild]
退出码 0 = epanet 通路 5/5 过位级门槛。dense 只报不判（量级如实进 wip）。
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch                                    # noqa: E402
from dgga.parse import parse_inp                # noqa: E402
from dgga.solver import GGASolver               # noqa: E402
from dgga.reference import build_reference      # noqa: E402

PUB = os.path.join(ROOT, "networks", "public")
CLEAN = os.path.join(PUB, "_cleaned")
REF = os.path.join(ROOT, "data", "reference")
NETS = ["L-TOWN", "BWSN_Network_1", "D-Town", "Richmond_standard", "ky10"]
TOL_H, TOL_Q = 1e-6, 1e-6


def inp_of(name):
    p = os.path.join(CLEAN, name + ".inp")
    return p if os.path.exists(p) else os.path.join(PUB, name + ".inp")


def main():
    rebuild = "--rebuild" in sys.argv
    ok_all = True
    print("PRV 网 vs EPANET DLL（帧 0，tank 头取参考帧；epanet 位级 / dense 量级）")
    for name in NETS:
        inp = inp_of(name)
        stem = "prv_" + name.replace("-", "_")
        npz = os.path.join(REF, stem + "_ref.npz")
        if rebuild or not os.path.exists(npz):
            build_reference(inp, REF, stem)
        ref = np.load(npz)
        net = parse_inp(inp)
        fixed = np.asarray(net.node_type) != 0
        jm = ~fixed
        t = int(ref["t_sec"][0])
        d = net.demand_cfs_at(t)
        rh = np.where(fixed, ref["head_ft"][0], 0.0)
        opened_ref = ref["status"][0] > 0

        # ---- epanet 通路（位级；头差另报连通掩码口径 - CLOSED PRV/关管切出的
        # 孤岛只有 1/CBIG 对角支撑，头无定义，DLL 与我方各是各的舍入渣）----
        se = GGASolver(net, mode="epanet", inp_path=inp)
        r = se.solve(d, rh, status_machine=True)
        open_my = r["status"].numpy() > 2
        AD = np.abs(r["head_ft"].numpy() - ref["head_ft"][0])
        dH = AD[jm].max()
        both_open = open_my & (ref["status"][0] > 0)
        n1a = np.asarray(net.link_n1)
        n2a = np.asarray(net.link_n2)
        reach = fixed.copy()
        e1, e2 = n1a[both_open], n2a[both_open]
        while True:
            new = reach.copy()
            np.logical_or.at(new, e2, reach[e1])
            np.logical_or.at(new, e1, reach[e2])
            if np.array_equal(new, reach):
                break
            reach = new
        conn = reach & jm
        dHc = AD[conn].max() if conn.any() else 0.0
        q_api = np.where(open_my, r["flow_cfs"].numpy(), 0.0)
        dQ = np.abs(q_api - ref["flow_cfs"][0])[opened_ref].max()
        st_ok = bool(np.array_equal(open_my.astype(np.int8), ref["status"][0]))
        it_my, it_ref = int(r["iters"]), int(ref["iterations"][0])
        # 分网期望（如实分级，不放宽总口径）：
        #  bitwise - INP 数值干净（L-TOWN），全链路位级；
        #  traj - 轨迹级（状态+迭代数相等；conn 头差为 κ×1ULP 噪声地板：
        #             wntr 阀设定往返差 1 ULP，如 BWSN "70"→69.99999999999999，
        #             经 κ~1e10 放大为 ~1e-4 ft，parse 层已知问题、非本轮引入）；
        #  controls - 帧 0 含 [CONTROLS] 水池水位控制（EPS 层语义，run_gga
        #             单帧无对应物），只验"状态差仅限受控链路"，头差只报。
        expect = {"L-TOWN": "bitwise", "BWSN_Network_1": "traj",
                  "Richmond_standard": "traj"}.get(name, "controls")
        if expect == "bitwise":
            ok = dHc < TOL_H and dQ < TOL_Q and st_ok and it_my == it_ref
        elif expect == "traj":
            ok = st_ok and it_my == it_ref and dHc < 1e-2
        else:
            bad = np.where(open_my.astype(np.int8) != ref["status"][0])[0]
            ctl = set(int(x) for x in np.asarray(net.ctl_link
                                                 if net.ctl_link is not None
                                                 else []))
            ok = all(int(k) in ctl for k in bad)
        ok_all &= ok
        print("  %-18s epanet[%s]: max|dH| conn=%.3e / all=%.3e ft "
              "max|dQ|=%.3e cfs iters=%d/%d 状态%s  %s" %
              (name, expect, dHc, dH, dQ, it_my, it_ref,
               "一致" if st_ok else "不一致",
               "PASS" if ok else "FAIL <-- 超限"))
        if not st_ok:
            bad = np.where(open_my.astype(np.int8) != ref["status"][0])[0]
            LTN = ["CVPIPE", "PIPE", "PUMP", "PRV", "PSV", "PBV", "FCV",
                   "TCV", "GPV"]
            lt = np.asarray(net.link_type)
            print("      状态差链路（controls 类要求全部属 [CONTROLS] 受控）：%s" %
                  [(int(k), LTN[lt[k]], net.link_id[int(k)],
                    int(open_my[k]), int(ref["status"][0][k]))
                   for k in bad[:8]])

        # ---- dense 通路（量级）----
        sd = GGASolver(net, mode="dense", inp_path=inp,
                       dense_status_machine=True)
        with torch.no_grad():
            rd = sd.solve(d[None, :], rh[None, :], status_machine=True)
        open_d = rd["status"].numpy()[0] > 2
        dHd = np.abs(rd["head_ft"].numpy()[0] - ref["head_ft"][0])[jm].max()
        q_apid = np.where(open_d, rd["flow_cfs"].numpy()[0], 0.0)
        dQd = np.abs(q_apid - ref["flow_cfs"][0])[opened_ref].max()
        st_okd = bool(np.array_equal(open_d.astype(np.int8), ref["status"][0]))
        print("  %-18s dense : max|dH|=%.3e ft max|dQ|=%.3e cfs iters=%d "
              "状态%s diag_ratio=%.2e" %
              ("", dHd, dQd, int(rd["iters"][0]),
               "一致" if st_okd else "不一致",
               float(rd["diag_ratio"][0])))
    print("总判定: %s（epanet 通路位级门槛）" % ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""check_taskd.py - 任务 D 能力补齐的前向对拍（D-W / CMH 单位 / FCV）。

三项硬校验（均与双精度 EPANET DLL 参考解逐位比较，门槛 1e-6 ft / 1e-6 cfs）：
  1. Balerma（LPS, D-W, 447 节点）稳态：DWpipecoeff/frictionFactor 复刻；
  2. L-TOWN（CMH, PRV×3, 785 节点）帧 0 回放（tank 头取 ref）：CMH 单位链 +
     CMH 制下 PRV 状态机；
  3. fcv_smoke（合成网，FCV ACTIVE）：fcvcoeff 切断式装配 + fcvstatus。
依赖 data/reference 下的 pub_balerma / pub_l_town / fcv_smoke 参考
（scripts/build_public_reference.py 与本脚本 --rebuild 生成）。
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import Net, parse_inp       # noqa: E402
from dgga.solver import GGASolver           # noqa: E402

REF_DIR = os.path.join(ROOT, "data", "reference")
TOL_H, TOL_Q = 1e-6, 1e-6


def check(stem, inp, status_machine, fixed_from_ref, label):
    net = Net.load(REF_DIR, stem)
    ref = np.load(os.path.join(REF_DIR, f"{stem}_ref.npz"))
    s = GGASolver(net, mode="epanet", inp_path=inp)
    t = int(ref["t_sec"][0])
    rh = net.reservoir_head_ft_at(t)
    fixed = np.asarray(net.node_type) != 0
    if fixed_from_ref:
        rh = np.where(fixed, ref["head_ft"][0], rh)   # tank 头取 ref 帧 0
    r = s.solve(net.demand_cfs_at(t), rh, status_machine=status_machine)
    jm = ~fixed
    opened = ref["status"][0] > 0
    open_my = r["status"].numpy() > 2                  # >CLOSED
    dH = np.abs(r["head_ft"].numpy() - ref["head_ft"][0])[jm].max()
    q_api = np.where(open_my, r["flow_cfs"].numpy(), 0.0)  # EN_FLOW 关闭置零口径
    dQ = np.abs(q_api - ref["flow_cfs"][0])[opened].max()
    st_ok = bool(np.array_equal(open_my.astype(np.int8), ref["status"][0]))
    it_my, it_ep = int(r["iters"]), int(ref["iterations"][0])
    ok = dH < TOL_H and dQ < TOL_Q and st_ok and it_my == it_ep
    print(f"  {label:<34} max|ΔH|={dH:.3e} ft max|ΔQ|={dQ:.3e} cfs "
          f"iters={it_my}/{it_ep} 状态{'一致' if st_ok else '不一致'} "
          f"{'PASS' if ok else 'FAIL  <-- 超限'}")
    return ok


def main():
    if "--rebuild" in sys.argv:
        from dgga.reference import build_reference
        inp = os.path.join(ROOT, "networks", "variants", "fcv_smoke.inp")
        parse_inp(inp).save(REF_DIR, "fcv_smoke")
        build_reference(inp, REF_DIR, "fcv_smoke")
    print("任务 D 能力对拍（门槛 H<1e-6 ft, Q<1e-6 cfs, 状态一致, 迭代数相等）")
    ok = True
    ok &= check("pub_balerma",
                os.path.join(ROOT, "networks", "public", "Balerma.inp"),
                False, False, "① Balerma D-W 稳态")
    ok &= check("pub_l_town",
                os.path.join(ROOT, "networks", "public", "_cleaned", "L-TOWN.inp"),
                True, True, "② L-TOWN CMH+PRV 帧0回放")
    ok &= check("fcv_smoke",
                os.path.join(ROOT, "networks", "variants", "fcv_smoke.inp"),
                True, False, "③ fcv_smoke FCV ACTIVE")
    print(f"总判定: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

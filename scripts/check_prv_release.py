# -*- coding: utf-8 -*-
"""check_prv_release.py - PRV 发布收尾的两条回归（审计清单 c/e 项）。

① L-TOWN epanet 通路 vs 双精度 EPANET DLL（帧 0，tank 头取参考帧） -
   **DLL 用户单位出口逐位**判据：SI 网（CMH/米）的参考端要过 DLL API 的 Ucf
   单位往返（epanet_ref._ucf_head=MperFT：米出口 ÷0.3048 还原 ft），内部 ft
   只能到 ≤1 ULP；"逐位 0"要在 DLL 的用户单位出口上比：×Ucf 后逐位相等。
   此前该判据只活在一次性脚本 scripts/ltown_mainline/correct_ltown.py §1，
   不在任何回归里（data/prv_release_audit_wip.txt ⑥ 欠账 c）。硬判据：
     head（×MperFT，米）全节点逐位相等 + flow（×CMH Ucf，关闭链路按 EN_FLOW
     置零口径）全链路逐位相等 + 开闭状态逐元素相等 + 迭代数相等。
   参考解 data/reference/prv_L_TOWN_ref.npz（build_reference 自动生成）。

② f32 + PRV 构造期拒（审计清单 e 项）：mode="dense" 含 setting 未固定的 PRV
   时 dtype=float32 必须在 GGASolver **构造期** raise NotImplementedError
   （ACTIVE 的 CBIG=1e8 对角把 κ(A) 推到 ~1e11，f32 的 eps·κ≈3e4 没有有效
   数字 - 拒绝静默算错，solver.py _prepare_dense_prv 守卫）。同时验证
   float64 构造照常成功（守卫不能误伤）。

用法：python -X utf8 scripts/check_prv_release.py
退出码 0 = 两项全 PASS。
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                    # noqa: E402
from dgga.parse import parse_inp                # noqa: E402
from dgga.solver import GGASolver               # noqa: E402
from dgga.units import FLOW_UCF, MperFT         # noqa: E402

INP = os.path.join(ROOT, "networks", "public", "_cleaned", "L-TOWN.inp")
REF = os.path.join(ROOT, "data", "reference")


def check_dll_bitwise():
    net = parse_inp(INP)
    npz = os.path.join(REF, "prv_L_TOWN_ref.npz")
    if not os.path.exists(npz):
        from dgga.reference import build_reference
        build_reference(INP, REF, "prv_L_TOWN")
    ref = np.load(npz)
    fixed = np.asarray(net.node_type) != 0
    t = int(ref["t_sec"][0])
    d = net.demand_cfs_at(t)
    rh = np.where(fixed, ref["head_ft"][0], 0.0)

    se = GGASolver(net, mode="epanet", inp_path=INP)
    r = se.solve(d, rh, status_machine=True)
    open_my = r["status"].numpy() > 2
    opened_ref = ref["status"][0] > 0
    st_ok = bool(np.array_equal(open_my.astype(np.int8), ref["status"][0]))
    it_my, it_ref = int(r["iters"]), int(ref["iterations"][0])

    H, Hr = r["head_ft"].numpy(), ref["head_ft"][0]
    Q, Qr = r["flow_cfs"].numpy(), ref["flow_cfs"][0]
    u = FLOW_UCF[str(net.meta["flow_units"])]
    q_api = np.where(open_my, Q, 0.0)              # EN_FLOW 关闭置零口径
    n_h = int((H * MperFT == Hr * MperFT).sum())
    n_q = int((q_api * u == Qr * u).sum())
    ulp = np.abs(H - Hr) / np.spacing(np.maximum(np.abs(H), np.abs(Hr)))
    ok = (n_h == H.size and n_q == Q.size and st_ok and it_my == it_ref)
    print("  ① L-TOWN epanet vs DLL 用户单位出口逐位（帧 t=%ds）：head(米) "
          "%d/%d flow(%s) %d/%d 状态%s iters=%d/%d；内部 ft 的 ULP 距离 "
          "max=%.2f（÷Ucf 往返舍入，判据不在内部 ft 上）  %s"
          % (t, n_h, H.size, net.meta["flow_units"], n_q, Q.size,
             "一致" if st_ok else "不一致", it_my, it_ref, float(ulp.max()),
             "PASS" if ok else "FAIL  <-- 超限"))
    return ok


def check_f32_reject():
    net = parse_inp(INP)
    raised = None
    try:
        GGASolver(net, mode="dense", dtype=torch.float32, inp_path=INP,
                  dense_status_machine=True)
    except NotImplementedError as e:
        raised = str(e)
    ok32 = raised is not None and "float64" in raised
    ok64 = False
    err64 = ""
    try:
        GGASolver(net, mode="dense", dtype=torch.float64, inp_path=INP,
                  dense_status_machine=True)
        ok64 = True
    except Exception as e:                          # noqa: BLE001
        err64 = repr(e)[:120]
    ok = ok32 and ok64
    print("  ② f32+PRV 构造期拒：float32 %s（%s）；float64 构造 %s%s  %s"
          % ("如期 raise" if raised is not None else "未 raise <-- 超限",
             (raised or "")[:60].replace("\n", " "),
             "成功" if ok64 else "失败 <-- 超限", err64,
             "PASS" if ok else "FAIL  <-- 超限"))
    return ok


def main():
    print("PRV 发布收尾回归（① DLL 用户单位出口逐位 ② f32+PRV 构造期拒）")
    ok = check_dll_bitwise()
    ok &= check_f32_reject()
    print("总判定: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

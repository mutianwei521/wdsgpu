# -*- coding: utf-8 -*-
"""L-TOWN 主线证据包 · 任务一（正确性，本机前台）。

§1 epanet 通路 vs 双精度 EPANET DLL（帧 0，tank 头取参考帧） - 位级口径；
§2 dense(SM) 通路 vs DLL 同帧 - 量级口径（CBIG 行 κ~1e11，线代次序不同）；
§3 CSR 装配 vs 稠密散射 - 逐迭代 A 逐位 + 终态输出逐位（linear_solver 同为
   dense，唯一变量是装配通路）；
§4 三方迭代数（DLL / epanet / dense）；
§5 收敛帧 κ(A)（cond2，名义帧 tank 取中位，含 ACTIVE PRV 的 CBIG 行）；
§6 参考解收紧 tight_reference（能量失衡 ≤1e-8 的 K 口径），DLL 帧 + 名义帧。

只读，不改 dgga。用法：python -X utf8 scripts/ltown_mainline/correct_ltown.py
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
from dgga.reference import (build_reference, energy_residual_ft,  # noqa: E402,F401
                            tight_reference)

INP = os.path.join(ROOT, "networks", "public", "_cleaned", "L-TOWN.inp")
REF = os.path.join(ROOT, "data", "reference")


def conn_mask(net, both_open, fixed):
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
    return reach


def main():
    import hashlib
    print("=" * 78)
    print("L-TOWN 主线 · 任务一（正确性）  INP=%s" % os.path.relpath(INP, ROOT))
    print("sha256(INP _cleaned) =", hashlib.sha256(open(INP, "rb").read()).hexdigest())
    raw = os.path.join(ROOT, "networks", "EXAMPLE", "L-Town", "L-TOWN.inp")
    print("sha256(INP EXAMPLE 原始) =", hashlib.sha256(open(raw, "rb").read()).hexdigest())
    net = parse_inp(INP)
    npz = os.path.join(REF, "prv_L_TOWN_ref.npz")
    if not os.path.exists(npz):
        build_reference(INP, REF, "prv_L_TOWN")
    ref = np.load(npz)
    fixed = np.asarray(net.node_type) != 0
    jm = ~fixed
    t = int(ref["t_sec"][0])
    d = net.demand_cfs_at(t)
    rh = np.where(fixed, ref["head_ft"][0], 0.0)
    opened_ref = ref["status"][0] > 0
    it_ref = int(ref["iterations"][0])

    # ---- §1 epanet vs DLL ----
    se = GGASolver(net, mode="epanet", inp_path=INP)
    r = se.solve(d, rh, status_machine=True)
    open_my = r["status"].numpy() > 2
    AD = np.abs(r["head_ft"].numpy() - ref["head_ft"][0])
    conn = conn_mask(net, open_my & opened_ref, fixed) & jm
    dH_all, dH_conn = AD[jm].max(), (AD[conn].max() if conn.any() else 0.0)
    q_api = np.where(open_my, r["flow_cfs"].numpy(), 0.0)
    dQ = np.abs(q_api - ref["flow_cfs"][0])[opened_ref].max()
    st_ok = bool(np.array_equal(open_my.astype(np.int8), ref["status"][0]))
    it_e = int(r["iters"])
    nz_h = int((AD[conn] != 0.0).sum())
    print("\n§1 epanet vs DLL（帧 t=%ds）：max|dH| conn=%.3e / all=%.3e ft "
          "(conn 非零元 %d/%d)  max|dQ|=%.3e cfs  iters=%d/%d  状态%s" %
          (t, dH_conn, dH_all, nz_h, int(conn.sum()), dQ, it_e, it_ref,
           "一致" if st_ok else "不一致"))
    print("    位级判定（<1e-6 ft/cfs + 状态/迭代数相等）: %s；内部 ft/cfs 逐位==0: %s" %
          ("PASS" if (dH_conn < 1e-6 and dQ < 1e-6 and st_ok
                      and it_e == it_ref) else "FAIL",
           "是" if dH_conn == 0.0 and dQ == 0.0 else "否"))
    # SI 网（CMH/米）的参考端要过 DLL API 的 Ucf 单位往返（epanet_ref.py:265
    # _ucf_head=MperFT：米出口 ÷0.3048 还原 ft），内部 ft 只能到 ≤1 ULP。
    # 判"逐位 0"要在 DLL 的用户单位出口上比：×Ucf 后逐位比较。
    from dgga.units import FLOW_UCF, MperFT
    H, Hr = r["head_ft"].numpy(), ref["head_ft"][0]
    Q, Qr = r["flow_cfs"].numpy(), ref["flow_cfs"][0]
    u = FLOW_UCF["CMH"]
    ulp = np.abs(H - Hr) / np.spacing(np.maximum(np.abs(H), np.abs(Hr)))
    print("    DLL 用户单位出口（米 / CMH）逐位相同：head %d/%d  flow %d/%d；"
          "内部 ft 的 ULP 距离 max=%.2f（÷Ucf 往返舍入）" %
          (int((H * MperFT == Hr * MperFT).sum()), H.size,
           int((Q * u == Qr * u).sum()), Q.size, float(ulp.max())))

    # ---- §2 dense(SM) vs DLL ----
    sd = GGASolver(net, mode="dense", inp_path=INP, dense_status_machine=True)
    with torch.no_grad():
        rd = sd.solve(d[None, :], rh[None, :], status_machine=True)
    open_d = rd["status"].numpy()[0] > 2
    dHd = np.abs(rd["head_ft"].numpy()[0] - ref["head_ft"][0])[jm].max()
    q_d = np.where(open_d, rd["flow_cfs"].numpy()[0], 0.0)
    dQd = np.abs(q_d - ref["flow_cfs"][0])[opened_ref].max()
    st_d = bool(np.array_equal(open_d.astype(np.int8), ref["status"][0]))
    it_d = int(rd["iters"][0])
    print("\n§2 dense(SM) vs DLL 同帧：max|dH|=%.3e ft  max|dQ|=%.3e cfs  "
          "iters=%d  状态%s  diag_ratio=%.3e" %
          (dHd, dQd, it_d, "一致" if st_d else "不一致",
           float(rd["diag_ratio"][0])))

    # ---- §3 CSR vs dense 装配（逐迭代 A + 终态，逐位）----
    caps = {"dense": [], "csr": []}
    orig = torch.linalg.cholesky_ex
    outs = {}
    for asm in ("dense", "csr"):
        def spy(A, *a, _k=asm, **kw):
            caps[_k].append(A.detach().clone())
            return orig(A, *a, **kw)
        torch.linalg.cholesky_ex = spy
        try:
            with torch.no_grad():
                outs[asm] = sd.solve(d[None, :], rh[None, :],
                                     status_machine=True, assemble=asm)
        finally:
            torch.linalg.cholesky_ex = orig
    nA = min(len(caps["dense"]), len(caps["csr"]))
    same_n = len(caps["dense"]) == len(caps["csr"])
    a_bit = all(torch.equal(caps["dense"][i], caps["csr"][i])
                for i in range(nA))
    maxdA = max((float((caps["dense"][i] - caps["csr"][i]).abs().max())
                 for i in range(nA)), default=0.0)
    o1, o2 = outs["dense"], outs["csr"]
    out_bit = all(torch.equal(o1[k], o2[k])
                  for k in ("head_ft", "flow_cfs", "emitter_cfs", "status",
                            "iters"))
    print("\n§3 CSR vs dense 装配（linear_solver 同为 dense）：逐迭代 A %d 轮"
          "%s逐位相同（max|dA|=%.1e）；终态 head/flow/emitter/status/iters "
          "%s逐位相同  =>  %s" %
          (nA, "" if (a_bit and same_n) else "**不**", maxdA,
           "" if out_bit else "**不**",
           "PASS" if (a_bit and same_n and out_bit) else "FAIL"))

    # ---- §4 三方迭代数 ----
    print("\n§4 三方迭代数（帧 0）：DLL=%d  epanet=%d  dense(SM)=%d  %s" %
          (it_ref, it_e, it_d,
           "相等" if it_ref == it_e == it_d else "不等"))

    # ---- §5 κ(A) 收敛帧（名义帧，tank 取中位）----
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                 dtype=np.float64))
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5 * (np.asarray(net.tank_hmin)[:tn.size]
                        + np.asarray(net.tank_hmax)[:tn.size])
    cap = {}

    def spy2(A, *a, **kw):
        cap["A"] = A.detach().clone()
        return orig(A, *a, **kw)
    torch.linalg.cholesky_ex = spy2
    try:
        with torch.no_grad():
            on = sd.solve(d0[None, :], rh0[None, :], status_machine=True)
    finally:
        torch.linalg.cholesky_ex = orig
    A = cap["A"].numpy()[0]
    k2 = float(np.linalg.cond(A, 2))
    nact = int((on["status"].numpy()[0][np.asarray(net.link_type) == 3] == 4)
               .sum())
    print("\n§5 名义帧（iters=%d converged=%s）收敛轮 A：cond2=%.3e  "
          "diag_ratio=%.3e  ACTIVE PRV=%d/3" %
          (int(on["iters"][0]), bool(on["converged"][0]), k2,
           float(on["diag_ratio"][0]), nact))
    # DLL 帧的 κ 一并给（同一表里两帧都会被引）
    cap.clear()
    torch.linalg.cholesky_ex = spy2
    try:
        with torch.no_grad():
            of = sd.solve(d[None, :], rh[None, :], status_machine=True)
    finally:
        torch.linalg.cholesky_ex = orig
    k2f = float(np.linalg.cond(cap["A"].numpy()[0], 2))
    nactf = int((of["status"].numpy()[0][np.asarray(net.link_type) == 3] == 4)
                .sum())
    print("    DLL 帧收敛轮 A：cond2=%.3e  ACTIVE PRV=%d/3" % (k2f, nactf))

    # ---- §6 参考解收紧 ----
    for tag, dd, rr in (("DLL帧", d, rh), ("名义帧", d0, rh0)):
        tr = tight_reference(se, dd, rr)
        print("\n§6 tight_reference（%s）：base 能量失衡 %.3e ft（iters0=%d）"
              " -> K_extra=%d 后 %.3e ft  met(<=1e-8)=%s  history=%s" %
              (tag, tr["base_energy_ft"], tr["iters0"], tr["K_extra"],
               tr["energy_ft"], tr["met"],
               [(k, "%.2e" % e) for k, e in tr["history"]]))
        # epanet 终态解在收紧参考下的头差（"INP 精度停机 vs 收紧解"的量级）
        rbase = se.run_gga(dd, rr, do_status=True)
        dtight = np.abs(rbase["head"] - tr["head"])[jm].max()
        print("    INP 精度停机解 vs 收紧解：max|dH|=%.3e ft" % dtight)
    print("\nCORRECT DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

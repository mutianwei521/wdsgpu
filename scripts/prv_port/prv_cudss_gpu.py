# -*- coding: utf-8 -*-
"""prv_cudss_gpu.py - PRV 轮 GPU 验收：cudss 通路（集群作业，自包含）。

含 PRV 网（L-TOWN / BWSN_Network_1 / D-Town / ky10）× B=32+逼阀 极端场景：
  · solve(status_machine=True, assemble="dense", linear_solver="dense")（GPU 批
    Cholesky）vs solve(..., assemble="csr", linear_solver="cudss")（cuDSS）：
    逐样本 iters 相等数 / 终态相等数 / 两路头差（连通掩码不做，报全 junction）/
    diag_ratio / 耗时。CSR 位型静态断言：PRV 槽位都在位（构造期已 assert）。
  · B=1 名义帧同口径。
判读：cudss 与 dense 在同一装配值上只差线性求解器；κ~1e10 下预期头差 ~1e-5 ft
量级；iters/终态的个别差异与 CPU 侧刀刃分岔同机制（如实报）。
torch.use_deterministic_algorithms(True)（CUDA scatter_add 原子序）。
"""
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch                                   # noqa: E402

torch.use_deterministic_algorithms(True)
from dgga.parse import parse_inp               # noqa: E402
from dgga.solver import GGASolver              # noqa: E402

NETS = [("L-TOWN", "networks_prv/L-TOWN.inp"),
        ("BWSN_Network_1", "networks_prv/BWSN_Network_1.inp"),
        ("D-Town", "networks_prv/D-Town.inp"),
        ("ky10", "networks_prv/ky10.inp")]
B = 32


def make_scenarios(net, B, seed, prv_links):
    import zlib                                            # noqa: F401
    rng = np.random.default_rng(seed)
    N = net.N
    nt = np.asarray(net.node_type)
    junc = np.where(nt == 0)[0]
    tanks = np.asarray(net.tank_node, dtype=np.int64)
    hmin = np.asarray(net.tank_hmin, dtype=np.float64)
    hmax = np.asarray(net.tank_hmax, dtype=np.float64)
    plen = max([len(p) for p in net.patterns], default=1)
    pstep = int(net.meta.get("pat_step_sec", 3600) or 3600)
    rows = []
    for b in range(B):
        t = int(rng.integers(0, max(plen, 1))) * pstep
        corner = (b % 4 == 0)
        mult = float(np.exp(rng.uniform(np.log(0.01), np.log(0.10)))) if corner \
            else float(np.exp(rng.uniform(np.log(0.05), np.log(4.0))))
        d = net.demand_cfs_at(t) * mult
        tot = float(np.abs(d[junc]).sum())
        if not corner:
            j = int(junc[rng.integers(0, junc.size)])
            d[j] += tot * float(rng.uniform(0.02, 0.60))
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(t),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            n = int(n)
            u = rng.random()
            if corner:
                rh[n] = hmax[i]
            elif u < 0.15:
                rh[n] = hmin[i]
            elif u < 0.30:
                rh[n] = hmax[i]
            else:
                lo, hi = float(hmin[i]), float(hmax[i])
                rh[n] = lo + (hi - lo) * float(rng.random()) if hi > lo \
                    else float(np.asarray(net.tank_h0)[i])
        rows.append((d, rh))
    d0 = net.demand_cfs_at(0)
    tot0 = float(np.abs(d0[junc]).sum())
    for k in prv_links:
        n2 = int(net.link_n2[k])
        d = d0 * 25.0
        if nt[n2] == 0:
            d[n2] += 15.0 * max(tot0, 1e-6)
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            rh[int(n)] = hmin[i]
        rows.append((d, rh))
        d = d0 * 0.2
        if nt[n2] == 0:
            d[n2] -= 1.0 * max(tot0, 1e-6)
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            rh[int(n)] = hmax[i]
        rows.append((d, rh))
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])


def run_one(name, rel):
    import zlib
    net = parse_inp(os.path.join(HERE, rel))
    prv = np.where(np.asarray(net.link_type) == 3)[0].tolist()
    D, RH = make_scenarios(net, B, zlib.crc32(name.encode()), prv)
    s = GGASolver(net, device="cuda", dtype=torch.float64, mode="dense",
                  inp_path=os.path.join(HERE, rel), dense_status_machine=True)
    jm = np.asarray(net.node_type) == 0

    def solve_path(asm, lin, dd, rr):
        excl = []
        keep = np.arange(dd.shape[0])
        while True:
            try:
                with torch.no_grad():
                    o = s.solve(dd[keep], rr[keep], status_machine=True,
                                assemble=asm, linear_solver=lin)
                return o, keep, excl
            except RuntimeError as e:
                import re
                m = re.search(r"样本 \[([0-9, ]+)\]", str(e))
                if not m:
                    raise
                bad = [int(x) for x in m.group(1).split(",")]
                excl += [int(keep[i]) for i in bad]
                keep = np.delete(keep, bad)

    t0 = time.time()
    od, keep_d, excl_d = solve_path("dense", "dense", D, RH)
    torch.cuda.synchronize()
    t_dense = time.time() - t0
    t0 = time.time()
    oc, keep_c, excl_c = solve_path("csr", "cudss", D[keep_d], RH[keep_d])
    torch.cuda.synchronize()
    t_cudss = time.time() - t0
    # cudss 侧再剔除的样本要在 dense 侧同样剔掉再比
    Sd = od["status"].cpu().numpy()[keep_c]
    Sc = oc["status"].cpu().numpy()
    itd = od["iters"].cpu().numpy()[keep_c]
    itc = oc["iters"].cpu().numpy()
    Hd = od["head_ft"].cpu().numpy()[keep_c]
    Hc = oc["head_ft"].cpu().numpy()
    n = Sd.shape[0]
    st_eq = int(sum(np.array_equal(Sd[b], Sc[b]) for b in range(n)))
    it_eq = int((itd == itc).sum())
    match = np.array([np.array_equal(Sd[b], Sc[b]) for b in range(n)])
    conv = od["converged"].cpu().numpy()[keep_c] & oc["converged"].cpu().numpy()
    m = match & conv
    dH = float(np.abs(Hd - Hc)[:, jm][m].max()) if m.any() else float("nan")
    print("[%s] n=%d(剔 dense=%s cudss=%s) iters相等 %d/%d 终态相等 %d/%d "
          "双收敛且状态同 %d：max|dH|=%.3e ft  diag_ratio(max)=%.2e  "
          "耗时 dense=%.1fs cudss=%.1fs" %
          (name, n, excl_d, excl_c, it_eq, n, st_eq, n, int(m.sum()), dH,
           float(od["diag_ratio"].max()), t_dense, t_cudss), flush=True)
    # B=1 名义帧
    d0 = net.demand_cfs_at(0)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                 dtype=np.float64))
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5 * (np.asarray(net.tank_hmin)[:tn.size]
                        + np.asarray(net.tank_hmax)[:tn.size])
    with torch.no_grad():
        o1 = s.solve(d0[None, :], rh0[None, :], status_machine=True)
        o2 = s.solve(d0[None, :], rh0[None, :], status_machine=True,
                     assemble="csr", linear_solver="cudss")
    same = np.array_equal(o1["status"].cpu().numpy(), o2["status"].cpu().numpy())
    dh1 = float(np.abs((o1["head_ft"] - o2["head_ft"]).cpu().numpy())[:, jm].max())
    print("       名义帧 B=1：iters %d/%d 状态%s max|dH|=%.3e ft" %
          (int(o1["iters"][0]), int(o2["iters"][0]),
           "同" if same else "不同", dh1), flush=True)
    s.cudss_free()
    return it_eq == n and st_eq == n


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    ok = True
    for name, rel in NETS:
        try:
            ok &= run_one(name, rel)
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok = False
            print("[%s] EXC %s" % (name, e), flush=True)
    print("PRV-CUDSS-RESULT all_exact=%s" % ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())

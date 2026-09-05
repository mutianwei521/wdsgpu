# -*- coding: utf-8 -*-
"""审计②：CSR 位型正确性 - 用 scipy 从拓扑独立重建 A 的位型，与上游存的
indptr/indices 逐元素对比；并检查行内列升序、对角落位、scatter 映射与 A_idx 一致。
覆盖全部参考网（位型在 __init__ 里建，与 mode 无关 ⇒ epanet-only 的大网也能验）
+ 合成病态网（孤立 junction / 并联管 / 自环）。"""
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audlib import REF, ROOT, Net, GGASolver, synth   # noqa: E402


def build_solver_any(net):
    """位型与 mode 无关：dense 建不出来就退 epanet，再退关守卫。"""
    for md in ("dense", "epanet"):
        for kw in ({}, dict(dense_tank_bound_check=False)):
            try:
                return GGASolver(net, mode=md, **kw), md
            except Exception:                             # noqa: BLE001
                continue
    return None, None


def scipy_pattern(net, Nj):
    """独立重建：非对角 (j1,j2)&(j2,j1)（两端皆 junction）、对角 (j1,j1)/(j2,j2)
    （该端为 junction）、以及全部 Nj 个对角。"""
    nt = np.asarray(net.node_type)
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    row = np.full(len(nt), -1, dtype=np.int64)
    row[nt == 0] = np.arange(Nj)
    m1, m2 = nt[n1] == 0, nt[n2] == 0
    both = m1 & m2
    R = np.concatenate([row[n1[both]], row[n2[both]], row[n1[m1]], row[n2[m2]],
                        np.arange(Nj)])
    C = np.concatenate([row[n2[both]], row[n1[both]], row[n1[m1]], row[n2[m2]],
                        np.arange(Nj)])
    M = sp.coo_matrix((np.ones(R.size), (R, C)), shape=(Nj, Nj)).tocsr()
    M.sum_duplicates()
    M.sort_indices()
    return M.indptr.astype(np.int64), M.indices.astype(np.int64)


def check(name, net):
    sv, md = build_solver_any(net)
    if sv is None:
        return [name, "-", "-", "构造失败"] + ["-"] * 6, 1
    Nj = sv.Nj
    ip = sv.A_csr_indptr.cpu().numpy().astype(np.int64)
    ic = sv.A_csr_indices.cpu().numpy().astype(np.int64)
    dp = sv.A_csr_dense_pos.cpu().numpy().astype(np.int64)
    sc = sv.A_csr_scatter.cpu().numpy().astype(np.int64)
    dg = sv.A_csr_diag.cpu().numpy().astype(np.int64)
    aix = sv.A_idx.cpu().numpy().astype(np.int64)
    sip, sic = scipy_pattern(net, Nj)

    r = {}
    r["scipy_ip"] = np.array_equal(ip, sip)
    r["scipy_ic"] = np.array_equal(ic, sic)
    r["dtype"] = (sv.A_csr_indptr.dtype == torch.int32
                  and sv.A_csr_indices.dtype == torch.int32
                  and sv.A_csr_scatter.dtype == torch.int64
                  and sv.A_csr_dense_pos.dtype == torch.int64
                  and sv.A_csr_diag.dtype == torch.int64)
    # 行内严格升序 + indptr 规范
    asc = all(np.all(np.diff(ic[ip[i]:ip[i + 1]]) > 0) for i in range(Nj))
    r["asc"] = bool(asc and ip[0] == 0 and ip[-1] == sv.A_csr_nnz
                    and ip.size == Nj + 1 and np.all(np.diff(ip) >= 0))
    # dense_pos ≡ row*Nj+col（CSR↔展平稠密一致）
    rows = np.repeat(np.arange(Nj), np.diff(ip))
    r["dpos"] = np.array_equal(dp, rows * Nj + ic)
    # 对角齐全且落位正确
    r["diag"] = np.array_equal(dp[dg], np.arange(Nj) * (Nj + 1))
    # **关键**：CSR 散射目标 == 稠密散射目标（逐元素），⇒ 两条通路加数序列同序同位
    r["map"] = np.array_equal(dp[sc], aix)
    # nnz = 独立位型的 nnz
    r["nnz"] = sv.A_csr_nnz == sic.size
    bad = sum(0 if v else 1 for v in r.values())
    return ([name, str(Nj), str(sv.A_csr_nnz), md]
            + ["OK" if r[k] else "**FAIL**" for k in
               ("scipy_ip", "scipy_ic", "asc", "dpos", "diag", "map")]), bad


def main():
    stems = sorted(os.path.basename(p)[:-8]
                   for p in glob.glob(os.path.join(REF, "*_net.npz")))
    rows, bad = [], 0
    for st in stems:
        r, b = check(st, Net.load(REF, st))
        rows.append(r)
        bad += b
    # 合成病态网
    cases = [("SYN net1 iso3", ("pub_net1", dict(iso=3))),
             ("SYN net1 iso3+ke", ("pub_net1", dict(iso=3, iso_ke=0.5))),
             ("SYN net1 par5", ("pub_net1", dict(par=5))),
             ("SYN net1 self3", ("pub_net1", dict(selfloop=3))),
             ("SYN net1 all", ("pub_net1", dict(iso=2, par=4, selfloop=2))),
             ("SYN hanoi all", ("pub_hanoi", dict(iso=3, par=6, selfloop=3))),
             ("SYN modena all", ("pub_modena", dict(iso=5, par=9, selfloop=4))),
             ("SYN ky4 all", ("pub_ky4", dict(iso=4, par=7, selfloop=3)))]
    for nm, (stem, kw) in cases:
        net, _ = synth(stem, **kw)
        r, b = check(nm, net)
        rows.append(r)
        bad += b

    hdr = ["网", "Nj", "nnz", "mode", "scipy.indptr", "scipy.indices",
           "行内升序", "dense_pos", "对角落位", "映射==A_idx"]
    w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(hdr)]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    print("-" * (sum(w) + 2 * len(w)))
    for r in rows:
        print("  ".join(r[i].ljust(w[i]) for i in range(len(hdr))))
    print(f"\n网数={len(rows)}  不合格项总数={bad}  "
          f"{'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    torch.set_num_threads(1)
    sys.exit(main())

# -*- coding: utf-8 -*-
"""aud_t4probe.py - 自写 T4 式探针：cuDSS 实际收到的 csr_data 在含 ACTIVE PRV
的帧上逐位对称（L-TOWN + BWSN，status_machine=True，这是上游 T4 网表没有的覆盖）。

用法：python3 aud_t4probe.py [pkg_dir]   pkg_dir 缺省 "."（其下的 dgga 被导入）。
转置槽映射自建（升序/有对偶/对合三断言）；拦 _cudss_forward 收每轮 data。
torch.use_deterministic_algorithms(True)（scatter_add 原子序固定）。
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else HERE
sys.path.insert(0, PKG)
import dgga                                     # noqa: E402
from dgga.parse import parse_inp               # noqa: E402
from dgga.solver import GGASolver              # noqa: E402

print("dgga from:", os.path.dirname(os.path.abspath(dgga.__file__)), flush=True)
torch.use_deterministic_algorithms(True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
DT = torch.float64


def tslots(s):
    Nj = int(s.Nj)
    row = s.A_csr_row.detach().cpu().numpy().astype(np.int64)
    col = s.A_csr_col.detach().cpu().numpy().astype(np.int64)
    key = row * Nj + col
    assert np.all(np.diff(key) > 0), "CSR 位型非严格升序"
    tk = col * Nj + row
    pos = np.searchsorted(key, tk)
    assert pos.max() < key.size and np.array_equal(key[pos], tk), "无对偶"
    assert np.array_equal(pos[pos], np.arange(key.size)), "非对合"
    return torch.as_tensor(pos, dtype=torch.int64, device=s.A_csr_row.device)


bad_total = 0
for stem in ("L-TOWN", "BWSN_Network_1"):
    inp = os.path.join(HERE, stem + ".inp")
    if not os.path.exists(inp):
        print("  %-16s SKIP（无 inp）" % stem)
        continue
    net = parse_inp(inp)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                 dtype=np.float64))
    tnn = np.asarray(net.tank_node, dtype=np.int64)
    if tnn.size:
        rh0[tnn] = 0.5 * (np.asarray(net.tank_hmin) + np.asarray(net.tank_hmax))
    rng = np.random.default_rng(909)
    B = 8
    D = torch.as_tensor(d0[None, :] * rng.uniform(0.8, 1.2, (B, net.N)),
                        dtype=DT, device="cuda")
    R = torch.as_tensor(np.repeat(rh0[None, :], B, axis=0), dtype=DT,
                        device="cuda")
    s = GGASolver(net, device="cuda", dtype=DT, mode="dense",
                  dense_status_machine=True)
    t = tslots(s)
    rec = dict(n=0, worst=0.0, nbad=0)
    orig = s._cudss_forward

    def probe(data, F, Bx, refine=None, slot=0, _o=orig, _r=rec, _t=t):
        w = float((data - data.index_select(1, _t)).abs().max())
        _r["n"] += 1
        _r["worst"] = max(_r["worst"], w)
        _r["nbad"] += int(w != 0.0)
        return _o(data, F, Bx, refine, slot)

    s._cudss_forward = probe
    with torch.no_grad():
        o = s.solve(D, R, status_machine=True, assemble="csr",
                    linear_solver="cudss")
    s._cudss_forward = orig
    st = o["status"].detach().cpu().numpy()
    lt = np.asarray(net.link_type)
    prv = np.where(lt == 3)[0]
    n_act = int((st[:, prv] == 4).sum()) if prv.size else 0
    ok = rec["n"] > 0 and rec["nbad"] == 0 and rec["worst"] == 0.0
    bad_total += 0 if ok else 1
    print("  %-16s 装配轮=%d ACTIVE-PRV槽(终态,B=8)=%d nnz=%d "
          "max|A-A^T|=%.3e 非对称轮=%d  %s"
          % (stem, rec["n"], n_act, s.A_csr_nnz, rec["worst"], rec["nbad"],
             "PASS" if ok else "<-- FAIL"), flush=True)
    s.cudss_free()
    del s
    torch.cuda.empty_cache()

print("AUD_T4PROBE_RESULT %s" % ("PASS" if bad_total == 0 else "FAIL"))
sys.exit(0 if bad_total == 0 else 1)

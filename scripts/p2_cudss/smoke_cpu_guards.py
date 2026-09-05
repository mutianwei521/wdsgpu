# -*- coding: utf-8 -*-
"""本机（无 CUDA）冒烟：缺省路径逐位不变 + cudss 的守卫全部明确 raise。"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/scripts/audit_p1_csr")
import numpy as np, torch
torch.set_num_threads(1)
from audlib import boundary, batchify, make_solver, bitfp
from dgga.parse import Net
from dgga.solver import GGASolver

REF = ROOT + "/data/reference"
for stem in ("pub_net1", "pub_hanoi", "pub_net3", "pub_modena", "pub_ky4"):
    net = Net.load(REF, stem)
    s = make_solver(net)
    d, rh = boundary(net, stem)
    D, R = batchify(d, rh, 4, 11)
    a = s.solve(D, R)
    b = s.solve(D, R, assemble="csr")
    c = s.solve(D, R, assemble="dense", linear_solver="dense")
    dH1 = float((a["head_ft"] - b["head_ft"]).abs().max())
    dH2 = float((a["head_ft"] - c["head_ft"]).abs().max())
    print(f"[{stem}] Nj={s.Nj} nnz={s.A_csr_nnz} iters={a['iters'].tolist()} "
          f"max|dH| csr={dH1:.3e} explicit-dense={dH2:.3e} fp={bitfp(a['head_ft'])}")
    assert dH1 == 0.0 and dH2 == 0.0
    B = 4
    g = torch.Generator().manual_seed(7)
    dat = torch.rand(B, s.A_csr_nnz, dtype=torch.float64, generator=g)
    x = torch.rand(B, s.Nj, dtype=torch.float64, generator=g)
    ref = torch.bmm(s._csr_to_dense(dat, B), x.unsqueeze(-1)).squeeze(-1)
    got = s._csr_spmv(dat, x, B)
    rel = float(((ref - got).abs() / ref.abs().clamp_min(1e-30)).max())
    print(f"        SpMV vs bmm: max|Δ|={float((ref - got).abs().max()):.3e} rel={rel:.3e}")
    assert rel < 1e-14

net = Net.load(REF, "pub_hanoi")
s = make_solver(net)
se = GGASolver(net, mode="epanet")
d, rh = boundary(net, "pub_hanoi")
s32 = GGASolver(net, mode="dense", dtype=torch.float32)
cases = [
    ("linear_solver='xyz'", lambda: s.solve(d, rh, linear_solver="xyz")),
    ("cudss + assemble 缺省(dense)", lambda: s.solve(d, rh, linear_solver="cudss")),
    ("cudss + mode='epanet'",
     lambda: se.solve(d, rh, assemble="csr", linear_solver="cudss")),
    ("cudss on CPU", lambda: s.solve(d, rh, assemble="csr", linear_solver="cudss")),
    ("cudss float32(CPU 先撞设备门)",
     lambda: s32.solve(d, rh, assemble="csr", linear_solver="cudss")),
]
for label, fn in cases:
    try:
        fn()
        print(f"  [FAIL] {label} 未抛错")
    except Exception as e:
        print(f"  [OK] {label} -> {type(e).__name__}: {str(e)[:78]}")
print("local smoke done")

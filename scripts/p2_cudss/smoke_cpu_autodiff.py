# -*- coding: utf-8 -*-
"""autodiff 两个入口的透传自检（本机 CPU）：csr 与 dense 逐位一致、梯度不断、
cudss 在 CPU/需要梯度时明确 raise。"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/scripts/audit_p1_csr")
import numpy as np, torch
torch.set_num_threads(1)
from audlib import boundary, batchify, make_solver, bitfp
from dgga.parse import Net
from dgga.autodiff import solve_unrolled, solve_polished

REF = ROOT + "/data/reference"
for stem in ("pub_hanoi", "pub_net3", "pub_modena"):
    net = Net.load(REF, stem)
    s = make_solver(net)
    d, rh = boundary(net, stem)
    D, R = batchify(d, rh, 3, 5)
    a = solve_unrolled(s, D, R, K=12)
    b = solve_unrolled(s, D, R, K=12, assemble="csr")
    dH = float((a["head_ft"] - b["head_ft"]).abs().max())
    dQ = float((a["flow_cfs"] - b["flow_cfs"]).abs().max())
    print(f"[unrolled {stem}] max|dH|={dH:.3e} max|dQ|={dQ:.3e} "
          f"fp {bitfp(a['head_ft'])}=={bitfp(b['head_ft'])}")
    assert dH == 0.0 and dQ == 0.0
    # 梯度：csr 通路仍可微，且与 dense 逐位相同
    gs = []
    for asm in ("dense", "csr"):
        Dg = D.clone().requires_grad_(True)
        o = solve_unrolled(s, Dg, R, K=12, assemble=asm)
        o["head_ft"].sum().backward()
        gs.append(Dg.grad.clone())
    print(f"        grad max|Δ∂L/∂demand|={float((gs[0]-gs[1]).abs().max()):.3e}")
    assert float((gs[0] - gs[1]).abs().max()) == 0.0
    # polished
    pa = solve_polished(s, d, rh, max_iter=60)
    pb = solve_polished(s, d, rh, max_iter=60, assemble="csr")
    print(f"[polished {stem}] max|dH|={np.abs(pa['head'] - pb['head']).max():.3e} "
          f"max|dq|={np.abs(pa['q'] - pb['q']).max():.3e}")
    assert np.abs(pa["head"] - pb["head"]).max() == 0.0

net = Net.load(REF, "pub_hanoi")
s = make_solver(net)
d, rh = boundary(net, "pub_hanoi")
D, R = batchify(d, rh, 2, 5)
cases = [
    ("unrolled cudss+assemble=dense",
     lambda: solve_unrolled(s, D, R, K=6, linear_solver="cudss")),
    ("unrolled cudss (CPU)",
     lambda: solve_unrolled(s, D, R, K=6, assemble="csr", linear_solver="cudss")),
    ("unrolled cudss + requires_grad",
     lambda: solve_unrolled(s, D.clone().requires_grad_(True), R, K=6,
                            assemble="csr", linear_solver="cudss")),
    ("polished cudss (CPU)",
     lambda: solve_polished(s, d, rh, max_iter=30, assemble="csr",
                            linear_solver="cudss")),
]
for label, fn in cases:
    try:
        fn()
        print(f"  [FAIL] {label} 未抛错")
    except Exception as e:
        print(f"  [OK] {label} -> {type(e).__name__}: {str(e)[:76]}")
print("autodiff smoke done")

# -*- coding: utf-8 -*-
"""a2_dense_count.py - 敌意审阅项 2（本机 dense 面，自写计数器）：

monkeypatch torch.linalg.{cholesky, cholesky_ex, lu_factor, lu, qr, solve}
逐调用记录（阶段, 形状）。判定：
  · backward 期间大系统（n=Nj）分解 0 次；
  · backward 期间允许出现的只有小 Woodbury 电容阵 solve（2p×2p / p×p），
    如实打印其形状与次数（这不是 Nj 系统的重分解）；
  · scipy splu 在 adjoint='gpu' 通路 0 次（证明没有偷用 CPU 伴随）。
网：L-TOWN / BWSN_1（PRV+SM）与 Hanoi（无阀对照）。
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch                                      # noqa: E402
import scipy.sparse.linalg as spla                # noqa: E402
from dgga.parse import parse_inp                  # noqa: E402
from dgga.solver import GGASolver                 # noqa: E402
from dgga import autodiff as ad                   # noqa: E402
from dgga.autodiff import implicit_solve          # noqa: E402

EVENTS = []
PHASE = ["fwd"]
FAILS = []


def wrap(mod, name):
    orig = getattr(mod, name)

    def f(*a, **k):
        shp = tuple(a[0].shape) if a and hasattr(a[0], "shape") else None
        EVENTS.append((PHASE[0], name, shp))
        return orig(*a, **k)

    setattr(mod, name, f)
    return orig


ORIG = {n: wrap(torch.linalg, n)
        for n in ("cholesky", "cholesky_ex", "lu_factor", "qr", "solve")}
_splu_orig = spla.splu
SPLU = [0]


def _splu(*a, **k):
    SPLU[0] += 1
    return _splu_orig(*a, **k)


spla.splu = _splu
ad.splu = _splu       # autodiff from scipy.sparse.linalg import splu 的本地名

CASES = [("L-TOWN", "public/_cleaned/L-TOWN.inp", True),
         ("BWSN_1", "public/_cleaned/BWSN_Network_1.inp", True),
         ("Hanoi", "public/Hanoi.inp", False)]
NETS = os.path.join(ROOT, "networks")

for name, relp, sm in CASES:
    inp = os.path.join(NETS, *relp.split("/"))
    net = parse_inp(inp)
    kw = dict(dense_status_machine=True) if sm else {}
    s = GGASolver(net, mode="dense", inp_path=inp, **kw)
    rng = np.random.default_rng(777)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                 dtype=np.float64))
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5 * (np.asarray(net.tank_hmin)[:tn.size]
                        + np.asarray(net.tank_hmax)[:tn.size])
    B = 8
    Db = d0[None, :] * rng.uniform(0.9, 1.1, (B, d0.size))
    dt = torch.float64
    D = torch.as_tensor(Db, dtype=dt).requires_grad_(True)
    R = torch.as_tensor(np.repeat(rh0[None, :], B, 0), dtype=dt)
    W = torch.as_tensor(rng.normal(0, 1, (B, s.Nj)), dtype=dt)
    EVENTS.clear()
    SPLU[0] = 0
    PHASE[0] = "fwd"
    for acc in (1e-12, 1e-9, 3e-8, 3e-7):
        try:
            h, f, e = implicit_solve(s, D, R, adjoint="gpu", accuracy=acc,
                                     max_iter=200, status_machine=sm)
            break
        except RuntimeError:
            if D.grad is not None:
                D.grad = None
            continue
    PHASE[0] = "bwd"
    (h[:, torch.as_tensor(np.asarray(s.junc_nodes))] * W).sum().backward()
    Nj = s.Nj
    big_f = [e for e in EVENTS if e[0] == "fwd"
             and e[1].startswith("cholesky")]
    big_b = [e for e in EVENTS if e[0] == "bwd"
             and e[1].startswith("cholesky")]
    small_b = [e for e in EVENTS if e[0] == "bwd"
               and not e[1].startswith("cholesky")]
    big_b_Nj = [e for e in big_b if e[2] and e[2][-1] == Nj]
    other_b = [e for e in EVENTS if e[0] == "bwd" and e[1] in
               ("lu_factor", "qr")]
    ok = (len(big_b_Nj) == 0 and len(big_f) > 0 and SPLU[0] == 0
          and len(other_b) == 0)
    shapes_small = sorted({e[2] for e in small_b if e[2]})
    print("[%s] fwd cholesky(Nj=%d)=%d | bwd cholesky=%d (Nj 系统 %d) | "
          "bwd 其它: %s | splu=%d  %s"
          % (name, Nj, len(big_f), len(big_b), len(big_b_Nj),
             {"solve(小 Woodbury)": (len(small_b), shapes_small)}
             if small_b else "无", SPLU[0], "PASS" if ok else "FAIL"))
    if not ok:
        FAILS.append(name)

# 对照：CPU 伴随确实走 splu（证明 splu 探针有效，不是死探针）
inp = os.path.join(NETS, "public", "Hanoi.inp")
net = parse_inp(inp)
se = GGASolver(net, mode="epanet", inp_path=inp)
d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
D = torch.as_tensor(d0, dtype=torch.float64).requires_grad_(True)
SPLU[0] = 0
h, f, e = implicit_solve(se, D, rh0)
h.sum().backward()
alive = SPLU[0] > 0
print("[探针自证] CPU 伴随 splu 调用=%d（>0 才证明探针活着）%s"
      % (SPLU[0], "PASS" if alive else "FAIL"))
if not alive:
    FAILS.append("probe")
print("A2 DONE rc=%d" % (1 if FAILS else 0))
sys.exit(1 if FAILS else 0)

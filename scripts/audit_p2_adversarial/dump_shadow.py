# -*- coding: utf-8 -*-
"""影子包对拍：用指定的 dgga 包根跑一整套缺省路径（dense + epanet + autodiff），
把每个数组原样存进 npz。用法：python dump_shadow.py <pkgroot> <out.npz>
两份 npz 逐位比较即可判定"缺省未变"。
"""
import os
import sys
import traceback

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
PKG, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, PKG)
import torch                                            # noqa: E402
from dgga.parse import parse_inp                        # noqa: E402
from dgga.solver import GGASolver                       # noqa: E402
import dgga.solver as _sm                               # noqa: E402
from dgga.autodiff import implicit_solve, solve_unrolled, solve_polished  # noqa: E402

print("dgga from", os.path.dirname(_sm.__file__))
import os as _os  # noqa: E402
R = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
NETS = [("Net1", R + "/networks/public/Net1.inp"),
        ("Net2", R + "/networks/public/Net2.inp"),
        ("Net3", R + "/networks/public/Net3.inp"),
        ("Hanoi", R + "/networks/public/Hanoi.inp"),
        ("Anytown", R + "/networks/public/Anytown.inp"),
        ("Fossolo", R + "/networks/public/Fossolo_poly1.inp"),
        ("Pescara", R + "/networks/public/Pescara.inp"),
        ("Modena", R + "/networks/public/Modena.inp"),
        ("city_d", R + "/networks/realInpData/city_d.inp")]
DT = torch.float64
res = {}


def put(k, v):
    res[k] = np.asarray(v)


for stem, f in NETS:
    try:
        net = parse_inp(f)
    except Exception:                                    # noqa: BLE001
        print("[parse-skip]", stem, traceback.format_exc().strip().split("\n")[-1][:90])
        continue
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    g = np.random.default_rng(31337)
    D4 = d[None, :] * g.uniform(0.7, 1.3, (4, d.size))
    R4 = rh[None, :] + g.uniform(-1.5, 1.5, (4, rh.size))
    R4 = np.where(np.isnan(rh)[None, :], np.nan, R4)
    jm = np.where(np.asarray(net.node_type) == 0)[0]
    ke = np.zeros(net.N)
    if jm.size:
        ke[jm[::7]] = 0.02 * float(np.abs(d).sum()) / max(1, jm[::7].size) / 6.3
    for mode in ("dense", "epanet"):
        try:
            s = GGASolver(net, mode=mode, inp_path=f, dtype=DT,
                          dense_tank_bound_check=False)
        except Exception:                                # noqa: BLE001
            print("[ctor-skip]", stem, mode,
                  traceback.format_exc().strip().split("\n")[-1][:90])
            continue
        for tag, kw, dd, rr in (("b1", {}, d, rh), ("b4", {}, D4, R4),
                                ("b1em", dict(ke_int=ke), d, rh),
                                ("b4em", dict(ke_int=ke), D4, R4)):
            try:
                o = s.solve(dd, rr, **kw)
                for k in ("head_ft", "flow_cfs", "emitter_cfs", "iters", "relerr"):
                    put("%s|%s|%s|%s" % (stem, mode, tag, k), o[k].numpy())
            except Exception:                            # noqa: BLE001
                put("%s|%s|%s|ERR" % (stem, mode, tag),
                    np.array(traceback.format_exc().strip().split("\n")[-1][:90]))
        if mode == "epanet":
            try:
                o = s.solve(D4, R4, status_machine=True)
                for k in ("head_ft", "flow_cfs", "iters", "status", "setting"):
                    put("%s|epanet|sm|%s" % (stem, k), o[k].numpy())
            except Exception:                            # noqa: BLE001
                put("%s|epanet|sm|ERR" % stem,
                    np.array(traceback.format_exc().strip().split("\n")[-1][:90]))
    # ---- autodiff 缺省通路（implicit + unrolled + polished）----
    try:
        s = GGASolver(net, mode="dense", inp_path=f, dtype=DT,
                      dense_tank_bound_check=False)
        dt_ = torch.tensor(D4, dtype=DT, requires_grad=True)
        rt_ = torch.tensor(R4, dtype=DT)
        h, q, e = implicit_solve(s, dt_, rt_)
        (h.sum() + q.sum()).backward()
        put("%s|ad|impl_h" % stem, h.detach().numpy())
        put("%s|ad|impl_q" % stem, q.detach().numpy())
        put("%s|ad|impl_g" % stem, dt_.grad.numpy())
        dt2 = torch.tensor(D4, dtype=DT, requires_grad=True)
        u = solve_unrolled(s, dt2, rt_, K=12)
        (u["head_ft"].sum() + u["flow_cfs"].sum()).backward()
        put("%s|ad|unr_h" % stem, u["head_ft"].detach().numpy())
        put("%s|ad|unr_g" % stem, dt2.grad.numpy())
        p = solve_polished(s, D4, R4)
        put("%s|ad|pol_h" % stem, p["head"])
        put("%s|ad|pol_q" % stem, p["q"])
    except Exception:                                    # noqa: BLE001
        put("%s|ad|ERR" % stem,
            np.array(traceback.format_exc().strip().split("\n")[-1][:90]))
    print("done", stem, flush=True)

np.savez(OUT, **res)
print("saved", OUT, len(res), "arrays")

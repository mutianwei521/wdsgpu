# -*- coding: utf-8 -*-
"""verify_exact_fix.py - 用 dgga.parse.exact_fixed_inputs_from_inp（新增的
非缺省函数）重跑五个受影响网的自主 EPS 对拍，给出改前/改后偏差。

用法: python scripts/exempt_diag/verify_exact_fix.py [stem ...]
"""

import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.parse import Net, exact_fixed_inputs_from_inp   # noqa: E402
from dgga.eps import EpsDriver                            # noqa: E402
from align import resolve_inp                             # noqa: E402

REF_DIR = os.path.join(ROOT, "data", "reference")
OUT_DIR = os.path.join(ROOT, "data", "exempt")
TOL_H, TOL_Q = 1e-6, 1e-6
STEMS = ["pub_anytown", "pub_net3", "pub_bwsn_network_1",
         "pub_bwsn_network_2", "pub_net6"]


def run(stem, apply_fix):
    net = Net.load(REF_DIR, stem)
    inp = resolve_inp(stem)
    nch = exact_fixed_inputs_from_inp(net, inp) if apply_fix else None
    ref = np.load(os.path.join(REF_DIR, f"{stem}_ref.npz"))
    drv = EpsDriver(net, inp_path=inp if os.path.isfile(inp) else None)
    s = drv.solver
    drv._inithyd()
    F = len(ref["t_sec"])
    dH, dQ = [], []
    st_all = it_all = t_all = True
    for f in range(F):
        t_all = t_all and int(drv.Htime) == int(ref["t_sec"][f])
        drv._demands()
        drv._controls()
        r = s.run_gga(drv.d, drv.H, q0=drv.q, e0=drv.e,
                      status0=drv.S, setting0=drv.K, do_status=True)
        drv.q, drv.e, drv.S = r["flow"], r["emitter"], r["status"]
        drv.K, drv.H = r["setting"], r["head"]
        drv.fixed_dem = r["fixed_demand"]
        drv.node_dem = np.where(s.is_fixed_node, drv.fixed_dem, drv.d + drv.e)
        op = drv.S > s.ST_CLOSED
        fa = np.where(op, drv.q, 0.0)
        opened = ref["status"][f] > 0
        dH.append(float(np.abs(drv.H - ref["head_ft"][f]).max()))
        dql = np.abs(fa - ref["flow_cfs"][f])
        dQ.append(float(dql[opened].max()) if opened.any() else 0.0)
        st_all = st_all and bool(np.array_equal(op.astype(np.int8), ref["status"][f]))
        it_all = it_all and int(r["iters"]) == int(ref["iterations"][f])
        drv._nexthyd(float(r["relerr"]))
    dH = np.asarray(dH)
    dQ = np.asarray(dQ)
    return {"F": int(F), "changed": nch, "max_dH": float(dH.max()),
            "argmax": int(dH.argmax()), "max_dQ": float(dQ.max()),
            "n_ge_tol": int((dH >= TOL_H).sum()), "status_all": bool(st_all),
            "iters_all": bool(it_all), "t_all": bool(t_all),
            "pass": bool(dH.max() < TOL_H and dQ.max() < TOL_Q and st_all and t_all),
            "bitwise_zero": bool(dH.max() == 0.0)}


def main(stems):
    rows = []
    for stem in stems:
        t0 = time.perf_counter()
        a = run(stem, False)
        b = run(stem, True)
        rows.append({"stem": stem, "before": a, "after": b})
        print(f"{stem:<24s} 改前 max|dH|={a['max_dH']:.3e} ft ({a['n_ge_tol']}/{a['F']} 帧超限, "
              f"{'PASS' if a['pass'] else 'FAIL'})  ->  改后 "
              f"max|dH|={b['max_dH']:.3e} ft, max|dQ|={b['max_dQ']:.3e} cfs, "
              f"逐位零={b['bitwise_zero']}, {'PASS' if b['pass'] else 'FAIL'}"
              f"  (改动 {b['changed']}, {time.perf_counter()-t0:.0f}s)", flush=True)
    p = os.path.join(OUT_DIR, "verify_exact_fix.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows},
                  f, ensure_ascii=False, indent=1)
    print("写出:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else STEMS))

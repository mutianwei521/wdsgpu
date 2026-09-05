# -*- coding: utf-8 -*-
"""ADVERSARIAL M4: is the "52/52 bit-identical" claim true?

The archived M4 evidence only covers the 5 networks whose deviation was
>= 1e-12 ft.  The published sweep table says only 20 of 52 agree EXACTLY;
the rest sit at 1.4e-14 .. 1.1e-13 ft.  If exact_fixed_inputs_from_inp does
not drive those to 0.0 as well, "52/52 bitwise" is an overclaim.

Independent script; does not import any exempt_diag helper.
"""
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.parse import Net, exact_fixed_inputs_from_inp   # noqa: E402
from dgga.eps import EpsDriver                            # noqa: E402
from align import resolve_inp                             # noqa: E402

REF = os.path.join(ROOT, "data", "reference")

# EPS-autonomous nets from data/benchmark_report.txt whose max|dH| is NOT 0.0
TARGETS = [
    ("EXA5", 5.684e-14),
    ("city_d", 1.421e-14),
    ("city_h", 1.421e-14),
    ("pub_c_town_batadal", 5.684e-14),
    ("pub_d_town", 5.684e-14),
    ("pub_l_town", 2.842e-14),
    ("pub_richmond_skeleton", 1.137e-13),
    ("pub_richmond_standard", 1.137e-13),
]
FIELD_SETS = [("none", None), ("res+vset", ("res", "vset")),
              ("res+vset+elev", ("res", "vset", "elev"))]


def eps_maxdh(stem, fields):
    net = Net.load(REF, stem)
    inp = resolve_inp(stem)
    nch = None
    if fields is not None:
        nch = exact_fixed_inputs_from_inp(net, inp, fields=fields)
    ref = np.load(os.path.join(REF, f"{stem}_ref.npz"))
    drv = EpsDriver(net, inp_path=inp if os.path.isfile(inp) else None)
    s = drv.solver
    drv._inithyd()
    F = len(ref["t_sec"])
    mdh = 0.0
    mdq = 0.0
    for f in range(F):
        drv._demands()
        drv._controls()
        r = s.run_gga(drv.d, drv.H, q0=drv.q, e0=drv.e, status0=drv.S,
                      setting0=drv.K, do_status=True)
        drv.q, drv.e, drv.S = r["flow"], r["emitter"], r["status"]
        drv.K, drv.H = r["setting"], r["head"]
        drv.fixed_dem = r["fixed_demand"]
        drv.node_dem = np.where(s.is_fixed_node, drv.fixed_dem, drv.d + drv.e)
        op = drv.S > s.ST_CLOSED
        fa = np.where(op, drv.q, 0.0)
        opened = ref["status"][f] > 0
        mdh = max(mdh, float(np.abs(drv.H - ref["head_ft"][f]).max()))
        dql = np.abs(fa - ref["flow_cfs"][f])
        if opened.any():
            mdq = max(mdq, float(dql[opened].max()))
        drv._nexthyd(float(r["relerr"]))
    return mdh, mdq, nch, F


def main():
    rows = []
    for stem, published in TARGETS:
        rec = {"stem": stem, "published_maxdH": published}
        for name, fields in FIELD_SETS:
            t0 = time.perf_counter()
            try:
                mdh, mdq, nch, F = eps_maxdh(stem, fields)
                rec[name] = {"max_dH": mdh, "max_dQ": mdq, "changed": nch,
                             "F": F, "bitzero": mdh == 0.0,
                             "sec": time.perf_counter() - t0}
            except Exception as exc:            # noqa: BLE001
                rec[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{stem:<24s} {name:<14s} {rec[name]}", flush=True)
        rows.append(rec)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "adv_m4_all52.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("written", out)


if __name__ == "__main__":
    main()

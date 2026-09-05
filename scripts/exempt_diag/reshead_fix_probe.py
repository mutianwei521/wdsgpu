# -*- coding: utf-8 -*-
"""reshead_fix_probe.py - 因果验证：把 reservoir 定水头改成 INP 原文的
strtod 值（而非 wntr 的 m→ft 往返值），我方与 DLL 的 EPS 偏差是否塌缩？

**不动 dgga/**：只在脚本里就地覆写 Net.elev_ft 的 reservoir 分量后重跑，
before/after 同一循环、同一门槛，逐帧对拍。

用法: python scripts/exempt_diag/reshead_fix_probe.py [stem ...]
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

from dgga.parse import Net                       # noqa: E402
from dgga.eps import EpsDriver                   # noqa: E402
from align import resolve_inp                    # noqa: E402
from fixedhead_bitcheck import raw_sections      # noqa: E402

REF_DIR = os.path.join(ROOT, "data", "reference")
OUT_DIR = os.path.join(ROOT, "data", "exempt")
TOL_H, TOL_Q, TOL_QC = 1e-6, 1e-6, 1e-5
SI = ("LPS", "LPM", "MLD", "CMH", "CMD")

STEMS = ["pub_net3", "pub_bwsn_network_1", "pub_bwsn_network_2", "pub_net6"]


def run_eps(stem, fix_reshead, max_frames=None):
    net = Net.load(REF_DIR, stem)
    inp = resolve_inp(stem)
    n_fixed = 0
    if fix_reshead:
        res_raw, _ = raw_sections(inp)
        hcf = 0.3048 if str(net.meta["flow_units"]) in SI else 1.0
        nt = np.asarray(net.node_type)
        el = np.asarray(net.elev_ft).copy()
        for i in np.where(nt == 1)[0]:
            nid = net.node_id[i]
            if nid in res_raw:
                v = float(res_raw[nid][1]) / hcf
                if v != el[i]:
                    n_fixed += 1
                el[i] = v
        net.elev_ft = el
    ref = np.load(os.path.join(REF_DIR, f"{stem}_ref.npz"))
    drv = EpsDriver(net, inp_path=inp if os.path.isfile(inp) else None)
    s = drv.solver
    drv._inithyd()
    F = len(ref["t_sec"]) if max_frames is None else min(max_frames, len(ref["t_sec"]))
    dHs, dQs, st_ok_all, it_ok_all, t_ok_all = [], [], True, True, True
    for f in range(F):
        t_ok_all = t_ok_all and int(drv.Htime) == int(ref["t_sec"][f])
        drv._demands()
        drv._controls()
        r = s.run_gga(drv.d, drv.H, q0=drv.q, e0=drv.e,
                      status0=drv.S, setting0=drv.K, do_status=True)
        drv.q, drv.e, drv.S = r["flow"], r["emitter"], r["status"]
        drv.K, drv.H = r["setting"], r["head"]
        drv.fixed_dem = r["fixed_demand"]
        drv.node_dem = np.where(s.is_fixed_node, drv.fixed_dem, drv.d + drv.e)
        open_api = drv.S > s.ST_CLOSED
        flow_api = np.where(open_api, drv.q, 0.0)
        opened = ref["status"][f] > 0
        dHs.append(float(np.abs(drv.H - ref["head_ft"][f]).max()))
        dQl = np.abs(flow_api - ref["flow_cfs"][f])
        dQs.append(float(dQl[opened].max()) if opened.any() else 0.0)
        st_ok_all = st_ok_all and bool(np.array_equal(
            open_api.astype(np.int8), ref["status"][f]))
        it_ok_all = it_ok_all and int(r["iters"]) == int(ref["iterations"][f])
        drv._nexthyd(float(r["relerr"]))
    dHs = np.asarray(dHs)
    dQs = np.asarray(dQs)
    return {"F": int(F), "n_reshead_fixed": n_fixed,
            "max_dH": float(dHs.max()), "argmax": int(dHs.argmax()),
            "median_dH": float(np.median(dHs)), "max_dQ": float(dQs.max()),
            "n_frames_ge_tol": int((dHs >= TOL_H).sum()),
            "status_all": bool(st_ok_all), "iters_all": bool(it_ok_all),
            "t_all": bool(t_ok_all),
            "pass": bool(dHs.max() < TOL_H and dQs.max() < TOL_Q
                         and st_ok_all and t_ok_all)}


def main(stems):
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": []}
    for stem in stems:
        print(f"\n=== {stem} ===", flush=True)
        t0 = time.perf_counter()
        a = run_eps(stem, False)
        b = run_eps(stem, True)
        print(f"  改前(wntr m→ft 往返): max|dH|={a['max_dH']:.3e} ft @帧{a['argmax']}"
              f"  max|dQ|={a['max_dQ']:.3e} cfs  超限帧={a['n_frames_ge_tol']}/{a['F']}"
              f"  判定={'PASS' if a['pass'] else 'FAIL'}", flush=True)
        print(f"  改后(INP 原文 strtod, 修 {b['n_reshead_fixed']} 个水库): "
              f"max|dH|={b['max_dH']:.3e} ft @帧{b['argmax']}"
              f"  max|dQ|={b['max_dQ']:.3e} cfs  超限帧={b['n_frames_ge_tol']}/{b['F']}"
              f"  判定={'PASS' if b['pass'] else 'FAIL'}", flush=True)
        print(f"  状态/迭代/时刻全等: 改前 {a['status_all']}/{a['iters_all']}/{a['t_all']}"
              f"  改后 {b['status_all']}/{b['iters_all']}/{b['t_all']}"
              f"   用时 {time.perf_counter() - t0:.1f}s", flush=True)
        out["rows"].append({"stem": stem, "before": a, "after": b})
    p = os.path.join(OUT_DIR, "reshead_fix_probe.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n写出:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else STEMS))

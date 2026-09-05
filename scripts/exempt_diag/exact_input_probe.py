# -*- coding: utf-8 -*-
"""exact_input_probe.py - 逐项定位"我方 1-ULP 输入偏差"的来源（模块四任务 2 根因）。

solver._apply_exact_props 已从 INP 原文位级重建 diam/len/r_hw/Km/需水/emitter，
但**没有覆盖**三类量，它们仍走 wntr 的 m 表示往返（x_m / 0.3048）：
  res - reservoir 定水头 base（直接是方程的边界条件）
  elev - 节点高程（DDA 下 junction 高程不进方程，但 PRV/PSV 的
          hset = El[n2] + setting 与 [CONTROLS] 的压力阈值 grade 用它）
  vset - PRV/PSV/FCV 的 setting（parse.py:634 已把该往返记为"发布审计欠账 d"）

本脚本把这三项**分别**、以及组合地换成 INP 原文 strtod 值，重跑自主 EPS，
看我方与官方参考解的偏差如何变化。**不动 dgga/**，只在脚本内就地覆写 Net。

用法: python scripts/exempt_diag/exact_input_probe.py <stem> [combo ...]
      combo ∈ {none,res,elev,vset,res+elev,res+vset,elev+vset,all}
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

REF_DIR = os.path.join(ROOT, "data", "reference")
OUT_DIR = os.path.join(ROOT, "data", "exempt")
TOL_H, TOL_Q = 1e-6, 1e-6
SI = ("LPS", "LPM", "MLD", "CMH", "CMD")
COMBOS = ["none", "res", "elev", "vset", "res+elev", "res+vset", "all"]


def scan_inp(inp):
    """[JUNCTIONS]/[RESERVOIRS]/[TANKS]/[VALVES] 原始 token。"""
    out = {"JUNCTIONS": {}, "RESERVOIRS": {}, "TANKS": {}, "VALVES": {}}
    sec = None
    with open(inp, "r", encoding="latin-1") as f:
        for raw in f:
            s0 = raw.strip()
            if not s0:
                continue
            if s0.startswith("["):
                sec = s0[1:s0.find("]")].strip().upper()
                continue
            body = s0.split(";", 1)[0].strip()
            if not body:
                continue
            tok = body.split()
            if sec in out:
                out[sec][tok[0]] = tok
    return out


def apply_fixes(net, inp, fixes):
    """就地把指定输入换成 INP 原文 strtod 值。返回被改动的元素数。"""
    raw = scan_inp(inp)
    hcf = 0.3048 if str(net.meta["flow_units"]) in SI else 1.0
    nt = np.asarray(net.node_type)
    n = {"res": 0, "elev": 0, "vset": 0}
    if "res" in fixes or "elev" in fixes:
        el = np.asarray(net.elev_ft).copy()
        for i in range(net.N):
            nid = net.node_id[i]
            if nt[i] == 1 and nid in raw["RESERVOIRS"]:
                if "res" in fixes or "elev" in fixes:
                    v = float(raw["RESERVOIRS"][nid][1]) / hcf
                    n["res"] += int(v != el[i])
                    el[i] = v
            elif nt[i] == 0 and nid in raw["JUNCTIONS"] and "elev" in fixes:
                v = float(raw["JUNCTIONS"][nid][1]) / hcf
                n["elev"] += int(v != el[i])
                el[i] = v
            elif nt[i] == 2 and nid in raw["TANKS"] and "elev" in fixes:
                v = float(raw["TANKS"][nid][1]) / hcf
                n["elev"] += int(v != el[i])
                el[i] = v
        net.elev_ft = el
    if "vset" in fixes:
        vs = np.asarray(net.valve_setting_user).copy()
        lt = np.asarray(net.link_type)
        for k in range(net.L):
            lid = net.link_id[k]
            if lt[k] >= 3 and lid in raw["VALVES"]:
                tok = raw["VALVES"][lid]
                if len(tok) > 5:
                    v = float(tok[5])
                    n["vset"] += int(v != vs[k])
                    vs[k] = v
        net.valve_setting_user = vs
    return n


def run_eps(stem, fixes, max_frames=None):
    net = Net.load(REF_DIR, stem)
    inp = resolve_inp(stem)
    nfix = apply_fixes(net, inp, fixes) if fixes else {"res": 0, "elev": 0, "vset": 0}
    ref = np.load(os.path.join(REF_DIR, f"{stem}_ref.npz"))
    drv = EpsDriver(net, inp_path=inp if os.path.isfile(inp) else None)
    s = drv.solver
    drv._inithyd()
    F = len(ref["t_sec"]) if max_frames is None else min(max_frames,
                                                         len(ref["t_sec"]))
    dHs, dQs = [], []
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
        dHs.append(float(np.abs(drv.H - ref["head_ft"][f]).max()))
        dQl = np.abs(fa - ref["flow_cfs"][f])
        dQs.append(float(dQl[opened].max()) if opened.any() else 0.0)
        st_all = st_all and bool(np.array_equal(op.astype(np.int8),
                                                ref["status"][f]))
        it_all = it_all and int(r["iters"]) == int(ref["iterations"][f])
        drv._nexthyd(float(r["relerr"]))
    dH = np.asarray(dHs)
    dQ = np.asarray(dQs)
    return {"F": int(F), "n_fixed": nfix,
            "max_dH": float(dH.max()), "argmax": int(dH.argmax()),
            "median_dH": float(np.median(dH)), "max_dQ": float(dQ.max()),
            "n_ge_tol": int((dH >= TOL_H).sum()),
            "status_all": bool(st_all), "iters_all": bool(it_all),
            "t_all": bool(t_all),
            "pass": bool(dH.max() < TOL_H and dQ.max() < TOL_Q
                         and st_all and t_all)}


def main(argv):
    stem = argv[0]
    combos = argv[1:] if len(argv) > 1 else COMBOS
    mf = None
    rows = []
    print(f"=== {stem} ===", flush=True)
    for c in combos:
        fixes = set() if c == "none" else (
            {"res", "elev", "vset"} if c == "all" else set(c.split("+")))
        t0 = time.perf_counter()
        r = run_eps(stem, fixes, mf)
        r["combo"] = c
        r["sec"] = time.perf_counter() - t0
        rows.append(r)
        print(f"  {c:<10s} 改动 res/elev/vset = "
              f"{r['n_fixed']['res']}/{r['n_fixed']['elev']}/{r['n_fixed']['vset']}"
              f"  max|dH|={r['max_dH']:.3e} ft @帧{r['argmax']}"
              f"  max|dQ|={r['max_dQ']:.3e}  超限帧={r['n_ge_tol']}/{r['F']}"
              f"  状态/迭代={r['status_all']}/{r['iters_all']}"
              f"  判定={'PASS' if r['pass'] else 'FAIL'}  ({r['sec']:.0f}s)",
              flush=True)
    p = os.path.join(OUT_DIR, f"exact_input_probe_{stem}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"stem": stem, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print("写出:", p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

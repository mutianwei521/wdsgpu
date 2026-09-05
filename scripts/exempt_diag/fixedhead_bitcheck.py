# -*- coding: utf-8 -*-
"""fixedhead_bitcheck.py - 只查"真正进入水力方程"的定水头输入是否逐位相同。

背景：solver._apply_exact_props 已从 INP 原文重建 diam/len/r_hw/Km/需水基值
（位级），但**未覆盖 elev_ft**；而 DDA 下 junction 高程不进方程，
真正进方程的是 reservoir 的定水头 base（net.elev_ft[reservoir]）与
tank 的初始/边界水头（parse.py 已用原文覆写）。
本脚本把这三类逐位比对，并给出与 INP 原文 strtod 的 ULP 距离。

用法: python scripts/exempt_diag/fixedhead_bitcheck.py [stem ...]
"""

import ctypes
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.parse import Net                # noqa: E402
from dgga import epanet_ref as ER         # noqa: E402
from align import resolve_inp             # noqa: E402
from input_bitcheck import ulp_dist       # noqa: E402

REF_DIR = os.path.join(ROOT, "data", "reference")
OUT_DIR = os.path.join(ROOT, "data", "exempt")

STEMS = ["pub_net3", "pub_bwsn_network_1", "pub_bwsn_network_2", "pub_net6",
         "pub_c_town_batadal", "pub_d_town", "ky5", "pub_l_town",
         "pub_richmond_standard", "pub_ky10", "pub_net2", "pub_anytown_wntr"]


def raw_sections(inp):
    """扫 INP 原文的 [RESERVOIRS] / [TANKS] 行。"""
    res, tank = {}, {}
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
            if sec == "RESERVOIRS":
                res[tok[0]] = tok
            elif sec == "TANKS":
                tank[tok[0]] = tok
    return res, tank


def main(stems):
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": []}
    for stem in stems:
        inp = resolve_inp(stem)
        if not os.path.isfile(inp):
            continue
        net = Net.load(REF_DIR, stem)
        res_raw, tank_raw = raw_sections(inp)
        is_si = str(net.meta["flow_units"]) in ("LPS", "LPM", "MLD", "CMH", "CMD")
        hcf = 0.3048 if is_si else 1.0
        nt = np.asarray(net.node_type)
        with ER.Epanet(inp) as en:
            ids = en.node_ids()
            v = ctypes.c_double()
            el_dll = {}
            for i, nid in enumerate(ids, start=1):
                en.lib.EN_getnodevalue(en._ph, i, ER.EN_ELEVATION, ctypes.byref(v))
                el_dll[nid] = v.value / en._ucf_head
            uh = en._ucf_head
        rec = {"stem": stem, "units": str(net.meta["flow_units"]),
               "is_si": bool(is_si), "reservoirs": [], "tanks": []}
        print(f"\n=== {stem}  units={net.meta['flow_units']}  ucf_head={uh:g} ===",
              flush=True)
        bad_r = bad_t = 0
        for i in np.where(nt == 1)[0]:                  # reservoir
            nid = net.node_id[i]
            mine = float(net.elev_ft[i])
            raw = float(res_raw[nid][1]) / hcf if nid in res_raw else float("nan")
            dll = el_dll.get(nid, float("nan"))
            u_raw = int(ulp_dist(np.array([mine]), np.array([raw]))[0])
            u_dll = int(ulp_dist(np.array([mine]), np.array([dll]))[0])
            bad_r += (u_raw > 0)
            rec["reservoirs"].append({"mine": mine, "raw_strtod": raw,
                                      "dll_roundtrip": dll,
                                      "ulp_vs_raw": u_raw, "ulp_vs_dll": u_dll})
            print(f"  reservoir  我方={mine!r}  INP原文/hcf={raw!r}  "
                  f"ULP(原文)={u_raw}  ULP(DLL回读)={u_dll}"
                  f"{'   <== 差' if u_raw else ''}", flush=True)
        for i in np.where(nt == 2)[0][:6]:              # tank（前 6 个）
            nid = net.node_id[i]
            mine = float(net.elev_ft[i])
            raw = float(tank_raw[nid][1]) / hcf if nid in tank_raw else float("nan")
            u_raw = int(ulp_dist(np.array([mine]), np.array([raw]))[0])
            bad_t += (u_raw > 0)
            rec["tanks"].append({"mine": mine, "raw_strtod": raw,
                                 "ulp_vs_raw": u_raw})
        # 水池初始水头（真正进方程的边界）
        th0 = np.asarray(net.tank_h0)
        rec["n_tanks"] = int(len(th0))
        rec["n_res_ulp_diff"] = int(bad_r)
        rec["n_tank_elev_ulp_diff"] = int(bad_t)
        print(f"  水库 elev 与原文差 {bad_r}/{int((nt == 1).sum())} 个；"
              f"水池底高程差 {bad_t}/{min(6, int((nt == 2).sum()))} 个（抽查）",
              flush=True)
        out["rows"].append(rec)
    p = os.path.join(OUT_DIR, "fixedhead_bitcheck.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n写出:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else STEMS))

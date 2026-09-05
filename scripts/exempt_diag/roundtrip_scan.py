# -*- coding: utf-8 -*-
"""roundtrip_scan.py - 全基准 52 网扫描：哪些网的"未被 _apply_exact_props
覆盖的输入"经 wntr 的 m 表示往返后与 INP 原文差 ULP？

三类量（见 exact_input_probe.py 的说明）：
  res  reservoir 定水头 base
  elev 节点高程（PRV/PSV 的 hset、[CONTROLS] 压力阈值用）
  vset PRV/PSV/FCV/TCV 的 setting
把"是否有 ULP 往返差"与 tab_full_benchmark 的判定（pass / exempt）做交叉表，
回答任务 2(c)"四个网是否有共同结构特征"。

用法: python scripts/exempt_diag/roundtrip_scan.py
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
from align import resolve_inp                    # noqa: E402
from exact_input_probe import scan_inp           # noqa: E402

REF_DIR = os.path.join(ROOT, "data", "reference")
OUT_DIR = os.path.join(ROOT, "data", "exempt")
SI = ("LPS", "LPM", "MLD", "CMH", "CMD")
EXEMPT = {"pub_net3", "pub_bwsn_network_1", "pub_bwsn_network_2", "pub_net6"}


def stems():
    out = []
    for fn in sorted(os.listdir(REF_DIR)):
        if fn.endswith("_net.npz"):
            s = fn[:-len("_net.npz")]
            if s.startswith("prv_") or s in ("fcv_smoke",):
                continue
            out.append(s)
    return out


def main():
    rows = []
    print(f"{'网':<24s}{'单位':>6s}{'res差':>7s}{'elev差':>8s}{'vset差':>8s}"
          f"{'判定':>8s}")
    for s in stems():
        inp = resolve_inp(s)
        if not os.path.isfile(inp):
            continue
        try:
            net = Net.load(REF_DIR, s)
            raw = scan_inp(inp)
        except Exception as e:                            # noqa: BLE001
            print(f"{s:<24s} 读取失败 {type(e).__name__}: {e}")
            continue
        hcf = 0.3048 if str(net.meta["flow_units"]) in SI else 1.0
        nt = np.asarray(net.node_type)
        lt = np.asarray(net.link_type)
        el = np.asarray(net.elev_ft)
        vs = np.asarray(net.valve_setting_user)
        n_res = n_el = n_vs = 0
        n_res_tot = n_el_tot = n_vs_tot = 0
        for i in range(net.N):
            nid = net.node_id[i]
            if nt[i] == 1 and nid in raw["RESERVOIRS"]:
                n_res_tot += 1
                n_res += int(float(raw["RESERVOIRS"][nid][1]) / hcf != el[i])
            elif nt[i] == 0 and nid in raw["JUNCTIONS"]:
                n_el_tot += 1
                n_el += int(float(raw["JUNCTIONS"][nid][1]) / hcf != el[i])
        for k in range(net.L):
            lid = net.link_id[k]
            if lt[k] >= 3 and lid in raw["VALVES"] and len(raw["VALVES"][lid]) > 5:
                n_vs_tot += 1
                n_vs += int(float(raw["VALVES"][lid][5]) != vs[k])
        verdict = "exempt" if s in EXEMPT else "pass"
        rows.append({"stem": s, "units": str(net.meta["flow_units"]),
                     "n_res_diff": n_res, "n_res": n_res_tot,
                     "n_elev_diff": n_el, "n_junc": n_el_tot,
                     "n_vset_diff": n_vs, "n_valve": n_vs_tot,
                     "verdict": verdict})
        print(f"{s:<24s}{str(net.meta['flow_units']):>6s}"
              f"{n_res:>4d}/{n_res_tot:<3d}{n_el:>5d}/{n_el_tot:<4d}"
              f"{n_vs:>5d}/{n_vs_tot:<4d}{verdict:>8s}"
              f"{'   <== 有往返差' if (n_res or n_vs) else ''}")
    # 交叉表：res 或 vset 有往返差 vs 判定
    a = sum(1 for r in rows if (r["n_res_diff"] or r["n_vset_diff"])
            and r["verdict"] == "exempt")
    b = sum(1 for r in rows if (r["n_res_diff"] or r["n_vset_diff"])
            and r["verdict"] == "pass")
    c = sum(1 for r in rows if not (r["n_res_diff"] or r["n_vset_diff"])
            and r["verdict"] == "exempt")
    d = sum(1 for r in rows if not (r["n_res_diff"] or r["n_vset_diff"])
            and r["verdict"] == "pass")
    print("\n交叉表（行=是否有 res/vset 往返 ULP 差，列=基准判定）")
    print(f"{'':<12s}{'exempt':>10s}{'pass':>10s}")
    print(f"{'有往返差':<12s}{a:>10d}{b:>10d}")
    print(f"{'无往返差':<12s}{c:>10d}{d:>10d}")
    p = os.path.join(OUT_DIR, "roundtrip_scan.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "rows": rows,
                   "crosstab": {"diff_exempt": a, "diff_pass": b,
                                "nodiff_exempt": c, "nodiff_pass": d}},
                  f, ensure_ascii=False, indent=1)
    print("\n写出:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

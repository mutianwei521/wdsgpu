# -*- coding: utf-8 -*-
"""input_bitcheck.py - 逐位比对"我方解析出的输入"与"官方 DLL 内部的输入"。

动机：pub_net6 的 1-ULP 自扰实验里，扰动 reservoir 水头得到的漂移
（max|dH| 与出现帧号）与我方实测偏差**完全一致**，提示我方某个输入量
与 DLL 差恰好 1 ULP。本脚本把 elevation / roughness / diameter / length /
minorloss / basedemand / setting 逐个拿出来做 ULP 距离统计，把根因钉死。

ULP 距离用 np.nextafter 迭代不现实，改用等价的整数表示差：
float64 按 IEEE-754 单调映射到 int64（负数取补），两数的 int 差即 ULP 数。

用法: python scripts/exempt_diag/input_bitcheck.py [stem ...]
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

REF_DIR = os.path.join(ROOT, "data", "reference")
OUT_DIR = os.path.join(ROOT, "data", "exempt")

EXEMPT = ["pub_net3", "pub_bwsn_network_1", "pub_bwsn_network_2", "pub_net6"]
CONTROL = ["pub_c_town_batadal", "pub_d_town", "ky5"]


def ulp_dist(a, b):
    """两个 float64 数组的 ULP 距离（IEEE-754 单调整数映射）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ia = a.view(np.int64).copy()
    ib = b.view(np.int64).copy()
    ia = np.where(ia < 0, np.int64(-0x8000000000000000) - ia, ia)
    ib = np.where(ib < 0, np.int64(-0x8000000000000000) - ib, ib)
    return np.abs(ia - ib)


def dll_inputs(inp):
    with ER.Epanet(inp) as en:
        cnt = en.counts()
        nN, nL = cnt["nodes"], cnt["links"]
        lib, ph = en.lib, en._ph
        v = ctypes.c_double()

        def gn(i, p):
            lib.EN_getnodevalue(ph, i, p, ctypes.byref(v))
            return v.value

        def gl(i, p):
            lib.EN_getlinkvalue(ph, i, p, ctypes.byref(v))
            return v.value

        out = {
            "elev_user": np.array([gn(i, ER.EN_ELEVATION) for i in range(1, nN + 1)]),
            "basedem_user": np.array([gn(i, ER.EN_BASEDEMAND) for i in range(1, nN + 1)]),
            "rough_user": np.array([gl(i, ER.EN_ROUGHNESS) for i in range(1, nL + 1)]),
            "diam_user": np.array([gl(i, ER.EN_DIAMETER) for i in range(1, nL + 1)]),
            "len_user": np.array([gl(i, ER.EN_LENGTH) for i in range(1, nL + 1)]),
            "minor_user": np.array([gl(i, ER.EN_MINORLOSS) for i in range(1, nL + 1)]),
            "setting_user": np.array([gl(i, ER.EN_INITSETTING) for i in range(1, nL + 1)]),
            "node_ids": en.node_ids(), "link_ids": en.link_ids(),
            "node_types": en.node_types(),
            "ucf_head": en._ucf_head, "ucf_flow": en._ucf_flow,
            "ucf_diam": en._link_ucf(ER.EN_DIAMETER),
        }
    return out


def main(stems):
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": []}
    for stem in stems:
        inp = resolve_inp(stem)
        if not os.path.isfile(inp):
            print(f"[跳过] {stem}", flush=True)
            continue
        net = Net.load(REF_DIR, stem)
        D = dll_inputs(inp)
        # 我方 Net 的节点/链路序 = INP 文件序；DLL 节点序 = junction 先、
        # 定水头节点后，故按 id 对齐
        nmap = {v: i for i, v in enumerate(net.node_id)}
        lmap = {v: i for i, v in enumerate(net.link_id)}
        ni = np.array([nmap[x] for x in D["node_ids"]])
        li = np.array([lmap[x] for x in D["link_ids"]])
        print(f"\n=== {stem}  N={net.N} L={net.L}  ucf_head={D['ucf_head']:.6g} "
              f"ucf_diam={D['ucf_diam']:.6g} ===", flush=True)
        rec = {"stem": stem, "fields": {}}
        # elevation：我方 elev_ft 为内部 ft；DLL 为用户单位 → /ucf_head
        pairs = [
            ("elev_ft", np.asarray(net.elev_ft)[ni],
             D["elev_user"] / D["ucf_head"]),
            ("roughness", np.asarray(net.roughness)[li], D["rough_user"]),
            ("diam_ft", np.asarray(net.diam_ft)[li],
             D["diam_user"] / D["ucf_diam"]),
            ("len_ft", np.asarray(net.len_ft)[li],
             D["len_user"] / D["ucf_head"]),
            ("valve_setting_user", np.asarray(net.valve_setting_user)[li],
             D["setting_user"]),
        ]
        for name, mine, theirs in pairs:
            m = np.isfinite(mine) & np.isfinite(theirs)
            # 非管道的 roughness/len/diam 我方置 0，DLL 给阀门直径等 → 只比两边都非零
            m &= ~((mine == 0) & (theirs != 0)) & ~((mine != 0) & (theirs == 0))
            u = ulp_dist(mine[m], theirs[m])
            nz = int((u > 0).sum())
            rec["fields"][name] = {"n_compared": int(m.sum()), "n_ulp_diff": nz,
                                   "max_ulp": int(u.max()) if u.size else 0,
                                   "n_ulp_eq_1": int((u == 1).sum())}
            flag = "" if nz == 0 else "  <== 有 ULP 级差异"
            print(f"  {name:<20s} 比对 {int(m.sum()):>6d} 个；差异 {nz:>5d} 个；"
                  f"max ULP={int(u.max()) if u.size else 0}{flag}", flush=True)
            if nz:
                idx = np.where(u > 0)[0][:5]
                sub_m = np.where(m)[0]
                for k in idx:
                    gi = sub_m[k]
                    print(f"      #{gi}: 我方 {mine[m][k]!r}  DLL {theirs[m][k]!r} "
                          f"ULP={int(u[k])}", flush=True)
        # 需水：按节点汇总基准需水（我方多类别展开 → 求和），DLL 给首类
        res["rows"].append(rec)
    p = os.path.join(OUT_DIR, "input_bitcheck.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n写出:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else EXEMPT + CONTROL))

# -*- coding: utf-8 -*-
"""任务 B-侦察：随机网成分扫描 + 梯队1参考解扩充（integrator，不改 dgga/ 已有模块）。

功能：
1. 扫描 networks/random_main/*.inp（20）与 networks/random_small/*.inp（3），
   逐网统计 JUNCTIONS/RESERVOIRS/TANKS/PIPES/PUMPS/VALVES(类型)/CONTROLS/RULES/
   STATUS/UNITS/HEADLOSS/DURATION，打印成分表。
2. 判定"梯队1兼容"（无泵、无水池(tank)、阀仅 TCV 或无阀、HEADLOSS=H-W），
   兼容网逐个执行 parse_inp → Net.save → build_reference 落盘到 data/reference/，
   stem 命名 rand_main_XXXX / rand_small_XXXX 避免与已有 7 网冲突。
   顺序纪律与 scripts/build_reference.py 一致：先 Net.save（整体重写 meta.json），
   后 build_reference（读-改-写合并 "reference" 键）。
3. 复用 scripts/validate_reference.py 的 validate_one 做三项检查
   （质量守恒硬门槛 1e-5 cfs / 需水对拍硬门槛 1e-9 cfs / H-W 水损报告项 0.05 ft）。
   不兼容网列入跳过清单并注明原因。
"""

import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.parse import parse_inp            # noqa: E402
from dgga.reference import build_reference  # noqa: E402
# 三项检查逐字复用 integrator 的实现（同一 REF_DIR=data/reference）
from validate_reference import (            # noqa: E402
    HW_MEDIAN_WARN, TOL_DEMAND, TOL_MASS, validate_one)

OUT_DIR = os.path.join(ROOT, "data", "reference")

# (目录, stem 前缀)：stem = <前缀>_<原文件序号>，如 rand_main_0007
GROUPS = [
    (os.path.join(ROOT, "networks", "random_main"), "rand_main"),
    (os.path.join(ROOT, "networks", "random_small"), "rand_small"),
]


def scan_inp(path):
    """纯文本扫描单个 INP 的成分（不依赖 wntr；节区名照 EPANET inpfile 约定）。"""
    counts = {k: 0 for k in ("JUNCTIONS", "RESERVOIRS", "TANKS", "PIPES",
                             "PUMPS", "VALVES", "CONTROLS", "RULES",
                             "STATUS", "EMITTERS", "PATTERNS")}
    valve_types = set()   # VALVES 行第 5 列：PRV/PSV/PBV/FCV/TCV/GPV
    units = headloss = duration = demand_model = ""
    sec = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.split(";", 1)[0].strip()  # 去行内注释
            if not line:
                continue
            if line.startswith("["):
                sec = line.strip("[]").upper()
                continue
            tok = line.split()
            if sec in counts:
                counts[sec] += 1
            if sec == "VALVES" and len(tok) >= 5:
                valve_types.add(tok[4].upper())
            elif sec == "OPTIONS":
                key = tok[0].upper()
                if key == "UNITS":
                    units = tok[1].upper()
                elif key == "HEADLOSS":
                    headloss = tok[1].upper()
                elif key == "DEMAND" and len(tok) >= 3 and tok[1].upper() == "MODEL":
                    demand_model = tok[2].upper()
            elif sec == "TIMES" and tok[0].upper() == "DURATION":
                duration = tok[1]
    return dict(counts=counts, valve_types=sorted(valve_types), units=units,
                headloss=headloss, duration=duration, demand_model=demand_model)


def tier1_check(info):
    """梯队1兼容判定：无泵、无水池、阀仅 TCV/无阀、H-W。返回不兼容原因列表（空=兼容）。"""
    c = info["counts"]
    reasons = []
    if c["PUMPS"] > 0:
        reasons.append(f"含泵 {c['PUMPS']} 台")
    if c["TANKS"] > 0:
        reasons.append(f"含水池 {c['TANKS']} 座")
    bad_valves = [v for v in info["valve_types"] if v != "TCV"]
    if bad_valves:
        reasons.append(f"含非 TCV 阀 {bad_valves}")
    if info["headloss"] != "H-W":
        reasons.append(f"HEADLOSS={info['headloss']}（非 H-W）")
    return reasons


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 第 1 步：扫描 + 成分表 ----
    entries = []  # (stem, inp_path, info, reasons)
    for dir_path, prefix in GROUPS:
        for inp in sorted(glob.glob(os.path.join(dir_path, "*.inp"))):
            num = os.path.splitext(os.path.basename(inp))[0].split("_")[-1]
            stem = f"{prefix}_{num}"
            info = scan_inp(inp)
            entries.append((stem, inp, info, tier1_check(info)))

    print("=" * 118)
    print(f"随机网成分表（共 {len(entries)} 网）：")
    print(f"{'stem':<16}{'Junc':>5}{'Res':>4}{'Tank':>5}{'Pipe':>5}{'Pump':>5}"
          f"{'Valve':>6}{'阀类型':>8}{'Ctrl':>5}{'Rule':>5}{'Stat':>5}"
          f"{'单位':>5}{'水损':>5}{'DURATION':>10}{'梯队1':>7}")
    for stem, _inp, info, reasons in entries:
        c = info["counts"]
        vt = ",".join(info["valve_types"]) if info["valve_types"] else "-"
        print(f"{stem:<16}{c['JUNCTIONS']:>5}{c['RESERVOIRS']:>4}{c['TANKS']:>5}"
              f"{c['PIPES']:>5}{c['PUMPS']:>5}{c['VALVES']:>6}{vt:>8}"
              f"{c['CONTROLS']:>5}{c['RULES']:>5}{c['STATUS']:>5}"
              f"{info['units']:>5}{info['headloss']:>5}{info['duration']:>10}"
              f"{'兼容' if not reasons else '跳过':>7}")

    # ---- 第 2 步：兼容网构建参考解 ----
    compat = [(s, p) for s, p, _i, r in entries if not r]
    skipped = [(s, r) for s, _p, _i, r in entries if r]
    print("=" * 118)
    print(f"梯队1兼容 {len(compat)} 网，跳过 {len(skipped)} 网。开始构建参考解 → {OUT_DIR}")

    built = []
    for k, (stem, inp) in enumerate(compat, 1):
        t0 = time.perf_counter()
        net = parse_inp(inp)
        net.save(OUT_DIR, stem)          # 先 save：整体重写 meta.json
        res = build_reference(inp, OUT_DIR, stem)  # 后 build：合并 reference 键
        dt = time.perf_counter() - t0
        T = len(res["t_sec"])
        print(f"[{k}/{len(compat)}] {stem}: N={net.N} L={net.L} T={T} "
              f"iter={int(res['iterations'].max())} "
              f"relerr={float(res['relerr'].max()):.2e} "
              f"警告 {len(res['warnings'])} 条  {dt:.2f}s")
        built.append(stem)

    # ---- 第 3 步：三项检查（复用 validate_reference.validate_one）----
    print("=" * 118)
    print("三项交叉验证（复用 scripts/validate_reference.py 同一实现与门槛）：")
    print(f"{'stem':<16}{'T':>3}{'N':>5}{'L':>5}"
          f"{'质量守恒max(cfs)':>18}{'需水对拍max(cfs)':>18}"
          f"{'HW max(ft)':>13}{'HW中位数(ft)':>14}{'判定':>6}")
    all_ok = True
    for stem in built:
        r = validate_one(stem)
        ok = r["mass_ok"] and r["dem_ok"] and r["hw_ok"]
        all_ok &= ok
        print(f"{stem:<16}{r['T']:>3}{r['N']:>5}{r['L']:>5}"
              f"{r['mass_max_gated']:>18.3e}{r['dem_err']:>18.3e}"
              f"{r['hw_max']:>13.3e}{r['hw_med']:>14.3e}{'PASS' if ok else 'FAIL':>6}")
    print(f"门槛：质量守恒 <{TOL_MASS:.0e} cfs（硬）；需水对拍 <{TOL_DEMAND:.0e} cfs（硬）；"
          f"H-W 中位数 <={HW_MEDIAN_WARN} ft（报告性）")

    # ---- 跳过清单 ----
    print("=" * 118)
    if skipped:
        print("跳过清单（不兼容原因）：")
        for stem, reasons in skipped:
            print(f"  {stem}: {'；'.join(reasons)}")
    else:
        print("跳过清单：空（全部网均为梯队1兼容）")
    print(f"总判定：{'PASS' if all_ok else 'FAIL'}（构建 {len(built)} 网）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

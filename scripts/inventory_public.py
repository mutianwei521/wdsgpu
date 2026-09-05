# -*- coding: utf-8 -*-
"""inventory_public.py - 纯文本扫描 networks/public/*.inp，输出特性清单。

不依赖 wntr/epanet，逐行解析 INP 节区（与 EPANET input3.c 的节名一致）：
节点/管段规模、UNITS、HEADLOSS、阀门类型分布、泵类型、水池/体积曲线、
CONTROLS/RULES 条数、DEMAND MODEL、EMITTERS、PATTERNS、DURATION。
结果存 data/public_inventory.json 并打印中文表。

能力判定基准（当前 dgga 求解器已验收能力）：
  HEADLOSS: H-W；UNITS: LPS/GPM；链路: PIPE/CV/TCV/PRV/PSV/泵/水池；
  CONTROLS+RULES；DEMAND MODEL: DDA。
缺口类别: D-W / C-M 水头损失、非 LPS/GPM 单位、FCV/PBV/GPV 阀、
  PDA 需水模型、EMITTERS 喷射器、水池体积曲线。
"""
import sys, os, json, re
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB = os.path.join(ROOT, "networks", "public")
OUT = os.path.join(ROOT, "data", "public_inventory.json")

# 下载来源登记（URL、许可）；本地 wntr 拷贝亦登记
SOURCES = {
    "Net1.inp":  ("wntr 4.x site-packages library/networks", "WNTR BSD-3 / EPA 公有领域"),
    "Net2.inp":  ("wntr 4.x site-packages library/networks", "WNTR BSD-3 / EPA 公有领域"),
    "Net3.inp":  ("wntr 4.x site-packages library/networks", "WNTR BSD-3 / EPA 公有领域"),
    "Net6.inp":  ("wntr 4.x site-packages library/networks", "WNTR BSD-3 / EPA 公有领域"),
    "ky4.inp":   ("wntr 4.x site-packages library/networks (原 uknowledge.uky.edu/wdst)", "WNTR BSD-3"),
    "ky10.inp":  ("wntr 4.x site-packages library/networks (原 uknowledge.uky.edu/wdst)", "WNTR BSD-3"),
    "Anytown_wntr.inp": ("wntr tests/networks_for_testing", "WNTR BSD-3"),
    "L-TOWN.inp": ("https://zenodo.org/api/records/4017659/files/L-TOWN.inp/content", "CC-BY-4.0 (BattLeDIM)"),
    "L-TOWN_Real.inp": ("https://zenodo.org/api/records/4017659/files/L-TOWN_Real.inp/content", "CC-BY-4.0 (BattLeDIM)"),
    "BWSN_Network_1.inp": ("raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/master/asce-tf-wdst/Battle of the Water Sensor Networks/", "仓库未声明许可(学术基准)"),
    "BWSN_Network_2.inp": ("raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/master/asce-tf-wdst/Battle of the Water Sensor Networks/", "仓库未声明许可(学术基准)"),
    "Anytown.inp": ("raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/master/asce-tf-wdst/Anytown/", "仓库未声明许可(学术基准)"),
    "Balerma.inp": ("raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/master/asce-tf-wdst/Balerma/", "仓库未声明许可(学术基准)"),
    "Hanoi.inp":   ("raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/master/asce-tf-wdst/Hanoi/", "仓库未声明许可(学术基准)"),
    "Fossolo_poly1.inp": ("raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/master/asce-tf-wdst/Fosspoly1/foss_poly_1.inp", "仓库未声明许可(学术基准)"),
    "Richmond_standard.inp": ("raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/master/collect-epanet-inp/", "仓库未声明许可(Exeter CWS 基准)"),
    "Richmond_skeleton.inp": ("raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/master/collect-epanet-inp/", "仓库未声明许可(Exeter CWS 基准)"),
    "D-Town.inp": ("raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/master/exeter-benchmarks/D-Town Water Distribution Network BWN-II/d-town.inp", "仓库未声明许可(BWN-II 基准)"),
    "C-Town_BATADAL.inp": ("raw.githubusercontent.com/scy-phy/www.batadal.net/master/data/CTOWN.INP", "CC-BY-4.0"),
    "Modena.inp": ("raw.githubusercontent.com/WaterFutures/WaterBenchmarkHub/main/docs/static/benchmarks/network-modena/modena.inp", "MIT(仓库)/Bragalli et al. 2008"),
    "Pescara.inp": ("raw.githubusercontent.com/WaterFutures/WaterBenchmarkHub/main/docs/static/benchmarks/network- Pescara/PES.inp", "MIT(仓库)/Bragalli et al. 2008"),
}

SUPPORTED_UNITS = {"LPS", "GPM"}
SUPPORTED_VALVES = {"TCV", "PRV", "PSV"}


def scan_inp(path):
    info = {
        "junctions": 0, "reservoirs": 0, "tanks": 0,
        "pipes": 0, "cv_pipes": 0, "pumps": 0, "valves": 0,
        "valve_types": {}, "pump_types": {"HEAD": 0, "POWER": 0, "SPEED/PATTERN": 0},
        "tank_volcurves": 0, "units": None, "headloss": None,
        "demand_model": "DDA", "controls": 0, "rules": 0,
        "emitters": 0, "patterns": 0, "duration": None,
        "quality": None,
    }
    section = None
    pattern_ids = set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                section = line.strip("[]").upper().strip()
                continue
            tok = line.split()
            up = [t.upper() for t in tok]
            if section == "JUNCTIONS":
                info["junctions"] += 1
            elif section == "RESERVOIRS":
                info["reservoirs"] += 1
            elif section == "TANKS":
                info["tanks"] += 1
                # ID Elev InitLvl MinLvl MaxLvl Diam MinVol [VolCurve] [Overflow]
                if len(tok) >= 8 and tok[7].upper() not in ("*",):
                    info["tank_volcurves"] += 1
            elif section == "PIPES":
                info["pipes"] += 1
                if len(tok) >= 8 and up[-1] == "CV":
                    info["cv_pipes"] += 1
            elif section == "PUMPS":
                info["pumps"] += 1
                if "HEAD" in up:
                    info["pump_types"]["HEAD"] += 1
                elif "POWER" in up:
                    info["pump_types"]["POWER"] += 1
                if "SPEED" in up or "PATTERN" in up:
                    info["pump_types"]["SPEED/PATTERN"] += 1
            elif section == "VALVES":
                info["valves"] += 1
                if len(tok) >= 5:
                    vt = tok[4].upper()
                    info["valve_types"][vt] = info["valve_types"].get(vt, 0) + 1
            elif section == "OPTIONS":
                if up[0] == "UNITS" and len(tok) >= 2:
                    info["units"] = tok[1].upper()
                elif up[0] == "HEADLOSS" and len(tok) >= 2:
                    info["headloss"] = tok[1].upper()
                elif up[0] == "DEMAND" and len(up) >= 3 and up[1] == "MODEL":
                    info["demand_model"] = up[2]
                elif up[0] == "QUALITY" and len(tok) >= 2:
                    info["quality"] = tok[1].upper()
            elif section == "CONTROLS":
                info["controls"] += 1
            elif section == "RULES":
                if up[0] == "RULE":
                    info["rules"] += 1
            elif section == "EMITTERS":
                info["emitters"] += 1
            elif section == "PATTERNS":
                pattern_ids.add(tok[0])
            elif section == "TIMES":
                if up[0] == "DURATION":
                    info["duration"] = " ".join(tok[1:])
    info["patterns"] = len(pattern_ids)
    info["nodes_total"] = info["junctions"] + info["reservoirs"] + info["tanks"]
    info["links_total"] = info["pipes"] + info["pumps"] + info["valves"]
    return info


def capability_gaps(info):
    gaps = []
    hl = (info["headloss"] or "H-W")
    if hl not in ("H-W", "HW"):
        gaps.append("水头损失 %s" % hl)
    un = info["units"] or "GPM"
    if un not in SUPPORTED_UNITS:
        gaps.append("单位 %s" % un)
    for vt, n in sorted(info["valve_types"].items()):
        if vt not in SUPPORTED_VALVES:
            gaps.append("阀门 %s(x%d)" % (vt, n))
    if info["demand_model"] == "PDA":
        gaps.append("需水模型 PDA")
    if info["emitters"] > 0:
        gaps.append("EMITTERS(x%d)" % info["emitters"])
    if info["tank_volcurves"] > 0:
        gaps.append("水池体积曲线(x%d)" % info["tank_volcurves"])
    return gaps


def main():
    rows = {}
    for fn in sorted(os.listdir(PUB)):
        if not fn.lower().endswith(".inp"):
            continue
        p = os.path.join(PUB, fn)
        info = scan_inp(p)
        info["file_size"] = os.path.getsize(p)
        src = SOURCES.get(fn, ("(未登记)", "(未知)"))
        info["source_url"] = src[0]
        info["license"] = src[1]
        info["gaps"] = capability_gaps(info)
        info["ready"] = (len(info["gaps"]) == 0)
        rows[fn] = info

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # 中文表
    hdr = ("网络", "节点", "管段", "单位", "损失", "阀门", "泵", "水池", "控制", "规则", "PAT", "历时", "需水", "喷射", "判定")
    print(("%-22s" + "%6s%6s" + "%6s%6s" + " %-16s" + "%4s%5s%5s%5s%5s" + " %-9s" + "%5s%5s" + "  %s") % hdr)
    for fn, r in rows.items():
        vd = ",".join("%s:%d" % kv for kv in sorted(r["valve_types"].items())) or "-"
        verdict = "立即可跑" if r["ready"] else "缺:" + ";".join(r["gaps"])
        print(("%-22s%6d%6d%6s%6s %-16s%4d%5d%5d%5d%5d %-9s%5d%5d  %s") % (
            fn[:22], r["nodes_total"], r["links_total"], r["units"] or "?",
            r["headloss"] or "?", vd[:16], r["pumps"], r["tanks"],
            r["controls"], r["rules"], r["patterns"],
            (r["duration"] or "?")[:9], r["demand_model"] == "PDA" and 1 or 0,
            r["emitters"], verdict))

    # 缺口频次
    freq = {}
    for r in rows.values():
        for g in r["gaps"]:
            key = re.sub(r"\(x\d+\)", "", g)
            freq[key] = freq.get(key, 0) + 1
    print("\n能力缺口频次（按出现网络数排序）:")
    for k, v in sorted(freq.items(), key=lambda kv: -kv[1]):
        print("  %-24s %d 个网络" % (k, v))
    n_ready = sum(1 for r in rows.values() if r["ready"])
    print("\n共 %d 个网络，立即可跑 %d 个；JSON 已存 %s" % (len(rows), n_ready, OUT))


if __name__ == "__main__":
    main()

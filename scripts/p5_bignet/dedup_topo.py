# -*- coding: utf-8 -*-
"""任务一·去重：字节哈希只抓到 7 组，但这批 collection 里大量副本是"同一个网、
不同字节"（改了 [TIMES]/[REPORT]/坐标/注释）。这里补一层**拓扑指纹**：

  topo = sha256( sorted(node_id) ‖ sorted((link_id, id(n1), id(n2))) ‖ N ‖ L )

拓扑指纹相同 = 同一个网的不同存档；再看水力属性指纹（直径/长度/糙率/需水基值）
区分"纯排版差异"与"同拓扑不同设计"（Design 类基准的多份解就是后者）。
"""
import hashlib
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from p5lib import dgga_provenance  # noqa: E402

import numpy as np  # noqa: E402
from dgga.parse import parse_inp  # noqa: E402


def fingerprints(path):
    net = parse_inp(path)
    nid = sorted(net.node_id)
    lid = sorted(f"{a}|{net.node_id[int(b)]}|{net.node_id[int(c)]}"
                 for a, b, c in zip(net.link_id, net.link_n1, net.link_n2))
    topo = hashlib.sha256(
        ("\x1f".join(nid) + "\x1e" + "\x1f".join(lid) +
         f"\x1e{net.N}\x1e{net.L}").encode("utf-8")).hexdigest()[:16]
    hyd = hashlib.sha256(b"".join(
        np.ascontiguousarray(a, dtype=np.float64).tobytes()
        for a in (net.diam_ft, net.len_ft, net.roughness, net.km_int,
                  net.elev_ft, net.dem_base_cfs))).hexdigest()[:16]
    return topo, hyd


def main():
    print("dgga imported from:", dgga_provenance(), flush=True)
    surv = json.load(open(os.path.join(HERE, "survey_inp.json"), encoding="utf-8"))
    out = []
    for r in surv:
        rec = dict(path=r["path"], sha256=r["sha256"], parse=r["parse"],
                   Nj=r.get("Nj"))
        if r["parse"] == "ok":
            try:
                rec["topo"], rec["hyd"] = fingerprints(os.path.join(ROOT, r["path"]))
            except Exception as e:
                rec["topo"] = rec["hyd"] = f"ERR:{type(e).__name__}"
        out.append(rec)
    p = os.path.join(HERE, "dedup_topo.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", p)

    groups = {}
    for r in out:
        if r.get("topo", "").startswith("ERR") or "topo" not in r:
            continue
        groups.setdefault(r["topo"], []).append(r)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"\n拓扑唯一网数 = {len(groups)}（可解析 {sum(len(v) for v in groups.values())} 个文件）")
    print(f"同拓扑多副本组 = {len(dup)}")
    for k, v in sorted(dup.items(), key=lambda x: -len(x[1])):
        hyds = {r["hyd"] for r in v}
        kind = "字节不同但水力属性也相同（纯排版/元数据差异）" if len(hyds) == 1 \
            else f"同拓扑但水力属性有 {len(hyds)} 种（不同设计解/工况）"
        print(f"\n  topo={k}  Nj={v[0]['Nj']}  {len(v)} 份 - {kind}")
        for r in v:
            print(f"      [{r['sha256']}/{r['hyd']}] {r['path']}")


if __name__ == "__main__":
    main()

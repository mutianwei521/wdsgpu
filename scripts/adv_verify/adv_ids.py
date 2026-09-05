# -*- coding: utf-8 -*-
"""ADVERSARIAL: element-identifier leakage scan over this round's readable output.

Two passes, because City D's own ids ARE bare integers and a naive match on
those is meaningless:
  (a) non-numeric ids (L-TOWN etc.) matched as whole tokens - a real leak test;
  (b) numeric ids matched only when they follow an element word
      (node/junction/link/pipe/sensor/节点/管/传感器/漏点) - catches the way a
      leak would actually be written.
"""
import glob
import os
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
REF = os.path.join(ROOT, "data", "reference")

ids_alpha, ids_num = set(), set()
for stem in ("city_d", "pub_l_town", "pub_l_town_real", "city_h"):
    p = os.path.join(REF, f"{stem}_meta.json")
    if not os.path.isfile(p):
        continue
    import json; z = json.load(open(p, encoding="utf-8"))
    for k in ("node_id", "link_id"):
        if k in z:
            for v in np.asarray(z[k]).tolist():
                v = str(v)
                (ids_num if v.isdigit() else ids_alpha).add(v)
print(f"collected ids: {len(ids_alpha)} non-numeric, {len(ids_num)} numeric")

TARGETS = ["data/cluster_localisation_wip.txt", "data/placement_metric_wip.txt",
           "data/optimizer_v2_wip.txt"]
TARGETS += [os.path.relpath(p, ROOT).replace("\\", "/")
            for p in glob.glob(os.path.join(ROOT, "data", "exempt", "*.log"))]
ELEMENT = re.compile(
    r"(?:node|junction|link|pipe|sensor|节点|管段|传感器|漏点|候选)\s*#?\s*(\d+)",
    re.I)

alpha_re = re.compile(r"(?<![A-Za-z0-9_-])(" +
                      "|".join(sorted((re.escape(x) for x in ids_alpha),
                                      key=len, reverse=True)) +
                      r")(?![A-Za-z0-9_-])") if ids_alpha else None
total = 0
for rel in TARGETS:
    p = os.path.join(ROOT, rel)
    if not os.path.isfile(p):
        print(f"  {rel}: MISSING")
        continue
    txt = open(p, encoding="utf-8", errors="replace").read()
    hits_a = sorted(set(alpha_re.findall(txt))) if alpha_re else []
    hits_n = sorted({m for m in ELEMENT.findall(txt) if m in ids_num})
    ctx = []
    for m in ELEMENT.finditer(txt):
        if m.group(1) in ids_num:
            ctx.append(txt[max(0, m.start() - 30):m.end() + 10].replace("\n", " "))
    total += len(hits_a) + len(hits_n)
    print(f"  {rel}: non-numeric-id hits {len(hits_a)}"
          f"{(' ' + str(hits_a[:8])) if hits_a else ''}; "
          f"element+number hits {len(hits_n)}"
          f"{(' ' + str(hits_n[:8])) if hits_n else ''}")
    for c in ctx[:6]:
        print("      ctx:", c)
print("TOTAL potential identifier leaks:", total)

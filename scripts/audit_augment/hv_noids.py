# -*- coding: utf-8 -*-
"""hv_noids.py: independent zero-identifier scan of the augmentation deliverables.

The identifier set is every node id and link id of the City D reference network. Text files
are tokenised (words, numbers, dotted / hyphenated tokens); JSON files are walked: every key
and every string value is tokenised the same way. Numeric JSON values are not identifiers
(counts and ids are indistinguishable), but for the text files every numeric token that
coincides with an id is listed with its line so it can be judged by eye.
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
from dgga.parse import Net  # noqa: E402

FILES = [
    "data/augment_city_d_wip.txt", "data/augment_pub_hanoi_wip.txt", "data/augment_public_wip.txt",
    "data/augment_suite_city_d.json", "data/augment_suite_pub_hanoi.json",
    "data/leak_augment_city_d.json", "data/placement_augment_city_d.json",
    "data/placement_augment_pub_hanoi.json", "data/placement_augment_pub_hanoi_synth25.json",
    "data/placement_augment_ltown.json", "data/augment_public_sigma_ladder.json",
    "data/calib_gc1_city_d.json", "data/audit_augment_fisher.json", "data/audit_augment_leak_fair.json",
    "data/audit_augment_wip.txt", "data/audit_leak_control.json", "data/audit_augment_fisher_v100.json",
    "docs/server_v100.md", "CHANGELOG.md",
    "scripts/augment_suite.py", "scripts/augment_suite_run.sh", "scripts/place_sensors.py",
    "scripts/audit_augment/hv_fisher.py", "scripts/audit_augment/hv_leak_fair.py",
    "scripts/audit_augment/hv_leak_control.py", "scripts/audit_augment/hv_noids.py",
    "scripts/audit_augment/hv_shadow.py",
    # coherence-driven augmentation round
    "data/augment_coherence_wip.txt", "data/augment_coherence.json",
    "data/placement_coh_city_d.json", "data/placement_coh_ltown.json", "data/placement_coh_hanoi.json",
    "scripts/augment_coherence.py", "scripts/augment_coherence_v100.sh",
]
LEAK_COH_DIR = "data/leak_coh_city_d"          # one json per spec, scanned when present
TOK = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-\.]*")


def main():
    net = Net.load(os.path.join(DATA, "reference"), "city_d")
    ids = set(str(x) for x in net.node_id) | set(str(x) for x in net.link_id)
    numeric_ids = {x for x in ids if x.isdigit()}
    print(f"[hv_noids] id set: {len(ids)} (numeric {len(numeric_ids)})")
    total_hits = 0
    files = list(FILES)
    d = os.path.join(ROOT, LEAK_COH_DIR)
    if os.path.isdir(d):
        files += [os.path.join(LEAK_COH_DIR, fn) for fn in sorted(os.listdir(d)) if fn.endswith(".json")]
    for rel in files:
        fp = os.path.join(ROOT, rel)
        if not os.path.isfile(fp):
            print(f"  {rel}: (absent)")
            continue
        hits = []
        if rel.endswith(".json"):
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)

            def walk(x, p):
                if isinstance(x, dict):
                    for k, v in x.items():
                        for t in TOK.findall(str(k)):
                            if t in ids:
                                hits.append((f"key {p}.{k}", t))
                        walk(v, f"{p}.{k}")
                elif isinstance(x, list):
                    for i, v in enumerate(x):
                        walk(v, f"{p}[{i}]")
                elif isinstance(x, str):
                    for t in TOK.findall(x):
                        if t in ids:
                            hits.append((f"str {p}", t))
            walk(obj, "$")
        else:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for ln, line in enumerate(f, 1):
                    for t in TOK.findall(line):
                        if t in ids:
                            hits.append((f"line {ln}: {line.strip()[:90]}", t))
        total_hits += len(hits)
        print(f"  {rel}: {len(hits)} id-token hit(s)")
        for where, t in hits[:12]:
            print(f"      [{t}] {where}")
    print(f"[hv_noids] total id-token hits {total_hits} (numeric hits in text need eye judgement: count vs id)")


if __name__ == "__main__":
    main()

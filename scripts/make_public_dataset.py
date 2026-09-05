#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the public, anonymised dataset release under ``datasets/``.

Sources (confidential, never published as-is)
---------------------------------------------
    --src-d     (default networks/realInpData/city_d.inp)   -> datasets/city_d.inp
    --src-h     (default networks/InpData/city_h.inp)       -> datasets/city_h.inp
    --src-leaks (default networks/field_records/city_d_leak_records.csv)
                                               -> datasets/city_d_leak_records.csv
                                               -> datasets/city_d_leak_records.json
The private source paths live under the untracked ``networks/`` tree and can be
overridden on the command line; this script contains no confidential names.

What the anonymisation does
---------------------------
1. ``[COORDINATES]`` / ``[VERTICES]`` are pushed through a fixed rigid-body
   affine map (translate-to-origin -> rotate by a seed-derived angle ->
   translate-to-origin again), then rounded to 3 decimals.  A rigid-body map
   destroys the georeference (the source files carry real projected CRS
   coordinates) while preserving every relative distance, so plots are
   unchanged up to rotation.  **Node/link coordinates are never read by the
   hydraulic solver**, therefore this step is provably lossless -- see
   ``--verify``.
2. ``[TITLE]`` is replaced by a neutral English caption; the leading
   ``; Filename: ...`` provenance comments (which hold a local absolute path
   and a Chinese file name) are dropped.
3. ``[LABELS]`` map annotations and ``[BACKDROP]`` file references are dropped;
   ``[BACKDROP] DIMENSIONS`` is recomputed from the transformed extent.
4. Every remaining non-ASCII character is removed (they only ever occur in
   comments; the script aborts if one is found in a data field).
5. **Node and link IDs are left untouched** -- the field leak records join to
   the model by node ID, and the authors chose not to obfuscate them.
6. Leak records: street addresses and metering-district names are deleted,
   Chinese categorical values are translated to English, and absolute dates
   are converted to day offsets from the earliest date in the table.

Usage
-----
    python -X utf8 scripts/make_public_dataset.py                # build
    python -X utf8 scripts/make_public_dataset.py --verify       # build + check
    python -X utf8 scripts/make_public_dataset.py --verify-only  # check only

Everything is a pure function of the build seed; re-running with the same seed
reproduces the release byte-for-byte.  The seed (and hence the derived rotation
angles) is NOT published: it is read from the private mapping file named by the
``HYDROGRAD_NAME_MAP`` environment variable (key ``build_seed``; that file exists only on
the authors' machines), or passed via ``--seed``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import fields as _dc_fields

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SRC_D_INP = os.path.join(ROOT, "networks", "realInpData", "city_d.inp")
DEFAULT_SRC_H_INP = os.path.join(ROOT, "networks", "InpData", "city_h.inp")
DEFAULT_SRC_LEAK_CSV = os.path.join(ROOT, "networks", "field_records",
                                    "city_d_leak_records.csv")

PRIVATE_MAP_JSON = os.environ.get("HYDROGRAD_NAME_MAP", "")


def _private_map():
    """gitignored 本地映射（禁词表 + 私有构建参数）；缺失时返回 None。"""
    if not PRIVATE_MAP_JSON or not os.path.exists(PRIVATE_MAP_JSON):
        return None
    with open(PRIVATE_MAP_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def default_seed():
    """The build seed is private (it determines the unpublished rotation
    angles).  It is never hard-coded here: read it from the private mapping
    file named by HYDROGRAD_NAME_MAP, or pass --seed explicitly."""
    m = _private_map()
    if m and "build_seed" in m:
        return int(m["build_seed"])
    return None


COORD_DECIMALS = 3

TITLES = {
    "city_d": "City D distribution network (anonymised)",
    "city_h": "City H distribution network (anonymised)",
}

# --------------------------------------------------------------------------
# Chinese -> English value dictionaries for the leak-record table.
# Keys are the verbatim source values; every distinct source value present in
# the 256-row table is covered, and the builder aborts on an unseen value so a
# future re-export cannot silently leak untranslated text.
# --------------------------------------------------------------------------
MAP_IN_MAIN_SCENARIO = {
    "\u662f": "yes",   # 是
    "\u5426": "no",    # 否
    "": "",
}

MAP_LEAK_TYPE = {
    "\u660e\u6f0f": "visible",  # 明漏  -- surfacing / reported by sight
    "\u6697\u6f0f": "hidden",   # 暗漏  -- non-surfacing / found by survey
    "": "",
}

MAP_MATERIAL = {
    "PE": "PE",
    "PPR": "PPR",
    "PVC": "PVC",
    "UPVC": "UPVC",
    "\u94a2": "steel",              # 钢
    "\u94c1": "iron",               # 铁
    "\u94f8\u94c1": "cast_iron",    # 铸铁
    "/": "unknown",                 # literal placeholder in the source sheet
    "": "",
}

MAP_LOCATION_TYPE = {
    "\u5ead\u9662\u7ba1\u7f51": "premise_network",                    # 庭院管网
    "\u4e2d\u533a\u7ba1\u7ebf": "mid_zone_main",                      # 中区管线
    "\u5e9f\u5f03\u7ba1\u7ebf": "abandoned_pipe",                     # 废弃管线
    "\u6d88\u706b\u6813": "fire_hydrant",                             # 消火栓
    "\u975e\u5c45\u4f9b\u6c34\u7ba1\u7ebf": "non_residential_main",   # 非居供水管线
    "\u9ad8\u533a\u7ba1\u7ebf": "high_zone_main",                     # 高区管线
    "\u5ead\u9662\u4e3b\u7ebf": "premise_trunk",                      # 庭院主线
    "\u5e02\u653f\u4e3b\u7ba1\u7ebf": "municipal_main",               # 市政主管线
    "\u697c\u524d\u7ba1\u7ebf": "building_frontage_pipe",             # 楼前管线
    "\u6d88\u706b\u6813\u7ba1\u7ebf": "fire_hydrant_branch",          # 消火栓管线
    "\u5e02\u653f\u652f\u7ebf": "municipal_branch",                   # 市政支线
    "\u5e02\u653f\u7ba1\u7f51": "municipal_network",                  # 市政管网
    "\u5165\u6237\u7ba1\u7ebf": "service_connection",                 # 入户管线
    "\u4e2d\u9ad8\u533a\u7ba1\u7ebf": "mid_high_zone_main",           # 中高区管线
    "\u9ad8\u533a\u7ba1\u7f51": "high_zone_network",                  # 高区管网
    "\u4f4e\u533a\u7ba1\u7ebf": "low_zone_main",                      # 低区管线
    "\u8bbe\u5907": "equipment",                                      # 设备
    "": "",
}

# Columns kept in the published CSV, in output order.
PUBLIC_COLUMNS = [
    "record_id",
    "leak_rate_t_per_day",
    "leak_lps",
    "node_id",
    "mapping_status",
    "snap_distance_m",
    "join_confidence",
    "in_main_scenario",
    "exclusion_reason",
    "geo_confidence",
    "nearest_pipe",
    "pipe_diameter_mm",
    "location_type",
    "material",
    "leak_type",
    "day_reported",
    "day_completed",
]

DROPPED_COLUMNS = ["address", "district", "reported", "completed"]


# ==========================================================================
# helpers
# ==========================================================================
def rotation_angle_deg(seed: int, tag: str) -> float:
    """Deterministic, platform-independent rotation angle in [0, 360)."""
    digest = hashlib.sha256(f"{seed}:{tag}".encode("ascii")).digest()
    frac = int.from_bytes(digest[:8], "big") / 2.0**64
    return round(frac * 360.0, 6)


def _read_text(path: str):
    raw = open(path, "rb").read()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode("utf-8-sig"), newline


def _nonascii(s: str):
    return [(i, ch) for i, ch in enumerate(s) if ord(ch) > 127]


# ==========================================================================
# INP anonymisation
# ==========================================================================
class InpAnonymiser:
    """Two-pass rewrite of an EPANET INP: collect coordinates, then emit."""

    GEOM_SECTIONS = ("COORDINATES", "VERTICES")

    def __init__(self, src: str, title: str, angle_deg: float, decimals: int = COORD_DECIMALS):
        self.src = src
        self.title = title
        self.angle_deg = angle_deg
        self.theta = math.radians(angle_deg)
        self.decimals = decimals
        self.report = {
            "source": src,
            "rotation_deg": angle_deg,
            "nonascii_lines": [],   # (lineno, section, repr(line))
            "dropped_lines": [],    # (lineno, section, reason, repr(line))
            "n_coordinates": 0,
            "n_vertices": 0,
        }

    # -- pass 1 ------------------------------------------------------------
    def _scan(self, lines):
        xs, ys = [], []
        section = None
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("["):
                section = s[1:s.find("]")].strip().upper() if "]" in s else None
                continue
            if section in self.GEOM_SECTIONS and not s.startswith(";"):
                tok = s.split(";", 1)[0].split()
                if len(tok) >= 3:
                    try:
                        xs.append(float(tok[1]))
                        ys.append(float(tok[2]))
                    except ValueError:
                        pass
        if not xs:
            return None
        return min(xs), min(ys)

    # -- transform ---------------------------------------------------------
    def _xy(self, x: float, y: float):
        # 1) translate so the source extent starts at (0, 0)  -- also keeps the
        #    magnitudes small, which protects float precision under rotation
        x -= self.x0
        y -= self.y0
        # 2) rotate about the origin by the seed-derived angle
        c, s = math.cos(self.theta), math.sin(self.theta)
        xr, yr = x * c - y * s, x * s + y * c
        # 3) translate again so the rotated extent starts at (0, 0)
        return xr - self.xr0, yr - self.yr0

    def _fmt(self, v: float) -> float:
        return round(v, self.decimals)

    # -- pass 2 ------------------------------------------------------------
    def run(self):
        text, newline = _read_text(self.src)
        lines = text.split("\n")
        lines = [ln[:-1] if ln.endswith("\r") else ln for ln in lines]

        mins = self._scan(lines)
        if mins is None:
            raise RuntimeError(f"{self.src}: no [COORDINATES] found")
        self.x0, self.y0 = mins

        # rotated extent -> second translation offset
        self.xr0 = self.yr0 = 0.0
        rx, ry = [], []
        section = None
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("["):
                section = s[1:s.find("]")].strip().upper() if "]" in s else None
                continue
            if section in self.GEOM_SECTIONS and not s.startswith(";"):
                tok = s.split(";", 1)[0].split()
                if len(tok) >= 3:
                    try:
                        x, y = float(tok[1]) - self.x0, float(tok[2]) - self.y0
                    except ValueError:
                        continue
                    c, s_ = math.cos(self.theta), math.sin(self.theta)
                    rx.append(x * c - y * s_)
                    ry.append(x * s_ + y * c)
        self.xr0, self.yr0 = min(rx), min(ry)

        out = []
        out.append("; Anonymised release -- see datasets/README.md")
        out.append("; Coordinates: metres under an unpublished rigid-body")
        out.append("; transform (shape/scale preserved); no georeference.")
        out.append("; Do not use for geolocation.")

        section = None
        seen_first_section = False
        title_written = False
        bbox = [None, None, None, None]  # xmin ymin xmax ymax (transformed)

        for lineno, ln in enumerate(lines, 1):
            s = ln.strip()

            if s.startswith("["):
                section = s[1:s.find("]")].strip().upper() if "]" in s else None
                seen_first_section = True
                out.append(s if s == ln.strip() else ln)
                if section == "TITLE":
                    out.append(self.title)
                    out.append("")
                    title_written = True
                continue

            if not seen_first_section:
                # provenance banner: absolute local path + Chinese file name
                if s:
                    self.report["dropped_lines"].append(
                        (lineno, "<header>", "provenance banner", ln))
                continue

            if section == "TITLE":
                if s and not title_written:
                    self.report["dropped_lines"].append(
                        (lineno, "TITLE", "original title/date", ln))
                elif s:
                    self.report["dropped_lines"].append(
                        (lineno, "TITLE", "original title/date", ln))
                continue

            if section == "LABELS":
                if s and not s.startswith(";"):
                    self.report["dropped_lines"].append(
                        (lineno, "LABELS", "map annotation text", ln))
                    continue
                out.append(ln)
                continue

            if section == "BACKDROP":
                tok = s.split(";", 1)[0].split()
                key = tok[0].upper() if tok else ""
                if key == "DIMENSIONS":
                    continue          # re-emitted after the loop
                if key == "FILE":
                    self.report["dropped_lines"].append(
                        (lineno, "BACKDROP", "backdrop image path", ln))
                    out.append(" FILE")
                    continue
                out.append(ln)
                continue

            if section in self.GEOM_SECTIONS and s and not s.startswith(";"):
                body, _, cmt = ln.partition(";")
                tok = body.split()
                if len(tok) >= 3:
                    try:
                        x, y = self._xy(float(tok[1]), float(tok[2]))
                    except ValueError:
                        out.append(ln)
                        continue
                    x, y = self._fmt(x), self._fmt(y)
                    if bbox[0] is None:
                        bbox = [x, y, x, y]
                    else:
                        bbox = [min(bbox[0], x), min(bbox[1], y),
                                max(bbox[2], x), max(bbox[3], y)]
                    out.append(" %-16s %-16.3f %-16.3f" % (tok[0], x, y))
                    if section == "COORDINATES":
                        self.report["n_coordinates"] += 1
                    else:
                        self.report["n_vertices"] += 1
                    continue
                out.append(ln)
                continue

            # ---- everything else passes through verbatim -------------------
            out.append(ln)

        # re-insert BACKDROP DIMENSIONS from the transformed extent
        if bbox[0] is not None:
            for i, ln in enumerate(out):
                if ln.strip().upper().startswith("[BACKDROP]"):
                    out.insert(i + 1, " DIMENSIONS      %-16.3f %-16.3f %-16.3f %-16.3f"
                               % tuple(bbox))
                    break

        # ---- final non-ASCII sweep ---------------------------------------
        cleaned = []
        section = None
        for i, ln in enumerate(out):
            s = ln.strip()
            if s.startswith("["):
                section = s[1:s.find("]")].strip().upper() if "]" in s else None
            bad = _nonascii(ln)
            if not bad:
                cleaned.append(ln)
                continue
            self.report["nonascii_lines"].append((i + 1, section, ln, bad))
            if s.startswith(";"):
                continue  # comment -> drop outright
            raise RuntimeError(
                f"{self.src}: non-ASCII found in a DATA line of [{section}] -- "
                f"manual handling required: {ln!r}")

        return newline.join(cleaned) + newline

    def write(self, dst: str):
        text = self.run()
        with open(dst, "w", encoding="ascii", newline="") as f:
            f.write(text)
        return self.report


# ==========================================================================
# leak-record anonymisation
# ==========================================================================
def _translate(value: str, table: dict, column: str):
    if value in table:
        return table[value]
    raise RuntimeError(
        f"leak records: unmapped value {value!r} in column {column!r} -- "
        "add it to the translation table before publishing")


def anonymise_leaks(src_csv: str, out_csv: str, out_json: str):
    raw = open(src_csv, "rb").read().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(raw)))
    report = {"n_rows": len(rows), "dropped_columns": list(DROPPED_COLUMNS)}

    # --- day-0 epoch ---------------------------------------------------
    dates = []
    for r in rows:
        for k in ("reported", "completed"):
            v = (r.get(k) or "").strip()
            if v:
                dates.append(_dt.date.fromisoformat(v))
    epoch = min(dates)
    report["epoch_note"] = "day 0 = earliest date present in the source table"
    report["n_dates_seen"] = len(dates)

    def day(v):
        v = (v or "").strip()
        return "" if not v else str((_dt.date.fromisoformat(v) - epoch).days)

    out_rows = []
    for r in rows:
        o = {
            "record_id": r["record_id"],
            "leak_rate_t_per_day": r["leak_rate_t_per_day"],
            "leak_lps": r["leak_lps"],
            "node_id": r["node_id"],
            "mapping_status": r["mapping_status"],
            "snap_distance_m": r["snap_distance_m"],
            "join_confidence": r["join_confidence"],
            "in_main_scenario": _translate(r["in_main_scenario"],
                                           MAP_IN_MAIN_SCENARIO, "in_main_scenario"),
            "exclusion_reason": r["exclusion_reason"],
            "geo_confidence": r["geo_confidence"],
            "nearest_pipe": r["nearest_pipe"],
            "pipe_diameter_mm": r["pipe_diameter_mm"],
            "location_type": _translate(r["location_type"],
                                        MAP_LOCATION_TYPE, "location_type"),
            "material": _translate(r["material"], MAP_MATERIAL, "material"),
            "leak_type": _translate(r["leak_type"], MAP_LEAK_TYPE, "leak_type"),
            "day_reported": day(r.get("reported")),
            "day_completed": day(r.get("completed")),
        }
        for k, v in o.items():
            bad = _nonascii(v)
            if bad:
                raise RuntimeError(f"leak records: non-ASCII left in {k}={v!r}")
        out_rows.append(o)

    with open(out_csv, "w", encoding="ascii", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PUBLIC_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)

    # --- summary JSON (no local paths, no source-workbook names) --------
    def _f(v):
        v = (v or "").strip()
        return None if v == "" else float(v)

    with_node = [r for r in out_rows if r["node_id"].strip()]
    distinct = sorted({r["node_id"] for r in with_node}, key=lambda s: int(s))
    per_node = {}
    for r in with_node:
        per_node[r["node_id"]] = per_node.get(r["node_id"], 0) + 1
    per_node = dict(sorted(per_node.items(), key=lambda kv: (-kv[1], int(kv[0]))))

    def _col(rows_, name):
        v = [_f(r[name]) for r in rows_]
        return sorted(x for x in v if x is not None)

    rates = _col(out_rows, "leak_rate_t_per_day")
    lps = _col(out_rows, "leak_lps")
    snap = _col(out_rows, "snap_distance_m")
    rates_wn = _col(with_node, "leak_rate_t_per_day")
    lps_wn = _col(with_node, "leak_lps")

    def _median(a):
        n = len(a)
        return None if n == 0 else (a[n // 2] if n % 2 else 0.5 * (a[n // 2 - 1] + a[n // 2]))

    def _pct(a, p):
        if not a:
            return None
        return a[min(len(a) - 1, int(round(p * (len(a) - 1))))]

    def _tally(col):
        t = {}
        for r in out_rows:
            t[r[col]] = t.get(r[col], 0) + 1
        return dict(sorted(t.items(), key=lambda kv: -kv[1]))

    summary = {
        "dataset": "city_d_leak_records",
        "description": ("Field leak repair work orders for the City D distribution "
                        "network, one calendar year, joined to model node IDs."),
        "anonymisation": [
            "street address column removed",
            "metering-district column removed",
            "absolute dates replaced by day offsets from the earliest record",
            "Chinese categorical values translated to English",
            "node and link IDs unchanged (join key to city_d.inp)",
        ],
        "columns": list(PUBLIC_COLUMNS),
        "n_records": len(out_rows),
        "n_with_node": len(with_node),
        "n_distinct_nodes": len(distinct),
        "distinct_nodes": distinct,
        "records_per_node": per_node,
        "day_span": {
            "day_reported_min": min(int(r["day_reported"]) for r in out_rows
                                    if r["day_reported"]),
            "day_reported_max": max(int(r["day_reported"]) for r in out_rows
                                    if r["day_reported"]),
        },
        "leak_rate_t_per_day": {"scope": "all records", "n": len(rates), "min": rates[0],
                                "median": _median(rates), "max": rates[-1],
                                "sum": sum(rates)},
        "leak_lps": {"scope": "all records", "n": len(lps), "min": lps[0],
                     "median": _median(lps), "max": lps[-1], "sum": sum(lps)},
        "leak_rate_t_per_day_with_node": {
            "scope": "records carrying a node_id", "n": len(rates_wn),
            "min": rates_wn[0], "median": _median(rates_wn), "max": rates_wn[-1],
            "sum": sum(rates_wn)},
        "leak_lps_with_node": {
            "scope": "records carrying a node_id", "n": len(lps_wn),
            "min": lps_wn[0], "median": _median(lps_wn), "max": lps_wn[-1],
            "sum": sum(lps_wn)},
        "snap_distance_m": {"n": len(snap), "median": _median(snap),
                            "p90": _pct(snap, 0.90), "max": snap[-1]},
        "join_confidence": _tally("join_confidence"),
        "geo_confidence": _tally("geo_confidence"),
        "in_main_scenario": _tally("in_main_scenario"),
        "leak_type": _tally("leak_type"),
        "material": _tally("material"),
        "location_type": _tally("location_type"),
        "mapping_status": _tally("mapping_status"),
        "note": ("The raw utility workbook also carries a reporter-contact column; "
                 "it was never extracted and is not part of this release."),
    }
    blob = json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=False)
    bad = _nonascii(blob)
    if bad:
        raise RuntimeError(f"leak JSON: non-ASCII survived: {bad[:5]}")
    with open(out_json, "w", encoding="ascii", newline="") as f:
        f.write(blob + "\n")

    report["summary"] = summary
    report["n_with_node"] = len(with_node)
    report["n_distinct_nodes"] = len(distinct)
    report["leak_lps_sum"] = summary["leak_lps"]["sum"]
    report["leak_lps_sum_with_node"] = summary["leak_lps_with_node"]["sum"]
    return report


# ==========================================================================
# README + checksums
# ==========================================================================
README_TEMPLATE = """\
# City D / City H water distribution network datasets

Two real municipal water distribution network models and one year of field
leak repair records, released in anonymised form.  The two utilities are
referred to only as **City D** and **City H**; the underlying networks are
operational systems in China, and the release is anonymised so that they cannot
be geolocated or named.

| File | Contents |
| --- | --- |
| `city_d.inp` | EPANET 2.2 model of the City D network ({d_junc} junctions, {d_pipes} pipes, LPS / Hazen-Williams, 24 h EPS). Primary case study. |
| `city_h.inp` | EPANET 2.2 model of the City H network ({h_junc} junctions, {h_pipes} pipes, LPS / Hazen-Williams, 24 h EPS). |
| `city_d_leak_records.csv` | {n_records} field leak repair work orders for City D, {n_with_node} of which are joined to a City D model node. |
| `city_d_leak_records.json` | Machine-readable summary of the same table (counts, distinct nodes, distributions). |
| `SHA256SUMS.txt` | SHA-256 checksums of every file above. |

Generated by `scripts/make_public_dataset.py` from the private source models.
The build is deterministic: the same (private, unpublished) build seed
reproduces these files byte for byte.

---

## 1. Anonymisation

### 1.1 Coordinates -- rigid-body transform, no georeference

> **The `[COORDINATES]` and `[VERTICES]` values in these files MUST NOT be
> used for geolocation.  Their unit of length is the metre -- the same unit
> as the `[PIPES]` `Length` column; the rigid-body transform changes neither
> scale nor shape -- but they have no datum, no origin and no north
> direction, and the transform parameters are not published.  Because shape
> and scale are fully preserved, the network outline could in principle
> still be matched against maps; do not attempt to locate the real systems.**

The source models carry real projected CRS coordinates.  Each released model
has had a fixed rigid-body map applied to every node coordinate and every pipe
vertex:

1. translate so the minimum X and minimum Y of the model become 0;
2. rotate about the origin by a fixed angle derived from the private build
   seed -- **the angles are not published** (publishing them would reduce
   inverting the transform to recovering a 2-D translation);
3. translate again so the rotated extent starts at (0, 0);
4. round to {decimals} decimals.

A rigid-body map preserves every pairwise distance, so pipe lengths, network
shape and any plot of the network are unchanged up to a rotation.  **EPANET
never reads node coordinates during a hydraulic solve** -- pipe lengths come
from the `[PIPES]` `Length` column, not from geometry -- so this step cannot
change a single simulated head or flow.  This is verified numerically by
`scripts/make_public_dataset.py --verify`, which runs a full extended-period
simulation of the original and the released model through the EPANET 2.2
toolkit and requires `max|delta| == 0.0` on every head and every flow at every
time step.

`[BACKDROP] DIMENSIONS` has been recomputed from the transformed extent and
any backdrop image reference removed.

### 1.2 Text

* `[TITLE]` replaced by a neutral English caption.
* The `; Filename: ...` provenance banner of the City D model, which contained
  a local absolute path and a Chinese file name, has been deleted.
* `[LABELS]` map annotations removed.
* Every file in this directory is pure ASCII and contains no absolute paths.

### 1.3 Identifiers -- deliberately NOT changed

Node IDs, pipe IDs, pump IDs, valve IDs, pattern IDs and curve IDs are
**identical to the source models**.  They are opaque integers assigned by the
utility's modelling software and carry no address information, and the leak
record table joins to the model through them.

### 1.4 Leak records

* The `address` column (building-level Chinese street addresses) is deleted.
* The `district` column (metering-district names) is deleted.
* `reported` / `completed` absolute dates are replaced by `day_reported` /
  `day_completed`, integer day offsets relative to an internal epoch
  (day 0 = earliest record in the table).  The absolute dates, the calendar
  date of the epoch and even its year are not published; only intervals and
  relative ordering are meaningful.
* Chinese categorical values are translated to English (see the dictionary in
  the build script).
* Nothing else is altered: `node_id`, `leak_lps` and `leak_rate_t_per_day` are
  bit-identical to the source table.

---

## 2. Field dictionary

### 2.1 `city_d.inp`, `city_h.inp`

Standard EPANET 2.2 INP format; see the EPANET 2.2 User Manual for section
semantics.  Units are `LPS` (litres per second) with `H-W`
(Hazen-Williams) head loss in both models.  Elevations and pipe lengths are in
metres, diameters in millimetres, as required by the SI flow unit setting.
Coordinates are the only non-standard element: they are metres under an
unpublished rigid-body transform, with no georeference (Section 1.1).

### 2.2 `city_d_leak_records.csv`

{n_records} rows, one per repair work order, UTF-8-free pure ASCII, `,`
separated, LF line endings.

| Column | Type | Description |
| --- | --- | --- |
| `record_id` | int | 1-based row identifier in the source work-order table. |
| `leak_rate_t_per_day` | float | Leak volume estimated by the utility at repair time, tonnes per day. Blank where the utility recorded no estimate. |
| `leak_lps` | float | Same quantity converted to litres per second (`t/d / 86.4`). This is the value used as the demand increment in leak scenarios. |
| `node_id` | str | ID of the `city_d.inp` junction the leak was mapped to. Blank when the leak could not be placed on the skeleton. |
| `mapping_status` | enum | `mapped_to_skeleton`, `unresolved_landmark`, `out_of_model`, `out_of_model_unresolved`. |
| `snap_distance_m` | float | Distance from the geocoded leak location to the assigned node, in metres, in the original CRS. A large value means the leak sits on a premise network branch that the skeleton does not represent. |
| `join_confidence` | enum | `high` / `medium` / `low` -- confidence of the leak-to-node assignment. |
| `in_main_scenario` | enum | `yes` / `no` -- whether the record is included in the primary leak scenario set. |
| `exclusion_reason` | str | Semicolon-separated reason codes when `in_main_scenario = no`; blank otherwise. |
| `geo_confidence` | enum | `high` / `medium` / `low` / `very_low` -- confidence of the geocoding step that preceded the node assignment. |
| `nearest_pipe` | str | ID of the nearest `city_d.inp` pipe. Blank when unresolved. |
| `pipe_diameter_mm` | float | Diameter of that pipe, millimetres. |
| `location_type` | enum | Where in the system the leak occurred: `premise_network`, `premise_trunk`, `municipal_main`, `municipal_branch`, `municipal_network`, `mid_zone_main`, `high_zone_main`, `mid_high_zone_main`, `low_zone_main`, `high_zone_network`, `non_residential_main`, `building_frontage_pipe`, `service_connection`, `fire_hydrant`, `fire_hydrant_branch`, `abandoned_pipe`, `equipment`. |
| `material` | enum | `PE`, `PPR`, `PVC`, `UPVC`, `steel`, `iron`, `cast_iron`, `unknown`. |
| `leak_type` | enum | `visible` -- the leak surfaced and was reported by sight; `hidden` -- it did not surface and was found by active leak detection. |
| `day_reported` | int | Days from the table's internal epoch (day 0 = earliest record; calendar date not published) to the day the leak was reported. |
| `day_completed` | int | Days from the same (unpublished) epoch to the day the repair was completed. |

### 2.3 `city_d_leak_records.json`

Summary of the CSV: record counts, the {n_distinct} distinct node IDs carrying
at least one leak, the number of records per node, and distributions of the
categorical and numeric columns.  Field names match the CSV.

---

## 3. Licence

These datasets are released under the **Creative Commons Attribution 4.0
International (CC BY 4.0)** licence: <https://creativecommons.org/licenses/by/4.0/>

Copyright (c) 2026 Tianwei Mu.

You may share and adapt the data for any purpose, including commercially,
provided you give appropriate credit.

## 4. How to cite

> Mu, T. (2026). *City D and City H water distribution network datasets*
> [Data set]. <https://github.com/mutianwei521/dgga> (`datasets/`).
> DOI: ADD ON ARCHIVAL RELEASE.

BibTeX:

```bibtex
@misc{{citydh_wdn_datasets,
  author       = {{Mu, Tianwei}},
  title        = {{City D and City H water distribution network datasets}},
  year         = {{2026}},
  note         = {{DOI to be added on archival release}},
  url          = {{https://github.com/mutianwei521/dgga}},
  howpublished = {{Data set}}
}}
```

## 5. Disclaimer

The coordinates in these models are metres under an unpublished rigid-body
transform and are not georeferenced.  Because the transform preserves shape
and scale, the network geometry could in principle be matched against maps;
the files must not be used to locate real infrastructure.  The leak
records describe historical repair work only and contain no personal data:
addresses, districts, calendar dates and reporter-contact information were all
removed before release.
"""


def _count_section(path, name):
    n = 0
    sec = None
    for ln in open(path, "r", encoding="ascii"):
        s = ln.strip()
        if s.startswith("["):
            sec = s[1:s.find("]")].strip().upper() if "]" in s else None
            continue
        if sec == name and s and not s.startswith(";"):
            n += 1
    return n


def write_readme(out_dir, leak_report):
    d_inp = os.path.join(out_dir, "city_d.inp")
    h_inp = os.path.join(out_dir, "city_h.inp")
    txt = README_TEMPLATE.format(
        decimals=COORD_DECIMALS,
        d_junc=_count_section(d_inp, "JUNCTIONS"),
        d_pipes=_count_section(d_inp, "PIPES"),
        h_junc=_count_section(h_inp, "JUNCTIONS"),
        h_pipes=_count_section(h_inp, "PIPES"),
        n_records=leak_report["n_rows"],
        n_with_node=leak_report["n_with_node"],
        n_distinct=leak_report["n_distinct_nodes"],
    )
    bad = _nonascii(txt)
    if bad:
        raise RuntimeError(f"README: non-ASCII {bad[:5]}")
    p = os.path.join(out_dir, "README.md")
    with open(p, "w", encoding="ascii", newline="") as f:
        f.write(txt)
    return p


def write_checksums(out_dir):
    names = sorted(n for n in os.listdir(out_dir)
                   if n != "SHA256SUMS.txt" and os.path.isfile(os.path.join(out_dir, n)))
    lines = []
    for n in names:
        h = hashlib.sha256(open(os.path.join(out_dir, n), "rb").read()).hexdigest()
        lines.append(f"{h}  {n}")
    p = os.path.join(out_dir, "SHA256SUMS.txt")
    with open(p, "w", encoding="ascii", newline="") as f:
        f.write("\n".join(lines) + "\n")
    return p, lines


# ==========================================================================
# verification
# ==========================================================================
def _read_coords(path):
    """{node_id: (x, y)} from [COORDINATES]."""
    out, sec = {}, None
    for ln in open(path, "r", encoding="utf-8-sig", errors="replace"):
        s = ln.strip()
        if s.startswith("["):
            sec = s[1:s.find("]")].strip().upper() if "]" in s else None
            continue
        if sec == "COORDINATES" and s and not s.startswith(";"):
            tok = s.split(";", 1)[0].split()
            if len(tok) >= 3:
                out[tok[0]] = (float(tok[1]), float(tok[2]))
    return out


def verify_rigid(src, dst, label, n_pairs=20000, seed=1):
    """The coordinate map must be a rigid body motion: all pairwise distances
    preserved (up to the 3-decimal rounding of the published values)."""
    import numpy as np
    a, b = _read_coords(src), _read_coords(dst)
    keys = [k for k in a if k in b]
    print(f"\n--- coordinate rigidity: {label} ---")
    print(f"    nodes in both files: {len(keys)} / src {len(a)} / dst {len(b)}  "
          f"ids identical={set(a) == set(b)}")
    A = np.array([a[k] for k in keys]); B = np.array([b[k] for k in keys])
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(keys), n_pairs)
    j = rng.integers(0, len(keys), n_pairs)
    da = np.hypot(*(A[i] - A[j]).T)
    db = np.hypot(*(B[i] - B[j]).T)
    err = np.abs(da - db)
    print(f"    {n_pairs} random node pairs, max distance error = {float(err.max()):.6g} "
          f"(rounding floor ~{10.0**-COORD_DECIMALS:.0e})")
    print(f"    published extent: x in [{B[:,0].min():.3f}, {B[:,0].max():.3f}], "
          f"y in [{B[:,1].min():.3f}, {B[:,1].max():.3f}]")
    print(f"    source extent   : x in [{A[:,0].min():.3f}, {A[:,0].max():.3f}], "
          f"y in [{A[:,1].min():.3f}, {A[:,1].max():.3f}]  (NOT published)")
    return bool(set(a) == set(b) and err.max() < 1e-2)


def verify_pair(src, dst, label):
    """Full-EPS bit-for-bit comparison of two INP files, plus Net-array diff."""
    import numpy as np
    sys.path.insert(0, ROOT)
    from dgga.epanet_ref import Epanet
    from dgga.parse import parse_inp, Net

    print(f"\n--- EPS comparison: {label} ---")
    print(f"    A = {src}")
    print(f"    B = {dst}")
    res = {}
    for tag, path in (("A", src), ("B", dst)):
        with Epanet(path) as en:
            r = en.solve_eps()
        res[tag] = r
        print(f"    {tag}: frames={len(r['t_sec'])} nodes={r['head_ft'].shape[1]} "
              f"links={r['flow_cfs'].shape[1]} warnings={len(r.get('warnings', []) or [])}")

    ok = True
    out = {}
    for key in ("head_ft", "pressure_ft", "demand_out_cfs", "flow_cfs", "setting"):
        a = np.asarray(res["A"][key], dtype=np.float64)
        b = np.asarray(res["B"][key], dtype=np.float64)
        if a.shape != b.shape:
            print(f"    {key}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            ok = False
            continue
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        nbits = int(np.count_nonzero(a != b))
        out[key] = d
        print(f"    max|delta| {key:<16} = {d!r}   (bitwise-unequal elements: {nbits}"
              f" / {a.size})")
        if d != 0.0 or nbits != 0:
            ok = False
    a = np.asarray(res["A"]["status"]); b = np.asarray(res["B"]["status"])
    same_status = bool(np.array_equal(a, b))
    print(f"    link status arrays identical: {same_status}")
    same_t = bool(np.array_equal(np.asarray(res["A"]["t_sec"]),
                                 np.asarray(res["B"]["t_sec"])))
    print(f"    time stamps identical: {same_t}")
    same_it = bool(np.array_equal(np.asarray(res["A"]["iterations"]),
                                  np.asarray(res["B"]["iterations"])))
    print(f"    iteration counts identical: {same_it}")
    same_re = bool(np.array_equal(np.asarray(res["A"]["relerr"]),
                                  np.asarray(res["B"]["relerr"])))
    print(f"    relative-error series identical: {same_re}")
    ok = ok and same_re
    ok = ok and same_status and same_t and same_it

    # ---- dgga.parse Net comparison ----
    print(f"    -- dgga.parse Net array diff --")
    na, nb = parse_inp(src), parse_inp(dst)
    def _deep_eq(x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.asarray(x), np.asarray(y)
            return x.shape == y.shape and bool(np.array_equal(x, y))
        if isinstance(x, (list, tuple)):
            return (isinstance(y, (list, tuple)) and len(x) == len(y)
                    and all(_deep_eq(a, b) for a, b in zip(x, y)))
        if isinstance(x, dict):
            return (isinstance(y, dict) and set(x) == set(y)
                    and all(_deep_eq(x[k], y[k]) for k in x))
        return bool(x == y)

    ndiff = 0
    nmax = {}
    for fl in _dc_fields(Net):
        va, vb = getattr(na, fl.name), getattr(nb, fl.name)
        same = _deep_eq(va, vb)
        if isinstance(va, np.ndarray) and va.dtype.kind in "fiu" and va.shape == vb.shape:
            nmax[fl.name] = float(np.max(np.abs(va.astype(float) - vb.astype(float)))) \
                if va.size else 0.0
        if not same:
            ndiff += 1
            print(f"       DIFF in Net.{fl.name}")
    worst = max(nmax.values()) if nmax else 0.0
    print(f"    Net fields compared: {len(_dc_fields(Net))}, differing: {ndiff}")
    print(f"    numeric Net arrays: {len(nmax)}, worst max|delta| across all = {worst!r}")
    ok = ok and ndiff == 0
    print(f"    RESULT {label}: {'LOSSLESS (max|delta| = 0.0 everywhere)' if ok else 'MISMATCH'}")
    return ok, out


def verify_leaks(src_csv, out_csv):
    print("\n--- leak record comparison ---")
    a = list(csv.DictReader(io.StringIO(open(src_csv, "rb").read().decode("utf-8-sig"))))
    b = list(csv.DictReader(io.StringIO(open(out_csv, "rb").read().decode("ascii"))))
    ok = len(a) == len(b)
    print(f"    rows: source={len(a)} public={len(b)}  equal={ok}")
    nid_a = [r["node_id"] for r in a]
    nid_b = [r["node_id"] for r in b]
    same_nid = nid_a == nid_b
    print(f"    node_id column identical row-by-row: {same_nid}")
    lps_a = [r["leak_lps"] for r in a]
    lps_b = [r["leak_lps"] for r in b]
    same_lps = lps_a == lps_b
    print(f"    leak_lps column identical row-by-row (string-exact): {same_lps}")
    fa = [float(v) for v in lps_a if v.strip()]
    fb = [float(v) for v in lps_b if v.strip()]
    dmax = max((abs(x - y) for x, y in zip(fa, fb)), default=0.0)
    print(f"    max|delta| leak_lps = {dmax!r}   n={len(fa)}  sum={sum(fa)!r}")
    sa = sorted({v for v in nid_a if v.strip()}, key=lambda s: int(s))
    sb = sorted({v for v in nid_b if v.strip()}, key=lambda s: int(s))
    print(f"    distinct nodes: source={len(sa)} public={len(sb)} identical={sa == sb}")
    rate_a = sum(float(r["leak_rate_t_per_day"]) for r in a if r["leak_rate_t_per_day"].strip())
    rate_b = sum(float(r["leak_rate_t_per_day"]) for r in b if r["leak_rate_t_per_day"].strip())
    print(f"    sum leak_rate_t_per_day: source={rate_a!r} public={rate_b!r} "
          f"delta={abs(rate_a - rate_b)!r}")
    ok = ok and same_nid and same_lps and sa == sb and dmax == 0.0 and rate_a == rate_b
    print(f"    RESULT leak records: {'IDENTICAL' if ok else 'MISMATCH'}")
    return ok


# Generic patterns: absolute paths and the author's handle.  The confidential
# source-network names themselves are NEVER written in this file: they are
# loaded at runtime from the private mapping file (HYDROGRAD_NAME_MAP).  Without
# that file the audit still runs, but the real-name part is reported as
# SKIPPED (never as a silent pass).
PATH_PATTERNS = ("C:\\", "D:\\", "c:\\", "d:\\", "/home/", "/Users/", "\\Users\\",
                 "mutianwei")


def realname_patterns():
    """Case variants of the confidential names from the local mapping, or
    None when the mapping file is absent."""
    m = _private_map()
    if m is None:
        return None
    pats = []
    for key in ("city_d", "city_h"):
        alias = m[key]["stem_alias"]
        pats += [alias.lower(), alias.capitalize(), alias.upper()]
        if m[key].get("cjk"):
            pats.append(m[key]["cjk"])
    for w in m.get("banned_extra", []):
        pats.append(w)
    return tuple(pats)


# Literals that legitimately contain a PATH_PATTERNS token and are deliberately
# published (the author's own repository URL). They are deleted from the text
# before the scan, so an accidental occurrence of the bare token still trips it.
ALLOWED_LITERALS = ("https://github.com/mutianwei521/dgga",)


def audit_output(out_dir):
    print("\n--- output audit (non-ASCII / absolute paths / source names) ---")
    clean = True
    rn = realname_patterns()
    if rn is None:
        print("    NOTE: private mapping (HYDROGRAD_NAME_MAP) absent -> the real-name "
              "part of this audit is SKIPPED (generic patterns still run).")
        rn = ()
    patterns = PATH_PATTERNS + rn
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if not os.path.isfile(p):
            continue
        raw = open(p, "rb").read()
        nb = [(i, b) for i, b in enumerate(raw) if b > 127]
        try:
            txt = raw.decode("ascii")
            dec = "ascii"
        except UnicodeDecodeError:
            txt = raw.decode("utf-8", "replace")
            dec = "NOT-ASCII"
        scan = txt
        for lit in ALLOWED_LITERALS:
            scan = scan.replace(lit, "")
        hits = []
        for pat in patterns:
            n = scan.count(pat)
            if n:
                # never echo a confidential name into the console/log
                shown = pat if pat in PATH_PATTERNS else "<real-name pattern>"
                hits.append((shown, n))
        crlf = raw.count(b"\r\n")
        print(f"    {name:<28} {len(raw):>9,d} B  decode={dec}  "
              f"non-ASCII bytes={len(nb)}  CRLF={crlf}  suspicious={hits if hits else 'none'}")
        if nb or hits:
            clean = False
    print(f"    RESULT audit: {'CLEAN' if clean else 'PROBLEMS FOUND'}")
    return clean


# ==========================================================================
def build(out_dir, seed, src_d, src_h, src_leaks):
    os.makedirs(out_dir, exist_ok=True)
    angles = {k: rotation_angle_deg(seed, k) for k in ("city_d", "city_h")}
    # The seed and the derived angles are private build parameters (publishing
    # them would let the rigid-body transform be partially inverted); they are
    # printed to the operator's console only and appear in no output file.
    print(f"seed = {seed}  (private, not written to any output)")
    print(f"rotation angles (deg): city_d = {angles['city_d']}, "
          f"city_h = {angles['city_h']}  (private, not written to any output)")

    reports = {}
    for tag, src in (("city_d", src_d), ("city_h", src_h)):
        dst = os.path.join(out_dir, f"{tag}.inp")
        an = InpAnonymiser(src, TITLES[tag], angles[tag])
        rep = an.write(dst)
        reports[tag] = rep
        print(f"\n[{tag}] {src}")
        print(f"   -> {dst}  ({os.path.getsize(dst):,d} B)")
        print(f"   coordinates transformed : {rep['n_coordinates']}")
        print(f"   vertices transformed    : {rep['n_vertices']}")
        print(f"   non-ASCII lines removed : {len(rep['nonascii_lines'])}")
        for lineno, sec, ln, bad in rep["nonascii_lines"]:
            print(f"      out-line {lineno} [{sec}] {ln!r}")
            print(f"         chars: {[(hex(ord(c)), c) for _, c in bad]}")
        print(f"   other lines dropped     : {len(rep['dropped_lines'])}")
        for lineno, sec, why, ln in rep["dropped_lines"]:
            print(f"      src-line {lineno} [{sec}] ({why}) {ln!r}")

    leak_rep = anonymise_leaks(
        src_leaks,
        os.path.join(out_dir, "city_d_leak_records.csv"),
        os.path.join(out_dir, "city_d_leak_records.json"),
    )
    print(f"\n[leak records] {src_leaks}")
    print(f"   rows                : {leak_rep['n_rows']}")
    print(f"   with node_id        : {leak_rep['n_with_node']}")
    print(f"   distinct nodes      : {leak_rep['n_distinct_nodes']}")
    print(f"   sum leak_lps (all)  : {leak_rep['leak_lps_sum']}")
    print(f"   sum leak_lps (node) : {leak_rep['leak_lps_sum_with_node']}")
    print(f"   dropped columns     : {leak_rep['dropped_columns']}")

    write_readme(out_dir, leak_rep)
    _, sums = write_checksums(out_dir)
    print("\n[SHA256SUMS.txt]")
    for ln in sums:
        print("   " + ln)
    return angles


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None,
                    help="private build seed; default comes from the private "
                         "mapping file named by HYDROGRAD_NAME_MAP (key build_seed)")
    ap.add_argument("--src-d", default=DEFAULT_SRC_D_INP,
                    help="private City D source INP (untracked)")
    ap.add_argument("--src-h", default=DEFAULT_SRC_H_INP,
                    help="private City H source INP (untracked)")
    ap.add_argument("--src-leaks", default=DEFAULT_SRC_LEAK_CSV,
                    help="private City D leak-record CSV (untracked)")
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--verify", action="store_true", help="build, then verify")
    ap.add_argument("--verify-only", action="store_true", help="verify an existing build")
    a = ap.parse_args(argv)

    seed = a.seed if a.seed is not None else default_seed()
    if seed is None and not a.verify_only:
        ap.error("no --seed given and the HYDROGRAD_NAME_MAP mapping has no "
                 "build_seed -- the seed is private and is not hard-coded "
                 "in this script")

    out_dir = os.path.abspath(a.out)
    if not a.verify_only:
        build(out_dir, seed, a.src_d, a.src_h, a.src_leaks)

    if a.verify or a.verify_only:
        print("\n" + "=" * 74)
        print("LOSSLESSNESS VERIFICATION")
        print("=" * 74)
        ok_d, _ = verify_pair(a.src_d, os.path.join(out_dir, "city_d.inp"),
                              "src(D)  vs  city_d.inp")
        ok_h, _ = verify_pair(a.src_h, os.path.join(out_dir, "city_h.inp"),
                              "src(H)  vs  city_h.inp")
        ok_d = verify_rigid(a.src_d, os.path.join(out_dir, "city_d.inp"),
                            "city_d") and ok_d
        ok_h = verify_rigid(a.src_h, os.path.join(out_dir, "city_h.inp"),
                            "city_h") and ok_h
        ok_l = verify_leaks(a.src_leaks,
                            os.path.join(out_dir, "city_d_leak_records.csv"))
        ok_a = audit_output(out_dir)
        print("\n" + "=" * 74)
        print(f"OVERALL: city_d={'PASS' if ok_d else 'FAIL'}  "
              f"city_h={'PASS' if ok_h else 'FAIL'}  "
              f"leaks={'PASS' if ok_l else 'FAIL'}  "
              f"audit={'PASS' if ok_a else 'FAIL'}")
        print("=" * 74)
        return 0 if (ok_d and ok_h and ok_l and ok_a) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

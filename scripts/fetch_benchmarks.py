# -*- coding: utf-8 -*-
"""fetch_benchmarks.py -- rebuild networks/public/ from its upstream sources.

This repository does not redistribute any third-party network file (see
THIRD_PARTY.md). This script recreates ``networks/public/`` by

  * downloading the files that live in public repositories / on Zenodo, and
  * copying the files that ship inside the installed ``wntr`` package,

then verifying every file against a SHA-256 manifest. Files that already exist
with the right hash are skipped, so the script is resumable.

Usage
-----
    python scripts/fetch_benchmarks.py                 # 20 networks, ~5 MB
    python scripts/fetch_benchmarks.py --with-large    # + L-TOWN_Real.inp (169 MiB)
    python scripts/fetch_benchmarks.py --only-licensed # only CC-BY-4.0 / MIT / BSD-3
    python scripts/fetch_benchmarks.py --list          # print the table, download nothing
    python scripts/fetch_benchmarks.py --verify        # re-hash what is already there
    python scripts/fetch_benchmarks.py Hanoi Net3      # only these

Licensing
---------
Each file prints its upstream licence before it is fetched. Ten of the
benchmarks come from a repository that states no licence at all; for those the
script prints an explicit "verify permissions yourself" warning. Nothing is
relicensed by downloading it.

Provenance of the SHA-256 values: computed on 2026-08-09 from the local copies
that produced data/benchmark_report.txt and data/regression_report.txt, and
cross-checked against the byte counts recorded in data/public_inventory.json.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "networks", "public")

UA = "dgga-fetch-benchmarks/0.1 (+https://example.invalid)"
CHUNK = 1 << 20

# Licence tags. "NOT_STATED" means the upstream repository carries no LICENSE
# file; those files are academic benchmarks of unclear redistribution status.
LIC_NOT_STATED = "not stated upstream (academic benchmark)"
LIC_CCBY = "CC-BY-4.0"
LIC_MIT = "MIT (upstream repository)"
LIC_BSD3 = "BSD-3-Clause (WNTR)"

# A licence is "clear" if the upstream repository states one.
CLEAR = {LIC_CCBY, LIC_MIT, LIC_BSD3}

KIOS = ("https://raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/"
        "master/")
WBH = ("https://raw.githubusercontent.com/WaterFutures/WaterBenchmarkHub/"
       "main/docs/static/benchmarks/")

# name -> dict(kind, url|wntr_rel, licence, note, large)
SOURCES: dict[str, dict] = {
    # --- shipped inside the installed wntr package ---------------------------
    "Net1.inp": dict(kind="wntr", rel="library/networks/Net1.inp",
                     licence=LIC_BSD3, note="EPANET example network (US EPA)"),
    "Net2.inp": dict(kind="wntr", rel="library/networks/Net2.inp",
                     licence=LIC_BSD3, note="EPANET example network (US EPA)"),
    "Net3.inp": dict(kind="wntr", rel="library/networks/Net3.inp",
                     licence=LIC_BSD3, note="EPANET example network (US EPA)"),
    "Net6.inp": dict(kind="wntr", rel="library/networks/Net6.inp",
                     licence=LIC_BSD3, note="EPANET example network (US EPA)"),
    "ky4.inp": dict(kind="wntr", rel="library/networks/ky4.inp",
                    licence=LIC_BSD3, note="Univ. of Kentucky WDST ky4"),
    "ky10.inp": dict(kind="wntr", rel="library/networks/ky10.inp",
                     licence=LIC_BSD3, note="Univ. of Kentucky WDST ky10"),
    "Anytown_wntr.inp": dict(kind="wntr",
                             rel="tests/networks_for_testing/Anytown.inp",
                             licence=LIC_BSD3,
                             note="WNTR's Anytown variant (3 pumps, 2 tanks)"),
    # --- KIOS-Research/EPANET-Benchmarks (no LICENSE upstream) ---------------
    "Anytown.inp": dict(kind="url", url=KIOS + "asce-tf-wdst/Anytown/Anytown.inp",
                        licence=LIC_NOT_STATED, note="ASCE TF WDST"),
    "Balerma.inp": dict(kind="url", url=KIOS + "asce-tf-wdst/Balerma/Balerma.inp",
                        licence=LIC_NOT_STATED, note="ASCE TF WDST, Darcy-Weisbach"),
    "Hanoi.inp": dict(kind="url", url=KIOS + "asce-tf-wdst/Hanoi/Hanoi.inp",
                      licence=LIC_NOT_STATED, note="ASCE TF WDST"),
    "Fossolo_poly1.inp": dict(
        kind="url", url=KIOS + "asce-tf-wdst/Fosspoly1/foss_poly_1.inp",
        licence=LIC_NOT_STATED, note="ASCE TF WDST (upstream name foss_poly_1.inp)"),
    "BWSN_Network_1.inp": dict(
        kind="url",
        url=KIOS + "asce-tf-wdst/Battle%20of%20the%20Water%20Sensor%20Networks/"
                   "BWSN_Network_1.inp",
        licence=LIC_NOT_STATED, note="BWSN benchmark"),
    "BWSN_Network_2.inp": dict(
        kind="url",
        url=KIOS + "asce-tf-wdst/Battle%20of%20the%20Water%20Sensor%20Networks/"
                   "BWSN_Network_2.inp",
        licence=LIC_NOT_STATED, note="BWSN benchmark, 12527 nodes"),
    "Richmond_skeleton.inp": dict(
        kind="url", url=KIOS + "collect-epanet-inp/Richmond_skeleton.inp",
        licence=LIC_NOT_STATED, note="Exeter CWS benchmark"),
    "Richmond_standard.inp": dict(
        kind="url", url=KIOS + "collect-epanet-inp/Richmond_standard.inp",
        licence=LIC_NOT_STATED, note="Exeter CWS benchmark"),
    "D-Town.inp": dict(
        kind="url",
        url=KIOS + "exeter-benchmarks/"
                   "D-Town%20Water%20Distribution%20Network%20BWN-II/d-town.inp",
        licence=LIC_NOT_STATED, note="BWN-II benchmark"),
    # --- licensed sources ----------------------------------------------------
    "C-Town_BATADAL.inp": dict(
        kind="url",
        url="https://raw.githubusercontent.com/scy-phy/www.batadal.net/master/"
            "data/CTOWN.INP",
        licence=LIC_CCBY, note="BATADAL C-Town"),
    "Modena.inp": dict(kind="url",
                       url=WBH + "network-modena/modena.inp",
                       licence=LIC_MIT,
                       note="WaterBenchmarkHub; network from Bragalli et al. 2008"),
    "Pescara.inp": dict(kind="url",
                        url=WBH + "network-%20Pescara/PES.inp",
                        licence=LIC_MIT,
                        note="WaterBenchmarkHub; network from Bragalli et al. 2008"),
    "L-TOWN.inp": dict(
        kind="url",
        url="https://zenodo.org/api/records/4017659/files/L-TOWN.inp/content",
        licence=LIC_CCBY, note="BattLeDIM 2020, Zenodo record 4017659"),
    "L-TOWN_Real.inp": dict(
        kind="url",
        url="https://zenodo.org/api/records/4017659/files/L-TOWN_Real.inp/content",
        licence=LIC_CCBY, note="BattLeDIM 2020, 107 demand patterns",
        large=True),
}

# SHA-256 of the exact bytes used for data/benchmark_report.txt (2026-08-09).
SHA256: dict[str, str] = {
    "Anytown.inp": "8a6ae1ed89f81326c4e2f5b65adf814d1ce0e0687b2b4b20adb464b793f29135",
    "Anytown_wntr.inp": "af3536342cba90cdfdbad4a22173fcf1a8cc7ec1313f4f4c9d2311f3fb917809",
    "Balerma.inp": "f18248aa7b1deff7646018a1081fcb98554e634dd9135259c86518f4eac99743",
    "BWSN_Network_1.inp": "510af942ec643eb87adcf26e5a7df1cc4c23eb0c1e470a8a1257ed055f1956f1",
    "BWSN_Network_2.inp": "232e17c02386dae436d8212346c757fa3ce52593837ef809caa29a3f73aceb3f",
    "C-Town_BATADAL.inp": "9198b0bb43f551b29bfb550e206ff95f1df220118fc0068c7f130cec78d58d9e",
    "D-Town.inp": "a4fc184845d7d7a9c7089b577d2b41ae0fcea9bc0b91996be5503884f914af4f",
    "Fossolo_poly1.inp": "de69a9741eda582fe96bc33858caa78b205f5ff2b12ea1527434378e4f50e01e",
    "Hanoi.inp": "941a08e1d9d2a698eb74e482a37df794affd059e3e0f462c8b3c77ba5f0fbe9c",
    "ky10.inp": "1cca0fe154cc80cb6ea63c085d3c34430e2c3dfea9efa99eb130c7288e8ca950",
    "ky4.inp": "0f776ada1c8fb17dad50d04b8035b4b96421de5c85ee7756697d6df28d4f2579",
    "L-TOWN.inp": "a7551b86745f4cc3433c78e60023077fa1386947d4d35372ffd3a0405622b436",
    "L-TOWN_Real.inp": "4f4765ed61afad1b79450ef491563599a998f8a4ea0adcbb9505803236c6803a",
    "Modena.inp": "65ad9991753c9280a5871aa247ed178be18713f93136ed88f929330feeb3ab32",
    "Net1.inp": "607510a01287d60d27b280a39df31a001363175a438a5de1b39e749cec6ddbc8",
    "Net2.inp": "7c140a40f9d43ec54c155783085f9f6403df6ea7e93df1f9ad4bbf35b6c28fb0",
    "Net3.inp": "ea3e825c4fef0b5cba47fb06301bc85253f18b6364dc96c44d9fb492c40faa52",
    "Net6.inp": "9a2ac6412469d4a5dc6352fc249f0c9841047ad1b908e0b7051faf1b55dcafab",
    "Pescara.inp": "3908c7b6304b1399eb53bb98960e7d522fef21062c24ff6406d21d68aa7541f5",
    "Richmond_skeleton.inp": "32737b69a99ad73a9b8eea5e19945204e42ae23a31b9524bb11c4ec8195d4741",
    "Richmond_standard.inp": "d3c270d8ff5d5ebd6fb8ccb4853673480734291a457ddbe35150d9004739dec4",
}


# --------------------------------------------------------------------------- utils
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n}"


def wntr_root() -> str:
    import wntr  # imported lazily so --list works without wntr
    return os.path.dirname(os.path.abspath(wntr.__file__))


def print_licence(name: str, spec: dict) -> None:
    lic = spec["licence"]
    print(f"  licence : {lic}")
    if lic not in CLEAR:
        print("  [LICENCE] not stated upstream - academic use; "
              "verify permissions yourself.")


# --------------------------------------------------------------------------- fetch
def download(url: str, dest: str) -> int:
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    got = 0
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        total = resp.headers.get("Content-Length")
        total = int(total) if total and total.isdigit() else None
        last = t0
        while True:
            blk = resp.read(CHUNK)
            if not blk:
                break
            out.write(blk)
            got += len(blk)
            now = time.time()
            if total and total > 8 * CHUNK and now - last > 2.0:
                print(f"    {human(got)} / {human(total)} "
                      f"({100.0 * got / total:.0f}%)", flush=True)
                last = now
    os.replace(tmp, dest)
    print(f"    downloaded {human(got)} in {time.time() - t0:.1f}s")
    return got


def fetch_one(name: str, spec: dict, force: bool) -> str:
    """Return one of: 'skip-exists', 'ok', 'hash-mismatch', 'error'."""
    dest = os.path.join(DEST, name)
    want = SHA256.get(name)

    if os.path.exists(dest) and not force:
        have = sha256_of(dest)
        if want is None:
            print(f"  present ({human(os.path.getsize(dest))}), no manifest entry "
                  f"-> kept, sha256={have[:16]}...")
            return "skip-exists"
        if have == want:
            print(f"  present and sha256 OK ({human(os.path.getsize(dest))}) -> skip")
            return "skip-exists"
        print(f"  present but sha256 MISMATCH\n    have {have}\n    want {want}")
        print("  re-fetching (use --keep-mismatched to leave it alone)")

    os.makedirs(DEST, exist_ok=True)
    try:
        if spec["kind"] == "wntr":
            src = os.path.join(wntr_root(), *spec["rel"].split("/"))
            if not os.path.exists(src):
                print(f"  ERROR: not found in the installed wntr: {src}")
                return "error"
            shutil.copyfile(src, dest)
            print(f"    copied from {src} ({human(os.path.getsize(dest))})")
        else:
            print(f"    GET {spec['url']}")
            download(spec["url"], dest)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}")
        if os.path.exists(dest + ".part"):
            os.remove(dest + ".part")
        return "error"

    if want is None:
        print(f"  sha256={sha256_of(dest)} (no manifest entry to compare)")
        return "ok"
    have = sha256_of(dest)
    if have != want:
        print(f"  sha256 MISMATCH after fetch\n    have {have}\n    want {want}")
        print("  the upstream file changed; do NOT use it to reproduce the "
              "published tables")
        return "hash-mismatch"
    print(f"  sha256 OK ({have[:16]}...)")
    return "ok"


# --------------------------------------------------------------------------- main
def select(args) -> list[str]:
    names = list(SOURCES)
    if args.names:
        want = {n.lower().removesuffix(".inp") for n in args.names}
        names = [n for n in names if n.lower().removesuffix(".inp") in want]
        missing = want - {n.lower().removesuffix(".inp") for n in names}
        if missing:
            print(f"unknown network(s): {sorted(missing)}")
            print(f"known: {sorted(SOURCES)}")
            return []
        return names
    if not args.with_large:
        names = [n for n in names if not SOURCES[n].get("large")]
    if args.only_licensed:
        names = [n for n in names if SOURCES[n]["licence"] in CLEAR]
    return names


def main(argv=None) -> int:
    global DEST
    default_dest = DEST
    ap = argparse.ArgumentParser(
        description="Rebuild networks/public/ from upstream sources.")
    ap.add_argument("names", nargs="*",
                    help="only these networks (e.g. Hanoi Net3); implies "
                         "--with-large for the ones you name")
    ap.add_argument("--with-large", action="store_true",
                    help="also fetch L-TOWN_Real.inp (169 MiB)")
    ap.add_argument("--only-licensed", action="store_true",
                    help="skip benchmarks whose upstream states no licence")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even if the file is present and hashes OK")
    ap.add_argument("--verify", action="store_true",
                    help="only re-hash the files already in networks/public/")
    ap.add_argument("--list", action="store_true",
                    help="print the source/licence table and exit")
    ap.add_argument("--dest", default=default_dest,
                    help=f"destination dir (default {default_dest})")
    args = ap.parse_args(argv)

    DEST = os.path.abspath(args.dest)

    if args.list:
        print(f"{'network':<24}{'licence':<44}source")
        for n, s in SOURCES.items():
            src = s["url"] if s["kind"] == "url" else f"wntr:{s['rel']}"
            print(f"{n:<24}{s['licence']:<44}{src}")
        print("\n'not stated upstream' = the upstream repository carries no "
              "LICENSE file.\nAcademic use; verify permissions yourself. "
              "Nothing here is redistributed by this project.")
        return 0

    names = select(args)
    if not names:
        return 2

    print(f"destination : {DEST}")
    print(f"networks    : {len(names)}"
          f"{'  (including L-TOWN_Real.inp, 169 MiB)' if 'L-TOWN_Real.inp' in names else ''}")
    print("=" * 78)

    if args.verify:
        bad = 0
        for n in names:
            p = os.path.join(DEST, n)
            if not os.path.exists(p):
                print(f"{n:<24} MISSING")
                bad += 1
                continue
            have = sha256_of(p)
            want = SHA256.get(n)
            ok = (want is None) or (have == want)
            print(f"{n:<24} {'OK  ' if ok else 'BAD '} {have}")
            bad += 0 if ok else 1
        print("=" * 78)
        print(f"verify: {len(names) - bad} OK, {bad} problem(s)")
        return 0 if bad == 0 else 1

    tally = {"ok": 0, "skip-exists": 0, "hash-mismatch": 0, "error": 0}
    problems = []
    for i, n in enumerate(names, 1):
        spec = SOURCES[n]
        print(f"[{i}/{len(names)}] {n}   ({spec['note']})")
        print_licence(n, spec)
        res = fetch_one(n, spec, args.force)
        tally[res] += 1
        if res in ("hash-mismatch", "error"):
            problems.append((n, res))
        print()

    print("=" * 78)
    print(f"fetched {tally['ok']}, already present {tally['skip-exists']}, "
          f"hash mismatch {tally['hash-mismatch']}, errors {tally['error']}")
    if problems:
        for n, r in problems:
            print(f"  {r:<14} {n}")
    n_unclear = sum(1 for n in names if SOURCES[n]["licence"] not in CLEAR)
    if n_unclear:
        print(f"\n{n_unclear} of these networks come from a repository that states "
              f"no licence.\nThey are used here as academic benchmarks; verify "
              f"permissions yourself before\nredistributing them. See THIRD_PARTY.md.")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

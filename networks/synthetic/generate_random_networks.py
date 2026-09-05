# -*- coding: utf-8 -*-
"""generate_random_networks.py: the generator behind networks/random_main and
networks/random_small.

The 23 synthetic networks shipped in this repository (20 under
``networks/random_main/``, 3 under ``networks/random_small/``) were produced by
this generator through the EPANET 2.2 toolkit (``EN_*`` project API, saved
with ``EN_saveinpfile``).  The generator logic below is the one that produced
them; only the toolkit binding was replaced by a self-contained ``ctypes``
wrapper around the EPANET 2.2 shared library that ships inside WNTR, so that
the script runs from this repository with no other code.

Two families, fixed seeds (see ``FAMILIES``):

    main   20 networks, seed 20260803  ->  networks/random_main/rand_0000..0019.inp
    small   3 networks, seed 11        ->  networks/random_small/rand_0000..0002.inp

Usage::

    python networks/synthetic/generate_random_networks.py --family main --out /tmp/rm
    python networks/synthetic/generate_random_networks.py --family small --out /tmp/rs
    python networks/synthetic/generate_random_networks.py --verify        # regenerate
        # both families into a temporary directory and compare SHA-256 with
        # networks/synthetic/SHA256SUMS.txt

Whether regeneration is byte-identical depends on the EPANET binary that
writes the file (``EN_saveinpfile`` formats every number with the C runtime's
``printf``).  The shipped files are the canonical artefacts; their SHA-256
values are recorded in ``SHA256SUMS.txt`` next to this script and the
``--verify`` mode reports, per file, whether your binary reproduces them.

How a network is built (unchanged from the original generator)
==============================================================
1. Geometry: Poisson-disk sampled points, Delaunay triangulated, reduced to the
   Gabriel graph (planar, road-like), then thinned towards a spanning tree by a
   tunable loop ratio.
2. Terrain: a tilted plane plus three low-frequency sinusoids.
3. Sources: reservoirs beside high-ground junctions, head set for a target
   service pressure.
4. Demands: log-normal per junction, rescaled to a target total.
5. Pipe sizing: demands accumulated along a shortest-path tree from the source,
   diameter from a target velocity, snapped to a commercial catalogue.
6. Acceptance: the candidate is solved once and kept only if service pressures
   and velocities land in operational bands.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import os
import sys
import tempfile
from ctypes import POINTER, byref, c_char_p, c_double, c_int, c_long, c_void_p
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
NETWORKS = os.path.dirname(HERE)

# (family, count, seed0, output directory relative to networks/)
FAMILIES = {
    "main": (20, 20260803, "random_main"),
    "small": (3, 11, "random_small"),
}

# ===========================================================================
# EPANET 2.2 toolkit constants used here (epanet2_enums.h)
# ===========================================================================
EN_JUNCTION, EN_RESERVOIR = 0, 1
EN_PIPE = 1
EN_ELEVATION, EN_PRESSURE = 0, 11
EN_VELOCITY = 9
EN_LINKCOUNT = 2
EN_DURATION = 0
EN_TRIALS, EN_ACCURACY = 0, 1
EN_NOSAVE = 0
EN_NOREPORT = 0
EN_LPS, EN_HW = 5, 0
EN_MAXMSG = 255


def find_epanet_library(explicit: str | None = None) -> str:
    """Locate the EPANET 2.2 shared library: an explicit path, or the one that
    ships inside the installed ``wntr`` package (the same library the rest of
    this repository uses as its reference engine)."""
    if explicit:
        return explicit
    import wntr  # noqa: WPS433 (optional dependency, imported lazily)

    base = os.path.join(os.path.dirname(wntr.__file__), "epanet", "libepanet")
    if sys.platform.startswith("win"):
        cand = os.path.join(base, "windows-x64", "epanet22.dll")
    elif sys.platform == "darwin":
        cand = os.path.join(base, "darwin-x64", "libepanet22.dylib")
    else:
        cand = os.path.join(base, "linux-x64", "libepanet22.so")
    if not os.path.exists(cand):
        raise FileNotFoundError("EPANET 2.2 shared library not found at " + cand)
    return cand


_SIG = [
    ("EN_createproject", [POINTER(c_void_p)]),
    ("EN_deleteproject", [c_void_p]),
    ("EN_init", [c_void_p, c_char_p, c_char_p, c_int, c_int]),
    ("EN_close", [c_void_p]),
    ("EN_saveinpfile", [c_void_p, c_char_p]),
    ("EN_getcount", [c_void_p, c_int, POINTER(c_int)]),
    ("EN_geterror", [c_int, c_char_p, c_int]),
    ("EN_setstatusreport", [c_void_p, c_int]),
    ("EN_openH", [c_void_p]),
    ("EN_initH", [c_void_p, c_int]),
    ("EN_runH", [c_void_p, POINTER(c_long)]),
    ("EN_closeH", [c_void_p]),
    ("EN_setoption", [c_void_p, c_int, c_double]),
    ("EN_settimeparam", [c_void_p, c_int, c_long]),
    ("EN_addnode", [c_void_p, c_char_p, c_int, POINTER(c_int)]),
    ("EN_getnodevalue", [c_void_p, c_int, c_int, POINTER(c_double)]),
    ("EN_setnodevalue", [c_void_p, c_int, c_int, c_double]),
    ("EN_setjuncdata", [c_void_p, c_int, c_double, c_double, c_char_p]),
    ("EN_setcoord", [c_void_p, c_int, c_double, c_double]),
    ("EN_addlink", [c_void_p, c_char_p, c_int, c_char_p, c_char_p, POINTER(c_int)]),
    ("EN_getlinkvalue", [c_void_p, c_int, c_int, POINTER(c_double)]),
    ("EN_setpipedata", [c_void_p, c_int, c_double, c_double, c_double, c_double]),
]

_LIB: ctypes.CDLL | None = None


def _lib(path: str | None = None) -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(find_epanet_library(path))
        for name, argtypes in _SIG:
            fn = getattr(_LIB, name)
            fn.restype = c_int
            fn.argtypes = argtypes
    return _LIB


class EpanetError(RuntimeError):
    pass


class EpanetProject:
    """Minimal EN_* project wrapper: errors (code >= 100) raise, warnings
    (1..99) are collected, exactly as in the original generator."""

    def __init__(self, dll: str | None = None) -> None:
        self.lib = _lib(dll)
        self._h = c_void_p()
        self._check(self.lib.EN_createproject(byref(self._h)), "EN_createproject")
        self._open = False
        self._h_open = False
        self.warnings: list[tuple[str, int, str]] = []

    def _check(self, code: int, func: str) -> int:
        if code == 0:
            return 0
        buf = ctypes.create_string_buffer(EN_MAXMSG + 1)
        self.lib.EN_geterror(int(code), buf, EN_MAXMSG)
        msg = buf.value.decode("utf-8", "replace")
        if code >= 100:
            raise EpanetError(f"{func}: EPANET error {code} - {msg}")
        self.warnings.append((func, code, msg))
        return code

    def init(self, flow_units: int, headloss: int, rpt_file: str, out_file: str) -> None:
        self._check(self.lib.EN_init(self._h, rpt_file.encode(), out_file.encode(),
                                     int(flow_units), int(headloss)), "EN_init")
        self._open = True

    def set_status_report(self, level: int) -> None:
        self._check(self.lib.EN_setstatusreport(self._h, int(level)), "EN_setstatusreport")

    def set_time_param(self, param: int, value: int) -> None:
        self._check(self.lib.EN_settimeparam(self._h, int(param), int(value)), "EN_settimeparam")

    def set_option(self, option: int, value: float) -> None:
        self._check(self.lib.EN_setoption(self._h, int(option), float(value)), "EN_setoption")

    def add_node(self, node_id: str, node_type: int) -> int:
        i = c_int()
        self._check(self.lib.EN_addnode(self._h, node_id.encode(), int(node_type), byref(i)),
                    "EN_addnode")
        return i.value

    def set_junction_data(self, index: int, elev: float, demand: float, pattern: str = "") -> None:
        self._check(self.lib.EN_setjuncdata(self._h, int(index), float(elev), float(demand),
                                            pattern.encode()), "EN_setjuncdata")

    def set_coord(self, index: int, x: float, y: float) -> None:
        self._check(self.lib.EN_setcoord(self._h, int(index), float(x), float(y)), "EN_setcoord")

    def set_node_value(self, index: int, prop: int, value: float) -> None:
        self._check(self.lib.EN_setnodevalue(self._h, int(index), int(prop), float(value)),
                    "EN_setnodevalue")

    def get_node_value(self, index: int, prop: int) -> float:
        v = c_double()
        self._check(self.lib.EN_getnodevalue(self._h, int(index), int(prop), byref(v)),
                    "EN_getnodevalue")
        return v.value

    def add_link(self, link_id: str, link_type: int, from_node: str, to_node: str) -> int:
        i = c_int()
        self._check(self.lib.EN_addlink(self._h, link_id.encode(), int(link_type),
                                        from_node.encode(), to_node.encode(), byref(i)),
                    "EN_addlink")
        return i.value

    def set_pipe_data(self, index: int, length: float, diam: float, rough: float,
                      minor_loss: float) -> None:
        self._check(self.lib.EN_setpipedata(self._h, int(index), float(length), float(diam),
                                            float(rough), float(minor_loss)), "EN_setpipedata")

    def get_link_value(self, index: int, prop: int) -> float:
        v = c_double()
        self._check(self.lib.EN_getlinkvalue(self._h, int(index), int(prop), byref(v)),
                    "EN_getlinkvalue")
        return v.value

    @property
    def num_links(self) -> int:
        n = c_int()
        self._check(self.lib.EN_getcount(self._h, EN_LINKCOUNT, byref(n)), "EN_getcount")
        return n.value

    def open_h(self) -> None:
        self._check(self.lib.EN_openH(self._h), "EN_openH")
        self._h_open = True

    def init_h(self, flag: int) -> None:
        self._check(self.lib.EN_initH(self._h, int(flag)), "EN_initH")

    def run_h(self) -> int:
        t = c_long()
        self._check(self.lib.EN_runH(self._h, byref(t)), "EN_runH")
        return t.value

    def close_h(self) -> None:
        if self._h_open:
            self._check(self.lib.EN_closeH(self._h), "EN_closeH")
            self._h_open = False

    def save_inp(self, path: str) -> None:
        self._check(self.lib.EN_saveinpfile(self._h, str(path).encode()), "EN_saveinpfile")

    def delete(self) -> None:
        if self._h:
            if self._h_open:
                self.lib.EN_closeH(self._h)
                self._h_open = False
            if self._open:
                self.lib.EN_close(self._h)
                self._open = False
            self.lib.EN_deleteproject(self._h)
            self._h = c_void_p()


# ===========================================================================
# Generator (verbatim logic of the original; the RNG call order is what makes
# the seeds reproduce the shipped files)
# ===========================================================================

#: Commercial ductile-iron / PVC nominal diameters, mm.
DIAMETER_CATALOGUE = np.array(
    [80, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800],
    dtype=float,
)


@dataclass
class NetworkSpec:
    """Knobs for one synthetic network."""

    n_junctions: int = 60
    extent_m: float = 3000.0            # side of the square service area
    loop_ratio: float = 0.75            # 1.0 = keep all Gabriel edges (looped)
                                        # 0.0 = spanning tree only (branched)
    n_reservoirs: int = 1
    elevation_range_m: float = 40.0
    service_head_m: float = 45.0        # target pressure at the critical node
    total_demand_lps: float = 60.0
    demand_cv: float = 0.8              # log-normal coefficient of variation
    target_velocity_ms: float = 1.0
    roughness_range: tuple[float, float] = (100.0, 140.0)   # Hazen-Williams C
    tortuosity: tuple[float, float] = (1.05, 1.25)
    seed: int = 0


def poisson_disk(rng: np.random.Generator, n: int, extent: float,
                 k: int = 30) -> np.ndarray:
    """Simple rejection-based blue-noise sampling of ``n`` points in a square."""
    min_dist = 0.7 * extent / math.sqrt(n)
    pts: list[np.ndarray] = [rng.uniform(0, extent, 2)]
    tries = 0
    while len(pts) < n and tries < 200 * n:
        cand = rng.uniform(0, extent, 2)
        arr = np.asarray(pts)
        if np.min(np.linalg.norm(arr - cand, axis=1)) >= min_dist:
            pts.append(cand)
        tries += 1
    while len(pts) < n:                     # fall back to pure uniform
        pts.append(rng.uniform(0, extent, 2))
    return np.asarray(pts)


def gabriel_edges(points: np.ndarray) -> list[tuple[int, int]]:
    """Gabriel graph: keep Delaunay edge (i,j) iff the disk with diameter ij
    contains no third point.  Yields a planar, road-like layout."""
    tri = Delaunay(points)
    cand: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                i, j = int(simplex[a]), int(simplex[b])
                cand.add((min(i, j), max(i, j)))

    kept: list[tuple[int, int]] = []
    for i, j in sorted(cand):
        centre = 0.5 * (points[i] + points[j])
        radius = 0.5 * np.linalg.norm(points[i] - points[j])
        d = np.linalg.norm(points - centre, axis=1)
        d[i] = d[j] = np.inf
        if np.all(d >= radius - 1e-9):
            kept.append((i, j))
    return kept


def _mst_edges(points: np.ndarray, edges: list[tuple[int, int]]
               ) -> set[tuple[int, int]]:
    """Prim MST over the given edge set, weighted by Euclidean length."""
    import heapq

    n = len(points)
    adj: dict[int, list[tuple[float, int, tuple[int, int]]]] = {i: [] for i in range(n)}
    for i, j in edges:
        w = float(np.linalg.norm(points[i] - points[j]))
        adj[i].append((w, j, (i, j)))
        adj[j].append((w, i, (i, j)))

    seen = {0}
    heap = list(adj[0])
    heapq.heapify(heap)
    out: set[tuple[int, int]] = set()
    while heap and len(seen) < n:
        w, nxt, e = heapq.heappop(heap)
        if nxt in seen:
            continue
        seen.add(nxt)
        out.add(e)
        for item in adj[nxt]:
            if item[1] not in seen:
                heapq.heappush(heap, item)
    return out


def thin_edges(points: np.ndarray, edges: list[tuple[int, int]],
               loop_ratio: float, rng: np.random.Generator
               ) -> list[tuple[int, int]]:
    """Drop non-MST edges to reach the requested looped/branched mix."""
    backbone = _mst_edges(points, edges)
    optional = [e for e in edges if e not in backbone]
    rng.shuffle(optional)
    keep = int(round(loop_ratio * len(optional)))
    return sorted(backbone | set(optional[:keep]))


def smooth_elevation(points: np.ndarray, extent: float, span: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Tilted plane + three low-frequency sinusoids, rescaled to ``span``."""
    x = points[:, 0] / extent
    y = points[:, 1] / extent
    theta = rng.uniform(0, 2 * np.pi)
    z = np.cos(theta) * x + np.sin(theta) * y
    for _ in range(3):
        kx, ky = rng.uniform(0.5, 2.5, 2)
        ph = rng.uniform(0, 2 * np.pi)
        z = z + 0.35 * np.sin(2 * np.pi * (kx * x + ky * y) + ph)
    z = z - z.min()
    if z.max() > 0:
        z = z / z.max()
    return z * span


def size_pipes(points: np.ndarray, edges: list[tuple[int, int]],
               lengths: np.ndarray, demands_lps: np.ndarray, source: int,
               target_velocity: float) -> np.ndarray:
    """Diameters (mm) from accumulated demand on a shortest-path tree."""
    import heapq

    n = len(points)
    adj: dict[int, list[tuple[int, float, int]]] = {i: [] for i in range(n)}
    for e, (i, j) in enumerate(edges):
        adj[i].append((j, float(lengths[e]), e))
        adj[j].append((i, float(lengths[e]), e))

    dist = np.full(n, np.inf)
    parent_edge = np.full(n, -1, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w, e in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                parent[v] = u
                parent_edge[v] = e
                heapq.heappush(heap, (dist[v], v))

    order = np.argsort(-dist)
    acc = demands_lps.astype(float).copy()
    edge_flow = np.zeros(len(edges))
    for v in order:
        pe = parent_edge[v]
        if pe >= 0:
            edge_flow[pe] += acc[v]
            acc[parent[v]] += acc[v]

    on_tree = edge_flow > 0
    q_m3s = np.maximum(edge_flow, 1e-6) / 1000.0
    d_m = np.sqrt(4.0 * q_m3s / (np.pi * target_velocity))
    d_mm = d_m * 1000.0

    if np.any(~on_tree):
        node_size = np.zeros(n)
        cnt = np.zeros(n)
        for e, (i, j) in enumerate(edges):
            if on_tree[e]:
                node_size[i] += d_mm[e]; cnt[i] += 1
                node_size[j] += d_mm[e]; cnt[j] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            node_size = np.where(cnt > 0, node_size / np.maximum(cnt, 1), 100.0)
        for e in np.flatnonzero(~on_tree):
            i, j = edges[e]
            d_mm[e] = 0.6 * min(node_size[i], node_size[j])

    idx = np.searchsorted(DIAMETER_CATALOGUE, d_mm)
    idx = np.clip(idx, 0, len(DIAMETER_CATALOGUE) - 1)
    return DIAMETER_CATALOGUE[idx]


def build_network(spec: NetworkSpec, out_inp: str, dll: str | None = None) -> dict:
    """Generate one network and write it as an ``.inp`` file."""
    rng = np.random.default_rng(spec.seed)
    n = spec.n_junctions

    points = poisson_disk(rng, n, spec.extent_m)
    edges = gabriel_edges(points)
    edges = thin_edges(points, edges, spec.loop_ratio, rng)
    elev = smooth_elevation(points, spec.extent_m, spec.elevation_range_m, rng)

    tort = rng.uniform(*spec.tortuosity, size=len(edges))
    lengths = np.array([np.linalg.norm(points[i] - points[j])
                        for i, j in edges]) * tort
    lengths = np.maximum(lengths, 10.0)

    sigma = math.sqrt(math.log(1.0 + spec.demand_cv**2))
    raw = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n)
    demands = raw / raw.sum() * spec.total_demand_lps

    source_junction = int(np.argmax(elev))
    diam_mm = size_pipes(points, edges, lengths, demands, source_junction,
                         spec.target_velocity_ms)
    rough = rng.uniform(*spec.roughness_range, size=len(edges))

    p = EpanetProject(dll)
    p.init(EN_LPS, EN_HW, os.devnull, "")
    p.set_status_report(EN_NOREPORT)
    p.set_time_param(EN_DURATION, 0)
    p.set_option(EN_ACCURACY, 1e-8)
    p.set_option(EN_TRIALS, 500)

    junction_ids = []
    for i in range(n):
        jid = f"J{i+1}"
        idx = p.add_node(jid, EN_JUNCTION)
        p.set_junction_data(idx, float(elev[i]), float(demands[i]), "")
        p.set_coord(idx, float(points[i, 0]), float(points[i, 1]))
        junction_ids.append(jid)

    src_candidates = list(np.argsort(-elev)[: max(spec.n_reservoirs * 3, 3)])
    rng.shuffle(src_candidates)
    for r in range(spec.n_reservoirs):
        j = int(src_candidates[r % len(src_candidates)])
        rid = f"R{r+1}"
        ridx = p.add_node(rid, EN_RESERVOIR)
        head = float(elev[j]) + spec.service_head_m + 0.5 * spec.elevation_range_m
        p.set_node_value(ridx, EN_ELEVATION, head)
        p.set_coord(ridx, float(points[j, 0]) - 50.0, float(points[j, 1]) - 50.0)
        lid = f"PS{r+1}"
        lidx = p.add_link(lid, EN_PIPE, rid, junction_ids[j])
        p.set_pipe_data(lidx, 50.0, float(DIAMETER_CATALOGUE[-1]), 140.0, 0.0)

    for e, (i, j) in enumerate(edges):
        lid = f"P{e+1}"
        lidx = p.add_link(lid, EN_PIPE, junction_ids[i], junction_ids[j])
        p.set_pipe_data(lidx, float(lengths[e]), float(diam_mm[e]),
                        float(rough[e]), 0.0)

    p.open_h()
    p.init_h(EN_NOSAVE)
    p.run_h()
    press = np.array([p.get_node_value(i, EN_PRESSURE) for i in range(1, n + 1)])
    vel = np.array([p.get_link_value(k, EN_VELOCITY)
                    for k in range(1, p.num_links + 1)])
    p.close_h()

    meta = {
        "n_junctions": n,
        "n_links": p.num_links,
        "n_reservoirs": spec.n_reservoirs,
        "loop_ratio": spec.loop_ratio,
        "seed": spec.seed,
        "extent_m": spec.extent_m,
        "total_demand_lps": float(spec.total_demand_lps),
        "min_pressure_m": float(press.min()),
        "max_pressure_m": float(press.max()),
        "mean_pressure_m": float(press.mean()),
        "max_velocity_ms": float(vel.max()),
        "mean_velocity_ms": float(vel.mean()),
        "n_edges_gabriel": len(edges),
    }

    p.save_inp(str(out_inp))
    p.delete()
    meta["inp"] = str(out_inp)
    return meta


def accept(meta: dict, min_pressure_m: float = 15.0,
           max_pressure_m: float = 120.0, max_velocity_ms: float = 4.0) -> bool:
    """Operational plausibility gate (generous bands: stressed systems stay,
    broken ones go)."""
    return (meta["min_pressure_m"] >= min_pressure_m
            and meta["max_pressure_m"] <= max_pressure_m
            and meta["max_velocity_ms"] <= max_velocity_ms)


def generate_family(out_dir: str, n_networks: int, seed0: int = 0,
                    max_attempts_each: int = 12, verbose: bool = True,
                    dll: str | None = None) -> list[dict]:
    """Generate ``n_networks`` accepted networks with randomised specs."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed0)
    accepted: list[dict] = []
    attempt = 0

    while len(accepted) < n_networks and attempt < n_networks * max_attempts_each:
        attempt += 1
        spec = NetworkSpec(
            n_junctions=int(rng.integers(30, 121)),
            extent_m=float(rng.uniform(1500, 5000)),
            loop_ratio=float(rng.uniform(0.15, 1.0)),
            n_reservoirs=int(rng.integers(1, 3)),
            elevation_range_m=float(rng.uniform(5, 60)),
            service_head_m=float(rng.uniform(35, 60)),
            total_demand_lps=float(rng.uniform(20, 150)),
            demand_cv=float(rng.uniform(0.4, 1.2)),
            target_velocity_ms=float(rng.uniform(0.7, 1.4)),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        name = f"rand_{len(accepted):04d}.inp"
        path = os.path.join(str(out_dir), name)
        try:
            meta = build_network(spec, path, dll)
        except Exception as exc:            # degenerate geometry, solver failure
            if verbose:
                print(f"  attempt {attempt}: rejected ({type(exc).__name__}: {exc})")
            continue
        if accept(meta):
            meta["spec"] = spec.__dict__
            accepted.append(meta)
            if verbose:
                print(f"  [{len(accepted)}/{n_networks}] {name}  "
                      f"n={meta['n_junctions']} links={meta['n_links']} "
                      f"loop={spec.loop_ratio:.2f} "
                      f"p=[{meta['min_pressure_m']:.1f},{meta['max_pressure_m']:.1f}]m "
                      f"vmax={meta['max_velocity_ms']:.2f}m/s")
        else:
            if os.path.exists(path):
                os.remove(path)
            if verbose:
                print(f"  attempt {attempt}: rejected "
                      f"(p=[{meta['min_pressure_m']:.1f},"
                      f"{meta['max_pressure_m']:.1f}]m "
                      f"vmax={meta['max_velocity_ms']:.2f})")
    return accepted


# ===========================================================================
# CLI
# ===========================================================================
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict[str, str]:
    out = {}
    with open(os.path.join(HERE, "SHA256SUMS.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(None, 1)
            out[name.strip()] = digest
    return out


def verify(dll: str | None, verbose: bool) -> int:
    manifest = _load_manifest()
    tmp = tempfile.mkdtemp(prefix="wdsgpu_synthetic_")
    n_ok = n_bad = 0
    for fam, (count, seed0, sub) in FAMILIES.items():
        out_dir = os.path.join(tmp, sub)
        print(f"== family {fam}: {count} networks, seed {seed0} -> {out_dir}")
        generate_family(out_dir, count, seed0=seed0, verbose=verbose, dll=dll)
        for i in range(count):
            rel = f"{sub}/rand_{i:04d}.inp"
            got = _sha256(os.path.join(out_dir, f"rand_{i:04d}.inp"))
            want = manifest.get(rel)
            ok = got == want
            n_ok += ok; n_bad += (not ok)
            print(f"  {'OK  ' if ok else 'DIFF'} {rel}  {got[:16]}  (shipped {str(want)[:16]})")
    print(f"regenerated {n_ok + n_bad} files: {n_ok} byte-identical to the shipped copies, {n_bad} differ")
    print(f"(temporary output left in {tmp})")
    return 0 if n_bad == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", choices=sorted(FAMILIES), help="which family to generate")
    ap.add_argument("--out", help="output directory (default: the family's directory under networks/)")
    ap.add_argument("--dll", default=None, help="explicit path to the EPANET 2.2 shared library")
    ap.add_argument("--verify", action="store_true",
                    help="regenerate both families into a temporary directory and compare "
                         "SHA-256 with SHA256SUMS.txt")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        return verify(a.dll, not a.quiet)
    if not a.family:
        ap.error("--family is required unless --verify is given")
    count, seed0, sub = FAMILIES[a.family]
    out = a.out or os.path.join(NETWORKS, sub)
    print(f"family {a.family}: {count} networks, seed {seed0} -> {out}")
    metas = generate_family(out, count, seed0=seed0, verbose=not a.quiet, dll=a.dll)
    print(f"wrote {len(metas)} networks")
    return 0 if len(metas) == count else 1


if __name__ == "__main__":
    sys.exit(main())

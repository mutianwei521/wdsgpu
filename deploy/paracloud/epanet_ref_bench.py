# -*- coding: utf-8 -*-
"""Same-machine EPANET 2.2 reference timing through the EN_* API, matched to
the paper's protocol: cold-start EN_initH(EN_INITFLOW) + one EN_runH per
scenario. EN_solveH would run the whole 24 h EPS and write the output file."""
import ctypes, json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.epanet_ref import Epanet, _dll_path, EN_BASEDEMAND, EN_HEAD

INP = os.path.join(ROOT, "city_d.inp")
print("library :", os.path.basename(_dll_path()))
SEED, B = 20260808, 64
EN_INITFLOW = 10
en = Epanet(INP)
cnt = en.counts(); nN = cnt["nodes"]
print("network : N=%d  L=%d" % (nN, cnt["links"]))

val = ctypes.c_double(); _t = ctypes.c_long()
base = np.empty(nN)
for i in range(1, nN + 1):
    en._check(en.lib.EN_getnodevalue(en._ph, i, EN_BASEDEMAND, ctypes.byref(val)), "get base")
    base[i - 1] = val.value
rng = np.random.default_rng(SEED)
fac = rng.uniform(0.8, 1.2, size=(B, nN))

en._check(en.lib.EN_openH(en._ph), "EN_openH")

def set_demands(b):
    for i in range(1, nN + 1):
        en._check(en.lib.EN_setnodevalue(en._ph, i, EN_BASEDEMAND,
                  ctypes.c_double(base[i - 1] * fac[b, i - 1])), "set base")
def solve():
    en._check(en.lib.EN_initH(en._ph, EN_INITFLOW), "EN_initH")
    en._check(en.lib.EN_runH(en._ph, ctypes.byref(_t)), "EN_runH")
def read_heads():
    h = np.empty(nN)
    for i in range(1, nN + 1):
        en._check(en.lib.EN_getnodevalue(en._ph, i, EN_HEAD, ctypes.byref(val)), "get head")
        h[i - 1] = val.value
    return h

set_demands(0); solve(); read_heads()

res = {}
t0 = time.perf_counter()
for _ in range(B): solve()
res["solve_only_ms"] = (time.perf_counter() - t0) / B * 1e3
t0 = time.perf_counter()
for b in range(B): set_demands(b); solve(); read_heads()
res["end_to_end_ms"] = (time.perf_counter() - t0) / B * 1e3
t0 = time.perf_counter()
for b in range(B): set_demands(b)
res["set_demands_ms"] = (time.perf_counter() - t0) / B * 1e3
t0 = time.perf_counter()
for _ in range(B): read_heads()
res["read_heads_ms"] = (time.perf_counter() - t0) / B * 1e3
for k in ("solve_only_ms", "set_demands_ms", "read_heads_ms", "end_to_end_ms"):
    print("EPANET %-16s %8.3f ms/scenario" % (k[:-3], res[k]))
json.dump(res, open(os.path.join(ROOT, "epanet_ref_bench.json"), "w"), indent=1)

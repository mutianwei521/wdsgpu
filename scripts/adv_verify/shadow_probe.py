# -*- coding: utf-8 -*-
"""Shadow-package probe: run the DEFAULT path and hash every float bit.

Invoked twice, once with the pre-session dgga on sys.path and once with the
current one.  If exact_fixed_inputs_from_inp really is non-default, the two
digests must be identical.
"""
import hashlib
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
PKG = sys.argv[1]                    # directory that CONTAINS the dgga package
import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import dgga                                          # noqa: E402
from dgga.parse import Net                           # noqa: E402
from dgga.eps import EpsDriver                       # noqa: E402
from align import resolve_inp                        # noqa: E402

assert os.path.dirname(os.path.abspath(dgga.__file__)) == \
    os.path.join(os.path.abspath(PKG), "dgga"), dgga.__file__
print("dgga from:", dgga.__file__)
print("has exact_fixed_inputs_from_inp:",
      hasattr(__import__("dgga.parse", fromlist=["x"]),
              "exact_fixed_inputs_from_inp"))

h = hashlib.sha256()
REF = os.path.join(ROOT, "data", "reference")
for stem, nframe in [("pub_net3", 12), ("pub_anytown", 9), ("city_d", 6),
                     ("pub_bwsn_network_1", 8), ("EXA5", 10)]:
    net = Net.load(REF, stem)
    inp = resolve_inp(stem)
    drv = EpsDriver(net, inp_path=inp if os.path.isfile(inp) else None)
    s = drv.solver
    drv._inithyd()
    for f in range(nframe):
        drv._demands()
        drv._controls()
        r = s.run_gga(drv.d, drv.H, q0=drv.q, e0=drv.e, status0=drv.S,
                      setting0=drv.K, do_status=True)
        drv.q, drv.e, drv.S = r["flow"], r["emitter"], r["status"]
        drv.K, drv.H = r["setting"], r["head"]
        drv.fixed_dem = r["fixed_demand"]
        drv.node_dem = np.where(s.is_fixed_node, drv.fixed_dem, drv.d + drv.e)
        for a in (drv.H, drv.q, drv.K):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(drv.S, dtype=np.int64).tobytes())
        drv._nexthyd(float(r["relerr"]))
    # also hash the parsed fixed inputs themselves
    h.update(np.ascontiguousarray(net.elev_ft, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(net.valve_setting_user, dtype=np.float64).tobytes())
print("DIGEST", h.hexdigest())

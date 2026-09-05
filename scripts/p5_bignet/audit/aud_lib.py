# -*- coding: utf-8 -*-
"""P5 敌意审阅 - 公共库（**不复用 scripts/p5_bignet/p5lib.py**，自己写一份）。

铁律零：全程 import HEAD 的只读副本。设 AUD_DGGA_ROOT 指向副本的父目录。
"""
import ctypes
import os
import sys

_RO = os.environ.get("AUD_DGGA_ROOT")
if _RO:
    sys.path.insert(0, _RO)

import numpy as np  # noqa: E402


def prov():
    import dgga
    import hashlib
    d = os.path.dirname(os.path.abspath(dgga.__file__))
    out = [("dgga path", d)]
    for f in ("solver.py", "autodiff.py", "parse.py"):
        p = os.path.join(d, f)
        out.append(("md5 " + f, hashlib.md5(open(p, "rb").read()).hexdigest()))
    return out


def fixed_head(net, t_sec=0):
    """定水头向量：reservoir 走 pattern，tank 用 H0；junction 位 nan。"""
    rh = np.array(net.reservoir_head_ft_at(t_sec), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh[tn] = np.asarray(net.tank_h0, dtype=np.float64)
    return rh


# ----------------------------------------------------------------- EPANET DLL
# 自己的 ctypes 绑定（不经 dgga.epanet_ref），只用到 6 个 API。
EN_HEAD = 10
EN_FLOW = 8
EN_ITERATIONS = 0
EN_RELATIVEERROR = 1
EN_JUNCTION = 0


def _dll():
    import wntr
    p = os.path.join(os.path.dirname(wntr.__file__), "epanet", "libepanet",
                     "windows-x64", "epanet22.dll")
    if not os.path.exists(p):
        raise RuntimeError("找不到 epanet22.dll: " + p)
    os.add_dll_directory(os.path.dirname(p))
    return ctypes.CDLL(p), p


class MyEpanet:
    """最小 EN_* 绑定：EN_createproject/open/openH/initH/runH/getnodevalue/
    getstatistic/getflowunits。返回**用户单位**，由调用方换算。"""

    def __init__(self, inp, rpt=None):
        self.lib, self.dllpath = _dll()
        self.ph = ctypes.c_void_p()
        self._ck(self.lib.EN_createproject(ctypes.byref(self.ph)), "createproject")
        rpt = rpt or (os.path.splitext(inp)[0] + ".audrpt")
        self._ck(self.lib.EN_open(self.ph, inp.encode("mbcs"),
                                  rpt.encode("mbcs"), b""), "open")
        self.rpt = rpt

    def _ck(self, code, where):
        if code > 100:
            buf = ctypes.create_string_buffer(256)
            try:
                self.lib.EN_geterror(code, buf, 255)
                msg = buf.value.decode("latin-1")
            except Exception:                       # noqa: BLE001
                msg = ""
            raise RuntimeError("EN_%s -> %d %s" % (where, code, msg))
        return code

    def flowunits(self):
        v = ctypes.c_int()
        self._ck(self.lib.EN_getflowunits(self.ph, ctypes.byref(v)), "getflowunits")
        return int(v.value)

    def counts(self):
        out = {}
        v = ctypes.c_int()
        for name, code in (("nodes", 0), ("tanks", 1), ("links", 2)):
            self._ck(self.lib.EN_getcount(self.ph, code, ctypes.byref(v)), "getcount")
            out[name] = int(v.value)
        return out

    def node_ids(self):
        n = self.counts()["nodes"]
        ids = []
        buf = ctypes.create_string_buffer(64)
        for i in range(1, n + 1):
            self._ck(self.lib.EN_getnodeid(self.ph, i, buf), "getnodeid")
            ids.append(buf.value.decode("latin-1"))
        return ids

    def run_first_step(self):
        """openH + initH(NOSAVE) + runH 一次 => t=0 的稳态帧。"""
        self._ck(self.lib.EN_openH(self.ph), "openH")
        self._ck(self.lib.EN_initH(self.ph, 0), "initH")
        t = ctypes.c_long(0)
        self._ck(self.lib.EN_runH(self.ph, ctypes.byref(t)), "runH")
        n = self.counts()
        val = ctypes.c_double()
        h = np.empty(n["nodes"])
        for i in range(1, n["nodes"] + 1):
            self._ck(self.lib.EN_getnodevalue(self.ph, i, EN_HEAD,
                                              ctypes.byref(val)), "getnodevalue")
            h[i - 1] = val.value
        q = np.empty(n["links"])
        for i in range(1, n["links"] + 1):
            self._ck(self.lib.EN_getlinkvalue(self.ph, i, EN_FLOW,
                                              ctypes.byref(val)), "getlinkvalue")
            q[i - 1] = val.value
        st = ctypes.c_double()
        self._ck(self.lib.EN_getstatistic(self.ph, EN_ITERATIONS,
                                          ctypes.byref(st)), "getstatistic")
        it = int(st.value)
        self._ck(self.lib.EN_getstatistic(self.ph, EN_RELATIVEERROR,
                                          ctypes.byref(st)), "getstatistic")
        return dict(head_user=h, flow_user=q, iters=it, relerr=float(st.value),
                    t=int(t.value))

    def close(self):
        try:
            self.lib.EN_closeH(self.ph)
        except Exception:                           # noqa: BLE001
            pass
        try:
            self.lib.EN_close(self.ph)
            self.lib.EN_deleteproject(self.ph)
        except Exception:                           # noqa: BLE001
            pass


def ucf_head_flow(flowunits):
    """用户单位 -> 内部单位（ft, cfs）的除数。types.h:68-83 的常数。"""
    GPMperCFS = 448.831
    LPSperCFS = 28.317
    LPMperCFS = 1699.0
    MLDperCFS = 2.4466
    CMHperCFS = 101.941
    CMDperCFS = 2446.6
    MGDperCFS = 0.64632
    IMGDperCFS = 0.5382
    AFDperCFS = 1.9837
    MperFT = 0.3048
    fl = {0: 1.0, 1: GPMperCFS, 2: MGDperCFS, 3: IMGDperCFS, 4: AFDperCFS,
          5: LPSperCFS, 6: LPMperCFS, 7: MLDperCFS, 8: CMHperCFS,
          9: CMDperCFS}[int(flowunits)]
    head = 1.0 if int(flowunits) <= 4 else MperFT
    return head, fl

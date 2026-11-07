# -*- coding: utf-8 -*-
"""diag_bwsn1.py - BWSN_Network_1 在 t=0 严格收敛设置下不收敛的诊断（WR 大修，2026-10-23）。

对照 EPANET 2.2 DLL 与 HydroGrad epanet 模式 run_gga（完整状态机，record_schedule=True）：
  精度/迭代：INP 自带 0.005/40；1e-8/1000（基线程序里 EPANET 的设置）；1e-12/200（solve_polished 内部）
  初始状态：INP 初始状态；以及执行 [CONTROLS] "LINK VALVE-180 Closed At Time 0" 后的状态
统计状态机轨迹里每条链路的切换次数、最后 10 次迭代仍在切换的链路、终态 PRV/泵状态。
只读，不改 dgga。运行：python -X utf8 scripts/diag_bwsn1.py
"""
import collections
import ctypes
import os
import shutil
import sys
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np                                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dgga import epanet_ref as er                              # noqa: E402
from dgga.parse import parse_inp                               # noqa: E402
from dgga.solver import GGASolver                              # noqa: E402

INP = os.path.join(ROOT, "networks", "public", "_cleaned", "BWSN_Network_1.inp")
EN_STATUS = 11
ST = {0: "XHEAD", 1: "TEMPCLOSED", 2: "CLOSED", 3: "OPEN", 4: "ACTIVE", 5: "XFLOW", 6: "XFCV", 7: "XPRESSURE"}


def ep_run(acc=None, trials=None, close180=False):
    e = er.Epanet(shutil.copy(INP, os.path.join(tempfile.mkdtemp(prefix="bw1_"), "net.inp")))
    if acc is not None:
        e.set_option(er.EN_ACCURACY, acc)
        e.set_option(er.EN_TRIALS, trials)
    lib, ph = e.lib, e._ph
    lib.EN_openH.argtypes = [ctypes.c_void_p]
    lib.EN_initH.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.EN_runH.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
    e._check(lib.EN_openH(ph), "EN_openH")
    lib.EN_initH(ph, 10)
    t = ctypes.c_long()
    rc = lib.EN_runH(ph, ctypes.byref(t))
    v = ctypes.c_double()
    lib.EN_getstatistic(ph, er.EN_ITERATIONS, ctypes.byref(v))
    it = int(v.value)
    lib.EN_getstatistic(ph, er.EN_RELATIVEERROR, ctypes.byref(v))
    rel = v.value
    nid = {s: i + 1 for i, s in enumerate(e.node_ids())}
    lid = {s: i + 1 for i, s in enumerate(e.link_ids())}

    def ng(name, prop):
        lib.EN_getnodevalue(ph, nid[name], prop, ctypes.byref(v))
        return v.value

    def lg(name, prop):
        lib.EN_getlinkvalue(ph, lid[name], prop, ctypes.byref(v))
        return v.value
    out = dict(iters=it, relerr=rel, rc=int(rc), uh=e._ucf_head, uf=e._ucf_flow, ng=ng, lg=lg, e=e)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    net = parse_inp(INP)
    se = GGASolver(net, mode="epanet", inp_path=INP)
    ids = list(net.link_id)
    prv = [k for k in range(se.L) if int(se.lt_np[k]) == 3]
    pumps = [k for k in range(se.L) if int(se.lt_np[k]) == 2]
    k180 = ids.index("VALVE-180")
    print(f"BWSN_Network_1: N={net.N} L={se.L} PRV={len(prv)} 泵={len(pumps)} | INP accuracy={net.meta['accuracy']} "
          f"trials={net.meta['trials']} checkfreq={net.meta['checkfreq']} maxcheck={net.meta['maxcheck']} "
          f"damp_limit={net.meta['damp_limit']}")
    # ---------------- EPANET t=0 ----------------
    ref = ep_run()
    junc, fixed = np.asarray(se.junc_nodes), np.asarray(se.fixed_nodes)
    nids = list(net.node_id)
    d = np.zeros(net.N)
    d[junc] = [ref["ng"](nids[j], er.EN_DEMAND) / ref["uf"] for j in junc]
    rh = np.zeros(net.N)
    rh[fixed] = [ref["ng"](nids[j], er.EN_HEAD) / ref["uh"] for j in fixed]
    H_ref = np.array([ref["ng"](nids[j], er.EN_HEAD) / ref["uh"] for j in range(net.N)])
    for tag, kw in (("INP 选项", {}), ("ACCURACY 1e-8 / TRIALS 1000", dict(acc=1e-8, trials=1000))):
        r = ref if not kw else ep_run(**kw)
        st = {ids[k]: int(r["lg"](ids[k], EN_STATUS)) for k in prv + pumps}
        print(f"[EPANET {tag}] iters={r['iters']} relerr={r['relerr']:.2e} 返回码={r['rc']} | "
              f"EN_STATUS(0关1开) PRV/泵: " + " ".join(f"{k[-3:]}={v}" for k, v in st.items()))
    print(f"  EPANET 在 t=0 对 VALVE-180 的状态 = {int(ref['lg']('VALVE-180', EN_STATUS))}（0=关闭：说明 t=0 控制已执行）")
    # ---------------- HydroGrad run_gga ----------------
    st_inp = np.asarray(se.init_status_int).copy()
    st_close = st_inp.copy()
    st_close[k180] = se.ST_CLOSED
    print(f"  HydroGrad 初始状态 VALVE-180 = {ST[int(st_inp[k180])]}")
    for sname, st0 in (("INP 初始状态", st_inp), ("执行 t=0 控制，VALVE-180 关闭", st_close)):
        for hacc, mi in ((float(net.meta["accuracy"]), int(net.meta["trials"])), (1e-8, 1000), (1e-12, 200)):
            res = se.run_gga(d, rh, status0=st0.copy(), do_status=True, hacc=hacc, max_iter=mi, record_schedule=True)
            sched = res.get("schedule") or []
            cnt = collections.Counter(k for _, _, ch in sched for k in ch)
            it_end = int(res["iters"])
            late = collections.Counter(k for it, _, ch in sched if it > it_end - 10 for k in ch)
            dH = float(np.abs(np.asarray(res["head"])[junc] - H_ref[junc]).max())
            fin = np.asarray(res["status"])
            print(f"[HG {sname} | 精度 {hacc:g} / 上限 {mi}] 收敛={bool(res['converged'])} 迭代={it_end} "
                  f"relerr={float(res['relerr']):.2e} | 与 EPANET(INP 选项) 的 max|ΔH|={dH:.2e} ft")
            print("      终态 PRV/泵: " + " ".join(f"{ids[k][-3:]}={ST[int(fin[k])]}" for k in prv + pumps))
            print("      状态切换次数最多的链路: " + ", ".join(f"{ids[k]}×{c}" for k, c in cnt.most_common(5))
                  + (f" | 最后 10 次迭代仍在切换: " + ", ".join(f"{ids[k]}×{c}" for k, c in late.most_common(5)) if late else " | 最后 10 次迭代无切换"))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""net6_full_eps.py - pub_net6 全 609 帧自主 EPS 对拍（模块四任务 1）。

与 scripts/bench_partial_eps.py 完全同一循环与同一判定门槛（H<1e-6 ft、
开启链路 Q<1e-6 cfs、关闭管 1e-5 cfs、帧时刻 int 相等、状态一致），
只是（a）不设帧数上限，（b）把逐帧偏差落盘成 JSON 便于统计分布，
（c）每 25 帧写一次 checkpoint，中断可续报已完成部分。

用法: python scripts/exempt_diag/net6_full_eps.py [stem] [n_frames]
"""

import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.parse import Net       # noqa: E402
from dgga.eps import EpsDriver   # noqa: E402
from align import resolve_inp    # noqa: E402

REF_DIR = os.path.join(ROOT, "data", "reference")
OUT_DIR = os.path.join(ROOT, "data", "exempt")
TOL_H, TOL_Q, TOL_QC = 1e-6, 1e-6, 1e-5


def main(stem="pub_net6", n_frames=10 ** 9):
    net = Net.load(REF_DIR, stem)
    ref = np.load(os.path.join(REF_DIR, f"{stem}_ref.npz"))
    inp = resolve_inp(stem)
    drv = EpsDriver(net, inp_path=inp if os.path.isfile(inp) else None)
    s = drv.solver
    drv._inithyd()
    T_ep = len(ref["t_sec"])
    F = min(n_frames, T_ep)
    print(f"=== {stem}: 自主 EPS {F}/{T_ep} 帧, N={net.N}, L={net.L} ===", flush=True)
    print("  帧    t(s)    max|dH|ft   max|dQ|cfs   闭管|dQ|   状态 迭代(我/EP)  秒", flush=True)

    rec = {"stem": stem, "N": int(net.N), "L": int(net.L), "T_ep": int(T_ep),
           "tol": {"H": TOL_H, "Q": TOL_Q, "Qc": TOL_QC},
           "frames": [], "python": sys.executable, "platform": sys.platform}
    out_json = os.path.join(OUT_DIR, f"{stem}_full_eps.json")
    t_wall0 = time.perf_counter()
    ok = True
    for f in range(F):
        tf0 = time.perf_counter()
        t = drv.Htime
        t_ok = int(t) == int(ref["t_sec"][f])
        drv._demands()
        drv._controls()
        r = s.run_gga(drv.d, drv.H, q0=drv.q, e0=drv.e,
                      status0=drv.S, setting0=drv.K, do_status=True)
        drv.q = r["flow"]
        drv.e = r["emitter"]
        drv.S = r["status"]
        drv.K = r["setting"]
        drv.H = r["head"]
        drv.fixed_dem = r["fixed_demand"]
        dem_out = np.where(s.is_fixed_node, drv.fixed_dem, drv.d + drv.e)
        drv.node_dem = dem_out

        open_api = drv.S > s.ST_CLOSED
        flow_api = np.where(open_api, drv.q, 0.0)
        opened = ref["status"][f] > 0
        dH = float(np.abs(drv.H - ref["head_ft"][f]).max())
        dQl = np.abs(flow_api - ref["flow_cfs"][f])
        dQ = float(dQl[opened].max()) if opened.any() else 0.0
        dQc = float(dQl[~opened].max()) if (~opened).any() else 0.0
        st_ok = bool(np.array_equal(open_api.astype(np.int8), ref["status"][f]))
        it_my, it_ep = int(r["iters"]), int(ref["iterations"][f])
        line_ok = bool(t_ok and dH < TOL_H and dQ < TOL_Q and dQc < TOL_QC and st_ok)
        ok = ok and line_ok
        dt = time.perf_counter() - tf0
        rec["frames"].append({"f": f, "t": int(t), "t_ok": bool(t_ok),
                              "dH": dH, "dQ": dQ, "dQc": dQc, "status_ok": st_ok,
                              "it_my": it_my, "it_ep": it_ep, "ok": line_ok,
                              "sec": dt})
        print(f"{f:>4d} {int(t):>7d} {dH:>12.3e} {dQ:>12.3e} {dQc:>11.3e} "
              f"{'Y' if st_ok else 'N'} {it_my:>4d}/{it_ep:<4d} {dt:6.2f}"
              f"{'' if line_ok else '  <-- 超限'}", flush=True)
        drv._nexthyd(float(r["relerr"]))
        if (f + 1) % 25 == 0 or f + 1 == F:
            rec["done"] = f + 1
            rec["wall_sec"] = time.perf_counter() - t_wall0
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=1)

    dHs = np.array([x["dH"] for x in rec["frames"]])
    dQs = np.array([x["dQ"] for x in rec["frames"]])
    it_eq = all(x["it_my"] == x["it_ep"] for x in rec["frames"])
    st_all = all(x["status_ok"] for x in rec["frames"])
    t_all = all(x["t_ok"] for x in rec["frames"])
    n_bad = int((dHs >= TOL_H).sum())
    print("-" * 84, flush=True)
    print(f"帧数 {F}/{T_ep}；耗时 {rec['wall_sec']:.0f}s（{rec['wall_sec']/F:.2f} s/帧）")
    print(f"max|dH| = {dHs.max():.3e} ft @帧{int(dHs.argmax())}；"
          f"中位 {np.median(dHs):.3e}；p95 {np.percentile(dHs, 95):.3e}")
    print(f"max|dQ| = {dQs.max():.3e} cfs；中位 {np.median(dQs):.3e}")
    print(f"dH<1e-6 的帧: {F - n_bad}/{F}；>=1e-6: {n_bad}")
    print(f"帧时刻 int 全等: {t_all}；状态逐帧一致: {st_all}；迭代数逐帧相等: {it_eq}")
    print(f"总判定: {'PASS' if ok else 'FAIL(仅门槛，见上分布)'}")
    rec["summary"] = {"F": int(F), "T_ep": int(T_ep),
                      "max_dH": float(dHs.max()), "argmax_dH": int(dHs.argmax()),
                      "median_dH": float(np.median(dHs)),
                      "p95_dH": float(np.percentile(dHs, 95)),
                      "max_dQ": float(dQs.max()), "median_dQ": float(np.median(dQs)),
                      "n_frames_dH_ge_tol": n_bad,
                      "t_all_equal": bool(t_all), "status_all_equal": bool(st_all),
                      "iters_all_equal": bool(it_eq), "verdict_pass": bool(ok),
                      "wall_sec": rec["wall_sec"]}
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    print("写出:", out_json)
    return 0


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "pub_net6"
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    sys.exit(main(st, nf))

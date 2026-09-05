# -*- coding: utf-8 -*-
"""任务一：把 networks/EXAMPLE 下 134 个公开 .inp 的准入情况一次查清。

对每个 .inp 报：内容哈希（去重）/ Nj / N / L / 各 link_type 计数 / 水池 / 水库 /
水损公式 / mode="epanet" 能否构造 / mode="dense" 能否构造（含要不要关水池守卫）/
**能否真的解出来**（单帧 B=1，res_head 里 tank 位用 tank_h0 填满）。

用法：
  set P5_DGGA_ROOT=<只读副本目录>
  python -X utf8 scripts/p5_bignet/survey_inp.py [--out <json>] [--only <substr>]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from p5lib import LINK_NAMES, base_case, dgga_provenance  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dgga.parse import parse_inp  # noqa: E402
from dgga.solver import GGASolver  # noqa: E402

# 单帧稠密 A[Nj,Nj] f64 的预算：solve 至少要 A + Cholesky 因子两份
SOLVE_NJ_CAP = 12000


def short(e):
    s = str(e).replace("\n", " ")
    return s if len(s) <= 240 else s[:237] + "..."


def one(path):
    rec = {"path": os.path.relpath(path, ROOT).replace("\\", "/")}
    with open(path, "rb") as f:
        blob = f.read()
    rec["sha256"] = hashlib.sha256(blob).hexdigest()[:16]
    rec["bytes"] = len(blob)

    # ---- 1. parse ----
    try:
        t0 = time.perf_counter()
        net = parse_inp(path)
        rec["parse_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        rec["parse"] = "ok"
    except Exception as e:
        rec["parse"] = "FAIL"
        rec["parse_err"] = f"{type(e).__name__}: {short(e)}"
        return rec

    nt = np.asarray(net.node_type)
    lt = np.asarray(net.link_type)
    rec["N"] = int(net.N)
    rec["Nj"] = int((nt == 0).sum())
    rec["n_res"] = int((nt == 1).sum())
    rec["n_tank"] = int((nt == 2).sum())
    rec["L"] = int(net.L)
    rec["links"] = {LINK_NAMES.get(int(k), str(k)): int(v)
                    for k, v in zip(*np.unique(lt, return_counts=True))}
    rec["headloss"] = str(net.meta.get("headloss", "?"))
    rec["flow_units"] = str(net.meta.get("flow_units", "?"))
    rec["demand_model"] = str(net.meta.get("demand_model", "DDA"))

    # ---- 2. mode="epanet" 构造 ----
    for mode, key in (("epanet", "epanet"), ("dense", "dense")):
        try:
            GGASolver(net, mode=mode, inp_path=path)
            rec[f"build_{key}"] = "ok"
        except Exception as e:
            rec[f"build_{key}"] = "FAIL"
            rec[f"build_{key}_err"] = f"{type(e).__name__}: {short(e)}"

    # ---- 2b. dense 关水池守卫后能否构造 ----
    rec["dense_needs_noguard"] = False
    if rec.get("build_dense") == "FAIL":
        try:
            GGASolver(net, mode="dense", inp_path=path,
                      dense_tank_bound_check=False)
            rec["build_dense_noguard"] = "ok"
            rec["dense_needs_noguard"] = True
        except Exception as e:
            rec["build_dense_noguard"] = "FAIL"
            rec["build_dense_noguard_err"] = f"{type(e).__name__}: {short(e)}"

    # ---- 3. 真的解一发（B=1，tank 位水头补齐）----
    dense_ok = rec.get("build_dense") == "ok" or rec.get("build_dense_noguard") == "ok"
    if not dense_ok:
        rec["solve_dense"] = "n/a"
    elif rec["Nj"] > SOLVE_NJ_CAP:
        rec["solve_dense"] = "skip(Nj>cap)"
    else:
        try:
            s = GGASolver(net, mode="dense", inp_path=path,
                          dense_tank_bound_check=not rec["dense_needs_noguard"])
            d, rh = base_case(net)
            t0 = time.perf_counter()
            out = s.solve(d, rh)
            rec["solve_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
            def _np(x):
                return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
            H = _np(out["head_ft"] if isinstance(out, dict) else out[0])
            conv = bool(np.all(_np(out["converged"])))
            rec["iters_dense"] = int(np.max(_np(out["iters"])))
            rec["relerr_dense"] = float(np.max(_np(out["relerr"])))
            rec["solve_dense"] = ("ok" if (np.isfinite(H).all() and conv)
                                  else ("NOTCONV" if np.isfinite(H).all()
                                        else "NONFINITE"))
        except Exception as e:
            rec["solve_dense"] = "FAIL"
            rec["solve_dense_err"] = f"{type(e).__name__}: {short(e)}"

    # ---- 4. epanet 模式也解一发（小网才做，逐样本 numpy 很慢）----
    if rec.get("build_epanet") == "ok" and rec["Nj"] <= 2000:
        try:
            s = GGASolver(net, mode="epanet", inp_path=path)
            d, rh = base_case(net)
            t0 = time.perf_counter()
            out = s.solve(d, rh)
            rec["solve_ms_epanet"] = round((time.perf_counter() - t0) * 1e3, 1)
            H = out["head_ft"]
            H = H.detach().cpu().numpy() if torch.is_tensor(H) else np.asarray(H)
            cv = out["converged"]
            cv = cv.detach().cpu().numpy() if torch.is_tensor(cv) else np.asarray(cv)
            rec["iters_epanet"] = int(np.max(
                out["iters"].detach().cpu().numpy() if torch.is_tensor(out["iters"])
                else np.asarray(out["iters"])))
            rec["solve_epanet"] = ("ok" if (np.isfinite(H).all() and bool(np.all(cv)))
                                   else ("NOTCONV" if np.isfinite(H).all()
                                         else "NONFINITE"))
        except Exception as e:
            rec["solve_epanet"] = "FAIL"
            rec["solve_epanet_err"] = f"{type(e).__name__}: {short(e)}"
    elif rec.get("build_epanet") == "ok":
        rec["solve_epanet"] = "skip(big)"
    else:
        rec["solve_epanet"] = "n/a"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "p5_bignet",
                                                  "survey_inp.json"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--root", default=os.path.join(ROOT, "networks", "EXAMPLE"))
    args = ap.parse_args()

    print("dgga imported from:", dgga_provenance(), flush=True)

    paths = []
    for dp, _, fns in os.walk(args.root):
        for fn in fns:
            if fn.lower().endswith(".inp"):
                paths.append(os.path.join(dp, fn))
    paths.sort()
    if args.only:
        _n = args.only.lower().replace("\\", "/")
        paths = [p for p in paths if _n in p.lower().replace("\\", "/")]
    print(f"found {len(paths)} .inp", flush=True)

    recs = []
    for i, p in enumerate(paths):
        try:
            r = one(p)
        except BaseException as e:            # MemoryError 等也要活着继续
            r = {"path": os.path.relpath(p, ROOT).replace("\\", "/"),
                 "parse": "CRASH", "parse_err": f"{type(e).__name__}: {short(e)}"}
            traceback.print_exc()
        recs.append(r)
        print(f"[{i+1}/{len(paths)}] Nj={r.get('Nj','?'):>6} "
              f"parse={r.get('parse')} dense={r.get('build_dense')} "
              f"solve={r.get('solve_dense')} {r['path']}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

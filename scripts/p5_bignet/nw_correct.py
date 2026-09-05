# -*- coding: utf-8 -*-
"""任务二·1：NW_Model 三方对拍 - **正确性优先，这步不过后面的性能数字一文不值**。

  (a) mode="dense"（论文主线通路）  vs  mode="epanet"（逐位复刻基准）
  (b) mode="dense"                  vs  EPANET DLL（dgga/epanet_ref.Epanet）
  (c) mode="epanet"                 vs  EPANET DLL（校准基准本身）
  (d) assemble="csr" + linear_solver="dense"（CSR 装配等价）vs 缺省 dense 装配

报 max|ΔH|（ft，只比 junction）与各自迭代数。对拍在 t=0 名义工况上做：
res_head_ft 的 tank 位用 tank_h0 填满（NW_Model 零水池，但代码走通用路径）。

用法：python -X utf8 scripts/p5_bignet/nw_correct.py [--inp <path>]
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from p5lib import base_case, dgga_provenance  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dgga.parse import parse_inp  # noqa: E402
from dgga.solver import GGASolver  # noqa: E402


def np_(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def run(inp, no_guard=False, csr_check=True, want_dll=True, acc=None, trials=None):
    print("=" * 74)
    print("INP :", os.path.relpath(inp, ROOT))
    print("dgga:", dgga_provenance())
    net = parse_inp(inp)
    nt = np.asarray(net.node_type)
    Nj = int((nt == 0).sum())
    print(f"N={net.N} Nj={Nj} L={net.L} headloss={net.meta.get('headloss')} "
          f"units={net.meta.get('flow_units')} tanks={(nt==2).sum()} res={(nt==1).sum()}")
    print(f"INP 自带 [OPTIONS]: ACCURACY={net.meta.get('accuracy')} "
          f"TRIALS={net.meta.get('max_trials', net.meta.get('trials'))}"
          f"   -> 本次对拍用 accuracy={acc} trials={trials}（None=用 INP 值）")
    kw = {}
    if acc is not None:
        kw["accuracy"] = acc
    if trials is not None:
        kw["max_iter"] = trials

    guard = not no_guard
    res = {}

    # ---- dense ----
    s_d = GGASolver(net, mode="dense", inp_path=inp, dense_tank_bound_check=guard)
    d, rh = base_case(net)                     # 必须在 solver 构造之后取
    t0 = time.perf_counter()
    o_d = s_d.solve(d, rh, **kw)
    res["dense"] = (np_(o_d["head_ft"]), int(np_(o_d["iters"])),
                    float(np_(o_d["relerr"])), (time.perf_counter() - t0) * 1e3,
                    np_(o_d["flow_cfs"]))

    # ---- CSR 装配等价（同样的稠密 Cholesky，只换装配路径）----
    if csr_check:
        t0 = time.perf_counter()
        o_c = s_d.solve(d, rh, assemble="csr", **kw)
        res["csr+dense"] = (np_(o_c["head_ft"]), int(np_(o_c["iters"])),
                            float(np_(o_c["relerr"])), (time.perf_counter() - t0) * 1e3,
                            np_(o_c["flow_cfs"]))

    # ---- epanet 逐位复刻 ----
    s_e = GGASolver(net, mode="epanet", inp_path=inp)
    t0 = time.perf_counter()
    o_e = s_e.solve(d, rh, **kw)
    res["epanet"] = (np_(o_e["head_ft"]), int(np_(o_e["iters"])),
                     float(np_(o_e["relerr"])), (time.perf_counter() - t0) * 1e3,
                     np_(o_e["flow_cfs"]))

    # ---- EPANET DLL ----
    if want_dll:
        from dgga.epanet_ref import Epanet, EN_ACCURACY, EN_TRIALS
        with Epanet(inp) as ep:
            if acc is not None:
                ep.set_option(EN_ACCURACY, acc)
            if trials is not None:
                ep.set_option(EN_TRIALS, trials)
            ids = ep.node_ids()
            t0 = time.perf_counter()
            o_r = ep.solve_single()
            dll_ms = (time.perf_counter() - t0) * 1e3
        # 按 node_id 对齐（parse 顺序 = INP 顺序，DLL 也是，但显式对齐更稳）
        pos = {nid: i for i, nid in enumerate(ids)}
        idx = np.array([pos[nid] for nid in net.node_id], dtype=np.int64)
        res["dll"] = (o_r["head_ft"][idx], int(o_r["iterations"]),
                      float(o_r["relerr"]), dll_ms,
                      None)

    jm = nt == 0
    print("\n迭代数 / relerr / 墙钟")
    for k, v in res.items():
        print(f"  {k:<11} iters={v[1]:>3}  relerr={v[2]:.3e}  {v[3]:9.1f} ms")

    def cmp(a, b):
        A, B = res[a][0], res[b][0]
        dj = np.abs(A[jm] - B[jm])
        dall = np.abs(A - B)
        return dj.max(), dall.max(), int(np.argmax(dj))

    print("\nmax|ΔH| (ft)                      junction位        全节点")
    pairs = [("dense", "epanet")]
    if csr_check:
        pairs.append(("csr+dense", "dense"))
    if want_dll:
        pairs += [("dense", "dll"), ("epanet", "dll")]
    out = {}
    for a, b in pairs:
        mj, ma, ij = cmp(a, b)
        out[f"{a}~{b}"] = mj
        print(f"  {a:<11} ~ {b:<11}      {mj:.6e}    {ma:.6e}   (argmax junction #{ij})")

    # 流量也看一眼（DLL 的 flow 顺序同 link 文件序）
    fa, fb = res["dense"][4], res["epanet"][4]
    print(f"\nmax|ΔQ| dense~epanet = {np.abs(fa-fb).max():.6e} cfs "
          f"(|Q|max={np.abs(fa).max():.4f})")
    Hd = res["dense"][0]
    print(f"H 范围 [{Hd[jm].min():.3f}, {Hd[jm].max():.3f}] ft")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=os.path.join(
        ROOT, "networks", "EXAMPLE", "epanet-example-networks",
        "epanet-tests", "large", "NW_Model.inp"))
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--no-dll", action="store_true")
    ap.add_argument("--accuracy", type=float, default=None)
    ap.add_argument("--trials", type=int, default=None)
    a = ap.parse_args()
    run(a.inp, no_guard=a.no_guard, want_dll=not a.no_dll,
        acc=a.accuracy, trials=a.trials)


if __name__ == "__main__":
    main()

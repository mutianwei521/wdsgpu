# -*- coding: utf-8 -*-
"""bench_scaling.py - 论文图 fig_scaling 的"网络规模 vs 单帧耗时"实测。

对每个网络测量 **单帧冷启动稳态求解** 的耗时（不含解析/建图/参考解读取）：
  - 我方 GGASolver(mode='epanet')：位级复刻路径（numpy 逐样本 + smatrix 稀疏 Cholesky）
  - 我方 GGASolver(mode='dense')：torch 稠密 Cholesky 路径（可微/可批量/可 GPU）；
    仅在网络元件受 dense 模式支持时可测（PIPE/TCV、无 tank），否则记 null
  - EPANET 2.2 DLL：EN_initH(EN_INITFLOW) 冷启动 + EN_runH(t=0)，同一 INP

边界条件取自 data/reference/<stem>_ref.npz 的第 0 帧：需水/水库水头用
parse.demand_cfs_at / reservoir_head_ft_at 现算（与 align.py 同源），
水池水头取 ref 第 0 帧（回放边界，不做 tanklevels 积分）。
状态机开启（status_machine=True），与 align/eps 同规。

耗时口径：min over N_REP 次（warmup 后），单位 ms。min 而非均值以压低
Windows 调度抖动；同一口径用于三条曲线。

输出：data/bench_scaling.json
运行：python -X utf8 scripts/bench_scaling.py
"""

import ctypes
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message=".*not writable.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.parse import Net           # noqa: E402
from dgga.solver import GGASolver    # noqa: E402
from dgga.epanet_ref import Epanet, EN_HEAD   # noqa: E402
from align import resolve_inp        # noqa: E402

REF_DIR = os.path.join(ROOT, "data", "reference")
OUT = os.path.join(ROOT, "data", "bench_scaling.json")
EN_INITFLOW = 10                     # epanet2_enums.h:371

# 覆盖 N=32 ~ 12527 三个数量级（公开网为主 + 两个真实网）
STEMS = [
    "pub_hanoi", "rand_main_0003", "pub_fossolo_poly1", "pub_pescara",
    "pub_net3", "pub_modena", "EXA4", "pub_balerma", "city_d",
    "pub_l_town", "pub_richmond_standard", "city_h", "pub_ky10", "pub_ky4",
    "pub_net6", "pub_bwsn_network_2",
]


def _timeit(fn, reps, warmup=1):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _reps(n):
    return 20 if n < 200 else (8 if n < 1500 else 3)


def bench_ours(stem, mode):
    """返回 (ms, iters) 或 (None, 原因)。"""
    net = Net.load(REF_DIR, stem)
    ref = np.load(os.path.join(REF_DIR, f"{stem}_ref.npz"))
    inp = resolve_inp(stem)
    try:
        s = GGASolver(net, mode=mode,
                      inp_path=inp if os.path.isfile(inp) else None)
    except NotImplementedError as e:
        return None, f"{mode} 模式不支持: {e}"
    t0 = int(ref["t_sec"][0])
    d = net.demand_cfs_at(t0)
    rh = net.reservoir_head_ft_at(t0)
    tanks = np.where(np.asarray(net.node_type) == 2)[0]
    rh = np.array(rh, dtype=np.float64)
    if tanks.size:
        rh[tanks] = ref["head_ft"][0][tanks]
    kw = dict(status_machine=True) if mode == "epanet" else {}
    try:
        r = s.solve(d, rh, **kw)
    except Exception as e:                       # noqa: BLE001
        return None, f"{mode} 求解失败: {type(e).__name__}: {e}"
    it = int(np.max(np.asarray(r["iters"])))
    sec = _timeit(lambda: s.solve(d, rh, **kw), _reps(net.N))
    return sec * 1e3, it


def bench_epanet(stem):
    inp = resolve_inp(stem)
    if not os.path.isfile(inp):
        return None, "无 INP"
    en = Epanet(inp)
    lib, ph = en.lib, en._ph
    t = ctypes.c_long()
    val = ctypes.c_double()
    nn = en.counts()["nodes"]

    def one():
        en._check(lib.EN_initH(ph, EN_INITFLOW), "EN_initH")
        rc = lib.EN_runH(ph, ctypes.byref(t))
        if rc > 100:
            raise RuntimeError(f"EN_runH rc={rc}")

    try:
        en._check(lib.EN_openH(ph), "EN_openH")
        sec = _timeit(one, _reps(nn))
        # 读全部节点水头的额外开销（与我方"求解即得全场"口径对齐时需计入）
        one()
        t0 = time.perf_counter()
        for i in range(1, nn + 1):
            lib.EN_getnodevalue(ph, i, EN_HEAD, ctypes.byref(val))
        sec_read = time.perf_counter() - t0
    finally:
        lib.EN_closeH(ph)
        en.close()
    return sec * 1e3, sec_read * 1e3


def main():
    out = {}
    for stem in STEMS:
        net = Net.load(REF_DIR, stem)
        row = dict(N=int(net.N), L=int(net.L),
                   units=str(net.meta.get("units")),
                   headloss=str(net.meta.get("headloss")))
        ms_e, it_e = bench_ours(stem, "epanet")
        row["ours_epanet_ms"] = ms_e
        row["ours_epanet_iters" if ms_e is not None else "ours_epanet_note"] = it_e
        ms_d, it_d = bench_ours(stem, "dense")
        row["ours_dense_ms"] = ms_d
        row["ours_dense_iters" if ms_d is not None else "ours_dense_note"] = it_d
        try:
            ms_ep, ms_rd = bench_epanet(stem)
        except Exception as e:                   # noqa: BLE001
            ms_ep, ms_rd = None, f"{type(e).__name__}: {e}"
        row["epanet_dll_ms"] = ms_ep
        row["epanet_read_heads_ms" if ms_ep is not None else "epanet_note"] = ms_rd
        out[stem] = row
        print(f"{stem:<24} N={row['N']:>6} L={row['L']:>6} "
              f"ours_epanet={_f(ms_e)} ms  ours_dense={_f(ms_d)} ms  "
              f"EPANET_DLL={_f(ms_ep)} ms")
        sys.stdout.flush()
    meta = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                python=sys.executable, torch=torch.__version__,
                torch_threads=int(torch.get_num_threads()),
                timing="min over repeats, cold-start single steady-state frame",
                note="tank heads from data/reference/<stem>_ref.npz frame 0")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, nets=out), f, ensure_ascii=False, indent=1)
    print(f"\n写出 {OUT}")
    return 0


def _f(x):
    return "     -" if x is None else f"{x:7.3f}"


if __name__ == "__main__":
    sys.exit(main())

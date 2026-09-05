# -*- coding: utf-8 -*-
"""bench_batch.py - 任务 B：批量一致性与性能基准（city_d）。

1. 一致性：B=64 个随机需水扰动场景（基准需水 × U(0.8,1.2)，逐节点独立采样），
   dense 模式（torch 纯广播批量路径）批量 solve 一次 vs 逐场景循环 solve，
   逐场景 max|ΔH| 必须 < 1e-12 ft（纯广播实现应比特级一致，否则=批维串扰 bug）。
2. 性能：CPU float64 B=1/64/256 每场景耗时；若有 CUDA 再测 GPU float64 与
   float32（float32 只测速度并报告 max|ΔH| 相对 float64 的量级，不设门槛）。
   对照组：dgga.epanet_ref 逐场景跑 EPANET（B=64） - 每场景
   EN_setbasedemand（逐类别缩放）→ EN_initH(EN_INITFLOW 冷启动) → EN_runH。

EPANET API 常数/原型均核对头文件原文：
- EN_INITFLOW = 10（ref/epanet2.2_toolkit/epanet2_enums.h:371，冷启动重置流量）
- EN_getnumdemands(ph, nodeIndex, int*)（epanet2_2.h:995）
- EN_getbasedemand(ph, nodeIndex, demandIndex, double*)（epanet2_2.h:1005）
- EN_setbasedemand(ph, nodeIndex, demandIndex, double)（epanet2_2.h:1016）
"""

import ctypes
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import Net           # noqa: E402
from dgga.solver import GGASolver    # noqa: E402
from dgga.epanet_ref import Epanet   # noqa: E402

STEM = "city_d"
INP = os.path.join(ROOT, "networks", "realInpData", f"{STEM}.inp")
REF_DIR = os.path.join(ROOT, "data", "reference")
SEED = 20260808
B_CONSIST = 64
B_LIST = (1, 64, 256)
TOL_CONSIST = 1e-12   # ft，批量 vs 逐场景

EN_INITFLOW = 10      # epanet2_enums.h:371


# ---------------------------------------------------------------- 计时工具
def _timeit(fn, repeats, warmup=1, sync=None):
    """返回 repeats 次中的最小耗时（秒）。sync=可调用（如 cuda.synchronize）。"""
    for _ in range(warmup):
        fn()
        if sync:
            sync()
    best = float("inf")
    for _ in range(repeats):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------- 场景构造
def build_scenarios(net, solver_for_props):
    """基准需水（t=0 名义需水）× U(0.8,1.2) 逐节点因子 → D[256,N]。

    注意：必须在构造带 inp_path 的 GGASolver 之后取 demand_cfs_at（
    _apply_exact_props 会就地把 net.dem_base_cfs 修正为位级基值）。"""
    assert solver_for_props is not None
    d0 = net.demand_cfs_at(0)                     # [N]，tank/reservoir 位=0
    rh0 = net.reservoir_head_ft_at(0)             # [N]，非水库位 nan
    rng = np.random.default_rng(SEED)
    factors = rng.uniform(0.8, 1.2, size=(max(B_LIST), net.N))   # [256,N]
    D = d0[None, :] * factors                     # 水库位 0×f=0，安全
    return d0, rh0, D, factors


# ---------------------------------------------------------------- 一致性
def check_consistency(solver, D, rh0):
    """dense 批量 solve 一次 vs 逐场景循环，逐场景 max|ΔH|。"""
    Db = D[:B_CONSIST]
    rb = solver.solve(Db, rh0)
    Hb = rb["head_ft"].cpu().numpy()              # [B,N]
    Qb = rb["flow_cfs"].cpu().numpy()
    ib = rb["iters"].cpu().numpy()
    assert bool(rb["converged"].all()), "批量 solve 有未收敛场景"

    dH = np.empty(B_CONSIST)
    dQ = np.empty(B_CONSIST)
    it_eq = True
    for b in range(B_CONSIST):
        r = solver.solve(Db[b], rh0)
        dH[b] = np.abs(r["head_ft"].cpu().numpy() - Hb[b]).max()
        dQ[b] = np.abs(r["flow_cfs"].cpu().numpy() - Qb[b]).max()
        it_eq = it_eq and int(r["iters"]) == int(ib[b])
    ok = bool(dH.max() < TOL_CONSIST)
    print(f"[一致性] dense 批量(B={B_CONSIST}) vs 逐场景循环：")
    print(f"  max|ΔH| = {dH.max():.3e} ft (门槛 {TOL_CONSIST:.0e})  "
          f"max|ΔQ| = {dQ.max():.3e} cfs  迭代数全等 = {it_eq}  "
          f"位级一致 = {bool(dH.max() == 0.0 and dQ.max() == 0.0)}")
    print(f"  判定: {'PASS' if ok else 'FAIL - 批维串扰 bug，需修 solver.py'}")
    print(f"  批量迭代数分布: min={ib.min()} max={ib.max()}")
    return ok, Hb


# ---------------------------------------------------------------- 我们的性能
def bench_ours(net, D, rh0, device, dtype, label):
    """dense 模式 B=1/64/256 每场景耗时（ms）。返回 {B: ms}，及 B=64 的 head。"""
    solver = GGASolver(net, device=device, dtype=dtype, mode="dense", inp_path=INP)
    sync = torch.cuda.synchronize if device == "cuda" else None
    out = {}
    H64 = None
    for B in B_LIST:
        Din = D[0] if B == 1 else D[:B]
        fn = lambda: solver.solve(Din, rh0)
        reps = {1: 20, 64: 5, 256: 3}[B]
        sec = _timeit(fn, repeats=reps, warmup=2, sync=sync)
        out[B] = sec / B * 1e3
        if B == 64:
            r = solver.solve(D[:64], rh0)
            H64 = r["head_ft"].to(torch.float64).cpu().numpy()
            it = r["iters"].cpu().numpy()
            nc = int((~r["converged"]).sum())
            print(f"  {label} B=64: 迭代数 min={it.min()} max={it.max()}"
                  f"{f'  [警告] {nc} 个场景未收敛' if nc else ''}")
    print(f"[性能] {label}: " + "  ".join(
        f"B={B}: {out[B]:.3f} ms/场景" for B in B_LIST))
    return out, H64


# ---------------------------------------------------------------- EPANET 对照
def bench_epanet(net, factors, B):
    """逐场景跑 EPANET：EN_setbasedemand 缩放全部需水类别 → EN_initH(EN_INITFLOW)
    → EN_runH(t=0) → 读全部节点 HEAD。返回 (分段耗时 dict, head_ft[B,N])。"""
    en = Epanet(INP)
    lib, ph = en.lib, en._ph
    # 追加声明本脚本所需原型（核对 epanet2_2.h:995/1005/1016）
    lib.EN_getnumdemands.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int)]
    lib.EN_getnumdemands.restype = ctypes.c_int
    lib.EN_getbasedemand.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_double)]
    lib.EN_getbasedemand.restype = ctypes.c_int
    lib.EN_setbasedemand.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_double]
    lib.EN_setbasedemand.restype = ctypes.c_int

    nn = en.counts()["nodes"]
    assert nn == net.N
    # 收集基准需水（用户单位）：仅 junction（node_type==0）
    ntypes = en.node_types()
    ci, cd = ctypes.c_int(), ctypes.c_double()
    base = []                                     # (node_idx1based, demand_idx, user_base)
    for i in range(1, nn + 1):
        if ntypes[i - 1] != 0:
            continue
        en._check(lib.EN_getnumdemands(ph, i, ctypes.byref(ci)),
                  f"EN_getnumdemands({i})")
        for j in range(1, ci.value + 1):
            en._check(lib.EN_getbasedemand(ph, i, j, ctypes.byref(cd)),
                      f"EN_getbasedemand({i},{j})")
            base.append((i, j, cd.value))

    from dgga.epanet_ref import EN_HEAD           # 枚举已核对（epanet_ref.py:37）
    t = ctypes.c_long()
    val = ctypes.c_double()
    heads = np.empty((B, nn), dtype=np.float64)
    t_set = t_solve = t_read = 0.0
    n_warn = 0
    en._check(lib.EN_openH(ph), "EN_openH")
    try:
        for b in range(B):
            f_b = factors[b]
            t0 = time.perf_counter()
            for (i, j, v) in base:                # 逐类别缩放 = 全节点需水 × f
                en._check(lib.EN_setbasedemand(ph, i, j, v * f_b[i - 1]),
                          f"EN_setbasedemand({i},{j})")
            t1 = time.perf_counter()
            en._check(lib.EN_initH(ph, EN_INITFLOW), "EN_initH")   # 冷启动
            rc = lib.EN_runH(ph, ctypes.byref(t))
            if rc > 100:
                raise RuntimeError(f"EN_runH 场景{b} 错误码 {rc}")
            if rc > 0:
                n_warn += 1
            t2 = time.perf_counter()
            for i in range(1, nn + 1):
                lib.EN_getnodevalue(ph, i, EN_HEAD, ctypes.byref(val))
                heads[b, i - 1] = val.value / en._ucf_head
            t3 = time.perf_counter()
            t_set += t1 - t0
            t_solve += t2 - t1
            t_read += t3 - t2
    finally:
        lib.EN_closeH(ph)
        en.close()
    ms = dict(set=t_set / B * 1e3, solve=t_solve / B * 1e3,
              read=t_read / B * 1e3,
              total=(t_set + t_solve + t_read) / B * 1e3)
    print(f"[性能] EPANET (B={B}, 冷启动): 设需水 {ms['set']:.3f} + "
          f"求解 {ms['solve']:.3f} + 读头 {ms['read']:.3f} = "
          f"{ms['total']:.3f} ms/场景  (运行警告 {n_warn} 场景)")
    return ms, heads


# ---------------------------------------------------------------- 主流程
def main():
    print(f"=== bench_batch: {STEM} ===")
    net = Net.load(REF_DIR, STEM)
    # dense CPU float64 求解器（inp_path → 位级属性重建，且就地修正需水基值）
    solver_cpu = GGASolver(net, device="cpu", dtype=torch.float64,
                           mode="dense", inp_path=INP)
    d0, rh0, D, factors = build_scenarios(net, solver_cpu)
    print(f"N={net.N} Nj={solver_cpu.Nj} L={net.L}  场景: 需水×U(0.8,1.2) "
          f"seed={SEED}  torch {torch.__version__}  threads={torch.get_num_threads()}")

    # ---- 1. 一致性 ----
    ok, H64_cpu = check_consistency(solver_cpu, D, rh0)
    if not ok:
        print("一致性 FAIL，终止（先修 solver.py 再重跑）")
        return 1

    # ---- 2. 性能：我们 CPU ----
    perf = {}
    perf["ours_cpu_f64"], _ = bench_ours(net, D, rh0, "cpu", torch.float64,
                                         "我们 dense CPU float64")

    # ---- GPU ----
    f32_note = "无 CUDA，未测"
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        perf["ours_gpu_f64"], H64_gpu = bench_ours(
            net, D, rh0, "cuda", torch.float64, "我们 dense GPU float64")
        dH_gpu = np.abs(H64_gpu - H64_cpu).max()
        print(f"  GPU f64 vs CPU f64 (B=64): max|ΔH| = {dH_gpu:.3e} ft")
        try:
            perf["ours_gpu_f32"], H64_f32 = bench_ours(
                net, D, rh0, "cuda", torch.float32, "我们 dense GPU float32")
            dH_f32 = np.abs(H64_f32 - H64_cpu).max()
            f32_note = f"max|ΔH| vs float64 = {dH_f32:.3e} ft"
            print(f"  float32 精度观察 (B=64): {f32_note}")
        except Exception as exc:            # κ~1e9 病态矩阵，float32 可能分解失败
            f32_note = f"求解失败: {type(exc).__name__}: {exc}"
            print(f"  float32 失败: {f32_note}")

    # ---- 我们 epanet 模式（逐样本精确路径，参考行）----
    solver_ep = GGASolver(net, mode="epanet", inp_path=INP)
    t0 = time.perf_counter()
    r_ep = solver_ep.solve(D[:B_CONSIST], rh0)
    ep_mode_ms = (time.perf_counter() - t0) / B_CONSIST * 1e3
    H64_ep = r_ep["head_ft"].cpu().numpy()
    print(f"[性能] 我们 epanet 模式 CPU (B=64): {ep_mode_ms:.3f} ms/场景")

    # ---- 3. EPANET 对照（B=64）----
    ms_en, H_en = bench_epanet(net, factors, B_CONSIST)

    # ---- 交叉观察（不设门槛）：验证 EPANET 确实解了同一批扰动场景 ----
    jm = np.asarray(net.node_type) == 0
    d_ep = np.abs(H64_ep - H_en)[:, jm].max()
    d_dn = np.abs(H64_cpu - H_en)[:, jm].max()
    print(f"[交叉] EPANET vs 我们epanet模式 (64场景 junction): max|ΔH|={d_ep:.3e} ft "
          f"（需水缩放结合律 1ulp 差 × κ 放大，观察值）")
    print(f"[交叉] EPANET vs 我们dense CPU f64: max|ΔH|={d_dn:.3e} ft")

    # ---- 汇总表 ----
    print("\n===== 汇总（每场景 ms，city_d，冷启动稳态）=====")
    hdr = f"{'实现':<26}" + "".join(f"{'B=' + str(B):>12}" for B in B_LIST)
    print(hdr)
    rows = [("我们 dense CPU f64", perf.get("ours_cpu_f64"))]
    if "ours_gpu_f64" in perf:
        rows.append(("我们 dense GPU f64", perf["ours_gpu_f64"]))
    if "ours_gpu_f32" in perf:
        rows.append(("我们 dense GPU f32", perf["ours_gpu_f32"]))
    for name, d in rows:
        print(f"{name:<26}" + "".join(f"{d[B]:>12.3f}" for B in B_LIST))
    print(f"{'我们 epanet 模式 CPU':<26}{'-':>12}{ep_mode_ms:>12.3f}{'-':>12}")
    print(f"{'EPANET DLL(仅求解)':<26}{'-':>12}{ms_en['solve']:>12.3f}{'-':>12}")
    print(f"{'EPANET DLL(含设需水+读头)':<26}{'-':>12}{ms_en['total']:>12.3f}{'-':>12}")
    print(f"float32 观察: {f32_note}")
    print(f"一致性判定: PASS (max|ΔH| 批量 vs 逐场景 < {TOL_CONSIST:.0e} ft)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

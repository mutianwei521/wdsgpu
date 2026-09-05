# -*- coding: utf-8 -*-
"""EPS 参考解收集：跑双精度 EPANET 得到全时段水力结果并落盘。

契约见 dgga/CONTRACT.md：
- build_reference(inp_path, out_dir, stem)：Epanet(inp).solve_eps() →
  np.savez_compressed(out_dir/<stem>_ref.npz, **arrays)；
  标量与 warnings 并入（若已存在则更新）<stem>_meta.json 的 "reference" 键。
- 落盘数值一律 EPANET 内部单位（ft/cfs），float64。

参考解收紧（p5 审计 §1.3 / prv 前置第④件）：
- energy_residual_ft(solver, head, flow, ...)：开管能量失衡 max|h_l − ΔH|（ft）。
- tight_reference(solver, d, rh, ...)：把单帧参考解收到能量失衡 ≤ tol
  （缺省 1e-8 ft），报所需追加迭代数 K。**验收/对拍脚本里凡当参考的解一律
  过这个 helper** - INP 自带 ACCURACY（EPANET 钳到 ≥1e-5）下"三方一致"只说明
  一致地停在同一个没收敛完的第 k 步，不说明解准了（NW_Model 实测 INP 精度下
  最差开管能量失衡 3.3e-2 ft，K≥20 才到 1e-8 量级）。
"""

import json
import os

import numpy as np

from dgga.epanet_ref import Epanet


def energy_residual_ft(solver, head, flow, status=None, setting=None):
    """开管能量失衡 max|h_l − (H[n1]−H[n2])|（ft），管道支（PIPE/CVPIPE）。

    h_l 用 solver._PY_np（EPANET hydcoeffs.c 的系数复刻）按 h_l = Y/P 还原
 - 对管道支 P=1/hgrad、Y=hloss/hgrad，Y/P 就是该状态/流量下的水头损失；
    关闭支没有能量方程，剔除（p5 审计 §1.3 的口径，0.73→3.3e-2 的订正正是
    剔关闭管）。泵/阀不进本残差（各有自己的特性方程）。"""
    s = solver
    S = np.asarray(s.init_status_int if status is None else status)
    q = np.asarray(flow, dtype=np.float64)
    H = np.asarray(head, dtype=np.float64)
    P, Y = s._PY_np(q, closed=S <= s.ST_CLOSED,
                    setting=s.init_setting if setting is None else setting,
                    status=S)
    mask = (s.lt_np <= 1) & (S > s.ST_CLOSED) & (P != 0.0)
    if not mask.any():
        return 0.0
    hl = Y[mask] / P[mask]
    dh = H[s.n1_np[mask]] - H[s.n2_np[mask]]
    return float(np.abs(hl - dh).max())


def tight_reference(solver, d, rh, ke=None, tol_energy=1e-8,
                    k_schedule=(10, 20, 40, 80, 160, 320), do_status=True,
                    hacc=None, max_iter=None):
    """单帧参考解收紧：能量失衡 ≤ tol_energy（缺省 1e-8 ft）。

    步骤：① run_gga（缺省带状态机）解到 INP 精度、拿到收敛状态/设定；
    ② 若能量失衡已 ≤ tol 直接返回；否则**冻结状态**（do_status=False +
    status0/setting0/q0/e0 热启动，hacc=0 强制跑满）追加 K 步 Newton，
    K 按 k_schedule 递增，直到 ≤ tol 或日程用尽（返回最优者并如实标
    met=False - 如 NW_Model 这类死支网存在 ~1e-8 的舍入地板）。

    返回 dict：head/flow/emitter/status/setting（numpy，最优那一步的解）、
    iters0（基解迭代数）、K_extra（追加步数；0=基解已达标）、
    energy_ft（最终失衡）、base_energy_ft、met（bool）、
    history（[(K, energy_ft), ...]）。"""
    base = solver.run_gga(d, rh, ke=ke, do_status=do_status,
                          hacc=hacc, max_iter=max_iter)
    e0 = energy_residual_ft(solver, base["head"], base["flow"],
                            base["status"], base["setting"])
    best = dict(head=base["head"], flow=base["flow"], emitter=base["emitter"],
                status=base["status"], setting=base["setting"],
                iters0=int(base["iters"]), K_extra=0, energy_ft=e0,
                base_energy_ft=e0, met=e0 <= tol_energy, history=[(0, e0)])
    if best["met"]:
        return best
    for k in k_schedule:
        out = solver.run_gga(d, rh, ke=ke, q0=base["flow"], e0=base["emitter"],
                             status0=base["status"], setting0=base["setting"],
                             do_status=False, max_iter=int(k), extra_iter=0,
                             hacc=0.0)
        e = energy_residual_ft(solver, out["head"], out["flow"],
                               base["status"], base["setting"])
        best["history"].append((int(k), e))
        if e < best["energy_ft"]:
            best.update(head=out["head"], flow=out["flow"],
                        emitter=out["emitter"], status=base["status"],
                        setting=base["setting"], K_extra=int(k), energy_ft=e)
        if e <= tol_energy:
            best["met"] = True
            return best
    return best


def build_reference(inp_path: str, out_dir: str, stem: str) -> dict:
    """跑完整 EPS 并写 <stem>_ref.npz 与 <stem>_meta.json 的 "reference" 键。

    返回 solve_eps() 的结果 dict（含 warnings），便于调用方直接复核。
    """
    os.makedirs(out_dir, exist_ok=True)

    with Epanet(inp_path) as en:
        version = en.version()
        cnt = en.counts()
        flow_units = en.flow_units
        res = en.solve_eps()
        open_warnings = list(en.warnings)  # 打开/读取阶段的警告（区别于运行警告）

    # ---- 数组落盘（warnings 是 list，不进 npz，只进 meta）----
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    npz_path = os.path.join(out_dir, f"{stem}_ref.npz")
    np.savez_compressed(npz_path, **arrays)

    # ---- 标量与警告并入 meta.json 的 "reference" 键 ----
    meta_path = os.path.join(out_dir, f"{stem}_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta["reference"] = {
        "epanet_version": int(version),
        "flow_units": int(flow_units),        # EN_FlowUnits 枚举值（1=GPM, 5=LPS）
        "n_frames": int(len(res["t_sec"])),
        "n_nodes": int(cnt["nodes"]),
        "n_links": int(cnt["links"]),
        "duration_sec": int(res["t_sec"][-1]),
        "iterations_max": int(res["iterations"].max()),
        "relerr_max": float(res["relerr"].max()),
        "warnings": res["warnings"],          # 运行警告 [[t_sec, code, msg], ...]
        "open_warnings": [list(w) for w in open_warnings],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return res


if __name__ == "__main__":
    # 冒烟测试：EXA6（GPM/US 制）与 city_d（LPS/SI 制）各跑一次完整 EPS 并落盘回读
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(_root, "data", "reference")
    cases = [
        (os.path.join(_root, "networks", "InpData", "EXA6.inp"), "EXA6"),
        (os.path.join(_root, "networks", "realInpData", "city_d.inp"), "city_d"),
    ]
    from dgga.epanet_ref import Epanet as _E  # 仅为打印版本

    for inp, stem in cases:
        print("=" * 72)
        print(f"[{stem}] {inp}")
        res = build_reference(inp, out_dir, stem)
        T = len(res["t_sec"])
        print(f"帧数 T = {T}，t_sec[0] = {res['t_sec'][0]}，t_sec[-1] = {res['t_sec'][-1]}")
        with _E(inp) as en:
            print(f"EN_getversion = {en.version()}，flow_units 枚举 = {en.flow_units}")
        print("逐帧 iterations / relerr：")
        for k in range(T):
            print(f"  t={res['t_sec'][k]:>6d}s  iter={res['iterations'][k]:>2d}  "
                  f"relerr={res['relerr'][k]:.3e}")
        print("head_ft  min/max = %.3f / %.3f ft"
              % (res["head_ft"].min(), res["head_ft"].max()))
        print("max|Q| = %.4f cfs" % np.abs(res["flow_cfs"]).max())
        print("运行警告 =", res["warnings"])

        # ---- 回读校验：npz 逐键与内存结果一致 ----
        npz_path = os.path.join(out_dir, f"{stem}_ref.npz")
        with np.load(npz_path) as z:
            for k in ("t_sec", "head_ft", "pressure_ft", "demand_out_cfs",
                      "flow_cfs", "status", "setting", "iterations", "relerr"):
                assert k in z.files, f"npz 缺键 {k}"
                assert np.array_equal(z[k], res[k]), f"npz 回读不一致: {k}"
        print(f"npz 回读校验通过: {npz_path}")
        meta_path = os.path.join(out_dir, f"{stem}_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            print("meta['reference'] =", json.dumps(json.load(f)["reference"],
                                                    ensure_ascii=False))

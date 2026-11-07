# -*- coding: utf-8 -*-
"""bc_mismatch.py - 边界条件失配试验（WR 大修 R2-M7，2026-10-23）。

真值观测用"真实"水库水头生成，反演模型只知道名义水头（pb.rh），不知道误差。
失配类型（只扰动水库节点，所有帧）：
  offDELTA   全部帧水库水头 +DELTA ft（液位计零点偏差），如 off0.3
  randS      每帧独立 N(0, S²) 误差（SCADA 液位噪声），如 rand0.3
  control    无边界误差（与 L1 同一观测，作配对对照）
其余与 calibrate.py 的 L1 完全一致：perpipe 真值 seed=7、σ=0.1 ft 量测噪声
（噪声种子 100+i，按节点生成）、λ=1e-4、缺省布点（hanoi 全 junction / city_d dopt40）、
同一 20% 传感器留出、20 训练帧 + 5 验证帧、同一校核器（Adam→L-BFGS→LM）。
评价用 calibrate.evaluate() 原样（模型侧用名义水头）。
输出 data/bc_mismatch_<net>.json（按 cond|i 合并）。
运行：python -X utf8 scripts/bc_mismatch.py --net hanoi --seeds 0,1,2,3,4 \
        --conds control,off0.1,off0.3,off1.0,rand0.3
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate as cal                                          # noqa: E402

BC_SEED0 = 700                       # rand 类边界误差种子 700+i（与噪声种子独立）


def make_obs_bc(pb, C_true, sigma, noise_seed, cond, bc_seed):
    rh_true = pb.rh.copy()
    res = np.asarray(pb.net.node_type) == 1
    if cond.startswith("off"):
        rh_true[:, res] += float(cond[3:])
    elif cond.startswith("rand"):
        rh_true[:, res] += float(cond[4:]) * np.random.default_rng(bc_seed).standard_normal(
            (rh_true.shape[0], int(res.sum())))
    elif cond != "control":
        raise ValueError(cond)
    sol = cal.solve_polished(pb.s, pb.d, rh_true, r_hw=pb.r_of_C(C_true), accuracy=1e-12,
                             max_iter=cal.OBS_MI, polish_steps=cal.OBS_POLISH)
    p = sol["head"] - pb.elev[None, :]
    noise = sigma * np.random.default_rng(noise_seed).standard_normal(p.shape)   # 与 make_obs 同式
    return dict(p_clean=p, obs=p + noise, d_used=pb.d, resid_inf=float(sol["resid_inf"].max()),
                n_obs_solve=1, bc_err_max_ft=float(np.abs(rh_true - pb.rh).max()))


def run_one(pb, cond, i, sigma=0.1, lam=1e-4):
    C_true, _ = cal.make_truth(pb, "perpipe", cal.TRUTH_SEED["perpipe"])
    nz = cal.NOISE_SEED0 + i
    scen = make_obs_bc(pb, C_true, sigma, nz, cond, BC_SEED0 + i)
    sensors = cal.get_sensors(pb, "default")
    sens_train, sens_hold = cal.split_holdout(sensors)
    reason, Vr, k_sub, rank_eps = cal.get_diag(pb, "default", sensors)
    t0 = time.perf_counter()
    c = cal.calibrate(pb, scen["obs"], sens_train, pb.d, lam=lam)
    ev = cal.evaluate(pb, C_true, c["C_hat"], reason, Vr, k_sub, rank_eps, scen,
                      sens_train, sens_hold, pb.d)
    keep = {k: float(v) for k, v in ev.items() if isinstance(v, (int, float, np.floating, np.integer))}
    rec = dict(net=pb.key, cond=cond, i=i, noise_seed=nz, sigma=sigma, lam=lam,
               bc_err_max_ft=scen["bc_err_max_ft"], t_sec=time.perf_counter() - t0,
               nfe=c["nfe"], nbwd=c["nbwd"], mse_train=c["mse_train"], **keep)
    print(f"[{pb.key}:{cond}:i{i}] info C-RMSE={keep.get('info_rmse', float('nan')):.3f} "
          f"sub={keep.get('sub_rmse', float('nan')):.3f} | train p={keep.get('train_rmse', float('nan')):.4f} "
          f"valF={keep.get('val_frame_rmse', float('nan')):.4f} valS={keep.get('val_sensor_rmse', float('nan')):.4f} ft "
          f"| {rec['t_sec']:.0f}s", flush=True)
    return rec


def summarize(recs):
    conds = sorted({r["cond"] for r in recs}, key=lambda c: (c != "control", c))
    keys = ("info_rmse", "sub_rmse", "train_rmse", "val_frame_rmse", "val_sensor_rmse")
    print("\ncond       n  " + "  ".join(f"{k:>16}" for k in keys) + "   （各种子中位数）")
    for c in conds:
        rs = [r for r in recs if r["cond"] == c]
        vals = [np.median([r[k] for r in rs if k in r]) if any(k in r for r in rs) else float("nan") for k in keys]
        print(f"{c:<9} {len(rs):2d}  " + "  ".join(f"{v:16.4f}" for v in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="hanoi", choices=["hanoi", "city_d"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--conds", default="control,off0.1,off0.3,off1.0,rand0.3")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pb = cal.Problem(a.net)
    import glob
    for cond in a.conds.split(","):
        path = os.path.join(cal.ROOT, "data", f"bc_mismatch_{a.net}_{cond}.json")   # 每种条件一个文件，可并行
        allr = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        for i in [int(x) for x in a.seeds.split(",")]:
            allr[str(i)] = run_one(pb, cond, i)
            json.dump(allr, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    recs = []
    for f in sorted(glob.glob(os.path.join(cal.ROOT, "data", f"bc_mismatch_{a.net}_*.json"))):
        recs += list(json.load(open(f, encoding="utf-8")).values())
    summarize(recs)
    print("saved -> data/bc_mismatch_%s_<cond>.json" % a.net)


if __name__ == "__main__":
    main()

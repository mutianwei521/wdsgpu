# -*- coding: utf-8 -*-
"""ulp_selfdrift.py - 官方 EPANET 引擎的 1-ULP 自扰实验（模块四任务 2a）。

**这条实验是"豁免"措辞的成败判据，不许绕过。**

做法：对同一 INP，用 wntr 自带的官方 EPANET 2.2 共享库（Windows epanet22.dll /
Linux libepanet22.so）跑两遍完整 EPS：
  run A：读出某个输入量的用户单位原值 v，用 EN_set* 写回**同一个 v**（保证两遍
         都经过 setter 路径，排除 setter 自身副作用）；
  run B：写回 nextafter(v, +inf)，即抬高恰好 1 个 ULP（相对量 ~2^-52）。
两遍之间只差 1 ULP 输入，其余完全相同。比较逐帧 max|ΔH| / max|ΔQ|。

**判据**：若官方引擎自身在该网上就漂移到我方偏差同量级，"数值伪影"成立；
若官方引擎自身很稳定（漂移远小于我方偏差），那就是我方的问题，必须如实承认。
对照组（表中判定 pass 的网）一并跑，用来证明这不是所有网的通病。

扰动量选择：
  · roughness：无量纲（Ucf=1），EN_setlinkvalue 直传，1 ULP 完全精确；
  · basedemand / elevation：用户单位原值直传（本脚本绕开 dgga 的 Ucf 换算，
    直接调 EN_set*，保证扰动就是 1 ULP 而非换算残差）。

用法:
  python scripts/exempt_diag/ulp_selfdrift.py            # 全部（四豁免网 + 对照）
  python scripts/exempt_diag/ulp_selfdrift.py pub_net3   # 单网
"""

import ctypes
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga import epanet_ref as ER   # noqa: E402
from align import resolve_inp       # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "exempt")

# 四个"豁免"网 + 对照组（表里判定 pass、同样含泵/池/阀/控制的网）
EXEMPT = ["pub_net3", "pub_bwsn_network_1", "pub_bwsn_network_2", "pub_net6"]
CONTROL = ["pub_c_town_batadal", "pub_d_town", "pub_richmond_standard",
           "ky5", "pub_ky10", "pub_net2", "pub_anytown_wntr", "pub_l_town"]

# 每网试验数（大网少几发，逐网如实记录）
N_TRIALS = {"pub_bwsn_network_2": 3, "pub_net6": 3, "pub_l_town": 2,
            "pub_ky10": 2}
DEFAULT_TRIALS = 5


# --------------------------------------------------------------------------
# 轻量 EPS：只读 head/flow/status/iters，比 solve_eps 少一半 ctypes 往返
# --------------------------------------------------------------------------
def eps_light(en):
    cnt = en.counts()
    nN, nL = cnt["nodes"], cnt["links"]
    lib, ph = en.lib, en._ph
    t = ctypes.c_long()
    tstep = ctypes.c_long()
    val = ctypes.c_double()
    stat = ctypes.c_double()
    ts, H, Q, S, IT, RE = [], [], [], [], [], []
    if lib.EN_openH(ph) > 100:
        raise RuntimeError("EN_openH 失败")
    try:
        if lib.EN_initH(ph, ER.EN_NOSAVE) > 100:
            raise RuntimeError("EN_initH 失败")
        while True:
            rc = lib.EN_runH(ph, ctypes.byref(t))
            if rc > 100:
                raise RuntimeError(f"EN_runH 失败 t={t.value} rc={rc}")
            h = np.empty(nN)
            for i in range(1, nN + 1):
                lib.EN_getnodevalue(ph, i, ER.EN_HEAD, ctypes.byref(val))
                h[i - 1] = val.value
            q = np.empty(nL)
            s = np.empty(nL, dtype=np.int8)
            for i in range(1, nL + 1):
                lib.EN_getlinkvalue(ph, i, ER.EN_FLOW, ctypes.byref(val))
                q[i - 1] = val.value
                lib.EN_getlinkvalue(ph, i, ER.EN_STATUS, ctypes.byref(val))
                s[i - 1] = int(val.value)
            lib.EN_getstatistic(ph, ER.EN_ITERATIONS, ctypes.byref(stat))
            IT.append(int(stat.value))
            lib.EN_getstatistic(ph, ER.EN_RELATIVEERROR, ctypes.byref(stat))
            RE.append(float(stat.value))
            ts.append(int(t.value))
            H.append(h)
            Q.append(q)
            S.append(s)
            if lib.EN_nextH(ph, ctypes.byref(tstep)) > 100:
                raise RuntimeError("EN_nextH 失败")
            if tstep.value == 0:
                break
    finally:
        lib.EN_closeH(ph)
    return {"t": np.asarray(ts), "H": np.vstack(H), "Q": np.vstack(Q),
            "S": np.vstack(S), "iters": np.asarray(IT),
            "relerr": np.asarray(RE), "ucf_head": en._ucf_head,
            "ucf_flow": en._ucf_flow}


# --------------------------------------------------------------------------
# 原始（用户单位）读写：绕开 Ucf 换算，保证扰动恰为 1 ULP
# --------------------------------------------------------------------------
def get_link_raw(en, idx, prop):
    v = ctypes.c_double()
    en.lib.EN_getlinkvalue(en._ph, idx, prop, ctypes.byref(v))
    return v.value


def set_link_raw(en, idx, prop, value):
    rc = en.lib.EN_setlinkvalue(en._ph, idx, prop, ctypes.c_double(value))
    if rc > 100:
        raise RuntimeError(f"EN_setlinkvalue rc={rc}")


def get_node_raw(en, idx, prop):
    v = ctypes.c_double()
    en.lib.EN_getnodevalue(en._ph, idx, prop, ctypes.byref(v))
    return v.value


def set_node_raw(en, idx, prop, value):
    rc = en.lib.EN_setnodevalue(en._ph, idx, prop, ctypes.c_double(value))
    if rc > 100:
        raise RuntimeError(f"EN_setnodevalue rc={rc}")


def pick_targets(inp, n_trials):
    """选扰动目标：n_trials 条管（按索引分位均匀取，可复现）+ 1 个最大基需水
    junction 的 basedemand + 1 个 reservoir 的 elevation。"""
    with ER.Epanet(inp) as en:
        lt = en.link_types()
        nt = en.node_types()
        pipes = np.where((lt == ER.EN_PIPE) | (lt == ER.EN_CVPIPE))[0] + 1
        tg = []
        if pipes.size:
            qs = np.linspace(0, pipes.size - 1, min(n_trials, pipes.size))
            for p in np.unique(np.round(qs).astype(int)):
                tg.append(("link", int(pipes[p]), ER.EN_ROUGHNESS, "roughness"))
        junc = np.where(nt == ER.EN_JUNCTION)[0] + 1
        if junc.size:
            bd = np.array([get_node_raw(en, int(i), ER.EN_BASEDEMAND)
                           for i in junc])
            j = int(junc[int(np.argmax(np.abs(bd)))])
            if abs(bd).max() > 0:
                tg.append(("node", j, ER.EN_BASEDEMAND, "basedemand"))
        res = np.where(nt == ER.EN_RESERVOIR)[0] + 1
        if res.size:
            tg.append(("node", int(res[0]), ER.EN_ELEVATION, "reservoir_head"))
    return tg


def run_once(inp, kind, idx, prop, perturb):
    """跑一遍 EPS。perturb=False → 写回原值；True → 写回 nextafter(原值)。"""
    with ER.Epanet(inp) as en:
        if kind == "link":
            v = get_link_raw(en, idx, prop)
            vv = np.nextafter(v, np.inf) if perturb else v
            set_link_raw(en, idx, prop, float(vv))
        else:
            v = get_node_raw(en, idx, prop)
            vv = np.nextafter(v, np.inf) if perturb else v
            set_node_raw(en, idx, prop, float(vv))
        r = eps_light(en)
        r["v0"] = float(v)
        r["v1"] = float(vv)
        r["n_warn"] = len(en.warnings)
    return r


def compare(a, b):
    """两次 EPS 结果的逐帧差（换算到内部单位 ft / cfs）。"""
    F = min(len(a["t"]), len(b["t"]))
    same_t = bool(np.array_equal(a["t"][:F], b["t"][:F])
                  and len(a["t"]) == len(b["t"]))
    dH = np.abs(a["H"][:F] - b["H"][:F]) / a["ucf_head"]
    dQ = np.abs(a["Q"][:F] - b["Q"][:F]) / a["ucf_flow"]
    same_S = bool(np.array_equal(a["S"][:F], b["S"][:F]))
    same_it = bool(np.array_equal(a["iters"][:F], b["iters"][:F]))
    fmax = int(dH.max(axis=1).argmax())
    return {"n_frames_a": int(len(a["t"])), "n_frames_b": int(len(b["t"])),
            "same_t": same_t, "same_status": same_S, "same_iters": same_it,
            "max_dH_ft": float(dH.max()),
            "max_dH_frame": fmax,
            "max_dH_node0": int(dH[fmax].argmax()),
            "p999_dH_ft": float(np.percentile(dH, 99.9)),
            "median_frame_max_dH_ft": float(np.median(dH.max(axis=1))),
            "max_dQ_cfs": float(dQ.max()),
            "n_frames_dH_ge_1e6": int((dH.max(axis=1) >= 1e-6).sum())}


def main(stems):
    rows = []
    for stem in stems:
        inp = resolve_inp(stem)
        if not os.path.isfile(inp):
            print(f"[跳过] {stem}: 找不到 INP {inp}", flush=True)
            continue
        nt = N_TRIALS.get(stem, DEFAULT_TRIALS)
        tg = pick_targets(inp, nt)
        print(f"\n=== {stem}  INP={os.path.relpath(inp, ROOT)}  "
              f"扰动目标 {len(tg)} 个 ===", flush=True)
        t0 = time.perf_counter()
        base = None
        for (kind, idx, prop, tag) in tg:
            try:
                a = run_once(inp, kind, idx, prop, False)
                b = run_once(inp, kind, idx, prop, True)
            except Exception as e:                       # noqa: BLE001
                print(f"  {tag}#{idx}: 失败 {type(e).__name__}: {e}", flush=True)
                continue
            if base is None:
                base = a
            c = compare(a, b)
            c.update({"stem": stem, "target": tag, "index_1based": int(idx),
                      "v0": a["v0"], "v1": b["v1"],
                      "rel_perturb": (abs(b["v1"] - a["v0"]) / abs(a["v0"])
                                      if a["v0"] else float("nan"))})
            rows.append(c)
            print(f"  {tag:<15s} idx={idx:<6d} 相对扰动={c['rel_perturb']:.3e}  "
                  f"帧数={c['n_frames_a']}/{c['n_frames_b']}  "
                  f"max|dH|={c['max_dH_ft']:.3e} ft (帧{c['max_dH_frame']})  "
                  f"max|dQ|={c['max_dQ_cfs']:.3e} cfs  "
                  f"状态同={c['same_status']} 迭代同={c['same_iters']}  "
                  f"超1e-6帧={c['n_frames_dH_ge_1e6']}/{c['n_frames_a']}",
                  flush=True)
        print(f"  [{stem}] 用时 {time.perf_counter() - t0:.1f}s", flush=True)
    out = os.path.join(OUT_DIR, "ulp_selfdrift.json"
                       if len(stems) > 1 else f"ulp_selfdrift_{stems[0]}.json")
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "python": sys.executable, "platform": sys.platform,
               "dll": ER._dll_path(), "rows": rows}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("\n写出:", out)
    # 汇总
    print("\n每网 1-ULP 自扰上确界（取该网所有扰动目标的最大值）")
    print(f"{'网':<24s}{'max|dH| ft':>13s}{'max|dQ| cfs':>13s}"
          f"{'状态全同':>10s}{'迭代全同':>10s}")
    for stem in stems:
        rs = [r for r in rows if r["stem"] == stem]
        if not rs:
            continue
        print(f"{stem:<24s}{max(r['max_dH_ft'] for r in rs):>13.3e}"
              f"{max(r['max_dQ_cfs'] for r in rs):>13.3e}"
              f"{str(all(r['same_status'] for r in rs)):>10s}"
              f"{str(all(r['same_iters'] for r in rs)):>10s}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else EXEMPT + CONTROL))

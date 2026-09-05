# -*- coding: utf-8 -*-
"""probe_status_headroom.py - 量化 EPANET 求解中"离散状态解析"占的迭代份额。

唯一问题：如果一个完美的"状态预测器"能在第 0 次迭代就给出收敛时的
LinkStatus/LinkSetting 构型，GGA 还需要几次迭代？

逐帧（完整 EPS，dgga.eps.EpsDriver）做三次求解，全部走 mode="epanet" 位级路径：
  A 常规解  ：EPANET 原样流程（do_status=True，CheckFreq=2/MaxCheck=10 原生节律）。
              记 n_A、最终 status/setting、求解中每一次状态翻转（迭代号+链路+来源）、
              是否进入 ExtraIter 段、是否 unbalanced。
  B 冻结解  ：把 A 的最终 status/setting 当作初始构型，do_status=False（状态检查
              全关），其余输入（d / 定水头 / q0 / e0）与 A 逐位相同。记 n_B。
              → 头空间比 = n_A / n_B，是"状态预测器"能夺回的迭代数上界。
  C 对照解  ：初始构型同 B，但状态检查仍开（do_status=True）。记 n_C。
              n_C - n_B = 状态检查本身的开销；n_A - n_C = 从错误初始构型出发的代价。
  At/Bt 紧解：同 A/B 但 hacc=HACC_TIGHT（1e-6），单独报告 - 用来回答"在比
              网络自带 ACCURACY 紧得多的容差下，状态机是不是收敛的绊脚石"。

有效性门槛（A 与 B 是否收敛到同一个解）：
  主判据 = 开启链路上的相对流量差 dQ_rel = max|Q_A-Q_B| / max|Q_A| <= DQ_REL_TOL。
  为什么不拿水头当主判据（两条实测理由，都在报告里）：
   1) 关闭/临时关闭链路切出的孤岛，在 EPANET 矩阵里只有 1/CBIG 量级的对角
      支撑，水头无定义。实测 ky5 某帧 max|ΔH|=9.19e3 ft，而开启链路流量差
      只有 7.7e-4 cfs - 纯假阳性。故水头差一律只在"经开启链路可达定水头节点"
      的连通掩码上统计（dH_conn），且只作参考列。
   2) 多数公开网 ACCURACY=1e-3~1e-2（c_town/d_town 是 0.01），A 与 B 在不同
      迭代停机，水头本来就允许差 O(1) ft，这是停机容差噪声不是解的分岔。
  副判据（更严）：dH_conn <= DH_TOL，只报通过率，不用于剔除。
不满足主判据的帧从头空间统计中剔除，并单独报告剔除比例与原因。

EPS 轨迹始终由 A 推进（B/C 是旁路实验，不污染 tank 积分 / 控制 / 规则）。

用法： python -X utf8 scripts/probe_status_headroom.py [--quick] [stem ...]
输出： data/status_headroom.json + 控制台中文表 + data/status_headroom_wip.txt
"""

import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.parse import Net        # noqa: E402
from dgga.eps import EpsDriver    # noqa: E402
from align import resolve_inp     # noqa: E402

REF_DIR = os.path.join(ROOT, "data", "reference")
OUT_JSON = os.path.join(ROOT, "data", "status_headroom.json")
WIP = os.path.join(ROOT, "data", "status_headroom_wip.txt")

# 主判据：开启链路相对流量差门槛（比多数网自带 ACCURACY=1e-2 严 10 倍）
DQ_REL_TOL = 1.0e-3
# 副判据（只报通过率）：连通节点上的绝对水头差门槛（ft）
DH_TOL = 1.0e-3
# 副实验：紧收敛容差（run_gga 的 hacc 不做 EPANET 的 1e-5 下钳）
HACC_TIGHT = 1.0e-6
# 紧容差解允许的最大迭代（超了记为未收敛，单列）
MAXIT_TIGHT = 200

# EN_LinkType：0=CVPIPE 1=PIPE 2=PUMP 3=PRV 4=PSV 5=PBV 6=FCV 7=TCV 8=GPV
# 可切换元件 = 带状态机的元件：CV 管、泵、全部阀
SWITCHABLE_TYPES = {0, 2, 3, 4, 5, 6, 7, 8}
# 但 TCV/GPV 在 hydstatus.c 中没有状态转换逻辑（只有 valvestatus 的 PRV/PSV、
# linkstatus 的 CV/PUMP/FCV），故另给一个"真状态机"口径
ACTIVE_SM_TYPES = {0, 2, 3, 4, 6}

# 分组
GROUP_SM = [           # 有阀/泵/控制/规则
    "pub_c_town_batadal", "pub_d_town", "EXA4", "EXA5", "ky3", "ky5",
    "pub_anytown", "pub_richmond_standard", "pub_bwsn_network_1",
    "pub_bwsn_network_2",   # 线索点名：参考解本身未收敛的大网（L=14831）
]
GROUP_CTRL = [         # 对照组：纯管道、无状态机
    "pub_hanoi", "pub_modena", "pub_fossolo_poly1",
    "rand_main_0000", "rand_main_0001", "rand_main_0002",
    "rand_main_0003", "rand_main_0004",
]
GROUP_RO = ["city_d"]  # 只读诊断，绝不进入任何统计/选型


class _Tracer:
    """在 GGASolver 实例上打补丁，记录一次 run_gga 内的迭代节律与状态翻转。"""

    HOOKS = ("_linkstatus", "_valvestatus", "_pswitch", "_badvalve")

    def __init__(self, solver):
        self.s = solver
        self._saved = {}
        self.npass = 0
        self.flips = []      # (iter, link, old, new, source)
        self.kchg = []       # (iter, link, old_setting, new_setting, source)

    def reset(self):
        self.npass = 0
        self.flips = []
        self.kchg = []

    def install(self):
        s = self.s
        orig_py = s._PY_np

        def py(*a, **kw):
            # _PY_np 在 while 循环体最顶端、每个 pass 恰好一次 → 迭代计数器
            self.npass += 1
            return orig_py(*a, **kw)

        self._saved["_PY_np"] = orig_py
        s._PY_np = py

        for name in self.HOOKS:
            orig = getattr(s, name)
            self._saved[name] = orig
            setattr(s, name, self._make(name, orig))

    def _make(self, name, orig):
        def wrapper(*a, **kw):
            # _badvalve(n_ep, S)；其余三个都是 (S, K, ...) 或 (S, K, H)
            S = a[1] if name == "_badvalve" else a[0]
            K = a[1] if name in ("_linkstatus", "_valvestatus", "_pswitch") else None
            S0 = S.copy()
            K0 = None if K is None else K.copy()
            r = orig(*a, **kw)
            for k in np.nonzero(S0 != S)[0].tolist():
                self.flips.append((self.npass, int(k), int(S0[k]), int(S[k]), name))
            if K0 is not None:
                for k in np.nonzero(K0 != K)[0].tolist():
                    self.kchg.append((self.npass, int(k), float(K0[k]),
                                      float(K[k]), name))
            return r
        return wrapper

    def remove(self):
        for name, fn in self._saved.items():
            try:
                delattr(self.s, name)      # 去掉实例属性，露出类方法
            except AttributeError:
                setattr(self.s, name, fn)
        self._saved = {}


def _connected_mask(n1, n2, open_mask, fixed_nodes, N):
    """经"开启链路"可达任一定水头节点的节点掩码（并查集）。

    关闭/临时关闭链路切出的孤岛在 EPANET 矩阵里只有 1/CBIG 量级的对角支撑，
    水头无物理意义（两次求解可以差上千 ft 而开启链路流量完全一致），
    比较水头时必须排除。"""
    parent = np.arange(N, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(n1[open_mask].tolist(), n2[open_mask].tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(N)], dtype=np.int64)
    good = {int(roots[i]) for i in np.asarray(fixed_nodes).tolist()}
    return np.isin(roots, list(good)) if good else np.zeros(N, dtype=bool)


class ProbeDriver(EpsDriver):
    """EpsDriver.run() 的探针版：逐帧做 A/B/C（原生容差）+ At/Bt（紧容差）求解。"""

    def run_probe(self, max_frames=100000):
        s = self.solver
        tr = _Tracer(s)
        self._inithyd()
        rows = []
        while True:
            t = self.Htime
            self._demands()
            self._controls()

            # ---- 快照：三次求解共用的完全相同的输入 ----
            d_in = np.array(self.d, dtype=np.float64, copy=True)
            H_in = np.array(self.H, dtype=np.float64, copy=True)
            q_in = np.array(self.q, dtype=np.float64, copy=True)
            e_in = None if self.e is None else np.array(self.e, copy=True)
            S_in = np.array(self.S, dtype=np.int8, copy=True)
            K_in = np.array(self.K, dtype=np.float64, copy=True)

            # ---- A：常规解（EPANET 原样）----
            tr.reset()
            tr.install()
            try:
                rA = s.run_gga(d_in, H_in, q0=q_in, e0=e_in, status0=S_in,
                               setting0=K_in, do_status=True)
            finally:
                tr.remove()
            nA = int(rA["iters"])
            S_A = np.array(rA["status"], dtype=np.int8, copy=True)
            K_A = np.array(rA["setting"], dtype=np.float64, copy=True)
            flips = list(tr.flips)
            kchg = list(tr.kchg)
            # nA > MaxIter：进入 hydsolver.c:168 的 ExtraIter 冻结段，或（当
            # ExtraIter=-1 即 UNBALANCED STOP 时）迭代用尽、EPANET 报
            # "WARNING: System unbalanced"。修正：本项目 19 个 INP 并非都是
            # STOP - c_town/d_town/EXA5/ky3/ky5/anytown/hanoi/modena/fossolo
            # 等 11 个是 UNBALANCED CONTINUE 10（extra_iter=10），
            # richmond/bwsn_1/bwsn_2/EXA4/rand_*/city_d 才是 -1。实测全池只有
            # bwsn_2 t=97200 一帧越过 MaxIter，而该网正是 extra_iter=-1，
            # 故此列在本次数据上等价于"用尽 MaxTrials"。
            hit_extra = nA > s.max_iter_default
            convA = bool(rA["converged"])
            # 预测器是否有事可做：A 的收敛构型 == 交给求解器的初始构型时，
            # "完美状态预测器"给出的就是热启动已经拿到的那份构型 → 零增益。
            cfg_same = bool(np.array_equal(S_A, S_in)
                            and np.array_equal(K_A, K_in))

            # ---- B：冻结解（状态检查全关，初始构型 = A 的最终构型）----
            # extra_iter 不覆写：与 A 用同一个 maxtrials（旧版强制 -1，
            # 在 extra_iter=10 的 11 个网上给 A 多 10 次迭代额度，不对称；
            # 实测无一帧触及该额度，故数值不变，但对比口径必须一致）
            rB = s.run_gga(d_in, H_in, q0=q_in, e0=e_in, status0=S_A,
                           setting0=K_A, do_status=False)
            nB = int(rB["iters"])
            openA = S_A > s.ST_CLOSED
            conn = _connected_mask(s.n1_np, s.n2_np, openA, s.fixed_nodes, s.N)
            dH = float(np.abs((rB["head"] - rA["head"])[conn]).max()) if conn.any() else 0.0
            dH_all = float(np.abs(rB["head"] - rA["head"]).max())
            dQ = (float(np.abs((rB["flow"] - rA["flow"])[openA]).max())
                  if openA.any() else 0.0)
            qscale = (float(np.abs(rA["flow"][openA]).max())
                      if openA.any() else 0.0)
            dQ_rel = dQ / qscale if qscale > 0.0 else 0.0

            # ---- C：初始构型同 B，但状态检查仍开 ----
            rC = s.run_gga(d_in, H_in, q0=q_in, e0=e_in, status0=S_A,
                           setting0=K_A, do_status=True)
            nC = int(rC["iters"])

            # ---- At/Bt：紧容差仲裁（判定"最终状态构型"是否良定义）----
            rAt = s.run_gga(d_in, H_in, q0=q_in, e0=e_in, status0=S_in,
                            setting0=K_in, do_status=True,
                            max_iter=MAXIT_TIGHT, hacc=HACC_TIGHT)
            S_At = np.array(rAt["status"], dtype=np.int8, copy=True)
            K_At = np.array(rAt["setting"], dtype=np.float64, copy=True)
            rBt = s.run_gga(d_in, H_in, q0=q_in, e0=e_in, status0=S_At,
                            setting0=K_At, do_status=False,
                            max_iter=MAXIT_TIGHT, hacc=HACC_TIGHT)
            openAt = S_At > s.ST_CLOSED
            connt = _connected_mask(s.n1_np, s.n2_np, openAt, s.fixed_nodes, s.N)
            dHt = (float(np.abs((rBt["head"] - rAt["head"])[connt]).max())
                   if connt.any() else 0.0)
            dQt = (float(np.abs((rBt["flow"] - rAt["flow"])[openAt]).max())
                   if openAt.any() else 0.0)
            same_cfg = bool(np.array_equal(S_At, S_A))

            rows.append(dict(
                t=int(t), n_A=nA, n_B=nB, n_C=nC,
                n_At=int(rAt["iters"]), n_Bt=int(rBt["iters"]),
                conv_At=bool(rAt["converged"]), conv_Bt=bool(rBt["converged"]),
                dH_t=dHt, dQ_t=dQt, cfg_tight_eq_loose=same_cfg,
                relerr_A=float(rA["relerr"]), relerr_B=float(rB["relerr"]),
                conv_A=convA, conv_B=bool(rB["converged"]),
                conv_C=bool(rC["converged"]),
                hit_extra=bool(hit_extra), cfg_same=cfg_same,
                dH_AB=dH, dH_AB_all=dH_all, dQ_AB=dQ, dQ_rel=dQ_rel,
                n_flip=len(flips), n_kchg=len(kchg),
                n_flip_links=len({f[1] for f in flips}),
                first_flip_iter=(min(f[0] for f in flips) if flips else 0),
                last_flip_iter=(max(f[0] for f in flips) if flips else 0),
                flip_links=sorted({f[1] for f in flips}),
                flip_iters=[f[0] for f in flips],
                flip_src={},
                status_A=S_A.tolist(),
            ))
            src = {}
            for f in flips:
                src[f[4]] = src.get(f[4], 0) + 1
            rows[-1]["flip_src"] = src

            # ---- EPS 轨迹用 A 推进（与原始 run() 逐位一致）----
            self.q = rA["flow"]
            self.e = rA["emitter"]
            self.S = rA["status"]
            self.K = rA["setting"]
            self.H = rA["head"]
            self.fixed_dem = rA["fixed_demand"]
            relerr = float(rA["relerr"])
            open_api = self.S > s.ST_CLOSED
            dem_out = d_in + self.e
            dem_out = np.where(s.is_fixed_node, self.fixed_dem, dem_out)
            self.node_dem = dem_out

            tstep = self._nexthyd(relerr)
            if tstep == 0:
                break
            if len(rows) >= max_frames:
                break
        return rows


def _stats(v):
    a = np.asarray(v, dtype=np.float64)
    if a.size == 0:
        return dict(n=0)
    return dict(n=int(a.size), min=float(a.min()), med=float(np.median(a)),
                mean=float(a.mean()), p90=float(np.percentile(a, 90)),
                max=float(a.max()))


def probe_one(stem, max_frames=100000):
    net = Net.load(REF_DIR, stem)
    inp = resolve_inp(stem)
    drv = ProbeDriver(net, inp_path=inp if os.path.isfile(inp) else None)
    t0 = time.time()
    rows = drv.run_probe(max_frames=max_frames)
    secs = time.time() - t0

    lt = np.asarray(net.link_type)
    n_switch = int(np.isin(lt, list(SWITCHABLE_TYPES)).sum())
    n_sm = int(np.isin(lt, list(ACTIVE_SM_TYPES)).sum())

    # 有效帧：主判据 = 开启链路相对流量差
    valid = [r for r in rows if r["dQ_rel"] <= DQ_REL_TOL]
    drop = [r for r in rows if r["dQ_rel"] > DQ_REL_TOL]
    # 副判据通过率（严格水头一致；不用于剔除）
    strict = [r for r in valid if r["dH_AB"] <= DH_TOL]

    nA = [r["n_A"] for r in valid]
    nB = [r["n_B"] for r in valid]
    nC = [r["n_C"] for r in valid]
    ratio = [r["n_A"] / r["n_B"] for r in valid]
    tt = [r for r in rows if r["conv_At"] and r["conv_Bt"]]
    nAt = [r["n_At"] for r in tt]
    nBt = [r["n_Bt"] for r in tt]
    ratio_t = [r["n_At"] / r["n_Bt"] for r in tt]

    flip_links = set()
    for r in rows:
        flip_links.update(r["flip_links"])
    # 帧间构型变化（含控制/规则驱动，非求解器内部）
    st = np.asarray([r["status_A"] for r in rows], dtype=np.int8)
    cfg_vary = int((st.min(axis=0) != st.max(axis=0)).sum()) if len(rows) else 0

    unconv_A = [r for r in rows if not r["conv_A"]]
    unconv_A_but_B = [r for r in unconv_A if r["conv_B"]]
    extra = [r for r in rows if r["hit_extra"]]

    f = np.asarray([r["n_flip"] for r in valid], dtype=np.float64)
    a = np.asarray(nA, dtype=np.float64)
    corr = (float(np.corrcoef(f, a)[0, 1])
            if f.size > 2 and f.std() > 0 and a.std() > 0 else None)

    return dict(
        stem=stem, N=int(net.N), L=int(net.L), secs=round(secs, 2),
        n_frames=len(rows), n_valid=len(valid), n_drop=len(drop),
        drop_frac=(len(drop) / len(rows)) if rows else 0.0,
        n_switchable=n_switch, n_state_machine=n_sm,
        n_flipped_in_solve=len(flip_links),
        n_config_varying_across_frames=cfg_vary,
        n_pumps=int(len(net.pump_link)), n_tanks=int(len(net.tank_node)),
        n_controls=int(len(net.ctl_link)), n_rules=int(len(net.rules_raw)),
        nA=_stats(nA), nB=_stats(nB), nC=_stats(nC), ratio=_stats(ratio),
        nAt=_stats(nAt), nBt=_stats(nBt), ratio_t=_stats(ratio_t),
        n_strict_dH_ok=len(strict),
        n_loose_dH_gt_tol=int(sum(1 for r in rows if r["dH_AB"] > DH_TOL)),
        max_dQ_rel=float(max(r["dQ_rel"] for r in rows)) if rows else 0.0,
        n_cfg_tight_ne_loose=int(sum(1 for r in rows
                                     if not r["cfg_tight_eq_loose"])),
        flips=_stats([r["n_flip"] for r in valid]),
        flip_links_per_frame=_stats([r["n_flip_links"] for r in valid]),
        last_flip_iter=_stats([r["last_flip_iter"] for r in valid
                               if r["n_flip"] > 0]),
        corr_flip_nA=corr,
        hacc_default=float(drv.solver.hacc_default),
        n_dQrel_gt_own_acc=int(sum(1 for r in rows
                                   if r["dQ_rel"] > drv.solver.hacc_default)),
        iter_weighted_ratio=(sum(r["n_A"] for r in valid)
                             / max(1, sum(r["n_B"] for r in valid))),
        n_unconv_A=len(unconv_A), n_unconv_A_but_B_conv=len(unconv_A_but_B),
        n_unconv_At=int(sum(1 for r in rows if not r["conv_At"])),
        n_unconv_At_but_Bt_conv=int(sum(1 for r in rows
                                        if not r["conv_At"] and r["conv_Bt"])),
        n_hit_extra=len(extra),
        max_dH_AB=float(max([r["dH_AB"] for r in rows])) if rows else 0.0,
        max_dH_AB_all=float(max([r["dH_AB_all"] for r in rows])) if rows else 0.0,
        max_dQ_AB=float(max([r["dQ_AB"] for r in rows])) if rows else 0.0,
        max_dH_t=float(max([r["dH_t"] for r in rows])) if rows else 0.0,
        max_dQ_t=float(max([r["dQ_t"] for r in rows])) if rows else 0.0,
        sum_nA=int(sum(r["n_A"] for r in valid)),
        sum_nB=int(sum(r["n_B"] for r in valid)),
        sum_nC=int(sum(r["n_C"] for r in valid)),
        sum_nAt=int(sum(r["n_At"] for r in valid)),
        sum_nBt=int(sum(r["n_Bt"] for r in valid)),
        frames=[{k: v for k, v in r.items() if k != "status_A"} for r in rows],
    )


def _fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    if args:
        groups = [("指定", args)]
    else:
        groups = [("状态机组", GROUP_SM), ("对照组(纯管道)", GROUP_CTRL),
                  ("只读诊断", GROUP_RO)]
    maxf = 60 if quick else 100000

    out = dict(dh_tol=DH_TOL, groups={}, nets={})
    for gname, stems in groups:
        out["groups"][gname] = stems
        for stem in stems:
            r = probe_one(stem, max_frames=maxf)
            out["nets"][stem] = r
            print(f"[{gname}] {stem}: 帧={r['n_frames']} 有效={r['n_valid']} "
                  f"n_A中位={r['nA'].get('med')} n_B中位={r['nB'].get('med')} "
                  f"头空间中位={_fmt(r['ratio'].get('med'))} "
                  f"p90={_fmt(r['ratio'].get('p90'))} ({r['secs']}s)")
            sys.stdout.flush()

    # ---------- 全池统计（只用 pub_*/rand_*/EXA/ky 的公开网；city_d 剔除） ----------
    def pool(stems):
        rat, ratt, fl, na = [], [], [], []
        sA = sB = sC = sAt = sBt = 0
        nfr = ndrop = nunc = nunc_b = nunct = nunct_b = 0
        tail = []
        # 按"预测器是否有事可做"拆分（cfg_same=收敛构型==热启动交进来的构型）
        eqA = eqB = eqC = neA = neB = neC = 0
        n_eq = n_ne = n_eq_AC_equal = 0
        for s in stems:
            r = out["nets"].get(s)
            if not r:
                continue
            nfr += r["n_frames"]
            ndrop += r["n_drop"]
            nunc += r["n_unconv_A"]
            nunc_b += r["n_unconv_A_but_B_conv"]
            nunct += r["n_unconv_At"]
            nunct_b += r["n_unconv_At_but_Bt_conv"]
            for fr in r["frames"]:
                if fr["conv_At"] and fr["conv_Bt"]:
                    ratt.append(fr["n_At"] / fr["n_Bt"])
                    sAt += fr["n_At"]
                    sBt += fr["n_Bt"]
                if fr["dQ_rel"] <= DQ_REL_TOL:
                    rat.append(fr["n_A"] / fr["n_B"])
                    fl.append(fr["n_flip"])
                    na.append(fr["n_A"])
                    sA += fr["n_A"]
                    sB += fr["n_B"]
                    sC += fr["n_C"]
                    tail.append((fr["n_A"], fr["n_B"]))
                    if fr.get("cfg_same", False):
                        n_eq += 1
                        eqA += fr["n_A"]
                        eqB += fr["n_B"]
                        eqC += fr["n_C"]
                        n_eq_AC_equal += int(fr["n_A"] == fr["n_C"])
                    else:
                        n_ne += 1
                        neA += fr["n_A"]
                        neB += fr["n_B"]
                        neC += fr["n_C"]
        c = None
        if len(fl) > 2 and np.std(fl) > 0 and np.std(na) > 0:
            c = float(np.corrcoef(fl, na)[0, 1])
        tails = {}
        for thr in (5, 8, 10, 15):
            sub = [a / b for a, b in tail if a >= thr]
            tails[f"n_A>={thr}"] = dict(n=len(sub), **(
                {k: v for k, v in _stats(sub).items() if k != "n"}))
        return dict(n_frames=nfr, n_drop=ndrop, n_unconv_A=nunc,
                    n_unconv_A_but_B_conv=nunc_b, n_unconv_At=nunct,
                    n_unconv_At_but_Bt_conv=nunct_b,
                    ratio=_stats(rat), ratio_tight=_stats(ratt),
                    sum_nA=sA, sum_nB=sB, sum_nC=sC, sum_nAt=sAt, sum_nBt=sBt,
                    total_speedup_AB=(sA / sB) if sB else None,
                    total_speedup_AC=(sA / sC) if sC else None,
                    total_speedup_CB=(sC / sB) if sB else None,
                    total_speedup_AtBt=(sAt / sBt) if sBt else None,
                    tail=tails, corr_flip_nA=c,
                    # 预测器可归因的加速：只有 cfg_same=False 的帧上、且只到 C
                    # 为止（C = 正确构型 + 状态检查照常开，这是 EPANET 等价解
                    # 唯一允许的部署方式；B 关掉了检查，等于取消校验，不能算）
                    cfg_eq=dict(n=n_eq, sum_nA=eqA, sum_nB=eqB, sum_nC=eqC,
                                n_AC_equal=n_eq_AC_equal),
                    cfg_ne=dict(n=n_ne, sum_nA=neA, sum_nB=neB, sum_nC=neC),
                    predictor_ceiling=(sA / (sA - (neA - neC))) if sA else None)

    have_sm = [s for s in GROUP_SM if s in out["nets"]]
    have_ct = [s for s in GROUP_CTRL if s in out["nets"]]
    out["pool_state_machine"] = pool(have_sm)
    out["pool_control"] = pool(have_ct)
    out["pool_all_public"] = pool(have_sm + have_ct)

    with open(OUT_JSON, "w", encoding="utf-8") as fjs:
        json.dump(out, fjs, ensure_ascii=False, indent=1)

    # ---------- 中文表 ----------
    lines = []
    lines.append("=" * 108)
    lines.append("离散状态解析的迭代占比探针（A=常规 / B=冻结构型+关状态检查 / "
                 "C=冻结构型+开状态检查）")
    lines.append(f"同解主判据：开启链路相对流量差 dQ_rel <= {DQ_REL_TOL:g}（不满足者剔除）；"
                 f"副判据：连通节点 max|ΔH| <= {DH_TOL:g} ft（只报通过率）")
    lines.append(f"紧容差副实验 hacc={HACC_TIGHT:g}，MaxIter={MAXIT_TIGHT}")
    lines.append("=" * 108)
    hdr = (f"{'网络':<24}{'帧':>5}{'剔':>4}{'nA中位':>8}{'nA最大':>8}"
           f"{'nB中位':>8}{'nB最大':>8}{'nC中位':>8}{'比中位':>8}{'比p90':>8}"
           f"{'比最大':>8}{'紧比中位':>10}{'紧比最大':>10}{'翻转/帧':>9}{'r(翻,nA)':>10}")
    for gname, stems in [("状态机组", [s for s in GROUP_SM if s in out["nets"]]),
                         ("对照组(纯管道)", [s for s in GROUP_CTRL if s in out["nets"]]),
                         ("只读诊断(不入统计)", [s for s in GROUP_RO if s in out["nets"]])]:
        if not stems:
            continue
        lines.append("")
        lines.append(f"--- {gname} ---")
        lines.append(hdr)
        for s in stems:
            r = out["nets"][s]
            lines.append(
                f"{s:<24}{r['n_frames']:>5}{r['n_drop']:>4}"
                f"{_fmt(r['nA'].get('med'),1):>8}{_fmt(r['nA'].get('max'),0):>8}"
                f"{_fmt(r['nB'].get('med'),1):>8}{_fmt(r['nB'].get('max'),0):>8}"
                f"{_fmt(r['nC'].get('med'),1):>8}"
                f"{_fmt(r['ratio'].get('med'),3):>8}{_fmt(r['ratio'].get('p90'),3):>8}"
                f"{_fmt(r['ratio'].get('max'),3):>8}"
                f"{_fmt(r['ratio_t'].get('med'),3):>10}{_fmt(r['ratio_t'].get('max'),3):>10}"
                f"{_fmt(r['flips'].get('mean'),2):>9}{_fmt(r['corr_flip_nA']):>10}")
    lines.append("")
    lines.append("--- 元件维度（可切换元件 = CV管+泵+全部阀；真状态机 = CV/泵/PRV/PSV/FCV）---")
    lines.append(f"{'网络':<24}{'L':>6}{'可切换':>8}{'真状态机':>10}"
                 f"{'求解中翻转过':>14}{'帧间构型变化':>14}{'A未收敛':>9}"
                 f"{'冻结后收敛':>12}{'用尽MaxTrials':>14}")
    for s in list(out["nets"]):
        r = out["nets"][s]
        lines.append(f"{s:<24}{r['L']:>6}{r['n_switchable']:>8}{r['n_state_machine']:>10}"
                     f"{r['n_flipped_in_solve']:>14}{r['n_config_varying_across_frames']:>14}"
                     f"{r['n_unconv_A']:>9}{r['n_unconv_A_but_B_conv']:>12}"
                     f"{r['n_hit_extra']:>14}")
    lines.append("")
    lines.append("--- 剔除诊断：A 与 B 是否真的收敛到同一个解 ---")
    lines.append(f"{'网络':<24}{'帧':>5}{'剔除':>6}{'严判据通过':>12}{'dQrel最大':>12}"
                 f"{'本网ACCURACY':>14}{'dQrel>本网acc':>15}"
                 f"{'dH连通最大':>12}{'dH全部最大':>12}{'紧解未收敛':>12}{'紧冻结后收敛':>14}")
    for s in list(out["nets"]):
        r = out["nets"][s]
        lines.append(f"{s:<24}{r['n_frames']:>5}{r['n_drop']:>6}"
                     f"{r['n_strict_dH_ok']:>12}{r['max_dQ_rel']:>12.2e}"
                     f"{r['hacc_default']:>14.1e}{r['n_dQrel_gt_own_acc']:>15}"
                     f"{r['max_dH_AB']:>12.2e}{r['max_dH_AB_all']:>12.2e}"
                     f"{r['n_unconv_At']:>12}{r['n_unconv_At_but_Bt_conv']:>14}")
    lines.append("")
    lines.append("--- 状态预测任务的维度（每帧真正需要预测对的元件数）---")
    lines.append(f"{'网络':<24}{'可切换':>8}{'求解中翻转过':>14}"
                 f"{'翻转链路/帧均':>14}{'翻转链路/帧最大':>16}"
                 f"{'末次翻转迭代中位':>18}{'末次翻转迭代最大':>18}{'迭代加权头空间':>16}")
    for s in list(out["nets"]):
        r = out["nets"][s]
        lines.append(f"{s:<24}{r['n_switchable']:>8}{r['n_flipped_in_solve']:>14}"
                     f"{_fmt(r['flip_links_per_frame'].get('mean'),2):>14}"
                     f"{_fmt(r['flip_links_per_frame'].get('max'),0):>16}"
                     f"{_fmt(r['last_flip_iter'].get('med'),1):>18}"
                     f"{_fmt(r['last_flip_iter'].get('max'),0):>18}"
                     f"{_fmt(r['iter_weighted_ratio'],4):>16}")
    lines.append("")
    for key, name in (("pool_state_machine", "全池·状态机组"),
                      ("pool_control", "全池·对照组"),
                      ("pool_all_public", "全池·全部公开网")):
        p = out[key]
        lines.append(
            f"{name}: 帧={p['n_frames']} 剔除={p['n_drop']}"
            f"({100*p['n_drop']/max(1,p['n_frames']):.1f}%) "
            f"头空间 中位={_fmt(p['ratio'].get('med'),3)} "
            f"p90={_fmt(p['ratio'].get('p90'),3)} 均值={_fmt(p['ratio'].get('mean'),3)} "
            f"最大={_fmt(p['ratio'].get('max'),3)} | 紧容差头空间 中位="
            f"{_fmt(p['ratio_tight'].get('med'),3)} p90={_fmt(p['ratio_tight'].get('p90'),3)} "
            f"最大={_fmt(p['ratio_tight'].get('max'),3)}")
        lines.append(
            f"    迭代总量 A={p['sum_nA']} B={p['sum_nB']} C={p['sum_nC']} → "
            f"总加速 A/B={_fmt(p['total_speedup_AB'],4)}（完美状态预测器上界）, "
            f"A/C={_fmt(p['total_speedup_AC'],4)}（正确初值的收益）, "
            f"C/B={_fmt(p['total_speedup_CB'],4)}（状态检查本身开销）; "
            f"紧容差 At/Bt={_fmt(p['total_speedup_AtBt'],4)}")
        lines.append(
            f"    A未收敛帧={p['n_unconv_A']} 其中冻结后收敛={p['n_unconv_A_but_B_conv']}; "
            f"紧解未收敛={p['n_unconv_At']} 其中冻结后收敛={p['n_unconv_At_but_Bt_conv']}; "
            f"r(翻转,nA)={_fmt(p['corr_flip_nA'])}")
        tl = " ".join(f"{k}:n={v['n']},中位={_fmt(v.get('med'),2)},"
                      f"最大={_fmt(v.get('max'),2)}" for k, v in p["tail"].items())
        lines.append(f"    困难帧尾部头空间 {tl}")
        eq, ne = p["cfg_eq"], p["cfg_ne"]
        lines.append(
            f"    【预测器归因拆分】收敛构型==热启动构型的帧 n={eq['n']}"
            f"（预测器零增益，实测 n_A==n_C 的 {eq['n_AC_equal']}/{eq['n']}）："
            f"A={eq['sum_nA']} C={eq['sum_nC']} B={eq['sum_nB']}；"
            f"构型需修正的帧 n={ne['n']}：A={ne['sum_nA']} C={ne['sum_nC']} "
            f"B={ne['sum_nB']} → 预测器真正省下 {ne['sum_nA']-ne['sum_nC']} 次迭代，"
            f"全池预测器上界={_fmt(p['predictor_ceiling'],4)}x"
            f"（A/B={_fmt(p['total_speedup_AB'],4)} 里剩下的部分是"
            f"'关掉状态检查'，等价解不允许）")
    txt = "\n".join(lines)
    print()
    print(txt)
    with open(WIP, "a", encoding="utf-8") as fw:
        fw.write("\n" + "=" * 108 + "\n")
        fw.write(time.strftime("%Y-%m-%d %H:%M:%S") + "  probe_status_headroom\n")
        fw.write(txt + "\n")
    print(f"\n落盘: {OUT_JSON}\n证据: {WIP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

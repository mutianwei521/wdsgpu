# -*- coding: utf-8 -*-
"""gate_b1_batch_sm.py - 门 B1：dense 批量状态机不动点 + CVPIPE 的验收。

四项硬验收（dense_gap_plan.md §7 第 5 步）：
  ① 逐位对拍：同一批极端场景，dense 批量状态机（mode="dense",
     dense_status_machine=True, solve(status_machine=True)）给出的**最终状态向量**
     与 mode="epanet" 的串行状态机（hydsolve hydsolver.c:150-189 的逐样本复刻）
     **逐元素相等**；头/流量差异只报量级（两条路的线性求解次序本就不同，
     dense 与 EPANET 的头差在无阀网上就已有 ~3e-6 ft，见 solver.py 类注释）。
  ② 批内异态：报每个网"状态随场景变化的可切换元件数"与"批内不同状态组合数
     （桶）"（口径同 dense_gap_plan.md §1(c)：可切换元件 = CVPIPE/PUMP/PRV/PSV/FCV）。
     桶数=1 说明场景不够极端。
  ③ 解锁计数：`_cleaned` 口径 21 网的 dense 可构造清单（实测，不照抄计划表）。
  ④ 梯度：新特性网上 unrolled / implicit 两条梯度与中央差分对拍（另见
     scripts/gate_b1_grad.py）。

极端场景配方（照 §1(c)：需水×0.05~4.0 + 随机时刻 + 单点大漏损 2%~60%），
本脚本再加一条**水池水位随机**（含贴边 Hmin/Hmax） - 那是 tankstatus
（hydstatus.c:401-476）唯一的触发口，不贴边就永远测不到 TEMPCLOSED 支。

用法： python -X utf8 scripts/gate_b1_batch_sm.py [--B 128] [网名 ...]
"""

import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                  # noqa: E402
from dgga.parse import parse_inp              # noqa: E402
from dgga.solver import GGASolver             # noqa: E402

PUB = os.path.join(ROOT, "networks", "public")
CLEAN = os.path.join(PUB, "_cleaned")
# 可切换元件（dense_gap_plan.md §1(c) 的 ACTIVE_SM_TYPES 口径）
SM_TYPES = {0, 2, 3, 4, 6}       # CVPIPE / PUMP / PRV / PSV / FCV


def inp_of(name):
    """`_cleaned` 口径：有清洗版用清洗版，否则用原始版。"""
    p = os.path.join(CLEAN, name + ".inp")
    return p if os.path.exists(p) else os.path.join(PUB, name + ".inp")


def all_public():
    return sorted(f[:-4] for f in os.listdir(PUB) if f.endswith(".inp"))


# ----------------------------------------------------------------------
def make_scenarios(net, B, seed):
    """极端场景批：返回 (demand [B,N], res_head [B,N])。

    · 需水乘子 ~ 对数均匀 [0.05, 4.0]（§1(c) 配方）
    · 随机时刻（从 pattern 长度里抽，命中不同的时段乘子）
    · 单点大漏损：随机 junction 上加 frac×全网需水，frac ~ U(0.02, 0.60)
    · 水池水位：U(Hmin, Hmax)，其中 30% 的场景贴到 Hmin / Hmax
      （tankstatus 的满/空池判据 hydstatus.c:444/:461 只在贴边时动作）
    """
    rng = np.random.default_rng(seed)
    N = net.N
    nt = np.asarray(net.node_type)
    junc = np.where(nt == 0)[0]
    tanks = np.asarray(net.tank_node, dtype=np.int64)
    h0 = np.asarray(net.tank_h0, dtype=np.float64)
    hmin = np.asarray(net.tank_hmin, dtype=np.float64)
    hmax = np.asarray(net.tank_hmax, dtype=np.float64)
    # 时刻候选：pattern 步长 × 长度（无 pattern 则只有 0）
    plen = max([len(p) for p in net.patterns], default=1)
    pstep = int(net.meta.get("pat_step_sec", 3600) or 3600)

    D = np.zeros((B, N), dtype=np.float64)
    RH = np.zeros((B, N), dtype=np.float64)
    for b in range(B):
        t = int(rng.integers(0, max(plen, 1))) * pstep
        # 前 1/4 场景走"泵截断角"：需水近零 + 全池满 - pumpstatus 的 XHEAD
        # （hydstatus.c:235 扬程增益 > ω²·Hmax + Htol）只在这种工况下才够得着。
        corner = (b % 4 == 0)
        if corner:
            mult = float(np.exp(rng.uniform(np.log(0.01), np.log(0.10))))
        else:
            mult = float(np.exp(rng.uniform(np.log(0.05), np.log(4.0))))
        d = net.demand_cfs_at(t) * mult
        tot = float(np.abs(d[junc]).sum())
        j = int(junc[rng.integers(0, junc.size)])
        if not corner:
            d[j] += tot * float(rng.uniform(0.02, 0.60))
        rh = net.reservoir_head_ft_at(t)
        rh = np.where(np.isnan(rh), 0.0, rh)
        for i, n in enumerate(tanks):
            n = int(n)
            u = rng.random()
            if corner:
                rh[n] = hmax[i]                  # 满池 + 近零需水 → 泵最易被截断
            elif u < 0.15:
                rh[n] = hmin[i]                  # 空池（:461 必触发）
            elif u < 0.30:
                rh[n] = hmax[i]                  # 满池（:444，CanOverflow=0 时触发）
            else:
                lo, hi = float(hmin[i]), float(hmax[i])
                rh[n] = lo + (hi - lo) * float(rng.random()) if hi > lo else float(h0[i])
        D[b] = d
        RH[b] = rh
    return D, RH


def _connected_mask(net, opened):
    """[B,N] 布尔：经"两边都开启"的链路可达任一定水头节点的节点。"""
    B = opened.shape[0]
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    fixed = np.asarray(net.node_type) != 0
    out = np.zeros((B, net.N), dtype=bool)
    for b in range(B):
        reach = fixed.copy()
        e1, e2 = n1[opened[b]], n2[opened[b]]
        while True:
            new = reach.copy()
            np.logical_or.at(new, e2, reach[e1])
            np.logical_or.at(new, e1, reach[e2])
            if np.array_equal(new, reach):
                break
            reach = new
        out[b] = reach
    return out


def bucket_stats(net, S):
    """§1(c) 口径：可切换元件的批内异态。S: int8[B,L] 最终状态。
    返回 (变态元件数, 可切换元件总数, 桶数, 逐元件明细 list[str])。"""
    lt = np.asarray(net.link_type)
    idx = np.where(np.isin(lt, sorted(SM_TYPES)))[0]
    nb_all = int(len(np.unique(S, axis=0)))          # 全链路口径（含被 tankstatus
    nv_all = int(sum(len(np.unique(S[:, k])) > 1     # 置 TEMPCLOSED 的普通管）
                     for k in range(S.shape[1])))
    if idx.size == 0:
        return 0, 0, 1, [], nv_all, nb_all
    sub = S[:, idx]
    varying = [i for i, k in enumerate(idx) if len(np.unique(sub[:, i])) > 1]
    nbucket = len(np.unique(sub, axis=0))
    NAME = {0: "CVPIPE", 1: "PIPE", 2: "PUMP", 3: "PRV", 4: "PSV", 6: "FCV", 7: "TCV"}
    ST = {0: "XHEAD", 1: "TEMPCLOSED", 2: "CLOSED", 3: "OPEN", 4: "ACTIVE",
          5: "XFLOW", 6: "XFCV", 7: "XPRESSURE"}
    detail = []
    show = varying[:6] if varying else list(range(min(idx.size, 6)))
    for i in show:
        k = int(idx[i])
        u, c = np.unique(sub[:, i], return_counts=True)
        detail.append(f"{NAME.get(int(lt[k]), lt[k])}#{k}({net.link_id[k]}): "
                      + " / ".join(f"{ST[int(a)]}:{int(b)}" for a, b in zip(u, c)))
    return len(varying), int(idx.size), int(nbucket), detail, nv_all, nb_all


# ----------------------------------------------------------------------
def check_net(name, B, seed):
    inp = inp_of(name)
    net = parse_inp(inp)
    D, RH = make_scenarios(net, B, seed)

    s_ep = GGASolver(net, mode="epanet", inp_path=inp)
    s_dn = GGASolver(net, mode="dense", inp_path=inp, dense_status_machine=True)

    t0 = time.time()
    r_ep = s_ep.solve(D, RH, status_machine=True)
    t_ep = time.time() - t0
    t0 = time.time()
    r_dn = s_dn.solve(D, RH, status_machine=True)
    t_dn = time.time() - t0

    S_ep = r_ep["status"].numpy().astype(np.int8)
    S_dn = r_dn["status"].numpy().astype(np.int8)
    same = bool(np.array_equal(S_ep, S_dn))
    nmis = int((S_ep != S_dn).sum())

    H_ep, H_dn = r_ep["head_ft"].numpy(), r_dn["head_ft"].numpy()
    Q_ep, Q_dn = r_ep["flow_cfs"].numpy(), r_dn["flow_cfs"].numpy()
    jm = np.asarray(net.node_type) == 0
    # 只在两边都判为"开启"的链路上比流量（关闭支的 Q 是残值，无物理意义）
    opened = (S_ep > 2) & (S_dn > 2)
    dH_all = float(np.abs(H_ep - H_dn)[:, jm].max())
    dQ = float(np.abs(Q_ep - Q_dn)[opened].max()) if opened.any() else 0.0
    # 连通掩码（口径同 probe_status_headroom.py）：被关闭链路切出的孤岛在
    # EPANET 矩阵里只有 1/CBIG 量级对角支撑，水头无定义，比它是假阳性。
    conn = _connected_mask(net, opened)
    m = conn & jm[None, :]
    AD = np.abs(H_ep - H_dn)
    dH = float(AD[m].max()) if m.any() else 0.0
    # 相对口径：极端场景里会出现 |H|~1e26 ft 的病态解（Anytown_wntr 的 INP 含一根
    # 阻力极端的管，EPANET 自己也给这个量级），绝对差没有意义，故并报相对差。
    RD = AD / np.maximum(np.abs(H_ep), 1.0)
    dHr = float(RD[m].max()) if m.any() else 0.0
    per = AD.copy()
    per[~m] = 0.0
    per = per.max(axis=1)
    dH_med, dH_p90 = float(np.median(per)), float(np.percentile(per, 90))
    # 只在"两条路都收敛"的场景上再报一次（未收敛帧的水头本就无定义可比性）
    bc = r_ep["converged"].numpy() & r_dn["converged"].numpy()
    dH_bc = float(per[bc].max()) if bc.any() else 0.0
    dQ_bc = (float(np.abs(Q_ep - Q_dn)[bc][opened[bc]].max())
             if bc.any() and opened[bc].any() else 0.0)
    it_ep = r_ep["iters"].numpy()
    it_dn = r_dn["iters"].numpy()
    cv_ep = int(r_ep["converged"].numpy().sum())
    cv_dn = int(r_dn["converged"].numpy().sum())
    nvary, ntot, nbuck, detail, nv_all, nb_all = bucket_stats(net, S_dn)

    print(f"\n=== {name}  (Nj={s_dn.Nj}, L={net.L}, B={B}) ===")
    print(f"  ① 状态逐元素相等 : {'是' if same else '否'}"
          f"（不等元素 {nmis} / {S_ep.size}）")
    print(f"     |ΔH| 连通口径  : max={dH:.3e} ft  p90={dH_p90:.3e}  "
          f"中位={dH_med:.3e}  相对 max={dHr:.3e}")
    print(f"     |ΔH| 全junction: max={dH_all:.3e} ft      max|ΔQ|={dQ:.3e} cfs")
    print(f"     两路均收敛子集  : {int(bc.sum())}/{B} 场景  max|ΔH|={dH_bc:.3e} ft "
          f" max|ΔQ|={dQ_bc:.3e} cfs")
    # ①b 批不变性：per-scenario 冻结掩码必须让"整批一次算"== "逐场景 B=1 算"
    nb1 = min(B, 16)
    S1 = np.empty((nb1, net.L), dtype=np.int8)
    H1 = np.empty((nb1, net.N), dtype=np.float64)
    for b in range(nb1):
        rb = s_dn.solve(D[b:b + 1], RH[b:b + 1], status_machine=True)
        S1[b] = rb["status"].numpy()[0]
        H1[b] = rb["head_ft"].numpy()[0]
    b1_st = bool(np.array_equal(S1, S_dn[:nb1]))
    b1_dh = float(np.abs(H1 - H_dn[:nb1]).max())
    print(f"     ①b 批不变性     : 前 {nb1} 场景 B=1 逐个跑 vs 整批  "
          f"状态{'相等' if b1_st else '不等'}  max|ΔH|={b1_dh:.3e} ft"
          f"{'（逐位）' if b1_dh == 0.0 else ''}")
    print(f"     迭代数         : epanet {it_ep.min()}..{it_ep.max()} / "
          f"dense {it_dn.min()}..{it_dn.max()}  "
          f"(逐场景相等 {int((it_ep == it_dn).sum())}/{B})")
    print(f"     收敛           : epanet {cv_ep}/{B}  dense {cv_dn}/{B}")
    print(f"     耗时           : epanet {t_ep:.2f}s  dense(批) {t_dn:.2f}s")
    print(f"  ② 批内异态       : §1(c)口径(CV/泵/阀) 变态元件 {nvary}/{ntot}，桶 {nbuck}"
          f"   |  全链路口径 变态 {nv_all}/{net.L}，桶 {nb_all}"
          f"{'   <-- 全链路也无异态' if nb_all <= 1 else ''}")
    for line in detail:
        print(f"        {line}")
    # 泵 XHEAD 的"逼近度"：批内最大扬程增益 / 截止扬程（>1+Htol 才会翻 XHEAD）。
    # 用来区分"场景不够极端"与"该网单帧稳态下根本够不着这条转移"。
    ratio = None
    if s_dn.n_pumps:
        pl = np.asarray(s_dn.pump_links)
        n1 = np.asarray(net.link_n1)[pl]
        n2 = np.asarray(net.link_n2)[pl]
        gain = H_dn[:, n2] - H_dn[:, n1]
        hmx = np.where(s_dn.is_chp_np[pl], np.inf,
                       np.asarray(s_dn.init_setting)[pl] ** 2 * s_dn.pl_hmax[pl])
        # 只统计"linkstatus 真会去查 pumpstatus"的泵（:144-148 的 S>=OPEN 且
        # LinkSetting>0）；关停泵是 1/CBIG 高阻支，其"扬程增益"无物理意义。
        live = (S_dn[:, pl] >= 3) & (np.asarray(s_dn.init_setting)[pl] > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(live, gain / hmx, -np.inf)
        if np.isfinite(rr).any():
            ratio = float(rr[np.isfinite(rr)].max())
            print(f"     泵 XHEAD 逼近度 : max(扬程增益/截止扬程)={ratio:.6f}"
                  f"（>1+Htol 才翻 XHEAD；截止扬程 "
                  f"{np.array2string(hmx, precision=1)} ft）")
        else:
            print("     泵 XHEAD 逼近度 : 不适用（批内全部泵为关停/CONST_HP）")
    return dict(name=name, same=same, nmis=nmis, dH=dH, dH_all=dH_all, dQ=dQ,
                pump_ratio=ratio,
                dHr=dHr, dH_med=dH_med, dH_p90=dH_p90,
                dH_bc=dH_bc, dQ_bc=dQ_bc, n_bc=int(bc.sum()),
                b1_st=b1_st, b1_dh=b1_dh,
                nv_all=nv_all, nb_all=nb_all,
                nvary=nvary, ntot=ntot, nbucket=nbuck, B=B,
                conv_ep=cv_ep, conv_dn=cv_dn,
                it_eq=int((it_ep == it_dn).sum()), t_ep=t_ep, t_dn=t_dn,
                detail=detail)


def dense_reach():
    """③ `_cleaned` 口径 21 网的 dense 可构造清单（三档实测）。"""
    rows = []
    for nm in all_public():
        p = inp_of(nm)
        used = "_cleaned" if p.startswith(CLEAN) else "public"
        try:
            net = parse_inp(p)
        except Exception as e:
            rows.append((nm, used, "-", "-", "-", f"parse: {type(e).__name__}"))
            continue
        res = []
        for lbl, kw in (("旧缺省", dict()),
                        ("旧+关水池静态守卫", dict(dense_tank_bound_check=False)),
                        ("门B1", dict(dense_status_machine=True))):
            try:
                GGASolver(net, mode="dense", inp_path=p, **kw)
                res.append("OK")
            except Exception as e:
                msg = str(e).split("\n")[0]
                res.append("×" if len(msg) > 0 else "×")
        rows.append((nm, used, res[0], res[1], res[2], ""))
    return rows


def main():
    argv, B, skip = [], 128, False
    for i, a in enumerate(sys.argv[1:]):
        if skip:
            skip = False
            continue
        if a == "--B":
            B = int(sys.argv[i + 2]); skip = True
        elif not a.startswith("--"):
            argv.append(a)
    nets = argv or ["Richmond_skeleton", "Anytown_wntr", "ky4", "Net3",
                    "Net1", "Net2", "Anytown", "Hanoi", "Modena",
                    "Fossolo_poly1", "Pescara"]
    print(f"门 B1 验收：dense 批量状态机不动点 + CVPIPE   B={B}")
    print("=" * 78)
    out = []
    for nm in nets:
        try:
            out.append(check_net(nm, B, seed=20260822))
        except Exception as e:
            import traceback
            print(f"\n=== {nm} ===\n  异常: {type(e).__name__}: {e}")
            traceback.print_exc()
            out.append(dict(name=nm, same=False, err=str(e)))

    print("\n" + "=" * 78)
    print("③ 解锁计数（_cleaned 口径，21 网 dense 可构造）")
    rows = dense_reach()
    hdr = ("网", "取文件", "旧缺省", "旧+关静态水池守卫", "门B1")
    print(f"  {hdr[0]:<20}{hdr[1]:<10}{hdr[2]:<8}{hdr[3]:<20}{hdr[4]:<6}")
    c = [0, 0, 0]
    for nm, used, a, b, cc, note in rows:
        print(f"  {nm:<20}{used:<10}{a:<8}{b:<20}{cc:<6}{note}")
        c[0] += a == "OK"; c[1] += b == "OK"; c[2] += cc == "OK"
    print(f"  合计: 旧缺省 {c[0]}/21   旧+关静态水池守卫 {c[1]}/21   门B1 {c[2]}/21")

    ok = all(r.get("same") and r.get("b1_st") for r in out)
    bad = [r["name"] for r in out if not (r.get("same") and r.get("b1_st"))]
    print("\n" + "=" * 78)
    print(f"总判定: {'PASS' if ok else 'FAIL'}" + ("" if ok else f"  超限网: {bad}"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

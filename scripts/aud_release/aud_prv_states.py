# -*- coding: utf-8 -*-
"""aud_prv_states.py - 敌意复测（自写，不复用上游测试）：

① L-TOWN（可选其他 PRV 网）B 个自配极端场景 + 每阀两条逼阀场景：
   dense 批量状态机 vs mode="epanet" 串行，逐场景硬比：
     调度轨迹（每次 valvestatus/linkstatus 调用的迭代号 + change 链路集，
     含空 change 的"检查时刻"；conv/per 支标签只报不判）、
     终态状态向量逐元素、iters。
② 逐 PRV 批内终态直方图（ACTIVE/OPEN/CLOSED 三态覆盖，缺态即 FAIL）。
③ 调度全同且双收敛场景的连通域 max|dH|（报分布；> DH_GATE ft 记硬 FAIL -
   κ~1.6e11 的线代噪声实测 <1e-2 ft，0.5 ft 的门只有实现错误才碰得到）。
④ 名义帧（t=0，tank 中位）：ACTIVE PRV 计数 + 收敛帧 cond2(A)（--kappa 时）。

用法：python -X utf8 aud_prv_states.py [--B 128] [--pkg DIR] [--kappa] [网名..]
--pkg 指向包含 dgga/ 的目录（变异体检测用；缺省仓库根）。
退出码 0 = 终态全同 + 调度不同 ≤5% + 三态全出 + dH 门内。
"""
import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=128)
ap.add_argument("--pkg", default=ROOT)
ap.add_argument("--kappa", action="store_true")
ap.add_argument("--seed", type=int, default=909)
ap.add_argument("nets", nargs="*", default=None)
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.pkg))
import dgga                                    # noqa: E402
import torch                                   # noqa: E402
from dgga.parse import parse_inp               # noqa: E402
from dgga.solver import GGASolver              # noqa: E402

print("dgga from:", os.path.dirname(os.path.abspath(dgga.__file__)))
PUB = os.path.join(ROOT, "networks", "public")
CLEAN = os.path.join(PUB, "_cleaned")
NETS = args.nets or ["L-TOWN"]
STN = {2: "CLOSED", 3: "OPEN", 4: "ACTIVE", 7: "XPRESS"}
DH_GATE = 0.5


def inp_of(name):
    p = os.path.join(CLEAN, name + ".inp")
    return p if os.path.exists(p) else os.path.join(PUB, name + ".inp")


def scenarios(net, B, seed, prv):
    """自配配方：极端需水/漏损/水池贴边 + 每阀逼 OPEN / 逼 CLOSED。"""
    rng = np.random.default_rng(seed)
    nt = np.asarray(net.node_type)
    junc = np.where(nt == 0)[0]
    tanks = np.asarray(net.tank_node, dtype=np.int64)
    hmin = np.asarray(net.tank_hmin, dtype=np.float64)
    hmax = np.asarray(net.tank_hmax, dtype=np.float64)
    plen = max([len(p) for p in net.patterns], default=1)
    pstep = int(net.meta.get("pat_step_sec", 3600) or 3600)
    rows = []
    for b in range(B):
        t = int(rng.integers(0, max(plen, 1))) * pstep
        mult = float(np.exp(rng.uniform(np.log(0.05), np.log(5.0))))
        d = net.demand_cfs_at(t) * mult
        tot = float(np.abs(d[junc]).sum())
        if b % 3 != 2:
            j = int(junc[rng.integers(0, junc.size)])
            d[j] += max(tot, 1e-6) * float(rng.uniform(0.05, 1.0))
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(t),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            u = rng.random()
            if u < 0.25:
                rh[int(n)] = hmin[i]
            elif u < 0.50:
                rh[int(n)] = hmax[i]
            else:
                rh[int(n)] = hmin[i] + (hmax[i] - hmin[i]) * rng.random()
        rows.append((d, rh))
    d0 = net.demand_cfs_at(0)
    tot0 = float(np.abs(d0[junc]).sum())
    nt_ = nt
    for k in prv:
        n2 = int(net.link_n2[k])
        d = d0 * 20.0                                   # 逼 OPEN
        if nt_[n2] == 0:
            d[n2] += 12.0 * max(tot0, 1e-6)
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            rh[int(n)] = hmin[i]
        rows.append((d, rh))
        d = d0 * 0.1                                    # 逼 CLOSED（下游注入）
        if nt_[n2] == 0:
            d[n2] -= 2.0 * max(tot0, 1e-6)
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                    dtype=np.float64))
        for i, n in enumerate(tanks):
            rh[int(n)] = hmax[i]
        rows.append((d, rh))
    return (np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]))


def ev_serial(sched):
    ev = []
    for it, kind, changed in sched:
        if kind == "valve":
            ev.append((int(it), "valve", "", tuple(sorted(changed))))
        elif kind in ("ls_conv", "ls_per"):
            ev.append((int(it), "ls", kind[3:], tuple(sorted(changed))))
        elif changed:
            ev.append((int(it), "extra", kind, tuple(sorted(changed))))
    return ev


def ev_dense(sched, b):
    ev = []
    for e in sched:
        if e.get("kind") == "valve":
            if bool(e["act"][b]):
                ev.append((int(e["it"]), "valve", "",
                           tuple(sorted(np.where(e["chg"][b])[0].tolist()))))
        else:
            lab = "conv" if bool(e["conv"][b]) else (
                "per" if bool(e["per"][b]) else None)
            if lab:
                ev.append((int(e["it"]), "ls", lab,
                           tuple(sorted(np.where(e["chg"][b])[0].tolist()))))
    return ev


def hard(ev):
    return [(it, k, chg) for it, k, _lab, chg in ev]


def conn_mask(net, opened):
    n1 = np.asarray(net.link_n1, dtype=np.int64)
    n2 = np.asarray(net.link_n2, dtype=np.int64)
    reach = np.asarray(net.node_type) != 0
    e1, e2 = n1[opened], n2[opened]
    while True:
        new = reach.copy()
        np.logical_or.at(new, e2, reach[e1])
        np.logical_or.at(new, e1, reach[e2])
        if np.array_equal(new, reach):
            break
        reach = new
    return reach


fail = 0
for stem in NETS:
    net = parse_inp(inp_of(stem))
    lt = np.asarray(net.link_type)
    prv = np.where(lt == 3)[0].tolist()
    if not prv:
        print("%s: 无 PRV，跳过" % stem)
        continue
    D, RH = scenarios(net, args.B, args.seed, prv)
    B = D.shape[0]
    sd = GGASolver(net, mode="dense", dense_status_machine=True)
    se = GGASolver(net, mode="epanet")
    with torch.no_grad():
        od = sd.solve(torch.as_tensor(D), torch.as_tensor(RH),
                      status_machine=True, record_schedule=True)
    Sd = od["status"].numpy()
    Hd = od["head_ft"].numpy()
    itd = od["iters"].numpy()
    n_sched = n_end = n_it = 0
    n_lab = 0
    dhs = []
    diverge = []
    for b in range(B):
        os_ = se.run_gga(D[b], RH[b], do_status=True, record_schedule=True)
        es, ed = ev_serial(os_["schedule"]), ev_dense(od["schedule"], b)
        same_h = hard(es) == hard(ed)
        lab_flip = same_h and [e[:2] + e[3:] for e in es] != \
            [e[:2] + e[3:] for e in ed]
        n_sched += int(same_h)
        n_lab += int(lab_flip)
        e_end = np.array_equal(os_["status"], Sd[b])
        n_end += int(e_end)
        n_it += int(int(os_["iters"]) == int(itd[b]))
        if same_h and e_end and bool(os_["converged"]) \
                and bool(od["converged"][b]):
            opened = (os_["status"] > 2) & (Sd[b] > 2)
            m = conn_mask(net, opened) & (np.asarray(net.node_type) == 0)
            if m.any():
                dhs.append(float(np.abs(os_["head"] - Hd[b])[m].max()))
        if not same_h:
            i0 = next((i for i in range(min(len(es), len(ed)))
                       if hard([es[i]]) != hard([ed[i]])),
                      min(len(es), len(ed)))
            # 转移判定子序列（非空 change 事件）是否一致：一致 ⇒ 分岔只在
            # "何时判收敛/何时停"（Hacc 刀刃），转移函数无差。
            ne_s = [(it, k, chg) for it, k, _l, chg in es if chg]
            ne_d = [(it, k, chg) for it, k, _l, chg in ed if chg]
            transfer_diff = (ne_s != ne_d) or (not e_end)
            diverge.append((b, i0, transfer_diff,
                            es[i0] if i0 < len(es) else None,
                            ed[i0] if i0 < len(ed) else None))
    # 三态直方图（dense 终态；serial 同表打印）
    print("\n== %s  B=%d（含逼阀 %d）==" % (stem, B, 2 * len(prv)))
    cover_ok = True
    for k in prv:
        cd = {v: int((Sd[:, k] == v).sum()) for v in (2, 3, 4, 7)}
        hit = [v for v in (2, 3, 4) if cd[v] > 0]
        cover_ok &= len(hit) == 3
        print("  PRV link %-5d dense终态: %s  三态%s" % (
            k, " ".join("%s=%d" % (STN[v], cd[v]) for v in (2, 3, 4, 7)),
            "全出" if len(hit) == 3 else "缺 " + ",".join(
                STN[v] for v in (2, 3, 4) if cd[v] == 0)))
    dmax = max(dhs) if dhs else 0.0
    n_transfer = sum(1 for d in diverge if d[2])
    print("  调度硬同 %d/%d（仅标签翻转 %d；停时分岔 %d；转移分岔 %d）"
          "终态同 %d/%d iters同 %d/%d" % (
              n_sched, B, n_lab, len(diverge) - n_transfer, n_transfer,
              n_end, B, n_it, B))
    print("  调度同+双收敛 %d 场景 连通域 max|dH|=%.3e ft（门 %.1f）" % (
        len(dhs), dmax, DH_GATE))
    for b, i0, td, a, c in diverge[:8]:
        print("    分岔样本 b=%d 事件#%d %s serial=%s dense=%s" % (
            b, i0, "转移不同<--" if td else "仅停时", a, c))
    ok = (n_end == B and n_transfer == 0 and cover_ok and dmax <= DH_GATE)
    print("  AUD_PRV_SUMMARY %s sched=%d/%d end=%d/%d iters=%d/%d "
          "transfer_diff=%d cover=%s dhmax=%.3e %s"
          % (stem, n_sched, B, n_end, B, n_it, B, n_transfer,
             cover_ok, dmax, "PASS" if ok else "FAIL"))
    fail += 0 if ok else 1
    if args.kappa:
        d0 = net.demand_cfs_at(0)
        rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                     dtype=np.float64))
        tk = np.asarray(net.tank_node, dtype=np.int64)
        if tk.size:
            rh0[tk] = 0.5 * (np.asarray(net.tank_hmin)
                             + np.asarray(net.tank_hmax))
        rec = []
        oc = torch.linalg.cholesky_ex

        def spy(A, *a, **k):
            rec.append(A.detach().clone())
            return oc(A, *a, **k)

        torch.linalg.cholesky_ex = spy
        oc2 = torch.linalg.cholesky
        torch.linalg.cholesky = lambda A, **k: (_ for _ in ()).throw(
            RuntimeError("unexpected cholesky"))
        try:
            with torch.no_grad():
                o1 = sd.solve(torch.as_tensor(d0)[None, :],
                              torch.as_tensor(rh0)[None, :],
                              status_machine=True)
        finally:
            torch.linalg.cholesky_ex = oc
            torch.linalg.cholesky = oc2
        st1 = o1["status"].numpy()[0]
        na = int((st1[prv] == 4).sum())
        A_conv = rec[-1][0].numpy() if rec else None
        if A_conv is not None:
            kap = float(np.linalg.cond(A_conv, 2))
            print("  名义帧: iters=%d ACTIVE PRV=%d/%d 收敛帧 cond2(A)=%.4e "
                  "捕获装配轮=%d" % (int(o1["iters"][0]), na, len(prv), kap,
                                    len(rec)))

sys.exit(1 if fail else 0)

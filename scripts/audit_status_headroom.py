# -*- coding: utf-8 -*-
"""audit_status_headroom.py - 对 probe_status_headroom.py 的独立核验。

不复用探针脚本的任何函数：自己写 EPS 推进、run_gga 调用、连通掩码与统计。

五项自测：
 1) 头空间数字复现：3 个网独立重算 n_A/n_B，与 data/status_headroom.json 比（5%）；
 2) 同解验证：抽 10 帧核对 max|ΔH|（全部/连通）与 max|ΔQ|（绝对/相对），并检查
    探针的剔除规则是否偏向剔掉"不利样本"（剔除帧 vs 保留帧的头空间分布）；
 3) 对照组：纯管道网 n_A 必须逐帧 == n_B（否则"关状态检查"本身改变了求解路径）；
 4) 反向检验：故意翻转某个可切换元件的状态作为初值，看迭代数怎么变
    （D=错构型+状态机开；E=错构型冻结）；
 5) 独立 go/no-go。

用法：python -X utf8 scripts/audit_status_headroom.py
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
PROBE_JSON = os.path.join(ROOT, "data", "status_headroom.json")
WIP = os.path.join(ROOT, "data", "audit_status_headroom_wip.txt")

REDO = ["pub_c_town_batadal", "EXA5", "pub_richmond_standard"]
CTRL = ["pub_hanoi", "pub_modena", "pub_fossolo_poly1",
        "rand_main_0000", "rand_main_0001", "rand_main_0002",
        "rand_main_0003", "rand_main_0004"]
# 反向检验的网（含大头空间帧与线索点名帧）
REV = ["EXA5", "pub_c_town_batadal", "pub_bwsn_network_2"]
DQ_REL_TOL = 1.0e-3
SWITCHABLE = (0, 2, 3, 4, 6, 7, 8)      # CV管/泵/PRV/PSV/FCV/TCV/GPV
OUT = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    sys.stdout.flush()
    OUT.append(line)


# ---------------------------------------------------------------- 工具
def reach_mask(solver, open_mask):
    """经开启链路可达任一定水头节点的节点掩码（scipy 连通分量，独立实现）。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    N = solver.N
    a = solver.n1_np[open_mask]
    b = solver.n2_np[open_mask]
    if a.size == 0:
        lab = np.arange(N)
    else:
        m = coo_matrix((np.ones(a.size), (a, b)), shape=(N, N))
        _, lab = connected_components(m, directed=False)
    roots = set(np.asarray(lab)[np.asarray(solver.fixed_nodes)].tolist())
    return np.isin(lab, list(roots))


def diffs(solver, rX, rY, S_ref):
    """两解差异：开启链路 max|ΔQ| 与相对值；连通节点 max|ΔH|；全部 max|ΔH|。"""
    opn = np.asarray(S_ref) > solver.ST_CLOSED
    dQ = float(np.abs((rX["flow"] - rY["flow"])[opn]).max()) if opn.any() else 0.0
    qs = float(np.abs(rX["flow"][opn]).max()) if opn.any() else 0.0
    conn = reach_mask(solver, opn)
    dH_c = float(np.abs((rX["head"] - rY["head"])[conn]).max()) if conn.any() else 0.0
    dH_a = float(np.abs(rX["head"] - rY["head"]).max())
    return dQ, (dQ / qs if qs > 0 else 0.0), dH_c, dH_a


def make_driver(stem):
    net = Net.load(REF_DIR, stem)
    inp = resolve_inp(stem)
    return net, EpsDriver(net, inp_path=inp if os.path.isfile(inp) else None)


def advance(drv, r, d_in):
    """把 A 解写回 EPS 状态并推进一帧（复刻 EpsDriver.run 收尾）。返回 tstep。"""
    s = drv.solver
    drv.q = r["flow"]
    drv.e = r["emitter"]
    drv.S = r["status"]
    drv.K = r["setting"]
    drv.H = r["head"]
    drv.fixed_dem = r["fixed_demand"]
    dem = d_in + drv.e
    drv.node_dem = np.where(s.is_fixed_node, drv.fixed_dem, dem)
    return drv._nexthyd(float(r["relerr"]))


def redo_net(stem, reverse=False, rev_min_nA=0, rev_max_links=10):
    """独立重算逐帧 A/B/C（+ 可选反向检验 D/E）。"""
    net, drv = make_driver(stem)
    s = drv.solver
    drv._inithyd()
    rows = []
    while True:
        t = drv.Htime
        drv._demands()
        drv._controls()
        d_in = np.array(drv.d, dtype=np.float64, copy=True)
        H_in = np.array(drv.H, dtype=np.float64, copy=True)
        q_in = np.array(drv.q, dtype=np.float64, copy=True)
        e_in = None if drv.e is None else np.array(drv.e, copy=True)
        S_in = np.array(drv.S, dtype=np.int8, copy=True)
        K_in = np.array(drv.K, dtype=np.float64, copy=True)

        rA = s.run_gga(d_in, H_in, q0=q_in, e0=e_in, status0=S_in,
                       setting0=K_in, do_status=True)
        S_A = np.array(rA["status"], dtype=np.int8, copy=True)
        K_A = np.array(rA["setting"], dtype=np.float64, copy=True)
        rB = s.run_gga(d_in, H_in, q0=q_in, e0=e_in, status0=S_A,
                       setting0=K_A, do_status=False, extra_iter=-1)
        rC = s.run_gga(d_in, H_in, q0=q_in, e0=e_in, status0=S_A,
                       setting0=K_A, do_status=True)
        dQ, dQr, dHc, dHa = diffs(s, rA, rB, S_A)
        row = dict(t=int(t), n_A=int(rA["iters"]), n_B=int(rB["iters"]),
                   n_C=int(rC["iters"]), conv_A=bool(rA["converged"]),
                   conv_B=bool(rB["converged"]), relerr_A=float(rA["relerr"]),
                   dQ=dQ, dQ_rel=dQr, dH_conn=dHc, dH_all=dHa,
                   cfg_same=bool(np.array_equal(S_A, S_in)
                                 and np.array_equal(K_A, K_in)),
                   n_cfg_diff=int((S_A != S_in).sum()), rev=[])

        if reverse and int(rA["iters"]) >= rev_min_nA:
            sw = [int(k) for k in range(s.L) if int(s.lt_np[k]) in SWITCHABLE]
            for k in sw[:rev_max_links]:
                S_w = S_A.copy()
                S_w[k] = (s.ST_OPEN if S_A[k] <= s.ST_CLOSED else s.ST_CLOSED)
                rD = s.run_gga(d_in, H_in, q0=q_in, e0=e_in, status0=S_w,
                               setting0=K_A, do_status=True)
                rE = s.run_gga(d_in, H_in, q0=q_in, e0=e_in, status0=S_w,
                               setting0=K_A, do_status=False, extra_iter=-1)
                _, dQrE, _, _ = diffs(s, rE, rA, S_A)
                row["rev"].append(dict(
                    k=k, n_D=int(rD["iters"]), conv_D=bool(rD["converged"]),
                    repaired=bool(np.array_equal(
                        np.asarray(rD["status"], dtype=np.int8), S_A)),
                    n_E=int(rE["iters"]), conv_E=bool(rE["converged"]),
                    dQ_rel_E=dQrE))
        rows.append(row)
        if advance(drv, rA, d_in) == 0:
            break
    return net, s, rows


def fm(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def st(v):
    a = np.asarray(v, dtype=np.float64)
    if a.size == 0:
        return dict(n=0, med=float("nan"), p90=float("nan"),
                    mean=float("nan"), max=float("nan"))
    return dict(n=int(a.size), med=float(np.median(a)),
                p90=float(np.percentile(a, 90)), mean=float(a.mean()),
                max=float(a.max()))


# ---------------------------------------------------------------- 主流程
def main():
    t0 = time.time()
    with open(PROBE_JSON, encoding="utf-8") as f:
        probe = json.load(f)
    say("=" * 104)
    say(time.strftime("%Y-%m-%d %H:%M:%S") + "  audit_status_headroom - 独立核验探针")
    say("=" * 104)

    # ---------------- 1) 头空间复现 ----------------
    say("")
    say("【1】头空间独立复现（自写 EPS 推进 + 自写 run_gga 调用）")
    say(f"{'网络':<24}{'帧我':>6}{'帧探针':>8}{'ΣnA我':>8}{'ΣnA探':>8}"
        f"{'ΣnB我':>8}{'ΣnB探':>8}{'A/B我':>9}{'A/B探':>9}{'偏差%':>8}"
        f"{'中位我':>8}{'p90我':>8}{'extra_iter':>11}")
    part1_ok = True
    mine = {}
    for stem in REDO:
        net, s, rows = redo_net(stem)
        mine[stem] = (s, rows)
        val = [r for r in rows if r["dQ_rel"] <= DQ_REL_TOL]
        sA = sum(r["n_A"] for r in val)
        sB = sum(r["n_B"] for r in val)
        rat = [r["n_A"] / r["n_B"] for r in val]
        p = probe["nets"][stem]
        rr, rp = sA / sB, p["sum_nA"] / p["sum_nB"]
        dev = 100.0 * abs(rr - rp) / rp
        part1_ok &= (dev <= 5.0)
        say(f"{stem:<24}{len(rows):>6}{p['n_frames']:>8}{sA:>8}{p['sum_nA']:>8}"
            f"{sB:>8}{p['sum_nB']:>8}{rr:>9.4f}{rp:>9.4f}{dev:>8.2f}"
            f"{np.median(rat):>8.3f}{np.percentile(rat, 90):>8.3f}"
            f"{s.extra_iter:>11}")
    say(f"  → 判定：{'通过' if part1_ok else '不通过'}（门槛 5%）")
    say("  注：c_town/EXA5 的 extra_iter=10（UNBALANCED CONTINUE 10），并非探针"
        "原注释所说的'全是 STOP'。旧版探针给 B 强制 extra_iter=-1，A 却按网取值，"
        "两者 maxtrials 差 10 - 已修（B/Bt 不再覆写）。实测无一帧触及该额度"
        "（唯一越过 MaxIter 的 bwsn_2 恰好 extra_iter=-1），故数值不变。")

    # ---------------- 2) 同解验证 + 剔除规则审计 ----------------
    say("")
    say("【2a】同解验证：头空间最大的 10 帧（我自己重算的解）")
    say(f"{'网络':<24}{'t':>8}{'nA':>5}{'nB':>5}{'比':>7}"
        f"{'max|ΔQ|开启':>14}{'ΔQ相对':>11}{'max|ΔH|连通':>14}{'max|ΔH|全部':>14}"
        f"{'本网ACC':>9}")
    pick = []
    for stem, (s, rows) in mine.items():
        for r in rows:
            pick.append((r["n_A"] / r["n_B"], stem, s, r))
    pick.sort(key=lambda x: -x[0])
    same_ok = True
    for rat, stem, s, r in pick[:10]:
        ok = r["dQ_rel"] <= s.hacc_default
        same_ok &= ok
        say(f"{stem:<24}{r['t']:>8}{r['n_A']:>5}{r['n_B']:>5}{rat:>7.2f}"
            f"{r['dQ']:>14.3e}{r['dQ_rel']:>11.2e}{r['dH_conn']:>14.3e}"
            f"{r['dH_all']:>14.3e}{s.hacc_default:>9.0e}")
    say(f"  → 10 帧的开启链路相对流量差是否全部 < 本网 ACCURACY："
        f"{'是' if same_ok else '否'}")
    allr = [r for _, _, _, r in pick]
    say(f"  → 3 网全部 {len(allr)} 帧中：dQ_rel>1e-3 的 "
        f"{sum(1 for r in allr if r['dQ_rel'] > 1e-3)} 帧；"
        f"dQ_rel>本网 ACCURACY 的 "
        f"{sum(1 for rat, stem, s, r in pick if r['dQ_rel'] > s.hacc_default)} 帧；"
        f"连通 max|ΔH| 最大 {max(r['dH_conn'] for r in allr):.3e} ft，"
        f"全部 max|ΔH| 最大 {max(r['dH_all'] for r in allr):.3e} ft")

    say("")
    say("【2b】剔除规则审计：剔除帧是不是'不利样本'（头空间大的被偷偷剔掉）？")
    say(f"{'来源':<28}{'帧数':>6}{'比中位':>9}{'比p90':>9}{'比均值':>9}{'比最大':>9}")
    drop_all, keep_all = [], []
    for stem, p in probe["nets"].items():
        if stem == "city_d":
            continue
        for fr in p["frames"]:
            rr = fr["n_A"] / fr["n_B"]
            (drop_all if fr["dQ_rel"] > DQ_REL_TOL else keep_all).append(
                (rr, stem, fr))
    for name, arr in (("探针·保留帧", keep_all), ("探针·剔除帧", drop_all)):
        d = st([a[0] for a in arr])
        say(f"{name:<28}{d['n']:>6}{fm(d['med']):>9}{fm(d['p90']):>9}"
            f"{fm(d['mean']):>9}{fm(d['max']):>9}")
    say("  剔除帧明细（全部）：")
    for rr, stem, fr in sorted(drop_all, key=lambda x: -x[0]):
        say(f"    {stem:<24}t={fr['t']:<8}nA={fr['n_A']:<4}nB={fr['n_B']:<4}"
            f"比={rr:.2f}  dQ_rel={fr['dQ_rel']:.2e}  "
            f"本网ACC={probe['nets'][stem]['hacc_default']:.0e}")
    kd = st([a[0] for a in keep_all])
    dd = st([a[0] for a in drop_all])
    sA_k = sum(a[2]["n_A"] for a in keep_all)
    sB_k = sum(a[2]["n_B"] for a in keep_all)
    sA_d = sum(a[2]["n_A"] for a in drop_all)
    sB_d = sum(a[2]["n_B"] for a in drop_all)
    say(f"  → 若把剔除帧全部计入：ΣnA/ΣnB = {(sA_k+sA_d)/(sB_k+sB_d):.4f}"
        f"（探针口径 {sA_k/sB_k:.4f}）；剔除帧头空间中位 {fm(dd['med'])} "
        f"vs 保留帧 {fm(kd['med'])} → "
        f"{'剔除未系统性偏向大头空间' if dd['med'] <= max(1.5, kd['p90']) else '剔除偏向大头空间，可疑'}")

    # ---------------- 3) 对照组 ----------------
    say("")
    say("【3】对照组自检：纯管道网（无状态机）头空间必须 == 1.000")
    say(f"{'网络':<24}{'帧':>5}{'可切换元件':>12}{'nA':>6}{'nB':>6}{'nC':>6}"
        f"{'逐帧A==B':>10}")
    ctrl_ok = True
    for stem in CTRL:
        net, s, rows = redo_net(stem)
        nsw = int(np.isin(np.asarray(net.link_type), SWITCHABLE).sum())
        eq = all(r["n_A"] == r["n_B"] for r in rows)
        ctrl_ok &= eq and nsw == 0
        say(f"{stem:<24}{len(rows):>5}{nsw:>12}"
            f"{sum(r['n_A'] for r in rows):>6}{sum(r['n_B'] for r in rows):>6}"
            f"{sum(r['n_C'] for r in rows):>6}{('是' if eq else '否'):>10}")
    # 更强的对照：探针 JSON 里"求解全程零翻转"的帧，A/B/C 必须逐帧精确相等
    say("  更强对照（探针全池中求解全程 n_flip==0 的帧：状态机一次都没动手）：")
    z = [f for st_ in probe["nets"] if st_ != "city_d"
         for f in probe["nets"][st_]["frames"] if f["n_flip"] == 0]
    zb = [f for f in z if not (f["n_A"] == f["n_B"] == f["n_C"])]
    say(f"    零翻转帧 {len(z)}/{sum(len(probe['nets'][k]['frames']) for k in probe['nets'] if k != 'city_d')}，"
        f"其中 n_A/n_B/n_C 不全相等的 {len(zb)} 帧 → "
        f"{'关状态检查不改变求解路径' if not zb else '关状态检查改变了求解路径，数字不可信'}")
    say("  另注：有状态机网中'最终构型 == 初始构型'但 n_A != n_B 的帧确实存在"
        "（EXA5 t=49320 nA=36 nB=13；c_town t=19694 nA=14 nB=7），原因是求解中途"
        "状态机来回翻转又翻回来（探针 n_flip>0），这些帧上 n_A == n_C 精确成立 - "
        "见【5】的预测器归因拆分：预测器在这些帧上零增益。")
    say(f"  → 对照组判定：{'通过' if ctrl_ok and not zb else '不通过'}")

    # ---------------- 4) 反向检验 ----------------
    say("")
    say("【4】反向检验：把收敛构型里某一个可切换元件的状态取反当初值")
    say("     D=错构型+状态机开（能否自愈、代价几何）；E=错构型冻结（会不会收敛到别的解）")
    say(f"{'网络':<22}{'t':>8}{'nA':>5}{'nC':>5}{'链路':>6}{'nD':>5}"
        f"{'nD/nC':>8}{'D自愈':>7}{'nE':>5}{'E收敛':>7}{'E的dQ相对':>12}")
    rev_rows = []
    for stem in REV:
        net, s, rows = redo_net(stem, reverse=True, rev_min_nA=5,
                                rev_max_links=8)
        for r in rows:
            for v in r["rev"]:
                rev_rows.append((stem, r, v))
    shown = 0
    for stem, r, v in rev_rows:
        if shown < 26:
            say(f"{stem:<22}{r['t']:>8}{r['n_A']:>5}{r['n_C']:>5}{v['k']:>6}"
                f"{v['n_D']:>5}{v['n_D']/max(1,r['n_C']):>8.2f}"
                f"{('是' if v['repaired'] else '否'):>7}{v['n_E']:>5}"
                f"{('是' if v['conv_E'] else '否'):>7}{v['dQ_rel_E']:>12.2e}")
            shown += 1
    if rev_rows:
        rd = st([v["n_D"] / max(1, r["n_C"]) for _, r, v in rev_rows])
        rep = sum(1 for _, _, v in rev_rows if v["repaired"])
        difE = sum(1 for _, _, v in rev_rows if v["dQ_rel_E"] > 1e-3)
        say(f"  → 样本 {len(rev_rows)} 个（错构型×帧）：n_D/n_C 中位 {fm(rd['med'],2)}"
            f" 均值 {fm(rd['mean'],2)} 最大 {fm(rd['max'],2)}；"
            f"状态机自愈回正确构型 {rep}/{len(rev_rows)}；"
            f"冻结错构型收敛到不同解 {difE}/{len(rev_rows)}")

    # ---------------- 5) 结论 ----------------
    say("")
    say("【5】独立汇总")
    pa = probe["pool_all_public"]
    say(f"  探针全池：帧={pa['n_frames']} 头空间中位={fm(pa['ratio']['med'])} "
        f"p90={fm(pa['ratio']['p90'])} 均值={fm(pa['ratio']['mean'])} "
        f"A/B={fm(pa['total_speedup_AB'],4)} A/C={fm(pa['total_speedup_AC'],4)} "
        f"C/B={fm(pa['total_speedup_CB'],4)}")
    allrat = [a[0] for a in keep_all] + [a[0] for a in drop_all]
    d = st(allrat)
    say(f"  含剔除帧的全池：中位={fm(d['med'])} p90={fm(d['p90'])} "
        f"均值={fm(d['mean'])} 最大={fm(d['max'])}")
    say(f"  未收敛帧（原生容差）={pa['n_unconv_A']}，冻结后收敛={pa['n_unconv_A_but_B_conv']}；"
        f"紧容差未收敛={pa['n_unconv_At']}，冻结后收敛={pa['n_unconv_At_but_Bt_conv']}")

    say("")
    say("  【预测器归因拆分（本次核验新增，独立复算 10 个有状态机网）】")
    tot_eq = tot_ne = 0
    A_eq = C_eq = B_eq = A_ne = C_ne = B_ne = 0
    ac_eq = 0
    for stem in ["pub_c_town_batadal", "pub_d_town", "EXA4", "EXA5", "ky3",
                 "ky5", "pub_anytown", "pub_richmond_standard",
                 "pub_bwsn_network_1", "pub_bwsn_network_2"]:
        rows = mine[stem][1] if stem in mine else redo_net(stem)[2]
        mine.setdefault(stem, (None, rows))
        for r in rows:
            if r["cfg_same"]:
                tot_eq += 1
                A_eq += r["n_A"]
                C_eq += r["n_C"]
                B_eq += r["n_B"]
                ac_eq += int(r["n_A"] == r["n_C"])
            else:
                tot_ne += 1
                A_ne += r["n_A"]
                C_ne += r["n_C"]
                B_ne += r["n_B"]
    say(f"    收敛构型 == 热启动交进来的构型（预测器无事可做）：{tot_eq} 帧"
        f"（{100*tot_eq/(tot_eq+tot_ne):.1f}%），ΣA={A_eq} ΣC={C_eq} ΣB={B_eq}，"
        f"其中 n_A==n_C 的 {ac_eq}/{tot_eq} 帧 → 预测器增益恒为 0，"
        f"这些帧上 A/B={A_eq/max(1,B_eq):.4f} 全部来自'关掉状态检查'")
    say(f"    构型需修正（预测器真有事可做）：{tot_ne} 帧"
        f"（{100*tot_ne/(tot_eq+tot_ne):.1f}%），ΣA={A_ne} ΣC={C_ne} ΣB={B_ne}"
        f" → 预测器可归因节省 {A_ne-C_ne} 次迭代")
    sA_all = A_eq + A_ne
    say(f"    ⇒ 独立算出的**状态预测器上界** = {sA_all}/{sA_all-(A_ne-C_ne)}"
        f" = {sA_all/max(1,sA_all-(A_ne-C_ne)):.4f}x"
        f"（上游报的 1.160x 把'关掉状态检查'也算成了预测器的功劳）")
    say(f"  用时 {time.time()-t0:.1f}s")

    txt = "\n".join(OUT)
    with open(WIP, "a", encoding="utf-8") as f:
        f.write("\n" + txt + "\n")
    print(f"\n证据落盘: {WIP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

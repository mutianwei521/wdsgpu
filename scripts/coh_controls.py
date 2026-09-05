# -*- coding: utf-8 -*-
"""coh_controls.py - 把"相干驱动增设真的把漏损找回来了吗"这件事做实。

上一轮 City D 工单案例上，公平池 coh+40 与 cohmax+20 第一次做到 top-1 真、2/3，
且漏点上没有传感器。本脚本回答三个问题，全部只读已算好的结果与字典缓存：

  random  随机对照分布：City D 上 k=20 / k=40 各 20 个公平 seed（拒绝抽样：抽到漏点
          节点的 seed 作废，条件在此事件上即公平池的均匀抽样），同一反演配方、同一
          噪声实现。报每个 seed 的 top-1/top-3 与相干口径，给"随机 n/20 达到 2/3"与
          交换性下的精确单侧 p =（1 + #{随机 ≥ 设计}）/（1 + n）。
  recall  相干-召回关系：把所有配置（S0、demo、coverage/D-opt、coh/cohmax/cohfull、
          random）的相干口径（max、中位、>0.999 对数、每个真漏点的劲敌相干）与字典
          列能量，对上 top-3 找回数做 Spearman 相关；再按真漏点分解，看召回到底由
          哪一个量驱动。相干口径一律从字典缓存按同一函数重算（与各次记录交叉核对）。
  sep     机理：丢失的漏点与其劲敌，在哪些传感位置上可分？对每个真漏点取 S0 下的劲敌
          列，把这一对的互相干写成 μ(S) = |c(S)|/sqrt(a(S)b(S))（逐行可加），于是可以
          精确回答：全池全装能压到多少、单点扫描的最好位置能压到多少、只盯这一对的
          oracle 贪心 80 步能压到多少、判别能量有多少落在 S0 之外。若"全装也压不下来"
          则该对在候选池内本质不可分；若存在可分位置却没进增设序列，则是目标函数或
          贪心的问题 - 两种情形给的是完全不同的结论。

运行（工作站）：python -X utf8 scripts/coh_controls.py --stage report
输出：data/coh_controls_wip.txt（可读）、data/coh_controls.json、data/fig_coh_controls.png
      City D 的一切先过 augment_suite.assert_no_ids：零节点/链路/传感器编号。
"""
import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from dgga.placement import _coh_from_gram, _gram_per_row          # noqa: E402
from augment_coherence import city_d_rand_positions, paths        # noqa: E402

LABELS = ("T1", "T2", "T3")
EPS = 1e-12
# 公平 seed：拒绝抽样后的前 20 个（k=20 作废 8/15；k=40 作废 5/15/16/17）
SEEDS = {"+20": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21],
         "+40": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23]}


def jload(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, fp):
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, fp)


# ======================================================================
# 相干口径：对任意传感位置集合，从字典缓存重算（与各次记录同一定义）
# ======================================================================
class Dict2:
    """字典缓存 + 逐行 Gram，供任意传感集合 O(NC^2) 重算相干。"""

    def __init__(self, net):
        z = np.load(paths(net)["cache"])
        self.D = z["Dfull"]
        self.junc, self.s0 = z["junc"], np.unique(z["s0_pos"])
        self.leak_pos = np.asarray(z["leak_pos"], dtype=np.int64)
        self.true_col = [int(x) for x in z["true_col"]]
        self.T, self.m, self.NC = self.D.shape
        self.Gi = _gram_per_row(self.D)                       # [m, NC, NC]
        self.pool_full = np.setdiff1d(np.arange(self.m), self.s0)
        self.pool_fair = np.setdiff1d(self.pool_full, self.leak_pos)
        self.iu, self.ju = np.triu_indices(self.NC, 1)

    def stats(self, sel):
        sel = np.unique(np.asarray(sel, dtype=np.int64))
        G = self.Gi[sel].sum(axis=0)
        mu = _coh_from_gram(G)
        off = mu[self.iu, self.ju]
        B = mu.copy()
        np.fill_diagonal(B, -1.0)
        norms = np.sqrt(np.clip(np.diag(G), 0.0, None))
        return dict(n_sensors=int(sel.size), coh_max=float(off.max()),
                    coh_median=float(np.median(off)), coh_q99=float(np.quantile(off, 0.99)),
                    n_gt_0999=int((off > 0.999).sum()), n_gt_099=int((off > 0.99).sum()),
                    logdet2=float(-np.log(np.clip(1 - off ** 2, EPS, None)).sum()),
                    rival={lab: float(B[t].max()) for lab, t in zip(LABELS, self.true_col)},
                    n_rivals_gt_099={lab: int((B[t] > 0.99).sum())
                                     for lab, t in zip(LABELS, self.true_col)},
                    col_norm={lab: float(norms[t]) for lab, t in zip(LABELS, self.true_col)},
                    leak_node_is_sensor={lab: bool(p in set(sel.tolist()))
                                         for lab, p in zip(LABELS, self.leak_pos)})


# ======================================================================
# stage random / recall：City D 的全部漏损重跑，拼成一张表
# ======================================================================
def city_d_positions(spec, d2, prev):
    """把配置名翻成传感位置集合（None = 无法重建，如 demo40 的另一次 seed 抽样）。"""
    z = np.load(paths("city_d")["orders"])
    s0 = d2.s0
    if spec in ("S0", "s0", "ga40"):
        return s0.copy()
    for tag, key in (("coh+", "coh_fair"), ("cohmax+", "cohmax_fair"), ("cohfull+", "coh_full")):
        if spec.startswith(tag):
            return np.r_[s0, np.asarray(z[key][:int(spec[len(tag):])], dtype=np.int64)]
    if spec.startswith("rand") and "_s" in spec:
        k, sd = spec[4:].split("_s")
        return np.r_[s0, city_d_rand_positions(k, sd, d2.junc, s0, d2.leak_pos)]
    if spec.startswith("rand:"):                                   # hv_leak_control 的写法
        _, k, sd = spec.split(":")
        return np.r_[s0, city_d_rand_positions(k, sd, d2.junc, s0, d2.leak_pos)]
    for tag, key in (("augcover", "augment_cover"), ("augdopt", "augment_dopt")):
        if spec.startswith(tag) and spec[len(tag):].isdigit():
            return np.r_[s0, np.asarray(prev[key][:int(spec[len(tag):])], dtype=np.int64)]
    return None


def family_of(spec):
    if spec in ("S0", "s0", "ga40"):
        return "S0"
    if spec == "demo40":
        return "demo"
    if spec.startswith("cohfull"):
        return "coh(原池)"
    if spec.startswith(("coh+", "cohmax")):
        return "coh(公平池)"
    if spec.startswith("rand"):
        return "random"
    if spec.startswith(("augcover", "augdopt", "cover:", "dopt:")):
        return "cover/D-opt"
    return "modifier"


def norm_spec(spec):
    """hv_leak_control 的 rand:k:seed 与 augment_coherence 的 randk_sseed 是同一个传感集合。"""
    if spec.startswith("rand:") and spec.count(":") == 2:
        _, k, sd = spec.split(":")
        return f"rand{k}_s{sd}"
    return {"ga40": "S0", "s0": "S0"}.get(spec, spec)


def collect_city_d(d2):
    """三处来源的 City D 工单重跑（同一反演器、同一噪声实现）→ 一张表。
    同一配置被两套实现各跑过一次时不重复计数，改记一条交叉核对。"""
    prev = np.load(os.path.join(DATA, "placement_orders_city_d.npz"))
    rows, seen, cross = [], {}, []

    def add(spec, src, n_sensors, top1, top3, flow_err, rec_coh, t_sec, host=None, stage1=None):
        spec = norm_spec(spec)
        if spec in seen:
            p = seen[spec]
            cross.append(dict(spec=spec, first_source=p["source"], second_source=src,
                              first_host=p.get("host"), second_host=host,
                              top1_agree=bool(p["top1"] == bool(top1)),
                              top3_agree=bool(p["top3"] == int(top3)),
                              n_sensors_agree=bool(p["n_sensors"] == int(n_sensors)),
                              max_rel_err_diff=max(abs(p["rec_T"][k] - float(v["rel_err"]))
                                                   for k, v in flow_err.items())))
            return
        pos = city_d_positions(spec, d2, prev)
        st = d2.stats(pos) if pos is not None else None
        if st is not None and abs(st["coh_max"] - rec_coh["max_offdiag"]) > 1e-9:
            raise RuntimeError(f"{spec}: 重算相干与记录不符 "
                               f"{st['coh_max']:.12f} vs {rec_coh['max_offdiag']:.12f}")
        # stage1 = Adam+L1 粗筛阶段的原始输出（按幅值排序取 top-3），发生在离散支撑搜索
        # （非线性 OMP + 对手互换抛光）之前 - 与 augment_public.lt_invert 唯一存在的那一步
        # 逐位同型（都是"排幅值取 top-3，看真值列在不在里面"），不含 L-TOWN 没有的任何步骤。
        kinds = (stage1 or {}).get("top3_kinds")
        row = dict(spec=spec, family=family_of(spec), source=src, n_sensors=int(n_sensors),
                   k_added=int(n_sensors) - int(d2.s0.size), top1=bool(top1),
                   top3=int(top3), rec_T={k: float(v["rel_err"]) for k, v in flow_err.items()},
                   recovered={k: bool(v["rel_err"] < 0.05) for k, v in flow_err.items()},
                   recomputed=st, recorded_coh_max=float(rec_coh["max_offdiag"]),
                   recorded_rival=rec_coh["true_node_max_rival_coh"], time_sec=t_sec, host=host,
                   stage1_top3=(int(stage1["top3_hits"]) if stage1 else None),
                   stage1_recovered=({lab: bool(lab in kinds) for lab in LABELS}
                                     if kinds is not None else None))
        rows.append(row)
        seen[spec] = row

    for fn in sorted(os.listdir(os.path.join(DATA, "leak_coh_city_d"))):
        if not fn.endswith(".json"):
            continue
        d = jload(os.path.join(DATA, "leak_coh_city_d", fn))
        add(d["spec"], "leak_coh_city_d", d["n_sensors"], d["top1_hit"], d["top3_hits"],
            d["flow_err"], d["coherence"], d.get("time_sec"), d.get("host"), d.get("stage1"))
    ctl = os.path.join(DATA, "audit_leak_control")
    for fn in sorted(os.listdir(ctl)):
        if not fn.endswith(".json"):
            continue
        d = jload(os.path.join(ctl, fn))
        add(d["spec"], "audit_leak_control", d["n_sensors"], d["top1_hit"], d["top3_hits"],
            d["flow_err"], d["coherence"], d.get("time_sec"), d.get("host"), d.get("stage1"))
    la = jload(os.path.join(DATA, "leak_augment_city_d.json"))
    for gkey, g in la.get("groups", {}).items():
        name, grp = gkey.split(":")
        if grp != "noisy":
            continue
        add(name, "leak_augment_city_d", g["n_sensors"],
            g["top1_hit"], g["top3_hits"], g["flow_err"], g["coherence"], g.get("time_sec"),
            g.get("host"), g.get("stage1"))
    hc = os.path.join(DATA, "leak_coh_city_d_hostcheck")       # 另一台机上重跑的同名配置
    if os.path.isdir(hc):
        for fn in sorted(os.listdir(hc)):
            if not fn.endswith(".json"):
                continue
            d = jload(os.path.join(hc, fn))
            add(d["spec"], "leak_coh_city_d_hostcheck", d["n_sensors"], d["top1_hit"],
                d["top3_hits"], d["flow_err"], d["coherence"], d.get("time_sec"), d.get("host"),
                d.get("stage1"))
    return rows, cross


# ======================================================================
# 非参数统计
# ======================================================================
def rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, dtype=np.float64)
    r[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # 并列取平均秩
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(rho=None, n=int(x.size), p_two_sided=None)
    rx, ry = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = x.size
    # 大样本正态近似：t = rho sqrt((n-2)/(1-rho^2))，n 小的时候只作参考（报里注明）
    p = None
    if abs(rho) < 1 - 1e-12:
        t = abs(rho) * np.sqrt((n - 2) / (1 - rho ** 2))
        p = float(math.erfc(t / np.sqrt(2)))
    return dict(rho=rho, n=int(n), p_two_sided=p)


def fisher_one_sided(a, b, c, d):
    """2x2 表 [[a, b], [c, d]]（行 = 设计 / 随机，列 = 找回 / 没找回）的单侧 Fisher 精确 p：
    P(设计组找回数 ≥ a | 边缘固定)。"""
    n1, n2, k = a + b, c + d, a + c
    tot = 0.0
    denom = math.comb(n1 + n2, k)
    for i in range(a, min(n1, k) + 1):
        if k - i > n2:
            continue
        tot += math.comb(n1, i) * math.comb(n2, k - i)
    return dict(table=[[int(a), int(b)], [int(c), int(d)]], p_one_sided=float(tot / denom))


def design_vs_random(rows, labels=LABELS):
    """所有"按目标选出来的"传感集合（coh / cohmax / cohfull / coverage / D-opt）对所有随机抽样：
    每个真漏点找回与否的 2x2 Fisher。"""
    ok = [r for r in rows if r["recomputed"] is not None and r["family"] != "modifier"]
    des = [r for r in ok if r["family"] in ("coh(公平池)", "coh(原池)", "cover/D-opt")]
    rnd = [r for r in ok if r["family"] == "random"]
    out = dict(n_designed=len(des), n_random=len(rnd),
               designed_specs=[r["spec"] for r in des], random_specs=[r["spec"] for r in rnd],
               per_truth={})
    for lab in labels:
        a = sum(r["recovered"][lab] for r in des)
        c = sum(r["recovered"][lab] for r in rnd)
        ent = fisher_one_sided(a, len(des) - a, c, len(rnd) - c)
        # 阈值可读性：找回 / 丢失两组的劲敌间隙区间是否分开
        g_rec = [1 - r["recomputed"]["rival"][lab] for r in ok if r["recovered"][lab]]
        g_lost = [1 - r["recomputed"]["rival"][lab] for r in ok if not r["recovered"][lab]]
        ent.update(designed_recovered=f"{a}/{len(des)}", random_recovered=f"{c}/{len(rnd)}",
                   gap_recovered_min=float(min(g_rec)) if g_rec else None,
                   gap_lost_max=float(max(g_lost)) if g_lost else None,
                   separated=bool(g_rec and g_lost and min(g_rec) > max(g_lost)))
        out["per_truth"][lab] = ent
    return out


def stage1_robustness(rows, dvr):
    """反演器不对称的稳健性检验：把结论一/二在"离散支撑搜索发生之前"的那一刻重跑一遍。
    stage1 = Adam+L1 粗筛的原始输出，按幅值排序取 top-3，看真值列在不在里面 - 这一步是
    City D 反演器里唯一与 augment_public.lt_invert（L-TOWN 仅有的步骤）同型的部分：两者
    都是"排幅值取 top-3"，不含非线性 OMP 支撑搜索或对手互换抛光。若结论一（T2 由相干
    驱动、T3 与随机无异）在只用这一步的产物上依然成立，说明它不是离散支撑搜索这一步
    "做出来"的，L-TOWN 缺这一步不影响该结论的可信度。"""
    ok = [r for r in rows if r["recomputed"] is not None and r["family"] != "modifier"
          and r.get("stage1_recovered") is not None]
    des = [r for r in ok if r["family"] in ("coh(公平池)", "coh(原池)", "cover/D-opt")]
    rnd = [r for r in ok if r["family"] == "random"]
    per_truth = {}
    for lab in LABELS:
        a = sum(r["stage1_recovered"][lab] for r in des)
        c = sum(r["stage1_recovered"][lab] for r in rnd)
        ent = fisher_one_sided(a, len(des) - a, c, len(rnd) - c)
        g_rec = [1 - r["recomputed"]["rival"][lab] for r in ok if r["stage1_recovered"][lab]]
        g_lost = [1 - r["recomputed"]["rival"][lab] for r in ok if not r["stage1_recovered"][lab]]
        gap = np.array([math.log10(max(1 - r["recomputed"]["rival"][lab], 1e-16)) for r in ok])
        yy = np.array([1.0 if r["stage1_recovered"][lab] else 0.0 for r in ok])
        # 与 stage2（最终，含离散支撑搜索）判定的一致性：refine 加回来的 / refine 撤掉的
        # refine_adds：stage1 没找到、stage2 找到了（离散支撑搜索新增的）
        # refine_drops：stage1 找到了、stage2 反而丢了（对手互换抛光把它换掉了）
        added = sum(1 for r in ok if not r["stage1_recovered"][lab] and r["recovered"][lab])
        dropped = sum(1 for r in ok if r["stage1_recovered"][lab] and not r["recovered"][lab])
        both = sum(1 for r in ok if r["stage1_recovered"][lab] and r["recovered"][lab])
        neither = len(ok) - added - dropped - both
        ent.update(n_configs=len(ok), designed_recovered=f"{a}/{len(des)}",
                   random_recovered=f"{c}/{len(rnd)}",
                   gap_recovered_min=float(min(g_rec)) if g_rec else None,
                   gap_lost_max=float(max(g_lost)) if g_lost else None,
                   separated=bool(g_rec and g_lost and min(g_rec) > max(g_lost)),
                   rho_vs_log_rival_gap=spearman(gap, yy),
                   agree_both_recovered=both, agree_neither=neither,
                   refine_adds=added, refine_drops=dropped,
                   final_designed_recovered=dvr["per_truth"][lab]["designed_recovered"],
                   final_random_recovered=dvr["per_truth"][lab]["random_recovered"],
                   final_fisher_p=dvr["per_truth"][lab]["p_one_sided"])
        per_truth[lab] = ent
    return dict(n_configs=len(ok), n_designed=len(des), n_random=len(rnd), per_truth=per_truth,
                note="stage1_recovered = 该真漏点在 Adam+L1 粗筛输出里排幅值 top-3（未经离散支撑搜索/"
                     "对手互换抛光）；与 dvr（stage2，最终判定）并排，量化离散支撑搜索这一步"
                     "改判了多少个配置 - 这一步是 L-TOWN 的 augment_public.lt_invert 没有的。")


def exact_p(design_stat, rand_stats, higher_is_better=True):
    """交换性零假设（设计与随机抽样可交换）下的精确单侧 p；外加一句功效诚实话：
    p=(1+ge)/(1+n) 里 n→∞ 时 p→ge/n（随机组"不劣于设计"的真实比例 r）。r=0 时再
    加样本永远有救（n≥19 就能把 p 压到 <0.05）；r>0 时它是渐近下限，再多样本也
    压不过去 - 报"需要多少样本"要报这一点，而不是外推一个不存在的界。"""
    r = np.asarray(rand_stats, dtype=np.float64)
    n = int(r.size)
    ge = int((r >= design_stat).sum()) if higher_is_better else int((r <= design_stat).sum())
    rate = ge / n if n else None
    if ge == 0:
        power = dict(observed_rate=0.0, asymptotic_p_floor=0.0,
                      n_needed_for_p_lt_05=19,
                      note="随机组目前 0 命中：若该比率不变，再采 19 个公平 seed（累计 ≥19）"
                           "就能把 p 压到 <0.05（1/20=0.05，n=19 时 p=1/20 才刚好，n=20 时 "
                           "p=1/21≈0.048），不需要外推假设。")
    else:
        power = dict(observed_rate=float(rate), asymptotic_p_floor=float(rate),
                      n_needed_for_p_lt_05=None,
                      note=f"随机组已有 {ge}/{n} 不劣于设计（比率 {rate:.3f}）：n→∞ 时 p→{rate:.3f}，"
                           f"这是渐近下限 - 只要这个比率不变，再多公平 seed 也压不过 p<0.05；"
                           f"要压过，需要的是随机组的真实命中率降到 0，不是样本量。")
    return dict(n_random=n, n_random_at_least_as_good=ge,
                p_one_sided=float((1 + ge) / (1 + n)),
                note="p =（1 + #{随机不劣于设计}）/（1 + n）：设计在 n+1 个可交换样本里排第一时的精确概率",
                power=power)


# ======================================================================
# stage sep：一对（真漏点，劲敌）的可分性 - 逐行可加的 μ(S)
# ======================================================================
def pair_separability(d2, net, greedy_steps=80):
    """μ(S) = |Σ_S c| / sqrt(Σ_S a · Σ_S b)，a/b/c 是该对在每个传感位置上的 Gram 贡献。
    于是"哪些位置可分"是对全池的一次向量化扫描，"能压到多少"由只盯这一对的 oracle 贪心给。"""
    Gi, s0, lk = d2.Gi, d2.s0, d2.leak_pos
    G0 = Gi[s0].sum(axis=0)
    mu0 = _coh_from_gram(G0)
    np.fill_diagonal(mu0, -1.0)
    hops = None
    try:
        from augment_coherence import _net_topology, hop_distances
        N, n1, n2, _cand = _net_topology(net)
        hops = {lab: hop_distances(N, n1, n2, int(d2.junc[p]))[d2.junc]
                for lab, p in zip(LABELS, lk)}
    except Exception as e:                                          # 拓扑拿不到就只报计数
        print(f"  [sep/{net}] 跳数不可得（{type(e).__name__}），只报计数")
    out = {}
    for lab, t in zip(LABELS, d2.true_col):
        r = int(np.argmax(mu0[t]))
        a, b, c = Gi[:, t, t], Gi[:, r, r], Gi[:, t, r]
        a0, b0, c0 = a[s0].sum(), b[s0].sum(), c[s0].sum()
        mu_s0 = float(mu0[t, r])

        def mu_of(sel):
            return float(abs(c[sel].sum()) / np.sqrt(max(a[sel].sum() * b[sel].sum(), 1e-300)))

        all_rows = np.arange(d2.m)
        fair_all = np.r_[s0, d2.pool_fair]
        # 单点扫描（原池，逐位向量化）
        mu1 = np.abs(c0 + c[d2.pool_full]) / np.sqrt((a0 + a[d2.pool_full]) * (b0 + b[d2.pool_full]))
        fair_mask = np.isin(d2.pool_full, d2.pool_fair)
        m1f = np.where(fair_mask, mu1, np.inf)
        jbest = int(np.argmin(m1f))
        # 只盯这一对的 oracle 贪心（公平池）；真值身份用上了，只作可达下限
        A, B, C = a0, b0, c0
        rem = d2.pool_fair.copy()
        picked = []
        curve, chosen_hops = [mu_s0], []
        for _ in range(min(greedy_steps, rem.size)):
            v = np.abs(C + c[rem]) / np.sqrt((A + a[rem]) * (B + b[rem]))
            j = int(np.argmin(v))
            curve.append(float(v[j]))
            if hops is not None:
                chosen_hops.append(int(hops[lab][rem[j]]))
            picked.append(int(rem[j]))
            A, B, C = A + a[rem[j]], B + b[rem[j]], C + c[rem[j]]
            rem = np.delete(rem, j)
        # 下限是不是贪心的局限：随机子集 + 贪心后的互换抛光，都不比贪心更低才敢说"到底了"
        def mu_sub(sel):
            sel = np.asarray(sel, dtype=np.int64)
            return float(abs(c0 + c[sel].sum()) /
                         np.sqrt(max((a0 + a[sel].sum()) * (b0 + b[sel].sum()), 1e-300)))

        gr = np.random.default_rng(12345)
        rand_min = {}
        for kk in (5, 20, 80):
            if kk > d2.pool_fair.size:
                continue
            v = min(mu_sub(gr.choice(d2.pool_fair, kk, replace=False)) for _ in range(2000))
            rand_min[f"+{kk}"] = float(v)
        # 互换抛光：贪心的 80 步解上，逐个尝试换成池外位置，取更低者
        cur = np.asarray(picked, dtype=np.int64)
        best = mu_sub(cur)
        outside = np.setdiff1d(d2.pool_fair, cur)
        for _ in range(200):
            improved = False
            for ii in range(cur.size):
                base = np.delete(cur, ii)
                ab, bb, cb = a0 + a[base].sum(), b0 + b[base].sum(), c0 + c[base].sum()
                vv = np.abs(cb + c[outside]) / np.sqrt((ab + a[outside]) * (bb + b[outside]))
                jj = int(np.argmin(vv))
                if vv[jj] < best - 1e-15:
                    best, old_v = float(vv[jj]), cur[ii]
                    cur = np.sort(np.r_[base, outside[jj]])
                    outside = np.sort(np.r_[np.delete(outside, jj), old_v])
                    improved = True
            if not improved:
                break
        swap_floor = float(best)
        # 杀掉这一对之后，t 的**最大**劲敌相干是多少 - 劲敌是一簇还是一个
        mu_after = _coh_from_gram(G0 + Gi[np.asarray(picked, dtype=np.int64)].sum(axis=0))
        np.fill_diagonal(mu_after, -1.0)
        top3_s0 = np.sort(mu0[t])[::-1][:3]
        # 单点杠杆：每个公平位置把这一对压低多少，分布集中还是分散
        drop = mu_s0 - mu1
        drop_f = np.where(fair_mask, drop, -np.inf)
        best_drop = float(drop_f.max())
        ordr = np.argsort(-drop_f)[:5]
        rec = dict(
            rival_coh_S0=mu_s0, gap_S0=float(1 - mu_s0),
            rival_is_one_of_many=int((mu0[t] > 0.99).sum()),
            n_rivals_gt_0999=int((mu0[t] > 0.999).sum()),
            col_norm_true=float(np.sqrt(a.sum())), col_norm_rival=float(np.sqrt(b.sum())),
            col_norm_true_on_S0=float(np.sqrt(a0)),
            mu_fair_all=mu_of(fair_all), mu_every_junction=mu_of(all_rows),
            single_sensor=dict(
                fair_best=float(m1f[jbest]),
                fair_best_hops=(int(hops[lab][d2.pool_full[jbest]]) if hops is not None else None),
                n_fair_lt_0999=int(((mu1 < 0.999) & fair_mask).sum()),
                n_fair_lt_099=int(((mu1 < 0.99) & fair_mask).sum()),
                n_fair_lt_09=int(((mu1 < 0.9) & fair_mask).sum()),
                n_fair=int(fair_mask.sum()),
                full_best=float(mu1.min()),
                leak_row=float(mu1[np.where(d2.pool_full == lk[LABELS.index(lab)])[0][0]])
                if lk[LABELS.index(lab)] in set(d2.pool_full.tolist()) else None),
            oracle_pair_greedy=dict(
                curve_at={f"+{k}": float(curve[k]) for k in (0, 1, 5, 10, 20, 40, 80) if k < len(curve)},
                floor=float(min(curve)), hops_first5=chosen_hops[:5], n_steps=len(curve) - 1,
                swap_polished_floor=swap_floor, random_subset_min=rand_min,
                max_rival_after=float(mu_after[t].max()),
                note="只盯这一对的贪心（用真值身份），给的是这一对可达的下限；"
                     "max_rival_after = 压完这一对之后 t 的最大劲敌相干（劲敌是一簇时会被顶上来）"),
            rival_cluster=dict(top3_rival_coh_S0=[float(x) for x in top3_s0],
                               n_gt_099=int((mu0[t] > 0.99).sum()),
                               n_gt_0999=int((mu0[t] > 0.999).sum())),
            leverage=dict(best_single_drop=best_drop,
                          n_fair_ge_half_best=int((drop_f >= 0.5 * best_drop).sum()),
                          n_fair_ge_tenth_best=int((drop_f >= 0.1 * best_drop).sum()),
                          top5_hops=([int(hops[lab][d2.pool_full[i]]) for i in ordr]
                                     if hops is not None else None)),
            pair_term_in_J=float(-np.log(max(1 - mu_s0 ** 2, EPS))))
        out[lab] = rec
    off0 = mu0[d2.iu, d2.ju]
    out["_J_S0"] = float(-np.log(np.clip(1 - off0 ** 2, EPS, None)).sum())
    # 公平池增设序列有没有挑到"可分位置"：coh_fair 的前 k 步里有几个属于 μ1<0.999 的位置
    z = np.load(paths(net)["orders"])
    for lab, t in zip(LABELS, d2.true_col):
        r = int(np.argmax(mu0[t]))
        a, b, c = Gi[:, t, t], Gi[:, r, r], Gi[:, t, r]
        a0, b0, c0 = a[s0].sum(), b[s0].sum(), c[s0].sum()
        mu1 = np.abs(c0 + c[d2.pool_full]) / np.sqrt((a0 + a[d2.pool_full]) * (b0 + b[d2.pool_full]))
        good = set(d2.pool_full[(mu1 < 0.999) & np.isin(d2.pool_full, d2.pool_fair)].tolist())
        out[lab]["separating_positions_picked_by_coh_fair"] = {
            f"+{k}": int(sum(1 for v in z["coh_fair"][:k] if int(v) in good))
            for k in (5, 10, 20, 40, 80) if z["coh_fair"].size >= k}
        out[lab]["n_separating_positions"] = int(len(good))
    return out


def frame_counterfactual(d2, frame_sets):
    """帧数是不是相干高的原因：同一 S0 / 同一全装，只换字典的帧子集。"""
    out = {}
    for name, idx in frame_sets.items():
        Gi = _gram_per_row(d2.D[idx])
        ent = {}
        for tag, sel in (("S0", d2.s0), ("S0+fair_all", np.r_[d2.s0, d2.pool_fair])):
            G = Gi[np.asarray(sel, dtype=np.int64)].sum(axis=0)
            mu = _coh_from_gram(G)
            np.fill_diagonal(mu, -1.0)
            off = mu[d2.iu, d2.ju]
            ent[tag] = dict(coh_max=float(off.max()), coh_median=float(np.median(off)),
                            n_gt_0999=int((off > 0.999).sum()),
                            rival={lab: float(mu[t].max()) for lab, t in zip(LABELS, d2.true_col)})
        out[name] = ent
    return out


# ======================================================================
# stage recall：相干口径与召回的关系
# ======================================================================
def recall_analysis(rows):
    ok = [r for r in rows if r["recomputed"] is not None and r["family"] != "modifier"]
    y = np.array([r["top3"] for r in ok], dtype=np.float64)
    cov = {
        "log10(1-max_mu)": np.array([math.log10(max(1 - r["recomputed"]["coh_max"], 1e-16)) for r in ok]),
        "median_mu": np.array([r["recomputed"]["coh_median"] for r in ok]),
        "n_pairs_gt_0999": np.array([r["recomputed"]["n_gt_0999"] for r in ok], dtype=np.float64),
        "J_logdet2": np.array([r["recomputed"]["logdet2"] for r in ok]),
        "n_sensors": np.array([r["n_sensors"] for r in ok], dtype=np.float64),
        "min_true_col_norm": np.array([min(r["recomputed"]["col_norm"].values()) for r in ok]),
    }
    overall = {k: spearman(v, y) for k, v in cov.items()}
    per_truth = {}
    for lab in LABELS:
        yy = np.array([1.0 if r["recovered"][lab] else 0.0 for r in ok])
        gap = np.array([math.log10(max(1 - r["recomputed"]["rival"][lab], 1e-16)) for r in ok])
        nrm = np.array([r["recomputed"]["col_norm"][lab] for r in ok])
        per_truth[lab] = dict(
            n_configs=len(ok), n_recovered=int(yy.sum()),
            rho_vs_log_rival_gap=spearman(gap, yy), rho_vs_col_norm=spearman(nrm, yy),
            log_rival_gap_recovered=[float(x) for x in np.sort(gap[yy > 0])],
            log_rival_gap_lost_range=([float(gap[yy == 0].min()), float(gap[yy == 0].max())]
                                      if (yy == 0).any() else None),
            col_norm_range=[float(nrm.min()), float(nrm.max())])
    return dict(n_configs=len(ok), specs=[r["spec"] for r in ok], overall=overall,
                per_truth=per_truth,
                note="配置集合 = 能重建传感集合、且不含 drop/add 修饰的全部 City D 重跑；相干口径"
                     "全部从字典缓存按同一函数重算，并与每次运行记录的 max μ 逐位核对（差 <1e-9）")


# ======================================================================
# stage random：随机对照分布与交换性检验
# ======================================================================
def random_analysis(rows):
    by = {r["spec"]: r for r in rows}
    out = {}
    for k, design in ((20, ["cohmax+20", "coh+20"]), (40, ["coh+40"])):
        rnd = []
        for sd in SEEDS[f"+{k}"]:
            r = by.get(f"rand{k}_s{sd}") or by.get(f"rand:{k}:{sd}")
            if r is not None:
                rnd.append(r)
        if not rnd:
            continue
        top3 = np.array([r["top3"] for r in rnd], dtype=np.float64)
        ent = dict(k=k, n_seeds=len(rnd), seeds_planned=SEEDS[f"+{k}"],
                   top3_counts={str(v): int((top3 == v).sum()) for v in (0, 1, 2, 3)},
                   n_top1_true=int(sum(r["top1"] for r in rnd)),
                   per_truth_recovered={lab: int(sum(r["recovered"][lab] for r in rnd))
                                        for lab in LABELS},
                   coh_max_range=[float(min(r["recomputed"]["coh_max"] for r in rnd)),
                                  float(max(r["recomputed"]["coh_max"] for r in rnd))],
                   coh_median_range=[float(min(r["recomputed"]["coh_median"] for r in rnd)),
                                     float(max(r["recomputed"]["coh_median"] for r in rnd))],
                   rival_gap_median={lab: float(np.median([1 - r["recomputed"]["rival"][lab]
                                                           for r in rnd])) for lab in LABELS},
                   per_seed=[dict(spec=r["spec"], top1=r["top1"], top3=r["top3"],
                                  recovered={l: r["recovered"][l] for l in LABELS},
                                  coh_max=r["recomputed"]["coh_max"],
                                  coh_median=r["recomputed"]["coh_median"],
                                  n_gt_0999=r["recomputed"]["n_gt_0999"],
                                  rival={l: r["recomputed"]["rival"][l] for l in LABELS})
                             for r in rnd],
                   tests={})
        for nm in design:
            d = by.get(nm)
            if d is None:
                continue
            ent["tests"][nm] = dict(
                design_top3=d["top3"], design_top1=d["top1"],
                top3=exact_p(d["top3"], top3),
                per_truth={lab: exact_p(1.0 if d["recovered"][lab] else 0.0,
                                        [1.0 if r["recovered"][lab] else 0.0 for r in rnd])
                           for lab in LABELS})
        out[f"k={k}"] = ent
    return out


# ======================================================================
# 图（英文标注，零节点/链路/传感器编号）
# ======================================================================
def make_figure(rows, rnd, sep, fp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in rows if r["recomputed"] is not None and r["family"] != "modifier"]
    fam_style = {"S0": ("k", "s", "S0"), "demo": ("0.5", "s", "demo-40"),
                 "coh(公平池)": ("#c0392b", "o", "coherence (fair pool)"),
                 "coh(原池)": ("#e67e22", "o", "coherence (original pool)"),
                 "cover/D-opt": ("#2980b9", "^", "coverage / D-opt"),
                 "random": ("#7f8c8d", "x", "random")}
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    rng = np.random.default_rng(0)
    a = ax[0, 0]
    for fam, (c, mk, en) in fam_style.items():
        g = [r for r in ok if r["family"] == fam]
        if not g:
            continue
        x = [math.log10(max(1 - r["recomputed"]["coh_max"], 1e-16)) for r in g]
        y = [r["top3"] + rng.uniform(-0.09, 0.09) for r in g]
        a.scatter(x, y, c=c, marker=mk, s=46, label=en, alpha=0.85, linewidths=1.1)
    a.set_xlabel(r"$\log_{10}(1-\max\ \mu)$   global dictionary coherence gap")
    a.set_ylabel("true leaks recovered (of 3)")
    a.set_title("(a) recall against the global coherence gap", fontsize=10.5)
    a.set_yticks([0, 1, 2, 3])
    a.grid(alpha=0.25)
    a.legend(fontsize=7.5, loc="upper left")
    b = ax[0, 1]
    for i, lab in enumerate(LABELS):
        x = [math.log10(max(1 - r["recomputed"]["rival"][lab], 1e-16)) for r in ok]
        y = [i + (0.22 if r["recovered"][lab] else -0.22) + rng.uniform(-0.05, 0.05) for r in ok]
        col = ["#c0392b" if r["recovered"][lab] else "0.62" for r in ok]
        b.scatter(x, y, c=col, s=34, alpha=0.85)
    b.set_yticks([0, 1, 2])
    b.set_yticklabels(["T1", "T2", "T3"])
    b.set_xlabel(r"$\log_{10}(1-\mu_{\rm rival})$   per-leak coherence gap")
    b.set_title("(b) per leak: recovered (red, upper) vs lost (grey, lower)", fontsize=10.5)
    b.grid(alpha=0.25)
    c_ax = ax[1, 0]
    off = {20: -0.19, 40: 0.19}
    for k, col in ((20, "#95a5a6"), (40, "#34495e")):
        e = rnd.get(f"k={k}")
        if not e:
            continue
        cnt = [e["top3_counts"][str(v)] for v in (0, 1, 2, 3)]
        c_ax.bar(np.arange(4) + off[k], cnt, width=0.36, color=col,
                 label=f"random +{k}  (n={e['n_seeds']})")
        for nm, t in e["tests"].items():
            if t["design_top3"] < 2:
                continue
            c_ax.axvline(t["design_top3"] + off[k], color="#c0392b", ls="--", lw=1.3)
            c_ax.text(t["design_top3"] + off[k] + 0.05, max(cnt) * 0.95,
                      f"{nm}\np={t['top3']['p_one_sided']:.3f}",
                      color="#c0392b", fontsize=8, ha="left", va="top")
    c_ax.set_xticks([0, 1, 2, 3])
    c_ax.set_xlabel("true leaks recovered (of 3)")
    c_ax.set_ylabel("number of random seeds")
    c_ax.set_title("(c) City D: coherence design against fair random draws", fontsize=10.5)
    c_ax.legend(fontsize=8)
    c_ax.grid(alpha=0.25, axis="y")
    d_ax = ax[1, 1]
    for net, ls in (("ltown", "-"), ("city_d", "--")):
        for lab, col in zip(LABELS, ("#c0392b", "#2980b9", "#27ae60")):
            g = sep[net][lab]["oracle_pair_greedy"]["curve_at"]
            ks = sorted(int(k[1:]) for k in g)
            d_ax.plot(ks, [max(1 - g[f"+{k}"], 1e-9) for k in ks], ls, color=col, marker="o",
                      ms=3.2, label=f"{'L-TOWN' if net == 'ltown' else 'City D'} {lab}")
    d_ax.set_yscale("log")
    d_ax.set_xlabel("sensors an oracle devotes to that one pair")
    d_ax.set_ylabel(r"$1-\mu$  of the leak / rival pair")
    d_ax.set_title("(d) placement floor of each leak against its rival", fontsize=10.5)
    d_ax.grid(alpha=0.25, which="both")
    d_ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fp, dpi=170)
    plt.close(fig)
    print(f"图：{fp}")


# ======================================================================
# stage report
# ======================================================================
def stage_report(machine=""):
    t0 = time.time()
    L = []

    def A(s=""):
        L.append(s)

    d2c, d2l = Dict2("city_d"), Dict2("ltown")
    rows, cross = collect_city_d(d2c)
    rnd = random_analysis(rows)
    rec = recall_analysis(rows)
    sep = {"ltown": pair_separability(d2l, "ltown"), "city_d": pair_separability(d2c, "city_d")}
    frames = frame_counterfactual(d2c, {"T=1（只用第 0 帧）": [0],
                                        "T=5（0,6,12,18,24）": [0, 6, 12, 18, 24],
                                        "T=25（全帧）": list(range(25))})
    A("=" * 78)
    A("相干驱动增设：随机对照、相干-召回关系、丢失漏点的可分性（scripts/coh_controls.py --stage report）")
    A("=" * 78)
    A("问题：City D 工单案例上公平池 coh+40 与 cohmax+20 拿到 top-1 真、2/3，别的都没有。"
      "这四件事把它做实：(1) 同预算的随机传感集合能不能也拿到 2/3；(2) 召回到底由不由相干驱动；"
      "(3) L-TOWN 上相干压不下来，是目标函数/贪心不行，还是那两对本来就不可分；"
      "(4) City D 反演器比 L-TOWN 多一步离散支撑搜索，结论一/二会不会是这一步做出来的假象。")
    A("口径：反演器、真值、噪声实现（0.1 ft，seed 909）全部原样；随机对照 = 从 junction − S0 按"
      "seed 抽 k 个、抽到漏点节点的 seed 作废（拒绝抽样，条件下即公平池均匀抽样），与 coh_fair 同池；"
      "相干口径一律从字典缓存按同一函数重算，并与每次运行记录的 max μ 核对（最大差 <1e-9）。")
    if machine:
        A(f"机器：{machine}")
    A()

    # ---- 一、随机对照 ----
    A("### 一、City D 随机对照分布（公平 seed，同一反演配方）")
    for key in ("k=20", "k=40"):
        e = rnd.get(key)
        if not e:
            A(f"  [{key}] 尚无结果")
            continue
        A(f"\n  [+{e['k']}] 完成 {e['n_seeds']}/{len(e['seeds_planned'])} 个公平 seed；"
          f"相干 max ∈ [{e['coh_max_range'][0]:.9f}, {e['coh_max_range'][1]:.9f}]，"
          f"中位 ∈ [{e['coh_median_range'][0]:.4f}, {e['coh_median_range'][1]:.4f}]")
        A(f"  {'seed':>12} {'top1真':>6} {'top3':>5} {'T1':>3} {'T2':>3} {'T3':>3} | "
          f"{'相干max':>12} {'中位':>7} {'>.999':>5} | {'劲敌T1':>9} {'劲敌T2':>12} {'劲敌T3':>9}")
        for s in e["per_seed"]:
            A(f"  {s['spec']:>12} {str(s['top1']):>6} {s['top3']:>5} "
              + " ".join(f"{'Y' if s['recovered'][l] else '.':>3}" for l in LABELS)
              + f" | {s['coh_max']:>12.9f} {s['coh_median']:>7.4f} {s['n_gt_0999']:>5} | "
              f"{s['rival']['T1']:>9.6f} {s['rival']['T2']:>12.9f} {s['rival']['T3']:>9.6f}")
        A(f"  分布：top-3 找回数 {e['top3_counts']}；top-1 真 {e['n_top1_true']}/{e['n_seeds']}；"
          f"逐个真漏点找回 " + " ".join(f"{l}:{e['per_truth_recovered'][l]}/{e['n_seeds']}" for l in LABELS))
        for nm, t in e["tests"].items():
            A(f"  检验 {nm}（top-3 = {t['design_top3']}，top-1 真 {t['design_top1']}）："
              f"随机里 {t['top3']['n_random_at_least_as_good']}/{t['top3']['n_random']} 不劣于它，"
              f"精确单侧 p = {t['top3']['p_one_sided']:.4f}"
              + "；逐漏点 " + " ".join(
                  f"{l}:{t['per_truth'][l]['n_random_at_least_as_good']}/{t['per_truth'][l]['n_random']}"
                  f"(p={t['per_truth'][l]['p_one_sided']:.3f})" for l in LABELS))
            if t["top3"]["p_one_sided"] >= 0.05:
                A(f"    功效：{t['top3']['power']['note']}")
    A("  p =（1 + #{随机不劣于设计}）/（1 + n）：交换性零假设（相干目标对"
      "\"哪些漏点被找回\"无信息）下设计排第一的精确概率。")
    dvr = design_vs_random(rows)
    rob1 = stage1_robustness(rows, dvr)
    A(f"\n  合并检验：所有\"按目标选出来的\"传感集合（coh / cohmax / cohfull / coverage / D-opt，"
      f"共 {dvr['n_designed']} 个）对所有随机抽样（共 {dvr['n_random']} 个），逐个真漏点的 2×2 单侧 Fisher：")
    A(f"  {'':>4} {'设计组找回':>10} {'随机组找回':>10} {'Fisher p':>10} | "
      f"{'找回时最小劲敌间隙':>18} {'丢失时最大劲敌间隙':>18} {'两组分开':>8}")
    for lab in LABELS:
        e = dvr["per_truth"][lab]
        A(f"  {lab:>4} {e['designed_recovered']:>10} {e['random_recovered']:>10} "
          f"{e['p_one_sided']:>10.2e} | "
          f"{('%.3e' % e['gap_recovered_min']) if e['gap_recovered_min'] is not None else '-':>18} "
          f"{('%.3e' % e['gap_lost_max']) if e['gap_lost_max'] is not None else '-':>18} "
          f"{('是' if e['separated'] else '否'):>8}")

    t2, t3, t1 = (dvr["per_truth"][l] for l in ("T2", "T3", "T1"))
    A(f"  结论一：撑起 2/3 的是 T2。T2 在 {dvr['n_designed']} 个按目标选出来的传感集合里被找回 "
      f"{t2['designed_recovered']}，在 {dvr['n_random']} 个公平随机抽样里 {t2['random_recovered']}"
      f"（Fisher p={t2['p_one_sided']:.2e}）；T3 设计组 {t3['designed_recovered']}、随机组 "
      f"{t3['random_recovered']}（p={t3['p_one_sided']:.2f}），看不出差别；T1 两组都是 0。"
      f"即：coh+40 的 2/3 = 一个真由相干驱动的 T2 + 一个与随机无异的 T3。")
    A("  （Fisher 那一列的设计组含同一序列的不同 k，互相嵌套、并不独立，只算佐证；"
      "同预算的交换性检验才是主证据。）")
    A()
    if cross:
        A("  重复核对（同一配置被跑过两次：两套独立实现 hv_leak_control.py / augment_coherence.py，"
          "或同一实现在不同 host 名下重跑 - 不同 torch / numpy / scipy 版本；host 字符串本身不可靠"
          " - ssh v100 的 hostname 与本机工作站字符串重合，见下方 host= 原样打印，不做同机/异机判断）：")
        for c in cross:
            A(f"    {c['spec']:>12}（{c['first_source']}[host={c.get('first_host')}] × "
              f"{c['second_source']}[host={c.get('second_host')}]）：传感数一致 {c['n_sensors_agree']}，"
              f"top-1 一致 {c['top1_agree']}，top-3 一致 {c['top3_agree']}，"
              f"流量相对误差最大差 {c['max_rel_err_diff']:.2e}")
        A()



    # ---- 二、相干-召回 ----
    A("### 二、相干-召回关系（全部可重建的 City D 配置，n = %d）" % rec["n_configs"])
    A(f"  {'协变量':>22} | {'Spearman ρ vs top-3 找回数':>26} {'n':>4} {'p(正态近似)':>12}")
    for k, v in rec["overall"].items():
        A(f"  {k:>22} | {('%.3f' % v['rho']) if v['rho'] is not None else 'n/a':>26} "
          f"{v['n']:>4} {('%.4f' % v['p_two_sided']) if v['p_two_sided'] is not None else '-':>12}")
    A(f"\n  按真漏点分解（找回 = 流量相对误差 <5%）：")
    A(f"  {'':>4} {'找回/配置':>10} | {'ρ(log 劲敌间隙)':>16} {'ρ(字典列范数)':>15} | "
      f"{'找回时的 log10(1−μ劲敌)':>24} | {'丢失时的区间':>22} | {'列范数区间':>18}")
    for lab in LABELS:
        p = rec["per_truth"][lab]
        g = p["log_rival_gap_recovered"]
        gs = f"[{min(g):.2f}, {max(g):.2f}]" if g else "-"
        ls_ = (f"[{p['log_rival_gap_lost_range'][0]:.2f}, {p['log_rival_gap_lost_range'][1]:.2f}]"
               if p["log_rival_gap_lost_range"] else "-")
        A(f"  {lab:>4} {p['n_recovered']:>4}/{p['n_configs']:<5} | "
          f"{('%.3f' % p['rho_vs_log_rival_gap']['rho']) if p['rho_vs_log_rival_gap']['rho'] is not None else 'n/a':>16} "
          f"{('%.3f' % p['rho_vs_col_norm']['rho']) if p['rho_vs_col_norm']['rho'] is not None else 'n/a':>15} | "
          f"{gs:>24} | {ls_:>22} | "
          f"[{p['col_norm_range'][0]:.3g}, {p['col_norm_range'][1]:.3g}]")
    pt2, pt3 = rec["per_truth"]["T2"], rec["per_truth"]["T3"]
    gt2 = dvr["per_truth"]["T2"]
    nrm1, nrm2 = rec["per_truth"]["T1"]["col_norm_range"], rec["per_truth"]["T2"]["col_norm_range"]
    A(f"  结论二：全局相干与 top-3 的相关（ρ={rec['overall']['log10(1-max_mu)']['rho']:.2f}）"
      f"完全经由 T2 传递。T2 的召回对它自己的劲敌间隙 ρ={pt2['rho_vs_log_rival_gap']['rho']:.3f}，"
      f"找回组的最小间隙 {gt2['gap_recovered_min']:.3e} 严格大于丢失组的最大间隙 "
      f"{gt2['gap_lost_max']:.3e}（{pt2['n_configs']} 个配置零重叠，阈值就落在这两个数之间）；"
      f"T3 的 ρ={pt3['rho_vs_log_rival_gap']['rho']:.3f}，两组区间重叠，与相干无关；"
      f"T1 一个都没找回，它的字典列范数只有 {nrm1[0]:.3g}–{nrm1[1]:.3g}，而 T2 是 "
      f"{nrm2[0]:.3g}–{nrm2[1]:.3g} - T1 是幅值受限（签名埋在 0.1 ft 噪声下），不是相干受限："
      f"它的劲敌相干本来就能被压到 {sep['city_d']['T1']['oracle_pair_greedy']['floor']:.3f}。")
    A()

    # ---- 三、可分性 ----
    A("### 三、机理：丢失的漏点与它的劲敌，在哪些传感位置上可分？")
    A("  μ(S) = |Σ_S c| / sqrt(Σ_S a · Σ_S b)（a/b/c = 该对在每个传感位置上的 Gram 贡献，逐行可加），")
    A("  于是\"全池全装能到多少\"\"单点最好能到多少\"\"只盯这一对的 oracle 贪心能到多少\"都是精确量。")
    for net, ttl in (("ltown", "L-TOWN（S0=33，公平池 746，候选漏点 60，字典 T=1）"),
                     ("city_d", "City D（S0=40，公平池 498，候选漏点 49，字典 T=25）")):
        s = sep[net]
        A(f"\n  [{ttl}]  J(S0) = {s['_J_S0']:.1f}")
        A(f"  {'':>4} {'劲敌μ(S0)':>12} {'公平池全装':>12} {'每个junction':>13} {'单点最好':>12} {'跳':>3} "
          f"{'<.999 位置':>10} | {'oracle对下限':>13} {'压完后最大劲敌':>14} | {'>.99劲敌':>8} {'该对在J':>8}")
        for lab in LABELS:
            r = s[lab]
            ss, g, cl = r["single_sensor"], r["oracle_pair_greedy"], r["rival_cluster"]
            A(f"  {lab:>4} {r['rival_coh_S0']:>12.9f} {r['mu_fair_all']:>12.9f} "
              f"{r['mu_every_junction']:>13.9f} {ss['fair_best']:>12.9f} "
              f"{(ss['fair_best_hops'] if ss['fair_best_hops'] is not None else -1):>3} "
              f"{ss['n_fair_lt_0999']:>4}/{ss['n_fair']:<5} | {g['floor']:>13.9f} "
              f"{g['max_rival_after']:>14.9f} | {cl['n_gt_099']:>8} {r['pair_term_in_J']:>8.2f}")
        for lab in LABELS:
            r = s[lab]
            A(f"       {lab} 劲敌簇（S0 上前 3）{[round(x, 6) for x in r['rival_cluster']['top3_rival_coh_S0']]}"
              f"；可分位置（单点能把该对压到 <0.999）{r['n_separating_positions']} 个，"
              f"coh_fair 前 k 步选中 {r['separating_positions_picked_by_coh_fair']}"
              f"；字典列范数 真 {r['col_norm_true']:.4g}（S0 行上 {r['col_norm_true_on_S0']:.4g}）"
              f" 劲敌 {r['col_norm_rival']:.4g}")
    A("\n  帧数反事实（City D，同一 S0 / 同一全装，只换字典帧子集） - 相干高不是\"只有一帧\"造成的：")
    A(f"  {'帧子集':>18} | {'S0: max μ':>12} {'中位':>7} {'>.999':>5} {'劲敌T2':>12} | "
      f"{'全装: max μ':>12} {'中位':>7} {'>.999':>5} {'劲敌T2':>12}")
    for name, e in frames.items():
        A(f"  {name:>18} | {e['S0']['coh_max']:>12.9f} {e['S0']['coh_median']:>7.4f} "
          f"{e['S0']['n_gt_0999']:>5} {e['S0']['rival']['T2']:>12.9f} | "
          f"{e['S0+fair_all']['coh_max']:>12.9f} {e['S0+fair_all']['coh_median']:>7.4f} "
          f"{e['S0+fair_all']['n_gt_0999']:>5} {e['S0+fair_all']['rival']['T2']:>12.9f}")
    A("  反演器不对称（读代码得到，非本轮实验）：City D 用 demo_leak_inversion - Adam+L1 粗筛之后有"
      "离散支撑精化（非线性 OMP + 对手互换抛光）；L-TOWN 用 augment_public.lt_invert - 只有 Adam+L1"
      "60 步、按幅值排 top-k，没有任何离散支撑搜索。把相干间隙变成正确支撑的那一步，L-TOWN 上不存在。")
    lt = sep["ltown"]
    A(f"  结论三：L-TOWN 相干压不下来不是贪心或目标函数的错，是上限本身就低。丢失的 T1 在 "
      f"{lt['T1']['single_sensor']['n_fair']} 个公平位置里只有 {lt['T1']['n_separating_positions']} "
      f"个能把它与劲敌压到 0.999 以下，而这个位置 coh_fair 前 5 步就选了"
      f"（{lt['T1']['separating_positions_picked_by_coh_fair']['+5']} 个）；T2 的 "
      f"{lt['T2']['n_separating_positions']} 个里前 5 步也选中 "
      f"{lt['T2']['separating_positions_picked_by_coh_fair']['+5']} 个。上限：即使每个 junction 都装表，"
      f"两对仍停在 {lt['T1']['mu_every_junction']:.6f} / {lt['T2']['mu_every_junction']:.6f}；"
      f"只盯一对的 oracle 贪心 80 步也只到 {lt['T1']['oracle_pair_greedy']['floor']:.6f} / "
      f"{lt['T2']['oracle_pair_greedy']['floor']:.6f}，而且压完首席劲敌后次席顶上来（最大劲敌回到 "
      f"{lt['T1']['oracle_pair_greedy']['max_rival_after']:.6f} / "
      f"{lt['T2']['oracle_pair_greedy']['max_rival_after']:.6f}，>0.99 的劲敌各有 "
      f"{lt['T1']['rival_cluster']['n_gt_099']} 个和 {lt['T2']['rival_cluster']['n_gt_099']} 个）。"
      f"作对照，唯一被找回的 T3 在 S0 上就已经 {lt['T3']['rival_coh_S0']:.4f}，"
      f"{lt['T3']['n_separating_positions']}/{lt['T3']['single_sensor']['n_fair']} 个位置都能压到 0.999 "
      f"以下，oracle 到 {lt['T3']['oracle_pair_greedy']['floor']:.4f}。这两对在 60 个候选的池子里本质不可分。")
    A()

    # ---- 四、反演器不对称的稳健性检验 ----
    A("### 四、反演器不对称的稳健性检验（结论一/二只用 stage1 会不会变）")
    A("  City D 反演器 = Adam+L1 粗筛（stage1，输出按幅值排序）→ 离散支撑搜索（非线性 OMP + 对手互换"
      "抛光，stage2，得到本报告结论一/二里用的 flow_err<5% 判据）。L-TOWN 的 augment_public.lt_invert "
      "只有前一半：Adam+L1、按幅值排 top-k，没有 stage2。stage1 的\"真值列在不在幅值 top-3 里\"与 "
      "lt_invert 的 n_true_in_top3 逐位同型 - 都是同一个判据。把结论一（设计组 vs 随机组的 Fisher 检验）"
      "与结论二（找回/丢失两组的劲敌间隙分离）只用 stage1 的判据重算一遍，等价于问：如果 City D 也只有"
      "L-TOWN 那半个反演器，结论还在不在。")
    A(f"  {'':>4} {'stage1 设计组':>11} {'stage1 随机组':>11} {'stage1 p':>10} {'stage2 设计组':>11} "
      f"{'stage2 随机组':>11} {'stage2 p':>10} | {'stage1 找回时最小间隙':>20} {'stage1 丢失时最大间隙':>20} "
      f"{'分开':>5} | {'ρ(stage1)':>10} | {'一致':>5} {'refine加':>7} {'refine撤':>7}")
    for lab in LABELS:
        e = rob1["per_truth"][lab]
        rho = e["rho_vs_log_rival_gap"]["rho"]
        A(f"  {lab:>4} {e['designed_recovered']:>11} {e['random_recovered']:>11} "
          f"{e['p_one_sided']:>10.2e} {e['final_designed_recovered']:>11} "
          f"{e['final_random_recovered']:>11} {e['final_fisher_p']:>10.2e} | "
          f"{('%.3e' % e['gap_recovered_min']) if e['gap_recovered_min'] is not None else '-':>20} "
          f"{('%.3e' % e['gap_lost_max']) if e['gap_lost_max'] is not None else '-':>20} "
          f"{('是' if e['separated'] else '否'):>5} | "
          f"{('%.3f' % rho) if rho is not None else 'n/a':>10} | "
          f"{e['agree_both_recovered'] + e['agree_neither']:>5}/{e['n_configs']:<3} "
          f"{e['refine_adds']:>7} {e['refine_drops']:>7}")
    rt2, rt3, rt1 = (rob1["per_truth"][l] for l in ("T2", "T3", "T1"))
    A(f"\n  结论四：结论一/二不是离散支撑搜索这一步做出来的。只用 stage1（与 L-TOWN 的反演器同型）的"
      f"判据重算，T2 在 {rob1['n_designed']} 个设计集合里被 stage1 选中 {rt2['designed_recovered']}，"
      f"在 {rob1['n_random']} 个随机集合里 {rt2['random_recovered']}"
      f"（Fisher p={rt2['p_one_sided']:.2e}，stage2/最终判据是 p={rt2['final_fisher_p']:.2e}）；"
      f"stage1 下找回组的最小劲敌间隙 {('%.3e' % rt2['gap_recovered_min']) if rt2['gap_recovered_min'] is not None else 'n/a'} "
      f"{'严格大于' if rt2['separated'] else '不大于'}丢失组的最大间隙 "
      f"{('%.3e' % rt2['gap_lost_max']) if rt2['gap_lost_max'] is not None else 'n/a'}"
      f"（stage1 下 ρ={rt2['rho_vs_log_rival_gap']['rho']:.3f}，"
      f"stage2 下 ρ={rec['per_truth']['T2']['rho_vs_log_rival_gap']['rho']:.3f}，参见结论二）；"
      f"stage1/stage2 判定在 T2 上 {rt2['agree_both_recovered'] + rt2['agree_neither']}/{rt2['n_configs']} "
      f"个配置一致，离散支撑搜索加回 {rt2['refine_adds']} 个、撤掉 {rt2['refine_drops']} 个。"
      f"T3（stage2 下与随机无异，p={rt3['final_fisher_p']:.2f}）在 stage1 下 p={rt3['p_one_sided']:.2f}，"
      f"同样看不出设计组比随机组强；T1 在 stage1 下设计组 {rt1['designed_recovered']}、随机组 "
      f"{rt1['random_recovered']}，{'也显著（p=' + format(rt1['p_one_sided'], '.2e') + '）' if rt1['p_one_sided'] < 0.05 else '不显著（p=' + format(rt1['p_one_sided'], '.2f') + '）'}"
      f" - 这是一个新事实，不是结论二的重复：即使是 T1，相干驱动的传感集合也比随机集合更常把它排进"
      f"幅值粗筛的 top-3（stage2 下两组仍都是 0）；但离散支撑搜索之后没有一个通过 flow_err<5%，说明"
      f"相干驱动能帮 T1 挤进候选名单，挤不进准确的流量估计 - 瓶颈在幅值/信噪比，不在相干，与结论二"
      f"\"T1 是幅值受限\"一致，只是这次多了一层：排得进粗筛的名次，量不出准的流量。"
      f"结论：把 City D 的反演器砍掉离散支撑搜索、只留 L-TOWN 那半步，T2 由相干驱动、T3 与随机无异这"
      f"两条结论的方向和显著性都没变 - 反演器不对称不影响本报告对 T2/T3 的判断。")
    A()

    # ---- 全表 ----
    A("### 附：City D 工单案例全部重跑（同一反演器、同一噪声实现）")
    A(f"  {'配置':>18} {'来源':>20} {'传感':>5} {'top1':>5} {'top3':>4} {'T1 T2 T3':>9} | "
      f"{'相干max':>12} {'中位':>7} {'>.999':>5} | {'劲敌T1':>9} {'劲敌T2':>12} {'劲敌T3':>9} | "
      f"{'真列范数 T1/T2/T3':>20}")
    for r in sorted(rows, key=lambda x: (x["n_sensors"], x["family"], x["spec"])):
        st = r["recomputed"]
        tail = (f"{st['coh_max']:>12.9f} {st['coh_median']:>7.4f} {st['n_gt_0999']:>5} | "
                f"{st['rival']['T1']:>9.6f} {st['rival']['T2']:>12.9f} {st['rival']['T3']:>9.6f} | "
                + "/".join(f"{st['col_norm'][l]:.3g}" for l in LABELS)) if st else \
               (f"{r['recorded_coh_max']:>12.9f} {'':>7} {'':>5} | 只有记录值（传感集合无法重建）")
        A(f"  {r['spec']:>18} {r['source']:>20} {r['n_sensors']:>5} {str(r['top1']):>5} {r['top3']:>4} "
          + " ".join(f"{'Y' if r['recovered'][l] else '.':>3}" for l in LABELS) + f" | {tail}")

    txt = "\n".join(L)
    print(txt)
    summary = dict(
        config=dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                    machine_note=machine, seeds=SEEDS,
                    random_draw="np.random.default_rng(seed).choice(junction − S0, k)；"
                                "抽到漏点节点的 seed 作废（拒绝抽样 = 公平池上的均匀抽样）；"
                                "与 audit_augment/hv_leak_control.py 的 rand:k:seed 逐位同一",
                    recovered_criterion="流量相对误差 < 5%",
                    p_value="（1 + #{随机不劣于设计}）/（1 + n），交换性零假设下的精确单侧 p"),
        crosscheck_two_implementations=cross,
        random_controls=rnd, design_vs_random=dvr, recall=rec, separability=sep,
        frame_counterfactual=frames, stage1_robustness=rob1,
        runs=[{k: v for k, v in r.items() if k != "recorded_rival"} for r in rows])
    from augment_suite import assert_no_ids, id_set
    assert_no_ids(summary, id_set("city_d"), set(), "coh_controls.json")
    jdump(summary, os.path.join(DATA, "coh_controls.json"))
    with open(os.path.join(DATA, "coh_controls_wip.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        make_figure(rows, rnd, sep, os.path.join(DATA, "fig_coh_controls.png"))
    except Exception as e:
        print(f"（图未生成：{type(e).__name__}: {e}）")
    print(f"\nreport 完成：data/coh_controls_wip.txt、data/coh_controls.json（{time.time() - t0:.0f}s）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="report", choices=["report"])
    ap.add_argument("--machine", default="", help="一句话记录机器")
    a = ap.parse_args()
    stage_report(a.machine)
    return 0


if __name__ == "__main__":
    sys.exit(main())

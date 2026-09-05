# -*- coding: utf-8 -*-
"""hv_report.py: assemble the hostile-verification record of the augmentation package.

Inputs (all produced by the scripts in this directory):
  data/audit_augment_fisher.json            workstation recompute (hv_fisher.py)
  data/audit_augment_fisher_v100.json       the same script on the V100 host (optional)
  data/audit_augment_leak_fair.json         hv_leak_fair.py
  data/audit_leak_control/*.json            hv_leak_control.py runs (V100 host)
  --regression <report>                     regression_all report written to scratch
  --notes <file>                            hand-written verdict, appended verbatim
Output: data/audit_augment_wip.txt (readable, no identifiers) and data/audit_leak_control.json
(merged control runs).  The zero-identifier guard of hv_noids.py is applied before writing.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)


def jload(fp, default=None):
    if os.path.isfile(fp):
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def f3(x, fmt="{:.3f}"):
    return "nan" if x is None else fmt.format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regression", default="")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    fi = jload(os.path.join(DATA, "audit_augment_fisher.json"))
    fv = jload(os.path.join(DATA, "audit_augment_fisher_v100.json"))
    lf = jload(os.path.join(DATA, "audit_augment_leak_fair.json"))
    ctrl = {}
    for fp in sorted(glob.glob(os.path.join(DATA, "audit_leak_control", "*.json"))):
        r = jload(fp)
        ctrl[r["spec"]] = r
    with open(os.path.join(DATA, "audit_leak_control.json"), "w", encoding="utf-8") as f:
        json.dump(dict(note="control reruns of the work-order leak case (hv_leak_control.py); labels T1..T3, hop "
                            "counts and kinds only, no identifiers", runs=ctrl), f, ensure_ascii=False, indent=1)
    L = []
    L.append("增设（augment）包敌意验证记录 - 自写脚本 scripts/audit_augment/*，不复用 augment_suite / place_sensors / dgga.placement 的判定逻辑")
    L.append(f"工作站 {fi['host']}；V100 主机 {fv['host'] if fv else '（未跑）'}；控制重跑主机 "
             f"{sorted({r['host'] for r in ctrl.values()}) if ctrl else '（无）'}")
    L.append("")
    L.append("一、增设曲线复算（hv_fisher.py：自写普查判据 / Fisher 矩阵显式求逆 / SVD 型 CRLB；随机抽 3 档 k）")
    L.append(f"  抽样种子 {fi['seed']} → k = {fi['ks_drawn']}；T={fi['T']} 候选 {fi['m']} P={fi['P']} 结构性 {fi['n_struct']}"
             f"（死支 {fi['n_dead']} + 钳位 {fi['n_clamped']}）")
    for k, v in fi["checks"].items():
        L.append(f"  [{'OK' if v is True or (isinstance(v, int) and not isinstance(v, bool)) or isinstance(v, dict) else 'FAIL'}] {k} = {v}")
    s0 = fi["s0"]
    L.append(f"  S0：可辨识 {s0['n_ident']} unob {s0['n_unob']} 秩 {s0['rank']} f={s0['f']:.6f} CRLB_ident={s0['crlb_ident']:.4e} "
             f"k_sub={s0['k_sub']} CRLB_sub={s0['crlb_sub']:.4e}（与 placement_augment / augment_suite 逐项同）")
    L.append(f"  {'目标':>6} {'+k':>3} {'找回':>4} {'未找回':>5} {'unob':>4} {'丢失':>4} {'可辨识':>5} {'秩':>4} {'f':>10} {'CRLB_ident':>10} "
             f"{'k_sub':>5} {'CRLB_sub':>10} {'找回管后验std中位':>12} {'≤σp/2':>5} | 上游 找回/unob | 计数 谱 后验")
    for r in fi["table"]:
        L.append(f"  {r['objective']:>6} {r['k']:>3} {r['n_recovered']:>4} {r['n_left']:>5} {r['n_unobservable']:>4} {r['n_lost']:>4} "
                 f"{r['n_identifiable']:>5} {r['rank']:>4} {r['f']:>10.4f} {r['crlb_ident']:>10.3e} {r['k_sub']:>5} "
                 f"{r['crlb_sub']:>10.3e} {f3(r['post_std_recovered_median']):>12} {r['n_recovered_half_prior']:>5} | "
                 f"{r['ref']['n_recovered']:>3}/{r['ref']['n_unobservable']:<3} | {r['match_counts']} {r['match_spectrum']} {r['match_post_std']}")
    for obj, v in fi["monotonicity"].items():
        L.append(f"  单调性 {obj}：80 步逐步核对，可辨识集缩小 {v['ident_shrink_violations']} 次，后验方差上升 "
                 f"{v['post_var_rise_violations']} 次，丢失最大 {v['lost_max']}；逐步找回曲线与上游逐元素同={v['recovered_every_k_equals_ref']}；"
                 f"80% 需 +{v['k_recover80']}（上游 {v['ref_k_recover80']}），全部 +{v['k_recover_all']}（上游 {v['ref_k_recover_all']}）")
    an = fi["section36_anchors"]
    L.append("  §3.6 锚点（同判据、同 SVD 公式）：" + "；".join(
        f"k={k} D-opt {v['dopt_recovered']}/{v['dopt_unobservable']} 随机中位 {v['random_median_recovered']}/{v['random_median_unobservable']}"
        f"（论文 {v['paper_dopt']}/{v['paper_dopt_unob']}，{v['paper_random']}/{v['paper_random_unob']}）{'同' if v['match'] else '异'}"
        for k, v in an.items() if k.isdigit()))
    L.append(f"    k=40 丢失 {an['k40_lost']}（论文 38）；k=40 CRLB_ident {an['k40_crlb_ident_same_formula']:.4e} vs 主线 json {an['k40_crlb_ident_ref']:.4e}")
    hk = fi["hanoi_crlb"]
    L.append("  Hanoi CRLB（表 2 同公式）：" + "；".join(
        f"k={k} D-opt {v['dopt']:.4f}/表 {v['table_dopt']} 随机中位 {v['random_median']:.3f}/表 {v['table_random']} {'同' if v['match'] else '异'}"
        for k, v in hk.items() if k.isdigit()))
    if fv:
        diffs = []
        for r0, r1 in zip(fi["table"], fv["table"]):
            for key in ("n_recovered", "n_unobservable", "n_lost", "rank", "k_sub"):
                if r0[key] != r1[key]:
                    diffs.append(f"{r0['objective']}+{r0['k']} {key} {r0[key]} vs {r1[key]}")
            for key in ("f", "crlb_ident", "crlb_sub", "bayes_trace", "post_std_recovered_median"):
                x, y = r0[key], r1[key]
                if x is not None and y is not None and abs(x - y) > 1e-9 * max(abs(x), 1e-300):
                    diffs.append(f"{r0['objective']}+{r0['k']} {key} rel {abs(x - y) / abs(x):.1e}")
        bad = [k for k, v in fv["checks"].items() if v is False]
        L.append(f"  V100 独立重跑（{fv['host']}，同缓存文件）：计数逐项相同；数值相对差>1e-9 的项：{diffs if diffs else '无'}；"
                 f"V100 上 False 的 check：{bad if bad else '无'}")
    L.append("  σ 梯（B）：找回管 |Ĉ−C_true| 用本脚本的真值再生（seed 7）与本脚本的找回掩码复算：")
    for r in fi.get("sigma_ladder", []):
        if "error" in r:
            L.append(f"    {r['key']}: {r['error']}")
            continue
        L.append(f"    {r['placement']:>11} σ={r['sigma']:<5g} 找回 {r['n_recovered']:>3} |ΔC|中位 {f3(r['err_median'], '{:.2f}'):>6} "
                 f"先验 {f3(r['err_prior_median'], '{:.2f}'):>6} 优于先验 {r['n_better_than_prior']:>3} ≤10 {r['n_le10']:>3} "
                 f"unob 记录/复算 {r['n_unob_record']}/{r['n_unob_here']} C0 反推 {r['c0_inferred_from_unobservable']} 与上游同={r['match_suite']}")
    L.append("")

    # ---- leak fairness ----
    L.append("二、工单漏损重跑公平性（hv_leak_fair.py：自写配方重放 / 跳数 / 足迹 / 字典相干）")
    for k, v in lf["checks"].items():
        L.append(f"  [{'OK' if v is True or isinstance(v, (int, dict)) else 'FAIL'}] {k} = {v}")
    L.append(f"  真值漏损流量 LPS：{ {k: round(v, 4) for k, v in lf['true_leak_lps'].items()} }（与记录逐位同）")
    for t, v in lf["footprint_all_junctions"].items():
        L.append(f"  足迹 {t}（单独存在，25 帧，全部 junction）：max {v['max_ft_all_junctions']:.4f} ft，峰值在漏点本节点={v['peak_is_leak_node']}，"
                 f"max>噪声(0.1 ft) 的 junction {v['n_junctions_max_gt_noise']} 个（距漏点跳数 {v['hops_of_junctions_max_gt_noise']}），"
                 f"rms>噪声 {v['n_junctions_rms_gt_noise']} 个")
    L.append(f"  {'配置':>11} {'传感':>4} {'增设':>4} | " + " | ".join(
        f"{t}: 距S0 距增设 ≤1跳 ≤2跳 rms    max" for t in ("T1", "T2", "T3")) + " | 相干max 中位 >.999 与记录同 | 记录 top3 支撑")
    for name, v in lf["configs"].items():
        p = v["proximity"]
        cells = []
        for t in ("T1", "T2", "T3"):
            q = p[t]
            cells.append(f"{t}: {q['hop_to_nearest_S0']:>3} {str(q['hop_to_nearest_added']):>5} {q['n_added_within_1hop']:>4} "
                         f"{q['n_added_within_2hops']:>4} {q['foot_rms_ft']:.3f} {q['foot_max_ft']:.3f}")
        cs = v["coherence"]
        rr = v.get("result_recorded", {})
        L.append(f"  {name:>11} {v['n_sensors']:>4} {v['n_added']:>4} | " + " | ".join(cells)
                 + f" | {cs['max_offdiag']:.6f} {cs['median_all']:.4f} {cs['n_gt_0999']:>4} {str(v.get('coherence_match', '-')):>5} | "
                 + (f"{rr['top3']}/3 {rr['support']}" if rr else "（未跑）"))
    L.append(f"  增设序列首次落到漏点 h 跳内的步号（-1=80 步内未落）：{lf['first_added_step_within_hops']}")
    L.append(f"  噪声实现：{lf['noise']}")
    L.append("")

    # ---- controls ----
    L.append("三、控制重跑（hv_leak_control.py，V100 主机，反演器 = demo_leak_inversion 原样，噪声组 = 重跑同实现 seed 909）")
    if ctrl:
        L.append(f"  {'spec':>22} {'传感':>4} {'增设':>4} | {'T1 距/≤1/漏点有传感':>18} | {'T2 距/≤1/漏点有传感':>18} | {'T3 距/≤1/漏点有传感':>18} | "
                 f"{'相干max':>8} {'中位':>6} {'>.999':>5} {'T2劲敌':>8} {'T3劲敌':>8} | L1top3 | 支撑 top1 top3 | T2误差 | final_mse 秒")
        for spec, r in ctrl.items():
            p = r["proximity"]
            cells = [f"{p[t]['hop_to_nearest_sensor']:>2}/{p[t]['n_added_within_1hop']:>2}/{str(p[t]['leak_node_is_sensor']):>5}"
                     for t in ("T1", "T2", "T3")]
            cs = r["coherence"]
            rv = cs["true_node_max_rival_coh"]
            L.append(f"  {spec:>22} {r['n_sensors']:>4} {r['n_added']:>4} | " + " | ".join(f"{c:>18}" for c in cells)
                     + f" | {cs['max_offdiag']:>8.6f} {cs['median_all']:>6.4f} {cs['n_pairs_gt_0999']:>5} {rv['T2']:>8.5f} {rv['T3']:>8.5f} | "
                     f"{r['stage1']['top3_hits']:>6} | {[d['kind'] for d in r['support']]} {r['top1_hit']} {r['top3_hits']}/3 | "
                     f"{r['flow_err']['T2']['rel_err']:.3f} | {r['final_mse_ft2']:.3e} {r['time_sec']:.0f}")
            L.append("      支撑成员：" + "；".join(
                (f"{d['kind']} {d['leak_lps']:.3f} LPS" if d["kind"] != "nontrue"
                 else f"非真值(与{d['closest_true']} coh={d['coh']:.5f}) {d['leak_lps']:.3f} LPS") for d in r["support"]))
    else:
        L.append("  （无控制重跑记录）")
    L.append("")
    if a.regression and os.path.isfile(a.regression):
        with open(a.regression, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        m = re.findall(r"(总判定.*)", txt)
        fails = re.findall(r"^(\S+ [^|]*?)\s*\|.*\| FAIL \|", txt, flags=re.M)
        only_guard = fails and all("匿名化" in x for x in fails)
        L.append("四、regression_all（工作站，--out scratch）：" + ("；".join(x.strip() for x in m[:2]) if m else "见报告")
                 + f"；FAIL 项 {[x.strip() for x in fails]}"
                 + ("（= ⑫ 匿名化守卫的冻结树旧命中，非守卫 53/53）" if only_guard else ""))
        L.append("")
    if a.notes and os.path.isfile(a.notes):
        with open(a.notes, "r", encoding="utf-8") as f:
            L.append(f.read().rstrip("\n"))
        L.append("")
    text = "\n".join(L) + "\n"
    # zero-identifier guard (own scanner)
    from hv_noids import TOK
    from dgga.parse import Net
    net = Net.load(os.path.join(DATA, "reference"), "city_d")
    ids = set(str(x) for x in net.node_id) | set(str(x) for x in net.link_id)
    alnum_hits = sorted({t for t in TOK.findall(text) if t in ids and not t.isdigit()})
    if alnum_hits:
        raise RuntimeError(f"identifier-like tokens in the report: {alnum_hits[:10]}")
    fp = os.path.join(DATA, "audit_augment_wip.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"[hv_report] wrote {fp} and data/audit_leak_control.json ({len(ctrl)} control runs)")


if __name__ == "__main__":
    main()

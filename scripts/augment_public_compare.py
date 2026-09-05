# -*- coding: utf-8 -*-
"""augment_public_compare.py - 公开网增设全套的 V100 服务器复现：合并 + 逐项对拍。

输入（--v100-dir，从 V100 的 scratch 工作目录取回，见 scripts/augment_public_v100.sh）：
  placement_augment_ltown.json / placement_orders_ltown.npz            （ltown-sfull + ltown-augment）
  placement_augment_pub_hanoi_synth25.json / placement_orders_pub_hanoi_synth25.npz
  augment_public_sigma_ladder.json                                     （sigma-ladder）
  ltown_augment_leak_g0.json / _g1 / _g2                               （ltown-leak，三卡分跑）
  calib_gc1_pub_hanoi.json                                             （hanoi-calib，AUG_* 键）
做三件事：
  1. 把三卡的 ltown-leak 结果合并成 --merge-out（缺省 data/ltown_augment_leak_v100.json；
     主结果 data/ltown_augment_leak.json = 集群 5090 作业 1543441 的记录，**不动**），
     config 记录每个配置跑在哪张卡、哪份日志；
  2. 逐项与仓库里的参考结果对拍：选点序列（npz 逐元素）、增设曲线（计数逐位、
     实数相对差）、σ 梯（逐格）、漏损反演（与 --leak-ref 主结果、--leak-local
     本机 dense 路径：loss / top-5 / 真值名次 / 相干）、Hanoi σ 梯标定（AUG_* 逐键）；
     --v100-dir 里缺哪部分就跳过哪部分的对拍；
  3. 把对拍摘要写到 --out（缺省 data/augment_public_v100_repro.json）与同名 .txt。
     V100 的 Hanoi 标定记录另存 --calib-out（缺省 data/calib_gc1_pub_hanoi_v100.json），
     不动 calib_gc1_pub_hanoi.json。

V100 原始输出（--v100-dir）与作业日志（data/gpu/v100_augpub_*.log，gitignored）不入库：
对拍摘要 json/txt 与合并后的两份 V100 记录入库，原件留在服务器
/mnt/sda/$USER/scratch/augpub_v100/{cpu,calib,g0,g1,g2}。

运行：& python -X utf8 scripts/augment_public_compare.py --v100-dir data/v100_augpub
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CFG_ORDER = ["S0", "dopt+5", "dopt+10", "dopt+20", "dopt+40", "dopt+80",
             "cover+5", "cover+10", "cover+20", "cover+40", "cover+80"]


def jload(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, p):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, p)


def rel(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def rp(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


# ----------------------------------------------------------------------
def parse_logs(log_dir):
    """data/gpu/v100_augpub_<label>_<date>.log → 出处（主机 / GPU / commit / 版本 / 耗时 / 命令）。"""
    out = {}
    if not os.path.isdir(log_dir):
        return out
    for fn in sorted(os.listdir(log_dir)):
        m = re.match(r"v100_augpub_(\w+?)_(\d{8})\.log$", fn)
        if not m:
            continue
        label = m.group(1)
        txt = open(os.path.join(log_dir, fn), "r", encoding="utf-8", errors="replace").read()
        d = dict(log=rp(os.path.join(log_dir, fn)))
        mm = re.search(r"host=(\S+)\s+start=(\S+)\s+CUDA_VISIBLE_DEVICES='([^']*)'", txt)
        if mm:
            d.update(host=mm.group(1), start=mm.group(2), cuda_visible_devices=mm.group(3))
        mm = re.search(r"^(\d+), (Tesla [^,]+), (\d+) MiB, ([\d.]+)$", txt, re.M)
        if mm:
            d.update(gpu_name=mm.group(2).strip(), driver=mm.group(4))
        mm = re.search(r"commit=(\w+)", txt)
        if mm:
            d["commit"] = mm.group(1)
        mm = re.search(r"python (\S+) torch (\S+) numpy (\S+) scipy (\S+)", txt)
        if mm:
            d.update(python=mm.group(1), torch=mm.group(2), numpy=mm.group(3), scipy=mm.group(4))
        mm = re.search(r"CMD: (.*)", txt)
        if mm:
            d["cmd"] = mm.group(1).strip()
        mm = re.search(r"AUGPUB DONE rc=(\d+) elapsed=(\d+)s", txt)
        if mm:
            d.update(rc=int(mm.group(1)), elapsed_sec=int(mm.group(2)))
        d["md5"] = {name: h for h, name in re.findall(r"^([0-9a-f]{32})\s+(\S+)$", txt, re.M)}
        out[label] = d
    return out


# ----------------------------------------------------------------------
def cmp_orders(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    keys = sorted(set(a.files) & set(b.files))
    res = {}
    for k in keys:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        same = x.shape == y.shape and np.array_equal(x, y)
        d = dict(equal=bool(same), n=int(x.size))
        if not same and x.shape == y.shape:
            d["first_diff_pos"] = int(np.argmax(x != y))
            d["n_diff"] = int((x != y).sum())
        res[k] = d
    return dict(keys=res, all_equal=all(v["equal"] for v in res.values()),
                missing_keys=sorted(set(a.files) ^ set(b.files)))


def cmp_augment_json(a, b):
    """增设曲线可读 JSON 的逐项对拍（a = V100, b = 参考）。"""
    out = dict(counts_equal=True, max_rel_real=0.0, worst=None, items=[])

    def chk(path, x, y, kind):
        if kind == "int":
            ok = int(x) == int(y)
            if not ok:
                out["counts_equal"] = False
            out["items"].append(dict(path=path, v100=x, ref=y, equal=ok))
        else:
            if x is None or y is None:
                out["items"].append(dict(path=path, v100=x, ref=y, equal=(x is None) == (y is None)))
                return
            r = rel(x, y)
            if r > out["max_rel_real"]:
                out["max_rel_real"], out["worst"] = r, dict(path=path, v100=x, ref=y)

    if "sfull" in a and "sfull" in b:
        sa, sb = a["sfull"], b["sfull"]
        for k in ("N", "Nj", "L", "P", "n_frames", "n_dead", "n_clamped_allframes",
                  "n_zero_col_all_sensors", "unobservable_floor"):
            chk(f"sfull.{k}", sa[k], sb[k], "int")
        chk("sfull.iters_status_machine", int(sa["iters_status_machine"] == sb["iters_status_machine"]), 1, "int")
        chk("sfull.active_prv_per_frame", int(sa["active_prv_per_frame"] == sb["active_prv_per_frame"]), 1, "int")
        out["sfull"] = dict(resid_inf_max=dict(v100=sa["resid_inf_max"], ref=sb["resid_inf_max"]),
                            relerr_status_machine_max=dict(v100=max(sa["relerr_status_machine"]),
                                                           ref=max(sb["relerr_status_machine"])),
                            t_total_sec=dict(v100=sa["t_total_sec"], ref=sb["t_total_sec"]),
                            sha256_equal=sa["sha256"] == sb["sha256"])
    for k in ("n_fixed", "n_candidates", "P", "n_structural", "n_target"):
        chk(f"config.{k}", a["config"][k], b["config"][k], "int")
    for k in ("n_identifiable", "n_unobservable", "rank", "n_sensors"):
        chk(f"s0.{k}", a["s0"][k], b["s0"][k], "int")
    for k in ("f", "crlb_ident", "bayes_trace"):
        chk(f"s0.{k}", a["s0"][k], b["s0"][k], "real")
    chk("s0.post_std_all_nonstruct.median", a["s0"]["post_std_all_nonstruct"]["median"],
        b["s0"]["post_std_all_nonstruct"]["median"], "real")
    chk("s0.post_std_all_nonstruct.n_half_prior", a["s0"]["post_std_all_nonstruct"]["n_half_prior"],
        b["s0"]["post_std_all_nonstruct"]["n_half_prior"], "int")
    for obj in ("dopt", "cover"):
        A, B = a["augment"][obj], b["augment"][obj]
        for k in ("k_recover80", "k_recover_all", "n_evals"):
            chk(f"augment.{obj}.{k}", A[k], B[k], "int")
        chk(f"augment.{obj}.cover_saturated_at", int(A["cover_saturated_at"] == B["cover_saturated_at"]), 1, "int")
        chk(f"augment.{obj}.recovered_every_k", int(A["recovered_every_k"] == B["recovered_every_k"]), 1, "int")
        chk(f"augment.{obj}.ident_curve", int(A["ident_curve"] == B["ident_curve"]), 1, "int")
        for kk in A["per_k"]:
            ra, rb = A["per_k"][kk], B["per_k"][kk]
            for k in ("n_recovered", "n_target_left", "n_lost", "n_unobservable",
                      "n_identifiable", "rank", "n_sensors"):
                chk(f"augment.{obj}.per_k.{kk}.{k}", ra[k], rb[k], "int")
            for k in ("f", "df", "cert_ratio", "crlb_ident", "bayes_trace", "upper_df"):
                chk(f"augment.{obj}.per_k.{kk}.{k}", ra[k], rb[k], "real")
            for grp in ("post_std_all_nonstruct", "post_std_target", "post_std_recovered"):
                chk(f"augment.{obj}.per_k.{kk}.{grp}.median", ra[grp]["median"], rb[grp]["median"], "real")
                chk(f"augment.{obj}.per_k.{kk}.{grp}.n_half_prior", ra[grp]["n_half_prior"],
                    rb[grp]["n_half_prior"], "int")
        out[f"t_{obj}_sec"] = dict(v100=A["t_total_sec"], ref=B["t_total_sec"])
    for kk in a["reselect"]["per_k"]:
        ra, rb = a["reselect"]["per_k"][kk], b["reselect"]["per_k"][kk]
        for k in ("n_recovered", "n_lost", "n_unobservable", "rank"):
            chk(f"reselect.per_k.{kk}.{k}", ra[k], rb[k], "int")
        for k in ("f", "crlb_ident"):
            chk(f"reselect.per_k.{kk}.{k}", ra[k], rb[k], "real")
    for k in ("n_recovered", "n_lost", "n_unobservable"):
        chk(f"reselect.same_k_as_s0.{k}", a["reselect"]["same_k_as_s0"][k],
            b["reselect"]["same_k_as_s0"][k], "int")
    out["n_int_items"] = sum(1 for it in out["items"])
    out["n_int_mismatch"] = sum(1 for it in out["items"] if not it["equal"])
    out["items"] = [it for it in out["items"] if not it["equal"]]        # 只留不一致项
    return out


def cmp_sigma(a, b):
    out = {}
    for stem in a:
        mx, worst, n_int_bad = 0.0, None, 0
        for name, row in a[stem]["rows"].items():
            for sg, v in row.items():
                if sg == "n_sensors":
                    if v != b[stem]["rows"][name][sg]:
                        n_int_bad += 1
                    continue
                w = b[stem]["rows"][name][sg]
                for k in ("median_nonstruct", "mean_nonstruct", "median_target", "median_recovered"):
                    if v[k] is None or w[k] is None:
                        if (v[k] is None) != (w[k] is None):
                            n_int_bad += 1
                        continue
                    d = abs(v[k] - w[k])
                    if d > mx:
                        mx, worst = d, dict(row=name, sigma=sg, key=k, v100=v[k], ref=w[k])
                for k in ("n_half_prior", "n_tenth_prior", "n_recovered"):
                    if v[k] != w[k]:
                        n_int_bad += 1
        out[stem] = dict(max_abs_diff=mx, worst=worst, n_int_mismatch=n_int_bad,
                         n_rows=len(a[stem]["rows"]))
    return out


def merge_leak(paths, logs):
    merged = None
    split = {}
    for tag, p in paths.items():
        d = jload(p)
        if merged is None:
            merged = dict(config=dict(d["config"]), runs={})
        else:
            for k, v in d["config"].items():
                if merged["config"].get(k) != v:
                    raise RuntimeError(f"三卡 config 不一致：{k}: {merged['config'].get(k)} vs {v}（{p}）")
        lg = logs.get(f"ltleak_{tag}", {})
        split[tag] = dict(source=rp(p), configs=list(d["runs"].keys()),
                          cuda_visible_devices=lg.get("cuda_visible_devices"),
                          log=lg.get("log"), elapsed_sec=lg.get("elapsed_sec"), rc=lg.get("rc"))
        for name, r in d["runs"].items():
            if name in merged["runs"]:
                raise RuntimeError(f"配置 {name} 在两份分跑文件里都出现")
            r = dict(r)
            r["v100_split"] = tag
            merged["runs"][name] = r
    merged["runs"] = {n: merged["runs"][n] for n in CFG_ORDER if n in merged["runs"]}
    missing = [n for n in CFG_ORDER if n not in merged["runs"]]
    if missing:
        raise RuntimeError(f"缺配置 {missing}")
    merged["config"]["v100_split"] = split
    merged["config"]["run_note"] = ("V100 服务器（<v100-host>，Tesla V100-SXM2-32GB × 3，"
                                    "csr+cuDSS，chunk=256）三卡分跑后合并；同 seed/噪声/注入点；"
                                    "交叉核对对象 = 主结果 data/ltown_augment_leak.json（集群 5090 "
                                    "作业 1543441）与 data/ltown_augment_leak_local_dense.json（本机 dense）")
    return merged


def cmp_leak(a, b, groups=("noiseless", "noisy")):
    rows, mx_loss, mx_loss1, mx_C, mx_coh = [], 0.0, 0.0, 0.0, 0.0
    same_top5 = same_rank = same_hits = 0
    n = 0
    for name in CFG_ORDER:
        if name not in a["runs"] or name not in b["runs"]:
            continue
        ra, rb = a["runs"][name], b["runs"][name]
        ca, cb = ra["coherence"], rb["coherence"]
        coh = dict(coh_max_absdiff=abs(ca["coh_max"] - cb["coh_max"]),
                   n_pairs_gt_0999_equal=ca["n_pairs_gt_0999"] == cb["n_pairs_gt_0999"],
                   n_orth_equal=ca["n_orthogonal_pairs"] == cb["n_orthogonal_pairs"],
                   rivals_same_node=all(ca["rivals"][t][0] == cb["rivals"][t][0] for t in ca["rivals"]),
                   rivals_max_absdiff=max(abs(ca["rivals"][t][1] - cb["rivals"][t][1]) for t in ca["rivals"]))
        mx_coh = max(mx_coh, coh["coh_max_absdiff"], coh["rivals_max_absdiff"])
        for gp in groups:
            if gp not in ra or gp not in rb:
                continue
            ga, gb = ra[gp], rb[gp]
            n += 1
            r_end, r_1 = rel(ga["loss_end"], gb["loss_end"]), rel(ga["loss_step1"], gb["loss_step1"])
            t5 = [d["node"] for d in ga["top5"]] == [d["node"] for d in gb["top5"]]
            rk = ga["truth_rank"] == gb["truth_rank"]
            hits = (ga["n_true_in_top3"] == gb["n_true_in_top3"] and ga["top1_true"] == gb["top1_true"])
            dC = max(abs(ga["truth_C"][t] - gb["truth_C"][t]) for t in ga["truth_C"])
            same_top5 += t5
            same_rank += rk
            same_hits += hits
            mx_loss, mx_loss1, mx_C = max(mx_loss, r_end), max(mx_loss1, r_1), max(mx_C, dC)
            rows.append(dict(config=name, group=gp, loss_end_v100=ga["loss_end"], loss_end_ref=gb["loss_end"],
                             rel_loss_end=r_end, rel_loss_step1=r_1, top5_same=t5, truth_rank_same=rk,
                             hits_same=hits, truth_C_max_absdiff=dC,
                             t_step_median_v100=ga["t_step_median"], t_step_median_ref=gb["t_step_median"],
                             **{f"coh_{k}": v for k, v in coh.items()}))
    return dict(n_groups=n, max_rel_loss_end=mx_loss, max_rel_loss_step1=mx_loss1,
                max_truth_C_absdiff=mx_C, max_coh_absdiff=mx_coh,
                n_top5_same=same_top5, n_truth_rank_same=same_rank, n_hits_same=same_hits, rows=rows)


def cmp_calib(a, b):
    keys = sorted(k for k in a["runs"] if k.startswith("AUG_") and k in b["runs"])
    rows, mx = [], {}
    for k in keys:
        ra, rb = a["runs"][k], b["runs"][k]
        d = dict(key=k, placement=ra["placement"], sigma=ra["sigma"], noise_seed=ra["noise_seed"])
        for f in ("info_rmse", "sub_rmse", "val_frame_rmse", "val_sensor_rmse", "train_rmse", "mse_train_final"):
            r = rel(ra[f], rb[f])
            d[f"rel_{f}"] = r
            d[f"{f}_v100"], d[f"{f}_ref"] = ra[f], rb[f]
            mx[f] = max(mx.get(f, 0.0), r)
        d["n_unobservable_equal"] = ra["n_unobservable"] == rb["n_unobservable"]
        d["sub_rank_equal"] = ra["sub_rank"] == rb["sub_rank"]
        d["nfe"] = dict(v100=ra["nfe"], ref=rb["nfe"])
        d["t_total_sec"] = dict(v100=ra["t_total_sec"], ref=rb["t_total_sec"])
        rows.append(d)
    # 按布点 × σ 的 3 种子中位（V100 / 参考）
    table = {}
    for pl in sorted(set(r["placement"] for r in rows), key=lambda x: (0 if x == "augS0" else 1, x)):
        table[pl] = {}
        for sg in (0.03, 0.1, 0.3):
            rs = [r for r in rows if r["placement"] == pl and abs(r["sigma"] - sg) < 1e-12]
            if rs:
                table[pl][f"{sg:g}"] = {
                    f: dict(v100=float(np.median([r[f"{f}_v100"] for r in rs])),
                            ref=float(np.median([r[f"{f}_ref"] for r in rs])))
                    for f in ("info_rmse", "sub_rmse", "val_frame_rmse")}
    return dict(n_keys=len(keys), max_rel=mx, n_unob_equal=sum(r["n_unobservable_equal"] for r in rows),
                n_subrank_equal=sum(r["sub_rank_equal"] for r in rows),
                t_total_sec=dict(v100=sum(r["t_total_sec"]["v100"] for r in rows),
                                 ref=sum(r["t_total_sec"]["ref"] for r in rows)),
                median_rel_info_rmse=float(np.median([r["rel_info_rmse"] for r in rows])) if rows else None,
                table=table, rows=rows)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v100-dir", default=os.path.join(DATA, "v100_augpub"))
    ap.add_argument("--logs", default=os.path.join(DATA, "gpu"))
    ap.add_argument("--out", default=os.path.join(DATA, "augment_public_v100_repro.json"))
    ap.add_argument("--merge-out", default=os.path.join(DATA, "ltown_augment_leak_v100.json"))
    ap.add_argument("--calib-out", default=os.path.join(DATA, "calib_gc1_pub_hanoi_v100.json"))
    ap.add_argument("--leak-ref", default=os.path.join(DATA, "ltown_augment_leak.json"),
                    help="主结果（集群 5090 作业 1543441 的记录）")
    ap.add_argument("--leak-local", default=os.path.join(DATA, "ltown_augment_leak_local_dense.json"))
    ap.add_argument("--no-merge", action="store_true", help="只对拍，不重写 --merge-out / --calib-out")
    a = ap.parse_args()
    V = a.v100_dir
    logs = parse_logs(a.logs)
    rep = dict(v100_dir=rp(V), provenance=logs)

    # 1. 漏损反演：合并 + 对拍（三卡输出缺席则跳过）
    parts = {t: os.path.join(V, f"ltown_augment_leak_{t}.json") for t in ("g0", "g1", "g2")}
    parts = {t: p for t, p in parts.items() if os.path.isfile(p)}
    merged = merge_leak(parts, logs) if parts else None
    if merged is not None:
        if not a.no_merge:
            jdump(merged, a.merge_out)
        ref = jload(a.leak_ref)
        rep["leak_vs_5090"] = cmp_leak(merged, ref)
        rep["leak_vs_5090"]["ref"] = rp(a.leak_ref)
        rep["leak_vs_5090"]["ref_gpu"] = ref["config"].get("gpu")
        if os.path.isfile(a.leak_local):
            loc = jload(a.leak_local)
            rep["leak_vs_local_dense"] = cmp_leak(merged, loc)
            rep["leak_vs_local_dense"]["ref"] = rp(a.leak_local)
        rep["leak_merged"] = rp(a.merge_out)
        rep["leak_v100_gpu"] = merged["config"].get("gpu")
        rep["leak_v100_t_step_median"] = {n: dict(noiseless=r["noiseless"]["t_step_median"],
                                                   noisy=r["noisy"]["t_step_median"])
                                          for n, r in merged["runs"].items()}

    # 2. 选点序列 + 增设曲线
    for key, fn in (("orders_ltown", "placement_orders_ltown.npz"),
                    ("orders_hanoi", "placement_orders_pub_hanoi_synth25.npz")):
        if os.path.isfile(os.path.join(V, fn)):
            rep[key] = cmp_orders(os.path.join(V, fn), os.path.join(DATA, fn))
    for key, fn in (("augment_ltown", "placement_augment_ltown.json"),
                    ("augment_hanoi", "placement_augment_pub_hanoi_synth25.json")):
        if os.path.isfile(os.path.join(V, fn)):
            rep[key] = cmp_augment_json(jload(os.path.join(V, fn)), jload(os.path.join(DATA, fn)))
    # 3. σ 梯
    if os.path.isfile(os.path.join(V, "augment_public_sigma_ladder.json")):
        rep["sigma_ladder"] = cmp_sigma(jload(os.path.join(V, "augment_public_sigma_ladder.json")),
                                        jload(os.path.join(DATA, "augment_public_sigma_ladder.json")))
    # 4. Hanoi 标定
    if os.path.isfile(os.path.join(V, "calib_gc1_pub_hanoi.json")):
        cv = jload(os.path.join(V, "calib_gc1_pub_hanoi.json"))
        cv_aug = dict(config=dict(cv["config"]), structural=cv["structural"],
                      runs={k: v for k, v in cv["runs"].items() if k.startswith("AUG_")})
        lg = logs.get("hanoi_calib", {})
        cv_aug["config"]["provenance"] = dict(
            host=lg.get("host"), log=lg.get("log"), elapsed_sec=lg.get("elapsed_sec"),
            commit=lg.get("commit"), torch=lg.get("torch"), python=lg.get("python"),
            cmd=lg.get("cmd"), note="V100 服务器 CPU 跑 calibrate.py --stage l1hanoi_aug（同真值、同噪声"
                                    "种子、同布点），只含 AUG_* 键；参考 = data/calib_gc1_pub_hanoi.json"
                                    "（Windows 工作站）")
        if not a.no_merge:
            jdump(cv_aug, a.calib_out)
        rep["calib_hanoi"] = cmp_calib(cv_aug, jload(os.path.join(DATA, "calib_gc1_pub_hanoi.json")))
        rep["calib_v100"] = rp(a.calib_out)
    jdump(rep, a.out)

    # ---- 打印摘要 ----
    L = []
    L.append(f"V100 复现对拍（{rp(a.out)}）")
    for lab, d in logs.items():
        L.append(f"  作业 {lab:12s} host={d.get('host')} GPU={d.get('cuda_visible_devices')!r} "
                 f"rc={d.get('rc')} 耗时 {d.get('elapsed_sec')}s  commit={d.get('commit', '?')[:7]} "
                 f"torch {d.get('torch')}")
    for key, lab in (("orders_ltown", "L-TOWN"), ("orders_hanoi", "Hanoi")):
        if key in rep:
            o = rep[key]
            L.append(f"  {lab} 选点序列 npz 逐元素相等：{o['all_equal']}"
                     f"（{ {k: v['equal'] for k, v in o['keys'].items()} }）")
    for nm in ("augment_ltown", "augment_hanoi"):
        if nm not in rep:
            continue
        c = rep[nm]
        L.append(f"  {nm}：整数项 {c['n_int_items']} 个不一致 {c['n_int_mismatch']}；实数项最大相对差 "
                 f"{c['max_rel_real']:.2e}（{c['worst']['path'] if c['worst'] else '-'}）"
                 + (f"；sfull ‖F‖∞max V100 {c['sfull']['resid_inf_max']['v100']:.2e} / 参考 "
                    f"{c['sfull']['resid_inf_max']['ref']:.2e}，耗时 {c['sfull']['t_total_sec']['v100']:.0f}s / "
                    f"{c['sfull']['t_total_sec']['ref']:.0f}s"
                    if "sfull" in c else ""))
    for stem, c in rep.get("sigma_ladder", {}).items():
        L.append(f"  σ 梯 {stem}：{c['n_rows']} 行，整数格不一致 {c['n_int_mismatch']}，实数格最大绝对差 "
                 f"{c['max_abs_diff']:.2e}")
    for nm in ("leak_vs_5090", "leak_vs_local_dense"):
        if nm not in rep:
            continue
        c = rep[nm]
        L.append(f"  漏损反演 V100 vs {c['ref']}（{c.get('ref_gpu', '')}）：{c['n_groups']} 组；loss末 最大相对差 "
                 f"{c['max_rel_loss_end']:.2e}，第 1 步 {c['max_rel_loss_step1']:.2e}；top-5 同 "
                 f"{c['n_top5_same']}/{c['n_groups']}，真值名次同 {c['n_truth_rank_same']}/{c['n_groups']}，"
                 f"top-1/top-3 判定同 {c['n_hits_same']}/{c['n_groups']}；真值 C 最大绝对差 "
                 f"{c['max_truth_C_absdiff']:.2e}；相干最大绝对差 {c['max_coh_absdiff']:.2e}")
    if "calib_hanoi" in rep:
        c = rep["calib_hanoi"]
        L.append(f"  Hanoi σ 梯标定 AUG_* {c['n_keys']} 键：unobservable 同 {c['n_unob_equal']}/{c['n_keys']}，"
                 f"sub_rank 同 {c['n_subrank_equal']}/{c['n_keys']}；最大相对差 "
                 + " ".join(f"{k}={v:.2e}" for k, v in c["max_rel"].items())
                 + f"；info-RMSE 相对差中位 {c['median_rel_info_rmse']:.2e}；总耗时 V100 "
                 f"{c['t_total_sec']['v100'] / 60:.1f} min / 参考 {c['t_total_sec']['ref'] / 60:.1f} min")
        L.append(f"  {'布点':>10} | " + " | ".join(f"σ={sg:<4} info-RMSE V100/参考   sub-RMSE V100/参考"
                                                  for sg in ("0.03", "0.1", "0.3")))
        for pl, row in c["table"].items():
            cells = []
            for sg in ("0.03", "0.1", "0.3"):
                if sg in row:
                    v = row[sg]
                    cells.append(f"{v['info_rmse']['v100']:7.3f}/{v['info_rmse']['ref']:7.3f}   "
                                 f"{v['sub_rmse']['v100']:7.3f}/{v['sub_rmse']['ref']:7.3f}")
                else:
                    cells.append(" " * 35)
            L.append(f"  {pl:>10} | " + " | ".join(cells))
    if "leak_vs_5090" in rep:
        L.append(f"  漏损反演逐组（V100 vs 主结果 5090）：")
        L.append(f"  {'配置':>9} {'组':>9} | {'loss末 V100':>12} {'loss末 5090':>12} {'相对差':>8} | top5同 名次同 | "
                 f"{'步时 V100':>8} {'5090':>6}")
        for r in rep["leak_vs_5090"]["rows"]:
            L.append(f"  {r['config']:>9} {r['group']:>9} | {r['loss_end_v100']:12.6e} {r['loss_end_ref']:12.6e} "
                     f"{r['rel_loss_end']:8.1e} | {str(r['top5_same']):>6} {str(r['truth_rank_same']):>5} | "
                     f"{r['t_step_median_v100']:8.2f} {r['t_step_median_ref']:6.2f}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.splitext(a.out)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

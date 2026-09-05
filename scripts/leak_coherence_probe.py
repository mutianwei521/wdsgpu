# -*- coding: utf-8 -*-
"""leak_coherence_probe.py - 论文图 fig_leak_inversion 的"候选相干性"面板实测。

复用 scripts/demo_leak_inversion.py 的 Problem / build_dictionary，按同一随机
种子（SEED=2026）重放传感器抽样与签名字典构造，导出候选签名字典的互相干矩阵
统计。自检：重放得到的 40 个传感器 ID 必须与 data/demo_leak_inversion.json
的 config.sensors 完全一致（否则说明重放偏离，直接报错，不出数）。

互相干定义：D[:,j] = 候选 j 单位用户系数 C 的传感器压力响应（25 帧 x 40 传感器
展平，见 demo_leak_inversion.build_dictionary），列归一化后 coh = Dn^T Dn。
coh[i,j] 即两候选"漏损签名"的夹角余弦 - 接近 1 表示 40 个传感器无法线性区分。

输出：data/leak_coherence.json
运行：python -X utf8 scripts/leak_coherence_probe.py
"""

import json
import os
import sys
import time
import warnings

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message=".*not writable.*")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import demo_leak_inversion as dm      # noqa: E402

OUT = os.path.join(ROOT, "data", "leak_coherence.json")
DEMO = os.path.join(ROOT, "data", "demo_leak_inversion.json")


def main():
    t0 = time.time()
    pb = dm.Problem()
    rng = np.random.default_rng(dm.SEED)
    others = [n for n in pb.cand if n not in ("195", "3083")]
    third = str(rng.choice(others))                 # 消耗与 demo 相同的 rng 状态
    sol_base = pb.fsolve(max_iter=dm.OBS_MI)
    sens = np.sort(rng.choice(pb.s.junc_nodes, dm.N_SENSOR, replace=False))
    sens_ids = [pb.net.node_id[i] for i in sens]

    ref = json.load(open(DEMO, encoding="utf-8"))
    assert sens_ids == ref["config"]["sensors"], "传感器重放不一致，拒绝出数"
    assert pb.cand == ref["config"]["candidates"], "候选集重放不一致，拒绝出数"
    assert third in ref["config"]["true_nodes"], "第三真值节点重放不一致"
    print(f"重放自检 PASS：{len(sens_ids)} 传感器 / {pb.nc} 候选 / 第三真值节点 {third}")

    _D, _Dn, coh = dm.build_dictionary(pb, sol_base, sens)
    nc = pb.nc
    iu, ju = np.triu_indices(nc, 1)
    off = np.abs(coh[iu, ju])
    # 每个候选与"最像它的另一个候选"的相干（列最大离对角）
    A = np.abs(coh).copy()
    np.fill_diagonal(A, -1.0)
    per_cand_max = A.max(axis=1)

    true_nodes = list(ref["config"]["true_nodes"].keys())
    pairs = sorted(
        [(pb.cand[int(i)], pb.cand[int(j)], float(coh[i, j]))
         for i, j in zip(iu, ju)], key=lambda x: -abs(x[2]))

    # 真值节点各自的最强竞争者
    rivals = {}
    for n in true_nodes:
        i = pb.cand.index(n)
        j = int(A[i].argmax())
        rivals[n] = dict(rival=pb.cand[j], coh=float(coh[i, j]))

    res = dict(
        meta=dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                  python=sys.executable, seed=dm.SEED,
                  n_sensor=dm.N_SENSOR, frames=25, n_candidates=nc,
                  probe_C=0.3, elapsed_sec=round(time.time() - t0, 1),
                  definition="coh = Dn^T Dn, Dn = column-normalised sensor-pressure "
                             "signature dictionary (25 frames x 40 sensors)"),
        n_pairs=int(off.size),
        max_offdiag=float(off.max()),
        median_offdiag=float(np.median(off)),
        q90_offdiag=float(np.quantile(off, 0.90)),
        n_pairs_gt_0999=int((off > 0.999).sum()),
        n_pairs_gt_099=int((off > 0.99).sum()),
        n_pairs_gt_09=int((off > 0.9).sum()),
        per_candidate_max_coh=[float(x) for x in per_cand_max],
        candidates=pb.cand,
        true_nodes=true_nodes,
        true_node_rivals=rivals,
        top20_pairs=[dict(a=a, b=b, coh=c) for a, b, c in pairs[:20]],
        offdiag_values=[float(x) for x in off],
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"互相干: max={res['max_offdiag']:.6f} 中位={res['median_offdiag']:.4f} "
          f">0.999 的对数={res['n_pairs_gt_0999']}/{res['n_pairs']}")
    for n, v in rivals.items():
        print(f"  真值 {n} 最强竞争者 {v['rival']}  coh={v['coh']:.6f}")
    print(f"写出 {OUT}（{res['meta']['elapsed_sec']} s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

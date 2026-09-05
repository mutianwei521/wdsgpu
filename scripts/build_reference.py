# -*- coding: utf-8 -*-
"""集成 CLI（integrator）：遍历契约列出的 7 个 INP，逐个执行
    parse_inp → Net.save（写 <stem>_net.npz + <stem>_meta.json）
    build_reference（写 <stem>_ref.npz，并把标量并入 meta.json 的 "reference" 键）
落盘目录：data/reference/。打印进度与耗时。

注意顺序纪律：Net.save 会整体重写 meta.json（不保留旧的 "reference" 键），
而 build_reference 采用读-改-写合并；因此必须**先 save 再 build_reference**，
最终 meta.json 同时含解析标量与 reference 键。
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import parse_inp          # noqa: E402
from dgga.reference import build_reference  # noqa: E402

# 契约管网清单（7 个），路径写死
NETWORKS = [
    (os.path.join(ROOT, "networks", "InpData", "EXA4.inp"),        "EXA4"),
    (os.path.join(ROOT, "networks", "InpData", "EXA5.inp"),        "EXA5"),
    (os.path.join(ROOT, "networks", "InpData", "EXA6.inp"),        "EXA6"),
    (os.path.join(ROOT, "networks", "InpData", "city_h.inp"),     "city_h"),
    (os.path.join(ROOT, "networks", "InpData", "ky3.inp"),         "ky3"),
    (os.path.join(ROOT, "networks", "InpData", "ky5.inp"),         "ky5"),
    (os.path.join(ROOT, "networks", "realInpData", "city_d.inp"),  "city_d"),
]
OUT_DIR = os.path.join(ROOT, "data", "reference")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_all = time.perf_counter()
    rows = []
    for k, (inp, stem) in enumerate(NETWORKS, 1):
        print("=" * 78)
        print(f"[{k}/{len(NETWORKS)}] {stem}: {inp}")
        if not os.path.exists(inp):
            raise FileNotFoundError(inp)

        # ---- 解析 + 落盘 net ----
        t0 = time.perf_counter()
        net = parse_inp(inp)
        net.save(OUT_DIR, stem)
        t1 = time.perf_counter()
        print(f"  parse+save: N={net.N} L={net.L} "
              f"(junction={int(np.sum(net.node_type == 0))} "
              f"reservoir={int(np.sum(net.node_type == 1))} "
              f"tank={int(np.sum(net.node_type == 2))})  耗时 {t1 - t0:.2f}s")

        # ---- 参考解 EPS ----
        res = build_reference(inp, OUT_DIR, stem)
        t2 = time.perf_counter()
        T = len(res["t_sec"])
        it = res["iterations"]
        print(f"  build_reference: 帧数 T={T} (t=0..{int(res['t_sec'][-1])}s)  "
              f"iter [{int(it.min())},{int(it.max())}]  "
              f"relerr max={res['relerr'].max():.2e}  "
              f"警告 {len(res['warnings'])} 条  耗时 {t2 - t1:.2f}s")
        if res["warnings"]:
            for w in res["warnings"][:5]:
                print(f"    警告: t={w[0]}s code={w[1]} {w[2]}")
        rows.append((stem, T, net.N, net.L, int(it.min()), int(it.max()),
                     float(res["relerr"].max()), t2 - t0))

    print("=" * 78)
    print("构建汇总：")
    print(f"{'网络':<10}{'帧数':>6}{'节点':>8}{'管段':>8}{'迭代范围':>12}"
          f"{'relerr_max':>14}{'耗时s':>10}")
    for stem, T, N, L, i0, i1, re_, dt in rows:
        print(f"{stem:<10}{T:>6}{N:>8}{L:>8}{f'[{i0},{i1}]':>12}{re_:>14.2e}{dt:>10.2f}")
    print(f"全部 {len(NETWORKS)} 网构建完成，总耗时 {time.perf_counter() - t_all:.1f}s，"
          f"输出目录 {OUT_DIR}")


if __name__ == "__main__":
    main()

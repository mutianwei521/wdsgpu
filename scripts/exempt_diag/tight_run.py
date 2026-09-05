# -*- coding: utf-8 -*-
"""tight_run.py - 任务 2(b3)：把官方 DLL 的 ACCURACY 收紧到 1e-8、TRIALS 放到
1000，逐帧看参考引擎自己收不收敛（核对归档脚注里"bwsn1/bwsn2 持续振荡"的说法）。

用法: python scripts/exempt_diag/tight_run.py [stem ...]
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cond_struct import tight_probe    # noqa: E402
from align import resolve_inp          # noqa: E402

STEMS = ["pub_net3", "pub_bwsn_network_1", "pub_bwsn_network_2", "pub_net6",
         "pub_c_town_batadal", "ky5"]


def main(stems):
    rows = []
    for s in stems:
        inp = resolve_inp(s)
        if not os.path.isfile(inp):
            continue
        t0 = time.perf_counter()
        for acc, tri in ((1e-3, 40), (1e-8, 1000)):
            try:
                r = tight_probe(inp, acc=acc, trials=tri)
            except Exception as e:                        # noqa: BLE001
                print(f"{s} acc={acc:g}: 失败 {type(e).__name__}: {e}", flush=True)
                continue
            r["stem"] = s
            rows.append(r)
            print(f"{s:<24s} ACC={acc:<8g} TRIALS={tri:<5d} 帧={r['T']:<5d} "
                  f"最大迭代={r['max_iters']:<5d} 触顶帧={r['n_hit_trials']:<4d} "
                  f"未收敛帧={r['n_frames_not_converged']:<5d} "
                  f"max relerr={r['max_relerr']:.3e} 警告={r['n_runH_warnings']}",
                  flush=True)
        print(f"  [{s}] {time.perf_counter()-t0:.0f}s", flush=True)
    p = os.path.join(ROOT, "data", "exempt", "tight_run.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows},
                  f, ensure_ascii=False, indent=1)
    print("写出:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else STEMS))

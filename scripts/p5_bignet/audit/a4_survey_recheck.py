# -*- coding: utf-8 -*-
"""审阅项 4：盘点表可复现吗。**不读 survey_inp.json**，从 .inp 重跑。

(1) 内容哈希去重：sha256 全量；重点核对"同名不同目录"的到底是不是同一个网
    （工单点名：不要用文件名判重）。
(2) 抽查准入判定：>=10 个网独立重跑 parse -> epanet 构造 -> dense 构造 -> B=1 解，
    含 2 个 parse 失败的。
(3) ★ 本轮的树是 HEAD（已含 CVPIPE + 批量状态机），上一轮的表出自 524c3e8。
    所以要专门看：**只卡 CVPIPE、不卡 PRV/PSV/FCV 的网现在进不进得去**。
"""
import hashlib
import os
import sys
import time
import traceback

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                        # noqa: E402
import aud_lib as AL                                      # noqa: E402
import torch                                              # noqa: E402

import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
EX = os.path.join(ROOT, "networks", "EXAMPLE")
for k, v in AL.prov():
    print("   %-18s %s" % (k, v))

# ------------------------------------------------------------------ (1) 哈希
files = []
for dp, dn, fn in os.walk(EX):
    for f in fn:
        if f.lower().endswith(".inp"):
            files.append(os.path.join(dp, f))
files.sort()
print("\n" + "=" * 100)
print("(1) 内容哈希去重 - 文件总数 %d（小写 .inp %d / 大写 .INP %d）"
      % (len(files), sum(1 for f in files if f.endswith(".inp")),
         sum(1 for f in files if f.endswith(".INP"))))
print("=" * 100)
byhash, byname = {}, {}
for f in files:
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    byhash.setdefault(h, []).append(f)
    byname.setdefault(os.path.basename(f).lower(), []).append((f, h))
dups = {h: v for h, v in byhash.items() if len(v) > 1}
print("  唯一 sha256 = %d；完全相同的文件组 = %d" % (len(byhash), len(dups)))
for h, v in sorted(dups.items(), key=lambda x: -len(x[1])):
    print("    [%d 份] %s" % (len(v), " | ".join(os.path.relpath(x, EX) for x in v)))
print("\n  --- 同名不同目录的（工单点名：别用文件名判重）---")
nn = 0
for nm, v in sorted(byname.items()):
    if len(v) > 1:
        nn += 1
        hs = {h for _, h in v}
        print("    %-28s %d 份，字节%s：%s"
              % (nm, len(v), "相同" if len(hs) == 1 else "**不同**",
                 " | ".join(os.path.relpath(p, EX) for p, _ in v)))
print("  同名多份的文件名 = %d 个" % nn)

# ------------------------------------------------------------- (2)(3) 准入抽查
from dgga.parse import parse_inp                          # noqa: E402
from dgga.solver import GGASolver                         # noqa: E402

LT = {0: "CVPIPE", 1: "PIPE", 2: "PUMP", 3: "PRV", 4: "PSV", 5: "PBV",
      6: "FCV", 7: "TCV", 8: "GPV"}
SPOT = [
    # (相对路径, 上一轮表里的 Nj, 上一轮的 dense 判定)
    ("epanet-example-networks/epanet-tests/large/NW_Model.inp", 8566, "ok"),
    ("epanet-example-networks/epanet-tests/large/NW_Model1.inp", 8566, "ok"),
    ("asce-tf-wdst/Battle of the Water Sensor Networks/BWSN_Network_2.inp", 12523, "-"),
    ("pangaea/PacificCity.inp", 8715, "-"),
    ("asce-tf-wdst/ky12/ky12.inp", 2347, "-"),
    ("asce-tf-wdst/ky8/ky8.inp", 1325, "ok*"),
    ("asce-tf-wdst/ky4/ky4.inp", 959, "ok*"),
    ("asce-tf-wdst/KL/KL.inp", 935, "ok"),
    ("asce-tf-wdst/ky1/ky1.inp", 856, "ok"),
    ("asce-tf-wdst/ky13/ky13.inp", 778, "ok"),
    ("L-Town/L-TOWN.inp", 782, "-"),
    ("asce-tf-wdst/ky14/ky14.inp", 377, "-  (CVPIPE5)"),
    ("asce-tf-wdst/Richmond/Richmond_skeleton.inp", 41, "-  (CVPIPE8)"),
    ("epanet-example-networks/epanet-tests/small/sampletown.inp", 6, "-  (CVPIPE1)"),
    ("exeter-benchmarks/D-Town Water Distribution Network BWN-II/Original_BWNII/"
     "1_Wu.inp", 398, "-"),
    ("asce-tf-wdst/Balerma/Balerma.inp", 443, "-  (D-W)"),
    # 两个 parse 失败的
    ("epanet-example-networks/epanet-tests/large/57460.inp", 28899, "parse FAIL"),
    ("asce-tf-wdst/exnet/exnet.inp", None, "parse FAIL"),
    ("asce-tf-wdst/Micropolis_v1/MICROPOLIS_v1.inp", None, "parse FAIL"),
]
print("\n" + "=" * 100)
print("(2)(3) 准入抽查（本轮的树 = HEAD，已含 CVPIPE + 批量状态机）")
print("=" * 100)
print("%-26s %6s %6s %-34s %-9s %-24s %s"
      % ("net", "Nj", "L", "link_type", "上轮dense", "本轮 dense 判定", "备注"))
for rel, nj_old, old in SPOT:
    p = os.path.join(EX, rel)
    name = os.path.basename(rel)
    if not os.path.exists(p):
        alt = [f for f in files if os.path.basename(f).lower() == name.lower()]
        if not alt:
            print("%-26s  文件不存在: %s" % (name, rel))
            continue
        p = alt[0]
    try:
        net = parse_inp(p)
    except Exception as e:                                # noqa: BLE001
        print("%-26s %6s %6s %-34s %-9s %-24s %s"
              % (name, "-", "-", "-", old, "parse FAIL",
                 type(e).__name__ + ": " + str(e).splitlines()[0][:70]))
        continue
    lt = np.asarray(net.link_type)
    hist = "+".join("%s%d" % (LT[int(t)], int((lt == t).sum()))
                    for t in sorted(np.unique(lt)))
    Nj = int((np.asarray(net.node_type) == 0).sum())
    has_cv = bool((lt == 0).any())
    note, verdict = "", ""
    for tank_guard in (True, False):
        for sm in ([True] if has_cv else [False, True]):
            try:
                s = GGASolver(net, device="cpu", dtype=torch.float64,
                              mode="dense", inp_path=p,
                              dense_tank_bound_check=tank_guard,
                              dense_status_machine=sm)
                d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
                rh = AL.fixed_head(net, 0)
                t0 = time.perf_counter()
                with torch.no_grad():
                    o = s.solve(d0, rh, status_machine=sm)
                ms = (time.perf_counter() - t0) * 1e3
                it = int(np.atleast_1d(o["iters"].numpy())[0])
                cv = bool(np.atleast_1d(o["converged"].numpy())[0]) \
                    if "converged" in o else (it > 0)
                verdict = "ok  iters=%d %.0fms" % (it, ms)
                note = ("守卫关 " if not tank_guard else "") + \
                       ("状态机开" if sm else "")
                del s
                break
            except Exception as e:                        # noqa: BLE001
                verdict = "-"
                note = type(e).__name__ + ": " + str(e).splitlines()[0][:60]
        if verdict.startswith("ok"):
            break
    print("%-26s %6d %6d %-34s %-9s %-24s %s"
          % (name, Nj, net.L, hist[:34], old, verdict, note[:70]))
    sys.stdout.flush()
print("\nA4 DONE")

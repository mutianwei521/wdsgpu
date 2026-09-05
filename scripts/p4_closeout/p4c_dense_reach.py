# -*- coding: utf-8 -*-
"""P4 收尾 P1-b：21 个 public 网里，稠密通路到底够得着几个？

README 原写 "Eight ... reach the dense path with default settings, **nine** if
you pass `dense_tank_bound_check=False` (ky4)"。这里真去建一遍：每个
`networks/public/*.inp`（有 `_cleaned/` 版本的用 `_cleaned`）建两次
`GGASolver(mode="dense")` - 一次全缺省，一次只加 `dense_tank_bound_check=False`。

实测（本机 CPU 前台）：缺省 **8** 个（Anytown / Fossolo_poly1 / Hanoi / Modena /
Net1 / Net2 / Net3 / Pescara）；加那个开关再解锁 **2** 个（**Anytown_wntr** 与
ky4）⇒ 合计 **10/21**，不是 9 - README 漏了 Anytown_wntr。与
`data/dense_gap_plan.md` §3 的 "+水池+泵 → 10/21" 一致。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import torch                                                      # noqa: E402
from dgga.parse import parse_inp                                  # noqa: E402
from dgga.solver import GGASolver                                 # noqa: E402

ND = os.path.join(ROOT, "networks", "public")
CL = os.path.join(ND, "_cleaned")


def trybuild(p, **kw):
    try:
        net = parse_inp(p)
        GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                  inp_path=p, **kw)
        return True, ""
    except Exception as e:                                        # noqa: BLE001
        return False, ("%s: %s" % (type(e).__name__, e)).replace("\n", " ")[:110]


inps = sorted(f for f in os.listdir(ND) if f.lower().endswith(".inp"))
print("public/*.inp 共 %d 个（有 _cleaned 的用 _cleaned）" % len(inps))
d, t = [], []
for fn in inps:
    p = (os.path.join(CL, fn) if os.path.exists(os.path.join(CL, fn))
         else os.path.join(ND, fn))
    ok0, _ = trybuild(p)
    ok1, why1 = trybuild(p, dense_tank_bound_check=False)
    (d if ok0 else (t if ok1 else [])).append(fn[:-4])
    print("  %-26s %-16s %s"
          % (fn[:-4],
             "缺省✓" if ok0 else ("+bound=False✓" if ok1 else "×"),
             "" if ok0 else why1))
print("\n缺省可建 %d 个: %s" % (len(d), d))
print("加 dense_tank_bound_check=False 再解锁 %d 个: %s" % (len(t), t))
print("合计 %d / %d" % (len(d) + len(t), len(inps)))

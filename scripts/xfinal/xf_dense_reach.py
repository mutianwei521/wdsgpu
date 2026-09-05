# -*- coding: utf-8 -*-
"""XF-4b（本机前台）：21 个 public/*.inp 各建两遍，自己数"能进稠密批量通路"的网数。

判据我自己定，且比 README 那句更严：不止构造成功，还要真解一批 B=2 出来
（构造期通过、求解期才抛的网不该算"reach"）。分别量
  缺省               GGASolver(mode="dense")
  dense_tank_bound_check=False
并把每个失败网的第一行异常打出来，看它卡的是能力门还是别的。
"""
import os
import sys
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isdir(os.path.join(ROOT, "dgga")):
    ROOT = os.getcwd()
sys.path.insert(0, ROOT)
NETD = os.path.join(ROOT, "networks", "public")

from dgga.parse import parse_inp                            # noqa: E402
from dgga.solver import GGASolver                           # noqa: E402


def try_net(p, tbc):
    kw = {} if tbc else dict(dense_tank_bound_check=False)
    net = parse_inp(p)
    s = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                  inp_path=p, **kw)
    B = 2
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = 0.5 * (np.asarray(net.tank_hmin)[:tn.size]
                         + np.asarray(net.tank_hmax)[:tn.size])
    rh0 = np.nan_to_num(rh0)
    D = torch.as_tensor(np.repeat(d0[None, :], B, 0), dtype=torch.float64)
    R = torch.as_tensor(np.repeat(rh0[None, :], B, 0), dtype=torch.float64)
    with torch.no_grad():
        o = s.solve(D, R)
    return int(s.Nj), int(o["iters"].reshape(-1)[0])


def main():
    files = sorted(f for f in os.listdir(NETD) if f.endswith(".inp"))
    CL = os.path.join(NETD, "_cleaned")
    pick = {f: (os.path.join(CL, f) if os.path.exists(os.path.join(CL, f))
                else os.path.join(NETD, f)) for f in files}
    print("  用 _cleaned 版本的：%s"
          % ", ".join(f for f in files if pick[f].startswith(CL)))
    print("=" * 104)
    print("XF-4b 稠密批量通路可达网数（%d 个 public/*.inp，判据=构造+真解 B=2）"
          % len(files))
    print("=" * 104)
    ok_def, ok_flag, rows = [], [], []
    for f in files:
        p = pick[f]
        r = {}
        for tbc, tag in ((True, "def"), (False, "flag")):
            try:
                nj, it = try_net(p, tbc)
                r[tag] = ("OK Nj=%d it=%d" % (nj, it), True)
            except Exception:                              # noqa: BLE001
                msg = traceback.format_exc().strip().split("\n")[-1]
                r[tag] = (msg[:96], False)
        if r["def"][1]:
            ok_def.append(f[:-4])
        if r["flag"][1]:
            ok_flag.append(f[:-4])
        rows.append((f[:-4], r))
    for name, r in rows:
        print("  %-22s 缺省 %-6s | tbc=False %-6s | %s"
              % (name, "OK" if r["def"][1] else "×",
                 "OK" if r["flag"][1] else "×",
                 (r["flag"][0] if not r["flag"][1] else r["def"][0])))
    extra = [n for n in ok_flag if n not in ok_def]
    print("\n  缺省可达 %d / %d ：%s" % (len(ok_def), len(files), ", ".join(ok_def)))
    print("  加 dense_tank_bound_check=False 可达 %d / %d，多出来的 %d 个：%s"
          % (len(ok_flag), len(files), len(extra), ", ".join(extra) or "无"))
    print("  README 说：缺省 8，加旗标 10（多出 Anytown_wntr 与 ky4）")
    good = (len(ok_def) == 8 and len(ok_flag) == 10
            and sorted(extra) == ["Anytown_wntr", "ky4"])
    print("  核对：%s" % ("对得上" if good else "**对不上**"))
    print("=" * 104)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""偏差归属独立复核：cudss 与 dense 的差，究竟是"稀疏通路算错"还是"cuDSS 自身
运行间不可复现"。全程 use_deterministic_algorithms(True)，含 B=256（上游只到 64）。
另测 refine 步数对偏差的影响，以及残差范数（谁真的解得更准）。
"""
import gc
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                       # noqa: E402
from dgga.solver import GGASolver                      # noqa: E402

DT = torch.float64
NETDIR = os.path.join(ROOT, "p2nets")
FILES = {"Net1": "Net1.inp", "Net2": "Net2.inp", "Net3": "Net3.inp",
         "Pescara": "Pescara.inp", "Modena": "Modena.inp",
         "City_D": "City_D.inp", "ky4": "ky4.inp"}
torch.use_deterministic_algorithms(True)
print(torch.cuda.get_device_name(0), "| torch", torch.__version__,
      "| deterministic=True")


def mk(stem, dev="cuda"):
    f = os.path.join(NETDIR, FILES[stem])
    net = parse_inp(f)
    return net, GGASolver(net, device=dev, dtype=DT, mode="dense", inp_path=f,
                          dense_tank_bound_check=False)


def bc(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax),
                         net.tank_hmin + 0.3 * (net.tank_hmax - net.tank_hmin),
                         net.tank_hmax - 0.3 * (net.tank_hmax - net.tank_hmin))
    return d, rh


def scen(d, rh, B, seed, dev="cuda"):
    g = np.random.default_rng(20260822 + seed)
    D = d[None, :] * g.lognormal(0.0, 0.22, (B, d.size))
    R = rh[None, :] + g.normal(0.0, 1.5, (B, rh.size))
    R = np.where(np.isnan(rh)[None, :], np.nan, R)
    return (torch.as_tensor(D, dtype=DT, device=dev),
            torch.as_tensor(R, dtype=DT, device=dev))


print("\n%-8s %5s | %11s %11s %11s | %11s %11s | %s" %
      ("net", "B", "dense自重跑", "cudss自重跑", "cudss换实例", "|cudss-dense|",
       "|c-d|/自抖动", "判定"))
for stem in ("Net1", "Net2", "Net3", "Pescara", "Modena", "City_D", "ky4"):
    net, s = mk(stem)
    d, rh = bc(net)
    for B in (64, 256):
        try:
            D, R = scen(d, rh, B, 31 + B)
            dh = [s.solve(D, R)["head_ft"] for _ in range(3)]
            ch = [s.solve(D, R, assemble="csr", linear_solver="cudss")["head_ft"]
                  for _ in range(3)]
            # 换实例（重建 solver ⇒ 重新 plan）
            _, s2 = mk(stem)
            c2 = s2.solve(D, R, assemble="csr", linear_solver="cudss")["head_ft"]
            dj = max(float((dh[0] - x).abs().max()) for x in dh[1:])
            cj = max(float((ch[0] - x).abs().max()) for x in ch[1:])
            ci = float((ch[0] - c2).abs().max())
            cd = float((dh[0] - ch[0]).abs().max())
            r = cd / cj if cj > 0 else float("inf")
            print("%-8s %5d | %11.3e %11.3e %11.3e | %11.3e | %11.2f | %s" %
                  (stem, B, dj, cj, ci, cd, r,
                   "差 ≈ cuDSS 自抖动" if (cj > 0 and 0.2 <= r <= 5)
                   else ("cuDSS 自身可复现，差是系统性的" if cj == 0
                         else "量级不符，需追查")))
            del s2
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:                           # noqa: BLE001
            print("%-8s %5d | %s" % (stem, B, repr(e)[:110]))
    del s, net
    gc.collect()
    torch.cuda.empty_cache()

print("\n[R] 谁真的解得更准：末轮线性系统残差 ||A x - b||_inf / ||b||_inf")
print("    做法：用 CPU f64 解同一批场景得参考头 Href，比较两条 GPU 通路的 |H-Href|；")
print("    再看整解收敛后的 relerr（EPANET 口径）")
print("%-8s %5s | %11s %11s | %11s %11s" %
      ("net", "B", "|gpuDen-cpu|", "|cudss-cpu|", "relerr den", "relerr cud"))
for stem in ("Net3", "Pescara", "Modena", "City_D", "ky4"):
    try:
        net, s = mk(stem)
        _, sc = mk(stem, dev="cpu")
        d, rh = bc(net)
        B = 64
        D, R = scen(d, rh, B, 777)
        ref = sc.solve(D.cpu(), R.cpu())
        a = s.solve(D, R)
        b = s.solve(D, R, assemble="csr", linear_solver="cudss")
        print("%-8s %5d | %11.3e %11.3e | %11.3e %11.3e" %
              (stem, B,
               float((a["head_ft"].cpu() - ref["head_ft"]).abs().max()),
               float((b["head_ft"].cpu() - ref["head_ft"]).abs().max()),
               float(a["relerr"].max()), float(b["relerr"].max())))
        del s, sc, net
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:                               # noqa: BLE001
        print("%-8s | %s" % (stem, repr(e)[:110]))
print("ADV3 DONE")

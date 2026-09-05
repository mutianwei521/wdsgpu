# -*- coding: utf-8 -*-
"""P2 追问：
  §A ΔH 归属：cudss / GPU-dense 各自与 **CPU f64 dense 基准** 的距离 + A 的条件数
  §B matrix_type=SPD（cuDSS 走 Cholesky）vs 缺省 GENERAL（带主元 LU）：精度与速度
  §C 小网变慢的成因：整解耗时拆到 装配 / factorize / solve / stack / SpMV
  §D 精化步数的时间代价（refine=0/1/2）
  §E 显存墙倒下的直接后果：ky4 上 dense 与 cudss 各自能吃多大的 B
"""
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                      # noqa: E402
from dgga.solver import GGASolver                     # noqa: E402
from nvmath.sparse.advanced import DirectSolverMatrixType   # noqa: E402

DT = torch.float64
NETDIR = os.path.join(ROOT, "p2nets")
NETS = [("Net1", "Net1.inp"), ("Anytown", "Anytown.inp"), ("Hanoi", "Hanoi.inp"),
        ("Net2", "Net2.inp"), ("Fossolo", "Fossolo_poly1.inp"),
        ("Pescara", "Pescara.inp"), ("Net3", "Net3.inp"), ("Modena", "Modena.inp"),
        ("City_D", "City_D.inp"), ("ky4", "ky4.inp")]
NMAP = dict(NETS)


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, rh


def batchify(d, rh, B, seed, dev):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.6, 1.4, (B, 1)) * g.uniform(0.75, 1.25, (B, d.size))
    R = rh[None, :] + g.uniform(-2.0, 2.0, (B, rh.size))
    R = np.where(np.isnan(rh)[None, :], np.nan, R)
    return (torch.as_tensor(D, dtype=DT, device=dev),
            torch.as_tensor(R, dtype=DT, device=dev))


def mk(stem, dev):
    f = os.path.join(NETDIR, NMAP[stem])
    net = parse_inp(f)
    return net, GGASolver(net, device=dev, dtype=DT, mode="dense", inp_path=f,
                          dense_tank_bound_check=False)


def tg(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def last():
    return traceback.format_exc().strip().split("\n")[-1][:110]


print(torch.cuda.get_device_name(0), "| torch", torch.__version__)
torch.use_deterministic_algorithms(True)
_orig = GGASolver._csr_to_dense
_ORIG_CU = GGASolver._cudss_solve

# =====================================================================
print("\n" + "=" * 108)
print("§A ΔH 归属：两条 GPU 通路各自与 **CPU f64 dense 基准**（回归里与 EPANET "
      "对到 1e-14 的那条）的距离；κ 为最后一轮 A 的 2-范数条件数")
print("%-9s %5s %6s | %11s %11s %11s | %10s | %s" %
      ("net", "B", "iters", "|cudss-cpu|", "|gpuDen-cpu|", "|cudss-gpuD|",
       "κ(A_末轮)", "|H|max"))
for stem, _ in NETS:
    for B in (8,):
        try:
            netc, sc = mk(stem, "cpu")
            netg, sg = mk(stem, "cuda")
            d, rh = boundary(netc)
            Dc, Rc = batchify(d, rh, B, 4242, "cpu")
            Dg, Rg = Dc.cuda(), Rc.cuda()
            hc = sc.solve(Dc, Rc)["head_ft"]
            hd = sg.solve(Dg, Rg)["head_ft"].cpu()
            hs = sg.solve(Dg, Rg, assemble="csr",
                          linear_solver="cudss")["head_ft"].cpu()
            # 末轮 A 的条件数：借 CPU 通路重跑到收敛后再装一次矩阵太绕，
            # 直接对 CPU 解处的 A 估：用 solve 的最后一次装配值（钩子）
            box = {}

            def _hook(self, data, Bn, _o=_orig, _b=box):
                out = _o(self, data, Bn)
                _b["A"] = out
                return out
            GGASolver._csr_to_dense = _hook
            sc.solve(Dc, Rc, assemble="csr")
            GGASolver._csr_to_dense = _orig
            A = box["A"] + torch.diag_embed(torch.zeros(B, sc.Nj, dtype=DT))
            kap = float(torch.linalg.cond(A).max())
            print("%-9s %5d %6d | %11.3e %11.3e %11.3e | %10.3e | %.1f" %
                  (stem, B, int(sc.solve(Dc, Rc)["iters"].max()),
                   float((hs - hc).abs().max()), float((hd - hc).abs().max()),
                   float((hs - hd).abs().max()), kap,
                   float(hc.abs().max())))
            del sc, sg, netc, netg
            torch.cuda.empty_cache()
        except Exception:                              # noqa: BLE001
            print("%-9s %5d | %s" % (stem, B, last()))
            GGASolver._csr_to_dense = _orig
            torch.cuda.empty_cache()

# =====================================================================
print("\n" + "=" * 108)
print("§B matrix_type：GENERAL（缺省，带主元 LU）vs SPD（Cholesky）")
print("%-9s %5s | %13s %13s | %13s %13s | %9s" %
      ("net", "B", "GEN |Δ vs cpu|", "SPD |Δ vs cpu|", "GEN ms/sc", "SPD ms/sc",
       "SPD/GEN"))
for stem in ("Net3", "Pescara", "Modena", "City_D", "ky4"):
    for B in (64, 256):
        try:
            netc, sc = mk(stem, "cpu")
            d, rh = boundary(netc)
            Dc, Rc = batchify(d, rh, B, 808, "cpu")
            hc = sc.solve(Dc, Rc)["head_ft"]
            netg, sg = mk(stem, "cuda")
            Dg, Rg = Dc.cuda(), Rc.cuda()
            res = {}
            for tag, mt in (("GEN", None), ("SPD", DirectSolverMatrixType.SPD)):
                sg.cudss_matrix_type = mt
                h = sg.solve(Dg, Rg, assemble="csr",
                             linear_solver="cudss")["head_ft"].cpu()
                torch.use_deterministic_algorithms(False)
                t = tg(lambda: sg.solve(Dg, Rg, assemble="csr",
                                        linear_solver="cudss")) / B * 1e3
                torch.use_deterministic_algorithms(True)
                res[tag] = (float((h - hc).abs().max()), t)
            print("%-9s %5d | %13.3e %13.3e | %13.5f %13.5f | %8.2fx" %
                  (stem, B, res["GEN"][0], res["SPD"][0], res["GEN"][1],
                   res["SPD"][1], res["SPD"][1] / res["GEN"][1]))
            del sc, sg, netc, netg
            torch.cuda.empty_cache()
        except Exception:                              # noqa: BLE001
            print("%-9s %5d | %s" % (stem, B, last()))
            torch.use_deterministic_algorithms(True)
            torch.cuda.empty_cache()

# =====================================================================
torch.use_deterministic_algorithms(False)
print("\n" + "=" * 108)
print("§C 单轮迭代耗时拆解（B=256，ms/轮，整批不除以 B）")
print("%-9s %6s | %9s %9s %9s %9s %9s | %9s %9s" %
      ("net", "Nj", "csr装配", "factorize", "solve×1", "stack×1", "SpMV×1",
       "稠密装配", "chol+2解"))
for stem in ("Net1", "Net3", "Modena", "City_D", "ky4"):
    try:
        net, s = mk(stem, "cuda")
        d, rh = boundary(net)
        B = 256
        D, R = batchify(d, rh, B, 11, "cuda")
        # 抓真实迭代点的 vals / csr_data / F（随机矩阵可能非正定，factorize 会炸）
        box = {}

        def _asm(self, v, Bn, _o=GGASolver._assemble_csr, _b=box):
            _b["vals"] = v
            return _o(self, v, Bn)

        def _cu(self, data, F, Bn, refine=None, _o=_ORIG_CU, _b=box):
            _b["data"] = data
            _b["F"] = F
            return _o(self, data, F, Bn, refine)
        GGASolver._assemble_csr = _asm
        GGASolver._cudss_solve = _cu
        s.solve(D, R, assemble="csr", linear_solver="cudss")
        GGASolver._assemble_csr = _asm.__defaults__[0]
        GGASolver._cudss_solve = _ORIG_CU
        vals, data, F = box["vals"], box["data"], box["F"]
        t_asm = tg(lambda: s._assemble_csr(vals, B)) * 1e3
        st = [v for k, v in s._cudss_cache.items() if k[0] == B][0]
        st["vals"].copy_(data)
        st["rhs"].copy_(F)
        ds = st["solver"]
        t_fac = tg(lambda: ds.factorize()) * 1e3
        ds.factorize()
        t_sol = tg(lambda: ds.solve()) * 1e3
        xl = ds.solve()
        t_stk = tg(lambda: torch.stack(xl)) * 1e3
        Hj = torch.stack(xl)
        t_spmv = tg(lambda: s._csr_spmv(data, Hj, B)) * 1e3
        Ad = s._csr_to_dense(data, B)
        t_dasm = tg(lambda: torch.zeros(B, s.Nj * s.Nj, dtype=DT, device="cuda")
                    .scatter_add(1, s.A_idx.expand(B, -1), vals)) * 1e3

        def dchol():
            c = torch.linalg.cholesky(Ad)
            Fc = F.unsqueeze(-1)
            H = torch.cholesky_solve(Fc, c)
            for _ in range(2):
                H = H + torch.cholesky_solve(
                    Fc - (Ad * H.transpose(-2, -1)).sum(-1, keepdim=True), c)
            return H
        t_dch = tg(dchol, 3, 1) * 1e3
        print("%-9s %6d | %9.4f %9.4f %9.4f %9.4f %9.4f | %9.4f %9.4f" %
              (stem, s.Nj, t_asm, t_fac, t_sol, t_stk, t_spmv, t_dasm, t_dch))
        del s, net, Ad
        torch.cuda.empty_cache()
    except Exception:                                  # noqa: BLE001
        print("%-9s | %s" % (stem, last()))
        torch.cuda.empty_cache()

# =====================================================================
print("\n" + "=" * 108)
print("§D 精化步数的时间代价（B=256，整解 ms/场景）")
print("%-9s | %11s %11s %11s | %11s" %
      ("net", "refine=0", "refine=1", "refine=2", "dense 基准"))
for stem in ("Net3", "Modena", "City_D", "ky4"):
    try:
        net, s = mk(stem, "cuda")
        d, rh = boundary(net)
        B = 256
        D, R = batchify(d, rh, B, 12, "cuda")
        ts = []
        for r in (0, 1, 2):
            s.cudss_refine = r
            ts.append(tg(lambda: s.solve(D, R, assemble="csr",
                                         linear_solver="cudss")) / B * 1e3)
        s.cudss_refine = 2
        td = tg(lambda: s.solve(D, R)) / B * 1e3
        print("%-9s | %11.5f %11.5f %11.5f | %11.5f" %
              (stem, ts[0], ts[1], ts[2], td))
        del s, net
        torch.cuda.empty_cache()
    except Exception:                                  # noqa: BLE001
        print("%-9s | %s" % (stem, last()))
        torch.cuda.empty_cache()

# =====================================================================
print("\n" + "=" * 108)
print("§E 显存墙倒下的直接后果：ky4 上两条通路各自能吃多大的 B（单卡 32GiB）")
print("%-6s | %-26s | %-26s" % ("B", "dense（峰值 MiB / ms/场景）",
                                "cudss（峰值 MiB / ms/场景）"))
net, s = mk("ky4", "cuda")
d, rh = boundary(net)
for B in (256, 512, 1024, 2048):
    row = [None, None]
    for i, kw in enumerate(({}, dict(assemble="csr", linear_solver="cudss"))):
        try:
            D, R = batchify(d, rh, B, 33, "cuda")
            torch.cuda.empty_cache()
            s.solve(D, R, **kw)
            torch.cuda.reset_peak_memory_stats()
            t = tg(lambda: s.solve(D, R, **kw), 3, 1) / B * 1e3
            row[i] = "%10.1f MiB / %8.4f" % (
                torch.cuda.max_memory_allocated() / 2 ** 20, t)
            del D, R
            torch.cuda.empty_cache()
        except Exception:                              # noqa: BLE001
            row[i] = last()[:26]
            torch.cuda.empty_cache()
    print("%-6d | %-26s | %-26s" % (B, row[0], row[1]))

print("\nP2b 结束")

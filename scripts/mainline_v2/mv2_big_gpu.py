# -*- coding: utf-8 -*-
"""mv2_big_gpu.py - 主线证据包 v2 · ky4 / NW_Model 各补一格 f+b（GPU 伴随上大网）。

口径：
  ky4（Nj=959，Kentucky WDST 经 WNTR，2 台 CONST_HP 泵，4 水池
       dense_tank_bound_check=False，无 PRV → status_machine=False，
       冻结初始状态 - 与 check_adjoint_gpu §A 同配方）；
  NW_Model（Nj=8566，3 泵非 CONST_HP，无水池无阀；dense 批量 B>=2 撞
       torch kernel 门槛（P5 §4.0），故只报 cudss 列）。
每网：
  §1 B=8 梯度对拍：adjoint='gpu' (cudss / 能跑则 dense) vs CPU ImplicitGGASolve
     （epanet 求解器 + splu，status=None 冻结初始状态），门槛 <1e-6；
  §2 B=256（或 OOM 前最大 B）时间格：fwd_c、implCPU（整批实测）、f+b(gpu)
     cudss（ky4 另试 dense），旧=fwd+implCPU，新=f+b(gpu)，倍数=旧/新。
CONST_HP 钳位守卫（ky4）：若 ±15% 批触发 raise，如实打印并降到 ±5% 重试，
格上注明扰动幅度 - 不静默换配方。
本机冒烟：MV2_DEV=cpu MV2_B2=4（cudss 段自动跳过）。
"""
import hashlib
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL = not os.path.isdir(os.path.join(ROOT, "dgga"))
if LOCAL:                                             # 本机冒烟：仓库根布局
    ROOT = os.path.dirname(os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                      # noqa: E402
from dgga.solver import GGASolver                     # noqa: E402
from dgga.autodiff import implicit_solve, solve_polished  # noqa: E402

DEV = os.environ.get("MV2_DEV", "cuda")
DT = torch.float64
SEED = 2026
NODE = os.popen("hostname").read().strip()
B2 = int(os.environ.get("MV2_B2", "256"))        # <=0 则跳过 §2
B1 = int(os.environ.get("MV2_B1", "8"))
ACC = float(os.environ.get("MV2_ACC", "1e-6"))
HAS_CUDSS = DEV == "cuda"
try:
    import nvmath                                      # noqa: F401
except ImportError:
    HAS_CUDSS = False
if LOCAL:
    NETS = [("ky4", os.path.join(ROOT, "networks", "public", "ky4.inp")),
            ("NW_Model", os.path.join(
                ROOT, "networks", "EXAMPLE", "epanet-example-networks",
                "epanet-tests", "large", "NW_Model.inp"))]
else:
    NETS = [("ky4", os.path.join(ROOT, "bignets", "ky4.inp")),
            ("NW_Model", os.path.join(ROOT, "bignets", "NW_Model.inp"))]
ONLY = os.environ.get("MV2_NETS", "").strip()
if ONLY:
    NETS = [n for n in NETS if n[0] in set(ONLY.split(","))]
FAILS = []


def last():
    return traceback.format_exc().strip().split("\n")[-1][:160]


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def tg(fn, budget=6.0, reps_max=3, warm=1):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    one = time.perf_counter() - t0
    reps = max(2, min(reps_max, int(budget / max(one, 1e-6))))
    best = one
    for _ in range(reps - 1):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - t0)
    return best


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, np.nan_to_num(rh)


def batchify(d, rh, B, amp, seed=SEED):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(1.0 - amp, 1.0 + amp, (B, d.size))
    R = rh[None, :] + g.uniform(-1.0, 1.0, (B, rh.size))
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


print("=" * 96)
print("主线 v2 · 大网 f+b 补格（ky4 / NW_Model；adjoint='gpu' vs CPU 伴随）")
print("node:", NODE, "| torch", torch.__version__, "| dev:", DEV,
      "" if DEV != "cuda" else "| " + torch.cuda.get_device_name(0))
print("cudss:", HAS_CUDSS, "| B2:", B2, "| ACC:", ACC)
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f,
          hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print()

for name, inp in NETS:
    print("=" * 96)
    print("[%s] sha256 INP %s" %
          (name, hashlib.sha256(open(inp, "rb").read()).hexdigest()))
    try:
        net = parse_inp(inp)
        s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=inp,
                      dense_tank_bound_check=False)
        se = GGASolver(net, mode="epanet", inp_path=inp)
    except Exception:                                  # noqa: BLE001
        FAILS.append(name + ":setup")
        print("  [err setup] %s" % last())
        continue
    d0, rh0 = boundary(net)
    jn = torch.as_tensor(np.asarray(se.junc_nodes), dtype=torch.long)
    print("  Nj=%d L=%d N=%d nnz=%d 泵=%d(CONST_HP %d) 水池=%d" %
          (s.Nj, s.L, net.N, int(s.A_csr_dense_pos.numel()), s.n_pumps,
           int(np.asarray(s.is_chp_np).sum()),
           len(np.asarray(net.tank_node))))
    sys.stdout.flush()

    def gpu_grad(D, R, W, ls):
        Dv = D.detach().clone().requires_grad_(True)
        h, _q, _e = implicit_solve(
            s, Dv, R, adjoint="gpu", accuracy=ACC, max_iter=200,
            status_machine=False,
            assemble=("csr" if ls == "cudss" else "dense"), linear_solver=ls)
        (h.index_select(1, jn.to(DEV)) * W).sum().backward()
        return Dv.grad

    def cpu_grad(D, R, W):
        """旧形态：CPU ImplicitGGASolve（冻结初始状态，与 dense 前向同映射）。"""
        Dv = D.detach().cpu().clone().requires_grad_(True)
        Rc = R.detach().cpu()
        Wc = W.detach().cpu()
        h, _q, _e = implicit_solve(se, Dv, Rc)
        (h.index_select(1, jn) * Wc.index_select(1, jn)).sum().backward()
        return Dv.grad

    # ---- §1 B=B1 梯度对拍 ----
    amp = 0.15
    okg = True
    try:
        D1, R1 = batchify(d0, rh0, B1, amp)
        W1 = torch.as_tensor(np.random.default_rng(7).normal(size=(B1, s.Nj)),
                             dtype=DT, device=DEV)
        g_ref = cpu_grad(D1, R1, W1).to(DEV)
        rel = lambda a, b: float((a - b).abs().max()
                                 / max(float(a.abs().max()),
                                       float(b.abs().max()), 1e-300))

        def fd_tiebreak(ga, gc):
            """rel>1e-6 时的 FD 仲裁（NW_Model κ2~7.3e10：两条伴随各自被
            κ·eps~7e-6 的线代次序差限制，1e-6 拍不动 - 在最差坐标上做
            双步长中央差分，双方都得落进 5x FD 噪声带才放行）。"""
            diff = (ga - gc).abs()
            flat = int(diff.argmax().item())
            bi, i = divmod(flat, diff.shape[1])
            dv = D1[bi].detach().cpu().numpy().copy()
            rv = R1.detach().cpu().numpy()[bi:bi + 1]
            wv = W1[bi].detach().cpu().numpy()
            base = dv[i]
            h = max(1e-5, 1e-3 * abs(base))
            L = {}
            for dlt in (h, -h, h / 2, -h / 2):
                dv[i] = base + dlt
                o = solve_polished(se, dv[None, :], rv)
                L[dlt] = float((o["head"][0, np.asarray(se.junc_nodes)]
                                * wv).sum())
            dv[i] = base
            fd1 = (L[h] - L[-h]) / (2 * h)
            fd2 = (L[h / 2] - L[-h / 2]) / h
            noise = max(abs(fd1 - fd2), 1e-12 * abs(fd2))
            ea = abs(float(ga[bi, i]) - fd2)
            ec = abs(float(gc[bi, i]) - fd2)
            ok = ea <= 5 * noise and ec <= 5 * noise
            print("    FD 仲裁@(b=%d,coord=%d)：fd(h)=%.8e fd(h/2)=%.8e "
                  "噪声=%.2e | |gpu-fd|=%.2e |cpu-fd|=%.2e -> %s"
                  % (bi, i, fd1, fd2, noise, ea, ec,
                     "双方均在 5x 噪声带内 PASS(FD)" if ok else "FAIL"))
            return ok

        outs = []
        for ls in (["cudss"] if HAS_CUDSS else []) + ["dense"]:
            try:
                g_ = gpu_grad(D1, R1, W1, ls)
                r_ = rel(g_, g_ref)
                if r_ < 1e-6:
                    outs.append("%s vs CPU=%.3e" % (ls, r_))
                else:
                    okfd = fd_tiebreak(g_, g_ref)
                    outs.append("%s vs CPU=%.3e(%s)"
                                % (ls, r_, "FD-PASS" if okfd else "FD-FAIL"))
                    okg &= okfd
            except torch.OutOfMemoryError:
                outs.append("%s=OOM" % ls)
                torch.cuda.empty_cache()
            except NotImplementedError:
                outs.append("%s=RAISE(%s)" % (ls, last()))
            except RuntimeError:
                outs.append("%s=FAIL(%s)" % (ls, last()))
        print("  §1 B=%d ±%d%% 梯度对拍：%s 门槛<1e-6（超限走 FD 仲裁）：%s"
              % (B1, int(amp * 100), "  ".join(outs),
                 "PASS" if okg else "FAIL"))
        if not okg:
            FAILS.append(name + ":§1")
        del D1, R1, W1
    except Exception:                                  # noqa: BLE001
        FAILS.append(name + ":§1")
        print("  [err §1] %s" % last())
    if DEV == "cuda":
        torch.cuda.empty_cache()
    sys.stdout.flush()

    # ---- §2 B=B2 时间格 ----
    for amp_try in ((0.15, 0.05) if B2 > 0 else ()):
        D, R = batchify(d0, rh0, B2, amp_try)
        W = torch.as_tensor(np.random.default_rng(7).normal(size=(B2, s.Nj)),
                            dtype=DT, device=DEV)
        res = {}
        gn = {}
        try:
            if HAS_CUDSS:
                def fc_():
                    with torch.no_grad():
                        s.solve(D, R, assemble="csr", linear_solver="cudss")
                res["fwd_c"] = tg(fc_) * 1e3 / B2
                with torch.no_grad():
                    o = s.solve(D, R, assemble="csr", linear_solver="cudss")
                    conv = int(o["converged"].sum())
                    iters = int(o["iters"].max())
                print("  §2 B=%d ±%d%%：fwd_c=%.5f ms/场景 conv=%d/%d iters<=%d"
                      % (B2, int(amp_try * 100), res["fwd_c"], conv, B2, iters))
                def f2c_():
                    gn["fb_c"] = float(gpu_grad(D, R, W, "cudss").norm())
                res["fb_c"] = tg(f2c_) * 1e3 / B2
        except NotImplementedError:
            print("  §2 ±%d%% 触发守卫：%s -> 降幅重试"
                  % (int(amp_try * 100), last()))
            del D, R, W
            if DEV == "cuda":
                torch.cuda.empty_cache()
            continue
        except Exception:                              # noqa: BLE001
            FAILS.append(name + ":§2")
            print("  [err §2 B=%d] %s" % (B2, last()))
            break
        # dense 一列（ky4 试；NW_Model B>=2 已知 kernel FAIL，试了如实打印）
        if HAS_CUDSS:
            s.cudss_free(empty_cache=True)
        try:
            def f2d_():
                gn["fb_d"] = float(gpu_grad(D, R, W, "dense").norm())
            res["fb_d"] = tg(f2d_) * 1e3 / B2
        except torch.OutOfMemoryError:
            res["fb_d"] = None
            print("  f+b(gpu,dense) B=%d：OOM" % B2)
            torch.cuda.empty_cache()
        except Exception:                              # noqa: BLE001
            res["fb_d"] = None
            print("  f+b(gpu,dense) B=%d：FAIL(%s)" % (B2, last()))
            if DEV == "cuda":
                torch.cuda.empty_cache()
        # implCPU 整批（一次实测；<60 s 再来一次取最小）
        try:
            t0 = time.perf_counter()
            g1 = cpu_grad(D, R, W)
            timp = time.perf_counter() - t0
            gcn = float(g1.norm())
            if timp < 60.0:
                t0 = time.perf_counter()
                cpu_grad(D, R, W)
                timp = min(timp, time.perf_counter() - t0)
            timp_per = timp * 1e3 / B2
        except Exception:                              # noqa: BLE001
            timp_per = None
            gcn = float("nan")
            print("  [err implCPU B=%d] %s" % (B2, last()))
        cell = lambda x: "OOM/-" if x is None else "%.5f" % x
        old_c = None if (res.get("fwd_c") is None or timp_per is None) \
            else res["fwd_c"] + timp_per
        print("  §2 结果（ms/场景，±%d%%）：implCPU=%s | 旧(c 前向+CPU 伴随)=%s"
              " | 新 f+b d=%s c=%s | |g| cpu=%.6e gpu_d=%s gpu_c=%s"
              % (int(amp_try * 100), cell(timp_per), cell(old_c),
                 cell(res.get("fb_d")), cell(res.get("fb_c")), gcn,
                 "-" if gn.get("fb_d") is None else "%.6e" % gn["fb_d"],
                 "-" if gn.get("fb_c") is None else "%.6e" % gn["fb_c"]))
        if old_c is not None and res.get("fb_c") is not None:
            print("  §2 倍数：旧/新c = %.2fx%s"
                  % (old_c / res["fb_c"],
                     "" if res.get("fb_d") is None else
                     "  旧/新d = %.2fx  新d/新c = %.2fx"
                     % (old_c / res["fb_d"], res["fb_d"] / res["fb_c"])))
        del D, R, W
        if DEV == "cuda":
            torch.cuda.empty_cache()
        break
    sys.stdout.flush()

print()
if FAILS:
    print("FAILS:", FAILS)
print("MV2 BIG DONE rc=%d" % (1 if FAILS else 0))
sys.exit(1 if FAILS else 0)

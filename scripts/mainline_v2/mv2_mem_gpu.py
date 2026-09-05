# -*- coding: utf-8 -*-
"""mv2_mem_gpu.py - 主线证据包 v2 · L-TOWN 三列 f+b 显存全表 + OOM 边界。

R5 统一口径（lt_mem_gpu.py 一个字不改）：
    torch_peak  torch.cuda.max_memory_allocated
    torch_resv  torch.cuda.max_memory_reserved
    ctx         CUDA 初始化后、任何张量之前的设备常驻
    nontorch    收尾时设备已用 − torch reserved − ctx
    total       torch_resv + nontorch                   ← 决定 OOM 的量
**每个配置一个全新进程**（父进程只派发，不碰 CUDA）。

模式（与时间表三列一一对应，另留纯前向对照）：
  fwdd / fwdc - 纯前向（状态机，no_grad）；上轮 ltm 表的复测对照。
  oldd - 旧形态完整训练步：dense 前向(no_grad) + CPU 隐式伴随
                 （GPU 四量应与 fwdd 同格 + CPU RSS 峰值增量）。
  fbd / fbc - 新形态完整训练步：implicit_solve(adjoint='gpu') 前向+反向
                 （dense / csr+cudss），loss=head 加权和，backward 到 demand。
OOM 边界：fbd 从主网格最大跑通 B 起细扫（640,768,896,1024）；fbc 倍增扫
  （1536..16384）。oldd 的边界 = fwdd 的边界（伴随在 CPU，§M2 已证），只扫 fwdd。
"""
import os
import subprocess
import sys
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
INP = os.path.join(ROOT, "networks_prv", "L-TOWN.inp")
DEV, DT = "cuda", torch.float64
NODE = os.popen("hostname").read().strip()
SEED = 2026
BS = [int(x) for x in os.environ.get("MV2_BS", "1,8,64,256,512,1024").split(",")]
OLD_BS = [int(x) for x in os.environ.get("MV2_OLD_BS", "1,64,256").split(",")]
SCAN_FBD = [int(x) for x in os.environ.get(
    "MV2_SCAN_FBD", "640,768,896,1024").split(",")]
SCAN_FWDD = [int(x) for x in os.environ.get(
    "MV2_SCAN_FWDD", "1280,1536").split(",")]
SCAN_FBC = [int(x) for x in os.environ.get(
    "MV2_SCAN_FBC", "1536,2048,3072,4096,6144,8192,12288,16384").split(",")]


def err():
    return traceback.format_exc().strip().split("\n")[-1][:140]


def used():
    torch.cuda.synchronize()
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


def resv():
    return torch.cuda.memory_reserved() / 2 ** 20


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + .3 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - .3 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, np.nan_to_num(rh)


def rss_mib():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmHWM"):
                    return int(ln.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def worker(B, mode):
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    from dgga.autodiff import implicit_solve
    net = parse_inp(INP)
    d0, rh0 = boundary(net)
    torch.cuda.init()
    torch.zeros(1, device=DEV)
    torch.cuda.synchronize()
    ctx = used() - resv()
    s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=INP,
                  dense_status_machine=True)
    g = np.random.default_rng(SEED)
    D = torch.as_tensor(d0[None, :] * g.uniform(.85, 1.15, (B, d0.size)),
                        dtype=DT, device=DEV)
    R = torch.as_tensor(rh0[None, :] + g.uniform(-1., 1., (B, rh0.size)),
                        dtype=DT, device=DEV)
    W = torch.as_tensor(np.random.default_rng(7).normal(size=(B, s.Nj)),
                        dtype=DT, device=DEV)
    jn = torch.as_tensor(np.asarray(s.junc_nodes), dtype=torch.long,
                         device=DEV)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        if mode in ("fwdd", "fwdc"):
            kw = dict(assemble="csr", linear_solver="cudss") \
                if mode == "fwdc" else {}
            with torch.no_grad():
                o = s.solve(D, R, status_machine=True, **kw)
            torch.cuda.synchronize()
            extra = "conv=%d/%d iters=%d" % (int(o["converged"].sum()), B,
                                             int(o["iters"].max()))
        elif mode in ("fbd", "fbc"):
            ls = "cudss" if mode == "fbc" else "dense"
            Dv = D.detach().clone().requires_grad_(True)
            h, _q, _e = implicit_solve(
                s, Dv, R, adjoint="gpu", accuracy=1e-6, max_iter=200,
                status_machine=True,
                assemble=("csr" if ls == "cudss" else "dense"),
                linear_solver=ls)
            (h.index_select(1, jn) * W).sum().backward()
            torch.cuda.synchronize()
            extra = "gnorm=%.6e" % float(Dv.grad.norm())
        elif mode == "oldd":
            rss0 = rss_mib()
            with torch.no_grad():
                o = s.solve(D, R, status_machine=True)
            S_all = o["status"].cpu().numpy()
            uniq, inv = np.unique(S_all, axis=0, return_inverse=True)
            se = GGASolver(net, mode="epanet", inp_path=INP)
            K0set = se.run_gga(d0, rh0, do_status=True)["setting"].copy()
            jnc = torch.as_tensor(np.asarray(se.junc_nodes), dtype=torch.long)
            Dc = D.detach().cpu().clone().requires_grad_(True)
            Rc = R.detach().cpu()
            Wc = W.detach().cpu()
            loss = 0.0
            for gi in range(uniq.shape[0]):
                ii = torch.as_tensor(np.where(inv == gi)[0])
                h, _q, _e = implicit_solve(se, Dc.index_select(0, ii),
                                           Rc.index_select(0, ii),
                                           speed=K0set, status=uniq[gi])
                loss = loss + (h.index_select(1, jnc)
                               * Wc.index_select(0, ii)).sum()
            loss.backward()
            torch.cuda.synchronize()
            extra = "gnorm=%.6e rss_hwm_delta=%.1fMiB groups=%d" % (
                float(Dc.grad.norm()), rss_mib() - rss0, uniq.shape[0])
        else:
            raise ValueError(mode)
        tp = torch.cuda.max_memory_allocated() / 2 ** 20
        tr = torch.cuda.max_memory_reserved() / 2 ** 20
        nt = used() - resv() - ctx
        print("RESULT %d %s ok ctx=%.2f torch_peak=%.2f torch_resv=%.2f "
              "nontorch=%.2f total=%.2f %s"
              % (B, mode, ctx, tp, tr, nt, tr + nt, extra))
    except torch.OutOfMemoryError:
        print("RESULT %d %s OOM ctx=%.2f" % (B, mode, ctx))
    except Exception:                                   # noqa: BLE001
        print("RESULT %d %s ERR %s" % (B, mode, err()))


if __name__ == "__main__" and len(sys.argv) > 1:
    worker(int(sys.argv[1]), sys.argv[2])
    raise SystemExit(0)

print("=" * 100)
print("主线 v2 · L-TOWN 三列 f+b 显存全表（R5 口径，MiB，每配置全新进程） | node:",
      NODE)
import hashlib                                          # noqa: E402
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f,
          hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print("sha256 INP",
      hashlib.sha256(open(INP, "rb").read()).hexdigest())
print("B 列:", BS, "| oldd 抽查:", OLD_BS, "| 细扫 fbd:", SCAN_FBD,
      "fwdd:", SCAN_FWDD, "fbc:", SCAN_FBC)
print()


def dispatch(B, mode, timeout=7200):
    cp = subprocess.run([sys.executable, "-X", "utf8", __file__, str(B), mode],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", cwd=ROOT, timeout=timeout)
    line = [x for x in (cp.stdout or "").splitlines() if x.startswith("RESULT")]
    if not line:
        tail = ((cp.stdout or "") + (cp.stderr or "")).strip()
        tail = tail.splitlines()[-1][:120] if tail else "(no output)"
        print("RESULT %d %s CRASH rc=%d %s" % (B, mode, cp.returncode, tail))
        sys.stdout.flush()
        return None
    print(line[-1])
    sys.stdout.flush()
    tok = line[-1].split()
    status = tok[3] if len(tok) > 3 else "ERR"
    if status == "ok":
        return dict(x.split("=", 1) for x in tok if "=" in x)
    return "OOM" if status == "OOM" else None


CELL = {}
for B in BS:
    for mode in ("fwdd", "fwdc", "fbd", "fbc"):
        CELL[(B, mode)] = dispatch(B, mode)
for B in OLD_BS:
    CELL[(B, "oldd")] = dispatch(B, "oldd")

# ---- OOM 细扫 ----
EDGE = {}
for mode, scan in (("fbd", SCAN_FBD), ("fwdd", SCAN_FWDD), ("fbc", SCAN_FBC)):
    ok_max = max([B for B in BS if isinstance(CELL.get((B, mode)), dict)],
                 default=None)
    oom_min = min([B for B in BS if CELL.get((B, mode)) == "OOM"],
                  default=None)
    for B in scan:
        if oom_min is not None and B >= oom_min:
            break
        r = dispatch(B, mode)
        CELL[(B, mode)] = r
        if isinstance(r, dict):
            ok_max = max(ok_max or 0, B)
        elif r == "OOM":
            oom_min = B
            break
        else:
            break
    EDGE[mode] = (ok_max, oom_min)

print()
print("=" * 100)
print("§M1 显存全表（R5 口径；三列 f+b + 纯前向对照）  node=%s" % NODE)
print("B     | fwdd total | fwdc total | fbd  peak      resv     非torch   "
      "total    | fbc  peak      resv     非torch   total    | fbd/fbc")
for B in BS:
    a, b = CELL.get((B, "fwdd")), CELL.get((B, "fwdc"))
    x, y = CELL.get((B, "fbd")), CELL.get((B, "fbc"))

    def tot(v):
        if v is None:
            return "  CRASH  "
        if v == "OOM":
            return "   OOM   "
        return "%9.1f" % float(v["total"])

    def four(v):
        if v is None:
            return "   CRASH                                 "
        if v == "OOM":
            return "   OOM                                   "
        return "%9.1f %9.1f %8.1f %9.1f" % (
            float(v["torch_peak"]), float(v["torch_resv"]),
            float(v["nontorch"]), float(v["total"]))
    r = "   -   "
    if isinstance(x, dict) and isinstance(y, dict):
        r = "%6.2fx" % (float(x["total"]) / float(y["total"]))
    elif x == "OOM" and isinstance(y, dict):
        r = "OOM/ok "
    print("%-5d | %s | %s | %s | %s | %s"
          % (B, tot(a), tot(b), four(x), four(y), r))

print()
print("§M2 oldd 抽查（旧形态完整训练步：dense 前向 + CPU 隐式伴随）  node=%s" % NODE)
for B in OLD_BS:
    x = CELL.get((B, "oldd"))
    print("B=%-5d %s" % (B, x if not isinstance(x, dict) else
                         " ".join("%s=%s" % kv for kv in x.items())))

print()
print("§M3 OOM 边界（每候选 B 全新进程）  node=%s" % NODE)
for mode in ("fbd", "fwdd", "fbc"):
    print("%-6s 最大跑通 B=%s  首个 OOM B=%s" %
          (mode, EDGE[mode][0], EDGE[mode][1]))
print("MV2 MEM DONE")

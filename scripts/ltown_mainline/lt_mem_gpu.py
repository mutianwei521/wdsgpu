# -*- coding: utf-8 -*-
"""L-TOWN 主线证据包 · 任务二（GPU 全表，显存半）。R5 统一口径，一个字不改：

    torch_peak  torch.cuda.max_memory_allocated
    torch_resv  torch.cuda.max_memory_reserved
    ctx         CUDA 初始化后、任何张量之前的设备常驻（历史两节点实测 588.19 MiB）
    nontorch    收尾时设备已用 − torch reserved − ctx
    total       torch_resv + nontorch                   ← 决定 OOM 的量

**每个配置一个全新进程**（父进程只派发，不碰 CUDA）。

模式：
  dense / cudss - 纯前向（状态机，no_grad）。L-TOWN 的实际训练形态里 GPU 侧
                    只有这一段（反向 = ImplicitGGASolve 的 CPU 伴随，见时间脚本），
                    所以 GPU 显存表就是前向表。
  impl_dense - 前向(dense) + 隐式伴随 f+b 的完整训练步：GPU R5 四量 + CPU
                    RSS 峰值增量（证"伴随不吃 GPU 显存"这句话，抽 B∈{1,64,256}）。
OOM 边界：主网格外，对 dense/cudss 前向从 1024 起倍增细扫（1536,2048,3072,...），
  每个候选 B 全新进程，报"最大跑通 / 首个 OOM"。
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
BS = [int(x) for x in os.environ.get("LT_BS", "1,8,64,256,512,1024").split(",")]
SCAN = [int(x) for x in os.environ.get(
    "LT_SCAN", "1536,2048,3072,4096,6144,8192,12288,16384").split(",")]
IMPL_BS = [int(x) for x in os.environ.get("LT_IMPL_BS", "1,64,256").split(",")]


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
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        if mode in ("dense", "cudss"):
            kw = dict(assemble="csr", linear_solver="cudss") \
                if mode == "cudss" else {}
            with torch.no_grad():
                o = s.solve(D, R, status_machine=True, **kw)
            torch.cuda.synchronize()
            extra = "conv=%d/%d iters=%d" % (int(o["converged"].sum()), B,
                                             int(o["iters"].max()))
        elif mode == "impl_dense":
            rss0 = rss_mib()
            with torch.no_grad():
                o = s.solve(D, R, status_machine=True)
            S_all = o["status"].cpu().numpy()
            uniq, inv = np.unique(S_all, axis=0, return_inverse=True)
            se = GGASolver(net, mode="epanet", inp_path=INP)
            K0set = se.run_gga(d0, rh0, do_status=True)["setting"].copy()
            jn = torch.as_tensor(np.asarray(se.junc_nodes), dtype=torch.long)
            Dc = D.detach().cpu().clone().requires_grad_(True)
            Rc = R.detach().cpu()
            Wc = torch.as_tensor(np.random.default_rng(7).normal(
                size=(B, net.N)), dtype=DT)
            loss = 0.0
            for gi in range(uniq.shape[0]):
                ii = torch.as_tensor(np.where(inv == gi)[0])
                h, _q, _e = implicit_solve(se, Dc.index_select(0, ii),
                                           Rc.index_select(0, ii),
                                           speed=K0set, status=uniq[gi])
                loss = loss + (h.index_select(1, jn) *
                               Wc.index_select(0, ii).index_select(1, jn)).sum()
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
print("L-TOWN 主线 · 显存全表（R5 统一口径，MiB，每配置全新进程） | node:", NODE)
import hashlib                                          # noqa: E402
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f,
          hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print("sha256 INP",
      hashlib.sha256(open(INP, "rb").read()).hexdigest())
print("B 列:", BS, "| OOM 细扫:", SCAN, "| impl 抽查:", IMPL_BS)
print()


def dispatch(B, mode, timeout=3600):
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
    for mode in ("dense", "cudss"):
        CELL[(B, mode)] = dispatch(B, mode)
for B in IMPL_BS:
    CELL[(B, "impl_dense")] = dispatch(B, "impl_dense", timeout=7200)

# ---- OOM 细扫（前向；从主网格最大 B 之后倍增）----
EDGE = {}
for mode in ("dense", "cudss"):
    ok_max = max([B for B in BS if isinstance(CELL.get((B, mode)), dict)],
                 default=None)
    oom_min = min([B for B in BS if CELL.get((B, mode)) == "OOM"],
                  default=None)
    if oom_min is None:
        for B in SCAN:
            r = dispatch(B, mode)
            if isinstance(r, dict):
                ok_max = B
            elif r == "OOM":
                oom_min = B
                break
            else:
                break
    EDGE[mode] = (ok_max, oom_min)

print()
print("=" * 100)
print("§M1 显存全表（R5 口径；纯前向=SM，训练形态的 GPU 段）  node=%s" % NODE)
print("B     | dense peak     resv     非torch   total    | "
      "cudss peak    resv     非torch   total    | total 倍数")
for B in BS:
    a, b = CELL.get((B, "dense")), CELL.get((B, "cudss"))

    def four(x):
        if x is None:
            return "   CRASH                                 "
        if x == "OOM":
            return "   OOM                                   "
        return "%9.1f %9.1f %8.1f %9.1f" % (
            float(x["torch_peak"]), float(x["torch_resv"]),
            float(x["nontorch"]), float(x["total"]))
    r = "   -   "
    if isinstance(a, dict) and isinstance(b, dict):
        r = "%6.2fx" % (float(a["total"]) / float(b["total"]))
    elif a == "OOM":
        r = "OOM/ok "
    print("%-5d | %s | %s | %s" % (B, four(a), four(b), r))

print()
print("§M2 impl_dense 抽查（完整训练步：GPU 前向 + CPU 隐式伴随）  node=%s" % NODE)
for B in IMPL_BS:
    x = CELL.get((B, "impl_dense"))
    print("B=%-5d %s" % (B, x if not isinstance(x, dict) else
                         " ".join("%s=%s" % kv for kv in x.items())))

print()
print("§M3 OOM 边界（纯前向，每候选 B 全新进程）  node=%s" % NODE)
for mode in ("dense", "cudss"):
    print("%-6s 最大跑通 B=%s  首个 OOM B=%s" %
          (mode, EDGE[mode][0], EDGE[mode][1]))
print("LT MEM DONE")

# -*- coding: utf-8 -*-
"""P4 全面重测 - R7：ragged + slots>1 的逐出压测（任务三）。

D2 的悬崖（槽被 LRU 逐出 ⇒ 反向变成"重建 DirectSolver + replan"，ky4 B=128
实测 52.5x）在合成压力下已量过；R7 要的是**真实训练形态**：带 Adam 优化器、
带分桶 loader、逐 step 计时，并给出"什么参数配什么训练形态"的建议表。

可微参数取 **ke（漏损/emitter 系数）**，不取 demand - 审计 R4 已判定
solve_unrolled 的 demand 梯度在 ky4/City_D/Net3 上不可用（见 autodiff 模块
docstring）。逐出与否只关线性解，与选哪个参数无关。

训练形态（六种，都是现实里会出现的）：
  S1 fixed        单一批量，前向完立刻反向                （最省心）
  S2 bucket4      4 个桶轮转，前向完立刻反向
  S3 bucket16     16 个桶轮转，前向完立刻反向
  S4 accum4       4 个不同批量的微批累积后一次反向
  S5 accum12      12 个不同批量的微批累积后一次反向        （踩法①）
  S6 val-interl   训练批前向 → 9 个不同批量的验证前向 → 再反向（踩法③）

参数组合：(cudss_cache_max, cudss_grad_slots)
  (8,1) 缺省 / (16,1) / (None,1) / (8,4) 踩法④ / (32,8) / (64,1)

每个 (net, 形态, 参数) 跑两轮同样的 step 列表：第 1 轮冷（含 plan），
第 2 轮热（稳态）。报两轮总时、热轮的 中位/p90/最大 step 时间、计数器增量、
以及该格的**设备增量**（本格开始前 / 收尾时 mem_get_info 之差，含 cuDSS
自己那块） - 用增量是因为 cudss_free() 只把一小部分还给驱动（§10.2.4），
同一进程里前一格的常驻会污染绝对值。
"""
import os
import statistics as st
import sys
import time
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                       # noqa: E402
from dgga.solver import GGASolver                      # noqa: E402
from dgga.autodiff import solve_unrolled               # noqa: E402

DEV, DT = "cuda", torch.float64
NETD = os.path.join(ROOT, "p2nets")
NODE = os.popen("hostname").read().strip()
SEED = 2026
NETS = [x for x in os.environ.get("P4R_R7_NETS", "Modena,ky4").split(",") if x]
FILE = {"Modena": "Modena.inp", "ky4": "ky4.inp", "City_D": "City_D.inp",
        "Net3": "Net3.inp"}

BUCKETS = [32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240,
           256, 272]

SHAPES = [
    ("S1 fixed", "imm", [128]),
    ("S2 bucket4", "imm", BUCKETS[:4]),
    ("S3 bucket16", "imm", BUCKETS),
    ("S4 accum4", "accum", BUCKETS[:4]),
    ("S5 accum12", "accum", BUCKETS[:12]),
    ("S6 val-interl", "val", BUCKETS[:10]),
]
PARAMS = [(8, 1), (16, 1), (None, 1), (8, 4), (32, 8), (64, 1)]
STEPS = int(os.environ.get("P4R_R7_STEPS", "6"))


def last():
    return traceback.format_exc().strip().split("\n")[-1][:120]


# cudss_counters() 不记 plan 次数（它只记 factorize/solve/bwd_*），而 R7 要看的
# 正是"槽被逐出后重建 DirectSolver + 重做 plan"。这里在**脚本层**给
# _cudss_state 挂一个只读计数器：每返回一个全新的 state 字典就 +1。dgga 不改。
_ORIG_STATE = GGASolver._cudss_state
PLANC = {"n": 0}


def _state_counted(self, *a, **k):
    st = _ORIG_STATE(self, *a, **k)
    if "_p4r_seen" not in st:
        st["_p4r_seen"] = True
        PLANC["n"] += 1
    return st


GGASolver._cudss_state = _state_counted


def used():
    torch.cuda.synchronize()
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


def boundary(net):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + .3 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - .3 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, np.nan_to_num(rh)


torch.cuda.init()
torch.zeros(1, device=DEV)
CTX = used() - torch.cuda.memory_reserved() / 2 ** 20
print("=" * 104)
print("P4 R7 · ragged + slots>1 逐出压测 | node:", NODE, "| torch",
      torch.__version__, "|", torch.cuda.get_device_name(0), "| ctx %.2f MiB" % CTX)
import nvmath                                          # noqa: E402
import hashlib                                         # noqa: E402
print("nvmath", nvmath.__version__, "| steps/轮 =", STEPS, "| 桶:", BUCKETS)
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f,
          hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print()

TAB = []
for stem in NETS:
    p = os.path.join(NETD, FILE[stem])
    net = parse_inp(p)
    d0, rh0 = boundary(net)
    g = np.random.default_rng(SEED)
    POOL = {}
    for B in sorted(set(sum([sh[2] for sh in SHAPES], []))):
        D = torch.as_tensor(d0[None, :] * g.uniform(.85, 1.15, (B, d0.size)),
                            dtype=DT, device=DEV)
        R = torch.as_tensor(rh0[None, :] + g.uniform(-1., 1., (B, rh0.size)),
                            dtype=DT, device=DEV)
        POOL[B] = (D, R)
    s0 = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=p,
                   dense_tank_bound_check=False)
    with torch.no_grad():
        K = int(s0.solve(*POOL[32])["iters"].max())
    junc = np.asarray(s0.junc_nodes)
    em = np.random.default_rng(SEED + 1).choice(junc, size=min(60, junc.size),
                                                replace=False)
    print("### %-7s Nj=%d  K=%d  emitter 节点 %d 个" % (stem, s0.Nj, K, em.size))
    del s0
    torch.cuda.empty_cache()

    for shname, kind, bl in SHAPES:
        for cap, slots in PARAMS:
            tagp = "cap=%-4s slots=%d" % ("None" if cap is None else cap, slots)
            try:
                s = GGASolver(net, device=DEV, dtype=DT, mode="dense",
                              inp_path=p, dense_tank_bound_check=False)
                s.cudss_cache_max = cap
                s.cudss_grad_slots = slots
                theta = torch.zeros(net.N, dtype=DT, device=DEV,
                                    requires_grad=True)
                mask = torch.zeros(net.N, dtype=DT, device=DEV)
                mask[torch.as_tensor(em, dtype=torch.long, device=DEV)] = 1.0
                opt = torch.optim.Adam([theta], lr=1e-3)
                base = 0.5

                def fwd(B):
                    D, R = POOL[B]
                    ke = (base + 0.1 * torch.tanh(theta)) * mask
                    o = solve_unrolled(s, D, R, ke=ke, K=K,
                                       assemble="csr", linear_solver="cudss")
                    return o["head_ft"]

                def step(i):
                    if kind == "imm":
                        B = bl[i % len(bl)]
                        loss = (fwd(B) ** 2).mean()
                    elif kind == "accum":
                        loss = 0.0
                        for B in bl:
                            loss = loss + (fwd(B) ** 2).mean()
                    else:                      # val 交叉：反向前塞验证前向
                        B = bl[i % len(bl)]
                        h = fwd(B)
                        with torch.no_grad():
                            for Bv in bl:
                                if Bv != B:
                                    fwd(Bv)
                        loss = (h ** 2).mean()
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    return float(loss.detach())

                # 逐出/形态之间只比**增量**：cuDSS free() 只把一小部分还给驱动
                # （§10.2.4），同一进程里前一格的常驻会污染绝对值。
                torch.cuda.empty_cache()
                dev0 = used()
                # 立刻反向的形态：步数至少走遍所有桶，否则 bucket16 会退化成 bucket-STEPS
                nstep = max(STEPS, len(bl)) if kind == "imm" else STEPS
                s.cudss_counters(reset=True)
                PLANC["n"] = 0
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for i in range(nstep):
                    step(i)
                torch.cuda.synchronize()
                cold = time.perf_counter() - t0
                c_cold = s.cudss_counters(reset=True)
                plan_cold = PLANC["n"]
                PLANC["n"] = 0
                per = []
                for i in range(nstep):
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    step(i)
                    torch.cuda.synchronize()
                    per.append((time.perf_counter() - t1) * 1e3)
                warm = sum(per) / 1e3
                c_warm = s.cudss_counters(reset=False)
                c_warm["plan"] = PLANC["n"]
                c_warm["plan_cold"] = plan_cold
                nstate = len(s.cudss_cache_info())
                dev_mib = used() - dev0
                med, mx = st.median(per), max(per)
                p90 = sorted(per)[max(0, int(0.9 * len(per)) - 1)]
                TAB.append((stem, shname, tagp, cold, warm, med, p90, mx,
                            c_warm, nstate, dev_mib))
                print("  %-13s %-18s | 冷 %8.3f s (plan %3d) | 热 %8.3f s | step 中位"
                      " %9.2f p90 %9.2f max %9.2f ms | plan %-4s fact %-5s"
                      " bwd_reuse %-5s bwd_refact %-5s | states %-3d | 设备增量 %8.1f MiB"
                      % (shname, tagp, cold, plan_cold, warm, med, p90, mx,
                         c_warm.get("plan"), c_warm.get("factorize"),
                         c_warm.get("bwd_reuse"),
                         c_warm.get("bwd_refactorize"), nstate, dev_mib))
                sys.stdout.flush()
                s.cudss_free(empty_cache=True)
                del s, theta, opt
                torch.cuda.empty_cache()
            except torch.OutOfMemoryError:
                print("  %-13s %-18s | OOM" % (shname, tagp))
                TAB.append((stem, shname, tagp, None, None, None, None, None,
                            {}, 0, None))
                torch.cuda.empty_cache()
            except Exception:                          # noqa: BLE001
                print("  %-13s %-18s | ERR %s" % (shname, tagp, last()))
                torch.cuda.empty_cache()
    print()

print("=" * 104)
print("§R7-1 全表  node=%s" % NODE)
print("net     形态           参数              | 冷(s)  plan冷 | 热(s)    中位ms"
      "    p90ms     maxms    | plan fact bwdreuse bwdref | states 设备增量MiB")
for r in TAB:
    (stem, sh, tp, cold, warm, med, p90, mx, c, ns, dv) = r
    if cold is None:
        print("%-7s %-13s %-18s | OOM" % (stem, sh, tp))
        continue
    print("%-7s %-13s %-18s | %6.3f %5s | %7.3f %8.2f %8.2f %9.2f | %4s %4s %8s"
          " %6s | %5d %8.1f"
          % (stem, sh, tp, cold, c.get("plan_cold"), warm, med, p90, mx,
             c.get("plan"), c.get("factorize"), c.get("bwd_reuse"),
             c.get("bwd_refactorize"), ns, dv))

print()
print("§R7-2 相对缺省 (8,1) 的热轮倍数（同网同形态）  node=%s" % NODE)
base = {}
for r in TAB:
    if r[2].startswith("cap=8 ") and r[2].endswith("slots=1") and r[4] is not None:
        base[(r[0], r[1])] = r[4]
print("net     形态           参数              | 热(s)    ÷缺省    设备MiB  ÷缺省")
bmem = {}
for r in TAB:
    if r[2].startswith("cap=8 ") and r[2].endswith("slots=1") and r[10] is not None:
        bmem[(r[0], r[1])] = r[10]
for r in TAB:
    (stem, sh, tp, cold, warm, med, p90, mx, c, ns, dv) = r
    if warm is None:
        continue
    b = base.get((stem, sh))
    bm = bmem.get((stem, sh))
    print("%-7s %-13s %-18s | %7.3f %7s  %8.1f %7s"
          % (stem, sh, tp, warm,
             "-" if not b else "%.2fx" % (warm / b), dv,
             "-" if not bm else "%.2fx" % (dv / bm)))
print("P4R R7 DONE")

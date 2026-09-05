# -*- coding: utf-8 -*-
"""mv2_train_gpu.py - 主线证据包 v2 · L-TOWN 训练形态压测（真实反演循环）。

配方（demo_leak_inversion.py 的 L-TOWN 版，单阶段 Adam）：
  · 合成真值：60 个候选 junction 里挑 3 个挂 emitter（用户系数 C_true=1.5，
    约 8 CMH @30 m；Ke_int = ucf_e / C^Qexp，input1.c:567-573 同链）；
    观测 = B 个需水场景（±15%，水库 ±1 ft）× 33 个随机传感器 junction 的 head，
    由 cudss 前向（accuracy=1e-6 + 状态机）生成。
  · 反演：theta ∈ R^60，C = softplus(theta)（C>0 链式可微，C 小 = 无漏损），
    损失 = 传感器 head MSE (ft^2) + 1e-4 * sum(C)；Adam(lr=0.1) STEPS=60 步。
    每步前向 = implicit_solve(adjoint='gpu', csr+cudss, 状态机)，反向 = GPU
    约化伴随。**缓存规则（sparse_gpu_plan.md §11.4）**：一步之内经过求解器的
    不同批量数(1) × cudss_grad_slots(1) <= cudss_cache_max(8) - 缺省即满足，
    不调参；计数器逐步打印坐实（稳态 plan=0、factorize=迭代+polish+1、
    bwd factorize 增量=0）。
  · 报告：ms/step（首步含 plan 预热单列；稳态=第 6..STEPS 步中位数）、R5 显存
    四量、损失轨迹（首/末）、top-5 恢复的 C（对照真值支撑）、旧形态（CPU 伴随）
    同损失 2 步对照。
本机冒烟：MV2_DEV=cpu MV2_TB=4 MV2_STEPS=5（cudss 换 dense）。
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
if not os.path.isdir(os.path.join(ROOT, "dgga")):     # 本机冒烟：仓库根布局
    ROOT = os.path.dirname(os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
from dgga.parse import parse_inp                      # noqa: E402
from dgga.solver import GGASolver                     # noqa: E402
from dgga.autodiff import implicit_solve              # noqa: E402
from dgga.units import FLOW_UCF, MperFT               # noqa: E402

DEV = os.environ.get("MV2_DEV", "cuda")
DT = torch.float64
INP = os.path.join(ROOT, "networks_prv", "L-TOWN.inp")
if not os.path.exists(INP):
    INP = os.path.join(ROOT, "networks", "public", "_cleaned", "L-TOWN.inp")
SEED = 2026
NODE = os.popen("hostname").read().strip()
TBS = [int(x) for x in os.environ.get("MV2_TB", "256,1024").split(",")]
STEPS = int(os.environ.get("MV2_STEPS", "60"))
OLD_STEPS = int(os.environ.get("MV2_OLD_STEPS", "2"))
LR = float(os.environ.get("MV2_LR", "0.1"))
ACC = float(os.environ.get("MV2_ACC", "1e-6"))
HAS_CUDSS = DEV == "cuda"
try:
    import nvmath                                      # noqa: F401
except ImportError:
    HAS_CUDSS = False
LS = "cudss" if HAS_CUDSS else "dense"
ASM = "csr" if HAS_CUDSS else "dense"
FAILS = []


def last():
    return traceback.format_exc().strip().split("\n")[-1][:160]


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def used():
    torch.cuda.synchronize()
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 2 ** 20


def resv():
    return torch.cuda.memory_reserved() / 2 ** 20


def softplus_inv(x):
    return float(np.log(np.expm1(x)))


print("=" * 96)
print("主线 v2 · L-TOWN 训练形态压测（Adam 反演循环，adjoint='gpu'）")
print("node:", NODE, "| torch", torch.__version__, "| dev:", DEV,
      "" if DEV != "cuda" else "| " + torch.cuda.get_device_name(0))
print("cudss:", HAS_CUDSS, "| B:", TBS, "| STEPS:", STEPS, "| ACC:", ACC)
for _f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", _f,
          hashlib.md5(open(os.path.join(ROOT, _f), "rb").read()).hexdigest())
print("sha256 INP", hashlib.sha256(open(INP, "rb").read()).hexdigest())
print()

CTX = None
if DEV == "cuda":
    torch.cuda.init()
    torch.zeros(1, device=DEV)
    torch.cuda.synchronize()
    CTX = used() - resv()

net = parse_inp(INP)
s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=INP,
              dense_status_machine=True)
se = GGASolver(net, mode="epanet", inp_path=INP)
d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
tn = np.asarray(net.tank_node, dtype=np.int64)
if tn.size:
    lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
    hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
    rh0[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
base = se.run_gga(d0, rh0, do_status=True)
K0set = base["setting"].copy()
jn_np = np.asarray(se.junc_nodes)
jn = torch.as_tensor(jn_np, dtype=torch.long)

# ---- 漏损参数化（input1.c:567-573 同链）----
qexp = float(net.meta["qexp"])
spgrav = float(net.meta["spgrav"])
press = str(net.meta.get("press_units", "") or "METERS")
KPAperPSI, PSIperFT = 6.895, 0.4333
pcf = KPAperPSI * PSIperFT * spgrav if press == "KPA" else MperFT * spgrav
qcf = FLOW_UCF[str(net.meta["flow_units"])]
ucf_e = qcf ** qexp / pcf
print("### L-TOWN Nj=%d | qexp=%g qcf=%g pcf=%g ucf_e=%g | "
      "cudss_cache_max=%s cudss_grad_slots=%s（缺省；规则：一步内不同批量数 1 "
      "x slots 1 <= cap 8 满足，不调参）"
      % (s.Nj, qexp, qcf, pcf, ucf_e,
         getattr(s, "cudss_cache_max", "-"), getattr(s, "cudss_grad_slots", "-")))

rng = np.random.default_rng(909)
NC, NS = 60, 33
cand = np.sort(rng.choice(jn_np, size=NC, replace=False))
truth_pos = rng.choice(NC, size=3, replace=False)
truth_nodes = cand[truth_pos]
C_TRUE = 1.5
sens = np.sort(rng.choice(np.setdiff1d(jn_np, truth_nodes), size=NS,
                          replace=False))
cand_t = torch.as_tensor(cand, dtype=torch.long, device=DEV)
sens_t = torch.as_tensor(sens, dtype=torch.long, device=DEV)
ke_true = np.zeros(net.N, dtype=np.float64)
ke_true[truth_nodes] = ucf_e / C_TRUE ** qexp
print("### 候选 %d / 传感器 %d / 真值节点 %s（C_true=%.2f，Ke_int=%.4e）"
      % (NC, NS, truth_nodes.tolist(), C_TRUE, float(ke_true[truth_nodes[0]])))
sys.stdout.flush()


def batchify(B):
    g = np.random.default_rng(SEED)
    D = d0[None, :] * g.uniform(0.85, 1.15, (B, d0.size))
    R = rh0[None, :] + g.uniform(-1.0, 1.0, (B, rh0.size))
    return (torch.as_tensor(D, dtype=DT, device=DEV),
            torch.as_tensor(R, dtype=DT, device=DEV))


def ke_of_theta(theta):
    C = torch.nn.functional.softplus(theta)
    ke = torch.zeros(net.N, dtype=DT, device=DEV) \
        .index_copy(0, cand_t, ucf_e / C ** qexp)
    return ke, C


LAM = 1e-4
for B in TBS:
    print("=" * 96)
    print("§B=%d" % B)
    try:
        D, R = batchify(B)
        with torch.no_grad():
            obs = s.solve(D, R, ke_int=torch.as_tensor(
                ke_true, dtype=DT, device=DEV).unsqueeze(0).expand(B, -1),
                status_machine=True, accuracy=ACC, max_iter=200,
                assemble=ASM, linear_solver=LS)
            H_obs = obs["head_ft"].index_select(1, sens_t).clone()
            print("  观测：conv=%d/%d iters=%d 状态组=%d"
                  % (int(obs["converged"].sum()), B, int(obs["iters"].max()),
                     np.unique(obs["status"].cpu().numpy(), axis=0).shape[0]))
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if HAS_CUDSS:
            s.cudss_counters(reset=True)
        theta = torch.full((NC,), softplus_inv(0.02), dtype=DT, device=DEV,
                           requires_grad=True)
        opt = torch.optim.Adam([theta], lr=LR)
        times = []
        losses = []
        cnt5 = None
        for it in range(STEPS):
            sync()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            ke, C = ke_of_theta(theta)
            h, _q, _e = implicit_solve(s, D, R, ke=ke, adjoint="gpu",
                                       accuracy=ACC, max_iter=200,
                                       status_machine=True, assemble=ASM,
                                       linear_solver=LS)
            loss = ((h.index_select(1, sens_t) - H_obs) ** 2).mean() \
                + LAM * C.sum()
            loss.backward()
            opt.step()
            sync()
            times.append((time.perf_counter() - t0) * 1e3)
            losses.append(float(loss))
            if HAS_CUDSS and it == 4:
                cnt5 = dict(s.cudss_counters())
        steady = float(np.median(times[5:])) if len(times) > 6 else \
            float(np.median(times))
        print("  Adam %d 步：首步 %.1f ms | 稳态中位 %.2f ms/step "
              "(%.4f ms/场景) | min/max %.1f/%.1f ms"
              % (STEPS, times[0], steady, steady / B,
                 min(times[1:]), max(times[1:])))
        print("  损失：step1=%.6e -> step%d=%.6e（降 %.1fx）"
              % (losses[0], STEPS, losses[-1],
                 losses[0] / max(losses[-1], 1e-300)))
        with torch.no_grad():
            C_fin = torch.nn.functional.softplus(theta).cpu().numpy()
        top = np.argsort(-C_fin)[:5]
        print("  top-5 C：%s" % "  ".join(
            "n%d=%.3f%s" % (cand[i], C_fin[i],
                            "(真)" if cand[i] in truth_nodes else "")
            for i in top))
        hit = sorted(cand[np.argsort(-C_fin)[:3]].tolist()) \
            == sorted(truth_nodes.tolist())
        print("  top-3 支撑 == 真值：%s" % hit)
        if HAS_CUDSS and cnt5 is not None:
            cnt = s.cudss_counters()
            dsteps = STEPS - 5
            ci = s.cudss_cache_info()
            print("  cudss 计数（第 6..%d 步均摊）：factorize=%.2f solve=%.2f "
                  "bwd_solve=%.2f bwd_refactorize=%.2f /step | 缓存 state 数=%d"
                  "（键=(B,dtype,dev,mtype,slot)，稳态不再 plan）"
                  % (STEPS, (cnt["factorize"] - cnt5["factorize"]) / dsteps,
                     (cnt["solve"] - cnt5["solve"]) / dsteps,
                     (cnt["bwd_solve"] - cnt5["bwd_solve"]) / dsteps,
                     (cnt["bwd_refactorize"] - cnt5["bwd_refactorize"]) / dsteps,
                     len(ci)))
        if DEV == "cuda":
            tp = torch.cuda.max_memory_allocated() / 2 ** 20
            tr = torch.cuda.max_memory_reserved() / 2 ** 20
            nt = used() - resv() - CTX
            print("  显存 R5：ctx=%.2f torch_peak=%.2f torch_resv=%.2f "
                  "nontorch=%.2f total=%.2f MiB" % (CTX, tp, tr, nt, tr + nt))
        # ---- 旧形态对照（CPU 伴随；同损失，OLD_STEPS 步）----
        if OLD_STEPS > 0:
            theta2 = torch.full((NC,), softplus_inv(0.02), dtype=DT,
                                requires_grad=True)
            opt2 = torch.optim.Adam([theta2], lr=LR)
            H_obs_c = H_obs.cpu()
            t_old = []
            for it in range(OLD_STEPS):
                sync()
                t0 = time.perf_counter()
                opt2.zero_grad(set_to_none=True)
                C2 = torch.nn.functional.softplus(theta2)
                ke2 = torch.zeros(net.N, dtype=DT) \
                    .index_copy(0, cand_t.cpu(), ucf_e / C2 ** qexp)
                with torch.no_grad():
                    S_all = s.solve(D, R, ke_int=ke2.detach().to(DEV)
                                    .unsqueeze(0).expand(B, -1),
                                    status_machine=True, accuracy=ACC,
                                    max_iter=200, assemble=ASM,
                                    linear_solver=LS)["status"].cpu().numpy()
                uniq, inv = np.unique(S_all, axis=0, return_inverse=True)
                Dc, Rc = D.cpu(), R.cpu()
                loss2 = 0.0
                nobs = 0
                for gi in range(uniq.shape[0]):
                    ii = torch.as_tensor(np.where(inv == gi)[0])
                    h2, _q2, _e2 = implicit_solve(
                        se, Dc.index_select(0, ii), Rc.index_select(0, ii),
                        ke=ke2, speed=K0set, status=uniq[gi])
                    loss2 = loss2 + ((h2.index_select(1, torch.as_tensor(
                        sens, dtype=torch.long))
                        - H_obs_c.index_select(0, ii)) ** 2).sum()
                    nobs += ii.numel() * NS
                loss2 = loss2 / nobs + LAM * C2.sum()
                loss2.backward()
                opt2.step()
                sync()
                t_old.append((time.perf_counter() - t0) * 1e3)
            print("  旧形态（GPU 前向 + CPU 伴随）%d 步：%s ms/step "
                  "(%.2f ms/场景) | 加速（旧/新稳态）= %.1fx"
                  % (OLD_STEPS,
                     " ".join("%.1f" % x for x in t_old),
                     min(t_old) / B, min(t_old) / steady))
        del D, R
        if DEV == "cuda":
            torch.cuda.empty_cache()
    except Exception:                                  # noqa: BLE001
        FAILS.append("B=%d" % B)
        print("  [err B=%d] %s" % (B, last()))
        if DEV == "cuda":
            torch.cuda.empty_cache()
    sys.stdout.flush()

print()
if FAILS:
    print("FAILS:", FAILS)
print("MV2 TRAIN DONE rc=%d" % (1 if FAILS else 0))
sys.exit(1 if FAILS else 0)

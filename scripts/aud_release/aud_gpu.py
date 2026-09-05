# -*- coding: utf-8 -*-
"""aud_gpu.py - 敌意复测：L-TOWN GPU 时间表抽 6 格（B=8/256/1024 x dense/cudss）。

自写脚本，不复用 lt_time_gpu.py。公平性老三样当场验证并打印：
  ① 计时区外：warmup（含 cuDSS plan/JIT/缓存分配）单独计时打印，不入计时区；
     计时区 = sync; t0; solve; sync; t1，只包 solve 调用；打印每次重复的原始值
     （spread 大 = 计时区里有一次性开销，公平性存疑）。
  ② sync：每次计时前后 torch.cuda.synchronize()。
  ③ 等工作量：断言 dense_refine == cudss_refine（都打印数值）；逐场景 iters
     dense vs cudss 完全相等、状态逐元素相等、converged 全 True 才算该格有效
     （不等则该格作废并打印计数 - 不做任何剔除）。
场景配方（自配，确定性种子 909，两节点同一批）：名义盆地小扰动
  demand x U(0.8,1.2)，tank 水位取中位 - 保证两路状态轨迹一致，倍数才可比。
"""
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dgga.parse import parse_inp               # noqa: E402
from dgga.solver import GGASolver              # noqa: E402

DT = torch.float64
DEV = "cuda"
SEED = 909
BS = [int(x) for x in os.environ.get("AUD_BS", "8,256,1024").split(",")]
REPS = 4

node = os.popen("hostname").read().strip()
print("=" * 88)
print("aud_gpu 敌意复测 | node:", node, "| torch", torch.__version__,
      "|", torch.cuda.get_device_name(0), flush=True)

net = parse_inp(os.path.join(HERE, "L-TOWN.inp"))
N = net.N
d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
rh0 = np.nan_to_num(np.array(net.reservoir_head_ft_at(0), dtype=np.float64))
tn = np.asarray(net.tank_node, dtype=np.int64)
if tn.size:
    rh0[tn] = 0.5 * (np.asarray(net.tank_hmin) + np.asarray(net.tank_hmax))

rng = np.random.default_rng(SEED)
Bmax = max(BS)
Dall = d0[None, :] * rng.uniform(0.8, 1.2, (Bmax, N))
Rall = np.repeat(rh0[None, :], Bmax, axis=0)

rows = []
for B in BS:
    D = torch.as_tensor(Dall[:B], dtype=DT, device=DEV)
    R = torch.as_tensor(Rall[:B], dtype=DT, device=DEV)
    out = {}
    for tag, kw in (("dense", dict()),
                    ("cudss", dict(assemble="csr", linear_solver="cudss"))):
        s = GGASolver(net, device=DEV, dtype=DT, mode="dense",
                      dense_status_machine=True)
        assert s.dense_refine == s.cudss_refine == 2, \
            (s.dense_refine, s.cudss_refine)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            tw0 = time.perf_counter()
            o = s.solve(D, R, status_machine=True, **kw)   # warmup（不计时）
            torch.cuda.synchronize()
            tw = time.perf_counter() - tw0
            reps = []
            for _ in range(REPS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                o = s.solve(D, R, status_machine=True, **kw)
                torch.cuda.synchronize()
                reps.append(time.perf_counter() - t0)
        mem = torch.cuda.max_memory_allocated() / 2**20
        out[tag] = dict(o=o, reps=reps, warm=tw, mem=mem)
        print("  B=%-5d %-6s warmup=%.3fs reps(s)=%s peak_alloc=%.0fMiB"
              % (B, tag, tw, ["%.4f" % r for r in reps], mem), flush=True)
        if tag == "cudss":
            s.cudss_free()
        del s
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    od, oc = out["dense"]["o"], out["cudss"]["o"]
    it_eq = int((od["iters"] == oc["iters"]).sum())
    st_eq = int((od["status"] == oc["status"]).all(dim=1).sum())
    cv_d = int(od["converged"].sum())
    cv_c = int(oc["converged"].sum())
    dh = float((od["head_ft"] - oc["head_ft"]).abs().max())
    td = min(out["dense"]["reps"]) * 1000.0 / B
    tc = min(out["cudss"]["reps"]) * 1000.0 / B
    td_med = sorted(out["dense"]["reps"])[REPS // 2] * 1000.0 / B
    tc_med = sorted(out["cudss"]["reps"])[REPS // 2] * 1000.0 / B
    valid = (it_eq == B and st_eq == B and cv_d == B and cv_c == B)
    print("  B=%-5d 公平性: iters同 %d/%d 状态同 %d/%d conv %d+%d/%d "
          "max|dH|=%.3e ft  %s" % (B, it_eq, B, st_eq, B, cv_d, cv_c, B,
                                   dh, "有效" if valid else "<-- 作废"),
          flush=True)
    print("  B=%-5d 结果: fwd_d=%.3f(min)/%.3f(med) fwd_c=%.3f/%.3f ms/场景 "
          "倍数(min)=%.2fx" % (B, td, td_med, tc, tc_med, td / tc), flush=True)
    rows.append((B, td, tc, td / tc, valid))

print("\n【汇总 node=%s】" % node)
for B, td, tc, x, valid in rows:
    print("  B=%-5d fwd_d=%.3f fwd_c=%.3f ms/场景 倍数=%.2fx %s"
          % (B, td, tc, x, "" if valid else "<-- 作废"))
ok = all(v for *_, v in rows)
print("AUD_GPU_RESULT %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)

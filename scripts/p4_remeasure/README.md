# scripts/p4_remeasure - P4 全面重测（统一口径全表 / 倍数分解 / R7 逐出压测）

结论与全部数字在 `data/p4_remeasure_wip.txt`；面向使用者的那一份写进了
`README.md` 的 **“Choosing a solve path”** 一节；索引挂在
`data/sparse_gpu_plan.md` §11。

**本轮零 `dgga` 改动**：`dgga/solver.py` md5 `45b0b0ee…`、`dgga/autodiff.py`
md5 `e4b8cfe4…`（= 本地 HEAD `f29215e` 的字节，每个作业开头自行打印核过）。
`scripts/regression*.py` 一字未动。缺省 `mode="epanet"` / `assemble="dense"` /
`linear_solver="dense"` 未被触碰。审计 R9 遵守：`p4r_time_gpu.py` 里的
“稠密 + 手写伴随”（`DenseAdjoint`）只是**测量脚本内的对照**，没有进 `dgga`，
也没有新增 `linear_solver`。

| 脚本 | 回答的问题 | 作业 / 节点 | 原始输出 |
|---|---|---|---|
| `p4r_time_gpu.py` | 任务一 ①② + 任务二：10 网 × B∈{1,8,64,256,512,1024} 的纯前向 / 前向+反向 ms/场景，以及**倍数的两因子分解**（稀疏 vs 稠密 ‖ 稠密走通用 autograd 的欠账） | 1459647 / <node-18> | `data/gpu/5090_p4rt_1459647.out` |
| 同上 | 第二节点 | 1459681 / <node-19> | `data/gpu/5090_p4rt_1459681.out` |
| 同上 | 第三节点（另一机型，`nah`） | 1459702 / <node-13> | `data/gpu/5090_p4rt_1459702.out` |
| `p4r_mem_gpu.py` | 任务一 ③④：R5 统一口径显存全表 + OOM 边界，**每配置一个全新进程** | 1459648 / <node-20> | `data/gpu/5090_p4rm_1459648.out` |
| 同上 | 第二节点（两节点 120 格逐位相同） | 1459700 / <node-21> | `data/gpu/5090_p4rm_1459700.out` |
| `p4r_r7_gpu.py` | 任务三 R7：真 Adam + 分桶 loader 的六种训练形态 × 六组 `(cache_max, grad_slots)` 逐 step 压测 | 1459649 / <node-20> | `data/gpu/5090_p4r7_1459649.out` |
| 同上 | 第二节点 | 1459710 / <node-13> | `data/gpu/5090_p4r7_1459710.out` |
| `p4r_merge.py` | 把多节点的 `.out` 合成**区间表**（只读，不重算任何数） | 本机 | - |

集群侧目录 `<cluster-work-dir>/wdsgpu/hgp4rm`；`p2nets/` 从 `../hgp4` 复制。
全部 `--gpus=1 -p gpu_5090`（**绝不用 `hp_4090`**）；第二/第三节点用
`--exclude=` 强制换机器（审计 R6：凡引用倍数至少两节点）。

跑法：

```
tar -czf - dgga/*.py scripts/p4_remeasure | ssh ... "cd .../hgp4rm && tar -xzf - --strip-components=2"
ssh ... "cd .../hgp4rm && sbatch p4rt.sh && sbatch p4rm.sh && sbatch p4r7.sh"
# 第二轮换节点
ssh ... "cd .../hgp4rm && sbatch --exclude=<第一轮节点> p4rt.sh"
```

环境变量：`P4R_BS`（批量表，缺省 `1,8,64,256,512,1024`）、`P4R_NETS`（只跑某几个网）、
`P4R_R7_NETS`（缺省 `Modena,ky4`）、`P4R_R7_STEPS`（每轮 step 数，缺省 6；
`.sh` 里设成 4）。

本机（有 CUDA 无 nvmath）跑不了这三个脚本，但守卫可以本机验：
`python -X utf8 scripts/p2_cudss/smoke_cpu_guards.py`（5/5 明确 raise）。

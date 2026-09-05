# scripts/p4_corrections - P4 口径订正的实测脚本

结论与数字在 `data/p4_corrections_wip.txt`，回填在 `data/sparse_gpu_plan.md` §10。
**这一轮零代码改动**：`dgga/` 与 `scripts/regression*.py` 一个字节未动。

| 脚本 | 回答的问题 | 作业 / 节点 | 原始输出 |
|---|---|---|---|
| `p4a_gpu.py` | 审计 R3：自重跑抖动表（7 网 × B∈{8,64,256} × {前向水头 + 4 个梯度} × 两条通路，每格 5 次自重跑） | 1459528 / <node-01> | `data/gpu/5090_p4a_1459528.out` |
| `p4b.sh`（跑的是 `../p3_adversarial/aud4_gpu.py`，**没有 p4b_gpu.py**） | 审计 D1：显存表换 R5 统一口径，并做第二节点 | 1459529 / <node-20> | `data/gpu/5090_p4b_1459529.out` |
| `p4c_gpu.py` | 审计 D2/D3：逐出槽的代价 = 重建 + replan；`cudss_plan` 只热 slot 0 | 1459530 / <node-20> | `data/gpu/5090_p4c_1459530.out` |
| `p4d_gpu.py` | 审计 D5：`cudss_cache_max=8` 的定价（纯前向 / 带反向，每格全新进程 ×2） | 1459531 / <node-22> | `data/gpu/5090_p4d_1459531.out` |
| `p4e_gpu.py` | D5 的决定性对照：批量表里 B **重不重复** | 1459588 / <node-20> | `data/gpu/5090_p4e_1459588.out` |

集群侧目录 `<cluster-work-dir>/wdsgpu/hgp4`，`dgga/` 与 `p2nets/` 从
`../hgaudit` 复制（提交版字节，`solver.py` md5 `45b0b0ee…`，五个作业各自打印核过）。
全部 `--gpus=1 -p gpu_5090`，且 `--exclude=<node-21>` - 为的是强制换到与
P3 审计不同的节点（审计 R6：凡引用倍数至少两节点）。

跑法：
```
tar -czf - scripts/p4_corrections | ssh ... "cd .../hgp4 && tar -xzf -"
ssh ... "cd .../hgp4 && sbatch p4a.sh"
```
`p4a_gpu.py` 认两个环境变量：`P4_REP`（每格自重跑次数，缺省 5）、
`P4_BS`（批量表，缺省 `8,64,256`）。`p4d_gpu.py` 认 `P4D_NB`（批量个数，缺省 80）。

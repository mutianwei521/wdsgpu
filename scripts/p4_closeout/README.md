# scripts/p4_closeout - P4 终审修复清单的落地（脚本 ↔ 作业 ↔ 原始输出）

终审报告：`data/p4_adversarial_wip.txt`（§9 是它的修复清单）。
本轮结论与全部实测数：`data/p4_closeout_wip.txt`。

改动落在 `scripts/regression_gpu.py`（+T4、T3 探针改成观察被测代码）、
`scripts/regression_gpu.sh`（退出码传递）、`README.md`、`data/sparse_gpu_plan.md`。
**`dgga/` 一个字节没动**，三个缺省（`mode="epanet"` / `assemble="dense"` /
`linear_solver="dense"`）的数值与梯度未被触碰。

| 脚本 | 在哪跑 | 作业 / 节点 | 原始输出 | 查什么 |
|---|---|---|---|---|
| `regression_gpu.sh` → `regression_gpu.py` | 集群 | 1459913 / <node-20> | `data/gpu/5090_regp4_1459913.out` | 加了 T4 之后的全量回归，**50/50 PASS**、`rc=0`、作业 `COMPLETED 0:0` |
| `scripts/aud_p4/aud_symgap_gpu.py`（终审的注入脚本，原样复用） | 集群 | 1459914 / <node-21>、1459918 / <node-22> | `data/gpu/5090_audsg_{1459914,1459918}.out`、`5090_audsg_T4_{clean,patched}_detail.txt` | **T4 自证有牙**：同一个 1e-9 单侧写注入，正品 50/50 PASS，注入后 **40/50 FAIL**，红的恰是 10 条 T4，T1/T2/T3 一条不误报 |
| `regression_gpu.sh`（新）+ 故意退 3 的桩 | 集群 | 1459916 / <node-13> | `data/gpu/5090_rcnew_1459916.out` | 修好之后作业 **FAILED，ExitCode 3:0** |
| `regression_gpu.sh`（修前的末行 `echo`）+ 同一个桩 | 集群 | 1459917 / <node-13> | `data/gpu/5090_rcold_1459917.out` | 对照：同样退 3，作业却是 **COMPLETED，ExitCode 0:0** - 这就是被吞掉的那个退出码 |
| `p4c_dec_gpu.py` (+`p4cdec.sh`) | 集群 | 1459999 / <node-24>、1460001 / <node-25> | `data/gpu/5090_p4cdec_{1459999,1460001}.out` | 分解表的**等工作量口径**（稠密对照的反向也做 2 步精化），同一次运行同时重测现印口径以验证可比 |
| `p4c_agree_check.py` | 本机 CPU 前台 | - | 见 wip | 从三份原始 `.out` 重算 README 那句一致性区间（348 个测量） |
| `p4c_dense_reach.py` | 本机 CPU 前台 | - | 见 wip | 21 个 public 网逐个真建 `mode="dense"`，数缺省 / 加 `dense_tank_bound_check=False` 各够得着几个 |

集群目录：`<cluster-work-dir>/wdsgpu/hgclose`（回归 + 退出码对照）、
`hgclosedec`（分解）、`hgaudit`（注入）。全部 `--gpus=1 -p gpu_5090`。

`.out` 在 `.gitignore` 里，本轮用 `git add -f` 收进 `data/gpu/`。

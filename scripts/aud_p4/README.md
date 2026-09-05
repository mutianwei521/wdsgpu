# scripts/aud_p4 - P4 对抗审阅（脚本 ↔ 作业 ↔ 原始输出）

被审对象：`f29215e`（R1/R2/R4 守卫）、`2b8345c`+`4dd344c`（D1–D5+R3 订正）、
`9ee0ba3`（全面重测 + README "Choosing a solve path"）。
结论与全部实测数写在 `data/p4_adversarial_wip.txt`。
**本轮零改动**：`dgga/` 与 `scripts/regression*.py` 一个字节没动，这里全是只读脚本。

| 脚本 | 在哪跑 | 作业 / 节点 | 原始输出 | 查什么 |
|---|---|---|---|---|
| `aud_r1_mutants.py` | 本机 CPU 前台 | - | 见 wip §1 | 我自己的 13 个单侧装配变体 vs `check_symmetry.py` 的判据；含 1 ULP、只写上三角、只某轮/某 batch/某类网；外加 3 组换种子的不误报对照 |
| `aud_default_shadow.py` | 本机 CPU 前台 | - | 见 wip §7 | 影子包（`git archive ad36c31 dgga`）vs 工作区，10 网 × {epanet, dense} 的缺省输出与四类梯度逐字节 sha256 对拍（270 项） |
| `aud_readme_check.py` | 本机 CPU | - | 见 wip §8 | 不看 wip，直接解析 `data/gpu/5090_p4r{t,m}_*.out` 重算 README 三张表 + 分解表（144 格） |
| `aud_r2_gpu.py` (+`aud_r2_run.py`, `audr2.sh`) | 集群 | 1459819 / <node-13> | `data/gpu/5090_audr2_1459819.out` | `regression_gpu.py` 的基线 40/40、T1/T2/T3 打印行数、空哨兵、6 个源码级哨兵是否如期变红 |
| `aud_tbl_gpu.py` + `aud_mem_gpu.py` (`audtbl.sh`) | 集群 | 1459820 / 1459821（<node-20>）、1459839（<node-22>） | `data/gpu/5090_audtbl_145982{0,1}.out`、`5090_audtbl_1459839.out` | 8 个格（含 2 个 OOM 边界格）独立重测、计时公平性全查（计时区内 plan 次数、逐样本迭代数、精化步数、warmup/同步/重复数）、倍数分解 + 两个公平性变体、12 个显存配置 |
| `aud_symgap_gpu.py` (`audsg2.sh`) | 集群 | 1459838 / <node-13>（注入 1e-9）、1459848 / <node-02>（注入 1e-2） | `data/gpu/5090_audsg_1459838.out`、`5090_audsg2_1459848.out` | cuDSS 专属装配分支的对称守卫缺口：正品逐位对称 + 注入后 40/40 仍 PASS |

集群目录 `<cluster-work-dir>/wdsgpu/hgaudit`，全部 `--gpus=1 -p gpu_5090`。
`audsg2.sh` 用 `AUD_MAG` 控制注入的相对量（1e-9 那次用的是同一脚本、`AUD_MAG` 缺省）。

# dense（批量可微）路径的能力缺口与移植方案

任务 D 综合判断。基于任务 A（阀）/ B（泵·水池·D-W）/ C（我方两条路径）的逐行精读，
外加本轮新做的 5 组实测。**本文只做判断与方案，未改动 `dgga/` 与 `paper/`。**

依据文件：
- EPANET 2.2 源码 `ref/EPANET2.2-2.2.0/SRC_engines/`（hydcoeffs.c / hydsolver.c / hydstatus.c / smatrix.c / hydraul.c）
- 我方 `dgga/solver.py`、`dgga/autodiff.py`
- 明细笔记：`data/valve_batch_wip.txt`、`data/taskB_pump_tank_dw_wip.txt`、`data/x_taskC_twopath_wip.txt`、`data/x_taskD_dense_gap_wip.txt`

---

## 0. 先纠正一个要写进论文的数字：是 4/21，不是 2/21

任务书给的前提是"dense 路径 21 个公开网只跑得起 2 个"。**实测不成立，正确的数是 4。**

对 `networks/public/*.inp` 全部 21 个文件逐个构造 `GGASolver(net, mode="dense")`：

| 取文件 | dense 可构造 |
|---|---|
| 原始 `networks/public/*.inp` | 2（Hanoi, Modena） - 另有 3 个（BWSN_Network_1 / Fossolo_poly1 / Pescara）在 **parse 阶段**就失败，与 dense 无关 |
| 项目自己在用的清洗版 `networks/public/_cleaned/*.inp`（见 `dgga/mlds.py:68-76`） | **4（Hanoi Nj=31、Fossolo_poly1 Nj=36、Pescara Nj=68、Modena Nj=268）** |

`Fossolo_poly1`(58 管 / 0 泵 / 0 阀 / 0 池 / H-W) 与 `Pescara`(99 管 / 0 泵 / 0 阀 / 0 池 / H-W) 清洗后 dense 直接 OK。
"2 个"这个数是拿未清洗文件数出来的，把两个 parse 故障算成了 dense 缺口。论文里如果要量化限制第 (1) 条，
**必须写 4/21，并注明口径是 `_cleaned` 版**，否则是可被审稿人一条命令证伪的自我贬低。

21 网的逐网阻塞原因（构造期异常，`dgga/solver.py:104-118`）：

| 阻塞特性 | 触发处 | 被拒网数 |
|---|---|---|
| PUMP(=2) | `solver.py:104-111` allowed 集合 | 16 |
| tank(node_type=2) | `solver.py:118` | 15 |
| CVPIPE(=0) | `solver.py:104-111` | 7 |
| PRV(=3) | 同上 | 7 |
| FCV(=6) | 同上 | 2 |
| PSV(=4) | 同上 | 1 |
| D-W | `solver.py:116` | 1（Balerma） |

---

## 1. 重点判定：批内异态**不会**改变 A 的稀疏结构

这是本轮最关键的一条，先给结论：**不会。批量稠密 Cholesky 这条路走得通，
离散状态在 EPANET 里只改矩阵的"值"，不改"位型"。** 三条硬证据：

**(a) 位型在 openhyd 阶段定死一次，求解全程不重建。**
`createsparse` 的调用点全网只有一个：`hydraul.c:61  ERRCODE(createsparse(pr));`（在 `openhyd` 里），
`smatrix.c:16` 的文件头注释也写明 "called from openhyd() in HYDRAUL.C"。
`hydsolve` 主循环（`hydsolver.c:114-189`）里没有任何一处重建稀疏结构；
`matrixcoeffs`（hydcoeffs.c:164-195）每轮只是 memset 后往同一批槽位累加。
链路 k 的非对角槽位 `Ndx[k]` 在 `paralink`（smatrix.c:271-297）按并联关系一次定死。

**(b) 状态切换的全部效果都是"往固定槽位写不同的数"。** 三种典型：
- 关闭（`S<=CLOSED`）：`pipecoeff` hydcoeffs.c:531-536 → `P=1/CBIG=1e-8, Y=Q`，槽位照写，只是值极小；
- ACTIVE PRV：`prvcoeff` hydcoeffs.c:972-975
  ```c
  hyd->P[k] = 0.0;
  hyd->Y[k] = hyd->LinkFlow[k] + hyd->Xflow[n2];
  sm->F[j] += (hset * CBIG);      // CBIG = 1e8, hydcoeffs.c:35
  sm->Aii[j] += CBIG;
  ```
  注意它在 `return`（:980）前**从不写** `Aij[Ndx[k]]` - 这是"跳过写入"，
  非对角保持 0，不是从已有值里减去。批量里把 `Aij -= P` 换成 `Aij -= (1-m_active)·P` 即可，
  位型不变；
- ACTIVE FCV：`fcvcoeff` hydcoeffs.c:1073-1084 把定流量当外部需水打进 `Xflow/F`，`P=1/CBIG`，同样不动位型。

`linkcoeffs` 的 `if (hyd->P[k] == 0.0) continue;`（hydcoeffs.c:218）配合
`headlosscoeffs` 对非固定 PRV/PSV/FCV 的 `else hyd->P[k] = 0.0;`（hydcoeffs.c:154-158）
是唯一的"退出常规装配"开关 - 而它在批量里就是一个 `[B,L]` 的 0/1 掩码乘法。

**(c) 批内确实会异态 - 实测过了，但这只要求掩码，不要求分桶。**
用 B=128 个极端场景（需水 ×0.05~4.0、随机时刻、单点大漏损占全网 2%~60%）跑 epanet 路径的状态机：

| 网 | 状态随场景变化的离散元件 | 批内不同状态组合数 | 收敛 |
|---|---|---|---|
| BWSN_Network_1 | 2/10（PRV#176 CLOSED:127/ACTIVE:1；PRV#177 CLOSED:117/ACTIVE:11） | 3 | 128/128 |
| D-Town | 5/16（PRV#458 一个阀就吃满 CLOSED:87 / OPEN:31 / ACTIVE:10） | **16** | 128/128 |
| Richmond_skeleton | 3/15（全是 CVPIPE 的开闭） | 4 | 128/128 |

同一个阀在同一批里取三种状态是真实发生的。但**"按状态分桶"是错的对策**：D-Town 在 B=128 时已有 16 个桶，
桶数随 B 增长，批量的意义被吃掉。正确对策是 (b)：三支全算 + `torch.where`，
代价是常数倍算力（PRV 三支、CVPIPE 两支），换来位型恒定、一次批量 Cholesky。

**真正的代价不在结构，在数值。** 实测收敛帧的 Schur 补条件数（`np.linalg.cond`，2-范数，
从 `sm.linsolve` 截获 Aii/Aij 重建稠密 A）：

| 网 | 对角范围 | ACTIVE PRV/PSV 数 | cond2(A) |
|---|---|---|---|
| Hanoi（dense 现能跑） | 1.5e-1 .. 1.6e+1 | 0 | **1.85e3** |
| Modena（dense 现能跑） | 2.0e-2 .. 1.9e+2 | 0 | **3.86e5** |
| BWSN_Network_1 | 2.4e-2 .. **1.000e+8** | 5 | **5.57e9** |
| D-Town | 6.8e-3 .. **1.000e+8** | 3 | **2.46e11** |

对角上限 `1.000e+8` 就是 `CBIG` 本身（hydcoeffs.c:35），一个 ACTIVE PRV 把 κ 抬高 4~6 个数量级。
结论：
- **f64 仍可用**：eps·κ ≈ 2.2e-16 × 2.5e11 ≈ 5e-5 相对误差，`solver.py:1587-1594` 已有的两步迭代精化能压回；但必须把精化步数做成可调，并对 κ 做在线监控。
- **f32 彻底出局**：1.2e-7 × 2.5e11 ≈ 3e4。`solver.py:1580-1600` 已记录现有 κ≈1e9 下 f32 Cholesky 必败；再叠 big-M 只会更糟。论文 lim:dense-path 里"单精度不可用，126.6 ft 漂移"的说法在含 PRV 后要改成"数量级更差"。

**另一条更硬的天花板：显存 O(B·Nj²)。** A 是 `[B,Nj,Nj]` f64（`solver.py:1549-1552`），
展开可微路径还要留 K 份计算图。B=256 时单份 A：

| 网 | Nj | A 单份 |
|---|---|---|
| D-Town | 399 | 0.30 GB |
| L-TOWN | 782 | 1.17 GB |
| ky4 | 959 | 1.75 GB |
| **Net6** | 3323 | **21.1 GB** |
| **BWSN_Network_2** | 12523 | **299 GB** |

即使把全部特性移植完，Net6 与 BWSN_Network_2 在 B=256 下**物理上进不了稠密路径**（×K 份计算图后更甚）。
所以"移植特性能解锁 21 个网"这句话本身要打折：**在 B=256 下真正能用的上限是 19 个。**

---

## 2. 逐特性移植表

工作量按"已有 numpy 逐位实现可抄"估（`_PY_np:707-818`、`_valvecoeffs:847-917`、
`_dw_PY_np:635-682`、九个状态机函数 `solver.py:936-1150` 全在），单位为人日。

| 特性 | 移植难度 | 被批量结构阻挡？ | 工作量 | 单独解锁 | 关键依据 |
|---|---|---|---|---|---|
| **水池 tank** | **低** | 否 | 0.5 | +1（Net2） | 单帧内 tank 就是定水头节点：`inithyd` 把 `NodeHead[tank]=H0` 设死（hydraul.c:110），`tanklevels` 只在时段间跑（hydraul.c:247,655-657）。dense 的 `g1_row/g2_row` 定水头接地装配（`solver.py:1558-1561`）已就位，零新分支；只需放开 `solver.py:118` 并补 `tankstatus`（hydstatus.c:401-476）满/空池置 TEMPCLOSED 的掩码。reservoir 就是 A==0 的 tank（hydstatus.c:436-438 "Ignore reservoirs"）。 |
| **D-W** | **低** | 否 | 1.0 | +1（Balerma） | `DWpipecoeff`（hydcoeffs.c:578-620）纯逐元素；`frictionFactor`（:623-670）无内迭代，Re≥4000 用 Swamee-Jain 显式式，**导数在源码里就是闭式**：`*dfdq = 1.8*f*y1*A9/y2/y3/q`（:650）。两个 `where` 阈值（Re=2000/4000）值连续、非严格 C1，是可微性的唯一隐患。 |
| **CVPIPE** | **低** | 否 | 0.5 | +0 | 系数侧与 PIPE 完全同分支（headlosscoeffs:138-141 → pipecoeff）；唯一差别是 `cvstatus`（hydstatus.c:177-202）的状态覆盖。实测 Richmond_skeleton 批内 3 个 CVPIPE 会异态，用 `where` 直译即可。 |
| **PUMP** | **中** | 否 | 2.5 | +1（Anytown） | `pumpcoeff`（hydcoeffs.c:673-791）只改系数不改结构；关停 `pumpstatus`（hydstatus.c:205-239）的 XHEAD 也只是把 P 压到 1/CBIG（:231,:235）。**`autodiff.py:817-895` 里 `solve_unrolled` 已有完整 torch 泵支可直接复用**。剩余工作：CUSTOM 曲线选段的 O(K·B·泵) python 双循环（`autodiff.py:853-866`）换 `searchsorted`+`gather`；CONST_HP 的 `dq` 半步限幅（hydsolver.c:437-443）批量化。 |
| **FCV** | **中** | 否（但要保 O(Nvalves) 循环） | 2.0 | +0 | `fcvcoeff`（hydcoeffs.c:1047-1097）ACTIVE 时把定流量当外需水切断管网，改 `Xflow/F`；`fcvstatus`（hydstatus.c:362-398）三条分片规则。 |
| **PRV / PSV** | **高** | **否 - 但它是唯一"改方程含义"的元件** | 5.0 | +0 | ACTIVE 时不是消元降维，是把该行换成 Dirichlet 约束（big-M，hydcoeffs.c:972-975 / 1024-1027）。批量化的三个硬点见下。 |
| **批量状态机不动点** | **高** | 部分 | 3.5 | - （是上面全部的前置） | `hydsolve:150-189`：每轮 `valvestatus`(:156/:161)，收敛才 `linkstatus`(:173)+`pswitch`(:174)，未收敛按 CheckFreq 周期查(:183-187)。批内每个场景的状态轨迹不同 ⇒ 需 per-scenario 冻结掩码 + "全批都无状态变化"才算收敛。 |
| **f64 数值验收（κ→1e11）** | **中** | - | 2.5 | - | 见 §1 的 κ 表；精化步数可调 + κ 在线监控 + f32 明确禁用。 |

PRV/PSV 的三个硬点（都可解，但都要写代码）：
1. **装配顺序绑死**：`Y[k] = LinkFlow[k] + Xflow[n2]`（prvcoeff:973）读的是**累加完需水后**的 Xflow，
   所以 `valvecoeffs` 必须排在 `nodecoeffs` 之后（matrixcoeffs.c:184-194）。批量里就是两趟 scatter，不构成障碍。
2. **跨阀依赖**：`fcvcoeff:1075-1076` 改 `Xflow`，后续 `prvcoeff:973` 读它 ⇒ 阀之间有真串行依赖。
   对策：保留 `for k in valve_links` 的 O(Nvalves) 循环（Nvalves ≤ 8，见 §0 表），**循环体内对 B 全向量**。
3. **状态转移是依赖旧状态的分片函数**，不是 (h,Q) 的纯函数（`prvstatus` hydstatus.c:242-299 四个 case、
   `psvstatus`:302-359、`fcvstatus`:362-398）。对策：one-hot(旧状态) 加权四个候选，全向量化。
   另外 `badvalve`（hydsolver.c:215-265）在矩阵奇异时把肇事 ACTIVE 阀改判 XFCV(:256)/XPRESSURE(:257)，
   **这是批量路径唯一无法直译的语义** - 批 Cholesky 任一样本失败整批抛，没有"只重试第 b 个"
   （我方对应实现 `solver.py:1262-1271`）。可接受的降级：把失败样本标记后剔出该批重跑，
   或直接对 ACTIVE 行做对称对角均衡以避免触发。

梯度侧不构成新问题：状态是零测度阶跃，收敛后冻结、隐函数定理仍成立（论文 lim:frozen-status 已声明）；
串行侧 `ImplicitGGASolve` 已经把 ACTIVE PRV/PSV 当约束行处理（`autodiff.py:_valve_act_masks:173`、
`_residual_np:364`、`_build_J:383`），公式可直接搬。

---

## 3. 性价比排序：只做一件事，做"泵 + 水池"

单特性解锁数全都是 +1 或 +0（见 §2 表最后一列），因为绝大多数网是**多重阻塞**。
只有捆绑才有意义（口径：`_cleaned` 版，21 网）：

| 累计移植 | dense 可跑 | 增量 |
|---|---|---|
| 现状 | 4/21 | - |
| +水池 | 5/21 | +1 |
| **+水池 +泵** | **10/21** | **+6** |
| +水池 +泵 +CVPIPE | 11/21 | +1 |
| +水池 +泵 +CVPIPE +PRV | 18/21 | +7 |
| + FCV | 19/21 | +1 |
| + PSV | 20/21 | +1 |
| + D-W | 21/21 | +1 |

**排序（性价比 = 解锁数 / 工作量）：**

1. **泵 + 水池（3 人日 → 4→10 网，+6）。** 断层第一。两者都不改结构、不含 big-M，
   水池在单帧内退化成定水头节点（hydraul.c:110），泵的 torch 实现 `autodiff.py:817-895` 已经写好了。
   解锁 Anytown、Anytown_wntr、Net1、Net2、Net3、ky4 - 且这 6 个里最大的 ky4 Nj=959，
   B=256 时 A 只要 1.75 GB，显存也扛得住。
2. **CVPIPE + PRV（5.5 人日 → 10→18 网，+7）。** 解锁数同样可观，但工作量翻倍且引入 κ=1e11 的数值风险，
   还要连带做批量状态机不动点（+3.5 人日）。**注意 PRV 单独做解锁 0 个** - 7 个需要 PRV 的网
   （BWSN_1、D-Town、L-TOWN、L-TOWN_Real、Net6、Richmond_standard、ky10）**同时**需要泵和水池。
   所以 PRV 永远排在泵/水池之后，没有例外。
3. **D-W（1 人日 → +1，Balerma）。** 工作量极低但只值 1 个网。真正的价值不在解锁数，
   在于它是唯一一个**导数闭式就写在源码里**（hydcoeffs.c:650）、可以零风险论证"我们没有近似"的特性。
4. **FCV / PSV（4 人日 → +2）。** 最后做。

---

## 4. 对论文的影响：**不做进这一篇**，但要改两处措辞

**建议：全部留作后续工作。理由三条。**

1. **不做也不损伤论文的主张。** 这篇的核心贡献是"逐位复刻 + 可微"，
   逐位复刻的那条路（epanet 模式）**已经支持全部这些特性** - 实测 21 个公开网里
   `mode="epanet"` 能构造 18 个（另 3 个是 parse/INP 故障，非能力缺口）。
   梯度结果在 Hanoi/Modena 上成立，是 claim 的充分支撑。dense 路径的覆盖面是"规模"问题，不是"正确性"问题。
2. **做进来会引入一个论文当前不打算辩护的数值风险。** §1 的 κ 表显示 ACTIVE PRV 把条件数从 1e5 推到 2.5e11。
   论文 lim:dense-path 现在的说法（"GPU f64 与 CPU f64 差 9.42e-6 ~ 2.16e-5 ft"）
   是在 κ≈1e5 的两个网上量的；含 PRV 后这个数会明显变大，
   等于要重做四机 GPU 表（Table tab:fourgpu）与整个 §sec:exemption 的豁免论证。这不是加一个 feature，是重开一轮验收。
3. **工作量与收益不成比例。** 完整对齐 ≈ 3 周（含验收），而收益是"把批量路径的可跑网数从 4 提到 19"，
   属于工程覆盖面，不产生新的科学主张。

**但必须改两处措辞（这两处现在是不准确的，属于必改）：**

- **限制第 (1) 条 lim:unimplemented（`paper/_body.tex:1129-1136`）**：现在只列了 PBV/GPV、PDA、
  带体积曲线的水池、水质模块。它**没有说清"未实现"是分路径的** - PRV/PSV/FCV/CVPIPE/泵/水池/D-W
  在复刻路径上是实现了的，缺的只是批量可微路径。建议增补一句量化的：
  "可微的批量稠密路径目前只覆盖 H-W 管道与 TCV，21 个公开基准网里可直接构造 4 个（Hanoi、Fossolo、Pescara、Modena）；
  泵、水池、CVPIPE 与调压阀在复刻路径上完整实现、在批量路径上留待后续。"
 - **数字写 4，不写 2**（§0）。
- **`solver.py:104-111` 的错误信息**里"（PBV/GPV 留接口）"这句措辞已过时且误导：
  dense 模式真正的缺口是 PUMP/PRV/PSV/FCV/CVPIPE 五类，不是 PBV/GPV 两类。属于一行字的修正（不在本轮范围）。

---

## 5. 明确判断"不该做"的

- **`badvalve` 的批量直译（hydsolver.c:215-265 / `solver.py:1262-1271`） - 不要做。**
  它的语义是"某一次 linsolve 遇到零主元 → 改这个阀的状态 → **不递增 iter** 重试本次迭代"。
  批量下没有"只重试第 b 个样本"的语义（批 Cholesky 一个样本崩整批崩），
  强行做只能退化成逐样本循环，等于把批量路径退回串行。
  应做的是**回避**而不是复刻：对 ACTIVE 行做对称对角均衡，并把触发 badvalve 的样本剔出该批单独走 epanet 路径。
- **f32 的批量路径 - 不要为含阀网做。** §1 实测 κ=2.5e11，f32 的 eps·κ≈3e4，
  这不是精度问题是完全没有有效数字。`solver.py:1580-1600` 已记录 κ≈1e9 时 f32 就必败。
  应在构造期直接拒绝 `dtype=float32 且含 PRV/PSV`，而不是留一条会静默算错的路。
- **PBV / GPV - 不要做。** 21 个公开网里出现次数为 0（`data/public_inventory.json` 全表 valve_types 无 PBV/GPV）。
- **PDA - 不要做进这一篇。** 同样 0 个公开网使用（全表 demand_model 均为 DDA），
  且 `demandcoeffs/newdemandflows` 在 DDA 下直接 return（hydcoeffs.c:438 / hydsolver.c:541），
  改成 PDA 会动到每一轮的装配与收敛判据。

---

## 6. 附带发现（与本任务同源，但严重度更高，建议单独处理）

**`dgga/autodiff.py:696` `solve_unrolled` 对 D-W 与 PRV/PSV/FCV 静默算错，且完全无守卫。**

`solve_unrolled` 是真正的 torch 批量展开路径，它接收一个 `GGASolver` 实例 -
可以是 `mode="epanet"` 的、含 D-W / PRV / 泵 / 水池的实例（构造期不拦）。
但扫描 `autodiff.py:696-1000` 全段：**没有任何一处出现 `_PRV/_PSV/_FCV`、`headloss_form` 或 `NotImplementedError`**。
非 TCV 的阀会掉进管道支，用 `s.r_hw` - 而 `parse.py:224` 明写 `r_hw ... 非管道 0` -
于是 `hgrad = 0` 被 RQtol 钳到 1e-7，PRV 被当成一根近乎无阻力的开管。D-W 网则会用去掉摩阻因子 f 的 r 配 H-W 的指数。

实测（与同一个 solver 的 epanet 路径对比，junction 水头）：

| 网 | 条件 | max\|ΔH\| | mean\|ΔH\| | 是否抛异常 |
|---|---|---|---|---|
| Balerma（D-W，无泵无阀） | K=30 | **1.82e+4 ft** | 7332 ft | 否 |
| BWSN_Network_1（PRV×8 + 泵） | K=40，**已传入冻结状态与 setting** | **384.9 ft** | 55.9 ft | 否 |

BWSN_1 那一行是把 epanet 路径收敛后的 `status`/`setting` 原样传给 `solve_unrolled` 的，
所以差异不是状态机不同步造成的，就是 5 个 ACTIVE PRV 的 big-M 行根本没装配。
建议在 `solve_unrolled` 入口加一句与 `solver.py:104-118` 同样口径的类型/水损公式守卫
（本轮不改代码，仅记录）。

---

## 7. 建议的执行顺序（若将来做）

1. `solve_unrolled` 加守卫（0.2 人日，止血，优先级最高）
2. 水池 → 定水头 + `tankstatus` 掩码（0.5）
3. 泵（复用 `autodiff.py:817-895`）+ CUSTOM 选段 `searchsorted`（2.5）→ **到此 10/21，收益最大的停点**
4. D-W（1.0，独立、零风险）
5. CVPIPE + 批量状态机不动点（4.0）
6. PRV/PSV + big-M + κ 验收（7.5）→ 18/21
7. FCV / PSV 收尾（2.0）→ 20/21，Net6 与 BWSN_Network_2 受显存限制仍不可用（§1）

# dgga 阶段 A 接口契约（v1）

所有模块必须遵守本契约。偏离契约 = 集成失败。

## 运行环境

- 解释器：`<python>`（Python 3.12, numpy 2.5, wntr 1.5.0）
- 双精度 EPANET DLL：`<wntr包目录>\epanet\libepanet\windows-x64\epanet22.dll`（运行时用
  `os.path.join(os.path.dirname(wntr.__file__), "epanet", "libepanet", "windows-x64", "epanet22.dll")` 定位，
  已验证导出 `EN_createproject / EN_open / EN_runH / EN_getnodevalue` 等双精度 API）。
- API 原型以 `ref/epanet2.2_toolkit/epanet2_2.h` 为准，枚举值以 `ref/epanet2.2_toolkit/epanet2_enums.h` 为准
  （**必须读文件抄枚举值，不许凭记忆**）。
- `ref/epanet2.2_toolkit/epanet2.dll`（32 位旧版 float API）一律不用。

## 单位约定（铁律）

- 一切落盘数值均为 **EPANET 内部单位：长度/水头 ft，流量 cfs**，dtype 一律 float64。
- 换算常数**逐字抄自** `ref/EPANET2.2-2.2.0/SRC_engines/types.h:68-83`（GPMperCFS、LPSperCFS、MperFT 等）。
- 方向约定与源码一致：用户值 = 内部值 × Ucf；因此从 API 读到用户值后 **除以** Ucf 得内部值。
- 七个网只有两种流量单位：LPS（SI 制，水头 m）与 GPM（US 制，水头 ft）。

## 管网清单（7 个）

```
networks/InpData/EXA4.inp  EXA5.inp  EXA6.inp  city_h.inp  ky3.inp  ky5.inp
networks/realInpData/city_d.inp
```

## 目录与产物

```
dgga/units.py         # 常数 + Ucf（chain-1）
dgga/parse.py         # wntr 解析 → Net 数据类（chain-1）
dgga/epanet_ref.py    # ctypes 双精度 EN_* 封装（chain-2）
dgga/reference.py     # EPS 参考解收集（chain-2）
scripts/build_reference.py     # 集成 CLI（integrator）
scripts/validate_reference.py  # 三项交叉验证（integrator）
data/reference/<stem>_net.npz  <stem>_ref.npz  <stem>_meta.json
```

## `dgga/units.py`（chain-1）

```python
class UCF:
    """UCF(flow_units: str)，flow_units ∈ {"LPS","GPM"}（大写）。
    属性（均为 float，用户单位/内部单位）：
      .flow   # LPS: LPSperCFS；GPM: GPMperCFS
      .head   # LPS(SI): MperFT；GPM(US): 1.0
      .length # 同 .head
      .diam   # SI: 1000*MperFT (mm/ft)；US: 12.0 (in/ft)
    """
```

## `dgga/parse.py`（chain-1）

`parse_inp(path: str) -> Net`；`Net.save(dir_path, stem)` 写 `<stem>_net.npz` + `<stem>_meta.json`；
`Net.load(dir_path, stem)` 逆操作。

Net 字段（numpy 数组，节点/管段均按 EPANET 文件顺序，**0 起始索引**）：

| 键 | 类型 | 含义 |
|---|---|---|
| node_id | list[str]（存 meta.json） | 节点 ID |
| node_type | int8[N] | 0=junction 1=reservoir 2=tank |
| elev_ft | float64[N] | 高程；reservoir 为其定水头 base |
| node_ke | float64[N] | emitter 内部系数（本期全 0 占位） |
| link_id | list[str]（meta） | 管段 ID |
| link_type | int8[L] | epanet2_enums.h 的 EN_LinkType 值（0=CV 1=PIPE 2=PUMP 3=PRV 4=PSV 5=PBV 6=FCV 7=TCV 8=GPV） |
| link_n1 / link_n2 | int32[L] | 0 起始节点索引 |
| diam_ft / len_ft | float64[L] | 泵：0 |
| roughness | float64[L] | H-W 的 C（无量纲，原值）；非管道 0 |
| km_int | float64[L] | 局部损失内部系数 = 0.02517·K/D_ft⁴（Q² 基准）；K 为用户局损系数 |
| r_hw | float64[L] | 管道 H-W 内部阻力 = 4.727·len_ft / C^1.852 / diam_ft^4.871；非管道 0 |
| init_status | int8[L] | 0=Closed 1=Open 2=Active(阀带设定) |
| valve_setting_user | float64[L] | 阀设定（用户单位原值，本期不换算）；泵为转速比 |
| res_head_pat | int32[N] | reservoir 水头模式索引（-1=无）；tank/junction=-1 |

方法：
- `demand_cfs_at(t_sec: int) -> float64[N]`：t 时刻全网名义需水（junction 多类别求和；
  阶梯取值公式 `F[ floor((t+Pstart)/Pstep) mod len ]`，乘全局 Demand Multiplier；
  tank/reservoir 置 0）。**必须处理多类别需水（city_d 节点 162 有 3 类）与全局默认 pattern。**
- `reservoir_head_ft_at(t_sec) -> float64[N]`（非 reservoir 置 nan；有模式则 base×F[t]）。
- meta.json 保存：flow_units、headloss（恒 "H-W"）、trials、accuracy、demand_model、
  emitter_exponent（用户 γ）、qexp（=1/γ）、duration_sec、hyd_step_sec、pat_step_sec、
  pat_start_sec、demand_multiplier、patterns（id→list）、控制/规则原始行（字符串数组，本期不解析）。

数据来源：wntr `WaterNetworkModel`（内部 SI：m、m³/s），换算到 ft/cfs 时用 types.h 常数
（m³/s → cfs 用 ×1000/LPSperCFS，保持与 EPANET 同一常数体系）。

## `dgga/epanet_ref.py`（chain-2）

```python
class Epanet:
    """ctypes 封装双精度 EN_* API。上下文管理器。
    Epanet(inp_path, rpt_path=None)  # rpt 默认写临时文件并在 close 时删除
    方法（全部返回内部单位 ft/cfs，即读到用户值后除以 UCF；UCF 用 EN_getflowunits 判定）：
      .version() -> int                      # EN_getversion
      .counts() -> dict(nodes, links, ...)   # EN_getcount
      .node_ids() / .link_ids() -> list[str]
      .solve_eps() -> dict                   # 见下
    错误处理：返回码>100 抛 RuntimeError（附 EN_geterror 文本）；1..100 为警告，收集到结果 warnings。
    """
```

`solve_eps()`：EN_openH → EN_initH(EN_NOSAVE) → 循环 { EN_runH(t) → 逐节点/逐管段读值 → EN_nextH }
直到 tstep==0。每个 EN_runH 返回时刻记一帧。返回 dict：

| 键 | 类型 | 来源（EN_getnodevalue / EN_getlinkvalue / EN_getstatistic） |
|---|---|---|
| t_sec | int64[T] | EN_runH 的 t |
| head_ft | float64[T,N] | EN_HEAD ÷ ucf.head |
| pressure_ft | float64[T,N] | EN_PRESSURE ÷ ucf.pressure（SI: MperFT×1？以 enums/实测校准 - US 制 psi、SI 制 m，换算到 ft 水柱） |
| demand_out_cfs | float64[T,N] | EN_DEMAND ÷ ucf.flow（求解后=实际总出流；tank=净流入） |
| flow_cfs | float64[T,L] | EN_FLOW ÷ ucf.flow（注意：关闭管被 API 置 0） |
| status | int8[T,L] | EN_STATUS |
| setting | float64[T,L] | EN_SETTING（原值不换算） |
| iterations | int32[T] | EN_ITERATIONS |
| relerr | float64[T] | EN_RELATIVEERROR |
| warnings | list（存 meta） | 运行警告码 |

注意：pressure 的用户单位（US=psi, SI=m）→ ft 水柱换算必须与 units.py 一致；
US: ÷PSIperFT；SI: ÷MperFT。

## `dgga/reference.py`（chain-2）

`build_reference(inp_path, out_dir, stem)`：跑 `Epanet(inp).solve_eps()`，
`np.savez_compressed(out_dir/<stem>_ref.npz, **arrays)`；标量与 warnings 并入
（若存在则更新）`<stem>_meta.json` 的 `"reference"` 键。

## `scripts/validate_reference.py`（integrator）

对每个网做三项检查并打印汇总表（中文）：

1. **质量守恒（结构性，必须过）**：对每个 junction、每帧：
   `residual = Σ(入流) − Σ(出流) − demand_out_cfs`，用 net 的关联索引 + ref 的 flow_cfs。
   门槛 `max|residual| < 1e-5 cfs`（关闭管被 API 置零会留 ~1e-6 级残差，属预期）。
2. **需水装配对拍（必须过）**：DDA 下对每个 junction、每帧：
   `|parse.demand_cfs_at(t) − demand_out_cfs| < 1e-9 cfs`（验证 pattern 阶梯索引 + 多类别求和 + 单位链）。
   注：全部 7 网均为 DDA、无 emitter，故实际出流 ≡ 名义需水。
3. **H-W 水损对拍（报告性，不设硬门槛）**：对开启管道（status=Open 且 |Q|>1e-8）：
   `err = |(H1−H2) − sign(Q)·(r_hw·|Q|^1.852 + km_int·Q²)|`，报告 max/中位数。
   预期中位数 ~1e-3 ft 量级（EPANET 收敛容差所致，非我们的错）；若 >0.05 ft 说明 r_hw 单位链有错。

## B2 增补（泵 / 水池 / 控制 / EPS；net.npz 新键向后兼容 - 老 npz 缺键时默认空）

### Net 新增字段（parse.py；数值一律 EPANET 内部单位 ft/cfs/ft²/ft³，float64）

水池（仅真实柱形水池，顺序 = [TANKS] 文件行序；换算照抄 input1.c:575-592）：

| 键 | 类型 | 含义 |
|---|---|---|
| tank_node | int32[Nt] | 节点索引 |
| tank_h0/hmin/hmax | float64[Nt] | 初始/最低/最高水位折算绝对水头（El+lvl/Ucf[ELEV]，input1.c:581-583） |
| tank_area | float64[Nt] | 截面积 PI·SQR(D/dcf)/4（input1.c:584；PI=3.141592654） |
| tank_vmin/v0/vmax | float64[Nt] | 体积（用户单位按 input3.c:247-251 先算，再 ÷Ucf[VOLUME]=hcf³） |
| tank_overflow | int8[Nt] | CanOverflow（input3.c:207-212） |

泵（顺序 = [PUMPS] 文件行序 = EPANET 泵序；拟合照抄 input3.c:2030-2129 /
input2.c:372-462（1 点曲线补 h0=1.33334·h1、qmax=2·q1；3 点曲线 X[0]==0），
恒功率 R=−8.814·P（input2.c:392-400，SI 再 ÷KWperHP input1.c:633），
换算照抄 input1.c:625-648；pow/log 走 msvcrt CRT）：

| 键 | 类型 | 含义 |
|---|---|---|
| pump_link | int32[Np] | 管段索引 |
| pump_ptype | int8[Np] | PumpType（types.h:172-177：0=CONST_HP 1=POWER_FUNC 3=NOCURVE） |
| pump_h0/r/n | float64[Np] | 曲线系数（内部单位；hloss = h0 + r·q^n 的负扬程口径） |
| pump_q0/qmax/hmax | float64[Np] | 设计流量/最大流量/截止扬程（CONST_HP: 1.0/BIG/BIG） |
| pump_upat | int32[Np] | 转速 PATTERN 索引（−1=无）；SPEED 存于 valve_setting_user |

[CONTROLS] 结构化（controldata input3.c:817-926；阈值折算 input1.c:672-706）：

| 键 | 类型 | 含义 |
|---|---|---|
| ctl_link / ctl_node | int32[C] | 受控管段 / 监测节点（−1=时间控制） |
| ctl_type | int8[C] | 0=LOWLEVEL 1=HILEVEL 2=TIMER 3=TIMEOFDAY（types.h:186-191） |
| ctl_status | int8[C] | 目标内部状态（StatusType：2=CLOSED 3=OPEN 4=ACTIVE） |
| ctl_setting | float64[C] | 目标设定（泵转速原值；阀按 input1.c:691-705 换内部；MISSING=−1e10） |
| ctl_grade | float64[C] | 阈值绝对水头 ft（tank: El+grade/Ucf[ELEV]；junction: El+grade/Ucf[PRESSURE]） |
| ctl_time | int64[C] | 触发时刻 s（TIMEOFDAY 已 mod 86400） |

meta 新增标量（老 meta 缺键按 EPANET 默认补齐）：extra_iter（UNBALANCED，默认 −1）、
checkfreq=2、maxcheck=10、damp_limit=0、tstart_sec、report_step_sec、report_start_sec。

### solver.py（B2）

- epanet 模式支持 PIPE/CVPIPE/TCV/PUMP 与 tank（定水头节点，头由调用方逐帧传入
  res_head_ft 的 tank 位）；dense 模式维持梯队 1 范围。
- `run_gga(d, fixed_head, ke, q0, e0, status0, setting0, do_status, ...)`：
  hydsolve（hydsolver.c:57-212）单样本复刻。status0/setting0 为内部 StatusType
  编码与 LinkSetting；do_status=True 启用状态机（pumpstatus/tankstatus/cvstatus
  照抄 hydstatus.c，CheckFreq/MaxCheck 节律 + 收敛后 statChange 复核 + pswitch）；
  pumpcoeff 照抄 hydcoeffs.c:673-791；恒功率泵 dq=Q/2 半步照抄 hydsolver.c:437-444。
  返回含 status/setting/fixed_demand（定水头节点净流入）。
- `solve(..., link_status=, link_setting=, status_machine=)` 直通上述能力（epanet 模式）。

### dgga/eps.py（B2）

`EpsDriver(net, inp_path).run()`：完整 EPS 复刻 hydraul.c - inithyd（:85-179）一次、
runhyd 循环（demands :465-535 / controls :538-619 含 vplus 一秒容差与 TIMER 精确命中）、
nexthyd/timestep（:622-659 = min{Hstep, 模式边界, Rtime, tanktimestep :662-707 的
ROUND, controltimestep :710-783}）、tanklevels（:998-1035 显式欧拉+1 秒前瞻钳位）、
tankgrade（:1089 柱形仿射）；全程 warm start。输出帧数组与 ref.npz 同口径
（flow_cfs 按 EN_FLOW 关闭置零，另存内部 flow_int_cfs）。

### 验收脚本

- scripts/align.py：泵/水池网自动切快照回放（状态冻结，tank 头/状态/设定取 ref，
  需水取 parse，warm start 用上一帧我方流量；恒功率泵 关→开 重置 Q0）。
- scripts/align_eps.py：4 网 t=0 完整自主 EPS 对拍；帧时刻 int 相等 +
  每帧 max|ΔH|<1e-6 ft、max|ΔQ|<1e-6 cfs；ky5 死端隔离支管保留按帧豁免通道
  （实测 4 网位级对齐、迭代数逐帧相等，豁免未触发）。

## 通用约定

- 编码 UTF-8；npz 用 `np.savez_compressed`；不引入 wntr 之外的新依赖（ctypes/numpy/json/tempfile 标准库）。
- 每个模块底部带 `if __name__ == "__main__":` 冒烟测试（用 EXA6 或 city_d），
  开发时必须真的用 wm-course 解释器跑通再交付。
- 路径全部 ASCII，无中文路径进 EN_open。

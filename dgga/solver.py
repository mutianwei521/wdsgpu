# -*- coding: utf-8 -*-
"""dgga.solver - 忠实复刻 EPANET 2.2 GGA 的 PyTorch 前向求解器。

范围（按通路分级）：
- mode="epanet"（逐位复刻）：H-W/D-W/C-M 管道、CVPIPE、TCV、PUMP、PRV/PSV/FCV、
  水库/水池、DDA 需水、emitter、完整状态机（hydstatus 九个转移 + badvalve）。
- mode="dense"（批量可微）：H-W 管道、TCV、PUMP、水库/水池；开
  dense_status_machine=True 再放行 CVPIPE 与 PRV（批量状态机 + 批量
  prvcoeff/prvstatus，PRV 轮）。PSV/PBV/FCV/GPV 与 D-W 仍构造期
  NotImplementedError（留接口）。

所有公式、常数、分支均逐字核对 EPANET 2.2 源码（ref/EPANET2.2-2.2.0/SRC_engines/），
注释标注 源文件:行号。float64。矩阵为稠密 [Nj,Nj]，用 scatter_add 装配，天然支持批维。
"""

import collections
import contextlib
import math
import time

import numpy as np
import torch

try:
    from dgga.parse import Net
    from dgga.smatrix import EpanetSmatrix
    from dgga.units import FLOW_UCF, SI_FLOW_UNITS
except ImportError:  # pragma: no cover
    from parse import Net
    from smatrix import EpanetSmatrix
    from units import FLOW_UCF, SI_FLOW_UNITS

# ---------------------------------------------------------------- 常数（照抄源码）
CSMALL = 1.0e-6        # hydcoeffs.c:34  const double CSMALL = 1.e-6;
CBIG = 1.0e8           # hydcoeffs.c:35  const double CBIG   = 1.e8;
BIG = 1.0e10           # types.h:48      #define BIG 1.E10
TINY = 1.0e-6          # types.h:49      #define TINY 1.E-6
MISSING = -1.0e10      # types.h:50      #define MISSING -1.E10
QZERO = 1.0e-6         # hydraul.c:23    const double QZERO = 1.e-6; (cfs 零流量)
RQTOL_DEFAULT = 1.0e-7  # input1.c:35    #define RQTOL 1E-7
HEXP_HW = 1.852        # input1.c:294    if (hyd->Formflag == HW) hyd->Hexp = 1.852;
HACC_DEFAULT = 0.001   # input1.c:26     #define HACC 0.001
VISCOS_DEFAULT = 1.1e-5  # types.h:53    #define VISCOS 1.1E-5
# D-W 摩阻因子常数（hydcoeffs.c:23-31，逐字照抄）
A1 = 3.14159265358979323850e+03   # hydcoeffs.c:23  1000*PI
A2 = 1.57079632679489661930e+03   # hydcoeffs.c:24  500*PI
A8 = 4.61841319859066668690e+00   # hydcoeffs.c:27  5.74*(PI/4)^.9
A9 = -8.68588963806503655300e-01  # hydcoeffs.c:28  -2/ln(10)
AB = 3.28895476345399058690e-03   # hydcoeffs.c:30  5.74/(4000^.9)
AC = -5.14214965799093883760e-03  # hydcoeffs.c:31  AA*AB
MAXITER_DEFAULT = 200  # input1.c:25     #define MAXITER 200
DAMPLIMIT_DEFAULT = 0.0  # input1.c:38   #define DAMPLIMIT 0
# types.h:57-61：#ifdef M_PI → M_PI，否则 3.141592654。实测 wntr 的 epanet22.dll
# 用回退字面量 3.141592654（构建未定义 M_PI）：k=1 截断对拍下用 M_PI 差
# 6.6e-7 ft、用 3.141592654 差 0（逐位一致）（初始流量 πD²/4 经病态矩阵放大所致）。
PI = 3.141592654       # types.h:60

# epanet22.dll（MinGW 系构建）的静态 pow 与 msvcrt.dll 位级一致（与 UCRT 有
# ~0.2% 的 1ulp 差）。位级对拍必须用同一 pow；非 Windows 环境退回 math.pow。
try:
    _pow_crt = __import__("ctypes").CDLL("msvcrt.dll").pow
    _pow_crt.restype = __import__("ctypes").c_double
    _pow_crt.argtypes = [__import__("ctypes").c_double] * 2
    _log_crt = __import__("ctypes").CDLL("msvcrt.dll").log
    _log_crt.restype = __import__("ctypes").c_double
    _log_crt.argtypes = [__import__("ctypes").c_double]
except OSError:  # pragma: no cover
    _pow_crt = math.pow
    _log_crt = math.log

# EN_LinkType（epanet2_enums.h:183-191，与 parse.py 一致）
_CVPIPE, _PIPE, _PUMP, _PRV, _PSV, _PBV, _FCV, _TCV, _GPV = range(9)


class _CudssSolveFn(torch.autograd.Function):
    """cuDSS 线性解 x = A^{-1}b 的自定义可微封装（sparse_gpu_plan.md §1c，P3）。

    forward：A 由 CSR 值 `data` [B,nnz] 给出（位型是构造期定死的 self.A_csr_*），
      b = `F` [B,Nj]；数值分解 + (1+refine) 次三角回代，全部由 cuDSS 做。

    backward：把整段当作**精确解算子** x = A^{-1}b 来微分（不是穿过迭代精化的
      那几步表达式），
          ∂L/∂b = A^{-T}·∂L/∂x = A^{-1}·∂L/∂x   （A 对称，见下）
          ∂L/∂A|_{ij} = −λ_i·x_j                （λ = ∂L/∂b）
      落到 CSR：grad_data[k] = −λ[row[k]]·x[col[k]]。

      **A 为什么是对称的**：装配里非对角两处 (i,j)/(j,i) 取自同一段
      `-Pm[:, lk_both]`，且槽位按 (min,max) 定序（solver.__init__ 的
      idx_off1/idx_off2） - 两个三角对每个无序节点对按**同一链路次序**累加，
      对角是 +P 之和，emitter 只加对角 ⇒ CPU 串行归约下按构造逐位对称，
      位型也对称（CUDA 上 scatter_add_ 原子序不定，对称到 ~1 ULP）。
      所以 A^T = A，伴随方程 A^T λ = g 与前向共用**同一次数值分解**。
      这里**不**对 grad_data 做对称化：CSR 的 (i,j) 与 (j,i) 是两个独立槽，
      各自的 -λ_i x_j / -λ_j x_i 会在上游 scatter_add 的反向里对同一条链路的
      P 求和，正好等于对称结构下的 λ_i x_j + λ_j x_i（手算与实测都对上）。

    "复用同一次分解"是有条件的（诚实说明，见 GGASolver.cudss_grad_slots）：
    cuDSS 的一份 state 同一时刻只持有**一个**数值分解。展开 K 步的图里
    前向按 1..K 分解、反向按 K..1 用，只有最后一次天然还在。slots≥K 时每步
    占一份 state（各自的分解都活着）⇒ 反向零重分解；slots<K 时越界的那些步
    在反向里按保存的 `data` 重新 factorize 一次（计数在 bwd_refactorize，
    绝不静默）。
    """

    @staticmethod
    def forward(ctx, data, F, solver, B, refine, slot):
        Hj, st, gen = solver._cudss_forward(data, F, B, refine, slot)
        ctx.solver = solver
        ctx.B = int(B)
        ctx.st = st
        ctx.gen = gen
        ctx.slot = slot
        ctx.save_for_backward(data, Hj)
        return Hj

    @staticmethod
    def backward(ctx, gH):
        # 二阶导（create_graph=True）：反向里那次 solve 是 cuDSS 的黑箱，接不进
        # autograd，硬走下去只会得到一个断开的图（torch 随后报的是"张量没参与
        # 计算"这种误导性错误）。这里提前明确 raise - 宁可少支持，不可静默给错。
        if torch.is_grad_enabled():
            raise NotImplementedError(
                "linear_solver='cudss' 的反向只做一阶：cuDSS 的三角回代不是 "
                "autograd 算子，create_graph=True / 二阶导数（Hessian-vector 等）"
                "在这条通路上做不到。请对需要二阶导的那段改用 "
                "linear_solver='dense'（稠密 Cholesky 全程是 torch 算子）。")
        s = ctx.solver
        data, Hj = ctx.saved_tensors
        lam = s._cudss_adjoint(data, gH.contiguous(), ctx.B, ctx.st, ctx.gen,
                               ctx.slot)
        g_data = g_F = None
        if ctx.needs_input_grad[0]:
            g_data = -(lam.index_select(1, s.A_csr_row)
                       * Hj.index_select(1, s.A_csr_col))
        if ctx.needs_input_grad[1]:
            g_F = lam
        return g_data, g_F, None, None, None, None


class GGASolver:
    """EPANET GGA 前向求解器（梯队 1）。

    GGASolver(net: Net, device='cpu', dtype=torch.float64)
    solve(demand_cfs, res_head_ft, ke_int=None, q0=None, e0=None,
          max_iter=None, accuracy=None) -> dict(head_ft, flow_cfs, emitter_cfs,
                                                iters, relerr)
    demand_cfs/res_head_ft 支持 [N] 或 [B,N]；q0/e0 为热启动（EPANET EPS 逐帧续用
    上一帧的 LinkFlow / EmitterFlow，hydraul.c 的 runhyd 循环不重跑 inithyd）。

    线性求解两种模式（mode）：
    - 'epanet'（默认）：忠实复刻 EPANET 稀疏 Cholesky（dgga.smatrix：genmmd MMD
      重排 + smatrix.c linsolve 逐句移植，装配次序也按 hydcoeffs.c 链路循环序）。
      系统含 1/CSMALL=1e6 与 RQtol 钳位 1e7 级对角、条件数 ~1e9，float64 前向误差
      ~eps·κ·‖H‖≈4e-6 ft；实测稠密 Cholesky 与 EPANET 的解差 1e-6~4e-6 ft，
      过不了对拍硬门槛 1e-6，只能复刻其运算次序。逐样本 numpy 路径。
    - 'dense'：torch.linalg.cholesky + cholesky_solve 的稠密批量路径（天然支持
      批维/GPU/自动微分），与 EPANET 差 ~3e-6 ft 量级。
    """

    def __init__(self, net: Net, device="cpu", dtype=torch.float64, mode="epanet",
                 inp_path=None, dense_tank_bound_check=True,
                 dense_status_machine=False):
        if mode not in ("epanet", "dense"):
            raise ValueError(f"未知 mode: {mode}")
        self.mode = mode
        # ---- 门 B1（新准入，缺省 False ⇒ 全部既有行为逐位不变）----
        # dense_status_machine=True 打开"批量状态机不动点"（hydsolve
        # hydsolver.c:150-189 的批量复刻，见 _linkstatus_batch / solve 的
        # status_machine 分支），并因此放行 CVPIPE（cvstatus hydstatus.c:177-202
        # 是它与 PIPE 的唯一差别，系数侧同分支 headlosscoeffs:138-141）。
        # PRV 轮：同一开关下再放行 PRV（批量 prvcoeff/prvstatus，见
        # _prvcoeffs_batch / _prvstatus_batch；此前构造期即 raise ⇒ 新准入，
        # 不改变任何既有网的行为）。PSV/FCV 仍拒绝。
        # 只开门不改缺省：solve() 仍需显式 status_machine=True 才走状态机。
        self.dense_status_machine = bool(dense_status_machine)
        if self.dense_status_machine and mode != "dense":
            raise ValueError("dense_status_machine 只对 mode='dense' 有意义")
        # 稠密通路迭代精化步数（PRV 轮做成可调；缺省 2 = 历史值，逐位不变）。
        # 依据 dense_gap_plan.md §1：ACTIVE PRV 的 CBIG 对角把 κ 从 ~1e5 推到
        # ~1e11 量级，f64 的 eps·κ ≈ 5e-5 需靠精化压回；κ 在线监控见 solve()
        # 输出的 diag_ratio（对角比值代理，仅含 PRV 时计算）。
        self.dense_refine = 2
        # dense 批量 PRV 静态数据的缺省（无 PRV / epanet 模式恒为 0/False；
        # 真值由 _build_dense_prv_data 填，仅 dense_status_machine=True 时调）
        self._dense_prv_np = 0          # setting!=MISSING 的 PRV 数（valvecoeffs 承接）
        self._dense_prv_np_all = 0      # 全部 PRV 数
        self._dense_prv_fixed_any = False   # setting==MISSING 的固定 PRV 存在？
        self._inp_path = inp_path
        self.net = net
        self.device = torch.device(device)
        self.dtype = dtype

        lt = np.asarray(net.link_type)
        nt = np.asarray(net.node_type)

        # ---- 范围检查 ----
        # epanet 模式（梯队3/任务D 起）：PIPE/CVPIPE/TCV/PUMP/PRV/PSV/FCV + tank；
        # dense 模式（任务E 起）：PIPE/TCV/PUMP + tank（tank 见下方开区间守卫）；
        # dense + dense_status_machine（门 B1 起）另放行 CVPIPE，PRV 轮起再放行
        # PRV（批量 prvcoeff/prvstatus，_build_dense_prv_data 的守卫另查
        # 端点/精度）。dense 仍缺：PSV（ACTIVE 罚函数行 hydcoeffs.c:1024-1027）、
        # FCV（ACTIVE 切流 :1073-1084 且改 Xflow ⇒ 跨阀依赖）、PBV/GPV。
        allowed = ({_CVPIPE, _PIPE, _TCV, _PUMP, _PRV, _PSV, _FCV}
                   if mode == "epanet" else
                   ({_CVPIPE, _PIPE, _TCV, _PUMP, _PRV}
                    if self.dense_status_machine
                    else {_PIPE, _TCV, _PUMP}))
        bad_links = set(lt.tolist()) - allowed
        if bad_links:
            _lname = {_CVPIPE: "CVPIPE", _PIPE: "PIPE", _PUMP: "PUMP", _PRV: "PRV",
                      _PSV: "PSV", _PBV: "PBV", _FCV: "FCV", _TCV: "TCV",
                      _GPV: "GPV"}
            raise NotImplementedError(
                f"mode={mode} 只支持 link_type∈{sorted(allowed)}，"
                f"出现 link_type={sorted(bad_links)}"
                f"（{', '.join(_lname.get(b, str(b)) for b in sorted(bad_links))}）")
        # 水损公式（input1.c:294-295：HW→Hexp=1.852，否则 2.0）
        self.headloss_form = str(net.meta.get("headloss", "H-W")).upper()
        if self.headloss_form not in ("H-W", "D-W", "C-M"):
            raise NotImplementedError(f"未知水损公式 {self.headloss_form}")
        if self.headloss_form == "D-W" and mode != "epanet":
            raise NotImplementedError("D-W 仅 epanet 模式（dense 路径留接口）")
        # 水池：单帧稳态下 tank 逐位等价于定水头节点 - 矩阵只有 Njuncs 阶
        # （hydsolver.c:123），tank/reservoir 编号排在 Njuncs 之后（types.h:816-818），
        # 只经 linkcoeffs 的 F += P*NodeHead（hydcoeffs.c:240,251）进右端；单帧内
        # NodeHead[tank]=tank->H0 由 inithyd 设死（hydraul.c:110），tanklevels 只在
        # 时段之间调用（hydraul.c:247,655-657）。dense 因此复用既有的定水头装配。
        # 唯一差别是 tankstatus（hydstatus.c:401-476）：满池/空池会把连接管置
        # TEMPCLOSED - dense 无状态机，复刻不了 ⇒ 见 __init__ 末尾的开区间守卫
        # _check_dense_tank_bounds()（需 self.htol，故推迟到 _build_b2_data 之后）。
        # 需水模型：demandcoeffs/newdemandflows 在 DDA 下直接 return
        # （hydcoeffs.c:438 / hydsolver.c:541）；PDA 分支未实现，防止静默算错
        if str(net.meta.get("demand_model", "DDA")).upper() != "DDA":
            raise NotImplementedError("梯队1 只支持 DDA 需水模型（PDA 留接口）")

        N, L = net.N, net.L
        self.N, self.L = N, L

        # ---- junction 掩码与编号（junction → 稠密矩阵行号）----
        is_junc = nt == 0
        self.junc_nodes = np.where(is_junc)[0]          # 行号 → 节点索引
        self.Nj = int(is_junc.sum())
        junc_row = np.full(N, -1, dtype=np.int64)
        junc_row[self.junc_nodes] = np.arange(self.Nj)
        self.fixed_nodes = np.where(~is_junc)[0]        # 定水头节点（本期全为水库）

        t64 = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)
        ti = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=self.device)

        n1 = np.asarray(net.link_n1, dtype=np.int64)
        n2 = np.asarray(net.link_n2, dtype=np.int64)
        self.n1, self.n2 = ti(n1), ti(n2)

        # ---- 管段静态属性 ----
        self.is_pipe = torch.as_tensor(lt == _PIPE, device=self.device)
        self.is_tcv = torch.as_tensor(lt == _TCV, device=self.device)
        # 关闭支：init_status==0（本期无状态机，状态恒定）
        self.closed = torch.as_tensor(np.asarray(net.init_status) == 0,
                                      device=self.device)
        self.r_hw = t64(net.r_hw)
        self.km_pipe = t64(net.km_int)   # 管道局损 Km（0.02517*K/D^4，parse 已算）
        self.diam = t64(net.diam_ft)

        # TCV 的有效 Km：
        # - Active(2)（setting 非 MISSING）：hydcoeffs.c:929-932
        #   link->Km = 0.02517 * setting / (SQR(Diam)*SQR(Diam))
        # - 固定 Open(1)（setting=MISSING）：沿用 [VALVES] 行局损 Km（tcvcoeff 不改写）
        st = np.asarray(net.init_status)
        setting = np.asarray(net.valve_setting_user, dtype=np.float64)
        d4 = np.asarray(net.diam_ft) ** 2 * np.asarray(net.diam_ft) ** 2
        km_tcv = np.where(st == 2,
                          np.divide(0.02517 * setting, d4,
                                    out=np.zeros_like(setting), where=d4 > 0),
                          np.asarray(net.km_int))
        self.km_tcv = t64(km_tcv)
        # 阀最小损 Km（link->Km；valvecoeff:1126 与 prvstatus:267 / psvstatus:327 用；
        # inp_path 时由 _apply_exact_props 按 input1.c:655 精确重算覆盖）
        self.km_valve_ml = np.asarray(net.km_int, dtype=np.float64).copy()
        # INP [OPTIONS] 覆盖前的默认（meta 或源码默认；_apply_exact_props 可覆盖）
        self._spgrav = float(net.meta.get("spgrav", 1.0))   # input1.c:121
        self._press = str(net.meta.get("press_units", "") or "")

        # ---- 装配索引（一次预处理）----
        j1 = junc_row[n1]
        j2 = junc_row[n2]
        m1 = is_junc[n1]              # n1 端为 junction
        m2 = is_junc[n2]              # n2 端为 junction
        both = m1 & m2
        Nj = self.Nj
        # A 的 scatter 目标（展平 [Nj*Nj]）：链路序 = [两端 junction 的 ±非对角×2,
        # n1 junction 对角, n2 junction 对角]（hydcoeffs.c:228-246 的语义）
        #
        idx_off1 = j1[both] * Nj + j2[both]
        idx_off2 = j2[both] * Nj + j1[both]
        # 对称装配定序（p5 审计 §1.2 / commit a78c185 的最小复现）：
        # 旧映射在"同一无序节点对 ≥3 条链路且方向不全同"时，(i,j)/(j,i) 两槽
        # 吃到同一组加数的**不同交错次序**（正向链路先进 (i,j) 槽、反向先进
        # (j,i) 槽），CPU 串行归约下差 ~1 ULP（NW_Model 40 对触发，
        # max|A-A^T|=2.84e-14）。修法：**只对这类节点对**把两个槽都按
        # (min,max) 定序 - 上三角段与下三角段对该对的链路按同一链路次序累加，
        # A 按构造逐位对称。其余链路的 A_idx **逐字节不变**：
        #   · 方向全同的并联对：两槽本就各按链路序吃同一序列；
        #   · 恰 2 条混向：两槽各吃同 2 个加数、次序互换，散射基是 0 ⇒
        #     (0+a)+b == (0+b)+a 逐位相等（IEEE 加法两元可交换）。
        # 不触发的网（全部回归网 + 门 B1 11 网实测 0 对）A_idx 逐字节同旧版，
        # 前向与反向（scatter_add 的 gather 反向按槽取 gA，gA 不逐位对称，
        # 换槽会把 ~1e-14 的舍入序抖进梯度 - 所以不能全局换 (min,max)）
        # 全通路逐位不变；影子包 sha256 对拍见 data/prv_pre_wip.txt。
        pair_lo = np.minimum(j1[both], j2[both])
        pair_hi = np.maximum(j1[both], j2[both])
        pkey = pair_lo * Nj + pair_hi
        uq, inv, cnt = np.unique(pkey, return_inverse=True, return_counts=True)
        n_fwd = np.zeros(uq.size, dtype=np.int64)
        np.add.at(n_fwd, inv, (j1[both] == pair_lo).astype(np.int64))
        trig = ((cnt >= 3) & (n_fwd != 0) & (n_fwd != cnt))[inv]
        idx_off1 = np.where(trig, pair_lo * Nj + pair_hi, idx_off1)
        idx_off2 = np.where(trig, pair_hi * Nj + pair_lo, idx_off2)
        idx_d1 = j1[m1] * Nj + j1[m1]
        idx_d2 = j2[m2] * Nj + j2[m2]
        self.A_idx = ti(np.concatenate([idx_off1, idx_off2, idx_d1, idx_d2]))
        self.lk_both = ti(np.where(both)[0])
        self.lk_m1 = ti(np.where(m1)[0])         # n1 端为 junction 的链路
        self.lk_m2 = ti(np.where(m2)[0])
        self.f_idx1 = ti(j1[m1])                 # F[j1] += Y（hydcoeffs.c:236）
        self.f_idx2 = ti(j2[m2])                 # F[j2] -= Y（hydcoeffs.c:247）
        # 定水头端"接地"贡献（hydcoeffs.c:240,251：定水头端只进对侧 junction 的 RHS）
        g1 = (~m1) & m2                          # n1 定水头、n2 junction
        g2 = (~m2) & m1                          # n2 定水头、n1 junction
        self.lk_g1 = ti(np.where(g1)[0])
        self.g1_row = ti(j2[g1])                 # F[Row[n2]] += P*Head[n1]
        self.g1_src = ti(n1[g1])
        self.lk_g2 = ti(np.where(g2)[0])
        self.g2_row = ti(j1[g2])                 # F[Row[n1]] += P*Head[n2]
        self.g2_src = ti(n2[g2])

        # ---- CSR 位型（sparse_gpu_plan.md §1a；与稠密装配并行的第二条通路）----
        # A 的位型由拓扑定死、迭代中不变 ⇒ 一次预计算即可。稠密路径每轮开
        # [B, Nj*Nj]（BWSN_2 的 Nj=12523、B=256 时实测口径 299 GiB）就是显存墙所在；
        # CSR 值张量只有 [B, nnz]（同网 nnz=41155 ⇒ 80 MiB，3811x）。
        # **本类的缺省仍是 dense，此处只是多存几个索引张量。**
        #
        # 位型 = 四段链路贡献的并集 ∪ **全部对角**。对角无条件全入的理由：
        #   稠密路径下孤立 junction（不接任何链路）的 A[j,j] 来自 zeros 初值 = 0.0，
        #   随后 emittercoeffs 的 diag_embed 再往上加 em/hgrad（无 emitter 则加 0.0，
        #   A[j,j] 恒 0 ⇒ Cholesky 必失败，这是稠密路径既有的行为，不在此处改变）。
        #   CSR 显式存下这个结构零，才能逐位复现稠密的 A，也才有位置承接 emitter 对角。
        # 次序：uniq 按 row*Nj+col 升序 = (row,col) 字典序 = CSR 标准次序，
        #   故 searchsorted 直接给出"链路贡献 → CSR data 下标"的映射，且该映射
        #   在四段拼接的**源序不变**，scatter_add 的累加次序与稠密路径逐位一致。
        a_flat = np.concatenate([idx_off1, idx_off2, idx_d1, idx_d2])
        diag_flat = np.arange(Nj, dtype=np.int64) * (Nj + 1)
        csr_pos = np.unique(np.concatenate([a_flat, diag_flat]))
        self.A_csr_nnz = int(csr_pos.size)
        # indptr/indices 存 int32：最大网 BWSN_2 的 nnz=41155、Nj=12523，远小于 2^31，
        # 且 cuSPARSE/cuDSS 的默认索引类型就是 int32（省一半索引带宽，接入不需转换）。
        # A_csr_scatter/A_csr_diag/A_csr_dense_pos 必须 int64 - torch 的
        # scatter_add_/scatter/index_select 只接受 int64 索引。
        self.A_csr_indptr = torch.as_tensor(
            np.concatenate([[0], np.cumsum(np.bincount(csr_pos // Nj,
                                                       minlength=Nj))]
                           ).astype(np.int32), device=self.device)
        self.A_csr_indices = torch.as_tensor((csr_pos % Nj).astype(np.int32),
                                             device=self.device)
        self.A_csr_scatter = ti(np.searchsorted(csr_pos, a_flat))
        self.A_csr_diag = ti(np.searchsorted(csr_pos, diag_flat))
        self.A_csr_dense_pos = ti(csr_pos)   # CSR data 下标 → 展平稠密位置
        # 稀疏 SpMV（§1b 迭代精化的残差）用的行/列号（int64，torch 索引要求）
        self.A_csr_row = ti(csr_pos // Nj)
        self.A_csr_col = ti(csr_pos % Nj)
        # cuDSS 有状态求解器缓存：键 (B, dtype, device, matrix_type)，值见 _cudss_state()。
        # plan() 是每(拓扑,批量)一次性成本（P0 实测 ~0.98 ms/矩阵），绝不进迭代循环。
        # **有界 LRU**（P2 审计 F4）：每份 state 除了 torch 侧的 [B,nnz]/[B,Nj]
        # 缓冲，还带一块 cuDSS 内部的分解缓冲（ky4 实测 228 MiB @B=256、
        # 546 MiB @B=1024，边际份 97.6 MiB @B≈216 - 作业 1459303/1459312），
        # 那块**不走 torch 缓存分配器**，torch.cuda.empty_cache() 一个字节都
        # 收不回，只有 DirectSolver.free() 能还。无界缓存下变批量训练会按每个 B
        # 各存一份单调堆积（ky4 80 个不同批量 → 8464 MiB，外推 ~325 个吃满 32 GiB）。
        # 见 cudss_cache_max / cudss_free() / cudss_cache_info()。
        self._cudss_cache = collections.OrderedDict()
        # None = 用 nvmath 的缺省 DirectSolverMatrixType.GENERAL（带主元 LU）。
        # A 对称正定，可改 DirectSolverMatrixType.SPD 让 cuDSS 走 Cholesky；
        # 改了会换分解算法，因此不做缺省，由调用方显式赋值。
        self.cudss_matrix_type = None
        self.cudss_refine = 2          # 迭代精化步数（与稠密通路同为 2）
        # cuDSS state 的 LRU 上限（None = 不限，即 P2 首版行为）。
        # 缺省 8 的定价依据（全部实测，见 data/p2_f4_wip.txt §3；
        # 作业 1459303 / 1459312，节点 <node-20>，ky4）：
        #   · **逐出很贵**：B=256 重建一份要 978 ms（3.82 ms/矩阵），常驻一份只要
        #     228 MiB - 上一步刚把该配置的整解峰值从 7302 砍到 135 MiB，
        #     回吐 0.2 GiB 换 1 秒是划算的；
        #   · **卡太紧会砸训练**：B 在 {1,8,64,256} 间轮转时上限 1/2 让 LRU 次次
        #     miss，每轮 232 → 1466 ms（**6.3x**），只省 96/64 MiB；
        #     "整批 + 尾批"形态下上限 1 多付 **2.04x**（6644 vs 3251 ms/4 epoch）
        #     而末态显存一模一样（848.2 MiB），一分钱不省；
        #   · 上限 4 与 8 都与"不限"逐项同表现（232.0 / 229.3 vs 232.4 ms/轮），
        #     取 8 是给分桶式 DataLoader（常见 4~8 桶）留一档余量；
        #   · 上限是硬界：最坏占用 = 上限 × 单份（单份可由 cudss_cache_info() 读）。
        #     ragged 变批量实测：不限单调爬到 8464 MiB，上限 8 持平在 1680 MiB
        #     （5.04x 更低），而两者耗时相同（75.4 vs 74.9 s） - **这个形态下
        #     逐出不要钱**。
        #   · 轮转周期 k > 上限时 LRU 必然次次 miss（任何计数式策略都一样）。
        #     真要在 k>8 个批量间轮转：调大上限，或在数据侧分桶/补齐 -
        #     cuDSS 的 plan 按 (拓扑, B) 定死（nvmath 1.0.0 只有显式批量），
        #     批量数就是 plan 的份数，这一点靠缓存策略换不掉。
        #   · 不做"按显存预算逐出"：单份占用只能透过全局 mem_get_info 观察，
        #     会被同卡上别的分配污染，做成缺省不可复现。
        self.cudss_cache_max = 8
        # ---- P3（autograd，sparse_gpu_plan.md §1c）----
        # 反向解 A·λ = ∂L/∂x 的迭代精化步数。缺省 2 = 与前向同等待遇
        # （前向 1 主解 + 2 精化）。设 0 可省两次 solve（那条与 Nj 无关的地板），
        # 代价见 data/p3_autograd_wip.txt §2.3。
        self.cudss_grad_refine = 2
        # 反向复用前向数值分解的**并发槽数**。一份 cuDSS state 同一时刻只持有
        # 一个数值分解，而展开 K 步的反向要按 K..1 逐个回用 ⇒ 想让每一步都
        # 零重分解，就得给每一步一份 state（前向轮转取槽 slot = i mod slots）。
        #   · slots=1（缺省）：只有"反向紧跟前向"的那一次能复用（单次线性解、
        #     隐式伴随、以及展开图的最后一步）。其余步在反向里重 factorize 一次，
        #     计入 cudss_counters()["bwd_refactorize"]，**不静默**。
        #   · slots>=K：反向零重分解（实测 factorize 计数 == K，见 wip §2.2），
        #     代价是 K 份 state 的常驻显存与 K 次 plan（都是每(拓扑,批量)一次性）。
        #     必须同时把 cudss_cache_max 调到 >= slots，否则 LRU 会把先建的槽
        #     逐出、反向又落回重分解（本类会在 slots>cache_max 时直接 raise）。
        self.cudss_grad_slots = 1
        self._cudss_slot_rr = 0        # 前向取槽的轮转指针（仅可微通路用）
        # 线性代数调用计数（纯 int 自增；缺省稠密通路一个都不碰）。
        # 用途：坐实"反向没有让 factorize 次数翻倍"。见 cudss_counters()。
        self._cudss_counters = dict(factorize=0, solve=0, bwd_solve=0,
                                    bwd_reuse=0, bwd_refactorize=0)

        self.junc_nodes_t = ti(self.junc_nodes)
        self.fixed_nodes_t = ti(self.fixed_nodes)
        # 节点序 [junc_nodes, fixed_nodes] → 原始节点序 的逆置换（dense 路径用
        # cat+index_select 重建全头向量，替代就地 index_put_，保证 autograd 干净）
        self._hperm_inv = ti(np.argsort(np.concatenate([self.junc_nodes,
                                                        self.fixed_nodes])))
        self.el_junc = t64(np.asarray(net.elev_ft)[self.junc_nodes])
        self.node_ke_default = t64(net.node_ke)

        # ---- 标量参数（meta / 源码默认值）----
        # Hexp：HW=1.852，D-W/C-M=2.0（input1.c:294-295）
        self.hexp = HEXP_HW if self.headloss_form == "H-W" else 2.0
        # D-W 所需：内部运动黏度（input1.c:271-282，parse 已换算）与粗糙度内部值
        self.viscos = float(net.meta.get("viscosity", VISCOS_DEFAULT))
        self.kc_np = np.asarray(net.roughness, dtype=np.float64).copy()
        self.qexp = float(net.meta["qexp"])       # emitter 指数 Qexp = 1/γ
        # RQtol：默认 1e-7（input1.c:35），INP 可设（input3.c:2022-2026）
        self.rqtol = float(net.meta.get("rqtol", RQTOL_DEFAULT))
        self.max_iter_default = int(net.meta["trials"])
        # INP 的 ACCURACY 被 EPANET 钳位到 [1e-5, 1e-1]（input3.c:2014-2019：
        # y=MAX(y,1.e-5); y=MIN(y,1.e-1)），仅 EN_setoption API 才允许更小值；
        # 随机网 INP 写 1e-8 但 DLL 实际用 1e-5，不钳位会多迭代一次
        self.hacc_default = min(max(float(net.meta["accuracy"]), 1.0e-5), 1.0e-1)
        # DampLimit 默认 0（input1.c:38）→ RelaxFactor 恒 1.0（hydsolver.c:149-162）；
        # DampLimit>0 的 0.6 阻尼与 valvestatus 延迟未实现，防静默算错
        self.damp_limit = float(net.meta.get("damp_limit", DAMPLIMIT_DEFAULT))
        if self.damp_limit > 0.0:
            raise NotImplementedError(
                "DAMPLIMIT>0（RelaxFactor=0.6 阻尼，hydsolver.c:152-158）不在本期范围")
        # HeadErrorLimit/FlowChangeLimit 默认 0（input1.c:108-109）；>0 时 hasconverged
        # 需 checkhydbalance（hydsolver.c:631-635），未实现，防静默算错
        if float(net.meta.get("headerror", 0.0)) > 0.0 \
                or float(net.meta.get("flowchange", 0.0)) > 0.0:
            raise NotImplementedError(
                "HEADERROR/FLOWCHANGE>0 收敛判据（hydsolver.c:631-635）不在本期范围")

        # ---- 可选：用 INP 原文按 EPANET 精确运算链重建管段属性（位级对拍所需）----
        if inp_path is not None:
            self._apply_exact_props(inp_path, st)

        # ---- epanet 模式：EPANET 稀疏机制 + 装配次序预处理（numpy）----
        self._build_epanet_path(n1, n2, nt)

        # ---- B2：泵/水池静态数据（numpy 路径）----
        self._build_b2_data(lt, nt)

        # ---- 任务E：dense（批量可微）路径的泵/水池静态数据与能力守卫 ----
        if self.mode != "epanet":
            # F1（门 B1 审计）：pswitch 可达性守卫下沉到 dense 构造期通用位置。
            # 旧位置在 _build_dense_sm_data（只有 dense_status_machine=True 才查），
            # 缺省 dense 完全没有 pswitch（hydsolver.c:268-355），凡 INP 含
            # "监测 junction 水头"的简单控制就会静默算错（实测 Net1 注入一条
            # 该类控制后 epanet 侧改 1 条状态、头差 80.27 ft，而缺省 dense
            # 不 raise）。21 个公开网该类控制数实测为 0 ⇒ 缺省行为不变。
            self._check_dense_pswitch()
            self._build_dense_pump_data()
            self._build_dense_tank_data()
            if self.dense_status_machine:
                # 状态机在位 ⇒ tankstatus（hydstatus.c:401-476）由
                # _linkstatus_batch 真正执行，构造期那条"H0 必须落在开区间内"
                # 的静态守卫是为"没有状态机"准备的，此时必须撤掉（否则
                # Anytown_wntr / ky4 这类 H0==Hmin 的网会被误拒）。
                self._build_dense_sm_data()
                self._build_dense_prv_data()
            elif dense_tank_bound_check:
                self._check_dense_tank_bounds()

    # ------------------------------------------------------------------
    def _check_dense_tank_bounds(self):
        """dense 路径把 tank 当定水头节点接入，前提是 tankstatus（hydstatus.c:401-476）
        全程不动作。

        tankstatus 只有两个触发条件：
          - 满池 `NodeHead[n1] >= tank->Hmax - Htol && !tank->CanOverflow`（:444）
          - 空池 `NodeHead[n1] <= tank->Hmin + Htol`（:461）
        命中后把连接管置 TEMPCLOSED（:449/:454/:466/:471），dense 无状态机、复刻不了。
        input1.c:365-367 已校验 Hmin<=H0<=Hmax，单帧内 NodeHead[tank]≡H0
        （inithyd hydraul.c:110），所以 H0 严格落在开区间内 ⇒ 该函数全程不动作；
        落在边界上必须 raise（铁律：不静默近似）。Htol 用项目既有口径 self.htol
        （input1.c:27 HTOL=0.0005，INP 可经 [OPTIONS] HTOL 覆盖）。"""
        if not self.n_tanks:
            return
        bad = []
        for n in self.tank_nodes:
            n = int(n)
            h0, hmin, hmax = self.tn_h0[n], self.tn_hmin[n], self.tn_hmax[n]
            if h0 >= hmax - self.htol and not self.tn_overflow[n]:
                bad.append(f"{self.net.node_id[n]}: H0={h0:.6f} >= Hmax-Htol="
                           f"{hmax - self.htol:.6f}（满池, CanOverflow=0）")
            if h0 <= hmin + self.htol:
                bad.append(f"{self.net.node_id[n]}: H0={h0:.6f} <= Hmin+Htol="
                           f"{hmin + self.htol:.6f}（空池）")
        if bad:
            raise NotImplementedError(
                "mode=dense 的水池按定水头节点接入，要求 H0 严格落在 (Hmin+Htol, "
                "Hmax-Htol) 内；下列水池落在边界上，tankstatus（hydstatus.c:401-476）"
                "会把其连接管置 TEMPCLOSED，dense 路径未实现该状态切换：\n  "
                + "\n  ".join(bad)
                + f"\n（Htol={self.htol:g}，input1.c:27）改用 mode='epanet' 并开"
                  " status_machine=True。\n"
                  "注：H0 落在边界只是 tankstatus 动作的**必要**条件（还要叠加泵方向"
                  ":449/:466 或 cvstatus :454/:471）。若确认某场景不触发，可用 "
                  "GGASolver(..., dense_tank_bound_check=False) 关掉这条静态守卫 - "
                  "solve() 里 _check_dense_tank_status 会对收敛解逐字复核完整判据，"
                  "真触发时照样 raise，不会静默算错。")

    def _build_dense_tank_data(self):
        """tankstatus（hydstatus.c:401-476）事后复核所需的静态索引。

        选端逻辑逐字照抄 :423-438：先看 n1 是否为定水头节点（`i = n1 - Njuncs`
        大于 0），是则就用 n1、**不再回退到 n2**；否则看 n2 并把流量取反；
        随后 `if (tank->A == 0.0) return;` 把 reservoir 排除（:436-438）。
        因此 reservoir--tank 的链路在 EPANET 里被整条跳过，此处必须同样跳过。"""
        n1, n2 = self.n1_np, self.n2_np
        fx = self.is_fixed_node
        use_n2 = (~fx[n1]) & fx[n2]                      # :425-433 交换
        cand = np.where(fx[n1], n1, np.where(fx[n2], n2, -1))
        sel = (cand >= 0) & self.is_tank_node[np.where(cand >= 0, cand, 0)]
        k = np.where(sel)[0]
        dev = self.device
        self.tl_k = torch.as_tensor(k, dtype=torch.int64, device=dev)
        if k.size == 0:
            return
        tank = cand[k]
        other = np.where(use_n2[k], n1[k], n2[k])
        ti_ = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=dev)
        tf_ = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64),
                                        dtype=self.dtype, device=dev)
        self.tl_tank = ti_(tank)
        self.tl_other = ti_(other)
        self.tl_qsign = tf_(np.where(use_n2[k], -1.0, 1.0))
        self.tl_hmax = tf_(self.tn_hmax[tank])
        self.tl_hmin = tf_(self.tn_hmin[tank])
        self.tl_ovf = torch.as_tensor(self.tn_overflow[tank], device=dev)
        self.tl_is_pump = torch.as_tensor(self.is_pump_np[k], device=dev)
        self.tl_pump_in = torch.as_tensor(self.is_pump_np[k] & (n2[k] == tank),
                                          device=dev)   # :449 link->N2 == n1
        self.tl_pump_out = torch.as_tensor(self.is_pump_np[k] & (n1[k] == tank),
                                           device=dev)  # :466 link->N1 == n1

    def _cvstatus_closed_t(self, s_is_open, dh, q):
        """cvstatus（hydstatus.c:177-202）的张量版，返回"结果为 CLOSED"的布尔掩码。
        s_is_open: 传入的当前状态 s 是否为 OPEN（:200 的 `return s` 分支用）。"""
        big = dh.abs() > self.htol                       # :191
        closed_big = (dh < -self.htol) | (q < -self.qtol)         # :193-195
        rev = q < -self.qtol                                      # :199
        closed_small = rev if s_is_open else torch.ones_like(rev)  # :200 return s
        return torch.where(big, closed_big, closed_small)

    def _check_dense_tank_status(self, H, q):
        """收敛后逐字复核 tankstatus（hydstatus.c:401-476）是否会动作。

        dense 把 tank 当定水头节点，成立的唯一前提就是该函数全程不动作。构造期的
        `_check_dense_tank_bounds` 只查了 INP 的 H0（必要条件）；这里用**实际收敛解**
        （水头可由调用方经 res_head_ft 批量给定、与 H0 无关）查完整判据 - 满池
        （:444）/空池（:461）叠加泵方向（:449/:466）或 cvstatus（:454/:471）。
        命中即 raise，绝不静默把连接管当成一直开着。"""
        k = getattr(self, "tl_k", None)
        if k is None or k.numel() == 0:
            return
        live = ~self.closed_dense.index_select(0, k)     # :421 LinkStatus<=CLOSED
        ht = H.index_select(1, self.tl_tank)             # 水池端水头
        dh = ht - H.index_select(1, self.tl_other)       # :441
        qk = q.index_select(1, k) * self.tl_qsign        # :423-433
        full = (ht >= self.tl_hmax - self.htol) & (~self.tl_ovf)      # :444
        empty = ht <= self.tl_hmin + self.htol                        # :461
        cut_full = torch.where(self.tl_is_pump, self.tl_pump_in,
                               self._cvstatus_closed_t(True, dh, qk))  # :449/:454
        cut_empty = torch.where(self.tl_is_pump, self.tl_pump_out,
                                ~self._cvstatus_closed_t(False, dh, qk))  # :466/:471
        hit = live & ((full & cut_full) | (empty & cut_empty))
        if bool(hit.any()):
            b, j = [int(x) for x in torch.nonzero(hit)[0]]
            kk = int(k[j])
            n = int(self.tl_tank[j])
            raise NotImplementedError(
                f"mode=dense：场景 b={b} 的水池 {self.net.node_id[n]}（水头 "
                f"{float(ht[b, j]):.6f} ft, Hmin={self.tn_hmin[n]:.6f}, "
                f"Hmax={self.tn_hmax[n]:.6f}, Htol={self.htol:g}）满/空，"
                f"tankstatus（hydstatus.c:401-476）会把连接管 "
                f"{self.net.link_id[kk]} 置 TEMPCLOSED。dense 路径没有状态机，"
                "继续算下去等于把该管当成一直开着 - 拒绝静默近似。请改用 "
                "mode='epanet' 并开 status_machine=True。")

    # ------------------------------------------------------------------
    # 门 B1：批量状态机不动点（hydsolve hydsolver.c:150-189 的批量复刻）
    # ------------------------------------------------------------------
    def _check_dense_pswitch(self):
        """F1：pswitch 可达性守卫（dense 构造期通用，含缺省 dense 与批量状态机）。

        pswitch（hydsolver.c:268-355）在收敛后检查简单控制里"监测 junction
        水头"的条目并改 LinkStatus/LinkSetting（含泵转速）。dense 路径（无论
        是否开 dense_status_machine）都没有 pswitch 的对应物：缺省 dense 完全
        不查状态，批量状态机也只覆盖 linkstatus 三条转移 - 这类 INP 放行就是
        静默算错（门 B1 审计 F1：Net1 注入一条 junction 压力控制后 epanet 侧
        改 1 条状态、头差 80.27 ft，缺省 dense 不报错）。监测 tank 水位的控制
        属 EPS 时段间语义（不进 hydsolve），不在此守卫范围。
        21 个公开网实测该类控制数为 0 ⇒ 守卫对现有网表不改变任何行为。"""
        net = self.net
        nt = np.asarray(net.node_type)
        cn = np.asarray(net.ctl_node, dtype=np.int64) if net.ctl_node is not None \
            else np.zeros(0, dtype=np.int64)
        ct = np.asarray(net.ctl_type, dtype=np.int64) if net.ctl_type is not None \
            else np.zeros(0, dtype=np.int64)
        if cn.size:
            hit = (cn >= 0) & (ct < 2) & (nt[np.where(cn >= 0, cn, 0)] == 0)
            if bool(hit.any()):
                i = int(np.where(hit)[0][0])
                raise NotImplementedError(
                    "mode=dense：本网含监测 junction 水头的简单控制"
                    f"（控制 #{i}，节点 {net.node_id[int(cn[i])]} → 链路 "
                    f"{net.link_id[int(net.ctl_link[i])]}），pswitch"
                    "（hydsolver.c:268-355）会在收敛后改其状态/设定；dense 路径"
                    "（含批量状态机）没有 pswitch 的对应物，放行会静默算错。"
                    "请改用 mode='epanet' 并开 status_machine=True。")

    def _build_dense_sm_data(self):
        """批量状态机所需的静态张量 + 本轮**明确不做**的语义的准入守卫。

        已实现的转移（与 dense 现放行的元件一一对应）：
          · 重开临时关闭链路 XHEAD/TEMPCLOSED → OPEN（hydstatus.c:131-137）
          · cvstatus（:177-202） - CVPIPE
          · pumpstatus（:205-239） - PUMP 的 XHEAD 截断
          · tankstatus（:401-476） - 满/空池把连接管置 TEMPCLOSED
          · valvestatus→prvstatus（:33-98, :242-299） - PRV（PRV 轮；批量版
            _prvstatus_batch，每轮迭代跑，节律见 solve 的 valvestatus 块）
        **不做**（dense 不放行这些元件，故不构成缺口）：
          · psvstatus（:302-359） - PSV；fcvstatus（:362-398） - FCV
          · badvalve（hydsolver.c:215-265） - 批量无"只重试第 b 个样本"的
            语义；批 Cholesky 失败显式捕获+报样本号（solve 的 cholesky_ex 支）
        **拒绝**（会静默算错，故构造期 raise）：
          · pswitch（hydsolver.c:268-355）里"监测 junction 水头"的简单控制：
            它会在收敛后改 LinkStatus/LinkSetting（含泵转速），批量化需要
            per-scenario 的 [B,L] setting 与随之批量化的 pumpcoeff；21 个公开网
            实测该类控制数为 0（其余控制的监测点都是 tank，属 EPS 时段间语义、
            不进 hydsolve），本轮不做。
        """
        dev = self.device
        tb = lambda a: torch.as_tensor(np.asarray(a), device=dev)
        self.is_cv_t = tb(self.lt_np == _CVPIPE)
        sp = np.asarray(self.init_setting, dtype=np.float64)
        # 泵转速为 0 的高阻支（hydcoeffs.c:696-701 的 `setting == 0.0`）；
        # 与 closed_dense 的口径一致，只是状态位改由 S 逐场景给出。
        self.sm_zero_speed_t = tb(self.is_pump_np & (sp == 0.0))
        # linkstatus :144-148 的 `LinkSetting > 0` 前置条件
        self.sm_pump_live_t = tb(self.is_pump_np & (sp > 0.0))
        # pumpstatus 的截止扬程：CONST_HP→BIG（:223-227），否则 ω²·Hmax（:231）
        hmax = np.where(self.is_chp_np, BIG, sp * sp * self.pl_hmax)
        self.sm_pump_hmax_t = torch.as_tensor(hmax, dtype=self.dtype, device=dev)
        # 触及定水头节点的链路（tankstatus 的入口条件 hydstatus.c:158-161）已由
        # _build_dense_tank_data 的 tl_k 给出（reservoir 端按 :436-438 剔除）。
        # 准入守卫：pswitch 的 junction 压力控制 - F1 起由构造期通用位置
        # _check_dense_pswitch()（__init__ 对一切 mode='dense' 调用）承担。

    def _cvstatus_t(self, s, dh, q):
        """cvstatus（hydstatus.c:177-202）的张量版，返回**新状态**（int8）。

        与 _cvstatus_closed_t 的差别：那个只回答"是否 CLOSED"（tankstatus 用），
        这个要逐字给出 :195/:200 的三分支结果（含 `return s` 的保持支）。"""
        big = dh.abs() > self.htol                          # :191
        rev = q < -self.qtol
        cl = torch.full_like(s, self.ST_CLOSED)
        op = torch.full_like(s, self.ST_OPEN)
        new_big = torch.where((dh < -self.htol) | rev, cl, op)   # :193-195
        new_small = torch.where(rev, cl, s)                      # :199-200
        return torch.where(big, new_big, new_small)

    def _linkstatus_batch(self, S, H, q):
        """linkstatus（hydstatus.c:101-174）的 [B,L] 批量复刻。

        返回 (S_new int8[B,L], changed bool[B])。changed 的口径与 C 一致：
        与**函数入口**的状态比较（:126 `status = S[k]` 取在重开 :131-137 之前），
        所以"把 TEMPCLOSED 重开成 OPEN"本身就算一次变化。

        dense 只放行 PIPE/CVPIPE/PUMP/TCV，故 :151-155 的 FCV 支恒不触发；
        PRV/PSV 的 valvestatus 在 hydsolve 的另一处（:156/:161），同样不触发。"""
        st0 = S                                              # :126 入口状态
        # 重开临时关闭链路（:131-137）
        s = torch.where((st0 == self.ST_XHEAD) | (st0 == self.ST_TEMPCLOSED),
                        torch.full_like(st0, self.ST_OPEN), st0)
        dh = H.index_select(1, self.n1) - H.index_select(1, self.n2)   # :129
        # CVPIPE（:139-143）
        if bool(self.is_cv_t.any()):
            s = torch.where(self.is_cv_t, self._cvstatus_t(s, dh, q), s)
        # PUMP（:144-148）：status>=OPEN 且 LinkSetting>0 时查 pumpstatus(-dh)
        if self.n_pumps:
            gain = -dh                                       # 扬程增益
            xh = torch.where(gain > self.sm_pump_hmax_t + self.htol,
                             torch.full_like(s, self.ST_XHEAD),
                             torch.full_like(s, self.ST_OPEN))   # :235,:238
            s = torch.where(self.sm_pump_live_t & (s >= self.ST_OPEN), xh, s)
        # tankstatus（:158-161 → :401-476）：满/空池置 TEMPCLOSED
        k = getattr(self, "tl_k", None)
        if k is not None and k.numel():
            live = s.index_select(1, k) > self.ST_CLOSED     # :421
            ht = H.index_select(1, self.tl_tank)
            dht = ht - H.index_select(1, self.tl_other)      # :441
            qk = q.index_select(1, k) * self.tl_qsign        # :423-433
            full = (ht >= self.tl_hmax - self.htol) & (~self.tl_ovf)   # :444
            empty = ht <= self.tl_hmin + self.htol                     # :461
            cut_f = torch.where(self.tl_is_pump, self.tl_pump_in,
                                self._cvstatus_closed_t(True, dht, qk))
            cut_e = torch.where(self.tl_is_pump, self.tl_pump_out,
                                ~self._cvstatus_closed_t(False, dht, qk))
            hit = live & ((full & cut_f) | (empty & cut_e))
            sk = torch.where(hit, torch.full_like(hit, self.ST_TEMPCLOSED,
                                                  dtype=s.dtype),
                             s.index_select(1, k))
            s = s.scatter(1, k.view(1, -1).expand(s.shape[0], -1), sk)
        return s, (s != st0).any(dim=1)                       # :164

    # ------------------------------------------------------------------
    # PRV 轮：dense 批量 PRV（prvcoeff hydcoeffs.c:942-992 + prvstatus
    # hydstatus.c:242-299 + valvestatus 节律 hydsolver.c:156/:161 的批量复刻）
    # ------------------------------------------------------------------
    def _build_dense_prv_data(self):
        """dense 批量 PRV 的静态张量与准入守卫（仅 dense_status_machine=True 调）。

        分两类（headlosscoeffs hydcoeffs.c:154-158 的分派）：
          · setting != MISSING（含 ACTIVE/后续被状态机改成 OPEN/CLOSED 的）：
            headloss 阶段 P=0 ⇒ linkcoeffs 整条跳过（:218，连 Xflow 都不进），
            系数改由 valvecoeffs→prvcoeff 在 nodecoeffs 之后接管
            （_prvcoeffs_batch）；这是"pcv"集合，也是状态机 valvestatus 的对象。
          · setting == MISSING（[STATUS] 固定 OPEN/CLOSED，changestatus
            input3.c:2168）：headloss 阶段走 valvecoeff（:1100-1151）开启支
            （_valve_PY_ml），关闭支由 closed_now 覆盖；valvestatus 跳过（:63）。

        守卫（新准入，宁 raise 不静默）：
          · PRV 两端必须都是 junction：ACTIVE 支写 Aii[Row[n2]]/F[Row[n2]]/
            F[Row[n1]] 且读 Xflow[n2] - 定水头端在 EPANET 里行号 >Njuncs 不进
            求解、Xflow 也另有口径，批量版不实现（ImplicitGGASolve 的
            _valve_act_masks 同口径拒绝）；
          · 含 pcv PRV 时 dtype 必须 float64：ACTIVE 的 CBIG=1e8 对角把 κ 推到
            ~1e11（dense_gap_plan §1 实测 D-Town 2.46e11），f32 的 eps·κ≈3e4
            完全没有有效数字，构造期直接拒绝。"""
        dev, dt = self.device, self.dtype
        prv_all = np.where(self.lt_np == _PRV)[0]
        self._dense_prv_np_all = int(prv_all.size)
        if not prv_all.size:
            return
        K0 = np.asarray(self.init_setting, dtype=np.float64)
        pcv = prv_all[K0[prv_all] != MISSING]     # valvecoeffs 承接（:308 跳 MISSING）
        fixedv = prv_all[K0[prv_all] == MISSING]
        self._dense_prv_np = int(pcv.size)
        self._dense_prv_fixed_any = bool(fixedv.size)
        m_fixed = np.zeros(self.L, dtype=bool)
        m_fixed[fixedv] = True
        self.prv_fixed_t = torch.as_tensor(m_fixed, device=dev)
        m_pcv = np.zeros(self.L, dtype=bool)
        m_pcv[pcv] = True
        self.pcv_static_t = torch.as_tensor(m_pcv, device=dev)
        if self._dense_prv_fixed_any:
            self.km_valve_ml_t = torch.as_tensor(self.km_valve_ml_np,
                                                 dtype=dt, device=dev)
        if not pcv.size:
            return
        if dt != torch.float64:
            raise NotImplementedError(
                "mode=dense 含 PRV（setting 未固定）时 dtype 必须 float64："
                "ACTIVE PRV 的 CBIG=1e8 对角把 κ(A) 推到 ~1e11 量级"
                "（dense_gap_plan.md §1 实测），f32 的 eps·κ≈3e4 完全没有"
                "有效数字 - 拒绝静默算错。")
        n1 = self.n1_np[pcv]
        n2 = self.n2_np[pcv]
        if self.is_fixed_node[n1].any() or self.is_fixed_node[n2].any():
            bad = pcv[self.is_fixed_node[n1] | self.is_fixed_node[n2]]
            raise NotImplementedError(
                "mode=dense 的批量 PRV 要求两端均为 junction；下列 PRV 有"
                "定水头端（EPANET 该行不进求解、Xflow 口径不同，且 ACTIVE 时"
                "触发 badvalve 病态修复）：%s。请改用 mode='epanet'。"
                % [self.net.link_id[int(k)] for k in bad])
        junc_row = np.full(self.N, -1, dtype=np.int64)
        junc_row[self.junc_nodes] = np.arange(self.Nj)
        j1 = junc_row[n1]
        j2 = junc_row[n2]
        ti = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=dev)
        tf = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64),
                                       dtype=dt, device=dev)
        self.prv_k_t = ti(pcv)                    # 文件序 = EPANET 阀序
        self.prv_n1_t = ti(n1)
        self.prv_n2_t = ti(n2)
        self.prv_j1_t = ti(j1)
        self.prv_j2_t = ti(j2)
        # hset = Node[n2].El + LinkSetting（prvcoeff:962-963；两者迭代中不变）
        self.prv_hset_np = self.elev_np[n2] + K0[pcv]
        self.prv_hset_t = tf(self.prv_hset_np)
        self.prv_km_np = self.km_valve_ml_np[pcv].copy()   # link->Km（:267/:1126）
        self.prv_km_t = tf(self.prv_km_np)
        # A 槽位（每阀 5 槽）：非 ACTIVE 的 (j1,j2)/(j2,j1)/(j1,j1)/(j2,j2)
        # （prvcoeff:987-989）+ ACTIVE 的 (j2,j2) CBIG（:975）。两个三角吃同一
        # 段值、同一次 scatter 的同一元素序 ⇒ A 按构造逐位对称（守卫 ⑨ 口径）。
        Nj = self.Nj
        flat = np.stack([j1 * Nj + j2, j2 * Nj + j1,
                         j1 * Nj + j1, j2 * Nj + j2, j2 * Nj + j2],
                        axis=1).reshape(-1)
        self.prv_A_idx = ti(flat)
        csr_pos = self.A_csr_dense_pos.cpu().numpy()
        pos = np.searchsorted(csr_pos, flat)
        if not (csr_pos[np.clip(pos, 0, csr_pos.size - 1)] == flat).all():
            raise RuntimeError("PRV 槽位不在 CSR 位型内（不应发生：位型含全部"
                               "链路两端槽与全对角）")
        self.prv_A_csr_idx = ti(pos)              # CSR 位型不变，只是值多两笔
        self.prv_F_idx = ti(np.stack([j1, j2], axis=1).reshape(-1))

    def _valve_PY_ml(self, q):
        """valvecoeff（hydcoeffs.c:1100-1151）开启支的批量版，Km = link->Km。

        仅供 setting==MISSING 的固定 PRV（headlosscoeffs:154-157 对 PRV/PSV/FCV
        的 MISSING 分派）；关闭支由调用方 closed_now 覆盖（:1118-1123 同式）。"""
        km = self.km_valve_ml_t
        qa = torch.abs(q)
        hgrad = 2.0 * km * qa                            # :1129
        lin = hgrad < self.rqtol                         # :1132
        hgrad = torch.where(lin, torch.full_like(hgrad, self.rqtol), hgrad)
        hloss = torch.where(lin, q * hgrad, q * hgrad / 2.0)   # :1135,:1137
        km_pos = km > 0.0                                # :1126
        P = torch.where(km_pos, 1.0 / hgrad, torch.full_like(q, 1.0 / CSMALL))
        Y = torch.where(km_pos, hloss / hgrad, q)        # :1140-1141/:1148-1149
        return P, Y

    def _prvcoeffs_batch(self, P, Y, F, Xflow, q, S, A=None, csr_data=None):
        """valvecoeffs→prvcoeff（hydcoeffs.c:282-330 / :942-992）的批量复刻。

        必须在 nodecoeffs 之后调用（ACTIVE 的 Y=LinkFlow+Xflow[n2] 读的是
        累加完 emitter 与需水之后的 Xflow，matrixcoeffs :184-194 的次序）。
        三支全算 + 掩码加权，不分桶；按文件序逐阀循环（循环体内对 B 全向量） -
        PRV 不写 Xflow 故无跨阀依赖，循环保留是给 FCV（fcvcoeff:1075-1076 改
        Xflow）留位。ACTIVE 只加对角 CBIG、不写非对角（"跳过写入"，A 保持
        对称 - cudss 反向复用分解的前提）；非 ACTIVE 两个三角吃同一段值 ⇒
        逐位对称。返回 (P, Y, F, A, csr_data)，A/csr_data 只更新传入的那个。"""
        dt = P.dtype
        B = P.shape[0]
        a_cols, f_cols, p_cols, y_cols = [], [], [], []
        for i in range(self._dense_prv_np):
            ki = int(self.prv_k_t[i])
            j2 = int(self.prv_j2_t[i])
            qk = q[:, ki]
            sk = S[:, ki]
            m_act = sk == self.ST_ACTIVE                 # :965
            m_cl = sk <= self.ST_CLOSED                  # valvecoeff:1118
            # ---- valvecoeff（:1100-1151）：OPEN/XPRESSURE 开启支 ----
            km = float(self.prv_km_np[i])
            qa = torch.abs(qk)
            if km > 0.0:                                 # :1126
                hgrad = 2.0 * km * qa                    # :1129
                lin = hgrad < self.rqtol
                hgrad = torch.where(lin, torch.full_like(hgrad, self.rqtol),
                                    hgrad)               # :1134
                hloss = torch.where(lin, qk * hgrad, qk * hgrad / 2.0)
                pk_o = 1.0 / hgrad                       # :1140
                yk_o = hloss / hgrad                     # :1141
            else:
                pk_o = torch.full_like(qk, 1.0 / CSMALL)  # :1148
                yk_o = qk                                # :1149
            pk_na = torch.where(m_cl, torch.full_like(qk, 1.0 / CBIG), pk_o)
            yk_na = torch.where(m_cl, qk, yk_o)          # :1120-1121
            mna = (~m_act).to(dt)
            mact = m_act.to(dt)
            xf2 = Xflow[:, j2]
            # ---- 装配（非 ACTIVE :987-991；ACTIVE :972-978）----
            pna = pk_na * mna
            a_cols += [-pna, -pna, pna, pna, CBIG * mact]
            dy = (yk_na - qk) * mna                      # :990-991
            f1 = dy + torch.where(m_act & (xf2 < 0.0), xf2,
                                  torch.zeros_like(xf2))  # :976-978
            f2 = -dy + float(self.prv_hset_np[i]) * CBIG * mact  # :974
            f_cols += [f1, f2]
            p_cols.append(pna)                           # ACTIVE → P=0（:972）
            y_cols.append(torch.where(m_act, qk + xf2, yk_na))   # :973
        a_vals = torch.stack(a_cols, dim=1)              # [B, 5*Np]
        f_vals = torch.stack(f_cols, dim=1)              # [B, 2*Np]
        P = P.index_copy(1, self.prv_k_t, torch.stack(p_cols, dim=1))
        Y = Y.index_copy(1, self.prv_k_t, torch.stack(y_cols, dim=1))
        F = F.scatter_add(1, self.prv_F_idx.view(1, -1).expand(B, -1), f_vals)
        if csr_data is not None:
            csr_data = csr_data.scatter_add(
                1, self.prv_A_csr_idx.view(1, -1).expand(B, -1), a_vals)
        if A is not None:
            Nj = self.Nj
            A = A.reshape(B, Nj * Nj).scatter_add(
                1, self.prv_A_idx.view(1, -1).expand(B, -1), a_vals) \
                .reshape(B, Nj, Nj)
        return P, Y, F, A, csr_data

    def _prvstatus_batch(self, S, H, q):
        """valvestatus（hydstatus.c:33-98）→ prvstatus（:242-299）的 [B,·] 批量版。

        转移依赖旧状态 ⇒ one-hot(旧状态) 加权四个候选，全向量化（不分桶）。
        节律由调用方保证：每轮迭代都调（hydsolver.c:156/:161；DampLimit>0 的
        延迟支构造期已拒）。返回 (S_new int8[B,L], changed bool[B])。"""
        k = self.prv_k_t
        h1 = H.index_select(1, self.prv_n1_t)            # :74 H[n1]
        h2 = H.index_select(1, self.prv_n2_t)
        qk = q.index_select(1, k)
        sk = S.index_select(1, k)
        hset = self.prv_hset_t                           # :74（静态）
        hml = self.prv_km_t * (qk * qk)                  # :267 Km*SQR(LinkFlow)
        htol, qtol = self.htol, self.qtol
        neg = qk < -qtol
        CL = torch.full_like(sk, self.ST_CLOSED)
        OP = torch.full_like(sk, self.ST_OPEN)
        AC = torch.full_like(sk, self.ST_ACTIVE)
        c_act = torch.where(neg, CL,
                            torch.where(h1 - hml < hset - htol, OP, AC))  # :273-276
        c_open = torch.where(neg, CL,
                             torch.where(h2 >= hset + htol, AC, OP))      # :279-282
        c_cl = torch.where((h1 >= hset + htol) & (h2 < hset - htol), AC,
                           torch.where((h1 < hset - htol) & (h1 > h2 + htol),
                                       OP, CL))                           # :285-288
        c_xp = torch.where(neg, CL, sk)                  # :291-292
        s_new = torch.where(sk == self.ST_ACTIVE, c_act,
                            torch.where(sk == self.ST_OPEN, c_open,
                                        torch.where(sk == self.ST_CLOSED, c_cl,
                                                    torch.where(
                                                        sk == self.ST_XPRESSURE,
                                                        c_xp, sk))))
        S_out = S.scatter(1, k.view(1, -1).expand(S.shape[0], -1), s_new)
        return S_out, (s_new != sk).any(dim=1)           # :88/:94

    # ------------------------------------------------------------------
    def _build_dense_pump_data(self):
        """dense 泵支的静态张量（pumpcoeff hydcoeffs.c:673-791 的批量复刻所需）。

        与 numpy 路径 `_PY_np` 的泵分支同源；CUSTOM 曲线的分段查找由
        `_curvecoeff` 的 python 线性扫描（hydcoeffs.c:818-822）改写成
        `torch.searchsorted` + `gather` 的批量版：`while (k2<npts && x[k2]<q) k2++`
        与 searchsorted(side='left') 语义完全一致（首个 x[k2]>=q 的下标），
        前提是曲线 x 严格递增 - 构造期在此断言，不满足则 raise。"""
        dev, dt = self.device, self.dtype
        tb = lambda a: torch.as_tensor(np.asarray(a), device=dev)
        tf = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64),
                                       dtype=dt, device=dev)
        self.is_pump_t = tb(self.is_pump_np)
        if not self.n_pumps:
            self.pump_custom_links_t = None
            self.closed_dense = self.closed
            return
        # 转速 ω = LinkSetting 初值（inithyd hydraul.c:131 → link->Kc）
        sp = np.asarray(self.init_setting, dtype=np.float64)
        self.pump_speed_t = tf(sp)
        self.pump_h0_t = tf(self.pl_h0)
        self.pump_r_t = tf(self.pl_r)
        n_np = np.asarray(self.pl_n, dtype=np.float64).copy()
        n_np[np.abs(n_np - 1.0) < TINY] = 1.0            # hydcoeffs.c:739
        self.pump_n_t = tf(n_np)
        self.pump_lin_t = tb((n_np == 1.0)) & self.is_pump_t   # :781-785 线性支
        self.pump_pf_t = tb(self.pl_ptype == 1)          # POWER_FUNC（幂函数支）
        self.pump_chp_t = tb(self.is_chp_np)             # CONST_HP :743-763
        self.pump_nocurve_t = tb(self.pl_ptype == 3)     # NOCURVE  :709-714
        custom_np = self.pl_ptype == 2                   # CUSTOM   :716-733
        self.pump_custom_t = tb(custom_np)
        # dense 无状态机：关闭支 = 初始 CLOSED（自 self.closed）∪ ω==0
        # （hydcoeffs.c:696-701 的 `setting == 0.0`）。self.closed 本身不改动
        # （epanet 路径的 _PY_np 默认参数依赖它，且泵的 ω==0 在那边单独处理）。
        self.closed_dense = self.closed | (self.is_pump_t & tb(sp == 0.0))
        # ---- CUSTOM 曲线：padded [Nc, Kmax] + 每泵点数 ----
        cl = np.where(custom_np)[0]
        self.pump_custom_links_t = None
        if cl.size:
            xs = [self.pump_curve_x[int(self.pl_pumpidx[int(k)])] for k in cl]
            ys = [self.pump_curve_y[int(self.pl_pumpidx[int(k)])] for k in cl]
            for k, x in zip(cl, xs):
                if any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
                    raise NotImplementedError(
                        f"CUSTOM 泵 {self.net.link_id[int(k)]} 的曲线流量非严格递增"
                        f"（{x}）：dense 的 searchsorted 分段查找与 hydcoeffs.c:818-822"
                        " 的线性扫描仅在严格递增时等价，拒绝静默近似。")
            kmax = max(len(x) for x in xs)
            X = np.full((len(cl), kmax + 1), np.inf, dtype=np.float64)
            Yc = np.zeros((len(cl), kmax + 1), dtype=np.float64)
            for i, (x, y) in enumerate(zip(xs, ys)):
                X[i, :len(x)] = x
                Yc[i, :len(y)] = y
            # 尾部 +inf 填充：q 高于末点时 searchsorted 返回 npts，随后被
            # clamp 到 npts-1，正好等于 hydcoeffs.c:821 的末段外推。
            self.pump_custom_links_t = torch.as_tensor(cl, dtype=torch.int64,
                                                       device=dev)
            self.pump_curve_x_t = tf(X)
            self.pump_curve_y_t = tf(Yc)
            self.pump_curve_np_t = torch.as_tensor(
                np.asarray([len(x) for x in xs], dtype=np.int64), device=dev)
            self.pump_custom_sp_t = tf(sp[cl])

    # ------------------------------------------------------------------
    def _apply_exact_props(self, inp_path, st):
        """从 INP 原文重算 diam/len/r_hw/Km，逐运算复刻 EPANET 解析链。

        动机：wntr 链（mm→m→ft）与 EPANET 链（mm/(1000.0*MperFT)）存在 1ulp 差，
        经 κ~1e9 病态矩阵放大成 1e-8~1e-7 ft 的头差，破坏 1e-6 对拍门槛。
        换算与公式照抄：
        - dcf=1000.0*MperFT, hcf=MperFT（input1.c:443,450，SI）；US: dcf=12, hcf=1
          （input1.c:469,475）
        - 管道: Diam/=dcf; Len/=hcf; Km=0.02517*K/SQR(D)/SQR(D)（input1.c:614-618）
        - 阀:   Diam/=dcf; Km 同上（input1.c:654-655）
        - R = 4.727*L/pow(C,Hexp)/pow(D,4.871)（hydcoeffs.c:95，H-W）
        - TCV 有效 Km = 0.02517*setting/(SQR(D)*SQR(D))（hydcoeffs.c:931，
          注意分母先乘后除，与 convertunits 的连除分组不同）
        - float() 与 EPANET strtod 同为正确舍入解析，位级一致。
        """
        MperFT = 0.3048                      # types.h:79
        _is_si = str(self.net.meta["flow_units"]) in SI_FLOW_UNITS
        if _is_si:                           # SI（input1.c:428-453）
            dcf = 1000.0 * MperFT
            hcf = MperFT
        else:                                # US（input1.c:455-477）
            dcf = 12.0
            hcf = 1.0

        # 扫描 INP 原始 token：[PIPES]/[VALVES]（ID → 字段）、
        # [JUNCTIONS]/[DEMANDS]（重建位级需水基值）
        pipes, valves = {}, {}
        junc_rows, demand_rows = [], []
        emit_rows, opt_rows = [], []
        sec = None
        with open(inp_path, "r", encoding="latin-1") as f:
            for raw in f:
                s0 = raw.strip()
                if not s0:
                    continue
                if s0.startswith("["):
                    sec = s0[1:s0.find("]")].strip().upper()
                    continue
                body = s0.split(";", 1)[0].strip()
                if not body:
                    continue
                tok = body.split()
                if sec == "PIPES":
                    pipes[tok[0]] = tok
                elif sec == "VALVES":
                    valves[tok[0]] = tok
                elif sec == "JUNCTIONS":
                    junc_rows.append(tok)
                elif sec == "DEMANDS":
                    demand_rows.append(tok)
                elif sec == "EMITTERS":
                    emit_rows.append(tok)
                elif sec == "OPTIONS":
                    opt_rows.append(tok)

        # ---- 需水基值位级重建（input3.c:761-814 语义）----
        # [JUNCTIONS] 第 3 列为初始需水类别；[DEMANDS] 首行覆盖之、后续行追加；
        # 内部值 = strtod(token)/Ucf[DEMAND]（convertunits input1.c:559），
        # Ucf[DEMAND]=qcf 按流量单位查表（initunits input1.c:443-448/:469-474）。
        qcf = FLOW_UCF[str(self.net.meta["flow_units"])]
        dem_user = {}                       # node_id → [user_base,...]（类别序）
        for tok in junc_rows:
            if len(tok) >= 3:
                dem_user[tok[0]] = [float(tok[1 + 1])]
        replaced = set()
        for tok in demand_rows:
            if tok[0].upper() == "MULTIPLY":
                continue
            nid = tok[0]
            y = float(tok[1])
            if nid in dem_user and nid not in replaced:
                dem_user[nid] = [y]         # 首行覆盖（input3.c:797-810）
                replaced.add(nid)
            else:
                dem_user.setdefault(nid, []).append(y)   # 追加（:813）
        # 按 net.dem_node 的类别顺序逐一替换基值
        dem_base = np.asarray(self.net.dem_base_cfs, dtype=np.float64).copy()
        n_fixed = 0
        for i_node, bases in ((self.net.node_id.index(nid), v)
                              for nid, v in dem_user.items()
                              if nid in self.net.node_id):
            idx = np.where(np.asarray(self.net.dem_node) == i_node)[0]
            if len(idx) == len(bases):
                for pos, b in zip(idx, bases):
                    dem_base[pos] = b / qcf
                n_fixed += len(idx)
        self.net.dem_base_cfs = dem_base    # 注：就地修正共享 Net 的需水基值

        lt = np.asarray(self.net.link_type)
        diam = self.diam.cpu().numpy().copy()
        r_hw = self.r_hw.cpu().numpy().copy()
        km_pipe = self.km_pipe.cpu().numpy().copy()
        km_tcv = self.km_tcv.cpu().numpy().copy()
        for k, lid in enumerate(self.net.link_id):
            if lt[k] in (0, 1) and lid in pipes:
                tok = pipes[lid]
                # [PIPES]: ID N1 N2 Length Diameter Roughness (MinorLoss) (Status)
                L_ = float(tok[3]) / hcf                 # input1.c:615
                D_ = float(tok[4]) / dcf                 # input1.c:614
                C_ = float(tok[5])
                K_ = float(tok[6]) if len(tok) > 6 else 0.0
                if self.headloss_form == "D-W":
                    # D-W 粗糙度 mm(或毫英尺)→ft：Kc /= 1000.0*Ucf[ELEV]
                    # （convertunits input1.c:613）
                    C_ = C_ / (1000.0 * hcf)
                    self.kc_np[k] = C_
                diam[k] = D_
                km_pipe[k] = 0.02517 * K_ / (D_ * D_) / (D_ * D_)   # input1.c:618
                if self.headloss_form == "H-W":
                    r_hw[k] = 4.727 * L_ / _pow_crt(C_, self.hexp) \
                        / _pow_crt(D_, 4.871)            # hydcoeffs.c:95（pow 同 DLL）
                elif self.headloss_form == "D-W":
                    # hydcoeffs.c:98  R = L/2.0/32.2/d/SQR(PI*SQR(d)/4.0)
                    r_hw[k] = L_ / 2.0 / 32.2 / D_ \
                        / ((PI * (D_ * D_) / 4.0) * (PI * (D_ * D_) / 4.0))
                else:  # C-M（hydcoeffs.c:101-102）
                    r_hw[k] = ((4.0 * C_ / (1.49 * PI * (D_ * D_)))
                               * (4.0 * C_ / (1.49 * PI * (D_ * D_)))) \
                        * _pow_crt(D_ / 4.0, -1.333) * L_
            elif lt[k] >= 3 and lid in valves:
                tok = valves[lid]
                # [VALVES]: ID N1 N2 Diameter Type Setting (MinorLoss)
                D_ = float(tok[3]) / dcf                 # input1.c:654
                K_ = float(tok[6]) if len(tok) > 6 else 0.0
                setting = float(tok[5])
                diam[k] = D_
                km_ml = 0.02517 * K_ / (D_ * D_) / (D_ * D_)        # input1.c:655
                self.km_valve_ml[k] = km_ml       # link->Km（valvecoeff/prvstatus 用）
                if lt[k] == _TCV:
                    if st[k] == 2:                       # Active：hydcoeffs.c:931
                        km_tcv[k] = 0.02517 * setting / ((D_ * D_) * (D_ * D_))
                    else:                                # 固定 Open：沿用局损 Km
                        km_tcv[k] = km_ml
        t64 = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)
        self.diam = t64(diam)
        self.r_hw = t64(r_hw)
        self.km_pipe = t64(km_pipe)
        self.km_tcv = t64(km_tcv)

        # ---- [OPTIONS] SPECIFIC GRAVITY / PRESSURE 单位（Ucf[PRESSURE] 所需）----
        spgrav = 1.0                       # input1.c:121 默认 SpGrav=1.0
        press = "METERS" if self.net.meta["flow_units"] == "LPS" else "PSI"
        for tok in opt_rows:
            t0 = tok[0].upper()
            if t0.startswith("SPECIFIC") and len(tok) >= 3:
                spgrav = float(tok[2])     # SPECIFIC GRAVITY x（input3.c w_SPECGRAV）
            elif t0.startswith("PRESSURE") and len(tok) >= 2 \
                    and tok[1].upper() in ("PSI", "KPA", "METERS"):
                press = tok[1].upper()     # input3.c:1780-1782
        self._spgrav = spgrav              # INP 原文覆盖 meta 默认
        self._press = press

        # ---- emitter Ke 位级重建（pow 走 msvcrt，与 DLL 位级一致）----
        # 读入：input3.c:996-1026（非 junction 忽略 :1019；同节点后行覆盖 :1024）。
        # 换算：input1.c:567-573  ucf = pow(Ucf[FLOW],Qexp)/Ucf[PRESSURE]（:568），
        #       Ke = ucf/pow(C,Qexp)，仅 C>0（:572）。Qexp=1/γ（input3.c:2029）。
        # Ucf[PRESSURE]：SI METERS→MperFT*SpGrav，SI KPA→KPAperPSI*PSIperFT*SpGrav
        # （input1.c:451-452）；US→PSIperFT*SpGrav（input1.c:476）。
        if emit_rows:
            KPAperPSI, PSIperFT = 6.895, 0.4333    # types.h:81,80
            if self.net.meta["flow_units"] == "LPS":
                pcf = KPAperPSI * PSIperFT * spgrav if press == "KPA" \
                    else MperFT * spgrav
            else:
                pcf = PSIperFT * spgrav
            ucf_e = _pow_crt(qcf, self.qexp) / pcf         # input1.c:568（qcf=Ucf[FLOW]）
            node_ke = np.zeros(self.N, dtype=np.float64)
            nt_ = np.asarray(self.net.node_type)
            for tok in emit_rows:
                if tok[0] in self.net.node_id:
                    i = self.net.node_id.index(tok[0])
                    if nt_[i] != 0:
                        continue               # 非 junction 忽略（input3.c:1019）
                    c = float(tok[1])
                    node_ke[i] = ucf_e / _pow_crt(c, self.qexp) if c > 0.0 else 0.0
            self.net.node_ke = node_ke         # 注：就地修正共享 Net（与 dem_base 同策略）
            self.node_ke_default = t64(node_ke)

    # ------------------------------------------------------------------
    def _build_epanet_path(self, n1, n2, nt):
        """预构 EPANET 编号、MMD/符号分解结构、与 hydcoeffs.c 链路循环一致的
        装配索引序（np.add.at 按元素顺序累加 = 复刻 C 的逐链路 += 次序）。"""
        N, L, Nj = self.N, self.L, self.Nj
        # EPANET 1 基编号：junction 先（文件序 1..Nj），定水头节点续后（文件序）
        ep_of = np.zeros(N, dtype=np.int64)
        ep_of[self.junc_nodes] = np.arange(1, Nj + 1)
        ep_of[self.fixed_nodes] = np.arange(Nj + 1, N + 1)
        a1 = ep_of[n1]                      # 链路两端的 EPANET 节点号
        a2 = ep_of[n2]
        sm = EpanetSmatrix(N, Nj, list(zip(a1.tolist(), a2.tolist())))
        self.sm = sm
        Row = np.asarray(sm.Row, dtype=np.int64)          # ep 节点 → 行号
        self.Ndx_np = np.asarray(sm.Ndx[1:], dtype=np.int64)  # 链路 → Aij 槽位
        self.row_junc = Row[1:Nj + 1]        # ep junction i=1..Nj 的行号

        m1 = a1 <= Nj                        # n1 端为 junction
        m2 = a2 <= Nj
        # Aii 累加序（hydcoeffs.c:235,246）：链路序内 n1 先、n2 后
        aii_rows, aii_lnk = [], []
        # F 链路贡献序（hydcoeffs.c:236,240,247,251）：
        # type0:+Y(n1 junc)  type2:+P*Head[n1]（n1 定水头→加到 Row[n2]）
        # type1:-Y(n2 junc)  type2:+P*Head[n2]（n2 定水头→加到 Row[n1]）
        f_rows, f_lnk, f_type, f_hnode = [], [], [], []
        for k in range(L):
            if m1[k]:
                aii_rows.append(Row[a1[k]])
                aii_lnk.append(k)
                f_rows.append(Row[a1[k]]); f_lnk.append(k)
                f_type.append(0); f_hnode.append(-1)
            else:
                f_rows.append(Row[a2[k]]); f_lnk.append(k)
                f_type.append(2); f_hnode.append(int(n1[k]))
            if m2[k]:
                aii_rows.append(Row[a2[k]])
                aii_lnk.append(k)
                f_rows.append(Row[a2[k]]); f_lnk.append(k)
                f_type.append(1); f_hnode.append(-1)
            else:
                f_rows.append(Row[a1[k]]); f_lnk.append(k)
                f_type.append(2); f_hnode.append(int(n2[k]))
        self.aii_rows = np.asarray(aii_rows, dtype=np.int64)
        self.aii_lnk = np.asarray(aii_lnk, dtype=np.int64)
        self.f_rows = np.asarray(f_rows, dtype=np.int64)
        self.f_lnk = np.asarray(f_lnk, dtype=np.int64)
        f_type = np.asarray(f_type, dtype=np.int64)
        self.f_t0 = np.where(f_type == 0)[0]
        self.f_t1 = np.where(f_type == 1)[0]
        self.f_t2 = np.where(f_type == 2)[0]
        self.f_hnode = np.asarray(f_hnode, dtype=np.int64)
        # Xflow 累加序（hydcoeffs.c:225-226）：链路序内 -Q@n1、+Q@n2
        xf_nodes = np.empty(2 * L, dtype=np.int64)
        xf_nodes[0::2] = a1
        xf_nodes[1::2] = a2
        self.xf_nodes = xf_nodes
        xf_lnk = np.empty(2 * L, dtype=np.int64)
        xf_lnk[0::2] = np.arange(L)
        xf_lnk[1::2] = np.arange(L)
        self.xf_lnk = xf_lnk
        xf_sign = np.empty(2 * L, dtype=np.float64)
        xf_sign[0::2] = -1.0
        xf_sign[1::2] = 1.0
        self.xf_sign = xf_sign
        # numpy 版静态属性
        self.a1_np = np.asarray(a1, dtype=np.int64)   # 链路两端 EPANET 节点号
        self.a2_np = np.asarray(a2, dtype=np.int64)
        self.Row_np = np.asarray(sm.Row, dtype=np.int64)   # ep 节点 → 行号（全节点）
        self.elev_np = np.asarray(self.net.elev_ft, dtype=np.float64)
        self.n1_np = np.asarray(n1, dtype=np.int64)
        self.n2_np = np.asarray(n2, dtype=np.int64)
        self.r_np = self.r_hw.cpu().numpy()
        self.km_pipe_np = self.km_pipe.cpu().numpy()
        self.km_tcv_np = self.km_tcv.cpu().numpy()
        self.diam_np = self.diam.cpu().numpy()
        self.is_tcv_np = self.is_tcv.cpu().numpy()
        self.closed_np = self.closed.cpu().numpy()
        self.el_junc_np = self.el_junc.cpu().numpy()

    # ------------------------------------------------------------------
    def _build_b2_data(self, lt, nt):
        """B2：泵曲线系数（link 索引化）、水池（node 索引化）、控制、内部状态编码。"""
        net = self.net
        N, L = self.N, self.L
        # 内部 StatusType 编码（types.h:193-205）：XHEAD=0 TEMPCLOSED=1 CLOSED=2
        # OPEN=3 ACTIVE=4 XFLOW=5 XFCV=6 XPRESSURE=7。
        # Net.init_status（0=Closed 1=Open 2=Active）→ 内部编码。
        self.ST_XHEAD, self.ST_TEMPCLOSED = 0, 1
        self.ST_CLOSED, self.ST_OPEN, self.ST_ACTIVE = 2, 3, 4
        self.ST_XFLOW, self.ST_XFCV, self.ST_XPRESSURE = 5, 6, 7
        map_init = {0: self.ST_CLOSED, 1: self.ST_OPEN, 2: self.ST_ACTIVE}
        self.init_status_int = np.asarray(
            [map_init[int(s)] for s in np.asarray(net.init_status)], dtype=np.int8)
        # ---- 阀静态索引 ----
        self.lt_np = np.asarray(lt, dtype=np.int64)
        self.is_valve_np = self.lt_np >= _PRV          # PRV..GPV
        self.valve_links = np.where(self.is_valve_np)[0]   # 文件序 = EPANET 阀序
        # PRV/PSV/FCV：headlosscoeffs 里 setting!=MISSING 时 P=0（hydcoeffs.c:157-158）
        self.is_pcv_np = np.isin(self.lt_np, (_PRV, _PSV, _FCV))
        self.km_valve_ml_np = self.km_valve_ml         # link->Km（可能已被 exact 覆盖）
        # ---- Ucf 换算因子（input1.c:443-453 SI / :469-477 US）----
        MperFT_, PSIperFT_, KPAperPSI_ = 0.3048, 0.4333, 6.895   # types.h:79,80,81
        spgrav, press = self._spgrav, self._press
        if str(net.meta["flow_units"]) in SI_FLOW_UNITS:   # input1.c:255-265
            self.ucf_flow = FLOW_UCF[str(net.meta["flow_units"])]  # input1.c:444-448
            self.ucf_head = MperFT_                    # input1.c:450 hcf = MperFT
            if not press:
                press = "METERS"
            # input1.c:451-452：METERS→MperFT*SpGrav，否则 KPA 公式
            self.ucf_pressure = (MperFT_ * spgrav if press != "KPA"
                                 else KPAperPSI_ * PSIperFT_ * spgrav)
        else:
            self.ucf_flow = FLOW_UCF[str(net.meta["flow_units"])]  # input1.c:470-474
            self.ucf_head = 1.0                        # input1.c:475 hcf = 1.0
            self.ucf_pressure = PSIperFT_ * spgrav     # input1.c:476
        # LinkSetting 初值 = link->Kc（inithyd hydraul.c:131）：泵为转速；
        # 阀为内部设定（convertunits input1.c:656-668：PRV/PSV/PBV /=Ucf[PRESSURE]、
        # FCV /=Ucf[FLOW]、TCV 无量纲原值）；[STATUS] 固定 OPEN/CLOSED 的阀
        # Kc=MISSING（changestatus input3.c:2168）；管道无意义（置 0）。
        self.init_setting = np.asarray(net.valve_setting_user, dtype=np.float64).copy()
        for k in self.valve_links:
            k = int(k)
            if self.init_status_int[k] != self.ST_ACTIVE:
                self.init_setting[k] = MISSING         # input3.c:2168
            elif self.lt_np[k] in (_PRV, _PSV, _PBV):
                self.init_setting[k] = self.init_setting[k] / self.ucf_pressure  # input1.c:664
            elif self.lt_np[k] == _FCV:
                self.init_setting[k] = self.init_setting[k] / self.ucf_flow      # input1.c:661
        # Htol/Qtol 默认值（input1.c:27-28）；INP 可设 HTOL/QTOL（input3.c:2020-2021）
        self.htol = float(net.meta.get("htol", 0.0005))   # input1.c:27 HTOL 0.0005
        self.qtol = float(net.meta.get("qtol", 0.0001))   # input1.c:28 QTOL 0.0001
        self.checkfreq = int(net.meta.get("checkfreq", 2))    # input1.c:36
        self.maxcheck = int(net.meta.get("maxcheck", 10))     # input1.c:37
        self.extra_iter = int(net.meta.get("extra_iter", -1))  # input1.c:115

        # ---- 泵：link 索引化曲线系数 ----
        self.is_pump_np = lt == _PUMP
        pl = np.asarray(net.pump_link, dtype=np.int64)
        self.n_pumps = len(pl)
        self.pump_links = pl                      # 泵管段索引（文件序 = EPANET 泵序）
        self.pl_ptype = np.full(L, -1, dtype=np.int64)
        self.pl_h0 = np.zeros(L, dtype=np.float64)
        self.pl_r = np.zeros(L, dtype=np.float64)
        self.pl_n = np.zeros(L, dtype=np.float64)
        self.pl_q0 = np.zeros(L, dtype=np.float64)
        self.pl_hmax = np.zeros(L, dtype=np.float64)
        if self.n_pumps:
            self.pl_ptype[pl] = np.asarray(net.pump_ptype, dtype=np.int64)
            self.pl_h0[pl] = np.asarray(net.pump_h0)
            self.pl_r[pl] = np.asarray(net.pump_r)
            self.pl_n[pl] = np.asarray(net.pump_n)
            self.pl_q0[pl] = np.asarray(net.pump_q0)
            self.pl_hmax[pl] = np.asarray(net.pump_hmax)
        self.is_chp_np = self.pl_ptype == 0       # CONST_HP 泵管段掩码
        # ---- CUSTOM 多段泵曲线（hydcoeffs.c:716-733 + curvecoeff :794-831）----
        # 曲线点存用户原值（untransformed，:810 注释），curvecoeff 每次调用按
        # :811/:829-830 做 Ucf 换算 - 运算次序照抄 C 以保位级。
        self.pl_pumpidx = np.full(L, -1, dtype=np.int64)   # 泵管段 → 泵序号
        self.pump_curve_x = []                    # 逐泵曲线流量（用户单位，list[float]）
        self.pump_curve_y = []                    # 逐泵曲线扬程（用户单位）
        if self.n_pumps:
            self.pl_pumpidx[pl] = np.arange(self.n_pumps)
            cq = np.asarray(net.pump_curve_q, dtype=np.float64)
            ch = np.asarray(net.pump_curve_h, dtype=np.float64)
            cptr = np.asarray(net.pump_curve_ptr, dtype=np.int64)
            has_ptr = cptr.size == self.n_pumps + 1
            for j in range(self.n_pumps):
                if has_ptr:
                    a, b = int(cptr[j]), int(cptr[j + 1])
                    self.pump_curve_x.append([float(v) for v in cq[a:b]])
                    self.pump_curve_y.append([float(v) for v in ch[a:b]])
                else:
                    self.pump_curve_x.append([])
                    self.pump_curve_y.append([])
                if int(net.pump_ptype[j]) == 2 and len(self.pump_curve_x[j]) < 2:
                    raise RuntimeError(
                        "CUSTOM 泵缺曲线点（Net 为旧版 npz？请用当前 parse_inp "
                        "重新生成 <stem>_net.npz/<stem>_meta.json）")

        # ---- 水池：node 索引化 ----
        tn = np.asarray(net.tank_node, dtype=np.int64)
        self.n_tanks = len(tn)
        self.tank_nodes = tn
        self.is_tank_node = np.zeros(N, dtype=bool)
        self.tn_hmax = np.zeros(N, dtype=np.float64)
        self.tn_hmin = np.zeros(N, dtype=np.float64)
        self.tn_h0 = np.zeros(N, dtype=np.float64)
        self.tn_overflow = np.zeros(N, dtype=bool)
        if self.n_tanks:
            self.is_tank_node[tn] = True
            self.tn_hmax[tn] = np.asarray(net.tank_hmax)
            self.tn_hmin[tn] = np.asarray(net.tank_hmin)
            self.tn_h0[tn] = np.asarray(net.tank_h0)
            self.tn_overflow[tn] = np.asarray(net.tank_overflow) != 0
        # 定水头节点集合（tankstatus 的 n>Njuncs 判据）
        self.is_fixed_node = nt != 0
        # 触及定水头节点的链路（newlinkflows :459-464 的净流入累加；文件序）
        self._fixed_links = np.where(self.is_fixed_node[self.n1_np]
                                     | self.is_fixed_node[self.n2_np])[0]

    # ------------------------------------------------------------------
    @staticmethod
    def _pow_ucrt(base, expo):
        """逐元素 pow，用 msvcrt.dll 的 pow（见模块级 _pow_crt）。

        实测：wntr 的 epanet22.dll 为 MinGW 系构建，其静态 pow 与 msvcrt.dll
        位级一致、与 UCRT/numpy 的 pow 有 ~0.2% 的 1ulp 差；用 UCRT pow 时
        迭代 2 起出现 1-2ulp 头差，经 TCV 的 P=1e6 放大为 1e-8 cfs 流差并逐迭代
        增长（k=5 时 ~3e-6 ft）；换 msvcrt pow 后 k=1..5 与 DLL 逐位一致。"""
        return np.fromiter((_pow_crt(float(x), expo) for x in base),
                           dtype=np.float64, count=len(base))

    def _dw_PY_np(self, q):
        """DWpipecoeff（hydcoeffs.c:578-620）+ frictionFactor（:623-670）的逐管道
        标量复刻（pow/log 走 msvcrt，与 DLL 位级一致；无 RQtol 钳位 - D-W 层流支
        天然线性）。返回 (P, Y)，非管道位为 0（调用方的阀/泵/关闭支随后覆盖）。"""
        L = self.L
        P = np.zeros(L, dtype=np.float64)
        Y = np.zeros(L, dtype=np.float64)
        lt = self.lt_np
        for k in range(L):
            if lt[k] > 1:                               # 仅 CVPIPE/PIPE
                continue
            Qk = float(q[k])
            qa = abs(Qk)                                # :591 q = ABS(LinkFlow)
            r = float(self.r_np[k])                     # :592 R
            ml = float(self.km_pipe_np[k])              # :593 Km
            e = float(self.kc_np[k]) / float(self.diam_np[k])  # :594 相对粗糙度
            s = self.viscos * float(self.diam_np[k])    # :595 Viscos*Diam
            if qa <= A2 * s:                            # :600 层流 Re<=2000
                r_ = 16.0 * PI * s * r                  # :602（Hagen-Poiseuille）
                hloss = Qk * (r_ + ml * qa)             # :603
                hgrad = r_ + 2.0 * ml * qa              # :604
            else:                                       # :607-615 紊流
                w = qa / s                              # :640 w = Re*Pi/4
                if w >= A1:                             # :644 Swamee-Jain 显式
                    y1 = A8 / _pow_crt(w, 0.9)          # :646
                    y2 = e / 3.7 + y1                   # :647
                    y3 = A9 * _log_crt(y2)              # :648
                    f = 1.0 / (y3 * y3)                 # :649
                    dfdq = 1.8 * f * y1 * A9 / y2 / y3 / qa  # :650
                else:                                   # :655-668 Dunlop 三次过渡
                    y2 = e / 3.7 + AB                   # :657
                    y3 = A9 * _log_crt(y2)              # :658
                    fa = 1.0 / (y3 * y3)                # :659
                    fb = (2.0 + AC / (y2 * y3)) * fa    # :660
                    r2 = w / A2                         # :661
                    x1 = 7.0 * fa - fb                  # :662
                    x2 = 0.128 - 17.0 * fa + 2.5 * fb   # :663
                    x3 = -0.128 + 13.0 * fa - (fb + fb)  # :664
                    x4 = 0.032 - 3.0 * fa + 0.5 * fb    # :665
                    f = x1 + r2 * (x2 + r2 * (x3 + r2 * x4))          # :666
                    dfdq = (x2 + r2 * (2.0 * x3 + r2 * 3.0 * x4)) / s / A2  # :667
                r1 = f * r + ml                         # :612
                hloss = r1 * qa * Qk                    # :613
                hgrad = (2.0 * r1 * qa) + (dfdq * r * qa * qa)   # :614
            P[k] = 1.0 / hgrad                          # :618
            Y[k] = hloss / hgrad                        # :619
        return P, Y

    def _curvecoeff(self, j, q):
        """curvecoeff（hydcoeffs.c:794-831）：泵 j 的多段曲线在流量 q（内部 cfs，
        调用方已除转速）处的局部线性模型，返回 (h0, r) 内部单位截距/斜率。
        曲线点存用户原值（untransformed，:810 注释）；段选择：首个 x[k2]>=q 的
        点为右端；q 低于首点 → 用首段外推（:820），q 高于末点 → 用末段外推
        （:821）。运算次序逐字照抄 C。"""
        q = q * self.ucf_flow                        # :811 q *= Ucf[FLOW]
        x = self.pump_curve_x[j]                     # :813 x = flow
        y = self.pump_curve_y[j]                     # :814 y = head
        npts = len(x)                                # :815
        k2 = 0                                       # :818
        while k2 < npts and x[k2] < q:               # :819
            k2 += 1
        if k2 == 0:                                  # :820
            k2 += 1
        elif k2 == npts:                             # :821
            k2 -= 1
        k1 = k2 - 1                                  # :822
        r = (y[k2] - y[k1]) / (x[k2] - x[k1])        # :825
        h0 = y[k1] - r * x[k1]                       # :826
        h0 = h0 / self.ucf_head                      # :829
        r = r * self.ucf_flow / self.ucf_head        # :830
        return h0, r

    def _PY_np(self, q, closed=None, setting=None, status=None):
        """numpy 版 P/Y（公式与 torch 路径同源，标注见 _pipe_PY/_tcv_PY）。

        B2 起支持动态状态：closed = (内部状态 <= CLOSED)；泵按 pumpcoeff
        （hydcoeffs.c:673-791）覆写。缺省参数（旧调用）行为逐位不变。"""
        if closed is None:
            closed = self.closed_np
        if setting is None:
            setting = self.init_setting
        if status is None:
            status = self.init_status_int
        qa = np.abs(q)
        if self.headloss_form == "D-W":
            # D-W：pipecoeff:539-543 转 DWpipecoeff（关闭支由下方统一覆盖）
            P, Y = self._dw_PY_np(q)
        else:
            # 管道（hydcoeffs.c:545-574，H-W/C-M 通用支）；pow 走 CRT（见 _pow_ucrt）
            hgrad = self.hexp * self.r_np * self._pow_ucrt(qa, self.hexp - 1.0)
            lin = hgrad < self.rqtol
            hgrad = np.where(lin, self.rqtol, hgrad)
            hloss = np.where(lin, hgrad * qa, hgrad * qa / self.hexp)
            mlp = self.km_pipe_np > 0.0
            hloss = np.where(mlp, hloss + self.km_pipe_np * qa * qa, hloss)
            hgrad = np.where(mlp, hgrad + 2.0 * self.km_pipe_np * qa, hgrad)
            hloss = hloss * np.where(q < 0, -1.0, 1.0)
            P = 1.0 / hgrad
            Y = hloss / hgrad
        # 阀（headlosscoeffs 分派 hydcoeffs.c:134-159：TCV→tcvcoeff:912-939，
        # PRV/PSV/FCV setting==MISSING→valvecoeff:1100-1151，否则 P=0）
        # TCV Active：Km = 0.02517*setting/(SQR(Diam)*SQR(Diam))（hydcoeffs.c:931）；
        # 其余阀/固定 TCV：Km = link->Km（阀行局损，input1.c:655）
        km = self.km_valve_ml_np
        tcv_act = self.is_tcv_np & (setting != MISSING)
        if tcv_act.any():
            d2 = self.diam_np * self.diam_np
            km = np.where(tcv_act,
                          np.divide(0.02517 * setting, d2 * d2,
                                    out=np.zeros_like(km), where=d2 > 0.0),
                          km)
        hgrad_v = 2.0 * km * qa
        lin_v = hgrad_v < self.rqtol
        hgrad_v = np.where(lin_v, self.rqtol, hgrad_v)
        hloss_v = np.where(lin_v, q * hgrad_v, q * hgrad_v / 2.0)
        P_v = np.where(km > 0.0, 1.0 / hgrad_v, 1.0 / CSMALL)
        Y_v = np.where(km > 0.0, hloss_v / hgrad_v, q)
        P = np.where(self.is_valve_np, P_v, P)
        Y = np.where(self.is_valve_np, Y_v, Y)
        # 关闭支（pipecoeff:531-536 / valvecoeff:1118-1123）
        P = np.where(closed, 1.0 / CBIG, P)
        Y = np.where(closed, q, Y)
        # PRV/PSV/FCV 且 setting!=MISSING：P=0（hydcoeffs.c:157-158），
        # 系数改由 valvecoeffs（prvcoeff/psvcoeff）在 nodecoeffs 之后接管
        pcv = self.is_pcv_np & (setting != MISSING)
        if pcv.any():
            P = np.where(pcv, 0.0, P)
        # ---- 泵：pumpcoeff（hydcoeffs.c:673-791）逐泵覆写 ----
        if self.n_pumps:
            for k in self.pump_links:
                k = int(k)
                sp = float(setting[k])
                # 关闭或 ω=0：高阻管（hydcoeffs.c:696-701）
                if int(status[k]) <= self.ST_CLOSED or sp == 0.0:
                    P[k] = 1.0 / CBIG
                    Y[k] = q[k]
                    continue
                # 无曲线：视作开阀（hydcoeffs.c:709-714）
                if self.pl_ptype[k] == 3:
                    P[k] = 1.0 / CSMALL
                    Y[k] = q[k]
                    continue
                qk = float(q[k])
                qa_k = abs(qk)                       # hydcoeffs.c:704 q = ABS(LinkFlow)
                if self.pl_ptype[k] == 2:            # CUSTOM（hydcoeffs.c:718-733）
                    # 转速折算流量 q/setting 所在线段的局部线性模型（:722）
                    h0c, rc = self._curvecoeff(int(self.pl_pumpidx[k]), qa_k / sp)
                    H0 = -h0c                        # :726 泵曲线扬程 → 水损取负
                    R = -rc                          # :727
                    # pump->N = 1.0（:728）
                    hgrad_p = R * sp                 # :731 hgrad = R*setting（相似律）
                    hloss_p = H0 * (sp * sp) + hgrad_p * qk   # :732
                    P[k] = 1.0 / hgrad_p             # :789
                    Y[k] = hloss_p / hgrad_p         # :790
                    continue
                # 转速修正（hydcoeffs.c:737-740）
                h0 = (sp * sp) * self.pl_h0[k]       # :737 h0 = SQR(setting)*H0
                n = float(self.pl_n[k])
                if abs(n - 1.0) < TINY:
                    n = 1.0                          # :739
                r = self.pl_r[k] * _pow_crt(sp, 2.0 - n)   # :740 r = R*pow(w, 2-N)
                if self.pl_ptype[k] == 0:            # CONST_HP（hydcoeffs.c:743-763）
                    hgrad_p = -r / qa_k / qa_k       # :746
                    if hgrad_p > CBIG:               # :748-752
                        hgrad_p = CBIG
                        hloss_p = -hgrad_p * qk
                    elif hgrad_p < self.rqtol:       # :753-757
                        hgrad_p = self.rqtol
                        hloss_p = -hgrad_p * qk
                    else:                            # :759-762
                        hloss_p = r / qk
                elif n != 1.0:                       # 非线性曲线（hydcoeffs.c:767-779）
                    hgrad_p = n * r * _pow_crt(qa_k, n - 1.0)   # :770
                    if hgrad_p < self.rqtol:         # :772-776
                        hgrad_p = self.rqtol
                        hloss_p = h0 + hgrad_p * qk
                    else:                            # :778
                        hloss_p = h0 + hgrad_p * qk / n
                else:                                # 线性曲线（hydcoeffs.c:781-785）
                    hgrad_p = r
                    hloss_p = h0 + hgrad_p * qk
                P[k] = 1.0 / hgrad_p                 # :789
                Y[k] = hloss_p / hgrad_p             # :790
        return P, Y

    def _emitter_hloss_np(self, e, ke):
        """numpy 版 emitterheadloss（hydcoeffs.c:378-409）。"""
        ke_adj = np.maximum(CSMALL, ke)
        hgrad = self.qexp * ke_adj * self._pow_ucrt(np.abs(e), self.qexp - 1.0)
        lin = hgrad < self.rqtol
        hgrad = np.where(lin, self.rqtol, hgrad)
        hloss = np.where(lin, hgrad * e, hgrad * e / self.qexp)
        return hloss, hgrad

    # ------------------------------------------------------------------
    # 梯队3：PRV/PSV 装配（hydcoeffs.c prvcoeff/psvcoeff/valvecoeff）
    def _valvecoeff_scalar(self, k, flow, s):
        """valvecoeff（hydcoeffs.c:1100-1151）标量版。Km = link->Km（阀行局损）。"""
        if s <= self.ST_CLOSED:                      # :1118
            return 1.0 / CBIG, flow                  # :1120-1121
        km = float(self.km_valve_ml_np[k])
        if km > 0.0:                                 # :1126
            q_ = abs(flow)                           # :1128
            hgrad = 2.0 * km * q_                    # :1129
            if hgrad < self.rqtol:                   # :1132
                hgrad = self.rqtol                   # :1134
                hloss = flow * hgrad                 # :1135
            else:
                hloss = flow * hgrad / 2.0           # :1137
            return 1.0 / hgrad, hloss / hgrad        # :1140-1141
        return 1.0 / CSMALL, flow                    # :1148-1149

    def _valvecoeffs(self, P, Y, Aii, Aij, F, Xf, q, S, K):
        """valvecoeffs（hydcoeffs.c:282-330）：逐阀（文件序 = EPANET 阀序）调
        prvcoeff（:942-992）/ psvcoeff（:995-1044）。setting==MISSING 跳过（:308）。
        必须在 nodecoeffs 之后调用（Xflow 已扣需水，hydcoeffs.c:164-195 次序）。"""
        lt = self.lt_np
        for k in self.valve_links:
            k = int(k)
            if K[k] == MISSING:                      # :308
                continue
            a1 = int(self.a1_np[k])
            a2 = int(self.a2_np[k])
            i_r = int(self.Row_np[a1])               # prvcoeff:960-961
            j_r = int(self.Row_np[a2])
            if lt[k] == _PRV:
                # hset = Node[n2].El + LinkSetting（prvcoeff:962-963）
                hset = self.elev_np[int(self.n2_np[k])] + K[k]
                if S[k] == self.ST_ACTIVE:           # :965
                    P[k] = 0.0                       # :972
                    Y[k] = q[k] + Xf[a2]             # :973 强制流量平衡
                    F[j_r] += hset * CBIG            # :974 罚函数强制 H[n2]=hset
                    Aii[j_r] += CBIG                 # :975
                    if Xf[a2] < 0.0:                 # :976
                        F[i_r] += Xf[a2]             # :978
                    continue
                pk, yk = self._valvecoeff_scalar(k, float(q[k]), int(S[k]))  # :986
                P[k], Y[k] = pk, yk
                Aij[int(self.Ndx_np[k])] -= pk       # :987
                Aii[i_r] += pk                       # :988
                Aii[j_r] += pk                       # :989
                F[i_r] += (yk - q[k])                # :990
                F[j_r] -= (yk - q[k])                # :991
            elif lt[k] == _PSV:
                # hset = Node[n1].El + LinkSetting（psvcoeff:1015-1016）
                hset = self.elev_np[int(self.n1_np[k])] + K[k]
                if S[k] == self.ST_ACTIVE:           # :1018
                    P[k] = 0.0                       # :1024
                    Y[k] = q[k] - Xf[a1]             # :1025
                    F[i_r] += hset * CBIG            # :1026
                    Aii[i_r] += CBIG                 # :1027
                    if Xf[a1] > 0.0:                 # :1028
                        F[j_r] += Xf[a1]             # :1030
                    continue
                pk, yk = self._valvecoeff_scalar(k, float(q[k]), int(S[k]))  # :1038
                P[k], Y[k] = pk, yk
                Aij[int(self.Ndx_np[k])] -= pk       # :1039
                Aii[i_r] += pk                       # :1040
                Aii[j_r] += pk                       # :1041
                F[i_r] += (yk - q[k])                # :1042
                F[j_r] -= (yk - q[k])                # :1043
            elif lt[k] == _FCV:
                # fcvcoeff（hydcoeffs.c:1047-1097）
                qs = float(K[k])                     # :1065 q = LinkSetting
                if S[k] == self.ST_ACTIVE:           # :1073
                    # 切断管网：设定流量作上游外需水/下游外供给（:1069-1071）
                    Xf[a1] -= qs                     # :1075
                    Xf[a2] += qs                     # :1076
                    Y[k] = q[k] - qs                 # :1077
                    F[i_r] -= qs                     # :1078
                    F[j_r] += qs                     # :1079
                    P[k] = 1.0 / CBIG                # :1080
                    Aij[int(self.Ndx_np[k])] -= P[k]  # :1081
                    Aii[i_r] += P[k]                 # :1082
                    Aii[j_r] += P[k]                 # :1083
                else:                                # :1086-1096 视作开管
                    pk, yk = self._valvecoeff_scalar(k, float(q[k]), int(S[k]))  # :1090
                    P[k], Y[k] = pk, yk
                    Aij[int(self.Ndx_np[k])] -= pk   # :1091
                    Aii[i_r] += pk                   # :1092
                    Aii[j_r] += pk                   # :1093
                    F[i_r] += (yk - q[k])            # :1094
                    F[j_r] -= (yk - q[k])            # :1095

    def _badvalve(self, n_ep, S):
        """badvalve（hydsolver.c:215-265）：病态行节点若属 ACTIVE 控制阀，
        置 XPRESSURE（FCV 为 XFCV）并返回 True 以重试。n_ep 为 EPANET 节点号。"""
        lt = self.lt_np
        for k in self.valve_links:
            k = int(k)
            if n_ep == int(self.a1_np[k]) or n_ep == int(self.a2_np[k]):  # :243
                if lt[k] in (_PRV, _PSV, _FCV):      # :246
                    if S[k] == self.ST_ACTIVE:       # :248
                        S[k] = (self.ST_XFCV if lt[k] == _FCV
                                else self.ST_XPRESSURE)   # :256-257
                        return True                  # :258
                return False                         # :261
        return False                                 # :264

    # ------------------------------------------------------------------
    # 梯队3：valvestatus（hydstatus.c:33-98 + prvstatus/psvstatus）
    def _prvstatus(self, k, s, hset, h1, h2, qk):
        """prvstatus（hydstatus.c:242-299）。"""
        htol = self.htol                             # :263
        hml = self.km_valve_ml_np[k] * (qk * qk)     # :267 Km*SQR(LinkFlow)
        status = s                                   # :270
        if s == self.ST_ACTIVE:                      # :273
            if qk < -self.qtol:                      # :274
                status = self.ST_CLOSED
            elif h1 - hml < hset - htol:             # :275
                status = self.ST_OPEN
            else:
                status = self.ST_ACTIVE              # :276
        elif s == self.ST_OPEN:                      # :279
            if qk < -self.qtol:                      # :280
                status = self.ST_CLOSED
            elif h2 >= hset + htol:                  # :281
                status = self.ST_ACTIVE
            else:
                status = self.ST_OPEN                # :282
        elif s == self.ST_CLOSED:                    # :285
            if h1 >= hset + htol and h2 < hset - htol:      # :286
                status = self.ST_ACTIVE
            elif h1 < hset - htol and h1 > h2 + htol:       # :287
                status = self.ST_OPEN
            else:
                status = self.ST_CLOSED              # :288
        elif s == self.ST_XPRESSURE:                 # :291
            if qk < -self.qtol:                      # :292
                status = self.ST_CLOSED
        return status                                # :298

    def _psvstatus(self, k, s, hset, h1, h2, qk):
        """psvstatus（hydstatus.c:302-359）。"""
        htol = self.htol                             # :323
        hml = self.km_valve_ml_np[k] * (qk * qk)     # :327
        status = s                                   # :330
        if s == self.ST_ACTIVE:                      # :333
            if qk < -self.qtol:                      # :334
                status = self.ST_CLOSED
            elif h2 + hml > hset + htol:             # :335
                status = self.ST_OPEN
            else:
                status = self.ST_ACTIVE              # :336
        elif s == self.ST_OPEN:                      # :339
            if qk < -self.qtol:                      # :340
                status = self.ST_CLOSED
            elif h1 < hset - htol:                   # :341
                status = self.ST_ACTIVE
            else:
                status = self.ST_OPEN                # :342
        elif s == self.ST_CLOSED:                    # :345
            if h2 > hset + htol and h1 > h2 + htol:         # :346
                status = self.ST_OPEN
            elif h1 >= hset + htol and h1 > h2 + htol:      # :347
                status = self.ST_ACTIVE
            else:
                status = self.ST_CLOSED              # :348
        elif s == self.ST_XPRESSURE:                 # :351
            if qk < -self.qtol:                      # :352
                status = self.ST_CLOSED
        return status                                # :358

    def _valvestatus(self, S, K, H, q):
        """valvestatus（hydstatus.c:33-98）：PRV/PSV（setting 未固定）状态复核。"""
        change = False
        lt = self.lt_np
        for k in self.valve_links:
            k = int(k)
            if K[k] == MISSING:                      # :63 固定 OPEN/CLOSED 跳过
                continue
            n1 = int(self.n1_np[k])
            n2 = int(self.n2_np[k])
            status = int(S[k])                       # :68
            if lt[k] == _PRV:                        # :73-77
                hset = self.elev_np[n2] + K[k]       # :74
                S[k] = self._prvstatus(k, status, hset, H[n1], H[n2], float(q[k]))
            elif lt[k] == _PSV:                      # :78-82
                hset = self.elev_np[n1] + K[k]       # :79
                S[k] = self._psvstatus(k, status, hset, H[n1], H[n2], float(q[k]))
            else:
                continue                             # :83-84
            if status != S[k]:                       # :88
                change = True                        # :94
        return change

    # ------------------------------------------------------------------
    # B2：hydstatus.c 三个状态机（逐字复刻；标量逐链路）
    def _cvstatus(self, s, dh, q):
        """cvstatus（hydstatus.c:177-202）。s=当前内部状态。"""
        if abs(dh) > self.htol:                      # :191
            if dh < -self.htol:                      # :193
                return self.ST_CLOSED
            elif q < -self.qtol:                     # :194
                return self.ST_CLOSED
            else:
                return self.ST_OPEN                  # :195
        else:
            if q < -self.qtol:                       # :199
                return self.ST_CLOSED
            return s                                 # :200

    def _pumpstatus(self, k, dh, setting):
        """pumpstatus（hydstatus.c:205-239）。dh = 泵扬程（head gain）。"""
        if self.pl_ptype[k] == 0:                    # CONST_HP：hmax=BIG（:223-227）
            hmax = BIG
        else:                                        # ω² 修正截止扬程（:231）
            hmax = (setting * setting) * self.pl_hmax[k]
        if dh > hmax + self.htol:                    # :235
            return self.ST_XHEAD
        return self.ST_OPEN                          # :238

    def _fcvstatus(self, k, s, h1, h2, qk, kc):
        """fcvstatus（hydstatus.c:362-398）。s=当前状态，kc=LinkSetting。
        反流 → XFCV；XFCV 且流量回升到设定以上 → ACTIVE。"""
        status = s                                   # :384
        if h1 - h2 < -self.htol:                     # :385
            status = self.ST_XFCV                    # :387
        elif qk < -self.qtol:                        # :389
            status = self.ST_XFCV                    # :391
        elif s == self.ST_XFCV and qk >= kc:         # :393
            status = self.ST_ACTIVE                  # :395
        return status                                # :397

    def _tankstatus(self, S, H, q, k, n1, n2):
        """tankstatus（hydstatus.c:401-476）：满/空池临时关闭进出流链路。"""
        if S[k] <= self.ST_CLOSED:                   # :421
            return
        qk = float(q[k])
        # 让 n1 指向 tank（:423-434）；n1 为定水头（含 reservoir）则不交换
        if not self.is_fixed_node[n1]:
            if not self.is_fixed_node[n2]:
                return
            n1, n2 = n2, n1
            qk = -qk
        if not self.is_tank_node[n1]:                # reservoir 忽略（:436-438）
            return
        h = H[n1] - H[n2]                            # :441
        is_pump = self.is_pump_np[k]
        # 满池阻止流入（:444-458）
        if H[n1] >= self.tn_hmax[n1] - self.htol and not self.tn_overflow[n1]:
            if is_pump:
                if self.n2_np[k] == n1:              # :449 泵出流入池
                    S[k] = self.ST_TEMPCLOSED
            elif self._cvstatus(self.ST_OPEN, h, qk) == self.ST_CLOSED:  # :454
                S[k] = self.ST_TEMPCLOSED
        # 空池阻止流出（:461-475）
        if H[n1] <= self.tn_hmin[n1] + self.htol:
            if is_pump:
                if self.n1_np[k] == n1:              # :466 泵从池抽水
                    S[k] = self.ST_TEMPCLOSED
            elif self._cvstatus(self.ST_CLOSED, h, qk) == self.ST_OPEN:  # :471
                S[k] = self.ST_TEMPCLOSED

    def _linkstatus(self, S, K, H, q):
        """linkstatus（hydstatus.c:101-174）：泵/CV/通池管的状态复核。"""
        change = False
        lt = np.asarray(self.net.link_type)
        for k in range(self.L):
            n1 = int(self.n1_np[k])
            n2 = int(self.n2_np[k])
            dh = H[n1] - H[n2]                       # :129
            status = int(S[k])
            # 重开临时关闭链路（:131-137）
            if status == self.ST_XHEAD or status == self.ST_TEMPCLOSED:
                S[k] = self.ST_OPEN
            if lt[k] == _CVPIPE:                     # :139-143
                S[k] = self._cvstatus(int(S[k]), dh, float(q[k]))
            if lt[k] == _PUMP and S[k] >= self.ST_OPEN and K[k] > 0.0:  # :144-148
                S[k] = self._pumpstatus(k, -dh, float(K[k]))
            # 非固定 FCV（hydstatus.c:151-155）：注意传入的是重开前的 status
            if lt[k] == _FCV and K[k] != MISSING:
                S[k] = self._fcvstatus(k, status, H[n1], H[n2],
                                       float(q[k]), float(K[k]))
            if self.is_fixed_node[n1] or self.is_fixed_node[n2]:  # :158-161
                self._tankstatus(S, H, q, k, n1, n2)
            if status != S[k]:                       # :164
                change = True
        return change

    def _pswitch(self, S, K, H):
        """pswitch（hydsolver.c:268-355）：junction 压力控制的收敛后复核。"""
        net = self.net
        anychange = False
        nt = np.asarray(net.node_type)
        lt = np.asarray(net.link_type)
        for i in range(len(net.ctl_link)):
            k = int(net.ctl_link[i])
            n = int(net.ctl_node[i])
            # 仅 junction 节点控制（:299-300）
            if n < 0 or nt[n] != 0:
                continue
            reset = 0
            if net.ctl_type[i] == 0 and H[n] <= net.ctl_grade[i] + self.htol:
                reset = 1                            # LOWLEVEL :303-307
            if net.ctl_type[i] == 1 and H[n] >= net.ctl_grade[i] - self.htol:
                reset = 1                            # HILEVEL :308-312
            if reset == 1:                           # :316-352
                change = 0
                s = int(S[k])
                if lt[k] == _PIPE and s != net.ctl_status[i]:
                    change = 1                       # :321-323
                if lt[k] == _PUMP and K[k] != net.ctl_setting[i]:
                    change = 1                       # :325-327
                if lt[k] >= _PRV:                    # :329-336
                    if K[k] != net.ctl_setting[i]:
                        change = 1
                    elif K[k] == MISSING and s != net.ctl_status[i]:
                        change = 1
                if change:                           # :339-351
                    S[k] = net.ctl_status[i]
                    if lt[k] > _PIPE:
                        K[k] = net.ctl_setting[i]
                    anychange = True
        return anychange

    # ------------------------------------------------------------------
    def run_gga(self, d, fixed_head, ke=None, q0=None, e0=None, status0=None,
                setting0=None, do_status=False, max_iter=None, extra_iter=None,
                hacc=None, record_schedule=False):
        """完整 GGA 内层求解（hydsolve hydsolver.c:57-212 的单样本 numpy 复刻）。

        do_status=False：状态冻结（快照回放/旧管网），数值收敛即终止；
        do_status=True：完整状态机 - 收敛后 statChange 复核（hydsolver.c:171-175），
        未收敛时 CheckFreq=2/MaxCheck=10 节律的周期 linkstatus（:181-187）。
        d/fixed_head 为全节点数组（定水头位从 fixed_head 取）。
        返回 dict（numpy）：head/flow/emitter/status/setting/iters/relerr/
        converged/fixed_demand（定水头节点净流入，newlinkflows :459-464 口径）。

        record_schedule=True（F2，调度级验收）：额外返回 "schedule" - 状态机
        每次被调用的轨迹 [(it, kind, changed), ...]，kind ∈ {"valve"
        （valvestatus，hydsolver.c:156/161 每迭代）、"ls_conv"（收敛支
        linkstatus :173）、"pswitch"（:174）、"ls_per"（周期支 linkstatus
        :185）}，changed = 该次调用改动的链路号列表（状态或设定，与调用前
        快照逐元素比）。缺省 False：零快照零开销，数值路径逐位不变。"""
        N, L, Nj = self.N, self.L, self.Nj
        d = np.asarray(d, dtype=np.float64)
        ke = np.asarray(self.net.node_ke if ke is None else ke, dtype=np.float64)
        ke_j = ke[self.junc_nodes]
        has_em = ke_j > 0.0
        any_em = bool(has_em.any())

        S = np.array(self.init_status_int if status0 is None else status0,
                     dtype=np.int8, copy=True)
        K = np.array(self.init_setting if setting0 is None else setting0,
                     dtype=np.float64, copy=True)
        closed0 = S <= self.ST_CLOSED
        if q0 is None:
            # 初始流量（hydraul.c:344-374 initlinkflow）：关闭=QZERO（:362-365），
            # 泵=Kc*Q0（:366-369），其余=1 fps 流速 PI*D²/4（:370-373）
            q = np.where(closed0, QZERO, PI * self.diam_np ** 2 / 4.0)
            if self.n_pumps:
                for k in self.pump_links:
                    k = int(k)
                    if not closed0[k]:
                        q[k] = K[k] * self.pl_q0[k]
        else:
            q = np.array(q0, dtype=np.float64, copy=True)
        if e0 is None:
            e_j = np.where(has_em, 1.0, 0.0)        # hydraul.c:115-121
        else:
            e_j = np.array(e0, dtype=np.float64)[self.junc_nodes]

        H = np.zeros(N, dtype=np.float64)
        H[self.fixed_nodes] = np.asarray(fixed_head,
                                         dtype=np.float64)[self.fixed_nodes]
        d_j = d[self.junc_nodes]
        row_junc = self.row_junc

        max_iter = self.max_iter_default if max_iter is None else int(max_iter)
        extra_iter = self.extra_iter if extra_iter is None else int(extra_iter)
        hacc = self.hacc_default if hacc is None else float(hacc)
        # maxtrials = MaxIter + ExtraIter（hydsolver.c:111-112）
        maxtrials = max_iter + (extra_iter if extra_iter > 0 else 0)
        nextcheck = self.checkfreq                   # hydsolver.c:99

        fixed_dem = np.zeros(N, dtype=np.float64)
        sched = [] if record_schedule else None       # F2：调度轨迹（缺省 None）
        iters = 0
        relerr = np.inf
        it = 1
        while it <= maxtrials:
            closed = S <= self.ST_CLOSED
            P, Y = self._PY_np(q, closed=closed, setting=K, status=S)
            # ---- matrixcoeffs（hydcoeffs.c:164-195）----
            Aii = np.zeros(N + 1, dtype=np.float64)
            Aij = np.zeros(self.sm.Ncoeffs + 1, dtype=np.float64)
            F = np.zeros(N + 1, dtype=np.float64)
            Xf = np.zeros(N + 1, dtype=np.float64)
            # linkcoeffs：np.add.at 按元素顺序累加 = C 的链路循环次序；
            # P==0 的链路整条跳过（hydcoeffs.c:218，ACTIVE PRV/PSV/FCV 由
            # valvecoeffs 接管） - 过滤保持元素顺序，全 True 时与不过滤逐位一致
            m_l = P != 0.0
            if m_l.all():
                np.add.at(Xf, self.xf_nodes, self.xf_sign * q[self.xf_lnk])
                np.add.at(Aij, self.Ndx_np, -P)                   # :229
                np.add.at(Aii, self.aii_rows, P[self.aii_lnk])    # :235,246
                fv = np.empty(self.f_rows.size, dtype=np.float64)
                fv[self.f_t0] = Y[self.f_lnk[self.f_t0]]          # :236
                fv[self.f_t1] = -Y[self.f_lnk[self.f_t1]]         # :247
                fv[self.f_t2] = P[self.f_lnk[self.f_t2]] * H[self.f_hnode[self.f_t2]]  # :240,251
                np.add.at(F, self.f_rows, fv)
            else:
                m_xf = m_l[self.xf_lnk]
                np.add.at(Xf, self.xf_nodes[m_xf],
                          (self.xf_sign * q[self.xf_lnk])[m_xf])  # :225-226
                np.add.at(Aij, self.Ndx_np[m_l], -P[m_l])         # :229
                m_aii = m_l[self.aii_lnk]
                np.add.at(Aii, self.aii_rows[m_aii],
                          P[self.aii_lnk][m_aii])                 # :235,246
                fv = np.empty(self.f_rows.size, dtype=np.float64)
                fv[self.f_t0] = Y[self.f_lnk[self.f_t0]]          # :236
                fv[self.f_t1] = -Y[self.f_lnk[self.f_t1]]         # :247
                fv[self.f_t2] = P[self.f_lnk[self.f_t2]] * H[self.f_hnode[self.f_t2]]  # :240,251
                m_f = m_l[self.f_lnk]
                np.add.at(F, self.f_rows[m_f], fv[m_f])
            # emittercoeffs（hydcoeffs.c:333-375）
            if any_em:
                hloss_e, hgrad_e = self._emitter_hloss_np(e_j, ke_j)
                Aii[row_junc[has_em]] += 1.0 / hgrad_e[has_em]
                F[row_junc[has_em]] += ((hloss_e + self.el_junc_np + 0.0)[has_em]
                                        / hgrad_e[has_em])
                Xf[1:Nj + 1][has_em] -= e_j[has_em]
            # nodecoeffs（hydcoeffs.c:256-279）：Xflow 就地扣需水（:276），
            # valvecoeffs 依赖扣完需水后的 Xflow（prvcoeff:973/976）
            Xf[1:Nj + 1] -= d_j                                   # :276
            F[row_junc] += Xf[1:Nj + 1]                           # :277
            # ---- valvecoeffs（hydcoeffs.c:282-330）：装配顺序依赖 - 必须在
            # nodecoeffs 之后读取已完成的 Xflow ----
            if self.valve_links.size:
                self._valvecoeffs(P, Y, Aii, Aij, F, Xf, q, S, K)
            # ---- linsolve（smatrix.c:729-871 移植；行空间）----
            Aii_l = Aii.tolist()
            Aij_l = Aij.tolist()
            F_l = F.tolist()
            errrow = self.sm.linsolve(Aii_l, Aij_l, F_l)
            if errrow:
                # 病态矩阵：badvalve 修复（hydsolver.c:127-131） - 若病态行属
                # 某 ACTIVE 控制阀，改其状态为 XPRESSURE/XFCV 后重试本迭代
                # （continue 不递增 iter，与 C 的 continue 一致）
                if self._badvalve(int(self.sm.Order[errrow]), S):
                    continue
                raise RuntimeError(f"linsolve 病态（行 {errrow}）")
            Fs = np.asarray(F_l)
            H[self.junc_nodes] = Fs[row_junc]       # hydsolver.c:135-138
            # ---- newflows（hydsolver.c:358-514；RelaxFactor=1.0）----
            dh = H[self.n1_np] - H[self.n2_np]
            dq = Y - P * dh                          # :430-431
            # 恒功率泵半步防穿零（hydsolver.c:437-444）：dq>Q 时 dq=Q/2
            if self.n_pumps:
                chp_hit = self.is_chp_np & (dq > q)
                if chp_hit.any():
                    dq = np.where(chp_hit, q / 2.0, dq)
            q = q - dq                               # :447
            # dqsum/qsum 逐元素顺序累加（hydsolver.c:447-449 的标量循环序；
            # np.sum 是 pairwise 舍入，与 C 顺序和的末位不同，可能在
            # relerr≈Hacc 边界翻转迭代数，故照抄 C 的左折叠次序）
            dqsum = sum(np.abs(dq).tolist())
            qsum = sum(np.abs(q).tolist())
            # 定水头节点净流入（newlinkflows :410-414 清零 + :459-464 累加，
            # 仅 status>CLOSED 的链路；EPS 的 tanklevels/controls 依赖此值）
            fixed_dem = np.zeros(N, dtype=np.float64)
            for k in self._fixed_links:
                if S[k] > self.ST_CLOSED:
                    kn1 = self.n1_np[k]
                    kn2 = self.n2_np[k]
                    if self.is_fixed_node[kn1]:
                        fixed_dem[kn1] -= q[k]       # :462
                    if self.is_fixed_node[kn2]:
                        fixed_dem[kn2] += q[k]       # :463
            if any_em:
                dh_e = H[self.junc_nodes] - self.el_junc_np
                dq_e = np.where(has_em, (hloss_e - dh_e) / hgrad_e, 0.0)
                e_j = e_j - dq_e
                # newemitterflows（hydsolver.c:488-504）：链路之后按节点序
                # 续折进同一累加器（不能先局部求和再一次加入，舍入序会变）
                for v in np.abs(dq_e[has_em]).tolist():
                    dqsum += v
                for v in np.abs(e_j[has_em]).tolist():
                    qsum += v
            relerr = dqsum / qsum if qsum > hacc else dqsum   # hydsolver.c:386-387
            iters = it
            if do_status:
                # DampLimit=0 → RelaxFactor 恒 1.0、每迭代 valvestatus
                # （hydsolver.c:149-162；DampLimit>0 已在构造时拒绝）
                if self.valve_links.size:
                    if sched is None:
                        valve_change = self._valvestatus(S, K, H, q)
                    else:                             # F2：快照仅记录时做
                        S_pre = S.copy(); K_pre = K.copy()
                        valve_change = self._valvestatus(S, K, H, q)
                        sched.append((it, "valve", np.where(
                            (S != S_pre) | (K != K_pre))[0].tolist()))
                else:
                    valve_change = False
                if relerr <= hacc:                   # hasconverged（hydsolver.c:611-643；
                    # HeadErrorLimit/FlowChangeLimit=0 → 只看 relerr；DDA 无 pdaconverged）
                    if it > max_iter:                # 已入额外迭代（hydsolver.c:168）
                        break
                    stat_change = valve_change
                    if sched is None:
                        if self._linkstatus(S, K, H, q):  # :173
                            stat_change = True
                        if self._pswitch(S, K, H):        # :174
                            stat_change = True
                    else:                             # F2：同一调用加前后快照
                        S_pre = S.copy()
                        if self._linkstatus(S, K, H, q):  # :173
                            stat_change = True
                        sched.append((it, "ls_conv",
                                      np.where(S != S_pre)[0].tolist()))
                        S_pre = S.copy(); K_pre = K.copy()
                        if self._pswitch(S, K, H):        # :174
                            stat_change = True
                        sched.append((it, "pswitch", np.where(
                            (S != S_pre) | (K != K_pre))[0].tolist()))
                    if not stat_change:               # :175
                        break
                    nextcheck = it + self.checkfreq   # :178
                elif it <= self.maxcheck and it == nextcheck:   # :183
                    if sched is None:
                        self._linkstatus(S, K, H, q)  # :185
                    else:
                        S_pre = S.copy()
                        self._linkstatus(S, K, H, q)  # :185
                        sched.append((it, "ls_per",
                                      np.where(S != S_pre)[0].tolist()))
                    nextcheck += self.checkfreq       # :186
            else:
                # 状态冻结：数值收敛即终止（旧 hasconverged 语义）
                if relerr <= hacc:
                    break
            it += 1
        # 迭代数计数口径（hydsolver.c:109 *iter=1 起、:189 (*iter)++ 后条件失败
        # 退出、:206 Iterations=*iter）：未收敛用尽 maxtrials 时 C 报
        # maxtrials+1；break 支保持当前 it（break 在 ++ 之前）。
        if it > maxtrials:
            iters = it

        emitter = np.zeros(N, dtype=np.float64)
        emitter[self.junc_nodes] = np.where(has_em, e_j, 0.0)
        out = dict(head=H, flow=q, emitter=emitter, status=S, setting=K,
                   iters=iters, relerr=relerr, converged=relerr <= hacc,
                   fixed_demand=fixed_dem)
        if sched is not None:
            out["schedule"] = sched
        return out

    def _solve_epanet_single(self, d, rh, ke, q0, e0, max_iter, hacc,
                             status0=None, setting0=None, do_status=False):
        """epanet 模式单样本求解（run_gga 的兼容包装）。"""
        r = self.run_gga(d, rh, ke=ke, q0=q0, e0=e0, status0=status0,
                         setting0=setting0, do_status=do_status,
                         max_iter=max_iter, hacc=hacc,
                         extra_iter=None if do_status else -1)
        return (r["head"], r["flow"], r["emitter"], r["iters"], r["relerr"],
                r["status"], r["setting"], r["fixed_demand"])

    # ------------------------------------------------------------------
    def _init_flow(self):
        """初始流量：hydraul.c:344-374 initlinkflow - 关闭支 = QZERO（hydraul.c:362-365），
        泵 = Kc*Q0（hydraul.c:366-369），管道/阀 = 1 fps 流速 = PI*D^2/4（:370-373）。

        注：无泵网（此前 dense 唯一支持的形态）下泵支为空掩码，返回值逐位不变。"""
        q = PI * self.diam ** 2 / 4.0
        if self.n_pumps:                                 # :366-369（先于 else 分支）
            qp = torch.as_tensor(self.init_setting * self.pl_q0,
                                 dtype=self.dtype, device=self.device)
            is_pump = torch.as_tensor(self.is_pump_np, device=self.device)
            q = torch.where(is_pump, qp, q)
        return torch.where(self.closed, torch.full_like(q, QZERO), q)

    def _pipe_PY(self, q):
        """pipecoeff：hydcoeffs.c:510-575（H-W 支）。q: [B,L]。返回 (P, Y)。"""
        # SGN(x) = x<0 ? -1 : 1（types.h:107），SGN(0)=+1。
        # qa = q*sgn ≡ |q|（前向数值逐位同 torch.abs，仅 ±0 的符号位可能不同、
        # 不影响任何非零结果），但 autograd 得 dqa/dq = sgn（q=0 处 = +1），
        # 与 EPANET 钳位支 hloss=RQtol·q 的解析导数 RQtol（hydcoeffs.c:554-558
        # 线性支）自洽；torch.abs 在 0 的 subgradient=0 会把该导数变 0，
        # 展开反传的误差经 P=1/RQtol=1e7 逐迭代放大而发散（city_d 死支
        # q 恰为 0 的钳位管实测 gA 随 K 指数增长）。
        sgn = torch.where(q < 0, -1.0, 1.0)
        qa = q * sgn
        # 摩阻梯度 hgrad = Hexp * R * |q|^(Hexp-1)（hydcoeffs.c:550）
        # autograd 防 0·inf→nan：|q|=0 时 pow 反传出 inf、与 where 选择支的 0 相乘成
        # nan。clamp_min(1e-30) 后 hgrad ≤ r·1e-25 级，必落 RQtol 钳位支（:554），
        # 前向逐位不变；反传变 0·有限=0。
        hgrad = self.hexp * self.r_hw * qa.clamp_min(1e-30) ** (self.hexp - 1.0)
        lin = hgrad < self.rqtol                        # RQtol 钳位支（:554-558）
        hgrad = torch.where(lin, torch.full_like(hgrad, self.rqtol), hgrad)
        hloss = torch.where(lin, hgrad * qa, hgrad * qa / self.hexp)  # :557,:560
        # 局损 Km 项（:563-567）：ml>0 时 hloss += ml*q^2, hgrad += 2*ml*q
        ml_pos = self.km_pipe > 0.0
        hloss = torch.where(ml_pos, hloss + self.km_pipe * qa * qa, hloss)
        hgrad = torch.where(ml_pos, hgrad + 2.0 * self.km_pipe * qa, hgrad)
        # 符号（:570）：SGN(x) = x<0 ? -1 : 1（types.h:107）
        hloss = hloss * sgn
        return 1.0 / hgrad, hloss / hgrad               # :573-574

    def _tcv_PY(self, q):
        """tcvcoeff→valvecoeff：hydcoeffs.c:912-939 与 :1100-1151（开启支）。"""
        km = self.km_tcv
        qa = torch.abs(q)
        # Km>0：hgrad = 2*Km*|q|（:1129），RQtol 钳位（:1132-1136），
        # 否则 hloss = flow*hgrad/2（:1137）；P=1/hgrad, Y=hloss/hgrad（:1140-1141）
        hgrad = 2.0 * km * qa
        lin = hgrad < self.rqtol
        hgrad = torch.where(lin, torch.full_like(hgrad, self.rqtol), hgrad)
        hloss = torch.where(lin, q * hgrad, q * hgrad / 2.0)
        P_km = 1.0 / hgrad
        Y_km = hloss / hgrad
        # Km==0（setting=0 → Km=0）：低阻线性支 P=1/CSMALL, Y=flow（:1146-1150）
        km_pos = km > 0.0
        P = torch.where(km_pos, P_km, torch.full_like(q, 1.0 / CSMALL))
        Y = torch.where(km_pos, Y_km, q)
        return P, Y

    def _pump_PY(self, q):
        """pumpcoeff（hydcoeffs.c:673-791）的 torch 批量复刻。q: [B,L]。

        返回 (P, Y)，仅泵位有意义（调用方用 is_pump_t 选择）。三条曲线分支
        （幂函数 :767-785 / CUSTOM :716-733 / CONST_HP :743-763）与各自钳位齐备；
        关闭与 ω==0（:696-701）由调用方的 closed_dense 统一覆盖为 P=1/CBIG, Y=Q。

        各 clamp_min / where 换基仅为"非选中支"的 0/inf 反传防护，前向选中支数值
        与 C 逐字相同（同 _pipe_PY 的 0·inf→nan 注释）；**幂函数支的 pow(0,N-1)
        不能靠 clamp 兜**，见下方注释。移植自 autodiff.solve_unrolled 的泵支，
        差别在 CUSTOM 分段查找改成 searchsorted+gather（原为 O(K·B·泵) 双循环）
        与上述 pow / CONST_HP 除法次序两处修正。
        """
        dt = self.dtype
        sgn = torch.where(q < 0, -1.0, 1.0)              # SGN，types.h:107
        qa = q * sgn
        spc = self.pump_speed_t.clamp_min(1e-12)         # ω>0（ω==0 走 closed 支）
        h0e = spc * spc * self.pump_h0_t                 # :737 h0 = SQR(setting)*H0
        re = self.pump_r_t * spc ** (2.0 - self.pump_n_t)   # :740 r = R*pow(w,2-N)

        # ---- CONST_HP（:743-763）：hgrad = -r/q/q，钳位到 [RQtol, CBIG] ----
        # 除法次序照抄 C 的 (-r/q)/q：写成 -r/(q*q) 与之差 1ulp（实测 ky4 泵位
        # |ΔP|~3e-18），没有理由白丢一位。q==0 时 clamp 后得巨值 → 落 :748 的
        # CBIG 支，与 C 的 -r/0/0=+inf 同支。
        qa_c = qa.clamp_min(1e-30)
        hg_chp_raw = -re / qa_c / qa_c                   # :746
        big_c = hg_chp_raw > CBIG                        # :748
        sml_c = hg_chp_raw < self.rqtol                  # :753
        hg_chp = torch.where(big_c, torch.full_like(qa, CBIG),
                             torch.where(sml_c, torch.full_like(qa, self.rqtol),
                                         hg_chp_raw))
        q_c = sgn * qa.clamp_min(1e-30)                  # 防 re/0
        hl_chp = torch.where(big_c | sml_c, -hg_chp * q,  # :750,:756
                             re / q_c)                    # :761

        # ---- 幂函数曲线（:767-779 非线性；:781-785 线性 n==1）----
        # pow(|q|, N-1) 必须逐字等于 C：|q|==0 时 C 的 pow 给 0（N>1）/ 1（N==1）
        # / +inf（N<1）。**不能沿用 _pipe_PY 的 clamp_min(1e-30) 写法** - 管道的
        # 指数 Hexp-1=0.852 固定，1e-30^0.852≈2.5e-26 必然落进 RQtol 钳位支、前向
        # 不变；泵的指数 N-1 由曲线拟合决定、可任意接近 0（实测 Net3 泵 N=1.088：
        # 1e-30^0.088=2.3e-3 ⇒ hgrad=6.7e-3 ≫ RQtol，钳位支被整支改写，P 从 1e7
        # 变成 1.5e2）。故显式取 C 的 pow(0,·)：基用 where 换成 1.0 只为反传安全
        # （0^负 的导数 inf 与 where 选择支的 0 相乘会成 nan，同 _pipe_PY 注释），
        # 前向值由 pw0 给定，逐字等于 C。
        # +inf 只发给"真正会选中这一支"的 POWER_FUNC 泵：CONST_HP/CUSTOM/NOCURVE
        # 与非泵位随后都会被覆盖，但 torch.where 的反传是 0×(未选中支导数)，
        # 未选中支若出 nan/inf 会把 0×nan 变成 nan 梯度 ⇒ 这些位一律给有限值。
        q_zero = qa == 0.0
        nm1 = self.pump_n_t - 1.0
        pw0 = torch.where(nm1 > 0.0, torch.zeros_like(nm1),
                          torch.where((nm1 < 0.0) & self.pump_pf_t,
                                      torch.full_like(nm1, float("inf")),
                                      torch.ones_like(nm1)))
        qa_pf = torch.where(q_zero, torch.ones_like(qa), qa)
        hg_pf_raw = self.pump_n_t * re * torch.where(q_zero,
                                                     pw0.expand_as(qa),
                                                     qa_pf ** nm1)
        sml_p = hg_pf_raw < self.rqtol                   # :772
        hg_pf = torch.where(sml_p, torch.full_like(qa, self.rqtol), hg_pf_raw)
        hl_pf = torch.where(sml_p, h0e + hg_pf * q,      # :774-775
                            h0e + hg_pf * q / self.pump_n_t.clamp_min(1e-30))  # :778
        hg_pf = torch.where(self.pump_lin_t, re, hg_pf)             # :783
        hl_pf = torch.where(self.pump_lin_t, h0e + re * q, hl_pf)   # :784

        hg_p = torch.where(self.pump_chp_t, hg_chp, hg_pf)
        hl_p = torch.where(self.pump_chp_t, hl_chp, hl_pf)

        # ---- CUSTOM 多段曲线（:716-733 + curvecoeff :794-831）----
        # 段选择离散（(H0,R) 为段内常数），用 detach 的 |q|/ω 查段；
        # hloss = H0·ω² + R·ω·Q 对 q（与 ω）保持可微。
        cl = self.pump_custom_links_t
        if cl is not None:
            B = q.shape[0]
            spc_c = self.pump_custom_sp_t.clamp_min(1e-12)          # [Nc]
            qsel = q.index_select(1, cl)                            # [B,Nc]
            # :722 curvecoeff(..., q/setting, ...) → :811 q *= Ucf[FLOW]
            qq = (qsel.detach().abs() / spc_c) * self.ucf_flow
            qq = qq.transpose(0, 1).contiguous()                    # [Nc,B]
            idx = torch.searchsorted(self.pump_curve_x_t, qq)       # :819 首个 x>=q
            k2 = torch.minimum(idx.clamp_min(1),                    # :820-821 夹取
                               (self.pump_curve_np_t - 1).unsqueeze(1))
            k1 = k2 - 1                                             # :822
            X, Yc = self.pump_curve_x_t, self.pump_curve_y_t
            x1 = X.gather(1, k1); x2 = X.gather(1, k2)
            y1 = Yc.gather(1, k1); y2 = Yc.gather(1, k2)
            rc = (y2 - y1) / (x2 - x1)                              # :825
            h0c = y1 - rc * x1                                      # :826
            h0c = h0c / self.ucf_head                               # :829
            rc = rc * self.ucf_flow / self.ucf_head                 # :830
            H0c = (-h0c).transpose(0, 1)                            # :726  [B,Nc]
            Rc = (-rc).transpose(0, 1)                              # :727
            hg_cu = Rc * spc_c                                      # :731（N=1.0）
            hl_cu = H0c * (spc_c * spc_c) + hg_cu * qsel            # :732
            zeros = torch.zeros(B, self.L, dtype=dt, device=q.device)
            hg_p = torch.where(self.pump_custom_t,
                               zeros.index_copy(1, cl, hg_cu), hg_p)
            hl_p = torch.where(self.pump_custom_t,
                               zeros.index_copy(1, cl, hl_cu), hl_p)

        P = 1.0 / hg_p                                   # :789
        Y = hl_p / hg_p                                  # :790
        # NOCURVE：当开阀（:709-714）
        P = torch.where(self.pump_nocurve_t,
                        torch.full_like(P, 1.0 / CSMALL), P)
        Y = torch.where(self.pump_nocurve_t, q, Y)
        return P, Y

    def _check_dense_pump_shutoff(self, H, q):
        """dense 收敛后复核 pumpstatus（hydstatus.c:205-239）的 XHEAD 判据。

        dense 路径没有状态机（solve() 明确拒绝 status/setting 相关入参），泵一旦
        被 EPANET 判为 XHEAD 就会被置成 P=1/CBIG 的高阻支（hydcoeffs.c:696-701），
        解将完全不同。此处只报错、不静默把泵当"永远开着"：
          linkstatus（hydstatus.c:147-151）对 status>=OPEN 且 LinkSetting>0 的泵调
          pumpstatus(k, -dh)，dh=H[n1]-H[n2] ⇒ 扬程增益 = H[n2]-H[n1]；
          hmax = CONST_HP ? BIG(:223-227) : SQR(setting)*Pump.Hmax(:231)；
          扬程增益 > hmax + Htol ⇒ XHEAD（:235）。2.2 不检查最大流量（:237 注释）。
        """
        if not self.n_pumps:
            return
        pl = torch.as_tensor(self.pump_links, dtype=torch.int64, device=H.device)
        # 只查"本可开着"的泵（关闭/ω==0 的泵 linkstatus 不会调 pumpstatus）
        live = ~self.closed_dense.index_select(0, pl)
        if not bool(live.any()):
            return
        n1 = self.n1.index_select(0, pl)
        n2 = self.n2.index_select(0, pl)
        gain = H.index_select(1, n2) - H.index_select(1, n1)     # 扬程增益 -dh
        sp = self.pump_speed_t.index_select(0, pl)
        hmax_np = np.where(self.is_chp_np[self.pump_links], BIG,
                           self.pl_hmax[self.pump_links])
        hmax = torch.as_tensor(hmax_np, dtype=H.dtype, device=H.device)
        hmax = torch.where(
            torch.as_tensor(self.is_chp_np[self.pump_links], device=H.device),
            hmax, sp * sp * hmax)                                # :223-231
        hit = (gain > hmax + self.htol) & live                   # :235
        if bool(hit.any()):
            b, j = [int(x) for x in torch.nonzero(hit)[0]]
            k = int(self.pump_links[j])
            raise NotImplementedError(
                f"mode=dense：场景 b={b} 的泵 {self.net.link_id[k]} 扬程增益 "
                f"{float(gain[b, j]):.6f} ft > 截止扬程 {float(hmax[j]):.6f} ft "
                f"+ Htol({self.htol:g})，EPANET 会把它判为 XHEAD 并置高阻关断"
                "（hydstatus.c:235 / hydcoeffs.c:696-701）。dense 路径没有状态机，"
                "继续算下去等于把泵当成永远开着 - 拒绝静默近似。请改用 "
                "mode='epanet' 并开 status_machine=True。")

    def _emitter_hloss(self, e, ke):
        """emitterheadloss：hydcoeffs.c:378-409。e/ke: [B,Nj]。返回 (hloss, hgrad)。"""
        ke_adj = torch.clamp(ke, min=CSMALL)            # ke = MAX(CSMALL, Ke)（:394）
        # clamp_min(1e-30)：同 _pipe_PY 的 0·inf 防护（qexp≠2 时 |e|=0 反传 inf；
        # 钳位值仍落 RQtol 支，前向逐位不变）
        hgrad = self.qexp * ke_adj * torch.abs(e).clamp_min(1e-30) ** (self.qexp - 1.0)  # :398
        lin = hgrad < self.rqtol                        # :401-405
        hgrad = torch.where(lin, torch.full_like(hgrad, self.rqtol), hgrad)
        hloss = torch.where(lin, hgrad * e, hgrad * e / self.qexp)       # :404,:408
        return hloss, hgrad

    # ------------------------------------------------------------------
    def _assemble_csr(self, vals, B):
        """CSR 值装配（sparse_gpu_plan.md §1a）：把 linkcoeffs 的四段贡献
        `vals` [B, len(A_idx)] 散射进 [B, nnz] 的 CSR data，而不是 [B, Nj*Nj]。

        位型见构造期 self.A_csr_*（row-major、(row,col) 字典序，含全部对角）。
        源序与稠密路径完全相同、目标映射是单射，故每个目标槽收到的加数序列
        逐项相同 ⇒ 与稠密 scatter_add 逐位一致（CPU 串行归约；CUDA 上两条
        通路都用 atomicAdd，本就不保证跨运行位级可复现）。
        """
        data = torch.zeros(B, self.A_csr_nnz, dtype=vals.dtype,
                           device=vals.device)
        data.scatter_add_(1, self.A_csr_scatter.expand(B, -1), vals)
        return data

    def _csr_to_dense(self, data, B):
        """[B, nnz] → [B, Nj, Nj]（纯搬运，无算术 ⇒ 逐位无损）。

        只服务 assemble="csr" + linear_solver="dense" 的组合（验装配等价）。
        linear_solver="cudss" 这条路**整段不执行**这个函数 - 显存墙就倒在
        这里（实测 ky4 B=256 的 solve 峰值 7343 → 195 MiB，见 p2_cudss_wip.txt §5）。
        """
        Nj = self.Nj
        return torch.zeros(B, Nj * Nj, dtype=data.dtype, device=data.device) \
            .scatter(1, self.A_csr_dense_pos.expand(B, -1), data) \
            .view(B, Nj, Nj)

    def _csr_spmv(self, data, x, B):
        """批量稀疏 SpMV：A·x，A 由 [B,nnz] 的 CSR 值给出，x [B,Nj] → [B,Nj]。

        只服务 cudss 通路的迭代精化残差（稠密通路用 (A*Hj^T).sum(-1)，不走这里）。
        逐 nnz 乘再按行 scatter_add：CUDA 上 scatter_add_ 的归约次序不定，
        故 GPU 验收须开 torch.use_deterministic_algorithms(True)。
        """
        prod = data * x.index_select(1, self.A_csr_col)          # [B,nnz]
        return torch.zeros(B, self.Nj, dtype=data.dtype, device=data.device) \
            .scatter_add(1, self.A_csr_row.expand(B, -1), prod)

    def _cudss_state(self, B, dtype, device, slot=0):
        """取（或首次构造）该 (B,dtype,device,matrix_type,slot) 的 cuDSS 有状态求解器。

        slot 只服务 P3 的反向复用（cudss_grad_slots）：纯前向恒取 slot=0，
        键与 P2 相比多一位常数，行为逐位不变。

        nvmath 1.0.0 只有**显式批量**（B 个独立 CSR 的列表，P0 实测），没有
        cuDSS 的 uniform-batch，所以这里建 B 个 sparse_csr_tensor，其 values
        全部是同一块 [B,nnz] 连续缓冲的行视图 - 每轮迭代只 copy_ 进缓冲，
        不重建对象、不重做 plan()。plan() 在此**只做一次**（耗时缓存于 state）。
        """
        device = torch.device(device)
        if device.type != "cuda":
            raise NotImplementedError(
                "linear_solver='cudss' 需要 CUDA 设备，当前 device=%s。"
                "本机无 CUDA 时请显式用 linear_solver='dense'（不做静默降级）。" % device)
        # 归一化卡号：构造期给的常是 torch.device('cuda')（无号），而张量带的是
        # 'cuda:0'。不归一化就会按两个键各建一份 state、各 plan 一次（实测
        # 1459212 号作业的 §4 里 cudss_plan() 预热完 solve 又 plan 了一次）。
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (int(B), dtype, str(device), str(self.cudss_matrix_type), int(slot))
        st = self._cudss_cache.get(key)
        if st is not None:
            self._cudss_cache.move_to_end(key)      # LRU：命中即最近使用
            return st
        if dtype != torch.float64:
            raise NotImplementedError(
                "linear_solver='cudss' 本轮只接 float64，收到 %s" % dtype)
        try:
            from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions
        except Exception as e:            # noqa: BLE001
            raise NotImplementedError(
                "linear_solver='cudss' 需要 nvmath-python（nvmath.sparse.advanced."
                "DirectSolver），导入失败：%r。请 pip install nvmath-python[cu12]，"
                "或显式改用 linear_solver='dense' - 不做静默降级。" % (e,)) from e
        Nj, nnz = self.Nj, self.A_csr_nnz
        vals = torch.zeros(B, nnz, dtype=dtype, device=device)
        # plan 只做重排 + 符号分解（与数值无关），但构造期仍给一个非奇异的值，
        # 免得后端在建对象时就摸数值：位型含全部对角 ⇒ 单位阵即可。
        vals.index_fill_(1, self.A_csr_diag, 1.0)
        rhs = torch.zeros(B, Nj, dtype=dtype, device=device)
        ip, ii = self.A_csr_indptr, self.A_csr_indices
        a_list = [torch.sparse_csr_tensor(ip, ii, vals[i], size=(Nj, Nj))
                  for i in range(B)]
        b_list = [rhs[i] for i in range(B)]
        # 写入目标：**必须**是 cuDSS 在 plan 时拿到指针的那块显存。
        # reset_operands(a=...) 会无条件把 solver_planned 置 False
        # （nvmath/sparse/advanced/direct_solver.py:1530） - 那等于每轮迭代作废 plan，
        # 正是 nvmath 自己警告要避开的用法（"update the values in place and
        # refactorize"）。所以这里先确认 sparse_csr_tensor 有没有拷贝 values：
        #   拷贝了 ⇒ 逐个写 values()；没拷贝 ⇒ 一句 copy_ 写整块 [B,nnz]。
        vviews = [a.values() for a in a_list]
        alias = all(int(v.data_ptr()) == int(vals[i].data_ptr())
                    for i, v in enumerate(vviews))
        opts = None if self.cudss_matrix_type is None else \
            DirectSolverOptions(sparse_system_type=self.cudss_matrix_type)
        solver = DirectSolver(a_list, b_list, options=opts)
        torch.cuda.synchronize(device)
        free0, tot0 = torch.cuda.mem_get_info(device)
        t0 = time.perf_counter()
        solver.plan()
        torch.cuda.synchronize(device)
        plan_ms = (time.perf_counter() - t0) * 1e3
        # 该 state 建立期间设备已用的增量（MiB），只作定 cudss_cache_max 的参考：
        # cuDSS 释放后的字节进的是可复用池，所以复用池时这个增量会偏小甚至为 0。
        try:
            free1, tot1 = torch.cuda.mem_get_info(device)
            dev_mib = ((tot1 - free1) - (tot0 - free0)) / 2 ** 20
        except Exception:                 # noqa: BLE001
            dev_mib = float("nan")
        st = dict(solver=solver, vals=vals, rhs=rhs, a=a_list, b=b_list,
                  vviews=vviews, alias=bool(alias),
                  plan_ms=plan_ms, dev_mib=dev_mib, B=int(B), slot=int(slot),
                  gen=0)     # gen：每 factorize() 一次 +1，反向据此判断能否复用
        self._cudss_cache[key] = st
        self._cudss_evict()
        return st

    @staticmethod
    def _cudss_release(st):
        """释放一份 cuDSS state：**先** DirectSolver.free()（还 cuDSS 内部那块
        不归 torch 管的分解/工作缓冲 + cudss handle/config/data 对象），**再**
        丢掉 torch 侧的 vals/rhs/稀疏张量引用。

        顺序不能反：a_list 里的 sparse_csr_tensor 的 values 是 vals 的行视图，
        cuDSS 在 plan 时记的就是这些指针；先放 torch 缓冲再 free() 等于让库在
        已归还的显存上做收尾。free() 自己会先 wait 上一次计算事件（nvmath
        sparse/advanced/direct_solver.py:1859 起），所以不用额外 synchronize。
        """
        solver = st.pop("solver", None)
        if solver is not None:
            solver.free()               # 幂等：valid_state 为 False 时直接 return
        for k in ("a", "b", "vviews", "vals", "rhs"):
            st.pop(k, None)

    def _cudss_evict(self):
        """把缓存压回 cudss_cache_max 份（LRU，从最久未用的一端逐出）。"""
        cap = self.cudss_cache_max
        if cap is None:
            return 0
        cap = int(cap)
        if cap < 1:
            raise ValueError("cudss_cache_max 必须 >=1 或 None（不限），"
                             "收到 %r" % (self.cudss_cache_max,))
        n = 0
        while len(self._cudss_cache) > cap:
            _, st = self._cudss_cache.popitem(last=False)
            self._cudss_release(st)
            n += 1
        return n

    def cudss_free(self, empty_cache=False):
        """显式释放**全部** cuDSS state（返回释放的份数）。

        为什么需要它：cuDSS 的分解缓冲不在 torch 缓存分配器里，
        `torch.cuda.empty_cache()` 一个字节都收不回（实测 ky4 B=256 一份 228 MiB，
        empty_cache 前后设备已用同为 816.2 MiB）。只有 DirectSolver.free() 能还。
        缺省的 LRU 上限（cudss_cache_max）已经托住了变批量训练，这个接口是给
        "用完这段就彻底放干净"的场合：评测完切下一张网、把显存让给别的模块、
        或者做显存实验。

        **口径提醒（实测，别夸大）**：free() 并不把字节还给驱动，只还给一个
        **可复用池** - B=256 释放后设备已用只掉 34 MiB（816.2 → 782.2），
        但同一 B 释放-重建 20 轮完全不单调，且释放 B=256 后建 B=1024 得到的
        占用与全新进程直接建 B=1024 一模一样（1328.2 MiB），说明那块被整块复用。
        所以本接口降的是**常驻水位**（新分配能从池里拿到），不是"归零"；
        对 OOM 起决定作用的正是常驻水位。

        empty_cache=True 时顺带 torch.cuda.empty_cache()，把 torch 侧那点
        [B,nnz]/[B,Nj] 缓冲也从缓存分配器还给驱动（缺省 False：清空 torch 缓存
        会让后续分配重新走 cudaMalloc，不该由一个释放接口悄悄替调用方决定）。
        """
        n = 0
        while self._cudss_cache:
            _, st = self._cudss_cache.popitem(last=False)
            self._cudss_release(st)
            n += 1
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return n

    @contextlib.contextmanager
    def cudss_session(self, empty_cache=False):
        """上下文管理器：退出时（含异常）无条件 cudss_free()。

            with s.cudss_session():
                out = s.solve(D, R, assemble="csr", linear_solver="cudss")
            # 到这里 cuDSS 那块设备内存已经还了
        """
        try:
            yield self
        finally:
            self.cudss_free(empty_cache=empty_cache)

    def cudss_cache_info(self):
        """当前缓存的 cuDSS state 一览（LRU 从旧到新）：
        [(B, dtype, device, matrix_type, plan_ms, 建立期设备内存增量 MiB,
          slot), ...]。用来给 cudss_cache_max 定价：上限 × 单份 MiB ≈
        缓存的常驻上限。末位 slot 见 cudss_grad_slots（纯前向恒为 0）。"""
        return [(k[0], k[1], k[2], k[3], v.get("plan_ms"), v.get("dev_mib"), k[4])
                for k, v in self._cudss_cache.items()]

    def cudss_counters(self, reset=False):
        """cuDSS 线性代数调用计数（dict 副本）：
          factorize        数值分解总次数（前向 + 反向重分解）
          solve            前向三角回代次数（每轮 = 1 主解 + cudss_refine 精化）
          bwd_solve        反向三角回代次数（每次 = 1 + cudss_grad_refine）
          bwd_reuse        反向**复用**前向数值分解的次数
          bwd_refactorize  反向不得不重做数值分解的次数（已含在 factorize 里）
        验收口径："factorize 没有翻倍" ⇔ bwd_refactorize == 0 ⇔
        factorize == 前向线性解次数。"""
        out = dict(self._cudss_counters)
        if reset:
            for k in self._cudss_counters:
                self._cudss_counters[k] = 0
        return out

    def cudss_plan(self, B, dtype=None, device=None):
        """显式预热：把该批量的 plan() 提前做掉（返回 plan 耗时 ms）。
        不调也可以 - 首次 solve 时会自动 plan 并缓存。

        注意：预热的 state 同样受 cudss_cache_max 的 LRU 约束 - 想一次预热
        k 个批量，先把 cudss_cache_max 调到 >=k，否则先热的会被后热的挤掉。"""
        st = self._cudss_state(int(B), dtype or self.dtype,
                               self.device if device is None else device)
        return st["plan_ms"]

    @staticmethod
    def _cudss_load(st, data):
        """把 [B,nnz] 的 CSR 值写进 cuDSS 在 plan 时拿到指针的那块缓冲。"""
        if st["alias"]:
            st["vals"].copy_(data)
        else:
            for i, v in enumerate(st["vviews"]):
                v.copy_(data[i])

    def _cudss_grad_slot(self):
        """可微通路的取槽（轮转）。见 cudss_grad_slots 的注释。"""
        n = int(self.cudss_grad_slots)
        if n < 1:
            raise ValueError("cudss_grad_slots 必须 >=1，收到 %r"
                             % (self.cudss_grad_slots,))
        cap = self.cudss_cache_max
        if cap is not None and n > int(cap):
            raise ValueError(
                "cudss_grad_slots=%d 大于 cudss_cache_max=%s：LRU 会把先建的槽"
                "逐出，反向就落回重分解，多占的显存白花。请把 cudss_cache_max "
                "调到 >=%d（或设 None=不限）。" % (n, cap, n))
        slot = self._cudss_slot_rr % n
        self._cudss_slot_rr = (self._cudss_slot_rr + 1) % n
        return slot

    def _cudss_forward(self, data, F, B, refine=None, slot=0):
        """cuDSS 解 A·H=F（A 由 [B,nnz] CSR 值给出）+ refine 步迭代精化。

        精化复用同一次数值分解（只 solve，不重 factorize），与稠密通路复用同一个
        Cholesky 因子是同一件事；残差用稀疏 SpMM 算。
        返回 (Hj, state, gen) - 后两个给 P3 的反向判断分解是否还在。
        """
        refine = self.cudss_refine if refine is None else refine
        st = self._cudss_state(B, data.dtype, data.device, slot)
        ds = st["solver"]
        self._cudss_load(st, data)
        st["rhs"].copy_(F)                 # b_list 是 rhs 的行视图，指针不变
        ds.factorize()                     # 数值分解；符号分解（plan）永不重做
        st["gen"] += 1
        self._cudss_counters["factorize"] += 1
        Hj = torch.stack(ds.solve())
        self._cudss_counters["solve"] += 1
        for _ in range(int(refine)):
            st["rhs"].copy_(F - self._csr_spmv(data, Hj, B))
            Hj = Hj + torch.stack(ds.solve())   # 复用同一次分解
            self._cudss_counters["solve"] += 1
        return Hj, st, st["gen"]

    def _cudss_adjoint(self, data, g, B, st, gen, slot):
        """反向：解 A^T·λ = g。A 对称 ⇒ 用前向那一次数值分解直接回代。

        只有当该 state 的分解还是前向那一次（gen 未被后续 factorize 顶掉、
        也没被 LRU 逐出）才复用；否则按保存的 data 重做一次数值分解，
        计入 bwd_refactorize - 这条退路会被计数器看见，不静默。
        """
        reuse = (st.get("solver") is not None and st.get("gen") == gen
                 and int(st.get("B", -1)) == int(B))
        if reuse:
            ds = st["solver"]
            self._cudss_counters["bwd_reuse"] += 1
        else:
            st = self._cudss_state(B, data.dtype, data.device, slot)
            ds = st["solver"]
            self._cudss_load(st, data)
            ds.factorize()
            st["gen"] += 1
            self._cudss_counters["factorize"] += 1
            self._cudss_counters["bwd_refactorize"] += 1
        st["rhs"].copy_(g)
        lam = torch.stack(ds.solve())
        self._cudss_counters["bwd_solve"] += 1
        for _ in range(int(self.cudss_grad_refine)):
            # A 对称 ⇒ 残差 g − A^T·λ 就是 g − A·λ，同一个 SpMV。
            st["rhs"].copy_(g - self._csr_spmv(data, lam, B))
            lam = lam + torch.stack(ds.solve())
            self._cudss_counters["bwd_solve"] += 1
        return lam

    def _cudss_factor(self, data, B, slot=0):
        """只做数值分解不回代（隐式伴随 GPU 轮的精抛光/终态分解用）。
        返回 (state, gen)；计数入 factorize（前向侧，诚实计数）。"""
        st = self._cudss_state(B, data.dtype, data.device, slot)
        self._cudss_load(st, data)
        st["solver"].factorize()
        st["gen"] += 1
        self._cudss_counters["factorize"] += 1
        return st, st["gen"]

    def _cudss_solve(self, data, F, B, refine=None):
        """线性解入口：需要梯度时走 _CudssSolveFn（P3，§1c），否则纯前向。"""
        if torch.is_grad_enabled() and (data.requires_grad or F.requires_grad):
            return _CudssSolveFn.apply(
                data, F, self, B,
                self.cudss_refine if refine is None else refine,
                self._cudss_grad_slot())
        return self._cudss_forward(data, F, B, refine, 0)[0]

    # ------------------------------------------------------------------
    def solve(self, demand_cfs, res_head_ft, ke_int=None, q0=None, e0=None,
              max_iter=None, accuracy=None, link_status=None, link_setting=None,
              status_machine=False, assemble="dense", linear_solver="dense",
              record_schedule=False):
        """单/批量稳态求解。demand_cfs [N]|[B,N]，res_head_ft [N]|[B,N]（非水库位 nan；
        tank 位需给定水头）。B2 增补（仅 epanet 模式）：link_status（内部 StatusType
        编码 int8[L]，None=初始状态）、link_setting（泵转速等 float64[L]）、
        status_machine（True=启用 hydstatus 状态机）。

        门 B1：构造时 dense_status_machine=True 后，mode="dense" 也接受
        status_machine=True - 走**批量状态机不动点**（hydsolve hydsolver.c:150-189
        的 [B,·] 复刻）：per-scenario 的状态向量 S[B,L] 与 nextcheck[B] 节律，
        "全批都无状态变化"才收工；覆盖 cvstatus / pumpstatus / tankstatus 三条转移，
        输出多一个 status（int8[B,L]）。含 CVPIPE 的网必须开（否则 raise）。
        link_status/link_setting 在该分支仍不支持（需要 [B,L] 的入口，见守卫文案）。

        PRV 轮：dense_status_machine=True 再放行 PRV。setting!=MISSING 的 PRV
        走批量 valvecoeffs（_prvcoeffs_batch，nodecoeffs 之后装配；ACTIVE =
        CBIG 罚函数约束行 prvcoeff:972-978，A 保持逐位对称）+ 每轮迭代的
        valvestatus（_prvstatus_batch，hydsolver.c:156/:161 节律），必须
        status_machine=True。含 pcv PRV 时输出多一个 diag_ratio（[B]，κ 的
        对角比值代理，ACTIVE 的 CBIG 对角是 κ~1e11 的来源）；批 Cholesky 失败
        显式 raise 并报样本号（badvalve 无批量对应物，dense_gap_plan §5）。
        f32 + pcv PRV 在构造期即拒。

        assemble（仅 dense 模式）："dense"=缺省，A 走 [B,Nj*Nj] 稠密散射（数值
        与历史逐位不变）；"csr"=走 [B,nnz] 的 CSR 值装配（sparse_gpu_plan.md
        §1a）。linear_solver='dense' 时 CSR 通路装配完仍散射回稠密走原 Cholesky。

        linear_solver（仅 dense 模式）："dense"=缺省，批量 Cholesky（数值与历史
        逐位不变）；"cudss"=nvmath 的 cuDSS DirectSolver（§1b），**必须配
        assemble='csr'**，全程不再开 [B,Nj*Nj]；需 CUDA + float64 + nvmath，
        缺一不可时明确 raise（不静默降级）。**P3 起可微**：需要梯度时线性解
        自动走 _CudssSolveFn（伴随复用前向的数值分解，§1c），其余节点仍是普通
        torch 算子 - 所以 d/rh/r_hw/ke/emitter 各类参数一视同仁，没有"某类参数
        做不到"的分支。

        record_schedule=True（F2，调度级验收；仅 dense + status_machine=True）：
        额外返回 "schedule" - 每次 _linkstatus_batch 被调用的迭代号与逐场景
        change 位图：[{it, conv[B], per[B], chg[B,L]}, ...]（conv/per = 该场景
        走的是收敛支 :173 还是周期支 :185；chg = 与调用入口状态逐元素比的改动
        位图，已按 run_ls 掩码）。缺省 False：零快照零拷贝，数值路径逐位不变。
        对拍器见 scripts/check_schedule.py。

        **契约（F3/F4，调用方必读）**：
        · iters 口径：收敛场景 = 收敛那一轮的迭代号；**未收敛场景 = maxtrials+1**
          （maxtrials = max_iter + ExtraIter[若>0]），与 run_gga / EPANET
          hydsolver.c:109/:189/:206 的计数口径一致（C 里 while 条件失败退出时
          *iter 已 ++ 到 maxtrials+1）。F3 前 dense 报 maxtrials、epanet 报
          maxtrials+1，两路不一致，已统一为后者。
        · 批量状态机分支（status_machine=True）**不因个别场景不收敛而 raise**：
          极端场景可把整网 TEMPCLOSED 孤岛化（门 B1 审计 Net2 8/128），此时
          该场景的解无物理意义，out["converged"][b]=False - **调用方必须逐
          场景检查 converged**，不查就消费 head/flow 属于用错 API。旧缺省
          dense（无状态机）的事后守卫命中即 raise 的行为不变。"""
        dev, dt = self.device, self.dtype
        if assemble not in ("dense", "csr"):
            raise ValueError(f"assemble 只能是 'dense'/'csr'，收到 {assemble!r}")
        if linear_solver not in ("dense", "cudss"):
            raise ValueError(
                f"linear_solver 只能是 'dense'/'cudss'，收到 {linear_solver!r}")
        if assemble == "csr" and self.mode != "dense":
            raise NotImplementedError(
                "assemble='csr' 只对 mode='dense' 有效；mode='epanet' 是逐位复刻"
                "基准（自己的 smatrix 稀疏通路），不接受装配替换")
        if linear_solver == "cudss":
            if self.mode != "dense":
                raise NotImplementedError(
                    "linear_solver='cudss' 只对 mode='dense' 有效；mode='epanet' 是逐位"
                    "复刻基准，不接受换线性求解器")
            if assemble != "csr":
                raise ValueError(
                    "linear_solver='cudss' 必须配 assemble='csr'（不再散射回稠密，"
                    "显存墙就倒在这里）；收到 assemble=%r" % (assemble,))
        if self.mode != "epanet" and not self.dense_status_machine:
            if link_status is not None or link_setting is not None or status_machine:
                raise NotImplementedError(
                    "link_status/link_setting/status_machine 仅 epanet 模式"
                    "（dense 需在构造时开 dense_status_machine=True）")
        if self.mode == "dense" and self.dense_status_machine:
            if link_setting is not None or link_status is not None:
                raise NotImplementedError(
                    "dense 批量状态机本轮只从 INP 初始构型（init_status_int）起步："
                    "link_setting 需要 per-scenario 的 [B,L] 设定（泵转速）并随之"
                    "批量化 pumpcoeff；link_status 的热启动同理要 [B,L] 入口。"
                    "两者都请走 mode='epanet'。")
            if bool(self.is_cv_t.any()) and not status_machine:
                raise NotImplementedError(
                    "本网含 CVPIPE，dense 路径必须 solve(..., status_machine=True)："
                    "CV 与 PIPE 的系数侧完全同分支（headlosscoeffs:138-141），唯一"
                    "差别就是 cvstatus（hydstatus.c:177-202）的状态覆盖；关掉状态机"
                    "等于把止回阀当成一根普通管，会静默算错。")
            if self._dense_prv_np and not status_machine:
                raise NotImplementedError(
                    "本网含 PRV（setting 未固定），dense 路径必须 "
                    "solve(..., status_machine=True)：valvestatus"
                    "（hydsolver.c:156/:161）每轮迭代都要跑，ACTIVE/OPEN/CLOSED "
                    "三态由 prvstatus（hydstatus.c:242-299）决定；冻结状态跑等于"
                    "把调压行为写死在初始构型上，会静默算错。")
        if record_schedule and not (self.mode == "dense" and status_machine):
            raise ValueError(
                "record_schedule=True 只覆盖 dense 批量状态机"
                "（mode='dense' + status_machine=True）；串行路径请直接用 "
                "run_gga(..., record_schedule=True)。")

        # 输入转换：torch.Tensor 走 .to()（保留 autograd 图；np.asarray 会对
        # requires_grad 张量抛错并断图），其余走 numpy 桥接（数值不变）。
        def _cvt(x):
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dt, device=dev)
            return torch.as_tensor(np.asarray(x), dtype=dt, device=dev)

        d = _cvt(demand_cfs)
        rh = _cvt(res_head_ft)
        single = d.dim() == 1
        if single:
            d = d.unsqueeze(0)
        if rh.dim() == 1:
            rh = rh.unsqueeze(0)
        B = d.shape[0]
        if rh.shape[0] == 1 and B > 1:
            rh = rh.expand(B, -1)

        ke = self.node_ke_default if ke_int is None else _cvt(ke_int)
        if ke.dim() == 1:
            ke = ke.unsqueeze(0)
        if ke.shape[0] == 1 and B > 1:
            ke = ke.expand(B, -1)
        ke_j = ke[:, self.junc_nodes_t]                  # [B,Nj]
        has_em = ke_j > 0.0

        max_iter = self.max_iter_default if max_iter is None else int(max_iter)
        hacc = self.hacc_default if accuracy is None else float(accuracy)

        # ---- epanet 模式：逐样本 numpy 精确路径 ----
        if self.mode == "epanet":
            B_ = d.shape[0]
            d_np = d.detach().cpu().numpy()
            rh_np = rh.detach().cpu().numpy() if rh.shape[0] == B_ else \
                np.broadcast_to(rh.detach().cpu().numpy(), (B_, self.N))
            ke_np = ke.detach().cpu().numpy() if ke.shape[0] == B_ else \
                np.broadcast_to(ke.detach().cpu().numpy(), (B_, self.N))
            q0_np = None if q0 is None else np.atleast_2d(np.asarray(q0, dtype=np.float64))
            e0_np = None if e0 is None else np.atleast_2d(np.asarray(e0, dtype=np.float64))
            Hs, Qs, Es, Is, Rs = [], [], [], [], []
            Ss, Ks, Fd = [], [], []
            for b in range(B_):
                Hb, Qb, Eb, ib, rb, Sb, Kb, fdb = self._solve_epanet_single(
                    d_np[b], rh_np[b], ke_np[b],
                    None if q0_np is None else q0_np[b % q0_np.shape[0]],
                    None if e0_np is None else e0_np[b % e0_np.shape[0]],
                    max_iter, hacc,
                    status0=link_status, setting0=link_setting,
                    do_status=status_machine)
                Hs.append(Hb); Qs.append(Qb); Es.append(Eb)
                Is.append(ib); Rs.append(rb)
                Ss.append(Sb); Ks.append(Kb); Fd.append(fdb)
            out = dict(
                head_ft=torch.as_tensor(np.stack(Hs), dtype=dt, device=dev),
                flow_cfs=torch.as_tensor(np.stack(Qs), dtype=dt, device=dev),
                emitter_cfs=torch.as_tensor(np.stack(Es), dtype=dt, device=dev),
                iters=torch.as_tensor(Is, dtype=torch.int64, device=dev),
                relerr=torch.as_tensor(Rs, dtype=dt, device=dev),
                converged=torch.as_tensor([r <= hacc for r in Rs], device=dev),
                status=torch.as_tensor(np.stack(Ss), dtype=torch.int8, device=dev),
                setting=torch.as_tensor(np.stack(Ks), dtype=dt, device=dev),
                fixed_demand_cfs=torch.as_tensor(np.stack(Fd), dtype=dt, device=dev),
            )
            if single:
                out = {k: v[0] for k, v in out.items()}
            return out

        # ---- 初值 ----
        if q0 is None:
            q = self._init_flow().unsqueeze(0).expand(B, -1).clone()
        else:
            q = _cvt(q0)
            q = q.unsqueeze(0).expand(B, -1).clone() if q.dim() == 1 else q.clone()
        if e0 is None:
            # EmitterFlow 初值：Ke>0 处 1.0（hydraul.c:115-121），否则 0
            e_j = torch.where(has_em, torch.ones_like(ke_j), torch.zeros_like(ke_j))
        else:
            e = _cvt(e0)
            e = e.unsqueeze(0).expand(B, -1) if e.dim() == 1 else e
            e_j = e[:, self.junc_nodes_t].clone()

        # 全头向量：定水头位 = res_head（demands()/inithyd 语义），junction 位后续覆盖。
        # autograd 干净化：不用就地 index_put_（H[:, idx] = ...），改为 cat + index_select
        # 的 out-of-place 重建（纯数据搬运，数值逐位不变，见 _hperm_inv）。
        rh_fix = rh[:, self.fixed_nodes_t]               # [B, N-Nj]，恒 = 边界水头
        if torch.isnan(rh_fix).any():
            raise ValueError("res_head_ft 在水库/水池位存在 nan")
        H_junc = torch.zeros(B, self.Nj, dtype=dt, device=dev)
        H = torch.cat([H_junc, rh_fix], dim=1).index_select(1, self._hperm_inv)

        d_j = d[:, self.junc_nodes_t]                    # DDA：DemandFlow=名义需水
        Nj = self.Nj

        active = torch.ones(B, dtype=torch.bool, device=dev)
        iters = torch.zeros(B, dtype=torch.int64, device=dev)
        relerr_out = torch.zeros(B, dtype=dt, device=dev)

        # ---- 门 B1：批量状态机不动点的 per-scenario 状态与节律 ----
        sm = bool(status_machine)
        maxtrials = max_iter
        S = nextcheck = None
        sched = [] if (record_schedule and sm and self.mode == "dense") else None
        # κ 代理（对角比值）：仅含 PRV 时逐轮更新并随 out 返回
        diag_ratio = torch.zeros(B, dtype=dt, device=dev) \
            if self._dense_prv_np else None
        if sm:
            S = torch.as_tensor(np.asarray(self.init_status_int, dtype=np.int8),
                                device=dev).view(1, -1).expand(B, -1).clone()
            # maxtrials = MaxIter + ExtraIter（hydsolver.c:111-112）
            maxtrials = max_iter + (self.extra_iter if self.extra_iter > 0 else 0)
            # nextcheck 逐场景独立（hydsolver.c:99 + :178/:186）
            nextcheck = torch.full((B,), self.checkfreq, dtype=torch.int64,
                                   device=dev)

        for it in range(1, maxtrials + 1):
            # ---- headlosscoeffs：逐链路 P/Y（hydcoeffs.c:119-161）----
            P_pipe, Y_pipe = self._pipe_PY(q)
            P_tcv, Y_tcv = self._tcv_PY(q)
            P = torch.where(self.is_tcv, P_tcv, P_pipe)
            Y = torch.where(self.is_tcv, Y_tcv, Y_pipe)
            # 泵（pumpcoeff hydcoeffs.c:673-791）
            if self.n_pumps:
                P_pump, Y_pump = self._pump_PY(q)
                P = torch.where(self.is_pump_t, P_pump, P)
                Y = torch.where(self.is_pump_t, Y_pump, Y)
            # 固定 PRV（setting==MISSING）：valvecoeff 开启支（headlosscoeffs
            # :154-157 的 MISSING 分派；关闭支由下方 closed_now 覆盖）
            if self._dense_prv_fixed_any:
                P_vf, Y_vf = self._valve_PY_ml(q)
                P = torch.where(self.prv_fixed_t, P_vf, P)
                Y = torch.where(self.prv_fixed_t, Y_vf, Y)
            # 关闭支（pipecoeff:531-536 / valvecoeff:1118-1123 / pumpcoeff:696-701）：
            # P=1/CBIG, Y=Q̂（closed_dense 含 ω==0 的泵）。开状态机时同一个口径
            # 改由 per-scenario 的 S 给出：closed = (S<=CLOSED) | ω==0，
            # 与 _PY_np(closed=S<=ST_CLOSED) + 泵支的 `sp == 0.0` 逐条对应。
            closed_now = ((S <= self.ST_CLOSED) | self.sm_zero_speed_t) if sm \
                else self.closed_dense
            if self._dense_prv_np:
                # setting!=MISSING 的 PRV：无论状态如何 headloss 阶段 P=0
                # （headlosscoeffs:157-158），CLOSED 的 1/CBIG 支由 valvecoeffs
                # 里的 valvecoeff（:1118-1123）接管 - closed 覆盖必须绕开它们
                closed_now = closed_now & ~self.pcv_static_t
            P = torch.where(closed_now, torch.full_like(P, 1.0 / CBIG), P)
            Y = torch.where(closed_now, q, Y)
            if self._dense_prv_np:
                # linkcoeffs:218 的 continue ⇒ 该链路不进 Xflow/A/F（两侧都不写）
                P = torch.where(self.pcv_static_t, torch.zeros_like(P), P)
                Y = torch.where(self.pcv_static_t, torch.zeros_like(Y), Y)
            pl = (P != 0.0).to(dt)                       # linkcoeffs:218 跳过 P==0

            # ---- matrixcoeffs（hydcoeffs.c:164-195 顺序：link→emitter→node）----
            # linkcoeffs（:198-253）
            Pm, Ym, qm = P * pl, Y * pl, q * pl
            # Xflow：n1 端 -Q，n2 端 +Q（:225-226）
            Xflow = torch.zeros(B, Nj, dtype=dt, device=dev)
            Xflow.scatter_add_(1, self.f_idx1.expand(B, -1), -qm[:, self.lk_m1])
            Xflow.scatter_add_(1, self.f_idx2.expand(B, -1), qm[:, self.lk_m2])
            # A：非对角 -P（:229），junction 端对角 +P（:235,246）
            vals = torch.cat([-Pm[:, self.lk_both], -Pm[:, self.lk_both],
                              Pm[:, self.lk_m1], Pm[:, self.lk_m2]], dim=1)
            if assemble == "csr":
                # §1a：值只落 [B,nnz]（不再开 [B,Nj*Nj]）。linear_solver='dense'
                # 时装配完再散射回稠密走原 Cholesky（只验装配等价）；
                # linear_solver='cudss' 时 **不做这一步**（§1b：显存墙倒在这里）。
                csr_data = self._assemble_csr(vals, B)
                A = None if linear_solver == "cudss" \
                    else self._csr_to_dense(csr_data, B)
            else:
                A = torch.zeros(B, Nj * Nj, dtype=dt, device=dev)
                A.scatter_add_(1, self.A_idx.expand(B, -1), vals)
                A = A.view(B, Nj, Nj)
            # F：junction 端 ±Y（:236,247）；定水头端接地 +P*Head（:240,251）
            F = torch.zeros(B, Nj, dtype=dt, device=dev)
            F.scatter_add_(1, self.f_idx1.expand(B, -1), Ym[:, self.lk_m1])
            F.scatter_add_(1, self.f_idx2.expand(B, -1), -Ym[:, self.lk_m2])
            if self.lk_g1.numel():
                F.scatter_add_(1, self.g1_row.expand(B, -1),
                               Pm[:, self.lk_g1] * H[:, self.g1_src])
            if self.lk_g2.numel():
                F.scatter_add_(1, self.g2_row.expand(B, -1),
                               Pm[:, self.lk_g2] * H[:, self.g2_src])

            # emittercoeffs（hydcoeffs.c:333-375）：对角 +1/hgrad，
            # F += (hloss+El)/hgrad（虚拟水库水头 = El），Xflow -= EmitterFlow
            hloss_e, hgrad_e = self._emitter_hloss(e_j, ke_j)
            em = has_em.to(dt)
            # 对角 += em/hgrad：diag_embed 加法与就地 index_put_ 累加逐位一致
            # （非对角 +0.0 不改变位型），且对 autograd 全程 out-of-place。
            # cudss 通路直接加到 CSR 的对角槽（A_csr_diag）上：同一次加法、
            # 被加数已逐位相等（P1）⇒ 结果也逐位相等。
            if linear_solver == "cudss":
                csr_data = csr_data.index_add(1, self.A_csr_diag, em / hgrad_e)
            else:
                A = A + torch.diag_embed(em / hgrad_e)
            F = F + em * (hloss_e + self.el_junc) / hgrad_e
            Xflow = Xflow - em * e_j

            # nodecoeffs（hydcoeffs.c:256-279）：Xflow -= DemandFlow；F += Xflow
            Xflow = Xflow - d_j
            F = F + Xflow

            # ---- valvecoeffs（hydcoeffs.c:282-330）：装配次序绑死 - 必须在
            # nodecoeffs 之后（ACTIVE 的 Y 读扣完需水/emitter 的 Xflow）----
            if self._dense_prv_np:
                if linear_solver == "cudss":
                    P, Y, F, _, csr_data = self._prvcoeffs_batch(
                        P, Y, F, Xflow, q, S, csr_data=csr_data)
                else:
                    P, Y, F, A, _ = self._prvcoeffs_batch(
                        P, Y, F, Xflow, q, S, A=A)

            # ---- 线性求解 A·H=F（对称正定 → Cholesky；稠密下无需 MMD 重排）----
            # 注：EPANET 稀疏 Cholesky（smatrix.c linsolve）与稠密分解的舍入路径不同；
            # 对含 1/CSMALL=1e6（TCV）与 1/CBIG=1e-8（关闭支）的病态 A，
            # 追加两步同精度迭代精化，把我们的解压到线性系统的"精确解"附近，
            # 消除分解顺序带来的 ~1e-6 ft 级差异（对拍硬门槛 1e-6 ft 所需）。
            # κ 在线监控（对角比值代理；仅含 PRV 时算 - ACTIVE 的 CBIG 对角是
            # κ~1e11 的来源，dense_gap_plan §1）。逐场景保留最后活跃轮的值。
            if self._dense_prv_np:
                diag_now = (csr_data.index_select(1, self.A_csr_diag)
                            if linear_solver == "cudss"
                            else torch.diagonal(A, dim1=-2, dim2=-1)).detach()
                dr_now = diag_now.abs().max(dim=1).values \
                    / diag_now.abs().min(dim=1).values.clamp_min(1e-300)
                diag_ratio = torch.where(active, dr_now, diag_ratio)
            if linear_solver == "cudss":
                # §1b：cuDSS 稀疏直接法（plan 已在构造/首解时做掉并缓存）+
                # 2 步迭代精化（残差走稀疏 SpMM）。全程无 [B,Nj*Nj]。
                Hj = self._cudss_solve(csr_data, F, B)
            elif dt == torch.float64:
                if self._dense_prv_np:
                    # badvalve（hydsolver.c:215-265）在批量下无对应物（"只重试
                    # 第 b 个样本"没有批语义，dense_gap_plan §2 硬点 3 / §5）：
                    # 批 Cholesky 失败显式捕获 + 报样本号，不做静默重试。
                    # cholesky_ex 与 cholesky 同一 LAPACK 例程，因子逐位相同。
                    chol, chol_info = torch.linalg.cholesky_ex(A)
                    if bool((chol_info != 0).any()):
                        bad = torch.nonzero(chol_info != 0).flatten().tolist()
                        raise RuntimeError(
                            "dense 批量 Cholesky 失败（含 PRV 网，迭代 %d）："
                            "样本 %s 的 A 非正定。EPANET 对此走 badvalve"
                            "（hydsolver.c:215-265）把肇事 ACTIVE 阀改判 "
                            "XPRESSURE 后重试，批量路径无对应物 - 请把这些"
                            "样本剔出该批改走 mode='epanet'。" % (it, bad))
                else:
                    chol = torch.linalg.cholesky(A)
                Fc = F.unsqueeze(-1)
                Hj = torch.cholesky_solve(Fc, chol)
                for _ in range(int(self.dense_refine)):
                    # 残差 A·Hj 不用 bmm：CPU 的批量 gemm 在 B=1 与 B≥2 走不同 MKL
                    # 路径（实测同数据 B=1 Δ=0、B≥2 Δ=8.9e-15，经 κ~1e9 病态放大为
                    # ~1e-6 ft 头差，破坏"批量=逐场景"位级一致）。改用逐元素乘 +
                    # 末维求和（两者实测对任意 B 位级不变），保证批不变性。
                    AHj = (A * Hj.transpose(-2, -1)).sum(-1, keepdim=True)
                    resid = Fc - AHj
                    Hj = Hj + torch.cholesky_solve(resid, chol)
                Hj = Hj.squeeze(-1)
            else:
                # float32 等低精度：A 对角横跨 1/CBIG=1e-8（关闭支）~1e6（TCV），
                # κ≈1e9 超出 1/eps_f32≈8e6，Cholesky 必报"非正定"（实测对称对角
                # 均衡后仍失败：TCV 强耦合行均衡后非对角≈1-5e-7，f32 下主元被舍入
                # 噪声淹没）。改用 D^-1/2·A·D^-1/2 均衡（精确算术下解不变）+ 带部分
                # 主元 LU：A·x=F ⇔ As·y=D^-1/2·F, x=D^-1/2·y。解的误差仍是
                # κ·eps_f32 量级，仅作速度/精度量级观察。float64 路径保持逐位不变。
                sdi = torch.rsqrt(torch.diagonal(A, dim1=-2, dim2=-1))   # [B,Nj]
                As = A * sdi.unsqueeze(-1) * sdi.unsqueeze(-2)
                # 均衡后 TCV 强耦合块 ≈ [[1,-1+5e-7],[-1+5e-7,1]]，f32 舍入后精确
                # 奇异（实测 LU 报 U 零主元）。加 λ=100·eps 对角正则（均衡后对角=1，
                # 相对扰动 ~1e-5）打破精确奇异 - f32 解本就只有 κ·eps_f32 量级精度，
                # 此扰动不改变其误差数量级。
                lam = 100.0 * torch.finfo(dt).eps
                Nj_ = As.shape[-1]
                As = As + lam * torch.eye(Nj_, dtype=dt, device=As.device)
                LUf, piv = torch.linalg.lu_factor(As)
                Fc = (F * sdi).unsqueeze(-1)
                Hj = torch.linalg.lu_solve(LUf, piv, Fc)
                for _ in range(int(self.dense_refine)):
                    AHj = (As * Hj.transpose(-2, -1)).sum(-1, keepdim=True)
                    Hj = Hj + torch.linalg.lu_solve(LUf, piv, Fc - AHj)
                Hj = Hj.squeeze(-1) * sdi
            # 冻结已收敛样本的头；out-of-place 重建全头向量（数值逐位同就地写）
            H_junc = torch.where(active.view(-1, 1), Hj, H_junc)
            H = torch.cat([H_junc, rh_fix], dim=1).index_select(1, self._hperm_inv)

            # ---- newflows（hydsolver.c:358-514）----
            # dq = Y - P*(H1-H2)（:430-431）；RelaxFactor=1.0（DampLimit=0，:149-162）
            dh = H[:, self.n1] - H[:, self.n2]
            dq = Y - P * dh
            if self.n_pumps:
                # 恒功率泵半步防穿零（hydsolver.c:437-443）：dq>Q 时 dq=Q/2
                dq = torch.where(self.pump_chp_t & (dq > q), q / 2.0, dq)
            q_new = q - dq                               # :447
            # emitter 更新（:488-513）：dq=(hloss-dh)/hgrad，dh=H-El
            dh_e = H[:, self.junc_nodes_t] - self.el_junc
            dq_e = (hloss_e - dh_e) / hgrad_e
            e_new = e_j - em * dq_e
            # relerr = Σ|dq| / Σ|Q|（更新后求和：:447-449,:500-504；分母含 emitter）
            dqsum = torch.sum(torch.abs(dq), dim=1) + \
                torch.sum(em * torch.abs(dq_e), dim=1)
            qsum = torch.sum(torch.abs(q_new), dim=1) + \
                torch.sum(em * torch.abs(e_new), dim=1)
            # qsum > Hacc 时取比值，否则取 dqsum（hydsolver.c:386-387）
            relerr = torch.where(qsum > hacc, dqsum / qsum, dqsum)

            # 冻结已收敛样本（EPANET 在收敛当轮更新完 q/H 后 break）
            am = active.view(-1, 1)
            q = torch.where(am, q_new, q)
            e_j = torch.where(am, e_new, e_j)
            iters = torch.where(active, torch.full_like(iters, it), iters)
            relerr_out = torch.where(active, relerr, relerr_out)
            # hasconverged（hydsolver.c:611-643）：relerr<=Hacc 即收敛
            # （HeadErrorLimit/FlowChangeLimit 默认 0 → 跳过；DDA 无 pdaconverged）
            if not sm:
                # 状态冻结：数值收敛即终止（本分支逐位不变）
                active = active & (relerr > hacc)
                if not bool(active.any()):
                    break
                continue
            # ---- 批量状态机不动点（hydsolve hydsolver.c:164-188）----
            # 每个场景有自己的状态轨迹与 nextcheck 节律 ⇒ per-scenario 冻结掩码；
            # "全批都无状态变化"才收工（active 全 False 才 break）。
            # **不按状态分桶**（dense_gap_plan.md §1(c)：D-Town B=128 已 16 桶，
            # 桶数随 B 增长）：三支全算 + torch.where，位型恒定。
            # ---- valvestatus（hydsolver.c:149-162）：DampLimit=0 ⇒ 每轮迭代
            # 都跑（:161），与 linkstatus 的"收敛/CheckFreq 周期"节律不同 -
            # 这条双节律照 hydsolve 复刻，用调度轨迹对拍器验收（F2）。
            if self._dense_prv_np:
                S_v, vchg = self._prvstatus_batch(S, H, q)
                if sched is not None:                  # F2：valve 调度事件
                    sched.append(dict(
                        it=it, kind="valve",
                        act=active.detach().cpu().numpy().copy(),
                        chg=((S_v != S) & active.view(-1, 1))
                        .detach().cpu().numpy().copy()))
                S = torch.where(active.view(-1, 1), S_v, S)
                valve_change = vchg & active
            else:
                valve_change = None
            conv = relerr <= hacc
            # (a) 收敛且仍在正常迭代段 ⇒ 复核状态（:171-179）
            br_conv = active & conv & (it <= max_iter)
            # (b) 未收敛的周期检查（:183-187）
            br_per = active & (~conv) & (it <= self.maxcheck) & (nextcheck == it)
            run_ls = br_conv | br_per
            if bool(run_ls.any()):
                S_new, chg = self._linkstatus_batch(S, H, q)
                if sched is not None:                  # F2：调度轨迹（缺省关）
                    sched.append(dict(
                        it=it,
                        conv=br_conv.detach().cpu().numpy().copy(),
                        per=br_per.detach().cpu().numpy().copy(),
                        chg=((S_new != S) & run_ls.view(-1, 1))
                        .detach().cpu().numpy().copy()))
                S = torch.where(run_ls.view(-1, 1), S_new, S)
                # pswitch（:174）在构造期已被守卫排除（junction 压力控制数=0）
                stat_change = chg & br_conv
            else:
                stat_change = torch.zeros(B, dtype=torch.bool, device=dev)
            if valve_change is not None:
                # statChange = valveChange || linkstatus()（hydsolver.c:171-173）
                stat_change = stat_change | (valve_change & br_conv)
            if bool(run_ls.any()):
                nextcheck = torch.where(br_conv & stat_change,
                                        torch.full_like(nextcheck, it + self.checkfreq),
                                        nextcheck)                     # :178
                nextcheck = torch.where(br_per, nextcheck + self.checkfreq,
                                        nextcheck)                     # :186
            # 收工条件：收敛 且（已入 ExtraIter 段 :168 或 本轮无状态变化 :175）
            done_now = active & conv & ((it > max_iter) | (~stat_change))
            active = active & (~done_now)
            if not bool(active.any()):
                break

        # F3：未收敛 iters 口径统一 - 与 run_gga / EPANET（hydsolver.c:109
        # `*iter=1` 起、:189 `(*iter)++` 后条件失败退出、:206 Iterations=*iter）
        # 一致：用尽 maxtrials 仍未收工的场景报 maxtrials+1（此前 dense 报
        # maxtrials）。收敛场景逐位不变（active 已 False，where 不触碰）。
        iters = torch.where(active,
                            torch.full_like(iters, maxtrials + 1), iters)

        if sm:
            # 状态机在位 ⇒ tankstatus/pumpstatus 已被真正执行，事后守卫不再适用
            # （它们是"没有状态机"时的止损，命中即 raise）。
            st_out = S
        else:
            # 事后状态复核：dense 无状态机，命中即 raise（不静默当"泵/管永远开着"）
            self._check_dense_pump_shutoff(H.detach(), q.detach())
            self._check_dense_tank_status(H.detach(), q.detach())
            st_out = None

        emitter = torch.zeros(B, self.N, dtype=dt, device=dev)
        emitter[:, self.junc_nodes_t] = e_j * has_em.to(dt)
        out = dict(head_ft=H, flow_cfs=q, emitter_cfs=emitter,
                   iters=iters, relerr=relerr_out,
                   converged=~active)
        if st_out is not None:
            out["status"] = st_out
        if diag_ratio is not None:
            # κ 代理（各场景最后活跃轮的 max|diag|/min|diag|；仅含 PRV 时返回）
            out["diag_ratio"] = diag_ratio
        if single:
            out = {k: v[0] for k, v in out.items()}
        if sched is not None:
            out["schedule"] = sched                    # F2：list，不参与去批维
        return out


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 冒烟测试：city_d 第 0 帧（冷启动）两种模式对拍参考解 + 批量/emitter 通路
    import os as _os  # noqa: E402
    ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    net = Net.load(ROOT + "/data/reference", "city_d")
    ref = np.load(ROOT + "/data/reference/city_d_ref.npz")
    inp = ROOT + "/networks/realInpData/city_d.inp"
    jm = np.asarray(net.node_type) == 0
    opened = ref["status"][0] == 1
    t = int(ref["t_sec"][0])
    d0, rh0 = net.demand_cfs_at(t), net.reservoir_head_ft_at(t)

    for mode in ("epanet", "dense"):
        s = GGASolver(net, mode=mode, inp_path=inp)
        r = s.solve(d0, rh0)
        dH = np.abs(r["head_ft"].numpy() - ref["head_ft"][0])[jm].max()
        dQ = np.abs(r["flow_cfs"].numpy() - ref["flow_cfs"][0])[opened].max()
        print(f"[{mode}] 帧0: iters={int(r['iters'])} (EPANET {int(ref['iterations'][0])}) "
              f"relerr={float(r['relerr']):.3e} max|ΔH|={dH:.3e} ft max|ΔQ|={dQ:.3e} cfs")
        assert int(r["iters"]) == int(ref["iterations"][0])
    # epanet 模式必须达位级（npz 往返 ulp 下限 ~1.5e-14）
    s = GGASolver(net, mode="epanet", inp_path=inp)
    r = s.solve(d0, rh0)
    assert np.abs(r["head_ft"].numpy() - ref["head_ft"][0])[jm].max() < 1e-12

    # 批量通路：两帧堆叠 == 逐帧
    t1 = int(ref["t_sec"][1])
    dB = np.stack([d0, net.demand_cfs_at(t1)])
    rB = np.stack([rh0, net.reservoir_head_ft_at(t1)])
    rb = s.solve(dB, rB)
    r1 = s.solve(net.demand_cfs_at(t1), net.reservoir_head_ft_at(t1))
    assert np.array_equal(rb["head_ft"][1].numpy(), r1["head_ft"].numpy())
    print("批量通路一致 OK")

    # emitter 通路：两模式互拍（city_d 无 emitter 参考，做交叉自洽）
    ke = np.zeros(net.N)
    ke[np.where(jm)[0][:20]] = 0.5
    re_ = GGASolver(net, mode="epanet", inp_path=inp).solve(d0, rh0, ke_int=ke)
    rd_ = GGASolver(net, mode="dense", inp_path=inp).solve(d0, rh0, ke_int=ke)
    dE = np.abs(re_["emitter_cfs"].numpy() - rd_["emitter_cfs"].numpy()).max()
    dHm = np.abs(re_["head_ft"].numpy() - rd_["head_ft"].numpy()).max()
    print(f"emitter 交叉自洽: max|Δe|={dE:.3e} cfs, max|ΔH|={dHm:.3e} ft "
          f"(iters {int(re_['iters'])}/{int(rd_['iters'])})")
    assert dE < 1e-6 and dHm < 1e-5
    print("solver.py 冒烟测试全部通过")

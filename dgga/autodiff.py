# -*- coding: utf-8 -*-
"""dgga.autodiff - GGA 稳态解的两条反传路线（阶段 C）。

路线一 solve_unrolled：固定迭代数 K 的 dense GGA 展开（全程 out-of-place torch 运算，
对 demand / 水库水头 / Ke / r_hw 可微）。数学同 solver.py dense 路径（hydcoeffs.c /
hydsolver.c 逐式对应，见各函数注释），只去掉收敛判定与冻结掩码。

路线二 ImplicitGGASolve：隐函数定理伴随法（torch.autograd.Function）。
forward 在 no_grad 下用 GGASolver(dense) 收敛后、再对完整残差系统 F(z;θ)=0 做
少量整装 Newton "精抛光"（scipy 稀疏 LU，不走 GGA 的 Schur 消元 - GGA 半迭代在
κ~1e9 病态网（city_d 的 TCV 1/CSMALL 与 RQtol 钳位对角）上 relerr 停滞 ~1e-7，
而整装 J 的 LU 能把 ‖F‖∞ 压到机器精度级，使 FD 对拍噪声降到 1e-12 ft 量级）。
backward 解伴随系统 J^T λ = ∂L/∂z 再收缩 ∂F/∂θ。

残差系统（EPANET 2.2 收敛条件的不动点形式；报告 7.1/7.5 节）：
  z = (Q[L], qE[发射器], H[junction])
  链路能量  r_k = φ_k(Q_k) − (H_{n1} − H_{n2})       （定水头端 H 取边界值）
            φ 与 hydcoeffs.c pipecoeff/valvecoeff 的 hloss 逐支一致
            （hydsolver.c:430-431 的 dq = (hloss−dh)/hgrad → 不动点即 r_k=0）
  emitter   r_e = Ke·qE^Qexp − (H_i − El_i)          （hydcoeffs.c:378-409 的 hloss，
            含 max(CSMALL,Ke) 与 RQtol 钳位；hydsolver.c:497 dq=(hloss−dh)/hgrad）
  junction  m_i = Σ_{n2=i}Q − Σ_{n1=i}Q − qE_i − d_i （hydcoeffs.c:225-226 Xflow 的
            符号约定：n1 端 −Q、n2 端 +Q；:373 Xflow−=emitter；:268 Xflow−=demand）
雅可比分块  J = [[D, 0, −A12], [0, E', S], [A21, P_e, 0]]，
  D = diag(φ'_k)＝求解器同款 hgrad（RQtol 钳位支的导数=钳位后 hgrad，与收敛残差自洽），
  E' = diag(emitter hgrad)，S = −I（emitter 行对 H_i），A21 = −A12^T 的质量侧，
  P_e = −I（质量行对 qE）。
支持 θ 及 ∂F/∂θ：demand（质量行 −I）、Ke（emitter 行 |qE|^{Qexp−1}·qE，钳位支为 0）、
水库水头（链路行经 A10：n1 定水头 −1 / n2 定水头 +1）、管道 r_hw（链路行
sign(Q)|Q|^Hexp，RQtol 钳位支为 0；阀/关闭支为 0）。

-------------------------------------------------------------------------------
**路线一（solve_unrolled）的 demand 梯度支持声明（P4 / 审计 R4，2026-08-22）**

展开路径给的是"截断 K 步的梯度"，不是不动点梯度。在**一部分网上它随 K 不收敛**，
此时它既不等于隐式伴随、重跑也不稳定 - 这条**不是**稀疏/cuDSS 通路引入的，
两条线代通路都一样（P3 对抗审计 §1.5 在 GPU 上量过，本机 CPU 复核见
scripts/p4_r4_probe.py 与 data/p4_guards_wip.txt）。分三档，判据是**实测的**：

  · **不建议（等同不支持）**：rel(K+5→K+10) ≥ 1e-5 或 vs 隐式伴随 ≥ 1e-3。
    实测命中 **City_D/city_d（rel(+5→+10)=1.000e+00，即 100% 变化；
    max|g_d|=1.36e+08）、ky4（1.082e-03）、Net3（1.818e-03）**。
    这三个网上请**改用 ImplicitGGASolve**（隐式伴随，见路线二）。
  · 有条件可用：rel(K+5→K+10) < 1e-5 且 vs 隐式 < 1e-3 - Modena / EXA6 /
    ky3 / ky5 / city_h。可当搜索方向用（一阶优化），**不能**当"可验证的导数"
    （别拿它去过 1e-6 的对拍门槛）。
  · 可用：rel(K+5→K+10) < 1e-8 且 vs 隐式 < 1e-6 - Hanoi / rand_main_0009。

**判定不能靠 κ(A)**：实测反例摆着 - EXA6 κ₂=1.013e+10、city_h κ₂=1.844e+10
两网都好到 1e-7，而 Net3 κ₂=1.594e+09、Nj 只有 92，坏到 2.175e-03。
所以审计 R4 原文里"κ≥7e8 的网"这个说法**不成立**，本轮予以订正：
每张网必须**现测**，接口是 `unrolled_grad_health()`（本模块，见其 docstring），
两次展开反传即可给出上面三档判定。root cause 也不是 RQtol 钳位支
（ky4 只有 1 条钳位支，却有 243 个不稳定坐标，且无一与钳位支相邻）。

B2 可微扩展（泵）：链路残差在泵管段用 pumpcoeff 的 hloss（hydcoeffs.c:673-791），
雅可比泵行 φ'_pump 取该 hloss 分支表达式的精确 ∂/∂Q（与求解器钳位语义自洽：
POWER_FUNC 各支 = 求解器 hgrad；CONST_HP 非钳位支 = 求解器 hgrad = −r/Q²，
钳位支 hloss=−hgrad_c·Q 的精确导数 = −hgrad_c - EPANET 迭代取 +hgrad_c 是
半迭代技巧而非残差导数，隐函数微分必须对残差本身求导）。
∂F/∂θ 新增：转速 ω（LinkSetting）、泵曲线参数 H0 与 R - 解析式见
_pump_coeffs_np 注释（钳位/关闭支导数为 0）。状态冻结帧经 status（内部
StatusType 编码 int8[L]，非微分量）传入，closed = status<=CLOSED。
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch

try:
    from dgga.solver import (GGASolver, CSMALL, CBIG, TINY, QZERO, PI, MISSING,
                             _pow_crt, _log_crt, A1, A2, A8, A9, AB, AC)
except ImportError:  # pragma: no cover
    from solver import (GGASolver, CSMALL, CBIG, TINY, QZERO, PI, MISSING,
                        _pow_crt, _log_crt, A1, A2, A8, A9, AB, AC)

# EN_LinkType（epanet2_enums.h:183-191，与 solver 一致）
_CVPIPE, _PIPE, _PUMP, _PRV, _PSV, _PBV, _FCV, _TCV, _GPV = range(9)


# ======================================================================
# numpy 侧：残差 / 雅可比 / Newton 精抛光（forward no_grad 与 backward 共用）
# ======================================================================
def _resolve_pump_args(s: GGASolver, speed, status, h0p, rp):
    """θ/状态缺省解析：speed→LinkSetting 初值、status→初始内部状态、
    h0p/rp→泵曲线系数 pl_h0/pl_r。注意 init_status_int<=ST_CLOSED ⇔
    net.init_status==0 ⇔ s.closed_np（映射 {0:CLOSED,1:OPEN,2:ACTIVE}），
    故旧调用（全 None）的 closed 掩码逐位不变。"""
    speed = s.init_setting if speed is None else np.asarray(speed, dtype=np.float64)
    status = s.init_status_int if status is None \
        else np.asarray(status, dtype=np.int8)
    h0p = s.pl_h0 if h0p is None else np.asarray(h0p, dtype=np.float64)
    rp = s.pl_r if rp is None else np.asarray(rp, dtype=np.float64)
    return speed, status, h0p, rp


def _pump_coeffs_np(s: GGASolver, q, hl, hg, dr, speed, status, h0p, rp):
    """泵管段覆写 (hloss, hgrad=∂hloss/∂Q 精确值, ∂hloss/∂r_hw=0)，
    返回 (∂hloss/∂ω, ∂hloss/∂H0, ∂hloss/∂R) 各 [L]（非泵位=0）。

    公式逐支照抄 pumpcoeff（hydcoeffs.c:673-791）：
      关闭/ω=0（:696-701）：hloss=CBIG·Q, hgrad=CBIG（高阻管），θ 导数 0；
      NOCURVE（:709-714）：hloss=CSMALL·Q, hgrad=CSMALL，θ 导数 0；
      h0 = ω²·H0（:737）；|N−1|<TINY → n=1（:739）；r = R·ω^{2−n}（:740）；
      CONST_HP（:743-763）：hgrad=−r/|Q|²（:746）；
        >CBIG 钳位（:748-752）：hloss=−CBIG·Q → ∂/∂Q=−CBIG，θ 导数 0；
        <RQtol 钳位（:753-757）：hloss=−RQtol·Q → ∂/∂Q=−RQtol，θ 导数 0；
        否则（:759-762）：hloss=r/Q → ∂/∂Q=−r/Q²=hgrad，
          ∂/∂ω=(2−n)·R·ω^{1−n}/Q，∂/∂R=ω^{2−n}/Q，∂/∂H0=0（h0 不进此支）；
      POWER_FUNC n≠1（:767-779）：hgrad=n·r·|Q|^{n−1}（:770）；
        <RQtol 钳位（:772-776）：hloss=h0+RQtol·Q → ∂/∂Q=RQtol，
          ∂/∂ω=2ω·H0，∂/∂H0=ω²，∂/∂R=0；
        否则（:778）：hloss=h0+hgrad·Q/n=ω²H0+R·ω^{2−n}·|Q|^{n−1}·Q →
          ∂/∂Q=n·r·|Q|^{n−1}=hgrad（Q<0 时 d(|Q|^{n−1}Q)/dQ=n|Q|^{n−1} 同式），
          ∂/∂ω=2ω·H0+(2−n)·R·ω^{1−n}·|Q|^{n−1}·Q，
          ∂/∂H0=ω²，∂/∂R=ω^{2−n}·|Q|^{n−1}·Q；
      线性 n=1（:781-785）：hloss=h0+r·Q, hgrad=r=R·ω →
        ∂/∂ω=2ω·H0+R·Q，∂/∂H0=ω²，∂/∂R=ω·Q。
    """
    dw = np.zeros(s.L, dtype=np.float64)
    dh0 = np.zeros(s.L, dtype=np.float64)
    drp = np.zeros(s.L, dtype=np.float64)
    for k in s.pump_links:
        k = int(k)
        sp_ = float(speed[k])
        if int(status[k]) <= s.ST_CLOSED or sp_ == 0.0:   # hydcoeffs.c:696-701
            hl[k] = CBIG * q[k]
            hg[k] = CBIG
            dr[k] = 0.0
            continue
        if s.pl_ptype[k] == 3:                            # NOCURVE（:709-714）
            hl[k] = CSMALL * q[k]
            hg[k] = CSMALL
            dr[k] = 0.0
            continue
        qk = float(q[k])
        qa_k = abs(qk)                                    # :704 q = ABS(LinkFlow)
        if s.pl_ptype[k] == 2:                            # CUSTOM（hydcoeffs.c:716-733）
            # 转速折算流量所在线段局部线性模型（curvecoeff :794-831；:722）
            h0c, rc = s._curvecoeff(int(s.pl_pumpidx[k]), qa_k / sp_)
            H0c = -h0c                                    # :726
            Rc = -rc                                      # :727
            # φ'(Q) = R·ω（:731；段内常数 - 分段线性曲线的精确导数，段边界
            # 次梯度按 curvecoeff 所选段取值，与求解器钳位语义自洽）
            hg[k] = Rc * sp_
            hl[k] = H0c * (sp_ * sp_) + hg[k] * qk        # :732
            # ∂hloss/∂ω = 2ω·H0 + R·Q（段冻结；段选择对 ω 分段常数，同上）
            dw[k] = 2.0 * sp_ * H0c + Rc * qk
            # 曲线系数 θ（pl_h0/pl_r）不进 CUSTOM 支（每次迭代由曲线重算）→ 导数 0
            dr[k] = 0.0
            continue
        H0 = float(h0p[k])
        R = float(rp[k])
        h0 = sp_ * sp_ * H0                               # :737
        n = float(s.pl_n[k])
        if abs(n - 1.0) < TINY:
            n = 1.0                                       # :739
        r = R * _pow_crt(sp_, 2.0 - n)                    # :740
        if s.pl_ptype[k] == 0:                            # CONST_HP（:743-763）
            hgrad = -r / qa_k / qa_k if qa_k > 0.0 else np.inf   # :746
            if hgrad > CBIG:                              # :748-752
                hl[k] = -CBIG * qk
                hg[k] = -CBIG        # 残差支 −CBIG·Q 的精确导数（见模块头注释）
            elif hgrad < s.rqtol:                         # :753-757
                hl[k] = -s.rqtol * qk
                hg[k] = -s.rqtol
            else:                                         # :759-762
                hl[k] = r / qk
                hg[k] = hgrad
                dw[k] = (2.0 - n) * R * _pow_crt(sp_, 1.0 - n) / qk
                drp[k] = _pow_crt(sp_, 2.0 - n) / qk
        elif n != 1.0:                                    # 非线性曲线（:767-779）
            hgrad = n * r * _pow_crt(qa_k, n - 1.0)       # :770
            if hgrad < s.rqtol:                           # :772-776
                hl[k] = h0 + s.rqtol * qk
                hg[k] = s.rqtol
                dw[k] = 2.0 * sp_ * H0
                dh0[k] = sp_ * sp_
            else:                                         # :778
                hl[k] = h0 + hgrad * qk / n
                hg[k] = hgrad
                dw[k] = 2.0 * sp_ * H0 + (2.0 - n) * R \
                    * _pow_crt(sp_, 1.0 - n) * _pow_crt(qa_k, n - 1.0) * qk
                dh0[k] = sp_ * sp_
                drp[k] = _pow_crt(sp_, 2.0 - n) * _pow_crt(qa_k, n - 1.0) * qk
        else:                                             # 线性曲线（:781-785）
            hl[k] = h0 + r * qk
            hg[k] = r
            dw[k] = 2.0 * sp_ * H0 + R * qk
            dh0[k] = sp_ * sp_
            drp[k] = sp_ * qk
        dr[k] = 0.0
    return dw, dh0, drp


def _valve_act_masks(s: GGASolver, speed, status):
    """PRV/PSV ACTIVE 掩码（setting!=MISSING 且状态 ACTIVE，梯队3 审计修复）。

    ACTIVE PRV/PSV 的能量行不是 hloss=dh，而是罚函数强制的水头约束
    （prvcoeff hydcoeffs.c:965-981：H[n2]=hset；psvcoeff :1018-1032：H[n1]=hset），
    其不动点 = 约束行 H−hset=0 + 两端正常质量平衡（推导见交付报告）。
    被约束端为定水头节点时约束行无未知量（EPANET 走 badvalve 病态修复），拒绝。"""
    act = (np.asarray(status) == s.ST_ACTIVE) & (np.asarray(speed) != MISSING)
    act_prv = act & (s.lt_np == _PRV)
    act_psv = act & (s.lt_np == _PSV)
    act_fcv = act & (s.lt_np == _FCV)
    if act_prv.any() and s.is_fixed_node[s.n2_np[act_prv]].any():
        raise NotImplementedError("ACTIVE PRV 下游为定水头节点（badvalve 病态）")
    if act_psv.any() and s.is_fixed_node[s.n1_np[act_psv]].any():
        raise NotImplementedError("ACTIVE PSV 上游为定水头节点（badvalve 病态）")
    return act_prv, act_psv, act_fcv


def _link_coeffs_np(s: GGASolver, q, r_hw, speed=None, status=None,
                    h0p=None, rp=None):
    """逐链路 (hloss, hgrad, ∂hloss/∂r_hw, 泵θ导数或 None)。公式支路与
    solver._PY_np 同源：管道 hydcoeffs.c:545-574、TCV :912-939/:1100-1151、
    关闭支 :531-536/:1118-1123、泵 :673-791（_pump_coeffs_np）。
    speed/status/h0p/rp=None 时取初始状态（无泵网行为逐位不变）。
    """
    speed, status, h0p, rp = _resolve_pump_args(s, speed, status, h0p, rp)
    qa = np.abs(q)
    sgn = np.where(q < 0, -1.0, 1.0)                     # SGN（types.h:107）
    if getattr(s, "headloss_form", "H-W") == "D-W":
        # D-W（DWpipecoeff hydcoeffs.c:578-620；hgrad 即精确 ∂hloss/∂Q，
        # :614 已含 dfdq 项）。∂hloss/∂R：层流 hloss=Q·(16πsR+ml·q) → 16πs·Q；
        # 紊流 hloss=(fR+ml)·q·Q → f·q·Q（f 只依赖 q，与 R 无关）。
        # 非管道位初值给 RQtol 线性支（与 H-W 路径 r=0 时的钳位行为一致，
        # 防雅可比奇异；TCV/关闭/泵支随后统一覆盖）
        hg = np.full(s.L, s.rqtol, dtype=np.float64)
        hl = s.rqtol * q
        dr = np.zeros(s.L, dtype=np.float64)
        for k in range(s.L):
            if s.lt_np[k] > 1:                           # 仅 CVPIPE/PIPE
                continue
            Qk = float(q[k])
            qk = abs(Qk)
            R = float(r_hw[k])
            ml = float(s.km_pipe_np[k])
            e = float(s.kc_np[k]) / float(s.diam_np[k])  # :594
            sv = s.viscos * float(s.diam_np[k])          # :595
            if qk <= A2 * sv:                            # :600 层流
                r16 = 16.0 * PI * sv                     # :602 的 16πs 因子
                hl[k] = Qk * (r16 * R + ml * qk)         # :603
                hg[k] = r16 * R + 2.0 * ml * qk          # :604
                dr[k] = r16 * Qk
            else:
                w = qk / sv                              # :640
                if w >= A1:                              # :644 Swamee-Jain
                    y1 = A8 / _pow_crt(w, 0.9)
                    y2 = e / 3.7 + y1
                    y3 = A9 * _log_crt(y2)
                    f = 1.0 / (y3 * y3)
                    dfdq = 1.8 * f * y1 * A9 / y2 / y3 / qk   # :650
                else:                                    # :655-668 Dunlop
                    y2 = e / 3.7 + AB
                    y3 = A9 * _log_crt(y2)
                    fa = 1.0 / (y3 * y3)
                    fb = (2.0 + AC / (y2 * y3)) * fa
                    r2 = w / A2
                    x1 = 7.0 * fa - fb
                    x2 = 0.128 - 17.0 * fa + 2.5 * fb
                    x3 = -0.128 + 13.0 * fa - (fb + fb)
                    x4 = 0.032 - 3.0 * fa + 0.5 * fb
                    f = x1 + r2 * (x2 + r2 * (x3 + r2 * x4))
                    dfdq = (x2 + r2 * (2.0 * x3 + r2 * 3.0 * x4)) / sv / A2
                r1 = f * R + ml                          # :612
                hl[k] = r1 * qk * Qk                     # :613
                hg[k] = (2.0 * r1 * qk) + (dfdq * R * qk * qk)   # :614
                dr[k] = f * qk * Qk
    else:
        # 管道：hgrad = Hexp*R*|q|^(Hexp-1)（:550），RQtol 钳位（:554-558）
        hg = s.hexp * r_hw * qa ** (s.hexp - 1.0)
        lin = hg < s.rqtol
        hg = np.where(lin, s.rqtol, hg)
        hl = np.where(lin, hg * qa, hg * qa / s.hexp)    # :557,:560
        # ∂hloss/∂r：非钳位支 hloss=sign·r·|q|^Hexp → sign(q)|q|^Hexp；钳位支与 r 无关 → 0
        dr = np.where(lin, 0.0, qa ** s.hexp)
        mlp = s.km_pipe_np > 0.0                         # 局损项（:563-567）
        hl = np.where(mlp, hl + s.km_pipe_np * qa * qa, hl)
        hg = np.where(mlp, hg + 2.0 * s.km_pipe_np * qa, hg)
        hl = hl * sgn                                    # :570
        dr = dr * sgn
    # TCV（Km>0：hgrad=2Km|q|（:1129）＋RQtol 钳位；Km=0：P=1/CSMALL,Y=q（:1146-1150）
    # ⇔ hloss=CSMALL·q, hgrad=CSMALL）
    km = s.km_tcv_np
    hgv = 2.0 * km * qa
    linv = hgv < s.rqtol
    hgv = np.where(linv, s.rqtol, hgv)
    hlv = np.where(linv, q * hgv, q * hgv / 2.0)         # :1132-1137
    kmp = km > 0.0
    hgv = np.where(kmp, hgv, CSMALL)
    hlv = np.where(kmp, hlv, CSMALL * q)
    tcv = s.is_tcv_np
    hl = np.where(tcv, hlv, hl)
    hg = np.where(tcv, hgv, hg)
    dr = np.where(tcv, 0.0, dr)
    # 非 TCV 阀 PRV/PSV/FCV（梯队3 审计修复；PBV/GPV 构造期已拒绝）：
    # 非 ACTIVE（含 setting=MISSING 固定阀与 XPRESSURE/XFCV）走 valvecoeff 开启支
    # （hydcoeffs.c:1126-1150：Km>0 → hgrad=2Km|q| + RQtol 钳位，Km=0 →
    # hloss=CSMALL·q）；关闭支由下方统一 CBIG 覆盖（:1118-1123 同式）。
    vmask = s.is_valve_np & ~s.is_tcv_np
    if vmask.any():
        act_prv, act_psv, act_fcv = _valve_act_masks(s, speed, status)
        kmv = s.km_valve_ml_np
        hgv2 = 2.0 * kmv * qa                            # :1129
        linv2 = hgv2 < s.rqtol
        hgv2 = np.where(linv2, s.rqtol, hgv2)            # :1132-1136
        hlv2 = np.where(linv2, q * hgv2, q * hgv2 / 2.0)  # :1135,:1137
        kmp2 = kmv > 0.0
        hgv2 = np.where(kmp2, hgv2, CSMALL)              # :1146-1150
        hlv2 = np.where(kmp2, hlv2, CSMALL * q)
        hl = np.where(vmask, hlv2, hl)
        hg = np.where(vmask, hgv2, hg)
        dr = np.where(vmask, 0.0, dr)
        # ACTIVE FCV：fcvcoeff 切断式（:1073-1084）的不动点
        # Q = qset + dh/CBIG ⇔ 残差 CBIG·(Q−qset) − dh = 0
        if act_fcv.any():
            hl = np.where(act_fcv, CBIG * (q - speed), hl)
            hg = np.where(act_fcv, CBIG, hg)
        # ACTIVE PRV/PSV：能量行改水头约束行（hl/hg 置 0，
        # 行内容由 _residual_np/_build_J 接管）
        actp = act_prv | act_psv
        if actp.any():
            hl = np.where(actp, 0.0, hl)
            hg = np.where(actp, 0.0, hg)
    # 关闭支：P=1/CBIG, Y=q ⇔ hloss=CBIG·q, hgrad=CBIG
    # （closed = status<=ST_CLOSED；旧调用逐位等于 s.closed_np，见 _resolve_pump_args）
    cl = status <= s.ST_CLOSED
    hl = np.where(cl, CBIG * q, hl)
    hg = np.where(cl, CBIG, hg)
    dr = np.where(cl, 0.0, dr)
    # 泵覆写（hydcoeffs.c:673-791）＋ θ 导数
    pd = None
    if s.n_pumps:
        pd = _pump_coeffs_np(s, q, hl, hg, dr, speed, status, h0p, rp)
    return hl, hg, dr, pd


def _emitter_coeffs_np(s: GGASolver, e, ke):
    """emitter (hloss, hgrad, ∂hloss/∂Ke)，[Nj] 全 junction 向量（调用方掩码）。
    hydcoeffs.c:378-409：ke_adj=max(CSMALL,Ke)（:394），hgrad=Qexp·ke·|e|^(Qexp-1)
    （:398），RQtol 钳位（:401-405）。非钳位支 hloss=ke_adj·|e|^(Qexp-1)·e →
    ∂/∂Ke=|e|^(Qexp-1)·e（Ke<CSMALL 时被 max 钳死 → 0；RQtol 支与 Ke 无关 → 0）。"""
    ke_adj = np.maximum(CSMALL, ke)
    ea = np.abs(e)
    hg = s.qexp * ke_adj * ea ** (s.qexp - 1.0)
    lin = hg < s.rqtol
    hg = np.where(lin, s.rqtol, hg)
    hl = np.where(lin, hg * e, hg * e / s.qexp)
    dke = np.where(lin | (ke < CSMALL), 0.0, ea ** (s.qexp - 1.0) * e)
    return hl, hg, dke


def _adj_cache(s: GGASolver):
    """节点关联索引缓存（挂在 solver 实例上）。"""
    c = getattr(s, "_ad_cache", None)
    if c is None:
        junc_row = np.full(s.N, -1, dtype=np.int64)
        junc_row[s.junc_nodes] = np.arange(s.Nj)
        m1 = junc_row[s.n1_np] >= 0                      # n1 端为 junction
        m2 = junc_row[s.n2_np] >= 0
        c = dict(junc_row=junc_row, m1=m1, m2=m2,
                 j1=junc_row[s.n1_np], j2=junc_row[s.n2_np],
                 f1=~m1, f2=~m2)                         # 定水头端掩码
        s._ad_cache = c
    return c


def _residual_np(s, q, e_j, Hj, d, rh, ke_j, r_hw, speed=None, status=None,
                 h0p=None, rp=None):
    """完整残差 F(z;θ) = (r_link[L], r_em[ne], m[Nj])；ne = Σ(ke_j>0)。

    梯队3 审计修复：ACTIVE PRV 的能量行改约束行 r=H[n2]−hset（prvcoeff
    :965-981 罚函数不动点；hset=El[n2]+setting，:962-963）；ACTIVE PSV 改
    r=hset−H[n1]（psvcoeff :1015-1032；符号取 −(H1−·) 以复用雅可比 −1@j1）。"""
    c = _adj_cache(s)
    speed, status, h0p, rp = _resolve_pump_args(s, speed, status, h0p, rp)
    hl, hg, dr, _ = _link_coeffs_np(s, q, r_hw, speed=speed, status=status,
                                    h0p=h0p, rp=rp)
    Hfull = np.empty(s.N, dtype=np.float64)
    Hfull[s.junc_nodes] = Hj
    Hfull[s.fixed_nodes] = rh[s.fixed_nodes]
    r_link = hl - (Hfull[s.n1_np] - Hfull[s.n2_np])
    act_prv = act_psv = None
    if bool((s.is_valve_np & ~s.is_tcv_np).any()):
        act_prv, act_psv, _af = _valve_act_masks(s, speed, status)
        for k in np.where(act_prv)[0]:
            n2 = int(s.n2_np[k])                         # hset：prvcoeff:962-963
            r_link[k] = Hfull[n2] - (s.elev_np[n2] + speed[k])
        for k in np.where(act_psv)[0]:
            n1 = int(s.n1_np[k])                         # hset：psvcoeff:1015-1016
            r_link[k] = (s.elev_np[n1] + speed[k]) - Hfull[n1]
    em = ke_j > 0.0
    hle, hge, dke = _emitter_coeffs_np(s, e_j, ke_j)
    r_em = (hle - (Hj - s.el_junc_np))[em]
    m = np.zeros(s.Nj, dtype=np.float64)
    np.add.at(m, c["j2"][c["m2"]], q[c["m2"]])           # n2 端 +Q（hydcoeffs.c:226）
    np.add.at(m, c["j1"][c["m1"]], -q[c["m1"]])          # n1 端 −Q（:225）
    m -= np.where(em, e_j, 0.0)                          # :373
    m -= d[s.junc_nodes]                                 # :268（DDA：DemandFlow=名义需水）
    return np.concatenate([r_link, r_em, m]), (hl, hg, dr, hle, hge, dke, em,
                                               act_prv, act_psv)


def _build_J(s, hg, hge, em, act_prv=None, act_psv=None):
    """稀疏雅可比 J（csc）。行/列均按 (Q[L], qE[ne], H[Nj]) 排布。

    梯队3：ACTIVE PRV 行 r=H[n2]−hset → ∂/∂Q=0（hg 已置 0）、∂/∂H[n1]=0
    （−1@j1 清零）、∂/∂H[n2]=+1（保留）；ACTIVE PSV 行 r=hset−H[n1] →
    ∂/∂H[n1]=−1（保留）、∂/∂H[n2]=0（+1@j2 清零）。"""
    c = _adj_cache(s)
    L, Nj = s.L, s.Nj
    em_rows = np.where(em)[0]
    ne = em_rows.size
    off_e, off_h = L, L + ne
    rows, cols, vals = [], [], []
    ar_L = np.arange(L)
    # 链路行：D = diag(hgrad)；−A12：∂r_k/∂H_{n1}=−1、∂r_k/∂H_{n2}=+1（junction 端）
    rows.append(ar_L); cols.append(ar_L); vals.append(hg)
    k1 = ar_L[c["m1"]]
    v1 = np.full(k1.size, -1.0)
    if act_prv is not None and act_prv.any():
        v1[act_prv[c["m1"]]] = 0.0                       # PRV 约束行不含 H[n1]
    rows.append(k1); cols.append(off_h + c["j1"][c["m1"]]); vals.append(v1)
    k2 = ar_L[c["m2"]]
    v2 = np.full(k2.size, 1.0)
    if act_psv is not None and act_psv.any():
        v2[act_psv[c["m2"]]] = 0.0                       # PSV 约束行不含 H[n2]
    rows.append(k2); cols.append(off_h + c["j2"][c["m2"]]); vals.append(v2)
    if ne:
        ar_e = np.arange(ne)
        # emitter 行：E' = diag(hgrad_e)；S：∂r_e/∂H_i = −1
        rows.append(off_e + ar_e); cols.append(off_e + ar_e); vals.append(hge[em_rows])
        rows.append(off_e + ar_e); cols.append(off_h + em_rows); vals.append(np.full(ne, -1.0))
        # 质量行对 qE：P_e = −I
        rows.append(off_h + em_rows); cols.append(off_e + ar_e); vals.append(np.full(ne, -1.0))
    # 质量行对 Q：A21（n1 端 −1、n2 端 +1，与 Xflow 符号一致）
    rows.append(off_h + c["j1"][c["m1"]]); cols.append(k1); vals.append(np.full(k1.size, -1.0))
    rows.append(off_h + c["j2"][c["m2"]]); cols.append(k2); vals.append(np.full(k2.size, 1.0))
    M = L + ne + Nj
    J = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(M, M)).tocsc()
    return J, em_rows


def _polish_np(s, d, rh, ke, r_hw, q, e_j, Hj, steps=3, speed=None, status=None,
               h0p=None, rp=None):
    """对 F(z;θ)=0 做整装 Newton：J·Δz = −F（scipy splu）。从 GGA 收敛态出发
    2-3 步即到 ‖F‖∞ ~ 机器精度级（Newton 二阶收敛；J 为 F 的精确分支雅可比）。
    返回 (q, e_j, Hj, ‖F‖∞_final)。"""
    ke_j = ke[s.junc_nodes]
    q = q.copy(); e_j = e_j.copy(); Hj = Hj.copy()
    pk = dict(speed=speed, status=status, h0p=h0p, rp=rp)
    for _ in range(int(steps)):
        F, (hl, hg, dr, hle, hge, dke, em, aprv, apsv) = _residual_np(
            s, q, e_j, Hj, d, rh, ke_j, r_hw, **pk)
        J, em_rows = _build_J(s, hg, hge, em, aprv, apsv)
        dz = splu(J).solve(-F)
        L, ne = s.L, em_rows.size
        q += dz[:L]
        e_j[em_rows] += dz[L:L + ne]
        Hj += dz[L + ne:]
    F, _ = _residual_np(s, q, e_j, Hj, d, rh, ke_j, r_hw, **pk)
    return q, e_j, Hj, float(np.abs(F).max())


class _override_r_hw:
    """临时替换 solver 的 r_hw（dense 路径 _pipe_PY 读张量）与 r_np
    （epanet 路径 _PY_np 读 numpy 版）。"""

    def __init__(self, s, r_hw_t):
        self.s = s
        self.r = r_hw_t

    def __enter__(self):
        self.old = self.s.r_hw
        self.old_np = self.s.r_np
        if self.r is not None:
            self.s.r_hw = self.r
            self.s.r_np = self.r.detach().cpu().numpy()
        return self.s

    def __exit__(self, *a):
        self.s.r_hw = self.old
        self.s.r_np = self.old_np
        return False


class _override_pump:
    """临时替换 solver.pl_h0 / pl_r（epanet 路径 _PY_np 泵支读这两个数组）。"""

    def __init__(self, s, h0p, rp):
        self.s = s
        self.h0p = h0p
        self.rp = rp

    def __enter__(self):
        self.old_h0 = self.s.pl_h0
        self.old_r = self.s.pl_r
        if self.h0p is not None:
            self.s.pl_h0 = np.asarray(self.h0p, dtype=np.float64)
        if self.rp is not None:
            self.s.pl_r = np.asarray(self.rp, dtype=np.float64)
        return self.s

    def __exit__(self, *a):
        self.s.pl_h0 = self.old_h0
        self.s.pl_r = self.old_r
        return False


def solve_polished(s: GGASolver, d, rh, ke=None, r_hw=None,
                   accuracy=1e-12, max_iter=200, polish_steps=3,
                   speed=None, status=None, pump_h0=None, pump_r=None,
                   assemble="dense", linear_solver="dense"):
    """GGA 收敛 + 整装 Newton 精抛光（numpy 出入）。d/rh/ke [N] 或 [B,N]，
    r_hw [L]（None=solver 自带）。B2 泵扩展（单配置、批共享）：speed=LinkSetting
    [L]、status=内部状态 int8[L]（状态冻结帧）、pump_h0/pump_r=曲线系数 [L]；
    含泵/水池网须用 mode='epanet' 的 solver（dense 前向不支持泵，构造时已拒绝）。
    返回 dict(q,e_j,Hj,head,flow,emitter,resid_inf,+已解析的 speed/status/h0p/rp)。

    assemble / linear_solver（sparse_gpu_plan.md §1a/§1b）直通给 s.solve()：
    缺省 "dense"/"dense" 与历史逐位一致；"csr"/"cudss" 只对 mode='dense'
    的 GPU f64 solver 有效。本函数本来就跑在 torch.no_grad() 下（后续
    抛光是 numpy 的），所以 cudss 通路在这里不会撞断图守卫。"""
    d = np.atleast_2d(np.asarray(d, dtype=np.float64))
    rh = np.atleast_2d(np.asarray(rh, dtype=np.float64))
    B = max(d.shape[0], rh.shape[0])
    if ke is None:
        ke = s.node_ke_default.cpu().numpy()
    ke = np.atleast_2d(np.asarray(ke, dtype=np.float64))
    d = np.broadcast_to(d, (B, s.N))
    rh = np.broadcast_to(rh, (B, s.N))
    ke = np.broadcast_to(ke, (B, s.N))
    r_np = s.r_hw.detach().cpu().numpy() if r_hw is None \
        else np.asarray(r_hw, dtype=np.float64)
    rt = None if r_hw is None else torch.as_tensor(r_np, dtype=s.dtype, device=s.device)
    sp_np = None if speed is None else np.asarray(speed, dtype=np.float64)
    st_np = None if status is None else np.asarray(status, dtype=np.int8)
    kw = {}
    if st_np is not None:
        kw["link_status"] = st_np
    if sp_np is not None:
        kw["link_setting"] = sp_np
    if assemble != "dense" or linear_solver != "dense":
        kw["assemble"] = assemble
        kw["linear_solver"] = linear_solver
    with torch.no_grad(), _override_r_hw(s, rt), _override_pump(s, pump_h0, pump_r):
        out = s.solve(d, rh, ke_int=ke, max_iter=max_iter, accuracy=accuracy, **kw)
    q_all = out["flow_cfs"].cpu().numpy().copy()
    e_all = out["emitter_cfs"].cpu().numpy()[:, s.junc_nodes].copy()
    H_all = out["head_ft"].cpu().numpy().copy()
    if q_all.ndim == 1:                     # epanet 模式 B=1 会去批维，补回
        q_all = q_all[None, :].copy()
        e_all = e_all[None, :].copy() if e_all.ndim == 1 else e_all
        H_all = H_all[None, :].copy()
    sp_res, st_res, h0_res, rp_res = _resolve_pump_args(s, sp_np, st_np,
                                                        pump_h0, pump_r)
    resid = np.zeros(B)
    for b in range(B):
        qb, eb, Hjb, rb = _polish_np(s, d[b], rh[b], ke[b], r_np,
                                     q_all[b], e_all[b], H_all[b, s.junc_nodes],
                                     steps=polish_steps, speed=sp_res,
                                     status=st_res, h0p=h0_res, rp=rp_res)
        q_all[b] = qb
        e_all[b] = eb
        H_all[b, s.junc_nodes] = Hjb
        resid[b] = rb
    emitter = np.zeros((B, s.N))
    emitter[:, s.junc_nodes] = np.where(ke[:, s.junc_nodes] > 0, e_all, 0.0)
    return dict(q=q_all, e_j=e_all, head=H_all, emitter=emitter,
                d=d, rh=rh, ke=ke, r_hw=r_np, resid_inf=resid,
                speed=sp_res, status=st_res, h0p=h0_res, rp=rp_res,
                iters=np.atleast_1d(out["iters"].cpu().numpy()))


# ======================================================================
# 路线二：隐函数定理伴随法
# ======================================================================
class ImplicitGGASolve(torch.autograd.Function):
    """(demand[N|B,N], res_head[N|B,N], ke[N|B,N], r_hw[L], solver, accuracy,
    max_iter, polish_steps[, speed[L], pump_h0[L], pump_r[L], status])
    -> (head[B,N], flow[B,L], emitter[B,N])。

    forward：no_grad 收敛（默认显式 accuracy=1e-12、max_iter=200）+ Newton 精抛光。
    backward：解 J^T λ = ∂L/∂z（scipy splu trans='T'），再按 ∂F/∂θ 收缩：
      dL/dθ = −λ^T ∂F/∂θ（+ 输出直通项：head 的定水头位恒等于 res_head）。
    B2 泵扩展：speed（转速 ω=LinkSetting）、pump_h0/pump_r（曲线系数）为可微
    张量 [L]（批共享；泵位以外梯度恒 0）；status 为冻结内部状态 int8[L]
    （非微分量，numpy/张量均可）。旧 8 参调用行为不变（autograd 允许 backward
    对未传入的可选输入返回多余的 None）。
    """

    @staticmethod
    def forward(ctx, demand, res_head, ke, r_hw, solver,
                accuracy=1e-12, max_iter=200, polish_steps=3,
                speed=None, pump_h0=None, pump_r=None, status=None):
        dims = (demand.dim(), res_head.dim(), ke.dim(), r_hw.dim())
        as_np = lambda x: None if x is None else \
            (x.detach().cpu().numpy() if isinstance(x, torch.Tensor)
             else np.asarray(x))
        sol = solve_polished(
            solver,
            demand.detach().cpu().numpy(), res_head.detach().cpu().numpy(),
            ke.detach().cpu().numpy(), r_hw.detach().cpu().numpy(),
            accuracy=accuracy, max_iter=max_iter, polish_steps=polish_steps,
            speed=as_np(speed), status=as_np(status),
            pump_h0=as_np(pump_h0), pump_r=as_np(pump_r))
        ctx.solver = solver
        ctx.sol = sol
        ctx.dims = dims
        ctx.pump_in = (isinstance(speed, torch.Tensor),
                       isinstance(pump_h0, torch.Tensor),
                       isinstance(pump_r, torch.Tensor))
        # autograd 引擎要求返回的梯度与各输入同 device（solver 在 GPU 而输入在
        # CPU 时不能一律用 solver.device），逐输入记录
        ctx.in_devs = (demand.device, res_head.device, ke.device, r_hw.device,
                       speed.device if ctx.pump_in[0] else None,
                       pump_h0.device if ctx.pump_in[1] else None,
                       pump_r.device if ctx.pump_in[2] else None)
        dt, dev = solver.dtype, solver.device
        head = torch.as_tensor(sol["head"], dtype=dt, device=dev)
        flow = torch.as_tensor(sol["q"], dtype=dt, device=dev)
        emitter = torch.as_tensor(sol["emitter"], dtype=dt, device=dev)
        if all(x == 1 for x in dims[:3]):
            head, flow, emitter = head[0], flow[0], emitter[0]
        return head, flow, emitter

    @staticmethod
    def backward(ctx, gH, gQ, gE):
        s = ctx.solver
        sol = ctx.sol
        c = _adj_cache(s)
        B = sol["q"].shape[0]
        gH = np.atleast_2d(gH.detach().cpu().numpy())
        gQ = np.atleast_2d(gQ.detach().cpu().numpy())
        gE = np.atleast_2d(gE.detach().cpu().numpy())
        gd = np.zeros((B, s.N))
        grh = np.zeros((B, s.N))
        gke = np.zeros((B, s.N))
        gr = np.zeros((B, s.L))
        gsp = np.zeros(s.L)
        gh0 = np.zeros(s.L)
        grp = np.zeros(s.L)
        L = s.L
        for b in range(B):
            q, e_j, Hj = sol["q"][b], sol["e_j"][b], sol["head"][b, s.junc_nodes]
            ke_j = sol["ke"][b, s.junc_nodes]
            _, hg, dr, pd = _link_coeffs_np(s, q, sol["r_hw"],
                                            speed=sol["speed"],
                                            status=sol["status"],
                                            h0p=sol["h0p"], rp=sol["rp"])
            _, hge, dke = _emitter_coeffs_np(s, e_j, ke_j)
            em = ke_j > 0.0
            aprv = apsv = None
            if bool((s.is_valve_np & ~s.is_tcv_np).any()):
                aprv, apsv, _af = _valve_act_masks(s, sol["speed"], sol["status"])
            J, em_rows = _build_J(s, hg, hge, em, aprv, apsv)
            ne = em_rows.size
            v = np.concatenate([gQ[b],
                                gE[b, s.junc_nodes][em_rows],
                                gH[b, s.junc_nodes]])
            lam = splu(J).solve(v, trans="T")            # J^T λ = ∂L/∂z
            lam_l, lam_e, lam_m = lam[:L], lam[L:L + ne], lam[L + ne:]
            # demand：∂m_i/∂d_i = −1 → dL/dd = +λ_m
            gd[b, s.junc_nodes] = lam_m
            # Ke：∂r_e/∂Ke = dke → dL/dKe = −λ_e·dke
            gke[b, s.junc_nodes[em_rows]] = -lam_e * dke[em_rows]
            # 水库水头（A10）：∂r_k/∂H_f：n1 定水头 −1 / n2 定水头 +1
            #  → dL/dH_f = +Σ_{n1=f}λ_l − Σ_{n2=f}λ_l；再加 head 输出直通 gH
            # 梯队3：ACTIVE PRV 行不含 H[n1]（约束行）→ n1 为定水头时该行
            # 对 rh 无导数，从散射中剔除（PSV 行对 H[n2] 同理；ACTIVE 被约束端
            # 为定水头已在 _valve_act_masks 拒绝）
            f1_eff, f2_eff = c["f1"], c["f2"]
            if aprv is not None and aprv.any():
                f1_eff = f1_eff & ~aprv
            if apsv is not None and apsv.any():
                f2_eff = f2_eff & ~apsv
            np.add.at(grh[b], s.n1_np[f1_eff], lam_l[f1_eff])
            np.add.at(grh[b], s.n2_np[f2_eff], -lam_l[f2_eff])
            grh[b, s.fixed_nodes] += gH[b, s.fixed_nodes]
            # r_hw：∂r_k/∂r = dr → dL/dr = −λ_l·dr
            gr[b] = -lam_l * dr
            # 泵 θ（批共享 → 逐样本累加）：dL/dθ = −Σ_b λ_l·∂φ/∂θ
            if pd is not None:
                dw, dh0_, drp_ = pd
                gsp += -lam_l * dw
                gh0 += -lam_l * dh0_
                grp += -lam_l * drp_
        dt = s.dtype

        def _fit(g, dim, dev):
            t = torch.as_tensor(g, dtype=dt, device=dev)
            return t.sum(0) if dim == 1 else t

        dv = ctx.in_devs
        pin = ctx.pump_in
        g_sp = torch.as_tensor(gsp, dtype=dt, device=dv[4]) if pin[0] else None
        g_h0 = torch.as_tensor(gh0, dtype=dt, device=dv[5]) if pin[1] else None
        g_rp = torch.as_tensor(grp, dtype=dt, device=dv[6]) if pin[2] else None
        return (_fit(gd, ctx.dims[0], dv[0]), _fit(grh, ctx.dims[1], dv[1]),
                _fit(gke, ctx.dims[2], dv[2]), _fit(gr, ctx.dims[3] + 1, dv[3]),
                None, None, None, None, g_sp, g_h0, g_rp, None)


# ======================================================================
# 路线二'：隐式伴随的 GPU 批量约化（adjoint="gpu"；CPU 伴随缺省逐位不动）
# ======================================================================
# 数学（数值证明见 scripts/adjoint_gpu/probe_math.py，L-TOWN 实测）：
# J^T λ = (gQ, gE, gH) 对链路/emitter 行消元后，节点方程的系数矩阵恰是
# 收敛态的 GGA 矩阵 Ā（D̄^{-1}=P 逐支同装配值 + em/hge 对角）⇒ 前向最后一轮
# 的分解（dense Cholesky / cuDSS）可直接复用，伴随只剩三角回代 + SpMV：
#   λ_l = P·(gQ − (λ_m[j2]−λ_m[j1]))，λ_e = (gE + λ_m)/hge，
#   Ā λ_m = B^T P gQ − em·gE/hge − gH。
# ACTIVE PRV（真雅可比 = 约束行 H[n2]−hset，D_kk=0）：Q_prv 列给出约束
#   λ_m[j2] − λ_m[j1] = gQ[prv]；其 λ_l 是自由量且不进任何 θ 梯度
#   （dr=0、无 rh 耦合、P=0 ⇒ 散射自动为 0）。前向分解的是 big-M 的
#   Â = Ā + CBIG·e_{j2}e_{j2}^T - **naive 复用是错的**（λ_m[j2]≈r/CBIG~1e-8，
#   L-TOWN 实测 PRV 下游 demand 梯度差 O(1)：238.297 vs 8.9e-8）；
#   正确做法 = Woodbury 行替换：M = Â + Σ e_{j2}(m_a−â_a)^T，
#   M^{-1}r = Â^{-1}r − S C^{-1}(x0[j2]−x0[j1]−r[j2])，S=Â^{-1}[e_{j2}]，
#   C_ab=(s_b)_{j2a}−(s_b)_{j1a}，再对真 M 做外层迭代精化（κ~1.6e11 护栏）。
#   实测 woodbury 后四类 θ 与 splu(J^T) 参考 rel ≤ 1.4e-10。
# 已知不可约化支（检测到即 raise，不静默）：CONST_HP 泵的 CBIG/RQtol 钳位支
#   （残差导数 −CBIG/−RQtol，装配用 +CBIG/+RQtol，符号翻转）；D-W（∂hloss/∂r
#   未在本路径实现）。二阶导（create_graph）同 _CudssSolveFn 明确拒绝。
def _adjgpu_cache(s):
    """GPU 约化伴随的静态索引缓存（挂 solver 实例；与 _adj_cache 独立）。"""
    c = getattr(s, "_adjgpu_idx", None)
    if c is None:
        dev = s.device
        junc_row = np.full(s.N, -1, dtype=np.int64)
        junc_row[s.junc_nodes] = np.arange(s.Nj)
        m1 = junc_row[s.n1_np] >= 0
        m2 = junc_row[s.n2_np] >= 0
        ti = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=dev)
        fixmask = np.zeros(s.N, dtype=bool)
        fixmask[s.fixed_nodes] = True
        c = dict(
            m1f=torch.as_tensor(m1, device=dev).to(s.dtype),
            m2f=torch.as_tensor(m2, device=dev).to(s.dtype),
            j1c=ti(np.where(m1, junc_row[s.n1_np], 0)),   # 定水头端取槽 0，
            j2c=ti(np.where(m2, junc_row[s.n2_np], 0)),   # 值由 m1f/m2f 清零
            lk_m1=ti(np.where(m1)[0]), lk_m2=ti(np.where(m2)[0]),
            f_idx1=ti(junc_row[s.n1_np][m1]),
            f_idx2=ti(junc_row[s.n2_np][m2]),
            # res_head 散射（含两端皆定水头的链路；CPU 伴随 f1/f2 同口径）
            lk_f1=ti(np.where(~m1)[0]), f1_nodes=ti(s.n1_np[~m1]),
            lk_f2=ti(np.where(~m2)[0]), f2_nodes=ti(s.n2_np[~m2]),
            is_pipe=torch.as_tensor(s.lt_np <= 1, device=dev),
            fixmask=torch.as_tensor(fixmask, device=dev),
        )
        s._adjgpu_idx = c
    return c


def _adjgpu_coeffs(s, c, q, e_j, ke_j, S):
    """当前状态的伴随/精抛光系数（[B,·] 全向量，无逐场景循环）：
    P_eff、Y_eff（逐位复刻 solve() 装配语义：pcv/关闭 1/CBIG/泵/TCV/固定阀）、
    emh=em/hge、hge、em、dr=∂hloss/∂r_hw、dke=∂r_e/∂Ke、act[B,p]（ACTIVE PRV）。
    D̄^{-1} ≡ P_eff：ACTIVE PRV 位 = 0 ⇒ 该行自动从消元与全部 θ 散射中排除；
    hloss = Y_eff/P_eff（关闭支 = CBIG·q、TCV Km=0 支 = CSMALL·q，逐支同
    _link_coeffs_np 的残差口径；ACTIVE PRV 行的残差由调用方另算 H[n2]−hset）。"""
    dt = s.dtype
    B = q.shape[0]
    if str(getattr(s, "headloss_form", "H-W")).upper() == "D-W":
        raise NotImplementedError(
            "adjoint='gpu' 未实现 Darcy-Weisbach 的 ∂hloss/∂r 收缩 - "
            "请改用缺省 CPU 伴随（ImplicitGGASolve）。")
    P, Y = s._pipe_PY(q)
    P_t, Y_t = s._tcv_PY(q)
    P = torch.where(s.is_tcv, P_t, P)
    Y = torch.where(s.is_tcv, Y_t, Y)
    closed_now = ((S <= s.ST_CLOSED) | s.sm_zero_speed_t) if S is not None \
        else s.closed_dense
    if s.n_pumps:
        P_p, Y_p = s._pump_PY(q)
        P = torch.where(s.is_pump_t, P_p, P)
        Y = torch.where(s.is_pump_t, Y_p, Y)
        # CONST_HP 钳位支：残差导数是 −CBIG/−RQtol（见 _pump_coeffs_np 注释），
        # 而装配 P 用 +CBIG/+RQtol ⇒ Schur 补 ≠ 真雅可比的约化，检测到即拒。
        if bool(s.pump_chp_t.any()):
            sgn = torch.where(q < 0, -1.0, 1.0)
            qa_c = (q * sgn).clamp_min(1e-30)
            re = s.pump_r_t * s.pump_speed_t.clamp_min(1e-12) \
                ** (2.0 - s.pump_n_t)
            hg_raw = -re / qa_c / qa_c
            bad = s.pump_chp_t & ~closed_now \
                & ((hg_raw > CBIG) | (hg_raw < s.rqtol))
            if bool(bad.any()):
                raise NotImplementedError(
                    "adjoint='gpu'：收敛态存在 CONST_HP 泵落在 CBIG/RQtol 钳位支"
                    "（残差导数与装配对角符号相反，约化不成立） - "
                    "请对该批改用 CPU 伴随。")
    if s._dense_prv_fixed_any:
        P_v, Y_v = s._valve_PY_ml(q)
        P = torch.where(s.prv_fixed_t, P_v, P)
        Y = torch.where(s.prv_fixed_t, Y_v, Y)
    if s._dense_prv_np:
        closed_now = closed_now & ~s.pcv_static_t
    P = torch.where(closed_now, torch.full_like(P, 1.0 / CBIG), P)
    Y = torch.where(closed_now, q, Y)
    act = None
    if s._dense_prv_np:
        if S is None:
            raise RuntimeError("含 pcv PRV 必然走状态机（solve 已守卫），"
                               "S 不应为 None")
        P = torch.where(s.pcv_static_t, torch.zeros_like(P), P)
        Y = torch.where(s.pcv_static_t, torch.zeros_like(Y), Y)
        Yd = torch.zeros_like(q)
        Fd = torch.zeros(B, s.Nj, dtype=dt, device=q.device)
        Xd = torch.zeros(B, s.Nj, dtype=dt, device=q.device)
        P2, Y2, _, _, _ = s._prvcoeffs_batch(P, Yd, Fd, Xd, q, S)
        # 非 ACTIVE pcv 的 P/Y 取 prvcoeff 值；ACTIVE 的 Y2=qk+Xflow（EPANET
        # 半迭代技巧）不进残差，P=0 掩掉即可
        P = P2
        Y = Y.index_copy(1, s.prv_k_t, Y2.index_select(1, s.prv_k_t))
        act = S.index_select(1, s.prv_k_t) == s.ST_ACTIVE      # [B,p]
        j2_np = s.prv_j2_t.cpu().numpy()
        j1_np = s.prv_j1_t.cpu().numpy()
        if np.unique(j2_np).size < j2_np.size:
            raise NotImplementedError(
                "adjoint='gpu'：多个 pcv PRV 共享同一下游 junction，"
                "约束行替换的散射会冲突 - 请改用 CPU 伴随。")
        if np.intersect1d(j2_np, j1_np).size:
            raise NotImplementedError(
                "adjoint='gpu'：存在级联 pcv PRV（某阀下游 = 另一阀上游 "
                "junction），行替换会叠在被替换行上 - 请改用 CPU 伴随。")
    # emitter（与 _emitter_coeffs_np 同支语义）
    _, hge = s._emitter_hloss(e_j, ke_j)
    em = (ke_j > 0.0).to(dt)
    emh = em / hge
    # dr = ∂hloss/∂r_hw（H-W/C-M 幂律；仅 CVPIPE/PIPE 非钳位支非关闭）
    sgn = torch.where(q < 0, -1.0, 1.0)
    qa = q * sgn
    hgr = s.hexp * s.r_hw * qa.clamp_min(1e-30) ** (s.hexp - 1.0)
    dr = torch.where(c["is_pipe"] & ~closed_now & (hgr >= s.rqtol),
                     qa ** s.hexp * sgn, torch.zeros_like(q))
    # dke = ∂r_e/∂Ke（非钳位支且 Ke≥CSMALL；|e|=0 时值为 0，clamp 只防 0^负）
    ea = torch.abs(e_j)
    hge_raw = s.qexp * torch.clamp(ke_j, min=CSMALL) \
        * ea.clamp_min(1e-30) ** (s.qexp - 1.0)
    dke = torch.where((hge_raw < s.rqtol) | (ke_j < CSMALL),
                      torch.zeros_like(e_j),
                      ea.clamp_min(1e-30) ** (s.qexp - 1.0) * e_j)
    return P, Y, emh, hge, em, dr, dke, act


def _adjgpu_matvec(s, c, P, emh, act, x):
    """Â·x（[B,Nj]）：Ā（P 装配语义）+ em/hge 对角 + ACTIVE PRV 下游 CBIG 对角。
    与前向 scatter 装配的矩阵动作一致（用于迭代精化残差，不落 [B,Nj,Nj]）。"""
    l1 = x.index_select(1, c["j1c"]) * c["m1f"]
    l2 = x.index_select(1, c["j2c"]) * c["m2f"]
    t = P * (l1 - l2)                                    # [B,L]
    B = x.shape[0]
    out = torch.zeros_like(x)
    out = out.scatter_add(1, c["f_idx1"].expand(B, -1),
                          t.index_select(1, c["lk_m1"]))
    out = out.scatter_add(1, c["f_idx2"].expand(B, -1),
                          -t.index_select(1, c["lk_m2"]))
    out = out + emh * x
    if act is not None:
        j2 = s.prv_j2_t
        xa = x.index_select(1, j2) * act.to(x.dtype) * CBIG
        out = out.scatter_add(1, j2.view(1, -1).expand(B, -1), xa)
    return out


def _kf_prepare(s, kf):
    """cudss：确认前向最后一轮的数值分解仍在（gen 未被顶掉、未被 LRU 逐出）；
    不在则按保存的 CSR 值重做一次（计入 bwd_refactorize，不静默）。dense：无操作。"""
    if kf["kind"] != "cudss":
        return
    st, gen = kf["st"], kf["gen"]
    if (st.get("solver") is not None and st.get("gen") == gen
            and int(st.get("B", -1)) == int(kf["B"])):
        s._cudss_counters["bwd_reuse"] += 1
        return
    st = s._cudss_state(kf["B"], kf["data"].dtype, kf["data"].device, 0)
    s._cudss_load(st, kf["data"])
    st["solver"].factorize()
    st["gen"] += 1
    s._cudss_counters["factorize"] += 1
    s._cudss_counters["bwd_refactorize"] += 1
    kf["st"], kf["gen"] = st, st["gen"]


def _kf_solve(s, kf, rhs):
    """一次三角回代（复用前向分解，零 factorize）。rhs [B,Nj]。"""
    if kf["kind"] == "dense":
        return torch.cholesky_solve(rhs.unsqueeze(-1), kf["chol"]).squeeze(-1)
    st = kf["st"]
    st["rhs"].copy_(rhs)
    out = torch.stack(st["solver"].solve())
    s._cudss_counters["bwd_solve"] += 1
    return out


def _adjgpu_factor(s, c, P, emh, act, linear_solver):
    """在**当前状态**装配 Â 并数值分解一次（前向侧计数，反向零新增分解）。

    为什么必须在当前状态重分解（BWSN_1 实测教训）：κ(Â)~1e10-1e12 时，
    用"上一状态"的分解去解当前状态的系统，单次回代的相对误差 ~κ·‖ΔÂ‖/‖Â‖
    可以超过 1（γ>1 连迭代精化都发散，λ 直接是垃圾且不报错）。分解与
    matvec 同态 ⇒ γ 回到 eps·κ 量级，精化必收缩。"""
    B = P.shape[0]
    dt, dev = P.dtype, P.device
    vals = torch.cat([-P[:, s.lk_both], -P[:, s.lk_both],
                      P[:, s.lk_m1], P[:, s.lk_m2]], dim=1)
    if linear_solver == "cudss":
        data = s._assemble_csr(vals, B)
        data = data.index_add(1, s.A_csr_diag, emh)
        if act is not None and bool(act.any()):
            cb_idx = s.prv_A_csr_idx.view(-1, 5)[:, 4]
            data = data.scatter_add(1, cb_idx.view(1, -1).expand(B, -1),
                                    CBIG * act.to(dt))
        st, gen = s._cudss_factor(data, B)
        return dict(kind="cudss", st=st, gen=gen, B=int(B), data=data,
                    linear_solver="cudss")
    A = torch.zeros(B, s.Nj * s.Nj, dtype=dt, device=dev)
    A = A.scatter_add(1, s.A_idx.expand(B, -1), vals).view(B, s.Nj, s.Nj)
    A = A + torch.diag_embed(emh)
    if act is not None and bool(act.any()):
        j2d = (s.prv_j2_t * (s.Nj + 1)).view(1, -1).expand(B, -1)
        A = A.reshape(B, s.Nj * s.Nj) \
            .scatter_add(1, j2d, CBIG * act.to(dt)).view(B, s.Nj, s.Nj)
    chol, info = torch.linalg.cholesky_ex(A)
    if bool((info != 0).any()):
        bad = torch.nonzero(info != 0).flatten().tolist()
        raise RuntimeError(
            "adjoint='gpu'：样本 %s 的 Â 非正定（精抛光/终态分解阶段） - "
            "参见 solve() 的 badvalve 说明，请把这些样本剔出该批。" % bad)
    return dict(kind="dense", chol=chol, linear_solver="dense")


def _adjgpu_primal_solve(s, c, kf, P, emh, act, rhs_p, r_prv, n_ref=2):
    """原始（primal）约化 Newton 系统 M_p ΔH = r_p：非 PRV 行 = Â 行；
    ACTIVE PRV 下游行 = Dirichlet ΔH[j2] = −r_prv（约束行 H[n2]=hset 的
    Newton）；其上游行 = (Ā_{j1}+Ā_{j2}) 行（把阀流量 Δq_prv 从两条质量行中
    消去）。M_p 与 Â 差 2p 行替换 ⇒ Woodbury（与伴随侧同一套证明）。"""
    mv = lambda x: _adjgpu_matvec(s, c, P, emh, act, x)

    def solveA(rhs):
        x = _kf_solve(s, kf, rhs)
        for _ in range(int(n_ref)):
            x = x + _kf_solve(s, kf, rhs - mv(x))
        return x

    if act is None or not bool(act.any()):
        return solveA(rhs_p)
    B = rhs_p.shape[0]
    p = int(act.shape[1])
    dt = rhs_p.dtype
    j1a, j2a = s.prv_j1_t, s.prv_j2_t
    actf = act.to(dt)
    idx1 = j1a.view(1, -1).expand(B, -1)
    idx2 = j2a.view(1, -1).expand(B, -1)
    # r_p：行 j1a += rhs_p[j2a]（scatter_add 兼容共享上游）；行 j2a := −r_prv
    inc = torch.zeros_like(rhs_p).scatter_add(
        1, idx1, torch.where(act, rhs_p.gather(1, idx2),
                             torch.zeros_like(actf)))
    rp = rhs_p + inc
    rp = rp.scatter(1, idx2, torch.where(act, -r_prv, rhs_p.gather(1, idx2)))
    x0 = solveA(rp)
    cols = []
    for u in list(j2a.tolist()) + list(j1a.tolist()):
        eb = torch.zeros_like(rhs_p)
        eb[:, int(u)] = 1.0
        cols.append(solveA(eb))
    Smat = torch.stack(cols, dim=-1)                     # [B,Nj,2p]
    R2 = Smat.index_select(1, j2a)                       # [B,p,2p]
    eye2 = torch.eye(p, dtype=dt, device=rhs_p.device).repeat(1, 2) \
        .view(1, p, 2 * p)
    C = torch.cat([R2, eye2.expand(B, -1, -1) - CBIG * R2], dim=1)
    actf2 = torch.cat([actf, actf], dim=1)               # [B,2p]
    Ceff = C * actf2.unsqueeze(2) + torch.diag_embed(1.0 - actf2)

    def wood(x, rr):
        top = (x.index_select(1, j2a) - rr.index_select(1, j2a)) * actf
        bot = (rr.index_select(1, j2a)
               - CBIG * x.index_select(1, j2a)) * actf
        wty = torch.cat([top, bot], dim=1)
        y = torch.linalg.solve(Ceff, wty.unsqueeze(-1)).squeeze(-1)
        return x - torch.einsum("bnp,bp->bn", Smat, y)

    dH = wood(x0, rp)
    for _ in range(2):                                   # 外层精化（真 M_p）
        base = mv(dH)
        row1 = (base.gather(1, idx1)
                + base.gather(1, idx2) - CBIG * dH.gather(1, idx2))
        res = rp - base
        res = res.scatter(1, idx1,
                          torch.where(act, rp.gather(1, idx1) - row1,
                                      res.gather(1, idx1)))
        res = res.scatter(1, idx2,
                          torch.where(act, rp.gather(1, idx2)
                                      - dH.gather(1, idx2),
                                      res.gather(1, idx2)))
        dH = dH + wood(solveA(res), res)
    return dH


def _adjgpu_residuals(s, c, q, e_j, H, d_j, ke_j, P, Y, emh, hge, em, act):
    """当前状态的完整残差（约化 Newton / 诊断共用）：
    r_l（P=0 的 ACTIVE PRV 位无效，调用方用 r_prv）、r_prv、r_e、m。"""
    B = q.shape[0]
    hl = torch.where(P != 0.0, Y / P, torch.zeros_like(Y))
    dh = H.index_select(1, s.n1) - H.index_select(1, s.n2)
    r_l = hl - dh
    r_prv = None
    if act is not None:
        r_prv = (H.index_select(1, s.prv_n2_t) - s.prv_hset_t) * act.to(q.dtype)
    hle, _ = s._emitter_hloss(e_j, ke_j)
    r_e = hle - (H.index_select(1, s.junc_nodes_t) - s.el_junc)
    m = torch.zeros(B, s.Nj, dtype=q.dtype, device=q.device)
    m = m.scatter_add(1, c["f_idx2"].expand(B, -1),
                      q.index_select(1, c["lk_m2"]))
    m = m.scatter_add(1, c["f_idx1"].expand(B, -1),
                      -q.index_select(1, c["lk_m1"]))
    m = m - em * e_j - d_j
    return r_l, r_prv, r_e, m


def _adjgpu_polish(s, c, q, e_j, H, d_j, ke_j, S, steps, linear_solver):
    """GPU 批量整装 Newton 精抛光（约化形式；对应 CPU 的 _polish_np）。

    每步在**当前状态**重装配+数值分解（前向侧 +1 factorize/步，见
    _adjgpu_factor 注释），解 M_p ΔH 后回代 Δq（ACTIVE PRV 的 Δq 由其
    下游质量行给出）与 Δe。步数走完后在**终态**再分解一次交给反向复用
    （反向零新增分解；γ 回到装配序 ULP 级）。返回 (q, e_j, H, kf, resid_inf)。"""
    B = q.shape[0]
    dt = q.dtype
    for _ in range(int(steps)):
        P, Y, emh, hge, em, dr, dke, act = _adjgpu_coeffs(s, c, q, e_j, ke_j, S)
        kf = _adjgpu_factor(s, c, P, emh, act, linear_solver)
        r_l, r_prv, r_e, m = _adjgpu_residuals(s, c, q, e_j, H, d_j, ke_j,
                                               P, Y, emh, hge, em, act)
        Pr = P * r_l
        bt = torch.zeros(B, s.Nj, dtype=dt, device=q.device)
        bt = bt.scatter_add(1, c["f_idx2"].expand(B, -1),
                            Pr.index_select(1, c["lk_m2"]))
        bt = bt.scatter_add(1, c["f_idx1"].expand(B, -1),
                            -Pr.index_select(1, c["lk_m1"]))
        rhs_p = m - bt + emh * r_e
        dH = _adjgpu_primal_solve(s, c, kf, P, emh, act, rhs_p, r_prv)
        # Δq = −P·(r_l + (ΔH[j2]−ΔH[j1]))；ACTIVE PRV（P=0）由下游质量行回代
        b1 = dH.index_select(1, c["j1c"]) * c["m1f"]
        b2 = dH.index_select(1, c["j2c"]) * c["m2f"]
        dq = -P * (r_l + (b2 - b1))
        if act is not None and bool(act.any()):
            base = _adjgpu_matvec(s, c, P, emh, act, dH)
            j2a = s.prv_j2_t
            abar2 = (base.index_select(1, j2a)
                     - CBIG * dH.index_select(1, j2a) * act.to(dt))
            dq_prv = rhs_p.index_select(1, j2a) - abar2
            kidx = s.prv_k_t.view(1, -1).expand(B, -1)
            dq = dq.scatter(1, kidx,
                            torch.where(act, dq_prv, dq.gather(1, kidx)))
        de = em * (-r_e + dH) / hge
        q = q + dq
        e_j = e_j + de
        H = H.index_copy(1, s.junc_nodes_t,
                         H.index_select(1, s.junc_nodes_t) + dH)
    # 终态：重装配+分解（反向复用这一份）＋残差报告
    P, Y, emh, hge, em, dr, dke, act = _adjgpu_coeffs(s, c, q, e_j, ke_j, S)
    kf = _adjgpu_factor(s, c, P, emh, act, linear_solver)
    r_l, r_prv, r_e, m = _adjgpu_residuals(s, c, q, e_j, H, d_j, ke_j,
                                           P, Y, emh, hge, em, act)
    r_l_eff = torch.where(P != 0.0, r_l, torch.zeros_like(r_l))
    resid = torch.maximum(r_l_eff.abs().max(dim=1).values,
                          (em * r_e).abs().max(dim=1).values)
    resid = torch.maximum(resid, m.abs().max(dim=1).values)
    if r_prv is not None:
        resid = torch.maximum(resid, r_prv.abs().max(dim=1).values)
    return q, e_j, H, kf, resid


def _adjgpu_node_solve(s, c, kf, P, emh, act, r_std, gQ_prv, n_ref=2):
    """解节点伴随 M λ_m = r：非约束行 = Â 行，ACTIVE PRV 下游行 = 约束行
    λ[j2]−λ[j1]=gQ[prv]（Woodbury 行替换 + 对真 M 的外层精化；probe_math 已证）。"""
    mv = lambda x: _adjgpu_matvec(s, c, P, emh, act, x)

    def solveA(rhs):
        x = _kf_solve(s, kf, rhs)
        for _ in range(int(n_ref)):
            x = x + _kf_solve(s, kf, rhs - mv(x))
        return x

    if act is None or not bool(act.any()):
        return solveA(r_std)
    B = r_std.shape[0]
    p = int(act.shape[1])
    j1a, j2a = s.prv_j1_t, s.prv_j2_t
    dt = r_std.dtype
    actf = act.to(dt)
    # r2：ACTIVE 行替换为 gQ[prv]
    r2 = r_std.scatter(1, j2a.view(1, -1).expand(B, -1),
                       torch.where(act, gQ_prv, r_std.index_select(1, j2a)))
    x0 = solveA(r2)
    cols = []
    for b in range(p):
        eb = torch.zeros_like(r_std)
        eb[:, int(j2a[b])] = 1.0
        cols.append(solveA(eb))
    Smat = torch.stack(cols, dim=-1)                     # [B,Nj,p]
    # C_eff：ACTIVE 行/列 = (s_b)_{j2a}−(s_b)_{j1a}，非 ACTIVE 行退化为单位
    Cw = Smat.index_select(1, j2a) - Smat.index_select(1, j1a)   # [B,p,p]
    mask2 = actf.unsqueeze(2) * actf.unsqueeze(1)
    Ceff = Cw * mask2 + torch.diag_embed(1.0 - actf)

    def wood(x, rr):
        wty = (x.index_select(1, j2a) - x.index_select(1, j1a)
               - rr.index_select(1, j2a)) * actf
        y = torch.linalg.solve(Ceff, wty.unsqueeze(-1)).squeeze(-1)
        return x - torch.einsum("bnp,bp->bn", Smat, y)

    lam = wood(x0, r2)
    for _ in range(2):                                   # 外层精化（真 M 残差）
        res = r2 - mv(lam)
        con = (r2.index_select(1, j2a)
               - (lam.index_select(1, j2a) - lam.index_select(1, j1a)))
        res = res.scatter(1, j2a.view(1, -1).expand(B, -1),
                          torch.where(act, con, res.index_select(1, j2a)))
        lam = lam + wood(solveA(res), res)
    return lam


class ImplicitGGASolveGPU(torch.autograd.Function):
    """(demand, res_head, ke, r_hw, solver, accuracy, max_iter, status_machine,
    assemble, linear_solver, polish_steps) -> (head[B,N], flow[B,L],
    emitter[B,N])。

    forward：mode='dense' 求解器 no_grad 批量收敛（含 PRV 网走状态机）→
    GPU 整装 Newton 精抛光（约化形式，polish_steps 步；对应 CPU 的
    _polish_np）→ 终态装配+数值分解一份留给反向。
    backward：GPU 批量约化伴随（见模块注释） - 复用前向终态分解，
    **零新增 factorize**，只有三角回代 + SpMV/scatter；每场景状态可不同
    （ACTIVE PRV 掩码 per-scenario，约束行 Woodbury 行替换）。
    不支持（明确 raise）：epanet 模式、f32、泵 θ（speed/h0/r）、D-W、
    CONST_HP 钳位支、级联/共下游 PRV、二阶导（create_graph）、未收敛场景。"""

    @staticmethod
    def forward(ctx, demand, res_head, ke, r_hw, solver,
                accuracy=1e-12, max_iter=200, status_machine=None,
                assemble="dense", linear_solver="dense", polish_steps=2):
        s = solver
        if s.mode != "dense":
            raise NotImplementedError(
                "adjoint='gpu' 需要 mode='dense' 的批量求解器（epanet 模式请用"
                "缺省 CPU 伴随 ImplicitGGASolve）")
        if s.dtype != torch.float64:
            raise NotImplementedError("adjoint='gpu' 只接 float64")
        dims = (demand.dim(), res_head.dim(), ke.dim(), r_hw.dim())
        dt, dev = s.dtype, s.device
        t = lambda x: x.detach().to(dtype=dt, device=dev)
        d = t(demand)
        rh = t(res_head)
        ke_t = t(ke)
        rt = t(r_hw)
        if d.dim() == 1:
            d = d.unsqueeze(0)
        if rh.dim() == 1:
            rh = rh.unsqueeze(0)
        if ke_t.dim() == 1:
            ke_t = ke_t.unsqueeze(0)
        B = max(d.shape[0], rh.shape[0], ke_t.shape[0])
        d = d.expand(B, -1)
        rh = rh.expand(B, -1)
        ke_t = ke_t.expand(B, -1)
        sm = bool(getattr(s, "dense_status_machine", False)) \
            if status_machine is None else bool(status_machine)
        c = _adjgpu_cache(s)
        with torch.no_grad(), _override_r_hw(s, rt):
            out = s.solve(d, rh, ke_int=ke_t, max_iter=max_iter,
                          accuracy=accuracy, status_machine=sm,
                          assemble=assemble, linear_solver=linear_solver)
            conv = out["converged"]
            if not bool(conv.all()):
                bad = torch.nonzero(~conv).flatten().tolist()
                raise RuntimeError(
                    "adjoint='gpu'：场景 %s 未收敛（relerr=%s > accuracy=%g） - "
                    "伴随在未收敛态无意义，请收紧初值/放宽 accuracy 或剔除场景。"
                    % (bad[:8], [float(out["relerr"][b]) for b in bad[:4]],
                       float(accuracy)))
            S = out.get("status")
            q = out["flow_cfs"]
            H = out["head_ft"]
            e_j = out["emitter_cfs"].index_select(1, s.junc_nodes_t)
            ke_j = ke_t.index_select(1, s.junc_nodes_t)
            d_j = d.index_select(1, s.junc_nodes_t)
            # GPU 整装 Newton 精抛光（约化形式）+ 终态分解（反向复用零新增）。
            # GGA 半迭代在病态网上有 ~1e-8 相对停机地板（模块 docstring），
            # 且 ACTIVE PRV 的罚函数定点与约束行定点差 O(1/CBIG) - 精抛光把
            # 状态压到与 CPU solve_polished 同级，梯度对拍才有 1e-9 的地。
            q, e_j, H, kf, resid = _adjgpu_polish(
                s, c, q, e_j, H, d_j, ke_j, S, polish_steps, linear_solver)
        em = (ke_j > 0.0).to(dt)
        emitter = torch.zeros(B, s.N, dtype=dt, device=dev) \
            .index_copy(1, s.junc_nodes_t, e_j * em)
        ctx.solver = s
        ctx.kf = kf
        ctx.S = S
        ctx.head = H
        ctx.flow = q
        ctx.emitter = emitter
        ctx.ke_t = ke_t
        ctx.rt = rt
        ctx.dims = dims
        ctx.in_devs = (demand.device, res_head.device, ke.device, r_hw.device)
        ctx.resid = resid
        head, flow, emit_o = H, q, emitter
        if all(x == 1 for x in dims[:3]):
            head, flow, emit_o = head[0], flow[0], emit_o[0]
        return head, flow, emit_o

    @staticmethod
    def backward(ctx, gH, gQ, gE):
        if torch.is_grad_enabled():
            raise NotImplementedError(
                "adjoint='gpu' 的反向只做一阶：create_graph=True / 二阶导请改用"
                "缺省 CPU 伴随（或 linear_solver='dense' 的展开路径）。")
        s = ctx.solver
        c = _adjgpu_cache(s)
        dt, dev = s.dtype, s.device
        q = ctx.flow
        B = q.shape[0]

        def _g(g, ref):
            if g is None:
                return torch.zeros_like(ref)
            g = g.detach().to(dtype=dt, device=dev)
            return g.unsqueeze(0).expand_as(ref) if g.dim() == 1 else g

        gH = _g(gH, ctx.head)
        gQ = _g(gQ, ctx.flow)
        gE = _g(gE, ctx.emitter)
        e_j = ctx.emitter.index_select(1, s.junc_nodes_t)
        ke_j = ctx.ke_t.index_select(1, s.junc_nodes_t)
        with _override_r_hw(s, ctx.rt):
            P, _Y, emh, hge, em, dr, dke, act = _adjgpu_coeffs(
                s, c, q, e_j, ke_j, ctx.S)
        gH_j = gH.index_select(1, s.junc_nodes_t)
        gE_j = gE.index_select(1, s.junc_nodes_t)
        # 标准右端 r = B^T P gQ − em·gE/hge − gH
        PgQ = P * gQ
        r = torch.zeros(B, s.Nj, dtype=dt, device=dev)
        r = r.scatter_add(1, c["f_idx2"].expand(B, -1),
                          PgQ.index_select(1, c["lk_m2"]))
        r = r.scatter_add(1, c["f_idx1"].expand(B, -1),
                          -PgQ.index_select(1, c["lk_m1"]))
        r = r - emh * gE_j - gH_j
        gQ_prv = gQ.index_select(1, s.prv_k_t) if act is not None else None
        _kf_prepare(s, ctx.kf)
        lam_m = _adjgpu_node_solve(s, c, ctx.kf, P, emh, act, r, gQ_prv)
        # 回代
        l1 = lam_m.index_select(1, c["j1c"]) * c["m1f"]
        l2 = lam_m.index_select(1, c["j2c"]) * c["m2f"]
        lam_l = P * (gQ - (l2 - l1))
        lam_e = em * (gE_j + lam_m) / hge
        # θ 收缩
        gd = torch.zeros(B, s.N, dtype=dt, device=dev) \
            .index_copy(1, s.junc_nodes_t, lam_m)
        gke = torch.zeros(B, s.N, dtype=dt, device=dev) \
            .index_copy(1, s.junc_nodes_t, -(lam_e * dke))
        grh = torch.zeros(B, s.N, dtype=dt, device=dev)
        if c["lk_f1"].numel():
            grh = grh.scatter_add(1, c["f1_nodes"].expand(B, -1),
                                  lam_l.index_select(1, c["lk_f1"]))
        if c["lk_f2"].numel():
            grh = grh.scatter_add(1, c["f2_nodes"].expand(B, -1),
                                  -lam_l.index_select(1, c["lk_f2"]))
        grh = grh + gH * c["fixmask"].to(dt)             # head 定水头位直通
        gr = -(lam_l * dr)
        dv = ctx.in_devs
        dm = ctx.dims

        def _fit(g, dim, dev_):
            g = g.to(device=dev_)
            return g.sum(0) if dim == 1 else g

        return (_fit(gd, dm[0], dv[0]), _fit(grh, dm[1], dv[1]),
                _fit(gke, dm[2], dv[2]), _fit(gr, dm[3] + 1, dv[3]),
                None, None, None, None, None, None, None)


def implicit_solve(solver, demand, res_head, ke=None, r_hw=None,
                   accuracy=1e-12, max_iter=200, polish_steps=3,
                   speed=None, pump_h0=None, pump_r=None, status=None,
                   adjoint="cpu", status_machine=None,
                   assemble="dense", linear_solver="dense"):
    """便捷包装：补默认 ke/r_hw 张量后调 ImplicitGGASolve。返回 (head, flow, emitter)。

    adjoint（本轮新开关）："cpu"=缺省，行为逐位不变（epanet 收敛+Newton 精抛光、
    scipy splu 伴随）；"gpu"=ImplicitGGASolveGPU - mode='dense' 批量前向 +
    GPU 约化整装 Newton 精抛光（polish_steps 步，语义对应 CPU 的 polish）+
    GPU 约化伴随（复用前向终态分解，dense Cholesky 或 cudss 皆可），
    status_machine/assemble/linear_solver 直通前向；speed/pump_h0/pump_r/status
    是 CPU 伴随的参数，gpu 路径不接（含 PRV 网的状态由状态机 per-scenario
    决定；泵 θ 梯度请走 CPU 伴随）。"""
    dt, dev = solver.dtype, solver.device
    if ke is None:
        ke = solver.node_ke_default
    if r_hw is None:
        r_hw = solver.r_hw
    t = lambda x: x.to(dtype=dt, device=dev) if isinstance(x, torch.Tensor) \
        else torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
    tn = lambda x: None if x is None else t(x)
    if adjoint == "cpu":
        return ImplicitGGASolve.apply(t(demand), t(res_head), t(ke), t(r_hw),
                                      solver, accuracy, max_iter, polish_steps,
                                      tn(speed), tn(pump_h0), tn(pump_r), status)
    if adjoint != "gpu":
        raise ValueError("adjoint 只能是 'cpu'/'gpu'，收到 %r" % (adjoint,))
    if (speed is not None or pump_h0 is not None or pump_r is not None
            or status is not None):
        raise NotImplementedError(
            "adjoint='gpu' 不接 speed/pump_h0/pump_r/status：dense 路径的状态由"
            "状态机 per-scenario 决定，泵 θ 梯度请走缺省 CPU 伴随。")
    return ImplicitGGASolveGPU.apply(t(demand), t(res_head), t(ke), t(r_hw),
                                     solver, accuracy, max_iter,
                                     status_machine, assemble, linear_solver,
                                     int(polish_steps))


# ======================================================================
# 路线一：固定 K 次迭代的 dense GGA 展开（autograd 直通）
# ======================================================================
def solve_unrolled(solver: GGASolver, demand, res_head, ke=None, r_hw=None,
                   K=None, q0=None, e0=None, speed=None, status=None,
                   pump_h0=None, pump_r=None,
                   assemble="dense", linear_solver="dense"):
    """固定迭代数 K 的 dense GGA（无收敛判定/冻结掩码），输出对
    demand/res_head/ke/r_hw（及泵 θ：speed/pump_h0/pump_r）可微。
    K 默认建议取该场景 epanet 模式实测迭代数。
    公式与 solver.solve(dense) 相同（hydcoeffs.c / hydsolver.c 行号见 solver.py）；
    B2 泵支为 pumpcoeff（hydcoeffs.c:673-791）的 torch 复刻＋恒功率泵半步防穿零
    （hydsolver.c:437-444）。status=冻结内部状态 int8[L]（None=初始状态；
    含泵/水池网建议由状态冻结帧传入）。
    返回 dict(head_ft[B,N], flow_cfs[B,L], emitter_cfs[B,N], relerr[B](detached))。

    **demand 梯度不是在所有网上都可用**（P4 / 审计 R4）：展开路径给的是截断 K 步
    的梯度，在 City_D/city_d、ky4、Net3 上它随 K **不收敛**（实测 K+5→K+10 的相对
    变化分别是 1.000e+00 / 1.082e-03 / 1.818e-03），这三个网上**不建议**用本函数求
    demand 梯度，请改用 ImplicitGGASolve。判定**不能按 κ(A) 拍**（反例：EXA6
    κ₂=1.0e10、city_h κ₂=1.8e10 都好到 1e-7），必须逐网现测 - 调
    `unrolled_grad_health(solver, demand, res_head, ...)` 即可。完整三档判据、实测表
    与订正理由见本模块 docstring 与 data/p4_guards_wip.txt。

    assemble / linear_solver（sparse_gpu_plan.md §1a/§1b）：缺省 "dense"/"dense"
    与历史逐位一致。assemble="csr" 把 A 装进 [B,nnz]（autograd 仍通）；
    linear_solver="cudss" 走 nvmath 的 cuDSS；**P3 起可微** - 线性解那一段换成
    _CudssSolveFn（solver.py），反向按 A^{-1} 的精确伴随算，且复用前向的数值分解
    （§1c）。展开 K 步时想让每一步都零重分解，把 solver.cudss_grad_slots 调到 >=K
    （并把 cudss_cache_max 调到 >=slots）；缺省 slots=1 只有最后一步能复用，
    其余步在反向里重 factorize 一次，计数见 solver.cudss_counters()。
    """
    s = solver
    if assemble not in ("dense", "csr"):
        raise ValueError(f"assemble 只能是 'dense'/'csr'，收到 {assemble!r}")
    if linear_solver not in ("dense", "cudss"):
        raise ValueError(
            f"linear_solver 只能是 'dense'/'cudss'，收到 {linear_solver!r}")
    if linear_solver == "cudss" and assemble != "csr":
        raise ValueError(
            "linear_solver='cudss' 必须配 assemble='csr'；收到 assemble=%r" % (assemble,))
    # 能力守卫。本路径复刻 pipecoeff/pumpcoeff（hydcoeffs.c:498-620 / 673-791）与
    # 关闭支 P=1/CBIG（:531-536），冻结 status 覆盖 CVPIPE 与泵的通断，所以
    # CVPIPE/PIPE/PUMP/TCV 都是正确的。但 PRV/PSV 的 ACTIVE 支是罚函数约束行
    # （prvcoeff:972-975 / psvcoeff:1024-1027），FCV 的 ACTIVE 支是切流
    # （fcvcoeff:1073-1084），两者本路径都未实现；若放行，这些阀会落进管道支并被
    # RQtol 钳位，结果错误且无提示。宁可在入口拒绝，也不静默近似。
    # D-W 的阻力随流量变（DWpipecoeff hydcoeffs.c:578-620 + frictionFactor :623-670），
    # 本路径用的是固定 r 配 s.hexp 的幂律，对 D-W 网实测差 1.8e+4 ft。C-M 与 H-W 同为
    # 定阻力幂律、结构相容，故只拒 D-W。
    if str(getattr(s, 'headloss_form', 'H-W')).upper() == 'D-W':
        raise NotImplementedError(
            'solve_unrolled 未实现 Darcy-Weisbach：其摩擦系数随雷诺数变化，'
            '按固定阻力幂律算会静默算错 - 请改用 ImplicitGGASolve（隐式伴随路径，已支持）。')
    _UNROLL_OK = {_CVPIPE, _PIPE, _PUMP, _TCV}
    _bad = sorted(set(np.asarray(s.lt_np).tolist()) - _UNROLL_OK)
    if _bad:
        _nm = {_PRV: 'PRV', _PSV: 'PSV', _FCV: 'FCV', _PBV: 'PBV', _GPV: 'GPV'}
        raise NotImplementedError(
            'solve_unrolled 支持 CVPIPE/PIPE/PUMP/TCV；出现 link_type=' + str(_bad)
            + ' (' + ', '.join(_nm.get(b, str(b)) for b in _bad) + ')。'
            'PRV/PSV 的 ACTIVE 罚函数约束行与 FCV 的切流支未在展开路径实现，'
            '放行会静默算错 - 请改用 ImplicitGGASolve（隐式伴随路径，已支持）。')
    dt, dev = s.dtype, s.device
    t = lambda x: x.to(dtype=dt, device=dev) if isinstance(x, torch.Tensor) \
        else torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
    d = t(demand)
    rh = t(res_head)
    single = d.dim() == 1
    if single:
        d = d.unsqueeze(0)
    if rh.dim() == 1:
        rh = rh.unsqueeze(0)
    B = max(d.shape[0], rh.shape[0])
    d = d.expand(B, -1)
    rh = rh.expand(B, -1)
    ke = s.node_ke_default if ke is None else t(ke)
    if ke.dim() == 1:
        ke = ke.unsqueeze(0)
    ke = ke.expand(B, -1)
    r_hw = s.r_hw if r_hw is None else t(r_hw)
    if K is None:
        K = int(s.max_iter_default)

    # ---- 冻结状态 / 泵静态数据 ----
    status_np = s.init_status_int if status is None \
        else np.asarray(status, dtype=np.int8)
    # closed = status<=CLOSED（旧调用 status=None 时逐位 == s.closed，
    # 见 _resolve_pump_args 注释）
    closed = torch.as_tensor(status_np <= s.ST_CLOSED, device=dev)
    has_pump = bool(s.n_pumps)
    if has_pump:
        sp_np = s.init_setting if speed is None else None
        sp = t(sp_np) if speed is None else t(speed)
        h0_t = t(s.pl_h0) if pump_h0 is None else t(pump_h0)
        r_t = t(s.pl_r) if pump_r is None else t(pump_r)
        is_pump_t = torch.as_tensor(s.is_pump_np, device=dev)
        chp_t = torch.as_tensor(s.is_chp_np, device=dev)
        nocurve_t = torch.as_tensor(s.pl_ptype == 3, device=dev)
        custom_np = s.pl_ptype == 2
        custom_t = torch.as_tensor(custom_np, device=dev)
        custom_links = np.where(custom_np)[0]
        n_np = s.pl_n.copy()
        n_np[np.abs(n_np - 1.0) < 1e-6] = 1.0            # :739（TINY=1e-6）
        n_t = t(n_np)
        lin_t = torch.as_tensor(n_np == 1.0, device=dev) & is_pump_t
        # ω=0 的泵与关闭同支（hydcoeffs.c:697 setting==0.0）
        closed = closed | (is_pump_t & (sp.detach() == 0.0))
    else:
        closed = closed

    ke_j = ke[:, s.junc_nodes_t]
    has_em = (ke_j > 0.0).detach()
    em = has_em.to(dt)
    d_j = d[:, s.junc_nodes_t]
    rh_fix = rh[:, s.fixed_nodes_t]
    Nj = s.Nj

    if q0 is None:
        if has_pump or status is not None:
            # initlinkflow（hydraul.c:344-374）按冻结状态：关闭=QZERO、
            # 泵=Kc*Q0（:366-369）、其余=PI*D²/4（常量初值，不参与反传）
            cl0 = status_np <= s.ST_CLOSED
            q_np = np.where(cl0, QZERO, PI * s.diam_np ** 2 / 4.0)
            if has_pump:
                sp0 = sp.detach().cpu().numpy()
                for k in s.pump_links:
                    k = int(k)
                    if not cl0[k]:
                        q_np[k] = sp0[k] * s.pl_q0[k]    # hydraul.c:366-369
            q = t(q_np).unsqueeze(0).expand(B, -1)
        else:
            q = s._init_flow().unsqueeze(0).expand(B, -1)
    else:
        q = t(q0)
        q = q.unsqueeze(0).expand(B, -1) if q.dim() == 1 else q
    if e0 is None:
        e_j = torch.where(has_em, torch.ones_like(ke_j), torch.zeros_like(ke_j))
    else:
        e = t(e0)
        e = e.unsqueeze(0).expand(B, -1) if e.dim() == 1 else e
        e_j = e[:, s.junc_nodes_t]

    H = None
    relerr = None
    for _ in range(int(K)):
        # ---- P/Y（_pipe_PY/_tcv_PY 同式；r_hw 为传入张量）----
        # qa = q*sgn ≡ |q|（SGN(0)=+1，types.h:107）：见 solver._pipe_PY 注释 -
        # torch.abs 在 q=0 的 subgradient=0 会把钳位支 hloss=RQtol·q 的导数变 0，
        # 展开反传经 P=1/RQtol=1e7 逐迭代放大而发散；q*sgn 前向逐位同 |q|。
        sgn = torch.where(q < 0, -1.0, 1.0)
        qa = q * sgn
        hgrad = s.hexp * r_hw * qa.clamp_min(1e-30) ** (s.hexp - 1.0)  # hydcoeffs.c:550
        lin = hgrad < s.rqtol
        hgrad = torch.where(lin, torch.full_like(hgrad, s.rqtol), hgrad)
        hloss = torch.where(lin, hgrad * qa, hgrad * qa / s.hexp)      # :557,:560
        mlp = s.km_pipe > 0.0
        hloss = torch.where(mlp, hloss + s.km_pipe * qa * qa, hloss)   # :563-567
        hgrad = torch.where(mlp, hgrad + 2.0 * s.km_pipe * qa, hgrad)
        hloss = hloss * sgn                                            # :570
        P_pipe, Y_pipe = 1.0 / hgrad, hloss / hgrad
        km = s.km_tcv                                                  # TCV（:1129-1150）
        hgv = 2.0 * km * qa
        linv = hgv < s.rqtol
        hgv = torch.where(linv, torch.full_like(hgv, s.rqtol), hgv)
        hlv = torch.where(linv, q * hgv, q * hgv / 2.0)
        kmp = km > 0.0
        P_tcv = torch.where(kmp, 1.0 / hgv, torch.full_like(q, 1.0 / CSMALL))
        Y_tcv = torch.where(kmp, hlv / hgv, q)
        P = torch.where(s.is_tcv, P_tcv, P_pipe)
        Y = torch.where(s.is_tcv, Y_tcv, Y_pipe)
        # ---- 泵 P/Y（pumpcoeff hydcoeffs.c:673-791 的 torch 复刻）----
        # 各 clamp_min 仅为非选中支的 0/inf 反传防护（前向选中支数值不变，
        # 参见 _pipe_PY 的 0·inf→nan 注释）；EPANET 迭代在钳位支取 +hgrad
        # （半迭代技巧），展开路径逐式复刻求解器。
        if has_pump:
            spc = sp.clamp_min(1e-12)                    # ω>0（关闭/ω=0 走 closed 支）
            h0e = spc * spc * h0_t                       # :737 h0 = SQR(setting)*H0
            re = r_t * spc ** (2.0 - n_t)                # :740 r = R*pow(w, 2-N)
            # CONST_HP（:743-763）：hgrad=-r/q/q，钳位 [RQtol, CBIG]
            qa2 = (qa * qa).clamp_min(1e-30)
            hg_chp_raw = -re / qa2                       # :746
            big_c = hg_chp_raw > CBIG
            sml_c = hg_chp_raw < s.rqtol
            hg_chp = torch.where(big_c, torch.full_like(qa, CBIG),
                                 torch.where(sml_c, torch.full_like(qa, s.rqtol),
                                             hg_chp_raw))
            q_c = sgn * qa.clamp_min(1e-30)              # 防 re/0
            hl_chp = torch.where(big_c | sml_c, -hg_chp * q,  # :750-756
                                 re / q_c)               # :761
            # POWER_FUNC 非线性（:767-779）：hgrad=n·r·|q|^(n-1)，RQtol 钳位
            hg_pf_raw = n_t * re * qa.clamp_min(1e-30) ** (n_t - 1.0)   # :770
            sml_p = hg_pf_raw < s.rqtol
            hg_pf = torch.where(sml_p, torch.full_like(qa, s.rqtol), hg_pf_raw)
            hl_pf = torch.where(sml_p, h0e + hg_pf * q,  # :774-775
                                h0e + hg_pf * q / n_t.clamp_min(1e-30))  # :778
            # 线性 n=1（:781-785）：hgrad=r，hloss=h0+r·q
            hg_pf = torch.where(lin_t, re, hg_pf)
            hl_pf = torch.where(lin_t, h0e + re * q, hl_pf)
            hg_p = torch.where(chp_t, hg_chp, hg_pf)
            hl_p = torch.where(chp_t, hl_chp, hl_pf)
            # CUSTOM（hydcoeffs.c:716-733）：段选择用 detach 的 |q|/ω（离散选段，
            # (H0,R) 为段内常数），hloss=H0·ω²+R·ω·Q 对 q/ω 保持可微
            if custom_links.size:
                qd = q.detach().cpu().numpy()
                spd = spc.detach().cpu().numpy()
                Bq = qd.shape[0]
                h0m = np.zeros((Bq, s.L))
                rm = np.zeros((Bq, s.L))
                for k in custom_links:
                    k = int(k)
                    j = int(s.pl_pumpidx[k])
                    for b in range(Bq):
                        h0c, rc = s._curvecoeff(j, abs(qd[b, k]) / spd[k])  # :722
                        h0m[b, k] = -h0c                 # :726
                        rm[b, k] = -rc                   # :727
                hg_cu = torch.as_tensor(rm, dtype=dt, device=dev) * spc     # :731
                hl_cu = torch.as_tensor(h0m, dtype=dt, device=dev) \
                    * (spc * spc) + hg_cu * q            # :732
                hg_p = torch.where(custom_t, hg_cu, hg_p)
                hl_p = torch.where(custom_t, hl_cu, hl_p)
            P_pump = 1.0 / hg_p                          # :789
            Y_pump = hl_p / hg_p                         # :790
            # NOCURVE：视作开阀（:709-714）
            P_pump = torch.where(nocurve_t, torch.full_like(P_pump, 1.0 / CSMALL),
                                 P_pump)
            Y_pump = torch.where(nocurve_t, q, Y_pump)
            P = torch.where(is_pump_t, P_pump, P)
            Y = torch.where(is_pump_t, Y_pump, Y)
        P = torch.where(closed, torch.full_like(P, 1.0 / CBIG), P)   # 关闭支
        Y = torch.where(closed, q, Y)

        # ---- matrixcoeffs（装配序同 dense；scatter_add_ 到新建零张量 = out-of-place）
        Xflow = torch.zeros(B, Nj, dtype=dt, device=dev)
        Xflow = Xflow.scatter_add(1, s.f_idx1.expand(B, -1), -q[:, s.lk_m1])
        Xflow = Xflow.scatter_add(1, s.f_idx2.expand(B, -1), q[:, s.lk_m2])
        vals = torch.cat([-P[:, s.lk_both], -P[:, s.lk_both],
                          P[:, s.lk_m1], P[:, s.lk_m2]], dim=1)
        if assemble == "csr":
            csr_data = s._assemble_csr(vals, B)
            A = None if linear_solver == "cudss" else s._csr_to_dense(csr_data, B)
        else:
            A = torch.zeros(B, Nj * Nj, dtype=dt, device=dev)
            A = A.scatter_add(1, s.A_idx.expand(B, -1), vals).view(B, Nj, Nj)
        F = torch.zeros(B, Nj, dtype=dt, device=dev)
        F = F.scatter_add(1, s.f_idx1.expand(B, -1), Y[:, s.lk_m1])
        F = F.scatter_add(1, s.f_idx2.expand(B, -1), -Y[:, s.lk_m2])
        if s.lk_g1.numel():                                            # 定水头接地
            F = F.scatter_add(1, s.g1_row.expand(B, -1),
                              P[:, s.lk_g1] * rh[:, s.g1_src])
        if s.lk_g2.numel():
            F = F.scatter_add(1, s.g2_row.expand(B, -1),
                              P[:, s.lk_g2] * rh[:, s.g2_src])
        # emitter（hydcoeffs.c:333-375）
        ke_adj = torch.clamp(ke_j, min=CSMALL)
        hge = s.qexp * ke_adj * torch.abs(e_j).clamp_min(1e-30) ** (s.qexp - 1.0)
        line = hge < s.rqtol
        hge = torch.where(line, torch.full_like(hge, s.rqtol), hge)
        hle = torch.where(line, hge * e_j, hge * e_j / s.qexp)
        if linear_solver == "cudss":
            csr_data = csr_data.index_add(1, s.A_csr_diag, em / hge)
        else:
            A = A + torch.diag_embed(em / hge)
        F = F + em * (hle + s.el_junc) / hge
        Xflow = Xflow - em * e_j
        # nodecoeffs（:256-279）
        Xflow = Xflow - d_j
        F = F + Xflow

        # ---- 线性解 + 2 步迭代精化（与 dense 同款批不变写法）----
        if linear_solver == "cudss":
            Hj = s._cudss_solve(csr_data, F, B)
        else:
            chol = torch.linalg.cholesky(A)
            Fc = F.unsqueeze(-1)
            Hj = torch.cholesky_solve(Fc, chol)
            for _r in range(2):
                AHj = (A * Hj.transpose(-2, -1)).sum(-1, keepdim=True)
                Hj = Hj + torch.cholesky_solve(Fc - AHj, chol)
            Hj = Hj.squeeze(-1)
        H = torch.cat([Hj, rh_fix], dim=1).index_select(1, s._hperm_inv)

        # ---- newflows（hydsolver.c:358-514）----
        dh = H[:, s.n1] - H[:, s.n2]
        dq = Y - P * dh
        if has_pump:
            # 恒功率泵半步防穿零（hydsolver.c:437-444）：dq>Q 时 dq=Q/2
            chp_hit = chp_t & (dq > q)
            dq = torch.where(chp_hit, q / 2.0, dq)
        q = q - dq
        dh_e = Hj - s.el_junc
        dq_e = (hle - dh_e) / hge
        e_j = e_j - em * dq_e
        with torch.no_grad():
            dqsum = dq.abs().sum(1) + (em * dq_e).abs().sum(1)
            qsum = q.abs().sum(1) + (em * e_j).abs().sum(1)
            relerr = dqsum / qsum

    emitter = torch.zeros(B, s.N, dtype=dt, device=dev)
    emitter = emitter.index_copy(1, s.junc_nodes_t, e_j * em)
    out = dict(head_ft=H, flow_cfs=q, emitter_cfs=emitter, relerr=relerr)
    if single:
        out = {k: v[0] for k, v in out.items()}
    return out


# ======================================================================
# 展开梯度的可用性自检（P4 / 审计 R4）
# ======================================================================
# 三档判据的门槛（标定实测见 scripts/p4_r4_probe.py 与 data/p4_guards_wip.txt）。
# 判据量：rel_K = max|g(K+ΔK) − g(K+2ΔK)| / max|g|（展开梯度随 K 还动不动）。
GRAD_HEALTH_OK = 1e-8        # 低于此 = "可用"
GRAD_HEALTH_COND = 1e-5      # 低于此 = "有条件"（只能当搜索方向）
# 高于 GRAD_HEALTH_COND = "不建议"，改用 ImplicitGGASolve。


def unrolled_grad_health(solver: GGASolver, demand, res_head, ke=None,
                         r_hw=None, K=None, dK=5, param="demand", weights=None,
                         vs_implicit=False, seed=0):
    """展开路径梯度的**逐网现测**可用性自检（审计 R4 的落地接口）。

    为什么必须现测：展开路径给的是截断 K 步的梯度，不是不动点梯度；它在一部分
    网上随 K 不收敛，而**这件事按 κ(A) 判不出来**（实测反例：EXA6 κ₂=1.013e+10、
    city_h κ₂=1.844e+10 两网都好到 1e-7，Net3 κ₂=1.594e+09 却坏到 2.175e-03），
    按"钳位支/网规模"也判不出来（ky4 只有 1 条 RQtol 钳位支、243 个不稳定坐标
    与钳位支毫不相邻）。所以判据只能是**行为量**：把 K 加长一截，梯度还动不动。

    做法：分别在 K+dK 与 K+2dK 上跑一次展开反传，量
        rel_K = max|g1 − g2| / max(|g1|,|g2|)
    可选 `vs_implicit=True` 再跟 ImplicitGGASolve（不动点梯度）对一次
    （慢得多：forward 走 scipy 精抛光、backward 逐场景 splu）。

    返回 dict(verdict, rel_K, rel_implicit, gmax, K, dK, param)，
    verdict ∈ {"可用", "有条件", "不建议"}：
      · "可用"   rel_K < 1e-8（且给了 vs_implicit 时 <1e-6）；
      · "有条件" rel_K < 1e-5：只能当一阶优化的搜索方向，别拿去过 1e-6 的对拍门槛；
      · "不建议" 其余：**改用 ImplicitGGASolve**。实测命中 City_D/city_d
        （rel_K=1.000e+00）、ky4（1.082e-03）、Net3（1.818e-03）。

    本函数**不改任何缺省**，只是跑两次现成的 solve_unrolled；不传就完全不执行。
    """
    if param not in ("demand", "res_head", "ke", "r_hw"):
        raise ValueError("param 只能是 demand/res_head/ke/r_hw，收到 %r" % (param,))
    s = solver
    dt, dev = s.dtype, s.device
    t = lambda x: x.to(dtype=dt, device=dev) if isinstance(x, torch.Tensor) \
        else torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
    d0, rh0 = t(demand), t(res_head)
    ke0 = s.node_ke_default if ke is None else t(ke)
    r0 = s.r_hw if r_hw is None else t(r_hw)
    if K is None:
        with torch.no_grad():
            K = int(torch.as_tensor(
                s.solve(d0, rh0, ke_int=ke0)["iters"]).max())
    K, dK = int(K), int(dK)
    B = max(1, d0.dim() == 2 and d0.shape[0] or 1)
    if weights is None:
        g = np.random.default_rng(seed)
        shp = (B, s.N) if d0.dim() == 2 else (s.N,)
        weights = torch.as_tensor(g.normal(0, 1, shp), dtype=dt, device=dev)
    W = t(weights)

    def grad_at(kk):
        th = dict(demand=d0.clone(), res_head=rh0.clone(),
                  ke=ke0.clone(), r_hw=r0.clone())
        th[param].requires_grad_(True)
        o = solve_unrolled(s, th["demand"], th["res_head"], ke=th["ke"],
                           r_hw=th["r_hw"], K=kk)
        gg, = torch.autograd.grad((o["head_ft"] * W).sum(), th[param])
        return gg.detach()

    g1, g2 = grad_at(K + dK), grad_at(K + 2 * dK)
    den = float(torch.max(g1.abs().max(), g2.abs().max()).clamp_min(1e-300))
    rel_K = float((g1 - g2).abs().max()) / den
    rel_imp = None
    if vs_implicit:
        th = dict(demand=d0.clone(), res_head=rh0.clone(),
                  ke=ke0.clone(), r_hw=r0.clone())
        th[param].requires_grad_(True)
        h, _q, _e = ImplicitGGASolve.apply(th["demand"], th["res_head"],
                                           th["ke"], th["r_hw"], s,
                                           1e-12, 200, 3)
        gi, = torch.autograd.grad((h * W).sum(), th[param])
        gi = gi.detach()
        dn = float(torch.max(g1.abs().max(), gi.abs().max()).clamp_min(1e-300))
        rel_imp = float((g1 - gi).abs().max()) / dn
    if rel_K < GRAD_HEALTH_OK and (rel_imp is None or rel_imp < 1e-6):
        verdict = "可用"
    elif rel_K < GRAD_HEALTH_COND and (rel_imp is None or rel_imp < 1e-3):
        verdict = "有条件"
    else:
        verdict = "不建议"
    return dict(verdict=verdict, rel_K=rel_K, rel_implicit=rel_imp,
                gmax=den, K=K, dK=dK, param=param)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dgga.parse import Net

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net = Net.load(os.path.join(ROOT, "data", "reference"), "rand_main_0009")
    s = GGASolver(net, mode="dense",
                  inp_path=os.path.join(ROOT, "networks", "random_main", "rand_0009.inp"))
    d0 = net.demand_cfs_at(0)
    rh0 = np.nan_to_num(net.reservoir_head_ft_at(0))
    ke = np.zeros(net.N)
    ke[s.junc_nodes[:10]] = 0.5
    sol = solve_polished(s, d0, rh0, ke)
    print("polish 后 ‖F‖inf = %.3e (GGA iters=%d)" % (sol["resid_inf"][0], sol["iters"][0]))
    assert sol["resid_inf"][0] < 1e-10

    dt_ = torch.tensor(d0, dtype=torch.float64, requires_grad=True)
    rt = torch.tensor(rh0, dtype=torch.float64, requires_grad=True)
    kt = torch.tensor(ke, dtype=torch.float64, requires_grad=True)
    r0 = s.r_hw.clone().requires_grad_(True)
    head, flow, emit = ImplicitGGASolve.apply(dt_, rt, kt, r0, s, 1e-12, 200, 3)
    w = torch.tensor(np.random.default_rng(0).normal(size=s.Nj))
    Lb = (w * head[s.junc_nodes]).sum()
    Lb.backward()
    gB = dt_.grad.clone()

    dt2 = torch.tensor(d0, dtype=torch.float64, requires_grad=True)
    out = solve_unrolled(s, dt2, rh0, ke=ke, K=12)
    La = (w * out["head_ft"][s.junc_nodes]).sum()
    La.backward()
    rel = float(((dt2.grad - gB)[s.junc_nodes].abs() /
                 gB[s.junc_nodes].abs().clamp_min(1e-12)).max())
    print("unrolled(K=12) vs implicit: demand 梯度最大相对差 = %.3e" % rel)
    assert rel < 1e-4
    print("autodiff.py 冒烟测试通过")

# -*- coding: utf-8 -*-
"""check_symmetry.py - A 的逐位精确对称性守卫（P4 / 审计 R1，最高优先级）。

**为什么要有这一项**：cuDSS 通路的伴随（solver.py `_CudssSolveFn`）把前向的
**数值分解直接拿来解 A^T λ = g**，其全部合法性只挂在一条恒等式上 - A 逐位精确
对称。稠密通路的 `torch.linalg.cholesky` 同样只读下三角，A 若失去对称，前向照样
收敛、迭代数照样对、结果看不出异样，**只有梯度会静静地错**。P3 对抗审计
（data/p3_adversarial_wip.txt §1.4）在 6 网 × 8 轮上实测 max|A−A^T| = 0.000e+00，
但这条恒等式此前**零自动保护**：谁往装配里加一项单侧写入（PRV 的 ACTIVE 罚函数
约束行、非对称阀模型、单向 CV 的特殊处理），没有任何测试会红。本脚本把它写死。

判据是**逐位**（== 0.0），不是容差，理由是实测的：§C 的变异 M1（转置那一格差
1e-13 相对量的非对称阀）在 Hanoi 上只把 max|A−A^T| 抬到 **9.635e-12**、在
sym_torture 上只有 **2.065e-13** - 一条 1e-9 的容差门槛会把这两网整网放行，
只有 TCV 在场（P=1/CSMALL=1e6）的 Net3/city_d 才被放大到 9.984e-07。
换句话说：容差判据能不能抓到单侧装配，取决于**恰好测了哪张网**，逐位判据不取决。

------------------------------------------------------------------------------
本脚本不改 dgga 一行代码。A 的捕获点是 `torch.linalg.cholesky` 的入参 - 那正是
"线性求解器实际看到的矩阵"，也正是伴随要复用其分解的那个矩阵。捕获点若被后人
换掉（比如改用 cholesky_ex），本脚本会因为"捕获轮数 ≠ 期望轮数"直接 FAIL，
而不是静默地什么都不检查（§B 的 liveness 断言）。

检查三层：
  §A 位型层（静态，不解方程）：CSR 位型转置闭合 + Nj 个对角全部在位。
  §B 数值层（动态，逐 Newton 轮）：
      · 稠密 A（solve / solve_unrolled，assemble=dense 与 csr 两条装配）
        max|A − A^T| == 0.0，逐位；
      · CSR data（`_assemble_csr` 的返回值，cuDSS 通路唯一经过的那份）
        max|data[k] − data[t(k)]| == 0.0，逐位。
  §C 变异层（证明断言真能抓错）：把 dgga 复制一份、在**源码级**注入四种单侧装配，
      子进程里跑同一套 §A/§B，要求每一种都变红。只加恒真检查是不算数的。

覆盖（"足够多的网与状态"）：
  · 并联管（同一节点对两条链路 → 两笔贡献落进同一 CSR 槽）：ky4 21 对、
    sym_torture 1 对；
  · **同一节点对 ≥3 条链路且方向不全同**（sym_mixed3：一对 3 条 2 正 1 反 +
    一对 4 条 2 正 2 反）：p5 审计 §1.2 / commit a78c185 实测这是旧装配把
    逐位对称打破的**唯一**触发形态（NW_Model 40 对、~1 ULP），而 2 条并联
    因 IEEE 加法两元可交换天然测不到 - 本网表此前恰好全是 ≤2 条并联，
    结构上就照不出这类破缺，故补此网（solver.__init__ 的定序修复后应恒 0）；
  · 孤立 junction（不接任何链路，对角只由 emittercoeffs 提供）：sym_torture；
  · 有/无 emitter：city_d_emit（5 个）、sym_torture（3 个）、以及各网的注入 ke；
  · 泵 / 水池：ky4（2 泵 4 池）、Net3（2 泵 3 池）、ky5（11 泵 3 池）；
  · 跨数量级的 P：关闭支 P=1/CBIG=1e-8（ky4/Net3/city_d/city_h/sym_torture）
    与 ACTIVE TCV P=1/CSMALL=1e6（city_d 79 个、sym_torture 1 个）同时在场，
    再叠加 ×1e-3 / ×1e3 的需水缩放把管道支的 P 拉开；每个配置都打印实测的
    |A| 非零动态范围，覆盖是量出来的不是声称的。

用法：`python -X utf8 scripts/check_symmetry.py`（CPU，无需 CUDA/nvmath）。
退出码 0 = 全部通过。regression_all.py 第 ⑨ 项调用本脚本。
"""

import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETS = os.path.join(ROOT, "networks")

# 逐位判据：不是容差。
EXACT = 0.0


# ======================================================================
# sym_torture 合成网：孤立 junction + 并联管 + ACTIVE TCV + 关闭支 + 水池 + emitter
# networks/ 整目录在 .gitignore 里（真实管网保密 + 第三方 benchmark 不转发），
# 所以这张**我们自己写的**合成网的原文就存在本脚本里，运行时按需落盘 -
# 否则守卫会在一份干净 clone 上因为缺网络文件而静默少测一批状态。
# ======================================================================
SYM_TORTURE_INP = r"""[TITLE]
Synthetic torture network for the A-symmetry regression guard (R1).
Covers in one file: parallel pipes (two links on the same node pair, so two
link contributions land in the same CSR slot), an isolated junction that carries
only an emitter (structural-zero row whose diagonal comes from emittercoeffs),
an ACTIVE TCV (P = 1/CSMALL = 1e6), a Closed pipe (P = 1/CBIG = 1e-8), a tank,
and emitters on three junctions.

[JUNCTIONS]
;ID              Elev         Demand
 J1              100          50
 J2              90           40
 J3              80           30
 J4              70           60
 J5              60           20
 J6              50           10
 JISO            40           5

[RESERVOIRS]
;ID              Head
 R1              250

[TANKS]
;ID              Elev         InitLevel    MinLevel     MaxLevel     Diameter     MinVol
 T1              120          20           2            38           50           0

[PIPES]
;ID              Node1        Node2        Length       Diameter     Roughness    MinorLoss    Status
 P1              R1           J1           1000         12           100          0            Open
 P2A             J1           J2           1500         10           100          0            Open
 P2B             J1           J2           1500         8            100          0            Open
 P3              J2           J3           1200         10           100          0            Open
 P4              J3           J4           1000         8            100          0            Open
 P5              J4           J5           900          8            100          0            Open
 P6              J5           J6           800          6            100          0            Open
 P7              J6           J1           1100         6            100          0            Closed
 P8              J4           T1           700          12           100          0            Open

[VALVES]
;ID              Node1        Node2        Diameter     Type         Setting      MinorLoss
 V1              J3           J5           10           TCV          8.5          0

[EMITTERS]
;Junction        Coefficient
 J2              0.5
 J6              1.2
 JISO            2.0

[OPTIONS]
 Units            GPM
 Headloss         H-W
 Specific Gravity 1.0
 Viscosity        1.0
 Trials           200
 Accuracy         0.00000001
 Unbalanced       Continue 10
 Pattern          1
 Demand Multiplier 1.0
 Emitter Exponent 0.5
 Quality          None
 Tolerance        0.01

[TIMES]
 Duration              0
 Hydraulic Timestep    1:00
 Pattern Timestep      1:00

[REPORT]
 Status               No
 Summary              No
 Page                 0

[END]
"""


# ======================================================================
# sym_mixed3 合成网：同一节点对 ≥3 条链路且方向不全同（p5 审计 §1.2 的触发形态）
# J1-J2 三条（PA/PB 正向 + PC 反向）、J3-J4 四条（2 正 2 反）。
# 旧装配下 (i,j)/(j,i) 两槽对同一组加数按不同交错次序累加 ⇒ A 差 ~1 ULP；
# solver.__init__ 的定序修复后按构造逐位对称。2 条并联测不到（两元可交换）。
# ======================================================================
SYM_MIXED3_INP = r"""[TITLE]
Synthetic torture network for the >=3 mixed-direction parallel-link case.
Two junction pairs each carry multiple links with inconsistent orientation:
J1-J2 has three links (two forward, one reversed) and J3-J4 has four links
(two forward, two reversed). With the pre-fix assembly the (i,j) and (j,i)
slots accumulate the same addends in different interleaved orders, which
breaks bit-exact symmetry by about one ULP; two parallel links can never
show this because two-term IEEE addition commutes bitwise.

[JUNCTIONS]
;ID              Elev         Demand
 J1              100          40
 J2              90           35
 J3              80           30
 J4              70           25

[RESERVOIRS]
;ID              Head
 R1              200

[PIPES]
;ID              Node1        Node2        Length       Diameter     Roughness    MinorLoss    Status
 PR              R1           J1           1000         12           100          0            Open
 PA              J1           J2           900          10           100          0            Open
 PB              J1           J2           700          8            130          0            Open
 PC              J2           J1           533          6            110          0            Open
 P23             J2           J3           800          8            100          0            Open
 QA              J3           J4           600          8            100          0            Open
 QB              J3           J4           500          6            120          0            Open
 QC              J4           J3           450          6            110          0            Open
 QD              J4           J3           400          5            120          0            Open

[EMITTERS]
;Junction        Coefficient
 J4              0.8

[OPTIONS]
 Units            GPM
 Headloss         H-W
 Trials           200
 Accuracy         0.00000001
 Unbalanced       Continue 10
 Emitter Exponent 0.5

[TIMES]
 Duration              0
 Hydraulic Timestep    1:00

[REPORT]
 Status               No
 Summary              No

[END]
"""


def _ensure_inp(fname, content):
    p = os.path.join(NETS, "variants", fname)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    old = None
    if os.path.exists(p):
        with open(p, encoding="utf-8", newline="") as f:
            old = f.read()
    if old != content:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    return p


def ensure_sym_torture():
    _ensure_inp("sym_mixed3.inp", SYM_MIXED3_INP)
    return _ensure_inp("sym_torture.inp", SYM_TORTURE_INP)


# ======================================================================
# 用例表：(标签, inp 相对路径, B, 注入 emitter 的 junction 比例, 需水缩放, K)
# ======================================================================
CASES = [
    # 小网：孤立 junction + 并联管 + ACTIVE TCV + 关闭支 + 水池 + emitter
    ("sym_torture",      "variants/sym_torture.inp",   4, 0.0,  1.0,   8),
    ("sym_torture x1e-3", "variants/sym_torture.inp",  4, 0.0,  1e-3,  8),
    ("sym_torture x1e3",  "variants/sym_torture.inp",  3, 0.4,  1e3,   8),
    # ≥3 条链路方向不全同的节点对（p5 审计 §1.2 的破缺触发形态；修复前必红）
    ("sym_mixed3",       "variants/sym_mixed3.inp",    4, 0.0,  1.0,   8),
    ("sym_mixed3 x1e3",  "variants/sym_mixed3.inp",    3, 0.3,  1e3,   8),
    # 纯管网（无泵无阀），batch 与单场景各一
    ("Hanoi",            "public/Hanoi.inp",           1, 0.0,  1.0,   8),
    ("Hanoi +emit",      "public/Hanoi.inp",           4, 0.35, 1.0,   8),
    ("Modena",           "public/Modena.inp",          4, 0.0,  1.0,   8),
    ("rand_main_0009",   "random_main/rand_0009.inp",  4, 0.25, 1.0,   8),
    ("rand_main_0014",   "random_main/rand_0014.inp",  4, 0.0,  1e-3,  8),
    # 泵 + 水池 + 关闭支
    ("Net1 pump+tank",   "public/Net1.inp",            4, 0.0,  1.0,   8),
    ("Net3 pump+tank",   "public/Net3.inp",            4, 0.20, 1.0,   8),
    ("EXA6 pump+tank",   "InpData/EXA6.inp",           4, 0.0,  1.0,   8),
    ("ky5 11pump",       "InpData/ky5.inp",            2, 0.10, 1.0,   6),
    # 并联管 21 对 + 2 泵 4 池 + 关闭支
    ("ky4 parallel",     "public/ky4.inp",             2, 0.0,  1.0,   5),
    ("ky4 +emit",        "public/ky4.inp",             2, 0.15, 1.0,   5),
    # 79 个 ACTIVE TCV（P=1/CSMALL）+ 4 关闭支
    ("city_d 79TCV",     "realInpData/city_d.inp",     2, 0.0,  1.0,   5),
    ("city_d_emit",      "variants/city_d_emit.inp",   2, 0.0,  1.0,   5),
    ("city_d x1e3",      "realInpData/city_d.inp",     2, 0.0,  1e3,   5),
    # 12 关闭支的大网
    ("city_h 6pump",    "InpData/city_h.inp",        2, 0.0,  1.0,   5),
    # PRV 轮：ACTIVE PRV 的 CBIG 罚函数行（本守卫 docstring 点名的头号风险
    # 形态 - "只写 (r,c) 不写 (c,r) 的大 M"）。第 7 元 "sm" ⇒ 构造带
    # dense_status_machine、solve 带 status_machine=True，只跑 solve 两条装配
    # （unrolled 对 PRV 维持 raise）；批内 ACTIVE/CLOSED/OPEN 混态由需水缩放
    # 与 B 内扰动触发。
    ("BWSN_1 PRV8 sm",   "public/_cleaned/BWSN_Network_1.inp", 4, 0.0, 1.0, 8,
     "sm"),
    ("BWSN_1 PRV8 x0.05", "public/_cleaned/BWSN_Network_1.inp", 3, 0.2, 0.05,
     8, "sm"),
    ("L-TOWN PRV3 sm",   "public/_cleaned/L-TOWN.inp", 2, 0.0,  1.0,   8,
     "sm"),
]

# 变异体：(名字, 文件, 原文, 替换, 说明)。全部是"单侧装配"的真实形态。
MUTANTS = [
    ("M1_asym_valve_1ulp", "solver.py",
     "            vals = torch.cat([-Pm[:, self.lk_both], -Pm[:, self.lk_both],",
     "            vals = torch.cat([-Pm[:, self.lk_both], "
     "-Pm[:, self.lk_both] * (1.0 + 1e-13),",
     "非对称阀模型：转置那一格差 1e-13 相对量（容差判据抓不到）"),
    ("M2_prv_penalty_row", "solver.py",
     "                A = A.view(B, Nj, Nj)",
     "                A = A.view(B, Nj, Nj)\n"
     "                _k = int((self.A_csr_row != self.A_csr_col)"
     ".nonzero()[0, 0])\n"
     "                _E = torch.zeros(Nj, Nj, dtype=dt, device=dev)\n"
     "                _E[int(self.A_csr_row[_k]), int(self.A_csr_col[_k])] "
     "= -1e6\n"
     "                A = A + _E",
     "PRV ACTIVE 罚函数行：只写 (r,c) 不写 (c,r) 的大 M"),
    ("M3_csr_one_sided", "solver.py",
     "        data.scatter_add_(1, self.A_csr_scatter.expand(B, -1), vals)",
     "        data.scatter_add_(1, self.A_csr_scatter.expand(B, -1), vals)\n"
     "        data[:, 1] = data[:, 1] - 1e3",
     "CSR 装配里单侧改一个非对角槽（cuDSS 通路唯一经过的那份数据）"),
    ("M4_unrolled_one_sided", "autodiff.py",
     "            A = A.scatter_add(1, s.A_idx.expand(B, -1), vals).view(B, Nj, Nj)",
     "            A = A.scatter_add(1, s.A_idx.expand(B, -1), vals).view(B, Nj, Nj)\n"
     "            A = A + torch.nn.functional.pad("
     "torch.full((1, 1), 1e-9, dtype=A.dtype, device=A.device),\n"
     "                (1, Nj - 2, 0, Nj - 1))",
     "展开路径（solve_unrolled）单侧加一项 1e-9（前向看不出）"),
]


# ======================================================================
# 捕获：torch.linalg.cholesky 的入参 = 线性求解器实际看到的 A
# ======================================================================
class Capture:
    """捕获点 = torch.linalg.cholesky **与 cholesky_ex** 的入参（PRV 轮起，
    含 PRV 的 dense 走 cholesky_ex 以显式捕获 badvalve 类失败并报样本号；
    两者是同一 LAPACK 例程，只是 ex 带 info），加 _assemble_csr 的返回值。"""

    def __init__(self, solver_mod):
        self.A = []
        self.csr = []
        self._chol = torch.linalg.cholesky
        self._chol_ex = torch.linalg.cholesky_ex
        self._asm = solver_mod.GGASolver._assemble_csr
        self._mod = solver_mod

    def __enter__(self):
        cap = self

        def chol(A, *a, **kw):
            cap.A.append(A.detach().clone())
            return cap._chol(A, *a, **kw)

        def chol_ex(A, *a, **kw):
            cap.A.append(A.detach().clone())
            return cap._chol_ex(A, *a, **kw)

        def asm(self_, vals, B):
            out = cap._asm(self_, vals, B)
            cap.csr.append(out.detach().clone())
            return out

        torch.linalg.cholesky = chol
        torch.linalg.cholesky_ex = chol_ex
        self._mod.GGASolver._assemble_csr = asm
        return self

    def __exit__(self, *e):
        torch.linalg.cholesky = self._chol
        torch.linalg.cholesky_ex = self._chol_ex
        self._mod.GGASolver._assemble_csr = self._asm
        return False


def transpose_slots(s):
    """CSR 槽 k → 其转置槽 t(k)。返回 (tidx, 位型是否转置闭合, 缺失槽数)。"""
    Nj = s.Nj
    row = s.A_csr_row.cpu().numpy().astype(np.int64)
    col = s.A_csr_col.cpu().numpy().astype(np.int64)
    key = row * Nj + col                     # 构造期 np.unique ⇒ 已升序
    tkey = col * Nj + row
    pos = np.searchsorted(key, tkey)
    posc = np.clip(pos, 0, key.size - 1)
    ok = (pos < key.size) & (key[posc] == tkey)
    return torch.as_tensor(posc, dtype=torch.int64), bool(ok.all()), int((~ok).sum())


def build_case(mods, inp_rel, B, ke_frac, dscale, seed=20260822, sm=False):
    """按 inp 造 (solver, D, R, KE)。水池水头取上下限中点，emitter 按比例注入。
    sm=True（PRV 轮）：构造带 dense_status_machine=True（PRV 准入）。"""
    parse_inp, GGASolver = mods["parse_inp"], mods["GGASolver"]
    p = os.path.join(NETS, inp_rel)
    net = parse_inp(p)
    if sm:
        s = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                      inp_path=p, dense_status_machine=True)
    else:
        s = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                      inp_path=p, dense_tank_bound_check=False)
    g = np.random.default_rng(seed)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64) * dscale
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = np.asarray(net.tank_hmin)[:tn.size]
        hi = np.asarray(net.tank_hmax)[:tn.size]
        rh0[tn] = 0.5 * (lo + hi)
    rh0 = np.nan_to_num(rh0)
    D = torch.as_tensor(d0[None, :] * g.uniform(0.85, 1.15, (B, d0.size)),
                        dtype=torch.float64)
    R = torch.as_tensor(np.repeat(rh0[None, :], B, 0) +
                        g.uniform(-1.0, 1.0, (B, rh0.size)) *
                        (np.asarray(net.node_type) != 0)[None, :],
                        dtype=torch.float64)
    ke = np.asarray(net.node_ke, dtype=np.float64).copy()
    if ke_frac > 0.0:
        jn = np.asarray(s.junc_nodes)
        ke[jn[g.random(jn.size) < ke_frac]] = 0.5
    KE = torch.as_tensor(np.repeat(ke[None, :], B, 0), dtype=torch.float64)
    return net, s, D, R, KE


def worst_sym(cap, tidx):
    """返回 (稠密最差, CSR 最差, 稠密轮数, CSR 轮数, |A| 非零动态范围)。"""
    wd = wc = 0.0
    rng_lo, rng_hi = np.inf, 0.0
    for A in cap.A:
        wd = max(wd, float((A - A.transpose(-2, -1)).abs().max()))
        nz = A[A != 0].abs()
        if nz.numel():
            rng_lo = min(rng_lo, float(nz.min()))
            rng_hi = max(rng_hi, float(nz.max()))
    for dd in cap.csr:
        wc = max(wc, float((dd - dd.index_select(1, tidx)).abs().max()))
        nz = dd[dd != 0].abs()
        if nz.numel():
            rng_lo = min(rng_lo, float(nz.min()))
            rng_hi = max(rng_hi, float(nz.max()))
    dyn = (rng_hi / rng_lo) if rng_lo not in (np.inf, 0.0) else float("nan")
    return wd, wc, len(cap.A), len(cap.csr), dyn


def run_battery(mods, cases, verbose=True):
    """跑整套 §A/§B。返回 (rows, n_bad)。rows 每行 = 一个配置的实测。"""
    solve_unrolled = mods["solve_unrolled"]
    smod = mods["smod"]
    rows, bad = [], 0
    for case in cases:
        label, inp_rel, B, ke_frac, dscale, K = case[:6]
        sm = len(case) > 6 and case[6] == "sm"
        try:
            net, s, D, R, KE = build_case(mods, inp_rel, B, ke_frac, dscale,
                                          sm=sm)
            tidx, pat_ok, pat_miss = transpose_slots(s)
            # 位型层：转置闭合 + 全对角在位
            diag_ok = bool((s.A_csr_row.index_select(0, s.A_csr_diag) ==
                            s.A_csr_col.index_select(0, s.A_csr_diag)).all()
                           and s.A_csr_diag.numel() == s.Nj)
            if sm:
                # PRV 网：只跑 solve 两条装配（含批量状态机；unrolled 对 PRV
                # 维持 raise）。CSR 捕获点在 _assemble_csr（valve 贡献在其后
                # 加入 A），稠密 A 捕获在 cholesky_ex 入参 = 含 PRV 贡献。
                cfgs = (
                    ("solve/dense", lambda: s.solve(D, R, ke_int=KE,
                                                    status_machine=True)),
                    ("solve/csr", lambda: s.solve(D, R, ke_int=KE,
                                                  status_machine=True,
                                                  assemble="csr")))
            else:
                cfgs = (
                    ("solve/dense", lambda: s.solve(D, R, ke_int=KE)),
                    ("solve/csr", lambda: s.solve(D, R, ke_int=KE,
                                                  assemble="csr")),
                    ("unrolled/dense", lambda: solve_unrolled(
                        s, D, R, ke=KE, K=K)),
                    ("unrolled/csr", lambda: solve_unrolled(
                        s, D, R, ke=KE, K=K, assemble="csr")))
            for cfgname, fn in cfgs:
                exc = ""
                with Capture(smod) as cap:
                    try:
                        with torch.no_grad():
                            out = fn()
                    except Exception as e:      # 变异体可能把前向也搞崩；
                        out = None              # 捕获到的 A 仍要判对称
                        exc = type(e).__name__
                wd, wc, nA, nC, dyn = worst_sym(cap, tidx)
                # liveness：捕获点还在不在？稠密线性解每轮恰好一次 cholesky
                if out is None:
                    exp = None
                elif cfgname.startswith("solve/"):
                    conv_all = bool(out["converged"].all()) if B > 1 \
                        else bool(out["converged"])
                    # 未收敛场景 iters=maxtrials+1（F3 口径）≠ 实际轮数，
                    # 只在全收敛时才断言"每轮恰一次分解"
                    exp = (int(out["iters"].max()) if B > 1
                           else int(out["iters"])) if conv_all else None
                else:
                    exp = K
                live = (nA >= 1 and (exp is None or nA == exp)
                        and (nC >= 1 if cfgname.endswith("/csr") else nC == 0))
                ok = (wd == EXACT and wc == EXACT and pat_ok and diag_ok
                      and live and not exc)
                bad += 0 if ok else 1
                rows.append((f"{label} [{cfgname}]", s.Nj, s.A_csr_nnz,
                             nA, nC, wd, wc, dyn, pat_ok, live, exc, ok))
                if verbose:
                    print("  %-34s Nj=%-4d nnz=%-5d 轮=%-2d/%-2d "
                          "max|A-A^T|=%.3e CSR=%.3e |A|动态范围=%.2e "
                          "位型对称=%s 捕获活=%s %s%s"
                          % (f"{label} [{cfgname}]", s.Nj, s.A_csr_nnz, nA, nC,
                             wd, wc, dyn, "是" if pat_ok else "否",
                             "是" if live else "否",
                             "PASS" if ok else "<-- FAIL",
                             f" ({exc})" if exc else ""), flush=True)
            del s
        except Exception as e:
            bad += 1
            rows.append((label, -1, -1, 0, 0, float("nan"), float("nan"),
                         float("nan"), False, False, type(e).__name__, False))
            if verbose:
                print("  %-34s <-- FAIL 构造/运行异常 %s: %s"
                      % (label, type(e).__name__, str(e)[:90]), flush=True)
    return rows, bad


def load_mods(pkg_root):
    """从 pkg_root 导入 dgga（变异体走临时目录的那一份）。"""
    if pkg_root is not None:
        sys.path.insert(0, pkg_root)
    else:
        sys.path.insert(0, ROOT)
    import dgga.solver as smod
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    from dgga.autodiff import solve_unrolled
    assert os.path.abspath(smod.__file__).startswith(
        os.path.abspath(pkg_root if pkg_root else ROOT)), smod.__file__
    return dict(smod=smod, parse_inp=parse_inp, GGASolver=GGASolver,
                solve_unrolled=solve_unrolled)


# ======================================================================
# 子进程模式：把 dgga 复制一份、源码级注入单侧装配、跑同一套检查
# ======================================================================
def child_mutant(name):
    ent = [m for m in MUTANTS if m[0] == name]
    if not ent:
        print(f"MUTANT-RESULT {name} ERROR 未知变异体")
        return 2
    _, fname, old, new, _desc = ent[0]
    tmp = tempfile.mkdtemp(prefix="symmut_")
    shutil.copytree(os.path.join(ROOT, "dgga"), os.path.join(tmp, "dgga"))
    tgt = os.path.join(tmp, "dgga", fname)
    with open(tgt, encoding="utf-8") as f:
        src = f.read()
    if src.count(old) != 1:
        print(f"MUTANT-RESULT {name} ERROR 锚点命中 {src.count(old)} 次（应为 1）")
        return 2
    with open(tgt, "w", encoding="utf-8") as f:
        f.write(src.replace(old, new))
    shutil.rmtree(os.path.join(tmp, "dgga", "__pycache__"), ignore_errors=True)
    ensure_sym_torture()
    mods = load_mods(tmp)
    # 变异体只跑一小撮用例（够红即可）
    sub = [c for c in CASES if c[0] in ("sym_torture", "Hanoi +emit",
                                        "Net3 pump+tank", "city_d 79TCV")]
    print(f"--- 变异体 {name}：{_desc}")
    rows, bad = run_battery(mods, sub, verbose=True)
    worst = max([r[5] for r in rows if r[5] == r[5]] + [0.0])
    worstc = max([r[6] for r in rows if r[6] == r[6]] + [0.0])
    print(f"MUTANT-RESULT {name} {'RED' if bad else 'GREEN'} "
          f"bad={bad}/{len(rows)} worstDense={worst:.3e} worstCSR={worstc:.3e}")
    return 0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        return child_mutant(sys.argv[2])

    print("=" * 100)
    print("check_symmetry.py - A 逐位精确对称守卫（审计 R1）")
    print("判据：max|A − A^T| == 0.0（逐位，非容差）；CSR data 同理按转置槽逐位比。")
    print("=" * 100)
    ensure_sym_torture()
    mods = load_mods(None)

    print("\n【§A+§B 正品 dgga：%d 网/状态 × 4 条装配-求解组合】" % len(CASES))
    rows, bad = run_battery(mods, CASES, verbose=True)
    n = len(rows)
    print(f"\n  小计：{n - bad}/{n} PASS；"
          f"最差 max|A-A^T| = "
          f"{max([r[5] for r in rows if r[5] == r[5]] + [0.0]):.3e}（稠密）/ "
          f"{max([r[6] for r in rows if r[6] == r[6]] + [0.0]):.3e}（CSR）")
    print(f"  实测 |A| 非零动态范围最大 = "
          f"{max([r[7] for r in rows if r[7] == r[7]] + [0.0]):.2e}"
          f"（跨数量级的 P 覆盖是量出来的）")

    print("\n【§C 变异层：证明这条断言真能抓错（不是恒真检查）】")
    mut_bad = 0
    mut_lines = []
    for name, fname, _o, _nn, desc in MUTANTS:
        p = subprocess.run([sys.executable, "-X", "utf8",
                            os.path.abspath(__file__), "--child", name],
                           cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1800)
        out = (p.stdout or "") + (p.stderr or "")
        line = ""
        for ln in out.splitlines():
            if ln.startswith("MUTANT-RESULT"):
                line = ln.strip()
        red = " RED " in f" {line} "
        mut_bad += 0 if red else 1
        print(f"  {name:<24} {'RED (断言变红，符合预期)' if red else '<-- 未变红！'}"
              f"  [{fname}] {desc}")
        print(f"      {line if line else out.strip().splitlines()[-1:]}")
        mut_lines.append(line)

    ok = (bad == 0) and (mut_bad == 0)
    print("\n" + "=" * 100)
    print(f"总判定: {'PASS' if ok else 'FAIL'}  "
          f"（正品 {n - bad}/{n} 逐位对称；变异 {len(MUTANTS) - mut_bad}/"
          f"{len(MUTANTS)} 如期变红）")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

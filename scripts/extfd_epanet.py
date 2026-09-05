# -*- coding: utf-8 -*-
"""extfd_epanet.py - 任务 C 外部对拍：EPANET(双精度 DLL) 有限差分 vs 解析梯度。

对象：city_d_emit（LPS / H-W / EMITTER EXPONENT 0.5 -> 内部 Qexp=2；
水库 2506 带水头模式 2516；25 帧 EPS，只取 t=0 帧）。

损失：L = sum_j w_j * H_j（junction，内部 ft），w = default_rng(2026).normal(Nj)
 - 与 scripts/gradcheck_3way.py 同种子（2026）同分布；gradcheck 的 run_case 在抽 w
之前先用同一 rng 布置随机 emitter，city_d_emit 的 emitter 来自 INP 无需布置，
故此处 w 为种子后的首次抽样（报告注明）。

坐标（8 个）：3 个 emitter 节点用户系数 C、3 个"单类别、正需水、非 emitter"
junction 的基础需水、水库水头、1 根非 RQtol 钳位管道 roughness(H-W C)。

自变量对齐（用户单位 <-> 内部单位换算链，全部对齐解析梯度的自变量定义）：
  demand   解析 g 对内部施加需水 d_cfs(t=0)。EPANET 改用户基值(EN_BASEDEMAND, LPS)，
           t=0 施加值 = 基值*pattern 因子；FD 分母用 EN_DEMAND 实测施加值之差
           （DDA、非 emitter 节点 EN_DEMAND = 施加需水），直接得 dL/dd_int，
           免去 pattern 因子与需水类别归属的换算。
  Ke       解析 g 对内部 Ke_int。EPANET 改用户 C（EN_EMITTER 直传原值）。
           Ke_int = Ucf[FLOW]^Qexp / (Ucf[PRESSURE] * C^Qexp)（input1.c:567-573，
           与 parse.py:350-368 同式）-> dKe/dC = -Qexp*Ke_int/C（链式法则），
           对比量取 dL/dC = dL/dKe_int * dKe/dC。
  水库水头 解析 g 对内部施加水头(ft)。EPANET 改 EN_ELEVATION（水库=定水头基值，m），
           施加水头 = 基值*pattern2516 因子；FD 分母用 EN_HEAD 实测施加水头之差，
           直接得 dL/dh_int。
  r_hw     解析 g 对内部阻力 r = 4.727*len_ft/C^1.852/diam_ft^4.871（parse.py:397，
           同 hydcoeffs.c resistcoeff 的 HW 支）-> dr/dC = -Hexp*r/C（Hexp=1.852），
           对比量取 dL/dC = dL/dr * dr/dC。

EPANET 收敛设置与实测噪声地板（决定 FD 方案）：
  EN_setoption 把 ACCURACY 收到 1e-8（epanet.c:1241-1243 API 下限；INP 解析路径
  才有 input3.c:2014-2019 的 [1e-5,1e-1] 钳位）、TRIALS 基准 200、DAMPLIMIT=1e-6
  （epanet2_enums.h:315）。实测 city_d_emit 与 city_d 同属 κ~1e9 病态网：GGA 在
  relerr ~3e-8..9e-8 进入极限环（damp/CHECKFREQ/MAXCHECK 扫描均无法达 1e-8，
  每次 EN_runH 以 warn=1 到满迭代停）。以解析精抛光不动点为基准实测极限环误差
  e = L_ep - L*：随 trials 变化有稳定偏置 ~-2e-5（中心差分自动相消）、
  典型波动 ~1.5e-5、偶发漂移到 -1.2e-4（trials=175）；e 与 EN_RELATIVEERROR
  无相关（corr=0.07），故不能按 relerr 择优相位。

FD 方案（"步长按坐标量级自适应"+ 抗噪）：
  1) 每个损失评估取 trials∈{190,200,210} 三次求解的 L 中位数（剔除极限环
     偶发漂移；偏置部分在中心差分中相消）；
  2) 对每坐标取对称步 delta = H*{1, 0.75, 0.5, 0.25}，4 对中心差分联立
     最小二乘拟合奇模型 dL(delta) = 2g*delta + b*delta^3（等价于消 O(h^2)
     截断的广义 Richardson，但噪声由最大步主导，优于放大小步噪声的
     4/3,-1/3 外推权）；截断残余 O(delta^5)；
  3) 步长基准按信号量级配齐（err ~ e_amp/(2H|g|)，e_amp~1.5e-5）：
     emitter C 取 0.30C、demand 取 0.04x、水库水头 0.002x、roughness 0.04C；
     demand/水头的 FD 分母用实测施加值（输入直驱量，无收敛噪声）。
  坐标选择按信号强度：demand 取 |g*x| 最大的 3 个候选，emitter 取 |gC*C|
  最大 3 个（5 个里选），roughness 取 |gC| 最大的非钳位管道。

门槛：逐坐标 |gFD - gA| / max(|gFD|, 1e-12) < 1e-4（EPANET 收敛噪声所限）。
"""

import ctypes
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import Net                                    # noqa: E402
from dgga.solver import GGASolver                             # noqa: E402
from dgga.autodiff import ImplicitGGASolve, solve_polished    # noqa: E402
from dgga import epanet_ref as er                             # noqa: E402

SEED = 2026
TOL = 1e-4
INP = os.path.join(ROOT, "networks", "variants", "city_d_emit.inp")
GGA_MI = 80          # city_d 族 κ~1e9：relerr ~15 迭代进平台，80 远超；精抛光兜底
ACC_EP = 1e-8        # EN_setoption(EN_ACCURACY) 的 API 下限（epanet.c:1241-1243）
EN_DAMPLIMIT = 17    # epanet2_enums.h:315（epanet_ref 未列，此处补）
FRACS = (1.0, 0.75, 0.5, 0.25)   # 对称步长梯队（相对各坐标步长基准 H）
CAPS = (190, 200, 210)           # 每个评估取三个 trials 的 L 中位数（见文件头）


# ---------------------------------------------------------------------------
# EPANET 侧：getter 与 t=0 单帧求解
# ---------------------------------------------------------------------------
def get_node(en, idx1, prop):
    """EN_getnodevalue 用户单位原值（idx1 为 1 基索引）。"""
    v = ctypes.c_double()
    en._check(en.lib.EN_getnodevalue(en._ph, idx1, prop, ctypes.byref(v)),
              f"EN_getnodevalue({prop},{idx1})")
    return v.value


def get_link(en, idx1, prop):
    v = ctypes.c_double()
    en._check(en.lib.EN_getlinkvalue(en._ph, idx1, prop, ctypes.byref(v)),
              f"EN_getlinkvalue({prop},{idx1})")
    return v.value


def solve_t0(en, nperm):
    """EN_openH -> EN_initH(EN_NOSAVE) -> 单次 EN_runH（即 t=0 帧）-> 读全节点
    HEAD/DEMAND（内部 ft/cfs，按 net 节点序 nperm 重排）。只需 t=0 帧，不跑完整 EPS。
    返回 (head_ft[N], demand_out_cfs[N], iterations, relerr, warn_code)。"""
    lib, ph = en.lib, en._ph
    t = ctypes.c_long()
    en._check(lib.EN_openH(ph), "EN_openH")
    try:
        en._check(lib.EN_initH(ph, er.EN_NOSAVE), "EN_initH")
        rc = lib.EN_runH(ph, ctypes.byref(t))
        if rc > 100:
            raise RuntimeError(f"EN_runH(t=0) 失败（错误码 {rc}）")
        warn = int(rc)
        n = nperm.size
        h = np.empty(n, dtype=np.float64)
        dem = np.empty(n, dtype=np.float64)
        v = ctypes.c_double()
        for i in range(n):
            j = int(nperm[i])
            en._check(lib.EN_getnodevalue(ph, j, er.EN_HEAD, ctypes.byref(v)),
                      f"EN_getnodevalue(HEAD,{j})")
            h[i] = v.value / en._ucf_head
            en._check(lib.EN_getnodevalue(ph, j, er.EN_DEMAND, ctypes.byref(v)),
                      f"EN_getnodevalue(DEMAND,{j})")
            dem[i] = v.value / en._ucf_flow
        st = ctypes.c_double()
        en._check(lib.EN_getstatistic(ph, er.EN_ITERATIONS, ctypes.byref(st)),
                  "EN_getstatistic(ITERATIONS)")
        iters = int(st.value)
        en._check(lib.EN_getstatistic(ph, er.EN_RELATIVEERROR, ctypes.byref(st)),
                  "EN_getstatistic(RELATIVEERROR)")
        relerr = float(st.value)
    finally:
        lib.EN_closeH(ph)
    return h, dem, iters, relerr, warn


def eval_median(en, nperm, sjunc, w):
    """一次损失评估：trials∈CAPS 三次求解取 L 中位数（剔除极限环偶发漂移，
    见文件头噪声分析）。返回 (L_median, 该次的 head, demand_out, relerr_max)。"""
    runs = []
    re_max = 0.0
    for k in CAPS:
        en.set_option(er.EN_TRIALS, k)
        h, dem, _, rel, _ = solve_t0(en, nperm)
        runs.append((float(w @ h[sjunc]), h, dem))
        re_max = max(re_max, rel)
    runs.sort(key=lambda r: r[0])
    L, h, dem = runs[len(runs) // 2]
    return L, h, dem, re_max


def fd_slope(en, nperm, sjunc, w, apply_fn, restore_fn, H,
             meas_idx=None, meas_what=None):
    """多步长对称中心差分 + 奇模型最小二乘：dL(delta) = 2g*delta + b*delta^3。

    apply_fn(delta) 把坐标置为 基值+delta（绝对设置，非增量）；restore_fn() 复位。
    meas_what='demand'/'head' 时 FD 分母用实测施加值之差（EN_DEMAND / EN_HEAD，
    二者为输入直驱量、无收敛噪声），否则用名义 delta（setter 直传的用户原值
    坐标：EMITTER C、ROUGHNESS C）。返回 (g_LS, g_max_step, delta_eff[4],
    relerr_max)。"""
    de_list, dl_list = [], []
    re_max = 0.0
    for frac in FRACS:
        dlt = H * frac
        apply_fn(+dlt)
        Lp, hp, demp, rep = eval_median(en, nperm, sjunc, w)
        apply_fn(-dlt)
        Lm, hm, demm, rem = eval_median(en, nperm, sjunc, w)
        re_max = max(re_max, rep, rem)
        if meas_what == "demand":
            de = 0.5 * (demp[meas_idx] - demm[meas_idx])
        elif meas_what == "head":
            de = 0.5 * (hp[meas_idx] - hm[meas_idx])
        else:
            de = dlt
        de_list.append(de)
        dl_list.append(Lp - Lm)
    restore_fn()
    d = np.asarray(de_list)
    y = np.asarray(dl_list)
    X = np.stack([2.0 * d, d ** 3], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    g_ls = float(coef[0])
    g_big = float(y[0] / (2.0 * d[0]))
    return g_ls, g_big, d, re_max


def main():
    torch.set_default_dtype(torch.float64)
    net = Net.load(os.path.join(ROOT, "data", "reference"), "city_d_emit")
    s = GGASolver(net, mode="dense", inp_path=INP)
    d0 = net.demand_cfs_at(0)
    rh0 = np.nan_to_num(net.reservoir_head_ft_at(0))
    ke0 = s.node_ke_default.cpu().numpy().copy()
    r0 = s.r_hw.detach().cpu().numpy().copy()
    rng = np.random.default_rng(SEED)
    w = rng.normal(size=s.Nj)
    w_t = torch.tensor(w, dtype=torch.float64)
    print(f"=== city_d_emit: N={s.N} Nj={s.Nj} L={s.L} "
          f"emitters={int((ke0 > 0).sum())} Qexp={s.qexp} Hexp={s.hexp} ===")

    # ---- 解析梯度（ImplicitGGASolve，显式 accuracy=1e-12 + Newton 精抛光）----
    dT = torch.tensor(d0, requires_grad=True)
    rhT = torch.tensor(rh0, requires_grad=True)
    keT = torch.tensor(ke0, requires_grad=True)
    rT = torch.tensor(r0, requires_grad=True)
    head, _, _ = ImplicitGGASolve.apply(dT, rhT, keT, rT, s, 1e-12, GGA_MI, 3)
    (w_t * head[s.junc_nodes]).sum().backward()
    gA = dict(demand=dT.grad.numpy(), rh=rhT.grad.numpy(),
              ke=keT.grad.numpy(), r=rT.grad.numpy())
    sol = solve_polished(s, d0, rh0, ke0, r0, accuracy=1e-12,
                         max_iter=GGA_MI, polish_steps=3)
    print(f"解析侧: GGA iters={int(sol['iters'][0])} 精抛光后 ||F||inf = "
          f"{sol['resid_inf'][0]:.3e}")
    h_ana = sol["head"][0]
    q_ana = sol["q"][0]
    L_ana = float(w @ h_ana[s.junc_nodes])

    # ---- 坐标候选 ----
    em_nodes = np.where(ke0 > 0.0)[0]
    ncat = np.bincount(net.dem_node, minlength=net.N)
    dem_cand = np.array([i for i in s.junc_nodes
                         if d0[i] > 0.0 and ncat[i] == 1 and ke0[i] == 0.0])
    # 信号强度 |g|*x 最大的 3 个（FD 信噪比 ~ |g|*H, H 正比 x；见文件头噪声分析）
    dem_pick = dem_cand[np.argsort(-np.abs(gA["demand"][dem_cand]
                                           * d0[dem_cand]))[:3]]
    # 非钳位管道（gradcheck_3way 同款掩码：钳位支 dphi/dr=0，hydcoeffs.c:554-558）
    hg_fric = s.hexp * r0 * np.abs(q_ana) ** (s.hexp - 1.0)
    pipe_ok = (s.is_pipe.cpu().numpy() & ~s.closed_np
               & (hg_fric > 10.0 * s.rqtol) & (np.asarray(net.roughness) > 0))
    rough_c = np.asarray(net.roughness)
    gA_rC_all = np.where(pipe_ok, gA["r"] * (-s.hexp) * r0
                         / np.where(rough_c > 0, rough_c, 1.0), 0.0)
    rough_pick = int(np.argmax(np.abs(gA_rC_all)))
    res_pick = [int(i) for i in s.fixed_nodes]

    # ---- EPANET：ACCURACY->1e-8、TRIALS->200、DAMPLIMIT->1e-6，ID 映射 ----
    en = er.Epanet(INP)
    try:
        en.set_option(er.EN_ACCURACY, ACC_EP)
        en.set_option(er.EN_TRIALS, 200)    # 基准；FD 评估内按 CAPS 轮换
        en.set_option(EN_DAMPLIMIT, 1e-6)   # 实测把极限环 relerr 8.6e-8 -> 3.5e-8
        idx_of = {nid: i + 1 for i, nid in enumerate(en.node_ids())}
        nperm = np.array([idx_of[nid] for nid in net.node_id], dtype=np.int64)
        lidx_of = {lid: i + 1 for i, lid in enumerate(en.link_ids())}
        acc_rb = ctypes.c_double()
        en._check(en.lib.EN_getoption(en._ph, er.EN_ACCURACY, ctypes.byref(acc_rb)),
                  "EN_getoption(ACCURACY)")
        print(f"EPANET: ACCURACY 回读 = {acc_rb.value:.1e}（EN_setoption 未被钳位）")

        # ---- 基准解与两侧一致性 ----
        h_base, dem_base, it0, re0, warn0 = solve_t0(en, nperm)
        L_base = float(w @ h_base[s.junc_nodes])
        print(f"EPANET t=0 基准: iters={it0} relerr={re0:.2e} warn={warn0} "
              f"（极限环平台，达不到 1e-8，见文件头） L={L_base:.10e}")
        print(f"基准一致性: max|dH|(EPANET vs 解析) = "
              f"{np.abs(h_base - h_ana).max():.3e} ft, |dL| = "
              f"{abs(L_base - L_ana):.3e}")

        # Ke_int(C) 换算链自检（input1.c:567-573 vs parse.py:368）
        ke_chk = 0.0
        C_user = {}
        for i in em_nodes:
            C_user[i] = get_node(en, int(nperm[i]), er.EN_EMITTER)
            ke_pred = (en._ucf_flow ** s.qexp / en._ucf_press) / C_user[i] ** s.qexp
            ke_chk = max(ke_chk, abs(ke_pred - ke0[i]) / ke0[i])
        print(f"Ke_int(C) 换算链自检: max 相对偏差 = {ke_chk:.2e}")

        # emitter 坐标按 |gC*C| 选 3（信噪比 ~ |gC|*H, H=0.15C）
        gC_em = {i: gA["ke"][i] * (-s.qexp * ke0[i] / C_user[i]) for i in em_nodes}
        em_pick = sorted(em_nodes, key=lambda i: -abs(gC_em[i] * C_user[i]))[:3]

        rows = []     # (kind, id, x0, gFD, gA, g_big, relerr_max)

        # ---- 1) 3 个 emitter 用户系数 C（步长基准 0.30C，信噪比所需）----
        for i in [int(x) for x in em_pick]:
            j = int(nperm[i])
            C0 = C_user[i]
            gfd, gbig, dde, rem = fd_slope(
                en, nperm, s.junc_nodes, w,
                lambda dlt, j=j, C0=C0: en.set_node_value(j, er.EN_EMITTER, C0 + dlt),
                lambda j=j, C0=C0: en.set_node_value(j, er.EN_EMITTER, C0),
                0.30 * C0)
            rows.append(("emitter C", net.node_id[i], C0, gfd, gC_em[i], gbig, rem))

        # ---- 2) 3 个基础需水（步长基准 0.04*基值；分母 = EN_DEMAND 实测差）----
        for i in [int(x) for x in dem_pick]:
            j = int(nperm[i])
            b_int = get_node(en, j, er.EN_BASEDEMAND) / en._ucf_flow
            gfd, gbig, dde, rem = fd_slope(
                en, nperm, s.junc_nodes, w,
                lambda dlt, j=j, b=b_int: en.set_node_value(j, er.EN_BASEDEMAND,
                                                            b + dlt),
                lambda j=j, b=b_int: en.set_node_value(j, er.EN_BASEDEMAND, b),
                0.04 * b_int, meas_idx=i, meas_what="demand")
            rows.append((f"demand(f0={d0[i] / b_int:.3f})", net.node_id[i], d0[i],
                         gfd, gA["demand"][i], gbig, rem))

        # ---- 3) 水库水头（步长基准 0.002*基值；分母 = EN_HEAD 实测差）----
        for i in res_pick:
            j = int(nperm[i])
            E_int = get_node(en, j, er.EN_ELEVATION) / en._ucf_head
            gfd, gbig, dde, rem = fd_slope(
                en, nperm, s.junc_nodes, w,
                lambda dlt, j=j, E=E_int: en.set_node_value(j, er.EN_ELEVATION,
                                                            E + dlt),
                lambda j=j, E=E_int: en.set_node_value(j, er.EN_ELEVATION, E),
                2e-3 * abs(E_int), meas_idx=i, meas_what="head")
            rows.append(("res head", net.node_id[i], rh0[i],
                         gfd, gA["rh"][i], gbig, rem))

        # ---- 4) 1 根管道 roughness（H-W 用户 C；步长基准 0.04C）----
        k = rough_pick
        jl = int(lidx_of[net.link_id[k]])
        C0 = get_link(en, jl, er.EN_ROUGHNESS)
        gfd, gbig, dde, rem = fd_slope(
            en, nperm, s.junc_nodes, w,
            lambda dlt, jl=jl, C0=C0: en.set_link_value(jl, er.EN_ROUGHNESS,
                                                        C0 + dlt),
            lambda jl=jl, C0=C0: en.set_link_value(jl, er.EN_ROUGHNESS, C0),
            0.04 * C0)
        ga = gA["r"][k] * (-s.hexp * r0[k] / C0)          # dr/dC = -Hexp*r/C
        rows.append(("rough C", net.link_id[k], C0, gfd, ga, gbig, rem))
    finally:
        en.close()

    # ---- 逐坐标对比表 ----
    print("=" * 100)
    print(f"{'θ':>16} {'ID':>6} {'x0':>12} {'EPANET-FD':>18} {'解析':>18} "
          f"{'相对误差':>10} {'单步偏离':>10} {'relerr':>8}")
    ok = True
    worst = (-1.0, None)
    for kind, nid, x0, gfd, ga, gbig, rem in rows:
        rel = abs(gfd - ga) / max(abs(gfd), 1e-12)
        drift = abs(gbig - gfd) / max(abs(gfd), 1e-12)
        ok = ok and rel < TOL
        if rel > worst[0]:
            worst = (rel, (kind, nid))
        print(f"{kind:>16} {nid:>6} {x0:>12.5g} {gfd:>18.10e} {ga:>18.10e} "
              f"{rel:>10.2e} {drift:>10.2e} {rem:>8.1e}"
              f"{'' if rel < TOL else '  <-- 超限'}")
    print("=" * 100)
    print(f"最差坐标: {worst[1]} rel={worst[0]:.2e}")
    print(f"外部对拍(EPANET-FD vs 解析, 门槛 {TOL:.0e}): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

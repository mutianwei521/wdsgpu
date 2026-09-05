# -*- coding: utf-8 -*-
"""regression_gpu.py - cuDSS 通路的不变量回归（P4 / 审计 R2）。

与 regression_all.py **分开**：这里的每一条都需要真实的 CUDA + nvmath(cuDSS)，
CPU 上无法执行。**缺 CUDA 或缺 nvmath 时明确 SKIP（退出码 2）并打印缺的是什么，
绝不静默 PASS** - 静默 PASS 才是这类"只在特定硬件上才成立"的断言最危险的失效
方式（本机 RTX 5060 Laptop 有 CUDA 无 nvmath，就是那个会被静默放行的环境）。

退出码：0 = PASS，1 = FAIL，2 = SKIP（环境不满足，已打印原因）。

四条断言（T1-T3 出自审计 data/p3_adversarial_wip.txt §9 R2；
T4 出自 P4 终审 data/p4_adversarial_wip.txt §2 实测到的守卫缺口）：

  T1  迭代数逐项相等：cudss 与 dense 在同一批场景上 `iters` **逐样本 int 相等**。
      线性解的精度不应改变牛顿轨迹；若不等，先查精化步数（验收 2 的原判据）。
      顺带报 max|ΔH|，但**不作门槛**（改判据的理由见 sparse_gpu_plan.md §7.2）。

  T2  `bwd_refactorize == 0`（slots 足够时）：展开 K 步、cudss_grad_slots=K、
      cudss_cache_max>=slots，反向必须零重分解，且 factorize 恰等于前向线性解
      次数 K。**同时验反面**：slots=1 时必须是 factorize==2K−1、
      bwd_refactorize==K−1 - 否则这个"0"可能只是计数器压根没在数。
      注：T1 单独**不足以**守住线性解的精度 - 审计实测把 cudss_refine 由 2 改成 0，
      10 个配置的迭代数一个没变（T1 全绿），是 T2 里 `solve == K*(1+cudss_refine)`
      这一项把它抓住的（p4_adversarial_wip.txt §3.4a）。

  T3  复用即逐位核对：在 `_cudss_adjoint` 入口拦截，凡是**被测代码自己判定**
      "可以复用前向分解"的那一次，就把 cuDSS 手上那块缓冲 `st["vals"]` 与 autograd
      保存的 `data` **逐位**比。stale 必须为 0，且事件数必须 > 0（不能靠"一次都没
      复用"蒙混）。**"被测代码自己判定"是要害**：探针早先的版本把复用谓词从
      `solver._cudss_adjoint` 抄了一份自己算，于是谁把 solver 侧的谓词放宽
      （例如丢掉 gen 检查），那些真正被错误复用的事件根本进不了比对，T3 全绿
      （p4_adversarial_wip.txt §3.4b）。现改为**观察** solver 的副作用：
      `bwd_reuse` / `bwd_refactorize` 两个计数器在这一次调用里的增量。

  T4  cuDSS 拿到的 csr_data 逐位对称：拦 `_cudss_load`（[B,nnz] 的值写进 cuDSS
      缓冲的唯一入口，前向每轮牛顿与反向重分解都过它），按转置槽比
      `max|data − data^T| == 0.0`，**无容差**。
      为什么必须在这里单列一条：regression_all 第 ⑨ 项 `check_symmetry.py` 是
      CPU-only 的，`linear_solver == "cudss"` 分支（含 emitter 对角那次
      `index_add(1, A_csr_diag, em/hgrad_e)`）它永远走不到，那段装配此前**零守卫**。
      终审实测：往那条分支注入 1e-9 相对量的单侧写，max|A−A^T| 到 3.647e-05
      （Net3），迭代数一个没变、max|dH| 只有 2.075e-06 ft，T1/T2/T3 **全绿放行**；
      要放大到 1e-2、把牛顿轨迹打崩（ky4 迭代 7→100）才被 T1 间接看见
      （p4_adversarial_wip.txt §2）。整条伴随都建在"A 对称 ⇒ 用前向那一次分解
      直接回代"上，所以这条按逐位判，1 ULP 也算红。

  发布审计（data/prv_release_audit_wip.txt ①/洞 B）补充：T1-T4 原网表
  （Hanoi/Net3/Modena/City_D/ky4）**没有任何含 PRV 的网**、不走状态机 -
  ACTIVE PRV 的 CBIG=1e8 罚函数行在 cuDSS 实收数据上零守卫，"ACTIVE 单侧写
  CBIG 非对角"类移植错误（MF 变异体）在这 50 项上全绿逃逸。现 NETS_PRV
  （L-TOWN，dense_status_machine）进 T1（加终态逐元素判据）与 T4（要求批内
  ACTIVE PRV 槽 >0，否则罚函数行根本没被测到，直接 FAIL）；T2/T3 打印跳过
  原因（solve_unrolled 对 PRV 构造期 raise，由 CPU 回归把守）。变异体自证
  见 data/holes_abc_wip.txt（MF 在 T4 L-TOWN 上必须红）。

用法（集群）：`sbatch regression_gpu.sh`，或直接 `python3 -X utf8 regression_gpu.py`。
网络文件按序找：$DGGA_NETS → <repo>/networks/public → <脚本目录>/p2nets
（每个目录先看其下 _cleaned/ 再看本级）。
"""

import os
import sys
import traceback

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DT = torch.float64

# (标签, 文件名, K, [B...])
NETS = [("Hanoi", "Hanoi.inp", 6, [1, 8]),
        ("Net3", "Net3.inp", 6, [1, 8]),
        ("Modena", "Modena.inp", 6, [8, 64]),
        ("City_D", "City_D.inp", 5, [8, 64]),
        ("ky4", "ky4.inp", 5, [8, 64])]

# 含 PRV 的网（发布审计洞 B：上述 T1-T4 网表无 PRV 网、不走状态机，ACTIVE PRV
# 的 CBIG 罚函数行在 cuDSS 实收数据上此前零守卫 - MF 类"单侧写 CBIG"变异体
# 50/50 全绿逃逸，见 data/prv_release_audit_wip.txt ①）。这些网走
# dense_status_machine=True + solve(status_machine=True)：
#   T1：iters 逐样本相等**且终态状态向量逐元素相等**（dense vs cudss）；
#   T4：SymProbe 拦 _cudss_load，含 ACTIVE PRV 帧的 csr_data 必须逐位对称。
#   T2/T3 不适用：solve_unrolled 对 PRV 网构造期 raise（该拒绝本身由 CPU 侧
#   回归 regression_all ⑪ 把守），跳过时打印原因，不静默。
NETS_PRV = [("L-TOWN", "L-TOWN.inp", None, [8, 64])]


def netdir():
    for d in (os.environ.get("DGGA_NETS"),
              os.path.join(ROOT, "networks", "public"),
              os.path.join(HERE, "p2nets"),
              os.path.join(os.getcwd(), "p2nets")):
        if d and os.path.isdir(d):
            return d
    return None


def find_inp(fn):
    """networks/public 优先 _cleaned 版（与 CPU 回归同源），其余按 netdir。"""
    for d in (os.path.join(netdir(), "_cleaned"), netdir()):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn + " (netdir=" + str(netdir()) + ")")


def preflight():
    """返回 (ok, 原因)。不满足就 SKIP，并把缺的东西说清楚。"""
    if not torch.cuda.is_available():
        return False, ("无 CUDA 设备（torch.cuda.is_available() == False）。"
                       "cuDSS 通路只在 GPU 上存在，本项全部 SKIP。")
    try:
        import nvmath  # noqa: F401
        from nvmath.sparse.advanced import DirectSolver  # noqa: F401
    except Exception as e:                                # noqa: BLE001
        return False, ("有 CUDA（%s）但 nvmath.sparse.advanced.DirectSolver "
                       "导入失败：%r。请 pip install nvmath-python[cu12]。"
                       "**本项 SKIP，不是 PASS** - cuDSS 的四条不变量本次未被"
                       "验证过。" % (torch.cuda.get_device_name(0), e))
    if netdir() is None:
        return False, "找不到网络目录（$DGGA_NETS / networks/public / p2nets）。"
    missing = []
    for _s, fn, _k, _b in NETS + NETS_PRV:
        try:
            find_inp(fn)
        except FileNotFoundError:
            missing.append(fn)
    if missing:
        return False, ("网络目录 %s 缺文件：%s（PRV 网 L-TOWN 是洞 B 修复的"
                       "关键覆盖，缺了必须显式 SKIP 而非静默少跑）。"
                       % (netdir(), missing))
    return True, ""


# ----------------------------------------------------------------------
def build(fn, B, seed=20260822, emit=True, sm=False):
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    p = find_inp(fn)
    net = parse_inp(p)
    if sm:                       # PRV 网：批量状态机路径（emitter 不注入）
        s = GGASolver(net, device="cuda", dtype=DT, mode="dense", inp_path=p,
                      dense_status_machine=True)
    else:
        s = GGASolver(net, device="cuda", dtype=DT, mode="dense", inp_path=p,
                      dense_tank_bound_check=False)
    g = np.random.default_rng(seed)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = 0.5 * (np.asarray(net.tank_hmin)[:tn.size]
                         + np.asarray(net.tank_hmax)[:tn.size])
    rh0 = np.nan_to_num(rh0)
    D = torch.as_tensor(d0[None, :] * g.uniform(.9, 1.1, (B, d0.size)),
                        dtype=DT, device="cuda")
    R = torch.as_tensor(np.repeat(rh0[None, :], B, 0)
                        + g.uniform(-1., 1., (B, rh0.size))
                        * (np.asarray(net.node_type) != 0)[None, :],
                        dtype=DT, device="cuda")
    ke = np.asarray(net.node_ke, dtype=np.float64).copy()
    if emit:
        jn = np.asarray(s.junc_nodes)
        ke[jn[g.random(jn.size) < .2]] = 1e-3
    KE = torch.as_tensor(np.repeat(ke[None, :], B, 0), dtype=DT, device="cuda")
    W = torch.as_tensor(g.normal(0, 1, (B, s.Nj)), dtype=DT, device="cuda")
    return net, s, D, R, KE, W


# ----------------------------------------------------------------------
# T3 探针：**观察**被测代码的复用判定，不重实现判据
# ----------------------------------------------------------------------
class ReuseProbe:
    """拦 `_cudss_adjoint`，"这一次到底复用了没有"取自 solver 自己的计数器增量。

    复用分支唯一的可观察副作用是 `bwd_reuse += 1`，退路分支是
    `bwd_refactorize += 1`。探针在调用前后读 `solver._cudss_counters`，用增量
    认定走了哪条路 - 于是**谓词怎么改，探针就跟着看到什么**，不会像"抄一份
    谓词自己算"那样在 solver 侧放宽判据时把错误复用的事件筛掉
    （p4_adversarial_wip.txt §3.4b）。

    两个增量必须恰是 (1,0) 或 (0,1)，否则记 incoh：谁把计数器静音，这里就会
    看见，而不是被当成"没复用"悄悄放过。
    快照必须在**调用前**取 - 退路分支会 `_cudss_load(st, data)` 把缓冲刷成
    data，事后再比就永远相等。
    """

    def __init__(self):
        from dgga.solver import GGASolver
        self.cls = GGASolver
        self.raw = GGASolver._cudss_adjoint
        self.reset()

    def reset(self):
        self.events = self.reuse = self.stale = self.incoh = 0
        self.worst = 0.0

    def __enter__(self):
        pr = self

        def probe(self_, data, g, B, st, gen, slot):
            pr.events += 1
            v = st.get("vals")
            snap = v.detach().clone() if v is not None else None
            c = self_._cudss_counters
            b_reu, b_ref = int(c["bwd_reuse"]), int(c["bwd_refactorize"])
            out = pr.raw(self_, data, g, B, st, gen, slot)
            d_reu = int(c["bwd_reuse"]) - b_reu
            d_ref = int(c["bwd_refactorize"]) - b_ref
            if (d_reu, d_ref) not in ((1, 0), (0, 1)):
                pr.incoh += 1
            if d_reu == 1:                       # 被测代码自己说：复用了
                pr.reuse += 1
                if snap is None:
                    pr.stale += 1                # 声称复用却没有缓冲
                else:
                    pr.worst = max(pr.worst,
                                   float((snap - data).abs().max()))
                    if not bool(torch.equal(snap, data)):
                        pr.stale += 1
            return out

        self.cls._cudss_adjoint = probe
        return self

    def __exit__(self, *e):
        self.cls._cudss_adjoint = self.raw
        return False


# ----------------------------------------------------------------------
# T4 探针：cuDSS 真正收到的 [B,nnz] 值，按转置槽逐位比
# ----------------------------------------------------------------------
def transpose_slots(s):
    """CSR data 下标 k → 转置槽 t(k)（即 A[col[k], row[k]] 的下标）。

    位型由拓扑定死（四段链路贡献成对入表 ∪ 全部对角），必须结构对称，
    因此 t 必须是一个精确的对合。升序、有对偶、是对合这三件事都**验出来**
    再返回 - 映射不对的话，"逐位对称"那句话就是在比错位置。
    """
    Nj = int(s.Nj)
    row = s.A_csr_row.detach().cpu().numpy().astype(np.int64)
    col = s.A_csr_col.detach().cpu().numpy().astype(np.int64)
    key = row * Nj + col                        # CSR 标准次序 ⇒ 严格升序
    if key.size == 0 or not bool(np.all(np.diff(key) > 0)):
        raise AssertionError("CSR 位型不是严格升序，转置槽映射无效")
    tk = col * Nj + row
    pos = np.searchsorted(key, tk)
    inb = pos < key.size
    if not bool(inb.all()) or not np.array_equal(key[np.where(inb, pos, 0)],
                                                 tk):
        raise AssertionError("CSR 位型不是结构对称：有槽找不到转置对偶")
    if not np.array_equal(pos[pos], np.arange(key.size)):
        raise AssertionError("转置槽映射不是对合")
    return torch.as_tensor(pos, dtype=torch.int64, device=s.A_csr_row.device)


class SymProbe:
    """拦 `_cudss_load` - [B,nnz] 的值写进 cuDSS 缓冲的唯一入口。

    `_cudss_forward`（前向每一轮牛顿的装配）与 `_cudss_adjoint` 的重分解退路
    都经过它，所以拦这一处就等于看住了"cuDSS 实际拿到的那份 A"，包括 CPU
    不可达的那段 emitter 对角 index_add。判据：max|data − data[t]| == 0.0。
    """

    def __init__(self, tidx):
        from dgga.solver import GGASolver
        self.cls = GGASolver
        self.raw = GGASolver._cudss_load         # staticmethod ⇒ 取到原函数
        self.tidx = tidx
        self.rounds = 0
        self.bad = 0
        self.worst = 0.0

    def __enter__(self):
        pr = self

        def probe(st, data):
            with torch.no_grad():
                d = data.detach()
                w = float((d - d.index_select(1, pr.tidx)).abs().max())
            pr.rounds += 1
            pr.worst = max(pr.worst, w)
            if w != 0.0:
                pr.bad += 1
            return pr.raw(st, data)

        self.cls._cudss_load = staticmethod(probe)
        return self

    def __exit__(self, *e):
        self.cls._cudss_load = staticmethod(self.raw)
        return False


# ----------------------------------------------------------------------
def t1_iters(rows):
    """T1：cudss vs dense 的迭代数逐样本相等（PRV 网另加终态逐元素相等）。"""
    print("\n【T1 迭代数逐项相等（cudss vs dense；PRV 网走状态机并加终态判据）】")
    bad = 0
    for stem, fn, _K, Bs in NETS + NETS_PRV:
        sm = (stem, fn, _K, Bs) in NETS_PRV
        for B in Bs:
            try:
                net, s, D, R, KE, _W = build(fn, B, sm=sm)
                kw = dict(status_machine=True) if sm else dict(ke_int=KE)
                with torch.no_grad():
                    od = s.solve(D, R, **kw)
                    oc = s.solve(D, R, assemble="csr",
                                 linear_solver="cudss", **kw)
                itd = od["iters"].reshape(-1).tolist()
                itc = oc["iters"].reshape(-1).tolist()
                same = itd == itc
                dH = float((od["head_ft"] - oc["head_ft"]).abs().max())
                st_same = True
                st_note = ""
                if sm:
                    st_same = bool(torch.equal(od["status"], oc["status"]))
                    st_note = " 终态逐元素=%s" % ("是" if st_same else "否")
                ok = same and st_same
                bad += 0 if ok else 1
                print("  %-8s B=%-4d 迭代 dense=%s cudss=%s 逐项相等=%s%s "
                      "| max|dH|=%.3e ft（仅报告，非门槛） %s"
                      % (stem, B, _brief(itd), _brief(itc), "是" if same else "否",
                         st_note, dH, "PASS" if ok else "<-- FAIL"), flush=True)
                rows.append(("T1 %s B=%d" % (stem, B), ok))
                s.cudss_free()
                del s
            except Exception:
                bad += 1
                rows.append(("T1 %s B=%d" % (stem, B), False))
                print("  %-8s B=%-4d <-- FAIL %s"
                      % (stem, B, traceback.format_exc().strip()
                         .split("\n")[-1][:120]), flush=True)
    return bad


def _brief(v):
    u = sorted(set(v))
    return str(u[0]) if len(u) == 1 else ("{%s}" % ",".join(map(str, u[:4])))


def t2_refactorize(rows):
    """T2：slots>=K 零重分解；slots=1 时计数器如实记 K-1（证明"0"不是空的）。"""
    print("\n【T2 bwd_refactorize（slots 足够时必须为 0；slots=1 时必须为 K−1）】")
    print("  （PRV 网 %s 不适用：solve_unrolled 对 PRV 构造期 raise，该拒绝由 "
          "CPU 回归 regression_all ⑪ 把守）"
          % [n[0] for n in NETS_PRV])
    bad = 0
    for stem, fn, K, Bs in NETS:
        B = Bs[0]
        for slots in (K, 1):
            try:
                from dgga.autodiff import solve_unrolled
                net, s, D, R, KE, W = build(fn, B)
                s.cudss_grad_slots = slots
                s.cudss_cache_max = max(8, slots + 2)
                s.cudss_counters(reset=True)
                Dg = D.clone().requires_grad_(True)
                out = solve_unrolled(s, Dg, R, ke=KE, K=K, assemble="csr",
                                     linear_solver="cudss")
                (out["head_ft"][:, s.junc_nodes_t] * W).sum().backward()
                c = s.cudss_counters()
                exp_fac = K if slots >= K else 2 * K - 1
                exp_ref = 0 if slots >= K else K - 1
                exp_reu = K if slots >= K else 1
                ok = (c["bwd_refactorize"] == exp_ref
                      and c["factorize"] == exp_fac
                      and c["bwd_reuse"] == exp_reu
                      and c["solve"] == K * (1 + s.cudss_refine)
                      and c["bwd_solve"] == K * (1 + s.cudss_grad_refine)
                      and torch.isfinite(Dg.grad).all())
                bad += 0 if ok else 1
                print("  %-8s B=%-4d K=%d slots=%-2d | fac=%-3d(期望%-3d) "
                      "bwd_refac=%-2d(期望%d) bwd_reuse=%-2d(期望%d) "
                      "solve=%-3d bwd_solve=%-3d  %s"
                      % (stem, B, K, slots, c["factorize"], exp_fac,
                         c["bwd_refactorize"], exp_ref, c["bwd_reuse"], exp_reu,
                         c["solve"], c["bwd_solve"],
                         "PASS" if ok else "<-- FAIL"), flush=True)
                rows.append(("T2 %s slots=%d" % (stem, slots), ok))
                s.cudss_free()
                del s
            except Exception:
                bad += 1
                rows.append(("T2 %s slots=%d" % (stem, slots), False))
                print("  %-8s slots=%-2d <-- FAIL %s"
                      % (stem, slots, traceback.format_exc().strip()
                         .split("\n")[-1][:120]), flush=True)
    return bad


def t3_reuse_bitwise(rows):
    """T3：每次声称复用，cuDSS 手上的缓冲与 saved data 必须逐位相等。"""
    print("\n【T3 复用即逐位核对（stale 必须 0，且事件数必须 >0）】")
    print("  （PRV 网 %s 不适用：同 T2，solve_unrolled 对 PRV 构造期 raise）"
          % [n[0] for n in NETS_PRV])
    bad = 0
    for stem, fn, K, Bs in NETS:
        for B in Bs:
            for slots in (K, 1):
                try:
                    from dgga.autodiff import solve_unrolled
                    net, s, D, R, KE, W = build(fn, B)
                    s.cudss_grad_slots = slots
                    s.cudss_cache_max = max(8, slots + 2)
                    Dg = D.clone().requires_grad_(True)
                    with ReuseProbe() as pr:
                        out = solve_unrolled(s, Dg, R, ke=KE, K=K,
                                             assemble="csr",
                                             linear_solver="cudss")
                        (out["head_ft"][:, s.junc_nodes_t] * W).sum().backward()
                        ev, ru, stale, worst, ic = (pr.events, pr.reuse,
                                                    pr.stale, pr.worst,
                                                    pr.incoh)
                    ok = (stale == 0 and ru > 0 and ev == K and ic == 0)
                    bad += 0 if ok else 1
                    print("  %-8s B=%-4d slots=%-2d | 伴随事件=%-2d 其中复用=%-2d "
                          "stale=%d incoh=%d worstDelta=%.3e  %s"
                          % (stem, B, slots, ev, ru, stale, ic, worst,
                             "PASS" if ok else "<-- FAIL"), flush=True)
                    rows.append(("T3 %s B=%d slots=%d" % (stem, B, slots), ok))
                    s.cudss_free()
                    del s
                except Exception:
                    bad += 1
                    rows.append(("T3 %s B=%d slots=%d" % (stem, B, slots), False))
                    print("  %-8s B=%-4d slots=%-2d <-- FAIL %s"
                          % (stem, B, slots, traceback.format_exc().strip()
                             .split("\n")[-1][:120]), flush=True)
    return bad


def t4_csr_symmetric(rows):
    """T4：cuDSS 实际收到的 csr_data 逐位对称（含 CPU 不可达的 emitter 对角段；
    PRV 网走状态机前向 - 含 ACTIVE PRV 的 CBIG 罚函数行，洞 B 修复）。"""
    print("\n【T4 cuDSS 收到的 csr_data 逐位对称（max|A−A^T| 必须 ==0.0，无容差）】")
    bad = 0
    for stem, fn, K, Bs in NETS + NETS_PRV:
        sm = (stem, fn, K, Bs) in NETS_PRV
        for B in Bs:
            try:
                from dgga.autodiff import solve_unrolled
                net, s, D, R, KE, W = build(fn, B, sm=sm)
                tidx = transpose_slots(s)
                with SymProbe(tidx) as pr:
                    with torch.no_grad():        # 前向：每一轮牛顿的装配
                        if sm:                   # PRV：状态机（ACTIVE 罚函数行）
                            o = s.solve(D, R, status_machine=True,
                                        assemble="csr", linear_solver="cudss")
                        else:
                            s.solve(D, R, ke_int=KE, assemble="csr",
                                    linear_solver="cudss")
                    n_fwd = pr.rounds
                    if (not sm) and B == Bs[0]:  # 反向：展开 K 步 + backward
                        Dg = D.clone().requires_grad_(True)
                        out = solve_unrolled(s, Dg, R, ke=KE, K=K,
                                             assemble="csr",
                                             linear_solver="cudss")
                        (out["head_ft"][:, s.junc_nodes_t]
                         * W).sum().backward()
                    n_all, worst, nbad = pr.rounds, pr.worst, pr.bad
                n_act = 0
                if sm:                           # ACTIVE PRV 在场才算测到了 CBIG 行
                    lt_ = torch.as_tensor(np.asarray(net.link_type),
                                          device=o["status"].device)
                    n_act = int(((o["status"] == 4)
                                 & (lt_ == 3)[None, :]).sum().item())
                ok = (n_all > 0 and nbad == 0 and worst == 0.0
                      and (n_act > 0 or not sm))
                bad += 0 if ok else 1
                print("  %-8s B=%-4d | 装配次数=%-3d(前向 %-2d) nnz=%-6d "
                      "max|A−A^T|=%.3e 非对称次数=%d%s  %s"
                      % (stem, B, n_all, n_fwd, s.A_csr_nnz, worst, nbad,
                         (" ACTIVE-PRV槽=%d" % n_act) if sm else "",
                         "PASS" if ok else "<-- FAIL"), flush=True)
                rows.append(("T4 %s B=%d" % (stem, B), ok))
                s.cudss_free()
                del s
            except Exception:
                bad += 1
                rows.append(("T4 %s B=%d" % (stem, B), False))
                print("  %-8s B=%-4d <-- FAIL %s"
                      % (stem, B, traceback.format_exc().strip()
                         .split("\n")[-1][:120]), flush=True)
    return bad


def main():
    print("=" * 96)
    print("regression_gpu.py - cuDSS 通路不变量回归（审计 R2 + P4 终审 T4）")
    ok, why = preflight()
    if not ok:
        print("总判定: SKIP（退出码 2）")
        print("原因: " + why)
        print("=" * 96)
        return 2
    dev = torch.cuda.get_device_name(0)
    import nvmath
    print("设备: %s | torch %s | nvmath %s | 节点 %s"
          % (dev, torch.__version__, getattr(nvmath, "__version__", "?"),
             os.popen("hostname").read().strip()))
    print("网络目录: %s" % netdir())
    print("=" * 96)
    rows = []
    bad = (t1_iters(rows) + t2_refactorize(rows) + t3_reuse_bitwise(rows)
           + t4_csr_symmetric(rows))
    n = len(rows)
    print("\n" + "=" * 96)
    print("总判定: %s （%d/%d 项通过）" % ("PASS" if bad == 0 else "FAIL",
                                          n - bad, n))
    if bad:
        print("未通过项: " + ", ".join(k for k, v in rows if not v))
    print("=" * 96)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

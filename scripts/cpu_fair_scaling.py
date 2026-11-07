# -*- coding: utf-8 -*-
"""cpu_fair_scaling.py - 公平 CPU 对比 v2（2026-10-23，回应用户意见与 R2-M5/M6）。

v1（cpu_compare_hanoi.py）的问题：用单次 20 帧调用的墙钟比较 C 引擎与 Python/PyTorch 实现，
比到的是实现语言的固定调用开销；EPANET 侧每次 EN_solveH 还要重新分配求解器、重做排序、写结果文件。
v2：
 1. EPANET 用最强的常规用法：EN_openH 一次；每工况 EN_initH(10)（冷启动，与 EN_solveH 同解）+ EN_runH，
    有限差分的扰动解用 EN_initH(0) 从同一工况基解热启动；另累计纯 C 求解时间（EN_initH+EN_runH）作为下界。
 2. 工况数 B 从 1 扫到 4096：单工况延迟 t(1)、边际成本 c（大 B 斜率）分开报告。
 3. 梯度：EPANET 前向/中心差分（热启动，B<=256 另报冷启动）vs HydroGrad 一次批量前向 + 一次伴随。
 4. 端到端校准：论文原题 paper20（与 v1 同题面）+ 大量调用版 rand200；同一 L-BFGS-B 设置，报告达标时间。
 5. 全部单线程；两者对工况都可并行，单核吞吐是公平单位。正则项梯度两臂都用解析式。
运行：python -X utf8 scripts/cpu_fair_scaling.py --stage scale | calib [--sets paper20,rand200] [--seeds 100,101,102,103,104]
（calib 的 --seeds 是配对的噪声种子：同一种子下两臂用同一份观测；10-23 补 5 个配对种子）
"""
import argparse
import ctypes
import json
import os
import shutil
import sys
import tempfile
import time

import cpu_compare_hanoi as v1          # 复用 v1 的题面、常量、环境设置（单线程、float64）
import numpy as np
import torch
from scipy.optimize import minimize

cal, er = v1.cal, v1.er
from dgga.autodiff import ImplicitGGASolve, solve_polished   # noqa: E402
from dgga.calib import hw_resistance                          # noqa: E402

ROOT, LAM, SIGMA, C0 = v1.ROOT, v1.LAM, v1.SIGMA, v1.C0
REG = v1.REG_SCALE
JSON = os.path.join(ROOT, "data", "cpu_fair_scaling_hanoi.json")
RPT = os.path.join(ROOT, "data", "cpu_fair_scaling_hanoi_report.txt")
H_FD = 1e-5      # v1 步长扫描：中心差分在 1e-5 最准；前向差分误差随 h 单调降，同取 1e-5


def fast_cpu():
    """Windows：退出 EcoQoS 后台降频、提高优先级，并用微基准挑一个最快的逻辑核绑定
    （混合架构 CPU 避开能效核；两臂在同一进程、同一个核上测）。"""
    if os.name != "nt":
        return {}
    import ctypes.wintypes as wt
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wt.HANDLE
    k32.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    k32.SetPriorityClass.argtypes = [wt.HANDLE, wt.DWORD]
    k32.SetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.c_size_t]
    h = k32.GetCurrentProcess()

    class PPT(ctypes.Structure):
        _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]
    st = PPT(1, 0x1, 0x0)            # EXECUTION_SPEED 受控且关闭 = 不做后台降频
    ok_t = bool(k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st)))  # ProcessPowerThrottling
    ok_p = bool(k32.SetPriorityClass(h, 0x80))                                          # HIGH_PRIORITY_CLASS

    def bench():
        t0, x = time.perf_counter(), 0
        for i in range(300000):
            x += i * i
        return time.perf_counter() - t0
    res = {}
    for c in range(0, os.cpu_count(), 2):
        k32.SetProcessAffinityMask(h, ctypes.c_size_t(1 << c))
        time.sleep(0.02)
        res[c] = min(bench() for _ in range(3))
    best = min(res, key=res.get)
    k32.SetProcessAffinityMask(h, ctypes.c_size_t(1 << best))
    return dict(throttle_off=ok_t, high_prio=ok_p, core=best,
                bench_ms={c: round(v * 1e3, 2) for c, v in res.items()})


class ScenSet:
    """B 个工况 × m 个传感器的压力观测；参数 = 34 根自由管的 C。"""
    def __init__(self, pb, D, RH, sens, C_true, obs=None, noise_seed=100, tag=""):
        self.pb, self.tag = pb, tag
        self.D, self.RH = np.ascontiguousarray(D), np.ascontiguousarray(RH)
        self.sens, self.B = np.asarray(sens), D.shape[0]
        self.free, self.C_true = pb.free_idx, C_true
        self.elev_s = pb.elev[self.sens]
        if obs is None:
            sol = solve_polished(pb.s, self.D, self.RH, r_hw=pb.r_of_C(C_true),
                                 accuracy=1e-12, max_iter=60, polish_steps=3)
            p = sol["head"][:, self.sens] - self.elev_s[None, :]
            obs = p + SIGMA * np.random.default_rng(noise_seed).standard_normal(p.shape)
        self.obs = obs

    def reg(self, C):
        return float(np.mean((np.asarray(C) - C0) ** 2) / REG)

    def dreg(self, C):
        C = np.asarray(C)
        return 2 * (C - C0) / (len(C) * REG)


def rand_set(pb, B, sens, C_true, seed=2027, noise_seed=100):
    rng = np.random.default_rng(seed)       # 同种子：小 B 的工况是大 B 的前 B 行
    D = pb.net.demand_cfs_at(0)[None, :] * rng.uniform(0.7, 1.3, (B, pb.net.N))
    RH = np.tile(np.nan_to_num(pb.net.reservoir_head_ft_at(0)), (B, 1))
    return ScenSet(pb, D, RH, sens, C_true, noise_seed=noise_seed, tag=f"rand{B}")


class EpanetFast:
    def __init__(self, pb, sens):
        tmpd = tempfile.mkdtemp(prefix="hanoi_fair_")
        inp = shutil.copy(os.path.join(ROOT, pb.cfg["inp"]), os.path.join(tmpd, "Hanoi.inp"))
        ep = self.ep = er.Epanet(inp)
        ep.set_option(er.EN_ACCURACY, v1.EP_ACC)
        ep.set_option(er.EN_TRIALS, v1.EP_TRIALS)
        ep.set_option(v1.EN_DAMPLIMIT, v1.EP_DAMP)
        lib, ph = self.lib, self.ph = ep.lib, ep._ph
        lib.EN_openH.argtypes = [ctypes.c_void_p]
        lib.EN_closeH.argtypes = [ctypes.c_void_p]
        lib.EN_initH.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.EN_runH.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
        ep._check(lib.EN_openH(ph), "EN_openH")
        nid = {s: i + 1 for i, s in enumerate(ep.node_ids())}
        lid = {s: i + 1 for i, s in enumerate(ep.link_ids())}
        self.junc = pb.s.junc_nodes
        self.junc_ep = [nid[pb.net.node_id[j]] for j in self.junc]
        self.sens_ep = [nid[pb.net.node_id[j]] for j in sens]
        self.free_ep = [lid[pb.net.link_id[k]] for k in pb.free_idx]
        self.uf, self.uh = ep._ucf_flow, ep._ucf_head
        self.tt, self.v = ctypes.c_long(), ctypes.c_double()
        # 需水口径统一（10-23 修正）：带用水模式或多需水类别的网络（如 City D），EN_BASEDEMAND 只改第一类基值，
        # t=0 实际施加需水 = Σ 基值×模式系数。这里把第一类的模式设为无（系数 1）、其余类别基值清零，
        # 之后 set_dem 写入的就是实际施加需水。Hanoi 无模式、单类别，此步不改变任何值。
        lib.EN_getnumdemands.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.EN_setbasedemand.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        nd = ctypes.c_int()
        for nj in self.junc_ep:
            ep._check(lib.EN_setnodevalue(ph, nj, 2, 0.0), "EN_setnodevalue(EN_PATTERN)")   # EN_PATTERN = 2
            ep._check(lib.EN_getnumdemands(ph, nj, ctypes.byref(nd)), "EN_getnumdemands")
            for c in range(2, nd.value + 1):
                ep._check(lib.EN_setbasedemand(ph, nj, c, 0.0), "EN_setbasedemand")
        self.t_c, self.n_solve = 0.0, 0

    def close(self):
        self.lib.EN_closeH(self.ph)
        self.ep.close()

    def set_C(self, C):
        for li, c in zip(self.free_ep, C):
            self.lib.EN_setlinkvalue(self.ph, li, er.EN_ROUGHNESS, float(c))

    def set_Ck(self, k, c):
        self.lib.EN_setlinkvalue(self.ph, self.free_ep[k], er.EN_ROUGHNESS, float(c))

    def set_dem(self, drow):
        for j, nj in zip(self.junc, self.junc_ep):
            self.lib.EN_setnodevalue(self.ph, nj, er.EN_BASEDEMAND, float(drow[j]) * self.uf)

    def solve(self, cold):
        t0 = time.perf_counter()
        e1 = self.lib.EN_initH(self.ph, 10 if cold else 0)
        e2 = self.lib.EN_runH(self.ph, ctypes.byref(self.tt))
        self.t_c += time.perf_counter() - t0
        if e1 > 100 or e2 > 100:
            raise RuntimeError(f"EPANET error {e1}/{e2}")
        self.n_solve += 1

    def heads(self):
        out = np.empty(len(self.sens_ep))
        for i, ni in enumerate(self.sens_ep):
            self.lib.EN_getnodevalue(self.ph, ni, er.EN_HEAD, ctypes.byref(self.v))
            out[i] = self.v.value / self.uh
        return out

    def loss(self, S, C):
        self.set_C(C)
        acc = 0.0
        for j in range(S.B):
            self.set_dem(S.D[j])
            self.solve(True)
            r = self.heads() - S.elev_s - S.obs[j]
            acc += r @ r
        return acc / (S.B * len(S.sens)) + LAM * S.reg(C)

    def loss_grad_fd(self, S, C, central, warm, h_rel=H_FD):
        n, hs = len(C), h_rel * np.asarray(C)
        self.set_C(C)
        acc, g = 0.0, np.zeros(n)
        for j in range(S.B):
            self.set_dem(S.D[j])
            self.solve(True)                                  # 基解冷启动
            r0 = self.heads() - S.elev_s - S.obs[j]
            f0 = r0 @ r0
            acc += f0
            for k in range(n):
                self.set_Ck(k, C[k] + hs[k])
                self.solve(not warm)
                rp = self.heads() - S.elev_s - S.obs[j]
                if central:
                    self.set_Ck(k, C[k] - hs[k])
                    self.solve(not warm)
                    rm = self.heads() - S.elev_s - S.obs[j]
                    g[k] += (rp @ rp - rm @ rm) / (2 * hs[k])
                else:
                    g[k] += (rp @ rp - f0) / hs[k]
                self.set_Ck(k, C[k])
        m = S.B * len(S.sens)
        return acc / m + LAM * S.reg(C), g / m + LAM * S.dreg(C)


class HydroFast:
    def __init__(self, S, max_iter=60, polish=3):
        pb, self.S, self.mi, self.po = S.pb, S, max_iter, polish
        self.dt, self.rht = torch.tensor(S.D), torch.tensor(S.RH)
        self.ket = torch.zeros(S.B, pb.net.N)
        self.r_base = torch.tensor(pb.r_base)
        self.free_t = torch.as_tensor(pb.free_idx)
        self.ln, self.dm = torch.tensor(pb.len_ft[pb.free_idx]), torch.tensor(pb.diam_ft[pb.free_idx])
        self.obs_t, self.elev_t = torch.tensor(S.obs), torch.tensor(S.elev_s)
        self.sens_t = torch.as_tensor(S.sens)
        self.n_fwd = self.n_adj = 0

    def _r(self, Ct):
        r = self.r_base.clone()
        r[self.free_t] = hw_resistance(Ct, self.ln, self.dm)
        return r

    def loss(self, C):
        with torch.no_grad():
            r = self._r(torch.tensor(np.asarray(C, float))).numpy()
        sol = solve_polished(self.S.pb.s, self.S.D, self.S.RH, r_hw=r, accuracy=1e-12,
                             max_iter=self.mi, polish_steps=self.po)
        self.n_fwd += 1
        P = sol["head"][:, self.S.sens] - self.S.elev_s[None, :]
        return float(np.mean((P - self.S.obs) ** 2)) + LAM * self.S.reg(C)

    def loss_grad(self, C):
        Ct = torch.tensor(np.asarray(C, float), requires_grad=True)
        head, _, _ = ImplicitGGASolve.apply(self.dt, self.rht, self.ket, self._r(Ct),
                                            self.S.pb.s, 1e-12, self.mi, self.po)
        pred = head[:, self.sens_t] - self.elev_t
        loss = ((pred - self.obs_t) ** 2).mean() + LAM * ((Ct - C0) ** 2).mean() / REG
        loss.backward()
        self.n_fwd += 1
        self.n_adj += 1
        return float(loss.detach()), Ct.grad.numpy().copy()


def tmed(fn, reps):
    ts, res = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        res = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), res


def stage_scale(case, B_list):
    pb, sens, Ct = case.pb, case.sens_train, case.C_true
    Cv = np.full(len(pb.free_idx), C0)
    ep = EpanetFast(pb, sens)
    TT = {"gga": 0.0}                       # HydroGrad 前向中“批量 GGA（torch 向量化）”部分的耗时
    orig_solve = pb.s.solve

    def timed_solve(*a, **k):
        t0 = time.perf_counter()
        r = orig_solve(*a, **k)
        TT["gga"] += time.perf_counter() - t0
        return r
    pb.s.solve = timed_solve
    rows = []
    print("B | EP 前向(墙钟, 纯C) | HG 前向(其中批量GGA) | HG 前向+伴随 | EP 中心差分-热 | EP 前向差分-热 | 梯度耗时比 中心/HG 前向/HG | 目标值相对差")
    for B in B_list:
        S = rand_set(pb, B, sens, Ct)
        hg = HydroFast(S)
        reps = 3 if B <= 256 else 1
        hg.loss_grad(Cv)                                      # 预热（首次调用的一次性开销不计）
        tc0 = ep.t_c
        t_epf, L_ep = tmed(lambda: ep.loss(S, Cv), reps)
        tc_epf = (ep.t_c - tc0) / reps
        g0 = TT["gga"]
        t_hgf, L_hg = tmed(lambda: hg.loss(Cv), reps)
        t_gga = (TT["gga"] - g0) / reps
        t_hgg, (_, g_hg) = tmed(lambda: hg.loss_grad(Cv), reps)
        fd = {}
        for name, central, warm in (("ctr_warm", True, True), ("fwd_warm", False, True),
                                    ("ctr_cold", True, False), ("fwd_cold", False, False)):
            if not warm and B > 256:
                continue
            tc0, n0, t0 = ep.t_c, ep.n_solve, time.perf_counter()
            _, gf = ep.loss_grad_fd(S, Cv, central, warm)
            fd[name] = dict(wall=time.perf_counter() - t0, c_only=ep.t_c - tc0, solves=ep.n_solve - n0,
                            rel_err=float(np.linalg.norm(gf - g_hg) / np.linalg.norm(g_hg)))
        row = dict(B=B, ep_fwd_wall=t_epf, ep_fwd_c=tc_epf, hg_fwd=t_hgf, hg_fwd_gga=t_gga, hg_fwd_adj=t_hgg,
                   obj_rel_diff=abs(L_ep - L_hg) / abs(L_hg), fd=fd)
        rows.append(row)
        cw, fw = fd["ctr_warm"], fd["fwd_warm"]
        print(f"{B:5d} | {t_epf*1e3:9.2f} ms, {tc_epf*1e3:8.2f} ms | {t_hgf*1e3:9.2f} ms ({t_gga*1e3:8.2f}) | {t_hgg*1e3:9.2f} ms | "
              f"{cw['wall']:8.3f} s (err {cw['rel_err']:.1e}) | {fw['wall']:8.3f} s (err {fw['rel_err']:.1e}) | "
              f"{cw['wall']/t_hgg:7.1f}x {fw['wall']/t_hgg:7.1f}x | {row['obj_rel_diff']:.1e}")
        if "ctr_cold" in fd:
            cc, fc = fd["ctr_cold"], fd["fwd_cold"]
            print(f"      冷启动差分：中心 {cc['wall']:8.3f} s (err {cc['rel_err']:.1e}, {cc['solves']} 次求解)，"
                  f"前向 {fc['wall']:8.3f} s (err {fc['rel_err']:.1e})；热启动中心 {cw['solves']} 次求解，纯C {cw['c_only']:.3f} s")
    ep.close()
    pb.s.solve = orig_solve
    # 单工况延迟 t(1)；边际成本 c 与截距 t0 由 B>=64 各点最小二乘 t = t0 + c·B 得到
    Bs = np.array([r["B"] for r in rows], float)
    series = dict(ep_fwd_wall=[r["ep_fwd_wall"] for r in rows], ep_fwd_c=[r["ep_fwd_c"] for r in rows],
                  hg_fwd=[r["hg_fwd"] for r in rows], hg_fwd_gga=[r["hg_fwd_gga"] for r in rows],
                  hg_fwd_adj=[r["hg_fwd_adj"] for r in rows],
                  ep_ctrFD_warm=[r["fd"]["ctr_warm"]["wall"] for r in rows],
                  ep_ctrFD_warm_c=[r["fd"]["ctr_warm"]["c_only"] for r in rows],
                  ep_fwdFD_warm=[r["fd"]["fwd_warm"]["wall"] for r in rows])
    fits, big = {}, Bs >= 64
    for k, ts in series.items():
        ts = np.array(ts)
        (t0, c), *_ = np.linalg.lstsq(np.c_[np.ones(big.sum()), Bs[big]], ts[big], rcond=None)
        fits[k] = dict(latency_B1_ms=ts[0] * 1e3, marginal_us_per_scen=c * 1e6, fixed_overhead_ms=t0 * 1e3)
        print(f"[fit] {k:<16} 单工况延迟 {ts[0]*1e3:9.3f} ms | 边际成本 {c*1e6:9.2f} us/工况 | 截距(固定开销) {t0*1e3:9.3f} ms")
    for ep_k in ("ep_fwd_wall", "ep_fwd_c"):
        a, b = fits["hg_fwd"], fits[ep_k]
        d = b["marginal_us_per_scen"] - a["marginal_us_per_scen"]
        bstar = (a["fixed_overhead_ms"] - b["fixed_overhead_ms"]) * 1e3 / d if d > 0 else None
        fits[f"crossover_hg_fwd_vs_{ep_k}"] = bstar
        print(f"[cross] HG 前向 vs {ep_k}: " + (f"B* ≈ {bstar:.0f} 个工况后 HG 每工况更省" if bstar else "边际成本 HG 不低于 EPANET，前向无交点"))
    return dict(rows=rows, fits=fits)


def run_cal(name, fun, S, counters, case=None):
    C_init = np.full(len(S.free), C0)
    hist, cb, t0 = [], counters(), time.perf_counter()

    def f(C):
        L, g = fun(C)
        hist.append((time.perf_counter() - t0, L))
        return L, g

    res = minimize(f, C_init, jac=True, method="L-BFGS-B", bounds=[(v1.C_LO, v1.C_HI)] * len(S.free),
                   options=v1.LBFGSB_OPTS)
    wall, ca = time.perf_counter() - t0, counters()
    ttt = lambda th: next((t for t, L in hist if L <= th), None)          # noqa: E731
    rec = dict(arm=name, set=S.tag, B=S.B, wall_s=wall, nit=int(res.nit), nfev=int(res.nfev),
               final_loss=float(res.fun), C_rmse=float(np.sqrt(np.mean((res.x - S.C_true[S.free]) ** 2))),
               t_to_1p05sig2=ttt(1.05 * SIGMA ** 2), t_to_sig2=ttt(SIGMA ** 2),
               counters={k: ca[k] - cb[k] for k in ca}, message=str(res.message))
    if case is not None:
        rec["val_frame_rmse_clean"] = case.evaluate(res.x, name)["val_frame_rmse_clean"]
    fmt = lambda x: "未达到" if x is None else f"{x:7.2f}s"               # noqa: E731
    print(f"[calib {S.tag}] {name:<20} 总 {wall:8.2f}s | 达 1.05σ² {fmt(rec['t_to_1p05sig2'])} | 达 σ² {fmt(rec['t_to_sig2'])} | "
          f"迭代 {res.nit} 调用 {res.nfev} | loss {res.fun:.6e} C_rmse {rec['C_rmse']:.3f} | {rec['counters']}")
    return rec


def stage_calib(case, sets):
    pb, out = case.pb, []
    for tag in sets:
        if tag == "paper20":
            S = ScenSet(pb, case.d_used[case.tr], pb.rh[case.tr], case.sens_train, case.C_true,
                        obs=case.obs_tr, tag="paper20")
            arms = ("hg", "ctr_warm", "ctr_cold")
        else:
            S = rand_set(pb, int(tag[4:]), case.sens_train, case.C_true, noise_seed=case.noise_seed)
            arms = ("hg", "ctr_warm")
        ep, hg = EpanetFast(pb, S.sens), HydroFast(S)
        cE = lambda: dict(solves=ep.n_solve, c_only_s=ep.t_c)                 # noqa: E731
        cH = lambda: dict(model_calls=hg.n_fwd, adjoint_solves=hg.n_adj)     # noqa: E731
        cs = case if tag == "paper20" else None
        for a in arms:
            if a == "hg":
                out.append(run_cal("HydroGrad-adjoint", hg.loss_grad, S, cH, cs))
            else:
                warm = a.endswith("warm")
                out.append(run_cal(f"EPANET-ctrFD-{'warm' if warm else 'cold'}",
                                   lambda C, w=warm: ep.loss_grad_fd(S, C, True, w), S, cE, cs))
        ep.close()
    for r in out:
        r["seed"] = case.noise_seed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="scale", choices=["scale", "calib"])
    ap.add_argument("--B", default="1,4,16,64,256,1024,4096")
    ap.add_argument("--sets", default="paper20,rand200")
    ap.add_argument("--seeds", default="100")
    a = ap.parse_args()
    sys.stdout = v1.Tee(RPT)
    print("=" * 110)
    print(f"cpu_fair_scaling  {time.strftime('%Y-%m-%d %H:%M:%S')}  stage={a.stage}  threads={torch.get_num_threads()}")
    env = fast_cpu()
    bm = env.get("bench_ms", {})
    print(f"[env] 关闭后台降频={env.get('throttle_off')} 高优先级={env.get('high_prio')} 绑定逻辑核={env.get('core')} "
          f"微基准 最快 {min(bm.values(), default=0)} ms / 最慢 {max(bm.values(), default=0)} ms（偶数号逻辑核）")
    allrec = json.load(open(JSON, encoding="utf-8")) if os.path.exists(JSON) else {}
    allrec[f"env_{a.stage}"] = env
    if a.stage == "scale":
        allrec["scale"] = stage_scale(v1.Case(100), [int(x) for x in a.B.split(",")])
    else:
        key = "calib" if a.seeds == "100" else "calib_seeds"     # 单种子 100 的旧结果留在 calib
        allrec.setdefault(key, {})
        for sd in [int(x) for x in a.seeds.split(",")]:
            print(f"--- 噪声种子 {sd} ---", flush=True)
            for r in stage_calib(v1.Case(sd), a.sets.split(",")):
                allrec[key][f"{r['set']}|{r['arm']}|s{sd}" if key == "calib_seeds" else f"{r['set']}|{r['arm']}"] = r
            json.dump(allrec, open(JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    allrec["protocol_v2"] = dict(date=time.strftime("%Y-%m-%d"), epanet="openH once; initH(10) cold base; initH(0) warm FD",
                                 h_fd=H_FD, threads=1, lbfgsb=v1.LBFGSB_OPTS, sigma=SIGMA, lam=LAM)
    json.dump(allrec, open(JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(f"saved -> {JSON}")


if __name__ == "__main__":
    main()

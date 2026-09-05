# -*- coding: utf-8 -*-
"""dgga.eps - 完整 EPS 驱动器（阶段 B2 里程碑 3；hydraul.c 逐字复刻）。

复刻范围（源文件:行号在各方法注释）：
- inithyd（hydraul.c:85-179，openhyd 的 initlinkflow :77-81 并入） - 仅一次；
- runhyd 循环（:182-223）：demands（:465-535 阶梯需水 + 变水头 reservoir +
  转速 pattern 的 setlinksetting :418-462）→ controls（:538-619，含 vplus
  一秒容差与 TIMER 精确命中）→ hydsolve（GGASolver.run_gga，含状态机）；
- nexthyd/timestep（:225-275, :622-659）= min{Hstep, 模式边界, Rtime,
  tanktimestep（:662-707 的 ROUND）, controltimestep（:710-783）}，
  收尾 tanklevels（:998-1035 显式欧拉 + 1 秒前瞻钳位）+ tankgrade
  （:1071-1101 柱形仿射）；
- 梯队3：规则引擎（dgga.rules，rules.c 逐字复刻）接入 ruletimestep
  （hydraul.c:786-853：Rulestep 子步推进、逐子步 tanklevels+checkrules、
  规则开火截断本步）；Rulestep 未设时 = Hstep/10（input1.c:240-241）；
- 全程 warm start：runhyd 不重跑 inithyd，LinkFlow/EmitterFlow/状态逐帧续用。

不支持（构造时报错）：体积曲线水池、PBV/FCV/GPV。float64。
"""

import numpy as np

try:
    from dgga.parse import Net
    from dgga.solver import GGASolver, QZERO, PI, MISSING
    from dgga.rules import RuleEngine
except ImportError:  # pragma: no cover
    from parse import Net
    from solver import GGASolver, QZERO, PI, MISSING
    from rules import RuleEngine

SECperDAY = 86400      # types.h:83


def _round_c(x):
    """ROUND(x)（types.h:103）：(x>=0) ? (int)(x+.5) : (int)(x-.5)。"""
    return int(x + 0.5) if x >= 0.0 else int(x - 0.5)


class EpsDriver:
    """完整 EPS 驱动器。EpsDriver(net, inp_path).run() -> dict 帧数组。"""

    def __init__(self, net: Net, inp_path=None):
        self.net = net
        self.solver = GGASolver(net, mode="epanet", inp_path=inp_path)
        m = net.meta

        # ---- 时间参数（setdefaults input1.c:143-151 + adjustdata :220-227）----
        self.Dur = int(m["duration_sec"])
        self.Pstep = int(m["pat_step_sec"]) if int(m["pat_step_sec"]) > 0 else 3600
        self.Rstep = int(m["report_step_sec"])
        if self.Rstep == 0:
            self.Rstep = self.Pstep            # adjustdata input1.c:222
        self.Hstep = int(m["hyd_step_sec"]) if int(m["hyd_step_sec"]) > 0 else 3600
        if self.Hstep > self.Pstep:
            self.Hstep = self.Pstep            # adjustdata input1.c:226
        if self.Hstep > self.Rstep:
            self.Hstep = self.Rstep            # adjustdata input1.c:227
        self.Pstart = int(m["pat_start_sec"])
        self.Tstart = int(m["tstart_sec"])
        self.hacc = self.solver.hacc_default
        self.extra_iter = self.solver.extra_iter
        # Rulestep（input1.c:150 默认 0；adjustdata input1.c:240-241：
        # 未设 → Hstep/10（长整除），并钳到 <= Hstep）
        self.Rulestep = int(m.get("rule_step_sec", 0))
        if self.Rulestep == 0:
            self.Rulestep = self.Hstep // 10       # input1.c:240
        self.Rulestep = min(self.Rulestep, self.Hstep)   # input1.c:241

        # ---- 静态索引 ----
        s = self.solver
        self.N, self.L = net.N, net.L
        self.tank_nodes = np.asarray(net.tank_node, dtype=np.int64)
        self.pump_links = np.asarray(net.pump_link, dtype=np.int64)
        self.res_nodes = np.where(np.asarray(net.node_type) == 1)[0]
        # tank 节点 → tank 序号
        self._tank_of_node = {int(n): j for j, n in enumerate(self.tank_nodes)}
        self._elev = np.asarray(net.elev_ft, dtype=np.float64)

        # ---- 规则引擎（梯队3；rules.c）----
        self.rules = RuleEngine(net, self.solver, self) if len(net.rules_raw) else None
        self.Dsystem = 0.0                         # hydraul.c:487（demands 更新）
        self.node_dem = np.zeros(self.N, dtype=np.float64)   # EN_DEMAND 口径

    # ------------------------------------------------------------------
    def _tankvolume(self, j, h):
        """tankvolume（hydraul.c:1038-1068，无体积曲线支 :1057）。"""
        net = self.net
        return net.tank_vmin[j] + (h - net.tank_hmin[j]) * net.tank_area[j]

    def _tankgrade(self, j, v):
        """tankgrade（hydraul.c:1071-1101，柱形仿射支 :1089）。"""
        net = self.net
        return net.tank_hmin[j] + (v - net.tank_vmin[j]) / net.tank_area[j]

    # ------------------------------------------------------------------
    def _inithyd(self):
        """inithyd（hydraul.c:85-179）+ openhyd 的 initlinkflow（:77-81）。"""
        s = self.solver
        net = self.net
        self.S = s.init_status_int.copy()          # :130 LinkStatus = link->Status
        self.K = s.init_setting.copy()             # :131 LinkSetting = link->Kc
        closed = self.S <= s.ST_CLOSED
        # initlinkflow（:344-374）：关闭=QZERO，泵=Kc*Q0，其余=PI*D²/4
        q = np.where(closed, QZERO, PI * s.diam_np ** 2 / 4.0)
        for j, k in enumerate(self.pump_links):
            k = int(k)
            if not closed[k]:
                q[k] = self.K[k] * float(net.pump_q0[j])
        self.q = q
        self.e = None                              # EmitterFlow：首帧由 run_gga 置 1
        # tank：V=V0、头=H0（:106-113）；NodeDemand=0
        self.tankV = np.asarray(net.tank_v0, dtype=np.float64).copy()
        H = np.zeros(self.N, dtype=np.float64)
        H[self.res_nodes] = self._elev[self.res_nodes]   # reservoir 头 = El（H0=El）
        for j, n in enumerate(self.tank_nodes):
            H[int(n)] = float(net.tank_h0[j])
        self.H = H
        self.fixed_dem = np.zeros(self.N, dtype=np.float64)  # :111 NodeDemand=0
        self.Htime = 0                             # :176
        self.Rtime = self.Rstep                    # :178
        self.Haltflag = 0                          # :175

    # ------------------------------------------------------------------
    def _demands(self):
        """demands（hydraul.c:465-535）：阶梯需水、变水头 reservoir、转速 pattern。"""
        net = self.net
        s = self.solver
        t = self.Htime
        # p = (Htime + Pstart) / Pstep（:484）
        p = (t + self.Pstart) // self.Pstep
        # junction 需水（多类别求和 × Dmult；:488-504）＝ parse.demand_cfs_at
        self.d = net.demand_cfs_at(t)
        # Dsystem（hydraul.c:487-499）：逐类别 djunc>0 左折叠累加（规则引擎
        # SYSTEM DEMAND 前提用；类别序 = 文件序 = EPANET 需水链序）
        dmult = float(net.meta["demand_multiplier"])
        fac = net._pattern_factors_at(t)
        npat = len(net.patterns)
        pidx = np.where(net.dem_pat < 0, npat, net.dem_pat)
        w = net.dem_base_cfs * fac[pidx] * dmult   # :496 djunc = Base*F[k]*Dmult
        ds = 0.0
        for vv in w.tolist():
            if vv > 0.0:                           # :497 djunc>0 计入
                ds += vv
        self.Dsystem = ds
        # 变水头 reservoir（:507-520）：仅有 pattern 者更新
        for i in self.res_nodes:
            pi = int(net.res_head_pat[i])
            if pi >= 0:
                F = net.patterns[pi]
                k = int(p % len(F))
                self.H[i] = self._elev[i] * F[k]   # :517 El * F[k]
        # 泵转速 pattern（:522-534）→ setlinksetting（:418-462 泵支 :437-447）
        for j, k in enumerate(self.pump_links):
            up = int(net.pump_upat[j])
            if up < 0:
                continue
            k = int(k)
            F = net.patterns[up]
            value = float(F[int(p % len(F))])
            self.K[k] = value                      # :439 *k = value
            if value > 0.0 and self.S[k] <= s.ST_CLOSED:   # :440-445
                # resetpumpflow（:1103-1116）：仅恒功率泵重置 Q0
                if net.pump_ptype[j] == 0:
                    self.q[k] = float(net.pump_q0[j])
                self.S[k] = s.ST_OPEN
            if value == 0.0 and self.S[k] > s.ST_CLOSED:   # :446
                self.S[k] = s.ST_CLOSED

    # ------------------------------------------------------------------
    def _controls(self):
        """controls（hydraul.c:538-619）：水池水位/TIMER/TIMEOFDAY 简单控制。"""
        net = self.net
        s = self.solver
        lt = np.asarray(net.link_type)
        setsum = 0
        for i in range(len(net.ctl_link)):
            reset = 0
            k = int(net.ctl_link[i])
            n = int(net.ctl_node[i])
            ctype = int(net.ctl_type[i])
            # 水池水位控制（:570-578）：n > Njuncs（tank/reservoir）
            if n >= 0 and s.is_fixed_node[n]:
                if n not in self._tank_of_node:
                    raise NotImplementedError("reservoir 上的水位控制不在本期范围")
                j = self._tank_of_node[n]
                h = float(self.H[n])
                vplus = abs(float(self.fixed_dem[n]))      # :573 一秒钟容差体积
                v1 = self._tankvolume(j, h)                # :574
                v2 = self._tankvolume(j, float(net.ctl_grade[i]))  # :575
                if ctype == 0 and v1 <= v2 + vplus:        # LOWLEVEL :576
                    reset = 1
                if ctype == 1 and v1 >= v2 - vplus:        # HILEVEL :577
                    reset = 1
            if ctype == 2:                                 # TIMER 精确命中（:581-584）
                if int(net.ctl_time[i]) == self.Htime:
                    reset = 1
            if ctype == 3:                                 # TIMEOFDAY（:586-593）
                if (self.Htime + self.Tstart) % SECperDAY == int(net.ctl_time[i]):
                    reset = 1
            if reset == 1:                                 # :596-616
                s1 = s.ST_CLOSED if self.S[k] <= s.ST_CLOSED else s.ST_OPEN  # :598-599
                s2 = int(net.ctl_status[i])                # :600
                k1 = float(self.K[k])                      # :601
                k2 = k1
                if lt[k] > 1:                              # link->Type > PIPE（:603）
                    k2 = float(net.ctl_setting[i])
                # 恒功率泵重开 → resetpumpflow（:606-607）
                if lt[k] == 2 and s1 == s.ST_CLOSED and s2 == s.ST_OPEN:
                    jj = int(np.where(self.pump_links == k)[0][0])
                    if net.pump_ptype[jj] == 0:            # resetpumpflow :1113-1115
                        self.q[k] = float(net.pump_q0[jj])
                if s1 != s2 or k1 != k2:                   # :609-615
                    self.S[k] = s2
                    self.K[k] = k2
                    setsum += 1
        return setsum

    # ------------------------------------------------------------------
    def _tanktimestep(self, tstep):
        """tanktimestep（hydraul.c:662-707）：最快满/空水池的 ROUND 时间。"""
        net = self.net
        for j, n in enumerate(self.tank_nodes):
            n = int(n)
            h = float(self.H[n])                   # :689
            qn = float(self.fixed_dem[n])          # :690 NodeDemand
            if abs(qn) <= QZERO:                   # :691
                continue
            if qn > 0.0 and h < net.tank_hmax[j]:  # :694
                v = net.tank_vmax[j] - self.tankV[j]
            elif qn < 0.0 and h > net.tank_hmin[j]:  # :695
                v = net.tank_vmin[j] - self.tankV[j]
            else:
                continue                           # :696
            t = _round_c(v / qn)                   # :699
            if 0 < t < tstep:                      # :700
                tstep = t
        return tstep

    def _controltimestep(self, tstep):
        """controltimestep（hydraul.c:710-783）：最快触发控制的时间。"""
        net = self.net
        s = self.solver
        lt = np.asarray(net.link_type)
        for i in range(len(net.ctl_link)):
            t = 0
            n = int(net.ctl_node[i])
            ctype = int(net.ctl_type[i])
            if n >= 0:
                if not s.is_fixed_node[n]:         # :739 junction → 跳过整条
                    continue
                j = self._tank_of_node.get(n)
                if j is None:
                    continue                       # reservoir：tankvolume A=0 语义外
                h = float(self.H[n])               # :742
                qn = float(self.fixed_dem[n])      # :743
                if abs(qn) <= QZERO:               # :744
                    continue
                grade = float(net.ctl_grade[i])
                if ((h < grade and ctype == 1 and qn > 0.0)      # HILEVEL :747
                        or (h > grade and ctype == 0 and qn < 0.0)):  # LOWLEVEL :748
                    v = self._tankvolume(j, grade) - self.tankV[j]    # :750
                    t = _round_c(v / qn)           # :751
            if ctype == 2:                         # TIMER（:756-762）
                if int(net.ctl_time[i]) > self.Htime:
                    t = int(net.ctl_time[i]) - self.Htime
            if ctype == 3:                         # TIMEOFDAY（:765-771）
                t1 = (self.Htime + self.Tstart) % SECperDAY
                t2 = int(net.ctl_time[i])
                t = t2 - t1 if t2 >= t1 else SECperDAY - t1 + t2
            if 0 < t < tstep:                      # :774-781
                k = int(net.ctl_link[i])
                if ((lt[k] > 1 and self.K[k] != net.ctl_setting[i])
                        or (self.S[k] != net.ctl_status[i])):
                    tstep = t
        return tstep

    def _tanklevels(self, tstep):
        """tanklevels（hydraul.c:998-1035）：显式欧拉 + 1 秒前瞻钳位。"""
        net = self.net
        for j, n in enumerate(self.tank_nodes):
            n = int(n)
            dv = self.fixed_dem[n] * tstep         # :1021
            self.tankV[j] += dv                    # :1022
            # 1 秒前瞻钳位（:1025-1031）
            if self.tankV[j] + self.fixed_dem[n] >= net.tank_vmax[j]:
                self.tankV[j] = net.tank_vmax[j]
            elif self.tankV[j] - self.fixed_dem[n] <= net.tank_vmin[j]:
                self.tankV[j] = net.tank_vmin[j]
            self.H[n] = self._tankgrade(j, self.tankV[j])   # :1033
        return

    def _timestep(self):
        """timestep（hydraul.c:622-659）。"""
        tstep = self.Hstep                          # :637
        # 需水模式边界（:641-643）
        n = (self.Htime + self.Pstart) // self.Pstep + 1
        t = n * self.Pstep - self.Htime
        if 0 < t < tstep:
            tstep = t
        # 报告时刻（:646-647）
        t = self.Rtime - self.Htime
        if 0 < t < tstep:
            tstep = t
        tstep = self._tanktimestep(tstep)           # :650
        tstep = self._controltimestep(tstep)        # :653
        if self.rules is not None:                  # :656（Nrules>0）
            tstep = self._ruletimestep(tstep)
        else:
            self._tanklevels(tstep)                 # :657（Nrules==0）
        return tstep

    def _ruletimestep(self, tstep):
        """ruletimestep（hydraul.c:786-853）：按 Rulestep 子步推进、逐子步
        tanklevels+checkrules，规则开火即截断本步；Htime 借用后复原。"""
        tnow = self.Htime                           # :805
        tmax = tnow + tstep                         # :806
        dt = self.Rulestep                          # :820
        dt1 = self.Rulestep - (tnow % self.Rulestep)  # :821
        dt = min(dt, tstep)                         # :825
        dt1 = min(dt1, tstep)                       # :826
        if dt1 == 0:
            dt1 = dt                                # :827
        while True:                                 # :840-847 do-while(dt>0)
            self.Htime += dt1                       # :842 借用全局时钟
            self._tanklevels(dt1)                   # :843
            if self.rules.checkrules(dt1):          # :844 规则开火即停
                break
            dt = min(dt, tmax - self.Htime)         # :845
            dt1 = dt                                # :846
            if dt <= 0:
                break
        tstep = self.Htime - tnow                   # :851
        self.Htime = tnow                           # :852 复原
        return tstep

    def _nexthyd(self, relerr):
        """nexthyd（hydraul.c:225-275）。返回 tstep（0=结束）。"""
        # runhyd 的 Haltflag（:214-217）：relerr>Hacc 且 ExtraIter==-1
        if relerr > self.hacc and self.extra_iter == -1:
            self.Haltflag = 1
        if self.Haltflag:                           # :245
            self.Htime = self.Dur
        hydstep = 0
        if self.Htime < self.Dur:                   # :250
            hydstep = self._timestep()
        if self.Htime < self.Dur:                   # :258-265
            self.Htime += hydstep
            if self.Htime >= self.Rtime:
                self.Rtime += self.Rstep
        else:
            self.Htime += 1                         # :270
        return hydstep

    # ------------------------------------------------------------------
    def run(self, max_frames=100000):
        """从 t=0 完整自主 EPS。返回帧数组 dict（与 ref.npz 同口径）。"""
        s = self.solver
        self._inithyd()
        frames = dict(t_sec=[], head_ft=[], flow_cfs=[], flow_int_cfs=[],
                      status=[], setting=[], demand_out_cfs=[],
                      iterations=[], relerr=[])
        while True:
            t = self.Htime                          # runhyd :201
            self._demands()                         # :202
            self._controls()                        # :203
            r = s.run_gga(self.d, self.H, q0=self.q, e0=self.e,
                          status0=self.S, setting0=self.K, do_status=True)
            self.q = r["flow"]
            self.e = r["emitter"]
            self.S = r["status"]
            self.K = r["setting"]
            self.H = r["head"]
            self.fixed_dem = r["fixed_demand"]
            relerr = float(r["relerr"])

            # ---- 记帧（EN_* API 口径）----
            open_api = self.S > s.ST_CLOSED
            flow_api = np.where(open_api, self.q, 0.0)        # EN_FLOW（epanet.c:3657-3659）
            # EN_DEMAND：junction = DemandFlow+EmitterFlow（hydsolver.c:199-204），
            # 定水头节点 = 净流入（newlinkflows :459-464）
            dem_out = self.d + self.e
            dem_out = np.where(s.is_fixed_node, self.fixed_dem, dem_out)
            # NodeDemand（hydsolve 收尾 hydsolver.c:199-204 + newlinkflows
            # :459-464）：规则引擎 checkvalue 的 r_DEMAND/FILLTIME 读此值
            self.node_dem = dem_out
            setting_api = np.where(self.K == MISSING, 0.0, self.K)  # epanet.c:3693
            # EN_SETTING 阀设定换算回用户单位（epanet.c:3695-3706）
            if s.valve_links.size:
                vl = s.valve_links
                lt_v = s.lt_np[vl]
                mult = np.where(np.isin(lt_v, (3, 4, 5)), s.ucf_pressure,
                                np.where(lt_v == 6, s.ucf_flow, 1.0))
                setting_api[vl] = setting_api[vl] * mult
            frames["t_sec"].append(t)
            frames["head_ft"].append(self.H.copy())
            frames["flow_cfs"].append(flow_api)
            frames["flow_int_cfs"].append(self.q.copy())
            frames["status"].append(open_api.astype(np.int8))
            frames["setting"].append(setting_api)
            frames["demand_out_cfs"].append(dem_out)
            frames["iterations"].append(int(r["iters"]))
            frames["relerr"].append(relerr)

            tstep = self._nexthyd(relerr)           # EN_nextH
            if tstep == 0:
                break
            if len(frames["t_sec"]) >= max_frames:
                raise RuntimeError("EPS 帧数超限（疑似死循环）")
        return {k: np.asarray(v) for k, v in frames.items()}


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stem = sys.argv[1] if len(sys.argv) > 1 else "EXA6"
    net = Net.load(os.path.join(ROOT, "data", "reference"), stem)
    inp = os.path.join(ROOT, "networks", "InpData", f"{stem}.inp")
    ref = np.load(os.path.join(ROOT, "data", "reference", f"{stem}_ref.npz"))
    out = EpsDriver(net, inp_path=inp).run()
    print(f"[{stem}] 帧数 {len(out['t_sec'])} (ref {len(ref['t_sec'])})")
    print("t 序列相等:", np.array_equal(out["t_sec"], ref["t_sec"]))
    T = min(len(out["t_sec"]), len(ref["t_sec"]))
    for f in range(T):
        dH = np.abs(out["head_ft"][f] - ref["head_ft"][f]).max()
        dQ = np.abs(out["flow_cfs"][f] - ref["flow_cfs"][f]).max()
        print(f"  f={f} t={out['t_sec'][f]} iters={out['iterations'][f]}/"
              f"{int(ref['iterations'][f])} dH={dH:.3e} dQ={dQ:.3e}")

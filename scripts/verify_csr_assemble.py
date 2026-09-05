# -*- coding: utf-8 -*-
"""verify_csr_assemble.py - sparse_gpu_plan.md §1a 的验收：CSR 值装配通路。

只验**装配**，不换求解器、不碰 autograd、不碰 cuDSS：
  §0  位型统计：nnz、nnz/Nj²、B=256 下 [B,nnz] vs [B,Nj²] 的显存对比
  §1  位型自检：CSR 合法性（indptr 单调 / 行内列升序 / 对角全在 / 覆盖稠密非零）
  §2  装配核逐位等价：随机 Pm 下 CSR 还原的 A vs 稠密 scatter_add 的 A
  §3  端到端逐位等价：solve(assemble="dense") vs solve(assemble="csr")
      的 head/flow/relerr 逐位、iters 逐项

用法：python -X utf8 scripts/verify_csr_assemble.py
"""

import os
import sys
import time
import unicodedata

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dgga.parse import Net                                        # noqa: E402
from dgga.solver import GGASolver                                 # noqa: E402

REF = os.path.join(ROOT, "data", "reference")
WIP = os.path.join(ROOT, "data", "p1_csr_wip.txt")

# 任务指定的 10 个网（8 个 dense 可跑的公开网 + City_D + ky4）
NETS = [("Net1", "pub_net1"), ("Anytown", "pub_anytown"), ("Hanoi", "pub_hanoi"),
        ("Net2", "pub_net2"), ("Fossolo", "pub_fossolo_poly1"),
        ("Pescara", "pub_pescara"), ("Net3", "pub_net3"),
        ("Modena", "pub_modena"), ("City_D", "city_d"), ("ky4", "pub_ky4")]
# 只做位型统计（dense 求解被特性门挡住，但构造期索引照样建得出来）
PATTERN_ONLY = [("ky10", "pub_ky10"), ("L-TOWN", "pub_l_town"),
                ("Net6", "pub_net6"), ("BWSN_2", "pub_bwsn_network_2")]


# ---------------------------------------------------------------- 排版
def wl(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s, w, right=True):
    d = w - wl(s)
    return (" " * d + s) if right else (s + " " * d)


def table(rows, header, right=None):
    c = len(header)
    right = right or [False] + [True] * (c - 1)
    ws = [max(wl(header[i]), max(wl(r[i]) for r in rows)) for i in range(c)]
    out = ["  ".join(pad(header[i], ws[i], right[i]) for i in range(c)),
           "-" * (sum(ws) + 2 * (c - 1))]
    for r in rows:
        out.append("  ".join(pad(r[i], ws[i], right[i]) for i in range(c)))
    return out


BUF = []


def emit(lines):
    for ln in lines:
        print(ln)
        BUF.append(ln)
    sys.stdout.flush()


# ---------------------------------------------------------------- 场景
def load_solver(stem, mode="dense"):
    """ky4 的水池 T-2 在 INP 里恰好空池（H0=765.00001 贴 Hmin），构造期的静态
    守卫会拦下 dense。本轮验的是两条装配通路在同一输入上的等价性，与水池贴边
    无关；关掉静态守卫、并把该池头推入区间内（见 boundary()）即可 - solve()
    末尾的 _check_dense_tank_status 仍会对收敛解逐字复核完整判据。"""
    net = Net.load(REF, stem)
    try:
        return net, GGASolver(net, mode=mode)
    except NotImplementedError:
        return net, GGASolver(net, mode=mode, dense_tank_bound_check=False)


def boundary(net, stem):
    """(d, rh)：需水与定水头边界。水池头取参考帧 0；落在 Hmin/Hmax 上的水池
    推入区间内 5% 处 - dense 路径不实现 tankstatus 的 TEMPCLOSED 切换，
    而本轮验的是"两条装配通路在**同一输入**上是否等价"，与水池是否贴边无关。"""
    ref = np.load(os.path.join(REF, f"{stem}_ref.npz"))
    t0 = int(ref["t_sec"][0])
    d = np.asarray(net.demand_cfs_at(t0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(t0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    nudged = []
    if tn.size:
        rh[tn] = ref["head_ft"][0][tn]
        lo = net.tank_hmin + 0.05 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.05 * (net.tank_hmax - net.tank_hmin)
        new = np.clip(rh[tn], lo, hi)
        for i in np.where(new != rh[tn])[0]:
            nudged.append(f"{net.node_id[tn[i]]} {rh[tn[i]]:.3f}→{new[i]:.3f}")
        rh[tn] = new
    return d, rh, nudged


def batch(net, d, rh, B, seed):
    """B 个场景：需水全局 ×U(0.6,1.4) × 逐节点 U(0.7,1.3)；定水头 ±2 ft。"""
    g = torch.Generator().manual_seed(seed)
    d0 = torch.as_tensor(d, dtype=torch.float64)
    r0 = torch.as_tensor(rh, dtype=torch.float64)
    D = d0.unsqueeze(0) * (0.6 + 0.8 * torch.rand(B, 1, dtype=torch.float64,
                                                  generator=g)) \
        * (0.7 + 0.6 * torch.rand(B, d0.numel(), dtype=torch.float64,
                                  generator=g))
    RH = r0.unsqueeze(0) + (torch.rand(B, r0.numel(), dtype=torch.float64,
                                       generator=g) - 0.5) * 4.0
    RH = torch.where(torch.isnan(r0).unsqueeze(0).expand(B, -1),
                     r0.unsqueeze(0).expand(B, -1), RH)
    return D, RH


# ---------------------------------------------------------------- 装配核
def _vals(sv, Pm):
    """solve() 里 linkcoeffs 的四段贡献（与 solver.py 内一字不差）。"""
    return torch.cat([-Pm[:, sv.lk_both], -Pm[:, sv.lk_both],
                      Pm[:, sv.lk_m1], Pm[:, sv.lk_m2]], dim=1)


def A_dense(sv, Pm):
    B = Pm.shape[0]
    A = torch.zeros(B, sv.Nj * sv.Nj, dtype=Pm.dtype)
    A.scatter_add_(1, sv.A_idx.expand(B, -1), _vals(sv, Pm))
    return A.view(B, sv.Nj, sv.Nj)


def A_csr(sv, Pm):
    B = Pm.shape[0]
    return sv._csr_to_dense(sv._assemble_csr(_vals(sv, Pm), B), B)


# ======================================================================
def sec0():
    emit(["=" * 96,
          "§0 CSR 位型规模与显存（位型 = 链路贡献并集 ∪ 全部对角；由拓扑定死，"
          "迭代中不变）",
          "    显存口径：f64，每轮迭代 A 的值张量 [B,nnz] vs 稠密 [B,Nj*Nj]，B=256"])
    rows, stats = [], {}
    for name, stem in NETS + PATTERN_ONLY:
        net = Net.load(REF, stem)
        sv = None
        for md in ("dense", "epanet"):
            for kw in ({}, dict(dense_tank_bound_check=False)):
                try:
                    sv = GGASolver(net, mode=md, **kw)
                    break
                except Exception:                                # noqa: BLE001
                    continue
            if sv is not None:
                break
        if sv is None:
            emit([f"    {name}: 两种模式构造均失败，跳过"])
            continue
        Nj, nnz = sv.Nj, sv.A_csr_nnz
        m_d = 256 * Nj * Nj * 8 / 2**30                          # GiB
        m_s = 256 * nnz * 8 / 2**20                              # MiB
        rows.append([name, f"{Nj}", f"{sv.L}", f"{nnz}",
                     f"{nnz / Nj**2:.5f}", f"{nnz / Nj:.2f}",
                     f"{m_d:.4f}", f"{m_s:.4f}", f"{Nj * Nj / nnz:.1f}x"])
        stats[name] = dict(Nj=Nj, L=sv.L, nnz=nnz, dense_GiB=m_d, csr_MiB=m_s,
                           ratio=Nj * Nj / nnz)
    rows.sort(key=lambda r: int(r[1]))
    emit(table(rows, ["网", "Nj", "L", "nnz", "nnz/Nj²", "nnz/Nj",
                      "稠密[B,Nj²] GiB", "CSR[B,nnz] MiB", "降幅"]))
    emit(["    降幅 = Nj²/nnz，随 Nj 单调增长（nnz≈3.1·Nj 近乎线性，稠密是 Nj²）。"])
    return stats


# ======================================================================
def sec1():
    emit(["", "=" * 96,
          "§1 位型自检：indptr 单调且末位=nnz / 行内列严格升序 / Nj 个对角全在位型内 /",
          "   位型 ⊇ 稠密装配的非零集（随机 Pm 下实测）"])
    rows, bad = [], 0
    for name, stem in NETS:
        net, sv = load_solver(stem)
        ip = sv.A_csr_indptr.numpy().astype(np.int64)
        ic = sv.A_csr_indices.numpy().astype(np.int64)
        ok_ip = bool(ip[0] == 0 and ip[-1] == sv.A_csr_nnz
                     and np.all(np.diff(ip) >= 0) and ip.size == sv.Nj + 1)
        ok_sorted = all(np.all(np.diff(ic[ip[r]:ip[r + 1]]) > 0)
                        for r in range(sv.Nj))
        # 对角：A_csr_diag[j] 必须落在第 j 行且列号 = j
        dg = sv.A_csr_diag.numpy()
        rowof = np.searchsorted(ip, dg, side="right") - 1
        ok_diag = bool(np.array_equal(rowof, np.arange(sv.Nj))
                       and np.array_equal(ic[dg], np.arange(sv.Nj)))
        # 覆盖性：随机 Pm 下稠密 A 的非零位置必须是位型的子集
        g = torch.Generator().manual_seed(7)
        Pm = 0.1 + torch.rand(2, sv.L, dtype=torch.float64, generator=g)
        nzd = torch.nonzero(A_dense(sv, Pm)[0].flatten()).flatten().numpy()
        ok_cov = bool(np.all(np.isin(nzd, sv.A_csr_dense_pos.numpy())))
        # 孤立 junction（不接任何链路）的行数 - 位型必须仍给它留对角
        deg = np.bincount(np.concatenate([
            sv.f_idx1.numpy(), sv.f_idx2.numpy()]), minlength=sv.Nj)
        n_iso = int((deg == 0).sum())
        ok = ok_ip and ok_sorted and ok_diag and ok_cov
        bad += (not ok)
        rows.append([name, f"{sv.Nj}", f"{sv.A_csr_nnz}",
                     "OK" if ok_ip else "FAIL", "OK" if ok_sorted else "FAIL",
                     "OK" if ok_diag else "FAIL", "OK" if ok_cov else "FAIL",
                     f"{n_iso}",
                     f"{sv.A_csr_indices.dtype}".replace("torch.", ""),
                     f"{sv.A_csr_scatter.dtype}".replace("torch.", "")])
    emit(table(rows, ["网", "Nj", "nnz", "indptr", "行内升序", "对角齐全",
                      "覆盖非零", "孤立junc", "indices型", "scatter型"]))
    emit([f"    不合格 {bad} 个。索引 dtype：indptr/indices=int32"
          f"（nnz、Nj 均 ≪ 2³¹，且 cuSPARSE/cuDSS 默认 int32，接入零转换）；",
          "    scatter/diag/dense_pos=int64（torch 的 scatter_add_/scatter 只收 int64）。"])
    return bad == 0


# ======================================================================
def sec2():
    emit(["", "=" * 96,
          "§2 装配核逐位等价：同一 Pm，CSR 装配还原的 A vs 稠密 scatter_add 的 A",
          "   判据：**max|ΔA| == 0.000e+00**（源序相同 + 目标映射单射 ⇒ 每个槽的"
          "加数序列逐项相同）",
          "   Pm 取 4 组：U(0.1,1.1)、跨 12 个数量级 10^U(-6,6)（考验求和次序）、"
          "含 30% 精确 0、常数 1"])
    rows, worst_all, bad = [], 0.0, 0
    for name, stem in NETS:
        net, sv = load_solver(stem)
        g = torch.Generator().manual_seed(20260822)
        B = 8
        cases = {
            "U(0.1,1.1)": 0.1 + torch.rand(B, sv.L, dtype=torch.float64,
                                           generator=g),
            "10^U(-6,6)": 10.0 ** (12.0 * torch.rand(B, sv.L,
                                                     dtype=torch.float64,
                                                     generator=g) - 6.0),
            "30%零": (0.1 + torch.rand(B, sv.L, dtype=torch.float64,
                                       generator=g))
            * (torch.rand(B, sv.L, dtype=torch.float64, generator=g) > 0.3),
            "常数1": torch.ones(B, sv.L, dtype=torch.float64),
        }
        cells = [name, f"{sv.Nj}"]
        w = 0.0
        for tag, Pm in cases.items():
            Ad, Ac = A_dense(sv, Pm), A_csr(sv, Pm)
            same = bool(torch.equal(Ad, Ac))
            dmax = float((Ad - Ac).abs().max())
            w = max(w, dmax)
            if not same:
                bad += 1
            cells.append(f"{dmax:.3e}" + ("" if same else " !!"))
        worst_all = max(worst_all, w)
        rows.append(cells)
    emit(table(rows, ["网", "Nj", "U(0.1,1.1)", "10^U(-6,6)", "30%零", "常数1"]))
    emit([f"    10 网 × 4 组 × B=8：max|ΔA| = {worst_all:.3e}，"
          f"逐位不同的组数 = {bad}   {'PASS' if bad == 0 else 'FAIL'}"])
    return bad == 0


# ======================================================================
def sec3():
    emit(["", "=" * 96,
          "§3 端到端逐位等价：solve(assemble='dense') vs solve(assemble='csr')",
          "   判据：head/flow/emitter/relerr **逐位相同**，iters 逐项相等"])
    rows, bad = [], 0
    note = []
    for name, stem in NETS:
        net, sv = load_solver(stem)
        d, rh, nudged = boundary(net, stem)
        if nudged:
            note.append(f"    {name} 水池推入区间：{'; '.join(nudged)}")
        cells = [name, f"{sv.Nj}"]
        it_txt, ok_net = "", True
        for B, seed in ((1, 0), (8, 4242)):
            if B == 1:
                D, RH = d, rh
            else:
                D, RH = batch(net, d, rh, B, seed)
            r_d = sv.solve(D, RH, assemble="dense")
            r_c = sv.solve(D, RH, assemble="csr")
            eq = all(torch.equal(r_d[k], r_c[k])
                     for k in ("head_ft", "flow_cfs", "emitter_cfs", "relerr"))
            it_eq = bool(torch.equal(r_d["iters"], r_c["iters"]))
            dH = float((r_d["head_ft"] - r_c["head_ft"]).abs().max())
            dQ = float((r_d["flow_cfs"] - r_c["flow_cfs"]).abs().max())
            ok_net = ok_net and eq and it_eq
            cells += [f"{dH:.3e}", f"{dQ:.3e}", "OK" if it_eq else "FAIL"]
            it = np.atleast_1d(r_d["iters"].numpy())
            it_txt = f"{it.min()}-{it.max()}" if B > 1 else f"{int(it[0])}"
        cells.append(it_txt)
        bad += (not ok_net)
        rows.append(cells)
    emit(table(rows, ["网", "Nj", "B=1 max|ΔH|", "B=1 max|ΔQ|", "B=1 iters",
                      "B=8 max|ΔH|", "B=8 max|ΔQ|", "B=8 iters", "迭代数"]))
    if note:
        emit(["", "   水池贴边处理（只影响输入场景，两条通路吃同一份输入）："] + note)
    emit([f"    10 网 × (B=1, B=8)：不一致的网 = {bad}   "
          f"{'PASS' if bad == 0 else 'FAIL'}"])
    return bad == 0


# ======================================================================
def sec4():
    emit(["", "=" * 96,
          "§4 B=256 实跑一次装配：CSR 值张量的**实测**字节数 vs 稠密所需字节数",
          "   （稠密一列是算出来的：BWSN_2 的 [256,12523²] f64 = 299 GiB，"
          "本机开不出来 - 这正是墙）"])
    rows = []
    for name, stem in [("Modena", "pub_modena"), ("City_D", "city_d"),
                       ("ky4", "pub_ky4"), ("Net6", "pub_net6"),
                       ("BWSN_2", "pub_bwsn_network_2")]:
        net = Net.load(REF, stem)
        sv = None
        for md in ("dense", "epanet"):
            for kw in ({}, dict(dense_tank_bound_check=False)):
                try:
                    sv = GGASolver(net, mode=md, **kw)
                    break
                except Exception:                                # noqa: BLE001
                    continue
            if sv is not None:
                break
        B = 256
        g = torch.Generator().manual_seed(5)
        Pm = 0.1 + torch.rand(B, sv.L, dtype=torch.float64, generator=g)
        t0 = time.perf_counter()
        data = sv._assemble_csr(_vals(sv, Pm), B)
        dt = time.perf_counter() - t0
        got = data.numel() * data.element_size()
        need = B * sv.Nj * sv.Nj * 8
        rows.append([name, f"{sv.Nj}", f"{tuple(data.shape)}",
                     f"{got/2**20:.2f}", f"{need/2**30:.2f}",
                     f"{need/got:.0f}x", f"{dt*1e3:.1f}"])
        del data
    emit(table(rows, ["网", "Nj", "CSR值张量形状", "实测 MiB", "稠密需 GiB",
                      "降幅", "装配墙钟 ms"]))
    emit(["    BWSN_2 上 299 GiB → 80 MiB（3811x）：显存墙由这一条通路拆掉；",
          "    但 BWSN_2/Net6 的**求解**仍卡在 CVPIPE/PRV/PSV/FCV 特性门上"
          "（dense_gap_plan.md §2），",
          "    §1a 只拆显存墙，特性墙是另一件事 - 如实记，不越界宣称。"])
    return True


# ======================================================================
def main():
    torch.set_num_threads(1)
    t0 = time.time()
    emit(["#" * 96,
          "sparse_gpu_plan.md §1a - CSR 值装配通路验收（前台实测）",
          f"日期 {time.strftime('%Y-%m-%d %H:%M:%S')}   torch {torch.__version__}"
          f"   dtype float64   线程 1   CPU",
          "范围：只新增装配通路；不换线性求解器、不碰 autograd、不碰 cuDSS；"
          "dense 仍为缺省",
          "#" * 96])
    st = sec0()
    ok1 = sec1()
    ok2 = sec2()
    ok3 = sec3()
    sec4()
    emit(["", "=" * 96,
          f"总用时 {time.time()-t0:.1f}s",
          f"判定：§1 位型自检 {'PASS' if ok1 else 'FAIL'}   "
          f"§2 装配核逐位 {'PASS' if ok2 else 'FAIL'}   "
          f"§3 端到端逐位 {'PASS' if ok3 else 'FAIL'}"])
    with open(WIP, "a", encoding="utf-8") as f:
        f.write("\n".join(BUF) + "\n\n")
    print(f"\n证据已追加 {WIP}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())

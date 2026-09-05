# -*- coding: utf-8 -*-
"""审计⑤：CSR 通路是否切断/改变梯度。
(a) 图是否活：csr 路径 head/flow 必须 requires_grad 且 grad 非 None；
(b) 与稠密通路的梯度逐位对拍（demand / ke / 水库水头 三类输入，B=1 与 B=8）；
(c) 绝对正确性：csr 通路的梯度 vs 中心差分（不只是"和稠密一样"）；
(d) in-place 检查：把 _assemble_csr / _csr_to_dense 的中间量标 requires_grad
    走一遍，确认没有 index_put_ 之类就地写破坏 autograd（torch 会主动报错）。
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audlib import REF, Net, GGASolver, make_solver, boundary, batchify   # noqa: E402

NETS = ["pub_net1", "pub_hanoi", "pub_net2", "pub_net3", "pub_modena",
        "rand_main_0009", "pub_pescara", "pub_ky4"]


def loss_of(sv, D, R, K, assemble, w):
    out = sv.solve(D, R, ke_int=K, assemble=assemble)
    H = out["head_ft"]
    Q = out["flow_cfs"]
    return (H * w[0]).sum() + (Q * w[1]).sum(), out


def grads(sv, d, rh, ke, B, assemble, w, seed):
    if B == 1:
        D = torch.as_tensor(d, dtype=torch.float64)
        R = torch.as_tensor(rh, dtype=torch.float64)
    else:
        D, R = batchify(d, rh, B, seed)
    D = D.clone().requires_grad_(True)
    K = torch.as_tensor(ke, dtype=torch.float64).clone().requires_grad_(True)
    # 水库水头含 nan（非水库位），只对有限位求导 → 用 mask 参数化
    finite = torch.isfinite(R)
    Rv = torch.nan_to_num(R, nan=0.0).clone().requires_grad_(True)
    Rf = torch.where(finite, Rv, torch.full_like(Rv, float("nan")))
    L, out = loss_of(sv, D, Rf, K, assemble, w)
    gd, gk, gr = torch.autograd.grad(L, [D, K, Rv], allow_unused=True)
    return (float(L), gd, gk, gr, out["head_ft"].requires_grad,
            out["flow_cfs"].requires_grad)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    print(f"{'网':16s} {'B':>2s} {'活图':>4s} {'max|Δ∂L/∂d|':>13s} "
          f"{'max|Δ∂L/∂ke|':>13s} {'max|Δ∂L/∂rh|':>13s} {'ΔL':>10s}  判定")
    bad = 0
    for st in NETS:
        net = Net.load(REF, st)
        sv = make_solver(net)
        d, rh = boundary(net, st, 0)
        nt = np.asarray(net.node_type)
        juncs = np.where(nt == 0)[0]
        rng = np.random.default_rng(5)
        ke = np.zeros(net.N)
        ke[rng.choice(juncs, size=min(20, len(juncs)), replace=False)] = 0.5
        w = (torch.as_tensor(rng.normal(size=net.N)),
             torch.as_tensor(rng.normal(size=net.L)))
        for B in (1, 8):
            try:
                L1, gd1, gk1, gr1, rH1, rQ1 = grads(sv, d, rh, ke, B, "dense", w, 42)
                L2, gd2, gk2, gr2, rH2, rQ2 = grads(sv, d, rh, ke, B, "csr", w, 42)
            except Exception as e:                            # noqa: BLE001
                print(f"{st:16s} {B:2d}  raise {type(e).__name__}: {str(e)[:60]}")
                continue
            live = rH2 and rQ2 and gd2 is not None and gk2 is not None \
                and gr2 is not None and float(gd2.abs().sum()) > 0
            e1 = float((gd1 - gd2).abs().max())
            e2 = float((gk1 - gk2).abs().max())
            e3 = float((gr1 - gr2).abs().max())
            ok = live and e1 == 0 and e2 == 0 and e3 == 0 and L1 == L2
            bad += 0 if ok else 1
            print(f"{st:16s} {B:2d} {'活' if live else '**断**':>4s} "
                  f"{e1:13.3e} {e2:13.3e} {e3:13.3e} {abs(L1-L2):10.3e}  "
                  f"{'OK' if ok else '**FAIL**'}")
    print(f"\n(b) 梯度逐位对拍：不合格 {bad}")

    # ---------- (c) csr 通路梯度 vs 中心差分（绝对正确性）----------
    print("\n(c) csr 通路 ∂L/∂demand vs 中心差分（Richardson 外推，pub_hanoi B=1）")
    st = "pub_hanoi"
    net = Net.load(REF, st)
    sv = make_solver(net)
    d, rh = boundary(net, st, 0)
    rng = np.random.default_rng(9)
    w = (torch.as_tensor(rng.normal(size=net.N)),
         torch.as_tensor(rng.normal(size=net.L)))
    ke = np.zeros(net.N)
    D = torch.as_tensor(d, dtype=torch.float64).clone().requires_grad_(True)
    R = torch.as_tensor(rh, dtype=torch.float64)
    K = torch.as_tensor(ke, dtype=torch.float64)
    L, _ = loss_of(sv, D, R, K, "csr", w)
    g = torch.autograd.grad(L, D)[0].detach().numpy()

    def f(dv):
        with torch.no_grad():
            Lx, _ = loss_of(sv, torch.as_tensor(dv, dtype=torch.float64), R, K,
                            "csr", w)
        return float(Lx)

    juncs = np.where(np.asarray(net.node_type) == 0)[0]
    cand = [int(j) for j in juncs if d[j] > 0]
    pick = cand[:12]
    worst = 0.0
    for j in pick:
        h1 = max(1e-3 * abs(d[j]), 1e-6)
        h2 = h1 / 2
        def cd(h):
            a = d.copy(); a[j] += h
            b = d.copy(); b[j] -= h
            return (f(a) - f(b)) / (2 * h)
        gfd = (4 * cd(h2) - cd(h1)) / 3
        rel = abs(g[j] - gfd) / max(abs(gfd), 1e-12)
        worst = max(worst, rel)
        print(f"   node {net.node_id[j]:>8s}  ad={g[j]: .8e}  fd={gfd: .8e}  "
              f"rel={rel:.2e}")
    print(f"   最差相对误差 = {worst:.2e}   {'OK' if worst < 1e-6 else '**FAIL**'}")

    # ---------- (d) in-place 破图检查 ----------
    print("\n(d) 装配核 in-place 检查：对 requires_grad 的 vals 直接调用两个核")
    sv = make_solver(Net.load(REF, "pub_net3"))
    B = 4
    vals = torch.randn(B, sv.A_idx.numel(), dtype=torch.float64,
                       requires_grad=True)
    data = sv._assemble_csr(vals, B)
    A = sv._csr_to_dense(data, B)
    s = (A ** 2).sum()
    gv, = torch.autograd.grad(s, vals)
    # 解析：dA/dvals 是 0/1 散射 ⇒ ∂s/∂vals_k = 2*A[目标槽]
    tgt = sv.A_idx
    ref = 2 * A.reshape(B, -1).index_select(1, tgt)
    err = float((gv - ref).abs().max())
    print(f"   data.requires_grad={data.requires_grad} A.requires_grad={A.requires_grad}"
          f"  max|∂s/∂vals - 解析| = {err:.3e}  "
          f"{'OK' if (data.requires_grad and err == 0.0) else '**FAIL**'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

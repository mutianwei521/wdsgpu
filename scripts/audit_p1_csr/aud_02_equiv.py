# -*- coding: utf-8 -*-
"""审计①：CSR 与稠密两条通路的 **A 矩阵**与**最终解**独立对拍。

A 矩阵：不落盘、不复用上游断言 - 直接给 _assemble_csr 套壳，在每次迭代拿到
同一份 vals 时，就地用 A_idx 做一遍稠密 scatter_add，与 _csr_to_dense 的结果
逐元素比 max|ΔA|（真实迭代点上的值，不是人造 Pm）。
最终解：head/flow/emitter/relerr/iters 逐位比较 + 位指纹。
覆盖：dense 可跑的全部参考网 × B∈{1,8,64} × 有/无 emitter × 多个时刻 +
合成病态网（孤立 junction / 并联管 / 自环）。
"""
import os
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audlib import (REF, Net, GGASolver, SMALL, BIG, synth, make_solver,   # noqa: E402
                    boundary, batchify, bitfp)

_ORIG_ASM = GGASolver._assemble_csr
_ORIG_C2D = GGASolver._csr_to_dense
PROBE = {"maxdA": 0.0, "n": 0, "nz_mismatch": 0}
CHOL_FP = []
_ORIG_CHOL = torch.linalg.cholesky


def _asm_probe(self, vals, B):
    data = _ORIG_ASM(self, vals, B)
    got = _ORIG_C2D(self, data, B).reshape(B, -1)
    ref = torch.zeros(B, self.Nj * self.Nj, dtype=vals.dtype, device=vals.device)
    ref.scatter_add_(1, self.A_idx.expand(B, -1), vals)
    d = float((got - ref).abs().max().item())
    PROBE["maxdA"] = max(PROBE["maxdA"], d)
    PROBE["n"] += 1
    # 位型覆盖：稠密的非零位置必须全在 CSR 位型内（否则会被静默吞掉）
    nzpos = torch.nonzero(ref[0]).reshape(-1)
    inpat = torch.isin(nzpos, self.A_csr_dense_pos)
    if not bool(inpat.all()):
        PROBE["nz_mismatch"] += 1
    del ref, got
    return data


def _chol_probe(A, **kw):
    CHOL_FP.append(bitfp(A))
    return _ORIG_CHOL(A, **kw)


def run(sv, d, rh, ke, B, seed, assemble):
    D, R = batchify(d, rh, B, seed) if B > 1 else (
        torch.as_tensor(d, dtype=torch.float64),
        torch.as_tensor(rh, dtype=torch.float64))
    CHOL_FP.clear()
    out = sv.solve(D, R, ke_int=ke, assemble=assemble)
    return out, list(CHOL_FP)


def case(name, net, stem, Bs, tlist, seeds):
    sv = make_solver(net)
    nt = np.asarray(net.node_type)
    juncs = np.where(nt == 0)[0]
    rng = np.random.default_rng(11)
    ke_on = np.zeros(net.N)
    pick = rng.choice(juncs, size=max(1, min(30, len(juncs) // 3)), replace=False)
    ke_on[pick] = 0.5
    # 合成网上若有 iso+ke 节点，node_ke 已带值 → 合并
    ke_on = np.maximum(ke_on, np.asarray(net.node_ke, dtype=np.float64))
    ke_base = np.asarray(net.node_ke, dtype=np.float64)

    rows = []
    for t in tlist:
        d, rh = boundary(net, stem, t_sec=t)
        for B in Bs:
            for emit_name, ke in (("no-em", ke_base), ("em", ke_on)):
                PROBE.update(maxdA=0.0, n=0, nz_mismatch=0)
                try:
                    o1, fp1 = run(sv, d, rh, ke, B, seeds, "dense")
                except Exception as e:                     # noqa: BLE001
                    rows.append([name, str(t), str(B), emit_name,
                                 f"dense_raise:{type(e).__name__}", "", "", ""])
                    continue
                try:
                    o2, fp2 = run(sv, d, rh, ke, B, seeds, "csr")
                except Exception as e:                     # noqa: BLE001
                    rows.append([name, str(t), str(B), emit_name,
                                 f"csr_raise:{type(e).__name__}", "", "", ""])
                    continue
                dH = float((o1["head_ft"] - o2["head_ft"]).abs().max())
                dQ = float((o1["flow_cfs"] - o2["flow_cfs"]).abs().max())
                dE = float((o1["emitter_cfs"] - o2["emitter_cfs"]).abs().max())
                dR = float((o1["relerr"] - o2["relerr"]).abs().max())
                same_it = bool(torch.equal(o1["iters"], o2["iters"]))
                same_fp = (fp1 == fp2)
                fpH = bitfp(o1["head_ft"]) == bitfp(o2["head_ft"])
                fpQ = bitfp(o1["flow_cfs"]) == bitfp(o2["flow_cfs"])
                ok = (dH == 0 and dQ == 0 and dE == 0 and dR == 0 and same_it
                      and same_fp and fpH and fpQ and PROBE["maxdA"] == 0.0
                      and PROBE["nz_mismatch"] == 0)
                rows.append([name, str(t), str(B), emit_name,
                             "OK" if ok else "**FAIL**",
                             f"{PROBE['maxdA']:.3e}({PROBE['n']})",
                             f"{dH:.3e}/{dQ:.3e}/{dE:.3e}",
                             f"it={'=' if same_it else 'X'} "
                             f"Afp={'=' if same_fp else 'X'} "
                             f"HQfp={'=' if (fpH and fpQ) else 'X'}"])
    return rows


def main():
    torch.set_num_threads(1)
    GGASolver._assemble_csr = _asm_probe
    torch.linalg.cholesky = _chol_probe
    t0 = time.time()
    allrows, nfail = [], 0

    jobs = []
    for st in SMALL:
        jobs.append((st, Net.load(REF, st), st, [1, 8, 64], [0, 21600, 46800]))
    for st in BIG:
        jobs.append((st, Net.load(REF, st), st, [1, 8, 64], [0, 21600]))
    syn = [("SYN net1 par5", "pub_net1", dict(par=5)),
           ("SYN net1 self3", "pub_net1", dict(selfloop=3)),
           ("SYN net1 iso2+ke", "pub_net1", dict(iso=2, iso_ke=0.5)),
           ("SYN net1 iso2", "pub_net1", dict(iso=2)),
           ("SYN net1 all", "pub_net1", dict(iso=2, par=4, selfloop=2, iso_ke=0.5)),
           ("SYN hanoi all", "pub_hanoi", dict(iso=3, par=6, selfloop=3, iso_ke=0.5)),
           ("SYN net3 all", "pub_net3", dict(iso=3, par=6, selfloop=3, iso_ke=0.5)),
           ("SYN modena all", "pub_modena", dict(iso=5, par=9, selfloop=4, iso_ke=0.5)),
           ("SYN ky4 all", "pub_ky4", dict(iso=4, par=7, selfloop=3, iso_ke=0.5))]
    for nm, stem, kw in syn:
        net, _ = synth(stem, **kw)
        jobs.append((nm, net, stem, [1, 8, 64], [0, 21600]))

    for nm, net, stem, Bs, tl in jobs:
        rs = case(nm, net, stem, Bs, tl, seeds=hash(nm) % 9973)
        for r in rs:
            if r[4] != "OK":
                nfail += 1
            print("  ".join(x.ljust(w) for x, w in
                            zip(r, [17, 6, 4, 6, 24, 16, 34, 30])))
            sys.stdout.flush()
        allrows += rs
    print(f"\n场景数={len(allrows)}  非 OK={nfail}  用时 {time.time()-t0:.1f}s")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

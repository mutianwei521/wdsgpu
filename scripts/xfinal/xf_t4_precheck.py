# -*- coding: utf-8 -*-
"""XF-1 前置（本机 CPU，秒级）：先证明"我造的变体确实制造了不对称、
我造的对照确实没有"，再拿去 GPU 上问 T4 看不看得见。

做法：用真网的 A_csr 位型造一份**严格对称**的 csr_data，把 xf_t4_teeth.py 里
那几段扰动代码原样 exec 一遍（同一份字符串，不另写一份），量 max|A−A^T|。
"""
import io
import os
import sys
import textwrap

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import xf_t4_teeth as X                                    # noqa: E402

from dgga.parse import parse_inp                           # noqa: E402
from dgga.solver import GGASolver                          # noqa: E402

NETS = [("Hanoi", "Hanoi.inp"), ("Net3", "Net3.inp"),
        ("Modena", "Modena.inp"), ("ky4", "ky4.inp")]
NETD = os.path.join(ROOT, "networks", "public")


def tidx_of(s):
    Nj = int(s.Nj)
    row = s.A_csr_row.cpu().numpy().astype(np.int64)
    col = s.A_csr_col.cpu().numpy().astype(np.int64)
    key = row * Nj + col
    assert np.all(np.diff(key) > 0)
    pos = np.searchsorted(key, col * Nj + row)
    assert np.array_equal(key[pos], col * Nj + row)
    assert np.array_equal(pos[pos], np.arange(key.size))
    return torch.as_tensor(pos, dtype=torch.int64)


BLOCKS = {}
for name, exp, anc, rep in X.variants():
    if anc is X.A_EMIT and rep is not None:
        BLOCKS[name] = (exp, textwrap.dedent(rep[len(X.A_EMIT):]).strip("\n"))
    elif anc is X.A_ASM and rep is not None:
        body = rep[len(X.A_ASM):].split(':', 1)[1]
        BLOCKS[name] = (exp, textwrap.dedent(body).strip("\n"))


def main():
    print("=" * 92)
    print("XF-1 前置：我造的扰动块在真位型上到底造出多大不对称（CPU，float64）")
    print("=" * 92)
    bad = 0
    for stem, fn in NETS:
        p = os.path.join(NETD, fn)
        net = parse_inp(p)
        s = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                      inp_path=p, dense_tank_bound_check=False)
        t = tidx_of(s)
        B, nnz = 3, int(s.A_csr_nnz)
        g = torch.Generator().manual_seed(7)
        base = torch.rand(B, nnz, dtype=torch.float64, generator=g) + 0.5
        sym = 0.5 * (base + base.index_select(1, t))       # 严格对称
        sym = 0.5 * (sym + sym.index_select(1, t))
        w0 = float((sym - sym.index_select(1, t)).abs().max())
        print("\n%-7s Nj=%-4d nnz=%-5d 基线 max|A−A^T|=%.3e （必须 0）"
              % (stem, s.Nj, nnz, w0))
        if w0 != 0.0:
            print("   基线不对称，前置无效"); bad += 1; continue
        for name, (exp, blk) in BLOCKS.items():
            ns = dict(self=s, torch=torch, csr_data=sym.clone())
            exec(blk, ns)                                  # noqa: S102
            d = ns["csr_data"]
            w = float((d - d.index_select(1, t)).abs().max())
            dv = float((d - sym).abs().max())
            good = (w != 0.0) if exp else (w == 0.0)
            bad += 0 if good else 1
            print("   %-18s 期望不对称=%-5s | max|A−A^T|=%.6e 值变了=%.3e  %s"
                  % (name, exp, w, dv, "OK" if good else "**不合格**"))
    print("\n" + "=" * 92)
    print("前置总判定: %s" % ("OK（变体真造出不对称、对照真没有）" if bad == 0
                              else "**有 %d 项不合格**" % bad))
    print("=" * 92)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""aud_sym.py - 敌意复测（自写）：装配矩阵 A 的逐位对称。

覆盖（判据全部 max|A−A^T| == 0.0 逐位，逐 Newton 轮、逐样本，无容差）：
  ① 自建最小触发网 mixA（同节点对 3 链路 2正1反）/ mixB（4 链路 2正2反）
 - 不是上游 check_symmetry 的内嵌网，INP 文本自写；
  ② NW_Model（p5 审计的原触发网）；
  ③ L-TOWN / BWSN_Network_1，dense_status_machine=True，B=4 批
     （名义帧 + 逼 OPEN + 逼 CLOSED 场景各在批内） - **含 ACTIVE/CLOSED PRV
     的 CBIG 行在场的每一轮都要逐位对称**；终态逐阀状态当场打印作证；
  ④ 每网各跑 assemble="dense" 与 assemble="csr"（linear_solver 同为 dense）。

用法：python -X utf8 aud_sym.py [--pkg DIR]（--pkg 指变异体包，缺省仓库根）
退出码 0 = 全部轮次逐位对称。
"""
import argparse
import os
import sys
import tempfile

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", default=ROOT)
args = ap.parse_args()
sys.path.insert(0, os.path.abspath(args.pkg))
import dgga                                    # noqa: E402
import torch                                   # noqa: E402
from dgga.parse import parse_inp               # noqa: E402
from dgga.solver import GGASolver              # noqa: E402

print("dgga from:", os.path.dirname(os.path.abspath(dgga.__file__)))

MIX_A = """[TITLE]
audit mixA: node pair with 3 links, mixed direction
[JUNCTIONS]
 J1  50  10
 J2  40  120
[RESERVOIRS]
 R1  100
[PIPES]
 P0  R1  J1  1000  12  100  0  Open
 PA  J1  J2  800   10  110  0  Open
 PB  J1  J2  900   8   120  0  Open
 PC  J2  J1  700   6   130  0  Open
[OPTIONS]
 Units  GPM
 Headloss  H-W
 Trials  40
 Accuracy 0.001
[END]
"""
MIX_B = MIX_A.replace("mixA: node pair with 3 links, mixed direction",
                      "mixB: node pair with 4 links, 2 fwd 2 rev") + ""
MIX_B = MIX_B.replace(" PC  J2  J1  700   6   130  0  Open",
                      " PC  J2  J1  700   6   130  0  Open\n"
                      " PD  J2  J1  650   7   125  0  Open")

PUB = os.path.join(ROOT, "networks", "public")
CLEAN = os.path.join(PUB, "_cleaned")
NW = os.path.join(ROOT, "networks", "EXAMPLE", "epanet-example-networks",
                  "epanet-tests", "large", "NW_Model.inp")


def inp_of(name):
    p = os.path.join(CLEAN, name + ".inp")
    return p if os.path.exists(p) else os.path.join(PUB, name + ".inp")


def probe_solve(s, D, RH, sm, assemble):
    """跑 solve 并抓每轮 cholesky 输入 A，返回 [(it, worst, nbad_samples)]。"""
    rec = []
    oc_ex, oc = torch.linalg.cholesky_ex, torch.linalg.cholesky

    def spy_ex(A, *a, **k):
        rec.append(A.detach().clone())
        return oc_ex(A, *a, **k)

    def spy(A, *a, **k):
        rec.append(A.detach().clone())
        return oc(A, *a, **k)

    torch.linalg.cholesky_ex, torch.linalg.cholesky = spy_ex, spy
    exc = None
    out = {}
    try:
        with torch.no_grad():
            out = s.solve(torch.as_tensor(D), torch.as_tensor(RH),
                          status_machine=sm, assemble=assemble)
    except Exception as e:                              # noqa: BLE001
        exc = str(e)[:90]
    finally:
        torch.linalg.cholesky_ex, torch.linalg.cholesky = oc_ex, oc
    stats = []
    for A in rec:
        d = (A - A.transpose(-2, -1)).abs()
        w = float(d.max())
        nb = int((d.amax(dim=(-2, -1)) != 0.0).sum())
        stats.append((w, nb))
    return out, stats, exc


def scen(net, mode):
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                dtype=np.float64))
    tn = np.asarray(net.tank_node, dtype=np.int64)
    hmin = np.asarray(net.tank_hmin, dtype=np.float64)
    hmax = np.asarray(net.tank_hmax, dtype=np.float64)
    nt = np.asarray(net.node_type)
    junc = np.where(nt == 0)[0]
    tot = float(np.abs(d[junc]).sum())
    lt = np.asarray(net.link_type)
    prv = np.where(lt == 3)[0]
    if mode == "nom":
        for i, n in enumerate(tn):
            rh[int(n)] = 0.5 * (hmin[i] + hmax[i])
    elif mode == "open":
        d = d * 20.0
        if prv.size:
            n2 = int(net.link_n2[prv[0]])
            if nt[n2] == 0:
                d[n2] += 12.0 * max(tot, 1e-6)
        for i, n in enumerate(tn):
            rh[int(n)] = hmin[i]
    elif mode == "closed":
        d = d * 0.1
        if prv.size:
            n2 = int(net.link_n2[prv[0]])
            if nt[n2] == 0:
                d[n2] -= 2.0 * max(tot, 1e-6)
        for i, n in enumerate(tn):
            rh[int(n)] = hmax[i]
    return d, rh


fail = 0
cases = []
tmp = tempfile.mkdtemp(prefix="audsym_")
for nm, txt in (("mixA", MIX_A), ("mixB", MIX_B)):
    p = os.path.join(tmp, nm + ".inp")
    with open(p, "w", encoding="ascii") as f:
        f.write(txt)
    cases.append((nm, p, False))
cases.append(("NW_Model", NW, False))
cases.append(("L-TOWN", inp_of("L-TOWN"), True))
cases.append(("BWSN_Network_1", inp_of("BWSN_Network_1"), True))

for nm, path, sm in cases:
    net = parse_inp(path)
    lt = np.asarray(net.link_type)
    prv = np.where(lt == 3)[0]
    if sm:
        rows = [scen(net, m) for m in ("nom", "nom", "open", "closed")]
        rows[1] = (rows[1][0] * 2.5, rows[1][1])
        D = np.stack([r[0] for r in rows])
        RH = np.stack([r[1] for r in rows])
    else:
        d, rh = scen(net, "nom")
        D, RH = d[None, :], rh[None, :]
    s = GGASolver(net, mode="dense", dense_status_machine=sm)
    for asm in ("dense", "csr"):
        out, stats, exc = probe_solve(s, D, RH, sm, asm)
        worst = max((w for w, _ in stats), default=-1.0)
        nbad = sum(nb for _, nb in stats)
        extra = ""
        if prv.size and "status" in out:
            st = out["status"].numpy()
            hist = {v: int((st[:, prv] == v).sum()) for v in (2, 3, 4, 7)}
            extra = " PRV槽[CL=%d OP=%d AC=%d XP=%d]" % (
                hist[2], hist[3], hist[4], hist[7])
        ok = (len(stats) > 0 and worst == 0.0 and nbad == 0 and exc is None)
        fail += 0 if ok else 1
        print("  %-14s %-5s B=%d 轮=%d max|A-A^T|=%.3e 非对称样本轮=%d%s%s %s"
              % (nm, asm, D.shape[0], len(stats), worst, nbad, extra,
                 (" EXC:" + exc) if exc else "",
                 "PASS" if ok else "<-- FAIL"))

print("AUD_SYM_RESULT %s" % ("PASS" if fail == 0 else "FAIL"))
sys.exit(1 if fail else 0)

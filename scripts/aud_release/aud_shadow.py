# -*- coding: utf-8 -*-
"""aud_shadow.py - 敌意影子对拍（自写）：工作树 dgga vs git archive fb5c6be。

对同一批回归网跑全通路，把每个输出张量（含梯度张量）的 sha256 写进 JSON；
两个包各跑一次后用 --compare 逐键比对。通路：
  A epanet run_gga(do_status=True/False) - 缺省串行（铁律三 缺省1）
  B solve 缺省 dense/dense（B=2 批） - 缺省2+3；构造/求解 RAISE 也入键
  C solve dense+status_machine（新准入网除外）
  D solve csr 装配 + status_machine
  E solve_unrolled K=4 → d/r_hw 两个梯度张量
  F implicit_solve → d 梯度张量
PRV 网（L-TOWN/BWSN_Network_1）在 fb5c6be 上 C/D/E/F 构造期 RAISE，
现 HEAD 是新准入 - 比对时按"预期新能力"单列，不算不一致；其 A/B 必须逐位同。

用法：python -X utf8 aud_shadow.py --pkg DIR --out FILE.json
      python -X utf8 aud_shadow.py --compare old.json new.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PUB = os.path.join(ROOT, "networks", "public")
CLEAN = os.path.join(PUB, "_cleaned")

NETS = ["Hanoi", "Net1", "Net2", "Net3", "ky4", "Anytown_wntr",
        "Richmond_skeleton", "Modena", "Pescara", "Fossolo_poly1",
        "Balerma", "L-TOWN", "BWSN_Network_1"]
GRAD_NETS = {"Hanoi", "Net2", "Modena", "Pescara"}     # 无泵/水池的稳网做梯度


def inp_of(name):
    p = os.path.join(CLEAN, name + ".inp")
    return p if os.path.exists(p) else os.path.join(PUB, name + ".inp")


def h(x):
    import torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.ascontiguousarray(np.asarray(x))
    return hashlib.sha256(
        (str(a.dtype) + str(a.shape)).encode() + a.tobytes()).hexdigest()[:16]


def hs(s):
    return "EXC:" + hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:12]


def run_pkg(pkg, out_path):
    sys.path.insert(0, os.path.abspath(pkg))
    import dgga
    import torch
    from dgga.parse import parse_inp
    from dgga.solver import GGASolver
    from dgga.autodiff import solve_unrolled, implicit_solve
    print("dgga from:", os.path.dirname(os.path.abspath(dgga.__file__)))
    torch.manual_seed(0)
    res = {}
    for nm in NETS:
        net = parse_inp(inp_of(nm))
        d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
        rh = np.nan_to_num(np.array(net.reservoir_head_ft_at(0),
                                    dtype=np.float64))
        tn = np.asarray(net.tank_node, dtype=np.int64)
        if tn.size:
            rh[tn] = 0.5 * (np.asarray(net.tank_hmin)
                            + np.asarray(net.tank_hmax))
        D2 = np.stack([d, d * 1.5])
        R2 = np.stack([rh, rh])
        # A epanet
        try:
            se = GGASolver(net, mode="epanet")
            for tag, ds in (("st1", True), ("st0", False)):
                o = se.run_gga(d, rh, do_status=(tag == "st1"))
                for k in ("head", "flow", "emitter", "status", "setting"):
                    res[f"{nm}/epanet_{tag}/{k}"] = h(o[k])
                res[f"{nm}/epanet_{tag}/iters"] = int(o["iters"])
                res[f"{nm}/epanet_{tag}/relerr"] = h(np.float64(o["relerr"]))
        except Exception as e:                          # noqa: BLE001
            res[f"{nm}/epanet/EXC"] = hs(e)
        # B/C/D dense 路
        for tag, sm, asm in (("plain", False, "dense"),
                             ("sm", True, "dense"), ("sm_csr", True, "csr")):
            try:
                sd = GGASolver(net, mode="dense",
                               dense_status_machine=sm)
                with torch.no_grad():
                    o = sd.solve(torch.as_tensor(D2), torch.as_tensor(R2),
                                 status_machine=sm, assemble=asm)
                for k in ("head_ft", "flow_cfs", "emitter_cfs"):
                    res[f"{nm}/dense_{tag}/{k}"] = h(o[k])
                res[f"{nm}/dense_{tag}/iters"] = h(o["iters"])
                if "status" in o:
                    res[f"{nm}/dense_{tag}/status"] = h(o["status"])
            except Exception as e:                      # noqa: BLE001
                res[f"{nm}/dense_{tag}/EXC"] = hs(e)
        # E/F 梯度
        if nm in GRAD_NETS:
            try:
                sd = GGASolver(net, mode="dense")
                dt_ = torch.as_tensor(D2[:1])
                dg = dt_.clone().requires_grad_(True)
                rw = sd.r_hw.clone().requires_grad_(True)
                o = solve_unrolled(sd, dg, torch.as_tensor(R2[:1]),
                                   r_hw=rw, K=4)
                w = torch.linspace(0.5, 1.5, sd.Nj, dtype=sd.dtype)
                (o["head_ft"][:, sd.junc_nodes_t] * w).sum().backward()
                res[f"{nm}/unrolled/gd"] = h(dg.grad)
                res[f"{nm}/unrolled/gr"] = h(rw.grad)
            except Exception as e:                      # noqa: BLE001
                res[f"{nm}/unrolled/EXC"] = hs(e)
            try:
                se2 = GGASolver(net, mode="epanet")
                dg = torch.as_tensor(d)[None, :].clone().requires_grad_(True)
                head, flow, em = implicit_solve(
                    se2, dg, torch.as_tensor(rh)[None, :])
                w = torch.linspace(0.5, 1.5, head.shape[1], dtype=head.dtype)
                (head * w).sum().backward()
                res[f"{nm}/implicit/gd"] = h(dg.grad)
                res[f"{nm}/implicit/head"] = h(head)
            except Exception as e:                      # noqa: BLE001
                res[f"{nm}/implicit/EXC"] = hs(e)
        print("  done", nm, flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=0, sort_keys=True)
    print("keys:", len(res), "->", out_path)


def compare(a_path, b_path):
    A = json.load(open(a_path, encoding="utf-8"))
    B = json.load(open(b_path, encoding="utf-8"))
    keys = sorted(set(A) | set(B))
    same = diff = newcap = 0
    prv = ("L-TOWN", "BWSN_Network_1")
    for k in keys:
        va, vb = A.get(k), B.get(k)
        base = k.split("/")[0]
        is_new = base in prv and ("/dense_sm" in k or "/unrolled" in k
                                  or "/implicit" in k)
        if va == vb:
            same += 1
        elif is_new or (va is None) != (vb is None) and is_new:
            newcap += 1
        else:
            diff += 1
            print("  DIFF %-46s old=%s new=%s" % (k, va, vb))
    print("SHADOW_COMPARE same=%d diff=%d 新准入差=%d 总键=%d %s"
          % (same, diff, newcap, len(keys), "PASS" if diff == 0 else "FAIL"))
    return 0 if diff == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    run_pkg(a.pkg, a.out)

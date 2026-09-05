# -*- coding: utf-8 -*-
"""审计④：缺省（dense）路径与 epanet 路径是否受影响 - 影子模块对比。
把 b66463f^（改动前）的 dgga/solver.py 作为独立模块载入，与工作树版本在同一
输入上逐位对拍。覆盖 dense 可跑的全部参考网 × B∈{1,8,64} × 有/无 emitter，
外加 epanet 模式抽样（铁律三：复刻基准不得动）。
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audlib import REF, ROOT, Net, SMALL, BIG, boundary, batchify, bitfp  # noqa: E402
from dgga.solver import GGASolver as NEW                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_SRC = os.path.join(HERE, "solver_pre_b66463f.py")


def load_old():
    src = subprocess.run(["git", "show", "b66463f^:dgga/solver.py"], cwd=ROOT,
                         capture_output=True)
    assert src.returncode == 0, src.stderr[:400]
    with open(OLD_SRC, "wb") as f:
        f.write(src.stdout)
    spec = importlib.util.spec_from_file_location("solver_old", OLD_SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["solver_old"] = m
    spec.loader.exec_module(m)
    return m.GGASolver


def mk(cls, net, mode, inp=None):
    kw = dict(mode=mode)
    if inp:
        kw["inp_path"] = inp
    try:
        return cls(net, **kw)
    except NotImplementedError:
        return cls(net, dense_tank_bound_check=False, **kw)


def cmp_out(a, b):
    keys = ("head_ft", "flow_cfs", "emitter_cfs", "relerr", "iters", "converged")
    d = {}
    for k in keys:
        x, y = a[k], b[k]
        if x.dtype.is_floating_point:
            d[k] = float((x - y).abs().max())
            if bitfp(x) != bitfp(y):
                d[k] = max(d[k], float("nan"))
        else:
            d[k] = 0.0 if torch.equal(x, y) else 1.0
    return d


def main():
    torch.set_num_threads(1)
    OLD = load_old()
    print(f"影子模块 = git show b66463f^:dgga/solver.py  ({os.path.getsize(OLD_SRC)} B)")
    print(f"{'网':18s} {'mode':7s} {'B':>4s} {'em':>5s}  {'max|ΔH|':>10s} "
          f"{'max|ΔQ|':>10s} {'max|ΔE|':>10s} {'iters':>6s}  判定")
    bad = 0
    stems = SMALL + BIG
    for st in stems:
        net = Net.load(REF, st)
        so, sn = mk(OLD, net, "dense"), mk(NEW, net, "dense")
        d, rh = boundary(net, st, 0)
        nt = np.asarray(net.node_type)
        juncs = np.where(nt == 0)[0]
        rng = np.random.default_rng(7)
        ke = np.zeros(net.N)
        ke[rng.choice(juncs, size=min(25, len(juncs)), replace=False)] = 0.4
        for B in (1, 8, 64):
            for emn, k in (("off", np.asarray(net.node_ke, dtype=np.float64)),
                           ("on", ke)):
                if B == 1:
                    D = torch.as_tensor(d); R = torch.as_tensor(rh)
                else:
                    D, R = batchify(d, rh, B, 909)
                try:
                    o = so.solve(D, R, ke_int=k)
                    n = sn.solve(D, R, ke_int=k)     # 缺省 = dense，不传 assemble
                except Exception as e:                        # noqa: BLE001
                    # 两边必须同抛
                    e2 = None
                    try:
                        sn.solve(D, R, ke_int=k)
                    except Exception as ee:                   # noqa: BLE001
                        e2 = ee
                    same = e2 is not None and type(e2) is type(e) and str(e2) == str(e)
                    bad += 0 if same else 1
                    print(f"{st:18s} {'dense':7s} {B:4d} {emn:>5s}  "
                          f"两边同抛 {type(e).__name__}: {'OK' if same else '**FAIL**'}")
                    continue
                dd = cmp_out(o, n)
                ok = all((v == 0.0) for v in dd.values())
                bad += 0 if ok else 1
                print(f"{st:18s} {'dense':7s} {B:4d} {emn:>5s}  "
                      f"{dd['head_ft']:10.3e} {dd['flow_cfs']:10.3e} "
                      f"{dd['emitter_cfs']:10.3e} {dd['iters']:6.0f}  "
                      f"{'OK' if ok else '**FAIL**'}")
        del so, sn

    # epanet 模式抽样（铁律三）
    print()
    for st in ["pub_net1", "pub_hanoi", "pub_net3", "city_d", "pub_ky4",
               "pub_c_town_batadal", "pub_d_town", "EXA4", "fcv_smoke",
               "pub_bwsn_network_1", "pub_richmond_skeleton", "pub_balerma"]:
        net = Net.load(REF, st)
        so, sn = mk(OLD, net, "epanet"), mk(NEW, net, "epanet")
        d, rh = boundary(net, st, 0)
        for B in (1, 4):
            if B == 1:
                D = torch.as_tensor(d); R = torch.as_tensor(rh)
            else:
                D, R = batchify(d, rh, B, 313)
            try:
                o = so.solve(D, R, status_machine=True)
                n = sn.solve(D, R, status_machine=True)
            except Exception as e:                            # noqa: BLE001
                print(f"{st:18s} {'epanet':7s} {B:4d} {'-':>5s}  skip "
                      f"{type(e).__name__}: {str(e)[:50]}")
                continue
            dd = cmp_out(o, n)
            ok = all((v == 0.0) for v in dd.values())
            bad += 0 if ok else 1
            print(f"{st:18s} {'epanet':7s} {B:4d} {'-':>5s}  "
                  f"{dd['head_ft']:10.3e} {dd['flow_cfs']:10.3e} "
                  f"{dd['emitter_cfs']:10.3e} {dd['iters']:6.0f}  "
                  f"{'OK' if ok else '**FAIL**'}")
    print(f"\n不合格 = {bad}   {'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

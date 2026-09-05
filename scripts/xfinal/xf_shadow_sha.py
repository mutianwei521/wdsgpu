# -*- coding: utf-8 -*-
"""XF-5（本机前台，CPU float64）：影子包对拍 - 三个缺省通路的**逐字节** sha256。

影子基线取 **0a80e63**（`b66463f` 引入 CSR 装配之前的最后一次 dgga 改动），
也就是"cuDSS/CSR 那一整串工作开始之前"的 dgga。把它整棵取到临时目录，与工作树
各跑一个子进程，对下面每个量取 `tobytes()` 的 sha256 并逐项比：

  P1 mode="epanet" 逐位复刻     s.run_gga(...)：head / flow / iters / status
  P2 mode="dense" 缺省批量      s.solve(...)（assemble="dense", linear_solver="dense"）
  P3 缺省可微通路              solve_unrolled(...) 的 head/flow **与三个梯度**
                               （demand / res_head / ke），以及
                               implicit_solve(...) 的 head **与三个梯度**

梯度必须进 hash - 只比前向的话，"给稠密通路悄悄写了个伴随"这类改动查不出来。
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if not os.path.isdir(os.path.join(ROOT, "dgga")):
    ROOT = os.getcwd()
BASE = os.environ.get("XF_BASE", "0a80e63")

WORKER = r'''
import hashlib, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ["XF_TREE"])
sys.stdout.reconfigure(encoding="utf-8")
torch.use_deterministic_algorithms(True)
from dgga.parse import parse_inp
from dgga.solver import GGASolver
from dgga.autodiff import solve_unrolled, implicit_solve
ND = os.environ["XF_NETD"]
DT = torch.float64
H = {}

def sha(t):
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:24]

def scen(net, s, B, seed):
    g = np.random.default_rng(seed)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5*(np.asarray(net.tank_hmin)[:tn.size]
                      + np.asarray(net.tank_hmax)[:tn.size])
    rh0 = np.nan_to_num(rh0)
    D = torch.as_tensor(d0[None,:]*g.uniform(.9,1.1,(B,d0.size)), dtype=DT)
    R = torch.as_tensor(np.repeat(rh0[None,:],B,0)
                        + g.uniform(-1.,1.,(B,rh0.size))
                        * (np.asarray(net.node_type)!=0)[None,:], dtype=DT)
    ke = np.asarray(net.node_ke, dtype=np.float64).copy()
    jn = np.asarray(s.junc_nodes); ke[jn[g.random(jn.size) < .2]] = 1e-3
    KE = torch.as_tensor(np.repeat(ke[None,:],B,0), dtype=DT)
    W = torch.as_tensor(g.normal(0,1,(B,s.Nj)), dtype=DT)
    return D, R, KE, W

NETS = [("Net1","Net1.inp",4),("Hanoi","Hanoi.inp",4),("Net2","Net2.inp",5),
        ("Net3","Net3.inp",5),("Anytown","Anytown.inp",6),
        ("Modena","_cleaned/Modena.inp",5)]
for stem, fn, K in NETS:
    p = os.path.join(ND, fn)
    net = parse_inp(p)
    # ---- P1 mode="epanet"（逐位复刻通路，缺省） ----
    se = GGASolver(net, device="cpu", dtype=DT, mode="epanet", inp_path=p)
    d0 = torch.as_tensor(np.asarray(net.demand_cfs_at(0), dtype=np.float64), dtype=DT)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5*(np.asarray(net.tank_hmin)[:tn.size]
                      + np.asarray(net.tank_hmax)[:tn.size])
    fh = torch.as_tensor(np.nan_to_num(rh0), dtype=DT)
    with torch.no_grad():
        oe = se.run_gga(d0, fh)
    for k in ("head","flow","emitter","setting","iters","relerr",
              "fixed_demand"):
        if k in oe and oe[k] is not None:
            H["P1|%s|%s"%(stem,k)] = sha(np.asarray(oe[k], dtype=np.float64))
    if oe.get("status") is not None:
        H["P1|%s|status"%stem] = sha(np.asarray(oe["status"], dtype=np.int64))
    if not any(kk.startswith("P1|%s|"%stem) for kk in H):
        H["P1|%s|EMPTY"%stem] = repr(sorted(oe))[:80]
    # ---- P2/P3 mode="dense" ----
    try:
        s = GGASolver(net, device="cpu", dtype=DT, mode="dense", inp_path=p,
                      dense_tank_bound_check=False)
    except Exception as e:
        H["P2|%s|SKIP"%stem] = repr(e)[:60]; continue
    D, R, KE, W = scen(net, s, 4, 20260822)
    with torch.no_grad():
        od = s.solve(D, R, ke_int=KE)              # 三缺省：dense/dense/dense
    for k in ("head_ft","flow_cfs","iters"):
        if k in od: H["P2|%s|%s"%(stem,k)] = sha(od[k])
    # ---- P3a solve_unrolled 缺省 + 三个梯度 ----
    Dg = D.clone().requires_grad_(True); Rg = R.clone().requires_grad_(True)
    Kg = KE.clone().requires_grad_(True)
    ou = solve_unrolled(s, Dg, Rg, ke=Kg, K=K)
    L = (ou["head_ft"][:, s.junc_nodes_t]*W).sum()
    L.backward()
    H["P3u|%s|head"%stem] = sha(ou["head_ft"])
    if "flow_cfs" in ou: H["P3u|%s|flow"%stem] = sha(ou["flow_cfs"])
    H["P3u|%s|gD"%stem] = sha(Dg.grad); H["P3u|%s|gR"%stem] = sha(Rg.grad)
    H["P3u|%s|gK"%stem] = sha(Kg.grad)
    # ---- P3b implicit_solve 缺省 + 三个梯度 ----
    try:
        D2 = D.clone().requires_grad_(True); R2 = R.clone().requires_grad_(True)
        K2 = KE.clone().requires_grad_(True)
        oi = implicit_solve(s, D2, R2, ke=K2)
        Hi = oi[0] if isinstance(oi, (tuple, list)) else oi["head_ft"]
        (Hi[:, s.junc_nodes_t]*W).sum().backward()
        H["P3i|%s|head"%stem] = sha(Hi)
        H["P3i|%s|gD"%stem] = sha(D2.grad); H["P3i|%s|gR"%stem] = sha(R2.grad)
        H["P3i|%s|gK"%stem] = sha(K2.grad)
    except Exception as e:
        H["P3i|%s|ERR"%stem] = repr(e)[:70]
print("XF5-JSON " + json.dumps(H))
'''


def shadow_tree():
    tmp = tempfile.mkdtemp(prefix="xfsh_")
    r = subprocess.run(["git", "archive", BASE, "dgga"], cwd=ROOT,
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit("git archive 失败: " + r.stderr.decode()[:300])
    tar = os.path.join(tmp, "a.tar")
    open(tar, "wb").write(r.stdout)
    subprocess.run(["tar", "-xf", tar], cwd=tmp, check=True)
    os.remove(tar)
    return tmp


def run(tree, netd):
    wd = tempfile.mkdtemp(prefix="xfw_")
    wf = os.path.join(wd, "xfw.py")
    io.open(wf, "w", encoding="utf-8").write(WORKER)
    env = dict(os.environ, XF_TREE=tree, XF_NETD=netd, PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, "-X", "utf8", wf],
                       capture_output=True, text=True, env=env, cwd=tree)
    for ln in r.stdout.splitlines():
        if ln.startswith("XF5-JSON "):
            return json.loads(ln[9:])
    raise SystemExit("worker 没出 JSON：\n" + r.stdout[-800:] + "\n" + r.stderr[-1500:])


def main():
    netd = os.path.join(ROOT, "networks", "public")
    sh = shadow_tree()
    print("=" * 100)
    print("XF-5 影子包对拍：工作树 vs %s（cuDSS/CSR 那串工作开始之前的 dgga）" % BASE)
    print("影子树: %s" % sh)
    sha_cur = hashlib.sha256(io.open(os.path.join(ROOT, "dgga", "solver.py"),
                                     "rb").read()).hexdigest()[:16]
    sha_old = hashlib.sha256(io.open(os.path.join(sh, "dgga", "solver.py"),
                                     "rb").read()).hexdigest()[:16]
    print("solver.py sha256: 工作树 %s | 影子 %s（**必须不同**，否则影子没取对）"
          % (sha_cur, sha_old))
    print("=" * 100, flush=True)
    a = run(ROOT, netd)
    b = run(sh, netd)
    keys = sorted(set(a) | set(b))
    same = diff = only = 0
    for k in keys:
        if k not in a or k not in b:
            only += 1
            print("  %-28s 只在 %s 侧：%s" % (k, "工作树" if k in a else "影子",
                                              a.get(k, b.get(k))))
        elif a[k] == b[k]:
            same += 1
        else:
            diff += 1
            print("  %-28s **不同**  工作树 %s | 影子 %s" % (k, a[k], b[k]))
    print("\n  逐字节相同 %d 项 | 不同 %d 项 | 单边 %d 项（总 %d）"
          % (same, diff, only, len(keys)))
    if diff == 0 and only == 0:
        print("  抽样（前 6 条 sha256）：")
        for k in keys[:6]:
            print("    %-28s %s" % (k, a[k]))
    print("\n总判定: %s" % ("缺省通路逐字节未变" if (diff == 0 and only == 0)
                            else "**缺省通路有变化**"))
    print("=" * 100)
    shutil.rmtree(sh, ignore_errors=True)
    return 0 if (diff == 0 and only == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

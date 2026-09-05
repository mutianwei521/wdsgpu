# -*- coding: utf-8 -*-
"""AUDIT R1-b - A 对称守卫在 **cuDSS 专属分支** 上的覆盖缺口（GPU 上求证）。

本机 CPU 上我已经量出：往 solver.py 里 `linear_solver == "cudss"` 那条分支
（csr_data 的 emitter index_add 之后）注入一个单侧写，check_symmetry.py
**全绿放行**（72/72 PASS），因为那条分支在 CPU 上根本不可达。

这里在真 GPU 上把这件事坐实，并量出后果：
  P1 正品：cuDSS 通路实际拿到的 csr_data（含 emitter 对角）逐位对称吗？
     判据 max|data[k] − data[t(k)]| == 0.0（与 check_symmetry 同一判据）。
  P2 注入一个单侧写（源码级、独立 dgga 树）后：
     · 前向 head 与迭代数变不变（会不会被 T1 抓到）；
     · 反向梯度错多少（伴随复用的正是这个分解）；
     · regression_gpu.py 的三条断言会不会红。
  ⇒ 若 P2 的前向/迭代数/三条断言全绿而梯度是错的，缺口就成立。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
NETD = os.path.join(HERE, "p2nets")
NODE = os.popen("hostname").read().strip()

INJECT = (
    "solver.py",
    "                csr_data = csr_data.index_add(1, self.A_csr_diag, "
    "em / hgrad_e)",
    "                csr_data = csr_data.index_add(1, self.A_csr_diag, "
    "em / hgrad_e)\n"
    "                _off = (self.A_csr_row != self.A_csr_col).nonzero()"
    ".reshape(-1)\n"
    "                csr_data = csr_data.index_add(\n"
    "                    1, _off[:1], "
    "csr_data.index_select(1, _off[:1]) * MAG_PLACEHOLDER)",
)

WORKER = r'''
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ["AUD_TREE"])
sys.stdout.reconfigure(encoding="utf-8")
from dgga.parse import parse_inp
from dgga.solver import GGASolver
from dgga.autodiff import solve_unrolled
NETD = os.environ["AUD_NETD"]
DT = torch.float64
res = {}
for stem, fn, K in (("Hanoi", "Hanoi.inp", 6), ("Net3", "Net3.inp", 6),
                    ("Modena", "Modena.inp", 6), ("ky4", "ky4.inp", 5)):
    p = os.path.join(NETD, fn)
    net = parse_inp(p)
    s = GGASolver(net, device="cuda", dtype=DT, mode="dense", inp_path=p,
                  dense_tank_bound_check=False)
    g = np.random.default_rng(20260822)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5*(np.asarray(net.tank_hmin)[:tn.size]
                      + np.asarray(net.tank_hmax)[:tn.size])
    rh0 = np.nan_to_num(rh0)
    B = 8
    D = torch.as_tensor(d0[None,:]*g.uniform(.9,1.1,(B,d0.size)), dtype=DT, device="cuda")
    R = torch.as_tensor(np.repeat(rh0[None,:],B,0)
                        + g.uniform(-1.,1.,(B,rh0.size))
                        * (np.asarray(net.node_type)!=0)[None,:], dtype=DT, device="cuda")
    ke = np.asarray(net.node_ke, dtype=np.float64).copy()
    jn = np.asarray(s.junc_nodes); ke[jn[g.random(jn.size) < .2]] = 1e-3
    KE = torch.as_tensor(np.repeat(ke[None,:],B,0), dtype=DT, device="cuda")
    W = torch.as_tensor(g.normal(0,1,(B,s.Nj)), dtype=DT, device="cuda")
    # 转置槽
    Nj = s.Nj
    row = s.A_csr_row.cpu().numpy().astype(np.int64)
    col = s.A_csr_col.cpu().numpy().astype(np.int64)
    key = row*Nj+col; tk = col*Nj+row
    pos = np.clip(np.searchsorted(key, tk), 0, key.size-1)
    tidx = torch.as_tensor(pos, dtype=torch.int64, device="cuda")
    # 捕获 cuDSS 真正拿到的 csr_data
    caps = []
    raw = GGASolver._cudss_forward
    def cap(self_, data_, F_, B_, refine=None, slot=0):
        caps.append(data_.detach().clone())
        return raw(self_, data_, F_, B_, refine, slot)
    GGASolver._cudss_forward = cap
    try:
        with torch.no_grad():
            od = s.solve(D, R, ke_int=KE)
            oc = s.solve(D, R, ke_int=KE, assemble="csr", linear_solver="cudss")
    finally:
        GGASolver._cudss_forward = raw
    worst = max(float((c - c.index_select(1, tidx)).abs().max()) for c in caps)
    it_d = od["iters"].reshape(-1).tolist(); it_c = oc["iters"].reshape(-1).tolist()
    dH = float((od["head_ft"]-oc["head_ft"]).abs().max())
    # 梯度：cudss vs dense（同一 K，同一批）
    dv = D.clone().requires_grad_(True)
    o = solve_unrolled(s, dv, R, ke=KE, K=K, assemble="csr", linear_solver="cudss")
    (o["head_ft"][:, s.junc_nodes_t]*W).sum().backward()
    gc = dv.grad.detach().clone()
    dv2 = D.clone().requires_grad_(True)
    o2 = solve_unrolled(s, dv2, R, ke=KE, K=K)
    (o2["head_ft"][:, s.junc_nodes_t]*W).sum().backward()
    gd = dv2.grad.detach().clone()
    rel = float((gc-gd).abs().max()/gd.abs().max())
    res[stem] = dict(nrounds=len(caps), worst_sym=worst, iters_same=(it_d==it_c),
                     iters_d=sorted(set(it_d)), iters_c=sorted(set(it_c)),
                     maxdH=dH, grad_rel=rel)
    s.cudss_free(); del s
print("WORKER-JSON " + json.dumps(res))
'''


def build(tree_patch):
    tmp = tempfile.mkdtemp(prefix="audsg_")
    shutil.copytree(os.path.join(HERE, "dgga"), os.path.join(tmp, "dgga"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(tmp, "scripts"))
    shutil.copy2(os.path.join(HERE, "regression_gpu.py"),
                 os.path.join(tmp, "scripts", "regression_gpu.py"))
    if tree_patch:
        f, old, new = INJECT
        new = new.replace("MAG_PLACEHOLDER",
                          repr(float(os.environ.get("AUD_MAG", "1e-9"))))
        tgt = os.path.join(tmp, "dgga", f)
        src = open(tgt, encoding="utf-8").read()
        if src.count(old) != 1:
            raise SystemExit("锚点命中 %d 次" % src.count(old))
        open(tgt, "w", encoding="utf-8").write(src.replace(old, new))
    return tmp


def run_worker(tree):
    wf = os.path.join(tree, "w.py")
    open(wf, "w", encoding="utf-8").write(WORKER)
    env = dict(os.environ, AUD_TREE=tree, AUD_NETD=NETD,
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    cp = subprocess.run([sys.executable, "-X", "utf8", wf], env=env, cwd=tree,
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=5400)
    out = (cp.stdout or "") + (cp.stderr or "")
    ln = [x for x in out.splitlines() if x.startswith("WORKER-JSON")]
    return (json.loads(ln[-1][len("WORKER-JSON "):]) if ln else None), out


def run_reg(tree):
    env = dict(os.environ, DGGA_NETS=NETD, PYTHONPATH=tree,
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    cp = subprocess.run([sys.executable, "-X", "utf8",
                         os.path.join(tree, "scripts", "regression_gpu.py")],
                        env=env, cwd=tree, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=7200)
    out = (cp.stdout or "") + (cp.stderr or "")
    v = [x for x in out.splitlines() if x.startswith("总判定")]
    return cp.returncode, (v[-1] if v else "(无总判定)"), out


print("=" * 100)
print("AUDIT R1-b - cuDSS 专属分支的对称守卫缺口 | node:", NODE,
      "| 注入相对量 AUD_MAG =", os.environ.get("AUD_MAG", "1e-9"))
import torch                                                     # noqa: E402
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
print("md5 dgga/solver.py",
      hashlib.md5(open(os.path.join(HERE, "dgga/solver.py"), "rb").read())
      .hexdigest())
print("=" * 100)

for tag, patched in (("P1 正品", False), ("P2 注入单侧写(仅 cudss 分支)", True)):
    tree = build(patched)
    d, out = run_worker(tree)
    print("\n【%s】" % tag)
    if d is None:
        print("  worker 无输出，尾部：", out.strip().splitlines()[-4:])
    else:
        for k, v in d.items():
            print("  %-8s 捕获轮=%-2d  max|data-data^T|=%.3e  迭代数相等=%s "
                  "%s/%s  max|dH|=%.3e ft  梯度 cudss vs dense 相对差=%.3e"
                  % (k, v["nrounds"], v["worst_sym"], v["iters_same"],
                     v["iters_d"], v["iters_c"], v["maxdH"], v["grad_rel"]))
    rc, verdict, rout = run_reg(tree)
    # 项数不写死：P4 收尾给 regression_gpu.py 加了第四条 T4，40 -> 50 项。
    print("  regression_gpu.py（全量）: rc=%d  %s" % (rc, verdict))
    open(os.path.join(HERE, "aud_symgap_%s.txt"
                      % ("patched" if patched else "clean")), "w",
         encoding="utf-8").write(out + "\n\n===== regression_gpu =====\n" + rout)
    shutil.rmtree(tree, ignore_errors=True)
    sys.stdout.flush()
print("\n" + "=" * 100)

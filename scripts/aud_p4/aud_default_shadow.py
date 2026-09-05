# -*- coding: utf-8 -*-
"""AUDIT 5 - 缺省通路逐字节对拍（影子包）。

问题：P4 这三个提交（f29215e / 2b8345c+4dd344c / 9ee0ba3）有没有动到
  mode="epanet" 逐位复刻路径、assemble="dense"、linear_solver="dense"
这三个缺省？特别是**有没有人顺手给稠密通路写了伴随**（铁律三 / 审计 R9） -
那会静默改掉缺省梯度，源码 diff 看着"只加了个函数"也可能被 monkeypatch 之类
的东西绕过。所以这里不看 diff，直接**跑**：

  影子包 = git archive <P4 之前的提交> dgga  →  临时目录
  正品包 = 工作区 dgga
两边各起一个子进程，用**完全缺省**的调用跑同一批网/批量，把每个输出张量的
原始 float64 字节做 sha256，逐项比。前向与**梯度**都比。

用法：python -X utf8 scripts/aud_p4/aud_default_shadow.py [BASE_COMMIT]
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

CASES = [("Hanoi", "public/Hanoi.inp", 1), ("Hanoi", "public/Hanoi.inp", 4),
         ("Net3", "public/Net3.inp", 4), ("Modena", "public/Modena.inp", 4),
         ("Net1", "public/Net1.inp", 4), ("ky4", "public/ky4.inp", 2),
         ("city_d", "realInpData/city_d.inp", 2),
         ("city_h", "InpData/city_h.inp", 2),
         ("EXA6", "InpData/EXA6.inp", 4),
         ("rand9", "random_main/rand_0009.inp", 4)]

WORKER = r'''
# -*- coding: utf-8 -*-
import hashlib, json, os, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
PKG = os.environ["AUD_PKG"]; NETS = os.environ["AUD_NETS"]
sys.path.insert(0, PKG)
import dgga.solver as smod
from dgga.parse import parse_inp
from dgga.solver import GGASolver
from dgga.autodiff import solve_unrolled, ImplicitGGASolve, solve_polished
assert os.path.abspath(smod.__file__).startswith(os.path.abspath(PKG)), smod.__file__
CASES = json.loads(os.environ["AUD_CASES"])

def h(t):
    if t is None: return "None"
    a = np.ascontiguousarray(t.detach().cpu().numpy())
    return hashlib.sha256(a.tobytes()).hexdigest()[:32] + "|" + str(a.dtype) + str(a.shape)

out = {}
for label, rel, B in CASES:
    p = os.path.join(NETS, rel)
    net = parse_inp(p)
    g = np.random.default_rng(20260822)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5*(np.asarray(net.tank_hmin)[:tn.size] + np.asarray(net.tank_hmax)[:tn.size])
    rh0 = np.nan_to_num(rh0)
    D = torch.as_tensor(d0[None,:]*g.uniform(.85,1.15,(B,d0.size)), dtype=torch.float64)
    R = torch.as_tensor(np.repeat(rh0[None,:],B,0) + g.uniform(-1.,1.,(B,rh0.size))
                        * (np.asarray(net.node_type)!=0)[None,:], dtype=torch.float64)
    ke = np.asarray(net.node_ke, dtype=np.float64).copy()
    for mode in ("epanet", "dense"):
        try:
            s = GGASolver(net, mode=mode, inp_path=p, dense_tank_bound_check=False)
        except Exception as e:
            out["%s/%s/CTOR" % (label, mode)] = "CTOR-ERR " + type(e).__name__
            continue
        jn = np.asarray(s.junc_nodes)
        ke2 = ke.copy(); ke2[jn[g.random(jn.size) < .3]] = 0.5
        KE = torch.as_tensor(np.repeat(ke2[None,:],B,0), dtype=torch.float64)
        W = torch.as_tensor(g.normal(0,1,(B,net.N)), dtype=torch.float64)
        pre = "%s/%s" % (label, mode)
        # --- 缺省 solve（一个可选参数都不传） ---
        try:
            with torch.no_grad():
                o = s.solve(D, R, ke_int=KE)
            for k in ("head_ft","flow_cfs","emitter_cfs","iters","relerr"):
                if k in o: out["%s/solve/%s" % (pre,k)] = h(o[k])
        except Exception as e:
            out["%s/solve" % pre] = "ERR " + type(e).__name__ + ":" + str(e)[:60]
        if mode != "dense":
            del s; continue
        K = 6
        # --- 缺省 solve_unrolled + 四类参数的梯度 ---
        for param in ("demand","res_head","ke","r_hw"):
            try:
                th = dict(demand=D.clone(), res_head=R.clone(), ke=KE.clone(),
                          r_hw=s.r_hw.clone())
                th[param].requires_grad_(True)
                o = solve_unrolled(s, th["demand"], th["res_head"], ke=th["ke"],
                                   r_hw=th["r_hw"], K=K)
                (o["head_ft"]*W).sum().backward()
                out["%s/unrolled/%s/head" % (pre,param)] = h(o["head_ft"])
                out["%s/unrolled/%s/flow" % (pre,param)] = h(o["flow_cfs"])
                out["%s/unrolled/%s/grad" % (pre,param)] = h(th[param].grad)
            except Exception as e:
                out["%s/unrolled/%s" % (pre,param)] = "ERR " + type(e).__name__ + ":" + str(e)[:60]
        # --- ImplicitGGASolve（隐式伴随）demand 梯度 ---
        try:
            dv = D.clone().requires_grad_(True)
            hh,_q,_e = ImplicitGGASolve.apply(dv, R.clone(), KE.clone(),
                                              s.r_hw.clone(), s, 1e-12, 200, 3)
            (hh*W).sum().backward()
            out["%s/implicit/head" % pre] = h(hh)
            out["%s/implicit/grad_d" % pre] = h(dv.grad)
        except Exception as e:
            out["%s/implicit" % pre] = "ERR " + type(e).__name__ + ":" + str(e)[:60]
        # --- solve_polished ---
        try:
            with torch.no_grad():
                op = solve_polished(s, D, R, ke=KE)
            for kk in ("head","q","e_j","emitter","resid_inf","iters"):
                if kk in op:
                    out["%s/polished/%s" % (pre, kk)] = h(torch.as_tensor(np.asarray(op[kk])))
        except Exception as e:
            out["%s/polished" % pre] = "ERR " + type(e).__name__ + ":" + str(e)[:60]
        del s
print("WORKER-JSON " + json.dumps(out))
'''


def run(pkg_root, cases):
    wf = os.path.join(tempfile.gettempdir(), "aud_shadow_worker.py")
    open(wf, "w", encoding="utf-8").write(WORKER)
    env = dict(os.environ, AUD_PKG=pkg_root,
               AUD_NETS=os.path.join(ROOT, "networks"),
               AUD_CASES=json.dumps(cases), PYTHONPATH=pkg_root)
    cp = subprocess.run([sys.executable, "-X", "utf8", wf], env=env, cwd=ROOT,
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=5400)
    o = (cp.stdout or "") + (cp.stderr or "")
    ln = [x for x in o.splitlines() if x.startswith("WORKER-JSON")]
    if not ln:
        print("worker 失败 rc=%d:\n%s" % (cp.returncode, o[-3000:]))
        raise SystemExit(2)
    return json.loads(ln[-1][len("WORKER-JSON "):])


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "ad36c31"
    import tarfile
    tmp = tempfile.mkdtemp(prefix="audshadow_")
    tf = os.path.join(tmp, "_pkg.tar")
    with open(tf, "wb") as fh:
        cp = subprocess.run(["git", "archive", base, "dgga"], cwd=ROOT,
                            stdout=fh)
    assert cp.returncode == 0, cp
    with tarfile.open(tf) as t:
        t.extractall(tmp)
    os.remove(tf)
    print("=" * 100)
    print("AUDIT 5 - 缺省通路逐字节对拍（影子包 = %s，正品 = 工作区）" % base)
    print("影子包路径:", tmp)
    for f in ("solver.py", "autodiff.py"):
        a = hashlib.md5(open(os.path.join(tmp, "dgga", f), "rb").read()).hexdigest()
        b = hashlib.md5(open(os.path.join(ROOT, "dgga", f), "rb").read()).hexdigest()
        print("  md5 %-12s 影子=%s 正品=%s %s"
              % (f, a[:16], b[:16], "同" if a == b else "**不同**"))
    print("=" * 100)
    old = run(os.path.join(tmp), CASES)
    new = run(ROOT, CASES)
    keys = sorted(set(old) | set(new))
    diff = [k for k in keys if old.get(k) != new.get(k)]
    err = [k for k in keys if str(new.get(k)).startswith("ERR")
           or str(new.get(k)).startswith("CTOR-ERR")]
    print("比对项数: %d（其中报错项 %d，两边同样报错也算比对通过）" % (len(keys), len(err)))
    if err:
        for k in err[:12]:
            print("   [两边同样报错] %-52s %s" % (k, str(new.get(k))[:70]))
    print("**不一致项: %d**" % len(diff))
    for k in diff[:40]:
        print("   %-56s 影子=%s\n   %-56s 正品=%s"
              % (k, str(old.get(k))[:60], "", str(new.get(k))[:60]))
    ngrad = sum(1 for k in keys if "/grad" in k)
    print("其中梯度项 %d 个，全部逐字节相同: %s"
          % (ngrad, "是" if not any("/grad" in k for k in diff) else "**否**"))
    print("=" * 100)
    print("AUD-SHADOW-VERDICT base=%s 比对=%d 不一致=%d 梯度项=%d"
          % (base, len(keys), len(diff), ngrad))
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main())

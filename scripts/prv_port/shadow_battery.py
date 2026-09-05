# -*- coding: utf-8 -*-
"""shadow_battery.py - PRV 轮的影子包对拍（铁律三：三个缺省逐位不变）。

影子包 = `git archive <BASE> dgga`（缺省 BASE = HEAD，即本轮改动的父提交），
正品包 = 工作树 dgga。两边各起一个子进程跑同一套确定性 battery，把每个输出
张量的原始字节 sha256 逐键比对。**梯度必须进 hash**（只比前向查不出"顺手给
缺省通路写了伴随"这类改动）。

覆盖（26 网，逐网按能力选通路）：
  · epanet run_gga：do_status=False / True（含 schedule 关），head/flow/emitter/
    status/setting/iters/relerr/fixed_demand 全 hash；
  · 缺省 dense：solve(D,R,ke)（assemble=dense/linear_solver=dense 三缺省）；
  · dense+批量状态机：solve(..., status_machine=True)，assemble=dense 与 csr；
  · dense 冻结（无 SM）+ csr 装配；
  · solve_unrolled 前向 + gd/gR/gK 三梯度；implicit_solve 前向 + 三梯度；
  · solve_polished 的 q/head/resid；
  · RAISE 行为：**plain dense** 的构造异常（类型+消息，两侧文案未变）；
    dense+SM 的构造异常只比类型（放行集合文案本轮扩了 PRV，属预期准入变化）。
  · 含 PRV 的网（L-TOWN / BWSN_Network_1 / D-Town / Richmond_standard / ky10）
    只进 epanet 通路与 plain-dense RAISE 的比对 - dense+SM 对它们是**新准入**
    （影子包构造期 raise），不构成"既有行为"。

用法：python -X utf8 scripts/prv_port/shadow_battery.py [BASE_COMMIT]
退出码 0 = 全部键逐字节相同。
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
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = sys.argv[1] if len(sys.argv) > 1 else "HEAD"

# (标签, inp 相对 networks/ 路径, B, unrolled K, 含 PRV？)
CASES = [
    ("Richmond_skeleton", "public/Richmond_skeleton.inp", 3, 6, False),
    ("Anytown_wntr", "public/Anytown_wntr.inp", 3, 6, False),
    ("Net1", "public/Net1.inp", 4, 6, False),
    ("Net2", "public/Net2.inp", 3, 6, False),
    ("Net3", "public/Net3.inp", 3, 6, False),
    ("Anytown", "public/Anytown.inp", 3, 6, False),
    ("Hanoi", "public/Hanoi.inp", 4, 6, False),
    ("Modena", "public/_cleaned/Modena.inp", 3, 6, False),
    ("Fossolo", "public/_cleaned/Fossolo_poly1.inp", 3, 6, False),
    ("Pescara", "public/_cleaned/Pescara.inp", 3, 6, False),
    ("ky4", "public/ky4.inp", 2, 5, False),
    ("EXA4", "InpData/EXA4.inp", 2, 5, True),    # 含 PRV3（+CV/TCV/泵）
    ("EXA5", "InpData/EXA5.inp", 2, 5, True),    # 含 PRV3（+泵）
    ("EXA6", "InpData/EXA6.inp", 2, 5, False),
    ("city_h", "InpData/city_h.inp", 2, 5, False),
    ("ky3", "InpData/ky3.inp", 2, 5, False),
    ("ky5", "InpData/ky5.inp", 2, 5, False),
    ("city_d", "realInpData/city_d.inp", 2, 5, False),
    ("city_d_emit", "variants/city_d_emit.inp", 2, 5, False),
    ("sym_torture", "variants/sym_torture.inp", 3, 6, False),
    ("fcv_smoke", "variants/fcv_smoke.inp", 2, 5, False),
    ("rand9", "random_main/rand_0009.inp", 3, 6, False),
    ("rand14", "random_main/rand_0014.inp", 3, 6, False),
    ("rands0", "random_small/rand_0000.inp", 3, 6, False),
    ("Balerma", "public/Balerma.inp", 2, 5, False),
    # 含 PRV：只比 epanet 通路 + plain-dense RAISE
    ("L-TOWN", "public/_cleaned/L-TOWN.inp", 1, 0, True),
    ("BWSN_1", "public/_cleaned/BWSN_Network_1.inp", 1, 0, True),
    ("D-Town", "public/D-Town.inp", 1, 0, True),
    ("Richmond_std", "public/Richmond_standard.inp", 1, 0, True),
    ("ky10", "public/ky10.inp", 1, 0, True),
]

WORKER = r'''
# -*- coding: utf-8 -*-
import hashlib, json, os, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
PKG = os.environ["SB_PKG"]; NETS = os.environ["SB_NETS"]
sys.path.insert(0, PKG)
import dgga.solver as smod
from dgga.parse import parse_inp
from dgga.solver import GGASolver
from dgga.autodiff import solve_unrolled, implicit_solve, solve_polished
assert os.path.abspath(smod.__file__).startswith(os.path.abspath(PKG)), smod.__file__
CASES = json.loads(os.environ["SB_CASES"])

def h(t):
    if t is None:
        return "None"
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()[:24] + "|" + str(a.dtype) + str(a.shape)

def scen(net, s, B, seed):
    g = np.random.default_rng(seed)
    d0 = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh0 = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh0[tn] = .5*(np.asarray(net.tank_hmin)[:tn.size]
                      + np.asarray(net.tank_hmax)[:tn.size])
    rh0 = np.nan_to_num(rh0)
    D = torch.as_tensor(d0[None, :]*g.uniform(.9, 1.1, (B, d0.size)),
                        dtype=torch.float64)
    R = torch.as_tensor(np.repeat(rh0[None, :], B, 0)
                        + g.uniform(-1., 1., (B, rh0.size))
                        * (np.asarray(net.node_type) != 0)[None, :],
                        dtype=torch.float64)
    ke = np.asarray(net.node_ke, dtype=np.float64).copy()
    jn = np.asarray(s.junc_nodes) if s is not None else np.where(
        np.asarray(net.node_type) == 0)[0]
    ke[jn[g.random(jn.size) < .2]] = 1e-3
    KE = torch.as_tensor(np.repeat(ke[None, :], B, 0), dtype=torch.float64)
    Nj = jn.size
    W = torch.as_tensor(g.normal(0, 1, (B, Nj)), dtype=torch.float64)
    return D, R, KE, W, d0, rh0, ke

H = {}
for label, rel, B, K, has_prv in CASES:
    p = os.path.join(NETS, rel)
    if not os.path.exists(p):
        H["%s|MISSINGINP" % label] = rel
        continue
    net = parse_inp(p)
    # ---- epanet：run_gga do_status False/True ----
    se = GGASolver(net, device="cpu", dtype=torch.float64, mode="epanet",
                   inp_path=p)
    D, R, KE, W, d0, rh0, ke = scen(net, se, B, 20260824)
    for tag, ds in (("sm0", False), ("sm1", True)):
        oe = se.run_gga(d0, rh0, ke=ke, do_status=ds)
        for k in ("head", "flow", "emitter", "setting", "relerr",
                  "fixed_demand"):
            H["P1%s|%s|%s" % (tag, label, k)] = h(np.asarray(oe[k],
                                                             dtype=np.float64))
        H["P1%s|%s|status" % (tag, label)] = h(np.asarray(oe["status"],
                                                          dtype=np.int64))
        H["P1%s|%s|iters" % (tag, label)] = str(int(oe["iters"]))
    # ---- plain dense 构造（PRV/泵网等的 RAISE 行为要逐字比）----
    try:
        sp = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                       inp_path=p)
        plain_ok = True
        H["P2raise|%s" % label] = "ok"
    except Exception as e:
        plain_ok = False
        H["P2raise|%s" % label] = type(e).__name__ + "|" + str(e)[:160]
    if has_prv:
        continue                      # PRV 网：dense/SM 是新准入，不进影子比对
    # ---- dense+SM 构造（异常只比类型：放行集合文案属预期准入变化）----
    try:
        s = GGASolver(net, device="cpu", dtype=torch.float64, mode="dense",
                      inp_path=p, dense_status_machine=True)
        sm_ok = True
        H["P2smraise|%s" % label] = "ok"
    except Exception as e:
        sm_ok = False
        H["P2smraise|%s" % label] = type(e).__name__
    # ---- 缺省 dense（无 SM 冻结解）----
    if plain_ok:
        with torch.no_grad():
            try:
                od = sp.solve(D, R, ke_int=KE)
                for k in ("head_ft", "flow_cfs", "emitter_cfs", "iters",
                          "relerr"):
                    H["P2|%s|%s" % (label, k)] = h(od[k])
            except Exception as e:
                H["P2|%s|RAISE" % label] = type(e).__name__ + "|" + str(e)[:120]
    if not sm_ok:
        continue
    # ---- dense+SM：dense 与 csr 装配 ----
    for asm in ("dense", "csr"):
        with torch.no_grad():
            try:
                osm = s.solve(D, R, ke_int=KE, status_machine=True,
                              assemble=asm)
                for k in ("head_ft", "flow_cfs", "iters", "status"):
                    H["P2sm%s|%s|%s" % (asm, label, k)] = h(osm[k])
            except Exception as e:
                H["P2sm%s|%s|RAISE" % (asm, label)] = \
                    type(e).__name__ + "|" + str(e)[:120]
    # ---- dense 冻结 + csr ----
    with torch.no_grad():
        try:
            ofz = s.solve(D, R, ke_int=KE, assemble="csr")
            for k in ("head_ft", "flow_cfs", "iters"):
                H["P2fz|%s|%s" % (label, k)] = h(ofz[k])
        except Exception as e:
            H["P2fz|%s|RAISE" % label] = type(e).__name__ + "|" + str(e)[:120]
    # ---- unrolled 前向 + 梯度 ----
    try:
        Dg = D.clone().requires_grad_(True)
        Rg = R.clone().requires_grad_(True)
        Kg = KE.clone().requires_grad_(True)
        ou = solve_unrolled(s, Dg, Rg, ke=Kg, K=max(K, 3))
        (ou["head_ft"][:, s.junc_nodes_t] * W).sum().backward()
        H["P3u|%s|head" % label] = h(ou["head_ft"])
        H["P3u|%s|gD" % label] = h(Dg.grad)
        H["P3u|%s|gR" % label] = h(Rg.grad)
        H["P3u|%s|gK" % label] = h(Kg.grad)
    except Exception as e:
        H["P3u|%s|RAISE" % label] = type(e).__name__ + "|" + str(e)[:120]
    # ---- implicit 前向 + 梯度 ----
    try:
        D2 = D.clone().requires_grad_(True)
        R2 = R.clone().requires_grad_(True)
        K2 = KE.clone().requires_grad_(True)
        oi = implicit_solve(s, D2, R2, ke=K2)
        Hi = oi[0]
        (Hi[:, s.junc_nodes_t] * W).sum().backward()
        H["P3i|%s|head" % label] = h(Hi)
        H["P3i|%s|gD" % label] = h(D2.grad)
        H["P3i|%s|gR" % label] = h(R2.grad)
        H["P3i|%s|gK" % label] = h(K2.grad)
    except Exception as e:
        H["P3i|%s|RAISE" % label] = type(e).__name__ + "|" + str(e)[:120]
    # ---- solve_polished ----
    try:
        po = solve_polished(s, D.numpy(), R.numpy(), ke=KE.numpy())
        H["P4|%s|q" % label] = h(po["q"])
        H["P4|%s|head" % label] = h(po["head"])
        H["P4|%s|resid" % label] = h(po["resid_inf"])
    except Exception as e:
        H["P4|%s|RAISE" % label] = type(e).__name__ + "|" + str(e)[:120]

with open(os.environ["SB_OUT"], "w", encoding="utf-8") as f:
    json.dump(H, f, ensure_ascii=False, indent=0, sort_keys=True)
print("keys:", len(H))
'''


def run_side(pkg_root, out_json):
    env = dict(os.environ)
    env["SB_PKG"] = pkg_root
    env["SB_NETS"] = os.path.join(ROOT, "networks")
    env["SB_CASES"] = json.dumps(CASES)
    env["SB_OUT"] = out_json
    wf = os.path.join(tempfile.gettempdir(), "sb_worker_%d.py" % os.getpid())
    with open(wf, "w", encoding="utf-8") as f:
        f.write(WORKER)
    pr = subprocess.run([sys.executable, "-X", "utf8", wf], cwd=ROOT,
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=7200, env=env)
    if pr.returncode != 0:
        print(pr.stdout[-2000:])
        print(pr.stderr[-4000:])
        raise RuntimeError("worker rc=%d (%s)" % (pr.returncode, pkg_root))
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def main():
    tmp = tempfile.mkdtemp(prefix="prvshadow_")
    shadow_root = os.path.join(tmp, "shadow")
    os.makedirs(shadow_root)
    # git archive BASE dgga → shadow_root/dgga
    tar = os.path.join(tmp, "dgga.tar")
    subprocess.run(["git", "archive", "-o", tar, BASE, "dgga"], cwd=ROOT,
                   check=True)
    subprocess.run(["tar", "-xf", tar, "-C", shadow_root], check=True)
    base_sha = subprocess.run(["git", "rev-parse", "--short", BASE], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    print("影子基线 = %s（git archive dgga）" % base_sha)

    h_shadow = run_side(shadow_root, os.path.join(tmp, "shadow.json"))
    h_work = run_side(ROOT, os.path.join(tmp, "work.json"))

    keys = sorted(set(h_shadow) | set(h_work))
    bad = []
    for k in keys:
        a, b = h_shadow.get(k), h_work.get(k)
        if a != b:
            bad.append((k, a, b))
    print("battery 键数：影子 %d / 工作树 %d / 并集 %d" %
          (len(h_shadow), len(h_work), len(keys)))
    if bad:
        print("不一致 %d 键：" % len(bad))
        for k, a, b in bad[:40]:
            print("  %s\n    shadow: %s\n    work  : %s" % (k, a, b))
    same = len(keys) - len(bad)
    print("sha256 键 %d/%d 完全相同 - %s" %
          (same, len(keys), "PASS" if not bad else "FAIL"))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

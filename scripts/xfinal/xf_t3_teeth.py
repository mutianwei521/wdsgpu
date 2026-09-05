# -*- coding: utf-8 -*-
"""XF-2 - T3 改成"观察被测代码"之后，放宽复用谓词还抓不抓得住。

我自己放宽 `_cudss_adjoint` 的复用谓词（**改被测代码**，不是改探针），然后
在同一棵变异树上跑两份探针：
  new = 出厂那份 scripts/regression_gpu.py（读 bwd_reuse/bwd_refactorize 增量）
  old = 把同一个文件里的 probe 闭包换回 6ed0808 的旧写法（把谓词抄一份自己算）
期望：old 全绿（这正是终审说的缺口），new 变红。若 new 也绿，判不合格。

变异（我自己挑的）：
  W1_drop_gen  丢掉 `st.get("gen") == gen` - 分解已被后续 factorize 顶掉也照用
  W2_drop_all  连 B 检查一起丢，`reuse = st.get("solver") is not None`
对照：
  C0_clean     不改，new/old 都必须全绿
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
NODE = os.popen("hostname").read().strip()
SRC_DGGA = os.environ.get("XF_DGGA", os.path.join(HERE, "dgga"))
SRC_REG = os.environ.get("XF_REG", os.path.join(HERE, "regression_gpu.py"))
NETD = os.environ.get("XF_NETD", os.path.join(HERE, "p2nets"))

A_REUSE = ('        reuse = (st.get("solver") is not None and st.get("gen") == gen\n'
           '                 and int(st.get("B", -1)) == int(B))')
W1 = ('        reuse = (st.get("solver") is not None\n'
      '                 and int(st.get("B", -1)) == int(B))')
W2 = '        reuse = (st.get("solver") is not None)'

NEW_PROBE = '''        def probe(self_, data, g, B, st, gen, slot):
            pr.events += 1
            v = st.get("vals")
            snap = v.detach().clone() if v is not None else None
            c = self_._cudss_counters
            b_reu, b_ref = int(c["bwd_reuse"]), int(c["bwd_refactorize"])
            out = pr.raw(self_, data, g, B, st, gen, slot)
            d_reu = int(c["bwd_reuse"]) - b_reu
            d_ref = int(c["bwd_refactorize"]) - b_ref
            if (d_reu, d_ref) not in ((1, 0), (0, 1)):
                pr.incoh += 1
            if d_reu == 1:                       # 被测代码自己说：复用了
                pr.reuse += 1
                if snap is None:
                    pr.stale += 1                # 声称复用却没有缓冲
                else:
                    pr.worst = max(pr.worst,
                                   float((snap - data).abs().max()))
                    if not bool(torch.equal(snap, data)):
                        pr.stale += 1
            return out'''

OLD_PROBE = '''        def probe(self_, data, g, B, st, gen, slot):
            # [XF] 复原 6ed0808 的旧探针：把 solver 的复用谓词抄一份自己算
            reuse = (st.get("solver") is not None and st.get("gen") == gen
                     and int(st.get("B", -1)) == int(B))
            pr.events += 1
            if reuse:
                pr.reuse += 1
                v = st["vals"]
                d = float((v - data).abs().max())
                pr.worst = max(pr.worst, d)
                if not bool(torch.equal(v, data)):
                    pr.stale += 1
            return pr.raw(self_, data, g, B, st, gen, slot)'''

MUT = [("C0_clean", None), ("W1_drop_gen", W1), ("W2_drop_all", W2)]


def build(mut):
    tmp = tempfile.mkdtemp(prefix="xft3_")
    shutil.copytree(SRC_DGGA, os.path.join(tmp, "dgga"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(tmp, "scripts"))
    reg = io.open(SRC_REG, encoding="utf-8").read()
    if reg.count(NEW_PROBE) != 1:
        raise SystemExit("regression_gpu.py 里 new probe 闭包命中 %d 次"
                         % reg.count(NEW_PROBE))
    io.open(os.path.join(tmp, "scripts", "reg_new.py"), "w",
            encoding="utf-8").write(reg)
    io.open(os.path.join(tmp, "scripts", "reg_old.py"), "w",
            encoding="utf-8").write(reg.replace(NEW_PROBE, OLD_PROBE))
    if mut is not None:
        p = os.path.join(tmp, "dgga", "solver.py")
        src = io.open(p, encoding="utf-8").read()
        if src.count(A_REUSE) != 1:
            raise SystemExit("复用谓词锚点命中 %d 次" % src.count(A_REUSE))
        io.open(p, "w", encoding="utf-8").write(src.replace(A_REUSE, mut))
    return tmp


def parse_t3(out):
    rows, sec = [], None
    for ln in out.splitlines():
        s = ln.strip()
        if s.startswith("【T"):
            sec = s[1:3]
        elif s.startswith("总判定"):
            sec = None
        elif sec == "T3" and ("PASS" in s or "FAIL" in s):
            rows.append(("FAIL" in s, s))
    return rows


def parse_all(out):
    sec, red, tot, verdict = None, {}, {}, None
    for ln in out.splitlines():
        s = ln.strip()
        if s.startswith("【T"):
            sec = s[1:3]
        elif s.startswith("总判定"):
            verdict = s
            sec = None
        elif sec and ("PASS" in s or "FAIL" in s):
            tot[sec] = tot.get(sec, 0) + 1
            red[sec] = red.get(sec, 0) + (1 if "FAIL" in s else 0)
    return red, tot, verdict


def main():
    t00 = time.time()
    print("=" * 100)
    print("XF-2  T3 探针改完还抓得住吗（放宽被测代码的复用谓词）| 节点 %s" % NODE)
    print("=" * 100, flush=True)
    summ, bad = [], 0
    for name, mut in MUT:
        tree = build(mut)
        env = dict(os.environ, DGGA_NETS=NETD, PYTHONPATH=tree,
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        got = {}
        for tag in ("new", "old"):
            r = subprocess.run(
                [sys.executable, "-X", "utf8",
                 os.path.join(tree, "scripts", "reg_%s.py" % tag)],
                capture_output=True, text=True, env=env, cwd=tree)
            red, tot, verdict = parse_all(r.stdout)
            got[tag] = (r.returncode, red, tot, verdict, parse_t3(r.stdout))
            io.open(os.path.join(os.getcwd(), "xf_t3_%s_%s.txt" % (name, tag)),
                    "w", encoding="utf-8").write(r.stdout)
            if r.returncode not in (0, 1):
                print("  [%s/%s] STDERR %s" % (name, tag,
                                               r.stderr.strip()[-300:]))
        print("\n### %s" % name)
        for tag in ("new", "old"):
            rc, red, tot, verdict, t3 = got[tag]
            print("  探针=%-3s rc=%d | " % (tag, rc)
                  + " ".join("%s %d/%d红" % (k, red.get(k, 0), tot.get(k, 0))
                             for k in sorted(tot))
                  + " | " + str(verdict))
        # T3 逐行（只印新探针红的那些 + 全部 stale/incoh 摘要）
        for tag in ("new", "old"):
            t3 = got[tag][4]
            nred = sum(1 for b, _ in t3 if b)
            print("  [T3/%s] 红 %d/%d 行" % (tag, nred, len(t3)))
            for b, s in t3:
                if b or nred == 0:
                    print("      %s%s" % ("RED " if b else "    ", s))
                if not b and nred:
                    pass
        n_new = sum(1 for b, _ in got["new"][4] if b)
        n_old = sum(1 for b, _ in got["old"][4] if b)
        if name == "C0_clean":
            good = (n_new == 0 and n_old == 0)
        else:
            good = (n_new > 0)
        bad += 0 if good else 1
        summ.append(dict(name=name, t3_red_new=n_new, t3_red_old=n_old,
                         rc_new=got["new"][0], rc_old=got["old"][0],
                         red_new=got["new"][1], tot_new=got["new"][2],
                         verdict_new=got["new"][3], verdict_old=got["old"][3]))
        print("  => 新探针 T3 红 %d 行 / 旧探针 T3 红 %d 行  %s"
              % (n_new, n_old, "OK" if good else "**不合格**"))
        shutil.rmtree(tree, ignore_errors=True)
    print("\n" + "=" * 100)
    print("总判定: %s | 用时 %.0f s"
          % ("T3 新探针有牙" if bad == 0 else "**不合格 %d 项**" % bad,
             time.time() - t00))
    print("XF2-JSON " + json.dumps(summ, ensure_ascii=False))
    print("=" * 100)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

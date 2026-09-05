# -*- coding: utf-8 -*-
"""XF-3a（本机）：全仓扫"末行 echo 吞掉退出码"。

一个 shell 脚本的退出码 = 最后一条执行到的命令的退出码。若末条是 `echo ...`
（哪怕文本里写着 rc=$?），脚本恒退 0 - sbatch 记的作业退出码就永远是 0:0，
CI 按作业状态接就把 FAIL/SKIP 全当通过。这里把仓里每个 .sh / .ps1 的"末条有效
命令"找出来分类，并单独查 regression_all.py / regression_gpu.py 的 py 侧退出码。
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.isdir(os.path.join(ROOT, "scripts")):
    ROOT = os.getcwd()


def last_cmd(path):
    txt = io.open(path, encoding="utf-8", errors="replace").read()
    lines = [l.rstrip() for l in txt.splitlines()]
    for l in reversed(lines):
        s = l.strip()
        if not s or s.startswith("#"):
            continue
        return s
    return ""


def main():
    shs = []
    for dp, dns, fns in os.walk(os.path.join(ROOT, "scripts")):
        dns[:] = [d for d in dns if d not in ("__pycache__", ".git")]
        for f in fns:
            if f.endswith((".sh", ".ps1")):
                shs.append(os.path.join(dp, f))
    shs.sort()
    print("=" * 100)
    print("XF-3a 末行 echo 吞退出码扫描：%d 个 shell 脚本" % len(shs))
    print("=" * 100)
    swallow, propagate, other = [], [], []
    for p in shs:
        lc = last_cmd(p)
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        if re.match(r"^exit\s+\$?\w", lc):
            propagate.append((rel, lc))
        elif lc.startswith("echo"):
            swallow.append((rel, lc))
        else:
            other.append((rel, lc))
    print("\n【A 退出码会传出去（末条是 exit $rc）】%d 个" % len(propagate))
    for r, l in propagate:
        print("  %-52s | %s" % (r, l))
    print("\n【B 末条是 echo ⇒ 脚本恒退 0，退出码被吞】%d 个" % len(swallow))
    for r, l in swallow:
        print("  %-52s | %s" % (r, l))
    print("\n【C 末条既非 echo 也非 exit（退出码 = 该命令的退出码）】%d 个" % len(other))
    for r, l in other:
        print("  %-52s | %s" % (r, l[:70]))

    print("\n" + "=" * 100)
    print("【D python 侧：判据脚本自己有没有把非零退出码 return 出去】")
    for rel in ("scripts/regression_all.py", "scripts/regression_gpu.py",
                "scripts/check_symmetry.py"):
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        txt = io.open(p, encoding="utf-8").read()
        has_sysexit = "sys.exit(main())" in txt
        rets = sorted(set(re.findall(r"^\s*return (\d+)$", txt, re.M)))
        print("  %-32s sys.exit(main())=%s | main 的 return 值集合=%s"
              % (os.path.basename(rel), has_sysexit, rets or "无字面量"))
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())

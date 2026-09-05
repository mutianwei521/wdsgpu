# -*- coding: utf-8 -*-
"""AUDIT R2 - regression_gpu.py 的三条断言是不是真的在执行、真的有牙。

敌意审阅口径：**不复用上游任何测试的结论**，只把上游那个脚本当被测对象。
做法 = 源码级哨兵（sentinel mutation）：
  1. 基线：原封不动跑一次 upstream scripts/regression_gpu.py（全量 40 项），
     记 rc 与 T1/T2/T3 的**打印行数**（行数少了 = 某条根本没执行）。
  2. 对每个哨兵：把 dgga 复制一份、在**源码级**破坏一条不变量，
     再跑同一个 regression_gpu.py（子集，省时间；断言一个字没改），
     要求 rc==1 且**恰好是**对应那条 T 变红。
  3. 空哨兵（不改任何东西）必须仍然 PASS - 证明我不是把什么都弄红了。

每个哨兵单独一个子进程 + 独立的临时 dgga 树，互不污染。
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
NETS = os.path.join(HERE, "p2nets")
NODE = os.popen("hostname").read().strip()

# ----------------------------------------------------------------------
# 哨兵表：(名字, 目标文件, 锚点原文, 替换文本, 期望变红的那条, 说明)
# 全部是"真实会发生的回归"，不是随手改坏。
# ----------------------------------------------------------------------
SENTINELS = [
    # --- T1：线性解精度改变牛顿轨迹 ---
    ("S1a_refine0", "solver.py",
     "        refine = self.cudss_refine if refine is None else refine\n"
     "        st = self._cudss_state(B, data.dtype, data.device, slot)",
     "        refine = 0\n"
     "        st = self._cudss_state(B, data.dtype, data.device, slot)",
     "T1", "cuDSS 前向的迭代精化被去掉（refine 2->0）：线性解变粗"),
    ("S1b_perturb", "solver.py",
     "            self._cudss_counters[\"solve\"] += 1\n"
     "        return Hj, st, st[\"gen\"]",
     "            self._cudss_counters[\"solve\"] += 1\n"
     "        Hj = Hj * (1.0 + 1e-9)\n"
     "        return Hj, st, st[\"gen\"]",
     "T1", "cuDSS 解被乘上 (1+1e-9)：牛顿轨迹被扰动"),
    # --- T2：计数器根本不数 / 复用判据被放宽 ---
    ("S2a_counter_mute", "solver.py",
     "            self._cudss_counters[\"bwd_refactorize\"] += 1",
     "            self._cudss_counters[\"bwd_refactorize\"] += 0",
     "T2", "bwd_refactorize 计数器不再自增（就是审计怕的\"不数的计数器\"）"),
    ("S2b_reuse_widened", "solver.py",
     "        reuse = (st.get(\"solver\") is not None and st.get(\"gen\") == gen\n"
     "                 and int(st.get(\"B\", -1)) == int(B))\n"
     "        if reuse:\n"
     "            ds = st[\"solver\"]",
     "        reuse = (st.get(\"solver\") is not None\n"
     "                 and int(st.get(\"B\", -1)) == int(B))\n"
     "        if reuse:\n"
     "            ds = st[\"solver\"]",
     "T2", "反向复用判据丢掉 gen 检查：会拿别的步的分解当自己的（真实的陈旧 bug）"),
    # --- T3：cuDSS 手上那块缓冲与 saved data 脱节 ---
    ("S3a_buffer_scribble", "solver.py",
     "            self._cudss_counters[\"solve\"] += 1\n"
     "        return Hj, st, st[\"gen\"]",
     "            self._cudss_counters[\"solve\"] += 1\n"
     "        st[\"vals\"][:, 0] += 1e-9\n"
     "        return Hj, st, st[\"gen\"]",
     "T3", "前向返回前把 cuDSS 的值缓冲改掉（前向结果逐位不变，只有反向会错）"),
    ("S3b_skip_reload", "solver.py",
     "        if st[\"alias\"]:\n"
     "            st[\"vals\"].copy_(data)",
     "        if st.get(\"_seen\"):\n"
     "            return\n"
     "        st[\"_seen\"] = True\n"
     "        if st[\"alias\"]:\n"
     "            st[\"vals\"].copy_(data)",
     "T3", "\"优化\"掉重复的值搬运：第二次起 cuDSS 拿的还是上一轮的值"),
]

SUBSET = os.environ.get("AUD_SUBSET", "Net3,ky4")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def build_tree(patch=None):
    """造一棵 tmp/{dgga, scripts/regression_gpu.py}；patch=(file, old, new)。"""
    tmp = tempfile.mkdtemp(prefix="audr2_")
    shutil.copytree(os.path.join(HERE, "dgga"), os.path.join(tmp, "dgga"))
    os.makedirs(os.path.join(tmp, "scripts"))
    shutil.copy2(os.path.join(HERE, "regression_gpu.py"),
                 os.path.join(tmp, "scripts", "regression_gpu.py"))
    shutil.copy2(os.path.join(HERE, "aud_r2_run.py"),
                 os.path.join(tmp, "scripts", "aud_r2_run.py"))
    shutil.rmtree(os.path.join(tmp, "dgga", "__pycache__"), ignore_errors=True)
    if patch is not None:
        f, old, new = patch
        tgt = os.path.join(tmp, "dgga", f)
        src = open(tgt, encoding="utf-8").read()
        n = src.count(old)
        if n != 1:
            shutil.rmtree(tmp, ignore_errors=True)
            return None, "锚点命中 %d 次（应为 1）" % n
        open(tgt, "w", encoding="utf-8").write(src.replace(old, new))
    return tmp, ""


def parse(out):
    """从 regression_gpu 的输出里抽出：每条 T 的打印行数、失败项、总判定。"""
    rows = dict(T1=0, T2=0, T3=0, T4=0)
    for ln in out.splitlines():
        m = re.match(r"\s{2}(\S+)\s+B=", ln)
        if m:
            pass
    # 直接数各段内的配置行（以两个空格 + 网名开头、含 PASS/FAIL）
    sec = None
    for ln in out.splitlines():
        if "【T1" in ln:
            sec = "T1"
        elif "【T2" in ln:
            sec = "T2"
        elif "【T3" in ln:
            sec = "T3"
        elif "【T4" in ln:
            sec = "T4"
        elif ln.startswith("=") or ln.startswith("总判定"):
            sec = None
        if sec and ("PASS" in ln or "FAIL" in ln) and ln.startswith("  "):
            rows[sec] += 1
    verdict = ""
    npass = ntot = -1
    failed = ""
    for ln in out.splitlines():
        if ln.startswith("总判定"):
            verdict = ln.strip()
            m = re.search(r"（(\d+)/(\d+) 项通过）", ln)
            if m:
                npass, ntot = int(m.group(1)), int(m.group(2))
        if ln.startswith("未通过项"):
            failed = ln.strip()[len("未通过项: "):]
    return rows, verdict, npass, ntot, failed


def run(tree, subset=None, tag=""):
    env = dict(os.environ)
    env["DGGA_NETS"] = NETS
    env["PYTHONPATH"] = tree
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if subset:
        env["AUD_SUBSET"] = subset
        script = os.path.join(tree, "scripts", "aud_r2_run.py")
    else:
        script = os.path.join(tree, "scripts", "regression_gpu.py")
    cp = subprocess.run([sys.executable, "-X", "utf8", script],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", env=env, cwd=tree, timeout=7200)
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode, out


print("=" * 100)
print("AUDIT R2 - regression_gpu.py 哨兵证伪 | node:", NODE)
import torch                                                    # noqa: E402
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "no cuda")
try:
    import nvmath
    print("nvmath", nvmath.__version__)
except Exception as e:                                          # noqa: BLE001
    print("nvmath 导入失败:", repr(e))
for f in ("dgga/solver.py", "dgga/autodiff.py"):
    print("md5", f, md5(os.path.join(HERE, f)))
print("md5 regression_gpu.py", md5(os.path.join(HERE, "regression_gpu.py")))
print("=" * 100)

# ---------------- 基线：全量、原封不动 ----------------
print("\n【B0 基线：原封不动的 regression_gpu.py，全量】")
tree0, why = build_tree(None)
rc0, out0 = run(tree0, subset=None)
rows0, v0, p0, t0, f0 = parse(out0)
print("  rc=%d  %s" % (rc0, v0))
print("  各条打印的配置行数：T1=%d T2=%d T3=%d T4=%d（期望 10/10/20/10）"
      % (rows0["T1"], rows0["T2"], rows0["T3"], rows0["T4"]))
print("  未通过项:", f0 or "(无)")
# P4 收尾把 T4（cuDSS 收到的 csr_data 逐位对称）加进了 regression_gpu.py：
# 40 项 -> 50 项，T4 自己 10 行。
base_ok = (rc0 == 0 and rows0["T1"] == 10 and rows0["T2"] == 10
           and rows0["T3"] == 20 and rows0["T4"] == 10 and p0 == t0 == 50)
print("  基线判定:", "OK" if base_ok else "<-- 与上游声称不符")
sys.stdout.flush()

# 保存基线原文（便于逐行核对）
open(os.path.join(HERE, "aud_r2_baseline.txt"), "w", encoding="utf-8").write(out0)

# ---------------- 空哨兵：子集、无改动，必须 PASS ----------------
print("\n【B1 空哨兵（子集 %s，零改动）：必须仍然 PASS】" % SUBSET)
rc1, out1 = run(tree0, subset=SUBSET)
rows1, v1, p1, t1, f1 = parse(out1)
print("  rc=%d  %s  行数 T1=%d T2=%d T3=%d T4=%d"
      % (rc1, v1, rows1["T1"], rows1["T2"], rows1["T3"], rows1["T4"]))
null_ok = (rc1 == 0)
print("  空哨兵判定:", "OK（不是把什么都弄红）" if null_ok else "<-- 空哨兵就红了，测法有问题")
shutil.rmtree(tree0, ignore_errors=True)
sys.stdout.flush()

# ---------------- 逐个哨兵 ----------------
print("\n【B2 哨兵证伪（每个哨兵一棵独立的 dgga 树，子集 %s）】" % SUBSET)
summary = []
for name, f, old, new, want, desc in SENTINELS:
    tree, why = build_tree((f, old, new))
    if tree is None:
        print("  %-22s <-- 无法注入：%s" % (name, why))
        summary.append((name, want, "INJECT-FAIL", "", ""))
        continue
    rc, out = run(tree, subset=SUBSET)
    rows, v, p, t, fl = parse(out)
    hit = sorted(set(x.split()[0] for x in fl.split(", ") if x.strip()))
    red = rc == 1
    exact = red and hit == [want]
    print("  %-22s rc=%d 变红=%s 红在=%s（期望 %s）%s"
          % (name, rc, "是" if red else "**否**", ",".join(hit) or "-", want,
             "  [恰好只红这一条]" if exact else
             ("  [红了但不止这一条/不是这一条]" if red else "  <-- 没抓到！")))
    print("      %s | %s | 行数 T1=%d T2=%d T3=%d T4=%d"
          % (desc, v, rows["T1"], rows["T2"], rows["T3"], rows["T4"]))
    if fl:
        print("      未通过项: " + fl[:300])
    summary.append((name, want, "RED" if red else "GREEN",
                    ",".join(hit), v))
    open(os.path.join(HERE, "aud_r2_%s.txt" % name), "w",
         encoding="utf-8").write(out)
    shutil.rmtree(tree, ignore_errors=True)
    sys.stdout.flush()

print("\n" + "=" * 100)
print("AUDIT-R2-SUMMARY 基线rc=%d 基线行数=%d/%d/%d/%d 空哨兵rc=%d"
      % (rc0, rows0["T1"], rows0["T2"], rows0["T3"], rows0["T4"], rc1))
for name, want, st, hit, v in summary:
    print("AUDIT-R2-SENT %-22s want=%s got=%-6s hit=%s" % (name, want, st, hit))
nred = sum(1 for _n, _w, st, _h, _v in summary if st == "RED")
print("AUDIT-R2-VERDICT 哨兵 %d/%d 变红；基线 %s；空哨兵 %s"
      % (nred, len(SENTINELS), "OK" if base_ok else "BAD",
         "OK" if null_ok else "BAD"))
print("=" * 100)

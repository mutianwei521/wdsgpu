# -*- coding: utf-8 -*-
"""XF-1 - T4 到底有没有牙：我自己造的单侧变体阶梯（1 ULP → 1e-2）+ 三个对称对照。

敌意立场，不复用上游 aud_symgap_gpu.py 的注入内容：
  · 注入点我自己挑三处（上游只挑了一处 emitter index_add 之后）；
  · 幅度自己排一条阶梯：**恰好 1 ULP（torch.nextafter）**、1e-14、1e-12、1e-9、
    1e-6、1e-4、1e-2 相对；
  · 三个"对称但不同"的对照（整体缩放 / 对角平移 / **成对**扰动同一对非对角槽）
    必须**不红** - 否则 T4 红的是"值变了"而不是"不对称了"，等于没有牙。

判据：跑**出厂那份** scripts/regression_gpu.py（原样拷进影子树），解析它自己印的
50 行，看 T4 十格是否全红、T1/T2/T3 有没有跟着乱红。

注入点：
  S1  emitter 对角 index_add **之后**（linear_solver=="cudss" 分支内）
  S2  `_assemble_csr` **之后**、`A=None` 之前（同样只在 cudss 分支内加判断）
  S3  `_cudss_load` **内部**（值写进 cuDSS 缓冲之前） - 这是"够不够得着"的探边：
      T4 的探针量的是**传给** _cudss_load 的实参，S3 改的是它写进去的东西。
用法：python3 -X utf8 xf_t4_teeth.py
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
ROOT = os.path.dirname(os.path.dirname(HERE)) if os.path.basename(
    os.path.dirname(HERE)) == "scripts" else os.path.dirname(HERE)
NODE = os.popen("hostname").read().strip()

SRC_DGGA = os.environ.get("XF_DGGA", os.path.join(HERE, "dgga"))
SRC_REG = os.environ.get("XF_REG", os.path.join(HERE, "regression_gpu.py"))
NETD = os.environ.get("XF_NETD", os.path.join(HERE, "p2nets"))

A_EMIT = "                csr_data = csr_data.index_add(1, self.A_csr_diag, em / hgrad_e)"
A_ASM = "                csr_data = self._assemble_csr(vals, B)"
A_LOAD = '        if st["alias"]:'

# --- 我自己写的扰动块：挑当轮 |值| 最大的**非对角**槽，只写一侧 -----------------
_ONE_SIDED = '''_xoff = (self.A_csr_row != self.A_csr_col).nonzero().reshape(-1)
_xk = _xoff[int(csr_data[0].detach().abs().index_select(0, _xoff).argmax())].reshape(1)
_xv = csr_data.index_select(1, _xk)
csr_data = csr_data.index_copy(1, _xk, {PERT})'''

# --- 对称对照 C3：**同一对**非对角槽一起改（结构与 S1 一样，只是补了转置那半）---
_PAIRED = '''_xoff = (self.A_csr_row != self.A_csr_col).nonzero().reshape(-1)
_xk = _xoff[int(csr_data[0].detach().abs().index_select(0, _xoff).argmax())].reshape(1)
_xt = ((self.A_csr_row == self.A_csr_col[_xk[0]]) & (self.A_csr_col == self.A_csr_row[_xk[0]])).nonzero().reshape(-1)
_xb = torch.cat([_xk, _xt])
_xv = csr_data.index_select(1, _xb)
csr_data = csr_data.index_copy(1, _xb, {PERT})'''

_SYM_SCALE = '''csr_data = csr_data * (1.0 + 1e-6)'''
_SYM_DIAG = '''csr_data = csr_data.index_add(1, self.A_csr_diag, torch.full_like(csr_data[:, :self.A_csr_diag.numel()], 1e-6))'''


def _ind(block, n):
    pad = " " * n
    return "\n" + "\n".join(pad + ln for ln in block.split("\n"))

# _cudss_load 是 staticmethod，拿不到 self - 只能按下标动。CSR 是
# (row,col) 字典序且位型含全部对角 ⇒ 槽 0 恒是 (0,0) 对角，槽 1 是 (0,c>0)
# 非对角。改槽 1 即单侧写。
_LOAD_PATCH = '''        data = data.clone()
        data[:, 1] = {PERT_LOAD}
        if st["alias"]:'''


def _pert(mag):
    if mag == "ulp":
        return "torch.nextafter(_xv, torch.full_like(_xv, float('inf')))"
    return "_xv * (1.0 + %r)" % float(mag)


def _pert_load(mag):
    if mag == "ulp":
        return ("torch.nextafter(data[:, 1], "
                "torch.full_like(data[:, 1], float('inf')))")
    return "data[:, 1] * (1.0 + %r)" % float(mag)


# (名字, 期望 T4 红?, 锚点, 替换文本)
def variants():
    V = []
    V.append(("C0_clean", False, None, None))
    # S1：A_EMIT 已经在 `if linear_solver == "cudss":` 体内（缩进 16）
    for tag, mag in (("ulp", "ulp"), ("1e-14", 1e-14), ("1e-12", 1e-12),
                     ("1e-09", 1e-9), ("1e-06", 1e-6), ("1e-04", 1e-4),
                     ("1e-02", 1e-2)):
        V.append(("S1_emit_%s" % tag, True, A_EMIT,
                  A_EMIT + _ind(_ONE_SIDED.replace("{PERT}", _pert(mag)), 16)))
    # S2：A_ASM 在 `if assemble == "csr":` 体内，**自己补一层 cudss 判断**，
    #     保证变体仍是 cuDSS 专属（不碰 assemble="csr"+linear_solver="dense"）。
    for tag, mag in (("ulp", "ulp"), ("1e-09", 1e-9)):
        V.append(("S2_asm_%s" % tag, True, A_ASM,
                  A_ASM + '\n                if linear_solver == "cudss":'
                  + _ind(_ONE_SIDED.replace("{PERT}", _pert(mag)), 20)))
    V.append(("S3_load_1e-06", None, A_LOAD,
              _LOAD_PATCH.replace("{PERT_LOAD}", _pert_load(1e-6))))
    V.append(("C1_symscale", False, A_EMIT, A_EMIT + _ind(_SYM_SCALE, 16)))
    V.append(("C2_symdiag", False, A_EMIT, A_EMIT + _ind(_SYM_DIAG, 16)))
    V.append(("C3_sympair_1e-04", False, A_EMIT,
              A_EMIT + _ind(_PAIRED.replace("{PERT}", _pert(1e-4)), 16)))
    return V


def build_tree(anchor, repl):
    tmp = tempfile.mkdtemp(prefix="xft4_")
    shutil.copytree(SRC_DGGA, os.path.join(tmp, "dgga"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(tmp, "scripts"))
    shutil.copy2(SRC_REG, os.path.join(tmp, "scripts", "regression_gpu.py"))
    if anchor is not None:
        p = os.path.join(tmp, "dgga", "solver.py")
        src = io.open(p, encoding="utf-8").read()
        n = src.count(anchor)
        if n != 1:
            raise SystemExit("锚点命中 %d 次，拒绝注入：%r" % (n, anchor[:50]))
        io.open(p, "w", encoding="utf-8").write(src.replace(anchor, repl))
    return tmp


def parse(out):
    """把 regression_gpu.py 自己印的结果解析成 {段: [(标签, ok, 数值行)]}。"""
    sec, res, verdict = None, {}, None
    for ln in out.splitlines():
        s = ln.strip()
        if s.startswith("【T1"):
            sec = "T1"
        elif s.startswith("【T2"):
            sec = "T2"
        elif s.startswith("【T3"):
            sec = "T3"
        elif s.startswith("【T4"):
            sec = "T4"
        elif s.startswith("总判定"):
            verdict = s
            sec = None
        elif sec and ("PASS" in s or "FAIL" in s) and not s.startswith("="):
            res.setdefault(sec, []).append(("FAIL" in s, s))
    return res, verdict


def main():
    t00 = time.time()
    print("=" * 100)
    print("XF-1  T4 有牙吗：自造单侧变体阶梯 + 对称对照 | 节点 %s" % NODE)
    print("dgga=%s | reg=%s | nets=%s" % (SRC_DGGA, SRC_REG, NETD))
    print("=" * 100, flush=True)
    summ = []
    for name, expect_red, anchor, repl in variants():
        t0 = time.time()
        tree = build_tree(anchor, repl)
        env = dict(os.environ, DGGA_NETS=NETD, PYTHONPATH=tree,
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        r = subprocess.run([sys.executable, "-X", "utf8",
                            os.path.join(tree, "scripts", "regression_gpu.py")],
                           capture_output=True, text=True, env=env, cwd=tree)
        res, verdict = parse(r.stdout)
        nred = {k: sum(1 for b, _ in v if b) for k, v in res.items()}
        ntot = {k: len(v) for k, v in res.items()}
        t4worst = []
        for b, s in res.get("T4", []):
            j = s.find("max|A−A^T|=")
            t4worst.append(s[j + 11:j + 20] if j > 0 else "?")
        line = ("%-18s rc=%d | T1 %d/%d红 T2 %d/%d红 T3 %d/%d红 T4 %d/%d红 | %s"
                % (name, r.returncode,
                   nred.get("T1", 0), ntot.get("T1", 0),
                   nred.get("T2", 0), ntot.get("T2", 0),
                   nred.get("T3", 0), ntot.get("T3", 0),
                   nred.get("T4", 0), ntot.get("T4", 0), verdict))
        print(line, flush=True)
        print("    T4 十格 max|A−A^T| = %s" % " ".join(t4worst), flush=True)
        if r.returncode not in (0, 1):
            print("    STDERR: " + r.stderr.strip()[-400:], flush=True)
        summ.append(dict(name=name, expect_red=expect_red, rc=r.returncode,
                         red=nred, tot=ntot, t4worst=t4worst, verdict=verdict,
                         secs=round(time.time() - t0, 1)))
        io.open(os.path.join(os.getcwd(), "xf_t4_%s.txt" % name), "w",
                encoding="utf-8").write(r.stdout)
        shutil.rmtree(tree, ignore_errors=True)
    print("\n" + "=" * 100)
    ok = True
    for d in summ:
        e, r4 = d["expect_red"], d["red"].get("T4", 0)
        n4 = d["tot"].get("T4", 0)
        if e is True:
            good = (r4 == n4 and n4 == 10)
        elif e is False:
            good = (r4 == 0 and n4 == 10)
        else:
            good = None
        ok = ok and (good is not False)
        print("%-18s 期望T4红=%-5s 实测T4红=%d/%d  %s"
              % (d["name"], d["expect_red"], r4, n4,
                 "OK" if good else ("**不合格**" if good is False else "（探边，无期望）")))
    print("总判定: %s | 用时 %.0f s" % ("T4 有牙" if ok else "T4 **不合格**",
                                        time.time() - t00))
    print("XF1-JSON " + json.dumps(summ, ensure_ascii=False))
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

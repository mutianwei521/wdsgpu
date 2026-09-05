# -*- coding: utf-8 -*-
"""AUDIT R1 - check_symmetry.py 的断言到底能不能抓错（敌意口径）。

被测对象 = scripts/check_symmetry.py 里的 §A/§B 判据。做法：
  · 我自己写 13 个**单侧装配**变体（位置/量级/条件全部与上游那 4 个不同，
    含"恰好 1 ULP"与"只写上三角"这两种最难抓的形态），
  · 每个变体复制一份 dgga、源码级注入、在**全部 18 个用例 × 4 条通路**上
    跑上游那套判据（不是他们挑的 4 个用例子集），
  · 期望：故意破坏对称的必须 RED；对称保持的改动（X12 缩放）必须 GREEN；
    换掉捕获点（X10）必须因 liveness 断言 FAIL。
  · 另外跑 3 组不同随机种子的正品，确认**不误报**。

用法：python -X utf8 scripts/aud_p4/aud_r1_mutants.py [变体名]
（带变体名 = 子进程模式，父进程自己会调。）
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# ----------------------------------------------------------------------
# 我自己的变体表：(名字, 文件, 锚点, 替换, 期望, 说明)
# 期望 ∈ {"RED", "GREEN", "LIVENESS"}
# ----------------------------------------------------------------------
ANCH_DENSE = ("                A = A.scatter_add(1, s.A_idx.expand(B, -1), vals)"
              ".view(B, Nj, Nj)")
M = [
    # ---- 1 ULP：最小可能的单侧扰动 ----
    ("X1_ulp_dense_mid", "solver.py",
     "                A = A.view(B, Nj, Nj)",
     "                A = A.view(B, Nj, Nj)\n"
     "                _off = (self.A_csr_row != self.A_csr_col).nonzero()"
     ".reshape(-1)\n"
     "                _k = int(_off[_off.numel() // 2])\n"
     "                _r = int(self.A_csr_row[_k]); _c = int(self.A_csr_col[_k])\n"
     "                _v = A[:, _r, _c]\n"
     "                _d = torch.nextafter(_v, torch.full_like(_v, float('inf'))) - _v\n"
     "                _E = torch.zeros(Nj, Nj, dtype=dt, device=dev)\n"
     "                _E[_r, _c] = 1.0\n"
     "                A = A + _E * _d.view(B, 1, 1)",
     "RED", "稠密 A 的中间那个非对角槽 +1 ULP（能想到的最小单侧扰动）"),
    ("X2_ulp_csr", "solver.py",
     "        data.scatter_add_(1, self.A_csr_scatter.expand(B, -1), vals)",
     "        data.scatter_add_(1, self.A_csr_scatter.expand(B, -1), vals)\n"
     "        _off = (self.A_csr_row != self.A_csr_col).nonzero().reshape(-1)\n"
     "        _k = int(_off[_off.numel() // 3])\n"
     "        _v = data[:, _k]\n"
     "        data[:, _k] = torch.nextafter("
     "_v, torch.full_like(_v, float('inf')))",
     "RED", "CSR data 的某个非对角槽 +1 ULP（cuDSS 通路唯一经过的那份）"),
    # ---- 只写上三角：Cholesky 只读下三角 ⇒ 前向与稠密梯度都看不出来 ----
    ("X3_upper_only", "solver.py",
     "                A = A.view(B, Nj, Nj)",
     "                A = A.view(B, Nj, Nj)\n"
     "                _off = (self.A_csr_row < self.A_csr_col).nonzero()"
     ".reshape(-1)\n"
     "                _k = int(_off[0])\n"
     "                _E = torch.zeros(Nj, Nj, dtype=dt, device=dev)\n"
     "                _E[int(self.A_csr_row[_k]), int(self.A_csr_col[_k])] "
     "= -1e-6\n"
     "                A = A + _E",
     "RED", "只往**严格上三角**写 -1e-6：cholesky 只读下三角，前向逐位不变"),
    # ---- 只在某一轮 / 某个 batch / 某类网上发生 ----
    ("X4_last_iter_only", "solver.py",
     "                A = A.view(B, Nj, Nj)",
     "                A = A.view(B, Nj, Nj)\n"
     "                self._xcnt = getattr(self, '_xcnt', 0) + 1\n"
     "                if self._xcnt % 4 == 0:\n"
     "                    _off = (self.A_csr_row != self.A_csr_col).nonzero()"
     ".reshape(-1)\n"
     "                    _k = int(_off[0])\n"
     "                    _E = torch.zeros(Nj, Nj, dtype=dt, device=dev)\n"
     "                    _E[int(self.A_csr_row[_k]), "
     "int(self.A_csr_col[_k])] = 1e-7\n"
     "                    A = A + _E",
     "RED", "只有每第 4 轮牛顿迭代才单侧写（考验\"逐轮都查\"）"),
    ("X5_batch1_only", "solver.py",
     "                A = A.view(B, Nj, Nj)",
     "                A = A.view(B, Nj, Nj)\n"
     "                if B > 1:\n"
     "                    _off = (self.A_csr_row != self.A_csr_col).nonzero()"
     ".reshape(-1)\n"
     "                    _k = int(_off[0])\n"
     "                    _E = torch.zeros(B, Nj, Nj, dtype=dt, device=dev)\n"
     "                    _E[1, int(self.A_csr_row[_k]), "
     "int(self.A_csr_col[_k])] = 1e-8\n"
     "                    A = A + _E",
     "RED", "只有 batch 里第 2 个场景单侧写（考验\"批内每个场景都查\"）"),
    ("X6_bignet_only", "solver.py",
     "                A = A.view(B, Nj, Nj)",
     "                A = A.view(B, Nj, Nj)\n"
     "                if Nj > 500:\n"
     "                    _off = (self.A_csr_row != self.A_csr_col).nonzero()"
     ".reshape(-1)\n"
     "                    _k = int(_off[0])\n"
     "                    _E = torch.zeros(Nj, Nj, dtype=dt, device=dev)\n"
     "                    _E[int(self.A_csr_row[_k]), "
     "int(self.A_csr_col[_k])] = 1e-10\n"
     "                    A = A + _E",
     "RED", "只有 Nj>500 的大网才单侧写（考验用例表覆盖到大网）"),
    ("X7_pump_only", "solver.py",
     "                A = A.view(B, Nj, Nj)",
     "                A = A.view(B, Nj, Nj)\n"
     "                if int(self.n_pumps) > 0:\n"
     "                    _off = (self.A_csr_row != self.A_csr_col).nonzero()"
     ".reshape(-1)\n"
     "                    _k = int(_off[0])\n"
     "                    _E = torch.zeros(Nj, Nj, dtype=dt, device=dev)\n"
     "                    _E[int(self.A_csr_row[_k]), "
     "int(self.A_csr_col[_k])] = 1e-9\n"
     "                    A = A + _E",
     "RED", "只有含泵的网才单侧写（考验用例表里真有泵网）"),
    # ---- 装配的其它位置 ----
    ("X8_csr_to_dense", "solver.py",
     "    def _csr_to_dense(self, data, B):",
     "    def _csr_to_dense(self, data, B):\n"
     "        data = data.index_copy(1, self.A_csr_diag[:1],\n"
     "                               data.index_select(1, self.A_csr_diag[:1]))",
     "GREEN", "对照：_csr_to_dense 里放一个**保持对称**的恒等改写（必须不误报）"),
    ("X9_scatter_slot_shift", "solver.py",
     "        vals = torch.cat([-Pm[:, self.lk_both], -Pm[:, self.lk_both],",
     "        vals = torch.cat([-Pm[:, self.lk_both], "
     "-Pm[:, self.lk_both] - 1e-12 * Pm[:, self.lk_both].abs(),",
     "RED", "阀/管的转置那半边差一个 1e-12 相对量（另一个量级的非对称阀）"),
    ("X10_cholesky_ex", "solver.py",
     "                chol = torch.linalg.cholesky(A)",
     "                chol = torch.linalg.cholesky_ex(A)[0]",
     "LIVENESS", "把捕获点 cholesky 换成 cholesky_ex（守卫应因捕获轮数不符而 FAIL）"),
    ("X11_cudss_branch", "solver.py",
     "                csr_data = csr_data.index_add(1, self.A_csr_diag, "
     "em / hgrad_e)",
     "                csr_data = csr_data.index_add(1, self.A_csr_diag, "
     "em / hgrad_e)\n"
     "                csr_data = csr_data.index_add(\n"
     "                    1, (self.A_csr_row != self.A_csr_col).nonzero()"
     ".reshape(-1)[:1],\n"
     "                    torch.full((B, 1), -1e3, dtype=dt, device=dev))",
     "GREEN", "只在 linear_solver=='cudss' 分支里单侧写（CPU 上这条分支不可达 ⇒ 预期漏网）"),
    ("X12_sym_scale", "solver.py",
     "                A = A.view(B, Nj, Nj)",
     "                A = A.view(B, Nj, Nj)\n"
     "                A = A * (1.0 + 1e-9)",
     "GREEN", "对照：**保持对称**的整体缩放（必须不误报 - 判据要有分辨率）"),
    ("X13_unrolled_csr", "autodiff.py",
     "            A = A.scatter_add(1, s.A_idx.expand(B, -1), vals).view(B, Nj, Nj)",
     "            A = A.scatter_add(1, s.A_idx.expand(B, -1), vals)"
     ".view(B, Nj, Nj)\n"
     "            _off = (s.A_csr_row != s.A_csr_col).nonzero().reshape(-1)\n"
     "            _k = int(_off[_off.numel() // 2])\n"
     "            _v = A[:, int(s.A_csr_row[_k]), int(s.A_csr_col[_k])]\n"
     "            _E = torch.zeros(Nj, Nj, dtype=A.dtype, device=A.device)\n"
     "            _E[int(s.A_csr_row[_k]), int(s.A_csr_col[_k])] = 1.0\n"
     "            A = A + _E * (torch.nextafter("
     "_v, torch.full_like(_v, float('inf'))) - _v).view(B, 1, 1)",
     "RED", "展开路径的稠密装配 +1 ULP（只影响 solve_unrolled）"),
]


def child(name):
    import check_symmetry as CS
    ent = [m for m in M if m[0] == name]
    if not ent:
        print("CHILD-ERR 未知变体")
        return 2
    _, fname, old, new, want, desc = ent[0]
    tmp = tempfile.mkdtemp(prefix="audr1_")
    shutil.copytree(os.path.join(ROOT, "dgga"), os.path.join(tmp, "dgga"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    tgt = os.path.join(tmp, "dgga", fname)
    src = open(tgt, encoding="utf-8").read()
    if src.count(old) != 1:
        print("CHILD-ERR 锚点命中 %d 次" % src.count(old))
        return 2
    open(tgt, "w", encoding="utf-8").write(src.replace(old, new))
    CS.ensure_sym_torture()
    mods = CS.load_mods(tmp)
    rows, bad = CS.run_battery(mods, CS.CASES, verbose=True)
    red_cfg = [r[0] for r in rows if not r[11]]
    worstD = max([r[5] for r in rows if r[5] == r[5]] + [0.0])
    worstC = max([r[6] for r in rows if r[6] == r[6]] + [0.0])
    # 分辨"红的原因"：对称度量真的 != 0，还是只是 liveness/异常
    sym_red = sum(1 for r in rows if (r[5] == r[5] and r[5] != 0.0)
                  or (r[6] == r[6] and r[6] != 0.0))
    live_red = sum(1 for r in rows if not r[9])
    exc_red = sum(1 for r in rows if r[10])
    print("CHILD-RESULT " + json.dumps(dict(
        name=name, want=want, bad=bad, n=len(rows), sym_red=sym_red,
        live_red=live_red, exc_red=exc_red, worstD=worstD, worstC=worstC,
        red_cfg=red_cfg[:80])))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


def child_clean(seed):
    """正品 + 换随机种子，检查不误报。"""
    import check_symmetry as CS
    CS.ensure_sym_torture()
    orig = CS.build_case

    def bc(mods, inp_rel, B, ke_frac, dscale, seed_=None):
        return orig(mods, inp_rel, B, ke_frac, dscale, seed=int(seed))
    CS.build_case = bc
    mods = CS.load_mods(None)
    rows, bad = CS.run_battery(mods, CS.CASES, verbose=False)
    worstD = max([r[5] for r in rows if r[5] == r[5]] + [0.0])
    worstC = max([r[6] for r in rows if r[6] == r[6]] + [0.0])
    print("CHILD-RESULT " + json.dumps(dict(
        name="clean_seed_%s" % seed, want="GREEN", bad=bad, n=len(rows),
        sym_red=0, live_red=sum(1 for r in rows if not r[9]),
        exc_red=sum(1 for r in rows if r[10]),
        worstD=worstD, worstC=worstC,
        red_cfg=[r[0] for r in rows if not r[11]][:80])))
    return 0


def main():
    if len(sys.argv) >= 2:
        a = sys.argv[1]
        return child_clean(a.split(":")[1]) if a.startswith("clean:") \
            else child(a)
    print("=" * 104)
    print("AUDIT R1 - 我自己的 13 个单侧装配变体 vs check_symmetry.py 的判据")
    print("=" * 104)
    res = []
    jobs = [m[0] for m in M] + ["clean:1", "clean:777", "clean:20260101"]
    for name in jobs:
        p = subprocess.run([sys.executable, "-X", "utf8",
                            os.path.abspath(__file__), name], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=3600)
        out = (p.stdout or "") + (p.stderr or "")
        line = [l for l in out.splitlines() if l.startswith("CHILD-RESULT")]
        if not line:
            print("  %-22s <-- 子进程无结果 rc=%d %s"
                  % (name, p.returncode, out.strip().splitlines()[-1:]))
            res.append(dict(name=name, want="?", bad=-1))
            continue
        d = json.loads(line[-1][len("CHILD-RESULT "):])
        res.append(d)
        got = "RED" if d["bad"] else "GREEN"
        okmark = {"RED": got == "RED", "GREEN": got == "GREEN",
                  "LIVENESS": got == "RED" and d["live_red"] > 0}.get(
                      d["want"], False)
        print("  %-22s 期望=%-8s 实得=%-5s 红%d/%d "
              "(对称度量!=0 的配置 %d, liveness 失效 %d, 抛异常 %d) "
              "worstDense=%.3e worstCSR=%.3e  %s"
              % (d["name"], d["want"], got, d["bad"], d["n"], d["sym_red"],
                 d["live_red"], d["exc_red"], d["worstD"], d["worstC"],
                 "OK" if okmark else "<<< 不符预期"))
        if d["bad"] and d["bad"] < 20:
            print("       红在: " + ", ".join(d["red_cfg"][:8])
                  + (" ..." if len(d["red_cfg"]) > 8 else ""))
        sys.stdout.flush()
    print("\n" + "=" * 104)
    print("AUD-R1-JSON " + json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())

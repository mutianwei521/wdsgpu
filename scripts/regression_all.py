# -*- coding: utf-8 -*-
"""regression_all.py - 一键全量回归扫掠（任务 B2）。

跑全部验收项并汇总成中文总表（每行：项目/关键指标/门槛/实测/判定）：
  ① align.py            静态网 25 项：city_d、city_d_emit、rand_main_0000..0019、
                        rand_small_0000..0002
  ② align.py 快照回放   EXA6 / city_h / ky3 / ky5（align.py 对泵/水池网自动走
                        replay_b2 冻结回放）
  ③ align_eps.py        EXA4/EXA5/EXA6/city_h/ky3/ky5 + pub_anytown
                        （完整自主 EPS；pub_anytown 为 CUSTOM 多段泵曲线网）
  ④ validate_reference.py  7 网三项交叉验证
  ⑤ gradcheck_3way.py   三方梯度对拍 + torch.gradcheck + B=8 批一致性
  ⑥ audit_grad_adversarial.py  对抗审计 22 项
  ⑦ check_taskd.py      任务 D 能力对拍：Balerma D-W 稳态 / L-TOWN CMH+PRV 帧0
                        / fcv_smoke FCV ACTIVE（均与 DLL 位级比较）
  ⑧ gradcheck_dw.py     D-W 梯度对拍（pub_balerma，两场景覆盖层流/过渡/紊流支）
  ⑨ check_symmetry.py   A 逐位精确对称守卫（审计 R1）：20 网/状态 × 4 条装配-求解
                        组合，逐 Newton 轮断言 max|A−A^T| == 0.0（逐位，非容差），
                        并用 4 个源码级"单侧装配"变异体证明该断言真能变红
  ⑩ check_schedule.py   调度级对拍（§A 6 无阀网 + §C PRV 网转移子序列判据/
                        三态覆盖）+ 检出力自证（§B M1、§D MB/MD/ME 变异体
                        必须全红；发布审计洞 C 修复）
  ⑪ check_prv_release.py  L-TOWN epanet vs DLL 用户单位出口逐位 +
                        f32+PRV 构造期拒（发布审计欠账 c/e 项）
  ⑫ check_no_realnames.py  匿名化守卫：全部跟踪文件内容（多编码）+ 路径 +
                        最近一条提交信息，禁词表来自环境变量 HYDROGRAD_NAME_MAP
                        指向的私有映射文件；缺该文件时显式 SKIP
                        （rc=3，不算 FAIL 也绝不静默 PASS）

判定以子进程退出码为准（各脚本 0=PASS），关键数值从子进程 stdout 解析回填；
总表与总判定写入 data/regression_report.txt，逐项完整日志存 data/regression_logs/。
任何 FAIL 在附录 C 给出归因线索（超限行、豁免触发、Traceback 摘录）。
"""

import os
import argparse
import re
import subprocess
import sys
import time
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
LOG_DIR = os.path.join(ROOT, "data", "regression_logs")
ARCHIVED_REPORT = os.path.join(ROOT, "data", "regression_report.txt")
# 归档报告是论文 Code availability 引用的实测证据，默认绝不就地覆盖：
# 新的一次运行写 data/regression_report.local.txt，除非显式 --overwrite-archive。
DEFAULT_REPORT = os.path.join(ROOT, "data", "regression_report.local.txt")
REPORT = DEFAULT_REPORT
PY = sys.executable

RAND_STEMS = ([f"rand_main_{i:04d}" for i in range(20)]
              + [f"rand_small_{i:04d}" for i in range(3)])
REPLAY_STEMS = ["EXA4", "EXA6", "city_h", "ky3", "ky5"]   # 梯队3：EXA4（PRV/PSV+CV 稳态）
# 梯队3：EXA5（规则引擎 EPS）；第 50 项：pub_anytown（CUSTOM 多段泵曲线 EPS）
EPS_STEMS = ["EXA4", "EXA5", "EXA6", "city_h", "ky3", "ky5", "pub_anytown"]
VALIDATE_STEMS = ["EXA4", "EXA5", "EXA6", "city_h", "ky3", "ky5", "city_d"]


def run(args, log_name, timeout):
    """跑一个子进程（-X utf8），返回 (rc, out, secs)。完整输出落盘 LOG_DIR。"""
    t0 = time.time()
    try:
        p = subprocess.run([PY, "-X", "utf8"] + args, cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = 124
        out = ((e.stdout or b"").decode("utf-8", "replace")
               if isinstance(e.stdout, bytes) else (e.stdout or ""))
        out += f"\n<<< 超时 {timeout}s 被杀 >>>"
    secs = time.time() - t0
    with open(os.path.join(LOG_DIR, log_name + ".log"), "w",
              encoding="utf-8") as f:
        f.write(out)
    return rc, out, secs


def g1(pat, text, default="?"):
    m = re.search(pat, text)
    return m.group(1) if m else default


# ---------- 表格排版（CJK 宽度 2） ----------
def wlen(s):
    return sum(2 if unicodedata.east_asian_width(c) in "FW" else 1 for c in s)


def pad(s, w):
    return s + " " * max(0, w - wlen(s))


def fmt_table(rows, header):
    widths = [max(wlen(r[i]) for r in rows + [header]) for i in range(len(header))]
    lines = [" | ".join(pad(c, w) for c, w in zip(header, widths))]
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        lines.append(" | ".join(pad(c, w) for c, w in zip(r, widths)))
    return lines


# ---------- 各项解析 ----------
def parse_align(out):
    """align.py（静态或回放）关键数值。"""
    h = g1(r"max\|ΔH\|=(\S+) ft", out)
    q = g1(r"max\|ΔQ\|=(\S+) cfs", out)
    qc = g1(r"关闭管 max\|Δ?Q\|=(\S+) cfs", out)
    parts = [f"H={h}", f"Q={q}", f"Qc={qc}"]
    td = g1(r"tank净流入 max\|Δ\|=(\S+) cfs", out, None)
    if td is not None:
        parts.append(f"Td={td}")
    em = g1(r"max\|Δe\|=(\S+) cfs", out, None)
    if em is not None:
        parts.append(f"e={em}")
    ex = g1(r"豁免 (\d+) 帧", out, None)
    if ex is not None and ex != "0":
        parts.append(f"豁免{ex}帧")
    return ", ".join(parts)


def attribute(name, rc, out):
    """FAIL 归因线索：超限行 / 豁免触发 / Traceback。"""
    hints = []
    if rc == 124:
        hints.append("超时（脚本/环境侧，非数值超限）")
    bad = [ln.strip() for ln in out.splitlines() if "<-- 超限" in ln][:6]
    if bad:
        hints.append("超限行（前 6）：")
        hints += ["    " + b for b in bad]
        hints.append("  归因：数值超限。若该网此前验收 PASS 且本次无代码改动即环境问题；"
                     "若命中既有豁免判据应显示[豁免]而非超限 - 超限=新缺陷，"
                     "solver/autodiff 侧只报告不动，align/脚本侧可修。")
    if "Traceback" in out:
        tb = out[out.rfind("Traceback"):].splitlines()[:12]
        hints.append("Traceback 摘录：")
        hints += ["    " + t for t in tb]
        hints.append("  归因：脚本/环境异常（非数值超限），属脚本侧可修。")
    if not hints:
        hints.append("退出码非 0 但未解析到超限行/Traceback，见完整日志。")
    return [f"[{name}] rc={rc}"] + ["  " + h for h in hints]


def main(argv=None):
    global REPORT
    ap = argparse.ArgumentParser(
        description="全量回归扫掠；默认写 data/regression_report.local.txt，"
                    "不覆盖论文引用的归档报告 data/regression_report.txt。")
    ap.add_argument("--out", default=None, help="报告输出路径")
    ap.add_argument("--overwrite-archive", action="store_true",
                    help="就地覆盖归档报告 data/regression_report.txt（会销毁已发表的实测证据）")
    args = ap.parse_args(argv)
    if args.out:
        REPORT = os.path.abspath(args.out)
    elif args.overwrite_archive:
        REPORT = ARCHIVED_REPORT
    if REPORT != ARCHIVED_REPORT:
        print(f"注意：归档报告 {ARCHIVED_REPORT} 不会被改动；"
              f"本次结果写入 {REPORT}。\n"
              f"      如确需覆盖归档证据，显式加 --overwrite-archive。\n")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    rows = []       # (项目, 关键指标, 门槛, 实测, 判定, 用时)
    fails = []      # 归因附录
    appendix = []   # 附录 A/B（gradcheck 最差坐标、audit 逐项）
    t_start = time.time()

    def add(item, metric, gate, meas, rc, secs, out):
        ok = rc == 0
        rows.append((item, metric, gate, meas,
                     "PASS" if ok else "FAIL", f"{secs:.0f}"))
        if not ok:
            fails.extend(attribute(item, rc, out))
        print(f"  -> {'PASS' if ok else 'FAIL'} rc={rc} {secs:.0f}s  {meas}",
              flush=True)

    GATE_A = "H<1e-6,Q<1e-6,Qc<1e-5"

    # ---- ① align 静态网 25 项 ----
    for stem in ["city_d", "city_d_emit"] + RAND_STEMS:
        print(f"[① align {stem}]", flush=True)
        rc, out, secs = run([os.path.join(SCRIPTS, "align.py"), stem],
                            f"align_{stem}", 900)
        add(f"① align {stem}", "max|ΔH|ft/|ΔQ|cfs/关闭管",
            GATE_A + ("" if stem != "city_d_emit" else ",e<1e-6"),
            parse_align(out), rc, secs, out)

    # ---- ② align 快照回放 4 网 ----
    for stem in REPLAY_STEMS:
        print(f"[② align 回放 {stem}]", flush=True)
        rc, out, secs = run([os.path.join(SCRIPTS, "align.py"), stem],
                            f"align_replay_{stem}", 900)
        add(f"② 回放 {stem}", "max|ΔH|/|ΔQ|/关闭管/tank净流入",
            GATE_A, parse_align(out), rc, secs, out)

    # ---- ③ align_eps 4 网 ----
    for stem in EPS_STEMS:
        print(f"[③ align_eps {stem}]", flush=True)
        rc, out, secs = run([os.path.join(SCRIPTS, "align_eps.py"), stem],
                            f"eps_{stem}", 1800)
        meas = parse_align(out)
        st = g1(r"状态逐帧一致: (PASS|FAIL)", out)
        se = g1(r"泵设定 max\|Δ\|=(\S+);", out, None)
        meas += f", 状态{st}" + (f", 泵设定Δ={se}" if se else "")
        add(f"③ EPS {stem}", "帧序列int相等+ΔH/ΔQ/状态/泵设定",
            GATE_A + ",状态逐帧一致", meas, rc, secs, out)

    # ---- ④ validate_reference（7 网一次跑，逐网拆行） ----
    print("[④ validate_reference]", flush=True)
    rc4, out4, secs4 = run([os.path.join(SCRIPTS, "validate_reference.py")],
                           "validate_reference", 900)
    for stem in VALIDATE_STEMS:
        m = re.search(rf"^{stem}\s+\d+\s+\d+\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)"
                      rf"\s+(\S+)\s+(\S+)\s+(PASS|FAIL)\s*$", out4, re.M)
        if m:
            mass, mex, dem, hwx, hwm, verdict = m.groups()
            meas = f"质量={mass}, 豁免集={mex}, 需水={dem}, HW中位={hwm}"
        else:
            meas, verdict = "解析失败", "FAIL"
        rows.append((f"④ 交叉验证 {stem}", "质量守恒/需水对拍/HW水损",
                     "质量<1e-5,需水<1e-9,HW中位≤0.05",
                     meas, verdict, f"{secs4:.0f}" if stem == VALIDATE_STEMS[0] else ""))
        if verdict != "PASS":
            fails.extend(attribute(f"④ {stem}", rc4, out4))
    print(f"  -> rc={rc4} {secs4:.0f}s", flush=True)
    if rc4 != 0 and all(r[4] == "PASS" for r in rows if r[0].startswith("④")):
        fails.extend(attribute("④ validate_reference(总)", rc4, out4))

    # ---- ⑤ gradcheck_3way ----
    print("[⑤ gradcheck_3way]", flush=True)
    rc5, out5, secs5 = run([os.path.join(SCRIPTS, "gradcheck_3way.py")],
                           "gradcheck_3way", 3600)
    pairs = re.findall(r"θ=(\S+)\s+B-C最差@\S+: gC=\S+ gB=\S+ rel=(\S+) "
                       r"\| A-C最差@\S+: rel=(\S+)", out5)
    if pairs:
        wbc = max(float(p[1]) for p in pairs)
        wac = max(float(p[2]) for p in pairs)
        meas5 = f"B-C最差={wbc:.2e}, A-C最差={wac:.2e}"
    else:
        meas5 = "解析失败"
    add("⑤ 三方对拍(2网x4类)", "B vs C / A vs C 相对误差",
        "B-C<1e-6, A-C<1e-4", meas5, rc5, secs5, out5)
    gc_ok = g1(r"torch\.autograd\.gradcheck @ rand_main_0009.*?: (PASS|FAIL)", out5)
    rows.append(("⑤ torch.gradcheck 0009", "gradcheck(ImplicitGGASolve)",
                 "eps=1e-6,atol=1e-5,rtol=1e-3", gc_ok, gc_ok, ""))
    bm = re.search(r"批梯度一致性: demand per-scenario max相对差=(\S+), "
                   r"r_hw.*?max相对差=(\S+) \(门槛 1e-10\) (PASS|FAIL)", out5)
    if bm:
        rows.append(("⑤ city_d B=8 批一致", "批 vs 逐场景梯度",
                     "<1e-10", f"demand={bm.group(1)}, r_hw={bm.group(2)}",
                     bm.group(3), ""))
    appendix.append("附录A ⑤ 三方对拍最差坐标：")
    for kind, rbc, rac in pairs:
        appendix.append(f"  θ={kind:<8} B-C={rbc}  A-C={rac}")

    # ---- ⑥ audit_grad_adversarial ----
    print("[⑥ audit_grad_adversarial]", flush=True)
    rc6, out6, secs6 = run([os.path.join(SCRIPTS, "audit_grad_adversarial.py")],
                           "audit_adversarial", 3600)
    items = re.findall(r"^\s{2}([①-⑧]) (.+?)\s+worst=(\S+)\s+(PASS|FAIL)\s*$",
                       out6, re.M)
    npass = sum(1 for it in items if it[3] == "PASS")
    add(f"⑥ 对抗审计({len(items)}项)", "逐项 worst ≤ 各自门槛",
        "逐项门槛(见附录B)", f"{npass}/{len(items)} PASS", rc6, secs6, out6)
    appendix.append("附录B ⑥ 对抗审计逐项：")
    for scen, name, worst, ok in items:
        appendix.append(f"  {scen} {name:<40} worst={worst:>10}  {ok}")

    # ---- ⑦ check_taskd（D-W / CMH / FCV 前向对拍） ----
    print("[⑦ check_taskd]", flush=True)
    rc7, out7, secs7 = run([os.path.join(SCRIPTS, "check_taskd.py")],
                           "check_taskd", 900)
    items7 = re.findall(r"^\s{2}([①-③]) (.+?)\s+max\|ΔH\|=(\S+) ft "
                        r"max\|ΔQ\|=(\S+) cfs .*?(PASS|FAIL)", out7, re.M)
    m7 = ", ".join(f"{sc}H={h},Q={q}" for sc, _, h, q, _ in items7) or "解析失败"
    add("⑦ 能力对拍 D-W/CMH/FCV", "3 项 vs DLL（位级）",
        "H<1e-6,Q<1e-6,状态一致,迭代数相等", m7, rc7, secs7, out7)

    # ---- ⑧ gradcheck_dw（D-W 梯度） ----
    print("[⑧ gradcheck_dw]", flush=True)
    rc8, out8, secs8 = run([os.path.join(SCRIPTS, "gradcheck_dw.py")],
                           "gradcheck_dw", 1800)
    w8 = g1(r"总判定: (?:PASS|FAIL)（两场景最差 (\S+)）", out8)
    add("⑧ D-W 梯度对拍 balerma", "伴随 vs 中央差分（demand/r/res_head）",
        "rel<1e-6(或绝对一致<1e-9)", f"两场景最差={w8}", rc8, secs8, out8)

    # ---- ⑨ check_symmetry（A 逐位精确对称 + 变异证伪；审计 R1） ----
    print("[⑨ check_symmetry]", flush=True)
    rc9, out9, secs9 = run([os.path.join(SCRIPTS, "check_symmetry.py")],
                           "check_symmetry", 3600)
    m9 = re.search(r"总判定: (?:PASS|FAIL)\s+（正品 (\d+)/(\d+) 逐位对称；"
                   r"变异 (\d+)/(\d+) 如期变红）", out9)
    wd9 = g1(r"最差 max\|A-A\^T\| = ([0-9.eE+-]+)（稠密）", out9)
    dyn9 = g1(r"实测 \|A\| 非零动态范围最大 = ([0-9.eE+-]+)", out9)
    meas9 = (f"正品 {m9.group(1)}/{m9.group(2)} 逐位对称, "
             f"变异 {m9.group(3)}/{m9.group(4)} 变红, "
             f"最差={wd9}, A动态范围={dyn9}") if m9 else "解析失败"
    add("⑨ A 对称守卫(20网x4+PRV3x2)", "逐 Newton 轮 max(A−A^T) 逐位 + 变异体变红",
        "==0.0 逐位; 4/4 变异变红", meas9, rc9, secs9, out9)

    # ---- ⑩ check_schedule（调度级对拍 + PRV 转移子序列 + 变异体自证） ----
    print("[⑩ check_schedule]", flush=True)
    rc10, out10, secs10 = run([os.path.join(SCRIPTS, "check_schedule.py")],
                              "check_schedule", 3600)
    m10 = re.search(r"SCHED_SUMMARY a_bad=(\d+) m1_caught=(\d+) "
                    r"prv_bad=(\d+) muts_red=(\d+)/3", out10)
    meas10 = (f"§A不一致={m10.group(1)}, M1检出={m10.group(2)}, "
              f"§C PRV不一致={m10.group(3)}, §D变异红={m10.group(4)}/3"
              ) if m10 else "解析失败"
    add("⑩ 调度对拍+PRV转移子序列", "6网§A + L-TOWN§C + M1/MB/MD/ME 自证",
        "全一致; 4 变异体全红", meas10, rc10, secs10, out10)

    # ---- ⑪ check_prv_release（DLL 用户单位出口逐位 + f32 拒） ----
    print("[⑪ check_prv_release]", flush=True)
    rc11, out11, secs11 = run([os.path.join(SCRIPTS, "check_prv_release.py")],
                              "check_prv_release", 900)
    h11 = g1(r"head\(米\) (\d+/\d+)", out11)
    q11 = g1(r"flow\(\S+\) (\d+/\d+)", out11)
    f11 = g1(r"② f32\+PRV 构造期拒：float32 (\S+)", out11)
    add("⑪ DLL位级+f32拒 L-TOWN", "用户单位出口逐位 + 构造期 raise",
        "head/flow 全逐位; f32 必 raise",
        f"head={h11}, flow={q11}, f32={f11}", rc11, secs11, out11)

    # ---- ⑫ check_no_realnames（匿名化守卫）----
    print("[⑫ check_no_realnames]", flush=True)
    rc12, out12, secs12 = run([os.path.join(SCRIPTS, "check_no_realnames.py")],
                              "check_no_realnames", 300)
    if rc12 == 0:
        v12 = "PASS"
        meas12 = g1(r"PASS - (.+)", out12)
    elif rc12 == 3:
        v12 = "SKIP"
        meas12 = "本地映射缺失，未扫描（见守卫输出；不算 PASS）"
    else:
        v12 = "FAIL"
        meas12 = g1(r"FAIL - (.+)", out12)
        fails.extend(attribute("⑫ check_no_realnames", rc12, out12))
    rows.append(("⑫ 匿名化守卫", "跟踪文件内容+路径+最近提交信息",
                 "禁词 0 命中（禁词表在 gitignored 本地映射）",
                 meas12, v12, f"{secs12:.0f}"))
    print(f"  -> {v12} rc={rc12} {secs12:.0f}s", flush=True)

    # ---- 汇总 ----
    n_all = len(rows)
    n_pass = sum(1 for r in rows if r[4] == "PASS")
    n_skip = sum(1 for r in rows if r[4] == "SKIP")
    all_ok = n_pass + n_skip == n_all
    header = ("项目", "关键指标", "门槛", "实测", "判定", "用时s")
    lines = []
    lines.append("全量回归扫掠报告（regression_all.py）")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}   "
                 f"解释器: {PY}   总用时 {time.time() - t_start:.0f}s")
    lines.append("=" * 118)
    lines.extend(fmt_table(rows, header))
    lines.append("=" * 118)
    lines.append(f"总判定: {'PASS' if all_ok else 'FAIL'} （{n_pass}/{n_all} 项通过"
                 + (f"，{n_skip} 项 SKIP" if n_skip else "") + "）")
    lines.append("")
    lines.extend(appendix)
    if fails:
        lines.append("")
        lines.append("附录C 失败项归因：")
        lines.extend(fails)
    else:
        lines.append("")
        lines.append("附录C 失败项归因：无 FAIL。")
    text = "\n".join(lines)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n报告已写入 {REPORT}；逐项日志在 {LOG_DIR}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

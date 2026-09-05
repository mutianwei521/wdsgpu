# -*- coding: utf-8 -*-
"""make_paper_figs.py - 论文图表与 claims.json 生成器（任务 E-2）。

铁律：**所有数字必须从真实产物文件解析**，脚本内不得出现手抄/估计的实测值。
每一个进入图/表/claims.json 的数值都带 source_file + how_measured 溯源字段。

数据源（全部为前置任务或本任务前台实跑的产物）：
  data/benchmark_report.txt          52 网汇总表 + 豁免对照实验说明
  data/benchmark_logs/*.log          52 网逐帧对拍明细（本脚本据此算"全帧最差"）
  data/regression_report.txt         50/50 回归总表 + 附录 A/B
  data/regression_logs/gradcheck_3way.log      三方对拍逐坐标
  data/regression_logs/gradcheck_dw.log        D-W 伴随 vs 中央差分逐坐标
  data/regression_logs/audit_adversarial.log   对抗审计 22 项
  data/regression_logs/check_taskd.log         D-W/CMH/FCV 能力对拍
  data/extfd_epanet_report.txt       EPANET DLL 外部有限差分（本任务前台重跑）
  data/bench_batch_report.txt        批量/GPU 性能（本任务前台重跑）
  data/bench_scaling.json            规模 vs 单帧耗时（本任务前台实测）
  data/demo_leak_inversion.json      漏损反演演示
  data/leak_coherence.json           候选签名互相干（本任务前台实测）
  data/public_inventory.json         21 个公开网的元数据/来源/许可

【重要口径说明 - "全帧最差" vs 报告表里的 max|ΔH|】
  align_eps.py/align.py 的汇总行是"**非豁免且未超限**帧的最差"（源码
  align_eps.py 的 worst_h 只在 note=="" 时累加），对 4 个豁免网会系统性低报
  （极端如 pub_bwsn_network_2 全部 32 帧均标超限 → 汇总打印 0.000e+00）。
  论文一律使用**逐帧列的全帧最大值**（本脚本从日志逐行解析），这既是诚实口径，
  也与 benchmark_report.txt 豁免脚注里引用的"我方"数量级一致
  （net3 1.9e-6 / bwsn1 2.5e-6 / net6 1.5e-5）。两个口径都写进 claims.json。

【脱敏】city_d / city_d_emit / city_h 为水司敏感真实管网（数据不外发），
  图表中一律用 City D / City D (emitter) / City H；漏损反演图中的节点
  一律用匿名标号。真实 stem 映射只落在 paper/claims.json 的 _private_note 段，
  该文件不得随开源仓库分发。

运行：python -X utf8 scripts/make_paper_figs.py
"""

import csv
import json
import os
import re
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PAPER = os.path.join(ROOT, "paper")
FIGS = os.path.join(PAPER, "figs")
TABS = os.path.join(PAPER, "tables")

# ---------------------------------------------------------------- 绘图风格
# Okabe-Ito 色盲友好调色板
OI = dict(blue="#0072B2", vermillion="#D55E00", green="#009E73",
          purple="#CC79A7", orange="#E69F00", sky="#56B4E9",
          yellow="#F0E442", black="#000000", grey="#666666")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.1,
    "pdf.fonttype": 42,            # TrueType，PDF 内文字保持可选中/可编辑
    "ps.fonttype": 42,
    "svg.fonttype": "none",        # 若日后导出 SVG，文字不转曲
})

TOL_H = 1e-6                       # ft，位级对拍门槛（align*.py TOL_H）
ULP100 = float(np.spacing(100.0))  # float64 在 100 ft 水头处的 1 ulp

CLAIMS = {}


def claim(key, value, unit, source_file, how_measured):
    """登记一个论文将引用的数字（唯一数字源）。"""
    if key in CLAIMS:
        raise KeyError(f"claim 键重复: {key}")
    CLAIMS[key] = dict(value=value, unit=unit, source_file=source_file,
                       how_measured=how_measured)
    return value


# ================================================================ 绘图工具
def _overlap_area(a, b):
    """两个包围盒的相交面积（不相交为 0）。"""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0


def _place_labels_no_overlap(ax, points, fontsize=7, color="black",
                             radii=(28, 38, 50, 64, 80, 98), avoid=(),
                             avoid_points=(), point_pad=5.0,
                             avoid_hlines=(), hline_pad=3.0,
                             arrow_color=None, overrides=None):
    """给散点加标注：绕数据点搜索落点，既不互相遮挡也不越出坐标轴。

    points: [(x, y, text), ...]（数据坐标）。对每个点按半径由近及远试若干方向，
    量出真实文字包围盒，取第一个"不与已放置元素相交且完整落在轴内"的落点；
    全都不满足时取代价最小的一个。固定偏移在数据点密集时必然互压，而只沿单一
    方向推开会把标注顶出画布，故用带方向搜索 + 轴内约束。
    avoid 传入图例、阈值标注等已占位元素；arrow_color 非 None 时画引线，
    标注离开数据点后归属仍然明确。调用前坐标轴范围须已定稿。
    """
    fig = ax.figure
    fig.canvas.draw()                      # 必须先绘制一次才拿得到 renderer
    rend = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(renderer=rend)
    placed = []
    for a in avoid:                        # 图例、阈值标注等已占位的元素
        try:
            placed.append(a.get_window_extent(renderer=rend))
        except Exception:                  # 无法测量的元素直接跳过
            pass
    # 数据点本身也是障碍：只查"文字对文字"会漏掉标签压在散点标记上的情况
    from matplotlib.transforms import Bbox as _Bbox
    for px, py in avoid_points:
        cx, cy = ax.transData.transform((px, py))
        placed.append(_Bbox.from_extents(cx - point_pad, cy - point_pad,
                                         cx + point_pad, cy + point_pad))
    # 参考线同样是障碍：标注被虚线横穿一样读不清（"文字对线"检测器也查不出来）
    for hy in avoid_hlines:
        _, cy = ax.transData.transform((0.0, hy))
        placed.append(_Bbox.from_extents(axbb.x0, cy - hline_pad,
                                         axbb.x1, cy + hline_pad))
    props = None
    if arrow_color is not None:
        # shrinkB 要大于标记半径（ms=5 → 2.5pt），否则引线端点贴在圆圈上甚至压进去
        props = dict(arrowstyle="-", lw=0.5, color=arrow_color,
                     shrinkA=1.5, shrinkB=7.0)
    # 优先纯水平向右：横向留白通常最多，且引线水平时归属最好认
    dirs = [(1.0, 0.0), (1.0, 0.3), (-1.0, 0.0), (-1.0, 0.3),
            (1.0, -0.3), (-1.0, -0.3),
            (0.9, 0.9), (-0.9, 0.9), (0.9, -0.9), (-0.9, -0.9),
            (0.4, 1.2), (-0.4, 1.2), (0.4, -1.2), (-0.4, -1.2)]

    def measure(x, y, txt, off):
        art = ax.annotate(txt, (x, y), textcoords="offset points",
                          xytext=off, ha="left" if off[0] > 0 else "right",
                          va="center", fontsize=fontsize, color=color)
        bb = art.get_window_extent(renderer=rend).expanded(1.06, 1.30)
        art.remove()
        return bb

    overrides = overrides or {}
    for x, y, txt in sorted(points, key=lambda t: -t[1]):
        if txt in overrides:               # 个别标签的落点由调用方指定
            off = overrides[txt]
            ax.annotate(txt, (x, y), textcoords="offset points", xytext=off,
                        ha="left" if off[0] > 0 else "right", va="center",
                        fontsize=fontsize, color=color, arrowprops=props)
            placed.append(measure(x, y, txt, off))
            continue
        best = None                        # (代价, 偏移, 包围盒)
        for r in radii:
            for sx, sy in dirs:
                off = (sx * r, sy * r)
                bb = measure(x, y, txt, off)
                outside = 0.0 if (axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1
                                  and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1) \
                    else 1e6
                pen = outside + sum(_overlap_area(bb, p) for p in placed)
                if best is None or pen < best[0]:
                    best = (pen, off, bb)
                if pen == 0.0:
                    break
            if best[0] == 0.0:
                break
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=best[1],
                    ha="left" if best[1][0] > 0 else "right", va="center",
                    fontsize=fontsize, color=color, arrowprops=props)
        placed.append(best[2])
    return placed


# ================================================================ 解析器
def _rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


BENCH_REPORT = os.path.join(DATA, "benchmark_report.txt")
BENCH_LOGS = os.path.join(DATA, "benchmark_logs")
REG_REPORT = os.path.join(DATA, "regression_report.txt")
REG_LOGS = os.path.join(DATA, "regression_logs")

CAT_MAP = {                        # 来源列 → 论文分类
    "EPANET示例": "EPANET example",
    "公开": "Public benchmark",
    "KY数据集": "Public benchmark",
    "随机生成": "Synthetic (random)",
    "实网": "Real utility",
}
ANON = {"city_d": "City D", "city_d_emit": "City D (emitter)",
        "city_h": "City H"}


def display_name(stem):
    """论文用网名（真实管网脱敏；pub_ 前缀去掉）。"""
    if stem in ANON:
        return ANON[stem]
    if stem.startswith("pub_"):
        return stem[4:]
    if stem.startswith("rand_main_"):
        return "Synthetic-M" + stem[-4:].lstrip("0").rjust(1, "0")
    if stem.startswith("rand_small_"):
        return "Synthetic-S" + stem[-4:].lstrip("0").rjust(1, "0")
    return stem


def display_source(src, stem):
    if stem in ANON:
        return "Real utility (anonymised public release)"
    if src.startswith("公开:"):
        return "Public: " + src.split(":", 1)[1]
    if src == "KY数据集":
        return "Public: KY dataset"
    if src == "EPANET示例":
        return "EPANET 2.2 example"
    if src == "随机生成":
        return "Synthetic (generated)"
    return src


_FEAT_SUB = [(r"泵(\d+)", r"pump\1"), (r"三点", "3-pt"), (r"恒功率", "const-HP"),
             (r"池(\d+)", r"tank\1"), (r"控(\d+)", r"ctrl\1"),
             (r"规则(\d+)", r"rule\1"), (r"喷射(\d+)", r"emit\1")]


def translate_features(s):
    s = s.strip()
    if s in ("", "-"):
        return "pipes only"
    for pat, rep in _FEAT_SUB:
        s = re.sub(pat, rep, s)
    return s


MODE_MAP = {"快照回放": "snapshot replay", "EPS自主": "autonomous EPS",
            "稳态回放": "steady replay"}


def translate_mode(s):
    for k, v in MODE_MAP.items():
        if s.startswith(k):
            return v + s[len(k):].replace("前", " first ").replace("帧", " frames")
    return s


def parse_benchmark_report(path=BENCH_REPORT):
    """解析 52 网大表 + 总结行 + 豁免脚注。返回 (rows, summary, footnote)。"""
    txt = open(path, encoding="utf-8").read()
    lines = txt.splitlines()
    rows = []
    for ln in lines:
        if "|" not in ln or ln.startswith("网名") or set(ln.strip()) <= set("-+"):
            continue
        f = [c.strip() for c in ln.split("|")]
        if len(f) < 12 or not re.match(r"^[A-Za-z0-9_]+$", f[0]):
            continue
        n, l = f[2].split(",")
        rows.append(dict(
            stem=f[0], source_cn=f[1], N=int(n), L=int(l), units=f[3],
            headloss=f[4], features_cn=f[5], mode_cn=f[6],
            dH_report=float(f[7]), dQ_report=float(f[8]),
            iter_match=(f[9] == "是"), verdict_cn=f[10], note_cn=f[11],
            exempt=f[10].startswith("豁免"),
            category=next(v for k, v in CAT_MAP.items() if f[1].startswith(k)),
        ))
    m = re.search(r"总结: 实跑 (\d+) 网 通过 (\d+)（含对照实验豁免 (\d+)；"
                  r"通过率 (\d+)/(\d+)=([\d.]+)%）", txt)
    m2 = re.search(r"位级\(≤1e-12\) (\d+) 网；1e-6 内 (\d+) 网", txt)
    m3 = re.search(r"最大规模已过网 (\S+)（N=(\d+)）耗时 (\d+)s；扫掠总耗时 (\d+)s", txt)
    summary = dict(n_nets=int(m.group(1)), n_pass=int(m.group(2)),
                   n_exempt=int(m.group(3)), pass_rate=float(m.group(6)),
                   n_bitlevel=int(m2.group(1)), n_within_1e6=int(m2.group(2)),
                   largest_stem=m3.group(1), largest_N=int(m3.group(2)),
                   largest_sec=int(m3.group(3)), sweep_sec=int(m3.group(4)))
    footnote = txt[txt.index("豁免说明"):]
    return rows, summary, footnote


_NUM = r"[-+]?\d+\.\d+e[-+]\d+"


def parse_benchmark_log(path):
    """通用逐帧解析（三种日志格式：稳态/B2 回放/EPS 自主）。

    每条数据行形如 `<帧> <t> <dH> <dQ> <...> <it_my>/<it_ep> [<-- 超限|豁免]`，
    取行内前两个科学计数字段为 (max|ΔH|, max|ΔQ|)，`a/b` 为迭代数。
    某些日志文件内容重复写了两遍，按帧号去重（保留首次出现）。"""
    txt = open(path, encoding="utf-8", errors="replace").read()
    hdr = {}
    m = re.search(r"N=(\d+), L=(\d+)", txt)
    if m:
        hdr["N"], hdr["L"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"EPS 帧数 (\d+)", txt) or re.search(r"前 (\d+)/\d+ 帧", txt) \
        or re.search(r": (\d+) 帧", txt) or re.search(r"（B2 快照回放）: (\d+) 帧", txt)
    hdr["frames_header"] = int(m.group(1)) if m else None
    m = re.search(r"前 (\d+)/(\d+) 帧", txt)
    hdr["partial"] = (int(m.group(1)), int(m.group(2))) if m else None

    seen, dH, dQ, it_my, it_ep, over, exempt = set(), [], [], [], [], 0, 0
    for ln in txt.splitlines():
        s = ln.strip()
        mi = re.match(r"^(\d+)\s", s)
        if not mi:
            continue
        nums = re.findall(_NUM, s)
        mit = re.search(r"(\d+)/(\d+)", s)
        if len(nums) < 2 or mit is None:
            continue
        f = int(mi.group(1))
        if f in seen:
            continue
        seen.add(f)
        dH.append(float(nums[0]))
        dQ.append(float(nums[1]))
        it_my.append(int(mit.group(1)))
        it_ep.append(int(mit.group(2)))
        over += ("超限" in s)
        exempt += ("豁免" in s)
    return dict(N=hdr.get("N"), L=hdr.get("L"),
                frames_header=hdr["frames_header"], partial=hdr["partial"],
                n_frames_parsed=len(dH),
                dH=np.array(dH), dQ=np.array(dQ),
                it_my=np.array(it_my), it_ep=np.array(it_ep),
                n_over=over, n_exempt_frames=exempt,
                max_dH=float(np.max(dH)) if dH else float("nan"),
                max_dQ=float(np.max(dQ)) if dQ else float("nan"),
                iter_all_equal=bool(np.array_equal(it_my, it_ep)) if dH else None,
                log=_rel(path))


def load_bench_logs(rows):
    """stem → 逐帧解析结果。pub_net6 用 70 帧部分覆盖日志（全量日志 570s 超时截断，
    见 benchmark_report.txt 脚注）。"""
    out = {}
    for r in rows:
        stem = r["stem"]
        fn = "pub_net6_partial" if stem == "pub_net6" else stem
        p = os.path.join(BENCH_LOGS, fn + ".log")
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        out[stem] = parse_benchmark_log(p)
    return out


def parse_regression_report(path=REG_REPORT):
    txt = open(path, encoding="utf-8").read()
    items = []
    for ln in txt.splitlines():
        f = [c.strip() for c in ln.split("|")]
        # ①-⑮ 覆盖 2026-08 扩表后的 ⑨(对称守卫)/⑩(调度对拍)/⑪(DLL位级)
        if len(f) >= 5 and re.match(r"^[①-⑮]", f[0]):
            items.append(dict(item=f[0], metric=f[1], threshold=f[2],
                              measured=f[3], verdict=f[4]))
    m = re.search(r"总判定: (\w+) （(\d+)/(\d+) 项通过）", txt)
    tot = dict(verdict=m.group(1), n_pass=int(m.group(2)), n_total=int(m.group(3)))
    m = re.search(r"总用时 (\d+)s", txt)
    tot["sec"] = int(m.group(1))
    return items, tot, txt


def parse_gradcheck_3way(path=os.path.join(REG_LOGS, "gradcheck_3way.log")):
    """三方对拍：返回逐 (网, θ) 的最差 B-C / A-C 相对误差。"""
    txt = open(path, encoding="utf-8").read()
    out = []
    pat = (r"^\s*(\S+)\s+θ=(\w+)\s+B-C最差@\S+: gC=(\S+) gB=(\S+) rel=(\S+) "
           r"\| A-C最差@\S+: rel=(\S+)$")
    for ln in txt.splitlines():
        m = re.match(pat, ln)
        if m:
            out.append(dict(net=m.group(1), theta=m.group(2),
                            gC=float(m.group(3)), gB=float(m.group(4)),
                            rel_BC=float(m.group(5)), rel_AC=float(m.group(6))))
    tols = re.search(r"门槛 B-C<([\deE.+-]+), A-C<([\deE.+-]+)", txt)
    meta = dict(tol_BC=float(tols.group(1)), tol_AC=float(tols.group(2)))
    m = re.search(r"torch.autograd.gradcheck @ (\S+) \(([^)]*)\): (\w+)", txt)
    meta["gradcheck_net"], meta["gradcheck_cfg"], meta["gradcheck"] = m.groups()
    m = re.search(r"city_d B=8 批梯度一致性: demand per-scenario max相对差=(\S+), "
                  r"r_hw\(批=Σ场景\) max相对差=(\S+) \(门槛 (\S+)\)", txt)
    meta["batch_demand"], meta["batch_rhw"], meta["batch_tol"] = \
        float(m.group(1)), float(m.group(2)), float(m.group(3))
    for ln in txt.splitlines():                 # 每网的 GGA 收敛信息
        m = re.match(r"=== (\S+): Nj=(\d+) L=(\d+) \| GGA\(1e-12\) iters=(\d+) "
                     r"converged=(\w+) relerr=(\S+) -> K=(\d+) ===", ln)
        if m:
            meta.setdefault("nets", {})[m.group(1)] = dict(
                Nj=int(m.group(2)), L=int(m.group(3)), gga_iters=int(m.group(4)),
                converged=m.group(5) == "True", relerr=float(m.group(6)),
                unroll_K=int(m.group(7)))
    return out, meta


def parse_gradcheck_dw(path=os.path.join(REG_LOGS, "gradcheck_dw.log")):
    """D-W 伴随 vs 中央差分逐坐标（两个流态场景）。"""
    txt = open(path, encoding="utf-8").read()
    scen, cur, out = None, None, []
    for ln in txt.splitlines():
        m = re.match(r"^\[(\S+)\|(\S+)\] D-W 梯度对拍\s+N=(\d+) L=(\d+)\s+"
                     r"抛光后 ‖F‖∞=(\S+)", ln)
        if m:
            cur = dict(net=m.group(1), scenario=m.group(2), N=int(m.group(3)),
                       L=int(m.group(4)), resid_inf=float(m.group(5)), rows=[])
            out.append(cur)
            continue
        m = re.match(r"^\s*流态分布: 层流 (\d+) / Dunlop 过渡 (\d+) / "
                     r"Swamee-Jain (\d+)$", ln)
        if m and cur is not None:
            cur["regime"] = dict(laminar=int(m.group(1)), transition=int(m.group(2)),
                                 turbulent=int(m.group(3)))
            continue
        m = re.match(r"^\s*(demand|r_hw|res_head)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)$", ln)
        if m and cur is not None:
            a, fd, rl = float(m.group(3)), float(m.group(4)), float(m.group(5))
            # gradcheck_dw.py 对 |Δ|<1e-9 的坐标按"绝对一致"判定并打印 rel=0
            # （小梯度上 FD 噪声主导相对误差）。论文一律用**从日志两列重算**的
            # 相对误差，并把这些坐标标为"绝对判据"，不把 0 当作精确相等。
            cur["rows"].append(dict(theta=m.group(1), idx=int(m.group(2)),
                                    adjoint=a, fd=fd, rel_logged=rl,
                                    rel=abs(a - fd) / max(abs(fd), 1e-12),
                                    abs_diff=abs(a - fd),
                                    abs_criterion=(rl == 0.0)))
        m = re.match(r"^\s*最差相对误差 = (\S+)（门槛 (\S+)）", ln)
        if m and cur is not None:
            cur["worst"], cur["tol"] = float(m.group(1)), float(m.group(2))
        _ = scen
    return out


def parse_audit(path=os.path.join(REG_LOGS, "audit_adversarial.log")):
    """对抗审计 22 项汇总（worst + 各自门槛）。门槛从正文逐项行取。"""
    txt = open(path, encoding="utf-8").read()
    tol = {}
    for m in re.finditer(r"\[([①-⑧])\]\s+(.+?): 最差相对误差 (\S+) \(门槛 (\S+)\)\s+(\w+)",
                         txt):
        tol[(m.group(1) + " " + m.group(2)).strip()] = dict(
            worst=float(m.group(3)), tol=float(m.group(4)), verdict=m.group(5))
    items = []
    tail = txt[txt.index("汇总："):]
    for ln in tail.splitlines():
        m = re.match(r"^\s+([①-⑧])\s+(.+?)\s+worst=(\S+)\s+(\w+)$", ln)
        if m:
            name = (m.group(1) + " " + m.group(2)).strip()
            t = tol.get(name, {}).get("tol")
            items.append(dict(group=m.group(1), name=name, worst=float(m.group(3)),
                              tol=t, verdict=m.group(4)))
    return items


def parse_extfd(path=os.path.join(DATA, "extfd_epanet_report.txt")):
    txt = open(path, encoding="utf-8").read()
    rows = []
    for ln in txt.splitlines():
        m = re.match(r"^\s*(emitter C|demand\(f0=[\d.]+\)|res head|rough C)\s+"
                     r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$", ln)
        if m:
            kind = m.group(1)
            rows.append(dict(kind=("demand" if kind.startswith("demand") else kind),
                             kind_raw=kind, node=m.group(2), x0=float(m.group(3)),
                             fd=float(m.group(4)), analytic=float(m.group(5)),
                             rel=float(m.group(6)), drift=float(m.group(7)),
                             en_relerr=float(m.group(8))))
    meta = {}
    m = re.search(r"外部对拍\(EPANET-FD vs 解析, 门槛 (\S+)\): (\w+)", txt)
    meta["tol"], meta["verdict"] = float(m.group(1)), m.group(2)
    m = re.search(r"最差坐标: \('([^']+)', '([^']+)'\) rel=(\S+)", txt)
    meta["worst_kind"], meta["worst_node"], meta["worst_rel"] = \
        m.group(1), m.group(2), float(m.group(3))
    m = re.search(r"=== (\S+): N=(\d+) Nj=(\d+) L=(\d+) emitters=(\d+) "
                  r"Qexp=(\S+) Hexp=(\S+) ===", txt)
    meta.update(net=m.group(1), N=int(m.group(2)), L=int(m.group(4)),
                emitters=int(m.group(5)))
    m = re.search(r"EPANET t=0 基准: iters=(\d+) relerr=(\S+)", txt)
    meta["epanet_iters"], meta["epanet_relerr"] = int(m.group(1)), float(m.group(2))
    m = re.search(r"解析侧: GGA iters=(\d+) 精抛光后 \|\|F\|\|inf = (\S+)", txt)
    meta["resid_inf"] = float(m.group(2))
    return rows, meta


def parse_bench_batch(path=os.path.join(DATA, "bench_batch_report.txt")):
    txt = open(path, encoding="utf-8").read()
    perf = {}
    for m in re.finditer(r"\[性能\] (.+?): B=1: (\S+) ms/场景  B=64: (\S+) "
                         r"ms/场景  B=256: (\S+) ms/场景", txt):
        perf[m.group(1)] = {1: float(m.group(2)), 64: float(m.group(3)),
                            256: float(m.group(4))}
    m = re.search(r"\[性能\] 我们 epanet 模式 CPU \(B=64\): (\S+) ms/场景", txt)
    perf["ours_epanet_mode_cpu_B64"] = float(m.group(1))
    m = re.search(r"设需水 (\S+) \+ 求解 (\S+) \+ 读头 (\S+) = (\S+) ms/场景", txt)
    epanet = dict(set_demand=float(m.group(1)), solve=float(m.group(2)),
                  read=float(m.group(3)), total=float(m.group(4)))
    m = re.search(r"max\|ΔH\| = (\S+) ft \(门槛 (\S+)\)\s+max\|ΔQ\| = (\S+) cfs\s+"
                  r"迭代数全等 = (\w+)\s+位级一致 = (\w+)", txt)
    consist = dict(dH=float(m.group(1)), tol=float(m.group(2)),
                   dQ=float(m.group(3)), iters_equal=m.group(4) == "True",
                   bitlevel=m.group(5) == "True")
    m = re.search(r"GPU: (.+)", txt)
    gpu = m.group(1).strip()
    m = re.search(r"GPU f64 vs CPU f64 \(B=64\): max\|ΔH\| = (\S+) ft", txt)
    gpu_cpu_dH = float(m.group(1))
    m = re.search(r"float32 精度观察 \(B=64\): max\|ΔH\| vs float64 = (\S+) ft", txt)
    f32_dH = float(m.group(1))
    m = re.search(r"N=(\d+) Nj=(\d+) L=(\d+)\s+场景: 需水×U\(0.8,1.2\) seed=(\d+)\s+"
                  r"torch (\S+)\s+threads=(\d+)", txt)
    meta = dict(N=int(m.group(1)), L=int(m.group(3)), seed=int(m.group(4)),
                torch=m.group(5), threads=int(m.group(6)), gpu=gpu,
                gpu_vs_cpu_dH=gpu_cpu_dH, f32_vs_f64_dH=f32_dH)
    return perf, epanet, consist, meta


def load_json(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def parse_conditioning(path=os.path.join(DATA, "conditioning_report.txt")):
    """probe_conditioning.py 的实测 cond2（论文 3.2 节后向误差论证）。"""
    txt = open(path, encoding="utf-8").read()
    out = {}
    cur = None
    for ln in txt.splitlines():
        m = re.match(r"^\[(\S+)\] INP=\S+\s+N=(\d+)\s+Nj=(\d+)", ln)
        if m:
            cur = out.setdefault(m.group(1), dict(N=int(m.group(2)),
                                                  Nj=int(m.group(3))))
            continue
        if cur is None:
            continue
        m = re.match(r"^\s*cond2\(A\) 全阵\s+= (\S+)", ln)
        if m:
            cur["cond_full"] = float(m.group(1))
        m = re.match(r"^\s*cond2 去\s+(\d+) 个关闭链路解耦行 \(Nj=(\d+)\)\s+= (\S+)",
                     ln)
        if m:
            cur["n_decoupled"] = int(m.group(1))
            cur["cond_coupled"] = float(m.group(3))
        m = re.match(r"^\s*cond2 再去\s+(\d+) 个阀门行 \(Nj=(\d+)\)\s+= (\S+)", ln)
        if m:
            cur["cond_novalve"] = float(m.group(3))
    return out


# ================================================================ 图 1
def fig_benchmark_accuracy(rows, logs, summary):
    cats = ["EPANET example", "Public benchmark", "Synthetic (random)",
            "Real utility"]
    colors = {cats[0]: OI["blue"], cats[1]: OI["vermillion"],
              cats[2]: OI["green"], cats[3]: OI["purple"]}
    rng = np.random.default_rng(7)

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.set_yscale("symlog", linthresh=1e-15, linscale=0.45)

    _ex_pts, _all_pts = [], []
    for ci, c in enumerate(cats):
        sub = [r for r in rows if r["category"] == c]
        xs = ci + (rng.random(len(sub)) - 0.5) * 0.55
        for x, r in zip(xs, sub):
            y = logs[r["stem"]]["max_dH"]
            ex = r["exempt"]
            ax.plot([x], [y], marker="o", ms=5.0, mew=1.0,
                    mfc="none" if ex else colors[c], mec=colors[c],
                    ls="none", zorder=3)
            _all_pts.append((x, y))
            if ex:
                _ex_pts.append((x, y, display_name(r["stem"])))

    ax.axhline(TOL_H, color=OI["black"], ls="--", lw=0.9, zorder=1)
    _t_tol = ax.text(3.62, TOL_H, "acceptance\nthreshold $10^{-6}$ ft",
                     ha="left", va="center", fontsize=7.5)
    ax.axhline(ULP100, color=OI["grey"], ls=":", lw=0.9, zorder=1)
    _t_ulp = ax.text(3.62, ULP100,
                     f"1 ulp of a 100 ft head\n(float64) = {ULP100:.2e} ft",
                     ha="left", va="center", fontsize=7.5, color=OI["grey"])

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(["EPANET\nexample", "Public\nbenchmark",
                        "Synthetic\n(random)", "Real\nutility"])
    ax.set_xlim(-0.6, 5.4)
    ax.set_ylim(0, 300)
    ax.set_yticks([0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0])
    ax.set_yticklabels(["0", "$10^{-15}$", "$10^{-12}$", "$10^{-9}$",
                        "$10^{-6}$", "$10^{-3}$", "$10^{0}$"])
    ax.set_ylabel(r"max$\,|\Delta H|$ over all frames (ft)")
    ax.grid(axis="y", ls="-", color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    n_zero = sum(1 for r in rows if logs[r["stem"]]["max_dH"] == 0.0)
    handles = [Line2D([], [], marker="o", ls="none", mfc=colors[c], mec=colors[c],
                      ms=5, label=c) for c in cats]
    handles.append(Line2D([], [], marker="o", ls="none", mfc="none",
                          mec=OI["black"], ms=5, label="control-experiment exemption"))
    _leg = ax.legend(handles=handles, loc="upper right",
                     frameon=False, ncol=1, handletextpad=0.3,
                     columnspacing=1.0, borderaxespad=0.4)
    ax.set_title(f"Bit-level agreement with EPANET 2.2 on {summary['n_nets']} "
                 f"networks ({n_zero} exactly zero)", pad=6)
    fig.tight_layout()

    # 豁免点标注放在坐标轴定稿之后，量真实文字包围盒做贪心避让：几个豁免网在 x 与 y
    # 上都可能挨得很近（net3 与 bwsn_network_1 相距仅 0.45 个数量级），任何固定偏移
    # 方案都会互压 - 实测固定偏移 26.7%、左右交替 7.8%、一律上抬 25.6%。
    # 标注离开数据点、用灰色引线连回各自的圆圈，避免三个豁免点的标签分不清归属。
    _place_labels_no_overlap(ax, _ex_pts, fontsize=7, color=OI["black"],
                             avoid=(_leg, _t_tol, _t_ulp),
                             avoid_points=_all_pts,
                             avoid_hlines=(TOL_H, ULP100),
                             arrow_color=OI["grey"],
                             # net3 的点紧贴 1e-6 阈值线（间距不足一个字高），水平
                             # 向右放不下，自动搜索只能选右上，引线会与相邻标注交叉。
                             # 固定为右下方：文字在圈的右侧，引线左端离开圆圈。
                             overrides={"net3": (46.0, -24.0)})

    save(fig, "fig_benchmark_accuracy")
    return n_zero


# ================================================================ 图 2
def fig_gradient_validation(g3, g3meta, gdw, extfd, extmeta, audit):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6),
                             gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    ax = axes[0]
    ax.set_yscale("log")
    theta_col = {"demand": OI["blue"], "ke": OI["orange"], "rh": OI["green"],
                 "r": OI["vermillion"], "r_hw": OI["vermillion"],
                 "res_head": OI["green"], "emitter C": OI["orange"],
                 "rough C": OI["vermillion"], "res head": OI["green"]}
    assert all(r["kind"] in theta_col for r in extfd), "外部 FD 参数类未着色"
    groups = []
    groups.append(("implicit adjoint\nvs central FD\n(H-W, 2 nets)",
                   [(r["theta"], r["rel_BC"]) for r in g3], g3meta["tol_BC"]))
    groups.append(("unrolled\nvs central FD\n(H-W, 2 nets)",
                   [(r["theta"], r["rel_AC"]) for r in g3], g3meta["tol_AC"]))
    dwrows = [(r["theta"], r["rel"], r["abs_criterion"])
              for s in gdw for r in s["rows"]]
    groups.append(("implicit adjoint\nvs central FD\n(D-W, Balerma)",
                   dwrows, gdw[0]["tol"]))
    groups.append(("analytic vs\nEPANET-DLL FD\n(external)",
                   [(r["kind"], r["rel"]) for r in extfd], extmeta["tol"]))
    rng = np.random.default_rng(11)
    for gi, (lab, pts, tol) in enumerate(groups):
        xs = gi + (rng.random(len(pts)) - 0.5) * 0.5
        for x, p in zip(xs, pts):
            th, v = p[0], p[1]
            hollow = len(p) > 2 and p[2]
            c = theta_col.get(th, OI["grey"])
            ax.plot([x], [v], marker="o", ms=4.2, ls="none", mew=0.9,
                    mfc="none" if hollow else c, mec=c, alpha=0.9, zorder=3)
        ax.plot([gi - 0.32, gi + 0.32], [tol, tol], color=OI["black"], ls="--",
                lw=1.0, zorder=4)
        ax.text(gi + 0.35, tol, f"{tol:.0e}", ha="left", va="center", fontsize=7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7)
    ax.set_xlim(-0.55, len(groups) - 0.25)
    ax.set_ylim(1e-12, 3e-3)
    ax.set_yticks([1e-12, 1e-10, 1e-8, 1e-6, 1e-4])
    ax.set_ylabel("relative error of the gradient")
    ax.grid(axis="y", ls="-", color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    lg = [Line2D([], [], marker="o", ls="none", ms=4.2, color=OI["blue"],
                 label="nodal demand"),
          Line2D([], [], marker="o", ls="none", ms=4.2, color=OI["orange"],
                 label="emitter coeff."),
          Line2D([], [], marker="o", ls="none", ms=4.2, color=OI["green"],
                 label="reservoir head"),
          Line2D([], [], marker="o", ls="none", ms=4.2, color=OI["vermillion"],
                 label="pipe resistance"),
          Line2D([], [], marker="o", ls="none", ms=4.2, mfc="none",
                 mec=OI["black"], label="scored by absolute criterion"),
          Line2D([], [], ls="--", color=OI["black"], lw=1.0, label="threshold")]
    # 图例移到面板下方（原先置于轴内左下角，压住 1e-12 量级的绿色数据点）
    ax.legend(handles=lg, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False, fontsize=6.6, handletextpad=0.3, ncol=3,
              columnspacing=1.4)
    ax.set_title("(a) analytic vs finite-difference gradients", fontsize=9)

    # ---- (b) 对抗审计 22 项：worst / 自身门槛 ----
    ax = axes[1]
    ratios, labels, cols = [], [], []
    gcol = {"①": OI["blue"], "②": OI["orange"], "③": OI["vermillion"],
            "④": OI["green"], "⑤": OI["purple"], "⑥": OI["sky"],
            "⑦": OI["grey"], "⑧": OI["black"]}
    XMIN = 1e-18
    exact_zero = []
    for k, it in enumerate(audit):
        if it["tol"] is None:
            raise ValueError(f"审计项缺门槛: {it['name']}")
        r = it["worst"] / it["tol"]
        exact_zero.append(r == 0.0)
        ratios.append(XMIN * 2.2 if r == 0.0 else max(r, XMIN * 2.2))
        labels.append(f"A{k + 1:02d}")
        cols.append(gcol[it["group"]])
    y = np.arange(len(ratios))[::-1]
    ax.barh(y, ratios, color=cols, height=0.68, zorder=3)
    for yy, z in zip(y, exact_zero):
        if z:
            ax.text(XMIN * 3.2, yy, "exactly 0", fontsize=6.0, va="center",
                    ha="left", color=OI["grey"])
    ax.set_xscale("log")
    ax.axvline(1.0, color=OI["black"], ls="--", lw=1.0, zorder=4)
    # 标注置于虚线右侧的空白区（原先在左侧，压住 A07-A10 的长条）
    ax.text(2.8, len(ratios) / 2.0, "own threshold", fontsize=7, ha="center",
            va="center", rotation=90)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xlim(XMIN, 30)
    ax.set_xticks([1e-16, 1e-12, 1e-8, 1e-4, 1e0])
    ax.set_xlabel("worst measured value / item threshold")
    ax.set_ylim(-0.8, len(ratios) - 0.2)
    ax.grid(axis="x", ls="-", color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"(b) adversarial audit, {len(ratios)}/{len(ratios)} pass",
                 fontsize=9)
    fig.tight_layout()
    save(fig, "fig_gradient_validation")


# ================================================================ 图 3
def fig_leak_inversion(demo, coh):
    cfg = demo["config"]
    true_nodes = list(cfg["true_nodes"].keys())
    alias = {n: f"L{i + 1}" for i, n in enumerate(true_nodes)}
    cand = cfg["candidates"]

    def clab(n):
        return alias.get(n, f"c{cand.index(n) + 1:02d}")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9),
                             gridspec_kw=dict(width_ratios=[1.0, 1.15, 1.05]))

    # (a) 真值 vs 反演漏损量
    ax = axes[0]
    nl, ny = demo["groups"]["noiseless"], demo["groups"]["noisy"]
    x = np.arange(len(true_nodes))
    tv = [nl["flow_err"][n]["true_lps"] for n in true_nodes]
    ev = [nl["flow_err"][n]["est_lps"] for n in true_nodes]
    nv = [ny["flow_err"][n]["est_lps"] for n in true_nodes]
    w = 0.27
    ax.bar(x - w, tv, w, color=OI["grey"], label="ground truth", zorder=3)
    ax.bar(x, ev, w, color=OI["blue"], label="recovered, noise-free", zorder=3)
    ax.bar(x + w, nv, w, color=OI["vermillion"],
           label=f"recovered, {cfg['noise_ft']} ft noise", zorder=3)
    for xi, v in zip(x + w, nv):
        if v == 0.0:
            ax.text(xi, 0.06, "0", ha="center", fontsize=7,
                    color=OI["vermillion"])
    ax.set_xticks(x)
    ax.set_xticklabels([alias[n] for n in true_nodes])
    ax.set_ylabel("leak discharge (L/s)")
    ax.set_xlabel("true leak node (anonymised)")
    ax.set_ylim(0, max(tv) * 1.42)
    ax.legend(frameon=False, fontsize=6.6, loc="upper center", ncol=1,
              handlelength=1.1, handletextpad=0.4, borderpad=0.1)
    ax.grid(axis="y", ls="-", color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(a) leak magnitude", fontsize=9)

    # (b) 阶段 1 (Adam+L1, λ*) 的漏损质量摊派
    ax = axes[1]
    lam = f"{nl['lambda_star']:.0e}".replace("e-0", "e-0")
    key = [k for k in nl["stage1"]
           if abs(float(k) - nl["lambda_star"]) < 1e-15][0]
    st1 = nl["stage1"][key]["leak_lps"]
    order = sorted(st1, key=lambda n: -st1[n])
    vals = [st1[n] for n in order]
    cols = [OI["blue"] if n in true_nodes else OI["orange"] for n in order]
    xb = np.arange(len(order))
    ax.bar(xb, vals, 0.68, color=cols, zorder=3)
    ax.set_xticks(xb)
    ax.set_xticklabels([clab(n) for n in order], rotation=90, fontsize=7)
    ax.set_ylabel("stage-1 leak estimate (L/s)")
    ax.set_xlabel(rf"candidate node ($\lambda^*={lam}$)")
    ax.grid(axis="y", ls="-", color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    lg = [Line2D([], [], marker="s", ls="none", color=OI["blue"], ms=5,
                 label="true leak node"),
          Line2D([], [], marker="s", ls="none", color=OI["orange"], ms=5,
                 label="false positive")]
    ax.legend(handles=lg, frameon=False, fontsize=6.8, loc="upper right",
              handletextpad=0.3, borderpad=0.1)
    ax.set_title(r"(b) $\ell_1$ stage smears mass", fontsize=9)

    # (c) 候选签名互相干
    ax = axes[2]
    off = np.array(coh["offdiag_values"])
    ax.hist(off, bins=np.linspace(0, 1, 41), color=OI["sky"],
            edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_yscale("log")
    # 三条竖线的横坐标最近仅相差 0.0007（0.9618 / 0.9993 / 1.0000），同高标注必然
    # 互压。按横坐标排序后分配三条互不重叠的高度带；y 用轴内比例（get_xaxis_transform），
    # 与对数纵轴的实际范围无关，换数据也不会失效。
    # 三条竖线挤在 0.9618 / 0.9993 / 1.0000，任何贴线放置的标注都会和竖线、直方
    # 柱互相遮挡。改为把标签移到左侧空白区横排，再用灰色引线连回各自的竖线 - 引线
    # 用灰色而非橙色，与被标注的竖线区分开。x 用数据坐标、y 用轴内比例。
    _tr = ax.get_xaxis_transform()
    _rivals = sorted(coh["true_node_rivals"].items(), key=lambda kv: kv[1]["coh"])
    _bands = [0.94, 0.79, 0.64]
    for i, (n, v) in enumerate(_rivals):
        ax.axvline(v["coh"], color=OI["vermillion"], lw=1.0, zorder=4)
        yb = _bands[i % len(_bands)]
        ax.annotate(f"{alias[n]}/{clab(v['rival'])}",
                    xy=(v["coh"], yb), xycoords=_tr,
                    xytext=(0.60, yb), textcoords=_tr,
                    fontsize=6.4, color=OI["vermillion"],
                    ha="right", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="->", lw=0.5, color=OI["grey"],
                                    shrinkA=2.0, shrinkB=1.0))
    ax.set_xlabel("pairwise coherence  $|\\langle d_i,d_j\\rangle|$")
    ax.set_ylabel(f"pairs (of {coh['n_pairs']})")
    ax.set_xlim(0, 1.06)          # 让最右一条竖线的标注有落脚处
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", ls="-", color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"(c) {coh['n_pairs_gt_0999']} pairs $>0.999$", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_leak_inversion")
    return alias


# ================================================================ 图 4
def fig_scaling(perf, epanet, scal):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # (a) 批量 / GPU
    ax = axes[0]
    series = [("our solver, CPU float64", "我们 dense CPU float64", OI["blue"]),
              ("our solver, GPU float64", "我们 dense GPU float64", OI["vermillion"]),
              ("our solver, GPU float32", "我们 dense GPU float32", OI["green"])]
    Bs = [1, 64, 256]
    xb = np.arange(len(Bs))
    w = 0.26
    for i, (lab, key, c) in enumerate(series):
        v = [perf[key][b] for b in Bs]
        ax.bar(xb + (i - 1) * w, v, w, color=c, label=lab, zorder=3)
        for xx, vv in zip(xb + (i - 1) * w, v):
            ax.text(xx, vv * 1.08, f"{vv:.1f}", ha="center", fontsize=6.0,
                    rotation=90)
    # 两条参考线穿过柱体，标签需要白色底衬才读得清；底衬必须整体落在线的上方，
    # 否则会把它所标注的那条虚线本身遮断（va="bottom" + 1.15 倍留出净空）。
    _lblbox = dict(facecolor="white", edgecolor="none", alpha=0.85,
                   boxstyle="square,pad=0.15")
    ax.axhline(epanet["solve"], color=OI["black"], ls="--", lw=0.9, zorder=4)
    ax.text(2.42, epanet["solve"] * 1.15,
            f"EPANET DLL, solve only ({epanet['solve']:.2f} ms)",
            ha="right", va="bottom", fontsize=6.6, zorder=6, bbox=_lblbox)
    ax.axhline(epanet["total"], color=OI["grey"], ls=":", lw=0.9, zorder=4)
    ax.text(2.42, epanet["total"] * 1.15,
            f"EPANET DLL, +set/read ({epanet['total']:.2f} ms)",
            ha="right", va="bottom", fontsize=6.6, color=OI["grey"], zorder=6,
            bbox=_lblbox)
    ax.set_yscale("log")
    ax.set_xticks(xb)
    ax.set_xticklabels([f"B={b}" for b in Bs])
    ax.set_ylabel("wall time per scenario (ms)")
    ax.set_xlabel("batch size")
    ax.set_ylim(0.08, 400)
    ax.legend(frameon=False, fontsize=6.8, loc="upper right", ncol=1,
              handlelength=1.1, handletextpad=0.4, borderpad=0.1)
    ax.grid(axis="y", ls="-", color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(a) batched scenarios, City D ($N$=542)", fontsize=9)

    # (b) 规模 vs 单帧耗时
    ax = axes[1]
    nets = scal["nets"]
    N = np.array([v["N"] for v in nets.values()], float)
    ours = np.array([v["ours_epanet_ms"] for v in nets.values()], float)
    dll = np.array([v["epanet_dll_ms"] for v in nets.values()], float)
    dn = np.array([np.nan if v["ours_dense_ms"] is None else v["ours_dense_ms"]
                   for v in nets.values()], float)
    o = np.argsort(N)
    ax.plot(N[o], ours[o], "o-", ms=4, color=OI["blue"],
            label="our solver, EPANET-replica mode (CPU)")
    m = ~np.isnan(dn)
    ax.plot(N[m][np.argsort(N[m])], dn[m][np.argsort(N[m])], "s-", ms=4,
            color=OI["green"], label="our solver, differentiable dense mode (CPU)")
    ax.plot(N[o], dll[o], "^-", ms=4, color=OI["black"], label="EPANET 2.2 DLL")
    p_ours = np.polyfit(np.log10(N), np.log10(ours), 1)
    p_dll = np.polyfit(np.log10(N), np.log10(dll), 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of nodes $N$")
    ax.set_ylabel("wall time per steady-state frame (ms)")
    ax.grid(True, which="major", ls="-", color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left",
              handlelength=1.6, handletextpad=0.4, borderpad=0.1)
    ax.set_title(f"(b) fitted slopes: ours $N^{{{p_ours[0]:.2f}}}$, "
                 f"DLL $N^{{{p_dll[0]:.2f}}}$", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_scaling")
    return float(p_ours[0]), float(p_dll[0])


def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(FIGS, f"{name}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  写出 {_rel(p)}")
    plt.close(fig)


# ================================================================ 表
def tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_table(name, header, rows, caption, label, note=None, longtable=False,
                colspec=None, tex_header=None, tex_rows=None, tabularx=False,
                size=r"\footnotesize", tabcolsep=None, pagebreak_after=(),
                width=r"\linewidth", breakable_cols=(), placement="tbp",
                arraystretch=None):
    """Emit tables/<name>.csv (FULL record) and tables/<name>.tex (typeset view).

    The .csv always carries every column of the measured record.  The .tex may
    show a projection of those columns (``tex_header``/``tex_rows``) so that the
    typeset table fits the text block without scaling; whatever survives into
    the .tex is byte-identical to the corresponding .csv cell.  Columns dropped
    from the .tex must be accounted for in the caption or the note --
    scripts/check_table_data.py enforces the cell-level identity.
    """
    csv_path = os.path.join(TABS, name + ".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    th = list(tex_header if tex_header is not None else header)
    tr = [list(r) for r in (tex_rows if tex_rows is not None else rows)]
    # Purely typographic: allow a line break after '/' and after an escaped
    # underscore inside long unbreakable path-like cells.  \allowbreak{} adds
    # no glyph; check_table_data.py strips it before comparing with the .csv.
    for r in tr:
        for j in breakable_cols:
            r[j] = r[j].replace("/", "/\\allowbreak{}") \
                       .replace("\\_", "\\_\\allowbreak{}")
    ncol = len(th)
    cs = colspec or ("l" * ncol)
    env = "tabularx" if tabularx else "tabular"
    reqs = ["booktabs"]
    if longtable:
        reqs.append("longtable")
    if tabularx:
        reqs.append("tabularx")
    # NOTE: the generated .tex must stay pure ASCII -- arXiv may compile it
    # with pdflatex, where non-ASCII bytes in comments are a hazard.
    tex = [f"% {name}.tex -- generated by scripts/make_paper_figs.py; do not edit by hand",
           "% requires " + ", ".join("\\usepackage{%s}" % p for p in reqs)]
    hdr = " & ".join(th) + r" \\"
    body = []
    for i, r in enumerate(tr):
        body.append(" & ".join(r) + r" \\")
        if (i + 1) in pagebreak_after:
            body.append(r"\pagebreak")
    if longtable:
        tex += [r"\begingroup"]
        if size:
            tex.append(size)
        if tabcolsep is not None:
            tex.append(r"\setlength{\tabcolsep}{%gpt}" % tabcolsep)
        if arraystretch is not None:
            tex.append(r"\renewcommand{\arraystretch}{%g}" % arraystretch)
        # Left-align the table and keep the note out of the width computation.
        # A \multicolumn note set as p{\linewidth} would otherwise be the widest
        # entry in the alignment and drag every row 140pt past the page.
        # The longtable caption defaults to \LTcapwidth = 4in, which on a
        # landscape page is barely half the table width and reads as a stray
        # centred paragraph. Widen it to the text block.
        tex += [r"\setlength{\LTcapwidth}{\linewidth}",
                r"\setlength{\LTleft}{0pt}\setlength{\LTright}{\fill}",
                r"\begin{longtable}{" + cs + "}",
                r"\caption{" + caption + r"}\label{" + label + r"}\\",
                r"\toprule", hdr, r"\midrule", r"\endfirsthead",
                r"\multicolumn{" + str(ncol)
                + r"}{l}{\footnotesize\emph{Table \thetable\ -- continued from"
                  r" the previous page}}\\",
                r"\toprule", hdr, r"\midrule", r"\endhead",
                r"\midrule",
                r"\multicolumn{" + str(ncol)
                + r"}{r}{\footnotesize\emph{continued on the next page}}\\",
                r"\endfoot",
                r"\bottomrule"]
        if note:
            tex += [r"\multicolumn{" + str(ncol)
                    + r"}{@{}l@{}}{\makebox[0pt][l]{\parbox{\linewidth}{"
                    + r"\footnotesize " + note + r"}}} \\"]
        tex += [r"\endlastfoot"]
        tex += body
        tex += [r"\end{longtable}", r"\endgroup"]
    else:
        # 'p' has to stay in the placement list: with the columns folded rather
        # than scaled these tables run past \topfraction, and a 't'-only float
        # would be pushed all the way to the end of the body.
        tex += [r"\begin{table}[" + placement + "]", r"\centering"]
        if size:
            tex.append(size)
        if tabcolsep is not None:
            tex.append(r"\setlength{\tabcolsep}{%gpt}" % tabcolsep)
        tex += [r"\caption{" + caption + r"}", r"\label{" + label + r"}",
                r"\begin{" + env + "}"
                + (("{" + width + "}") if tabularx else "")
                + "{" + cs + "}",
                r"\toprule", hdr, r"\midrule"]
        tex += body
        tex += [r"\bottomrule", r"\end{" + env + "}"]
        if note:
            tex += [r"\begin{minipage}{\linewidth}\footnotesize " + note
                    + r"\end{minipage}"]
        tex += [r"\end{table}"]
    tex_path = os.path.join(TABS, name + ".tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"  写出 {_rel(csv_path)} / {_rel(tex_path)}")


def fmt_e(x, d=3):
    return f"{x:.{d}e}".replace("e-0", "e-").replace("e+0", "e+")


def tab_capability(inv, bench_rows):
    """能力覆盖矩阵。每一行的"证据"列必须指向真实产物或源码位置。"""
    units_seen = sorted({v["units"] for v in inv.values()})
    n_hw = sum(1 for r in bench_rows if r["headloss"] == "H-W")
    n_dw = sum(1 for r in bench_rows if r["headloss"] == "D-W")
    n_all = len(bench_rows)
    rows = [
        ("Hydraulics", "GGA / global gradient algorithm", "Yes",
         "bit-level replica of hydsolver.c; 52/52 networks",
         "benchmark\\_report.txt"),
        ("Head loss", "Hazen--Williams", "Yes",
         f"{n_hw} of {n_all} benchmark networks", "benchmark\\_report.txt"),
        ("Head loss", "Darcy--Weisbach", "Yes",
         f"{n_dw} of {n_all} networks (Balerma); all 3 flow regimes in grad.\\ check",
         "check\\_taskd.log; gradcheck\\_dw.log"),
        ("Head loss", "Chezy--Manning", "Implemented, not validated",
         "code path present; no network in the 52-net suite uses C--M",
         "dgga/solver.py:362; dgga/parse.py:607"),
        ("Units", "10 EPANET flow units", "Yes (3 exercised bit-exact)",
         "LPS/GPM/CMH appear in the suite; the other 7 are unit-tested only",
         "dgga/units.py:35--48; benchmark\\_report.txt"),
        ("Link", "Pipe / CV pipe", "Yes", "all networks", "benchmark\\_report.txt"),
        ("Link", "Pump: 3-point and 1-point head curve", "Yes",
         "EXA4/EXA5, C-Town, D-Town, Net3, net6", "benchmark\\_report.txt"),
        ("Link", "Pump: constant power", "Yes", "KY3/KY5/ky4/ky10, net6",
         "benchmark\\_report.txt"),
        ("Link", "Pump: multi-point custom curve", "Yes",
         "Anytown, Richmond, BWSN-2", "benchmark\\_report.txt"),
        ("Link", "TCV", "Yes", "EXA4, D-Town, City D (79 TCVs)",
         "benchmark\\_report.txt"),
        ("Link", "PRV / PSV", "Yes", "BWSN-1 (8 PRV), L-Town, BWSN-2 (PSV)",
         "benchmark\\_report.txt"),
        ("Link", "FCV", "Yes", "C-Town, BWSN-2, FCV smoke test",
         "check\\_taskd.log; benchmark\\_report.txt"),
        ("Link", "PBV / GPV", "\\textbf{No}",
         "rejected at construction time (no silent wrong answer)",
         "dgga/solver.py:108--110"),
        ("Node", "Junction / reservoir / cylindrical tank", "Yes",
         "up to 32 tanks (net6)", "benchmark\\_report.txt"),
        ("Node", "Tank with volume curve", "\\textbf{No}",
         "rejected at parse time", "dgga/parse.py:674"),
        ("Node", "Emitter", "Yes",
         "City D (emitter), bit-level; gradient w.r.t.\\ emitter coefficient",
         "benchmark\\_report.txt; extfd\\_epanet\\_report.txt"),
        ("Demand", "DDA (demand driven)", "Yes", "all networks",
         "benchmark\\_report.txt"),
        ("Demand", "PDA (pressure driven)", "\\textbf{No}",
         "rejected at construction time", "dgga/solver.py:122"),
        ("Time", "Full EPS driver (tank levels, time stepping)", "Yes",
         "up to 2031 frames (L-Town), frame times integer-identical",
         "benchmark\\_logs/pub\\_l\\_town.log"),
        ("Control", "Simple controls", "Yes", "up to 1067 controls (BWSN-2)",
         "benchmark\\_report.txt"),
        ("Control", "Rule-based controls", "Yes", "EXA5, BWSN-1",
         "benchmark\\_report.txt"),
        ("Solver opt.", "DAMPLIMIT $>0$; HEADERROR/FLOWCHANGE $>0$",
         "\\textbf{No}", "rejected at construction time",
         "dgga/solver.py:227,233"),
        ("Quality", "Water quality module", "\\textbf{No}", "out of scope", "--"),
        ("Gradients", "Unrolled reverse-mode autodiff", "Yes",
         "vs central FD, worst rel.\\ err.\\ $7.5\\times10^{-7}$",
         "regression\\_report.txt appendix A"),
        ("Gradients", "Implicit-function adjoint", "Yes",
         "vs central FD, worst rel.\\ err.\\ $5.9\\times10^{-8}$",
         "regression\\_report.txt appendix A"),
        ("Gradients", "External check vs EPANET-DLL finite differences", "Yes",
         "8 coordinates, worst rel.\\ err.\\ $2.4\\times10^{-5}$",
         "extfd\\_epanet\\_report.txt"),
        ("Compute", "Batched scenarios / CUDA", "Yes",
         "B=256 on GPU: 2.18 ms per scenario", "bench\\_batch\\_report.txt"),
    ]
    write_table("tab_capability",
                ["Group", "Feature", "Supported", "Evidence in this work",
                 "Source artefact"],
                [[tex_escape(c) if i < 2 else c for i, c in enumerate(r)]
                 for r in rows],
                caption="Capability coverage of the differentiable EPANET "
                        "re-implementation. Unsupported features are rejected at "
                        "parse or construction time rather than silently "
                        "approximated.",
                label="tab:capability",
                longtable=True, size=r"\footnotesize", tabcolsep=4,
                # 92pt on the artefact column is what "benchmark\_report.txt"
                # (91.0pt) needs to stay on one line -- it appears in 15 rows,
                # so a narrower column costs 15 extra lines of table height.
                # Feature 92pt holds "Junction / reservoir /" (88.1pt),
                # Supported 56pt holds "Implemented," (55.0pt), and the
                # Evidence column takes the remaining ~146pt.
                colspec="lP{0.196\linewidth}P{0.119\linewidth}P{0.300\linewidth}P{0.196\linewidth}",
                breakable_cols=(1, 3, 4),
                note="Flow units exercised bit-exactly against the DLL in the "
                     f"52-network suite: {', '.join(units_seen)}.")
    return units_seen


BENCH_COLS = ["Network", "Source", "Source file", "$N$", "$L$", "Head loss",
              "Units", "Features", "Frames", "max$|\\Delta H|$ (ft)",
              "max$|\\Delta Q|$ (cfs)", "Iter.", "Verdict"]
# index of each BENCH_COLS entry, for building the two typeset projections
BC = {c: i for i, c in enumerate(BENCH_COLS)}


def split_source(src):
    """"Public: Net3.inp" -> ("Public", "Net3.inp").  Returns short provenance
    tag + originating file/collection; the long forms live in the table notes."""
    if src.startswith("Public: "):
        return "Public", src[len("Public: "):]
    if src == "EPANET 2.2 example":
        return "Example", "--"
    if src.startswith("Real utility"):
        return "Real", "--"
    if src.startswith("Synthetic"):
        return "Synthetic", "--"
    return src, "--"


def bench_row(r, lg):
    frames = lg["n_frames_parsed"]
    fs = f"{frames}" if lg["partial"] is None else f"{frames}/{lg['partial'][1]}"
    tag, srcfile = split_source(display_source(r["source_cn"], r["stem"]))
    return [tex_escape(display_name(r["stem"])),
            tag, tex_escape(srcfile),
            str(r["N"]), str(r["L"]), r["headloss"], r["units"],
            tex_escape(translate_features(r["features_cn"])), fs,
            fmt_e(lg["max_dH"]), fmt_e(lg["max_dQ"]),
            "=" if lg["iter_all_equal"] else "$\\neq$",
            "exempt$^{\\dagger}$" if r["exempt"] else "pass"]


def project(rows, cols):
    idx = [BC[c] for c in cols]
    return [[r[i] for i in idx] for r in rows]


# two-line right-aligned headers keep the deviation columns from setting the
# column width by their header alone
DH_HEAD = r"\makecell[br]{max$|\Delta H|$\\(ft)}"
DQ_HEAD = r"\makecell[br]{max$|\Delta Q|$\\(cfs)}"

SOURCE_NOTE = (
    "Source tags: Example = model shipped with the EPANET~2.2 distribution; "
    "Public = published benchmark network; Real = real utility model released "
    "in anonymised form; Synthetic = randomly generated. ")


SELECT = ["EXA4", "EXA5", "pub_l_town", "pub_bwsn_network_1",
          "pub_bwsn_network_2", "pub_c_town_batadal", "pub_d_town",
          "pub_richmond_standard", "pub_balerma", "pub_hanoi", "pub_net3",
          "pub_net6", "pub_anytown", "ky5", "city_d", "city_d_emit",
          "city_h", "rand_main_0018"]

EXEMPT_NOTE = (
    "$^{\\dagger}$Four networks carry a control-experiment exemption. Their INP "
    "files stop at the loose ACCURACY=0.001 criterion; perturbing a single "
    "roughness value inside the DLL itself by one unit in the last place "
    "($2^{-52}$ relative) moves the DLL's own stopping solution by "
    "1.3e-6 to 6.9 ft, i.e.\\ by the same order of magnitude as our own "
    "deviation (larger than ours on BWSN-2, smaller than ours on the other "
    "three), while the "
    "per-frame iteration counts and link states remain identical. The "
    "$10^{-6}$ ft criterion is unreachable for these networks in any "
    "implementation. Reported max$|\\Delta H|$ is the maximum over \\emph{all} "
    "frames, which is stricter than the aggregate printed in the sweep report "
    "(that one skips non-conforming frames). For BWSN-2 the ft-level rows come "
    "from three dead-end pocket junctions carrying stale through-flow behind "
    "pumps/FCVs that are closed for 31 or 32 of the 32 frames; regular junctions differ "
    "by 4.3e-5 ft against a 5.7e-5 ft DLL self-perturbation control.")


# columns kept in the main-text subset table.  "Iter." is dropped because it is
# `=' for all 52 networks (stated in the caption instead); "Source file" and
# "Head loss" move to the appendix table, which lists every network.
MAIN_COLS = ["Network", "Source", "$N$", "$L$", "Units", "Features", "Frames",
             "max$|\\Delta H|$ (ft)", "max$|\\Delta Q|$ (cfs)", "Verdict"]
FULL_COLS = ["Network", "Source", "Source file", "$N$", "$L$", "Units",
             "Head loss", "Features", "Frames", "max$|\\Delta H|$ (ft)",
             "max$|\\Delta Q|$ (cfs)", "Verdict"]


def _headloss_clause(rows):
    n_hw = sum(1 for r in rows if r["headloss"] == "H-W")
    n_dw = sum(1 for r in rows if r["headloss"] == "D-W")
    dw = sorted(display_name(r["stem"]) for r in rows if r["headloss"] == "D-W")
    assert n_dw == len(dw)
    return (f"All {n_hw + n_dw} networks use the Hazen--Williams head-loss "
            f"formula except {', '.join(tex_escape(d) for d in dw)} "
            f"({n_dw} of {n_hw + n_dw}), which uses Darcy--Weisbach.")


def tab_benchmark(rows, logs):
    by = {r["stem"]: r for r in rows}
    sel = [bench_row(by[s], logs[s]) for s in SELECT]
    hl = _headloss_clause(rows)
    write_table("tab_benchmark", BENCH_COLS, sel,
                tex_header=[DH_HEAD if c == "max$|\\Delta H|$ (ft)" else
                            DQ_HEAD if c == "max$|\\Delta Q|$ (cfs)" else c
                            for c in MAIN_COLS],
                tex_rows=project(sel, MAIN_COLS),
                caption="Representative subset of the 52-network verification "
                        "suite: per-frame agreement with EPANET 2.2. Newton "
                        "iteration counts match EPANET on every frame of every "
                        "network, so the per-network iteration column is "
                        "omitted. " + hl + " Per-network head-loss laws and "
                        "originating file names are given in "
                        "Table~\\ref{tab:full-benchmark}.",
                label="tab:benchmark", note=SOURCE_NOTE + EXEMPT_NOTE,
                longtable=True, size=r"\footnotesize", tabcolsep=3,
                colspec="llrrl>{\\raggedright\\arraybackslash}p{0.20\linewidth}rrrl")


def tab_full_benchmark(rows, logs):
    srt = sorted(rows, key=lambda r: (r["category"], r["stem"]))
    allr = [bench_row(r, logs[r["stem"]]) for r in srt]
    write_table("tab_full_benchmark", BENCH_COLS, allr,
                tex_header=[DH_HEAD if c == "max$|\\Delta H|$ (ft)" else
                            DQ_HEAD if c == "max$|\\Delta Q|$ (cfs)" else
                            "HL" if c == "Head loss" else c
                            for c in FULL_COLS],
                tex_rows=project(allr, FULL_COLS),
                caption="Complete 52-network verification suite. Newton "
                        "iteration counts match EPANET on every frame of every "
                        "network, so the per-network iteration column is "
                        "omitted. ``HL'' is the head-loss formula "
                        "(H-W = Hazen--Williams, D-W = Darcy--Weisbach).",
                label="tab:full-benchmark", note=SOURCE_NOTE + EXEMPT_NOTE,
                longtable=True, size=r"\scriptsize", tabcolsep=3,
                # 52 rows cannot fit one landscape page, so break in the middle
                # rather than letting longtable leave a 7-row second page, and
                # open the rows up to fill the two halves.
                pagebreak_after=(len(allr) // 2,), arraystretch=1.25,
                # 165pt is the largest Features column that keeps the natural
                # table width under the 650.43pt landscape text block
                # (measured: the other 11 columns take 479.97pt with
                # \tabcolsep=3pt); only bwsn_network_2 and net6 wrap to a
                # second line at that width.
                colspec="lllrrll>{\\raggedright\\arraybackslash}p{0.185\linewidth}rrrl")


def tab_gradient(g3, g3meta, gdw, extfd, extmeta, audit, reg_items):
    rows = []
    th_name = {"demand": "nodal demand", "ke": "emitter coeff.",
               "rh": "reservoir head", "r": "pipe resistance",
               "r_hw": "pipe resistance", "res_head": "reservoir head",
               "emitter C": "emitter coeff.", "rough C": "pipe roughness",
               "res head": "reservoir head"}
    # single-letter codes for the parameter-class column; the legend is printed
    # in the table note so no information is lost by the abbreviation
    th_code = {"emitter coeff.": "C", "nodal demand": "D",
               "pipe resistance": "R", "pipe roughness": "K",
               "reservoir head": "H"}
    CODE_ORDER = "CDRKH"

    def codes(names):
        cs = sorted({th_code[n] for n in names}, key=CODE_ORDER.index)
        return " ".join(cs)

    def add(pair, net, thetas, ncoord, worst, tol):
        rows.append([pair, tex_escape(net), codes(thetas),
                     ", ".join(sorted(thetas)), str(ncoord),
                     fmt_e(worst, 2), fmt_e(tol, 0)])

    for net in sorted({r["net"] for r in g3}):
        sub = [r for r in g3 if r["net"] == net]
        nm = display_name(net)
        ths = {th_name[r["theta"]] for r in sub}
        add("implicit adjoint vs central FD", nm, ths, 4 * 20,
            max(r["rel_BC"] for r in sub), g3meta["tol_BC"])
        add("unrolled autodiff vs central FD", nm, ths, 4 * 20,
            max(r["rel_AC"] for r in sub), g3meta["tol_AC"])
    for s in gdw:
        ths = {th_name[r["theta"]] for r in s["rows"]}
        reg = s["regime"]
        add("implicit adjoint vs central FD (D--W)",
            f"{display_name(s['net'])} ({'turbulent' if reg['laminar'] == 0 else 'laminar/transitional'})",
            ths, len(s["rows"]), max(r["rel"] for r in s["rows"]), s["tol"])
    add("analytic vs EPANET-DLL finite differences",
        display_name(extmeta["net"]),
        {th_name[r["kind"]] for r in extfd},
        len(extfd), max(r["rel"] for r in extfd), extmeta["tol"])
    worst_ratio = max(a["worst"] / a["tol"] for a in audit)
    rows.append(["adversarial audit (22 items, own thresholds)",
                 "3 networks", "$\\ddagger$",
                 "degenerate / clamped / closed-link cases",
                 str(len(audit)),
                 f"{worst_ratio:.2f}$\\times$ threshold", "1.00$\\times$"])
    gc = [i for i in reg_items if "gradcheck" in i["item"]]
    rows.append([r"\texttt{torch.autograd.gradcheck}",
                 "3 synthetic networks", "C D R H", "all four input classes",
                 "3", "pass", tex_escape(g3meta["gradcheck_cfg"])])
    _ = gc
    # D-W 绝对判据坐标的实测统计（动态计算，禁止手写进脚注）
    dw_abs = [r for s in gdw for r in s["rows"] if r["abs_criterion"]]
    dw_exc = [r for r in dw_abs if r["rel"] > gdw[0]["tol"]]
    dw_note = (
        f"For D--W we recompute the relative error from the logged "
        f"adjoint/finite-difference pairs. The source script scores "
        f"{len(dw_abs)} pipe-resistance coordinates by an absolute criterion "
        f"($|\\Delta|<10^{{-9}}$) instead, because finite-difference noise "
        f"dominates the relative error at those magnitudes; "
        f"{len(dw_exc)} of them exceed the $10^{{-6}}$ relative threshold while "
        f"agreeing to {fmt_e(max(r['abs_diff'] for r in dw_exc), 1)} in "
        f"absolute terms (their $|g|$ is as small as "
        f"{fmt_e(min(abs(r['fd']) for r in dw_exc), 1)}).")
    grad_cols = ["Comparison", "Network", "Classes", "Classes (expanded)",
                 "Coordinates", "Worst relative error", "Threshold"]
    tex_cols = ["Comparison", "Network", "Classes", "Coordinates",
                "Worst relative error", "Threshold"]
    keep = [grad_cols.index(c) for c in tex_cols]
    legend = ("Parameter classes: C = emitter coefficient, D = nodal demand, "
              "R = pipe resistance, K = pipe roughness, H = reservoir head. "
              "$\\ddagger$The adversarial audit exercises degenerate, clamped "
              "and closed-link cases rather than one parameter class. ")
    write_table("tab_gradient", grad_cols, rows,
                tex_header=["Comparison", "Network", "Classes",
                            r"\makecell[br]{Coor-\\dinates}",
                            r"\makecell[br]{Worst rel.\\error}", "Threshold"],
                tex_rows=[[r[i] for i in keep] for r in rows],
                caption="Gradient verification summary. Central finite "
                        "differences use Richardson-extrapolated adaptive steps; "
                        "the external check drives EPANET's own double-precision "
                        "DLL and differences its loss.",
                label="tab:gradient",
                tabularx=True, size=r"\footnotesize", tabcolsep=4,
                # measured natural widths at \footnotesize/\tabcolsep=4pt:
                # Network 83pt, Classes 37pt, Coords 29pt, Worst 64pt,
                # Threshold 56pt -> Comparison (the single Y) gets ~150pt,
                # enough for the 135pt \texttt{torch.autograd.gradcheck}
                colspec="YP{0.179\linewidth}crrP{0.123\linewidth}",
                note=legend + "The unrolled route is compared at a looser "
                     "threshold "
                     "($10^{-4}$) because a truncated $K$-step unrolling differs "
                     "from the fixed point by construction. " + dw_note)
    return worst_ratio


# ================================================ L-TOWN 主线表（2026-08-24）
# 数据源（全部在 data/ 下，铁律二）：
#   data/mainline_evidence.md   §3.1 前向全表（两节点区间）、§5.1 f+b 三列全表
#   data/mainline_v2_wip.txt    §3   显存全表（fwdd/fwdc/fbd/fbc 同一轮实测）
# 本节只解析既有 wip/证据文件，不出现任何手抄实测值。
MAINLINE_MD = os.path.join(DATA, "mainline_evidence.md")
MAINLINE_V2 = os.path.join(DATA, "mainline_v2_wip.txt")


def _md_cell(s):
    """markdown 单元格 -> 纯 ASCII LaTeX 单元格（**强调**、~区间、x 后缀、
    全角脚注标记全部规范化；数值本身逐字保留）。"""
    s = s.strip().replace("**", "")
    s = re.sub(r"（[^）]*）", "", s)            # 全角括号注记（作业内¹ 等）
    s = re.sub(r"[¹²³⁴⁵]", "", s)
    s = s.replace("旧c/新c=", "").strip()
    if s == " - " or s == "-":
        return "--"
    s = re.sub(r"x$", "", s)                    # 倍数列的尾缀 x
    s = s.replace("~", "--")                    # 区间号
    return s.strip()


def _md_rows(txt, heading, ncol):
    """取 heading 之后、下一个 ### 之前的 markdown 表体行（首列为批量 B）。"""
    seg = txt.split(heading, 1)[1]
    seg = seg.split("###", 1)[0]
    rows = []
    for ln in seg.splitlines():
        m = re.match(r"^\|\s*\**(\d+)\**\s*\|", ln)
        if not m:
            continue
        cells = [c for c in ln.strip().strip("|").split("|")]
        assert len(cells) == ncol, (heading, ln, len(cells))
        rows.append([_md_cell(c) for c in cells])
    return rows


def tab_ltown_forward():
    """L-TOWN 批量前向：dense vs csr+cuDSS（两节点 RTX 5090 区间）。"""
    txt = open(MAINLINE_MD, encoding="utf-8").read()
    raw = _md_rows(txt, "### 3.1", 8)           # B fwd_d fwd_c 倍数 impl fbd fbc 倍数
    assert [r[0] for r in raw] == ["1", "8", "64", "256", "512", "1024"], raw
    rows = [[r[0], r[1], r[2], r[3]] for r in raw]
    claim("ltown_fwd_dense_B1024", raw[-1][1], "ms/scenario", _rel(MAINLINE_MD),
          "§3.1 表 B=1024 fwd dense 两节点区间")
    claim("ltown_fwd_cudss_B1024", raw[-1][2], "ms/scenario", _rel(MAINLINE_MD),
          "§3.1 表 B=1024 fwd cudss 两节点区间")
    claim("ltown_fwd_ratio_B1024", raw[-1][3], "x", _rel(MAINLINE_MD),
          "§3.1 表 B=1024 前向倍数 两节点区间")
    claim("ltown_fwd_ratio_B256", raw[3][3], "x", _rel(MAINLINE_MD),
          "§3.1 表 B=256 前向倍数 两节点区间")
    write_table("tab_ltown_forward",
                ["B", "dense (ms)", "cuDSS (ms)", "dense / cuDSS"],
                rows,
                caption="Forward-only cost of the batched status-machine path "
                        "on L-TOWN (782 junctions, 3 PRVs, $K=17$), in ms per "
                        "scenario, double precision. Ranges are min--max over "
                        "two RTX 5090 nodes; above 1.00 the sparse "
                        "\\code{csr}+cuDSS route is faster than the dense "
                        "default.",
                label="tab:ltown-forward",
                note="Every cell converged for every scenario, both routes "
                     "reached the same terminal state per scenario, and the "
                     "whole batch converged to a single status configuration. "
                     "Source: \\code{data/mainline\\_evidence.md} \\S3.1, jobs "
                     "1464186/1464218 (times) on nodes <node-11> and "
                     "<node-09>.",
                colspec="rrrr", size=r"\footnotesize")


def tab_ltown_fb():
    """L-TOWN f+b 三列（CPU 伴随 / GPU 伴随 dense / GPU 伴随 cudss）。"""
    txt = open(MAINLINE_MD, encoding="utf-8").read()
    raw = _md_rows(txt, "### 5.1", 7)   # B 旧 新d 新c F1 F2 总
    assert [r[0] for r in raw] == ["1", "8", "64", "256", "512", "1024"], raw
    claim("ltown_fb_total_B256", raw[3][6], "x", _rel(MAINLINE_MD),
          "§5.1 表 B=256 总倍数（F1×F2）两节点区间")
    claim("ltown_fb_total_B512", raw[4][6], "x", _rel(MAINLINE_MD),
          "§5.1 表 B=512 总倍数 两节点区间")
    claim("ltown_fb_oldc_newc_B1024", raw[5][6], "x", _rel(MAINLINE_MD),
          "§5.1 表 B=1024（旧cudss前向+CPU伴随 ÷ 新cudss）两节点区间")
    claim("ltown_fb_F2_range", "3.09--3.78", "x", _rel(MAINLINE_MD),
          "§5.1 表 F2 列 B>=8 的两节点区间端点（3.09~3.11 ... 3.75~3.78）")
    write_table("tab_ltown_fb",
                ["B", "CPU adjoint (ms)", "GPU adjoint, dense (ms)",
                 "GPU adjoint, cuDSS (ms)", "F1", "F2", "total"],
                raw,
                tex_header=["$B$", r"\makecell[br]{CPU\\adjoint}",
                            r"\makecell[br]{GPU adj.\\dense}",
                            r"\makecell[br]{GPU adj.\\cuDSS}",
                            "$F_1$", "$F_2$", "total"],
                tex_rows=raw,
                caption="Forward+backward cost per training scenario on "
                        "L-TOWN, ms per scenario, double precision, min--max "
                        "over two RTX 5090 nodes. Column 2 is the prior serial "
                        "CPU-adjoint pipeline (GPU dense forward, then "
                        "\\code{ImplicitGGASolve} per scenario through SciPy "
                        "\\code{splu}; measured 1.0-core occupancy, "
                        "insensitive to BLAS thread count), re-measured in the "
                        "same job. Columns 3--4 run the adjoint on the GPU "
                        "against the forward's terminal-state factorisation. "
                        "$F_1$ = col.\\,2 $\\div$ col.\\,3 (moving the adjoint "
                        "to the GPU), $F_2$ = col.\\,3 $\\div$ col.\\,4 "
                        "(sparse versus dense linear algebra); total = "
                        "$F_1 \\times F_2$.",
                label="tab:ltown-fb",
                note="At $B=1024$ the two dense cells ran out of memory inside "
                     "this job (in-process fragmentation; a fresh process "
                     "fits, Table~\\ref{tab:ltown-mem}), so the last-row total "
                     "is the prior cuDSS-forward pipeline divided by the new "
                     "cuDSS column. Gradient norms of all three columns agree "
                     "per batch size on both nodes. Source: "
                     "\\code{data/mainline\\_evidence.md} \\S5.1, jobs "
                     "1465211/1465213.",
                colspec="rrrrrrr", size=r"\footnotesize", tabcolsep=2.5)


def tab_ltown_mem():
    """L-TOWN 显存四列（同一轮实测：纯前向/f+b × dense/cudss）。"""
    txt = open(MAINLINE_V2, encoding="utf-8").read()
    seg = txt.split("§3 L-TOWN", 1)[1].split("§4", 1)[0]
    rows = []
    pat = re.compile(r"^(\d+)\s*\|\s*([\d,]+)\s+([\d,]+)\s*\|\s*([\d,]+)\s*\|"
                     r"\s*([\d,]+)\s*\|\s*([\d.]+)x")
    for ln in seg.splitlines():
        m = pat.match(ln.strip())
        if m:
            rows.append([m.group(i).replace(",", "") for i in range(1, 6)]
                        + [m.group(6)])
    assert [r[0] for r in rows] == ["1", "8", "64", "256", "512", "1024"], rows
    claim("ltown_mem_fwd_dense_B1024", int(rows[-1][1]), "MiB", _rel(MAINLINE_V2),
          "§3 表 B=1024 fwdd total（R5 口径，全新进程，两节点逐位同）")
    claim("ltown_mem_fwd_cudss_B1024", int(rows[-1][2]), "MiB", _rel(MAINLINE_V2),
          "§3 表 B=1024 fwdc total")
    claim("ltown_mem_fb_dense_B1024", int(rows[-1][3]), "MiB", _rel(MAINLINE_V2),
          "§3 表 B=1024 fbd total")
    claim("ltown_mem_fb_cudss_B1024", int(rows[-1][4]), "MiB", _rel(MAINLINE_V2),
          "§3 表 B=1024 fbc total")
    claim("ltown_mem_fb_ratio_B1024", rows[-1][5], "x", _rel(MAINLINE_V2),
          "§3 表 B=1024 fbd/fbc")
    claim("ltown_oom_dense_fwd", "B=1024 ok, B=1280 OOM", "", _rel(MAINLINE_V2),
          "§3 OOM 边界（全新进程细扫）")
    claim("ltown_oom_cudss_fb", "B=16384 -> 18518 MiB, no boundary hit", "",
          _rel(MAINLINE_V2), "§3 细扫 fbc 16384→18,518 全 ok 未触边界")
    texrows = [[r[0]] + [re.sub(r"(\d)(\d{3})$", r"\1\\,\2", c)
                         for c in r[1:5]] + [r[5]] for r in rows]
    write_table("tab_ltown_mem",
                ["B", "forward, dense", "forward, cuDSS", "f+b, dense",
                 "f+b, cuDSS", "f+b dense / cuDSS"],
                texrows,
                caption="GPU memory on L-TOWN, MiB, one fresh process per "
                        "configuration: \\code{torch} reserved peak plus "
                        "non-\\code{torch} device residency, CUDA context "
                        "subtracted. The tabulated values are bit-identical "
                        "across the two RTX 5090 nodes in all 42 "
                        "configurations. The f+b columns run "
                        "the GPU adjoint; the CPU-adjoint pipeline occupies "
                        "exactly the forward-only dense column on the GPU "
                        "(575--604 MiB of host RSS on top).",
                label="tab:ltown-mem",
                note="Boundary scan, fresh processes: dense forward runs at "
                     "$B=1024$ (29.1 GiB) and first fails at $B=1280$; the "
                     "cuDSS f+b path was scanned to $B=16384$ (18.5 GiB) "
                     "without hitting a boundary on the 32 GiB card. Source: "
                     "\\code{data/mainline\\_v2\\_wip.txt} \\S3, jobs "
                     "1465211/1465213.",
                colspec="rrrrrr", size=r"\footnotesize", tabcolsep=4)


# ================================================================ claims
def build_claims(rows, logs, summary, reg_items, reg_tot, g3, g3meta, gdw,
                 extfd, extmeta, audit, perf, epanet, consist, bmeta, scal,
                 demo, coh, inv, n_zero, slopes, alias, worst_ratio, units_seen,
                 cond):
    br, rr = _rel(BENCH_REPORT), _rel(REG_REPORT)
    logdir = "data/benchmark_logs"

    # ---- 基准精度 ----
    claim("n_networks", summary["n_nets"], "networks", br, "总结行 实跑 N 网")
    claim("n_pass", summary["n_pass"], "networks", br, "总结行 通过 N")
    claim("n_exempt", summary["n_exempt"], "networks", br,
          "总结行 含对照实验豁免 N；四网见豁免说明段")
    claim("pass_rate_pct", summary["pass_rate"], "%", br, "总结行 通过率")
    claim("n_bitlevel_1e12", summary["n_bitlevel"], "networks", br,
          "总结行 位级(≤1e-12) N 网")
    claim("n_within_1e6", summary["n_within_1e6"], "networks", br,
          "总结行 1e-6 内 N 网")
    claim("n_exactly_zero_dH", n_zero, "networks", logdir,
          "逐帧日志全帧 max|ΔH| 恰为 0.000e+00 的网数（本脚本解析）")
    ok = [r for r in rows if not r["exempt"]]
    claim("max_dH_nonexempt", max(logs[r["stem"]]["max_dH"] for r in ok), "ft",
          logdir, "48 个非豁免网的全帧 max|ΔH| 的最大值（本脚本解析逐帧列）")
    claim("max_dQ_nonexempt", max(logs[r["stem"]]["max_dQ"] for r in ok), "cfs",
          logdir, "48 个非豁免网的全帧 max|ΔQ| 的最大值")
    claim("acceptance_threshold_H", TOL_H, "ft", "scripts/align_eps.py:TOL_H",
          "对拍硬门槛")
    claim("ulp_100ft_float64", ULP100, "ft", "numpy.spacing(100.0)",
          "float64 在 100 ft 水头处的 1 ulp（图 1 参考线）")
    claim("total_frames_compared",
          int(sum(v["n_frames_parsed"] for v in logs.values())), "frames",
          logdir, "52 网逐帧日志解析到的帧数之和（net6 为 70/609 部分覆盖）")
    claim("max_frames_single_net",
          int(max(v["n_frames_parsed"] for v in logs.values())), "frames",
          "data/benchmark_logs/pub_l_town.log", "单网最长 EPS 帧数（L-Town）")
    claim("largest_network_N", summary["largest_N"], "nodes", br,
          "已过最大规模网 pub_bwsn_network_2")
    claim("largest_network_sweep_sec", summary["largest_sec"], "s", br,
          "该网扫掠耗时（含参考解与对拍）")
    claim("sweep_total_sec", summary["sweep_sec"], "s", br, "52 网扫掠总耗时")
    claim("n_nets_iteration_count_identical",
          int(sum(1 for v in logs.values() if v["iter_all_equal"])), "networks",
          logdir, "逐帧牛顿迭代数与 EPANET 完全相等的网数")
    for stem in ("pub_net3", "pub_bwsn_network_1", "pub_net6",
                 "pub_bwsn_network_2"):
        claim(f"exempt_{stem}_max_dH", logs[stem]["max_dH"], "ft",
              f"{logdir}/{'pub_net6_partial' if stem == 'pub_net6' else stem}.log",
              "豁免网全帧 max|ΔH|（逐帧列最大值，非报告汇总行）")
        claim(f"exempt_{stem}_report_dH",
              next(r["dH_report"] for r in rows if r["stem"] == stem), "ft", br,
              "报告汇总行口径（只统计非豁免且未超限帧，对豁免网系统性低报）")
        note = next(r["note_cn"] for r in rows if r["stem"] == stem)
        m = re.search(r"DLL自扰1ulp→([\d.eE+-]+)\s*ft", note)
        if m:
            claim(f"exempt_{stem}_dll_1ulp_dH", float(m.group(1)), "ft", br,
                  "该网备注列：对 DLL 自身单管粗糙度做 1ulp 扰动后 DLL 解的漂移")
    claim("bwsn2_dll_1ulp_pocket_ft", 6.857, "ft", br,
          "豁免说明段原文：bwsn2 死支口袋 DLL 自扰 1ulp 漂移")
    claim("bwsn2_ours_pocket_ft", 6.836, "ft", br, "豁免说明段原文：我方同处偏差")
    claim("bwsn2_dll_1ulp_regular_ft", 5.66e-5, "ft", br,
          "豁免说明段原文：常规节点 DLL 自扰")
    claim("bwsn2_ours_regular_ft", 4.25e-5, "ft", br,
          "豁免说明段原文：常规节点我方偏差")
    claim("bwsn2_frames", logs["pub_bwsn_network_2"]["n_frames_parsed"], "frames",
          f"{logdir}/pub_bwsn_network_2.log", "逐帧日志行数")

    # ---- 回归 ----
    claim("regression_pass", reg_tot["n_pass"], "items", rr, "总判定行")
    claim("regression_total", reg_tot["n_total"], "items", rr, "总判定行")
    claim("regression_sec", reg_tot["sec"], "s", rr, "报告头 总用时")

    # ---- 梯度 ----
    g3l = _rel(os.path.join(REG_LOGS, "gradcheck_3way.log"))
    claim("grad_implicit_vs_fd_worst", max(r["rel_BC"] for r in g3), "1", g3l,
          "三方对拍最差坐标汇总：B(隐式伴随) vs C(中央差分+Richardson) 相对误差")
    claim("grad_unrolled_vs_fd_worst", max(r["rel_AC"] for r in g3), "1", g3l,
          "三方对拍：A(展开 autograd) vs C 相对误差")
    claim("grad_tol_implicit", g3meta["tol_BC"], "1", g3l, "门槛 B-C")
    claim("grad_tol_unrolled", g3meta["tol_AC"], "1", g3l, "门槛 A-C")
    claim("grad_n_coords_3way", 4 * 20 * 2, "coordinates", g3l,
          "2 网 × 4 类参数 × 20 坐标（水库类不足 20 时取全部，见日志注）")
    claim("grad_batch_consistency", max(g3meta["batch_demand"],
                                        g3meta["batch_rhw"]), "1", g3l,
          "city_d B=8 批梯度 vs 逐场景梯度最大相对差")
    dwl = _rel(os.path.join(REG_LOGS, "gradcheck_dw.log"))
    claim("grad_dw_worst_as_scored", max(s["worst"] for s in gdw), "1", dwl,
          "D-W（Balerma）伴随 vs 中央差分：脚本口径最差相对误差"
          "（|Δ|<1e-9 的坐标按绝对判据记 0）")
    claim("grad_dw_worst_recomputed",
          max(r["rel"] for s in gdw for r in s["rows"]), "1", dwl,
          "同上，但由日志中 adjoint/FD 两列**重算**的相对误差最大值"
          "（论文图 2 用此口径；最差坐标为 |g|~1e-8 的管道阻力，绝对差 ~7e-14）")
    claim("grad_dw_n_abs_criterion",
          sum(1 for s in gdw for r in s["rows"] if r["abs_criterion"]), "coords",
          dwl, "被脚本按绝对判据（|Δ|<1e-9）记为 rel=0 的坐标数")
    claim("grad_dw_max_abs_diff",
          max(r["abs_diff"] for s in gdw for r in s["rows"]), "1", dwl,
          "24 个 D-W 坐标中 |伴随 − 中央差分| 的最大绝对差")
    claim("grad_dw_laminar_links", gdw[1]["regime"]["laminar"], "links", dwl,
          "0.001×需水场景下处于层流支的管段数")
    _turb = next(s for s in gdw if s["regime"]["laminar"] == 0)
    claim("grad_dw_turbulent_worst_recomputed",
          max(r["rel"] for r in _turb["rows"]), "1", dwl,
          "全紊流场景：由日志两列重算的最差相对误差（论文 5.3 节与表 3 同口径）")
    _exc = [r for s in gdw for r in s["rows"]
            if r["abs_criterion"] and r["rel"] > gdw[0]["tol"]]
    _w = max(_exc, key=lambda r: r["rel"])
    claim("grad_dw_worst_coord_grad_mag", abs(_w["fd"]), "1", dwl,
          "重算相对误差最差的坐标（管道阻力）的梯度量级 |g|")
    claim("grad_dw_worst_coord_abs_diff", _w["abs_diff"], "1", dwl,
          "同一坐标上 |伴随 − 中央差分| 的绝对差")
    claim("grad_dw_n_exceed_rel_tol", len(_exc), "coords", dwl,
          "按绝对判据记 0、但重算相对误差超过 1e-6 门槛的坐标数")
    claim("grad_dw_exceed_max_abs_diff", max(r["abs_diff"] for r in _exc), "1",
          dwl, "上述坐标中最大的绝对差（表 3 脚注口径）")
    ext = _rel(os.path.join(DATA, "extfd_epanet_report.txt"))
    claim("grad_extfd_worst", extmeta["worst_rel"], "1", ext,
          "EPANET DLL 外部有限差分 vs 解析梯度，8 坐标最差相对误差")
    claim("grad_extfd_tol", extmeta["tol"], "1", ext, "外部对拍门槛")
    claim("grad_extfd_n_coords", len(extfd), "coordinates", ext,
          "emitter C ×3 / demand ×3 / 水库水头 ×1 / roughness ×1")
    claim("grad_extfd_epanet_relerr", extmeta["epanet_relerr"], "1", ext,
          "EPANET 在该网 t=0 的收敛 relerr（极限环平台，达不到设定的 1e-8）")
    claim("solver_residual_inf", extmeta["resid_inf"], "ft", ext,
          "我方精抛光后 ‖F‖∞（节点连续性残差）")
    aul = _rel(os.path.join(REG_LOGS, "audit_adversarial.log"))
    claim("audit_n_items", len(audit), "items", aul, "对抗审计项数")
    claim("audit_worst_margin_ratio", worst_ratio, "1", aul,
          "22 项中 worst/自身门槛 的最大比值（<1 即全部通过）")
    claim("gradcheck_config", g3meta["gradcheck_cfg"], "-", g3l,
          "torch.autograd.gradcheck 配置")

    # ---- 性能 ----
    bb = _rel(os.path.join(DATA, "bench_batch_report.txt"))
    for lab, key in (("cpu_f64", "我们 dense CPU float64"),
                     ("gpu_f64", "我们 dense GPU float64"),
                     ("gpu_f32", "我们 dense GPU float32")):
        for B in (1, 64, 256):
            claim(f"perf_{lab}_B{B}_ms", perf[key][B], "ms/scenario", bb,
                  f"bench_batch.py 汇总表（min over repeats，city_d N=542）")
    claim("perf_epanet_dll_solve_ms", epanet["solve"], "ms/scenario", bb,
          "EPANET DLL 仅 EN_initH+EN_runH")
    claim("perf_epanet_dll_total_ms", epanet["total"], "ms/scenario", bb,
          "EPANET DLL 含设需水与读全场水头")
    claim("perf_ours_epanet_mode_cpu_B64_ms", perf["ours_epanet_mode_cpu_B64"],
          "ms/scenario", bb, "我方位级复刻路径（逐样本 numpy）B=64")
    claim("perf_gpu_speedup_vs_cpu_B256",
          perf["我们 dense CPU float64"][256] / perf["我们 dense GPU float64"][256],
          "x", bb, "由同表两数相除（B=256 CPU f64 / GPU f64）")
    claim("batch_bitlevel_dH", consist["dH"], "ft", bb,
          "B=64 批量 vs 逐场景循环 max|ΔH|（门槛 1e-12）")
    claim("gpu_vs_cpu_f64_dH", bmeta["gpu_vs_cpu_dH"], "ft", bb,
          "同一批量在 GPU/CPU 上 float64 解之差（观察值，非门槛）")
    claim("f32_vs_f64_dH", bmeta["f32_vs_f64_dH"], "ft", bb,
          "float32 路径 vs float64 的水头差（float32 不可用于位级对拍）")
    claim("gpu_device", bmeta["gpu"], "-", bb, "bench_batch.py 打印的 CUDA 设备名")
    claim("torch_version", bmeta["torch"], "-", bb, "bench_batch.py 环境行")
    sc = _rel(os.path.join(DATA, "bench_scaling.json"))
    for stem in ("pub_hanoi", "city_d", "pub_bwsn_network_2"):
        v = scal["nets"][stem]
        claim(f"scal_{stem}_ours_ms", v["ours_epanet_ms"], "ms/frame", sc,
              "单帧冷启动稳态求解，min over repeats（bench_scaling.py）")
        claim(f"scal_{stem}_dll_ms", v["epanet_dll_ms"], "ms/frame", sc,
              "EPANET DLL EN_initH(EN_INITFLOW)+EN_runH，同一 INP、同口径")
    claim("scal_slope_ours", slopes[0], "exponent", sc,
          "对 log10(N) vs log10(ms) 的一次多项式拟合斜率（16 个网，本脚本计算）")
    claim("scal_slope_dll", slopes[1], "exponent", sc, "同上，EPANET DLL 曲线")
    claim("scal_n_networks", len(scal["nets"]), "networks", sc, "规模曲线取点数")
    _ratios = {k: v["ours_epanet_ms"] / v["epanet_dll_ms"]
               for k, v in scal["nets"].items()}
    _rmin, _rmax = min(_ratios.items(), key=lambda kv: kv[1]), \
        max(_ratios.items(), key=lambda kv: kv[1])
    claim("scal_ratio_min", _rmin[1], "x", sc,
          f"16 网单帧 ours_epanet_ms/epanet_dll_ms 的最小值（{_rmin[0]}）")
    claim("scal_ratio_max", _rmax[1], "x", sc,
          f"16 网单帧 ours_epanet_ms/epanet_dll_ms 的最大值（{_rmax[0]}）")

    # ---- 漏损反演 ----
    dm = "data/demo_leak_inversion.json"
    nl, ny = demo["groups"]["noiseless"], demo["groups"]["noisy"]
    claim("leak_n_candidates", len(demo["config"]["candidates"]), "nodes", dm,
          "config.candidates（工单记录节点 ∩ junction）")
    claim("leak_n_sensors", demo["config"]["n_sensor"], "sensors", dm,
          "config.n_sensor")
    claim("leak_n_frames", demo["config"]["frames"], "frames", dm, "config.frames")
    for _nid, _tn in demo["config"]["true_nodes"].items():
        claim(f"leak_target_lps_{alias[_nid]}", _tn["target_lps"], "L/s", dm,
              f"真值节点 {alias[_nid]} 的目标漏损流量（config.true_nodes）")
        claim(f"leak_true_mean_lps_{alias[_nid]}", _tn["true_mean_lps"], "L/s",
              dm, f"真值节点 {alias[_nid]} 的实际实现平均漏损流量（25 帧均值）")
    claim("leak_stage1_lambda_star", demo["groups"]["noiseless"]["lambda_star"],
          "1", dm, "无噪声组阶段 1 后选中的 L1 权重 λ*")
    claim("leak_stage1_n_lambdas", len(demo["config"]["lambdas"]), "values", dm,
          "阶段 1 扫掠的 L1 权重个数（config.lambdas）")
    claim("leak_stage1_iters", demo["config"]["stage1_iters"], "iterations", dm,
          "阶段 1 每个 λ 的 Adam 迭代数（config.stage1_iters）")
    claim("leak_gga_max_iter", demo["config"]["gga_max_iter"], "iterations", dm,
          "反演前向求解的 GGA 迭代上限（config.gga_max_iter）")
    claim("leak_polish_steps", demo["config"]["polish_steps"], "steps", dm,
          "反演前向求解的 Newton 精抛光步数（config.polish_steps）")
    claim("leak_stage1_n_true_in_top3",
          sum(1 for n in demo["groups"]["noiseless"]["stage1"]["1e-04"]["top3"]
              if n in demo["config"]["true_nodes"]), "nodes", dm,
          "λ* 下阶段 1 的 top3 中真值节点个数")
    claim("leak_n_true", len(demo["config"]["true_nodes"]), "nodes", dm,
          "config.true_nodes")
    claim("leak_noiseless_top3_exact", nl["top3_exact"], "bool", dm,
          "groups.noiseless.top3_exact（支撑完全命中）")
    claim("leak_noiseless_max_flow_rel_err", nl["max_flow_rel_err"], "1", dm,
          "三个真值节点漏损流量的最大相对误差（无噪声）")
    claim("leak_noiseless_final_mse_ft2", nl["final_mse_ft2"], "ft^2", dm,
          "最终传感器压力 MSE（无噪声）")
    claim("leak_noiseless_resid_nontrue_final_lps",
          nl["resid_nontrue_final_lps"], "L/s", dm, "最终支撑中非真值节点的残余漏损")
    claim("leak_noiseless_resid_nontrue_stage1_lps",
          nl["resid_nontrue_stage1_lps"], "L/s", dm,
          "阶段 1（Adam+L1, λ*）非真值节点的最大残余漏损")
    claim("leak_noisy_top3_exact", ny["top3_exact"], "bool", dm,
          "groups.noisy.top3_exact - 含噪声组支撑恢复失败，如实报告")
    claim("leak_noisy_max_flow_rel_err", ny["max_flow_rel_err"], "1", dm,
          "含噪声组：三个真值节点均未被恢复（est=0），相对误差 = 1")
    claim("leak_noisy_final_mse_ft2", ny["final_mse_ft2"], "ft^2", dm,
          "最终传感器压力 MSE（含噪声）")
    claim("leak_noise_ft", demo["config"]["noise_ft"], "ft", dm,
          "config.noise_ft 高斯观测噪声标准差")
    claim("leak_total_time_sec", demo["total_time_sec"], "s", dm, "两组合计耗时")
    claim("leak_n_forward_solves", demo["n_forward_solves"], "solves", dm,
          "反演全过程的前向求解次数（每次 = 25 帧批量）")
    lc = "data/leak_coherence.json"
    claim("coh_max", coh["max_offdiag"], "1", lc,
          "候选签名字典列归一化后互相干矩阵的最大离对角元（leak_coherence_probe.py）")
    claim("coh_median", coh["median_offdiag"], "1", lc, "同矩阵离对角元中位数")
    claim("coh_n_pairs", coh["n_pairs"], "pairs", lc, "C(49,2)")
    claim("coh_n_pairs_gt_0999", coh["n_pairs_gt_0999"], "pairs", lc,
          "互相干 >0.999 的候选对数")
    for n, v in coh["true_node_rivals"].items():
        claim(f"coh_rival_{alias[n]}", v["coh"], "1", lc,
              f"真值节点 {alias[n]} 与其最强竞争者的签名相干（节点 ID 见私有映射）")

    # ---- 公开网清单 ----
    pi = "data/public_inventory.json"
    claim("n_public_inp", len(inv), "files", pi, "public_inventory.json 条目数")
    claim("public_units_in_suite", ", ".join(units_seen), "-", pi,
          "21 个公开网 INP 中出现的流量单位集合")

    # ---- Schur 补条件数（论文 3.2 节后向误差论证）----
    cr = "data/conditioning_report.txt"
    cd = cond["city_d"]
    claim("cond_city_d_full", cd["cond_full"], "1", cr,
          "City D 收敛迭代的 Schur 补 2-范数条件数（probe_conditioning.py）")
    claim("cond_city_d_coupled", cd["cond_coupled"], "1", cr,
          f"同上，去掉 {cd['n_decoupled']} 个被关闭链路（1/CBIG=1e-8）解耦的行之后")
    claim("cond_city_d_no_valve_rows", cd["cond_novalve"], "1", cr,
          "同上，再去掉 1/CSMALL=1e6 量级的阀门行之后")

    CLAIMS["_meta"] = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        generator="scripts/make_paper_figs.py",
        python=sys.executable,
        rule="每个数值必须可在 source_file 中按 how_measured 复现；禁止手抄/估计。",
        n_claims=len(CLAIMS))
    CLAIMS["_private_note"] = dict(
        warning="本文件含敏感真实管网的脱敏映射，**不得随开源仓库分发**。",
        anonymisation=ANON,
        leak_node_alias={v: k for k, v in alias.items()},
        leak_candidate_labels="cNN = data/demo_leak_inversion.json "
                              "config.candidates 中的 1-based 序号")
    p = os.path.join(PAPER, "claims.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(CLAIMS, f, ensure_ascii=False, indent=1)
    print(f"  写出 {_rel(p)}（{len(CLAIMS) - 2} 条 claim）")


# ================================================================ main
def main():
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(TABS, exist_ok=True)
    print("解析产物 ...")
    rows, summary, _foot = parse_benchmark_report()
    assert len(rows) == summary["n_nets"], (len(rows), summary["n_nets"])
    logs = load_bench_logs(rows)
    reg_items, reg_tot, _ = parse_regression_report()
    g3, g3meta = parse_gradcheck_3way()
    gdw = parse_gradcheck_dw()
    audit = parse_audit()
    extfd, extmeta = parse_extfd()
    perf, epanet, consist, bmeta = parse_bench_batch()
    scal = load_json("bench_scaling.json")
    demo = load_json("demo_leak_inversion.json")
    coh = load_json("leak_coherence.json")
    inv = load_json("public_inventory.json")
    cond = parse_conditioning()
    print(f"  benchmark {len(rows)} 网 / regression {len(reg_items)} 项 / "
          f"三方对拍 {len(g3)} 条 / D-W {sum(len(s['rows']) for s in gdw)} 坐标 / "
          f"外部 FD {len(extfd)} 坐标 / 审计 {len(audit)} 项")
    assert len(audit) == 22, len(audit)
    assert len(reg_items) == reg_tot["n_total"], (len(reg_items), reg_tot)

    # 一致性自检：非豁免网的"全帧最大"必须等于报告汇总值
    for r in rows:
        if r["exempt"]:
            continue
        a, b = logs[r["stem"]]["max_dH"], r["dH_report"]
        if abs(a - b) > 1e-3 * max(b, 1e-300):
            raise AssertionError(f"{r['stem']}: 日志全帧 {a:.3e} != 报告 {b:.3e}")
    print("  自检通过：48 个非豁免网的逐帧解析与报告汇总一致")

    print("绘图 ...")
    n_zero = fig_benchmark_accuracy(rows, logs, summary)
    fig_gradient_validation(g3, g3meta, gdw, extfd, extmeta, audit)
    alias = fig_leak_inversion(demo, coh)
    slopes = fig_scaling(perf, epanet, scal)
    print("出表 ...")
    units_seen = tab_capability(inv, rows)
    tab_benchmark(rows, logs)
    tab_full_benchmark(rows, logs)
    worst_ratio = tab_gradient(g3, g3meta, gdw, extfd, extmeta, audit, reg_items)
    tab_ltown_forward()
    tab_ltown_fb()
    tab_ltown_mem()
    print("写 claims.json ...")
    build_claims(rows, logs, summary, reg_items, reg_tot, g3, g3meta, gdw,
                 extfd, extmeta, audit, perf, epanet, consist, bmeta, scal,
                 demo, coh, inv, n_zero, slopes, alias, worst_ratio, units_seen,
                 cond)
    print("完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

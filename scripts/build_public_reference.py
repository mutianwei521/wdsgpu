# -*- coding: utf-8 -*-
"""build_public_reference.py [stem ...] - 公开网参考解生成（任务 D）。

对 networks/public/*.inp 逐网：
  1. dgga.parse.parse_inp → data/reference/pub_<stem>_{net.npz,meta.json}
  2. dgga.reference.build_reference（双精度 EPANET DLL 完整 EPS）→ pub_<stem>_ref.npz
stem 规范化：文件名去扩展、非字母数字折叠为 '_'、小写、前缀 "pub_"。
Pescara.inp 自带 NUL 填充字节（仓库原样），喂 DLL 前清洗到
networks/public/_cleaned/。成功/跳过与原因写 data/public_reference_index.json。
"""

import json
import os
import re
import sys
import time
import traceback

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import parse_inp                    # noqa: E402
from dgga.reference import build_reference          # noqa: E402

PUB_DIR = os.path.join(ROOT, "networks", "public")
CLEAN_DIR = os.path.join(PUB_DIR, "_cleaned")
REF_DIR = os.path.join(ROOT, "data", "reference")
INDEX = os.path.join(ROOT, "data", "public_reference_index.json")


def norm_stem(fname):
    base = os.path.splitext(os.path.basename(fname))[0]
    return "pub_" + re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_").lower()


def clean_inp(path):
    """最小化清洗（parse 与 DLL 喂同一份清洗文件，保证参考解与 Net 同源）：
    1. 去 NUL 字节（Pescara 源文件自带 14006 个 NUL 填充，仓库原样）；
    2. [OPTIONS] PATTERN 引用未定义 pattern → 删该行。语义等价论证：EPANET 对
       未定义默认 pattern 静默回退（input1.c:326-332：DefPat=findpattern=0 →
       demand->Pat 保持 0=常数 1.0）；删行后 DefPatID 退回 DEFPATID="1"
       （input1.c:94），仅当文件不存在名为 "1" 的 pattern 时行为不变 - 存在则拒绝清洗；
       wntr 对未定义 pattern 直接抛错，故必须删行。
    3. [OPTIONS] QUALITY 化学单位字段非法（BWSN1 的 "Chemical TIME"）→ 置
       QUALITY NONE。水质设置不进入水力方程，参考解（纯水力）不受影响。
    4. [COORDINATES] 引用未定义节点（Pescara 把管段 ID 误写进坐标表）→ 删该行。
       坐标纯绘图元数据，不进任何水力量；wntr 对此抛 KeyError 故必须删。
    返回可用路径。"""
    with open(path, "rb") as f:
        data = f.read()
    changed = []
    if b"\x00" in data:
        changed.append(f"移除 {data.count(b'\x00')} 个 NUL 字节")
        data = data.replace(b"\x00", b"")
    text = data.decode("latin-1")
    lines = text.splitlines(keepends=True)
    # 收集 pattern ID 与节点 ID
    pat_ids, node_ids, sec = set(), set(), None
    for ln in lines:
        s0 = ln.strip()
        if not s0 or s0.startswith(";"):
            continue
        if s0.startswith("["):
            sec = s0[1:s0.find("]")].strip().upper()
            continue
        body = s0.split(";", 1)[0].split()
        if not body:
            continue
        if sec == "PATTERNS":
            pat_ids.add(body[0])
        elif sec in ("JUNCTIONS", "RESERVOIRS", "TANKS"):
            node_ids.add(body[0])
    out_lines, sec = [], None
    for ln in lines:
        s0 = ln.strip()
        if s0.startswith("["):
            sec = s0[1:s0.find("]")].strip().upper()
        elif sec == "OPTIONS" and s0 and not s0.startswith(";"):
            tk = s0.split(";", 1)[0].split()
            if tk and tk[0].upper().startswith("PATTERN") and len(tk) >= 2 \
                    and tk[1] not in pat_ids:
                if "1" in pat_ids:
                    raise RuntimeError(
                        f"{path}: PATTERN {tk[1]!r} 未定义且存在 pattern '1'，"
                        f"删行会改变 EPANET 默认 pattern 语义，拒绝清洗")
                changed.append(f"删 OPTIONS 行 {s0!r}（未定义 pattern，"
                               f"EPANET 语义=常数 1.0 不变）")
                continue
            if tk and tk[0].upper().startswith("QUALITY") and len(tk) >= 3 \
                    and tk[1].upper() in ("CHEM", "CHEMICAL") \
                    and tk[2].upper() not in ("MG/L", "UG/L"):
                changed.append(f"OPTIONS {s0!r} → 'Quality NONE'（化学单位非法，"
                               f"水质不进水力方程）")
                out_lines.append(" Quality             NONE\n")
                continue
        elif sec == "COORDINATES" and s0 and not s0.startswith(";"):
            tk = s0.split(";", 1)[0].split()
            if tk and tk[0] not in node_ids:
                changed.append(f"删 [COORDINATES] 行 {s0!r}（未定义节点，纯绘图元数据）")
                continue
        out_lines.append(ln)
    if not changed:
        return path
    os.makedirs(CLEAN_DIR, exist_ok=True)
    out = os.path.join(CLEAN_DIR, os.path.basename(path))
    with open(out, "w", encoding="latin-1", newline="") as f:
        f.write("".join(out_lines))
    for c in changed:
        print(f"  [清洗] {os.path.basename(path)}: {c}")
    return out


def build_tight(argv):
    """对照实验（ky5 豁免同法）：ACCURACY=1e-8（EN_setoption 不受 INP 1e-5 钳位）、
    TRIALS=1000 重跑 EPS，落盘 pub_<stem>_tight_ref.npz。若宽松 Accuracy 下的
    质量残差在收紧后掉到 <1e-5 cfs，即证实其为 EPANET 收敛容差而非装配错误。"""
    from dgga.epanet_ref import Epanet, EN_ACCURACY, EN_TRIALS
    EN_UNBALANCED = 14   # epanet2_enums.h:312 不收敛时的额外迭代（>=0 即 CONTINUE）
    with open(INDEX, "r", encoding="utf-8") as f:
        index = json.load(f)
    for stem in argv:
        rec = index[stem]
        inp = os.path.join(ROOT, rec.get("inp_used", rec["inp"]))
        print(f"[{stem}] 对照实验 ACCURACY=1e-8 TRIALS=1000: {inp}", flush=True)
        t0 = time.time()
        with Epanet(inp) as en:
            en.set_option(EN_ACCURACY, 1e-8)
            en.set_option(EN_TRIALS, 1000)
            # UNBALANCED CONTINUE 10：1e-8 逼近 float64 噪声底，个别帧到不了也
            # 继续跑完 EPS（否则 INP 自带 STOP 的网在首帧就中止）
            en.set_option(EN_UNBALANCED, 10)
            res = en.solve_eps()
        arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(os.path.join(REF_DIR, f"{stem}_tight_ref.npz"), **arrays)
        print(f"  T={len(res['t_sec'])} 帧, iter_max={int(res['iterations'].max())}, "
              f"relerr_max={res['relerr'].max():.2e}, 警告 {len(res['warnings'])} 条, "
              f"{time.time() - t0:.1f}s", flush=True)
    return 0


def main(argv):
    if argv and argv[0] == "--tight":
        return build_tight(argv[1:])
    os.makedirs(REF_DIR, exist_ok=True)
    inps = sorted(f for f in os.listdir(PUB_DIR) if f.lower().endswith(".inp"))
    if argv:
        want = set(a.lower() for a in argv)
        inps = [f for f in inps if norm_stem(f) in want
                or norm_stem(f)[4:] in want or f.lower() in want]
    index = {}
    if os.path.exists(INDEX):
        with open(INDEX, "r", encoding="utf-8") as f:
            index = json.load(f)
    n_ok = n_skip = 0
    for fname in inps:
        stem = norm_stem(fname)
        inp0 = os.path.join(PUB_DIR, fname)
        rec = {"inp": os.path.relpath(inp0, ROOT).replace("\\", "/"), "stem": stem}
        print("=" * 78)
        print(f"[{stem}] {fname}", flush=True)
        t0 = time.time()
        try:
            inp = clean_inp(inp0)
            rec["inp_used"] = os.path.relpath(inp, ROOT).replace("\\", "/")
            net = parse_inp(inp)
            net.save(REF_DIR, stem)
            t1 = time.time()
            print(f"  parse: N={net.N} L={net.L} units={net.meta['flow_units']} "
                  f"headloss={net.meta['headloss']} dur={net.meta['duration_sec']}s "
                  f"({t1 - t0:.1f}s)", flush=True)
            res = build_reference(inp, REF_DIR, stem)
            t2 = time.time()
            T = len(res["t_sec"])
            it_max = int(res["iterations"].max())
            re_max = float(res["relerr"].max())
            nw = len(res["warnings"])
            rec.update(status="ok", N=net.N, L=net.L,
                       units=net.meta["flow_units"], headloss=net.meta["headloss"],
                       frames=T, iter_max=it_max, relerr_max=re_max,
                       n_run_warnings=nw,
                       parse_sec=round(t1 - t0, 2), solve_sec=round(t2 - t1, 2))
            print(f"  ref: T={T} 帧, iter_max={it_max}, relerr_max={re_max:.2e}, "
                  f"运行警告 {nw} 条, 求解耗时 {t2 - t1:.1f}s", flush=True)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            rec.update(status="skip", reason=f"{type(e).__name__}: {e}")
            print(f"  跳过: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=3)
            n_skip += 1
        index[stem] = rec
        with open(INDEX, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=1)
    print("=" * 78)
    print(f"完成: 成功 {n_ok} / 跳过 {n_skip}（索引 {INDEX}）")
    return 0 if n_skip == 0 else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

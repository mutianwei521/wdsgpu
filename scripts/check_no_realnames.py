# -*- coding: utf-8 -*-
"""check_no_realnames.py - 匿名化常态守卫（regression_all 第 ⑫ 项）。

扫描对象：
  1. 当前树全部**跟踪**文件的内容（git ls-files；字节级多编码：ASCII 变体
     小写/首字母大写/全大写 + 中文 UTF-8 与 GBK 两种字节序列）；
  2. 全部跟踪文件的**路径**；
  3. 最近一条提交信息（subject + body）。

禁词表**不入仓库**：从环境变量 HYDROGRAD_NAME_MAP 指向的私有 JSON 映射读（该文件只存在于作者机器上）
（city_d/city_h 的 stem_alias 与 cjk，及顶层 banned_extra）。
该文件不存在时：明确打印 SKIP 及原因并以退出码 3 结束 -
绝不静默 PASS（regression_all 把 rc=3 显示为 SKIP，不算 FAIL 也不算 PASS 数值项）。

任何命中 → 打印文件/位置/变体掩码标签（不回显禁词本身）并以退出码 1 FAIL。
全净 → 打印统计并退出 0。

自证方法（守卫必须能变红）：
  git add 一个含禁词的临时文件（进索引即被 git ls-files 看见）→ 本脚本必须
  FAIL；git rm --cached 后必须回到 PASS。见 data/anon_tree_wip.txt 的实测记录。
"""

import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_PATH = os.environ.get("HYDROGRAD_NAME_MAP", "")


def load_variants():
    if not MAP_PATH or not os.path.exists(MAP_PATH):
        print("check_no_realnames: SKIP - HYDROGRAD_NAME_MAP 未设置或文件不存在，"
              "禁词表不可得（禁词本身从不入仓库）。")
        print("  这不是 PASS：在能拿到本地映射的机器上必须重跑本守卫。")
        sys.exit(3)
    m = json.load(open(MAP_PATH, encoding="utf-8"))
    out = []
    for tag, key in (("A", "city_d"), ("B", "city_h")):
        alias = m[key]["stem_alias"]
        out.append((f"{tag}.lc", alias.lower().encode("ascii")))
        out.append((f"{tag}.Cap", alias.capitalize().encode("ascii")))
        out.append((f"{tag}.UC", alias.upper().encode("ascii")))
        cjk = m[key].get("cjk")
        if cjk:
            out.append((f"{tag}.cjk8", cjk.encode("utf-8")))
            out.append((f"{tag}.cjkGBK", cjk.encode("gbk")))
    for i, w in enumerate(m.get("banned_extra", [])):
        out.append((f"X{i}.lc", w.lower().encode("ascii")))
    return out


def hits_in(data: bytes, variants):
    return {lab: n for lab, pat in variants if (n := data.count(pat))}


def archive_hits(fp: str, variants):
    """压缩归档内部也要扫：gzip/tar（.tgz/.tar.gz）与 zip（.zip/.npz/.xlsx）。
    压缩会让字节级扫描失明 - 实测 deploy 的 .tgz 里藏过 48 处命中。
    解不开时回退为不扫内部（外层原始字节已扫过）。"""
    low = fp.lower()
    out = {}
    try:
        if low.endswith((".tgz", ".tar.gz", ".tar")):
            import tarfile
            with tarfile.open(fp) as t:
                for m in t.getmembers():
                    h = hits_in(m.name.encode("utf-8", "replace"), variants)
                    if m.isfile():
                        fobj = t.extractfile(m)
                        if fobj is not None:
                            for k, v in hits_in(fobj.read(), variants).items():
                                h[k] = h.get(k, 0) + v
                    for k, v in h.items():
                        out[f"{m.name}:{k}"] = v
        elif low.endswith((".zip", ".npz", ".xlsx", ".docx")):
            import zipfile
            with zipfile.ZipFile(fp) as z:
                for nm in z.namelist():
                    h = hits_in(nm.encode("utf-8", "replace"), variants)
                    for k, v in hits_in(z.read(nm), variants).items():
                        h[k] = h.get(k, 0) + v
                    for k, v in h.items():
                        out[f"{nm}:{k}"] = v
    except Exception:
        return {}
    return out


def main():
    variants = load_variants()
    files = [f for f in subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT,
        capture_output=True).stdout.decode("utf-8", "replace").split("\0") if f]
    bad = []
    n_bytes = 0
    for f in files:
        ph = hits_in(f.encode("utf-8"), variants)
        if ph:
            bad.append(("PATH", f, ph))
        fp = os.path.join(ROOT, f)
        try:
            data = open(fp, "rb").read()
        except OSError:
            # 索引里有但工作区缺的文件：读索引里的 blob 再扫
            data = subprocess.run(["git", "cat-file", "blob", f":{f}"],
                                  cwd=ROOT, capture_output=True).stdout
        n_bytes += len(data)
        h = hits_in(data, variants)
        if h:
            bad.append(("CONTENT", f, h))
        ah = archive_hits(fp, variants)
        if ah:
            bad.append(("ARCHIVE", f, ah))
    msg = subprocess.run(["git", "log", "-1", "--format=%s%n%b"], cwd=ROOT,
                         capture_output=True).stdout
    h = hits_in(msg, variants)
    if h:
        bad.append(("COMMIT-MSG", "HEAD", h))
    if bad:
        print(f"check_no_realnames: FAIL - {len(bad)} 处命中：")
        for kind, f, h in bad:
            lab = " ".join(f"{k}={v}" for k, v in sorted(h.items()))
            print(f"  [{kind}] {f}  ({lab})")
        return 1
    print(f"check_no_realnames: PASS - 跟踪文件 {len(files)} 个 "
          f"/ {n_bytes:,d} B 内容 + 全部路径 + 最近一条提交信息，"
          f"{len(variants)} 个禁词变体 0 命中")
    return 0


if __name__ == "__main__":
    sys.exit(main())

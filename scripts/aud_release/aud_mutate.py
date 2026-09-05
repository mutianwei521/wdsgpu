# -*- coding: utf-8 -*-
"""aud_mutate.py - 敌意变异体族（自造，与上游 check_* 内嵌变异体无共享）。

用法：python aud_mutate.py <mutant> <src_dgga_dir> <dst_dir>
把 src_dgga_dir 复制为 dst_dir/dgga 并对 solver.py 打指定变异（锚点必须恰好
命中 1 次，否则退出码 2）。变异体全部是"移植时真会犯"的形态：

  MA_earlyexit   收敛即退出，忽略 stat_change（门 B1 审计 M1 类）
  MB_valve_freq  valvestatus 降频：只在偶数迭代跑（正品每轮迭代 :161）
  MC_ls_every    linkstatus 周期支每轮都跑（正品 CheckFreq=2 节律 :183-187）
  MD_open_case   prvstatus 的 OPEN case 丢 h2>=hset+htol→ACTIVE 转移（:281）
  ME_y_predemand ACTIVE 的 Y 读 nodecoeffs 之前的 Xflow（+d_j 还原扣需水前）
  MF_oneside     ACTIVE 时把 CBIG 误写进一侧非对角（罚函数行单侧装配）
"""
import os
import shutil
import sys

MUTS = {
    "MA_earlyexit": (
        "            done_now = active & conv & ((it > max_iter) | "
        "(~stat_change))",
        "            done_now = active & conv"),
    "MB_valve_freq": (
        "            if self._dense_prv_np:\n"
        "                S_v, vchg = self._prvstatus_batch(S, H, q)",
        "            if self._dense_prv_np and (it % 2 == 0):\n"
        "                S_v, vchg = self._prvstatus_batch(S, H, q)"),
    "MC_ls_every": (
        "            br_per = active & (~conv) & (it <= self.maxcheck) & "
        "(nextcheck == it)",
        "            br_per = active & (~conv)"),
    "MD_open_case": (
        "        c_open = torch.where(neg, CL,\n"
        "                             torch.where(h2 >= hset + htol, AC, OP))"
        "      # :279-282",
        "        c_open = torch.where(neg, CL, OP)"),
    "ME_y_predemand": (
        "                    P, Y, F, A, _ = self._prvcoeffs_batch(\n"
        "                        P, Y, F, Xflow, q, S, A=A)",
        "                    P, Y, F, A, _ = self._prvcoeffs_batch(\n"
        "                        P, Y, F, Xflow + d_j, q, S, A=A)"),
    "MF_oneside": (
        "            a_cols += [-pna, -pna, pna, pna, CBIG * mact]",
        "            a_cols += [-pna - CBIG * mact, -pna, pna, pna, "
        "CBIG * mact]"),
}


def main():
    name, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    old, new = MUTS[name]
    pkg = os.path.join(dst, "dgga")
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    shutil.copytree(src, pkg)
    shutil.rmtree(os.path.join(pkg, "__pycache__"), ignore_errors=True)
    f = os.path.join(pkg, "solver.py")
    with open(f, "r", encoding="utf-8") as fh:
        s = fh.read()
    n = s.count(old)
    if n != 1:
        print("MUTATE ERROR %s 锚点命中 %d 次（应为 1）" % (name, n))
        sys.exit(2)
    with open(f, "w", encoding="utf-8") as fh:
        fh.write(s.replace(old, new))
    print("MUTATE OK %s -> %s" % (name, pkg))


if __name__ == "__main__":
    main()

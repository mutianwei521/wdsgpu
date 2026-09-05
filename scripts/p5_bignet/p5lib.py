# -*- coding: utf-8 -*-
"""P5 大网准入/性能公共库。

铁律零：本轮全程 import HEAD 的**只读副本**（另有研究员在并发改 dgga/solver.py）。
只读副本由 `git archive HEAD dgga | tar -x -C <scratch>` 生成，commit 记在
data/p5_bignet_wip.txt 抬头。设 P5_DGGA_ROOT 环境变量指向该副本的父目录。
"""
import os
import sys

_RO = os.environ.get("P5_DGGA_ROOT")
if _RO:
    # 只读副本必须排在最前，压过工作树里正在被改的 dgga/
    sys.path.insert(0, _RO)

import numpy as np  # noqa: E402


def dgga_provenance():
    """返回实际 import 到的 dgga 包路径（报告里要写明测的是哪一份）。"""
    import dgga
    return os.path.dirname(os.path.abspath(dgga.__file__))


def full_head(net, t_sec=0):
    """定水头节点的完整水头向量 float64[N]（ft），junction 位保持 nan。

    ★ 本轮踩过的坑：net.reservoir_head_ft_at() 只填 reservoir，**tank 位仍是 nan**。
      单帧稳态里 tank 逐位等价于定水头节点（NodeHead[tank]=tank->H0，
      hydraul.c:110 inithyd），必须把 tank_h0 一并给进 res_head_ft，
      否则 solve() 直接报 "res_head_ft 在水库/水池位存在 nan"。
    """
    rh = net.reservoir_head_ft_at(t_sec)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        rh[tn] = np.asarray(net.tank_h0, dtype=np.float64)
    return rh


def base_case(net, t_sec=0):
    """(demand[N] cfs, res_head[N] ft)。必须在 GGASolver(inp_path=...) 构造之后调用
 - 构造会就地把 net.dem_base_cfs 修正为 INP 位级基值。"""
    return net.demand_cfs_at(t_sec), full_head(net, t_sec)


def batch_case(net, B, seed=0, t_sec=0):
    """B 个需水场景（逐节点 U[0.3,1.7] × 全局 U[0.6,1.4]，与 mlds.sample_scenarios
    同分布），定水头不动。返回 (demand[B,N], res_head[B,N])。"""
    d0, rh0 = base_case(net, t_sec)
    rng = np.random.default_rng(seed)
    N = net.N
    nm = rng.uniform(0.3, 1.7, size=(B, N))
    gs = rng.uniform(0.6, 1.4, size=(B, 1))
    return d0[None, :] * nm * gs, np.broadcast_to(rh0, (B, N)).copy()


LINK_NAMES = {0: "CVPIPE", 1: "PIPE", 2: "PUMP", 3: "PRV", 4: "PSV",
              5: "PBV", 6: "FCV", 7: "TCV", 8: "GPV"}

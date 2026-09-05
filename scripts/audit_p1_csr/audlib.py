# -*- coding: utf-8 -*-
"""独立审计公共件：网集合、病态合成网、边界构造、位指纹。不复用上游测试。"""
import copy
import os
import sys

import numpy as np
import torch

import os as _os  # noqa: E402
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from dgga.parse import Net                       # noqa: E402
from dgga.solver import GGASolver                # noqa: E402

REF = os.path.join(ROOT, "data", "reference")

# dense 模式可构造的全部参考网（由 aud_00_scan 实测得出），按 Nj 升序
SMALL = ["pub_net1", "pub_anytown", "pub_anytown_wntr", "pub_hanoi", "pub_net2",
         "pub_fossolo_poly1", "rand_main_0009", "pub_pescara", "EXA6",
         "rand_small_0001", "pub_net3", "pub_modena", "ky3"]
BIG = ["ky5", "city_d", "city_d_emit", "city_h", "pub_ky4"]


# ---------------------------------------------------------------- 合成病态网
def _grow_node(net, kind=0, elev=100.0):
    """追加一个节点（默认 junction），返回新索引。"""
    net.node_id = list(net.node_id) + [f"AUD_N{len(net.node_id)}"]
    net.node_type = np.append(net.node_type, np.int8(kind))
    net.elev_ft = np.append(net.elev_ft, float(elev))
    net.node_ke = np.append(net.node_ke, 0.0)
    net.res_head_pat = np.append(net.res_head_pat, np.int32(-1))
    return len(net.node_id) - 1


def _clone_link(net, src, n1, n2):
    """复制链路 src 的全部属性，接到 (n1,n2)。"""
    net.link_id = list(net.link_id) + [f"AUD_L{len(net.link_id)}"]
    for k in ("link_type", "diam_ft", "len_ft", "roughness", "km_int", "r_hw",
              "init_status", "valve_setting_user"):
        a = getattr(net, k)
        setattr(net, k, np.append(a, a[src]))
    net.link_n1 = np.append(net.link_n1, np.int32(n1))
    net.link_n2 = np.append(net.link_n2, np.int32(n2))
    return len(net.link_id) - 1


def synth(stem, iso=0, par=0, selfloop=0, iso_ke=0.0, seed=1):
    """从参考网派生病态网：iso 个孤立 junction、par 条并联管（复制现有管，
    形成同一对节点上的多重边）、selfloop 条自环管（n1==n2）。"""
    net = copy.deepcopy(Net.load(REF, stem))
    rng = np.random.default_rng(seed)
    nt = np.asarray(net.node_type)
    juncs = np.where(nt == 0)[0]
    pipes = np.where(np.asarray(net.link_type) == 1)[0]
    # 并联管：挑既有管道，复制 par 条（第一条挑两端都是 junction 的，最能撞对角+非对角）
    both = [p for p in pipes if nt[net.link_n1[p]] == 0 and nt[net.link_n2[p]] == 0]
    for i in range(par):
        s = int(both[i % len(both)])
        _clone_link(net, s, int(net.link_n1[s]), int(net.link_n2[s]))
    # 自环：挑一个 junction，n1==n2
    for i in range(selfloop):
        s = int(both[i % len(both)])
        j = int(net.link_n1[s])
        _clone_link(net, s, j, j)
    # 孤立 junction（可选挂 emitter，否则 A[j,j]=0 → Cholesky 必失败）
    iso_nodes = []
    for i in range(iso):
        j = _grow_node(net, 0, elev=float(rng.uniform(50, 150)))
        iso_nodes.append(j)
        if iso_ke > 0:
            net.node_ke[j] = iso_ke
    net.__post_init__()
    return net, iso_nodes


# ---------------------------------------------------------------- 边界
def make_solver(net):
    try:
        return GGASolver(net, mode="dense")
    except NotImplementedError:
        return GGASolver(net, mode="dense", dense_tank_bound_check=False)


def boundary(net, stem, t_sec=None, tank_head=None):
    """(d, rh)。水池头：给了 tank_head 就用，否则取 elev 区间中点，避免贴边。"""
    if t_sec is None:
        t_sec = 0
    d = np.asarray(net.demand_cfs_at(t_sec), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(t_sec), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        h = np.asarray(tank_head) if tank_head is not None else None
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        base = h[tn] if h is not None else 0.5 * (net.tank_hmin + net.tank_hmax)
        rh[tn] = np.clip(base, lo, hi)
    return d, rh


def batchify(d, rh, B, seed, dtype=torch.float64):
    g = np.random.default_rng(seed)
    D = d[None, :] * g.uniform(0.6, 1.4, (B, 1)) * g.uniform(0.75, 1.25, (B, d.size))
    R = rh[None, :] + g.uniform(-2.0, 2.0, (B, rh.size))
    R = np.where(np.isnan(rh)[None, :], np.nan, R)
    return (torch.as_tensor(D, dtype=dtype), torch.as_tensor(R, dtype=dtype))


# ---------------------------------------------------------------- 位指纹
_MIX = 0x9E3779B97F4A7C15 - (1 << 64)


def bitfp(t):
    """f64 张量的位级指纹：按位重解释为 int64，混入位置后求和（回绕）。
    任一 bit 变化都会改变指纹（碰撞概率 ~2^-64）。"""
    it = torch.int32 if t.dtype == torch.float32 else torch.int64
    b = t.detach().contiguous().reshape(-1).view(it).to(torch.int64)
    idx = torch.arange(b.numel(), dtype=torch.int64, device=b.device)
    return int((b ^ (idx * _MIX)).sum().item())


def maxabs(a, b):
    return float((a.detach().double() - b.detach().double()).abs().max().item())

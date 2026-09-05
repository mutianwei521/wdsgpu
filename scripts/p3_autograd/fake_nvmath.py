# -*- coding: utf-8 -*-
"""本机（无 nvmath）用的 cuDSS 替身：把 nvmath.sparse.advanced 塞进 sys.modules。

**只用于本机跑 P3 的逻辑/梯度自检**，不参与任何性能或数值结论 - 集群上跑的
是真 cuDSS。替身逐条复刻我们依赖的那几条语义（其余一概不实现）：
  · DirectSolver(a_list, b_list, options) 记住**这些对象**（不拷贝）；
  · plan() 只做符号层的事（这里什么也不做）；
  · factorize() 在**调用那一刻**把 a_list 的值快照成分解（dense LU）；
  · solve() 用**那一次**分解 + **当前** b_list 的值回代；
  · free() 幂等。
关键点是 factorize/solve 的时序：如果 P3 的反向错误地复用了一次已被顶掉的
分解，替身会给出**明显错误**的结果（真 cuDSS 也一样），所以这套自检对
"复用判据写错了"是有杀伤力的，不是走过场。
"""
import sys
import types

import torch


class DirectSolverOptions:
    def __init__(self, sparse_system_type=None):
        self.sparse_system_type = sparse_system_type


class DirectSolverMatrixType:
    SPD = "SPD"
    GENERAL = "GENERAL"


class DirectSolver:
    def __init__(self, a, b, options=None, **kw):
        self.a, self.b = list(a), list(b)
        self.options = options
        self._lu = None
        self._alive = True

    def plan(self, **kw):
        self._planned = True

    def factorize(self, **kw):
        if not getattr(self, "_planned", False):
            raise RuntimeError("Factorization cannot be performed before plan() "
                               "has been called.")
        self._lu = [torch.linalg.lu_factor(x.to_dense()) for x in self.a]

    def solve(self, **kw):
        if self._lu is None:
            raise RuntimeError("solve() before factorize()")
        return [torch.linalg.lu_solve(lu, piv, self.b[i].unsqueeze(-1)).squeeze(-1)
                for i, (lu, piv) in enumerate(self._lu)]

    def free(self):
        self._lu = None
        self._alive = False


def install():
    """把替身注册进 sys.modules（幂等）。"""
    if "nvmath.sparse.advanced" in sys.modules:
        return
    root = types.ModuleType("nvmath")
    sp = types.ModuleType("nvmath.sparse")
    adv = types.ModuleType("nvmath.sparse.advanced")
    adv.DirectSolver = DirectSolver
    adv.DirectSolverOptions = DirectSolverOptions
    adv.DirectSolverMatrixType = DirectSolverMatrixType
    sp.advanced = adv
    root.sparse = sp
    root.__version__ = "FAKE-0"
    sys.modules["nvmath"] = root
    sys.modules["nvmath.sparse"] = sp
    sys.modules["nvmath.sparse.advanced"] = adv

# -*- coding: utf-8 -*-
"""dgga.smatrix - EPANET 2.2 稀疏矩阵机制的忠实移植（阶段 B）。

目的：city_d 对拍硬门槛 max|ΔH|<1e-6 ft。系统含 1/CSMALL=1e6（TCV 低阻支）与
RQtol 钳位 1e7 级对角，条件数 ~1e9，任何 float64 求解器的前向误差 ~eps·κ·‖H‖≈4e-6 ft。
要落在 1e-6 内只能复刻 EPANET 自身的运算次序：MMD 重排（genmmd.c 的整数算法）
+ 符号分解（smatrix.c factorize）+ 数值 Cholesky（smatrix.c linsolve，George & Liu
GSFCT/GSSLV）。本文件逐行移植：

- genmmd/mmdint/mmdelm/mmdupd/mmdnum：genmmd.c:79-1000（f2c 翻译版，纯整数，
  goto 改写为 while/for，语义逐分支核对）。
- localadjlists/paralink/xparalinks：smatrix.c:220-297（前插链表 → 遍历序为
  链路索引降序；平行管共享 Ndx 槽位）。
- reordernodes 的 MMD 输入邻接：smatrix.c:393-408（仅 junction-junction 边，
  遍历序与链表一致；delta=-1，maxint=INT_MAX）。
- factorize/growlist/newlink/addlink：smatrix.c:428-595（消元图填充，
  Ncoeffs 从 Nlinks 起增）。
- storesparse/sortsparse：smatrix.c:598-697（列内行号升序 = 双转置排序结果）。
- linsolve：smatrix.c:729-871（数值分解+前代回代，运算次序逐句照抄 - 这是
  与 EPANET 位级一致的关键）。

全部 1 基索引，与 C 源一致。
"""

import math

import numpy as np

INT_MAX = 2147483647  # limits.h INT_MAX（smatrix.c:360 maxint = INT_MAX）


# ======================================================================
# genmmd.c 移植（纯整数；数组 1 基，传入的 xadj/adjncy 为 python list）
# ======================================================================
def _mmdint(neqns, xadj, adjncy, dhead, dforw, dbakw, qsize, llist, marker):
    """genmmd.c:261-307 mmdint_：度双向链表初始化。"""
    for node in range(1, neqns + 1):
        dhead[node] = 0
        qsize[node] = 1
        marker[node] = 0
        llist[node] = 0
    for node in range(1, neqns + 1):
        ndeg = xadj[node + 1] - xadj[node] + 1     # genmmd.c:295（度+1）
        fnode = dhead[ndeg]
        dforw[node] = fnode
        dhead[ndeg] = node
        if fnode > 0:
            dbakw[fnode] = node
        dbakw[node] = -ndeg


def _mmdelm(mdnode, xadj, adjncy, dhead, dforw, dbakw, qsize, llist,
            marker, maxint, tag):
    """genmmd.c:339-549 mmdelm_：消元 mdnode 并做商图变换（adjncy 就地改写）。"""
    marker[mdnode] = tag
    istrt = xadj[mdnode]
    istop = xadj[mdnode + 1] - 1
    elmnt = 0
    rloc = istrt
    rlmt = istop
    # 找可达集（genmmd.c:377-398）
    for i in range(istrt, istop + 1):
        nabor = adjncy[i]
        if nabor == 0:
            break                                   # goto L300
        if marker[nabor] >= tag:
            continue                                # L200
        marker[nabor] = tag
        if dforw[nabor] < 0:                        # 已消元 → 挂元素链（L100）
            llist[nabor] = elmnt
            elmnt = nabor
        else:
            adjncy[rloc] = nabor
            rloc += 1
    # L300：合并来自广义元素的可达节点（genmmd.c:399-447）
    while elmnt > 0:
        adjncy[rlmt] = -elmnt
        link = elmnt
        restart = True
        while restart:                              # L400
            restart = False
            jstrt = xadj[link]
            jstop = xadj[link + 1] - 1
            for j in range(jstrt, jstop + 1):
                node = adjncy[j]
                link = -node
                if node < 0:
                    restart = True                  # goto L400
                    break
                if node == 0:
                    break                           # goto L900
                # L500
                if marker[node] >= tag or dforw[node] < 0:
                    continue                        # L800
                marker[node] = tag
                # L600：必要时借用已消元节点的存储
                while rloc >= rlmt:
                    link2 = -adjncy[rlmt]
                    rloc = xadj[link2]
                    rlmt = xadj[link2 + 1] - 1
                adjncy[rloc] = node                 # L700
                rloc += 1
        elmnt = llist[elmnt]                        # L900
    if rloc <= rlmt:                                # L1000
        adjncy[rloc] = 0
    # 对可达集中每个节点做度结构调整（genmmd.c:455-547）
    link = mdnode
    restart = True
    while restart:                                  # L1100
        restart = False
        istrt = xadj[link]
        istop = xadj[link + 1] - 1
        for i in range(istrt, istop + 1):
            rnode = adjncy[i]
            link = -rnode
            if rnode < 0:
                restart = True                      # goto L1100
                break
            if rnode == 0:
                return                              # goto L1800
            # L1200：从度链表移除 rnode
            pvnode = dbakw[rnode]
            if pvnode != 0 and pvnode != -maxint:
                nxnode = dforw[rnode]
                if nxnode > 0:
                    dbakw[nxnode] = pvnode
                if pvnode > 0:
                    dforw[pvnode] = nxnode
                npv = -pvnode
                if pvnode < 0:
                    dhead[npv] = nxnode
            # L1300：清除 rnode 的失效商图邻居
            jstrt = xadj[rnode]
            jstop = xadj[rnode + 1] - 1
            xqnbr = jstrt
            for j in range(jstrt, jstop + 1):
                nabor = adjncy[j]
                if nabor == 0:
                    break                           # L1500
                if marker[nabor] >= tag:
                    continue                        # L1400
                adjncy[xqnbr] = nabor
                xqnbr += 1
            # L1500
            nqnbrs = xqnbr - jstrt
            if nqnbrs <= 0:
                # 与 mdnode 合并为超节点（genmmd.c:524-529）
                qsize[mdnode] += qsize[rnode]
                qsize[rnode] = 0
                marker[rnode] = maxint
                dforw[rnode] = -mdnode
                dbakw[rnode] = -maxint
            else:
                # L1600：标记 rnode 待更新度，把 mdnode 加为其邻居
                dforw[rnode] = nqnbrs + 1
                dbakw[rnode] = 0
                adjncy[xqnbr] = mdnode
                xqnbr += 1
                if xqnbr <= jstop:
                    adjncy[xqnbr] = 0
            # L1700 继续
    # L1800（restart 循环自然结束等价 return）


def _mmdupd(ehead, neqns, xadj, adjncy, delta, mdeg, dhead, dforw, dbakw,
            qsize, llist, marker, maxint, tag):
    """genmmd.c:582-885 mmdupd_：多重消元后的度更新。返回 (mdeg, tag)。"""
    mdeg0 = mdeg + delta
    elmnt = ehead
    while True:                                     # L100
        if elmnt <= 0:
            return mdeg, tag
        mtag = tag + mdeg0
        if mtag >= maxint:
            tag = 1
            for i in range(1, neqns + 1):
                if marker[i] < maxint:
                    marker[i] = 0
            mtag = tag + mdeg0
        # L300：把 elmnt 中的节点分成 q2（恰两邻）/qx（多邻）两链表，算 deg0
        q2head = 0
        qxhead = 0
        deg0 = 0
        link = elmnt
        restart = True
        while restart:                              # L400
            restart = False
            istrt = xadj[link]
            istop = xadj[link + 1] - 1
            for i in range(istrt, istop + 1):
                enode = adjncy[i]
                link = -enode
                if enode < 0:
                    restart = True                  # goto L400
                    break
                if enode == 0:
                    break                           # goto L800
                # L500
                if qsize[enode] == 0:
                    continue                        # L700
                deg0 += qsize[enode]
                marker[enode] = mtag
                if dbakw[enode] != 0:
                    continue                        # 不需要度更新
                if dforw[enode] == 2:
                    llist[enode] = q2head           # L600
                    q2head = enode
                else:
                    llist[enode] = qxhead
                    qxhead = enode
        # L800：先处理 q2 链，再处理 qx 链
        enode = q2head
        iq2 = 1
        while True:
            if enode <= 0:                          # L900/L1600 链尾
                if iq2 == 1:
                    enode = qxhead                  # L1500
                    iq2 = 0
                    continue
                break                               # L2300
            if dbakw[enode] != 0:                   # 无需更新 → L2200
                enode = llist[enode]
                continue
            tag += 1
            deg = deg0
            if iq2 == 1:
                # q2 情形（genmmd.c:699-778）：找另一个相邻元素
                istrt = xadj[enode]
                nabor = adjncy[istrt]
                if nabor == elmnt:
                    nabor = adjncy[istrt + 1]
                link = nabor
                if dforw[nabor] >= 0:
                    deg += qsize[nabor]             # 未消元
                else:
                    # L1000：遍历第二个元素中的节点
                    restart = True
                    while restart:
                        restart = False
                        istrt2 = xadj[link]
                        istop2 = xadj[link + 1] - 1
                        for i in range(istrt2, istop2 + 1):
                            node = adjncy[i]
                            link = -node
                            if node == enode:
                                continue            # L1400
                            if node < 0:
                                restart = True      # goto L1000
                                break
                            if node == 0:
                                break               # goto L2100
                            # L1100
                            if qsize[node] == 0:
                                continue
                            if marker[node] < tag:
                                marker[node] = tag  # 首次遇到
                                deg += qsize[node]
                                continue
                            # L1200：与 enode 不可区分或被支配
                            if dbakw[node] != 0:
                                continue
                            if dforw[node] == 2:
                                qsize[enode] += qsize[node]
                                qsize[node] = 0
                                marker[node] = maxint
                                dforw[node] = -enode
                                dbakw[node] = -maxint
                            else:                   # L1300 outmatched
                                if dbakw[node] == 0:
                                    dbakw[node] = -maxint
            else:
                # qx 情形（genmmd.c:779-851）
                istrt = xadj[enode]
                istop = xadj[enode + 1] - 1
                for i in range(istrt, istop + 1):
                    nabor = adjncy[i]
                    if nabor == 0:
                        break                       # goto L2100
                    if marker[nabor] >= tag:
                        continue                    # L2000
                    marker[nabor] = tag
                    link = nabor
                    if dforw[nabor] >= 0:
                        deg += qsize[nabor]         # 未消元
                        continue
                    # L1700：已消元 → 计入该元素中未标记节点
                    restart = True
                    while restart:
                        restart = False
                        jstrt = xadj[link]
                        jstop = xadj[link + 1] - 1
                        for j in range(jstrt, jstop + 1):
                            node = adjncy[j]
                            link = -node
                            if node < 0:
                                restart = True      # goto L1700
                                break
                            if node == 0:
                                break               # goto L2000
                            if marker[node] >= tag:
                                continue            # L1900
                            marker[node] = tag
                            deg += qsize[node]
            # L2100：更新 enode 外部度并入度链表
            deg = deg - qsize[enode] + 1
            fnode = dhead[deg]
            dforw[enode] = fnode
            dbakw[enode] = -deg
            if fnode > 0:
                dbakw[fnode] = enode
            dhead[deg] = enode
            if deg < mdeg:
                mdeg = deg
            # L2200
            enode = llist[enode]
        # L2300
        tag = mtag
        elmnt = llist[elmnt]


def _mmdnum(neqns, perm, invp, qsize):
    """genmmd.c:916-1000 mmdnum_：产出最终 perm/invp。"""
    for node in range(1, neqns + 1):
        nqsize = qsize[node]
        if nqsize <= 0:
            perm[node] = invp[node]
        if nqsize > 0:
            perm[node] = -invp[node]
    for node in range(1, neqns + 1):
        if perm[node] > 0:
            continue                                # L500
        father = node
        while perm[father] <= 0:                    # L200
            father = -perm[father]
        root = father                               # L300
        num = perm[root] + 1
        invp[node] = -num
        perm[root] = num
        father = node                               # L400：缩短合并树
        while True:
            nextf = -perm[father]
            if nextf <= 0:
                break
            perm[father] = -root
            father = nextf
    for node in range(1, neqns + 1):                # L600
        num = -invp[node]
        invp[node] = num
        perm[num] = node


def genmmd(neqns, xadj, adjncy, delta=-1, maxint=INT_MAX):
    """genmmd.c:79-233 genmmd：多重最小度重排。

    输入 1 基 xadj(len neqns+2)/adjncy（python list，adjncy 会被就地破坏）。
    返回 (invp, perm)：invp[node]=新行号（EPANET 的 Row），perm[row]=节点（Order）。
    """
    invp = [0] * (neqns + 1)     # dforw / 最终 Row
    perm = [0] * (neqns + 1)     # dbakw / 最终 Order
    dhead = [0] * (neqns + 2)    # 度链表头（mmdint 的 ndeg=度+1 ≤ neqns+1）
    qsize = [0] * (neqns + 1)
    llist = [0] * (neqns + 1)
    marker = [0] * (neqns + 1)
    if neqns <= 0:
        return invp, perm
    nofsub = 0
    _mmdint(neqns, xadj, adjncy, dhead, invp, perm, qsize, llist, marker)
    num = 1
    # 消去孤立节点（度链表 1；genmmd.c:123-133）
    nextmd = dhead[1]
    while nextmd > 0:
        mdnode = nextmd
        nextmd = invp[mdnode]
        marker[mdnode] = maxint
        invp[mdnode] = -num
        num += 1
    # 主循环（genmmd.c:135-224）
    if num <= neqns:
        tag = 1
        dhead[1] = 0
        mdeg = 2
        finished = False
        while not finished:
            while dhead[mdeg] <= 0:                 # L300
                mdeg += 1
            mdlmt = mdeg + delta                    # L400
            ehead = 0
            while True:                             # L500
                mdnode = dhead[mdeg]
                while mdnode <= 0:
                    mdeg += 1
                    if mdeg > mdlmt:
                        break                       # goto L900
                    mdnode = dhead[mdeg]
                if mdnode <= 0:
                    break                           # → L900
                # L600：从度结构中移除 mdnode
                nextmd = invp[mdnode]
                dhead[mdeg] = nextmd
                if nextmd > 0:
                    perm[nextmd] = -mdeg
                invp[mdnode] = -num
                nofsub += mdeg + qsize[mdnode] - 2
                if num + qsize[mdnode] > neqns:
                    finished = True                 # goto L1000
                    break
                tag += 1
                if tag >= maxint:
                    tag = 1
                    for i in range(1, neqns + 1):
                        if marker[i] < maxint:
                            marker[i] = 0
                _mmdelm(mdnode, xadj, adjncy, dhead, invp, perm, qsize,
                        llist, marker, maxint, tag)
                num += qsize[mdnode]
                llist[mdnode] = ehead
                ehead = mdnode
                if delta >= 0:
                    continue                        # goto L500
                break                               # delta<0 → L900
            if finished:
                break
            # L900：度更新
            if num > neqns:
                break                               # goto L1000
            mdeg, tag = _mmdupd(ehead, neqns, xadj, adjncy, delta, mdeg,
                                dhead, invp, perm, qsize, llist, marker,
                                maxint, tag)
    _mmdnum(neqns, perm, invp, qsize)               # L1000
    return invp, perm


# ======================================================================
# smatrix.c 管线：邻接表 → MMD → 符号分解 → 稀疏存储 → 数值 linsolve
# ======================================================================
class EpanetSmatrix:
    """复刻 createsparse（smatrix.c:77-127）。

    输入（全部 1 基 EPANET 编号）：
      nnodes, njuncs, links = [(n1, n2), ...]（EPANET 链路顺序）
    产出：Row[node]、Ndx[link]、XLNZ/NZSUB/LNZ（列内已升序）、Ncoeffs。
    """

    def __init__(self, nnodes, njuncs, links):
        self.nnodes = nnodes
        self.njuncs = njuncs
        nlinks = len(links)
        self.nlinks = nlinks

        # ---- localadjlists（smatrix.c:220-268）：前插链表 + 平行管剔除 ----
        # adj[node] = [(nbr, slot), ...]，遍历序 = 前插序（链路索引降序）
        adj = [[] for _ in range(nnodes + 1)]
        Ndx = [0] * (nlinks + 1)
        for k1 in range(1, nlinks + 1):
            i, j = links[k1 - 1]
            # paralink（smatrix.c:271-297）：Adjlist[i] 中已有 node==j → 平行
            pmark = 0
            for (nbr, slot) in adj[i]:
                if nbr == j:
                    Ndx[k1] = slot                  # 共享首条链路的槽位
                    pmark = 1
                    break
            if not pmark:
                Ndx[k1] = k1
            # 平行管以 node=0 入链后被 xparalinks 删除 → 等价于不入链
            if not pmark:
                adj[i].insert(0, (j, k1))
                adj[j].insert(0, (i, k1))
        self.Ndx = Ndx

        # ---- reordernodes（smatrix.c:345-424）----
        # 默认序（含 tank：Row=Order=自身）
        Row = list(range(nnodes + 1))
        Order = list(range(nnodes + 1))
        # MMD 输入邻接：仅 junction-junction 边，遍历序与链表一致（:398-408）
        xadj = [0] * (njuncs + 2)
        adjncy = [0]                                # 1 基占位
        xadj[1] = 1
        m = 1
        for k in range(1, njuncs + 1):
            for (nbr, _slot) in adj[k]:
                if 0 < nbr <= njuncs:
                    adjncy.append(nbr)
                    m += 1
            xadj[k + 1] = m
        invp, perm = genmmd(njuncs, xadj, adjncy, delta=-1, maxint=INT_MAX)
        for k in range(1, njuncs + 1):
            Row[k] = invp[k]
            Order[k] = perm[k]
        self.Row = Row
        self.Order = Order

        # ---- factorize/growlist/newlink（smatrix.c:428-554）：符号填充 ----
        Ncoeffs = nlinks                            # smatrix.c:109
        degree = [0] * (nnodes + 1)
        for k in range(1, njuncs + 1):              # :452-458（含 tank 邻居计数）
            degree[k] = sum(1 for (nbr, _s) in adj[k] if nbr > 0)
        adjset = [set(nbr for (nbr, _s) in a) for a in adj]  # linked() 加速
        for kk in range(1, njuncs + 1):             # :463-472
            knode = Order[kk]
            # growlist（:478-509）
            entries = adj[knode]                    # knode 自身链表在本轮不变
            for idx in range(len(entries)):
                node = entries[idx][0]
                if node > 0 and degree[node] > 0:
                    degree[node] -= 1
                    # newlink（:512-554）：与后续表项两两连边
                    inode = node
                    for jdx in range(idx + 1, len(entries)):
                        jnode = entries[jdx][0]
                        if jnode > 0 and degree[jnode] > 0:
                            if jnode not in adjset[inode]:   # linked()
                                Ncoeffs += 1
                                adj[inode].insert(0, (jnode, Ncoeffs))
                                adj[jnode].insert(0, (inode, Ncoeffs))
                                adjset[inode].add(jnode)
                                adjset[jnode].add(inode)
                                degree[inode] += 1
                                degree[jnode] += 1
            degree[knode] = 0
        self.Ncoeffs = Ncoeffs

        # ---- storesparse + sortsparse（smatrix.c:598-697）----
        # 列 i 的下三角非零行号（升序）与 Aij 槽位
        XLNZ = [0] * (njuncs + 2)
        NZSUB = [0]
        LNZ = [0]
        XLNZ[1] = 1
        kpos = 0
        for i in range(1, njuncs + 1):
            ii = Order[i]
            colent = []
            for (nbr, slot) in adj[ii]:
                if nbr == 0:
                    continue
                jrow = Row[nbr]
                if i < jrow <= njuncs:
                    colent.append((jrow, slot))
            colent.sort()                           # sortsparse：列内行号升序
            for (jrow, slot) in colent:
                kpos += 1
                NZSUB.append(jrow)
                LNZ.append(slot)
            XLNZ[i + 1] = XLNZ[i] + len(colent)
        self.XLNZ = XLNZ
        self.NZSUB = NZSUB
        self.LNZ = LNZ

    # ------------------------------------------------------------------
    def linsolve(self, Aii, Aij, B):
        """smatrix.c:729-871 linsolve 逐句移植（数值运算次序 = 与 EPANET 位级一致）。

        Aii[1..n]、B[1..n]（行空间）、Aij[1..Ncoeffs] 均为 python list，就地改写；
        返回 0 或病态行号。解在 B 中。
        """
        n = self.njuncs
        XLNZ = self.XLNZ
        NZSUB = self.NZSUB
        LNZ = self.LNZ
        temp = [0.0] * (n + 1)
        link = [0] * (n + 1)
        first = [0] * (n + 1)

        # 数值分解 A → L（GSFCT）
        for j in range(1, n + 1):
            diagj = 0.0
            newk = link[j]
            k = newk
            while k != 0:
                newk = link[k]
                kfirst = first[k]
                ljk = Aij[LNZ[kfirst]]
                diagj += ljk * ljk                  # smatrix.c:787
                istrt = kfirst + 1
                istop = XLNZ[k + 1] - 1
                if istop >= istrt:
                    first[k] = istrt
                    isub = NZSUB[istrt]
                    link[k] = link[isub]
                    link[isub] = k
                    for i in range(istrt, istop + 1):
                        isub = NZSUB[i]
                        temp[isub] += Aij[LNZ[i]] * ljk   # :804
                k = newk
            diagj = Aii[j] - diagj                  # :812
            if diagj <= 0.0:
                return j                            # 病态（:813-816）
            diagj = math.sqrt(diagj)                # sqrt（:817；UCRT sqrt=硬件精确舍入）
            Aii[j] = diagj
            istrt = XLNZ[j]
            istop = XLNZ[j + 1] - 1
            if istop >= istrt:
                first[j] = istrt
                isub = NZSUB[istrt]
                link[j] = link[isub]
                link[isub] = j
                for i in range(istrt, istop + 1):
                    isub = NZSUB[i]
                    bj = (Aij[LNZ[i]] - temp[isub]) / diagj   # :830
                    Aij[LNZ[i]] = bj
                    temp[isub] = 0.0

        # 前代（:838-852）
        for j in range(1, n + 1):
            bj = B[j] / Aii[j]
            B[j] = bj
            istrt = XLNZ[j]
            istop = XLNZ[j + 1] - 1
            if istop >= istrt:
                for i in range(istrt, istop + 1):
                    isub = NZSUB[i]
                    B[isub] -= Aij[LNZ[i]] * bj

        # 回代（:855-869）
        for j in range(n, 0, -1):
            bj = B[j]
            istrt = XLNZ[j]
            istop = XLNZ[j + 1] - 1
            if istop >= istrt:
                for i in range(istrt, istop + 1):
                    isub = NZSUB[i]
                    bj -= Aij[LNZ[i]] * B[isub]
            B[j] = bj / Aii[j]
        return 0

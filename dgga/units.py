# -*- coding: utf-8 -*-
"""dgga.units - EPANET 单位换算常数与 UCF 类（契约 chain-1）。

换算常数逐字抄自 ref/EPANET2.2-2.2.0/SRC_engines/types.h 第 68-83 行
（"Flow units conversion factors" 段），数值一个字符都不改。
方向约定与 EPANET 源码一致（units.c convertunits）：
    用户值 = 内部值 × Ucf   ⇔   内部值 = 用户值 ÷ Ucf
内部单位恒为：长度/水头 ft，流量 cfs。
"""

# ---- 以下常数逐字抄自 types.h（行号见行尾注释）----
GPMperCFS = 448.831    # types.h:68  #define GPMperCFS   448.831
AFDperCFS = 1.9837     # types.h:69  #define AFDperCFS   1.9837
MGDperCFS = 0.64632    # types.h:70  #define MGDperCFS   0.64632
IMGDperCFS = 0.5382    # types.h:71  #define IMGDperCFS  0.5382
LPSperCFS = 28.317     # types.h:72  #define LPSperCFS   28.317
LPMperCFS = 1699.0     # types.h:73  #define LPMperCFS   1699.0
CMHperCFS = 101.94     # types.h:74  #define CMHperCFS   101.94
CMDperCFS = 2446.6     # types.h:75  #define CMDperCFS   2446.6
MLDperCFS = 2.4466     # types.h:76  #define MLDperCFS   2.4466
M3perFT3 = 0.028317    # types.h:77  #define M3perFT3    0.028317
LperFT3 = 28.317       # types.h:78  #define LperFT3     28.317
MperFT = 0.3048        # types.h:79  #define MperFT      0.3048
PSIperFT = 0.4333      # types.h:80  #define PSIperFT    0.4333
KPAperPSI = 6.895      # types.h:81  #define KPAperPSI   6.895
KWperHP = 0.7457       # types.h:82  #define KWperHP     0.7457
SECperDAY = 86400      # types.h:83  #define SECperDAY   86400


# ---- 流量单位全表（initunits input1.c:443-448 SI 支 / :469-474 US 支）----
# SI 判定照抄 input3.c 单位段 → input1.c:255-265：LPS/LPM/MLD/CMH/CMD 为 SI，
# 其余（CFS/GPM/MGD/IMGD/AFD）为 US。
SI_FLOW_UNITS = ("LPS", "LPM", "MLD", "CMH", "CMD")     # input1.c:256-261
US_FLOW_UNITS = ("CFS", "GPM", "MGD", "IMGD", "AFD")    # input1.c:263-264
FLOW_UCF = {
    # US（input1.c:470-474）：qcf=1.0，GPM/MGD/IMGD/AFD 覆盖
    "CFS": 1.0,          # input1.c:470  qcf = 1.0
    "GPM": GPMperCFS,    # input1.c:471
    "MGD": MGDperCFS,    # input1.c:472
    "IMGD": IMGDperCFS,  # input1.c:473
    "AFD": AFDperCFS,    # input1.c:474
    # SI（input1.c:444-448）：qcf=LPSperCFS，LPM/MLD/CMH/CMD 覆盖
    "LPS": LPSperCFS,    # input1.c:444
    "LPM": LPMperCFS,    # input1.c:445
    "MLD": MLDperCFS,    # input1.c:446
    "CMH": CMHperCFS,    # input1.c:447
    "CMD": CMDperCFS,    # input1.c:448
}


class UCF:
    """单位换算因子集合（用户单位 / 内部单位）。

    UCF(flow_units)，flow_units ∈ FLOW_UCF（10 种，大写）：
      .flow     FLOW_UCF[flow_units]（initunits input1.c:443-448/:469-474）
      .head     SI: MperFT（input1.c:450）；US: 1.0（input1.c:475）
      .length   同 .head
      .diam     SI: 1000*MperFT (mm/ft，input1.c:443)；US: 12.0 (in/ft，:469)
      .pressure SI: MperFT（用户 m 水柱，input1.c:451）；US: PSIperFT（:476）
 - 契约未要求，附送给 chain-2 的 EN_PRESSURE 换算用，
                   与 EPANET units.c 中 Ucf[PRESSURE] 的取法一致。
    """

    def __init__(self, flow_units: str):
        if flow_units not in FLOW_UCF:
            raise ValueError(f"flow_units 只支持 {sorted(FLOW_UCF)}，"
                             f"得到 {flow_units!r}")
        self.is_si = flow_units in SI_FLOW_UNITS
        self.flow = FLOW_UCF[flow_units]
        if self.is_si:                # SI 制（水头 m、管径 mm、压力 m 水柱）
            self.head = MperFT
            self.diam = 1000.0 * MperFT
            self.pressure = MperFT
        else:                         # US 制（水头 ft、管径 in、压力 psi）
            self.head = 1.0
            self.diam = 12.0
            self.pressure = PSIperFT
        self.length = self.head       # 契约：.length 同 .head
        self.flow_units = flow_units


if __name__ == "__main__":
    # 冒烟测试：两套单位制的因子逐项核对
    si = UCF("LPS")
    us = UCF("GPM")
    print("LPS:", "flow=", si.flow, "head=", si.head, "length=", si.length,
          "diam=", si.diam, "pressure=", si.pressure)
    print("GPM:", "flow=", us.flow, "head=", us.head, "length=", us.length,
          "diam=", us.diam, "pressure=", us.pressure)
    assert si.flow == 28.317 and si.head == 0.3048 and si.diam == 304.8
    assert us.flow == 448.831 and us.head == 1.0 and us.diam == 12.0
    # 1 cfs = 28.317 LPS；100 LPS = 100/28.317 cfs ≈ 3.5314 cfs
    print("100 LPS ->", 100.0 / si.flow, "cfs（应约 3.5314）")
    # 1 psi = 1/0.4333 ft ≈ 2.3079 ft 水柱
    print("1 psi ->", 1.0 / us.pressure, "ft 水柱（应约 2.3079）")
    cmh = UCF("CMH")
    assert cmh.flow == 101.94 and cmh.head == 0.3048 and cmh.is_si
    print("CMH:", "flow=", cmh.flow, "head=", cmh.head, "（types.h:74）")
    try:
        UCF("XXX")
    except ValueError as e:
        print("非法单位正确拒绝:", e)
    print("units.py 冒烟测试通过")

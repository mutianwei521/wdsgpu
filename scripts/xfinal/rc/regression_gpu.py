# -*- coding: utf-8 -*-
"""XF-3b 桩：**不是**真回归，只按 $XF_RC 退出。
放在与出厂 regression_gpu.sh 同一个目录里，用**未改一字的出厂 .sh** 提交，
看 sbatch/sacct 记到的作业退出码是不是跟着 py 走。"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
rc = int(os.environ.get("XF_RC", "0"))
print("XF-STUB 我是桩，不是回归。将以 rc=%d 退出（$XF_RC=%r），节点 %s"
      % (rc, os.environ.get("XF_RC"), os.popen("hostname").read().strip()))
print("总判定: %s（桩）" % {0: "PASS", 1: "FAIL", 2: "SKIP"}.get(rc, "?"))
sys.exit(rc)

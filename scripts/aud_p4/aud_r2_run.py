# -*- coding: utf-8 -*-
"""只做一件事：按 $AUD_SUBSET 缩小 regression_gpu.NETS 的网表，再调它的 main()。
**三条断言的代码一个字符没改**，只是少跑几张网（省 GPU 时）。"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regression_gpu as R                                       # noqa: E402

sub = os.environ.get("AUD_SUBSET", "").strip()
if sub:
    keep = set(sub.split(","))
    R.NETS = [n for n in R.NETS if n[0] in keep]
print("[aud_r2_run] 网表 =", [n[0] for n in R.NETS])
sys.exit(R.main())

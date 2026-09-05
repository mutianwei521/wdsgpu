#!/bin/bash
# 服务器（无 Slurm）后台跑 scripts/augment_suite.py 的包装：nohup + 日志 + rc 文件，
# 每个作业记 hostname / GPU / git 版本 / 关键文件 md5 / 耗时。
# 用法：nohup bash scripts/augment_suite_run.sh <net> <stage[,stage...]> [augment_suite.py 的其余参数] > /dev/null 2>&1 &
#   例：nohup bash scripts/augment_suite_run.sh city_d calib > /dev/null 2>&1 &
#       nohup bash scripts/augment_suite_run.sh city_d leak --configs demo40,ga40,augcover40,augdopt40 > /dev/null 2>&1 &
#       nohup bash scripts/augment_suite_run.sh pub_hanoi verify,calib,report,fig > /dev/null 2>&1 &
# 日志 logs/augment_suite_<net>_<stage>_<时间戳>.log；结束时写同名 .rc（内容 = 退出码）。
# 前台轮询：tail -f <log>，或 until [ -f <log 去掉 .log>.rc ]; do sleep 60; done
set -u
cd "$(dirname "$0")/.." || exit 2
net=$1; stage=$2; shift 2
mkdir -p logs
ts=$(date +%Y%m%d_%H%M%S)
log=logs/augment_suite_${net}_${stage//,/_}_${ts}.log
PY=${PYTHON:-python}
{
  echo "host=$(hostname)  start=$(date -Is)  python=$($PY -c 'import sys;print(sys.version.split()[0])')"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  git rev-parse HEAD 2>/dev/null || echo "(no git metadata: archive checkout)"
  md5sum dgga/placement.py dgga/calib.py dgga/autodiff.py scripts/augment_suite.py \
         scripts/calibrate.py scripts/demo_leak_inversion.py
  t0=$(date +%s)
  $PY -u -X utf8 scripts/augment_suite.py --net "$net" --stage "$stage" "$@"
  rc=$?
  echo "AUGMENT_SUITE DONE rc=$rc  elapsed=$(( $(date +%s) - t0 ))s  end=$(date -Is)  host=$(hostname)"
  echo $rc > "${log%.log}.rc"
} > "$log" 2>&1
exit $(cat "${log%.log}.rc")

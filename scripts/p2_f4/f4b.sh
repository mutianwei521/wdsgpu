#!/bin/bash
#SBATCH --job-name=p2f4b
#SBATCH --output=f4b_%j.out
#SBATCH --error=f4b_%j.out
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
rc=0
run() { python3 -X utf8 f4b_gpu.py "$1" "$2" 2>/dev/null || rc=$?; }
echo "################ A 池语义（free 掉的字节去哪了）################"
run pool 8
echo "################ B 轮转 B∈{1,8,64,256}：上限太小的代价 ################"
for c in none 1 2 4 8; do run cycle $c; done
echo "################ C 整批+尾批：定缺省的那一档 ################"
for c in none 1 2 8; do run tail $c; done
echo "################ D ragged 80 个批量：修复前爬升 vs 修复后持平 ################"
for c in none 8; do run ragged $c; done
echo "################ E 大批量 49 个：修复前的 OOM 边界 ################"
for c in none 8; do run big $c; done
echo "F4B JOB DONE rc=$rc"
exit $rc

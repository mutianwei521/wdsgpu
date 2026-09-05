#!/bin/bash
#SBATCH --job-name=advp2
#SBATCH --output=adv_%j.out
#SBATCH --error=adv_%j.out
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
rc=0
python3 -X utf8 adv_gpu.py || rc=$?
echo ""
echo "================ 显存：每配置一个全新进程（独立复核）================"
for net in Net3 Modena City_D ky4; do
  for B in 64 256; do
    for p in dense cudss; do python3 -X utf8 adv_mem.py $net $B $p || rc=$?; done
  done
done
echo ""
echo "================ 大批量 / OOM 边界（ky4）================"
for B in 512 1024 2048; do
  for p in dense cudss; do python3 -X utf8 adv_mem.py ky4 $B $p || rc=$?; done
done
echo "ADV JOB DONE rc=$rc"
exit $rc

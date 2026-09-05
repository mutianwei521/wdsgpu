#!/bin/bash
#SBATCH --job-name=advp2b
#SBATCH --output=adv2_%j.out
#SBATCH --error=adv2_%j.out
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CUBLAS_WORKSPACE_CONFIG=:4096:8
rc=0
python3 -X utf8 adv2_timing.py || rc=$?
echo ""
echo "================ 同节点复跑上游 p2.py（只为核对其 §2 数字是否可复现）================"
python3 -X utf8 p2.py || rc=$?
echo "ADV2 JOB DONE rc=$rc"
exit $rc

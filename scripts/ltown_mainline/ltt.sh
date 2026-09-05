#!/bin/bash
#SBATCH --job-name=ltt
#SBATCH --output=ltt_%j.out
#SBATCH --error=ltt_%j.out
#SBATCH -p gpu_5090
#SBATCH --gpus=1
#SBATCH --time=04:00:00
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -X utf8 lt_time_gpu.py
rc=$?
echo "LT TIME JOB DONE rc=$rc"
exit $rc

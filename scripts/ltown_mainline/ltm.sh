#!/bin/bash
#SBATCH --job-name=ltm
#SBATCH --output=ltm_%j.out
#SBATCH --error=ltm_%j.out
#SBATCH -p gpu_5090
#SBATCH --gpus=1
#SBATCH --time=04:00:00
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -X utf8 lt_mem_gpu.py
rc=$?
echo "LT MEM JOB DONE rc=$rc"
exit $rc

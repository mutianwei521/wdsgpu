#!/bin/bash
#SBATCH --job-name=ajg
#SBATCH --output=ajg_%j.out
#SBATCH --error=ajg_%j.out
#SBATCH -p gpu_5090
#SBATCH --gpus=1
#SBATCH --time=04:00:00
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -X utf8 adjgpu_gpu.py
rc=$?
echo "ADJGPU JOB DONE rc=$rc"
exit $rc

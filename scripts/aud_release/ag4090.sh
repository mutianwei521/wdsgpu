#!/bin/bash
#SBATCH --job-name=ag4090
#SBATCH --output=ag4090_%j.out
#SBATCH --error=ag4090_%j.out
#SBATCH -p gpu_4090
#SBATCH --gpus=1
#SBATCH --time=02:00:00
cd "$SLURM_SUBMIT_DIR"
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
AUD_BS=8,256,512 python3 -X utf8 aud_gpu.py
rc=$?
echo "AG4090 DONE rc=$rc"
exit $rc

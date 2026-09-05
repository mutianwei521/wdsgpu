#!/bin/bash
#SBATCH --job-name=audr2
#SBATCH --output=audr2_%j.out
#SBATCH --error=audr2_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 aud_r2_gpu.py
rc=$?
echo "AUDR2 JOB DONE rc=$rc"
exit $rc

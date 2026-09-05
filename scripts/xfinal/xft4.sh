#!/bin/bash
#SBATCH --job-name=xft4
#SBATCH --output=xft4_%j.out
#SBATCH --error=xft4_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 xf_t4_teeth.py
rc=$?
echo "XFT4 JOB DONE rc=$rc"
exit $rc

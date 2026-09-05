#!/bin/bash
#SBATCH --job-name=p4r7
#SBATCH --output=p4r7_%j.out
#SBATCH --error=p4r7_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
P4R_R7_STEPS=4 python3 -X utf8 p4r_r7_gpu.py
rc=$?
echo "P4R JOB DONE rc=$rc"
exit $rc

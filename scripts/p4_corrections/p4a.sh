#!/bin/bash
#SBATCH --job-name=p4a
#SBATCH --output=p4a_%j.out
#SBATCH --error=p4a_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090
#SBATCH --exclude=<node-21>

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 p4a_gpu.py
rc=$?
echo "P4 JOB DONE rc=$rc"
exit $rc

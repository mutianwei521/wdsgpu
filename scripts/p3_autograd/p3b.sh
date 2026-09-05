#!/bin/bash
#SBATCH --job-name=p3b
#SBATCH --output=p3b_%j.out
#SBATCH --error=p3b_%j.out
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 p3b_gpu.py
rc=$?
echo "P3B JOB DONE rc=$rc"
exit $rc

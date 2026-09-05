#!/bin/bash
#SBATCH --job-name=p3c
#SBATCH --output=p3c_%j.out
#SBATCH --error=p3c_%j.out
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 p3c_gpu.py
rc=$?
echo "P3C JOB DONE rc=$rc"
exit $rc

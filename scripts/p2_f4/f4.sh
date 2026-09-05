#!/bin/bash
#SBATCH --job-name=p2f4
#SBATCH --output=f4_%j.out
#SBATCH --error=f4_%j.out
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 f4_gpu.py
rc=$?
echo "F4 JOB DONE rc=$rc"
exit $rc

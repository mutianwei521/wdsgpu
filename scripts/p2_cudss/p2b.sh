#!/bin/bash
#SBATCH --job-name=p2b
#SBATCH --output=p2b_%j.out
#SBATCH --error=p2b_%j.out
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 p2b.py
rc=$?
echo "P2B JOB DONE rc=$rc"
exit $rc

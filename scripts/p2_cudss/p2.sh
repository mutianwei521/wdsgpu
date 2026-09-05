#!/bin/bash
#SBATCH --job-name=p2cudss
#SBATCH --output=p2_%j.out
#SBATCH --error=p2_%j.out
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 p2.py
rc=$?
echo "P2 JOB DONE rc=$rc"
exit $rc

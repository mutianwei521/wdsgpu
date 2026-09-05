#!/bin/bash
#SBATCH --job-name=p5diag
#SBATCH --output=p5d_%j.out
#SBATCH --error=p5d_%j.out
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 p5_diag_gpu.py
rc=$?
echo "P5 DIAG JOB DONE rc=$rc"
exit $rc

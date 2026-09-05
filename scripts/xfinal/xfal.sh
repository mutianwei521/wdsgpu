#!/bin/bash
#SBATCH --job-name=xfal
#SBATCH --output=xfal_%j.out
#SBATCH --error=xfal_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 xf_alias.py
rc=$?
echo "XFAL JOB DONE rc=$rc"
exit $rc

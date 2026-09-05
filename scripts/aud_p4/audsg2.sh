#!/bin/bash
#SBATCH --job-name=audsg2
#SBATCH --output=audsg2_%j.out
#SBATCH --error=audsg2_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export AUD_MAG=1e-2
python3 -X utf8 aud_symgap_gpu.py
rc=$?
echo "AUDSG2 JOB DONE rc=$rc"
exit $rc

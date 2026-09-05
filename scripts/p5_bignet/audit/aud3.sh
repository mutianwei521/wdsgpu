#!/bin/bash
#SBATCH --job-name=aud3time
#SBATCH --output=aud3_%j.out
#SBATCH --error=aud3_%j.out
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 aud_gpu_a3.py
rc=$?
echo "AUD A3 JOB DONE rc=$rc"
exit $rc

#!/bin/bash
#SBATCH --job-name=p5time
#SBATCH --output=p5t_%j.out
#SBATCH --error=p5t_%j.out
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 p5_time_gpu.py
rc=$?
echo "P5 TIME JOB DONE rc=$rc"
exit $rc

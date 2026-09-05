#!/bin/bash
#SBATCH --job-name=p5mem
#SBATCH --output=p5m_%j.out
#SBATCH --error=p5m_%j.out
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 p5_mem_gpu.py
rc=$?
echo "P5 MEM JOB DONE rc=$rc"
exit $rc

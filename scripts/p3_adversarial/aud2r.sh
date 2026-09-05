#!/bin/bash
#SBATCH --job-name=aud2r
#SBATCH --output=aud2r_%j.out
#SBATCH --error=aud2r_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090
#SBATCH --exclude=<node-21>
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 aud2_gpu.py
rc=$?
echo "AUD JOB DONE rc=$rc"
exit $rc

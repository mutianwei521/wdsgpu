#!/bin/bash
#SBATCH --job-name=p5th
#SBATCH --output=p5th_%j.out
#SBATCH --error=p5th_%j.out
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -X utf8 p5_thresh.py
rc=$?
echo "P5 THRESH JOB DONE rc=$rc"
exit $rc

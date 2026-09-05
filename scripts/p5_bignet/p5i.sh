#!/bin/bash
#SBATCH --job-name=p5iso
#SBATCH --output=p5i_%j.out
#SBATCH --error=p5i_%j.out
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -X utf8 p5_d1_isolate.py
rc=$?
echo "P5 ISOLATE JOB DONE rc=$rc"
exit $rc

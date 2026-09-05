#!/bin/bash
#SBATCH --job-name=rcold
#SBATCH --output=rcold_%j.out
#SBATCH --error=rcold_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090
# ===================== DO NOT "FIX" THIS FILE =====================
# Deliberate negative control: the old shape, kept so the difference can be
# re-demonstrated on demand. Slurm records this as COMPLETED 0:0 even though
# the Python inside exits 3.
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)"
python3 -X utf8 rcstub.py 3
echo "RCOLD JOB DONE rc=$?"

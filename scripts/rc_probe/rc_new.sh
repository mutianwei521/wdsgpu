#!/bin/bash
#SBATCH --job-name=rcnew
#SBATCH --output=rcnew_%j.out
#SBATCH --error=rcnew_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)"
python3 -X utf8 rcstub.py 3
rc=$?
echo "RCNEW JOB DONE rc=$rc"
exit $rc

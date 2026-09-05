#!/bin/bash
#SBATCH --job-name=rcok
#SBATCH --output=rcok_%j.out
#SBATCH --error=rcok_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)"
python3 -X utf8 rcstub.py 0
rc=$?
echo "RCOK JOB DONE rc=$rc"
exit $rc

#!/bin/bash
#SBATCH --job-name=rcacc
#SBATCH --output=rcacc_%j.out
#SBATCH --error=rcacc_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090
# Accumulate shape: the FIRST of two steps fails and the second succeeds.
# Under the old "last command wins" behaviour this job would be recorded as
# COMPLETED; it must come back FAILED 5:0.
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)"
rc=0
python3 -X utf8 rcstub.py 5 || rc=$?
python3 -X utf8 rcstub.py 0 || rc=$?
echo "RCACC JOB DONE rc=$rc"
exit $rc

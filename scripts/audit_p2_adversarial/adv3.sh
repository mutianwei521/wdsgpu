#!/bin/bash
#SBATCH --job-name=advp2c
#SBATCH --output=adv3_%j.out
#SBATCH --error=adv3_%j.out
cd "$SLURM_SUBMIT_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -X utf8 adv3_repro.py
rc=$?
echo "ADV3 JOB DONE rc=$rc"
exit $rc

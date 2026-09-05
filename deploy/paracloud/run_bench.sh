#!/bin/bash
#SBATCH --job-name=hydrograd
#SBATCH --gpus=1
#SBATCH --output=bench_%x_%j.out
#SBATCH --error=bench_%x_%j.out
cd "$SLURM_SUBMIT_DIR"
source "$PWD/venv/bin/activate"
echo "=============================================================="
echo "partition : $SLURM_JOB_PARTITION"
echo "node      : $(hostname)"
echo "cpus      : $(nproc)"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
echo "=============================================================="
# One card per job, so the visible device is always idle -- no occupancy
# check needed, unlike a shared workstation.
rc=0
python -X utf8 gpu_bench.py || rc=$?
echo
python -X utf8 epanet_ref_bench.py || rc=$?
echo "BENCH JOB DONE rc=$rc"
exit $rc

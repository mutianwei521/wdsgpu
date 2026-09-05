#!/bin/bash
#SBATCH --job-name=mv2
#SBATCH --output=mv2_%j.out
#SBATCH --error=mv2_%j.out
#SBATCH -p gpu_5090
#SBATCH --gpus=1
#SBATCH --time=04:00:00
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rc=0
for scr in mv2_time_gpu.py mv2_big_gpu.py mv2_train_gpu.py mv2_mem_gpu.py; do
  echo "########## RUN $scr ##########"
  python3 -X utf8 "$scr"
  r=$?
  echo "########## $scr rc=$r ##########"
  if [ "$r" -ne 0 ]; then rc=1; fi
done
echo "MV2 JOB DONE rc=$rc"
exit $rc

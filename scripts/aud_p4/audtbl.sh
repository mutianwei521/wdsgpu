#!/bin/bash
#SBATCH --job-name=audtbl
#SBATCH --output=audtbl_%j.out
#SBATCH --error=audtbl_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
rc=0
python3 -X utf8 aud_tbl_gpu.py || rc=$?
echo "--- MEM SPOTCHECK (fresh process per config) ---"
for cfg in "ky4.inp 64" "ky4.inp 256" "City_D.inp 512" "City_D.inp 1024" "Modena.inp 64" "Net3.inp 1024"; do
  set -- $cfg
  for m in dense cudss; do
    python3 -X utf8 aud_mem_gpu.py "$1" "$2" "$m" || rc=$?
  done
done
echo "AUDTBL JOB DONE rc=$rc"
exit $rc

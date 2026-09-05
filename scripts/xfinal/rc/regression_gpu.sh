#!/bin/bash
#SBATCH --job-name=regp4
#SBATCH --output=regp4_%j.out
#SBATCH --error=regp4_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# rc 必须先存再 echo，最后 exit $rc：末条命令若是 echo，脚本恒退 0，
# 作业状态就永远 COMPLETED - FAIL 与 SKIP 都会被 CI 当成通过
# （P4 终审 data/p4_adversarial_wip.txt §3.2）。
python3 -X utf8 regression_gpu.py
rc=$?
echo "REGRESSION_GPU JOB DONE rc=$rc"
exit $rc

#!/bin/bash
#SBATCH --job-name=xfrcold
#SBATCH --output=xfrcold_%j.out
#SBATCH --error=xfrcold_%j.out
#SBATCH --gpus=1
#SBATCH -p gpu_5090

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node: $(hostname) | cpus: $(nproc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# ============================ 请勿"修好"本文件 ============================
# 这是**故意保留的坏形式**，是 P4 终审用来证明 bug 存在的负对照：作业
# 1459917 用本脚本跑一个故意退 3 的桩，Slurm 记成 COMPLETED 0:0；同一个桩
# 换成 ../regression_gpu.sh 的正确形式（作业 1459916）记成 FAILED 3:0。
# 把这里改成 exit $rc 就等于销毁那份证据。
# ------------------------------------------------------------------------
# 正确形式见 scripts/regression_gpu.sh：rc 必须先存再 echo，最后 exit $rc。
# 末条命令若是 echo，脚本恒退 0，
# 作业状态就永远 COMPLETED - FAIL 与 SKIP 都会被 CI 当成通过
# （P4 终审 data/p4_adversarial_wip.txt §3.2）。
python3 -X utf8 regression_gpu.py
echo "REGRESSION_GPU JOB DONE rc=$?"

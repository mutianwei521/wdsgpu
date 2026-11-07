#!/bin/bash
#SBATCH --job-name=wrfreeze
#SBATCH -p gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=wrfreeze_%j.out
#SBATCH --error=wrfreeze_%j.out
# WR 大修（10-23）：代码冻结后，在同一个作业、同一节点上，用同一统计量（中位数，至少 5 次重复）
# 统一重测论文用到的全部 GPU 计时。运行目录为扁平布局（scripts/wr_freeze_bundle.sh 生成）：
#   dgga/、本脚本、mv2_time_gpu.py、mv2_train_gpu.py、p4r_time_gpu.py、p4r_r7_gpu.py、a3_gpu.py、gpu_transfer_timing.py、
#   p2nets/（P4 原测量所用的 10 个 INP）、networks/public/_cleaned/L-TOWN.inp、networks/realInpData/city_d.inp、COMMIT.txt
# 不重测：S6b 显存（确定量，两节点逐位相同）；未被引用的大网补格（mv2_big）；RTX 4090 上的优化器对比（另注明硬件）。
cd "$SLURM_SUBMIT_DIR"
echo "commit: $(cat COMMIT.txt)"; echo "node: $(hostname)  start: $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv,noheader
lscpu | grep -E "Model name|^CPU\(s\)|Thread|max MHz"; echo "nproc=$(nproc)"
python3 -c "import torch,nvmath,numpy,scipy;print('torch',torch.__version__,'cuda',torch.version.cuda,'nvmath',nvmath.__version__,'numpy',numpy.__version__,'scipy',scipy.__version__)"
md5sum dgga/*.py *.py p2nets/*.inp networks/public/_cleaned/L-TOWN.inp networks/realInpData/city_d.inp
export OMP_NUM_THREADS=8 TIMING_STAT=median TIMING_REPS_MIN=5
CD=networks/realInpData/city_d.inp
run() { echo "########## RUN $* ##########  $(date +%T)"; env "$@"; echo "########## rc=$? $(date +%T) ##########"; }
# 1. 三条路径（S6a；F1、F2；City D 为新增）。B=1024 先在全新进程单独跑，避开进程内碎片导致的 OOM。
run MV2_BS=1024 python3 -X utf8 mv2_time_gpu.py
run MV2_BS=1,8,64,256,512 python3 -X utf8 mv2_time_gpu.py
run MV2_INP=$CD MV2_BS=1024 python3 -X utf8 mv2_time_gpu.py
run MV2_INP=$CD MV2_BS=1,8,64,256,512 python3 -X utf8 mv2_time_gpu.py
# 2. 真实反演循环（"191 倍"；旧形态取 3 步的中位数）
run MV2_TB=256 MV2_OLD_STEPS=3 python3 -X utf8 mv2_train_gpu.py
run MV2_TB=1024 MV2_OLD_STEPS=3 python3 -X utf8 mv2_train_gpu.py
run MV2_INP=$CD MV2_TB=256 MV2_OLD_STEPS=3 python3 -X utf8 mv2_train_gpu.py
# 3. 稀疏/稠密交叉点、规模轴、A/B/C 分解（补充材料第 7 节）
run CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -X utf8 p4r_time_gpu.py
# 4. cuDSS 计划缓存规则（补充材料第 6 节）
run CUBLAS_WORKSPACE_CONFIG=:4096:8 P4R_R7_STEPS=4 P4R_R7_NETS=Modena,ky4 python3 -X utf8 p4r_r7_gpu.py
# 5. 反向不新增分解的独立计数（正文第 53 行）
run CUBLAS_WORKSPACE_CONFIG=:4096:8 A3_BS=8,64,256,1024 python3 -X utf8 a3_gpu.py
# 6. 数据传输
run python3 -X utf8 gpu_transfer_timing.py
echo "end: $(date '+%F %T')"; echo "JOB DONE"

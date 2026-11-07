#!/bin/bash
# WR 大修 10-23：重测 L-TOWN 三列对比与反演循环（旧形态多步取中位），并在 City D 上重复
cd "$SLURM_SUBMIT_DIR"
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
md5sum dgga/solver.py dgga/autodiff.py
export OMP_NUM_THREADS=8
run() { echo "########## RUN $* ##########"; env "$@"; echo "########## rc=$? ##########"; }
run MV2_BS=1,8,64,256,512,1024 python3 -X utf8 mv2_time_gpu.py
run MV2_TB=256 MV2_OLD_STEPS=10 python3 -X utf8 mv2_train_gpu.py
run MV2_TB=1024 MV2_OLD_STEPS=5 python3 -X utf8 mv2_train_gpu.py
run MV2_INP=networks/realInpData/city_d.inp MV2_BS=1,8,64,256,512,1024 python3 -X utf8 mv2_time_gpu.py
run MV2_INP=networks/realInpData/city_d.inp MV2_TB=256 MV2_OLD_STEPS=5 python3 -X utf8 mv2_train_gpu.py
echo "JOB DONE"

#!/bin/bash
#SBATCH --job-name=ag5090
#SBATCH --output=ag5090_%j.out
#SBATCH --error=ag5090_%j.out
#SBATCH -p gpu_5090
#SBATCH --gpus=1
#SBATCH --time=02:00:00
cd "$SLURM_SUBMIT_DIR"
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rcall=0
AUD_BS=8,256,1024 python3 -X utf8 aud_gpu.py; r1=$?; echo "STEP aud_gpu rc=$r1"; [ $r1 -ne 0 ] && rcall=1
python3 -X utf8 aud_t4probe.py .; r2=$?; echo "STEP t4probe_genuine rc=$r2"; [ $r2 -ne 0 ] && rcall=1
python3 -X utf8 aud_mutate.py MF_oneside dgga mut_mf; [ $? -ne 0 ] && rcall=1
cp *.inp mut_mf/ 2>/dev/null
cp aud_t4probe.py mut_mf/
python3 -X utf8 mut_mf/aud_t4probe.py mut_mf; r3=$?; echo "STEP t4probe_MF rc=$r3 期望非0"
[ $r3 -eq 0 ] && rcall=1
mkdir -p base/scripts mutrg/scripts
cp -r dgga base/ 2>/dev/null; cp regression_gpu.py base/scripts/
cp -r mut_mf/dgga mutrg/; cp regression_gpu.py mutrg/scripts/
export DGGA_NETS=$HOME/run/wdsgpu/hydrograd/p2nets   # adjust to your cluster work directory
python3 -X utf8 base/scripts/regression_gpu.py > rg_base.out 2>&1; r4=$?
echo "STEP upstream_rg_baseline rc=$r4"; tail -4 rg_base.out
python3 -X utf8 mutrg/scripts/regression_gpu.py > rg_mf.out 2>&1; r5=$?
echo "STEP upstream_rg_MF rc=$r5 rc0即T4网表洞实锤"; tail -4 rg_mf.out
rc=$rcall
echo "AG5090 DONE rc=$rc r1=$r1 r2=$r2 r3=$r3 r4=$r4 r5=$r5"
exit $rc

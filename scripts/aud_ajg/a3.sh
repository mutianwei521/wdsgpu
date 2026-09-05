#!/bin/bash
#SBATCH --job-name=audajg
#SBATCH --output=audajg_%j.out
#SBATCH --error=audajg_%j.out
#SBATCH -p gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
cd "$SLURM_SUBMIT_DIR"
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CUBLAS_WORKSPACE_CONFIG=:4096:8
rcall=0
echo "########## STEP1 a3_gpu.py ##########"
python3 -X utf8 a3_gpu.py
r1=$?
echo "STEP1 rc=$r1"
[ $r1 -ne 0 ] && rcall=1
echo "########## STEP2 regression_gpu genuine ##########"
export DGGA_NETS="$PWD/nets"
mkdir -p base/scripts mf/scripts
rm -rf base/dgga mf/dgga
cp -r dgga base/
cp regression_gpu.py base/scripts/
python3 -X utf8 aud_mutate.py MF_oneside dgga mf
rmut=$?
cp regression_gpu.py mf/scripts/
python3 -X utf8 base/scripts/regression_gpu.py > rg_base.out 2>&1
r2=$?
echo "STEP2 baseline rc=$r2"
tail -6 rg_base.out
grep -cE "PASS" rg_base.out
echo "########## STEP3 regression_gpu MF mutant (expect nonzero) ##########"
python3 -X utf8 mf/scripts/regression_gpu.py > rg_mf.out 2>&1
r3=$?
echo "STEP3 MF rc=$r3 expect_nonzero"
tail -6 rg_mf.out
echo "--- MF FAIL rows ---"
grep -E "FAIL" rg_mf.out | head -12
grep -cE "FAIL" rg_mf.out
[ $rmut -ne 0 ] && rcall=1
[ $r2 -ne 0 ] && rcall=1
[ $r3 -eq 0 ] && rcall=1
rc=$rcall
echo "AUDAJG DONE rc=$rc r1=$r1 rmut=$rmut r2=$r2 r3=$r3"
exit $rc

#!/bin/bash
#SBATCH --job-name=ltleak
#SBATCH --output=ltleak_%j.out
#SBATCH --error=ltleak_%j.out
#SBATCH -p gpu_5090,gpu_4090
#SBATCH --gpus=1
#SBATCH --time=03:00:00
# L-TOWN 漏损反演原样重跑（传感器 = S0 ∪ S_k），csr+cuDSS = 记录作业 mv2 的配置。
# 提交：cd hydrograd/augpub && sbatch scripts/augment_public_ltleak.sh
# （augpub/ = git archive HEAD dgga + 本脚本目录 + data/placement_orders_ltown.npz
#   + networks_prv/L-TOWN.inp，与 mv2/、ltmain/ 同样的自包含目录约定）
cd "$SLURM_SUBMIT_DIR"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
md5sum dgga/solver.py dgga/autodiff.py dgga/placement.py scripts/augment_public.py
sha256sum networks_prv/L-TOWN.inp
python3 -u -X utf8 scripts/augment_public.py --stage ltown-leak --linear cudss --chunk 256
rc=$?
echo "LTLEAK JOB DONE rc=$rc"
exit $rc

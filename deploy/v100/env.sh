# source this on the V100 server before any job:  source deploy/v100/env.sh
# (also installed as /mnt/sda/$USER/scratch/act.sh)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hydrograd
export CUBLAS_WORKSPACE_CONFIG=:4096:8          # same as scripts/regression_gpu.sh
export DGGA_NETS=/mnt/sda/$USER/scratch/p2nets  # networks/public + _cleaned + City_D.inp (-> datasets/city_d.inp), symlinks
cd /mnt/sda/$USER/hydrograd

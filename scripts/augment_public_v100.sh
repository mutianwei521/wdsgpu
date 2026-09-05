#!/bin/bash
# augment_public_v100.sh -- run the public-network augmentation suite (scripts/augment_public.py)
# on the V100 server (ssh v100, no Slurm): nohup + log + rc file per job; each log records
# host / GPU / commit / md5 of the code / sha256 of the .inp / elapsed time.
#
# Layout (one archive of HEAD per job so that concurrent jobs never share an output file):
#   /mnt/sda/$USER/scratch/augpub_v100/{cpu,calib,g0,g1,g2}   git archive HEAD + symlinks to
#   networks/ and data/reference/; the committed ltown_augment_leak*.json are moved aside so that
#   the leak stage recomputes every configuration instead of skipping it.
#
# Usage on the server:
#   bash scripts/augment_public_v100.sh setup      # create the five work dirs from HEAD
#   bash scripts/augment_public_v100.sh launch     # start the five jobs (3 GPUs + 2 CPU)
#   bash scripts/augment_public_v100.sh status     # tail the logs, show rc files
# Fetch afterwards (from the workstation), then merge + compare:
#   scripts/augment_public_compare.py --v100-dir data/v100_augpub
set -u
H=/mnt/sda/$USER/hydrograd
S=/mnt/sda/$USER/scratch/augpub_v100
R=$S/augpub_run.sh
PY="python -X utf8 scripts/augment_public.py"

write_runner() {
cat > "$R" <<'EOF'
#!/bin/bash
# background runner: <workdir> <gpu index|cpu> <label> '<shell command>'
set -u
W=$1; G=$2; LABEL=$3; CMD=$4
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hydrograd
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DGGA_NETS=/mnt/sda/$USER/scratch/p2nets
export PYTHONUNBUFFERED=1
if [ "$G" = cpu ]; then
  export CUDA_VISIBLE_DEVICES=""; export OMP_NUM_THREADS=8
else
  export CUDA_VISIBLE_DEVICES=$G; export OMP_NUM_THREADS=4
fi
cd "$W" || exit 2
mkdir -p logs
log=logs/$LABEL.log
rm -f "logs/$LABEL.rc"
{
  echo "host=$(hostname)  start=$(date -Is)  CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'  OMP_NUM_THREADS=$OMP_NUM_THREADS  cwd=$W"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  echo "commit=$(cat GIT_HEAD 2>/dev/null)"
  python -c 'import torch,numpy,scipy;print("python",__import__("sys").version.split()[0],"torch",torch.__version__,"numpy",numpy.__version__,"scipy",scipy.__version__)'
  md5sum dgga/placement.py dgga/solver.py dgga/autodiff.py dgga/calib.py dgga/sensitivity.py \
         scripts/augment_public.py scripts/calibrate.py scripts/place_sensors.py
  sha256sum networks/EXAMPLE/L-Town/L-TOWN.inp networks/public/Hanoi.inp
  echo "CMD: $CMD"
  t0=$(date +%s)
  bash -c "$CMD"
  rc=$?
  echo "AUGPUB DONE rc=$rc elapsed=$(( $(date +%s) - t0 ))s end=$(date -Is) host=$(hostname)"
  echo $rc > "logs/$LABEL.rc"
} > "$log" 2>&1
EOF
}

case ${1:-} in
  setup)
    mkdir -p "$S"; write_runner
    for d in cpu calib g0 g1 g2; do
      rm -rf "$S/$d"; mkdir -p "$S/$d"
      git -C "$H" archive HEAD | tar -x -C "$S/$d"
      git -C "$H" rev-parse HEAD > "$S/$d/GIT_HEAD"
      ln -sfn "$H/networks" "$S/$d/networks"
      ln -sfn "$H/data/reference" "$S/$d/data/reference"
      for f in ltown_augment_leak ltown_augment_leak_5090 ltown_augment_leak_local_dense; do
        [ -f "$S/$d/data/$f.json" ] && mv "$S/$d/data/$f.json" "$S/$d/data/$f.committed.json"
      done
    done
    rm -f "$S/cpu/data/placement_augment_ltown.json"     # sfull/augment start from a clean file
    echo "setup done: $S"
    ;;
  launch)
    nohup bash "$R" "$S/cpu"   cpu cpu_chain   "$PY --stage ltown-sfull && $PY --stage ltown-augment && $PY --stage hanoi-augment && $PY --stage sigma-ladder" > /dev/null 2>&1 &
    nohup bash "$R" "$S/calib" cpu hanoi_calib "$PY --stage hanoi-calib --placements augS0,augdopt5,augdopt10,augdopt20" > /dev/null 2>&1 &
    nohup bash "$R" "$S/g0" 0 ltleak_g0 "$PY --stage ltown-leak --linear cudss --chunk 256 --only S0,dopt+5,dopt+10,dopt+20" > /dev/null 2>&1 &
    nohup bash "$R" "$S/g1" 1 ltleak_g1 "$PY --stage ltown-leak --linear cudss --chunk 256 --only dopt+40,dopt+80,cover+5,cover+10" > /dev/null 2>&1 &
    nohup bash "$R" "$S/g2" 2 ltleak_g2 "$PY --stage ltown-leak --linear cudss --chunk 256 --only cover+20,cover+40,cover+80" > /dev/null 2>&1 &
    echo "launched 5 jobs; logs in $S/*/logs/"
    ;;
  status)
    for d in cpu calib g0 g1 g2; do
      echo "=== $d"; ls "$S/$d/logs/"*.rc 2>/dev/null && cat "$S/$d/logs/"*.rc
      grep -v "Warning\|^  return\|^  a_list\|^Consider\|^  traj\|multithreading" "$S/$d/logs/"*.log | tail -n 4
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    ;;
  *)
    echo "usage: $0 setup|launch|status"; exit 2
    ;;
esac

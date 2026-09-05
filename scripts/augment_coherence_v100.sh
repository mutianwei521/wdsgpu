#!/bin/bash
# augment_coherence_v100.sh -- run the leak-inversion stages of scripts/augment_coherence.py on the
# V100 server (ssh v100, no Slurm): nohup + log + rc file per job; each log records host / GPU /
# commit / md5 of the code / elapsed time. The selection stage (--stage select) runs on the
# workstation first; its orders (data/placement_orders_coh_*.npz) travel with the archive.
#
# Layout: one archive of HEAD in /mnt/sda/$USER/scratch/augcoh_v100/w (symlinks to networks/ and
# data/reference/); every job writes a distinct output file, so one work dir serves all jobs.
#   GPU jobs (3):  L-TOWN --stage leak --net ltown --linear cudss --chunk 256 --only <subset>
#                  -> data/ltown_coh_leak_g{0,1,2}.json   (merged on the workstation)
#   CPU jobs (8):  City D --stage leak --net city_d --specs <one spec>, 3 threads each
#                  -> data/leak_coh_city_d/<spec>.json
#
# Usage on the server (tarball = `git archive --format=tar.gz HEAD` made on the workstation):
#   bash scripts/augment_coherence_v100.sh setup <tarball> <commit>   # unpack + symlinks
#   bash scripts/augment_coherence_v100.sh launch            # start 3 GPU + 8 CPU jobs
#   bash scripts/augment_coherence_v100.sh controls [nslot]  # City D random controls, CPU only
#   bash scripts/augment_coherence_v100.sh runlist "<specs>" [nslot] [threads]   # any City D batch
#   bash scripts/augment_coherence_v100.sh status            # rc files + log tails + nvidia-smi
# Fetch afterwards (from the workstation): data/ltown_coh_leak_g*.json, data/leak_coh_city_d/*.json,
# logs/*.log -> data/gpu/v100_augcoh_<label>_<date>.log (gitignored), then
#   python -X utf8 scripts/augment_coherence.py --stage merge --parts data/ltown_coh_leak_g0.json,...
#   python -X utf8 scripts/augment_coherence.py --stage report
set -u
# without linger, systemd-logind kills every nohup'd child in this login session the moment the
# ssh connection that launched them drops (workstation reboot/crash included) -- no rc file, no
# error, jobs just vanish mid-run. Bit us once (2h of a 12-slot batch lost silently). Idempotent.
loginctl enable-linger 2>/dev/null || true
# HG_HOME / HG_SCRATCH let the same script drive the other lab server (128 CPU cores), whose
# checkout and private data live under a different mount; CONDA_SH points at that host's conda.
H=${HG_HOME:-/mnt/sda/$USER/hydrograd}
S=${HG_SCRATCH:-/mnt/sda/$USER/scratch}/augcoh_v100
W=$S/w
R=$S/augcoh_run.sh
export CONDA_SH=${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
PY="python -X utf8 scripts/augment_coherence.py"

write_runner() {
cat > "$R" <<'EOF'
#!/bin/bash
# background runner: <workdir> <gpu index|cpu> <threads> <label> '<shell command>'
set -u
W=$1; G=$2; TH=$3; LABEL=$4; CMD=$5
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate hydrograd
export CUBLAS_WORKSPACE_CONFIG=:4096:8
[ -d "${DGGA_NETS:-}" ] || export DGGA_NETS=/mnt/sda/$USER/scratch/p2nets
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH
if [ "$G" = cpu ]; then export CUDA_VISIBLE_DEVICES=""; else export CUDA_VISIBLE_DEVICES=$G; fi
cd "$W" || exit 2
mkdir -p logs
log=logs/$LABEL.log
rm -f "logs/$LABEL.rc"
{
  echo "host=$(hostname)  start=$(date -Is)  CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'  OMP_NUM_THREADS=$OMP_NUM_THREADS  cwd=$W"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
  echo "commit=$(cat GIT_HEAD 2>/dev/null)"
  python -c 'import torch,numpy,scipy;print("python",__import__("sys").version.split()[0],"torch",torch.__version__,"numpy",numpy.__version__,"scipy",scipy.__version__)'
  md5sum dgga/placement.py dgga/solver.py dgga/autodiff.py scripts/augment_coherence.py scripts/augment_public.py scripts/demo_leak_inversion.py
  md5sum data/placement_orders_coh_ltown.npz data/placement_orders_coh_city_d.npz
  echo "CMD: $CMD"
  t0=$(date +%s)
  bash -c "$CMD"
  rc=$?
  echo "AUGCOH DONE rc=$rc elapsed=$(( $(date +%s) - t0 ))s end=$(date -Is) host=$(hostname)"
  echo $rc > "logs/$LABEL.rc"
} > "$log" 2>&1
EOF
}

case ${1:-} in
  setup)
    TAR=${2:?tarball of git archive HEAD}
    COMMIT=${3:?commit hash the tarball was made from}
    mkdir -p "$S"; write_runner
    rm -rf "$W"; mkdir -p "$W"
    tar -xzf "$TAR" -C "$W"
    echo "$COMMIT" > "$W/GIT_HEAD"
    ln -sfn "$H/networks" "$W/networks"
    ln -sfn "$H/data/reference" "$W/data/reference"
    mkdir -p "$W/data/leak_coh_city_d"
    echo "setup done: $W (commit $(cat "$W/GIT_HEAD"))"
    ;;
  launch)
    nohup bash "$R" "$W" 0 4 lt_g0 "$PY --stage leak --net ltown --linear cudss --chunk 256 --out data/ltown_coh_leak_g0.json --only S0,coh+5,coh+10,coh+20" > /dev/null 2>&1 &
    nohup bash "$R" "$W" 1 4 lt_g1 "$PY --stage leak --net ltown --linear cudss --chunk 256 --out data/ltown_coh_leak_g1.json --only coh+40,coh+80,cohmax+20,rand+20:s0" > /dev/null 2>&1 &
    nohup bash "$R" "$W" 2 4 lt_g2 "$PY --stage leak --net ltown --linear cudss --chunk 256 --out data/ltown_coh_leak_g2.json --only rand+20:s1,rand+20:s2,rand+20:s3,rand+20:s4" > /dev/null 2>&1 &
    for sp in coh+5 coh+10 coh+20 coh+40 coh+80 cohfull+20 cohfull+40 cohmax+20; do
      lab=cd_$(echo "$sp" | tr '+' 'p')
      nohup bash "$R" "$W" cpu 3 "$lab" "$PY --stage leak --net city_d --specs $sp" > /dev/null 2>&1 &
    done
    echo "launched 3 GPU + 8 CPU jobs; logs in $W/logs/"
    ;;
  controls)
    # City D random controls for the coherence design: k = 20 and k = 40, twenty fair seeds each
    # (the first twenty seeds whose draw misses every leak node -- 8, 15 are rejected at k = 20 and
    # 5, 15, 16, 17 at k = 40), plus S0 through the very same code path as the coh runs.
    # Round-robin over NSLOT background jobs of 3 threads each (36 threads on this host).
    NSLOT=${2:-12}
    SPECS="S0"
    for s in 0 1 2 3 4 5 6 7 9 10 11 12 13 14 16 17 18 19 20 21; do SPECS="$SPECS rand20_s$s"; done
    for s in 0 1 2 3 4 6 7 8 9 10 11 12 13 14 18 19 20 21 22 23; do SPECS="$SPECS rand40_s$s"; done
    for j in $(seq 0 $((NSLOT - 1))); do
      grp=""; k=0
      for sp in $SPECS; do
        [ $((k % NSLOT)) -eq "$j" ] && grp="$grp,$sp"
        k=$((k + 1))
      done
      grp=${grp#,}
      [ -z "$grp" ] && continue
      nohup bash "$R" "$W" cpu 3 "cdctl_$j" "$PY --stage leak --net city_d --specs $grp" > /dev/null 2>&1 &
    done
    echo "launched $NSLOT City D control jobs (specs: $(echo $SPECS | wc -w)); logs in $W/logs/"
    ;;
  runlist)
    # generic City D batch: runlist "<space-separated specs>" [nslot] [threads]
    SPECS=${2:?spec list}
    NSLOT=${3:-12}
    TH=${4:-3}
    for j in $(seq 0 $((NSLOT - 1))); do
      grp=""; k=0
      for sp in $SPECS; do
        [ $((k % NSLOT)) -eq "$j" ] && grp="$grp,$sp"
        k=$((k + 1))
      done
      grp=${grp#,}
      [ -z "$grp" ] && continue
      nohup bash "$R" "$W" cpu "$TH" "cdrun_$j" "$PY --stage leak --net city_d --specs $grp" > /dev/null 2>&1 &
    done
    echo "launched $NSLOT jobs x $TH threads over $(echo $SPECS | wc -w) specs; logs in $W/logs/"
    ;;
  status)
    for f in "$W"/logs/*.log; do
      l=$(basename "$f" .log)
      rc=$(cat "$W/logs/$l.rc" 2>/dev/null || echo "-")
      echo "== $l rc=$rc :: $(grep -a -v 'Warning\|^  return\|^  a_list\|^Consider\|^  traj\|multithreading' "$f" | tail -n 1 | cut -c1-160)"
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
    uptime
    ;;
  *)
    echo "usage: $0 setup <tarball> <commit>|launch|controls [nslot]|runlist \"<specs>\" [nslot] [threads]|status"; exit 2
    ;;
esac

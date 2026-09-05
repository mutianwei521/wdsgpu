#!/bin/bash
# Run ONCE on the LOGIN node. Compute nodes have no internet, so every
# dependency has to be present before any job is submitted.
set -e
cd "$(dirname "$0")"
ENV_DIR="$PWD/venv"
echo "== creating $ENV_DIR =="
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip -q
# numpy must stay below 2: wntr's compiled extensions are built against 1.x
# and fail to import otherwise (seen on another host).
pip install -q "numpy<2" scipy wntr
# cu128 is required for Blackwell (RTX 5090, sm_120). It also covers 4090
# (sm_89) and 3090 (sm_86), so one wheel serves every queue here.
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
echo "== versions =="
python -c "import numpy,scipy,wntr,torch;print('numpy',numpy.__version__,'scipy',scipy.__version__,'wntr',wntr.__version__,'torch',torch.__version__,torch.version.cuda)"
echo "== done. submit with: sbatch --gpus=1 -p gpu_5090 ./run_bench.sh =="

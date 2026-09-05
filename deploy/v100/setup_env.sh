#!/bin/bash
# Build the `hydrograd` conda environment on the V100 server (3x Tesla V100 32 GB,
# sm_70, driver 580 -> CUDA 13.0 capable; wheels are cu126). Run once, from the
# server, with internet access (the campus DNS drops now and then; pip retries
# and resumes on its own, a full run took ~35 min at 0.5-6 MB/s).
#
#   bash deploy/v100/setup_env.sh 2>&1 | tee ~/hydrograd_setup.log
#
# Version pins that matter (see docs/linux_gpu_hosts.md, "V100 host"):
#   * torch cu126 wheel: the cu128+ wheels drop sm_70; every 2.6.0 .. 2.14.0+cu126
#     wheel still ships sm_70 (checked with torch.cuda.get_arch_list()).
#   * nvmath-python 1.0.0: the version the cuDSS path (P2-P4, regression_gpu.py)
#     was validated against on the 5090/4090 cluster.
#   * cuda-core 1.1.1: cuda-core 1.2.0 (pulled in by nvmath-python[cu12]) makes
#     nvmath's PinnedMemoryResource raise "CUDA device 0 does not support the
#     requested host memory pool" on this Volta + driver 580 box, which fails every
#     explicitly-batched DirectSolver.solve(). 1.1.1 (and 1.0.1) work.
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda create -y -n hydrograd python=3.11
conda activate hydrograd
python -m pip install --upgrade pip -q
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy scipy wntr pytest matplotlib pandas networkx
python -m pip install "nvmath-python[cu12]==1.0.0"
python -m pip install "cuda-core==1.1.1"
python - <<'PY'
import torch, numpy, scipy, wntr, nvmath, cuda.core
print("torch", torch.__version__, torch.version.cuda, "cuda_ok", torch.cuda.is_available())
print("arch", torch.cuda.get_arch_list())
print("ngpu", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("numpy", numpy.__version__, "scipy", scipy.__version__, "wntr", wntr.__version__,
      "nvmath", nvmath.__version__, "cuda-core", cuda.core.__version__)
assert "sm_70" in torch.cuda.get_arch_list(), "torch wheel lost sm_70 -- pin an older cu126 wheel"
PY
echo "== done. activate with: source deploy/v100/env.sh =="

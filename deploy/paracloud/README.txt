HydroGrad cross-GPU benchmark, ParaCloud (Slurm)

Target dir: <your cluster work directory>/wdsgpu

1. Upload and unpack, then ONCE on the login node:
       ./setup_env.sh
   This installs into ./venv. It must happen on the login node because the
   compute nodes have no internet. torch is the cu128 wheel, which is what
   Blackwell (RTX 5090, sm_120) requires; it also covers 4090 and 3090.

2. Submit one job per queue:
       sbatch --gpus=1 -p gpu_5090 ./run_bench.sh
       sbatch --gpus=1 -p gpu_4090 ./run_bench.sh

3. Watch with `parajobs`; results land in bench_hydrograd_<jobid>.out and in
   gpu_bench.json / epanet_ref_bench.json.

Notes
- Nothing is run on the login node; the guide forbids it and the benchmark
  would be meaningless there anyway.
- Each job gets a whole card, so unlike a shared workstation there is no need
  to check occupancy first.
- The CPU arm runs on the 6 to 8 cores the queue allocates, so CPU timings are
  not comparable with the 24-thread laptop or the 64-thread server. The GPU
  timings and the GPU-minus-CPU deviation are.

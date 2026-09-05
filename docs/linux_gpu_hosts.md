# Linux and GPU hosts: what was verified, and what changes off Windows

The bit-level claims in this repository were measured on Windows x64 against
the `epanet22.dll` that ships inside WNTR. This page records what the same
code does on the Linux hosts the experiments were run on, so that a user does
not mistake a platform difference for a defect. Host names and account paths
are deliberately omitted; every number below is copied from the raw records
under `data/gpu/` and `data/*_wip.txt` named in each section.

## Environments that were built and verified

| host class | GPU | driver / CUDA | torch | numpy / scipy / wntr | nvmath-python / cuDSS | record |
|--|--|--|--|--|--|--|
| workstation (reference) | RTX 5060 Laptop | 13.2 | 2.13.0+cu132 | 2.5.1 / 1.18.0 / 1.5.0 | not installed (cuDSS route unavailable) | `data/regression_report.txt` |
| cluster nodes, Slurm | RTX 5090 32 GiB, RTX 4090 | cu128 wheels | 2.11.0+cu128 | see job headers | 1.0.0 / 0.8 | `data/gpu/5090_*.out`, `data/gpu/4090_*.out` |
| lab server, 3 cards | Tesla V100 32 GiB (sm_70) | 580 (CUDA 13.0 capable), cu126 wheels | 2.14.0+cu126 | 2.4.6 / 1.17.1 / 1.5.0 | 1.0.0 / 0.8.0.10, `cuda-core` pinned to 1.1.1 | `data/gpu/v100_*_20260903.*` |
| lab server, 4 cards | RTX 3090 24 GiB (sm_86) | 550.90.07 / CUDA 12.4 | 2.6.0+cu124 | 2.4.6 / 1.17.1 / 1.5.0 | 1.0.0 / 0.8.0.10 | environment record only |

The two lab-server recipes are `deploy/v100/setup_env.sh` and, for a
driver-550 host that is capped at CUDA 12.4, the same recipe with the cu124
torch wheel and `nvidia-cublas-cu12==12.4.5.8` re-pinned after installing
`nvmath-python[cu12]` (that extra upgrades cuBLAS to 12.9 and makes `pip check`
complain against torch's pin). The Slurm bundle for the cluster runs is
`deploy/paracloud/`.

### Pins that matter

- **torch wheel and the GPU generation.** cu128 is required for Blackwell
  (RTX 5090, sm_120) and also covers sm_86/sm_89. The cu128 and later wheels
  drop sm_70, so a Volta card needs a cu126 wheel (every 2.6.0 to 2.14.0+cu126
  wheel still carries sm_70; `setup_env.sh` asserts it).
- **`cuda-core==1.1.1` on Volta with driver 580.** `nvmath-python[cu12]` pulls
  in `cuda-core` 1.2.0, whose pinned-memory pool raises
  "CUDA device 0 does not support the requested host memory pool" on that
  combination and fails every explicitly batched `DirectSolver.solve()`
  (`scripts/regression_gpu.py` went 0/54). Pinning 1.1.1 restores 54/54.
- **`CUBLAS_WORKSPACE_CONFIG=:4096:8`** is exported by `scripts/regression_gpu.sh`
  and `deploy/v100/env.sh`; keep it when comparing runs.
- **`DGGA_NETS`** tells `scripts/regression_gpu.py` where the public networks
  are (it looks in `$DGGA_NETS`, then `networks/public/`, each with a
  `_cleaned/` subdirectory first). The L-TOWN items additionally need the
  City D model under the file name `City_D.inp`; a symbolic link to
  `datasets/city_d.inp` is enough.

## What passes on Linux, and what does not

`scripts/regression_gpu.py` (the cuDSS-route invariants, 54 items) passed
54/54 on the V100 host, twice on two different cards
(`data/gpu/v100_regression_gpu_20260903.out`). The smoke tests in `tests/`
pass on every host above.

`scripts/regression_all.py` does **not** reach its Windows result on Linux:
42/54 on the V100 host, 468 s (`data/gpu/v100_regression_all_20260903.txt`;
the same commit gave 53/54 on the workstation the same morning, the one miss
being the anonymisation guard on a frozen manuscript tree that is not part of
this repository). Eleven of the twelve Linux misses are bit-level comparisons
against the EPANET binary bundled with WNTR, with iteration counts and link
statuses equal in every one of them and values drifting at the 1e-7 to
1e-5 ft level:

| item | V100 host (Linux) | Windows, same commit | gate |
|--|--|--|--|
| 1 align City D / City D + emitters | 1.217e-05 / 1.387e-05 ft | 1.421e-14 ft | H < 1e-6 ft |
| 2 replay EXA6 / City H / ky5 | 4.692e-07 / 7.029e-07 / 1.842e-08 ft, tank inflow off on EXA6 and ky5 | 0 to 1.4e-14 ft | H < 1e-6 ft per frame |
| 3 EPS EXA5 / EXA6 / City H | 7.071e-07 / 1.000e-06 / 7.029e-07 ft, statuses and pump settings identical | 5.7e-14 / 0 / 1.4e-14 ft | H < 1e-6 ft per frame |
| 3 EPS ky5 | control-event times differ at 4 of 31 frames (49189/60927/75716 s against 49199/60731/85103 s) | integer-identical | equality |
| 7 L-TOWN CMH+PRV frame 0 | 1.806e-06 ft, 7.104e-07 cfs, iterations 17/17 | pass | H < 1e-6 ft |
| 11 L-TOWN user-unit outlet | 5/785 heads, 18/909 flows not bit-identical | 785/785, 909/909 | bit-identical |

The 23 synthetic networks, the D-W network, the three-way gradient check,
the batch-consistency check, the 22-item adversarial audit, the exact
symmetry guard (72/72), the schedule mutants and the D-W gradient check all
pass on Linux as well.

The attribution (`data/gpu/v100_libm_attribution_20260903.txt`) has three parts:

1. `dgga/solver.py` calls `pow` and `log` through `msvcrt` on Windows so that
   transcendental results match the DLL it is compared against; on Linux it
   falls back to glibc's `math.pow` / `math.log`. Probing 25 000 arguments at
   each of the eight exponents the solver uses, the two libraries differ by
   exactly one ULP in 0.03 to 0.25 per cent of calls.
2. EPANET itself is not bit-stable across platforms: rebuilding the reference
   solutions with WNTR's Linux `libepanet22.so` and comparing them with the
   Windows references gives max |dH| = 8.41e-06 ft on City D (only 0.18 per
   cent of heads bit-equal) and 7.14e-06 ft on L-TOWN (2031 frames), with
   equal iteration counts; on ky5 the control-event times shift by up to 3600 s.
3. Linux `dgga` against the Linux library: 2.11e-05 ft on City D (still over
   the gate), 7.93e-07 ft on L-TOWN frame 0 (passes).

So the 1e-6 ft gate expresses "bit-level replication under one C runtime",
which is below EPANET's own cross-platform spread. The bit-level figures in
the README and the paper are Windows figures; the GPU experiments (sensor
placement, calibration, leak inversion, the batched and sparse routes) do
not depend on bit-level agreement with the DLL and were run on the Linux
hosts. If the 53/53 gate is ever needed on Linux, the reference arrays would
have to be rebuilt from the Linux binary and the exemption logic re-audited;
that has not been done.

## Running jobs on a host without a scheduler

The pattern used on both lab servers: `nohup`, one log per job, an `rc` file
written when the job exits, and the hostname, `CUDA_VISIBLE_DEVICES` and wall
time recorded at the top of the log. On a shared card, check `nvidia-smi`
before launching and again after finishing; a GPU timing taken on a busy card
is not a measurement (`data/gpu/README.md` records one such trap).

```bash
S=$HOME/scratch
nohup bash -c "source deploy/v100/env.sh; s=\$(date +%s); \
  CUDA_VISIBLE_DEVICES=1 python -X utf8 scripts/<job>.py > $S/<job>.log 2>&1; \
  echo \"rc=\$? secs=\$(( \$(date +%s) - s )) host=\$(hostname)\" > $S/<job>.rc" \
  >/dev/null 2>&1 < /dev/null &
```

Note that `deploy/v100/env.sh` assumes the project is checked out at
`/mnt/sda/$USER/hydrograd` and the public networks are staged at
`/mnt/sda/$USER/scratch/p2nets`; edit the two paths for your host.

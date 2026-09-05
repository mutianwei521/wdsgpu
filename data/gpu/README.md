# Cross-GPU measurements

`x_gpu_bench.py` and `x_epanet_ref_bench.py` are self-contained: they parse the
network from `.inp` and use the CPU float64 result as their own reference, so
they need neither the reference cache nor a Windows build. `deploy/paracloud/`
holds the Slurm bundle used for the 4090 and 5090 runs.

## Results, city_d (542 nodes, 554 links), B = 256, float64

| | RTX 5060 Laptop | RTX 3090 | RTX 4090 | RTX 5090 |
|---|---|---|---|---|
| CPU cores allocated | 24 | 64 | 6 | 8 |
| HydroGrad GPU (ms/scenario) | 2.179 | 0.997 | 0.641 | **0.461** |
| GPU minus CPU deviation (ft) | 1.02e-5 | 1.56e-5 | 9.42e-6 | 2.16e-5 |
| float32 drift (ft) | 126.6 | 126.6 | 126.6 | 126.6 |
| EPANET solve only (ms) | 0.178 | 0.457 | 0.547 | 0.430 |
| EPANET end to end (ms) | 1.100 | 1.772 | 2.122 | 1.586 |
| ours / EPANET solve only | 12.2x | 2.18x | 1.17x | 1.07x |
| ours / EPANET end to end | 1.98x | 0.56x | 0.30x | 0.29x |

The laptop column is Section `sec:performance` of the paper; the other three
were measured for this comparison. Raw job output is in `4090_bench.out` and
`5090_bench.out`.

## What four machines say that one cannot

**The GPU-CPU deviation is a property of the hardware.** It spans 9.42e-6 to
2.16e-5 ft, a factor of 2.3, and does not order by generation: the 4090 sits
closest to its own CPU and the 5090 furthest. Every value is seven orders of
magnitude above the 1e-12 ft at which the replica path meets the reference
engine. The differentiable path does not reproduce itself across GPUs, which is
the direct argument for keeping the bit-exact path on the CPU.

**float32 fails identically on all four**, 126.6 ft to four significant figures
on unrelated architectures, with iterations rising from 4-5 to 14-25. That
failure belongs to the problem.

**The comparison with EPANET has no single answer.** It runs from 12.2x slower
than EPANET's solve time on the laptop to parity on the 5090. Part of that is
the reference moving: EPANET is single-threaded and its own end-to-end cost
ranges from 1.100 to 2.122 ms across these hosts.

## Traps worth knowing about

- **Check GPU occupancy first.** The first 3090 run landed on a card already at
  93 per cent utilisation and reported 1.535 ms, 54 per cent slow. A GPU timing
  on a shared host without checking `nvidia-smi` is not a measurement. Slurm
  jobs get a whole card, so this only bites on the workstation.
- **Do not read the GPU-over-own-CPU ratio as a hardware figure.** The 4090
  scores 68.9x and the 5090 37.6x only because their queues allocate 6 and 8
  CPU cores. It measures the CPU allocation.
- **Batching is not optional.** At B=1 the GPU loses to the CPU on both cluster
  nodes, 0.86x and 0.95x.
- **Both cluster jobs wrote to the same `gpu_bench.json`**, so the JSON holds
  whichever finished last. The `.out` files are the complete record.
- wntr differs across hosts, 1.3.2 on the workstation against 1.5.0 on the
  cluster, but it changes nothing here. The reference timings call the same
  EPANET 2.2 shared library, which wntr only ships rather than implements, and
  the parser produced an identical network on every host: N=542, Nj=541, L=554
  throughout.

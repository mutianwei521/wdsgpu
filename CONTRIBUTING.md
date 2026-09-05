# Contributing to HydroGrad (`dgga`)

Thanks for taking an interest. This project has one unusual constraint that
shapes everything below: **`dgga` is a transcription of EPANET 2.2.0, not an
independent hydraulic solver.** A change that makes the code cleaner, faster or
more "correct" but moves the output away from EPANET's is a regression.

## Ground rules

1. **The EPANET C source is the specification.** When you add or change a
   numerical formula, cite the file and line it comes from
   (`hydcoeffs.c:673-791`, `input1.c:581-583`, ...) in a comment, the way the
   existing code does. Do not reconstruct a formula from a textbook or from
   memory: transcribe it.
2. **No silent approximations.** If an EPANET feature is not implemented,
   raise `NotImplementedError` at construction time. Never compute something
   plausible-looking for an input you do not fully support. See the
   `PBV`/`GPV`/PDA/volume-curve guards in `dgga/solver.py`, and the six guards
   on the cuDSS route.
3. **The three defaults are frozen.** `mode="epanet"`, `assemble="dense"` and
   `linear_solver="dense"` reproduce their recorded numbers bit for bit; every
   new capability is reached through a new keyword. A pull request that changes
   a default path's numbers, even by one ulp, needs the regression suite to say
   so and a reason that survives review.
4. **Internal units everywhere.** All arrays that cross a module boundary are
   `float64` in EPANET's internal units: ft for heads and lengths, cfs for
   flows. Conversions happen at the parse/report boundary only, using the
   constants transcribed in `dgga/units.py`.
5. **Network data.** The only network files this repository redistributes are
   the 23 synthetic networks under `networks/random_main` and
   `networks/random_small` (generated, see `networks/synthetic/README.md`) and
   the two anonymised operational models under `datasets/` (CC BY 4.0). Do not
   add third-party benchmark files: add a source URL and a SHA-256 to
   `scripts/fetch_benchmarks.py` instead, and record the licence in
   `THIRD_PARTY.md`. Do not add anything that could identify the two utilities.
6. **No fabricated numbers.** Every figure in the README, in `data/*.txt` or in
   a paper must be traceable to an artefact produced by a script in this
   repository. If something was not measured, say "not measured".

## Before you open a pull request

Run, and paste the actual output into the PR description:

```bash
python -m pytest -q tests                  # seconds, no downloads
python scripts/benchmark_sweep.py --stems <the networks your change touches>
python scripts/gradcheck_3way.py           # if you touched autodiff.py
python scripts/audit_grad_adversarial.py   # if you touched autodiff.py or solver.py
python scripts/check_symmetry.py           # if you touched assembly
```

`python scripts/regression_all.py` runs the full 54-item suite and writes
`data/regression_report.local.txt` (the archived `data/regression_report.txt`
is the evidence the paper cites and is never overwritten by default). Items
that need the undistributed EXA*/ky* models will report missing references on
a fresh clone; the 25 steady-state alignments, the gradient checks, the
symmetry guard and the schedule mutants run from the synthetic networks and the
released datasets alone. `python scripts/regression_gpu.py` (54 items) needs
CUDA and `nvmath-python`; it skips with exit code 2 and says why when they are
absent.

A change to the forward solver must keep the per-frame iteration counts, link
statuses and settings **exactly** equal to EPANET's. Head/flow agreement alone
is not enough: matching iteration counts is what distinguishes a faithful
transcription from a solver that happens to land in the same place.

## Code style

- Python 3.10 or later, `float64`, no new runtime dependencies beyond numpy /
  scipy / torch / wntr (the cuDSS route's `nvmath-python` stays optional).
- Module docstrings state the EPANET source files they mirror.
- Each module keeps its `if __name__ == "__main__":` smoke test working.
- Comments explaining *why* EPANET does something odd are valuable; comments
  restating what the line does are not.
- No em dashes in text (the corresponding author's rule); use a colon, a comma
  or parentheses.

## Reporting a mismatch with EPANET

Open an issue with: the INP file (or a public network that reproduces it), the
frame and node/link where the deviation appears, the deviation magnitude, and
the EPANET version and platform you compared against. Before filing, please
check whether the network has a loose `ACCURACY` setting; for those, run the
1-ulp control experiment described in the README's *Limitations* section, since
the reference solution itself may be the unstable quantity. Also check the
platform: the bit-level figures are Windows figures (`docs/linux_gpu_hosts.md`).

# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-05

First release under the name **HydroGrad** (package `dgga`, repository
`wdsgpu`), accompanying the paper (arXiv preprint, identifier to be added). The
repository is a fresh tree with a
single initial commit; it carries the code, the verification scripts, every
measurement report the paper cites, the 23 synthetic networks with their
generator, and the two anonymised operational models (CC BY 4.0).

### Release packaging

- Public repository name `wdsgpu`; software name HydroGrad; package name
  `dgga` unchanged. `pyproject.toml` version 0.2.0, author list of the
  paper, optional extras `cudss` (sparse GPU route) and `baselines`
  (CMA-ES baselines). `CITATION.cff` added.
- `networks/random_main` and `networks/random_small` (23 synthetic networks)
  are now redistributed, with the generator and a SHA-256 manifest under
  `networks/synthetic/`; the generator regenerates all 23 files byte for byte
  through WNTR's EPANET 2.2 library.
- `datasets/` (City D, City H, leak work orders; CC BY 4.0) shipped byte for
  byte with its own `SHA256SUMS.txt`.
- `data/`: the measurement reports cited by the paper (benchmark and
  regression reports, gradient checks, exemption diagnostics, the L-TOWN
  mainline evidence and raw cluster job outputs under `data/gpu/`, the
  calibration, placement, augmentation, clustering and optimiser records) and
  the paper tables as CSV under `data/tables/`. Working notes of the separate
  unrolled-solver research line are not part of this release.
- `scripts/`: the regression and guard suites, the reference builders and the
  fetch script, the gradient and adversarial checks, the GPU measurement
  drivers, and the application drivers named in the paper. The
  anonymisation guard and the dataset builder read their private mapping from
  the `HYDROGRAD_NAME_MAP` environment variable instead of a fixed path.
- `tests/`: eight CPU smoke tests that run on a fresh clone in seconds.
- `docs/linux_gpu_hosts.md`: what the suite does on Linux and on each GPU
  host, with the libm attribution; `docs/repository_map.md`.
- Typography: em dashes removed from every text file in the repository (the
  Chinese working records and code comments included); no numerical content
  changed.

### Added

- **Coherence-driven sensor augmentation** (`dgga.placement.coherence_augment`,
  `scripts/augment_coherence.py`, `scripts/augment_coherence_v100.sh`; record in
  `data/augment_coherence_wip.txt` / `data/augment_coherence.json`): the
  placement that Section 3.6 said "was not attempted". S0 stays fixed and k
  sensors are added greedily to minimise
  J(S) = sum_{i<j} -log(1 - mu_ij(S)^2 + 1e-12), the pairwise log-volume of the
  column-normalised leak-signature dictionary on the rows S0 + S (every pair of
  single-leak hypotheses, all candidates weighted alike). Its only inputs are
  the dictionary, S0 and the candidate-position pool; the leak identities never
  enter. The main result uses the fair pool (junction minus S0 minus the three
  leak nodes), so a sensor sitting on the leak node itself, which explained
  every 1/3 of the previous round, cannot be the design; the original pool is
  reported separately, as are the (max mu, J) lexicographic variant, the
  previous D-optimal / coverage orders and random +k draws under the same
  metric. `--stage mechanism` is a truth-dependent diagnostic computed after
  selection: a single-sensor scan over the whole pool and a five-step oracle
  greedy on each true leak's rival coherence.
  - Selection (workstation, seconds): City D fair pool +80 brings J 2576 to
    1554, the median coherence 0.928 to 0.760 and the >0.999 pairs 48 to 16;
    L-TOWN +80 brings J 2944 to 2302, the median 0.826 to 0.752, >0.999 pairs
    42 to 17. Hanoi runs as the smoke test (31 junctions, J 1216 to 856).
  - Mechanism: the rival coherence of the lost leaks has a placement floor. City
    D T2 (rival two hops away): 0.9999997 with S0, best single fair-pool sensor
    0.99997, five oracle steps 0.99997, and only the excluded leak node itself
    reaches 0.99905; T3 (rival three hops away, already carrying an S0 sensor):
    0.99939, best 0.99933, oracle 0.99932, the leak node itself 0.99948. L-TOWN
    T1: 0.999965, best single 0.99949, oracle 0.99919, leak node 0.99887; T2:
    0.99923 to 0.99841 / 0.99646. These pairs are 14 and 7 of J = 2576 (City D)
    and 10 and 6 of 2944 (L-TOWN), so no truth-independent objective has a
    reason to spend sensors on them, and no fair-pool sensor could separate
    them anyway. Given the original pool the objective does pick the leak
    nodes (they are candidate nodes: 13 of the first 20 City D positions and
    15 of 20 on L-TOWN land on candidate leak nodes).
  - L-TOWN leak inversion rerun (V100, csr+cuDSS, 12 configurations x 2 noise
    groups, 1372-1444 s per card; S0 reproduces the 5090 record digit for
    digit): every configuration, coherence-driven +5/10/20/40/80, the max
    variant, and five random +20 draws, returns S0's verdict, top-1 true and
    1/3 in the top three, with the two lost leaks at ranks 16-19 and 26-33
    (rank 5-7 at +80). What the coherence design does change is the amplitude
    of the found leak, C 0.17 (S0) to 0.44/0.58/0.68 at +20/40/80 against a
    true 1.5, as its rival coherence falls 0.987 to 0.942.

- **Hostile verification of the augmentation package** (`scripts/audit_augment/`,
  own implementations that do not call `dgga.placement`, `place_sensors.py` or
  `augment_suite.py`; record in `data/audit_augment_wip.txt`):
  - `hv_fisher.py` re-derives the census criterion, the Fisher matrix (explicit
    inverse), the eps-rank and 1e-2-subspace CRLB and the posterior std from the
    raw sensitivity cache for three randomly drawn k (seed 20260903: 5/10/80),
    checks monotonicity at every one of the 80 steps of both objectives, the
    same-threshold / same-sigma question (t=0 census == 25-frame census of S0,
    279/149/41/6 partition), the Section 3.6 anchors (from-scratch D-optimal
    32/43/52/68/73, random median 9/18.5/31.5/43/53, k=40 loses 38, Hanoi CRLB
    table) and the sigma-ladder errors with a regenerated truth: everything
    agrees with the stored JSONs; on the V100 host the same script agrees except
    CRLB_ident at +10 D-optimal (relative 1.1e-6, the eps-rank quantity is
    dominated by directions at ~1e-13 sigma_1 and is not a meaningful number).
  - `hv_shadow.py`: `dgga.placement` of the baseline commit (git archive 8b5e4f7)
    and of the working tree produce 180 bit-identical arrays on the City D and
    Hanoi caches; `bayes_dopt_augment(S0 = empty)` reproduces `bayes_dopt_greedy`
    bit for bit.
  - `hv_leak_fair.py` and `hv_leak_control.py` (13 control reruns on the V100
    host, 3093-3848 s each, the recorded inverter untouched): the recipe of the
    work-order case is replayed and matches; but the City D candidate pool
    contains the three leak nodes, the coverage order places a sensor on the T2
    leak node at step 3 (D-optimal at step 5), and the 1/3 of every augmented
    set is that sensor: removing it from +20 coverage returns 0/3, S0 plus that
    one sensor alone returns 1/3, and twenty random additions recover T2 in 0
    of 5 draws. T1's footprint is below the noise at every junction (confirmed).
    T3 is not a physics limit: 3 of 5 random +20 sets and D-optimal +20 without
    the T2 sensor recover it (rival coherence 0.9994-0.9997 throughout), while
    a sensor on the T3 node itself does not. Verdict: identifiability recovery
    (A, D) and calibration at sigma <= 0.1 (B) are verified; leak recovery (C)
    is not, and the 1/3 results must not be presented as an augmentation effect.
  - `hv_noids.py`: independent zero-identifier scan of every readable output;
    `hv_report.py` assembles the record. regression_all non-guard 53/53 on the
    workstation; frozen trees unchanged; `dgga/autodiff.py` untouched.

- Augmentation **suite driver** `scripts/augment_suite.py` (`--net city_d |
  pub_hanoi`, stages `verify | calib | leak | report | fig`): recomputes the
  augmentation curves independently from the sensitivity cache and the stored
  sensor orders and cross-checks them against `placement_augment_<stem>.json`
  (monotonicity re-asserted step by step), reruns the sigma-ladder calibration
  with S0 and S0 + k (`calibrate.py` engine, keys `AUG_*`), reruns the City D
  work-order leak case with the augmented sensor sets (dual-denominator
  coherence, top-1 / top-3, a `demo40` reproduction group that replays the
  recorded `demo_leak_inversion.json` run digit for digit), and writes a
  readable `data/augment_suite_<net>.json` / `_wip.txt` / `fig_augment_<net>.png`.
  Every readable output is scanned before it is written: any dict key or string
  equal to a node or link identifier of the network aborts the run.
  `scripts/calibrate.py` gains the City D placements `ga40` / `augdopt<k>` /
  `augcover<k>` and `scripts/demo_leak_inversion.py` accepts sensor sets of any
  size (both default paths unchanged). Virtual augmentation only: a design and
  simulation study, not a field installation.

  City D results (calib and leak stages run on the V100 server, dense CPU path,
  nine independent work trees in parallel, 15 calibrations of 980-1860 s and
  9 leak cases of 1110-2460 s; verify/report/fig on the workstation;
  `data/augment_city_d_wip.txt`, `data/augment_suite_city_d.json`,
  `data/leak_augment_city_d.json`, `data/calib_gc1_city_d.json` runs `AUG_*`):
  - Augmentation curve re-verified independently, zero monotonicity violations:
    coverage recovers 37/62/95/133/149 of the 149 sensor-starved pipes at
    +5/10/20/40/80 (80 % at +32, all at +56), D-optimal 21/23/35/52/76;
    reselecting |S0|+k from scratch loses 38/38/24/21/15 pipes S0 could see.
  - Sigma ladder with S0 and S0+k: the recovered pipes move off the prior
    (sigma 0.1: median |dC| 11.5 vs prior 21.9 for +20 D-optimal, 13.2 vs 18.3
    for +40; 15.5 vs 19.1 and 18.4 vs 19.5 for +20/+40 coverage); the
    well-conditioned-subspace C-RMSE falls 20.2 (S0) to 8.5 (+20 coverage);
    the network-wide informative C-RMSE barely moves (25.6 to 23.2-24.3), and at
    sigma 0.3 the +20 configurations are worse than S0 on that metric.
  - Work-order leak case, 0.1 ft noise: the recorded run is reproduced digit
    for digit on the server (same support, final MSE relative difference
    4.4e-15); S0 finds 0/3; every augmented set (coverage and D-optimal,
    +20/+40/+80) finds exactly 1/3 (T2, flow error 1-4 %), top-1 never.
    Median dictionary coherence falls 0.93 to 0.79-0.90 and T1's strongest rival
    from 0.96 to 0.34, but T1 (3.0 L/s) leaves a pressure footprint of at most
    0.09 ft anywhere in the network (0.003-0.01 ft RMS on the sensors), below
    the noise, and T3 keeps a near-twin at coherence 0.9993-0.9997 in every
    sensor set; both are physics limits, not placement limits. Reported as is.
  - Hanoi: the suite's calib stage reproduces `calibrate.py --stage l1hanoi_aug`
    exactly (`AUG_augcover20` = `AUG_augdopt20`, identical sensor set).

- Sensor **augmentation** mode (`dgga.placement.bayes_dopt_augment`,
  `recovery_report`, `posterior_std`; CLI `scripts/place_sensors.py --stage
  augment`): the existing sensor set S0 is kept fixed and k sensors are added
  greedily under either the Bayesian D-optimal objective or a coverage
  objective (number of pipes crossing the census identifiability criterion,
  `max|S| > atol`, same criterion and same sigma as the census). Monotonicity
  is asserted step by step (identifiable set never shrinks, posterior variance
  never rises) and a from-scratch reselection of |S0|+k sensors is reported as
  the control. Outputs `data/placement_<stem>.json["augment"]` (with candidate
  indices) and `data/placement_augment_<stem>.json` (readable, no node or
  candidate identifiers). Existing functions and the default stages are
  untouched; `bayes_dopt_augment(S, [], k)` reproduces `bayes_dopt_greedy`
  bit for bit. This is virtual augmentation on the calibrated model: a design
  and simulation study, not a field installation.

- Author, affiliation and repository metadata across `LICENSE`,
  `pyproject.toml`, `README.md`, `datasets/README.md` and `paper/main.tex`.
- Six bibliography entries in `paper/refs.bib` (`george1981computer`,
  `liu1985modification`, `sherman1950adjustment`, `swamee1976explicit`,
  `tropp2007signal`, `wagner1988water`) that `main.tex` cited with no matching
  record. Volume / issue / pages / DOI are still marked `% MISSING:`.

### Changed

- The two real networks are anonymised as **City D** / **City H** everywhere
  (previously Utility-A / Utility-B): `scripts/make_paper_figs.py` and all
  regenerated `paper/figs` and `paper/tables` artefacts, plus `paper/main.tex`.
- Generated `paper/tables/*.tex` headers are now pure ASCII, so the LaTeX
  sources build under pdflatex without non-ASCII bytes in comments.
- Data-availability text in `paper/main.tex`, `README.md` and `THIRD_PARTY.md`
  rewritten: the two utility models and the 256 leak work-order records are now
  published under `datasets/` (CC BY 4.0), with the anonymisation described and
  its hydraulic losslessness stated as a verified measurement
  (`max|delta| = 0.0` over the full 25-frame EPS on both networks).
- `scripts/make_public_dataset.py`: dataset README cites the repository URL;
  the output audit gained an allow-list so the author's own repository URL is
  not flagged as a leaked local path. Rebuild is byte-identical for every
  `.inp` / `.csv` / `.json` in `datasets/`.

## [0.1.0] - 2026-08-09

First public packaging of the differentiable EPANET 2.2 hydraulic engine.

### Added

- `dgga.parse` - EPANET INP → `Net`, with multi-category demands, pattern
  stepping, cylindrical tanks, pump curve fitting (1-point / 3-point / CUSTOM /
  constant-horsepower) and structured `[CONTROLS]`.
- `dgga.units` - the 10 EPANET flow units and the conversion constants
  transcribed from `types.h:68-83`.
- `dgga.solver` - `GGASolver`, the forward global gradient algorithm, in two
  modes: `epanet` (EPANET's own sparse Cholesky ordering, per-sample, bit-exact)
  and `dense` (batched torch Cholesky, CPU/GPU). Hazen–Williams,
  Darcy–Weisbach and Chezy–Manning head loss; pipes, CV pipes, TCV, PRV, PSV,
  FCV, pumps, tanks, reservoirs, emitters; EPANET's status machine.
- `dgga.smatrix` - the MMD reordering and `linsolve` of `smatrix.c`.
- `dgga.eps` - `EpsDriver`, a full extended-period simulation: EPANET's time
  step rule, tank level integration, warm starts.
- `dgga.rules` - the `[RULES]` engine, including priority arbitration.
- `dgga.epanet_ref` / `dgga.reference` - double-precision `EN_*` ctypes
  bindings used to generate reference solutions.
- `dgga.autodiff` - two backward routes: `ImplicitGGASolve` / `implicit_solve`
  (implicit function theorem, sparse LU adjoint) and `solve_unrolled`
  (fixed-`K` unrolled GGA). Differentiable w.r.t. demands, reservoir heads,
  pipe resistance, H-W roughness, emitter coefficients and pump `h0`/`r`/speed.
  Batched and CUDA-capable.
- `scripts/` - reference builders, alignment checks, the 52-network benchmark
  sweep, the 50-check regression suite, the 22-item adversarial gradient audit,
  the three-way gradient cross-check, the batch/GPU benchmark, and a leak
  inversion demo.
- `scripts/fetch_benchmarks.py` - rebuilds `networks/public/` from upstream
  sources with SHA-256 verification and per-file licence reporting.
- Packaging: `pyproject.toml`, MIT `LICENSE`, `README.md`, `THIRD_PARTY.md`,
  `CONTRIBUTING.md`, `.gitignore`.

### Verified

- 52 / 52 benchmark networks pass; 47 bit-exact at ≤ 5.684e-14 ft
  (`data/benchmark_report.txt`).
- 50 / 50 regression checks pass (`data/regression_report.txt`).
- Gradients agree across the unrolled route, the implicit route and central
  finite differences to 5.85e-08 / 7.46e-07 relative; `torch.gradcheck` passes.

### Known limitations

PBV and GPV valves, pressure-driven analysis, tanks with volume curves and the
water-quality module are not implemented. Bit-exactness is measured on Windows
x64 against the `epanet22.dll` bundled with `wntr`; other platforms should
expect ~1e-6 ft agreement. Four networks pass under a documented 1-ulp control
experiment rather than on the 1e-6 ft threshold. See the README.

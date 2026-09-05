# Repository map

What each directory holds, which script produced which report, and what is
deliberately not here.

## Code

| path | what it is |
|--|--|
| `dgga/` | the package. `parse.py` (INP to `Net`), `units.py`, `solver.py` (`GGASolver`: the EPANET-replica path, the dense batched path with its status machine, the CSR assembly and the cuDSS route), `smatrix.py` (EPANET's sparse ordering and solve), `eps.py` (extended-period driver), `rules.py` (rule engine), `autodiff.py` (`implicit_solve` / `ImplicitGGASolve`, the GPU implicit adjoint, `solve_unrolled`, `unrolled_grad_health`), `epanet_ref.py` / `reference.py` (double-precision `EN_*` bindings and reference-solution builder), `sensitivity.py`, `calib.py`, `placement.py`, `cluster.py`, `optim2.py` (the application modules), `symbolic.py`, `precond.py`, `ggaformer.py`, `mlds.py` (a separate learned-solver research line; present because the package is shipped whole, not used by the paper). `CONTRACT.md` is the original Chinese interface contract of the package. |
| `tests/` | `python -m pytest -q tests`: eight CPU smoke tests on a synthetic network and on the released manifests. |
| `networks/` | `random_main/`, `random_small/` (23 synthetic networks, redistributed); `synthetic/` (their generator, `SHA256SUMS.txt`, README); `public/` (created by `scripts/fetch_benchmarks.py`, git-ignored). |
| `datasets/` | City D, City H and the City D leak work orders, CC BY 4.0, with `README.md` and `SHA256SUMS.txt`. |
| `deploy/` | `paracloud/`: self-contained Slurm bundle for the cross-GPU benchmark (copy `dgga/` next to it); `v100/`: environment recipe, activation file and cuDSS smoke test for a Volta host. |
| `assets/` | the two README figures, rendered from the manuscript PDFs. |
| `docs/` | this map and `linux_gpu_hosts.md`. |

## Scripts, grouped by what they establish

Verification of the forward solver (Windows x64 against WNTR's `epanet22.dll`):

| script | produces |
|--|--|
| `fetch_benchmarks.py` | `networks/public/` from upstream sources, SHA-256 verified |
| `inventory_public.py` | `data/public_inventory.json` |
| `build_public_reference.py`, `build_random_reference.py`, `build_reference.py` | `data/reference/*.npz` (git-ignored) for the public, the synthetic and the private networks |
| `validate_reference.py` | mass-balance / demand / head-loss cross-checks of a reference |
| `align.py`, `align_eps.py`, `bench_partial_eps.py` | frame-by-frame alignment against the reference (steady, replay, autonomous EPS) |
| `benchmark_sweep.py` | `data/benchmark_report.txt` (52 networks) |
| `regression_all.py` | `data/regression_report.txt` (54 items; local runs go to `data/regression_report.local.txt`) |
| `check_symmetry.py`, `check_schedule.py`, `check_prv_release.py`, `check_taskd.py`, `check_no_realnames.py` | the guards inside the regression suite (exact symmetry of `A`, status-machine schedule with mutants, PRV release items, D-W / CMH / FCV capability, anonymisation) |
| `exempt_diag/` | the exemption diagnostics of the four loosely converged networks: `data/exempt/*.json` (1-ulp self-drift of the DLL, input bit-checks, tightened runs, Net6 full EPS) |
| `probe_conditioning.py`, `probe_status_headroom.py`, `audit_status_headroom.py` | `data/conditioning_report.txt`, `data/status_headroom.json` |

Gradients:

| script | produces |
|--|--|
| `gradcheck_3way.py`, `gradcheck_dw.py`, `gradcheck_b2.py` | the three-way cross-check (unrolled / implicit / central FD), the Darcy-Weisbach and the pump-network checks (regression items 5 and 8) |
| `audit_grad_adversarial.py`, `audit_d_adversarial.py`, `audit_prv_dw_adversarial.py` | the 22-item adversarial audit and its extensions (regression item 6) |
| `extfd_epanet.py`, `extfd_calib.py` | `data/extfd_epanet_report.txt`, `data/extfd_calib_report.txt`: finite differences driven through EPANET's own library against the analytic gradient |
| `sensitivity_check.py`, `p4_r4_probe.py` | sensitivity-matrix checks; usability of the truncated-K unrolled gradient |
| `adjoint_gpu/` | the GPU implicit adjoint: numerical proof of the Woodbury row replacement, local acceptance, cluster end-to-end table (`data/adjoint_gpu_wip.txt`) |
| `prv_port/` | PRV in the batched path: batch extremes, DLL comparison, frozen-status gradient check, shadow battery (`data/prv_port_wip.txt`) |
| `gate_b1_batch_sm.py`, `gate_b1_grad.py` | the batched status machine gate (`data/gate_b1_wip.txt`) |

Batched, sparse and GPU routes:

| script | produces |
|--|--|
| `bench_batch.py`, `bench_scaling.py`, `x_profile_dispatch.py` | `data/bench_batch_*.txt`, `data/bench_scaling.json`, `data/x_dispatch_profile.json` |
| `x_gpu_bench.py`, `x_epanet_ref_bench.py` (same files as `deploy/paracloud/`) | the cross-GPU table in `data/gpu/README.md` |
| `verify_csr_assemble.py`, `audit_p1_csr/` | CSR assembly bit-identical to dense assembly (`data/p1_csr_wip.txt`, `data/p1_csr_audit_wip.txt`) |
| `p2_cudss/`, `p2_f4/`, `p3_autograd/`, `p3_adversarial/`, `audit_p2_adversarial/` | the cuDSS route: acceptance, cache eviction, differentiability, adversarial rounds (`data/p2_*`, `data/p3_*`) |
| `p4_remeasure/`, `p4_corrections/`, `p4_closeout/`, `aud_p4/`, `xfinal/`, `rc_probe/` | the crossover tables, the two-factor decomposition, the memory tables and their audits (`data/p4_*`, `data/xf_final_wip.txt`, `data/rc_propagation_wip.txt`) |
| `p5_bignet/` | the scale axis beyond 1000 junctions (`data/p5_bignet_wip.txt`) |
| `regression_gpu.py`, `regression_gpu.sh` | the 54 cuDSS-route invariants (`data/gpu/*regp4*.out`, `data/gpu/v100_regression_gpu_20260903.out`) |
| `ltown_mainline/`, `mainline_v2/` | the L-TOWN forward, forward-plus-backward, memory and training-loop tables (`data/mainline_evidence.md`, `data/ltown_mainline_wip.txt`, `data/mainline_v2_wip.txt`, `data/tables/tab_ltown_*.csv`) |
| `aud_release/`, `aud_ajg/`, `adv_verify/` | hostile re-verification rounds of the PRV release, the GPU adjoint and the four application modules (`data/prv_release_audit_wip.txt`, `data/aud_ajg_wip.txt`, `data/hostile_verify_wip.txt`, `data/hostile_m*.json`) |

Applications (all on the released City D model, L-TOWN and Hanoi):

| script | produces |
|--|--|
| `calibrate.py` | roughness calibration engine and the sigma ladder (`data/calib_gc1_*.json`, `data/calib_augrand_city_d.json`) |
| `baselines_calib.py`, `compare_calib.py` | global-optimiser baselines and the paired comparison (`data/baselines_*.json`, `data/gd_comparison*`, `data/gd_audit_report.txt`) |
| `place_sensors.py`, `placement_metric.py`, `placement_metric_report.py` | Bayesian D-optimal placement, the census of identifiable coordinates and the augmentation significance (`data/placement_*`, `data/placement_metric_wip.txt`) |
| `augment_suite.py`, `augment_coherence.py`, `augment_public.py`, `augment_public_compare.py`, `coh_controls.py`, `audit_augment/` | sensor augmentation on City D, L-TOWN and Hanoi, the coherence-driven design, its random controls and the hostile verification (`data/augment_*`, `data/coh_controls*`, `data/leak_coh_*`, `data/audit_augment_*`, `data/audit_leak_control*`) |
| `cluster_localisation.py` | cluster-level leak localisation (`data/cluster_*`) |
| `optimizer_v2.py`, `optimizer_v2_report.py` | first-order optimiser study (`data/optimizer_v2_*`) |
| `demo_leak_inversion.py`, `leak_coherence_probe.py` | the leak-inversion demonstration (`data/demo_leak_inversion.json`, `data/leak_coherence.json`) |
| `make_public_dataset.py` | builds `datasets/` from the unpublished originals and verifies hydraulic losslessness (`--verify-only` needs both) |
| `make_paper_figs.py` | the manuscript figures and tables from `data/` (writes `paper/figs`, `paper/tables`) |

Most scripts print Chinese progress text and write Chinese-language working
records (`data/*_wip.txt`); the numbers, the file names and the code are what
the paper cites.

## What is not here

- The un-anonymised operational models and field records, and the private
  mapping used by the anonymisation guard and the dataset builder.
- The third-party benchmark networks (fetched, see `THIRD_PARTY.md`), the
  EPANET reference solutions (`data/reference/`, rebuilt by the scripts) and
  the per-network alignment logs.
- The manuscript sources and the separate unrolled-solver research line
  (its modules remain in `dgga/`; its scripts and records do not ship).

# -*- coding: utf-8 -*-
"""Smoke tests that run on a fresh clone with no downloads: they use the
synthetic networks shipped under networks/random_main and the released
datasets. Everything here is CPU-only and finishes in seconds.

    python -m pytest -q tests

The tolerances are the ones measured on the authors' machine (see
data/regression_report.txt for the full suite): the replica and the dense
path agree to reduction-order noise, batched and single-scenario solves are
bit-identical, the implicit adjoint matches a central finite difference to
better than 1e-6 relative, and the cuDSS route refuses rather than degrades.
"""
import hashlib
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dgga.parse import parse_inp            # noqa: E402
from dgga.solver import GGASolver           # noqa: E402
from dgga.autodiff import implicit_solve, solve_unrolled   # noqa: E402

INP = os.path.join(ROOT, "networks", "random_main", "rand_0009.inp")


@pytest.fixture(scope="module")
def net():
    return parse_inp(INP)


@pytest.fixture(scope="module")
def inputs(net):
    d = torch.as_tensor(net.demand_cfs_at(0))
    rh = torch.as_tensor(net.reservoir_head_ft_at(0))
    return d, rh


@pytest.fixture(scope="module")
def solvers(net):
    return (GGASolver(net, mode="epanet", inp_path=INP),
            GGASolver(net, mode="dense", inp_path=INP))


def test_replica_and_dense_path_agree(solvers, inputs):
    se, sd = solvers
    d, rh = inputs
    r_e = se.solve(d, rh)
    r_d = sd.solve(d, rh)
    assert int(r_e["iters"]) == int(r_d["iters"])
    assert bool(r_e["converged"]) and bool(r_d["converged"])
    assert float((r_e["head_ft"] - r_d["head_ft"]).abs().max()) < 1e-9
    assert float((r_e["flow_cfs"] - r_d["flow_cfs"]).abs().max()) < 1e-11


def test_batched_solve_is_bit_identical_to_single(solvers, inputs):
    _, sd = solvers
    d, rh = inputs
    factors = (0.9, 1.0, 1.1, 1.2)
    rb = sd.solve(torch.stack([d * f for f in factors]), torch.stack([rh] * len(factors)))
    for b, f in enumerate(factors):
        r1 = sd.solve(d * f, rh)
        assert int(rb["iters"][b]) == int(r1["iters"])
        assert float((rb["head_ft"][b] - r1["head_ft"]).abs().max()) == 0.0


def test_csr_assembly_matches_dense_assembly_bit_for_bit(solvers, inputs):
    _, sd = solvers
    d, rh = inputs
    r_dense = sd.solve(d, rh)
    r_csr = sd.solve(d, rh, assemble="csr", linear_solver="dense")
    assert int(r_dense["iters"]) == int(r_csr["iters"])
    assert float((r_dense["head_ft"] - r_csr["head_ft"]).abs().max()) == 0.0


def test_implicit_adjoint_matches_central_finite_difference(net, solvers, inputs):
    se, _ = solvers
    d, rh = inputs
    junc = np.flatnonzero(net.demand_cfs_at(0) > 0)
    i_theta, i_probe = int(junc[0]), int(junc[-1])
    dd = d.clone().requires_grad_(True)
    head, _flow, _emit = implicit_solve(se, dd, rh)
    head[i_probe].backward()
    g = float(dd.grad[i_theta])
    h = 1e-3 * float(d[i_theta])

    def head_at(delta):
        dp = d.clone()
        dp[i_theta] += delta
        return float(se.solve(dp, rh, accuracy=1e-12, max_iter=200)["head_ft"][i_probe])

    fd = (head_at(h) - head_at(-h)) / (2 * h)
    assert abs(g - fd) / abs(fd) < 1e-6


def test_unrolled_and_implicit_gradients_agree(solvers, inputs):
    se, sd = solvers
    d, rh = inputs
    junc = np.flatnonzero(d.numpy() > 0)
    i_theta, i_probe = int(junc[0]), int(junc[-1])
    d1 = d.clone().requires_grad_(True)
    implicit_solve(se, d1, rh)[0][i_probe].backward()
    d2 = d.clone().requires_grad_(True)
    K = int(se.solve(d, rh)["iters"]) + 5
    solve_unrolled(sd, d2, rh, K=K)["head_ft"][i_probe].backward()
    g1, g2 = float(d1.grad[i_theta]), float(d2.grad[i_theta])
    assert abs(g1 - g2) / abs(g1) < 1e-8


def test_cudss_route_refuses_instead_of_falling_back(solvers, inputs):
    se, sd = solvers
    d, rh = inputs
    with pytest.raises(ValueError):
        sd.solve(d, rh, assemble="dense", linear_solver="cudss")
    with pytest.raises(NotImplementedError):
        se.solve(d, rh, assemble="csr", linear_solver="cudss")
    if not torch.cuda.is_available():
        with pytest.raises(NotImplementedError):
            sd.solve(d, rh, assemble="csr", linear_solver="cudss")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_manifest(manifest, base):
    n = 0
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(None, 1)
            assert _sha256(os.path.join(base, name.strip())) == digest, name
            n += 1
    return n


def test_synthetic_networks_match_manifest():
    n = _check_manifest(os.path.join(ROOT, "networks", "synthetic", "SHA256SUMS.txt"),
                        os.path.join(ROOT, "networks"))
    assert n == 23


def test_released_datasets_match_manifest():
    n = _check_manifest(os.path.join(ROOT, "datasets", "SHA256SUMS.txt"),
                        os.path.join(ROOT, "datasets"))
    assert n == 5

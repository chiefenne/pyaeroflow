"""Stage 1: convection-diffusion with an exact solution on all mesh types."""

import numpy as np
import pytest

from pyaeroflow.cases import convection_diffusion as case
from pyaeroflow.physics.convection_diffusion import ConvectionDiffusion

LEVELS = (8, 16, 32, 64)
MODEL = ConvectionDiffusion()


@pytest.mark.parametrize("cell_type", ["quad", "polygon"])
def test_converges_and_is_accurate(cell_type):
    r = case.solve(MODEL, cell_type, 16, perturb=0.4)
    h = r.history
    assert r.converged
    assert h["residual"][-1] / h["residual"][0] < 1e-11
    assert len(h["iteration"]) - 1 < 40
    assert r.error < 2e-3


@pytest.mark.slow
@pytest.mark.parametrize("perturb", [0.0, 0.4])
@pytest.mark.parametrize("cell_type", ["quad", "triangle", "polygon", "mixed"])
def test_second_order(cell_type, perturb):
    results = [case.solve(MODEL, cell_type, n, perturb=perturb) for n in LEVELS]
    assert all(r.converged for r in results)
    orders = case.observed_orders(results)
    assert 1.8 <= orders[-1] <= 2.2, orders


@pytest.mark.slow
@pytest.mark.parametrize("cell_type", ["triangle", "polygon"])
@pytest.mark.parametrize(
    ("variant", "bounds"),
    [
        ({"limiter": "venkatakrishnan-wang"}, (1.8, 2.2)),
        ({"viscous_gradient": "edge-correction"}, (1.8, 2.2)),
        ({"spatial_order": 1}, (0.7, 1.3)),
    ],
    ids=["limiter", "edge-correction", "first-order"],
)
def test_variants(cell_type, variant, bounds):
    model = MODEL._replace(**variant)
    results = [case.solve(model, cell_type, n, perturb=0.4) for n in LEVELS]
    assert all(r.converged for r in results)
    orders = case.observed_orders(results)
    assert bounds[0] <= orders[-1] <= bounds[1], orders
    assert np.isfinite(orders).all()

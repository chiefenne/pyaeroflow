"""V1 plane Couette and V2 plane Poiseuille flow (periodic channels)."""

import numpy as np
import pytest

from pyaeroflow.cases import channel
from pyaeroflow.physics import incompressible as inc


@pytest.mark.parametrize(
    ("cell_type", "options"),
    [
        ("quad", {}),
        ("quad", {"cluster_y": "both"}),
        ("triangle", {}),
        ("mixed", {}),
    ],
    ids=["quad", "quad-clustered", "triangle", "mixed"],
)
def test_couette_exact_to_round_off(cell_type, options):
    """The linear profile is reproduced to round-off where the midpoint
    quadrature of the quadratic convective flux is exact (cells whose signed
    edge projections cancel: axis-aligned quads and their diagonal splits)."""
    r = channel.couette(8, cell_type, **options)
    assert r.converged
    assert r.max_error.max() < 1e-9


@pytest.mark.parametrize("cell_type", ["polygon", "quad"])
def test_couette_small_error_on_general_cells(cell_type):
    """Perturbed quads and polygons: second-order error, not round-off."""
    r = channel.couette(8, cell_type, perturb=0.4)
    assert r.converged
    assert r.max_error[inc.U] < 2e-3


def test_explicit_mode_matches_implicit():
    r = channel.couette(4, "triangle", perturb=0.3)
    explicit = r.problem.solve_explicit(cfl=1.5, max_iterations=5000, residual_drop=6.0)
    assert explicit.converged
    assert np.abs(explicit.u - r.w).max() < 1e-5


@pytest.mark.parametrize("cell_type", ["quad", "triangle"])
def test_poiseuille_converges(cell_type):
    r = channel.poiseuille(16, cell_type, perturb=0.4)
    assert r.converged
    assert len(r.history["iteration"]) - 1 < 60
    assert r.error[inc.U] < 2e-2


@pytest.mark.slow
@pytest.mark.parametrize("cell_type", ["quad", "triangle", "polygon", "mixed"])
def test_poiseuille_second_order(cell_type):
    results = [channel.poiseuille(n, cell_type, perturb=0.4) for n in (8, 16, 32, 64)]
    assert all(r.converged for r in results)
    orders = channel.observed_orders(results)
    assert 1.8 <= orders[-1] <= 2.2, orders

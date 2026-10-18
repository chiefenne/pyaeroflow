"""V3 Taylor-Green vortex and B2 DFG 2D-2 (unsteady, stage 4)."""

import numpy as np
import pytest

from pyaeroflow.cases import cylinder, taylor_green
from pyaeroflow.solver import bdf_coefficients


def test_bdf_coefficients():
    dt = 0.1
    assert bdf_coefficients(1, dt, "bdf2") == pytest.approx((10.0, -10.0, 0.0))
    assert bdf_coefficients(2, dt, "bdf2") == pytest.approx((15.0, -20.0, 5.0))
    assert bdf_coefficients(2, dt, "bdf1") == pytest.approx((10.0, -10.0, 0.0))


def test_taylor_green_short():
    r = taylor_green.solve(n=32, viscosity=0.05, time_step=0.05, end_time=1.0)
    assert r.converged
    assert r.velocity_error < 0.01
    assert r.pressure_error < 0.03
    assert r.kinetic_energy == pytest.approx(r.kinetic_energy_exact, rel=0.02)
    assert r.history["inner_iterations"].mean() < 4


@pytest.mark.slow
def test_taylor_green_time_order():
    """Error against a fine-dt run on the same mesh: BDF2 order 2, BDF1 order 1."""
    n, nu, t_end = 32, 0.05, 1.0
    inner = taylor_green.inner_settings(6.0)
    ref = taylor_green.solve(n, nu, 0.0125, t_end, inner=inner)
    area = ref.problem.mesh.cell_area
    for scheme, expected in (("bdf2", 2.0), ("bdf1", 1.0)):
        errors = []
        for dt in (0.2, 0.1, 0.05):
            r = taylor_green.solve(n, nu, dt, t_end, scheme=scheme, inner=inner)
            dv = r.w[:, 1:] - ref.w[:, 1:]
            errors.append(np.sqrt(np.sum(area[:, None] * dv**2) / np.sum(area)))
        orders = np.log2(errors[:-1]) - np.log2(errors[1:])
        assert np.all(np.abs(orders - expected) < 0.25), (scheme, orders)


@pytest.mark.slow
def test_dfg_2d2_coarse():
    """Periodic shedding on the coarse mesh (1.5k cells, 17 steps per period)."""
    r = cylinder.dfg_2d2(16, 12, time_step=0.02, end_time=8.0)
    assert r.converged and r.periodic
    assert 0.22 < r.strouhal < 0.32
    assert 2.9 < r.cd_max < 3.4
    assert 0.7 < r.cl_max < 1.2
    assert 2.0 < r.dp_half_period < 2.7


@pytest.mark.slow
def test_dfg_2d2_medium():
    """4.4k cells, dt = 0.01: St -7 %, cd_max -1 %, cl_max +5 %, dp -1.5 %."""
    r = cylinder.dfg_2d2(32, 20, time_step=0.01, end_time=8.0)
    ref = r.reference
    assert r.converged and r.periodic
    assert r.strouhal == pytest.approx(np.mean(ref["strouhal"]), rel=0.10)
    assert r.cd_max == pytest.approx(np.mean(ref["cd_max"]), rel=0.02)
    assert r.cl_max == pytest.approx(np.mean(ref["cl_max"]), rel=0.08)
    assert r.dp_half_period == pytest.approx(np.mean(ref["dp"]), rel=0.03)

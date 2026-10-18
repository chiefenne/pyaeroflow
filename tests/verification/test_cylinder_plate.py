"""V4 inviscid cylinder, B1 Blasius flat plate, B2 DFG cylinder in a channel."""

import numpy as np
import pytest

from pyaeroflow.cases import cylinder, flat_plate


def test_blasius_ode():
    fpp0, f = flat_plate.blasius()
    assert fpp0 == pytest.approx(0.332057, abs=1e-5)
    assert f(5.0)[1] == pytest.approx(0.99155, abs=1e-4)


def test_inviscid_cylinder_coarse():
    r = cylinder.inviscid(64, 32)
    assert r.converged
    assert np.abs(r.cp - r.cp_exact).max() < 0.12
    assert abs(r.cl) < 1e-6 and abs(r.cd) < 0.03
    assert r.cp.min() == pytest.approx(-3.0, abs=0.12)


@pytest.mark.slow
def test_inviscid_cylinder_converges_with_refinement():
    coarse = cylinder.inviscid(64, 32)
    fine = cylinder.inviscid(128, 48)
    assert fine.converged
    e_coarse = np.abs(coarse.cp - coarse.cp_exact).max()
    e_fine = np.abs(fine.cp - fine.cp_exact).max()
    assert e_fine < 0.6 * e_coarse
    assert abs(fine.cd) < 0.3 * abs(coarse.cd)


def test_dfg_2d1_coarse():
    r = cylinder.dfg_2d1(16, 12)
    assert r.converged
    assert r.cd == pytest.approx(r.reference["cd"], rel=0.02)
    assert r.dp == pytest.approx(r.reference["dp"], rel=0.06)
    assert r.cl == pytest.approx(r.reference["cl"], abs=0.002)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("m", "n_radial", "cell_type"), [(64, 32, "quad"), (32, 20, "triangle")]
)
def test_dfg_2d1(m, n_radial, cell_type):
    r = cylinder.dfg_2d1(m, n_radial, cell_type)
    assert r.converged
    assert r.cd == pytest.approx(r.reference["cd"], rel=0.006)
    assert r.dp == pytest.approx(r.reference["dp"], rel=0.02)
    assert r.cl == pytest.approx(r.reference["cl"], abs=0.0005)


# Cf is compared away from the leading edge (Blasius is asymptotic in Re_x)
# and away from the outlet corner, where the last plate face feels the outlet.
PLATE_RANGE = (0.2, 0.9)


def test_flat_plate_coarse():
    r = flat_plate.solve(80, 40, beta=4.0)
    assert r.converged
    sel = (r.x > PLATE_RANGE[0]) & (r.x < PLATE_RANGE[1])
    assert np.abs(r.cf[sel] / r.cf_blasius[sel] - 1.0).max() < 0.04
    assert np.abs(r.u_profile - r.u_blasius).max() < 0.02


@pytest.mark.slow
def test_flat_plate():
    r = flat_plate.solve()
    assert r.converged
    sel = (r.x > PLATE_RANGE[0]) & (r.x < PLATE_RANGE[1])
    ratio = r.cf[sel] / r.cf_blasius[sel]
    assert np.abs(ratio - 1.0).max() < 0.035
    assert abs(ratio.mean() - 1.0) < 0.015
    assert np.abs(r.u_profile - r.u_blasius).max() < 0.01

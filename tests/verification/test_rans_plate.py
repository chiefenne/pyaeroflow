"""R1: turbulent flat plate (NASA TMR domain), SA-neg vs the TMR values."""

import numpy as np
import pytest

from pyaeroflow.cases import tmr_flat_plate as tp
from pyaeroflow.solver import PseudoTimeSettings


def test_turbulent_plate_coarse():
    """48 x 48 cells: C_f(0.97) +1.2 %, C_D +2 % vs the TMR SA values."""
    r = tp.solve(48, 12, 48)
    assert r.converged and r.y_plus_max < 1.0
    assert r.cf_097 == pytest.approx(r.reference["cf_097"], rel=0.025)
    assert r.cd == pytest.approx(r.reference["cd"], rel=0.03)
    # Fully turbulent: no laminar Cf anywhere behind the leading-edge region.
    assert r.cf[r.x > 0.2].min() > 0.002


def test_turbulent_plate_coupled_matches_segregated():
    seg = tp.solve(24, 6, 32)
    cpl = tp.solve(24, 6, 32, coupling="coupled")
    assert cpl.converged
    assert cpl.cf_097 == pytest.approx(seg.cf_097, rel=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize(("nxp", "nxf", "ny"), [(96, 24, 64), (192, 48, 96)])
def test_turbulent_plate_refined(nxp, nxf, ny):
    """7.7k / 23k cells: C_f(0.97) +1.0 / +0.9 %, C_D +1.7 / +1.4 %."""
    r = tp.solve(nxp, nxf, ny)
    assert r.converged and r.y_plus_max < 1.0
    assert r.cf_097 == pytest.approx(r.reference["cf_097"], rel=0.015)
    assert r.cd == pytest.approx(r.reference["cd"], rel=0.025)


def test_bcm_transition_on_the_plate():
    """BCM (L3, step 1) on the plate at Re_L = 5e5: Tu = 0.1 % stays laminar
    (Blasius C_f within 5 % for 2.5e5 <= Re_x <= 9.5e5), Tu = 3.3 % (T3A) is laminar
    at Re_x = 0.75e5, transitions around Re_x = 1e5 and reaches the turbulent C_f
    by Re_x = 2.5e5.
    The BCM model needs a low freestream nu_tilde (0.1 nu)."""
    re = 5.0e5
    lam = tp.solve(
        48,
        12,
        48,
        reynolds=re,
        transition="bcm",
        turbulence_intensity=0.1,
        nu_tilde_ratio=0.1,
    )
    assert lam.converged
    blasius = 0.664 / np.sqrt(lam.x * re)
    sel = (lam.x >= 0.5) & (lam.x <= 1.9)  # clear of the leading edge and outlet
    assert np.allclose(lam.cf[sel], blasius[sel], rtol=0.05)

    # The transitional case needs the finer plate: with 48 cells along the plate
    # the converged transition location depends on the start-up path (the
    # intermittency feeds back on itself through nu_t, so the discrete problem
    # has more than one steady state), with 96 cells it is Re_x = 1e5 (T3A).
    trans = tp.solve(
        96,
        24,
        64,
        reynolds=re,
        transition="bcm",
        turbulence_intensity=3.3,
        nu_tilde_ratio=0.1,
    )
    assert trans.converged
    cf = np.interp([0.15, 0.5, 1.9], trans.x, trans.cf)
    assert cf[0] == pytest.approx(0.664 / np.sqrt(0.15 * re), rel=0.1)  # still laminar
    turbulent = 0.0576 * (np.array([0.5, 1.9]) * re) ** (-0.2)
    assert np.all(cf[1:] > 0.85 * turbulent)  # turbulent from Re_x = 2.5e5 on
    assert trans.cf[trans.x > 0.5].min() > 2.0 * blasius[sel].max()


def test_aft_transition_on_the_plate():
    """SA-AFT2019b on the plate at 1e6 per unit length with N_crit = 3: the
    e^N estimate for Blasius (H12 = 2.59: Re_theta0 = 237, dn/dRe_theta =
    0.0103) puts transition at Re_x = 6.3e5. Laminar C_f (48 cells, 5-20 %
    below Blasius) up to Re_x = 4.7e5, onset between 5.5e5 and 7e5,
    turbulent C_f from Re_x = 8.5e5 on; converges to machine precision."""
    re = 1.0e6
    r = tp.solve(
        48,
        12,
        48,
        reynolds=re,
        transition="aft",
        critical_amplification=3.0,
        nu_tilde_ratio=0.1,
        settings=PseudoTimeSettings(residual_drop=8.0, max_iterations=2000),
    )
    assert r.converged
    blasius = 0.664 / np.sqrt(r.x * re)
    turbulent = 0.0576 * (r.x * re) ** (-0.2)
    lam = (r.x > 0.2) & (r.x < 0.47)
    assert np.all(np.abs(r.cf[lam] / blasius[lam] - 1.0) < 0.2)
    assert np.all(r.cf[r.x > 0.85] > 0.85 * turbulent[r.x > 0.85])
    onset = r.x[np.argmax((r.x > 0.1) & (r.cf > 1.3 * blasius))] * re
    assert 5.5e5 < onset < 7.0e5
    gamma = np.exp(r.w[:, 4])
    assert gamma.min() == pytest.approx(1.0 / 50.0, rel=1e-2)  # laminar layer


def test_sst_turbulent_plate():
    """k-omega SST-2003m on the TMR plate, TMR SST freestream (k = 2.25e-7 U^2,
    mu_t/mu = 0.009): C_f(0.97) 0.002684 / 0.002684 / 0.002689 and C_D
    0.002869 / 0.002866 / 0.002864 on 48 x 48 / 96 x 64 / 192 x 96 cells, i.e.
    grid-converged, 0.8 % below and 0.3 % above the TMR SA values."""
    r = tp.solve(48, 12, 48, turbulence="sst")
    assert r.converged and r.y_plus_max < 1.0
    assert r.cf_097 == pytest.approx(r.reference["cf_097"], rel=0.015)
    assert r.cd == pytest.approx(r.reference["cd"], rel=0.015)
    assert r.cf[r.x > 0.2].min() > 0.002

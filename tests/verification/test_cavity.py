"""B3 lid-driven cavity vs Ghia, Ghia & Shin (1982)."""

import numpy as np
import pytest

from pyaeroflow.cases import cavity


def deviations(r):
    return np.abs(r.u_centre - r.ghia_u).max(), np.abs(r.v_centre - r.ghia_v).max()


def test_re100_coarse():
    r = cavity.solve(100, 32, cluster=False)
    assert r.converged
    du, dv = deviations(r)
    assert du < 0.015 and dv < 0.015


@pytest.mark.slow
def test_re100():
    r = cavity.solve(100, 64, cluster=False)
    assert r.converged
    du, dv = deviations(r)
    assert du < 0.01 and dv < 0.01


@pytest.mark.slow
def test_re1000():
    """Ghia's Re = 1000 data differ from grid-converged results by up to
    about 0.01-0.015 (e.g. Botella & Peyret 1998), hence the tolerances."""
    r = cavity.solve(1000, 128, cluster=True)
    assert r.converged
    du, dv = deviations(r)
    assert du < 0.01 and dv < 0.02
    assert r.u_centre.min() == pytest.approx(-0.385, abs=0.006)

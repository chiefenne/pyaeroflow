"""Incompressible energy equation: natural convection, heated plate (V1)."""

import pytest

from pyaeroflow.cases import heat_transfer as ht


def test_pohlhausen_similarity_solution():
    """theta'(0): 0.332 at Pr = 1 (Blasius), 0.2927 at Pr = 0.7."""
    assert ht.pohlhausen(1.0) == pytest.approx(0.33206, abs=2e-5)
    assert ht.pohlhausen(0.7) == pytest.approx(0.2927, abs=2e-4)


def test_lange_correlation():
    assert ht.lange_cylinder_nusselt(20.0) == pytest.approx(2.409, abs=2e-3)


def test_de_vahl_davis_ra1e3_coarse():
    """Ideal gas at Delta T / T = 1 %: Nu = 1.118 (de Vahl Davis 1983)."""
    r = ht.solve_cavity(1.0e3, 24)
    assert r.converged
    assert r.nusselt_hot == pytest.approx(1.118, rel=0.01)
    assert r.nusselt_cold == pytest.approx(r.nusselt_hot, rel=1e-8)  # energy balance


@pytest.mark.slow
@pytest.mark.parametrize(
    "rayleigh, n", [(1.0e4, 48), (1.0e5, 64), (1.0e6, 96)], ids=["1e4", "1e5", "1e6"]
)
def test_de_vahl_davis(rayleigh, n):
    r = ht.solve_cavity(rayleigh, n)
    assert r.converged
    assert r.nusselt_hot == pytest.approx(r.reference, rel=0.006)


def test_boussinesq_and_ideal_gas_agree_at_small_delta_t():
    a = ht.solve_cavity(1.0e3, 16, density_model="ideal_gas")
    b = ht.solve_cavity(1.0e3, 16, density_model="boussinesq")
    assert a.converged and b.converged
    assert a.nusselt_hot == pytest.approx(b.nusselt_hot, rel=2e-3)

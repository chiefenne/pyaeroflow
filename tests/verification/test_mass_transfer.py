"""Species equations of the energy solver: Stefan tube, thermosolutal cavity (S1)."""

import pytest

from pyaeroflow.cases import heat_transfer as ht
from pyaeroflow.cases import mass_transfer as mt


@pytest.mark.parametrize("molar_mass", [1.0, 0.622], ids=["equal", "vapour"])
def test_stefan_tube(molar_mass):
    """Net mass flux (rho D / L) ln((1 - Y_L) / (1 - Y_0)), any molar-mass ratio."""
    r = mt.solve_stefan_tube(16, molar_mass=molar_mass)
    assert r.converged
    assert r.mass_flux == pytest.approx(r.exact, rel=5e-4)
    assert r.profile_error < 2e-3


def test_thermosolutal_cavity_is_the_thermal_cavity():
    """Le = 1, N = 1: Sh = Nu, and both those of the thermal cavity at Ra_T + Ra_S."""
    r = mt.solve_thermosolutal_cavity(1.0e3, 16)
    thermal = ht.solve_cavity(1.0e3, 16, density_model="boussinesq")
    assert r.converged and thermal.converged
    assert r.sherwood == pytest.approx(r.nusselt, rel=1e-6)
    # The mean density differs by the mean mass fraction (Boussinesq in every
    # term, delta / 4 = 0.25 %).
    assert r.nusselt == pytest.approx(thermal.nusselt_hot, rel=1e-3)


@pytest.mark.slow
def test_thermosolutal_cavity_ra1e5():
    r = mt.solve_thermosolutal_cavity(1.0e5, 64)
    assert r.converged
    assert r.sherwood == pytest.approx(r.reference, rel=0.006)
    assert r.sherwood == pytest.approx(r.nusselt, rel=1e-6)

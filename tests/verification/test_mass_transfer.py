"""Species equations of the energy solver: Stefan tube, thermosolutal cavity (S1),
evaporating surfaces with RANS (S1b)."""

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


def test_stefan_tube_small_mass_flux():
    """Y_0 = 0.0072: the Stefan flow develops from rest (diffusion velocity in beta)."""
    r = mt.solve_stefan_tube(16, y_surface=0.0072, molar_mass=0.622, schmidt=0.71)
    assert r.converged
    assert r.mass_flux == pytest.approx(r.exact, rel=1e-6)


def test_evaporating_tube_wet_bulb():
    """Adiabatic water surface, Le = 1: T_s is the wet-bulb temperature, m exact."""
    t_s, y_s, b = mt.wet_bulb()
    assert t_s * 300.0 == pytest.approx(282.33, abs=0.01)  # 9.2 C in dry 27 C air
    r = mt.solve_evaporating_tube(16)
    assert r.converged
    assert abs(r.surface_temperature - t_s).max() < 1e-12
    assert r.mass_flux.mean() == pytest.approx(r.reference, rel=1e-5)


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


@pytest.mark.slow
@pytest.mark.parametrize("turbulence", ["sa-neg", "sst"])
def test_turbulent_evaporating_plate(turbulence):
    """Le = Le_t = 1: wet-bulb surface; St_m within -3..+9 % of Kays & Crawford."""
    import numpy as np

    r = mt.solve_evaporating_plate(turbulence)
    assert r.converged
    assert abs(r.surface_temperature - r.wet_bulb).max() < 1e-12
    re_x = 5.0e6 * r.x
    sel = (re_x > 4.0e5) & (re_x < 5.0e6)
    ratio = r.mass_flux[sel] / np.log1p(r.transfer_number) / r.reference[sel]
    assert ratio.min() > 0.97 and ratio.max() < 1.09

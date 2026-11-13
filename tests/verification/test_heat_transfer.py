"""Incompressible energy equation: convection, heated plates (V1, V2 RANS)."""

import pytest

from pyaeroflow.cases import heat_transfer as ht
from pyaeroflow.driver import FlowProblem
from pyaeroflow.physics import thermo as th
from pyaeroflow.physics import variable_density as vd
from pyaeroflow.solver import PseudoTimeSettings


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


def test_kays_crawford_stanton():
    st = float(ht.kays_crawford_stanton(1.0e6, 0.71))
    assert st == pytest.approx(0.0287 * 1.0e6**-0.2 * 0.71**-0.4)


@pytest.mark.slow
@pytest.mark.parametrize("turbulence", ["sa-neg", "sst"])
def test_turbulent_heated_plate(turbulence):
    """RANS energy equation: St within -4..+7 % of Kays & Crawford (Re_x 4e5-5e6)."""
    import numpy as np

    r = ht.solve_turbulent_plate(turbulence)
    assert r.converged
    re_x = 5.0e6 * r.x
    sel = (re_x > 4.0e5) & (re_x < 5.0e6)
    ratio = r.stanton[sel] / r.reference[sel]
    assert ratio.min() > 0.96 and ratio.max() < 1.07
    analogy = 2.0 * r.stanton[sel] / r.cf[sel]
    assert np.all((analogy > 1.1) & (analogy < 1.3))


def test_similarity_plate_limits():
    """Variable-property similarity: Pohlhausen / Blasius at T_w = T_inf."""
    nu, cf = ht.similarity_plate(1.0 + 1e-6, 0.71, 110.4 / 300.0)
    assert nu == pytest.approx(ht.pohlhausen(0.71), rel=1e-4)
    assert cf == pytest.approx(0.6641, abs=2e-4)
    # A hotter wall lowers both (lower density at the wall dominates).
    hot_nu, hot_cf = ht.similarity_plate(2.0, 0.71, 110.4 / 300.0)
    assert hot_nu < nu and hot_cf < cf


def test_heated_channel_developed_nusselt():
    """Uniform wall heat flux, Re_Dh = 40: Nu = 140/17 (Shah & London), developed."""
    import numpy as np

    r = ht.solve_channel()
    assert r.converged
    sel = (r.x > 1.5) & (r.x < 3.6)  # thermally developed, clear of the outlet
    assert np.all(np.abs(r.nusselt[sel] / r.reference - 1.0) < 0.01)


@pytest.mark.slow
def test_heated_cylinder_re40_lange():
    """Heated cylinder Re 40 (wall +3 K): mean Nu within 2 % of Lange et al. (1998)."""
    import numpy as np

    from pyaeroflow.mesh import library

    mesh = library.build("cylinder-ogrid", n_radial=72, outer_radius=60.0)
    model = vd.VariableDensityModel(
        density=1.0,
        thermo=th.Thermo(
            th.DensityModel("ideal_gas", pressure=1.0),
            th.PropertyLaw("constant", 1.0 / 40.0),
            th.PropertyLaw("prandtl", 0.71),
            th.PropertyLaw("constant", 1.0),
        ),
    )
    delta = 0.01
    problem = FlowProblem(
        mesh,
        model,
        {
            "farfield": {
                "type": "farfield",
                "velocity": (1.0, 0.0),
                "pressure": 0.0,
                "temperature": 1.0,
            },
            "cylinder": {"type": "wall", "temperature": 1.0 + delta},
        },
    )
    result = problem.solve(
        problem.initial_state((1.0, 0.0), 0.0, temperature=1.0),
        settings=PseudoTimeSettings(
            residual_drop=9.0, max_iterations=300, cfl_start=1000.0
        ),
    )
    assert result.converged
    faces = problem.boundary_faces(["cylinder"])
    q = problem.wall_heat_flux(result.u)[faces]
    length = np.asarray(problem.geom.b_length)[faces]
    k = 1.0 / (40.0 * 0.71)
    nu = float(np.sum(q * length) / (k * delta * length.sum()))
    assert nu == pytest.approx(ht.lange_cylinder_nusselt(40.0), rel=0.02)


@pytest.mark.slow
def test_rayleigh_benard_clever_busse(tmp_path):
    """Rayleigh-Benard rolls at Ra 1e4, Pr 0.71, one critical wavelength: Nu 2.652.

    Clever and Busse (1974) as tabulated by the Nektar++ tutorial (2.65179); the
    64 x 32 case file is within 0.3 % (about 800 iterations).
    """
    import re
    import shutil
    from pathlib import Path

    from pyaeroflow.config import load_case
    from pyaeroflow.mesh import library, write_su2
    from pyaeroflow.run import run_case

    case = tmp_path / "rb.toml"
    shutil.copy(
        Path(__file__).parents[2] / "examples" / "cases" / "rayleigh_benard_ra1e4.toml",
        case,
    )
    write_su2(tmp_path / "rayleigh_benard_64x32.su2", library.build("rayleigh-benard"))
    config = load_case(case)
    config.data["solver"]["max_iterations"] = 3000
    result = run_case(config, console=False)
    assert result.converged
    text = (result.output_dir / "rayleigh_benard_ra1e4.log").read_text()
    nu = float(re.search(r"heat 'bottom'.*mean Nu = ([0-9.]+)", text).group(1))
    assert abs(nu / 2.65179 - 1.0) < 0.005

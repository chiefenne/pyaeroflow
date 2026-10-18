"""R2: NACA 0012 at Re 6e6 (NASA TMR 2DN00), SA-neg on the generated O-grid."""

import numpy as np
import pytest

from pyaeroflow.cases import naca0012
from pyaeroflow.mesh import airfoil_ogrid, airfoil_points, naca0012_ogrid
from pyaeroflow.mesh import naca0012 as thickness


def test_airfoil_geometry_and_ogrid():
    pts = airfoil_points(64)
    assert pts.shape == (64 + 3, 2)  # 64 surface nodes + 3 base nodes
    assert np.isclose(pts[:, 0].max(), 1.0) and np.isclose(pts[:, 0].min(), 0.0)
    assert thickness(0.3) == pytest.approx(0.0595, abs=5e-4)  # 12 % thick (TMR: 11.9 %)
    with pytest.raises(ValueError, match="growth ratio"):
        naca0012_ogrid(64, 32, farfield_radius=50.0)
    mesh = naca0012_ogrid(64, 32, reynolds=1.0e4, farfield_radius=50.0)
    assert mesh.n_cells == 67 * 32 and mesh.cell_area.min() > 0.0
    far = mesh.nodes[-67:]
    radius = np.linalg.norm(far - [0.5, 0.0], axis=1)
    assert 0.95 * 50.0 < radius.min() and radius.max() < 1.05 * 50.0
    assert set(mesh.boundary_faces) == {"airfoil", "farfield"}


def test_naca0012_alpha10_coarse():
    """128 x 64 O-grid, 500 chords: C_L -7 % (coarse farfield/wake resolution).

    The O-grid results keep the frozen limiter: on this grid (trailing-edge
    fan, stretched outer cells) the unlimited solution of D54 converges
    slowly and gives C_L -14 %; the PyAero C-mesh is the D54 reference."""
    r = naca0012.solve(10.0, 128, 64, release_limiter=False)
    assert r.converged and r.y_plus_max < 2.0
    assert r.cl == pytest.approx(r.reference["cl"], rel=0.10)
    assert r.cd_viscous == pytest.approx(0.0057, abs=0.001)
    assert r.cp.min() < -3.0  # suction peak


@pytest.mark.slow
@pytest.mark.parametrize("alpha", [0.0, 10.0, 15.0])
def test_naca0012_medium(alpha):
    """256 x 128 (33k cells): C_L within 2 % (-0.008, 1.076, 1.536); the drag
    is 13-45 % high, dominated by the pressure drag on the coarse O-grid wake."""
    r = naca0012.solve(alpha, 256, 128, release_limiter=False)
    # The first spacing is sized for C_f = 0.003; at the 15-degree suction
    # peak C_f is twice that and y+ reaches 2.2.
    assert r.converged and r.y_plus_max < (3.0 if alpha == 15.0 else 2.0)
    ref = r.reference
    assert r.cl == pytest.approx(ref["cl"], abs=0.015 + 0.03 * abs(ref["cl"]))
    assert r.cd_viscous == pytest.approx(
        {0.0: 0.0069, 10.0: 0.0062, 15.0: 0.0051}[alpha], rel=0.1
    )
    assert ref["cd"] < r.cd < 1.6 * ref["cd"]


def euler_lift(mesh, vortex, alpha=10.0):
    from pyaeroflow.driver import IncompressibleProblem
    from pyaeroflow.physics import forces as fz
    from pyaeroflow.physics import incompressible as inc
    from pyaeroflow.solver import PseudoTimeSettings

    a = np.radians(alpha)
    conditions = {
        "airfoil": {"type": "slip-wall"},
        "farfield": {"type": "farfield", "velocity": (np.cos(a), np.sin(a))},
    }
    model = inc.IncompressibleModel(density=1.0, viscosity=0.0)
    problem = IncompressibleProblem(mesh, model, conditions)
    v = problem.farfield_vortex((0.25, 0.0), ["airfoil"]) if vortex else None
    r = problem.solve(
        problem.initial_state((np.cos(a), np.sin(a))),
        PseudoTimeSettings(residual_drop=5.0, max_iterations=150),
        vortex=v,
    )
    assert r.converged
    return fz.coefficients(
        problem.forces(r.u, ["airfoil"], (0.25, 0.0)), 0.5, 1.0, alpha
    ).cl


def test_vortex_correction_recovers_far_farfield_lift():
    """Euler at 10 deg: a 10-chord farfield with the point-vortex correction
    (P10) gives the lift of a 500-chord farfield within 2 %."""
    near = airfoil_ogrid(
        airfoil_points(96), 32, first_spacing=2e-3, farfield_radius=10.0
    )
    far = airfoil_ogrid(
        airfoil_points(96),
        48,
        first_spacing=2e-3,
        farfield_radius=500.0,
        max_growth=1.35,
    )
    cl_far = euler_lift(far, False)
    cl_near = euler_lift(near, False)
    cl_corrected = euler_lift(near, True)
    assert abs(cl_corrected - cl_far) < abs(cl_near - cl_far)
    assert cl_corrected == pytest.approx(cl_far, rel=0.02)


@pytest.mark.slow
def test_naca0012_pyaero_mesh_alpha10():
    """PyAero y+ = 1 mesh (100k cells), Re 6e6, 10 deg, vortex correction (D54):
    C_L -0.9 %, viscous drag +1.6 %, pressure drag +13 % vs TMR; about 8 min."""
    from pathlib import Path

    from pyaeroflow.mesh import read_su2

    mesh = read_su2(
        Path(__file__).parents[2] / "examples/meshes/naca0012_pyaero_100k_y1.su2"
    )
    r = naca0012.solve(10.0, mesh=mesh, vortex_correction=True)
    assert r.converged
    assert r.cl == pytest.approx(r.reference["cl"], rel=0.02)
    assert r.cd_viscous == pytest.approx(r.reference["cd_viscous"], rel=0.05)
    assert r.cd == pytest.approx(r.reference["cd"], rel=0.10)

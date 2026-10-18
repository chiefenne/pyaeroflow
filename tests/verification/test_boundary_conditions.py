"""Stage-3 boundary conditions and wall loads against exact solutions."""

import numpy as np
import pytest

from pyaeroflow.cases import channel
from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import forces as fz
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.solver import PseudoTimeSettings


def test_couette_wall_forces_and_skin_friction():
    """Shear mu U / H on both walls; the fluid drags the moving lid backwards."""
    mu, u_wall, height, length = 0.1, 1.0, 1.0, 2.0
    r = channel.couette(8, "quad", viscosity=mu, wall_velocity=u_wall, cluster_y="both")
    prob = r.problem
    shear = mu * u_wall / height
    top = prob.forces(r.w, ["top"])
    bottom = prob.forces(r.w, ["bottom"])
    assert top.fx == pytest.approx(-shear * length, rel=1e-8)
    assert bottom.fx == pytest.approx(shear * length, rel=1e-8)
    assert abs(top.fy) < 1e-8 and abs(bottom.fy) < 1e-8  # p = 0 (pinned)
    assert top.fx_pressure == pytest.approx(0.0, abs=1e-8)
    q = 0.5 * u_wall**2
    d = fz.wall_distribution(
        prob.wall_loads(r.w), prob.geom, prob.boundary_faces(["bottom"]), q
    )
    np.testing.assert_allclose(d["cf"], shear / q, rtol=1e-8)  # +x on the bottom wall
    np.testing.assert_allclose(d["tau_x"], shear, rtol=1e-8)


def test_coefficients_rotate_with_angle_of_attack():
    f = fz.Forces(
        fx=1.0,
        fy=2.0,
        fx_pressure=1.0,
        fy_pressure=2.0,
        fx_viscous=0.0,
        fy_viscous=0.0,
        mz=0.5,
    )
    c0 = fz.coefficients(f, q=0.5, length=2.0, angle_of_attack=0.0)
    assert (c0.cd, c0.cl, c0.cm) == (1.0, 2.0, 0.25)
    c90 = fz.coefficients(f, q=0.5, length=2.0, angle_of_attack=90.0)
    assert c90.cd == pytest.approx(2.0) and c90.cl == pytest.approx(-1.0)


def test_total_pressure_inlet_euler_channel():
    """Inviscid channel: |u| = sqrt(2 (p0 - p_out) / rho) along the direction."""
    rho, p0 = 1.3, 0.8
    mesh = rectangle(10, 5, x=(0.0, 2.0), cell_type="quad")
    conditions = {
        "left": {
            "type": "total-pressure-inlet",
            "total_pressure": p0,
            "flow_direction": (1.0, 0.0),
        },
        "right": {"type": "pressure-outlet", "pressure": 0.0},
        "bottom": {"type": "slip-wall"},
        "top": {"type": "symmetry"},
    }
    prob = IncompressibleProblem(
        mesh, inc.IncompressibleModel(density=rho, viscosity=0.0), conditions
    )
    w0 = prob.initial_state((0.5, 0.0), 0.0)
    res = prob.solve(w0, PseudoTimeSettings(residual_drop=9.0))
    assert res.converged
    speed = np.sqrt(2.0 * p0 / rho)
    np.testing.assert_allclose(res.u[:, inc.U], speed, rtol=1e-7)
    np.testing.assert_allclose(res.u[:, inc.V], 0.0, atol=1e-7)
    np.testing.assert_allclose(res.u[:, inc.P], 0.0, atol=1e-7)


@pytest.mark.parametrize("cell_type", ["quad", "triangle"])
def test_transpiration_channel_exact(cell_type):
    """Poiseuille flow with uniform blowing (bottom) and suction (top):
    u = G / v_w [ y - H (e^(k y) - 1) / (e^(k H) - 1) ], k = v_w / nu, v = v_w."""
    nu, g, v_w, height, length = 0.1, 1.0, 0.3, 1.0, 1.0
    n = 24
    mesh = rectangle(
        n,
        n,
        x=(0.0, length),
        y=(0.0, height),
        cell_type=cell_type,
        cluster_y="both",
        beta=1.0,
    )
    conditions = {
        "bottom": {"type": "transpiration-wall", "normal_velocity": v_w},
        "top": {"type": "transpiration-wall", "normal_velocity": -v_w},
        "left": {"type": "periodic", "partner": "right", "translation": (length, 0.0)},
    }
    model = inc.IncompressibleModel(density=1.0, viscosity=nu, body_force=(g, 0.0))
    prob = IncompressibleProblem(mesh, model, conditions)
    res = prob.solve(settings=PseudoTimeSettings(residual_drop=9.0, max_iterations=200))
    assert res.converged
    y = mesh.cell_centroid[:, 1]
    k = v_w / nu
    exact = g / v_w * (y - height * np.expm1(k * y) / np.expm1(k * height))
    area = mesh.cell_area
    err = np.sqrt(np.sum(area * (res.u[:, inc.U] - exact) ** 2) / area.sum())
    assert err < 3e-3, err
    np.testing.assert_allclose(res.u[:, inc.V], v_w, atol=2e-3)


def test_transpiration_mass_imbalance_is_rejected():
    mesh = rectangle(4, 4, cell_type="quad")
    conditions = {
        "bottom": {"type": "transpiration-wall", "normal_velocity": 0.3},
        "top": {"type": "wall"},
        "left": {"type": "wall"},
        "right": {"type": "wall"},
    }
    with pytest.raises(ValueError, match="net inflow"):
        IncompressibleProblem(mesh, inc.IncompressibleModel(), conditions)

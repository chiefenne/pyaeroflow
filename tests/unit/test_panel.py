"""Panel method and the boundary-layer initial field (initial.boundary_layer)."""

import dataclasses

import numpy as np
import pytest

from pyaeroflow.config import ConfigError, resolve
from pyaeroflow.ibl.field import mesh_ibl_field, wall_contours
from pyaeroflow.mesh import naca0012_ogrid
from pyaeroflow.mesh.mesh import MeshError
from pyaeroflow.numerics import panel


def joukowski(n, e=0.08, f=0.05):
    """Joukowski airfoil (cusped trailing edge) with its exact lift slope data."""
    mu = complex(-e, f)
    radius = abs(1.0 - mu)
    beta = np.arcsin(f / radius)
    zeta = mu + radius * np.exp(1j * (-beta + np.linspace(0.0, 2.0 * np.pi, n)))
    z = zeta + 1.0 / zeta
    return np.column_stack((z.real, z.imag))[:-1], radius, beta


def naca0012_contour(n=161, closed=True):
    b = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(b))
    a4 = -0.1036 if closed else -0.1015
    y = 0.6 * (
        0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4
    )
    xy = np.vstack(
        (np.column_stack((x[::-1], y[::-1])), np.column_stack((x[1:], -y[1:])))
    )
    return xy[:-1] if closed else xy


@pytest.mark.parametrize("alpha", [0.0, 5.0])
def test_joukowski_lift_matches_the_exact_solution(alpha):
    xy, radius, beta = joukowski(201)
    chord = np.ptp(xy[:, 0])
    a = np.radians(alpha)
    body = panel.contour_body(xy)
    assert not body.blunt
    sol = panel.solve(body, (np.cos(a), np.sin(a)))
    exact = 8.0 * np.pi * radius / chord * np.sin(a + beta)
    assert sol.lift_coefficient(chord) == pytest.approx(exact, rel=5e-4)


def test_contour_start_and_direction_do_not_matter():
    xy, _, _ = joukowski(101)
    ref = panel.solve(panel.contour_body(xy), (1.0, 0.1)).lift_coefficient()
    other = np.roll(xy, 37, axis=0)[::-1]
    assert panel.solve(panel.contour_body(other), (1.0, 0.1)).lift_coefficient() == (
        pytest.approx(ref, rel=1e-10)
    )


def test_body_is_a_streamline_and_far_field_is_the_freestream():
    body = panel.contour_body(naca0012_contour())
    a = np.radians(5.0)
    u = (np.cos(a), np.sin(a))
    sol = panel.solve(body, u)
    mid = 0.5 * (body.nodes[:-1] + body.nodes[1:])
    assert np.abs(panel.stream_function(sol, mid) - sol.psi0[0]).max() < 2e-4
    far = panel.velocity(sol, np.array([[300.0, 200.0], [-400.0, 50.0]]))
    assert far == pytest.approx(np.tile(u, (2, 1)), abs=2e-3)
    # stagnation: zero sheet strength somewhere near the leading edge
    speed = panel.surface_speed(sol)
    assert np.abs(speed).min() < 0.15
    assert np.abs(speed).max() == pytest.approx(1.8, abs=0.3)


def test_blunt_trailing_edge_has_a_base_panel_and_the_flow_leaves_it():
    blunt = panel.contour_body(naca0012_contour(closed=False))
    sharp = panel.contour_body(naca0012_contour(closed=True))
    assert blunt.blunt and not sharp.blunt
    assert blunt.nodes[0] == pytest.approx([1.0, 0.00126])
    assert blunt.nodes[-1] == pytest.approx([1.0, -0.00126])
    a = np.radians(5.0)
    u = (np.cos(a), np.sin(a))
    cl_blunt = panel.solve(blunt, u).lift_coefficient()
    cl_sharp = panel.solve(sharp, u).lift_coefficient()
    assert cl_sharp == pytest.approx(0.603, abs=2e-3)
    assert cl_blunt == pytest.approx(cl_sharp, rel=5e-3)
    behind = panel.velocity(panel.solve(blunt, (1.0, 0.0)), np.array([[1.002, 0.0]]))
    assert 0.5 < behind[0, 0] < 1.0
    assert abs(behind[0, 1]) < 1e-6


def test_base_with_inner_nodes_is_reduced_to_one_panel():
    xy = naca0012_contour(closed=False)
    base = np.array([[1.0, -0.0006], [1.0, 0.0], [1.0, 0.0006]])
    body = panel.contour_body(np.vstack((xy, base)))
    assert body.blunt
    assert body.nodes.shape[0] == xy.shape[0]


def test_two_bodies_are_solved_together():
    one = naca0012_contour(81)
    bodies = [panel.contour_body(one), panel.contour_body(one + [0.0, 50.0])]
    sol = panel.solve(bodies, (np.cos(0.1), np.sin(0.1)))
    single = panel.solve(bodies[0], (np.cos(0.1), np.sin(0.1)))
    assert sol.circulation(0) == pytest.approx(single.circulation(), rel=2e-3)
    assert sol.circulation(1) == pytest.approx(single.circulation(), rel=2e-3)


@pytest.fixture(scope="module")
def ogrid():
    return naca0012_ogrid(64, 40, reynolds=1.0e4, farfield_radius=20.0)


def test_wall_contours_of_a_mesh(ogrid):
    assert len(wall_contours(ogrid, "airfoil")) == 1
    part = dict(ogrid.boundary_faces, airfoil=ogrid.boundary_faces["airfoil"][:10])
    with pytest.raises(MeshError, match="closed"):
        wall_contours(dataclasses.replace(ogrid, boundary_faces=part), "airfoil")


@pytest.mark.slow
def test_boundary_layer_field_of_a_mesh(ogrid):
    a = np.radians(2.0)
    u = (np.cos(a), np.sin(a))
    f = mesh_ibl_field(ogrid, "airfoil", u, 1.0e-4)
    assert f.result.converged
    speed = np.linalg.norm(f.velocity, axis=1)
    far = np.linalg.norm(ogrid.cell_centroid, axis=1) > 15.0
    assert f.velocity[far] == pytest.approx(np.tile(u, (far.sum(), 1)), abs=0.02)
    assert speed.min() < 0.1  # no slip in the first cells
    # laminar layers at Re 1e4: eddy viscosity only in the (turbulent) wake
    ahead = ogrid.cell_centroid[:, 0] < 0.9
    assert np.all(f.eddy_viscosity[ahead] == 0.0)
    assert f.eddy_viscosity.max() > 0.0


def _case(**initial):
    return {
        "mesh": {"file": "m.su2"},
        "physics": {"model": "laminar"},
        "fluid": {"density": 1.0, "viscosity": 0.1},
        "boundary": {
            "airfoil": {"type": "wall"},
            "farfield": {"type": "farfield", "velocity": 1.0},
        },
        "initial": initial,
    }


def test_config_accepts_walls_only():
    cfg = resolve(_case(boundary_layer="airfoil"))
    assert cfg.data["initial"]["boundary_layer"] == "airfoil"
    assert resolve(_case()).data["initial"]["boundary_layer"] is None
    with pytest.raises(ConfigError, match="not a wall"):
        resolve(_case(boundary_layer="farfield"))
    with pytest.raises(ConfigError, match="did you mean 'airfoil'"):
        resolve(_case(boundary_layer="airfoi"))
    cfg = resolve(_case(boundary_layer="airfoil", restart="x.npz"))
    assert any("boundary_layer ignored" in n for n in cfg.notices)

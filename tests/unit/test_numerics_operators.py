"""Gradients, reconstruction, limiter and face gradients (stage 1)."""

import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.mesh import rectangle
from pyaeroflow.numerics import (
    boundary_face_gradient,
    build_geometry,
    interior_face_gradient,
    lsq_gradient,
    reconstruct,
    venkatakrishnan_wang,
)

MESHES = [
    (cell_type, perturb)
    for cell_type in ("quad", "triangle", "polygon", "mixed")
    for perturb in (0.0, 0.4)
]
A, B, C = 0.3, -1.7, 2.4  # linear field u = A + B x + C y


def linear(points):
    return (A + B * points[:, 0] + C * points[:, 1])[:, None]


@pytest.fixture(params=MESHES, ids=lambda m: f"{m[0]}-p{m[1]}")
def disc(request):
    cell_type, perturb = request.param
    mesh = rectangle(
        7,
        5,
        x=(0.0, 1.4),
        cell_type=cell_type,
        perturb=perturb,
        cluster_y="bottom",
        beta=1.5,
        diagonal="random",
        seed=11,
    )
    return build_geometry(mesh)


@pytest.mark.parametrize("q", [0.0, 1.0, 1.5, 2.0])
def test_lsq_exact_for_linear_field(disc, q):
    disc = build_geometry(disc.mesh, weight_exponent=q)
    g = disc.geom
    u, u_b = linear(np.asarray(g.centroid)), linear(np.asarray(g.b_centre))
    grad = np.asarray(lsq_gradient(g, jnp.asarray(u), jnp.asarray(u_b)))
    np.testing.assert_allclose(grad[:, 0, 0], B, atol=1e-11)
    np.testing.assert_allclose(grad[:, 0, 1], C, atol=1e-11)


def test_lsq_several_variables(disc):
    g = disc.geom
    c, b = np.asarray(g.centroid), np.asarray(g.b_centre)
    u = np.column_stack((linear(c)[:, 0], 2.0 * c[:, 0], -c[:, 1]))
    u_b = np.column_stack((linear(b)[:, 0], 2.0 * b[:, 0], -b[:, 1]))
    grad = np.asarray(lsq_gradient(g, jnp.asarray(u), jnp.asarray(u_b)))
    np.testing.assert_allclose(grad[:, 1, 0], 2.0, atol=1e-11)
    np.testing.assert_allclose(grad[:, 1, 1], 0.0, atol=1e-11)
    np.testing.assert_allclose(grad[:, 2, 0], 0.0, atol=1e-11)
    np.testing.assert_allclose(grad[:, 2, 1], -1.0, atol=1e-11)


def test_reconstruction_exact_for_linear_field(disc):
    g = disc.geom
    u = jnp.asarray(linear(np.asarray(g.centroid)))
    u_b = jnp.asarray(linear(np.asarray(g.b_centre)))
    grad = lsq_gradient(g, u, u_b)
    left, right, inner = reconstruct(g, u, grad, jnp.ones_like(u))
    exact_f = linear(np.asarray(g.f_centre))
    np.testing.assert_allclose(left, exact_f, atol=1e-11)
    np.testing.assert_allclose(right, exact_f, atol=1e-11)
    np.testing.assert_allclose(inner, linear(np.asarray(g.b_centre)), atol=1e-11)


@pytest.mark.parametrize("scheme", ["alpha-damping", "edge-correction"])
def test_face_gradients_exact_for_linear_field(disc, scheme):
    g = disc.geom
    u = jnp.asarray(linear(np.asarray(g.centroid)))
    u_b = jnp.asarray(linear(np.asarray(g.b_centre)))
    grad = lsq_gradient(g, u, u_b)
    gi = np.asarray(interior_face_gradient(g, u, grad, scheme))
    gb = np.asarray(boundary_face_gradient(g, u, u_b, grad, scheme))
    for gf in (gi, gb):
        np.testing.assert_allclose(gf[:, 0, 0], B, atol=1e-10)
        np.testing.assert_allclose(gf[:, 0, 1], C, atol=1e-10)


@pytest.mark.parametrize("cell_type", ["quad", "triangle", "polygon", "mixed"])
def test_limiter_vanishes_under_refinement(cell_type):
    """Wang's threshold is fixed relative to the global range of u, while the
    extrapolation differences shrink with h: for a smooth field the limiter
    tends to 1 under mesh refinement (it is not exactly 1 on coarse meshes)."""
    psi_min = []
    for k in (1, 8):
        mesh = rectangle(7 * k, 5 * k, cell_type=cell_type, perturb=0.3, seed=2)
        g = build_geometry(mesh).geom
        u = jnp.asarray(linear(np.asarray(g.centroid)))
        u_b = jnp.asarray(linear(np.asarray(g.b_centre)))
        grad = lsq_gradient(g, u, u_b)
        psi_min.append(float(venkatakrishnan_wang(g, u, u_b, grad, 0.2).min()))
    assert psi_min[1] > psi_min[0]
    assert psi_min[1] > 0.99


def face_bounds(disc, u, u_b):
    """Min/max over each cell and its face neighbours (incl. boundary values)."""
    m, g = disc.mesh, disc.geom
    lo, hi = u.copy(), u.copy()
    o, n = np.asarray(g.f_owner), np.asarray(g.f_neigh)
    np.minimum.at(lo, o, u[n]), np.maximum.at(hi, o, u[n])
    np.minimum.at(lo, n, u[o]), np.maximum.at(hi, n, u[o])
    b = np.asarray(g.b_owner)
    np.minimum.at(lo, b, u_b), np.maximum.at(hi, b, u_b)
    assert lo.shape[0] == m.n_cells
    return lo, hi


@pytest.mark.parametrize("threshold", [0.0, 0.05])
def test_limiter_bounds_for_discontinuity(disc, threshold):
    """Step in u: limited face states stay within the neighbour bounds (+eps)."""
    g = disc.geom
    c, bc = np.asarray(g.centroid), np.asarray(g.b_centre)
    step = lambda p: np.where(p[:, 0] + 0.3 * p[:, 1] > 0.7, 1.0, 0.0)[:, None]  # noqa: E731
    u, u_b = step(c), step(bc)
    grad = lsq_gradient(g, jnp.asarray(u), jnp.asarray(u_b))
    psi = venkatakrishnan_wang(g, jnp.asarray(u), jnp.asarray(u_b), grad, threshold)
    states = reconstruct(g, jnp.asarray(u), grad, psi)
    left, right, inner = (np.asarray(a) for a in states)
    lo, hi = face_bounds(disc, u, u_b)
    eps = threshold * 1.0  # global range is 1
    tol = 1e-12 + eps
    o, n, b = (np.asarray(a) for a in (g.f_owner, g.f_neigh, g.b_owner))
    for states, cells in ((left, o), (right, n), (inner, b)):
        assert np.all(states >= lo[cells] - tol)
        assert np.all(states <= hi[cells] + tol)
    unlimited = reconstruct(g, jnp.asarray(u), grad, jnp.ones_like(psi))[0]
    assert np.asarray(unlimited).max() > 1.0 + tol or np.asarray(unlimited).min() < -tol

"""Incompressible flux pieces, periodic geometry and problem setup checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.numerics import build_geometry, lsq_gradient
from pyaeroflow.physics import incompressible as inc


def test_abs_preconditioned_jacobian_matches_eigendecomposition():
    model = inc.IncompressibleModel(density=1.3)
    rho, beta2 = 1.3, 2.1**2
    w = jnp.array([[0.4, 0.7, -0.4]])
    th = 0.6
    n = np.array([np.cos(th), np.sin(th)])
    u, v = 0.7, -0.4
    vn = u * n[0] + v * n[1]
    # Flux Jacobian with d(rho)/dp = 1/beta^2 (SU2) and the preconditioner.
    a = np.array(
        [
            [vn / beta2, rho * n[0], rho * n[1]],
            [n[0] + u * vn / beta2, rho * (n[0] * u + vn), rho * n[1] * u],
            [n[1] + v * vn / beta2, rho * n[0] * v, rho * (vn + n[1] * v)],
        ]
    )
    g = np.array([[1 / beta2, 0, 0], [u / beta2, rho, 0], [v / beta2, 0, rho]])
    m = np.linalg.solve(g, a)
    lam, p = np.linalg.eig(m)
    np.testing.assert_allclose(
        np.sort(lam.real), sorted([vn, vn - np.sqrt(beta2), vn + np.sqrt(beta2)])
    )
    abs_m = (p @ np.diag(np.abs(lam)) @ np.linalg.inv(p)).real
    ours = np.asarray(
        inc.abs_preconditioned_jacobian(model, w, jnp.asarray(n)[None], beta2)
    )
    np.testing.assert_allclose(ours[0], abs_m, atol=1e-13)
    gamma = np.asarray(inc.preconditioner(model, w, beta2))[0]
    np.testing.assert_allclose(gamma, g)
    ginv = np.asarray(inc.preconditioner_inverse(model, w, beta2))[0]
    np.testing.assert_allclose(ginv @ gamma, np.eye(3), atol=1e-14)


def test_fds_flux_is_consistent():
    model = inc.IncompressibleModel(density=0.8)
    w = jnp.array([[1.5, 0.3, -0.2]])
    n = jnp.array([[0.6, 0.8]])
    np.testing.assert_allclose(
        inc.fds_flux(model, w, w, n, 3.0), inc.inviscid_flux(model, w, n), atol=1e-15
    )


@pytest.mark.parametrize("cell_type", ["quad", "triangle", "polygon", "mixed"])
def test_periodic_geometry_and_gradients(cell_type):
    mesh = rectangle(6, 5, x=(0.0, 1.5), cell_type=cell_type, perturb=0.3, seed=1)
    disc = build_geometry(mesh, periodic=[("left", "right", (1.5, 0.0))])
    g = disc.geom
    n_left = mesh.boundary_faces["left"].size
    assert disc.periodic_faces.shape == (n_left, 2)
    assert g.f_owner.size == np.count_nonzero(mesh.neighbour >= 0) + n_left
    assert sorted(disc.boundary_slices) == ["bottom", "top"]
    # A field linear in y is periodic in x: gradients exact everywhere.
    c, b = np.asarray(g.centroid), np.asarray(g.b_centre)
    u = jnp.asarray((0.2 + 1.7 * c[:, 1])[:, None])
    u_b = jnp.asarray((0.2 + 1.7 * b[:, 1])[:, None])
    grad = np.asarray(lsq_gradient(g, u, u_b))
    np.testing.assert_allclose(grad[:, 0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(grad[:, 0, 1], 1.7, atol=1e-12)
    # Periodic face: delta must be the translated centroid difference.
    k = g.f_owner.size - 1
    o, n = int(g.f_owner[k]), int(g.f_neigh[k])
    np.testing.assert_allclose(
        np.asarray(g.f_delta[k]), c[n] - np.array([1.5, 0.0]) - c[o]
    )


def test_periodic_mismatch_is_rejected():
    mesh = rectangle(6, 5, cell_type="quad", perturb=0.3, seed=1)
    with pytest.raises(Exception, match="do not match"):
        build_geometry(mesh, periodic=[("left", "right", (1.0, 0.01))])


def walls(top_velocity=(0.0, 0.0)):
    return {
        "bottom": {"type": "wall"},
        "left": {"type": "wall"},
        "right": {"type": "wall"},
        "top": {"type": "wall", "velocity": top_velocity},
    }


def test_pressure_is_pinned_without_pressure_boundary():
    mesh = rectangle(4, 4, cell_type="quad")
    prob = IncompressibleProblem(mesh, inc.IncompressibleModel(), walls((1.0, 0.0)))
    assert prob.pinned_cell is not None
    centre = mesh.cell_centroid[prob.pinned_cell]
    assert np.all(np.abs(centre - 0.5) < 0.2)
    conditions = dict(walls(), top={"type": "pressure-outlet", "pressure": 0.0})
    prob = IncompressibleProblem(mesh, inc.IncompressibleModel(), conditions)
    assert prob.pinned_cell is None and prob.model.pin is None


def test_net_inflow_without_pressure_boundary_is_rejected():
    mesh = rectangle(4, 4, cell_type="quad")
    conditions = dict(walls(), left={"type": "velocity-inlet", "velocity": (1.0, 0.0)})
    with pytest.raises(ValueError, match="net inflow"):
        IncompressibleProblem(mesh, inc.IncompressibleModel(), conditions)


def test_missing_boundary_condition_is_rejected():
    mesh = rectangle(4, 4, cell_type="quad")
    conditions = walls()
    del conditions["left"]
    with pytest.raises(ValueError, match="missing"):
        IncompressibleProblem(mesh, inc.IncompressibleModel(), conditions)


def test_jacobian_matches_dense_ad_for_euler():
    """Without viscosity the first-order residual is exactly compact."""
    mesh = rectangle(4, 3, cell_type="mixed", perturb=0.3, seed=4)
    conditions = {
        "bottom": {"type": "symmetry"},
        "top": {"type": "slip-wall"},
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.1)},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
    }
    model = inc.IncompressibleModel(viscosity=0.0, spatial_order=1)
    prob = IncompressibleProblem(mesh, model, conditions)
    rng = np.random.default_rng(0)
    w = 0.5 + rng.uniform(-0.3, 0.3, (mesh.n_cells, 3))
    beta2 = prob.beta2(w)
    sparse = prob.jacobian(w, beta2).toarray()
    dense = jax.jacfwd(
        lambda x: inc.residual(
            prob.model, prob.geom, prob.bc, x.reshape(w.shape), 1, beta2
        ).ravel()
    )(jnp.asarray(w.ravel()))
    np.testing.assert_allclose(sparse, np.asarray(dense), atol=1e-12)

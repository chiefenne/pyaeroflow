"""Residual assembly, AD Jacobian and pseudo-time control."""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from pyaeroflow.mesh import rectangle
from pyaeroflow.numerics import build_geometry
from pyaeroflow.solver import (
    PseudoTimeSettings,
    divergence,
    first_order_jacobian,
    solve_steady,
)


def nonlinear_interior_flux(params, u_o, u_n, normal, delta):
    """Two coupled variables, nonlinear in both states."""
    s = u_o + u_n + params
    return jnp.array(
        [
            normal[0] * s[0] ** 2 + normal[1] * s[0] * s[1],
            jnp.sin(s[1]) * normal[1] - (u_n[0] - u_o[0]) / jnp.dot(delta, normal),
        ]
    )


def nonlinear_boundary_flux(params, u_o, u_b, normal, delta):
    return jnp.array(
        [u_o[0] * u_o[1] * normal[0], jnp.exp(u_o[1] - u_b[1]) * normal[1]]
    )


def test_jacobian_matches_dense_ad():
    mesh = rectangle(4, 3, cell_type="mixed", perturb=0.3, seed=4)
    geom = build_geometry(mesh).geom
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.uniform(-1.0, 1.0, (mesh.n_cells, 2)))
    u_b = jnp.asarray(rng.uniform(-1.0, 1.0, (geom.b_owner.size, 2)))

    params = jnp.array(0.3)

    def r1(u):
        f = jax.vmap(nonlinear_interior_flux, in_axes=(None, 0, 0, 0, 0))(
            params, u[geom.f_owner], u[geom.f_neigh], geom.f_normal, geom.f_delta
        )
        b = jax.vmap(nonlinear_boundary_flux, in_axes=(None, 0, 0, 0, 0))(
            params, u[geom.b_owner], u_b, geom.b_normal, geom.b_delta
        )
        return divergence(geom, f, b)

    dense = jax.jit(jax.jacfwd(lambda x: r1(x.reshape(u.shape)).ravel()))(u.ravel())
    sparse = first_order_jacobian(
        geom, nonlinear_interior_flux, nonlinear_boundary_flux, u, params, (u_b,)
    )
    np.testing.assert_allclose(sparse.toarray(), np.asarray(dense), atol=1e-13)


def test_divergence_of_constant_vector_field_vanishes():
    mesh = rectangle(5, 4, cell_type="polygon", perturb=0.4, seed=1)
    geom = build_geometry(mesh).geom
    c = jnp.array([0.7, -1.3])
    f = (geom.f_normal @ c)[:, None]
    b = (geom.b_normal @ c)[:, None]
    assert float(jnp.abs(divergence(geom, f, b)).max()) < 1e-14


def scalar_problem(u0):
    """R(u) = sqrt(u) - 1 per cell; Newton from u0 = 9 jumps to u < 0."""
    residual = lambda u: np.sqrt(u) - 1.0  # noqa: E731  (NaN for u < 0)
    jacobian = lambda u: sp.diags(0.5 / np.sqrt(u.ravel()))  # noqa: E731
    time_term = lambda u, cfl: sp.diags(np.ones(u.size) / cfl)  # noqa: E731
    return residual, jacobian, time_term, np.full((3, 1), u0)


def test_rejected_update_cuts_cfl_and_recovers():
    residual, jacobian, time_term, u0 = scalar_problem(9.0)
    settings = PseudoTimeSettings(
        cfl_start=1.0e8, residual_drop=12.0, line_search_steps=0
    )
    with np.errstate(invalid="ignore"):
        result = solve_steady(residual, jacobian, time_term, u0, settings)
    h = result.history
    assert result.converged
    np.testing.assert_allclose(result.u, 1.0, rtol=1e-12)
    assert not h["accepted"][1]  # first full Newton step lands at u < 0
    assert h["cfl"][2] == 1.0e8 * settings.cut_factor


def test_line_search_accepts_a_shortened_step():
    """With backtracking (D54) the same step is halved and accepted."""
    residual, jacobian, time_term, u0 = scalar_problem(9.0)
    settings = PseudoTimeSettings(cfl_start=1.0e8, residual_drop=12.0)
    with np.errstate(invalid="ignore"):
        result = solve_steady(residual, jacobian, time_term, u0, settings)
    assert result.converged and result.history["accepted"].all()
    np.testing.assert_allclose(result.u, 1.0, rtol=1e-12)


def test_newton_krylov_waits_for_the_measured_drop():
    """A small absolute residual must not trigger the NK switch (per-equation
    measure, D54): u0 = 1.0001 gives |R| ~ 5e-5 from the start."""
    residual, jacobian, time_term, u0 = scalar_problem(1.0001)
    jvp = lambda u, du: 0.5 / np.sqrt(u) * du  # noqa: E731
    settings = PseudoTimeSettings(nk_switch_residual_drop=2.0, residual_drop=10.0)
    result = solve_steady(
        residual, jacobian, time_term, u0, settings, residual_startup=residual, jvp=jvp
    )
    h = result.history
    assert result.converged and not h["newton_krylov"][1]
    first_nk = int(np.argmax(h["newton_krylov"]))
    assert first_nk > 0 and h["measure"][first_nk - 1] <= 1e-2
    assert h["newton_krylov"][first_nk:].all()  # the switch latches


def test_ser_raises_cfl_while_residual_falls():
    residual, jacobian, time_term, u0 = scalar_problem(1.5)
    result = solve_steady(residual, jacobian, time_term, u0, PseudoTimeSettings())
    h = result.history
    assert result.converged and np.all(h["accepted"])
    assert np.all(np.diff(h["residual"]) < 0.0)
    assert np.all(np.diff(h["cfl"][1:]) > 0.0)


def test_proportional_ser_bounds_the_cfl_factor():
    """control = ser (D55): CFL factor per accepted step within [0.3, 2]."""
    residual, jacobian, time_term, u0 = scalar_problem(1.5)
    settings = PseudoTimeSettings(cfl_control="ser", residual_drop=10.0)
    result = solve_steady(residual, jacobian, time_term, u0, settings)
    h = result.history
    assert result.converged and np.all(h["accepted"])
    factor = h["cfl"][2:] / h["cfl"][1:-1]
    assert np.all(factor <= 2.0 + 1e-12) and np.all(factor >= 0.3 - 1e-12)
    assert factor.max() > 1.0


def test_reference_residual_follows_an_operator_change():
    """A callback that changes the residual operator (limiter release) must
    not leave the acceptance test comparing against the old operator's
    residual: here the root moves from u = 1 to u = 1.21 at iteration 2,
    when the first problem is three orders down."""
    state = {"target": 1.0}
    residual = lambda u: np.sqrt(u) - state["target"]  # noqa: E731
    jacobian = lambda u: sp.diags(0.5 / np.sqrt(u.ravel()))  # noqa: E731
    time_term = lambda u, cfl: sp.diags(np.ones(u.size) / cfl)  # noqa: E731

    def callback(it, u, residuals, cfl, accepted):
        if it == 2:
            state["target"] = 1.1
            callback.info["residual_changed"] = True

    callback.info = {}
    settings = PseudoTimeSettings(residual_drop=10.0, max_iterations=40)
    result = solve_steady(
        residual, jacobian, time_term, np.full((3, 1), 1.5), settings, callback=callback
    )
    assert result.converged and result.history["accepted"][3:].all()
    np.testing.assert_allclose(result.u, 1.21, rtol=1e-9)

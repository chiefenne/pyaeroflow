"""Newton-Krylov phase, GMRES-ILU and limiter freezing (stage 3)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.solver import (
    LinearSettings,
    PseudoTimeSettings,
    factorize,
    solve_matrix_free,
)

CAVITY = {
    "bottom": {"type": "wall"},
    "left": {"type": "wall"},
    "right": {"type": "wall"},
    "top": {"type": "wall", "velocity": (1.0, 0.0)},
}


def cavity(n=24, viscosity=1e-2, **model):
    mesh = rectangle(
        n, n, cell_type="quad", cluster_x="both", cluster_y="both", beta=1.0
    )
    return IncompressibleProblem(
        mesh, inc.IncompressibleModel(density=1.0, viscosity=viscosity, **model), CAVITY
    )


def test_jvp_matches_dense_jacobian():
    prob = cavity(6, limiter="venkatakrishnan-wang")
    rng = np.random.default_rng(1)
    w = 0.3 + rng.uniform(-0.2, 0.2, (prob.mesh.n_cells, 3))
    dw = rng.standard_normal(w.shape)
    beta2 = prob.beta2(w)
    psi = prob.limiter(w)
    dense = jax.jacfwd(
        lambda x: inc.residual(
            prob.model, prob.geom, prob.bc, x.reshape(w.shape), 2, beta2, psi
        ).ravel()
    )(jnp.asarray(w.ravel()))
    expected = np.asarray(dense) @ dw.ravel()
    np.testing.assert_allclose(
        prob.jvp(w, dw, beta2, psi).ravel(), expected, rtol=1e-10
    )


def test_newton_krylov_converges_in_fewer_iterations():
    prob = cavity()
    approx = prob.solve(
        settings=PseudoTimeSettings(residual_drop=9, newton_krylov=False)
    )
    nk = prob.solve(settings=PseudoTimeSettings(residual_drop=9))
    assert approx.converged and nk.converged
    n_approx = len(approx.history["iteration"]) - 1
    n_nk = len(nk.history["iteration"]) - 1
    assert nk.history["newton_krylov"].any()
    assert n_nk < n_approx / 2, (n_nk, n_approx)
    assert np.abs(nk.u - approx.u).max() < 1e-6


def test_gmres_ilu_matches_direct():
    prob = cavity()
    direct = prob.solve(settings=PseudoTimeSettings(residual_drop=9))
    ilu = prob.solve(
        settings=PseudoTimeSettings(
            residual_drop=9, linear=LinearSettings(type="gmres-ilu")
        )
    )
    assert ilu.converged
    assert np.abs(ilu.u - direct.u).max() < 1e-6
    assert ilu.history["linear_iterations"][1:].max() > 0


def test_limiter_freeze_restores_deep_convergence():
    prob = cavity(viscosity=2e-3, limiter="venkatakrishnan-wang")
    settings = PseudoTimeSettings(residual_drop=10, max_iterations=80)
    frozen = prob.solve(settings=settings, freeze_limiter_after_residual_drop=3.0)
    assert frozen.converged
    assert frozen.history["residual"][-1] / frozen.history["residual"][0] < 1e-10


def test_matrix_free_gmres_solves_sparse_system():
    rng = np.random.default_rng(0)
    n = 60
    a = np.diag(5.0 + rng.random(n)) + 0.3 * rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    import scipy.sparse as sp

    lu = factorize(sp.csr_matrix(a))
    x, its = solve_matrix_free(lambda v: a @ v, b, lu.precondition, tolerance=1e-12)
    np.testing.assert_allclose(a @ x, b, atol=1e-9)
    assert its <= 3  # exact preconditioner: converges at once


def test_nk_settings_in_config():
    from pyaeroflow.config import resolve

    raw = {
        "mesh": {"file": "m.su2"},
        "physics": {"model": "laminar"},
        "fluid": {"density": 1.0, "viscosity": 0.1},
        "reference": {"velocity": 1.0},
        "initial": {"velocity": [0.0, 0.0]},
        "boundary": {
            "bottom": {"type": "wall"},
            "top": {"type": "wall"},
            "left": {"type": "wall"},
            "right": {"type": "wall"},
        },
        "solver": {
            "newton_krylov": {"enabled": False, "switch_at_iteration": 40},
            "linear_solver": {"type": "gmres-ilu", "ilu_fill_factor": 5.0},
        },
        "numerics": {"limiter_freeze": {"after_residual_drop": 3.0}},
    }
    cfg = resolve(raw)
    assert cfg["solver"]["newton_krylov"]["enabled"] is False
    assert cfg["solver"]["newton_krylov"]["switch_at_residual_drop"] == 2.0
    assert cfg["solver"]["newton_krylov"]["switch_at_iteration"] == 40
    assert cfg["solver"]["linear_solver"]["ilu_fill_factor"] == 5.0
    assert cfg["numerics"]["limiter_freeze"]["after_iterations"] is None
    with pytest.raises(Exception, match="not one of"):
        resolve({**raw, "solver": {"linear_solver": {"type": "lu"}}})


def test_linearize_matches_jvp_and_newton_history():
    """C1: the stored linearization gives the jvp products and the same run."""
    prob = cavity(6, limiter="venkatakrishnan-wang")
    rng = np.random.default_rng(3)
    w = 0.3 + rng.uniform(-0.2, 0.2, (prob.mesh.n_cells, 3))
    dw = rng.standard_normal(w.shape)
    beta2 = prob.beta2(w)
    psi = prob.limiter(w)
    apply = prob.linearize(w, beta2, psi)
    np.testing.assert_allclose(
        apply(dw), prob.jvp(w, dw, beta2, psi), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        apply(2.0 * dw), 2.0 * prob.jvp(w, dw, beta2, psi), rtol=1e-12, atol=1e-14
    )
    settings = {"max_iterations": 30, "residual_drop": 6.0, "nk_switch_iteration": 3}
    with_lin = prob.solve(settings=PseudoTimeSettings(**settings))
    with_jvp = prob.solve(settings=PseudoTimeSettings(nk_linearize=False, **settings))
    assert with_lin.history["newton_krylov"].any()
    np.testing.assert_allclose(
        with_lin.history["residual"], with_jvp.history["residual"], rtol=1e-6
    )
    np.testing.assert_array_equal(
        with_lin.history["linear_iterations"], with_jvp.history["linear_iterations"]
    )

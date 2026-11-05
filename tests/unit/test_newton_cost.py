"""Phase B Newton-cost options: right-preconditioned GMRES, forcing safeguards,
R1 (frozen-limiter) products, adaptive preconditioner lagging (all opt-in)."""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.solver import PseudoTimeSettings
from pyaeroflow.solver.linear import gmres_right, solve_matrix_free

CAVITY = {
    "bottom": {"type": "wall"},
    "left": {"type": "wall"},
    "right": {"type": "wall"},
    "top": {"type": "wall", "velocity": (1.0, 0.0)},
}


def cavity(n=16, viscosity=2e-3, **model):
    mesh = rectangle(
        n, n, cell_type="quad", cluster_x="both", cluster_y="both", beta=1.0
    )
    return IncompressibleProblem(
        mesh, inc.IncompressibleModel(density=1.0, viscosity=viscosity, **model), CAVITY
    )


def system(n=300, seed=0):
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=0.03, random_state=seed) + sp.diags(
        3.0 + rng.random(n)
    )
    return a.tocsc(), rng.standard_normal(n)


@pytest.mark.parametrize("restart", [4, 50])
@pytest.mark.parametrize("tolerance", [1e-1, 1e-6])
def test_right_gmres_meets_the_true_residual(restart, tolerance):
    a, b = system()
    ilu = spla.spilu(a, drop_tol=0.05)
    x, its = gmres_right(
        a.dot, b, ilu.solve, tolerance=tolerance, restart=restart, max_iterations=400
    )
    assert np.linalg.norm(b - a @ x) <= tolerance * np.linalg.norm(b) * (1 + 1e-8)
    assert 0 < its < 400


def test_right_gmres_exact_preconditioner_converges_at_once():
    a, b = system(seed=2)
    lu = spla.splu(a)
    x, its = gmres_right(a.dot, b, lu.solve, tolerance=1e-12)
    np.testing.assert_allclose(a @ x, b, atol=1e-9)
    assert its == 1


def test_right_gmres_stops_at_max_iterations():
    a, b = system(seed=3)
    x, its = solve_matrix_free(
        a.dot, b, lambda r: r, tolerance=1e-14, restart=5, max_iterations=12,
        method="right",
    )
    assert its <= 12 + 2  # plus the residual checks between cycles
    assert np.linalg.norm(b - a @ x) < np.linalg.norm(b)


def test_unknown_gmres_method_is_refused():
    a, b = system()
    with pytest.raises(ValueError):
        solve_matrix_free(a.dot, b, lambda r: r, method="left")


def test_phase_b_options_reach_the_same_solution():
    prob = cavity(limiter="venkatakrishnan-wang")
    base = prob.solve(settings=PseudoTimeSettings(residual_drop=9, max_iterations=120))
    phase_b = prob.solve(
        settings=PseudoTimeSettings(
            residual_drop=9,
            max_iterations=120,
            nk_gmres="right",
            nk_forcing_safeguard=True,
            nk_approximate_drop=4.0,
            pc_lagging="adaptive",
        )
    )
    assert base.converged and phase_b.converged
    assert phase_b.history["newton_krylov"].any()
    assert np.abs(phase_b.u - base.u).max() < 1e-6
    # The exact products took over after 4 orders (the switch latches).
    assert phase_b.state["approximate_products"] is False


def test_adaptive_lagging_factorizes_less_often():
    prob = cavity()
    records = {"fixed": [], "adaptive": []}
    for mode, rec in records.items():
        result = prob.solve(
            settings=PseudoTimeSettings(
                residual_drop=9,
                max_iterations=120,
                nk_switch_residual_drop=1.0,
                pc_lagging=mode,
            ),
            debug=rec.append,
        )
        assert result.converged
    nk_factor = {
        m: sum(r["factorized"] for r in rec if r.get("nk")) for m, rec in records.items()
    }
    assert nk_factor["adaptive"] <= nk_factor["fixed"]

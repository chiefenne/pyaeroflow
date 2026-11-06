"""Phase C: local CFL of selected rows from the update size (opt-in)."""

import numpy as np
import scipy.sparse as sp

from pyaeroflow.cases import tmr_flat_plate as tp
from pyaeroflow.solver import PseudoTimeSettings, solve_steady


def two_column_problem(n=4):
    """R = u - 1 in both columns; column 1 starts far away (large relative updates).

    ``time_term`` honours ``row_cfl`` like the RANS driver and records it.
    """
    calls = []

    def residual(u):
        return u - 1.0

    def jacobian(u):
        return sp.identity(u.size, format="csr")

    def time_term(u, cfl, row_cfl=None):
        t = np.full(u.shape, 1.0 / cfl)
        if row_cfl is not None:
            cols, cap = row_cfl
            for c in cols:
                t[:, c] *= np.maximum(1.0, cfl / cap)
            calls.append(np.array(cap))
        return sp.diags(t.ravel())

    u0 = np.column_stack([np.zeros(n), np.linspace(50.0, 80.0, n)])
    u0[0, 1] = 1.0  # converged cell: small update, never capped
    return residual, jacobian, time_term, u0, calls


def test_local_cfl_cuts_large_updates_and_releases():
    residual, jacobian, time_term, u0, calls = two_column_problem()
    settings = PseudoTimeSettings(
        cfl_start=10.0,
        residual_drop=10.0,
        newton_krylov=False,
        local_cfl_columns=(1,),
        max_iterations=200,
    )
    result = solve_steady(residual, jacobian, time_term, u0, settings)
    assert result.converged
    np.testing.assert_allclose(result.u, 1.0, rtol=1e-8)
    assert calls, "a large update must cap some cells"
    first = calls[0]
    assert np.isinf(first[0])  # the converged cell keeps the global CFL
    assert np.isfinite(first[1:]).all()
    # the caps grow again once the updates are small
    assert (result.state["local_cfl"][1:] > first[1:]).all()


def test_local_cfl_off_by_default():
    residual, jacobian, time_term, u0, calls = two_column_problem()
    settings = PseudoTimeSettings(cfl_start=10.0, newton_krylov=False)
    result = solve_steady(residual, jacobian, time_term, u0, settings)
    assert not calls
    assert result.state["local_cfl"] is None


def test_driver_row_cfl_scales_only_the_given_rows():
    p = tp.solve(
        12,
        4,
        12,
        reynolds=1.0e5,
        turbulence="sst",
        settings=tp.PseudoTimeSettings(max_iterations=2),
    ).problem
    w = p.initial_state((1.0, 0.0))
    params = p.scheme_parameters(w)
    n, nv = w.shape
    cap = np.full(n, np.inf)
    cap[::3] = 10.0
    cfl = 1.0e3
    plain = p.time_term(w, params, cfl).diagonal().reshape(n, nv)
    local = p.time_term(w, params, cfl, row_cfl=([4], cap)).diagonal().reshape(n, nv)
    assert p.local_cfl_cells == int(np.count_nonzero(cap < cfl))
    np.testing.assert_allclose(local[::3, 4], plain[::3, 4] * cfl / 10.0)
    np.testing.assert_allclose(local[:, :4], plain[:, :4])
    mask = np.isinf(cap)
    np.testing.assert_allclose(local[mask], plain[mask])


def test_growth_max_bounds_the_cfl_factor():
    residual, jacobian, time_term, u0, _ = two_column_problem()
    kw = dict(cfl_start=1.0, newton_krylov=False, residual_drop=10.0)
    free = solve_steady(residual, jacobian, time_term, u0, PseudoTimeSettings(**kw))
    bounded = solve_steady(
        residual,
        jacobian,
        time_term,
        u0,
        PseudoTimeSettings(growth_max=3.0, **kw),
    )
    assert bounded.converged
    assert np.max(np.diff(np.log(free.history["cfl"]))) > np.log(3.0)
    assert np.max(np.diff(np.log(bounded.history["cfl"]))) <= np.log(3.0) + 1e-12


def test_equation_scaling_keeps_the_solution():
    from pyaeroflow.solver.pseudo_time import _equation_scaling

    rng = np.random.default_rng(0)
    n, nv = 5, 3
    a = np.eye(n * nv) * 4.0 + 0.1 * rng.standard_normal((n * nv, n * nv))
    b = rng.standard_normal(n * nv) * np.tile([1e-6, 1.0, 1e-3], n)
    op, pre, rhs = _equation_scaling(
        a.dot, lambda y: y, b, np.array([1e-6, 1.0, 1e-3]), n
    )
    m = np.column_stack([op(e) for e in np.eye(n * nv)])
    x = np.linalg.solve(m, rhs)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10, atol=1e-14)
    # every equation has the same share of the scaled right-hand side
    shares = (rhs.reshape(n, nv) ** 2).sum(axis=0)
    assert shares.max() / shares.min() < 50.0


def test_source_limit_release_waits_for_the_cfl():
    residual, jacobian, _, u0, _ = two_column_problem()
    seen = []

    def time_term(u, cfl, source_limit=None):
        seen.append((cfl, source_limit))
        return sp.diags(np.full(u.size, 1.0 / cfl))

    jvp = lambda u, du: du  # noqa: E731  (J = I)
    kw = dict(
        cfl_start=1.0,
        growth_factor=1.5,
        residual_drop=14.0,
        nk_switch_residual_drop=0.0,
        source_cfl_limit=0.9,
        source_cfl_release=2,
        max_iterations=60,
    )
    solve_steady(residual, jacobian, time_term, u0, PseudoTimeSettings(**kw), jvp=jvp)
    early = [c for c, lim in seen if lim is None]
    seen.clear()
    solve_steady(
        residual,
        jacobian,
        time_term,
        u0,
        PseudoTimeSettings(source_release_cfl=20.0, **kw),
        jvp=jvp,
    )
    late = [c for c, lim in seen if lim is None]
    assert early and late
    assert min(early) < 20.0 and min(late) >= 20.0


def test_source_limit_release_waits_for_the_residual_drop():
    residual, jacobian, _, u0, _ = two_column_problem()
    seen = []

    def time_term(u, cfl, source_limit=None):
        seen.append(float(np.abs(u - 1.0).max()))
        seen_limit.append(source_limit)
        return sp.diags(np.full(u.size, 1.0 / cfl))

    seen_limit = []
    kw = dict(
        cfl_start=1.0,
        growth_factor=1.5,
        residual_drop=14.0,
        nk_switch_residual_drop=0.0,
        source_cfl_limit=0.9,
        source_cfl_release=2,
        max_iterations=60,
    )
    jvp = lambda u, du: du  # noqa: E731
    solve_steady(residual, jacobian, time_term, u0, PseudoTimeSettings(**kw), jvp=jvp)
    first_free = seen_limit.index(None)
    seen_limit.clear()
    seen.clear()
    solve_steady(
        residual,
        jacobian,
        time_term,
        u0,
        PseudoTimeSettings(source_release_drop=3.0, **kw),
        jvp=jvp,
    )
    assert seen_limit.index(None) > first_free

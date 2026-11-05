"""Phase A control-loop options: unsteady-residual step control, CFL floor,
source-term time-step limit (all opt-in; the defaults keep the 1.2 loop)."""

import numpy as np
import pytest
import scipy.sparse as sp

from pyaeroflow.cases import tmr_flat_plate as tp
from pyaeroflow.solver import PseudoTimeSettings, solve_steady


def scalar_problem(u0, n=3):
    """R(u) = sqrt(u) - 1 per cell; Newton from u0 = 9 jumps to u < 0."""
    residual = lambda u: np.sqrt(u) - 1.0  # noqa: E731  (NaN for u < 0)
    jacobian = lambda u: sp.diags(0.5 / np.sqrt(u.ravel()))  # noqa: E731
    time_term = lambda u, cfl: sp.diags(np.ones(u.size) / cfl)  # noqa: E731
    return residual, jacobian, time_term, np.full((n, 1), u0)


def test_unsteady_step_control_under_relaxes_and_converges():
    residual, jacobian, time_term, u0 = scalar_problem(9.0)
    settings = PseudoTimeSettings(
        cfl_start=1.0e8, residual_drop=12.0, step_control="unsteady"
    )
    with np.errstate(invalid="ignore"):
        result = solve_steady(residual, jacobian, time_term, u0, settings)
    h = result.history
    assert result.converged
    np.testing.assert_allclose(result.u, 1.0, rtol=1e-12)
    # The full step lands at u < 0: the quadratic fit cuts it (not a halving
    # sequence), and the partial step holds the CFL.
    assert h["accepted"][1] and 0.1 <= h["step_factor"][1] <= 0.5
    assert h["cfl"][2] == h["cfl"][1]


def test_unsteady_retry_reuses_the_factorization():
    residual, jacobian, time_term, u0 = scalar_problem(9.0)
    settings = PseudoTimeSettings(
        cfl_start=1.0e8,
        residual_drop=12.0,
        step_control="unsteady",
        line_search_steps=0,
        refactor_interval=1,
    )
    records = []
    with np.errstate(invalid="ignore"):
        result = solve_steady(
            residual, jacobian, time_term, u0, settings, debug=records.append
        )
    h = result.history
    assert result.converged
    assert not h["accepted"][1] and h["cfl"][2] == 1.0e8 * settings.cut_factor
    # The retry at the lower CFL keeps the factorization of the failed step
    # (GMRES preconditioned by it); the step after it factorizes again.
    assert records[0]["factorized"] and not records[1]["factorized"]
    assert records[1]["gmres"] > 0
    assert records[2]["factorized"]


def test_merit_default_is_unchanged():
    """The default step control halves (1.2 behaviour)."""
    residual, jacobian, time_term, u0 = scalar_problem(9.0)
    settings = PseudoTimeSettings(cfl_start=1.0e8, residual_drop=12.0)
    with np.errstate(invalid="ignore"):
        result = solve_steady(residual, jacobian, time_term, u0, settings)
    assert set(result.history["step_factor"]) <= {1.0, 0.5, 0.25}


def test_cfl_floor_follows_the_residual_drop():
    residual, jacobian, time_term, u0 = scalar_problem(1.5)
    settings = PseudoTimeSettings(
        cfl_start=1.0e-2,
        growth_factor=1.0,
        cfl_floor_exponent=0.5,
        residual_drop=10.0,
        newton_krylov=False,
        max_iterations=400,
    )
    result = solve_steady(residual, jacobian, time_term, u0, settings)
    h = result.history
    assert result.converged
    r0 = h["residual"][0]
    for it in range(7, h["iteration"].size - 1):
        floor = 1.0e-2 * (r0 / h["residual"][it]) ** 0.5
        assert h["cfl"][it + 1] >= min(floor, settings.cfl_max_approximate) * (
            1 - 1e-12
        )


def test_source_limit_is_released_after_clean_newton_steps():
    residual, jacobian, _, u0 = scalar_problem(1.5)
    calls = []

    def time_term(u, cfl, source_limit=None):
        calls.append(source_limit)
        return sp.diags(np.ones(u.size) / cfl)

    jvp = lambda u, du: 0.5 / np.sqrt(u) * du  # noqa: E731
    settings = PseudoTimeSettings(
        nk_switch_residual_drop=0.0,
        residual_drop=14.0,
        source_cfl_limit=0.9,
        source_cfl_release=2,
    )
    result = solve_steady(residual, jacobian, time_term, u0, settings, jvp=jvp)
    assert result.converged
    assert calls[:2] == [0.9, 0.9]
    assert all(c is None for c in calls[2:])
    assert result.state["source_active"] is False


@pytest.mark.parametrize("turbulence", ["sa-neg", "sst"])
def test_source_cfl_cap_limits_the_time_term(turbulence):
    p = tp.solve(
        12,
        4,
        12,
        reynolds=1.0e5,
        turbulence=turbulence,
        settings=tp.PseudoTimeSettings(max_iterations=3),
    ).problem
    w = p.initial_state((1.0, 0.0))
    rng = np.random.default_rng(3)
    w = w * (1.0 + 0.1 * rng.random(w.shape))
    w[:, 1] = np.minimum(1.0, np.asarray(p.geom.wall_distance) / 0.01)
    params = p.scheme_parameters(w)
    j = np.asarray(p.physics.source_jacobian(p.model, p.geom, p.bc, w))
    nt = p.physics.NT
    assert j.shape == (w.shape[0], w.shape[1] - nt, w.shape[1] - nt)
    assert np.isfinite(j).all()
    if turbulence == "sa-neg":
        # Where the source decays, the full derivative is the implicit one.
        blocks = np.asarray(p.physics.source_blocks(p.model, p.geom, p.bc, w))
        decay = j[:, 0, 0] > 0.0
        np.testing.assert_allclose(j[decay, 0, 0], blocks[decay, nt, nt], rtol=1e-10)
    cap = p.source_cfl_cap(w, params, 0.9)
    assert (cap > 0.0).all()
    cfl = 10.0 * np.min(cap[np.isfinite(cap)]) if np.isfinite(cap).any() else 1e6
    plain = p.time_term(w, params, cfl).diagonal()
    limited = p.time_term(w, params, cfl, source_limit=0.9).diagonal()
    assert np.all(limited >= plain * (1 - 1e-12))
    assert p.source_limited_cells == int(np.count_nonzero(cap < cfl))
    assert p.source_limited_cells > 0


def test_control_loop_defaults_since_1_3():
    from pyaeroflow.config import resolve
    from pyaeroflow.run import _pseudo_time_settings

    raw = {
        "mesh": {"file": "m.su2"},
        "physics": {"model": "laminar"},
        "fluid": {"density": 1.0, "viscosity": 0.1},
        "reference": {"velocity": 1.0},
        "initial": {"velocity": [0.0, 0.0]},
        "boundary": {
            "bottom": {"type": "wall"},
            "top": {"type": "wall", "velocity": [1.0, 0.0]},
            "left": {
                "type": "periodic",
                "partner": "right",
                "translation": [2.0, 0.0],
            },
        },
    }
    s = _pseudo_time_settings(resolve(raw, "c.toml"))
    assert s.step_control == "unsteady" and s.source_cfl_limit == 0.9
    assert s.source_cfl_ramp == 1.0
    old = {
        **raw,
        "solver": {"step_control": "merit", "source_time_step": {"limit": 0.0}},
    }
    s = _pseudo_time_settings(resolve(old, "c.toml"))
    assert s.step_control == "merit" and s.source_cfl_limit is None


def test_source_limit_ramps_up_and_steps_back():
    residual, jacobian, _, u0 = scalar_problem(1.5)
    calls = []

    def time_term(u, cfl, source_limit=None):
        calls.append(source_limit)
        return sp.diags(np.ones(u.size) / cfl)

    jvp = lambda u, du: 0.5 / np.sqrt(u) * du  # noqa: E731
    settings = PseudoTimeSettings(
        nk_switch_residual_drop=0.0,
        residual_drop=14.0,
        source_cfl_limit=0.9,
        source_cfl_release=2,
        source_cfl_ramp=10.0,
    )
    result = solve_steady(residual, jacobian, time_term, u0, settings, jvp=jvp)
    assert result.converged
    # base for the first steps, then x10 per clean step, then off
    assert calls[0] == pytest.approx(0.9) and calls[1] == pytest.approx(0.9)
    levels = [c for c in calls if c is not None]
    assert levels == sorted(levels) and levels[-1] > 0.9 * 10
    assert calls[-1] is None or calls[-1] > 0.9

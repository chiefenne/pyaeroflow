"""Selective frequency damping in the steady solver (solver.sfd, off by default)."""

import numpy as np
import scipy.sparse as sp

from pyaeroflow.config import resolve
from pyaeroflow.solver import PseudoTimeSettings, solve_steady

SIGMA, OMEGA = 0.1, 1.0
MINIMAL = {
    "mesh": {"file": "m.su2"},
    "physics": {"model": "laminar"},
    "fluid": {"density": 1.0, "viscosity": 0.1},
    "reference": {"velocity": 1.0},
    "initial": {"velocity": [0.0, 0.0]},
    "boundary": {"wall": {"type": "wall"}},
}


def hopf():
    """Hopf normal form: unstable focus at u = 0, limit cycle of radius sqrt(sigma).

    du/dt = -R(u); at the fixed pseudo-time step 0.1 the implicit march
    follows the oscillation (period 2 pi, about 63 iterations) to the cycle.
    """

    def residual(u):
        x, y = u[:, 0], u[:, 1]
        g = SIGMA - (x * x + y * y)
        return -np.stack([g * x - OMEGA * y, OMEGA * x + g * y], axis=1)

    def jacobian(u):
        x, y = u[0]
        g = SIGMA - (x * x + y * y)
        return sp.csr_matrix(
            -np.array(
                [
                    [g - 2 * x * x, -OMEGA - 2 * x * y],
                    [OMEGA - 2 * x * y, g - 2 * y * y],
                ]
            )
        )

    def time_term(u, cfl):
        return sp.identity(2, format="csr") / cfl

    return residual, jacobian, time_term, np.array([[0.05, 0.0]])


def settings(**kw):
    base = dict(
        cfl_start=0.1,
        cfl_max=0.1,
        growth_factor=1.0,
        newton_krylov=False,
        refactor_interval=1,
        max_iterations=3000,
        residual_drop=8.0,
    )
    return PseudoTimeSettings(**{**base, **kw})


def test_sfd_damps_the_limit_cycle():
    residual, jacobian, time_term, u0 = hopf()
    weight = np.ones((1, 2))
    plain = solve_steady(
        residual, jacobian, time_term, u0, settings(), sfd_weight=weight
    )
    assert not plain.converged
    assert np.linalg.norm(plain.u) > 0.1  # on the cycle (radius 0.32)
    damped = solve_steady(
        residual,
        jacobian,
        time_term,
        u0,
        settings(sfd_chi=0.5, sfd_delta=60.0),
        sfd_weight=weight,
    )
    assert damped.converged
    np.testing.assert_allclose(damped.u, 0.0, atol=1e-8)
    # the history holds the residual of the problem, not the damped one
    np.testing.assert_allclose(
        damped.history["residual"][-1],
        np.sqrt(np.mean(residual(damped.u) ** 2)),
        rtol=1e-10,
    )


def test_sfd_off_without_chi_or_weight():
    residual, jacobian, time_term, u0 = hopf()
    a = solve_steady(residual, jacobian, time_term, u0, settings(max_iterations=50))
    b = solve_steady(
        residual,
        jacobian,
        time_term,
        u0,
        settings(max_iterations=50, sfd_chi=0.5),  # no weight: off
    )
    np.testing.assert_array_equal(a.u, b.u)
    assert "sfd_filter" not in a.state and "sfd_filter" not in b.state


def test_sfd_start_reset_and_continuation():
    residual, jacobian, time_term, u0 = hopf()
    weight = np.ones((1, 2))
    s = settings(sfd_chi=0.5, sfd_delta=60.0, sfd_reset=200, sfd_start=100)
    full = solve_steady(residual, jacobian, time_term, u0, s, sfd_weight=weight)
    assert full.converged
    # first part, then a continuation from its state: identical result
    part = solve_steady(
        residual,
        jacobian,
        time_term,
        u0,
        PseudoTimeSettings(**{**s.__dict__, "max_iterations": 300}),
        sfd_weight=weight,
    )
    assert part.state["sfd_since"] == 200
    rest = solve_steady(
        residual,
        jacobian,
        time_term,
        part.u,
        s,
        resume=part.state,
        sfd_weight=weight,
    )
    np.testing.assert_allclose(rest.u, full.u, atol=1e-14)
    assert rest.state["iteration"] == full.state["iteration"]


def test_sfd_options_reach_the_schema():
    raw = {
        **MINIMAL,
        "solver": {"sfd": {"chi": 2.0, "delta": 260, "reset_interval": 600}},
    }
    cfg = resolve(raw, "c.toml")
    sfd = cfg["solver"]["sfd"]
    assert (sfd["chi"], sfd["delta"], sfd["reset_interval"]) == (2.0, 260, 600)
    assert sfd["start_iteration"] == 0
    assert resolve(MINIMAL, "c.toml")["solver"]["sfd"]["chi"] is None

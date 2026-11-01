"""Dual time stepping for unsteady problems (D9, D37).

The physical time derivative is discretised with BDF2 (BDF1 for the first
step, or throughout with ``scheme="bdf1"``) and added to the steady residual:

    R*(u) = M (a0 u + a1 u^n + a2 u^(n-1)) + R(u),

with a diagonal mass ``M`` (cell area times density for the momentum rows,
zero for the continuity row of the incompressible equations). Every time
step drives R* to zero with the steady solver (:func:`solve_steady`:
pseudo-time, SER, Newton-Krylov), starting from the extrapolated state
2 u^n - u^(n-1).
"""

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

from pyaeroflow.solver.pseudo_time import (
    PseudoTimeSettings,
    rms_per_variable,
    solve_steady,
)


class DualTimeSettings(NamedTuple):
    """Settings of :func:`solve_unsteady`.

    Attributes
    ----------
    time_step : float
        Physical time step (non-dimensional inside the solver).
    end_time : float
        Final time; the number of steps is round(end_time / time_step).
    scheme : {"bdf2", "bdf1"}
    inner : PseudoTimeSettings
        Settings of the inner (pseudo-time) iterations; ``residual_drop`` is
        the inner drop per step, ``max_iterations`` the inner limit. The
        physical time term stabilises the inner system, so the pseudo-time
        CFL starts large (1e4 by default) and Newton-Krylov is used from the
        first inner iteration; a step then needs one or two inner iterations
        in smooth flow.
    extrapolate : bool
        Start every step from 2 u^n - u^(n-1) instead of u^n.
    """

    time_step: float
    end_time: float
    scheme: str = "bdf2"
    inner: PseudoTimeSettings = PseudoTimeSettings(
        cfl_start=1.0e4,
        residual_drop=3.0,
        max_iterations=50,
        nk_switch_residual_drop=0.0,
    )
    extrapolate: bool = True


class UnsteadyResult(NamedTuple):
    """Result of :func:`solve_unsteady`.

    Attributes
    ----------
    u : ndarray
        Solution at the final time.
    time : float
    converged : bool
        True if every time step reached the inner residual drop.
    history : dict of str to ndarray
        Per time step: ``step``, ``time``, ``residual`` (inner RMS norm at
        the end of the step), ``residuals`` (per variable), ``inner_iterations``,
        ``inner_drop`` (orders reached relative to R*(u^n)), ``converged``,
        ``linear_iterations``
        (GMRES iterations of the step).
    u_previous : ndarray or None
        Solution one time step before ``u`` (the second level of BDF2); a
        restart needs it to continue with BDF2 at once.
    """

    u: np.ndarray
    time: float
    converged: bool
    history: dict
    u_previous: np.ndarray = None


def bdf_coefficients(step, dt, scheme):
    """Coefficients (a0, a1, a2) of M (a0 u + a1 u^n + a2 u^(n-1))."""
    if scheme == "bdf1" or step == 1:
        return 1.0 / dt, -1.0 / dt, 0.0
    if scheme == "bdf2":
        return 1.5 / dt, -2.0 / dt, 0.5 / dt
    raise ValueError("scheme must be 'bdf2' or 'bdf1'")


def solve_unsteady(
    residual,
    jacobian,
    time_term,
    mass,
    u0,
    settings,
    callback=None,
    inner_callback=None,
    jvp=None,
    linearize=None,
    limit_step=None,
    resume=None,
):
    """Advance ``M du/dt + R(u) = 0`` from ``u0`` to ``settings.end_time``.

    Parameters
    ----------
    residual, jacobian, time_term, jvp, linearize
        As in :func:`solve_steady` (steady residual and its operators).
    mass : array, shape (n_cells, n_var)
        Diagonal of the mass matrix M.
    u0 : array, shape (n_cells, n_var)
    settings : DualTimeSettings
    callback : callable, optional
        ``callback(step, time, u, residuals, inner_iterations, inner_drop,
        converged)`` after every time step; ``callback.info`` may be a dict
        that receives ``linear_iterations``. Setting ``callback.info["stop"]``
        ends the run after that step.
    inner_callback : callable, optional
        Passed to the inner :func:`solve_steady` (per inner iteration).
    limit_step : callable, optional
        Passed to the inner :func:`solve_steady` (limits every inner update).
    resume : dict, optional
        ``step`` (the steps already done; ``u0`` is the solution after them)
        and ``u_previous`` (the solution after step - 1, or None: the first
        step then uses BDF1). ``end_time`` stays the total: the run continues
        with step + 1 and the time is step * time_step. ``callback.info``
        receives ``u_previous`` and ``time_step`` before every call.

    Returns
    -------
    UnsteadyResult
    """
    s = settings
    dt = s.time_step
    n_steps = int(round(s.end_time / dt))
    mass = np.asarray(mass, dtype=float)
    m_diag = sp.diags(mass.ravel())
    u_n = np.array(u0, dtype=float)
    resume = resume or {}
    step0 = int(resume.get("step", 0))
    u_nm1 = resume.get("u_previous")
    u_nm1 = None if u_nm1 is None else np.array(u_nm1, dtype=float)
    hist = {
        k: []
        for k in (
            "step",
            "time",
            "residual",
            "residuals",
            "inner_iterations",
            "inner_drop",
            "converged",
            "linear_iterations",
        )
    }
    all_converged = True
    t = step0 * dt

    for step in range(step0 + 1, n_steps + 1):
        a0, a1, a2 = bdf_coefficients(1 if u_nm1 is None else step, dt, s.scheme)
        source = mass * (a1 * u_n + (a2 * u_nm1 if u_nm1 is not None else 0.0))

        def res_t(u, a0=a0, source=source):
            return np.asarray(residual(u)) + a0 * mass * u + source

        def jac_t(u, a0=a0):
            return jacobian(u) + a0 * m_diag

        jvp_t = None
        if jvp is not None:

            def jvp_t(u, du, a0=a0):
                return np.asarray(jvp(u, du)) + a0 * mass * du

        lin_t = None
        if linearize is not None:

            def lin_t(u, a0=a0):
                product = linearize(u)
                return lambda du: np.asarray(product(du)) + a0 * mass * du

        # The inner residual drop refers to R*(u^n): extrapolation lowers the
        # starting residual, not the accuracy required.
        start, norm_n = u_n, None
        if s.extrapolate and u_nm1 is not None:
            start, norm_n = 2.0 * u_n - u_nm1, rms_per_variable(res_t(u_n))
        inner = solve_steady(
            res_t,
            jac_t,
            time_term,
            start,
            s.inner,
            callback=inner_callback,
            jvp=jvp_t,
            linearize=lin_t,
            reference_norm=norm_n,
            limit_step=limit_step,
        )
        h = inner.history
        t = step * dt
        u_nm1, u_n = u_n, inner.u
        # Inner drop in orders: the per-equation measure (relative to R*(u^n)
        # or to the inner start).
        drop = float(-np.log10(max(h["measure"][-1], np.finfo(float).tiny)))
        n_inner = int(h["iteration"][-1])
        lin = int(np.sum(h["linear_iterations"]))
        all_converged &= inner.converged
        hist["step"].append(step)
        hist["time"].append(t)
        hist["residual"].append(h["residual"][-1])
        hist["residuals"].append(h["residuals"][-1])
        hist["inner_iterations"].append(n_inner)
        hist["inner_drop"].append(drop)
        hist["converged"].append(inner.converged)
        hist["linear_iterations"].append(lin)
        if callback is not None:
            if hasattr(callback, "info"):
                callback.info["linear_iterations"] = lin
                callback.info["u_previous"] = u_nm1
                callback.info["time_step"] = dt
            callback(step, t, u_n, h["residuals"][-1], n_inner, drop, inner.converged)
            if getattr(callback, "info", {}).get("stop"):
                break  # the callback asked for a clean stop (control file)

    history = {k: np.asarray(v) for k, v in hist.items()}
    return UnsteadyResult(u_n, t, all_converged, history, u_nm1)

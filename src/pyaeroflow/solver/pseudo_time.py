"""Implicit pseudo-time stepping with SER CFL control (D25, D35, D36).

Each iteration solves

    (T(u, CFL) + J(u)) du = -R(u),

with the pseudo-time matrix T and a Jacobian J. For a scalar equation
T = diag(lambda_i / CFL) with the spectral radius lambda_i of cell i (the
pseudo-time step is dtau_i = CFL A_i / lambda_i); for the preconditioned flow
equations T holds the blocks Gamma_i lambda_i / CFL.

Two phases (D25, as SU2's ``NEWTON_KRYLOV``):

1. Start-up with the approximate first-order Jacobian J1, solved by the
   sparse linear solver (direct LU or GMRES-ILU).
2. Newton-Krylov once the residual has dropped by ``nk_switch_residual_drop``
   orders: the exact Jacobian of the full residual is applied matrix-free
   through ``jvp`` (JAX forward mode), solved by GMRES with the factorization
   of T + J1 as preconditioner.

The CFL number follows switched evolution relaxation (Mulder & van Leer
1985):

    CFL_n = min(CFL_max, CFL_(n-1) * max(|R_(n-1)| / |R_n|, growth_factor)),

i.e. the CFL number grows at least by ``growth_factor`` per accepted
iteration while the residual decreases (SU2 raises its CFL by a factor 2
under the same condition); during a slow transient SER alone would keep it
small. Safeguards: if the linear solve fails, the update is not finite, or the
residual grows by more than ``reject_factor``, the update is rejected and the
CFL number is multiplied by ``cut_factor``.

``cfl_control = "ser"`` replaces this growth control by proportional SER on
the RMS norm (D55):

    CFL_n = CFL_(n-1) * clip((|R_(n-1)| / |R_n|)^p, ratio_min, ratio_max),

with at least ``ser_growth_min`` while the norm falls, and in the
Newton-Krylov phase modulated by the
GMRES effort of the step instead of the adaptive cap: full factor below
``ser_gmres_hold`` iterations, at most 1 (hold) up to ``ser_gmres_reduce``,
at most 0.5 above.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from pyaeroflow.solver.linear import (
    LinearSettings,
    LinearSolverError,
    factorize,
    solve_matrix_free,
)
from pyaeroflow.solver.profile import timers


@dataclass(frozen=True)
class PseudoTimeSettings:
    """Settings of the pseudo-time iteration.

    Attributes
    ----------
    cfl_start : float
        Initial CFL number.
    cfl_max : float
        Upper limit of the CFL number.
    cfl_max_approximate : float
        Upper limit of the CFL number while the Newton-Krylov switch is
        pending (Newton-Krylov enabled and ``jvp`` given, but the switch
        condition not yet met). A pseudo-time step at a very large CFL is
        a Newton step with the approximate first-order Jacobian and does
        not converge for the second-order residual; SU2, which has only
        this Jacobian, is run at CFL <= 1e3. Without Newton-Krylov
        ``cfl_max`` applies throughout.
    cut_factor : float
        CFL reduction after a rejected update.
    reject_factor : float
        An update is rejected if it raises the residual norm by more than
        this factor.
    cfl_min : float
        The iteration stops (not converged) if the CFL number drops below.
    max_iterations : int
        Maximum number of iterations, rejected ones included.
    residual_drop : float
        Convergence: orders of magnitude of residual reduction.
    residual_floor : float
        Absolute RMS residual below which the iteration counts as converged
        regardless of the relative drop (a restart from a converged state).
    first_order_iterations : int
        Iterations evaluated with ``residual_startup`` before switching to
        ``residual`` (D36).
    growth_factor : float
        Minimum CFL growth per accepted iteration with decreasing residual.
    newton_krylov : bool
        Switch to Newton-Krylov (needs ``jvp``).
    nk_switch_residual_drop : float
        Residual drop (orders of magnitude) at which the switch happens;
        0 switches at the first iteration. The switch latches: Newton-Krylov
        stays on when the measure later rises again (e.g. while the
        turbulence field develops).
    nk_switch_iteration : int or None
        Switch at this iteration at the latest (the approximate-Jacobian
        iteration typically stalls two to three orders below the start).
    nk_linear_tolerance : float
        Relative tolerance of the GMRES solve in the Newton-Krylov phase.
    nk_restart : int
        GMRES restart length in the Newton-Krylov phase.
    nk_max_krylov : int
        Maximum GMRES iterations per Newton step.
    nk_forcing : bool
        Eisenstat-Walker forcing: the GMRES tolerance of a Newton step is
        0.9 (|R_k| / |R_{k-1}|)^2, bounded by ``nk_linear_tolerance`` from
        below and 0.5 from above, so that the linear solve is only as
        accurate as the nonlinear progress warrants (Knoll & Keyes 2004).
    line_search_steps : int
        Backtracking: an update that raises the residual measure is halved
        up to this many times before it is rejected (D54).
    nk_cfl_adapt_iterations : int
        GMRES iteration count above which the CFL of the Newton-Krylov phase
        is capped (quartered) for the following steps, and a quarter of
        which lets the cap grow again; keeps the first-order preconditioner
        effective on stretched meshes (D52, P14).
    refactor_interval : int
        Rebuild the factorized first-order system (Jacobian plus pseudo-time
        term) only every n-th iteration and reuse it otherwise, in both
        phases: as a lagged matrix for the pseudo-time step and as the
        preconditioner of Newton-Krylov. It is rebuilt earlier after a
        rejected update, when the CFL has grown by more than
        ``refactor_cfl_ratio`` since the factorization, and after a
        Newton-Krylov step that needed more than ``nk_cfl_adapt_iterations
        // 4`` GMRES iterations (a stale preconditioner costs more there than
        it saves). The factorization dominates the cost on large meshes.
        1 = every iteration.
    refactor_cfl_ratio : float
        CFL growth since the last factorization that forces a new one.
    cfl_control : {"growth", "ser"}
        CFL control: ``growth`` (SER on the RMS norm with at least
        ``growth_factor`` per accepted iteration, adaptive cap from the
        GMRES effort) or ``ser`` (proportional SER on the RMS norm,
        bounded per step, GMRES-modulated; D55).
    ser_exponent : float
        Exponent p of the SER ratio (``cfl_control = "ser"``).
    ser_growth_min : float
        Minimum CFL factor while the RMS norm falls (``ser``).
    ser_ratio_min, ser_ratio_max : float
        Bounds of the CFL factor per accepted iteration (``ser``).
    ser_gmres_hold : int
        Newton-Krylov steps with at least this many GMRES iterations do not
        raise the CFL (``ser``).
    ser_gmres_reduce : int
        Newton-Krylov steps with more GMRES iterations than this halve the
        CFL at least (``ser``).
    linear : LinearSettings
        Sparse linear solver of phase 1 and preconditioner of phase 2.
    """

    cfl_start: float = 10.0
    cfl_max: float = 1.0e10
    cfl_max_approximate: float = 1.0e3
    cut_factor: float = 0.1
    reject_factor: float = 10.0
    cfl_min: float = 1.0e-6
    max_iterations: int = 1000
    residual_drop: float = 8.0
    residual_floor: float = 1.0e-14
    first_order_iterations: int = 0
    growth_factor: float = 2.0
    newton_krylov: bool = True
    nk_switch_residual_drop: float = 2.0
    nk_switch_iteration: int = None
    nk_linear_tolerance: float = 1.0e-2
    nk_restart: int = 50
    nk_max_krylov: int = 200
    refactor_interval: int = 3
    refactor_cfl_ratio: float = 4.0
    nk_cfl_adapt_iterations: int = 60
    nk_forcing: bool = True
    line_search_steps: int = 2
    cfl_control: str = "growth"
    ser_exponent: float = 1.0
    ser_growth_min: float = 1.2
    ser_ratio_min: float = 0.3
    ser_ratio_max: float = 2.0
    ser_gmres_hold: int = 10
    ser_gmres_reduce: int = 30
    linear: LinearSettings = LinearSettings()


class SteadyResult(NamedTuple):
    """Result of :func:`solve_steady` and :func:`solve_explicit`.

    Attributes
    ----------
    u : ndarray, shape (n_cells, n_var)
        Final solution.
    converged : bool
        True if the residual dropped by ``residual_drop`` orders.
    history : dict of str to ndarray
        Per iteration: ``iteration``, ``residual`` (RMS norm after the
        iteration), ``measure`` (per-equation convergence measure, D54),
        ``residuals`` (RMS norm per variable), ``cfl``,
        ``accepted``, ``newton_krylov`` (phase flag), ``linear_iterations``
        (GMRES iterations of the step, 0 for a direct solve).
    state : dict or None
        Solver state at the end (``cfl``, ``cfl_cap``, ``scale``, ``nk_on``,
        ``iteration``), the ``resume`` argument of a continuation.
    """

    u: np.ndarray
    converged: bool
    history: dict
    state: dict = None


def rms(r):
    """Root-mean-square norm of a residual array (inf for a diverged trial step)."""
    with np.errstate(over="ignore"):
        return float(np.sqrt(np.mean(np.square(r))))


def rms_per_variable(r):
    """RMS norm of each column of a residual array."""
    with np.errstate(over="ignore"):
        return np.sqrt(np.mean(np.square(r), axis=0))


def solve_steady(
    residual,
    jacobian,
    time_term,
    u0,
    settings=None,
    residual_startup=None,
    callback=None,
    jvp=None,
    reference_norm=None,
    limit_step=None,
    resume=None,
):
    """Drive ``residual(u)`` to zero by implicit pseudo-time stepping.

    Parameters
    ----------
    residual : callable
        ``residual(u) -> (n_cells, n_var)`` array.
    jacobian : callable
        ``jacobian(u) -> scipy.sparse`` approximate Jacobian dR/du.
    time_term : callable
        ``time_term(u, cfl) -> scipy.sparse`` pseudo-time matrix T.
    u0 : array, shape (n_cells, n_var)
        Initial solution.
    settings : PseudoTimeSettings, optional
    residual_startup : callable, optional
        Residual used for the first ``settings.first_order_iterations``
        iterations (e.g. first-order convective states).
    callback : callable, optional
        ``callback(iteration, u, residual_norms, cfl, accepted)`` after every
        iteration (e.g. logging). Setting ``callback.info["stop"]`` ends the
        run after that iteration; ``callback.info["converged"]`` ends it as
        converged (e.g. a Cauchy criterion on the force coefficients). Setting
        ``callback.info["max_iterations"]`` or ``["residual_drop"]`` changes
        these limits of the running iteration. ``callback.info`` may be set
        beforehand to a dict that receives ``newton_krylov`` and
        ``linear_iterations``.
    jvp : callable, optional
        ``jvp(u, du) -> dR/du(u) @ du`` of the full residual; enables the
        Newton-Krylov phase.
    reference_norm : array, optional
        Per-variable residual norms that ``residual_drop`` and the
        Newton-Krylov switch refer to; default the norms of the initial
        residual.
    limit_step : callable, optional
        ``limit_step(u, du, r, tau) -> du`` applied to every computed update
        before the line search: ``r`` is the residual of ``u`` and ``tau`` the
        diagonal of the pseudo-time matrix at CFL 1 (area / dt per cell and
        variable, shaped like ``u``), so ``-r / tau`` is the explicit local
        step of a cell (e.g. a cap on the turbulence variable's change).
    resume : dict, optional
        Solver state of an earlier run (``SteadyResult.state``). With
        ``iteration`` the run continues that run: the counting starts after
        that iteration (``settings.max_iterations`` is the total), the
        per-variable residual peaks (``scale``) are kept, so that
        ``residual_drop`` still refers to the original start, and the
        iteration-triggered phases (first-order start-up, Newton-Krylov
        switch) are not repeated. Without ``iteration`` only ``cfl``,
        ``cfl_cap`` and ``nk_on`` are taken over (a new problem, e.g. the next
        angle of attack, that starts from the earlier solution). The initial
        callback (iteration 0) is not called on a continuation.

    Convergence is measured per equation (D54): every variable's RMS
    residual is divided by the largest value it has had so far (or by
    ``reference_norm``), and the residual measure is the maximum of these
    ratios, so that an equation whose residual is small in absolute terms
    (the turbulence equation) still has to converge. Variables whose peak
    residual stays below 10^-12 of the largest peak (round-off residuals of
    a symmetric or one-dimensional problem) are left out; the SA residual
    is about 1/Re times the momentum residual, so it stays in. Step acceptance,
    the line search, the CFL control and the forcing term use the plain RMS
    norm of the whole residual, as before.

    Returns
    -------
    SteadyResult
    """
    s = settings or PseudoTimeSettings()
    u = np.array(u0, dtype=float)

    def res(u, it):
        with timers.section("residual"):
            if residual_startup is not None and it <= s.first_order_iterations:
                return np.asarray(residual_startup(u))
            return np.asarray(residual(u))

    resume = resume or {}
    it0 = int(resume.get("iteration", 0))
    r = res(u, it0 + 1)
    tiny = np.finfo(float).tiny
    scale = np.maximum(
        rms_per_variable(r) if reference_norm is None else np.asarray(reference_norm),
        tiny,
    )
    if "scale" in resume and it0:
        scale = np.maximum(np.asarray(resume["scale"], dtype=float), tiny)

    def measure(r):
        active = scale >= 1e-12 * scale.max()
        return float(np.max(np.where(active, rms_per_variable(r) / scale, 0.0)))

    norm_abs = rms(r)
    norm = measure(r)
    target = 10.0 ** (-s.residual_drop)

    def done(norm, norm_abs):
        return norm <= target or norm_abs <= s.residual_floor

    nk_threshold = 10.0 ** (-s.nk_switch_residual_drop)
    cfl = float(resume.get("cfl", s.cfl_start))
    # Forcing term of the Newton-Krylov steps: part of the state that a
    # continuation takes over (restarted at the tight tolerance, the first step
    # solves the linear system far more accurately than the run before did).
    eta = float(resume.get("eta", s.nk_linear_tolerance))
    hist = {
        "iteration": [it0],
        "residual": [norm_abs],
        "measure": [norm],
        "residuals": [rms_per_variable(r)],
        "cfl": [cfl],
        "accepted": [True],
        "newton_krylov": [False],
        "linear_iterations": [0],
    }
    converged = done(norm, norm_abs)
    # CFL cap of the Newton-Krylov phase, adapted to the GMRES effort: the
    # first-order preconditioner degrades as the pseudo-time term vanishes,
    # so expensive linear solves lower the cap and cheap ones raise it.
    cfl_cap = float(resume.get("cfl_cap", s.cfl_max))
    nk_on = bool(resume.get("nk_on", False))  # the switch latches
    # Live view of the solver state for the callback (restart files).
    state = {}

    def snapshot():
        return {
            "iteration": state["it"],
            "cfl": cfl,
            "cfl_cap": cfl_cap,
            "scale": scale.copy(),
            "nk_on": nk_on,
            "eta": eta,
        }

    state["it"] = it0
    info = getattr(callback, "info", None)
    if info is not None:
        info["solver_state"] = snapshot
        # The residual array of the last accepted iteration (field output).
        info["cell_residual"] = lambda: r
    if callback is not None:
        if info is not None:
            info["measure"] = norm
        if not it0:
            callback(0, u, hist["residuals"][0], cfl, True)
    lin, steps_since_factor, cfl_at_factor = None, 0, cfl
    stale = False  # the preconditioner let GMRES work hard: rebuild it
    max_iterations = s.max_iterations
    it = it0
    while it < max_iterations:
        it += 1
        state["it"] = it - 1
        if converged or cfl < s.cfl_min:
            break
        info = getattr(callback, "info", None)
        if info is not None and info.pop("residual_changed", False):
            # The callback changed the residual operator (limiter frozen or
            # released): the reference for acceptance and convergence is the
            # residual of the current state under the new operator, not the
            # one the previous step produced.
            r = res(u, it)
            norm_abs = rms(r)
            scale = np.maximum(scale, rms_per_variable(r))
            norm = measure(r)
        startup = residual_startup is not None and it <= s.first_order_iterations
        if (
            s.first_order_iterations
            and it == s.first_order_iterations + 1
            and residual_startup is not None
        ):
            r = res(u, it)  # switch to the full residual
            norm_abs = rms(r)
            scale = np.maximum(scale, rms_per_variable(r))
            norm = measure(r)
        by_iteration = s.nk_switch_iteration is not None and it > s.nk_switch_iteration
        immediate = s.nk_switch_residual_drop <= 0.0  # dual-time inner iterations
        nk_on = nk_on or immediate or norm <= nk_threshold or by_iteration
        use_nk = s.newton_krylov and jvp is not None and not startup and nk_on
        with timers.section("time term"):
            t_matrix = time_term(u, cfl)
        accepted, lin_its = False, 0
        try:
            reuse = (
                lin is not None
                and steps_since_factor < s.refactor_interval
                and cfl <= cfl_at_factor * s.refactor_cfl_ratio
                and not stale
            )
            if not reuse:
                with timers.section("assemble matrix"):
                    matrix = jacobian(u) + t_matrix
                with timers.section("factorize"):
                    lin = factorize(matrix, s.linear)
                del matrix
                steps_since_factor, cfl_at_factor = 0, cfl
            steps_since_factor += 1
            if use_nk:
                with timers.section("nk solve (GMRES)"):
                    du, lin_its = solve_matrix_free(
                        _newton_operator(t_matrix, jvp, u),
                        -r.ravel(),
                        _timed("preconditioner", lin.precondition),
                        tolerance=eta if s.nk_forcing else s.nk_linear_tolerance,
                        restart=s.nk_restart,
                        max_iterations=s.nk_max_krylov,
                    )
            else:
                with timers.section("direct solve"):
                    du = lin.solve(-r.ravel())
                lin_its = lin.iterations
            du = du.reshape(u.shape)
            if limit_step is not None:
                tau = (t_matrix.diagonal() * cfl).reshape(u.shape)
                du = limit_step(u, du, r, tau)
            u_new = u + du
            r_new = res(u_new, it + 1)
            abs_new = rms(r_new)
            # Backtracking line search (D54): halve a step that raises the
            # residual norm before giving up on it.
            step = 1.0
            for _ in range(s.line_search_steps):
                if np.isfinite(abs_new) and abs_new <= norm_abs:
                    break
                step *= 0.5
                u_new = u + step * du
                r_new = res(u_new, it + 1)
                abs_new = rms(r_new)
            accepted = np.isfinite(abs_new) and abs_new <= s.reject_factor * norm_abs
        except LinearSolverError:
            pass
        hist["iteration"].append(it)
        hist["cfl"].append(cfl)
        hist["accepted"].append(accepted)
        hist["newton_krylov"].append(use_nk)
        hist["linear_iterations"].append(lin_its)
        if accepted:
            ser = s.cfl_control == "ser"
            if ser:
                # Proportional SER on the plain RMS norm (D54, D55), with a
                # minimum growth while it falls: the per-equation measure
                # stays at 1 while one equation sets new peaks (nu_tilde
                # during a slow transient) and would shrink the CFL.
                ratio = (norm_abs / max(abs_new, tiny)) ** s.ser_exponent
                if ratio > 1.0:
                    ratio = max(ratio, s.ser_growth_min)
                ratio = min(max(ratio, s.ser_ratio_min), s.ser_ratio_max)
                if use_nk and lin_its > s.ser_gmres_reduce:
                    ratio = min(ratio, 0.5)
                elif use_nk and lin_its >= s.ser_gmres_hold:
                    ratio = min(ratio, 1.0)
            else:
                ratio = norm_abs / max(abs_new, tiny)
                if ratio > 1.0:
                    ratio = max(ratio, s.growth_factor)
            if use_nk:
                if not ser:
                    if lin_its >= s.nk_cfl_adapt_iterations:
                        cfl_cap = max(cfl * 0.25, s.cfl_start)
                    elif lin_its <= s.nk_cfl_adapt_iterations // 4:
                        cfl_cap = min(cfl_cap * 4.0, s.cfl_max)
                stale = lin_its > s.nk_cfl_adapt_iterations // 4
            # Cap the CFL while the Newton-Krylov switch is pending: a huge
            # pseudo-time step with the approximate Jacobian does not converge.
            waiting = s.newton_krylov and jvp is not None and not use_nk
            limit = min(cfl_cap, s.cfl_max_approximate) if waiting else cfl_cap
            cfl = min(limit, cfl * ratio)
            if use_nk:
                eta = min(
                    0.5, max(s.nk_linear_tolerance, 0.9 * (abs_new / norm_abs) ** 2)
                )
            u, r, norm_abs = u_new, r_new, abs_new
            # Peaks of the per-variable residuals set the scales (D54).
            scale = np.maximum(scale, rms_per_variable(r))
            norm = measure(r)
            converged = done(norm, norm_abs)
        else:
            cfl *= s.cut_factor
            lin = None  # rebuild the preconditioner after a rejection
        hist["residual"].append(norm_abs)
        hist["measure"].append(norm)
        hist["residuals"].append(rms_per_variable(r))
        if callback is not None:
            info = getattr(callback, "info", None)
            if info is not None:
                info["newton_krylov"] = use_nk
                info["linear_iterations"] = lin_its
                info["measure"] = norm
            state["it"] = it
            callback(it, u, hist["residuals"][-1], hist["cfl"][-1], accepted)
            if info is not None and info.get("converged"):
                converged = True  # the callback judged a force criterion met
                break
            if info is not None and info.get("stop"):
                break  # the callback asked for a clean stop (control file)
            if info is not None:  # limits the callback changed (.read_toml_N)
                max_iterations = int(info.get("max_iterations", max_iterations))
                if "residual_drop" in info:
                    target = 10.0 ** (-float(info["residual_drop"]))
        state["it"] = it
    history = {k: np.asarray(v) for k, v in hist.items()}
    return SteadyResult(u, bool(converged), history, snapshot())


def _newton_operator(t_matrix, jvp, u):
    """Return v -> (T + J(u)) v for the matrix-free Newton-Krylov solve."""

    def apply(v):
        with timers.section("Jacobian product"):
            return t_matrix @ v + np.asarray(jvp(u, v.reshape(u.shape))).ravel()

    return apply


def _timed(name, func):
    """Wrap ``func`` so that every call is timed under ``name``."""

    def wrapped(x):
        with timers.section(name):
            return func(x)

    return wrapped


def solve_explicit(
    residual,
    update,
    u0,
    cfl=0.8,
    max_iterations=10000,
    residual_drop=8.0,
    stages=(1.0 / 3.0, 0.5, 1.0),
    callback=None,
):
    """Explicit multi-stage pseudo-time stepping, reference mode (D25).

    Stage k: u_k = u_n - alpha_k * update(u_(k-1), R(u_(k-1)), cfl), where
    ``update`` returns dtau_i Gamma_i^-1 R_i / A_i, i.e. the explicit
    increment for CFL = cfl.

    Parameters
    ----------
    residual : callable
    update : callable
        ``update(u, r, cfl) -> (n_cells, n_var)``.
    u0 : array
    cfl : float
    max_iterations : int
    residual_drop : float
    stages : tuple of float
        Runge-Kutta stage coefficients.
    callback : callable, optional

    Returns
    -------
    SteadyResult
    """
    u = np.array(u0, dtype=float)
    r = np.asarray(residual(u))
    norm = rms(r)
    target = max(norm, np.finfo(float).tiny) * 10.0 ** (-residual_drop)
    hist = {
        "iteration": [0],
        "residual": [norm],
        "residuals": [rms_per_variable(r)],
        "cfl": [cfl],
        "accepted": [True],
        "newton_krylov": [False],
        "linear_iterations": [0],
    }
    converged = norm <= target
    if callback is not None:
        callback(0, u, hist["residuals"][0], cfl, True)
    for it in range(1, max_iterations + 1):
        if converged:
            break
        u_stage = u
        for alpha in stages:
            r_stage = r if alpha == stages[0] else np.asarray(residual(u_stage))
            u_stage = u - alpha * np.asarray(update(u_stage, r_stage, cfl))
        u = u_stage
        r = np.asarray(residual(u))
        norm = rms(r)
        if not np.isfinite(norm):
            break
        converged = norm <= target
        hist["iteration"].append(it)
        hist["residual"].append(norm)
        hist["residuals"].append(rms_per_variable(r))
        hist["cfl"].append(cfl)
        hist["accepted"].append(True)
        hist["newton_krylov"].append(False)
        hist["linear_iterations"].append(0)
        if callback is not None:
            callback(it, u, hist["residuals"][-1], cfl, True)
            if getattr(callback, "info", {}).get("stop"):
                break  # the callback asked for a clean stop (control file)
    history = {k: np.asarray(v) for k, v in hist.items()}
    return SteadyResult(u, bool(converged), history)

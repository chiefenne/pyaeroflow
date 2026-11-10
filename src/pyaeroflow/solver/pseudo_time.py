"""Implicit pseudo-time stepping with SER CFL control (D25, D35, D36).

Each iteration solves

    (T(u, CFL) + J(u)) du = -R(u),

with the pseudo-time matrix T and a Jacobian J. For a scalar equation
T = diag(lambda_i / CFL) with the spectral radius lambda_i of cell i (the
pseudo-time step is dtau_i = CFL A_i / lambda_i); for the preconditioned flow
equations T holds the blocks Gamma_i lambda_i / CFL.

Two phases (D25, Jacobian-free Newton-Krylov, Knoll & Keyes 2004):

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
iteration while the residual decreases (a common choice is a factor 2
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
from functools import partial
from typing import NamedTuple

import numpy as np

from pyaeroflow.solver.linear import (
    LinearSettings,
    LinearSolverError,
    factorize,
    reset_analysis,
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
        not converge for the second-order residual; a solver with only
        this Jacobian is run at CFL <= 1e3. Without Newton-Krylov
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
    nk_linearize : bool
        Linearize the residual once per Newton step (``jax.linearize``) and
        apply the stored tangent in the GMRES products, instead of a full
        ``jax.jvp`` per product (C1). Needs ``linearize``; False uses ``jvp``
        (less memory, slower products).
    line_search_steps : int
        Backtracking: an update that raises the residual measure is halved
        up to this many times before it is rejected (D54).
    nk_variable_scaling : bool
        Measure the GMRES error of every variable relative to its size: the
        unknowns of the Newton-Krylov solve are du_j / s_j with s_j = 1 for
        p, u, v (non-dimensional) and the RMS of the field for the turbulence
        and transition variables. SciPy's GMRES is left-preconditioned, so it
        minimizes about the error of du in raw units: with Re_thetat ~ 1e3 and
        k ~ 1e-3 it solves the Re_thetat rows and leaves the gamma rows
        unsolved (320k SD7003 LM: 95 % of the gamma residual left in the
        linear model at tolerance 0.5).
    merit_columns : tuple of int or None
        Residual columns of the RMS norm that the line search, the step
        acceptance and the SER CFL control judge; None = all. Langtry-Menter
        runs can leave out gamma and Re_thetat: their raw residual is largest
        in the big outer cells (outlet, farfield), where Re_thetat^eq follows
        tiny velocity gradients, and would otherwise veto good flow steps.
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
    step_control : {"merit", "unsteady"}
        Step acceptance. ``merit`` (1.2 behaviour): halve a step that raises
        the RMS of the steady residual (``line_search_steps`` times) and
        reject it only if it still raised the RMS more than
        ``reject_factor``-fold. ``unsteady`` (HANIM, Wang, Diskin, Nielsen &
        Liu 2021): judge the step by the pseudo-time residual
        R_t(u + a du) = a T du + R(u + a du), which equals R(u) at a = 0 and
        vanishes to first order at a = 1 for the exact linear solve; a step
        must lower its RMS. A full step that does not is under-relaxed by the
        minimum of a quadratic fit of |R_t(a)|^2 (bounded to [0.1, 0.5] of
        the last trial, up to ``line_search_steps`` times); a partial step
        holds the CFL, and a step that still fails or needs a < ``alpha_min``
        is rejected: state restored, CFL times ``cut_factor``, and the retry
        reuses the existing factorization (direct-solve phase: GMRES on the
        assembled matrix with the old factorization as preconditioner). A
        second failure in a row rebuilds the factorization and the PARDISO
        analysis, as ``merit`` does at once.
    alpha_min : float
        Smallest under-relaxation an ``unsteady`` step may be accepted with.
    cfl_floor_exponent : float or None
        SER floor: after an accepted step and when no step failed in the last
        5 iterations, CFL >= cfl_start (R_0 / R)^p with R the merit RMS and R_0
        that of the start (bounded by the CFL limit of the phase). None = off.
    source_cfl_limit : float or None
        Source-term time-step limit (Piotrowski & Zingg 2020): the local CFL
        of every cell obeys lambda_max(dS/dQ) dt <= this value (0.9 in the
        paper); needs a ``time_term`` that takes a ``source_limit`` keyword
        (the RANS problems). None = off.
    source_cfl_release : int
        The source-term limit is switched off after this many consecutive
        Newton-Krylov steps that were accepted as full steps, and armed again
        by any cut or rejected step.
    source_cfl_ramp : float
        Gradual release (> 1): after ``source_cfl_release`` clean steps the
        limit is multiplied by this factor on every further clean step and off
        above 1e8 times the base; a cut or rejected step divides it by the
        factor (at least back to the base). 1 = the hard release and re-arm.
    nk_gmres : {"scipy", "right"}
        GMRES of the Newton-Krylov phase: SciPy's (left-preconditioned inner
        iteration, restarts from scratch when the true residual misses the
        tolerance; 1.2 behaviour) or the right-preconditioned
        :func:`~pyaeroflow.solver.linear.gmres_right`, which minimizes the
        true linear residual the forcing term refers to (Phase B).
    nk_unconverged : {"reject", "accept"}
        What a Newton-Krylov step does whose GMRES stops at ``nk_max_krylov``
        products above the forcing tolerance. ``"reject"`` (default since
        1.6.1): the step is rejected as a failed linear solve (CFL times
        ``cut_factor``, fresh factorization), log note ``GMRES unconverged``.
        ``"accept"`` (1.6.0 and before): the update is tried as it is.
    nk_forcing_max : float
        Upper bound of the Eisenstat-Walker forcing term (``nk_forcing``).
    nk_forcing_safeguard : bool
        Safeguards of Eisenstat & Walker (1996) choice 2 and Kelley (1995):
        eta_k >= 0.9 eta_(k-1)^2 whenever that exceeds 0.1 (no sudden
        tightening after one good step), and eta_k >= 0.5 target / measure
        (no over-solving of the last steps), the latter above
        ``nk_linear_tolerance`` too.
    nk_approximate_drop : float or None
        Approximate (R1) Jacobian products until the residual measure has
        dropped by this many orders (Yildirim et al. 2019: 4-5 orders), exact
        products afterwards (latched): the Newton-Krylov products then come
        from ``linearize_approximate`` (frozen limiter). None = exact
        products throughout (1.2).
    nk_product_precision : {"float64", "float32"}
        Precision of the Newton-Krylov Jacobian products (``linearize``).
        ``float32`` evaluates the tangent in single precision, about half the
        time per product, with a float64 fallback for non-finite products
        (:meth:`~pyaeroflow.driver.FlowProblem.linearize_float32`); the
        residual, GMRES and the update stay float64.
    pc_lagging : {"fixed", "adaptive"}
        Preconditioner lagging in the Newton-Krylov phase. ``fixed`` (1.2):
        ``refactor_interval``, ``refactor_cfl_ratio`` and the 15-iteration
        rule. ``adaptive``: keep the factorization while the GMRES count of a
        step stays below ``pc_lag_growth`` times (and 3 above) the count of
        the first step after the factorization, for at most ``pc_lag_max``
        steps, while the CFL has not fallen below half its value at the
        factorization; the pseudo-time term of the operator is always the
        current one, so a lagged factorization only costs GMRES iterations,
        which this rule watches. A step with a lagged preconditioner that hits
        ``nk_cfl_adapt_iterations`` refactorizes instead of cutting the CFL
        cap.
        The factorization is also rebuilt when the CFL has grown more than
        ``refactor_cfl_ratio`` times since it was made (as with ``fixed``):
        NACA 0012 alpha 15, CFL 967 -> 4273 on a lagged factorization, GMRES
        hit the product cap.
    pc_lag_max : int
        Largest number of Newton steps per factorization (``adaptive``).
    pc_lag_growth : float
        GMRES growth factor that triggers a new factorization (``adaptive``).
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
    nk_linearize: bool = True
    line_search_steps: int = 2
    merit_columns: tuple = None
    nk_variable_scaling: bool = False
    cfl_control: str = "growth"
    ser_exponent: float = 1.0
    ser_growth_min: float = 1.2
    ser_ratio_min: float = 0.3
    ser_ratio_max: float = 2.0
    ser_gmres_hold: int = 25
    ser_gmres_reduce: int = 60
    step_control: str = "merit"
    alpha_min: float = 0.1
    cfl_floor_exponent: float = None
    source_cfl_limit: float = None
    source_cfl_release: int = 5
    source_cfl_ramp: float = 1.0
    nk_gmres: str = "right"
    nk_unconverged: str = "reject"
    nk_forcing_max: float = 0.5
    nk_forcing_safeguard: bool = False
    nk_approximate_drop: float = None
    nk_product_precision: str = "float64"
    pc_lagging: str = "adaptive"
    pc_lag_max: int = 10
    pc_lag_growth: float = 1.5
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
        (GMRES iterations of the step, 0 for a direct solve),
        ``step_factor`` (the fraction of the Newton step the line search kept:
        1, 1/2, 1/4 ...; 1 for the start entry and for a failed linear solve).
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
    linearize=None,
    reference_norm=None,
    limit_step=None,
    resume=None,
    debug=None,
    linearize_approximate=None,
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
    linearize : callable, optional
        ``linearize(u) -> apply`` with ``apply(du) -> dR/du(u) @ du``: the
        residual linearized once per Newton step (``settings.nk_linearize``).
        Either ``jvp`` or ``linearize`` enables Newton-Krylov.
    linearize_approximate : callable, optional
        Like ``linearize``, the approximate (R1) linearization used for the
        Newton-Krylov products until ``settings.nk_approximate_drop`` orders
        (e.g. with the limiter frozen at the state).
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
    debug : callable, optional
        ``debug(record)`` after every iteration with a dict of the numerics:
        ``it``, ``cfl``, ``nk``, ``gmres``, ``factorized``, ``tolerance`` and
        ``linear_residual`` (Newton-Krylov: the true relative residual of the
        linear solve, one extra Jacobian product), ``re_evaluated`` (RMS
        before and after a change of the operator), ``reference`` (residual
        array the step is judged against), ``reference_rms``, ``du`` (update
        after ``limit_step``), ``limited`` (cells per variable changed by
        ``limit_step``), ``trials`` ((factor, RMS, RMS per variable) of every
        line-search evaluation), ``new`` (residual of the last trial),
        ``accepted``, ``step``, ``error`` (failed linear solve).

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

    cols = None if s.merit_columns is None else list(s.merit_columns)

    def merit(r):
        return rms(r if cols is None else np.asarray(r)[:, cols])

    norm_abs = merit(r)
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
        "step_factor": [1.0],
    }
    converged = done(norm, norm_abs)
    # CFL cap of the Newton-Krylov phase, adapted to the GMRES effort: the
    # first-order preconditioner degrades as the pseudo-time term vanishes,
    # so expensive linear solves lower the cap and cheap ones raise it.
    cfl_cap = float(resume.get("cfl_cap", s.cfl_max))
    nk_on = bool(resume.get("nk_on", False))  # the switch latches
    unsteady = s.step_control == "unsteady"
    # Merit RMS of the start: the reference of the SER floor.
    r_start = float(resume.get("residual_start", norm_abs))
    # Source-term time-step limit: armed at the start and after every cut or
    # rejected step, released after source_cfl_release clean Newton steps.
    source_active = bool(resume.get("source_active", s.source_cfl_limit is not None))
    source_level = float(resume.get("source_level", 1.0))  # gradual release
    source_off = 1.0e8
    ramp = s.source_cfl_ramp > 1.0
    clean_nk = 0
    fails_in_row, last_fail = 0, -(10**9)
    retry_pc = False  # retry a failed step with the existing factorization
    adaptive_pc = s.pc_lagging == "adaptive"
    pc_base_its = None  # GMRES count of the first NK step after a factorization
    # Approximate (R1) products until nk_approximate_drop orders, then exact.
    approx_on = bool(
        resume.get(
            "approximate_products",
            s.nk_approximate_drop is not None and linearize_approximate is not None,
        )
    )
    approx_threshold = (
        10.0 ** (-s.nk_approximate_drop) if s.nk_approximate_drop is not None else 0.0
    )
    eta_prev = None
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
            "residual_start": r_start,
            "source_active": source_active,
            "source_level": source_level,
            "approximate_products": approx_on,
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
    use_linearize = linearize is not None and s.nk_linearize
    has_jvp = jvp is not None or use_linearize
    lin, steps_since_factor, cfl_at_factor = None, 0, cfl
    lin_nk = False  # the factorization was built in the Newton-Krylov phase
    stale = False  # the preconditioner let GMRES work hard: rebuild it
    max_iterations = s.max_iterations
    it = it0
    while it < max_iterations:
        it += 1
        state["it"] = it - 1
        if converged or cfl < s.cfl_min:
            break
        info = getattr(callback, "info", None)
        dbg = {"it": it, "cfl": cfl} if debug is not None else None
        if info is not None and info.pop("residual_changed", False):
            # The callback changed the residual operator (limiter frozen or
            # released): the reference for acceptance and convergence is the
            # residual of the current state under the new operator, not the
            # one the previous step produced.
            before = norm_abs
            r = res(u, it)
            norm_abs = merit(r)
            if dbg is not None:
                dbg["re_evaluated"] = (before, norm_abs)
            scale = np.maximum(scale, rms_per_variable(r))
            norm = measure(r)
        startup = residual_startup is not None and it <= s.first_order_iterations
        if (
            s.first_order_iterations
            and it == s.first_order_iterations + 1
            and residual_startup is not None
        ):
            r = res(u, it)  # switch to the full residual
            norm_abs = merit(r)
            scale = np.maximum(scale, rms_per_variable(r))
            norm = measure(r)
        by_iteration = s.nk_switch_iteration is not None and it > s.nk_switch_iteration
        immediate = s.nk_switch_residual_drop <= 0.0  # dual-time inner iterations
        nk_on = nk_on or immediate or norm <= nk_threshold or by_iteration
        use_nk = s.newton_krylov and has_jvp and not startup and nk_on
        with timers.section("time term"):
            if s.source_cfl_limit is not None:
                t_matrix = time_term(
                    u,
                    cfl,
                    source_limit=s.source_cfl_limit * source_level
                    if source_active
                    else None,
                )
            else:
                t_matrix = time_term(u, cfl)
        # The approximate-Jacobian phase of a Newton-Krylov run keeps the
        # merit control: there the pseudo-time residual accepts steps that
        # raise the steady residual at low CFL (small AFT/BCM plates). Set before
        # the try block: the rejection branch reads it after an early failure.
        unsteady_now = unsteady and (use_nk or not (s.newton_krylov and has_jvp))
        accepted, lin_its, step, linear_failed = False, 0, 1.0, False
        try:
            if adaptive_pc and use_nk:
                reuse = lin is not None and (
                    retry_pc
                    or (
                        lin_nk
                        and steps_since_factor < s.pc_lag_max
                        and 0.5 * cfl_at_factor
                        <= cfl
                        <= cfl_at_factor * s.refactor_cfl_ratio
                        and not stale
                    )
                )
            else:
                reuse = lin is not None and (
                    retry_pc
                    or (
                        steps_since_factor < s.refactor_interval
                        and cfl <= cfl_at_factor * s.refactor_cfl_ratio
                        and not stale
                    )
                )
            if not reuse:
                with timers.section("assemble matrix"):
                    matrix = jacobian(u) + t_matrix
                with timers.section("factorize"):
                    lin = factorize(matrix, s.linear)
                del matrix
                steps_since_factor, cfl_at_factor = 0, cfl
                lin_nk, pc_base_its = use_nk, None
            steps_since_factor += 1
            if use_nk:
                if approx_on and norm <= approx_threshold:
                    approx_on = False
                    _note(callback, "exact Jacobian products")
                if approx_on:
                    with timers.section("linearize"):
                        product = linearize_approximate(u)
                elif use_linearize:
                    with timers.section("linearize"):
                        product = linearize(u)
                else:
                    product = partial(jvp, u)
                tolerance = eta if s.nk_forcing else s.nk_linear_tolerance
                operator = _newton_operator(t_matrix, product, u.shape)
                precondition = _timed("preconditioner", lin.precondition)
                if s.nk_variable_scaling:
                    operator, precondition, unscale = _variable_scaling(
                        operator, precondition, u
                    )
                with timers.section("nk solve (GMRES)"):
                    du, lin_its, lin_converged = solve_matrix_free(
                        operator,
                        -r.ravel(),
                        precondition,
                        tolerance=tolerance,
                        restart=s.nk_restart,
                        max_iterations=s.nk_max_krylov,
                        method=s.nk_gmres,
                        full_output=True,
                    )
                if not lin_converged and s.nk_unconverged == "reject":
                    # A solve stopped at the product cap above the forcing
                    # tolerance is no Newton step: reject it as a linear
                    # failure (CFL cut, fresh factorization) instead of
                    # trying an update of unknown quality.
                    _note(callback, "GMRES unconverged")
                    raise LinearSolverError(
                        f"GMRES missed the tolerance {tolerance:.2g} "
                        f"in {lin_its} products"
                    )
                if s.nk_variable_scaling:
                    du = unscale(du)
                if dbg is not None:
                    # True linear residual |(T + J) du + R| / |R| (one more
                    # Jacobian product): GMRES only knows its own estimate.
                    dbg["tolerance"] = tolerance
                    a_du = _newton_operator(t_matrix, product, u.shape)(du)
                    dbg["linear_residual"] = float(
                        np.linalg.norm(a_du + r.ravel())
                        / max(np.linalg.norm(r.ravel()), tiny)
                    )
                    dbg["linear_misfit"] = (a_du + r.ravel()).reshape(u.shape)
                product = None  # frees the stored linearization
            elif retry_pc and reuse:
                # Retry after a failed step at a lower CFL: the factorization
                # belongs to the old pseudo-time term, so it preconditions a
                # GMRES solve of the new matrix instead of being rebuilt (the
                # new matrix only has a larger diagonal).
                with timers.section("assemble matrix"):
                    matrix = jacobian(u) + t_matrix
                with timers.section("direct solve"):
                    du, lin_its = solve_matrix_free(
                        matrix.dot,
                        -r.ravel(),
                        _timed("preconditioner", lin.precondition),
                        tolerance=s.nk_linear_tolerance,
                        restart=s.nk_restart,
                        max_iterations=s.nk_max_krylov,
                        method=s.nk_gmres,
                    )
                del matrix
            else:
                with timers.section("direct solve"):
                    du = lin.solve(-r.ravel())
                lin_its = lin.iterations
            du = du.reshape(u.shape)
            if dbg is not None:
                dbg.update(nk=use_nk, gmres=lin_its, factorized=not reuse)
                dbg["reference"] = r
                du_raw = du
            if limit_step is not None:
                tau = (t_matrix.diagonal() * cfl).reshape(u.shape)
                du = limit_step(u, du, r, tau)
            if dbg is not None:
                # changed beyond round-off: a clip returns (u + du) - u
                changed = np.abs(du - du_raw) > 1e-10 * (np.abs(u) + np.abs(du_raw))
                dbg["limited"] = np.count_nonzero(changed, axis=0)
                dbg["du_raw"] = du_raw
                dbg["du"] = du
                dbg["trials"] = []
            u_new = u + du
            r_new = res(u_new, it + 1)
            abs_new = merit(r_new)
            if dbg is not None:
                dbg["trials"].append((1.0, abs_new, rms_per_variable(r_new)))
                dbg["full_step"] = r_new
            if unsteady_now:
                r_new, abs_new, step, accepted = _unsteady_line_search(
                    lambda a, u=u, du=du, it=it: res(u + a * du, it + 1),
                    merit,
                    (t_matrix @ du.ravel()).reshape(u.shape),
                    norm_abs,
                    r_new,
                    s.line_search_steps,
                    s.alpha_min,
                    dbg,
                )
                u_new = u + step * du
            else:
                # Backtracking line search (D54): halve a step that raises the
                # residual norm before giving up on it. The fraction kept is
                # logged (``step 1/2``, ``step 1/4``): an iteration that needs
                # it every time makes little progress however the forces look.
                for _ in range(s.line_search_steps):
                    if np.isfinite(abs_new) and abs_new <= norm_abs:
                        break
                    step *= 0.5
                    u_new = u + step * du
                    r_new = res(u_new, it + 1)
                    abs_new = merit(r_new)
                    if dbg is not None:
                        dbg["trials"].append((step, abs_new, rms_per_variable(r_new)))
                accepted = (
                    np.isfinite(abs_new) and abs_new <= s.reject_factor * norm_abs
                )
            if dbg is not None:
                dbg["new"] = r_new
        except LinearSolverError as error:
            linear_failed = True
            if dbg is not None:
                dbg["error"] = str(error)
        if dbg is not None:
            dbg.update(accepted=accepted, step=step, reference_rms=norm_abs)
            debug(dbg)
        hist["iteration"].append(it)
        hist["cfl"].append(cfl)
        hist["accepted"].append(accepted)
        hist["newton_krylov"].append(use_nk)
        hist["linear_iterations"].append(lin_its)
        hist["step_factor"].append(step)
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
            if unsteady_now:
                # HANIM: success never lowers the CFL (failures do, by the
                # retry); an under-relaxed step holds it. Lowering it by the
                # steady-residual ratio of a step that was accepted on the
                # pseudo-time residual lets the CFL decay while the steady
                # residual creeps up (flat-plate AFT stalled at CFL 2).
                ratio = min(ratio, 1.0) if step < 1.0 else max(ratio, 1.0)
            if use_nk:
                lagged = adaptive_pc and steps_since_factor > 1
                if not ser:
                    if lin_its >= s.nk_cfl_adapt_iterations and not lagged:
                        cfl_cap = max(cfl * 0.25, s.cfl_start)
                    elif lin_its >= s.nk_cfl_adapt_iterations:
                        # stale (lagged) preconditioner: refactorize next
                        # step but do not grow the CFL on a failed solve
                        ratio = min(ratio, 1.0)
                    elif lin_its <= s.nk_cfl_adapt_iterations // 4:
                        cfl_cap = min(cfl_cap * 4.0, s.cfl_max)
                if adaptive_pc:
                    if pc_base_its is None:
                        pc_base_its = max(lin_its, 1)
                    stale = (
                        lin_its > max(s.pc_lag_growth * pc_base_its, pc_base_its + 3)
                        or lin_its >= s.nk_cfl_adapt_iterations
                    )
                else:
                    stale = lin_its > s.nk_cfl_adapt_iterations // 4
            # Cap the CFL while the Newton-Krylov switch is pending: a huge
            # pseudo-time step with the approximate Jacobian does not converge.
            waiting = s.newton_krylov and has_jvp and not use_nk
            limit = min(cfl_cap, s.cfl_max_approximate) if waiting else cfl_cap
            cfl = min(limit, cfl * ratio)
            if s.cfl_floor_exponent is not None and it - last_fail > 5:
                floor = s.cfl_start * (r_start / max(abs_new, tiny)) ** (
                    s.cfl_floor_exponent
                )
                cfl = max(cfl, min(floor, limit))
            if use_nk:
                eta_new = 0.9 * (abs_new / norm_abs) ** 2
                if s.nk_forcing_safeguard:
                    if eta_prev is not None and 0.9 * eta_prev**2 > 0.1:
                        eta_new = max(eta_new, 0.9 * eta_prev**2)
                    new_measure = measure(r_new)
                    eta_new = max(eta_new, 0.5 * target / max(new_measure, tiny))
                eta = min(s.nk_forcing_max, max(s.nk_linear_tolerance, eta_new))
                eta_prev = eta
            u, r, norm_abs = u_new, r_new, abs_new
            # Peaks of the per-variable residuals set the scales (D54).
            scale = np.maximum(scale, rms_per_variable(r))
            norm = measure(r)
            converged = done(norm, norm_abs)
            fails_in_row, retry_pc = 0, False
        else:
            cfl *= s.cut_factor
            fails_in_row += 1
            last_fail = it
            if (
                unsteady_now
                and fails_in_row == 1
                and lin is not None
                and not linear_failed
            ):
                retry_pc = True  # first retry: keep the factorization
            else:
                retry_pc = False
                lin = None  # rebuild the preconditioner after a rejection
                reset_analysis()  # and the PARDISO analysis
        if s.source_cfl_limit is not None:
            was_active = source_active
            if accepted and use_nk and step >= 1.0:
                clean_nk += 1
                if clean_nk >= s.source_cfl_release:
                    if ramp and source_active:
                        source_level *= s.source_cfl_ramp
                        if source_level > source_off:
                            source_active = False
                    else:
                        source_active = False
            elif not accepted or step < 1.0:
                clean_nk = 0
                if ramp and source_level > 1.0:
                    # step back one level (from "off" to the last level before)
                    source_level = max(
                        1.0,
                        (source_off if not source_active else source_level)
                        / s.source_cfl_ramp,
                    )
                else:
                    source_level = 1.0
                source_active = True
            info = getattr(callback, "info", None)
            if info is not None and was_active != source_active:
                note = "source dt limit " + ("armed" if source_active else "released")
                info["note"] = f"{info['note']}, {note}" if info.get("note") else note
        hist["residual"].append(norm_abs)
        hist["measure"].append(norm)
        hist["residuals"].append(rms_per_variable(r))
        if callback is not None:
            info = getattr(callback, "info", None)
            if info is not None:
                info["newton_krylov"] = use_nk
                info["linear_iterations"] = lin_its
                info["measure"] = norm
                info["step_factor"] = step
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


def _note(callback, text):
    """Append ``text`` to the log note of the current iteration."""
    info = getattr(callback, "info", None)
    if info is not None:
        info["note"] = f"{info['note']}, {text}" if info.get("note") else text


def _unsteady_line_search(
    residual_at, merit, t_du, norm0, r_full, steps, alpha_min, dbg=None
):
    """Quadratic under-relaxation on the pseudo-time residual (HANIM).

    R_t(a) = a T du + R(u + a du); f(a) = |R_t(a)| on the merit columns,
    f(0) = ``norm0``. A full step is accepted if f(1) < f(0). Otherwise
    g = f^2 is modelled as g(0) - 2 g(0) a + c a^2 (slope of the exact Newton
    direction) through the last trial, and the next trial is its minimum
    a = g(0) / c, bounded to [0.1, 0.5] times the last trial.

    Returns ``(r_new, merit of r_new, a, accepted)``.
    """
    g0 = norm0 * norm0
    alpha, r_new = 1.0, r_full
    f_t = merit(r_new + t_du)
    for _ in range(steps):
        if np.isfinite(f_t) and f_t < norm0:
            break
        with np.errstate(over="ignore", invalid="ignore"):
            g_a = f_t * f_t if np.isfinite(f_t) else np.inf
            c = (g_a - g0 + 2.0 * g0 * alpha) / (alpha * alpha)
        guess = g0 / c if np.isfinite(c) and c > 0.0 else 0.0
        alpha = min(max(guess, 0.1 * alpha), 0.5 * alpha)
        r_new = residual_at(alpha)
        f_t = merit(r_new + alpha * t_du)
        if dbg is not None:
            dbg["trials"].append((alpha, merit(r_new), rms_per_variable(r_new)))
    abs_new = merit(r_new)
    accepted = bool(
        np.isfinite(f_t) and np.isfinite(abs_new) and f_t < norm0 and alpha >= alpha_min
    )
    if dbg is not None:
        dbg["unsteady_residual"] = f_t
    return r_new, abs_new, alpha, accepted


def _variable_scaling(operator, precondition, u):
    """Newton-Krylov system in the unknowns du_j / s_j (``nk_variable_scaling``).

    s_j = 1 for the first three variables (p, u, v, non-dimensional) and the
    RMS of the field for the others. Returns the scaled operator v -> A (S v),
    the preconditioner r -> S^-1 M^-1 r and the map back y -> S y.
    """
    scale = np.ones(u.shape[1])
    if u.shape[1] > 3:
        rms_tail = rms_per_variable(u[:, 3:])
        scale[3:] = np.where(rms_tail > 0.0, rms_tail, 1.0)
    full = np.tile(scale, u.shape[0])

    def apply(v):
        return operator(full * v)

    def pre(r):
        return precondition(r) / full

    def unscale(y):
        return full * y

    return apply, pre, unscale


def _newton_operator(t_matrix, product, shape):
    """Return v -> (T + J) v for the matrix-free Newton-Krylov solve.

    ``product(du)`` is ``J @ du`` with ``du`` of the state ``shape``.
    """

    def apply(v):
        with timers.section("Jacobian product"):
            return t_matrix @ v + np.asarray(product(v.reshape(shape))).ravel()

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
        "step_factor": [1.0],
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
        hist["step_factor"].append(1.0)
        if callback is not None:
            callback(it, u, hist["residuals"][-1], cfl, True)
            if getattr(callback, "info", {}).get("stop"):
                break  # the callback asked for a clean stop (control file)
    history = {k: np.asarray(v) for k, v in hist.items()}
    return SteadyResult(u, bool(converged), history)

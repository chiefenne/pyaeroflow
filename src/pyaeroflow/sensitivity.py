"""Sensitivity of a converged steady solution to the angle of attack.

At a converged state ``R(w, alpha) = 0`` the implicit function theorem gives
the tangent ``dw/dalpha = -(dR/dw)^-1 dR/dalpha``. It costs one linear solve
with the second-order Jacobian, which the Newton-Krylov solver already has
as a matrix-free product (``jax.linearize``) with the factorized first-order
Jacobian as preconditioner. Two uses:

* exact slopes ``dCL/dalpha``, ``dCD/dalpha``, ``dCM/dalpha`` (per degree);
* a first-order predictor ``w(alpha + d) ~ w + d dw/dalpha`` for the start of
  the next angle of a polar (:func:`write_predicted_restart`).

``dR/dalpha`` is a central difference of the residual at fixed ``w`` over the
farfield state of ``alpha -+ step``: the farfield data are the only place
where the angle enters the residual, and the residual is smooth in it.
"""

import time
from typing import NamedTuple

import numpy as np

from pyaeroflow.solver.linear import factorize, solve_matrix_free
from pyaeroflow.solver.pseudo_time import _variable_scaling


class AlphaTangent(NamedTuple):
    """Result of :func:`alpha_tangent`.

    Attributes
    ----------
    du : ndarray
        ``dw/dalpha`` per degree in the non-dimensional state units of the
        run (same shape as the state).
    dcl, dcd, dcm : float
        Total derivatives of the force coefficients per degree.
    products : int
        Jacobian-vector products of the linear solve.
    converged : bool
        The linear solve met its tolerance.
    seconds : float
    """

    du: np.ndarray
    dcl: float
    dcd: float
    dcm: float
    products: int
    converged: bool
    seconds: float


def _residual_at_alpha(problem, u, bc, psi, gamma):
    """Second-order residual of ``u`` with the boundary data ``bc``."""
    saved = problem.bc
    problem.bc = bc
    try:
        return problem.residual(u, problem.scheme_parameters(u), psi=psi, gamma=gamma)
    finally:
        problem.bc = saved


def alpha_tangent(
    result,
    state,
    config_at,
    step=0.05,
    tolerance=1.0e-6,
    max_products=400,
    restart=50,
):
    """Tangent ``dw/dalpha`` and the coefficient slopes of a converged run.

    Parameters
    ----------
    result : RunResult
        A steady run (``result.problem`` and ``result.scaling`` set).
    state : dict
        Solver state of its restart file (``load_restart_state``): the frozen
        limiter ``psi`` and intermittency ``gamma`` of the converged state
        and the final CFL (sets the diagonal of the preconditioner).
    config_at : callable
        ``config_at(alpha_deg) -> Resolved``: the case at another angle (the
        boundary data are taken from it; nothing is run).
    step : float
        Half step in degrees of the difference of ``dR/dalpha``.
    tolerance : float
        Relative tolerance of the linear solve.

    Raises
    ------
    ValueError
        Unsupported case (unsteady, farfield vortex correction active).
    """
    from pyaeroflow.run import Monitor, setup, state_scale

    t0 = time.perf_counter()
    problem, scaling = result.problem, result.scaling
    if problem is None or result.config.data["time"]["mode"] == "unsteady":
        raise ValueError("the angle tangent needs a steady run")
    if state is None or abs(float(state.get("vortex_gamma", 0.0))) > 0.0:
        raise ValueError(
            "the angle tangent needs the solver state and no farfield vortex"
        )
    u = np.asarray(result.w) / state_scale(problem, scaling)
    alpha = float(result.config.data["reference"]["angle_of_attack"])
    psi, gamma = state.get("psi"), state.get("gamma")

    bcs = []
    for a in (alpha - step, alpha + step):
        p, _, _, _ = setup(config_at(a), result.mesh)
        bcs.append(p.bc)
        del p
    dr = (
        _residual_at_alpha(problem, u, bcs[1], psi, gamma)
        - _residual_at_alpha(problem, u, bcs[0], psi, gamma)
    ) / (2.0 * step)

    params = problem.scheme_parameters(u)
    product = problem.linearize(u, params, psi=psi, gamma=gamma)
    shape = u.shape
    cfl = max(float(state.get("cfl", 1.0e3)), 1.0e3)
    matrix = problem.jacobian(u, params) + problem.time_term(u, params, cfl)
    lin = factorize(matrix, _linear_settings(result.config))
    del matrix

    def operator(v):
        return np.asarray(product(v.reshape(shape))).ravel()

    operator, precondition, unscale = _variable_scaling(operator, lin.precondition, u)
    y, products, converged = solve_matrix_free(
        operator,
        -dr.ravel(),
        precondition,
        tolerance=tolerance,
        restart=restart,
        max_iterations=max_products,
        method="right",
        full_output=True,
    )
    du = unscale(y).reshape(shape)

    monitor = Monitor(result.config, problem, scaling)

    def coefficients(eps):
        monitor.alpha = alpha + eps
        c = monitor.final(u + eps * du)
        return np.array([c.cl, c.cd, c.cm])

    h = step
    slope = (coefficients(h) - coefficients(-h)) / (2.0 * h)
    return AlphaTangent(
        du, *map(float, slope), int(products), bool(converged), time.perf_counter() - t0
    )


def start_residuals(result, state, config_at, du, alpha_next):
    """RMS residual at ``alpha_next`` of the plain and the predicted start.

    Both states are evaluated with the farfield data of ``alpha_next`` and
    the converged frozen limiter: the final state of the angle that was run
    (the plain restart chain) and ``w + (alpha_next - alpha) dw/dalpha``.
    The predictor is only worth using if the second is lower.
    """
    from pyaeroflow.run import setup, state_scale

    problem, scaling = result.problem, result.scaling
    u = np.asarray(result.w) / state_scale(problem, scaling)
    alpha = float(result.config.data["reference"]["angle_of_attack"])
    bc = setup(config_at(alpha_next), result.mesh)[0].bc
    psi, gamma = state.get("psi"), state.get("gamma")
    out = []
    for w in (u, u + (alpha_next - alpha) * du):
        r = _residual_at_alpha(problem, w, bc, psi, gamma)
        out.append(float(np.sqrt(np.mean(r**2))))
    return tuple(out)


def _linear_settings(config):
    from pyaeroflow.run import _pseudo_time_settings

    return _pseudo_time_settings(config.data).linear


def write_predicted_restart(source, target, du, delta):
    """Copy restart file ``source`` to ``target`` with ``w += delta * du``.

    All other entries (mesh fingerprint, solver state) are kept, so the file
    is a restart file of the same mesh and flow family.
    """
    import os

    with np.load(source) as data:
        content = {k: np.array(data[k]) for k in data.files}
    content["w"] = content["w"] + delta * du
    tmp = type(target)(str(target) + ".tmp")
    with open(tmp, "wb") as file:
        np.savez(file, **content)
    os.replace(tmp, target)
    return target

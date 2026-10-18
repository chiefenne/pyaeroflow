"""V3: decaying 2D Taylor-Green vortex (unsteady verification).

Doubly periodic box [0, 2 pi]^2 with the exact Navier-Stokes solution

    u = -cos(x) sin(y) exp(-2 nu t),  v = sin(x) cos(y) exp(-2 nu t),
    p = -(cos(2x) + cos(2y)) / 4 * exp(-4 nu t).

The convective terms are balanced by the pressure, so the velocity error
tests the convective and viscous discretisation together with the time
scheme. Without a pressure boundary the pressure is pinned in one cell;
errors are measured after removing the mean (D39 R2).
"""

import dataclasses
from typing import NamedTuple

import numpy as np

from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.solver import DualTimeSettings

SIZE = 2.0 * np.pi


def exact(x, y, t, viscosity):
    """Exact (p, u, v) at the points (x, y) and time t, shape (n, 3)."""
    decay = np.exp(-2.0 * viscosity * t)
    w = np.empty((np.size(x), 3))
    w[:, inc.P] = -0.25 * (np.cos(2 * x) + np.cos(2 * y)) * decay**2
    w[:, inc.U] = -np.cos(x) * np.sin(y) * decay
    w[:, inc.V] = np.sin(x) * np.cos(y) * decay
    return w


class TaylorGreenResult(NamedTuple):
    """V3 result."""

    problem: object
    w: np.ndarray
    converged: bool
    history: dict
    time: float
    exact: np.ndarray
    velocity_error: float
    pressure_error: float
    kinetic_energy: float
    kinetic_energy_exact: float


def problem(n, viscosity, cell_type="quad", limiter="none"):
    """Periodic Taylor-Green problem on an n x n mesh."""
    mesh = rectangle(n, n, x=(0.0, SIZE), y=(0.0, SIZE), cell_type=cell_type)
    conditions = {
        "left": {"type": "periodic", "partner": "right", "translation": (SIZE, 0.0)},
        "bottom": {"type": "periodic", "partner": "top", "translation": (0.0, SIZE)},
    }
    model = inc.IncompressibleModel(density=1.0, viscosity=viscosity, limiter=limiter)
    # Pin the exact initial pressure in the cell nearest the centre, so that
    # the pressure level is comparable with the exact solution.
    centre = np.array([0.5 * SIZE, 0.5 * SIZE])
    cell = int(np.argmin(np.sum((mesh.cell_centroid - centre) ** 2, axis=1)))
    x, y = mesh.cell_centroid[cell]
    p0 = float(exact(x, y, 0.0, viscosity)[0, inc.P])
    return IncompressibleProblem(
        mesh, model, conditions, pressure_point=centre, pin_pressure=p0
    )


def inner_settings(residual_drop=4.0):
    """Inner-iteration settings: the dual-time defaults with the given drop.

    Temporal-order studies need a tight inner convergence (6 orders), since
    the inner iteration error must stay below the temporal error of the
    smallest time step.
    """
    return dataclasses.replace(
        DualTimeSettings._field_defaults["inner"], residual_drop=residual_drop
    )


def solve(
    n=32,
    viscosity=0.05,
    time_step=0.05,
    end_time=1.0,
    scheme="bdf2",
    cell_type="quad",
    limiter="none",
    inner=None,
):
    """Run the vortex to ``end_time`` and compare with the exact solution.

    Returns
    -------
    TaylorGreenResult
        ``velocity_error``: RMS of |v - v_exact| over the cells divided by
        the exact RMS speed; ``pressure_error``: RMS of the mean-free
        pressure error relative to the exact RMS pressure.
    """
    prob = problem(n, viscosity, cell_type, limiter)
    xc, yc = prob.mesh.cell_centroid.T
    w0 = exact(xc, yc, 0.0, viscosity)
    settings = DualTimeSettings(
        time_step,
        end_time,
        scheme,
        inner or inner_settings(),
    )
    result = prob.solve_unsteady(w0, settings)
    w_ex = exact(xc, yc, result.time, viscosity)
    area = prob.mesh.cell_area
    w = result.u
    dv = w[:, 1:] - w_ex[:, 1:]
    v_err = np.sqrt(
        np.sum(area[:, None] * dv**2) / np.sum(area[:, None] * w_ex[:, 1:] ** 2)
    )
    dp = w[:, inc.P] - w_ex[:, inc.P]
    dp = dp - np.sum(area * dp) / np.sum(area)
    p_ex = w_ex[:, inc.P] - np.sum(area * w_ex[:, inc.P]) / np.sum(area)
    p_err = np.sqrt(np.sum(area * dp**2) / np.sum(area * p_ex**2))
    ke = 0.5 * np.sum(area[:, None] * w[:, 1:] ** 2)
    ke_ex = 0.5 * np.sum(area[:, None] * w_ex[:, 1:] ** 2)
    return TaylorGreenResult(
        prob,
        w,
        result.converged,
        result.history,
        result.time,
        w_ex,
        float(v_err),
        float(p_err),
        float(ke),
        float(ke_ex),
    )

"""Verification cases V1 (plane Couette) and V2 (plane Poiseuille).

Channel of height H between y = 0 and y = H, periodic in x.

- Couette: the top wall moves with velocity U_w, no pressure gradient;
  u = U_w y / H, v = 0, p = const.
- Poiseuille: both walls at rest, driven by the body force f_x = G
  (mean pressure gradient -G); u = G / (2 mu) y (H - y).
"""

from typing import NamedTuple

import numpy as np

from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.solver import PseudoTimeSettings


class ChannelResult(NamedTuple):
    """Solution and error of a channel case.

    Attributes
    ----------
    problem : IncompressibleProblem
    w : ndarray, shape (n_cells, 3)
    exact : ndarray, shape (n_cells, 3)
        Exact (p, u, v) at the cell centroids (p up to the pinned level).
    error : ndarray, shape (3,)
        Area-weighted RMS error per variable.
    max_error : ndarray, shape (3,)
        Maximum absolute error per variable.
    h : float
        Mesh size sqrt(area / n_cells).
    converged : bool
    history : dict
    """

    problem: object
    w: np.ndarray
    exact: np.ndarray
    error: np.ndarray
    max_error: np.ndarray
    h: float
    converged: bool
    history: dict


def _solve(mesh, model, conditions, exact, settings):
    problem = IncompressibleProblem(mesh, model, conditions)
    result = problem.solve(
        settings=settings or PseudoTimeSettings(residual_drop=10.0, max_iterations=300)
    )
    if problem.pinned_cell is not None:  # align the pressure level
        exact[:, inc.P] += (
            result.u[problem.pinned_cell, inc.P] - exact[problem.pinned_cell, inc.P]
        )
    diff = result.u - exact
    area = mesh.cell_area
    error = np.sqrt(np.sum(area[:, None] * diff**2, axis=0) / area.sum())
    return ChannelResult(
        problem,
        result.u,
        exact,
        error,
        np.abs(diff).max(axis=0),
        float(np.sqrt(area.sum() / mesh.n_cells)),
        result.converged,
        result.history,
    )


def couette(
    n,
    cell_type="quad",
    height=1.0,
    length=2.0,
    wall_velocity=1.0,
    viscosity=0.1,
    perturb=0.0,
    cluster_y="none",
    settings=None,
    seed=3,
):
    """Plane Couette flow on an n x n mesh (V1)."""
    mesh = rectangle(
        n,
        n,
        x=(0.0, length),
        y=(0.0, height),
        cell_type=cell_type,
        perturb=perturb,
        cluster_y=cluster_y,
        seed=seed,
        diagonal="random" if perturb else "alternating",
    )
    conditions = {
        "bottom": {"type": "wall"},
        "top": {"type": "wall", "velocity": (wall_velocity, 0.0)},
        "left": {"type": "periodic", "partner": "right", "translation": (length, 0.0)},
    }
    model = inc.IncompressibleModel(density=1.0, viscosity=viscosity)
    exact = np.zeros((mesh.n_cells, 3))
    exact[:, inc.U] = wall_velocity * mesh.cell_centroid[:, 1] / height
    return _solve(mesh, model, conditions, exact, settings)


def poiseuille(
    n,
    cell_type="quad",
    height=1.0,
    length=1.0,
    force=1.0,
    viscosity=0.1,
    perturb=0.0,
    settings=None,
    seed=3,
):
    """Plane Poiseuille flow driven by the body force ``force`` (V2)."""
    mesh = rectangle(
        n,
        n,
        x=(0.0, length),
        y=(0.0, height),
        cell_type=cell_type,
        perturb=perturb,
        seed=seed,
        diagonal="random" if perturb else "alternating",
    )
    conditions = {
        "bottom": {"type": "wall"},
        "top": {"type": "wall"},
        "left": {"type": "periodic", "partner": "right", "translation": (length, 0.0)},
    }
    model = inc.IncompressibleModel(
        density=1.0, viscosity=viscosity, body_force=(force, 0.0)
    )
    y = mesh.cell_centroid[:, 1]
    exact = np.zeros((mesh.n_cells, 3))
    exact[:, inc.U] = force / (2.0 * viscosity) * y * (height - y)
    return _solve(mesh, model, conditions, exact, settings)


def observed_orders(results, variable=inc.U):
    """Observed order of accuracy between successive refinement levels."""
    e = np.array([r.error[variable] for r in results])
    h = np.array([r.h for r in results])
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])

"""Verification case: convection-diffusion with an exact exponential solution.

Unit square, Dirichlet values from the exact solution on all boundaries,
solved by implicit pseudo-time stepping from u = 0. Used by the stage-1 tests
and by ``examples/stage1_convection_diffusion.py``.
"""

from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from pyaeroflow.mesh import rectangle
from pyaeroflow.numerics import build_geometry
from pyaeroflow.physics import convection_diffusion as cd
from pyaeroflow.solver import PseudoTimeSettings, first_order_jacobian, solve_steady


class CaseResult(NamedTuple):
    """Solution and error of one run.

    Attributes
    ----------
    mesh : Mesh
    u : ndarray, shape (n_cells, 1)
        Converged cell values.
    exact : ndarray, shape (n_cells, 1)
        Exact solution at the cell centroids.
    error : float
        Area-weighted RMS error at the cell centroids.
    h : float
        Mesh size measure sqrt(domain area / n_cells).
    converged : bool
    history : dict
        Pseudo-time history (see :class:`~pyaeroflow.solver.SteadyResult`).
    """

    mesh: object
    u: np.ndarray
    exact: np.ndarray
    error: float
    h: float
    converged: bool
    history: dict


def solve(model, cell_type, n, perturb=0.0, seed=5, settings=None):
    """Solve the case on an n x n rectangle mesh of the given cell type.

    Parameters
    ----------
    model : ConvectionDiffusion
    cell_type : {"quad", "triangle", "polygon", "mixed"}
    n : int
        Grid intervals per direction.
    perturb : float
        Node perturbation (random diagonals are used if > 0).
    seed : int
    settings : PseudoTimeSettings, optional
        Default: convergence by 11 orders of magnitude.

    Returns
    -------
    CaseResult
    """
    diagonal = "random" if perturb > 0.0 else "alternating"
    mesh = rectangle(
        n, n, cell_type=cell_type, perturb=perturb, seed=seed, diagonal=diagonal
    )
    geom = build_geometry(mesh).geom
    u_b = jnp.asarray(cd.exact_solution(model, np.asarray(geom.b_centre)))
    fi, fb = partial(cd.interior_flux, model), partial(cd.boundary_flux, model)
    lam = cd.spectral_radius(model, geom)
    result = solve_steady(
        residual=lambda u: cd.residual(model, geom, jnp.asarray(u), u_b),
        jacobian=lambda u: first_order_jacobian(
            geom, fi, fb, jnp.asarray(u), b_args=(u_b,)
        ),
        time_term=lambda u, cfl: sp.diags(lam / cfl),
        u0=np.zeros((mesh.n_cells, 1)),
        settings=settings or PseudoTimeSettings(residual_drop=11.0),
    )
    exact = cd.exact_solution(model, mesh.cell_centroid)
    area = mesh.cell_area
    error = float(
        np.sqrt(np.sum(area * (result.u[:, 0] - exact[:, 0]) ** 2) / area.sum())
    )
    h = float(np.sqrt(area.sum() / mesh.n_cells))
    return CaseResult(mesh, result.u, exact, error, h, result.converged, result.history)


def observed_orders(results):
    """Observed order of accuracy between successive refinement levels."""
    e = np.array([r.error for r in results])
    h = np.array([r.h for r in results])
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])

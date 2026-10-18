"""Benchmark B3: lid-driven cavity (Ghia, Ghia & Shin 1982).

Unit square, lid velocity 1, Re = U L / nu. Centreline profiles are compared
with Table I (u along x = 0.5) and Table II (v along y = 0.5) of Ghia et al.,
stored in ``data/ghia1982_u.csv`` and ``data/ghia1982_v.csv``.
"""

from importlib import resources
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.solver import PseudoTimeSettings


class CavityResult(NamedTuple):
    """Solution and comparison with Ghia et al.

    Attributes
    ----------
    problem : IncompressibleProblem
    w : ndarray, shape (n_cells, 3)
        Converged state (p, u, v).
    converged : bool
    history : dict
    ghia_y, ghia_u : ndarray
        Reference u along the vertical centreline.
    ghia_x, ghia_v : ndarray
        Reference v along the horizontal centreline.
    u_centre, v_centre : ndarray
        Computed values at the reference locations.
    """

    problem: object
    w: np.ndarray
    converged: bool
    history: dict
    ghia_y: np.ndarray
    ghia_u: np.ndarray
    ghia_x: np.ndarray
    ghia_v: np.ndarray
    u_centre: np.ndarray
    v_centre: np.ndarray


def ghia_data(reynolds):
    """Return (y, u, x, v) reference profiles for Re in {100, 400, 1000}."""
    out = []
    for name in ("ghia1982_u.csv", "ghia1982_v.csv"):
        text = resources.files("pyaeroflow.cases.data").joinpath(name).read_text()
        rows = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
        header = rows[0].split(",")
        data = np.array([[float(x) for x in ln.split(",")] for ln in rows[1:]])
        col = header.index(f"Re{int(reynolds)}")
        out += [data[:, 0], data[:, col]]
    return tuple(out)


def interpolate_profiles(problem, w, points_u, points_v):
    """Linear interpolation of the cell-centred solution at probe points.

    Boundary-face values are added to the interpolation points so that the
    profiles reach the walls.
    """
    geom, bc = problem.geom, problem.bc
    w_b = np.asarray(
        inc.boundary_state(bc, geom.b_normal, jnp.asarray(w)[geom.b_owner])
    )
    points = np.vstack((problem.mesh.cell_centroid, np.asarray(geom.b_centre)))
    values = np.vstack((w, w_b))
    interp = LinearNDInterpolator(points, values)
    return interp(points_u)[:, inc.U], interp(points_v)[:, inc.V]


def solve(
    reynolds,
    n,
    cell_type="quad",
    cluster=True,
    beta=1.5,
    limiter="none",
    settings=None,
):
    """Solve the cavity at the given Reynolds number on an n x n mesh.

    Parameters
    ----------
    reynolds : float
    n : int
    cell_type : str
    cluster : bool
        Cluster grid lines towards all four walls (tanh, strength ``beta``).
    beta : float
    limiter : str
    settings : PseudoTimeSettings, optional

    Returns
    -------
    CavityResult
    """
    clustering = (
        {"cluster_x": "both", "cluster_y": "both", "beta": beta} if cluster else {}
    )
    mesh = rectangle(n, n, cell_type=cell_type, **clustering)
    conditions = {
        "bottom": {"type": "wall"},
        "left": {"type": "wall"},
        "right": {"type": "wall"},
        "top": {"type": "wall", "velocity": (1.0, 0.0)},
    }
    model = inc.IncompressibleModel(
        density=1.0, viscosity=1.0 / reynolds, limiter=limiter
    )
    problem = IncompressibleProblem(mesh, model, conditions)
    result = problem.solve(
        settings=settings or PseudoTimeSettings(residual_drop=9.0, max_iterations=500)
    )
    y, u_ref, x, v_ref = ghia_data(reynolds)
    u_c, v_c = interpolate_profiles(
        problem,
        result.u,
        np.column_stack((np.full_like(y, 0.5), y)),
        np.column_stack((x, np.full_like(x, 0.5))),
    )
    return CavityResult(
        problem,
        result.u,
        result.converged,
        result.history,
        y,
        u_ref,
        x,
        v_ref,
        u_c,
        v_c,
    )

"""B1: laminar flat plate vs the Blasius similarity solution.

Plate from x = 0 to x = L = 1 in a uniform stream U = 1; the domain starts at
x = -0.5 with a symmetry boundary ahead of the plate. Reynolds number
Re_L = U L / nu. Compared: the skin-friction coefficient C_f(x) =
0.664 / sqrt(Re_x) and the velocity profile u / U = f'(eta) at x = 0.5 with
eta = y sqrt(U / (nu x)).

The Blasius solution is the asymptotic result for Re_x -> infinity. At finite
Reynolds numbers the leading-edge displacement raises the edge velocity and
the skin friction by O(Re_x^-1): about +3 % at Re_L = 1e4 and +0.5 % at
Re_L = 1e5 in PyAeroFlow runs (consistent with the second-order boundary-layer
theory of Imai and others). The case therefore uses Re_L = 1e5 by default.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import brentq

from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import rectangle, split_boundary
from pyaeroflow.physics import forces as fz
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.solver import PseudoTimeSettings


def blasius(eta_max=10.0):
    """Solve f''' + f f'' / 2 = 0, f(0) = f'(0) = 0, f'(inf) = 1 by shooting.

    Returns
    -------
    fpp0 : float
        f''(0) (0.332057...).
    f : callable
        ``f(eta) -> (f, f', f'')`` from the dense ODE solution.
    """

    def rhs(_, y):
        return [y[1], y[2], -0.5 * y[0] * y[2]]

    def shoot(fpp0):
        sol = solve_ivp(rhs, [0.0, eta_max], [0.0, 0.0, fpp0], rtol=1e-10, atol=1e-12)
        return sol.y[1, -1] - 1.0

    fpp0 = brentq(shoot, 0.2, 0.5, xtol=1e-12)
    sol = solve_ivp(
        rhs, [0.0, eta_max], [0.0, 0.0, fpp0], rtol=1e-10, atol=1e-12, dense_output=True
    )
    return fpp0, sol.sol


class FlatPlateResult(NamedTuple):
    """B1 result."""

    problem: object
    w: np.ndarray
    converged: bool
    history: dict
    x: np.ndarray
    cf: np.ndarray
    cf_blasius: np.ndarray
    eta: np.ndarray
    u_profile: np.ndarray
    u_blasius: np.ndarray
    x_profile: float


def solve(
    nx=160, ny=80, reynolds=1.0e5, beta=3.5, height=1.0, x_profile=0.5, settings=None
):
    """Laminar flat plate on a wall-clustered quadrilateral mesh.

    Parameters
    ----------
    nx, ny : int
        Cells in x (uniform over [-0.5, 1]) and y (tanh-clustered to the wall).
    reynolds : float
        Re_L with L = 1, U = 1 (sets nu = 1 / Re_L).
    beta : float
        Wall clustering strength.
    height : float
        Domain height.
    x_profile : float
        Station of the velocity-profile comparison.
    settings : PseudoTimeSettings, optional

    Returns
    -------
    FlatPlateResult
    """
    u_inf, nu = 1.0, 1.0 / reynolds
    mesh = rectangle(
        nx,
        ny,
        x=(-0.5, 1.0),
        y=(0.0, height),
        cell_type="quad",
        cluster_y="bottom",
        beta=beta,
    )
    mesh = split_boundary(
        mesh,
        "bottom",
        {"symmetry": lambda x, y: x < 0.0, "plate": lambda x, y: x >= 0.0},
    )
    conditions = {
        "left": {"type": "velocity-inlet", "velocity": (u_inf, 0.0)},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
        "top": {"type": "farfield", "velocity": (u_inf, 0.0), "pressure": 0.0},
        "symmetry": {"type": "symmetry"},
        "plate": {"type": "wall"},
    }
    model = inc.IncompressibleModel(density=1.0, viscosity=nu)
    problem = IncompressibleProblem(mesh, model, conditions)
    result = problem.solve(
        problem.initial_state((u_inf, 0.0)),
        settings or PseudoTimeSettings(residual_drop=8.0, max_iterations=300),
    )
    faces = problem.boundary_faces(["plate"])
    d = fz.wall_distribution(
        problem.wall_loads(result.u), problem.geom, faces, 0.5 * u_inf**2
    )
    order = np.argsort(d["x"])
    x, cf = d["x"][order], d["cf"][order]
    cf_blasius = 0.664 / np.sqrt(u_inf * x / nu)

    _, f = blasius()
    eta = np.linspace(0.0, 8.0, 41)
    y = eta * np.sqrt(nu * x_profile / u_inf)
    w_b = np.asarray(
        inc.boundary_state(
            problem.bc,
            problem.geom.b_normal,
            jnp.asarray(result.u)[problem.geom.b_owner],
        )
    )
    points = np.vstack((mesh.cell_centroid, np.asarray(problem.geom.b_centre)))
    interp = LinearNDInterpolator(points, np.vstack((result.u, w_b)))
    u_profile = (
        interp(np.column_stack((np.full_like(y, x_profile), y)))[:, inc.U] / u_inf
    )
    return FlatPlateResult(
        problem,
        result.u,
        result.converged,
        result.history,
        x,
        cf,
        cf_blasius,
        eta,
        u_profile,
        f(eta)[1],
        x_profile,
    )

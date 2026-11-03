"""Verification case: compressible laminar flat plate (L1, C2).

Plate from x = 0 to x = 1 in a uniform stream at Mach M, Reynolds number
Re_L = rho U L / mu of the freestream; the domain starts at x = -0.5 with a
symmetry boundary ahead of the plate, as the incompressible case B1
(:mod:`pyaeroflow.cases.flat_plate`).

Reference: the compressible similarity solution of the laminar boundary
layer (Howarth-Dorodnitsyn variables; White, Viscous Fluid Flow, 3rd ed.,
sec. 7-3) for an ideal gas with constant c_p and Prandtl number,

    (C f'')' + f f'' = 0,
    (C / Pr g')' + f g' + (gamma - 1) M^2 C f''^2 = 0,

with f' = u / U, g = T / T_e and the Chapman-Rubesin parameter
C = rho mu / (rho_e mu_e) = mu(g) / (mu_e g). Boundary conditions
f(0) = f'(0) = 0, f'(inf) = g(inf) = 1, and g'(0) = 0 (adiabatic wall) or
g(0) = T_w / T_e (isothermal wall). Then

    C_f sqrt(Re_x) = sqrt(2) C_w f''(0),
    q_w sqrt(Re_x) / (rho_e U c_p T_e) = -C_w g'(0) / (sqrt(2) Pr),

and the adiabatic wall temperature g(0) gives the recovery factor
r = (T_aw / T_e - 1) / ((gamma - 1) M^2 / 2), close to sqrt(Pr).
"""

from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_bvp

from pyaeroflow.driver import FlowProblem
from pyaeroflow.mesh import rectangle, split_boundary
from pyaeroflow.physics import compressible as comp
from pyaeroflow.physics import forces as fz
from pyaeroflow.solver import LinearSettings, PseudoTimeSettings


class Similarity(NamedTuple):
    """Similarity solution of the compressible laminar boundary layer."""

    cf_sqrt_re: float  # C_f sqrt(Re_x)
    heat_flux_sqrt_re: float  # q_w sqrt(Re_x) / (rho_e U c_p T_e)
    wall_temperature: float  # T_w / T_e
    recovery_factor: float  # adiabatic wall only, else None


def similarity(
    mach,
    prandtl=0.72,
    gamma=1.4,
    sutherland=110.4 / 288.15,
    wall_temperature=None,
    eta_max=12.0,
):
    """Solve the compressible similarity equations (``solve_bvp``).

    Parameters
    ----------
    mach, prandtl, gamma : float
    sutherland : float or None
        Sutherland's constant over the edge temperature; None = constant mu.
    wall_temperature : float or None
        T_w / T_e of an isothermal wall; None = adiabatic.
    eta_max : float
        Outer edge of the similarity coordinate.

    Returns
    -------
    Similarity
    """
    m2 = (gamma - 1.0) * mach**2

    def chapman(g):
        g = np.maximum(g, 1e-6)
        if sutherland is None:
            return 1.0 / g
        return np.sqrt(g) * (1.0 + sutherland) / (g + sutherland)

    # y = (f, f', C f'', g, C/Pr g')
    def rhs(_, y):
        c = chapman(y[3])
        fpp = y[2] / c
        gp = prandtl * y[4] / c
        return np.vstack((y[1], fpp, -y[0] * fpp, gp, -y[0] * gp - m2 * c * fpp**2))

    def bc(ya, yb):
        wall = ya[4] if wall_temperature is None else ya[3] - wall_temperature
        return np.array([ya[0], ya[1], yb[1] - 1.0, yb[3] - 1.0, wall])

    eta = np.linspace(0.0, eta_max, 400)
    t_w0 = 1.0 + 0.85 * 0.5 * m2 if wall_temperature is None else wall_temperature
    y0 = np.zeros((5, eta.size))
    y0[1] = np.tanh(eta)
    y0[0] = np.log(np.cosh(eta))
    y0[2] = 0.47 / np.cosh(eta) ** 2
    y0[3] = 1.0 + (t_w0 - 1.0) * np.exp(-eta)
    y0[4] = -(t_w0 - 1.0) * np.exp(-eta) / prandtl
    sol = solve_bvp(rhs, bc, eta, y0, tol=1e-9, max_nodes=100000)
    if not sol.success:
        raise RuntimeError(f"similarity solution failed: {sol.message}")
    ya = sol.y[:, 0]
    g_w = float(ya[3])
    recovery = (g_w - 1.0) / (0.5 * m2) if wall_temperature is None and m2 else None
    return Similarity(
        float(np.sqrt(2.0) * ya[2]),
        float(-ya[4] / np.sqrt(2.0)),
        g_w,
        recovery,
    )


class CompressiblePlateResult(NamedTuple):
    """Result of :func:`solve`."""

    problem: object
    w: np.ndarray
    converged: bool
    history: dict
    x: np.ndarray
    cf: np.ndarray  # C_f(x)
    wall_temperature: np.ndarray  # T_w / T_inf (cell values next to the wall)
    heat_flux: np.ndarray  # q_w / (rho U c_p T_inf)
    reference: Similarity


def solve(
    nx=160,
    ny=80,
    mach=0.5,
    reynolds=1.0e5,
    wall_temperature=None,
    beta=3.5,
    height=1.0,
    settings=None,
):
    """Compressible laminar flat plate on a wall-clustered quadrilateral mesh.

    Parameters
    ----------
    nx, ny : int
        Cells in x (uniform over [-0.5, 1]) and y (tanh-clustered to the wall).
    mach, reynolds : float
        Freestream Mach number and Re_L (L = 1).
    wall_temperature : float or None
        T_w / T_inf of an isothermal plate; None = adiabatic.
    beta, height : float
        Wall clustering strength and domain height.
    settings : PseudoTimeSettings, optional

    Returns
    -------
    CompressiblePlateResult
    """
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
    r_gas, ff = comp.freestream_conditions(mach)
    plate = {"type": "wall"}
    if wall_temperature is not None:
        plate["temperature"] = float(wall_temperature)
    conditions = {
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.0), "temperature": 1.0},
        "right": {"type": "pressure-outlet", "pressure": r_gas},
        "top": ff,
        "symmetry": {"type": "symmetry"},
        "plate": plate,
    }
    model = comp.CompressibleModel(gas_constant=r_gas, viscosity=1.0 / reynolds)
    problem = FlowProblem(mesh, model, conditions)
    result = problem.solve(
        problem.initial_state((1.0, 0.0), r_gas),
        settings
        or PseudoTimeSettings(
            residual_drop=8.0,
            max_iterations=200,
            linear=LinearSettings(type="pardiso"),
        ),
    )
    faces = problem.boundary_faces(["plate"])
    d = fz.wall_distribution(problem.wall_loads(result.u), problem.geom, faces, 0.5)
    order = np.argsort(d["x"])
    owner = np.asarray(problem.geom.b_owner)[faces][order]
    cp = comp.heat_capacity(model)
    q = problem.wall_heat_flux(result.u)[faces][order] / cp  # rho = U = T = 1
    ref = similarity(
        mach,
        model.prandtl,
        model.gamma,
        model.sutherland_temperature,
        wall_temperature,
    )
    return CompressiblePlateResult(
        problem,
        result.u,
        result.converged,
        result.history,
        d["x"][order],
        d["cf"][order],
        np.asarray(result.u)[owner, comp.T],
        q,
        ref,
    )

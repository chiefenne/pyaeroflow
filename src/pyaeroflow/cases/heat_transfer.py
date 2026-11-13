"""Validation of the incompressible energy equation (variable density, V1).

Three cases with reference data that needs no external file:

- de Vahl Davis (1983) natural-convection cavity: unit square, hot left and
  cold right wall, adiabatic top and bottom, Pr = 0.71; mean Nusselt number
  1.118, 2.243, 4.519, 8.800 at Ra = 1e3 .. 1e6 (the benchmark solution;
  Le Quere (1991) gives 8.825 at Ra = 1e6). Run here with the ideal-gas
  density at Delta T / T = 1 % (the Boussinesq limit).
- Laminar heated flat plate (Pohlhausen): Nu_x = theta'(0) Re_x^(1/2) of the
  similarity solution, computed by :func:`pohlhausen`.
- Heated cylinder in cross flow at low Reynolds number: mean Nu of the
  correlation of Lange, Durst & Breuer (1998), Pr = 0.71,
  Nu = 0.082 Re^0.5 + 0.734 Re^x, x = 0.05 + 0.226 Re^0.085.
- Turbulent heated flat plate (RANS, stage V2): local Stanton number of the
  correlation of Kays & Crawford for an isothermal plate, turbulent from the
  leading edge, St Pr^0.4 = 0.0287 Re_x^-0.2 (5e5 < Re_x < 5e6,
  0.5 < Pr < 1), on the NASA TMR plate domain.

Non-dimensional set-up of the cavity (reference length L, buoyancy velocity
U = sqrt(g beta Delta T L), T_ref the mean temperature): rho = 1 / T,
mu = sqrt(Pr / Ra), k = mu / Pr, c_p = 1, |g| = 1 / (Delta T / T_ref).
"""

import math
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from pyaeroflow.driver import FlowProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import thermo as th
from pyaeroflow.physics import variable_density as vd
from pyaeroflow.solver import PseudoTimeSettings

DE_VAHL_DAVIS_NUSSELT = {1.0e3: 1.118, 1.0e4: 2.243, 1.0e5: 4.519, 1.0e6: 8.800}


def lange_cylinder_nusselt(reynolds):
    """Mean Nu of a heated cylinder at Pr = 0.71 (Lange et al. 1998, Re <= 200)."""
    x = 0.05 + 0.226 * reynolds**0.085
    return 0.082 * reynolds**0.5 + 0.734 * reynolds**x


def pohlhausen(prandtl, eta_max=12.0):
    """theta'(0) of the laminar flat plate: Nu_x = pohlhausen(Pr) Re_x^(1/2).

    Blasius f''' + f f'' / 2 = 0 and the energy equation
    theta'' + Pr f theta' / 2 = 0 (theta = (T - T_w) / (T_inf - T_w)),
    solved by shooting on f''(0) and the quadrature
    theta'(0) = 1 / int_0^inf exp(-Pr/2 int_0^eta f) d eta.
    """

    def rhs(_, y):
        f, fp, fpp, intf, integral = y
        return [fp, fpp, -0.5 * f * fpp, f, math.exp(-0.5 * prandtl * intf)]

    def shoot(fpp0):
        sol = solve_ivp(
            rhs, (0.0, eta_max), [0.0, 0.0, fpp0, 0.0, 0.0], rtol=1e-11, atol=1e-13
        )
        return sol

    fpp0 = brentq(lambda a: shoot(a).y[1, -1] - 1.0, 0.2, 0.5, xtol=1e-13)
    return 1.0 / shoot(fpp0).y[4, -1]


class CavityResult(NamedTuple):
    """Natural-convection cavity: solution and mean Nusselt numbers."""

    problem: object
    w: np.ndarray
    converged: bool
    iterations: int
    nusselt_hot: float
    nusselt_cold: float
    reference: float


def cavity_model(rayleigh, prandtl=0.71, delta=0.01, density_model="ideal_gas"):
    """Non-dimensional model of the cavity (see the module docstring)."""
    mu = math.sqrt(prandtl / rayleigh)
    if density_model == "ideal_gas":
        dens = th.DensityModel("ideal_gas", pressure=1.0)
    else:
        dens = th.DensityModel(density_model, density=1.0, expansion=1.0)
    thermo = th.Thermo(
        dens,
        th.PropertyLaw("constant", mu),
        th.PropertyLaw("prandtl", prandtl),
        th.PropertyLaw("constant", 1.0),
    )
    return vd.VariableDensityModel(
        density=1.0, thermo=thermo, gravity=(0.0, -1.0 / delta)
    )


def solve_cavity(
    rayleigh, n, prandtl=0.71, delta=0.01, beta=1.5, settings=None, **model_args
):
    """Solve the cavity on an n x n mesh clustered at the walls (tanh, ``beta``).

    Returns
    -------
    CavityResult
        Mean Nu of the hot and cold wall, q L / (k Delta T).
    """
    model = cavity_model(rayleigh, prandtl, delta, **model_args)
    mesh = rectangle(n, n, cluster_x="both", cluster_y="both", beta=beta)
    conditions = {
        "left": {"type": "wall", "temperature": 1.0 + 0.5 * delta},
        "right": {"type": "wall", "temperature": 1.0 - 0.5 * delta},
        "top": {"type": "wall"},
        "bottom": {"type": "wall"},
    }
    problem = FlowProblem(mesh, model, conditions)
    w0 = problem.initial_state((0.0, 0.0), 0.0, temperature=1.0)
    result = problem.solve(
        w0,
        settings=settings or PseudoTimeSettings(residual_drop=9.0, max_iterations=300),
    )
    q = problem.wall_heat_flux(result.u)
    k = float(th.conductivity(model.thermo, np.array(1.0)))
    length = np.asarray(problem.geom.b_length)

    def mean_nu(name):
        faces = problem.boundary_faces([name])
        return float(np.sum(q[faces] * length[faces]) / (k * delta))

    return CavityResult(
        problem,
        result.u,
        result.converged,
        int(result.history["iteration"][-1]) if "iteration" in result.history else -1,
        mean_nu("left"),
        -mean_nu("right"),
        DE_VAHL_DAVIS_NUSSELT.get(float(rayleigh)),
    )


def kays_crawford_stanton(reynolds_x, prandtl=0.71):
    """Local St of the turbulent isothermal plate: 0.0287 Re_x^-0.2 Pr^-0.4."""
    return 0.0287 * np.asarray(reynolds_x, dtype=float) ** -0.2 * prandtl**-0.4


class TurbulentPlateResult(NamedTuple):
    """Turbulent heated flat plate: solution and local St, C_f along the plate."""

    problem: object
    w: np.ndarray
    converged: bool
    iterations: int
    x: np.ndarray
    stanton: np.ndarray
    cf: np.ndarray
    reference: np.ndarray


def solve_turbulent_plate(
    turbulence="sa-neg",
    nx_plate=96,
    ny=64,
    reynolds=5.0e6,
    prandtl=0.71,
    delta=0.01,
    density_model="ideal_gas",
    settings=None,
):
    """Turbulent heated plate on the TMR domain (Re per unit length ``reynolds``).

    Non-dimensional: U = rho_inf = c_p = 1, T_inf = 1, plate at 1 + ``delta``,
    mu = 1 / Re at T_inf with Sutherland's law (S = 110.4 K / 300 K), k from
    Pr, Pr_t = 0.9 (default law). SA-neg with nu_tilde / nu = 3 or SST with
    Tu = 0.1 %, nu_t / nu = 1 at the inflow (the solver defaults).

    Returns
    -------
    TurbulentPlateResult
        St = q_w / (rho c_p U (T_w - T_inf)) and C_f = 2 tau_w per plate face.
    """
    from pyaeroflow.cases.tmr_flat_plate import plate_mesh

    mu = 1.0 / reynolds
    if density_model == "ideal_gas":
        dens = th.DensityModel("ideal_gas", pressure=1.0)
    else:
        dens = th.DensityModel(density_model, density=1.0, expansion=1.0)
    thermo = th.Thermo(
        dens,
        th.PropertyLaw("sutherland", mu, (110.4 / 300.0,)),
        th.PropertyLaw("prandtl", prandtl),
        th.PropertyLaw("constant", 1.0),
    )
    model = vd.VariableDensityModel(density=1.0, thermo=thermo, turbulence=turbulence)
    mesh = plate_mesh(nx_plate, 24, ny, reynolds=reynolds)
    conditions = {
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.0)},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
        "top": {"type": "farfield", "velocity": (1.0, 0.0), "pressure": 0.0},
        "symmetry": {"type": "symmetry"},
        "plate": {"type": "wall", "temperature": 1.0 + delta},
    }
    problem = FlowProblem(mesh, model, conditions)
    w0 = problem.initial_state((1.0, 0.0), 0.0, temperature=1.0)
    result = problem.solve(
        w0,
        settings=settings
        or PseudoTimeSettings(residual_drop=8.0, max_iterations=300, cfl_start=1000.0),
    )
    faces = problem.boundary_faces(["plate"])
    x = np.asarray(problem.geom.b_centre)[faces, 0]
    q = problem.wall_heat_flux(result.u)[faces]
    loads = problem.wall_loads(result.u)
    tau = np.hypot(*np.asarray(loads[1])[faces].T)
    return TurbulentPlateResult(
        problem,
        result.u,
        result.converged,
        int(result.history["iteration"][-1]) if "iteration" in result.history else -1,
        x,
        q / delta,
        2.0 * tau,
        kays_crawford_stanton(reynolds * x, prandtl),
    )

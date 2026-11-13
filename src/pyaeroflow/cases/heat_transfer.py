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


def similarity_plate(wall_ratio, prandtl=0.71, sutherland=None):
    """Laminar flat plate with variable properties (Howarth-Levy-Lees similarity).

    Low speed (no dissipation, no pressure work), constant c_p and Pr, ideal-gas
    density rho ~ 1/T and mu(T) of Sutherland's law (``sutherland`` = S / T_inf,
    None: mu ~ T^0, i.e. constant mu), k = mu c_p / Pr. With the density-weighted
    coordinate eta = sqrt(U / (2 rho_inf mu_inf x)) int rho dy the equations are
    (C f'')' + f f'' = 0 and (C theta' / Pr)' + f theta' = 0,
    C = rho mu / (rho_inf mu_inf), theta = (T - T_inf) / (T_w - T_inf).

    Returns
    -------
    (nusselt, skin_friction) : float
        Nu_x / sqrt(Re_x) and C_f sqrt(Re_x) at the wall-to-freestream
        temperature ratio ``wall_ratio`` (Nu_x = q_w x / (k_inf (T_w - T_inf)),
        C_f = tau_w / (rho_inf U^2 / 2)). ``wall_ratio`` = 1 gives Pohlhausen.
    """

    def mu_of(t):
        if sutherland is None:
            return 1.0
        return t**1.5 * (1.0 + sutherland) / (t + sutherland)

    def c_of(t):
        return mu_of(t) / t

    def rhs(_, y):
        f, fp, fpp, th_, thp = y
        t = 1.0 + (wall_ratio - 1.0) * th_
        h = 1e-6
        dc = (c_of(t + h) - c_of(t - h)) / (2.0 * h)
        c = c_of(t)
        tp = (wall_ratio - 1.0) * thp
        fppp = (-f * fpp - dc * tp * fpp) / c
        thpp = (-prandtl * f * thp - dc * tp * thp) / c
        return [fp, fpp, fppp, thp, thpp]

    def shoot(guess):
        fpp0, thp0 = guess
        sol = solve_ivp(
            rhs,
            (0.0, 12.0),
            [0.0, 0.0, fpp0, 1.0, thp0],
            rtol=1e-10,
            atol=1e-12,
        )
        return sol.y[:, -1]

    def miss(guess):
        y = shoot(guess)
        return [y[1] - 1.0, y[3]]

    from scipy.optimize import fsolve

    fpp0, thp0 = fsolve(miss, [0.4696, -0.3 * (prandtl / 0.71) ** 0.33])
    t_w = wall_ratio
    mu_w, rho_w = mu_of(t_w), 1.0 / t_w
    nusselt = -mu_w * rho_w * thp0 / math.sqrt(2.0)
    skin = math.sqrt(2.0) * rho_w * mu_w * fpp0
    return float(nusselt), float(skin)


class ChannelResult(NamedTuple):
    """Heated channel: local Nusselt number along the walls."""

    problem: object
    w: np.ndarray
    converged: bool
    iterations: int
    x: np.ndarray
    nusselt: np.ndarray
    reference: float


def solve_channel(
    flux=3.0e-4,
    reynolds=40.0,
    prandtl=0.71,
    length_ratio=8.0,
    nx=64,
    ny=24,
    settings=None,
):
    """Parallel-plate channel with a uniform heat flux on both walls.

    Non-dimensional: height H = 1/2, hydraulic diameter D_h = 2 H = 1, mean
    velocity 1, Re_Dh = ``reynolds``, rho = 1/T (ideal gas, T_ref = 1), constant
    mu and c_p, k = mu / Pr, parabolic inflow at T = 1, ``flux`` into the fluid
    at both walls. The local Nusselt number Nu = q D_h / (k (T_w - T_b)) with
    the mixing-cup temperature T_b of the cross-section approaches the
    fully developed value 140 / 17 = 8.235 (Shah & London) beyond the thermal
    entrance, x / (D_h Re Pr) of about 0.05.
    """
    h = 0.5
    length = length_ratio * h
    mu = 1.0 / reynolds
    thermo = th.Thermo(
        th.DensityModel("ideal_gas", pressure=1.0),
        th.PropertyLaw("constant", mu),
        th.PropertyLaw("prandtl", prandtl),
        th.PropertyLaw("constant", 1.0),
    )
    model = vd.VariableDensityModel(density=1.0, thermo=thermo)
    mesh = rectangle(
        nx,
        ny,
        x=(0.0, length),
        y=(0.0, h),
        cell_type="quad",
        cluster_y="both",
        beta=1.5,
    )

    def inflow(x, y):
        return 6.0 * y / h * (1.0 - y / h), np.zeros_like(y)

    conditions = {
        "left": {"type": "velocity-inlet", "velocity": inflow, "temperature": 1.0},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
        "bottom": {"type": "wall", "heat_flux": flux},
        "top": {"type": "wall", "heat_flux": flux},
    }
    problem = FlowProblem(mesh, model, conditions)
    w0 = problem.initial_state((1.0, 0.0), 0.0, temperature=1.0)
    result = problem.solve(
        w0,
        settings=settings
        or PseudoTimeSettings(residual_drop=9.0, max_iterations=300, cfl_start=1000.0),
    )
    w = result.u
    faces = problem.boundary_faces(["bottom"])
    xf = np.asarray(problem.geom.b_centre)[faces, 0]
    t_wall = problem.wall_temperature(w)[faces]
    centre = mesh.cell_centroid
    area = mesh.cell_area
    k = 1.0 / (reynolds * prandtl)
    nu_x = np.empty(faces.size)
    for i, x in enumerate(xf):
        col = np.isclose(centre[:, 0], x, atol=1e-9)
        rho = np.asarray(th.density(thermo, w[col, 3]))
        weight = rho * w[col, 1] * area[col]
        t_b = float(np.sum(weight * w[col, 3]) / np.sum(weight))
        nu_x[i] = flux * 1.0 / (k * (t_wall[i] - t_b))
    return ChannelResult(
        problem,
        w,
        result.converged,
        int(result.history["iteration"][-1]) if "iteration" in result.history else -1,
        xf,
        nu_x,
        140.0 / 17.0,
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

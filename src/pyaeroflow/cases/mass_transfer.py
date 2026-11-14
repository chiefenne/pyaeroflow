"""Validation of the species equations (mixture of the energy solver, stage S1).

Cases with exact or already validated reference solutions:

- Stefan tube: a species evaporates from a surface (fixed mass fraction
  Y_0, the bulk gas does not cross it) and diffuses through the stagnant gas
  to the open end at Y_L. With constant rho D the net mass flux is
  m = (rho D / L) ln((1 - Y_L) / (1 - Y_0)) and (1 - Y) varies exponentially
  along the tube; this holds for any ratio of the molar masses, the density
  then varies along the tube.
- Thermosolutal cavity: the cavity of de Vahl Davis with a second, solutal
  buoyancy of the same sign, Boussinesq density rho_0 (1 - beta (T - T_ref)
  - beta_Y Y), equal Schmidt and Prandtl numbers (Lewis number 1) and
  buoyancy ratio N = beta_Y dY / (beta dT). The mass fraction is then a copy
  of the temperature, the flow is that of the thermal cavity at the Rayleigh
  number Ra_T (1 + N), and the mean Sherwood number equals the mean Nusselt
  number of that cavity.
- Evaporating surfaces (stage S1b): with a Lewis number of 1 (Sc = Pr,
  Sc_t = Pr_t), equal specific heats and an adiabatic surface, the mass
  and heat transfer numbers are equal, B = (Y_s - Y_inf) / (1 - Y_s) =
  c_p (T_inf - T_s) / L, whatever the flow (Spalding). With the saturation
  law Y_s(T_s) this fixes the surface ("wet-bulb") temperature exactly,
  uniform along the surface; in the Stefan tube the net mass flux is then
  (rho D / L) ln(1 + B). On a turbulent plate the mass flux m = g ln(1 + B)
  gives the Stanton number of mass transfer g / (rho U), compared with
  Kays and Crawford's 0.0287 Re_x^-0.2 Sc^-0.4.
"""

import math
from typing import NamedTuple

import numpy as np

from pyaeroflow.cases.heat_transfer import DE_VAHL_DAVIS_NUSSELT, kays_crawford_stanton
from pyaeroflow.driver import FlowProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import thermo as th
from pyaeroflow.physics import variable_density as vd
from pyaeroflow.solver import PseudoTimeSettings


class StefanTubeResult(NamedTuple):
    """Stefan tube: solution, net mass flux and its exact value."""

    problem: object
    w: np.ndarray
    converged: bool
    mass_flux: float
    exact: float
    profile_error: float


def solve_stefan_tube(
    n=24, y_surface=0.5, y_open=0.0, molar_mass=0.622, schmidt=0.6, settings=None
):
    """Solve the Stefan tube of length 1 on an n x n mesh (surface at y = 0).

    Non-dimensional: rho = 1 and mu = 1 for the bulk gas at T = 1, so that
    rho D = 1 / Sc; ``molar_mass`` is W_species / W_bulk. The open end is a
    farfield at rest with the mass fraction ``y_open``.

    Returns
    -------
    StefanTubeResult
        ``mass_flux``: mean net mass flux through the surface;
        ``profile_error``: largest error of Y in the cells.
    """
    species = th.Species("vapour", molar_mass, th.PropertyLaw("schmidt", schmidt))
    thermo = th.Thermo(
        th.DensityModel("ideal_gas", pressure=1.0),
        th.PropertyLaw("constant", 1.0),
        th.PropertyLaw("prandtl", 0.71),
        th.PropertyLaw("constant", 1.0),
        (species,),
    )
    model = vd.VariableDensityModel(density=1.0, thermo=thermo)
    mesh = rectangle(n, n)
    conditions = {
        "bottom": {
            "type": "wall",
            "temperature": 1.0,
            "mass_fraction": (y_surface,),
            "stefan_flow": True,
        },
        "top": {
            "type": "farfield",
            "velocity": (0.0, 0.0),
            "pressure": 0.0,
            "temperature": 1.0,
            "mass_fraction": (y_open,),
        },
        "left": {"type": "symmetry"},
        "right": {"type": "symmetry"},
    }
    problem = FlowProblem(mesh, model, conditions)
    w0 = problem.initial_state((0.0, 0.0), 0.0, temperature=1.0)
    result = problem.solve(
        w0,
        settings=settings or PseudoTimeSettings(residual_drop=9.0, max_iterations=300),
    )
    _, m = problem.wall_species_flux(result.u)
    faces = problem.boundary_faces(["bottom"])
    length = np.asarray(problem.geom.b_length)[faces]
    ratio = (1.0 - y_open) / (1.0 - y_surface)
    y_cell = np.asarray(mesh.cell_centroid)[:, 1]
    exact_y = 1.0 - (1.0 - y_surface) * ratio**y_cell
    return StefanTubeResult(
        problem,
        result.u,
        result.converged,
        float(np.sum(m[faces] * length) / length.sum()),
        math.log(ratio) / schmidt,
        float(np.max(np.abs(result.u[:, 4] - exact_y))),
    )


class ThermosolutalResult(NamedTuple):
    """Thermosolutal cavity: mean Nusselt and Sherwood numbers of the hot wall."""

    problem: object
    w: np.ndarray
    converged: bool
    nusselt: float
    sherwood: float
    reference: float


def solve_thermosolutal_cavity(
    rayleigh, n, buoyancy_ratio=1.0, prandtl=0.71, delta=0.01, beta=1.5, settings=None
):
    """Solve the thermosolutal cavity at the total Rayleigh number ``rayleigh``.

    Lewis number 1; the thermal Rayleigh number is rayleigh / (1 + N) with
    the buoyancy ratio N. Non-dimensional as the thermal cavity of
    :mod:`pyaeroflow.cases.heat_transfer` (buoyancy velocity of the total
    density difference). The hot left wall holds the species at Y = delta,
    the cold right wall at 0.

    Returns
    -------
    ThermosolutalResult
        ``reference``: mean Nu of de Vahl Davis at ``rayleigh`` (None if
        not tabulated).
    """
    mu = math.sqrt(prandtl / rayleigh)
    thermal = 1.0 / (1.0 + buoyancy_ratio)
    species = th.Species(
        "light",
        diffusivity=th.PropertyLaw("schmidt", prandtl),
        expansion=buoyancy_ratio * thermal,
    )
    thermo = th.Thermo(
        th.DensityModel("boussinesq", density=1.0, expansion=thermal),
        th.PropertyLaw("constant", mu),
        th.PropertyLaw("prandtl", prandtl),
        th.PropertyLaw("constant", 1.0),
        (species,),
    )
    model = vd.VariableDensityModel(
        density=1.0, thermo=thermo, gravity=(0.0, -1.0 / delta)
    )
    mesh = rectangle(n, n, cluster_x="both", cluster_y="both", beta=beta)
    conditions = {
        "left": {
            "type": "wall",
            "temperature": 1.0 + 0.5 * delta,
            "mass_fraction": (delta,),
        },
        "right": {
            "type": "wall",
            "temperature": 1.0 - 0.5 * delta,
            "mass_fraction": (0.0,),
        },
        "top": {"type": "wall"},
        "bottom": {"type": "wall"},
    }
    problem = FlowProblem(mesh, model, conditions)
    w0 = problem.initial_state((0.0, 0.0), 0.0, temperature=1.0)
    result = problem.solve(
        w0,
        settings=settings or PseudoTimeSettings(residual_drop=9.0, max_iterations=300),
    )
    faces = problem.boundary_faces(["left"])
    length = np.asarray(problem.geom.b_length)[faces]
    q = problem.wall_heat_flux(result.u)[faces]
    j, _ = problem.wall_species_flux(result.u)
    k = float(th.conductivity(thermo, np.array(1.0)))
    rho_d = mu / prandtl
    return ThermosolutalResult(
        problem,
        result.u,
        result.converged,
        float(np.sum(q * length) / (k * delta)),
        float(np.sum(j[faces, 0] * length) / (rho_d * delta)),
        DE_VAHL_DAVIS_NUSSELT.get(float(rayleigh)),
    )


# Water vapour in air at T_ref = 300 K, p_op = 1 atm (non-dimensional by
# c_p T_ref and T_ref): L = 2.45e6 J/kg, Clausius-Clapeyron through 3536 Pa
# at 300 K, W_v / W_air = 0.622.
WATER_VAPOUR = th.Species(
    "vapour",
    0.622,
    th.PropertyLaw("schmidt", 0.71),
    latent_heat=2.45e6 / (1006.0 * 300.0),
    saturation=(
        "clausius-clapeyron",
        math.log(3536.0 / 101325.0),
        2.45e6 * 18.015 / 8314.462618 / 300.0,
        1.0,
    ),
)


def wet_bulb(species=WATER_VAPOUR, t_inf=1.0, y_inf=0.0):
    """Exact surface temperature and transfer number (T_s, Y_s, B), Le = 1.

    Solves (Y_s(T_s) - Y_inf) / (1 - Y_s) = (T_inf - T_s) / L (c_p = 1) by
    bisection.
    """
    thermo = th.Thermo(species=(species,))

    def mass(t):
        y = float(th.saturation_mass_fraction(thermo, t))
        return y, (y - y_inf) / (1.0 - y)

    lo, hi = 0.5 * t_inf, t_inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mass(mid)[1] - (t_inf - mid) / species.latent_heat > 0.0:
            hi = mid
        else:
            lo = mid
    t_s = 0.5 * (lo + hi)
    y_s, b = mass(t_s)
    return t_s, y_s, b


class EvaporationResult(NamedTuple):
    """Evaporating surface: solution, surface values and the exact references."""

    problem: object
    w: np.ndarray
    converged: bool
    iterations: int
    x: np.ndarray
    surface_temperature: np.ndarray
    mass_flux: np.ndarray
    wet_bulb: float
    transfer_number: float
    reference: object


def solve_evaporating_tube(n=24, prandtl=0.71, settings=None):
    """Stefan tube of length 1 with an evaporating, adiabatic water surface.

    Non-dimensional as :func:`solve_stefan_tube` (rho D = 1 / Sc, Sc = Pr);
    dry air at T = 1 at the open end. ``reference`` is the exact net mass
    flux ln(1 + B) / Sc.
    """
    species = WATER_VAPOUR._replace(diffusivity=th.PropertyLaw("schmidt", prandtl))
    thermo = th.Thermo(
        th.DensityModel("ideal_gas", pressure=1.0),
        th.PropertyLaw("constant", 1.0),
        th.PropertyLaw("prandtl", prandtl),
        th.PropertyLaw("constant", 1.0),
        (species,),
    )
    model = vd.VariableDensityModel(density=1.0, thermo=thermo)
    mesh = rectangle(n, n)
    conditions = {
        "bottom": {"type": "wall", "evaporation": True},
        "top": {
            "type": "farfield",
            "velocity": (0.0, 0.0),
            "pressure": 0.0,
            "temperature": 1.0,
            "mass_fraction": (0.0,),
        },
        "left": {"type": "symmetry"},
        "right": {"type": "symmetry"},
    }
    problem = FlowProblem(mesh, model, conditions)
    w0 = problem.initial_state((0.0, 0.0), 0.0, temperature=1.0)
    result = problem.solve(
        w0,
        settings=settings or PseudoTimeSettings(residual_drop=9.0, max_iterations=300),
    )
    t_s, _, b = wet_bulb(species)
    return _evaporation_result(
        problem, result, ["bottom"], t_s, b, math.log1p(b) / prandtl
    )


def _evaporation_result(problem, result, walls, t_s, b, reference):
    faces = problem.boundary_faces(walls)
    _, m = problem.wall_species_flux(result.u)
    history = result.history
    return EvaporationResult(
        problem,
        result.u,
        result.converged,
        int(history["iteration"][-1]) if "iteration" in history else -1,
        np.asarray(problem.geom.b_centre)[faces, 0],
        problem.wall_temperature(result.u)[faces],
        m[faces],
        t_s,
        b,
        reference,
    )


def solve_evaporating_plate(
    turbulence="sa-neg",
    nx_plate=96,
    ny=64,
    reynolds=5.0e6,
    prandtl=0.71,
    prandtl_turbulent=0.9,
    settings=None,
):
    """Turbulent plate (TMR domain) with an evaporating, adiabatic water surface.

    Dry air at T = 1 and U = 1; mu = 1 / Re (Sutherland), Sc = Pr and
    Sc_t = Pr_t (Lewis number 1), c_p = 1 for both species. ``reference``
    is Kays and Crawford's St_m = 0.0287 Re_x^-0.2 Sc^-0.4 per plate face;
    compare it with m / (rho_inf U ln(1 + B)).
    """
    from pyaeroflow.cases.tmr_flat_plate import plate_mesh

    species = WATER_VAPOUR._replace(diffusivity=th.PropertyLaw("schmidt", prandtl))
    thermo = th.Thermo(
        th.DensityModel("ideal_gas", pressure=1.0),
        th.PropertyLaw("sutherland", 1.0 / reynolds, (110.4 / 300.0,)),
        th.PropertyLaw("prandtl", prandtl),
        th.PropertyLaw("constant", 1.0),
        (species,),
    )
    law = th.TurbulentPrandtl("constant", (("value", prandtl_turbulent),))
    model = vd.VariableDensityModel(
        density=1.0,
        thermo=thermo,
        turbulence=turbulence,
        turbulent_prandtl=law,
        turbulent_schmidt=law,
    )
    mesh = plate_mesh(nx_plate, 24, ny, reynolds=reynolds)
    inflow = {"temperature": 1.0, "mass_fraction": (0.0,)}
    conditions = {
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.0), **inflow},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
        "top": {
            "type": "farfield",
            "velocity": (1.0, 0.0),
            "pressure": 0.0,
            **inflow,
        },
        "symmetry": {"type": "symmetry"},
        "plate": {"type": "wall", "evaporation": True},
    }
    problem = FlowProblem(mesh, model, conditions)
    w0 = problem.initial_state((1.0, 0.0), 0.0, temperature=1.0)
    result = problem.solve(
        w0,
        settings=settings
        or PseudoTimeSettings(residual_drop=8.0, max_iterations=300, cfl_start=1000.0),
    )
    t_s, _, b = wet_bulb(species)
    out = _evaporation_result(problem, result, ["plate"], t_s, b, None)
    return out._replace(reference=kays_crawford_stanton(reynolds * out.x, prandtl))


def multicomponent_tube_reference(path, length=1.0):
    """Species fluxes of a multicomponent Stefan tube from its 1D equations.

    Reads the case file ``path`` (a Stefan-flow wall at y = 0 with the mass
    fractions ``Y_0`` of the transported species, a farfield at rest at y =
    ``length`` with ``Y_L``; isothermal, bulk gas does not cross the surface)
    and integrates the steady 1D equations with the thermodynamics of the
    case: m_k = m Y_k + J_k(Y, dY/dx) constant for all species with
    m = sum m_k (the bulk species has no net flux), by a boundary value
    problem with the constants m_k as unknowns. It tests the discretisation,
    the boundary conditions and the non-dimensionalisation of the solver
    with any diffusion model and mixture rule, not the model itself.

    Returns the mass fluxes m_k into the fluid in kg/(s m).
    """
    import jax
    import jax.numpy as jnp
    from scipy.integrate import solve_bvp

    from pyaeroflow.config import load_case
    from pyaeroflow.run import setup

    config = load_case(path)
    problem, scaling, _, _ = setup(config)
    thermo = problem.model.thermo
    b = config.data["boundary"]
    y0 = np.asarray(b["bottom"]["mass_fraction"], dtype=float)
    y1 = np.asarray(b["top"]["mass_fraction"], dtype=float)
    t = jnp.asarray(1.0)
    n = y0.size

    def gradient(y, m_k):
        """dY/dx with J_k(Y, g) = m_k - m Y_k; J is linear in the gradient."""
        y = jnp.asarray(y)
        j0 = th.diffusion_flux(thermo, t, y, jnp.zeros(n))
        matrix = jax.jacfwd(lambda g: th.diffusion_flux(thermo, t, y, g))(jnp.zeros(n))
        m = jnp.sum(m_k)
        return jnp.linalg.solve(matrix, m_k - m * y - j0)

    rhs_one = jax.jit(gradient)

    def rhs(x, y, p):
        out = [np.asarray(rhs_one(y[:, i], p)) for i in range(y.shape[1])]
        return np.stack(out, axis=1)

    def bc(ya, yb, p):
        return np.concatenate((ya - y0, yb - y1))

    # flux of the Stefan tube of the total vapour as the starting guess
    x = np.linspace(0.0, length / scaling.length, 41)
    m0 = float(np.sum(y0)) * 0.1
    guess = np.array([y0[k] + (y1[k] - y0[k]) * x / x[-1] for k in range(n)])
    sol = solve_bvp(rhs, bc, x, guess, p=np.full(n, m0 * 0.1), tol=1e-9, max_nodes=4000)
    if not sol.success:
        raise RuntimeError(f"reference boundary value problem failed: {sol.message}")
    return np.asarray(sol.p) * scaling.density * scaling.velocity

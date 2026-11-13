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
"""

import math
from typing import NamedTuple

import numpy as np

from pyaeroflow.cases.heat_transfer import DE_VAHL_DAVIS_NUSSELT
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

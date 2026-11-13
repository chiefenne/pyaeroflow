"""Variable-density incompressible flow with an energy equation (V1, laminar).

Low-Mach formulation inside the incompressible (artificial-compressibility)
solver after the concept of Economon (2020): the density depends on the
temperature through an equation of state at a constant thermodynamic
pressure, so that it acts on the flow (buoyancy, inertia, continuity), and
the energy equation is solved together with continuity and momentum in one
Newton system.

Working variables per cell: V = (p, u, v, T) with the gauge (dynamic)
pressure p and the temperature T (non-dimensional: T / T_ref). The state
layout is :func:`pyaeroflow.physics.layout.incompressible_layout` with the
energy block; later blocks (turbulence in V2) are inserted before it by name.

Steady equations, residual R(V) = net outflow minus sources:

    continuity   div(rho v)                               = 0
    momentum     div(rho v v + p I - tau)                 = (rho - rho_op) g
    energy       div(rho v h - k grad T)                  = 0

with rho(T) of the density model, h(T) = int_{T_ref}^T c_p dT, mu(T), k(T)
and c_p(T) of :mod:`pyaeroflow.physics.thermo`. Pressure work and viscous
dissipation are left out (low Mach number). With gravity, p is the pressure
minus the hydrostatic pressure of the operating density rho_op (the density
at the reference temperature, ``model.density``).

Pseudo-time preconditioning: Gamma = dU/dV of the conservative variables
U = (rho, rho u, rho v, rho h) with d(rho)/dp replaced by 1/beta^2,

    Gamma = [[1/beta^2,   0,   0,   rho_T            ],
             [u/beta^2,   rho, 0,   u rho_T          ],
             [v/beta^2,   0,   rho, v rho_T          ],
             [h/beta^2,   0,   0,   rho c_p + h rho_T]],   rho_T = d(rho)/dT.

The flux Jacobian is A = v_n Gamma + B with B independent of T, and
Gamma^-1 B couples only (p, u, v): Gamma^-1 A = v_n I + M, where the
(p, u, v) block of M is that of the constant-density solver (with the local
density) and the T row and column of M vanish. Therefore

    |Gamma^-1 A| = diag(|Gamma^-1 A|_(p,u,v), |v_n|)

exactly (eigenvalues v_n, v_n - beta, v_n + beta, v_n), and the
flux-difference splitting F = (F_L + F_R)/2 - Gamma |Gamma^-1 A| (V_R - V_L)/2
upwinds the temperature with |v_n| and the density jump with it. The energy
row of the dissipation is h times the continuity row plus rho c_p |v_n| dT,
so that the solution does not depend on the zero of h.

beta^2 = beta_factor max(|v|^2, U_b^2) with the buoyancy velocity
U_b^2 = |g| L_ref (rho_max - rho_min) / rho_op over the cells and the
prescribed boundary temperatures: a flow that starts at rest gets its beta^2
from the temperature differences.

Closed domains (stage V3): with an ideal gas and no boundary that exchanges
mass, the thermodynamic pressure is no longer the operating pressure but
follows from the total mass M_0 of the gas in the domain,

    sum_i A_i rho(p_th, T_i) = M_0   ->   p_th = p_op M_0 / sum_i A_i rho(p_op, T_i)

(rho = p_th / T is linear in p_th). :func:`thermodynamic_pressure` evaluates
it from the state inside the residual, so that p_th is a global unknown that
the Newton linearization (forward-mode AD) contains exactly;
:func:`closed_model` returns the model at that pressure. The first-order
Jacobian uses p_th of the current state but not its global derivative.

Species (stage S1, laminar): with ``thermo.species`` the state is
(p, u, v, T, Y_1 .. Y_n) with the mass fractions of the transported species
(the fluid of the case is the bulk species with 1 - sum Y_k), and

    species k    div(rho v Y_k + J_k)                              = 0
    energy       div(rho v h - k grad T + sum_k (h_k - h_bulk) J_k) = 0

with rho(T, Y), h(T, Y) and the diffusion fluxes J_k of
:func:`pyaeroflow.physics.thermo.diffusion_flux` (Fick's law with a
correction velocity). Gamma gets the columns d(U)/dY_k; the mass fractions
are upwinded with |v_n| like T. Species boundary conditions: walls have zero
flux (default), fixed mass fractions (``mass_fraction``), a prescribed
diffusion flux into the fluid (``species_flux``) or fixed mass fractions
with the Stefan flow of a wall that the bulk species does not cross
(``stefan_flow``): the net mass flux m = sum_k J_k / (1 - sum_k Y_k) of the
diffusion fluxes leaves the wall and carries mass, momentum, enthalpy and
species. Inlets and farfields prescribe Y (default 0), outlets take the
interior values. Without species every function is the one of stage V1-V3.

Thermal boundary conditions: walls (and transpiration walls) adiabatic
(default), isothermal (``temperature``) or with a prescribed heat flux into
the fluid (``heat_flux``); velocity inlets, total-pressure inlets and
farfields prescribe T; pressure outlets take the interior T; symmetry faces
carry no heat flux.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pyaeroflow.numerics import (
    boundary_face_gradient,
    interior_face_gradient,
    lsq_gradient,
    reconstruct,
    venkatakrishnan_wang,
)
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import thermo as th
from pyaeroflow.physics.incompressible import (
    FARFIELD,
    KIND_NAMES,
    PRESSURE_OUTLET,
    SYMMETRY,
    TOTAL_PRESSURE_INLET,
    VELOCITY_INLET,
    WALL,
)
from pyaeroflow.physics.layout import incompressible_layout
from pyaeroflow.solver import divergence

# Same flow family as the constant-density solver: restart files of either
# can start the other (the columns are matched by name).
FLOW = "incompressible"
P, U, V, T = 0, 1, 2, 3
# T stays positive (Kelvin / T_ref): positivity limiter of the update.
POSITIVE_COLUMNS = (T,)
RESIDUAL_NAMES = ("p", "u", "v", "T")
# The residual has a cell source (buoyancy) with a T derivative: the driver
# adds :func:`source_blocks` to the first-order Jacobian.
CELL_SOURCES = True

# Thermal kinds of wall faces.
ADIABATIC, ISOTHERMAL, HEAT_FLUX = 0, 1, 2
THERMAL_NAMES = {
    "adiabatic": ADIABATIC,
    "isothermal": ISOTHERMAL,
    "heat-flux": HEAT_FLUX,
}
# Species kinds of wall faces: zero flux, prescribed diffusion flux, fixed mass
# fractions, fixed mass fractions with Stefan flow.
ZERO_FLUX, SPECIES_FLUX, FIXED_FRACTION, STEFAN_FLOW = 0, 1, 2, 3
WALL_TYPES = ("wall", "transpiration-wall")
FIXED_TEMPERATURE_TYPES = ("velocity-inlet", "total-pressure-inlet", "farfield")


class VariableDensityModel(NamedTuple):
    """Model parameters (hashable; a static jit argument).

    Attributes
    ----------
    density : float
        Operating density rho_op (the density at the reference temperature):
        the buoyancy source is (rho - rho_op) g.
    thermo : Thermo
        Density model and property laws (:mod:`pyaeroflow.physics.thermo`).
    gravity : tuple of float
        Gravitational acceleration (g_x, g_y), non-dimensional g L / U^2.
    turbulent_prandtl : TurbulentPrandtl
        Pr_t law of the eddy conductivity (RANS, stage V2).
    beta_factor, spatial_order, limiter, limiter_threshold,
    viscous_gradient, alpha, body_force, pin :
        As in :class:`pyaeroflow.physics.incompressible.IncompressibleModel`.
    turbulence, transition, turbulence_spatial_order, coupling,
    turbulence_intensity, critical_amplification, lm_positive_jacobian,
    lm_jacobian_smoothing :
        RANS (stage V2), as in
        :class:`~pyaeroflow.physics.incompressible.IncompressibleModel`; the
        residual is then that of :mod:`pyaeroflow.physics.rans_variable_density`
        (SA, SA-BCM, SA-AFT) or
        :mod:`pyaeroflow.physics.rans_sst_variable_density` (SST, SST-LM).
    closed_mass : float or None
        Closed domain (stage V3): total mass M_0 of the gas (non-dimensional,
        rho times area); p_th then follows from it
        (:func:`thermodynamic_pressure`). None: constant p_th.
    pressure_work : float
        R / c_p,ref, the factor of dp_th/dt in the unsteady energy equation
        d(rho h)/dt + div(rho v h) = (R / c_p,ref) dp_th/dt + div(k grad T)
        (non-dimensional; zero without an ideal gas).
    """

    density: float = 1.0
    thermo: th.Thermo = th.Thermo()
    gravity: tuple = (0.0, 0.0)
    turbulent_prandtl: th.TurbulentPrandtl = th.TurbulentPrandtl()
    beta_factor: float = 4.1
    spatial_order: int = 2
    limiter: str = "none"
    limiter_threshold: float = 0.05
    viscous_gradient: str = "alpha-damping"
    alpha: float = 4.0 / 3.0
    body_force: tuple = (0.0, 0.0)
    pin: tuple = None
    turbulence: str = "none"
    turbulence_spatial_order: int = 1
    coupling: str = "segregated"
    transition: str = "none"
    turbulence_intensity: float = 0.0
    critical_amplification: float = None
    lm_positive_jacobian: bool = True
    lm_jacobian_smoothing: float = 0.0
    closed_mass: float = None
    pressure_work: float = 0.0

    @property
    def viscosity(self):
        """Dynamic viscosity at the reference temperature (mu(1))."""
        return float(self.thermo.viscosity.value)

    @property
    def layout(self):
        """State layout: flow block and the energy block."""
        return incompressible_layout(
            self.turbulence, self.transition, energy=True, species=self.n_species
        )

    @property
    def n_species(self):
        """Number of transported species."""
        return len(self.thermo.species)

    @property
    def n_var(self):
        """Number of variables (p, u, v, the turbulence columns and T)."""
        return self.layout.n_var


class BoundaryData(NamedTuple):
    """Per-boundary-face condition (a JAX pytree).

    Attributes
    ----------
    kind : int array (n_bfaces,)
        Flow kinds of :mod:`pyaeroflow.physics.incompressible`.
    value : float array (n_bfaces, 4)
        Prescribed (p, u, v, T) where the kind uses them (T: isothermal
        walls, inlets, farfields).
    thermal : int array (n_bfaces,)
        ADIABATIC, ISOTHERMAL or HEAT_FLUX at walls (ADIABATIC elsewhere).
    heat_flux : float array (n_bfaces,)
        Prescribed heat flux into the fluid at HEAT_FLUX walls.
    face_density : float array (n_bfaces,)
        rho of the prescribed temperature (total-pressure inlets).
    density : float
        Operating density.
    nu_tilde, turbulence : float arrays or None
        Turbulence boundary values (RANS, stage V2), as in
        :class:`pyaeroflow.physics.incompressible.BoundaryData`.
    thermo : StaticThermo or None
        Property laws (RANS: the SST wall omega uses nu of the wall
        temperature).
    """

    kind: jnp.ndarray
    value: jnp.ndarray
    thermal: jnp.ndarray
    heat_flux: jnp.ndarray
    face_density: jnp.ndarray
    density: float = 1.0
    nu_tilde: jnp.ndarray = None
    turbulence: jnp.ndarray = None
    thermo: th.StaticThermo = None  # RANS: property laws at the boundary faces
    # Species (stage S1): kind per face (ZERO_FLUX, ...) and prescribed
    # diffusion fluxes into the fluid (n_bfaces, n_species); None without.
    species: jnp.ndarray = None
    species_flux: jnp.ndarray = None


def species_kind(bc):
    """Species kind of a wall condition dict."""
    if bc.get("species_flux") is not None:
        return SPECIES_FLUX
    if bc.get("mass_fraction") is not None:
        return STEFAN_FLOW if bc.get("stefan_flow") else FIXED_FRACTION
    return ZERO_FLUX


def _species_values(entry, n_species, x, y):
    """Per-face values (n, n_species) of a list of numbers or callables."""
    if len(entry) != n_species:
        raise ValueError(f"expected {n_species} species values, got {len(entry)}")
    return np.column_stack([inc._evaluate(v, x, y) for v in entry])


def thermal_kind(bc):
    """Thermal kind of a wall condition dict."""
    if bc.get("heat_flux") is not None:
        return HEAT_FLUX
    if bc.get("temperature") is not None:
        return ISOTHERMAL
    return ADIABATIC


def build_boundary_data(disc, conditions, model):
    """:class:`BoundaryData` from a dict of boundary conditions.

    The flow keys are those of
    :func:`pyaeroflow.physics.incompressible.boundary_data`. Walls take
    ``temperature`` (isothermal) or ``heat_flux`` (into the fluid), neither
    for an adiabatic wall; inlets and farfields take ``temperature``
    (default 1, the reference temperature). Values are non-dimensional and
    may be callables of the face centres.
    """
    flow = inc.boundary_data(disc, conditions, model.density)
    nb = disc.boundary_face_ids.size
    t_val = np.ones(nb)
    thermal = np.zeros(nb, dtype=np.int64)
    heat = np.zeros(nb)
    centres = np.asarray(disc.geom.b_centre)
    for name, sl in disc.boundary_slices.items():
        bc = conditions[name]
        x, y = centres[sl, 0], centres[sl, 1]
        if bc["type"] in WALL_TYPES:
            kind = thermal_kind(bc)
            thermal[sl] = kind
            if kind == ISOTHERMAL:
                t_val[sl] = inc._evaluate(bc["temperature"], x, y)
            elif kind == HEAT_FLUX:
                heat[sl] = inc._evaluate(bc["heat_flux"], x, y)
        elif bc["type"] in FIXED_TEMPERATURE_TYPES:
            t_val[sl] = inc._evaluate(bc.get("temperature", 1.0), x, y)
        elif bc["type"] == "pressure-outlet" and bc.get("temperature") is not None:
            t_val[sl] = inc._evaluate(bc["temperature"], x, y)
    value = np.column_stack((np.asarray(flow.value), t_val))
    ns = model.n_species
    kinds = fluxes = y_val = None
    if ns:
        # Species: walls ``mass_fraction`` (fixed, with ``stefan_flow`` the
        # Stefan flow) or ``species_flux`` (into the fluid), neither for zero
        # flux; inlets and farfields ``mass_fraction`` (default 0).
        y_val, fluxes = np.zeros((nb, ns)), np.zeros((nb, ns))
        kinds = np.zeros(nb, dtype=np.int64)
        for name, sl in disc.boundary_slices.items():
            bc = conditions[name]
            x, y = centres[sl, 0], centres[sl, 1]
            if bc["type"] in WALL_TYPES:
                kinds[sl] = species_kind(bc)
                if kinds[sl.start] == SPECIES_FLUX:
                    fluxes[sl] = _species_values(bc["species_flux"], ns, x, y)
                elif kinds[sl.start] != ZERO_FLUX:
                    y_val[sl] = _species_values(bc["mass_fraction"], ns, x, y)
            elif bc.get("mass_fraction") is not None:
                y_val[sl] = _species_values(bc["mass_fraction"], ns, x, y)
        value = np.column_stack((value, y_val))
        kinds, fluxes = jnp.asarray(kinds), jnp.asarray(fluxes)
        y_val = jnp.asarray(y_val)
    face_density = np.asarray(th.density(model.thermo, jnp.asarray(t_val), y_val))
    # RANS (stage V2): the turbulence boundary values of the flow part.
    turbulent = model.turbulence != "none"
    return BoundaryData(
        flow.kind,
        jnp.asarray(value),
        jnp.asarray(thermal),
        jnp.asarray(heat),
        jnp.asarray(face_density),
        float(model.density),
        flow.nu_tilde if turbulent else None,
        flow.turbulence if turbulent else None,
        th.StaticThermo(model.thermo) if turbulent else None,
        kinds,
        fluxes,
    )


def jacobian_boundary_args(bc):
    """Per-face arrays of :func:`boundary_flux` (first-order Jacobian)."""
    args = (bc.kind, bc.value, bc.thermal, bc.heat_flux, bc.face_density)
    if bc.species is not None:
        args += (bc.species, bc.species_flux)
    return args


def lsq_include(conditions, n_var=4, n_species=0):
    """``lsq_include`` callback: boundary values that enter the gradients.

    As in the constant-density solver, plus T where it is known: isothermal
    walls, inlets, farfields and symmetry faces; not at adiabatic and
    heat-flux walls and pressure outlets. T is the last column (the energy
    block follows the turbulence and transition blocks, RANS stage V2), in
    front of the ``n_species`` mass fractions: these enter where they are
    prescribed (not at zero-flux and flux walls and pressure outlets).
    """
    t_col = n_var - 1 - n_species

    def include(slices, n_bfaces):
        mask = np.ones((n_bfaces, n_var), dtype=bool)
        for name, sl in slices.items():
            bc = conditions[name]
            k = KIND_NAMES[bc["type"]]
            if k in (WALL, VELOCITY_INLET, TOTAL_PRESSURE_INLET):
                mask[sl, P] = False
            elif k == PRESSURE_OUTLET:
                mask[sl, U:] = False
            if bc["type"] in WALL_TYPES and thermal_kind(bc) != ISOTHERMAL:
                mask[sl, t_col] = False
            if bc["type"] in WALL_TYPES and species_kind(bc) < FIXED_FRACTION:
                mask[sl, t_col + 1 :] = False
        return mask

    return include


def boundary_state(bc, normal, v_in):
    """Boundary-face state (p, u, v, T) from the interior state ``v_in``.

    Flow columns as in the constant-density solver; the total-pressure
    inlet uses the density of its prescribed temperature. T: prescribed at
    isothermal walls, inlets and farfields, interior elsewhere.
    """
    vn = v_in[:, U] * normal[:, 0] + v_in[:, V] * normal[:, 1]
    mirrored = v_in.at[:, U].add(-vn * normal[:, 0]).at[:, V].add(-vn * normal[:, 1])
    kind = bc.kind[:, None]
    out = jnp.where(kind == SYMMETRY, mirrored, v_in)
    out = jnp.where(kind == FARFIELD, bc.value, out)
    dyn = jnp.maximum(bc.value[:, P] - v_in[:, P], 0.0)
    speed = jnp.sqrt(2.0 * dyn / bc.face_density + 1e-300)
    inlet = (
        v_in.at[:, U].set(speed * bc.value[:, U]).at[:, V].set(speed * bc.value[:, V])
    )
    out = jnp.where(kind == TOTAL_PRESSURE_INLET, inlet, out)
    k = bc.kind
    prescribe_v = (k == WALL) | (k == VELOCITY_INLET)
    out = out.at[:, U].set(jnp.where(prescribe_v, bc.value[:, U], out[:, U]))
    out = out.at[:, V].set(jnp.where(prescribe_v, bc.value[:, V], out[:, V]))
    out = out.at[:, P].set(jnp.where(k == PRESSURE_OUTLET, bc.value[:, P], out[:, P]))
    fixed_t = (
        (k == VELOCITY_INLET)
        | (k == TOTAL_PRESSURE_INLET)
        | (k == FARFIELD)
        | ((k == WALL) & (bc.thermal == ISOTHERMAL))
    )
    out = out.at[:, T].set(jnp.where(fixed_t, bc.value[:, T], out[:, T]))
    if bc.species is None:
        return out
    fixed_y = (
        (k == VELOCITY_INLET)
        | (k == TOTAL_PRESSURE_INLET)
        | (k == FARFIELD)
        | ((k == WALL) & (bc.species >= FIXED_FRACTION))
    )
    return out.at[:, T + 1 :].set(
        jnp.where(fixed_y[:, None], bc.value[:, T + 1 :], out[:, T + 1 :])
    )


def _fractions(model, w):
    """Mass fractions of the states ``w`` (laminar layout), None without species."""
    return w[..., T + 1 :] if model.thermo.species else None


def _properties(model, t, y=None):
    """rho, d(rho)/dT, h and c_p at temperatures ``t`` (mass fractions ``y``)."""
    tm = model.thermo
    if y is None:
        return (
            th.density(tm, t),
            th.density_derivative(tm, t),
            th.enthalpy(tm, t),
            th.specific_heat(tm, t),
        )
    return (
        th.density(tm, t, y),
        th.density_derivative(tm, t, y),
        th.enthalpy(tm, t, y),
        th.specific_heat(tm, t, y),
    )


def _temperature_and_fractions(model, w):
    """T and Y (None without species) of states in any layout of the model.

    T is the last column in front of the mass fractions (the turbulence
    columns of the RANS modules come before it).
    """
    ns = len(model.thermo.species)
    if ns == 0:
        return w[..., -1], None
    return w[..., -1 - ns], w[..., w.shape[-1] - ns :]


def thermodynamic_pressure(model, geom, w):
    """Thermodynamic pressure p_th (non-dimensional) of the state ``w``.

    The operating pressure of the density model, or, in a closed domain
    (``model.closed_mass``), p_op M_0 / sum_i A_i rho(p_op, T_i): the total
    mass of the gas stays M_0. T is the last column of ``w`` (in front of the
    mass fractions of a mixture). Works on traced and on concrete arrays.
    """
    p_op = model.thermo.density.pressure
    if model.closed_mass is None:
        return p_op
    t, y = _temperature_and_fractions(model, w)
    rho_op = (
        th.density(model.thermo, t) if y is None else th.density(model.thermo, t, y)
    )
    return p_op * model.closed_mass / jnp.sum(geom.area * rho_op)


def at_pressure(model, pressure):
    """``model`` with the thermodynamic pressure ``pressure`` (ideal gas)."""
    tm = model.thermo
    new = tm._replace(density=tm.density._replace(pressure=pressure))
    if tm.species:  # D_k is given at the operating pressure
        new = new._replace(pressure_ratio=pressure / tm.density.pressure)
    return model._replace(thermo=new)


def closed_model(model, geom, w):
    """Return the model at p_th of ``w`` (closed domains), else ``model``.

    Inside traced code the pressure is a tracer: the returned model must not
    be passed as a static argument to a jitted function.
    """
    if model.closed_mass is None:
        return model
    return at_pressure(model, thermodynamic_pressure(model, geom, w))


def closed_boundary(model, bc):
    """``bc`` with the property laws of ``model`` (RANS: the SST wall omega)."""
    if model.closed_mass is None or bc.thermo is None:
        return bc
    return bc._replace(thermo=th.StaticThermo(model.thermo))


def with_pressure_argument(flux):
    """Compact face flux with p_th as the last per-face argument.

    ``flux(model, params, *args, normal, delta)`` becomes
    ``f(model, params, *args, p_th, normal, delta)``: the first-order
    Jacobian of a closed domain passes the current p_th per face.
    """

    def wrapped(model, params, *args):
        *rest, pressure, normal, delta = args
        return flux(at_pressure(model, pressure), params, *rest, normal, delta)

    return wrapped


def is_closed_domain(conditions):
    """Return True if no boundary condition exchanges mass.

    Walls, slip walls, symmetry and periodic faces; transpiration walls only
    with zero normal velocity; no wall with the Stefan flow of a species.
    """
    for bc in conditions.values():
        kind = bc["type"]
        if bc.get("stefan_flow"):  # species: a net mass flux through the wall
            return False
        if kind in ("wall", "slip-wall", "symmetry", "periodic"):
            continue
        if kind == "transpiration-wall":
            vn = bc.get("normal_velocity", 0.0)
            if not callable(vn) and float(vn) == 0.0:
                continue
        return False
    return True


def buoyancy_velocity_squared(
    model, t_cells, t_boundary, y_cells=None, y_boundary=None
):
    """U_b^2 = |g| (rho_max - rho_min) / rho_op (reference length 1)."""
    g = float(np.hypot(*model.gravity))
    if g == 0.0:
        return 0.0
    t = np.concatenate((np.ravel(t_cells), np.ravel(t_boundary)))
    if y_cells is None:
        rho = np.asarray(th.density(model.thermo, jnp.asarray(t)))
    else:
        y = np.concatenate((y_cells, y_boundary))
        rho = np.asarray(th.density(model.thermo, jnp.asarray(t), jnp.asarray(y)))
    return g * float(rho.max() - rho.min()) / model.density


def beta_squared(model, w, bc):
    """beta^2 = beta_factor max(max|v|^2, U_b^2) over cells and boundary values."""
    w = np.asarray(w)
    value = np.asarray(bc.value)
    v2 = max(
        float(np.max(np.sum(w[:, U : V + 1] ** 2, axis=1))),
        float(np.max(np.sum(value[:, U : V + 1] ** 2, axis=1))),
    )
    fixed = (np.asarray(bc.thermal) == ISOTHERMAL) | np.isin(
        np.asarray(bc.kind), (VELOCITY_INLET, TOTAL_PRESSURE_INLET, FARFIELD)
    )
    if model.thermo.species:
        # Species: also the faces with prescribed mass fractions (the
        # prescribed T and Y of either kind of face; defaults elsewhere).
        fixed = fixed | (np.asarray(bc.species) >= FIXED_FRACTION)
        ub2 = buoyancy_velocity_squared(
            model, w[:, T], value[fixed, T], w[:, T + 1 :], value[fixed, T + 1 :]
        )
        return float(model.beta_factor * max(v2, ub2, 1e-10))
    ub2 = buoyancy_velocity_squared(model, w[:, T], value[fixed, T])
    return float(model.beta_factor * max(v2, ub2, 1e-10))


def scheme_parameters(model, w, bc):
    """Frozen per-evaluation scheme parameters: beta^2 (flow-family interface)."""
    return beta_squared(model, w, bc)


def inviscid_flux(model, w, normal):
    """F.n = (rho v_n, rho u v_n + p n_x, rho v v_n + p n_y, rho h v_n, rho Y v_n)."""
    t = w[:, T]
    y = _fractions(model, w)
    rho, _, h, _ = _properties(model, t, y)
    vn = w[:, U] * normal[:, 0] + w[:, V] * normal[:, 1]
    m = rho * vn
    flux = jnp.stack(
        (
            m,
            m * w[:, U] + w[:, P] * normal[:, 0],
            m * w[:, V] + w[:, P] * normal[:, 1],
            m * h,
        ),
        axis=1,
    )
    if y is None:
        return flux
    return jnp.concatenate((flux, m[:, None] * y), axis=1)


def preconditioner(model, w, beta2):
    """Gamma for states w (n, n_var): shape (n, n_var, n_var)."""
    y = _fractions(model, w)
    rho, rho_t, h, cp = _properties(model, w[:, T], y)
    one = jnp.ones_like(rho)
    q = jnp.stack((one, w[:, U], w[:, V], h), axis=1)  # (1, u, v, h)
    if y is not None:
        q = jnp.concatenate((q, y), axis=1)  # (1, u, v, h, Y)
    nv = w.shape[1]
    g = jnp.zeros((w.shape[0], nv, nv))
    g = g.at[:, :, P].set(q / beta2)
    g = g.at[:, :, T].set(q * rho_t[:, None])
    g = g.at[:, T, T].add(rho * cp)
    g = g.at[:, U, U].set(rho)
    g = g.at[:, V, V].set(rho)
    if y is None:
        return g
    # Columns Y_k: q d(rho)/dY_k, rho (h_k - h_bulk) in the energy row and
    # rho on the diagonal.
    rho_y = th.density_species_derivative(model.thermo, w[:, T], y)
    g = g.at[:, :, T + 1 :].set(q[:, :, None] * rho_y[:, None, :])
    dh = th.species_enthalpy(model.thermo, w[:, T])
    g = g.at[:, T, T + 1 :].add(rho[:, None] * dh)
    k = jnp.arange(T + 1, nv)
    return g.at[:, k, k].add(rho[:, None])


def abs_preconditioned_jacobian(model, w, normal, beta2):
    """|Gamma^-1 A| for states w (n, n_var) and unit normals: (n, n_var, n_var).

    The (p, u, v) block of the constant-density solver with the local
    density, and |v_n| for T and the mass fractions.
    """
    rho = _properties(model, w[:, T], _fractions(model, w))[0]
    flow = inc.abs_preconditioned_jacobian(
        inc.IncompressibleModel(density=1.0), w[:, :3], normal, beta2
    )
    # Scale the density in the off-diagonal entries (the 3x3 block takes rho
    # from its model): row p carries rho, column p carries 1/rho.
    flow = flow.at[:, P, U:].multiply(rho[:, None])
    flow = flow.at[:, U:, P].divide(rho[:, None])
    vn = w[:, U] * normal[:, 0] + w[:, V] * normal[:, 1]
    nv = w.shape[1]
    m = jnp.zeros((w.shape[0], nv, nv))
    m = m.at[:, :3, :3].set(flow)
    k = jnp.arange(T, nv)
    return m.at[:, k, k].set(jnp.abs(vn)[:, None])


def dissipation(model, w_m, dw, normal, beta2):
    """Gamma |Gamma^-1 A| dw for mean states w_m and jumps dw, shape (n, n_var)."""
    y = _fractions(model, w_m)
    rho, rho_t, h, cp = _properties(model, w_m[:, T], y)
    beta = jnp.sqrt(beta2)
    nx, ny = normal[:, 0], normal[:, 1]
    tx, ty = -ny, nx
    vn = w_m[:, U] * nx + w_m[:, V] * ny
    l0, lm, lp = jnp.abs(vn), jnp.abs(vn - beta), jnp.abs(vn + beta)
    s, d = 0.5 * (lm + lp), 0.5 * (lp - lm)
    dvn = dw[:, U] * nx + dw[:, V] * ny
    dvt = dw[:, U] * tx + dw[:, V] * ty
    # x = |Gamma^-1 A| dw
    x_p = s * dw[:, P] + beta * rho * d * dvn
    x_n = d * dw[:, P] / (beta * rho) + s * dvn
    x_u = nx * x_n + l0 * tx * dvt
    x_v = ny * x_n + l0 * ty * dvt
    x_t = l0 * dw[:, T]
    # Gamma x
    mass = x_p / beta2 + rho_t * x_t
    if y is not None:
        x_y = l0[:, None] * dw[:, T + 1 :]
        rho_y = th.density_species_derivative(model.thermo, w_m[:, T], y)
        dh = th.species_enthalpy(model.thermo, w_m[:, T])
        mass = mass + jnp.sum(rho_y * x_y, axis=1)
        energy = mass * h + rho * cp * x_t + rho * jnp.sum(dh * x_y, axis=1)
        return jnp.concatenate(
            (
                jnp.stack(
                    (
                        mass,
                        mass * w_m[:, U] + rho * x_u,
                        mass * w_m[:, V] + rho * x_v,
                        energy,
                    ),
                    axis=1,
                ),
                mass[:, None] * y + rho[:, None] * x_y,
            ),
            axis=1,
        )
    return jnp.stack(
        (
            mass,
            mass * w_m[:, U] + rho * x_u,
            mass * w_m[:, V] + rho * x_v,
            mass * h + rho * cp * x_t,
        ),
        axis=1,
    )


def fds_flux(model, w_l, w_r, normal, beta2):
    """Flux-difference-splitting flux per unit length, states (n, 4)."""
    w_m = 0.5 * (w_l + w_r)
    central = 0.5 * (
        inviscid_flux(model, w_l, normal) + inviscid_flux(model, w_r, normal)
    )
    return central - 0.5 * dissipation(model, w_m, w_r - w_l, normal, beta2)


def viscous_flux(model, grad, t_f, normal, heat=True):
    """-(0, tau.n, k grad T.n) for face gradients grad (n, 4, 2) and face T.

    ``heat`` (bool or bool array) False drops the conduction.
    """
    mu = th.viscosity(model.thermo, t_f)
    flow = inc.viscous_flux(model, grad[:, U : V + 1], normal, mu)
    k = th.conductivity(model.thermo, t_f)
    q = k * jnp.einsum("nk,nk->n", grad[:, T, :], normal)
    q = jnp.where(heat, q, 0.0)
    return jnp.concatenate((flow, -q[:, None]), axis=1)


def species_flux(model, grad, t_f, y_f, normal):
    """Diffusion fluxes J_k.n of face gradients grad (n, n_var, 2), (n, n_species)."""
    grad_n = jnp.einsum("nsk,nk->ns", grad[:, T + 1 :, :], normal)
    return th.diffusion_flux(model.thermo, t_f, y_f, grad_n)


def _with_species(model, flux, j, t_f):
    """Append the species fluxes ``j`` and add their enthalpy flux to ``flux``."""
    dh = th.species_enthalpy(model.thermo, t_f)
    flux = flux.at[:, T].add(jnp.sum(dh * j, axis=1))
    return jnp.concatenate((flux, j), axis=1)


def boundary_viscous_flux(model, bc, grad, w_b, normal):
    """-F_v.n through boundary faces (n_bfaces, 4).

    Momentum as in the constant-density solver (mu of the face T). Energy:
    conduction at isothermal walls, inlets, farfields and outlets, minus the
    prescribed heat flux at heat-flux walls, nothing at adiabatic walls and
    symmetry faces.
    """
    k = bc.kind
    wall = k == WALL
    conduct = ~((wall & (bc.thermal != ISOTHERMAL)) | (k == SYMMETRY))
    flux = viscous_flux(model, grad, w_b[:, T], normal, heat=conduct)
    prescribed = wall & (bc.thermal == HEAT_FLUX)
    flux = flux.at[:, T].set(jnp.where(prescribed, -bc.heat_flux, flux[:, T]))
    if bc.species is None:
        return flux
    # Species: diffusion where the mass fractions are prescribed and at
    # outlets, minus the prescribed flux at flux walls, nothing at zero-flux
    # walls and symmetry faces.
    t_b, y_b = w_b[:, T], w_b[:, T + 1 :]
    j = species_flux(model, grad, t_b, y_b, normal)
    diffuse = ~((wall & (bc.species < FIXED_FRACTION)) | (k == SYMMETRY))
    j = jnp.where(diffuse[:, None], j, 0.0)
    j = jnp.where((wall & (bc.species == SPECIES_FLUX))[:, None], -bc.species_flux, j)
    flux = _with_species(model, flux, j, t_b)
    # Stefan flow: the bulk species does not cross the wall, the net mass
    # flux m = sum J_k / (1 - sum Y_k) carries mass, momentum (velocity
    # m / rho along the normal), enthalpy and species.
    stefan = wall & (bc.species == STEFAN_FLOW)
    bulk = jnp.maximum(1.0 - jnp.sum(y_b, axis=1), 1e-6)
    m = jnp.where(stefan, jnp.sum(j, axis=1) / bulk, 0.0)
    rho, _, h, _ = _properties(model, t_b, y_b)
    carried = jnp.stack(
        (m, m * m / rho * normal[:, 0], m * m / rho * normal[:, 1], m * h), axis=1
    )
    return flux + jnp.concatenate((carried, m[:, None] * y_b), axis=1)


def boundary_convective_flux(model, bc, normal, w_in, beta2):
    """Convective flux through boundary faces from the interior state w_in."""
    w_out = boundary_state(bc, normal, w_in)
    upwind = fds_flux(model, w_in, w_out, normal, beta2)
    # Walls: mass and enthalpy flux of the prescribed (transpiration)
    # velocity at the wall temperature, momentum flux of pressure.
    wall = (
        w_in.at[:, U]
        .set(bc.value[:, U])
        .at[:, V]
        .set(bc.value[:, V])
        .at[:, T]
        .set(w_out[:, T])
    )
    if bc.species is not None:
        wall = wall.at[:, T + 1 :].set(w_out[:, T + 1 :])
    wall_flux = inviscid_flux(model, wall, normal)
    sym_flux = inviscid_flux(model, w_in.at[:, U].set(0.0).at[:, V].set(0.0), normal)
    kind = bc.kind[:, None]
    return jnp.where(
        kind == WALL, wall_flux, jnp.where(kind == SYMMETRY, sym_flux, upwind)
    )


def _limiter(model, geom, w, w_b, grad):
    if model.limiter == "venkatakrishnan-wang":
        return venkatakrishnan_wang(geom, w, w_b, grad, model.limiter_threshold)
    if model.limiter == "venkatakrishnan":
        return venkatakrishnan_wang(
            geom, w, w_b, grad, model.limiter_threshold, local=True
        )
    if model.limiter == "none":
        return jnp.ones_like(w)
    raise ValueError(f"unknown limiter '{model.limiter}'")


@partial(jax.jit, static_argnums=0)
def limiter(model, geom, bc, w):
    """Limiter values psi of the current state, shape (n_cells, 4)."""
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    return _limiter(model, geom, w, w_b, lsq_gradient(geom, w, w_b))


def cell_sources(model, geom, w):
    """Buoyancy and body force per cell, shape (n_cells, 2): (rho - rho_op) g + f."""
    rho = _properties(model, w[:, T], _fractions(model, w))[0]
    g = jnp.asarray(model.gravity, dtype=w.dtype)
    f = jnp.asarray(model.body_force, dtype=w.dtype)
    return geom.area[:, None] * ((rho - model.density)[:, None] * g[None] + f[None])


@partial(jax.jit, static_argnums=(0, 4))
def residual(model, geom, bc, w, order, beta2, psi=None):
    """Residual R(V), shape (n_cells, 4): net outflow minus sources.

    Parameters as in :func:`pyaeroflow.physics.incompressible.residual`.
    In a closed domain the density follows p_th(w) (:func:`closed_model`).
    """
    model = closed_model(model, geom, w)
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    if order == 2:
        if psi is None:
            psi = _limiter(model, geom, w, w_b, grad)
        w_l, w_r, w_in = reconstruct(geom, w, grad, psi)
    else:
        w_l, w_r, w_in = w[geom.f_owner], w[geom.f_neigh], w[geom.b_owner]
    f_flux = fds_flux(model, w_l, w_r, geom.f_normal, beta2)
    b_flux = boundary_convective_flux(model, bc, geom.b_normal, w_in, beta2)
    g_f = interior_face_gradient(geom, w, grad, model.viscous_gradient, model.alpha)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    t_f = 0.5 * (w[geom.f_owner, T] + w[geom.f_neigh, T])
    v_flux = viscous_flux(model, g_f, t_f, geom.f_normal)
    if model.thermo.species:
        y_f = 0.5 * (w[geom.f_owner, T + 1 :] + w[geom.f_neigh, T + 1 :])
        j_f = species_flux(model, g_f, t_f, y_f, geom.f_normal)
        v_flux = _with_species(model, v_flux, j_f, t_f)
    f_flux = f_flux + v_flux
    b_flux = b_flux + boundary_viscous_flux(model, bc, g_b, w_b, geom.b_normal)
    r = divergence(geom, f_flux, b_flux)
    r = r.at[:, U : V + 1].add(-cell_sources(model, geom, w))
    if model.pin is not None:
        cell, p0 = model.pin
        r = r.at[cell, P].set(w[cell, P] - p0)
    return r


@partial(jax.jit, static_argnums=(0, 5))
def residual_jvp(model, geom, bc, w, dw, order, beta2, psi=None):
    """Exact Jacobian-vector product dR/dw(w) @ dw by forward-mode AD."""

    def f(x):
        return residual(model, geom, bc, x, order, beta2, psi)

    return jax.jvp(f, (w,), (dw,))[1]


@partial(jax.jit, static_argnums=(0, 4))
def residual_linearize(model, geom, bc, w, order, beta2, psi=None):
    """Residual and its linearization at ``w`` (``jax.linearize``)."""

    def f(x):
        return residual(model, geom, bc, x, order, beta2, psi)

    return jax.linearize(f, w)


def source_blocks(model, geom, bc, w):
    """dR/dw of the cell sources, (n_cells, n_var, n_var): -area rho_T g in (u, v; T).

    With species also -area d(rho)/dY_k g in (u, v; Y_k).
    """
    model = closed_model(model, geom, w)
    y = _fractions(model, w)
    rho_t = _properties(model, w[:, T], y)[1]
    g = jnp.asarray(model.gravity, dtype=w.dtype)
    nv = w.shape[1]
    blocks = jnp.zeros((w.shape[0], nv, nv), dtype=w.dtype)
    blocks = blocks.at[:, U, T].set(-geom.area * rho_t * g[0])
    blocks = blocks.at[:, V, T].set(-geom.area * rho_t * g[1])
    if y is None:
        return blocks
    rho_y = th.density_species_derivative(model.thermo, w[:, T], y)
    blocks = blocks.at[:, U, T + 1 :].set(-geom.area[:, None] * rho_y * g[0])
    return blocks.at[:, V, T + 1 :].set(-geom.area[:, None] * rho_y * g[1])


def interior_flux(model, beta2, w_o, w_n, normal, delta):
    """Compact first-order flux through an interior face (for the Jacobian)."""
    conv = fds_flux(model, w_o[None], w_n[None], normal[None], beta2)[0]
    dn = jnp.dot(delta, normal)
    if model.viscous_gradient == "alpha-damping":
        coeff = model.alpha / dn
    else:
        coeff = dn / jnp.dot(delta, delta)
    g = (coeff * (w_n - w_o))[:, None] * normal[None, :]
    t_f = 0.5 * (w_o[T] + w_n[T])
    flux = viscous_flux(model, g[None], t_f[None], normal[None])
    if model.thermo.species:
        y_f = 0.5 * (w_o[T + 1 :] + w_n[T + 1 :])
        j = species_flux(model, g[None], t_f[None], y_f[None], normal[None])
        flux = _with_species(model, flux, j, t_f[None])
    return conv + flux[0]


def boundary_flux(
    model, beta2, w_o, bc_kind, bc_value, bc_thermal, bc_heat, bc_rho, *args
):
    """Compact first-order flux through a boundary face (for the Jacobian).

    ``args``: the species kind and fluxes of the face (with species only),
    then the face normal and the vector from the cell centre to the face.
    """
    *species, normal, delta = args
    bc = BoundaryData(
        bc_kind[None],
        bc_value[None],
        bc_thermal[None],
        bc_heat[None],
        bc_rho[None],
        model.density,
    )
    if species:
        bc = bc._replace(species=species[0][None], species_flux=species[1][None])
    conv = boundary_convective_flux(model, bc, normal[None], w_o[None], beta2)[0]
    w_b = boundary_state(bc, normal[None], w_o[None])[0]
    dn = jnp.dot(delta, normal)
    g = ((w_b - w_o) / dn)[:, None] * normal[None, :]
    return conv + boundary_viscous_flux(model, bc, g[None], w_b[None], normal[None])[0]


def _diffusivity(model, t, y=None):
    """max(nu, k / (rho c_p), D_k) at temperatures ``t`` (mass fractions ``y``)."""
    tm = model.thermo
    if y is None:
        rho = th.density(tm, t)
        nu = th.viscosity(tm, t) / rho
        kappa = th.conductivity(tm, t) / (rho * th.specific_heat(tm, t))
        return jnp.maximum(nu, kappa)
    rho = th.density(tm, t, y)
    nu = th.viscosity(tm, t) / rho
    kappa = th.conductivity(tm, t) / (rho * th.specific_heat(tm, t, y))
    d = jnp.max(th.density_diffusivity(tm, t, y), axis=-1) / rho
    return jnp.maximum(jnp.maximum(nu, kappa), d)


@partial(jax.jit, static_argnums=0)
def spectral_radius(model, geom, w, beta2):
    """Sum over faces of (|v_n| + beta + 2 max(nu, kappa) / (d.n)) * length."""
    model = closed_model(model, geom, w)
    beta = jnp.sqrt(beta2)
    n = w.shape[0]
    o, nb, b = geom.f_owner, geom.f_neigh, geom.b_owner
    w_f = 0.5 * (w[o] + w[nb])
    dn_f = jnp.einsum("fk,fk->f", geom.f_delta, geom.f_normal)
    lam_f = (
        jnp.abs(jnp.einsum("fk,fk->f", w_f[:, U : V + 1], geom.f_normal))
        + beta
        + 2.0 * _diffusivity(model, w_f[:, T], _fractions(model, w_f)) / dn_f
    ) * geom.f_length
    dn_b = jnp.einsum("fk,fk->f", geom.b_delta, geom.b_normal)
    lam_b = (
        jnp.abs(jnp.einsum("fk,fk->f", w[b, U : V + 1], geom.b_normal))
        + beta
        + 2.0 * _diffusivity(model, w[b, T], _fractions(model, w[b])) / dn_b
    ) * geom.b_length
    return (
        jax.ops.segment_sum(lam_f, o, num_segments=n)
        + jax.ops.segment_sum(lam_f, nb, num_segments=n)
        + jax.ops.segment_sum(lam_b, b, num_segments=n)
    )


@partial(jax.jit, static_argnums=0)
def explicit_update(model, geom, w, r, beta2, cfl):
    """Explicit increment CFL / Lambda_i * Gamma_i^-1 R_i, shape (n_cells, 4)."""
    lam = spectral_radius(model, geom, w, beta2)
    gamma = preconditioner(closed_model(model, geom, w), w, beta2)
    du = jnp.linalg.solve(gamma, r[:, :, None])[:, :, 0]
    return du * (cfl / lam)[:, None]


def time_term_blocks(model, geom, w, beta2, cfl):
    """Pseudo-time blocks Gamma_i * Lambda_i / CFL, shape (n_cells, 4, 4)."""
    w = jnp.asarray(w)
    gamma = np.asarray(preconditioner(closed_model(model, geom, w), w, beta2))
    lam = np.asarray(spectral_radius(model, geom, w, beta2))
    return gamma * (lam / cfl)[:, None, None]


# -- unsteady time term (stage V3) ---------------------------------------------------


def _conservative_cell(model, w, area, row_scale, pressure_term):
    """Area times (rho, rho u, rho v, rho phi..., rho h - c p_th, rho Y) of cells."""
    t, y = _temperature_and_fractions(model, w)
    rho, _, h, _ = _properties(model, t, y)
    q = rho[..., None] * w
    q = q.at[..., P].set(rho)
    e = rho * h
    if pressure_term:
        e = e - model.pressure_work * model.thermo.density.pressure
    q = q.at[..., w.shape[-1] - 1 - len(model.thermo.species)].set(e)
    return area[..., None] * q * row_scale


@partial(jax.jit, static_argnums=0)
def conservative_variables(model, geom, w, row_scale):
    """Q(w): cell area times the conservative variables, (n_cells, n_var).

    (rho, rho u, rho v, rho phi..., rho h - c p_th) with rho of p_th(w) in a
    closed domain and c = ``model.pressure_work`` (the dp_th/dt work of the
    low-Mach energy equation; p_th is constant in an open domain, the term is
    left out there). The turbulence and transition columns phi are
    conservative in rho phi, as the residual rows of the variable-density
    RANS modules; ``row_scale`` (n_var,) multiplies the rows as the residual
    row scaling does (SST: nu_ref on omega). The pinned continuity row (the
    gauge-pressure level) carries no time term.
    """
    m = closed_model(model, geom, w)
    q = _conservative_cell(
        m, w, geom.area, row_scale[None, :], model.closed_mass is not None
    )
    if model.pin is not None:
        q = q.at[model.pin[0], P].set(0.0)
    return q


@partial(jax.jit, static_argnums=0)
def conservative_linearize(model, geom, w, row_scale):
    """Q(w) and its exact linearization (``jax.linearize``)."""
    return jax.linearize(lambda x: conservative_variables(model, geom, x, row_scale), w)


@partial(jax.jit, static_argnums=0)
def conservative_blocks(model, geom, w, row_scale):
    """Cell-local dQ_i/dw_i at the current p_th, (n_cells, n_var, n_var).

    The global part of a closed domain (dp_th/dT of every cell) is left out:
    these blocks serve the first-order Jacobian of the inner iterations.
    """
    m = closed_model(model, geom, w)
    closed = model.closed_mass is not None

    def cell(x, a):
        return _conservative_cell(m, x, a, row_scale, closed)

    blocks = jax.vmap(jax.jacfwd(cell))(w, geom.area)
    if model.pin is not None:
        blocks = blocks.at[model.pin[0], P, :].set(0.0)
    return blocks


# -- wall output ---------------------------------------------------------------------


def wall_traction(model, grad, w_b, normal):
    """tau.n on boundary faces (n, 2) with mu of the boundary temperature."""
    mu = th.viscosity(model.thermo, w_b[:, T])
    return -inc.viscous_flux(model, grad[:, U : V + 1], normal, mu)[:, 1:]


@partial(jax.jit, static_argnums=0)
def wall_heat_flux(model, geom, bc, w):
    """Heat flux into the fluid per boundary face (n_bfaces,), k dT/dn.

    ``n`` is the outward normal of the fluid domain (into the wall): a wall
    hotter than the fluid has q_w > 0. Zero at adiabatic walls, the
    prescribed value at heat-flux walls.
    """
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    return -boundary_viscous_flux(model, bc, g_b, w_b, geom.b_normal)[:, T]


@partial(jax.jit, static_argnums=0)
def wall_temperature(model, geom, bc, w):
    """Temperature per boundary face (n_bfaces,).

    The prescribed T at isothermal walls and fixed-T boundaries; elsewhere
    T_cell + q_w (d.n) / k (the one-sided gradient of the face heat flux).
    """
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    q = wall_heat_flux(model, geom, bc, w)
    t_c = w[geom.b_owner, T]
    dn = jnp.einsum("fk,fk->f", geom.b_delta, geom.b_normal)
    k = th.conductivity(model.thermo, t_c)
    walls = (bc.kind == WALL) & (bc.thermal != ISOTHERMAL)
    return jnp.where(walls, t_c + q * dn / k, w_b[:, T])


def _wall_species(model, geom, bc, w):
    model = closed_model(model, geom, w)
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    flux = boundary_viscous_flux(model, bc, g_b, w_b, geom.b_normal)
    return model, w_b, -flux[:, T + 1 :], -flux[:, P]


@partial(jax.jit, static_argnums=0)
def wall_species_flux(model, geom, bc, w):
    """Species mass fluxes into the fluid per boundary face, (n_bfaces, n_species).

    The diffusion flux -J_k.n plus, at Stefan-flow walls, the part m Y_k that
    the net mass flux carries; ``n`` is the outward normal of the fluid
    domain. Zero at zero-flux walls, the prescribed value at flux walls.
    Second return value: the net mass flux m into the fluid (n_bfaces,),
    non-zero at Stefan-flow walls.
    """
    _, _, j, m = _wall_species(model, geom, bc, w)
    return j, m


@partial(jax.jit, static_argnums=0)
def wall_mass_fraction(model, geom, bc, w):
    """Mass fractions per boundary face, (n_bfaces, n_species).

    The prescribed values where they are fixed; elsewhere Y_cell plus the
    one-sided gradient of the face flux, Y_cell + j_k (d.n) / (rho D_k).
    """
    model, w_b, j, _ = _wall_species(model, geom, bc, w)
    w_c = w[geom.b_owner]
    dn = jnp.einsum("fk,fk->f", geom.b_delta, geom.b_normal)
    rho_d = th.density_diffusivity(model.thermo, w_c[:, T], w_c[:, T + 1 :])
    walls = (bc.kind == WALL) & (bc.species < FIXED_FRACTION)
    extrapolated = w_c[:, T + 1 :] + j * dn[:, None] / rho_d
    return jnp.where(walls[:, None], extrapolated, w_b[:, T + 1 :])


def cell_properties(model, w, geom=None):
    """Per-cell rho, mu, k, c_p (dict of arrays) of the state ``w``.

    ``geom`` is needed in a closed domain (the density of p_th(w)).
    """
    if model.closed_mass is not None:
        model = closed_model(model, geom, jnp.asarray(w))
    t = jnp.asarray(w)[:, model.layout.column("T")]
    tm = model.thermo
    y = None
    if tm.species:
        y = jnp.asarray(w)[:, model.layout.slice("species")]
    rho, _, _, cp = _properties(model, t, y)
    return {
        "density": np.asarray(rho),
        "viscosity": np.asarray(th.viscosity(tm, t)),
        "conductivity": np.asarray(th.conductivity(tm, t)),
        "specific_heat": np.asarray(cp),
    }

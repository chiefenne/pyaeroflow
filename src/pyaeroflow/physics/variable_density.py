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

    @property
    def viscosity(self):
        """Dynamic viscosity at the reference temperature (mu(1))."""
        return float(self.thermo.viscosity.value)

    @property
    def layout(self):
        """State layout: flow block and the energy block."""
        return incompressible_layout(self.turbulence, self.transition, energy=True)

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
    face_density = np.asarray(th.density(model.thermo, jnp.asarray(t_val)))
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
    )


def jacobian_boundary_args(bc):
    """Per-face arrays of :func:`boundary_flux` (first-order Jacobian)."""
    return (bc.kind, bc.value, bc.thermal, bc.heat_flux, bc.face_density)


def lsq_include(conditions, n_var=4):
    """``lsq_include`` callback: boundary values that enter the gradients.

    As in the constant-density solver, plus T where it is known: isothermal
    walls, inlets, farfields and symmetry faces; not at adiabatic and
    heat-flux walls and pressure outlets. T is the last column (the energy
    block follows the turbulence and transition blocks, RANS stage V2).
    """
    t_col = n_var - 1

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
    return out.at[:, T].set(jnp.where(fixed_t, bc.value[:, T], out[:, T]))


def _properties(model, t):
    """rho, d(rho)/dT, h and c_p at temperatures ``t``."""
    tm = model.thermo
    return (
        th.density(tm, t),
        th.density_derivative(tm, t),
        th.enthalpy(tm, t),
        th.specific_heat(tm, t),
    )


def buoyancy_velocity_squared(model, t_cells, t_boundary):
    """U_b^2 = |g| (rho_max - rho_min) / rho_op (reference length 1)."""
    g = float(np.hypot(*model.gravity))
    if g == 0.0:
        return 0.0
    t = np.concatenate((np.ravel(t_cells), np.ravel(t_boundary)))
    rho = np.asarray(th.density(model.thermo, jnp.asarray(t)))
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
    ub2 = buoyancy_velocity_squared(model, w[:, T], value[fixed, T])
    return float(model.beta_factor * max(v2, ub2, 1e-10))


def scheme_parameters(model, w, bc):
    """Frozen per-evaluation scheme parameters: beta^2 (flow-family interface)."""
    return beta_squared(model, w, bc)


def inviscid_flux(model, w, normal):
    """F.n = (rho v_n, rho u v_n + p n_x, rho v v_n + p n_y, rho h v_n)."""
    t = w[:, T]
    rho = th.density(model.thermo, t)
    h = th.enthalpy(model.thermo, t)
    vn = w[:, U] * normal[:, 0] + w[:, V] * normal[:, 1]
    m = rho * vn
    return jnp.stack(
        (
            m,
            m * w[:, U] + w[:, P] * normal[:, 0],
            m * w[:, V] + w[:, P] * normal[:, 1],
            m * h,
        ),
        axis=1,
    )


def preconditioner(model, w, beta2):
    """Gamma for states w (n, 4): shape (n, 4, 4)."""
    rho, rho_t, h, cp = _properties(model, w[:, T])
    one = jnp.ones_like(rho)
    q = jnp.stack((one, w[:, U], w[:, V], h), axis=1)  # (1, u, v, h)
    g = jnp.zeros((w.shape[0], 4, 4))
    g = g.at[:, :, P].set(q / beta2)
    g = g.at[:, :, T].set(q * rho_t[:, None])
    g = g.at[:, T, T].add(rho * cp)
    g = g.at[:, U, U].set(rho)
    g = g.at[:, V, V].set(rho)
    return g


def abs_preconditioned_jacobian(model, w, normal, beta2):
    """|Gamma^-1 A| for states w (n, 4) and unit normals: (n, 4, 4).

    The (p, u, v) block of the constant-density solver with the local
    density, and |v_n| for T.
    """
    rho = th.density(model.thermo, w[:, T])
    flow = inc.abs_preconditioned_jacobian(
        inc.IncompressibleModel(density=1.0), w[:, :3], normal, beta2
    )
    # Scale the density in the off-diagonal entries (the 3x3 block takes rho
    # from its model): row p carries rho, column p carries 1/rho.
    flow = flow.at[:, P, U:].multiply(rho[:, None])
    flow = flow.at[:, U:, P].divide(rho[:, None])
    vn = w[:, U] * normal[:, 0] + w[:, V] * normal[:, 1]
    m = jnp.zeros((w.shape[0], 4, 4))
    m = m.at[:, :3, :3].set(flow)
    return m.at[:, T, T].set(jnp.abs(vn))


def dissipation(model, w_m, dw, normal, beta2):
    """Gamma |Gamma^-1 A| dw for mean states w_m and jumps dw, shape (n, 4)."""
    rho, rho_t, h, cp = _properties(model, w_m[:, T])
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
    return flux.at[:, T].set(jnp.where(prescribed, -bc.heat_flux, flux[:, T]))


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
    rho = th.density(model.thermo, w[:, T])
    g = jnp.asarray(model.gravity, dtype=w.dtype)
    f = jnp.asarray(model.body_force, dtype=w.dtype)
    return geom.area[:, None] * ((rho - model.density)[:, None] * g[None] + f[None])


@partial(jax.jit, static_argnums=(0, 4))
def residual(model, geom, bc, w, order, beta2, psi=None):
    """Residual R(V), shape (n_cells, 4): net outflow minus sources.

    Parameters as in :func:`pyaeroflow.physics.incompressible.residual`.
    """
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
    f_flux = f_flux + viscous_flux(model, g_f, t_f, geom.f_normal)
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
    """dR/dw of the cell sources, shape (n_cells, 4, 4): -area rho_T g in (u, v; T)."""
    rho_t = th.density_derivative(model.thermo, w[:, T])
    g = jnp.asarray(model.gravity, dtype=w.dtype)
    blocks = jnp.zeros((w.shape[0], 4, 4), dtype=w.dtype)
    blocks = blocks.at[:, U, T].set(-geom.area * rho_t * g[0])
    return blocks.at[:, V, T].set(-geom.area * rho_t * g[1])


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
    return conv + viscous_flux(model, g[None], t_f[None], normal[None])[0]


def boundary_flux(
    model, beta2, w_o, bc_kind, bc_value, bc_thermal, bc_heat, bc_rho, normal, delta
):
    """Compact first-order flux through a boundary face (for the Jacobian)."""
    bc = BoundaryData(
        bc_kind[None],
        bc_value[None],
        bc_thermal[None],
        bc_heat[None],
        bc_rho[None],
        model.density,
    )
    conv = boundary_convective_flux(model, bc, normal[None], w_o[None], beta2)[0]
    w_b = boundary_state(bc, normal[None], w_o[None])[0]
    dn = jnp.dot(delta, normal)
    g = ((w_b - w_o) / dn)[:, None] * normal[None, :]
    return conv + boundary_viscous_flux(model, bc, g[None], w_b[None], normal[None])[0]


def _diffusivity(model, t):
    """max(nu, k / (rho c_p)) at temperatures ``t``."""
    tm = model.thermo
    rho = th.density(tm, t)
    nu = th.viscosity(tm, t) / rho
    kappa = th.conductivity(tm, t) / (rho * th.specific_heat(tm, t))
    return jnp.maximum(nu, kappa)


@partial(jax.jit, static_argnums=0)
def spectral_radius(model, geom, w, beta2):
    """Sum over faces of (|v_n| + beta + 2 max(nu, kappa) / (d.n)) * length."""
    beta = jnp.sqrt(beta2)
    n = w.shape[0]
    o, nb, b = geom.f_owner, geom.f_neigh, geom.b_owner
    w_f = 0.5 * (w[o] + w[nb])
    dn_f = jnp.einsum("fk,fk->f", geom.f_delta, geom.f_normal)
    lam_f = (
        jnp.abs(jnp.einsum("fk,fk->f", w_f[:, U : V + 1], geom.f_normal))
        + beta
        + 2.0 * _diffusivity(model, w_f[:, T]) / dn_f
    ) * geom.f_length
    dn_b = jnp.einsum("fk,fk->f", geom.b_delta, geom.b_normal)
    lam_b = (
        jnp.abs(jnp.einsum("fk,fk->f", w[b, U : V + 1], geom.b_normal))
        + beta
        + 2.0 * _diffusivity(model, w[b, T]) / dn_b
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
    du = jnp.linalg.solve(preconditioner(model, w, beta2), r[:, :, None])[:, :, 0]
    return du * (cfl / lam)[:, None]


def time_term_blocks(model, geom, w, beta2, cfl):
    """Pseudo-time blocks Gamma_i * Lambda_i / CFL, shape (n_cells, 4, 4)."""
    w = jnp.asarray(w)
    gamma = np.asarray(preconditioner(model, w, beta2))
    lam = np.asarray(spectral_radius(model, geom, w, beta2))
    return gamma * (lam / cfl)[:, None, None]


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


def cell_properties(model, w):
    """Per-cell rho, mu, k, c_p (dict of arrays) of the state ``w``."""
    t = jnp.asarray(w)[:, model.layout.column("T")]
    tm = model.thermo
    return {
        "density": np.asarray(th.density(tm, t)),
        "viscosity": np.asarray(th.viscosity(tm, t)),
        "conductivity": np.asarray(th.conductivity(tm, t)),
        "specific_heat": np.asarray(th.specific_heat(tm, t)),
    }

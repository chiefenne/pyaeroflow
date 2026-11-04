"""Compressible Euler and laminar Navier-Stokes equations (L1).

Density-based.

Working variables per cell: V = (p, u, v, T), the primitive variables that
are reconstructed (MUSCL), with the density from the ideal-gas law
rho = p / (R T). The residual R(V) is the net outflow of the conservative
fluxes

    F.n = (rho v_n, rho u v_n + p n_x, rho v v_n + p n_y, rho H v_n),

with v_n = v.n, the total energy rho E = p / (gamma - 1) + rho |v|^2 / 2 and
the total enthalpy H = E + p / rho. The rows of R are the continuity, x- and
y-momentum and energy equations; the unknowns are V. Pseudo-time stepping
uses the conservative time derivative,

    (dU/dV) dV/dtau + R(V) = 0,

so that the pseudo-time blocks are dU/dV Lambda_i / CFL with the spectral
radius Lambda_i = sum over faces of (|v_n| + c) L.

Non-dimensionalisation: rho, |v| and T
of the freestream are 1, the reference length is the chord. Then
p_inf = 1 / (gamma M_inf^2) and the non-dimensional gas constant is
R = p_inf / (rho_inf T_inf) = 1 / (gamma M_inf^2) (see
:func:`gas_constant_for_mach`).

Convective flux (D21 for this solver): Roe's approximate Riemann solver
(Roe 1981) with Harten's entropy fix in its smooth form,
|lambda| -> (lambda^2 + delta^2) / (2 delta) for |lambda| < delta,
delta = ``entropy_fix`` (|v_n| + c) of the Roe-averaged state. Second order
by MUSCL reconstruction of V with the Venkatakrishnan(-Wang) limiter. A face
whose reconstructed state has p <= 0 or T <= 0 falls back to first order
(non-physical reconstruction check).

Viscous flux (C2): the residual subtracts

    F_v.n = (0, tau.n, (tau.v).n + k grad T.n),

tau = mu (grad v + grad v^T - 2/3 div v I) (Stokes), with the dynamic
viscosity mu(T) of Sutherland's law
or a constant, and the conductivity k = mu c_p / Pr (constant Prandtl
number, Pr = 0.72), c_p = gamma R / (gamma - 1). Face
gradients of (p, u, v, T) as in the incompressible solver (D24); mu, k and
v at the face from the mean of the two cell states.

No-slip walls (``wall`` in a laminar case): zero velocity, the pressure of
the interior cell, and either no heat flux (adiabatic, the default) or a
prescribed wall temperature (isothermal), whose heat flux follows from the one-sided
normal temperature gradient. Slip walls and symmetry faces carry no viscous
flux.

Boundary conditions: farfield (freestream state outside, Roe flux between the
interior and that state: the upwinding picks the incoming characteristics),
slip wall and symmetry (pressure flux only), static-pressure outlet
(prescribed p, entropy and
velocity from the interior; supersonic outflow extrapolates everything),
total-conditions inlet (p_0, T_0 and flow direction; p from the interior)
and velocity inlet (u, v, T prescribed, p from the interior).
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
from pyaeroflow.physics.incompressible import (
    FARFIELD,
    KIND_NAMES,
    PRESSURE_OUTLET,
    SYMMETRY,
    TOTAL_PRESSURE_INLET,
    VELOCITY_INLET,
    WALL,
)
from pyaeroflow.solver import divergence

FLOW = "compressible"
P, U, V, T = 0, 1, 2, 3
# Columns that must stay positive (step limiter, structural Jacobian samples).
POSITIVE_COLUMNS = (P, T)
# Names of the residual rows (log and history): the conservative rows.
RESIDUAL_NAMES = ("rho", "rhou", "rhov", "rhoE")

__all__ = [
    "FARFIELD",
    "KIND_NAMES",
    "PRESSURE_OUTLET",
    "SYMMETRY",
    "TOTAL_PRESSURE_INLET",
    "VELOCITY_INLET",
    "WALL",
]


class CompressibleModel(NamedTuple):
    """Model parameters (hashable; a static jit argument).

    Attributes
    ----------
    gamma : float
        Ratio of specific heats.
    gas_constant : float
        Non-dimensional gas constant R = p_inf / (rho_inf T_inf); with the
        freestream as reference 1 / (gamma M_inf^2).
    viscosity : float
        Dynamic viscosity at the reference temperature T = 1 (1 / Re of the
        reference state); 0 = Euler.
    viscosity_law : {"sutherland", "constant"}
    sutherland_temperature : float
        Sutherland's constant S / T_ref (110.4 K / T_ref for air).
    prandtl : float
        Laminar Prandtl number.
    spatial_order : int
        1 or 2 (MUSCL reconstruction of p, u, v, T).
    limiter : {"none", "venkatakrishnan-wang", "venkatakrishnan"}
    limiter_threshold : float
    entropy_fix : float
        Harten's coefficient: delta = entropy_fix (|v_n| + c); 0 = off.
    low_mach_correction : bool
        Rieper's low-Mach correction of the Roe flux on interior faces
        (Rieper 2011): the face velocities are blended towards their
        mean by z = min(max(M_L, M_R), 1), which scales the velocity-jump
        dissipation with the local Mach number.
    viscous_gradient : str
        Face-gradient scheme of the viscous flux and the wall loads.
    alpha : float
    density : float
        Reference density of the force coefficients (freestream, 1).
    turbulence : {"none", "sa-neg", "sst"}
        Turbulence model: "sa-neg" (C3) adds nu_tilde as fifth variable
        (:mod:`pyaeroflow.physics.rans_compressible`), "sst" (C4) k and omega
        as fifth and sixth (:mod:`pyaeroflow.physics.rans_sst_compressible`).
    turbulence_spatial_order : int
        1 or 2: convective states of the turbulence equation.
    prandtl_turbulent : float
        Turbulent Prandtl number of the eddy heat flux.
    pin, transition, coupling :
        Interface of the problem class: no pin, no transition model, the
        mean-flow/turbulence coupling of the first-order Jacobian.
    """

    gamma: float = 1.4
    gas_constant: float = 1.0 / 1.4
    viscosity: float = 0.0
    viscosity_law: str = "sutherland"
    sutherland_temperature: float = 110.4 / 288.15
    prandtl: float = 0.72
    spatial_order: int = 2
    limiter: str = "venkatakrishnan-wang"
    limiter_threshold: float = 0.05
    entropy_fix: float = 0.001
    low_mach_correction: bool = False
    viscous_gradient: str = "alpha-damping"
    alpha: float = 4.0 / 3.0
    density: float = 1.0
    pin: tuple = None
    turbulence: str = "none"
    turbulence_spatial_order: int = 1
    prandtl_turbulent: float = 0.9
    transition: str = "none"
    coupling: str = "segregated"

    @property
    def n_var(self):
        """Number of variables: p, u, v, T (and nu_tilde, or k and omega)."""
        return {"sa-neg": 5, "sst": 6}.get(self.turbulence, 4)


def gas_constant_for_mach(mach, gamma=1.4):
    """Non-dimensional R = 1 / (gamma M^2) of the freestream reference."""
    return 1.0 / (gamma * mach**2)


class BoundaryData(NamedTuple):
    """Per-boundary-face condition (a JAX pytree).

    Attributes
    ----------
    kind : int array, shape (n_bfaces,)
    value : float array, shape (n_bfaces, 4)
        Prescribed (p, u, v, T) where the kind uses them; the total-conditions
        inlet stores (p_0, d_x, d_y, T_0) with the unit flow direction d.
    gamma, gas_constant : float
    nu_tilde : float array, shape (n_bfaces,) or None
        Prescribed SA working variable at inflow faces (zero elsewhere).
    turbulence : float array, shape (n_bfaces, 2) or None
        SST (C4): prescribed (k, omega) at inflow faces; at walls (0,
        60 / (beta_1 d1^2)), the wall omega without its factor nu_w (set by
        the driver, see :mod:`pyaeroflow.physics.rans_sst_compressible`).
    viscosity : tuple of float or None
        (mu at T = 1, Sutherland's S / T_ref, 1 for Sutherland's law and 0
        for a constant viscosity): mu(T) of boundary states (the SST wall
        omega).
    """

    kind: jnp.ndarray
    value: jnp.ndarray
    gamma: float = 1.4
    gas_constant: float = 1.0 / 1.4
    nu_tilde: jnp.ndarray = None
    turbulence: jnp.ndarray = None
    viscosity: tuple = None


def _evaluate(spec, x, y):
    return np.asarray(spec(x, y) if callable(spec) else spec, dtype=float)


def build_boundary_data(disc, conditions, model):
    """Return :func:`boundary_data` (flow-family interface)."""
    return boundary_data(disc, conditions, model)


def boundary_data(disc, conditions, model):
    """Build :class:`BoundaryData` from a dict of boundary conditions.

    Parameters
    ----------
    disc : Discretization
    conditions : dict of str to dict
        Non-dimensional values: ``farfield`` takes ``velocity`` (u, v),
        ``pressure`` and ``temperature``; ``velocity-inlet`` ``velocity``
        and ``temperature``; ``pressure-outlet`` ``pressure``;
        ``total-pressure-inlet`` ``total_pressure``, ``total_temperature``
        and ``flow_direction``; ``wall`` an optional ``temperature``
        (isothermal; stored in the T column, 0 = adiabatic);
        ``slip-wall`` and ``symmetry`` nothing; inflow kinds an optional
        ``nu_tilde`` (SA-neg). Constants or callables of the boundary-face
        centres.
    model : CompressibleModel
    """
    nb = disc.boundary_face_ids.size
    kind = np.zeros(nb, dtype=np.int64)
    value = np.zeros((nb, 4))
    nu_tilde = np.zeros(nb)
    turbulence = np.zeros((nb, 2))
    missing = set(disc.boundary_slices) - set(conditions)
    extra = set(conditions) - set(disc.boundary_slices)
    if missing or extra:
        raise ValueError(
            f"boundary conditions missing for {sorted(missing)}, "
            f"unknown boundaries {sorted(extra)}"
        )
    centres = np.asarray(disc.geom.b_centre)
    for name, sl in disc.boundary_slices.items():
        bc = conditions[name]
        if bc["type"] == "transpiration-wall":
            raise ValueError("transpiration-wall: not available for compressible flow")
        kind[sl] = KIND_NAMES[bc["type"]]
        x, y = centres[sl, 0], centres[sl, 1]
        if bc["type"] == "total-pressure-inlet":
            d = np.asarray(_evaluate(bc["flow_direction"], x, y), dtype=float)
            d = np.broadcast_to(
                d.reshape(2, -1) if d.ndim > 1 else d[:, None], (2, x.size)
            )
            norm = np.hypot(d[0], d[1])
            cols = (
                bc["total_pressure"],
                d[0] / norm,
                d[1] / norm,
                bc["total_temperature"],
            )
        else:
            vel = bc.get("velocity", (0.0, 0.0))
            if callable(vel):
                vel = vel(x, y)
            cols = (
                bc.get("pressure", 0.0),
                vel[0],
                vel[1],
                bc.get("temperature", 0.0),
            )
        for col, spec in enumerate(cols):
            value[sl, col] = np.broadcast_to(_evaluate(spec, x, y), x.shape)
        nu_tilde[sl] = np.broadcast_to(
            _evaluate(bc.get("nu_tilde", 0.0), x, y), x.shape
        )
        for col, key in enumerate(("k", "omega")):
            turbulence[sl, col] = np.broadcast_to(
                _evaluate(bc.get(key, 0.0), x, y), x.shape
            )
    return BoundaryData(
        jnp.asarray(kind),
        jnp.asarray(value),
        float(model.gamma),
        float(model.gas_constant),
        jnp.asarray(nu_tilde),
        jnp.asarray(turbulence),
        viscosity_parameters(model),
    )


def viscosity_parameters(model):
    """(mu_ref, S, Sutherland flag) of :attr:`BoundaryData.viscosity`."""
    return (
        float(model.viscosity),
        float(model.sutherland_temperature),
        1.0 if model.viscosity_law == "sutherland" else 0.0,
    )


def boundary_viscosity(params, t):
    """mu(T) from :func:`viscosity_parameters` (traced inside BoundaryData)."""
    mu_ref, s, sutherland = params
    t = jnp.maximum(t, 1e-12)
    law = t**1.5 * (1.0 + s) / (t + s)
    return mu_ref * jnp.where(sutherland > 0.5, law, jnp.ones_like(t))


def lsq_include(conditions, n_var=4):
    """``lsq_include`` callback: boundary values enter a gradient where known.

    Farfield and symmetry faces: all variables. Walls: the velocity, and T
    at isothermal walls (the adiabatic wall leaves p and T to the
    interior). Velocity inlets: u, v, T. Total-conditions inlets: u, v, T
    (computed from p_0, T_0). Outlets: p.
    """

    def include(slices, n_bfaces):
        mask = np.ones((n_bfaces, n_var), dtype=bool)
        for name, sl in slices.items():
            k = KIND_NAMES[conditions[name]["type"]]
            if k == WALL:
                mask[sl, P] = False
                if not conditions[name].get("temperature"):
                    mask[sl, T] = False
            elif k in (VELOCITY_INLET, TOTAL_PRESSURE_INLET):
                mask[sl, P] = False
            elif k == PRESSURE_OUTLET:
                mask[sl, U:] = False
        return mask

    return include


# -- thermodynamics --------------------------------------------------------------


def density(gas_constant, w):
    """Return the density p / (R T) of states w (..., 4)."""
    return w[..., P] / (gas_constant * w[..., T])


def sound_speed(gamma, gas_constant, w):
    """Return the speed of sound sqrt(gamma R T)."""
    return jnp.sqrt(gamma * gas_constant * w[..., T])


def conservative(gamma, gas_constant, w):
    """U = (rho, rho u, rho v, rho E) of primitive states w (n, 4)."""
    rho = density(gas_constant, w)
    q2 = w[:, U] ** 2 + w[:, V] ** 2
    return jnp.stack(
        (
            rho,
            rho * w[:, U],
            rho * w[:, V],
            w[:, P] / (gamma - 1.0) + 0.5 * rho * q2,
        ),
        axis=1,
    )


def mach_number(gamma, gas_constant, w):
    """|v| / c per state."""
    return jnp.hypot(w[..., U], w[..., V]) / sound_speed(gamma, gas_constant, w)


def physical_flux(gamma, gas_constant, w, normal):
    """F.n for states w (n, 4) and unit normals (n, 2)."""
    rho = density(gas_constant, w)
    vn = w[:, U] * normal[:, 0] + w[:, V] * normal[:, 1]
    h = gamma / (gamma - 1.0) * w[:, P] / rho + 0.5 * (w[:, U] ** 2 + w[:, V] ** 2)
    return jnp.stack(
        (
            rho * vn,
            rho * w[:, U] * vn + w[:, P] * normal[:, 0],
            rho * w[:, V] * vn + w[:, P] * normal[:, 1],
            rho * h * vn,
        ),
        axis=1,
    )


def _harten(lam, delta):
    """|lam| with Harten's smooth entropy fix below ``delta``."""
    a = jnp.abs(lam)
    return jnp.where(a < delta, 0.5 * (lam**2 + delta**2) / delta, a)


def roe_flux(gamma, gas_constant, entropy_fix, w_l, w_r, normal):
    """Roe flux per unit length between primitive states (n, 4).

    F = (F_L + F_R) / 2 - sum_k |lambda_k| alpha_k r_k / 2 with the
    Roe-averaged state (Blazek, Computational Fluid Dynamics, 3rd ed.,
    sec. 4.3.3).
    """
    nx, ny = normal[:, 0], normal[:, 1]
    r_l, r_r = density(gas_constant, w_l), density(gas_constant, w_r)
    g1 = gamma / (gamma - 1.0)
    h_l = g1 * w_l[:, P] / r_l + 0.5 * (w_l[:, U] ** 2 + w_l[:, V] ** 2)
    h_r = g1 * w_r[:, P] / r_r + 0.5 * (w_r[:, U] ** 2 + w_r[:, V] ** 2)
    s_l, s_r = jnp.sqrt(r_l), jnp.sqrt(r_r)
    wt = 1.0 / (s_l + s_r)
    u = (s_l * w_l[:, U] + s_r * w_r[:, U]) * wt
    v = (s_l * w_l[:, V] + s_r * w_r[:, V]) * wt
    h = (s_l * h_l + s_r * h_r) * wt
    rho = s_l * s_r
    q2 = u**2 + v**2
    c2 = jnp.maximum((gamma - 1.0) * (h - 0.5 * q2), 1e-12)
    c = jnp.sqrt(c2)
    vn = u * nx + v * ny

    dp = w_r[:, P] - w_l[:, P]
    dr = r_r - r_l
    du = w_r[:, U] - w_l[:, U]
    dv = w_r[:, V] - w_l[:, V]
    dvn = du * nx + dv * ny

    if entropy_fix > 0.0:
        delta = entropy_fix * (jnp.abs(vn) + c)
        l1, l2, l3 = (
            _harten(vn - c, delta),
            _harten(vn, delta),
            _harten(vn + c, delta),
        )
    else:
        l1, l2, l3 = jnp.abs(vn - c), jnp.abs(vn), jnp.abs(vn + c)

    a1 = l1 * (dp - rho * c * dvn) / (2.0 * c2)
    a3 = l3 * (dp + rho * c * dvn) / (2.0 * c2)
    a2 = l2 * (dr - dp / c2)
    a4 = l2 * rho
    ones = jnp.ones_like(u)
    zeros = jnp.zeros_like(u)
    diss = (
        a1[:, None] * jnp.stack((ones, u - c * nx, v - c * ny, h - c * vn), axis=1)
        + a2[:, None] * jnp.stack((ones, u, v, 0.5 * q2), axis=1)
        + a4[:, None]
        * jnp.stack(
            (zeros, du - dvn * nx, dv - dvn * ny, u * du + v * dv - vn * dvn), axis=1
        )
        + a3[:, None] * jnp.stack((ones, u + c * nx, v + c * ny, h + c * vn), axis=1)
    )
    f_l = physical_flux(gamma, gas_constant, w_l, normal)
    f_r = physical_flux(gamma, gas_constant, w_r, normal)
    return 0.5 * (f_l + f_r - diss)


def _flux(model, w_l, w_r, normal):
    return roe_flux(
        model.gamma, model.gas_constant, model.entropy_fix, w_l, w_r, normal
    )


def low_mach_states(gamma, gas_constant, w_l, w_r):
    """Rieper's low-Mach correction of the face states (Rieper 2011).

    u_L* = (1 + z)/2 u_L + (1 - z)/2 u_R and u_R* likewise, with
    z = min(max(M_L, M_R), 1) (Rieper, J. Comput. Phys. 230, 2011): the
    velocity jump that drives the Roe dissipation is scaled by z, which
    removes the O(1/M) excess dissipation of the pressure field at low Mach
    numbers. p and T are unchanged.
    """
    c2 = gamma * gas_constant

    def mach2(w):
        return (w[:, U] ** 2 + w[:, V] ** 2) / (c2 * w[:, T])

    z = jnp.minimum(jnp.sqrt(jnp.maximum(mach2(w_l), mach2(w_r)) + 1e-30), 1.0)
    a, b = (0.5 * (1.0 + z))[:, None], (0.5 * (1.0 - z))[:, None]
    v_l, v_r = w_l[:, U:T], w_r[:, U:T]
    return (
        w_l.at[:, U:T].set(a * v_l + b * v_r),
        w_r.at[:, U:T].set(a * v_r + b * v_l),
    )


def interior_convective_flux(model, w_l, w_r, normal):
    """Roe flux through interior faces, with the low-Mach correction if set."""
    if model.low_mach_correction:
        w_l, w_r = low_mach_states(model.gamma, model.gas_constant, w_l, w_r)
    return _flux(model, w_l, w_r, normal)


def jacobian_convective_flux(model, w_l, w_r, normal):
    """Convective flux of the compact first-order Jacobian: plain Roe.

    The low-Mach correction stays out of the Jacobian (the residual and the
    Newton-Krylov products keep it): its velocity blending removes the
    dissipation of the velocity jump where M -> 0, so that at a stagnation
    point the first-order Jacobian plus the pseudo-time term becomes
    singular from CFL ~ 300 on (NACA 0012 at M 0.15, alpha 0, 449 x 129:
    50 perturbed PARDISO pivots at CFL 1000, solutions of 1e26).
    """
    return _flux(model, w_l, w_r, normal)


# -- boundary conditions ---------------------------------------------------------


def boundary_state(bc, normal, w_in):
    """Boundary-face state (p, u, v, T) from the interior state ``w_in``.

    Farfield: the freestream. Slip wall / symmetry: interior p and T, the
    interior velocity without its normal component. Wall: interior p, the
    wall velocity, the wall temperature if one is prescribed (T column > 0,
    isothermal) and the interior T otherwise (adiabatic). Velocity inlet:
    interior p, prescribed u, v, T. Pressure outlet: prescribed p, interior
    velocity, T on the interior isentrope (T_in (p_b / p_in)^((gamma - 1) /
    gamma)); a supersonic outflow (v_n >= c) takes the interior state.
    Total-conditions inlet: p = min(p_in, p_0), the Mach number from
    p_0 / p, T = T_0 / (1 + (gamma - 1) M^2 / 2) and the velocity along the
    prescribed direction.
    """
    gamma, r_gas = bc.gamma, bc.gas_constant
    vn = w_in[:, U] * normal[:, 0] + w_in[:, V] * normal[:, 1]
    mirrored = w_in.at[:, U].add(-vn * normal[:, 0]).at[:, V].add(-vn * normal[:, 1])
    kind = bc.kind[:, None]
    val = bc.value
    out = jnp.where(kind == SYMMETRY, mirrored, w_in)
    out = jnp.where(kind == FARFIELD, val, out)
    wall = w_in.at[:, U].set(val[:, U]).at[:, V].set(val[:, V])
    t_wall = jnp.where(val[:, T] > 0.0, val[:, T], w_in[:, T])
    out = jnp.where(kind == WALL, wall.at[:, T].set(t_wall), out)
    inlet_v = wall.at[:, T].set(val[:, T])
    out = jnp.where(kind == VELOCITY_INLET, inlet_v, out)

    # Static-pressure outlet (subsonic): p prescribed, isentropic T.
    p_in = w_in[:, P]
    ratio = jnp.maximum(val[:, P], 1e-300) / p_in
    t_out = w_in[:, T] * ratio ** ((gamma - 1.0) / gamma)
    outlet = w_in.at[:, P].set(val[:, P]).at[:, T].set(t_out)
    supersonic = vn >= sound_speed(gamma, r_gas, w_in)
    outlet = jnp.where(supersonic[:, None], w_in, outlet)
    out = jnp.where(kind == PRESSURE_OUTLET, outlet, out)

    # Total-conditions inlet.
    p0, t0 = val[:, P], val[:, T]
    p_b = jnp.minimum(p_in, p0)
    m2 = 2.0 / (gamma - 1.0) * ((p0 / p_b) ** ((gamma - 1.0) / gamma) - 1.0)
    t_b = t0 / (1.0 + 0.5 * (gamma - 1.0) * m2)
    speed = jnp.sqrt(m2 * gamma * r_gas * t_b + 1e-300)
    inlet_t = jnp.stack((p_b, speed * val[:, U], speed * val[:, V], t_b), axis=1)
    return jnp.where(kind == TOTAL_PRESSURE_INLET, inlet_t, out)


def boundary_convective_flux(model, bc, normal, w_in):
    """Convective flux through boundary faces from the interior state w_in.

    Walls and symmetry faces carry the pressure flux of the interior
    pressure (zero mass and energy flux); all other kinds the Roe flux
    between the interior and the boundary state.
    """
    w_out = boundary_state(bc, normal, w_in)
    upwind = _flux(model, w_in, w_out, normal)
    p = w_in[:, P]
    zero = jnp.zeros_like(p)
    wall_flux = jnp.stack((zero, p * normal[:, 0], p * normal[:, 1], zero), axis=1)
    kind = bc.kind[:, None]
    return jnp.where((kind == WALL) | (kind == SYMMETRY), wall_flux, upwind)


# -- viscous flux (C2) ---------------------------------------------------------------


def laminar_viscosity(model, t):
    """Dynamic viscosity mu(T), non-dimensional (mu(1) = ``model.viscosity``).

    Sutherland: mu = mu_ref T^(3/2) (1 + S) / (T + S) with T and S scaled by
    the reference temperature.
    """
    if model.viscosity_law == "constant":
        return model.viscosity * jnp.ones_like(t)
    if model.viscosity_law != "sutherland":
        raise ValueError(f"unknown viscosity law '{model.viscosity_law}'")
    s = model.sutherland_temperature
    t = jnp.maximum(t, 1e-12)
    return model.viscosity * t**1.5 * (1.0 + s) / (t + s)


def heat_capacity(model):
    """c_p = gamma R / (gamma - 1), non-dimensional."""
    return model.gamma * model.gas_constant / (model.gamma - 1.0)


def stress(mu, grad):
    """Viscous stress tau, shape (n, 2, 2).

    For dynamic viscosities mu (n,) and face gradients grad (n, 4, 2).
    """
    g = grad[:, U:T, :]
    div = g[:, 0, 0] + g[:, 1, 1]
    tau = mu[:, None, None] * (g + jnp.swapaxes(g, 1, 2))
    return tau - (2.0 / 3.0 * mu * div)[:, None, None] * jnp.eye(2)[None]


def viscous_flux(model, grad, w_f, normal, heat=True, mu_t=None):
    """-F_v.n per unit length, shape (n, 4) (added to the convective flux).

    With an eddy viscosity ``mu_t`` (n,) (RANS, C3) the stress uses
    mu + mu_t and the conductivity c_p (mu / Pr + mu_t / Pr_t) (Boussinesq,
    without the 2/3 rho k term, as usual for SA).

    Parameters
    ----------
    grad : (n, 4, 2)
        Face gradients of (p, u, v, T).
    w_f : (n, 4)
        Face states: mu(T), k and the velocity of the work term.
    normal : (n, 2)
    heat : bool or bool array (n,)
        False drops the heat flux (adiabatic walls).
    mu_t : (n,), optional
        Dynamic eddy viscosity at the faces.
    """
    mu = laminar_viscosity(model, w_f[:, T])
    k_over_cp = mu / model.prandtl
    if mu_t is not None:
        k_over_cp = k_over_cp + mu_t / model.prandtl_turbulent
        mu = mu + mu_t
    tn = jnp.einsum("nij,nj->ni", stress(mu, grad), normal)
    work = w_f[:, U] * tn[:, 0] + w_f[:, V] * tn[:, 1]
    k = k_over_cp * heat_capacity(model)
    q = k * jnp.einsum("nk,nk->n", grad[:, T, :], normal)
    q = jnp.where(heat, q, 0.0)
    return -jnp.stack((jnp.zeros_like(mu), tn[:, 0], tn[:, 1], work + q), axis=1)


def boundary_viscous_flux(model, bc, grad, w_b, normal, mu_t=None):
    """-F_v.n through boundary faces (n_bfaces, 4).

    Walls: stress of the face gradient, heat flux only where the wall is
    isothermal. Slip walls and symmetry: none. Other kinds: the full flux
    with the boundary state. ``mu_t``: the eddy viscosity at the faces (RANS).
    """
    kind = bc.kind
    adiabatic = (kind == WALL) & (bc.value[:, T] <= 0.0)
    flux = viscous_flux(model, grad, w_b, normal, heat=~adiabatic, mu_t=mu_t)
    return jnp.where((kind == SYMMETRY)[:, None], 0.0, flux)


def wall_traction(model, grad, w_b, normal):
    """tau.n on boundary faces (n, 2), for the wall loads (forces.py)."""
    mu = laminar_viscosity(model, w_b[:, T])
    return jnp.einsum("nij,nj->ni", stress(mu, grad), normal)


@partial(jax.jit, static_argnums=0)
def wall_heat_flux(model, geom, bc, w):
    """Heat flux q_w = -k dT/dn into the fluid per boundary face (n_bfaces,).

    ``n`` is the outward normal of the fluid domain (into the wall): a wall
    hotter than the fluid has q_w > 0. Zero at adiabatic walls.
    """
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    return -boundary_viscous_flux(model, bc, g_b, w_b, geom.b_normal)[:, 3]


# -- residual ---------------------------------------------------------------------


def _physical(w):
    """Return True where p > 0 and T > 0."""
    return (w[:, P] > 0.0) & (w[:, T] > 0.0)


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


def face_states(model, geom, bc, w, order, psi=None, w_b=None, grad=None):
    """Convective face states (left, right, boundary-inner) of order 1 or 2.

    Second order: MUSCL with the limiter (or the frozen ``psi``); faces with
    a non-physical reconstructed state (p <= 0 or T <= 0 on either side) use
    the cell values. ``w_b`` and ``grad``: the first-order boundary states
    and the cell gradients, if already computed.
    """
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    if order != 2:
        return w[o], w[n], w[b]
    if w_b is None:
        w_b = boundary_state(bc, geom.b_normal, w[b])
    if grad is None:
        grad = lsq_gradient(geom, w, w_b)
    if psi is None:
        psi = _limiter(model, geom, w, w_b, grad)
    w_l, w_r, w_in = reconstruct(geom, w, grad, psi)
    ok = (_physical(w_l) & _physical(w_r))[:, None]
    w_l = jnp.where(ok, w_l, w[o])
    w_r = jnp.where(ok, w_r, w[n])
    w_in = jnp.where(_physical(w_in)[:, None], w_in, w[b])
    return w_l, w_r, w_in


@partial(jax.jit, static_argnums=(0, 4))
def residual(model, geom, bc, w, order, params=None, psi=None):
    """Residual R(V), shape (n_cells, 4): net outflow of mass, momentum, energy.

    Convective minus viscous flux (the latter if ``model.viscosity`` > 0).
    ``params`` (the scheme parameters of the problem interface) is unused.
    """
    viscous = model.viscosity > 0.0
    w_b = grad = None
    if viscous or order == 2:
        w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
        grad = lsq_gradient(geom, w, w_b)
    w_l, w_r, w_in = face_states(model, geom, bc, w, order, psi, w_b, grad)
    f_flux = interior_convective_flux(model, w_l, w_r, geom.f_normal)
    b_flux = boundary_convective_flux(model, bc, geom.b_normal, w_in)
    if viscous:
        o, n = geom.f_owner, geom.f_neigh
        g_f = interior_face_gradient(geom, w, grad, model.viscous_gradient, model.alpha)
        g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
        f_flux = f_flux + viscous_flux(model, g_f, 0.5 * (w[o] + w[n]), geom.f_normal)
        b_flux = b_flux + boundary_viscous_flux(model, bc, g_b, w_b, geom.b_normal)
    return divergence(geom, f_flux, b_flux)


@partial(jax.jit, static_argnums=(0, 5))
def residual_jvp(model, geom, bc, w, dw, order, params=None, psi=None):
    """Exact Jacobian-vector product dR/dw(w) @ dw by forward-mode AD."""

    def f(x):
        return residual(model, geom, bc, x, order, params, psi)

    return jax.jvp(f, (w,), (dw,))[1]


@partial(jax.jit, static_argnums=(0, 4))
def residual_linearize(model, geom, bc, w, order, params=None, psi=None):
    """Residual and its linearization at ``w`` (``jax.linearize``)."""

    def f(x):
        return residual(model, geom, bc, x, order, params, psi)

    return jax.linearize(f, w)


@partial(jax.jit, static_argnums=(0, 4))
def non_physical_faces(model, geom, bc, w, order=2, psi=None):
    """Count the faces whose reconstruction falls back to first order."""
    if order != 2:
        return 0
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    if psi is None:
        psi = _limiter(model, geom, w, w_b, grad)
    w_l, w_r, w_in = reconstruct(geom, w, grad, psi)
    bad = ~(_physical(w_l) & _physical(w_r))
    return jnp.sum(bad) + jnp.sum(~_physical(w_in))


# -- first-order Jacobian (compact fluxes) --------------------------------------


def interior_flux(model, params, w_o, w_n, normal, delta):
    """Compact first-order flux through an interior face (for the Jacobian).

    Viscous part: the two-point face gradient (the normal derivative of the
    face-gradient scheme, as the incompressible solver).
    """
    conv = jacobian_convective_flux(model, w_o[None], w_n[None], normal[None])[0]
    if model.viscosity > 0.0:
        dn = jnp.dot(delta, normal)
        if model.viscous_gradient == "alpha-damping":
            coeff = model.alpha / dn
        else:
            coeff = dn / jnp.dot(delta, delta)
        g = (coeff * (w_n - w_o))[:, None] * normal[None, :]
        w_f = 0.5 * (w_o + w_n)
        conv = conv + viscous_flux(model, g[None], w_f[None], normal[None])[0]
    return conv


def boundary_flux(model, params, w_o, bc_kind, bc_value, normal, delta):
    """Compact first-order flux through a boundary face (for the Jacobian)."""
    bc = BoundaryData(bc_kind[None], bc_value[None], model.gamma, model.gas_constant)
    conv = boundary_convective_flux(model, bc, normal[None], w_o[None])[0]
    if model.viscosity > 0.0:
        w_b = boundary_state(bc, normal[None], w_o[None])[0]
        dn = jnp.dot(delta, normal)
        g = ((w_b - w_o) / dn)[:, None] * normal[None, :]
        visc = boundary_viscous_flux(model, bc, g[None], w_b[None], normal[None])
        conv = conv + visc[0]
    return conv


# -- pseudo time --------------------------------------------------------------------


def scheme_parameters(model, w, bc):
    """Frozen per-evaluation scheme parameters: none for this solver."""
    return 0.0


def face_eigenvalue(model, w_f, normal, delta, mu_t=None):
    """|v_n| + c + lambda_v per face for face states w_f (n, >= 4).

    Viscous part: lambda_v = (4/3 (mu + mu_t) + gamma (mu / Pr + mu_t / Pr_t))
    / (rho |d.n|), the two viscous eigenvalue terms over the face distance;
    ``mu_t`` (n,) is the eddy viscosity (RANS), None = laminar.
    """
    c = sound_speed(model.gamma, model.gas_constant, w_f)
    out = jnp.abs(jnp.einsum("fk,fk->f", w_f[:, U:T], normal)) + c
    if model.viscosity > 0.0:
        mu = laminar_viscosity(model, w_f[:, T])
        rho = density(model.gas_constant, w_f)
        dn = jnp.abs(jnp.einsum("fk,fk->f", delta, normal))
        visc = (4.0 / 3.0 + model.gamma / model.prandtl) * mu
        if mu_t is not None:
            visc = visc + (4.0 / 3.0 + model.gamma / model.prandtl_turbulent) * mu_t
        out = out + visc / (rho * dn)
    return out


def sum_face_eigenvalues(geom, lam_f, lam_b, n):
    """Sum of face eigenvalues times face length per cell."""
    lam_f = lam_f * geom.f_length
    lam_b = lam_b * geom.b_length
    return (
        jax.ops.segment_sum(lam_f, geom.f_owner, num_segments=n)
        + jax.ops.segment_sum(lam_f, geom.f_neigh, num_segments=n)
        + jax.ops.segment_sum(lam_b, geom.b_owner, num_segments=n)
    )


@partial(jax.jit, static_argnums=0)
def spectral_radius(model, geom, w, params=None):
    """Sum over faces of (|v_n| + c + lambda_v) * length, per cell.

    See :func:`face_eigenvalue`; face states are the mean of the two cells.
    """
    o, nb, b = geom.f_owner, geom.f_neigh, geom.b_owner
    lam_f = face_eigenvalue(model, 0.5 * (w[o] + w[nb]), geom.f_normal, geom.f_delta)
    lam_b = face_eigenvalue(model, w[b], geom.b_normal, geom.b_delta)
    return sum_face_eigenvalues(geom, lam_f, lam_b, w.shape[0])


@partial(jax.jit, static_argnums=0)
def transformation(model, w):
    """dU/dV per cell, shape (n_cells, 4, 4)."""

    def cons(x):
        return conservative(model.gamma, model.gas_constant, x[None])[0]

    return jax.vmap(jax.jacfwd(cons))(w)


def time_term_blocks(model, geom, w, params, cfl):
    """Pseudo-time blocks dU/dV Lambda_i / CFL, shape (n_cells, 4, 4)."""
    w = jnp.asarray(w)
    m = np.asarray(transformation(model, w))
    lam = np.asarray(spectral_radius(model, geom, w))
    return m * (lam / cfl)[:, None, None]


@partial(jax.jit, static_argnums=0)
def explicit_update(model, geom, w, r, params, cfl):
    """Explicit increment CFL / Lambda_i (dU/dV)^-1 R_i, shape (n_cells, 4)."""
    lam = spectral_radius(model, geom, w)
    du = jnp.linalg.solve(transformation(model, w), r[:, :, None])[:, :, 0]
    return du * (cfl / lam)[:, None]


def freestream_conditions(mach, angle_of_attack=0.0, gamma=1.4):
    """Non-dimensional freestream: (gas constant, farfield condition dict).

    rho = |v| = T = 1, p = 1 / (gamma M^2); the angle in degrees.
    """
    r_gas = gas_constant_for_mach(mach, gamma)
    a = np.radians(angle_of_attack)
    return r_gas, {
        "type": "farfield",
        "velocity": (float(np.cos(a)), float(np.sin(a))),
        "pressure": r_gas,
        "temperature": 1.0,
    }

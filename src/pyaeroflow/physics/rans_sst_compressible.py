"""Compressible RANS: density-based mean flow plus k-omega SST (L1 stage C4).

State per cell: (p, u, v, T, k, omega). The mean flow is that of
:mod:`pyaeroflow.physics.compressible` with the effective viscosity
mu + mu_t, mu_t = rho nu_t, in the stress, the conductivity
c_p (mu / Pr + mu_t / Pr_t) in the heat flux and the -2/3 rho k delta_ij
part of the Boussinesq stress (compressible SST; it enters the
momentum and energy fluxes like a pressure 2/3 rho k). The total energy
does not contain k (as in CFL3D).

The turbulence rows are the conservative compressible SST-2003m of the NASA
Turbulence Modeling Resource:

    d(rho k)/dt + div(rho v k) = rho P_k - beta* rho k omega
                                 + div((mu + sigma_k mu_t) grad k)
    d(rho omega)/dt + div(rho v omega) = rho P_omega - beta rho omega^2
                                 + div((mu + sigma_omega mu_t) grad omega)
                                 + rho CD,

the kinematic sources of :mod:`pyaeroflow.physics.sst` times rho, with the
divergence part of the production (-2/3 k div v in P_k) and
nu = mu(T) / rho in the blending functions. A uniform-density flow gives
the incompressible model back. The unknowns stay k and omega.

Numerics as the incompressible SST (:mod:`pyaeroflow.physics.rans_sst`)
and the compressible SA (:mod:`pyaeroflow.physics.rans_compressible`):
k and omega upwinded with the mass flux of the Roe flux of the same face
(first order by default), two-point normal derivative for their diffusion
with the face average of the cell coefficients mu + sigma mu_t, omega row
multiplied by the reference nu (row scaling, see rans_sst), the first-order
Jacobian with the residual's face viscosities. Boundary values: k = 0 and
omega = 60 nu_w / (beta_1 d1^2) at walls, nu_w = mu(T_w) / rho_w of the
wall-face state (a vertex-based code takes mu and rho of the first
interior node), d1 the
wall-cell height (``BoundaryData.turbulence`` holds 60 / (beta_1 d1^2));
the prescribed (k, omega) at inlets and inflowing farfield faces, the
interior values elsewhere.

Pseudo time: d(rho, rho u, rho v, rho E, rho k, rho omega)/dV times the
spectral radius over CFL, with the eddy viscosity rho k / omega (an upper
bound of the limited mu_t; cell-local) in the viscous eigenvalue.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pyaeroflow.numerics import (
    boundary_face_gradient,
    interior_face_gradient,
    lsq_gradient,
)
from pyaeroflow.physics import compressible as comp
from pyaeroflow.physics import sst
from pyaeroflow.physics.compressible import (
    FARFIELD,
    TOTAL_PRESSURE_INLET,
    VELOCITY_INLET,
    WALL,
    T,
    U,
    V,
)
from pyaeroflow.physics.rans import strain_rate, upwind_scalar_flux
from pyaeroflow.solver import divergence

FLOW = comp.FLOW
NK = 4  # column of k
NW = 5  # column of omega
NT = NK  # first turbulence column (the problem interface)
OMEGA_MIN = 1.0e-8  # floor of omega in denominators (non-dimensional)
POSITIVE_COLUMNS = comp.POSITIVE_COLUMNS
RESIDUAL_NAMES = comp.RESIDUAL_NAMES  # the driver appends "k", "om"

build_boundary_data = comp.build_boundary_data
lsq_include = comp.lsq_include
scheme_parameters = comp.scheme_parameters
wall_traction = comp.wall_traction


def reference_viscosity(model):
    """Return nu = mu(1) / rho_ref, the row scale of the omega residual."""
    return model.viscosity / model.density


def _row_scale(model):
    return jnp.array([1.0, reference_viscosity(model)])


def kinematic_viscosity(model, w):
    """Molecular nu = mu(T) / rho of states w (..., >= 4)."""
    return comp.laminar_viscosity(model, w[..., T]) / comp.density(
        model.gas_constant, w
    )


# -- boundary conditions ------------------------------------------------------


def boundary_turbulence(bc, normal, w_b4, t_in):
    """(k, omega) on boundary faces from the mean boundary state and the interior."""
    vn = w_b4[:, U] * normal[:, 0] + w_b4[:, V] * normal[:, 1]
    kind = bc.kind
    inflow = (
        (kind == VELOCITY_INLET)
        | (kind == TOTAL_PRESSURE_INLET)
        | ((kind == FARFIELD) & (vn < 0.0))
    )
    out = jnp.where(inflow[:, None], bc.turbulence, t_in)
    nu_w = comp.boundary_viscosity(bc.viscosity, w_b4[:, T]) / comp.density(
        bc.gas_constant, w_b4
    )
    wall = jnp.stack((jnp.zeros_like(nu_w), bc.turbulence[:, 1] * nu_w), axis=1)
    return jnp.where((kind == WALL)[:, None], wall, out)


def boundary_state(bc, normal, w_in):
    """Boundary-face state (p, u, v, T, k, omega) from the interior state."""
    w_b4 = comp.boundary_state(bc, normal, w_in[:, :NK])
    t = boundary_turbulence(bc, normal, w_b4, w_in[:, NK:])
    return jnp.concatenate((w_b4, t), axis=1)


# -- cell quantities -------------------------------------------------------------


def _omega(w):
    return jnp.maximum(w[:, NW], OMEGA_MIN)


def cell_quantities(model, geom, w, grad):
    """Per-cell strain rate, F1, F2, kinematic nu_t, grad k . grad omega, rho, mu."""
    rho = comp.density(model.gas_constant, w)
    mu = comp.laminar_viscosity(model, w[:, T])
    k = jnp.maximum(w[:, NK], 0.0)
    omega = _omega(w)
    strain = strain_rate(grad[:, U : V + 1, :])
    kgw = jnp.sum(grad[:, NK, :] * grad[:, NW, :], axis=1)
    f1, f2 = sst.blending(k, omega, mu / rho, geom.wall_distance, kgw)
    nu_t = sst.eddy_viscosity(k, omega, strain, f2)
    return strain, f1, f2, nu_t, kgw, rho, mu


def _face_coefficients(model, bc, w_b, quantities, o, n, b):
    """mu_t and the (k, omega) diffusion coefficients at interior and boundary faces.

    Interior faces: the means of the cell values mu_t = rho nu_t and
    mu + sigma mu_t. Boundary faces: mu_t the mean of the owner value
    and the owner value (0 at walls), the coefficients with mu(T) of the
    boundary state and the owner's F1.
    """
    _, f1, _, nu_t, _, rho, mu = quantities
    mu_t = rho * nu_t
    c_k, c_w = sst.diffusion_coefficients(mu, mu_t, f1)
    mu_t_f = 0.5 * (mu_t[o] + mu_t[n])
    dc_f = jnp.stack((0.5 * (c_k[o] + c_k[n]), 0.5 * (c_w[o] + c_w[n])), axis=1)
    mu_t_b = 0.5 * (mu_t[b] + jnp.where(bc.kind == WALL, 0.0, mu_t[b]))
    mu_b = comp.laminar_viscosity(model, w_b[:, T])
    dc_b = jnp.stack(sst.diffusion_coefficients(mu_b, mu_t_b, f1[b]), axis=1)
    return mu_t_f, dc_f, mu_t_b, dc_b


def _tke_flux(model, w_a, w_b, normal):
    """Flux part of the -2/3 rho k delta_ij stress: 2/3 rho k (0, n, v.n).

    rho k and v of the mean of the two states (n, 6); added to the flux
    (the residual is the net outflow).
    """
    w_f = 0.5 * (w_a + w_b)
    rho_k = 0.5 * (
        comp.density(model.gas_constant, w_a) * jnp.maximum(w_a[:, NK], 0.0)
        + comp.density(model.gas_constant, w_b) * jnp.maximum(w_b[:, NK], 0.0)
    )
    q = 2.0 / 3.0 * rho_k
    vn = w_f[:, U] * normal[:, 0] + w_f[:, V] * normal[:, 1]
    return jnp.stack(
        (jnp.zeros_like(q), q * normal[:, 0], q * normal[:, 1], q * vn), axis=1
    )


def _sources(model, geom, w, grad, quantities):
    """Return rho times the kinematic SST sources (s_k, s_omega) per unit volume."""
    strain, f1, f2, _, kgw, rho, _ = quantities
    div = grad[:, U, 0] + grad[:, V, 1]
    switch = 1.0 if geom.production_switch is None else geom.production_switch
    s_k, s_w = sst.sources(
        w[:, NK], _omega(w), strain, f1, f2, kgw, switch, divergence=div
    )
    return rho * s_k, rho * s_w


_cell_quantities_remat = jax.checkpoint(cell_quantities, static_argnums=(0,))


# -- residual ---------------------------------------------------------------------


@partial(jax.jit, static_argnums=0)
def limiter(model, geom, bc, w):
    """Limiter values psi of the current state, shape (n_cells, 6)."""
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    return comp._limiter(model, geom, w, w_b, lsq_gradient(geom, w, w_b))


def _dot_n(g, normal):
    return g[:, 0] * normal[:, 0] + g[:, 1] * normal[:, 1]


@partial(jax.jit, static_argnums=(0, 4))
def residual(model, geom, bc, w, order, params=None, psi=None):
    """Residual R(V), shape (n_cells, 6): mass, momentum, energy, rho k, rho omega.

    The omega row is multiplied by the reference nu. ``params`` (the scheme
    parameters of the problem interface) is unused.
    """
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    w_b = boundary_state(bc, geom.b_normal, w[b])
    grad = lsq_gradient(geom, w, w_b)
    if order == 2 or model.turbulence_spatial_order == 2:
        w_l, w_r, w_in = comp.face_states(model, geom, bc, w, 2, psi, w_b, grad)
    m_l, m_r, m_in = (
        (w_l[:, :NK], w_r[:, :NK], w_in[:, :NK])
        if order == 2
        else (w[o, :NK], w[n, :NK], w[b, :NK])
    )
    s_l, s_r, s_in = (
        (w_l[:, NK:], w_r[:, NK:], w_in[:, NK:])
        if model.turbulence_spatial_order == 2
        else (w[o, NK:], w[n, NK:], w[b, NK:])
    )

    f_mean = comp.interior_convective_flux(model, m_l, m_r, geom.f_normal)
    b_mean = comp.boundary_convective_flux(model, bc, geom.b_normal, m_in)
    t_f = upwind_scalar_flux(f_mean[:, 0:1], s_l, s_r)
    t_b = upwind_scalar_flux(b_mean[:, 0:1], s_in, w_b[:, NK:])

    q = _cell_quantities_remat(model, geom, w, grad)
    w4, g4, wb4 = w[:, :NK], grad[:, :NK], w_b[:, :NK]
    g_f = interior_face_gradient(geom, w4, g4, model.viscous_gradient, model.alpha)
    g_b = boundary_face_gradient(geom, w4, wb4, g4, model.viscous_gradient)
    mu_t_f, dc_f, mu_t_b, dc_b = _face_coefficients(model, bc, w_b, q, o, n, b)
    f_mean = (
        f_mean
        + comp.viscous_flux(
            model, g_f, 0.5 * (w4[o] + w4[n]), geom.f_normal, mu_t=mu_t_f
        )
        + _tke_flux(model, w[o], w[n], geom.f_normal)
    )
    b_mean = (
        b_mean
        + comp.boundary_viscous_flux(model, bc, g_b, wb4, geom.b_normal, mu_t=mu_t_b)
        + _tke_flux(model, w_b, w_b, geom.b_normal)
    )
    # Two-point normal derivative of k and omega (see rans_sst).
    dn_f = _dot_n(geom.f_delta, geom.f_normal)
    dn_b = _dot_n(geom.b_delta, geom.b_normal)
    t_f = t_f - dc_f * (w[n, NK:] - w[o, NK:]) / dn_f[:, None]
    t_b = t_b - dc_b * (w_b[:, NK:] - w[b, NK:]) / dn_b[:, None]

    r = divergence(
        geom,
        jnp.concatenate((f_mean, t_f), axis=1),
        jnp.concatenate((b_mean, t_b), axis=1),
    )
    s_k, s_w = _sources(model, geom, w, grad, q)
    r = r.at[:, NK].add(-geom.area * s_k)
    r = r.at[:, NW].add(-geom.area * s_w)
    return r.at[:, NK:].multiply(_row_scale(model)[None, :])


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


# -- first-order Jacobian (compact fluxes) --------------------------------------


@partial(jax.jit, static_argnums=0)
def jacobian_coefficients(model, geom, bc, w):
    """Face viscosities of the first-order Jacobian, from the current state.

    (mu_t_f, dc_f, mu_t_b, dc_b) as in the residual; given values in the
    compact fluxes (see :func:`pyaeroflow.physics.rans_sst.jacobian_coefficients`).
    """
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    w_b = boundary_state(bc, geom.b_normal, w[b])
    grad = lsq_gradient(geom, w, w_b)
    q = cell_quantities(model, geom, w, grad)
    return _face_coefficients(model, bc, w_b, q, o, n, b)


def interior_flux(model, params, w_o, w_n, mu_t, dc, normal, delta):
    """Compact first-order flux through an interior face (for the Jacobian)."""
    conv = comp.jacobian_convective_flux(
        model, w_o[None, :NK], w_n[None, :NK], normal[None]
    )[0]
    dn = jnp.dot(delta, normal)
    if model.viscous_gradient == "alpha-damping":
        coeff = model.alpha / dn
    else:
        coeff = dn / jnp.dot(delta, delta)
    g = (coeff * (w_n[:NK] - w_o[:NK]))[:, None] * normal[None, :]
    w_f = 0.5 * (w_o[:NK] + w_n[:NK])
    visc = comp.viscous_flux(model, g[None], w_f[None], normal[None], mu_t=mu_t[None])
    visc = visc + _tke_flux(model, w_o[None], w_n[None], normal[None])
    t = upwind_scalar_flux(conv[0], w_o[NK:], w_n[NK:])
    t = t - dc * (w_n[NK:] - w_o[NK:]) / dn
    return jnp.concatenate((conv + visc[0], t * _row_scale(model)))


def boundary_flux(
    model, params, w_o, bc_kind, bc_value, bc_turb, mu_t, dc, normal, delta
):
    """Compact first-order flux through a boundary face (for the Jacobian)."""
    bc = comp.BoundaryData(
        bc_kind[None],
        bc_value[None],
        model.gamma,
        model.gas_constant,
        None,
        bc_turb[None],
        comp.viscosity_parameters(model),
    )
    w_b = boundary_state(bc, normal[None], w_o[None])[0]
    conv = comp.boundary_convective_flux(model, bc, normal[None], w_o[None, :NK])[0]
    dn = jnp.dot(delta, normal)
    g = ((w_b[:NK] - w_o[:NK]) / dn)[:, None] * normal[None, :]
    visc = comp.boundary_viscous_flux(
        model, bc, g[None], w_b[None, :NK], normal[None], mu_t=mu_t[None]
    )
    visc = visc + _tke_flux(model, w_b[None], w_b[None], normal[None])
    t = upwind_scalar_flux(conv[0], w_o[NK:], w_b[NK:])
    t = t - dc * (w_b[NK:] - w_o[NK:]) / dn
    return jnp.concatenate((conv + visc[0], t * _row_scale(model)))


@partial(jax.jit, static_argnums=0)
def source_blocks(model, geom, bc, w):
    """Cell-local source Jacobian blocks (n_cells, 6, 6): the destruction terms.

    d(rho beta* k omega)/d(k, omega) and d(rho beta omega^2)/d omega (F1-
    blended beta) times the cell area, at frozen rho (the implicit part;
    production and cross diffusion are left to the exact Newton-Krylov
    product), the omega row scaled like the residual.
    """
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    _, f1, _, _, _, rho, _ = cell_quantities(model, geom, w, grad)
    k = jnp.maximum(w[:, NK], 0.0)
    omega = _omega(w)
    beta = sst.blend(f1, sst.BETA_1, sst.BETA_2)
    a = geom.area * rho
    blocks = jnp.zeros((w.shape[0], model.n_var, model.n_var))
    blocks = blocks.at[:, NK, NK].set(a * sst.BETA_STAR * omega)
    blocks = blocks.at[:, NK, NW].set(a * sst.BETA_STAR * k)
    return blocks.at[:, NW, NW].set(reference_viscosity(model) * a * 2.0 * beta * omega)


@partial(jax.jit, static_argnums=0)
def source_jacobian(model, geom, bc, w):
    """Full cell-local d(-area row_scale rho src)/d(k, omega), shape (n, 2, 2).

    Production, destruction and cross diffusion at frozen rho, strain, div v,
    F1, F2 and grad k . grad omega; for the source-term time-step limit.
    """
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    strain, f1, f2, _, kgw, rho, _ = cell_quantities(model, geom, w, grad)
    div = grad[:, U, 0] + grad[:, V, 1]
    switch = 1.0 if geom.production_switch is None else geom.production_switch
    switch = jnp.broadcast_to(jnp.asarray(switch, dtype=float), strain.shape)
    scale = _row_scale(model)

    def cell(t, s, a, b, c, sw, dv, r):
        s_k, s_w = sst.sources(
            t[0], jnp.maximum(t[1], OMEGA_MIN), s, a, b, c, sw, divergence=dv
        )
        return -r * jnp.stack([s_k, s_w]) * scale

    jac = jax.vmap(jax.jacfwd(cell))(w[:, NK:], strain, f1, f2, kgw, switch, div, rho)
    return geom.area[:, None, None] * jac


def segregated_mask(n_var):
    """Block mask of the segregated coupling: mean flow and turbulence apart."""
    mask = jnp.zeros((n_var, n_var))
    return mask.at[:NK, :NK].set(1.0).at[NK:, NK:].set(1.0)


# -- pseudo time ------------------------------------------------------------------


def conservative(model, w):
    """(rho, rho u, rho v, rho E, rho k, rho omega) of primitive states (n, 6)."""
    u4 = comp.conservative(model.gamma, model.gas_constant, w[:, :NK])
    return jnp.concatenate((u4, u4[:, 0:1] * w[:, NK:]), axis=1)


@partial(jax.jit, static_argnums=0)
def transformation(model, w):
    """dU/dV per cell, shape (n_cells, 6, 6)."""

    def cons(x):
        return conservative(model, x[None])[0]

    return jax.vmap(jax.jacfwd(cons))(w)


@partial(jax.jit, static_argnums=0)
def spectral_radius(model, geom, w, params=None):
    """Sum over faces of (|v_n| + c + lambda_v) * length with mu + rho k / omega."""
    o, nb, b = geom.f_owner, geom.f_neigh, geom.b_owner
    mu_t = comp.density(model.gas_constant, w) * jnp.maximum(w[:, NK], 0.0) / _omega(w)
    w_f = 0.5 * (w[o] + w[nb])
    lam_f = comp.face_eigenvalue(
        model, w_f, geom.f_normal, geom.f_delta, 0.5 * (mu_t[o] + mu_t[nb])
    )
    lam_b = comp.face_eigenvalue(model, w[b], geom.b_normal, geom.b_delta, mu_t[b])
    return comp.sum_face_eigenvalues(geom, lam_f, lam_b, w.shape[0])


@partial(jax.jit, static_argnums=0)
def _time_term_blocks(model, geom, w, cfl):
    m = transformation(model, w)
    m = m.at[:, NW, :].multiply(reference_viscosity(model))  # row scaling
    return m * (spectral_radius(model, geom, w) / cfl)[:, None, None]


def time_term_blocks(model, geom, w, params, cfl):
    """Pseudo-time blocks dU/dV Lambda_i / CFL, shape (n_cells, 6, 6)."""
    return np.asarray(_time_term_blocks(model, geom, jnp.asarray(w), cfl))


def explicit_update(model, geom, w, r, params, cfl):
    """Not available: the explicit mode has no RANS (as incompressible)."""
    raise NotImplementedError("explicit mode is not available with RANS")


# -- fields and wall quantities ------------------------------------------------------


@partial(jax.jit, static_argnums=0)
def _fields(model, geom, bc, w):
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    _, f1, f2, nu_t, _, _, _ = cell_quantities(model, geom, w, grad)
    return nu_t, f1, f2


def eddy_viscosity(model, geom, bc, w):
    """Kinematic eddy viscosity nu_t per cell."""
    return _fields(model, geom, bc, w)[0]


def blending_f1(model, geom, bc, w):
    """Menter's first blending function per cell."""
    return _fields(model, geom, bc, w)[1]


@partial(jax.jit, static_argnums=0)
def wall_heat_flux(model, geom, bc, w):
    """Heat flux q_w = -k dT/dn into the fluid per boundary face (n_bfaces,)."""
    b = geom.b_owner
    w_b = boundary_state(bc, geom.b_normal, w[b])
    grad = lsq_gradient(geom, w, w_b)
    q = cell_quantities(model, geom, w, grad)
    _, _, mu_t_b, _ = _face_coefficients(
        model, bc, w_b, q, geom.f_owner, geom.f_neigh, b
    )
    g_b = boundary_face_gradient(
        geom, w[:, :NK], w_b[:, :NK], grad[:, :NK], model.viscous_gradient
    )
    flux = comp.boundary_viscous_flux(
        model, bc, g_b, w_b[:, :NK], geom.b_normal, mu_t=mu_t_b
    )
    return -flux[:, 3]

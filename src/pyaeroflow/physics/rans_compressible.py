"""Compressible RANS: density-based mean flow plus SA-neg (L1 stage C3).

State per cell: (p, u, v, T, nu_tilde). The mean flow is that of
:mod:`pyaeroflow.physics.compressible` with the effective viscosity
mu + mu_t, mu_t = rho nu_tilde f_v1, in the stress and the conductivity
c_p (mu / Pr + mu_t / Pr_t) in the heat flux (Boussinesq without the
2/3 rho k term, as SU2 for SA).

The turbulence row is the conservative compressible form of SA-neg of the
NASA Turbulence Modeling Resource (SA page, "compressible" form; the
negative branch of Allmaras, Johnson & Spalart 2012):

    d(rho nu_t)/dt + div(rho v nu_t)
        = rho (P - D) + (1/sigma) [div(rho (nu + nu_t f_n) grad nu_t)
                                   + c_b2 rho |grad nu_t|^2]
          - (1/sigma) (nu + nu_t f_n) grad rho . grad nu_t,

nu_t standing for nu_tilde, nu = mu(T) / rho the local kinematic viscosity
and P, D the SA-neg production and destruction of
:mod:`pyaeroflow.physics.spalart_allmaras` (with f_t2, as the incompressible
SA-neg). The last term makes the equation rho times the kinematic one, so a
uniform-density flow gives the incompressible model back. The unknown stays
nu_tilde.

Convection: nu_tilde upwinded with the mass flux of the Roe flux of the same
face (first order by default, ``turbulence_spatial_order``), so that the
turbulence row is consistent with continuity. Diffusion: the face gradient
of the mean flow (alpha-damping) with rho_f (nu_f + nu_tilde_f f_n) / sigma
of the face-mean state. Boundary values of nu_tilde as in the incompressible
SA: zero at walls, prescribed at inlets and inflowing farfield faces, the
interior value elsewhere.

Pseudo time: the blocks are d(rho, rho u, rho v, rho E, rho nu_tilde)/dV
times the spectral radius over CFL, the conservative time derivative of all
five rows.
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
from pyaeroflow.physics import spalart_allmaras as sa
from pyaeroflow.physics.compressible import (
    FARFIELD,
    TOTAL_PRESSURE_INLET,
    VELOCITY_INLET,
    WALL,
    P,
    T,
    U,
    V,
)
from pyaeroflow.solver import divergence

FLOW = comp.FLOW
NT = 4  # column of nu_tilde
POSITIVE_COLUMNS = comp.POSITIVE_COLUMNS
RESIDUAL_NAMES = comp.RESIDUAL_NAMES  # the driver appends "nu"

build_boundary_data = comp.build_boundary_data
lsq_include = comp.lsq_include
scheme_parameters = comp.scheme_parameters
wall_traction = comp.wall_traction


# -- turbulence quantities ----------------------------------------------------


def kinematic_viscosity(model, w):
    """Molecular nu = mu(T) / rho of states w (..., >= 4)."""
    return comp.laminar_viscosity(model, w[..., T]) / comp.density(
        model.gas_constant, w
    )


def eddy_viscosity(model, w):
    """Kinematic eddy viscosity nu_tilde f_v1 per state (0 for nu_tilde < 0)."""
    return sa.eddy_viscosity(w[..., NT], kinematic_viscosity(model, w))


def dynamic_eddy_viscosity(model, w):
    """mu_t = rho nu_tilde f_v1 per state."""
    return comp.density(model.gas_constant, w) * eddy_viscosity(model, w)


def face_coefficients(model, w_a, w_b):
    """Eddy viscosity and nu_tilde diffusion coefficient at faces.

    For the states on both sides (n, 5): mu_t is the mean of the two
    mu_t values (as the incompressible SA, which averages nu_t), the
    diffusion coefficient rho_f (nu_f + nu_tilde_f f_n) / sigma of the mean
    state.
    """
    w_f = 0.5 * (w_a + w_b)
    mu_t = 0.5 * (
        dynamic_eddy_viscosity(model, w_a) + dynamic_eddy_viscosity(model, w_b)
    )
    rho_f = comp.density(model.gas_constant, w_f)
    dc = rho_f * sa.diffusion_coefficient(w_f[:, NT], kinematic_viscosity(model, w_f))
    return mu_t, dc


# -- boundary conditions ------------------------------------------------------


def boundary_nu_tilde(bc, normal, w_b4, nt_in):
    """nu_tilde on boundary faces from the mean boundary state and the interior."""
    vn = w_b4[:, U] * normal[:, 0] + w_b4[:, V] * normal[:, 1]
    kind = bc.kind
    prescribed = (
        (kind == VELOCITY_INLET)
        | (kind == TOTAL_PRESSURE_INLET)
        | ((kind == FARFIELD) & (vn < 0.0))
    )
    out = jnp.where(prescribed, bc.nu_tilde, nt_in)
    return jnp.where(kind == WALL, 0.0, out)


def boundary_state(bc, normal, w_in):
    """Boundary-face state (p, u, v, T, nu_tilde) from the interior state."""
    w_b4 = comp.boundary_state(bc, normal, w_in[:, :NT])
    nt = boundary_nu_tilde(bc, normal, w_b4, w_in[:, NT])
    return jnp.concatenate((w_b4, nt[:, None]), axis=1)


def upwind_scalar_flux(mass_flux, s_o, s_n):
    """Upwind flux mdot * s for the mass flux mdot (positive = outward)."""
    return jnp.maximum(mass_flux, 0.0) * s_o + jnp.minimum(mass_flux, 0.0) * s_n


# -- sources --------------------------------------------------------------------


def _source_inputs(geom, w, grad):
    """Stencil-dependent source inputs: vorticity, |grad nu_t|^2, grad rho . grad nu_t.

    The density gradient follows from the gradients of p and T,
    grad rho = rho (grad p / p - grad T / T).
    """
    vort = jnp.abs(grad[:, V, 0] - grad[:, U, 1])
    g_nt = grad[:, NT, :]
    grad_sq = jnp.sum(g_nt**2, axis=1)
    grad_rho_rel = grad[:, P, :] / w[:, P, None] - grad[:, T, :] / w[:, T, None]
    cross = jnp.sum(grad_rho_rel * g_nt, axis=1)  # (grad rho . grad nu_t) / rho
    switch = 1.0 if geom.production_switch is None else geom.production_switch
    switch = jnp.broadcast_to(jnp.asarray(switch, dtype=float), grad_sq.shape)
    return vort, grad_sq, cross, switch


def cell_source(model, w, vort, distance, grad_sq, cross, switch):
    """Source of the rho nu_tilde row per unit volume (cell-local).

    rho (P - D + c_b2 / sigma |grad nu_t|^2) - (nu + nu_t f_n) / sigma
    grad rho . grad nu_t, with ``cross`` = (grad rho . grad nu_t) / rho.
    """
    rho = comp.density(model.gas_constant, w)
    nu = kinematic_viscosity(model, w)
    nt = w[..., NT]
    src = sa.source(nt, nu, vort, distance, grad_sq, switch)
    return rho * (src - sa.diffusion_coefficient(nt, nu) * cross)


def _sources_of_state(model, geom, w, grad):
    vort, grad_sq, cross, switch = _source_inputs(geom, w, grad)
    return cell_source(model, w, vort, geom.wall_distance, grad_sq, cross, switch)


# -- residual ---------------------------------------------------------------------


@partial(jax.jit, static_argnums=0)
def limiter(model, geom, bc, w):
    """Limiter values psi of the current state, shape (n_cells, 5)."""
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    return comp._limiter(model, geom, w, w_b, lsq_gradient(geom, w, w_b))


def _dot_n(g, normal):
    return g[:, 0] * normal[:, 0] + g[:, 1] * normal[:, 1]


@partial(jax.jit, static_argnums=(0, 4))
def residual(model, geom, bc, w, order, params=None, psi=None):
    """Residual R(V), shape (n_cells, 5): mass, momentum, energy, rho nu_tilde.

    ``params`` (the scheme parameters of the problem interface) is unused.
    """
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    w_b = boundary_state(bc, geom.b_normal, w[b])
    grad = lsq_gradient(geom, w, w_b)
    if order == 2 or model.turbulence_spatial_order == 2:
        w_l, w_r, w_in = comp.face_states(model, geom, bc, w, 2, psi, w_b, grad)
    m_l, m_r, m_in = (
        (w_l[:, :NT], w_r[:, :NT], w_in[:, :NT])
        if order == 2
        else (w[o, :NT], w[n, :NT], w[b, :NT])
    )
    s_l, s_r, s_in = (
        (w_l[:, NT], w_r[:, NT], w_in[:, NT])
        if model.turbulence_spatial_order == 2
        else (w[o, NT], w[n, NT], w[b, NT])
    )

    f_mean = comp.interior_convective_flux(model, m_l, m_r, geom.f_normal)
    b_mean = comp.boundary_convective_flux(model, bc, geom.b_normal, m_in)
    f_nt = upwind_scalar_flux(f_mean[:, 0], s_l, s_r)
    b_nt = upwind_scalar_flux(b_mean[:, 0], s_in, w_b[:, NT])

    g_f = interior_face_gradient(geom, w, grad, model.viscous_gradient, model.alpha)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    mu_t_f, dc_f = face_coefficients(model, w[o], w[n])
    mu_t_b, dc_b = face_coefficients(model, w[b], w_b)
    w_f = 0.5 * (w[o] + w[n])
    f_mean = f_mean + comp.viscous_flux(
        model, g_f[:, :NT], w_f[:, :NT], geom.f_normal, mu_t=mu_t_f
    )
    b_mean = b_mean + comp.boundary_viscous_flux(
        model, bc, g_b[:, :NT], w_b[:, :NT], geom.b_normal, mu_t=mu_t_b
    )
    f_nt = f_nt - dc_f * _dot_n(g_f[:, NT, :], geom.f_normal)
    b_nt = b_nt - dc_b * _dot_n(g_b[:, NT, :], geom.b_normal)

    r = divergence(
        geom,
        jnp.concatenate((f_mean, f_nt[:, None]), axis=1),
        jnp.concatenate((b_mean, b_nt[:, None]), axis=1),
    )
    src = _sources_of_state(model, geom, w, grad)
    return r.at[:, NT].add(-geom.area * src)


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


def interior_flux(model, params, w_o, w_n, normal, delta):
    """Compact first-order flux through an interior face (for the Jacobian)."""
    conv = comp.jacobian_convective_flux(
        model, w_o[None, :NT], w_n[None, :NT], normal[None]
    )[0]
    dn = jnp.dot(delta, normal)
    if model.viscous_gradient == "alpha-damping":
        coeff = model.alpha / dn
    else:
        coeff = dn / jnp.dot(delta, delta)
    g = (coeff * (w_n - w_o))[:, None] * normal[None, :]
    mu_t, dc = face_coefficients(model, w_o[None], w_n[None])
    w_f = 0.5 * (w_o + w_n)
    visc = comp.viscous_flux(
        model, g[None, :NT], w_f[None, :NT], normal[None], mu_t=mu_t
    )[0]
    nt = upwind_scalar_flux(conv[0], w_o[NT], w_n[NT]) - dc[0] * g[NT] @ normal
    return jnp.concatenate((conv + visc, nt[None]))


def boundary_flux(model, params, w_o, bc_kind, bc_value, bc_nt, normal, delta):
    """Compact first-order flux through a boundary face (for the Jacobian)."""
    bc = comp.BoundaryData(
        bc_kind[None], bc_value[None], model.gamma, model.gas_constant, bc_nt[None]
    )
    w_b = boundary_state(bc, normal[None], w_o[None])[0]
    conv = comp.boundary_convective_flux(model, bc, normal[None], w_o[None, :NT])[0]
    dn = jnp.dot(delta, normal)
    g = ((w_b - w_o) / dn)[:, None] * normal[None, :]
    mu_t, dc = face_coefficients(model, w_o[None], w_b[None])
    visc = comp.boundary_viscous_flux(
        model, bc, g[None, :NT], w_b[None, :NT], normal[None], mu_t=mu_t
    )[0]
    nt = (
        upwind_scalar_flux(conv[0], w_o[NT], w_b[NT]) - dc[0] * (w_b[NT] - w_o[NT]) / dn
    )
    return jnp.concatenate((conv + visc, nt[None]))


@partial(jax.jit, static_argnums=0)
def source_blocks(model, geom, bc, w):
    """Cell-local source Jacobian blocks (n_cells, 5, 5).

    Only d(-area src)/d nu_tilde on the turbulence diagonal, clipped to its
    negative part, with the stencil inputs (vorticity, |grad nu_t|^2,
    grad rho . grad nu_t) and the mean-flow state frozen, as the
    incompressible SA (:func:`pyaeroflow.physics.rans.source_blocks`).
    """
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    vort, grad_sq, cross, switch = _source_inputs(geom, w, grad)

    def one(nt, wc, vo, d, gs, cr, sw):
        return cell_source(model, wc.at[NT].set(nt), vo, d, gs, cr, sw)

    d_src = jax.vmap(jax.grad(one))(
        w[:, NT], w, vort, geom.wall_distance, grad_sq, cross, switch
    )
    blocks = jnp.zeros((w.shape[0], model.n_var, model.n_var))
    return blocks.at[:, NT, NT].set(-geom.area * jnp.minimum(d_src, 0.0))


def segregated_mask(n_var):
    """Block mask of the segregated coupling: mean flow and turbulence apart."""
    mask = jnp.zeros((n_var, n_var))
    return mask.at[:NT, :NT].set(1.0).at[NT:, NT:].set(1.0)


# -- pseudo time ------------------------------------------------------------------


def conservative(model, w):
    """(rho, rho u, rho v, rho E, rho nu_tilde) of primitive states (n, 5)."""
    u4 = comp.conservative(model.gamma, model.gas_constant, w[:, :NT])
    rho_nt = u4[:, 0] * w[:, NT]
    return jnp.concatenate((u4, rho_nt[:, None]), axis=1)


@partial(jax.jit, static_argnums=0)
def transformation(model, w):
    """dU/dV per cell, shape (n_cells, 5, 5)."""

    def cons(x):
        return conservative(model, x[None])[0]

    return jax.vmap(jax.jacfwd(cons))(w)


@partial(jax.jit, static_argnums=0)
def spectral_radius(model, geom, w, params=None):
    """Sum over faces of (|v_n| + c + lambda_v) * length with mu + mu_t."""
    o, nb, b = geom.f_owner, geom.f_neigh, geom.b_owner
    mu_t = dynamic_eddy_viscosity(model, w)
    w_f = 0.5 * (w[o] + w[nb])
    lam_f = comp.face_eigenvalue(
        model, w_f, geom.f_normal, geom.f_delta, 0.5 * (mu_t[o] + mu_t[nb])
    )
    lam_b = comp.face_eigenvalue(model, w[b], geom.b_normal, geom.b_delta, mu_t[b])
    return comp.sum_face_eigenvalues(geom, lam_f, lam_b, w.shape[0])


def time_term_blocks(model, geom, w, params, cfl):
    """Pseudo-time blocks dU/dV Lambda_i / CFL, shape (n_cells, 5, 5)."""
    w = jnp.asarray(w)
    m = np.asarray(transformation(model, w))
    lam = np.asarray(spectral_radius(model, geom, w))
    return m * (lam / cfl)[:, None, None]


def explicit_update(model, geom, w, r, params, cfl):
    """Not available: the explicit mode has no RANS (as incompressible)."""
    raise NotImplementedError("explicit mode is not available with RANS")


# -- wall quantities ---------------------------------------------------------------


@partial(jax.jit, static_argnums=0)
def wall_heat_flux(model, geom, bc, w):
    """Heat flux q_w = -k dT/dn into the fluid per boundary face (n_bfaces,).

    As :func:`pyaeroflow.physics.compressible.wall_heat_flux`, with the
    face eddy viscosity of the residual.
    """
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    mu_t, _ = face_coefficients(model, w[geom.b_owner], w_b)
    flux = comp.boundary_viscous_flux(
        model, bc, g_b[:, :NT], w_b[:, :NT], geom.b_normal, mu_t=mu_t
    )
    return -flux[:, 3]

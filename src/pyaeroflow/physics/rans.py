"""RANS residual: incompressible mean flow plus the SA-neg equation (D11, D13).

State per cell: (p, u, v, nu_tilde). The mean-flow fluxes are those of
:mod:`pyaeroflow.physics.incompressible` with the effective viscosity
mu + mu_t at the faces; the turbulence equation uses an upwind convective
flux with the face-average normal velocity, the alpha-damping face gradient
for its diffusion and the SA-neg sources of
:mod:`pyaeroflow.physics.spalart_allmaras`. The transport-equation interface
(D13) is this module's pattern: an extra column, a boundary value rule, a
convective and a diffusive face flux, a cell source.

Boundary values of nu_tilde: zero at walls, the prescribed value at
velocity inlets, total-pressure inlets and inflowing farfield faces, the
interior value elsewhere (zero gradient).
"""

from functools import partial

import jax
import jax.numpy as jnp

from pyaeroflow.numerics import (
    boundary_face_gradient,
    interior_face_gradient,
    lsq_gradient,
    reconstruct,
)
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import spalart_allmaras as sa
from pyaeroflow.physics.incompressible import (
    FARFIELD,
    TOTAL_PRESSURE_INLET,
    VELOCITY_INLET,
    WALL,
    BoundaryData,
    U,
    V,
)
from pyaeroflow.solver import divergence

NT = 3  # column of nu_tilde


def kinematic_viscosity(model):
    """Kinematic viscosity mu / rho."""
    return model.viscosity / model.density


def boundary_nu_tilde(bc, normal, w_b3, nt_in):
    """nu_tilde on boundary faces from the mean boundary state and the interior."""
    vn = w_b3[:, U] * normal[:, 0] + w_b3[:, V] * normal[:, 1]
    kind = bc.kind
    prescribed = (
        (kind == VELOCITY_INLET)
        | (kind == TOTAL_PRESSURE_INLET)
        | ((kind == FARFIELD) & (vn < 0.0))
    )
    out = jnp.where(prescribed, bc.nu_tilde, nt_in)
    return jnp.where(kind == WALL, 0.0, out)


def boundary_state(bc, normal, w_in):
    """Boundary-face state (p, u, v, nu_tilde) from the interior state."""
    w_b3 = inc.boundary_state(bc, normal, w_in[:, :3])
    nt = boundary_nu_tilde(bc, normal, w_b3, w_in[:, NT])
    return jnp.concatenate((w_b3, nt[:, None]), axis=1)


def upwind_scalar_flux(vn, s_o, s_n):
    """Upwind flux vn * s for the normal velocity vn (positive = outward)."""
    return jnp.maximum(vn, 0.0) * s_o + jnp.minimum(vn, 0.0) * s_n


def face_viscosities(model, nt_o, nt_n):
    """Effective dynamic viscosity and SA diffusion coefficient at a face."""
    nu = kinematic_viscosity(model)
    mu_t = (
        0.5
        * model.density
        * (sa.eddy_viscosity(nt_o, nu) + sa.eddy_viscosity(nt_n, nu))
    )
    return model.viscosity + mu_t, sa.diffusion_coefficient(0.5 * (nt_o + nt_n), nu)


def vorticity(grad):
    """|dv/dx - du/dy| from cell gradients (n, n_var, 2)."""
    return jnp.abs(grad[:, V, 0] - grad[:, U, 1])


def _source(model, nu_tilde, vort, distance, grad_sq, switch):
    """SA source with the transition model's intermittency (cell-local)."""
    nu = kinematic_viscosity(model)
    gamma = intermittency(model, nu_tilde, vort, distance)
    return sa.source(
        nu_tilde,
        nu,
        vort,
        distance,
        grad_sq,
        switch,
        intermittency=gamma,
        ft2=model.transition == "none",
    )


def intermittency(model, nu_tilde, vort, distance):
    """Transition intermittency on the SA production: 1 without a model."""
    if model.transition == "bcm":
        return sa.bcm_intermittency(
            nu_tilde,
            kinematic_viscosity(model),
            vort,
            distance,
            model.turbulence_intensity,
        )
    return 1.0


def intermittency_field(model, geom, bc, w):
    """Intermittency per cell for the output (ones without a transition model)."""
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    vort = vorticity(lsq_gradient(geom, w, w_b))
    gamma = intermittency(model, w[:, NT], vort, geom.wall_distance)
    return jnp.broadcast_to(jnp.asarray(gamma, dtype=float), w[:, NT].shape)


def eddy_viscosity(model, w):
    """Kinematic eddy viscosity per cell."""
    return sa.eddy_viscosity(w[:, NT], kinematic_viscosity(model))


def _dot_n(g, normal):
    return g[:, 0] * normal[:, 0] + g[:, 1] * normal[:, 1]


@partial(jax.jit, static_argnums=0)
def limiter(model, geom, bc, w):
    """Limiter values psi of the current state, shape (n_cells, 4)."""
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    return inc._limiter(model, geom, w, w_b, lsq_gradient(geom, w, w_b))


@partial(jax.jit, static_argnums=(0, 4))
def residual(model, geom, bc, w, order, beta2, psi=None):
    """Residual R(w), shape (n_cells, 4): net outflow minus sources."""
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    sa_order = model.turbulence_spatial_order
    if order == 2 or sa_order == 2:
        if psi is None:
            psi = inc._limiter(model, geom, w, w_b, grad)
        w_l, w_r, w_in = reconstruct(geom, w, grad, psi)
    m_l, m_r, m_in = (
        (w_l[:, :3], w_r[:, :3], w_in[:, :3])
        if order == 2
        else (w[o, :3], w[n, :3], w[b, :3])
    )
    s_l, s_r, s_in = (
        (w_l[:, NT], w_r[:, NT], w_in[:, NT])
        if sa_order == 2
        else (w[o, NT], w[n, NT], w[b, NT])
    )

    f_flux = inc.fds_flux(model, m_l, m_r, geom.f_normal, beta2)
    b_flux = inc.boundary_convective_flux(model, bc, geom.b_normal, m_in, beta2)
    vn_f = _dot_n(0.5 * (m_l[:, 1:] + m_r[:, 1:]), geom.f_normal)
    vn_b = _dot_n(w_b[:, 1:3], geom.b_normal)
    sa_f = upwind_scalar_flux(vn_f, s_l, s_r)
    sa_b = upwind_scalar_flux(vn_b, s_in, w_b[:, NT])

    g_f = interior_face_gradient(geom, w, grad, model.viscous_gradient, model.alpha)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    mu_f, dc_f = face_viscosities(model, w[o, NT], w[n, NT])
    mu_b, dc_b = face_viscosities(model, w[b, NT], w_b[:, NT])
    f_flux = f_flux + inc.viscous_flux(model, g_f[:, 1:3], geom.f_normal, mu_f)
    b_flux = b_flux + inc.viscous_flux(model, g_b[:, 1:3], geom.b_normal, mu_b)
    sa_f = sa_f - dc_f * _dot_n(g_f[:, NT], geom.f_normal)
    sa_b = sa_b - dc_b * _dot_n(g_b[:, NT], geom.b_normal)

    r = divergence(
        geom,
        jnp.concatenate((f_flux, sa_f[:, None]), axis=1),
        jnp.concatenate((b_flux, sa_b[:, None]), axis=1),
    )
    grad_sq = jnp.sum(grad[:, NT, :] ** 2, axis=1)
    switch = 1.0 if geom.production_switch is None else geom.production_switch
    src = _source(model, w[:, NT], vorticity(grad), geom.wall_distance, grad_sq, switch)
    r = r.at[:, NT].add(-geom.area * src)
    force = jnp.asarray(model.body_force)
    r = r.at[:, 1:3].add(-geom.area[:, None] * force[None, :])
    if model.pin is not None:
        cell, p0 = model.pin
        r = r.at[cell, inc.P].set(w[cell, inc.P] - p0)
    return r


@partial(jax.jit, static_argnums=(0, 5))
def residual_jvp(model, geom, bc, w, dw, order, beta2, psi=None):
    """Exact Jacobian-vector product of the full residual (coupled)."""

    def f(x):
        return residual(model, geom, bc, x, order, beta2, psi)

    return jax.jvp(f, (w,), (dw,))[1]


def interior_flux(model, beta2, w_o, w_n, normal, delta):
    """Compact first-order flux through an interior face (for the Jacobian)."""
    conv = inc.fds_flux(model, w_o[None, :3], w_n[None, :3], normal[None], beta2)[0]
    dn = jnp.dot(delta, normal)
    if model.viscous_gradient == "alpha-damping":
        coeff = model.alpha / dn
    else:
        coeff = dn / jnp.dot(delta, delta)
    g = (coeff * (w_n[1:3] - w_o[1:3]))[:, None] * normal[None, :]
    mu, dc = face_viscosities(model, w_o[NT], w_n[NT])
    conv = conv + inc.viscous_flux(model, g[None], normal[None], mu[None])[0]
    vn = 0.5 * jnp.dot(w_o[1:3] + w_n[1:3], normal)
    flux = upwind_scalar_flux(vn, w_o[NT], w_n[NT]) - dc * coeff * (w_n[NT] - w_o[NT])
    return jnp.concatenate((conv, flux[None]))


def boundary_flux(model, beta2, w_o, bc_kind, bc_value, bc_nt, normal, delta):
    """Compact first-order flux through a boundary face (for the Jacobian)."""
    bc = BoundaryData(bc_kind[None], bc_value[None], model.density, bc_nt[None])
    w_b = boundary_state(bc, normal[None], w_o[None])[0]
    conv = inc.boundary_convective_flux(model, bc, normal[None], w_o[None, :3], beta2)[
        0
    ]
    dn = jnp.dot(delta, normal)
    g = ((w_b[1:3] - w_o[1:3]) / dn)[:, None] * normal[None, :]
    mu, dc = face_viscosities(model, w_o[NT], w_b[NT])
    conv = conv + inc.viscous_flux(model, g[None], normal[None], mu[None])[0]
    vn = jnp.dot(w_b[1:3], normal)
    flux = upwind_scalar_flux(vn, w_o[NT], w_b[NT]) - dc * (w_b[NT] - w_o[NT]) / dn
    return jnp.concatenate((conv, flux[None]))


@partial(jax.jit, static_argnums=0)
def source_blocks(model, geom, bc, w):
    """Cell-local source Jacobian blocks (n_cells, 4, 4): d(-area src)/d nu_tilde.

    Vorticity and |grad nu_tilde|^2 are frozen (they depend on the stencil).
    """
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    grad_sq = jnp.sum(grad[:, NT, :] ** 2, axis=1)
    switch = 1.0 if geom.production_switch is None else geom.production_switch
    d_src = jax.vmap(jax.grad(_source, argnums=1), in_axes=(None, 0, 0, 0, 0, 0))(
        model,
        w[:, NT],
        vorticity(grad),
        geom.wall_distance,
        grad_sq,
        jnp.broadcast_to(jnp.asarray(switch, dtype=float), w[:, NT].shape),
    )
    # Only the negative part (destruction-like) enters the implicit
    # operator: a positive dP/d nu_tilde larger than the pseudo-time
    # diagonal makes the implicit step unstable at large CFL (seen with the
    # BCM model, whose laminar state is only weakly damped). The residual
    # and the Newton-Krylov JVP are exact; this is the first-order
    # Jacobian used for pseudo-time steps and as preconditioner.
    blocks = jnp.zeros((w.shape[0], 4, 4))
    return blocks.at[:, NT, NT].set(-geom.area * jnp.minimum(d_src, 0.0))


@partial(jax.jit, static_argnums=0)
def time_term_blocks(model, geom, w, beta2, cfl):
    """Pseudo-time blocks: Gamma (mean flow) and 1 (nu_tilde), times Lambda / CFL."""
    lam = inc.spectral_radius(model, geom, w[:, :3], beta2)
    g = jnp.zeros((w.shape[0], 4, 4))
    g = g.at[:, :3, :3].set(inc.preconditioner(model, w[:, :3], beta2))
    g = g.at[:, NT, NT].set(1.0)
    return g * (lam / cfl)[:, None, None]


SEGREGATED_MASK = jnp.array(
    [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]], dtype=float
)

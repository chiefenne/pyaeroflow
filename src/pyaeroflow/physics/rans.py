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

With the SA-AFT2019b transition model (``model.transition == "aft"``,
:mod:`pyaeroflow.physics.aft`) the state has two more columns, the
log-intermittency gamma_tilde (NG) and the amplification factor n (NN). They
use the same convective flux as nu_tilde, the diffusion coefficients of the
model, the value 0 on inflow faces and zero gradient elsewhere (walls
included).
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
from pyaeroflow.physics import aft
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
NG = 4  # column of the log-intermittency gamma_tilde (SA-AFT)
NN = 5  # column of the amplification factor n (SA-AFT)


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
    """Boundary-face state (p, u, v, nu_tilde[, gamma_tilde, n]) from the interior."""
    w_b3 = inc.boundary_state(bc, normal, w_in[:, :3])
    nt = boundary_nu_tilde(bc, normal, w_b3, w_in[:, NT])
    cols = [w_b3, nt[:, None]]
    if w_in.shape[1] > NT + 1:
        vn = w_b3[:, U] * normal[:, 0] + w_b3[:, V] * normal[:, 1]
        kind = bc.kind
        inflow = (
            (kind == VELOCITY_INLET)
            | (kind == TOTAL_PRESSURE_INLET)
            | ((kind == FARFIELD) & (vn < 0.0))
        )
        cols.append(jnp.where(inflow[:, None], 0.0, w_in[:, NT + 1 :]))
    return jnp.concatenate(cols, axis=1)


def upwind_scalar_flux(vn, s_o, s_n):
    """Upwind flux vn * s for the normal velocity vn (positive = outward)."""
    return jnp.maximum(vn, 0.0) * s_o + jnp.minimum(vn, 0.0) * s_n


def face_viscosities(model, t_o, t_n):
    """Effective dynamic viscosity and turbulence diffusion coefficients at a face.

    ``t_o``, ``t_n`` are the turbulence columns (..., n_t) on both sides;
    the coefficients (..., n_t) are SA's (nu + nu_tilde)/sigma and, with
    AFT, nu + nu_t/sigma_y (gamma_tilde) and sigma_n (nu + nu_t) (n).
    """
    nu = kinematic_viscosity(model)
    nt_o, nt_n = t_o[..., 0], t_n[..., 0]
    nu_t = 0.5 * (sa.eddy_viscosity(nt_o, nu) + sa.eddy_viscosity(nt_n, nu))
    coeffs = [sa.diffusion_coefficient(0.5 * (nt_o + nt_n), nu)]
    if t_o.shape[-1] > 1:
        coeffs += [nu + nu_t / aft.SIGMA_Y, aft.SIGMA_N * (nu + nu_t)]
    return model.viscosity + model.density * nu_t, jnp.stack(coeffs, axis=-1)


def vorticity(grad):
    """|dv/dx - du/dy| from cell gradients (n, n_var, 2)."""
    return jnp.abs(grad[:, V, 0] - grad[:, U, 1])


def strain_rate(grad_v):
    """S = sqrt(2 S_ij S_ij) from velocity gradients (n, 2, 2)."""
    ux, uy = grad_v[:, 0, 0], grad_v[:, 0, 1]
    vx, vy = grad_v[:, 1, 0], grad_v[:, 1, 1]
    return sa._safe_sqrt(2.0 * (ux**2 + vy**2) + (uy + vx) ** 2)


def critical_amplification(model):
    """N_crit of SA-AFT: the given value, else from the turbulence intensity."""
    if model.critical_amplification is not None:
        return float(model.critical_amplification)
    return aft.critical_amplification(model.turbulence_intensity)


def local_shape_factor(model, geom, w, w_b):
    """H_L = d^2/nu grad(u . n_d) . n_d, n_d the unit wall-distance gradient.

    The gradient of q = u . n_d is the least-squares gradient with the
    velocity's coefficient set (q is known wherever the velocity is).
    """
    n_d = geom.wall_normal
    q = w[:, U] * n_d[:, 0] + w[:, V] * n_d[:, 1]
    nb = n_d[geom.b_owner]
    q_b = w_b[:, U] * nb[:, 0] + w_b[:, V] * nb[:, 1]
    coeff = geom.lsq_coeff
    if coeff.shape[1] > 1:
        coeff = coeff[:, U : U + 1]
    grad_q = lsq_gradient(geom._replace(lsq_coeff=coeff), q[:, None], q_b[:, None])
    dq_dn = grad_q[:, 0, 0] * n_d[:, 0] + grad_q[:, 0, 1] * n_d[:, 1]
    return geom.wall_distance**2 / kinematic_viscosity(model) * dq_dn


def _source(model, nu_tilde, vort, distance, grad_sq, switch, gamma=None):
    """SA source with the transition model's intermittency (cell-local).

    ``gamma`` (optional) is a given intermittency, e.g. the under-relaxed
    field of the driver; by default it is evaluated from the current state.

    gamma is frozen in every linearisation (Picard on gamma, as SU2 does):
    the residual uses its current value, but neither the Newton-Krylov JVP
    nor the first-order source Jacobian differentiates it. Both square roots
    of the BCM formula have unbounded slope (at onset and as nu_t -> 0), so
    the exact derivative makes the linear systems stiff and the Newton steps
    erratic.
    """
    nu = kinematic_viscosity(model)
    if gamma is None:
        gamma = intermittency(model, nu_tilde, vort, distance)
    gamma = jax.lax.stop_gradient(gamma)
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


def _turbulence_sources(model, t, grad_v, distance, grad_sq, switch, h_l, gamma):
    """Return the sources (n, n_t) of the turbulence columns ``t``, cell-local.

    ``grad_v`` (n, 2, 2) are the velocity gradients, ``h_l`` the SA-AFT local
    shape factor (unused otherwise). With AFT the SA equation uses
    f_t2 = c_t3 (1 - exp(gamma_tilde)) and no intermittency factor; nothing
    is frozen (the transition terms enter the JVP exactly).
    """
    vort = jnp.abs(grad_v[:, 1, 0] - grad_v[:, 0, 1])
    if model.transition != "aft":
        src = _source(model, t[:, 0], vort, distance, grad_sq, switch, gamma)
        return src[:, None]
    nu = kinematic_viscosity(model)
    nt, g, n = t[:, 0], t[:, NG - NT], t[:, NN - NT]
    src_sa = sa.source(nt, nu, vort, distance, grad_sq, switch, ft2_value=aft.ft2(g))
    strain = strain_rate(grad_v)
    src_n, src_g = aft.sources(
        n,
        g,
        nu,
        sa.eddy_viscosity(nt, nu),
        strain,
        vort,
        distance,
        h_l,
        critical_amplification(model),
    )
    return jnp.stack((src_sa, src_g, src_n), axis=1)


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


@partial(jax.jit, static_argnums=0)
def intermittency_field(model, geom, bc, w):
    """Intermittency per cell for the output (ones without a transition model)."""
    if model.transition == "aft":
        return jnp.exp(w[:, NG])
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    vort = vorticity(lsq_gradient(geom, w, w_b))
    gamma = intermittency(model, w[:, NT], vort, geom.wall_distance)
    return jnp.broadcast_to(jnp.asarray(gamma, dtype=float), w[:, NT].shape)


def eddy_viscosity(model, w):
    """Kinematic eddy viscosity per cell."""
    return sa.eddy_viscosity(w[:, NT], kinematic_viscosity(model))


def _dot_n(g, normal):
    return g[:, 0] * normal[:, 0] + g[:, 1] * normal[:, 1]


def _dot_n_cols(g, normal):
    """Return the normal component of gradients (n, k, 2) as (n, k)."""
    return g[:, :, 0] * normal[:, 0:1] + g[:, :, 1] * normal[:, 1:2]


def _source_inputs(model, geom, w, w_b, grad):
    """Return the stencil-dependent inputs of the sources.

    These are the velocity gradient, |grad nu_tilde|^2, the production
    switch and the AFT local shape factor (zeros without AFT).
    """
    grad_sq = jnp.sum(grad[:, NT, :] ** 2, axis=1)
    switch = 1.0 if geom.production_switch is None else geom.production_switch
    switch = jnp.broadcast_to(jnp.asarray(switch, dtype=float), grad_sq.shape)
    h_l = (
        local_shape_factor(model, geom, w, w_b)
        if model.transition == "aft"
        else jnp.zeros_like(grad_sq)
    )
    return grad[:, U : V + 1, :], grad_sq, switch, h_l


def _sources_of_state(model, geom, w, w_b, grad, gamma=None):
    grad_v, grad_sq, switch, h_l = _source_inputs(model, geom, w, w_b, grad)
    return _turbulence_sources(
        model, w[:, NT:], grad_v, geom.wall_distance, grad_sq, switch, h_l, gamma
    )


@partial(jax.jit, static_argnums=0)
def limiter(model, geom, bc, w):
    """Limiter values psi of the current state, shape (n_cells, n_var)."""
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    return inc._limiter(model, geom, w, w_b, lsq_gradient(geom, w, w_b))


@partial(jax.jit, static_argnums=(0, 4))
def residual(model, geom, bc, w, order, beta2, psi=None, gamma=None):
    """Residual R(w), shape (n_cells, n_var): net outflow minus sources.

    ``gamma`` (n_cells,) replaces the transition intermittency of the current
    state (under-relaxed intermittency of the driver); None = evaluate it.
    """
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
        (w_l[:, NT:], w_r[:, NT:], w_in[:, NT:])
        if sa_order == 2
        else (w[o, NT:], w[n, NT:], w[b, NT:])
    )

    f_flux = inc.fds_flux(model, m_l, m_r, geom.f_normal, beta2)
    b_flux = inc.boundary_convective_flux(model, bc, geom.b_normal, m_in, beta2)
    vn_f = _dot_n(0.5 * (m_l[:, 1:] + m_r[:, 1:]), geom.f_normal)
    vn_b = _dot_n(w_b[:, 1:3], geom.b_normal)
    sa_f = upwind_scalar_flux(vn_f[:, None], s_l, s_r)
    sa_b = upwind_scalar_flux(vn_b[:, None], s_in, w_b[:, NT:])

    g_f = interior_face_gradient(geom, w, grad, model.viscous_gradient, model.alpha)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    mu_f, dc_f = face_viscosities(model, w[o, NT:], w[n, NT:])
    mu_b, dc_b = face_viscosities(model, w[b, NT:], w_b[:, NT:])
    f_flux = f_flux + inc.viscous_flux(model, g_f[:, 1:3], geom.f_normal, mu_f)
    b_flux = b_flux + inc.viscous_flux(model, g_b[:, 1:3], geom.b_normal, mu_b)
    sa_f = sa_f - dc_f * _dot_n_cols(g_f[:, NT:], geom.f_normal)
    sa_b = sa_b - dc_b * _dot_n_cols(g_b[:, NT:], geom.b_normal)

    r = divergence(
        geom,
        jnp.concatenate((f_flux, sa_f), axis=1),
        jnp.concatenate((b_flux, sa_b), axis=1),
    )
    src = _sources_of_state(model, geom, w, w_b, grad, gamma)
    r = r.at[:, NT:].add(-geom.area[:, None] * src)
    force = jnp.asarray(model.body_force)
    r = r.at[:, 1:3].add(-geom.area[:, None] * force[None, :])
    if model.pin is not None:
        cell, p0 = model.pin
        r = r.at[cell, inc.P].set(w[cell, inc.P] - p0)
    return r


@partial(jax.jit, static_argnums=(0, 5))
def residual_jvp(model, geom, bc, w, dw, order, beta2, psi=None, gamma=None):
    """Exact Jacobian-vector product of the full residual (coupled)."""

    def f(x):
        return residual(model, geom, bc, x, order, beta2, psi, gamma)

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
    mu, dc = face_viscosities(model, w_o[NT:], w_n[NT:])
    conv = conv + inc.viscous_flux(model, g[None], normal[None], mu[None])[0]
    vn = 0.5 * jnp.dot(w_o[1:3] + w_n[1:3], normal)
    flux = upwind_scalar_flux(vn, w_o[NT:], w_n[NT:]) - dc * coeff * (
        w_n[NT:] - w_o[NT:]
    )
    return jnp.concatenate((conv, flux))


def boundary_flux(model, beta2, w_o, bc_kind, bc_value, bc_nt, normal, delta):
    """Compact first-order flux through a boundary face (for the Jacobian)."""
    bc = BoundaryData(bc_kind[None], bc_value[None], model.density, bc_nt[None])
    w_b = boundary_state(bc, normal[None], w_o[None])[0]
    conv = inc.boundary_convective_flux(model, bc, normal[None], w_o[None, :3], beta2)[
        0
    ]
    dn = jnp.dot(delta, normal)
    g = ((w_b[1:3] - w_o[1:3]) / dn)[:, None] * normal[None, :]
    mu, dc = face_viscosities(model, w_o[NT:], w_b[NT:])
    conv = conv + inc.viscous_flux(model, g[None], normal[None], mu[None])[0]
    vn = jnp.dot(w_b[1:3], normal)
    flux = upwind_scalar_flux(vn, w_o[NT:], w_b[NT:]) - dc * (w_b[NT:] - w_o[NT:]) / dn
    return jnp.concatenate((conv, flux))


@partial(jax.jit, static_argnums=0)
def source_blocks(model, geom, bc, w):
    """Cell-local source Jacobian blocks (n_cells, n_var, n_var).

    Only the diagonal derivatives d(-area src_k)/d w_k of the turbulence
    columns, clipped to their negative part; the velocity gradients,
    |grad nu_tilde|^2 and the AFT shape factor are frozen (they depend on
    the stencil).
    """
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    grad_v, grad_sq, switch, h_l = _source_inputs(model, geom, w, w_b, grad)

    def cell_source(t, gv, d, gs, sw, hl):
        return _turbulence_sources(
            model, t[None], gv[None], d[None], gs[None], sw[None], hl[None], None
        )[0]

    jac = jax.vmap(jax.jacfwd(cell_source))(
        w[:, NT:], grad_v, geom.wall_distance, grad_sq, switch, h_l
    )
    d_src = jnp.diagonal(jac, axis1=1, axis2=2)
    # Only the negative part (destruction-like) enters the implicit
    # operator: a positive dP/d nu_tilde larger than the pseudo-time
    # diagonal makes the implicit step unstable at large CFL (seen with the
    # BCM model, whose laminar state is only weakly damped). The residual
    # is exact and the Newton-Krylov JVP exact up to the frozen BCM
    # intermittency (see _source); this is the first-order
    # Jacobian used for pseudo-time steps and as preconditioner.
    n_var = w.shape[1]
    blocks = jnp.zeros((w.shape[0], n_var, n_var))
    k = jnp.arange(NT, n_var)
    return blocks.at[:, k, k].set(-geom.area[:, None] * jnp.minimum(d_src, 0.0))


@partial(jax.jit, static_argnums=0)
def time_term_blocks(model, geom, w, beta2, cfl):
    """Pseudo-time blocks: Gamma (mean flow) and 1 (turbulence), times Lambda / CFL."""
    lam = inc.spectral_radius(model, geom, w[:, :3], beta2)
    n_var = w.shape[1]
    g = jnp.zeros((w.shape[0], n_var, n_var))
    g = g.at[:, :3, :3].set(inc.preconditioner(model, w[:, :3], beta2))
    k = jnp.arange(NT, n_var)
    g = g.at[:, k, k].set(1.0)
    return g * (lam / cfl)[:, None, None]


def segregated_mask(n_var):
    """Block mask of the segregated coupling: mean flow and turbulence apart."""
    mask = jnp.zeros((n_var, n_var))
    return mask.at[:NT, :NT].set(1.0).at[NT:, NT:].set(1.0)


SEGREGATED_MASK = segregated_mask(4)

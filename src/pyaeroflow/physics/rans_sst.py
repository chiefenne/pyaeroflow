"""RANS residual with the k-omega SST model (SST-2003m, L2, D12).

State per cell: (p, u, v, k, omega). The module mirrors
:mod:`pyaeroflow.physics.rans` (the transport-equation interface of D13):
the mean-flow fluxes of :mod:`pyaeroflow.physics.incompressible` with
mu + rho nu_t at the faces, upwind convection of k and omega with the
face-average normal velocity, the two-point normal derivative for their
diffusion and the cell sources of :mod:`pyaeroflow.physics.sst`.

Cell quantities that need gradients (strain rate, F1, F2, nu_t) are
evaluated from the least-squares gradients; the face diffusion coefficients
are the averages of the two cell values (SU2). At boundary faces the owner
cell's F1 is used, and nu_t is zero at walls.

The omega equation is solved multiplied by nu (row scaling, the solution is
unchanged): its residual then has the units of the k residual. Unscaled,
omega (up to 1e7 U/L next to a wall) would dominate the RMS residual norm
that decides acceptance and line search, so that the mean flow could not
make progress (SU2 solves turbulence apart from the mean flow and does not
need this).

Boundary values: k = 0 and omega = 60 nu / (beta_1 d1^2) at walls (stored in
``BoundaryData.turbulence`` by the driver, d1 the wall cell's height), the
prescribed (k, omega) at velocity inlets, total-pressure inlets and
inflowing farfield faces, the interior values elsewhere.
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
from pyaeroflow.physics import sst
from pyaeroflow.physics.incompressible import (
    FARFIELD,
    TOTAL_PRESSURE_INLET,
    VELOCITY_INLET,
    WALL,
    BoundaryData,
    U,
    V,
)
from pyaeroflow.physics.rans import (
    _dot_n,
    kinematic_viscosity,
    segregated_mask,
    strain_rate,
    upwind_scalar_flux,
)
from pyaeroflow.physics.rans import time_term_blocks as rans_time_term_blocks
from pyaeroflow.solver import divergence

__all__ = [
    "NK",
    "NW",
    "NT",
    "boundary_state",
    "residual",
    "residual_jvp",
    "limiter",
    "interior_flux",
    "boundary_flux",
    "jacobian_coefficients",
    "source_blocks",
    "time_term_blocks",
    "segregated_mask",
    "cell_quantities",
    "eddy_viscosity",
]

NK = 3  # column of k
NW = 4  # column of omega
NT = NK  # first turbulence column (the interface of physics.rans)
OMEGA_MIN = 1.0e-8  # floor of omega in denominators (non-dimensional)


def boundary_turbulence(bc, normal, w_b3, t_in):
    """(k, omega) on boundary faces from the mean boundary state and the interior."""
    vn = w_b3[:, U] * normal[:, 0] + w_b3[:, V] * normal[:, 1]
    kind = bc.kind
    prescribed = (
        (kind == WALL)
        | (kind == VELOCITY_INLET)
        | (kind == TOTAL_PRESSURE_INLET)
        | ((kind == FARFIELD) & (vn < 0.0))
    )
    return jnp.where(prescribed[:, None], bc.turbulence, t_in)


def boundary_state(bc, normal, w_in):
    """Boundary-face state (p, u, v, k, omega) from the interior."""
    w_b3 = inc.boundary_state(bc, normal, w_in[:, :3])
    t = boundary_turbulence(bc, normal, w_b3, w_in[:, NK:])
    return jnp.concatenate((w_b3, t), axis=1)


def _omega(w):
    return jnp.maximum(w[:, NW], OMEGA_MIN)


def cell_quantities(model, geom, w, grad):
    """Per-cell strain rate, F1, F2, nu_t and grad k . grad omega."""
    nu = kinematic_viscosity(model)
    k = jnp.maximum(w[:, NK], 0.0)
    omega = _omega(w)
    strain = strain_rate(grad[:, U : V + 1, :])
    kgw = jnp.sum(grad[:, NK, :] * grad[:, NW, :], axis=1)
    f1, f2 = sst.blending(k, omega, nu, geom.wall_distance, kgw)
    nu_t = sst.eddy_viscosity(k, omega, strain, f2)
    return strain, f1, f2, nu_t, kgw


def _face_coefficients(model, nu_t, f1, o, n):
    """mu_eff and the (k, omega) diffusion coefficients at faces, cell averages."""
    nu = kinematic_viscosity(model)
    dk, dw = sst.diffusion_coefficients(nu, nu_t, f1)
    mu = model.viscosity + model.density * 0.5 * (nu_t[o] + nu_t[n])
    dc = jnp.stack((0.5 * (dk[o] + dk[n]), 0.5 * (dw[o] + dw[n])), axis=1)
    return mu, dc


def _boundary_coefficients(model, bc, nu_t, f1, b):
    """Face coefficients at boundary faces: nu_t = 0 at walls, owner F1."""
    nu = kinematic_viscosity(model)
    nu_t_b = 0.5 * (nu_t[b] + jnp.where(bc.kind == WALL, 0.0, nu_t[b]))
    dk, dw = sst.diffusion_coefficients(nu, nu_t_b, f1[b])
    return model.viscosity + model.density * nu_t_b, jnp.stack((dk, dw), axis=1)


@partial(jax.jit, static_argnums=0)
def limiter(model, geom, bc, w):
    """Limiter values psi of the current state, shape (n_cells, 5)."""
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    return inc._limiter(model, geom, w, w_b, lsq_gradient(geom, w, w_b))


@partial(jax.jit, static_argnums=(0, 4))
def residual(model, geom, bc, w, order, beta2, psi=None, gamma=None):
    """Residual R(w), shape (n_cells, 5): net outflow minus sources.

    ``gamma`` is accepted for the interface of :mod:`pyaeroflow.physics.rans`
    and ignored (no transition model with SST).
    """
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    w_b = boundary_state(bc, geom.b_normal, w[b])
    grad = lsq_gradient(geom, w, w_b)
    t_order = model.turbulence_spatial_order
    if order == 2 or t_order == 2:
        if psi is None:
            psi = inc._limiter(model, geom, w, w_b, grad)
        w_l, w_r, w_in = reconstruct(geom, w, grad, psi)
    m_l, m_r, m_in = (
        (w_l[:, :3], w_r[:, :3], w_in[:, :3])
        if order == 2
        else (w[o, :3], w[n, :3], w[b, :3])
    )
    s_l, s_r, s_in = (
        (w_l[:, NK:], w_r[:, NK:], w_in[:, NK:])
        if t_order == 2
        else (w[o, NK:], w[n, NK:], w[b, NK:])
    )

    f_flux = inc.fds_flux(model, m_l, m_r, geom.f_normal, beta2)
    b_flux = inc.boundary_convective_flux(model, bc, geom.b_normal, m_in, beta2)
    vn_f = _dot_n(0.5 * (m_l[:, 1:] + m_r[:, 1:]), geom.f_normal)
    vn_b = _dot_n(w_b[:, 1:3], geom.b_normal)
    t_f = upwind_scalar_flux(vn_f[:, None], s_l, s_r)
    t_b = upwind_scalar_flux(vn_b[:, None], s_in, w_b[:, NK:])

    strain, f1, f2, nu_t, kgw = cell_quantities(model, geom, w, grad)
    g_f = interior_face_gradient(geom, w, grad, model.viscous_gradient, model.alpha)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    mu_f, dc_f = _face_coefficients(model, nu_t, f1, o, n)
    mu_b, dc_b = _boundary_coefficients(model, bc, nu_t, f1, b)
    f_flux = f_flux + inc.viscous_flux(model, g_f[:, 1:3], geom.f_normal, mu_f)
    b_flux = b_flux + inc.viscous_flux(model, g_b[:, 1:3], geom.b_normal, mu_b)
    # k and omega diffuse with the two-point normal derivative: omega ~ 1/d^2
    # next to a wall, and the least-squares gradients of such a profile are
    # so large that the alpha-damping correction reverses the sign of the
    # face derivative between the first and second cell (omega is then
    # pumped out of the second cell).
    dn_f = _dot_n(geom.f_delta, geom.f_normal)
    dn_b = _dot_n(geom.b_delta, geom.b_normal)
    t_f = t_f - dc_f * (w[n, NK:] - w[o, NK:]) / dn_f[:, None]
    t_b = t_b - dc_b * (w_b[:, NK:] - w[b, NK:]) / dn_b[:, None]

    r = divergence(
        geom,
        jnp.concatenate((f_flux, t_f), axis=1),
        jnp.concatenate((b_flux, t_b), axis=1),
    )
    switch = 1.0 if geom.production_switch is None else geom.production_switch
    s_k, s_w = sst.sources(w[:, NK], _omega(w), strain, f1, f2, kgw, switch)
    r = r.at[:, NK].add(-geom.area * s_k)
    r = r.at[:, NW].add(-geom.area * s_w)
    r = r.at[:, NW].multiply(kinematic_viscosity(model))  # row scaling
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
        return residual(model, geom, bc, x, order, beta2, psi)

    return jax.jvp(f, (w,), (dw,))[1]


def _row_scale(model):
    """Row factors of the (k, omega) residuals: (1, nu)."""
    return jnp.array([1.0, kinematic_viscosity(model)])


@partial(jax.jit, static_argnums=0)
def jacobian_coefficients(model, geom, bc, w):
    """Face viscosities of the first-order Jacobian, from the current state.

    Returns (mu_f, dc_f, mu_b, dc_b): the effective viscosity and the (k,
    omega) diffusion coefficients at interior and boundary faces, as in the
    residual. They enter the compact fluxes as given values (SU2 does not
    differentiate the diffusion coefficients either); a nu_t estimated from
    the two face states alone (k / omega) misses the SST limiter and is off
    by orders of magnitude in a laminar boundary layer.
    """
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    w_b = boundary_state(bc, geom.b_normal, w[b])
    grad = lsq_gradient(geom, w, w_b)
    _, f1, _, nu_t, _ = cell_quantities(model, geom, w, grad)
    mu_f, dc_f = _face_coefficients(model, nu_t, f1, o, n)
    mu_b, dc_b = _boundary_coefficients(model, bc, nu_t, f1, b)
    return mu_f, dc_f, mu_b, dc_b


def interior_flux(model, beta2, w_o, w_n, mu, dc, normal, delta):
    """Compact first-order flux through an interior face (for the Jacobian).

    ``mu`` and ``dc`` are the face's given viscosities
    (:func:`jacobian_coefficients`).
    """
    conv = inc.fds_flux(model, w_o[None, :3], w_n[None, :3], normal[None], beta2)[0]
    dn = jnp.dot(delta, normal)
    if model.viscous_gradient == "alpha-damping":
        coeff = model.alpha / dn
    else:
        coeff = dn / jnp.dot(delta, delta)
    g = (coeff * (w_n[1:3] - w_o[1:3]))[:, None] * normal[None, :]
    conv = conv + inc.viscous_flux(model, g[None], normal[None], mu[None])[0]
    vn = 0.5 * jnp.dot(w_o[1:3] + w_n[1:3], normal)
    flux = upwind_scalar_flux(vn, w_o[NK:], w_n[NK:]) - dc * (w_n[NK:] - w_o[NK:]) / dn
    return jnp.concatenate((conv, flux * _row_scale(model)))


def boundary_flux(model, beta2, w_o, bc_kind, bc_value, bc_turb, mu, dc, normal, delta):
    """Compact first-order flux through a boundary face (for the Jacobian)."""
    bc = BoundaryData(bc_kind[None], bc_value[None], model.density, None, bc_turb[None])
    w_b = boundary_state(bc, normal[None], w_o[None])[0]
    conv = inc.boundary_convective_flux(model, bc, normal[None], w_o[None, :3], beta2)[
        0
    ]
    dn = jnp.dot(delta, normal)
    g = ((w_b[1:3] - w_o[1:3]) / dn)[:, None] * normal[None, :]
    conv = conv + inc.viscous_flux(model, g[None], normal[None], mu[None])[0]
    vn = jnp.dot(w_b[1:3], normal)
    flux = upwind_scalar_flux(vn, w_o[NK:], w_b[NK:]) - dc * (w_b[NK:] - w_o[NK:]) / dn
    return jnp.concatenate((conv, flux * _row_scale(model)))


@partial(jax.jit, static_argnums=0)
def source_blocks(model, geom, bc, w):
    """Cell-local source Jacobian blocks (n_cells, 5, 5), SU2's implicit part.

    Only the destruction terms: d(beta* k omega)/d(k, omega) and
    d(beta omega^2)/d omega with the F1-blended beta, times the cell area
    (the residual carries minus the source). Production and cross diffusion
    are left to the exact Newton-Krylov JVP.
    """
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    _, f1, _, _, _ = cell_quantities(model, geom, w, grad)
    k = jnp.maximum(w[:, NK], 0.0)
    omega = _omega(w)
    beta = sst.blend(f1, sst.BETA_1, sst.BETA_2)
    a = geom.area
    blocks = jnp.zeros((w.shape[0], 5, 5))
    blocks = blocks.at[:, NK, NK].set(a * sst.BETA_STAR * omega)
    blocks = blocks.at[:, NK, NW].set(a * sst.BETA_STAR * k)
    nu = kinematic_viscosity(model)
    return blocks.at[:, NW, NW].set(nu * a * 2.0 * beta * omega)


@partial(jax.jit, static_argnums=0)
def time_term_blocks(model, geom, w, beta2, cfl):
    """Pseudo-time blocks with the omega row scaled by nu.

    Those of :func:`pyaeroflow.physics.rans.time_term_blocks`, row-scaled
    like the omega residual.
    """
    g = rans_time_term_blocks(model, geom, w, beta2, cfl)
    return g.at[:, NW, NW].multiply(kinematic_viscosity(model))


@partial(jax.jit, static_argnums=0)
def _fields(model, geom, bc, w):
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    _, f1, f2, nu_t, _ = cell_quantities(model, geom, w, grad)
    return nu_t, f1, f2


def eddy_viscosity(model, geom, bc, w):
    """Kinematic eddy viscosity per cell."""
    return _fields(model, geom, bc, w)[0]


def blending_f1(model, geom, bc, w):
    """Menter's first blending function per cell."""
    return _fields(model, geom, bc, w)[1]

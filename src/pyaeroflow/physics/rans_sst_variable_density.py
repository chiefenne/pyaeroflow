"""RANS with the incompressible energy equation: variable density, k-omega SST (V2).

State per cell: (p, u, v, k, omega[, gamma, Re_thetat], T): the columns of
:mod:`pyaeroflow.physics.rans_sst` with T appended (energy block last).

Mean flow and energy as in :mod:`pyaeroflow.physics.rans_variable_density`:
the variable-density fluxes of :mod:`pyaeroflow.physics.variable_density`
with mu(T) + mu_t in the stress and k(T) + c_p mu_t / Pr_t in the heat flux
(pluggable Pr_t law). The isotropic -2/3 rho k part of the Boussinesq stress
stays in the pressure, as in the constant-density SST.

Turbulence: the conservative SST-2003m of the NASA Turbulence Modeling
Resource (as :mod:`pyaeroflow.physics.rans_sst_compressible`),

    div(rho v k)     = rho P_k - beta* rho k omega
                       + div((mu + sigma_k mu_t) grad k)
    div(rho v omega) = rho P_omega - beta rho omega^2 + rho CD
                       + div((mu + sigma_omega mu_t) grad omega),

the kinematic sources of :mod:`pyaeroflow.physics.sst` times the local
density, nu = mu(T) / rho(T) in the blending functions and the production
with its -2/3 k div v part, div v = -rho_T v . grad T / rho of the steady
low-Mach continuity equation (zero at constant density). Langtry-Menter:
the gamma and Re_thetat rows are rho times their kinematic sources with
div(rho D grad phi) diffusion.

Discretization as :mod:`pyaeroflow.physics.rans_sst`: k and omega (and the
LM variables) upwinded with the face mass flux rho_f v_n,f, two-point normal
derivative for the diffusion with the face mean of rho (nu + sigma nu_t) of
the cells, the omega and Re_thetat rows scaled by the reference nu and
RET_SCALE. Wall omega = 60 nu_w / (beta_1 d1^2) with nu_w = mu(T_w) /
rho(T_w) of the wall-face state (``BoundaryData.turbulence`` holds
60 / (beta_1 d1^2)). With ``density_model = constant`` at rho = 1 and
constant properties the flow and turbulence rows are those of
:mod:`pyaeroflow.physics.rans_sst` up to the round-off of nu_w.
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
from pyaeroflow.physics import lm, sst
from pyaeroflow.physics import rans_sst as base
from pyaeroflow.physics import rans_variable_density as rvd
from pyaeroflow.physics import thermo as th
from pyaeroflow.physics import variable_density as vd
from pyaeroflow.physics.incompressible import WALL, U, V
from pyaeroflow.physics.rans import _dot_n, strain_rate, upwind_scalar_flux
from pyaeroflow.physics.rans_variable_density import (
    assemble,
    bc_species,
    boundary_distance,
    boundary_energy_viscous_flux,
    cell_properties,
    energy_viscous_flux,
    face_distance,
    mean_density,
    mean_state,
    n_species,
    state_properties,
    t_column,
    thermal_state,
)
from pyaeroflow.solver import divergence

FLOW = vd.FLOW
NK, NW, NT = base.NK, base.NW, base.NT
NGL, NRT = base.NGL, base.NRT
RET_SCALE = base.RET_SCALE
OMEGA_MIN = base.OMEGA_MIN
RESIDUAL_NAMES = vd.RESIDUAL_NAMES
CELL_SOURCES = True

build_boundary_data = vd.build_boundary_data
lsq_include = vd.lsq_include
scheme_parameters = vd.scheme_parameters
wall_traction = rvd.wall_traction
_row_scale = base._row_scale  # (1, nu_ref[, 1, RET_SCALE]): reference nu


def boundary_turbulence(bc, normal, w_b4, t_in):
    """Turbulence columns on boundary faces; wall omega with nu(T_w)."""
    t = base.boundary_turbulence(bc, normal, w_b4[:, :3], t_in)
    _, nu_w = cell_properties(
        bc.thermo, w_b4[:, 3], w_b4[:, 4:] if bc_species(bc) else None
    )
    wall = bc.kind == WALL
    return t.at[:, 1].set(jnp.where(wall, t[:, 1] * nu_w, t[:, 1]))


def boundary_state(bc, normal, w_in):
    """Boundary-face state (p, u, v, k, omega[, gamma, Re_thetat], T[, Y])."""
    w_b4 = rvd.mean_boundary_state(bc, normal, w_in)
    tc = t_column(w_in, bc_species(bc))
    return assemble(w_b4, boundary_turbulence(bc, normal, w_b4, w_in[:, NK:tc]))


def _omega(w):
    return jnp.maximum(w[:, NW], OMEGA_MIN)


def _cell_state(model, geom, w, grad):
    """rho, nu and the SST cell quantities (strain, F1, F2, nu_t, kgw) per cell."""
    rho, nu = state_properties(model, w, n_species(model))
    return (rho, nu) + tuple(base.cell_quantities(model, geom, w, grad, nu))


def _divergence(model, w, grad):
    """Velocity divergence -v . grad rho / rho of the steady continuity.

    grad rho = rho_T grad T (+ sum rho_Y grad Y with species).
    """
    ns = n_species(model)
    tc = t_column(w, ns)
    t, y = thermal_state(w, ns)
    rho_t = th.density_derivative(model.thermo, t, y)
    v_grad_rho = rho_t * (w[:, U] * grad[:, tc, 0] + w[:, V] * grad[:, tc, 1])
    if y is not None:
        rho_y = th.density_species_derivative(model.thermo, t, y)
        v_grad_y = (
            w[:, U, None] * grad[:, tc + 1 :, 0] + w[:, V, None] * grad[:, tc + 1 :, 1]
        )
        v_grad_rho = v_grad_rho + jnp.sum(rho_y * v_grad_y, axis=1)
    return -v_grad_rho / th.density(model.thermo, t, y)


def _face_coefficients(model, nu_t, f1, rho, nu, o, n):
    """Face mu_t and turbulence diffusion coefficients: means of rho nu_t, rho D."""
    coeffs = base._cell_diffusion(model, nu, nu_t, f1)
    mu_t = 0.5 * (rho[o] * nu_t[o] + rho[n] * nu_t[n])
    dc = jnp.stack([0.5 * (rho[o] * c[o] + rho[n] * c[n]) for c in coeffs], axis=1)
    return mu_t, dc


def _boundary_coefficients(model, bc, nu_t, f1, rho, nu, b):
    """Boundary-face mu_t and coefficients: nu_t = 0 at walls, owner F1, rho, nu."""
    nu_t_b = 0.5 * (nu_t[b] + jnp.where(bc.kind == WALL, 0.0, nu_t[b]))
    coeffs = base._cell_diffusion(model, nu[b], nu_t_b, f1[b])
    return rho[b] * nu_t_b, rho[b][:, None] * jnp.stack(coeffs, axis=1)


@partial(jax.jit, static_argnums=0)
def limiter(model, geom, bc, w):
    """Limiter values psi of the current state, shape (n_cells, n_var)."""
    bc = vd.closed_boundary(vd.closed_model(model, geom, w), bc)
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    return inc._limiter(model, geom, w, w_b, lsq_gradient(geom, w, w_b))


_cell_state_remat = jax.checkpoint(_cell_state, static_argnums=(0,))
_sst_sources_remat = jax.checkpoint(sst.sources, static_argnums=(9,))


def _lm_sources_remat(*args, positive, smooth=0.0):
    return jax.checkpoint(partial(lm.sources, positive=positive, smooth=smooth))(*args)


@partial(jax.jit, static_argnums=(0, 4))
def residual(model, geom, bc, w, order, beta2, psi=None, gamma=None):
    """Residual R(w), shape (n_cells, n_var): net outflow minus sources.

    Parameters as in :func:`pyaeroflow.physics.rans_sst.residual` (``gamma``
    is ignored). Closed domains: the density follows p_th(w).
    """
    cell_state = _cell_state_remat
    if model.closed_mass is not None:
        model = vd.closed_model(model, geom, w)
        bc = vd.closed_boundary(model, bc)
        # the model now holds a traced p_th: close over it (not static)
        cell_state = jax.checkpoint(partial(_cell_state, model))
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    ns = n_species(model)
    tc = t_column(w, ns)
    w_b = boundary_state(bc, geom.b_normal, w[b])
    grad = lsq_gradient(geom, w, w_b)
    t_order = model.turbulence_spatial_order
    if order == 2 or t_order == 2:
        if psi is None:
            psi = inc._limiter(model, geom, w, w_b, grad)
        w_l, w_r, w_in = reconstruct(geom, w, grad, psi)
    m_l, m_r, m_in = (
        (mean_state(w_l, ns), mean_state(w_r, ns), mean_state(w_in, ns))
        if order == 2
        else (mean_state(w[o], ns), mean_state(w[n], ns), mean_state(w[b], ns))
    )
    s_l, s_r, s_in = (
        (w_l[:, NK:tc], w_r[:, NK:tc], w_in[:, NK:tc])
        if t_order == 2
        else (w[o, NK:tc], w[n, NK:tc], w[b, NK:tc])
    )
    f_flux = vd.fds_flux(model, m_l, m_r, geom.f_normal, beta2)
    b_flux = vd.boundary_convective_flux(model, bc, geom.b_normal, m_in, beta2)
    vn_f = _dot_n(0.5 * (m_l[:, 1:3] + m_r[:, 1:3]), geom.f_normal)
    mass_f = 0.5 * (mean_density(model, m_l) + mean_density(model, m_r)) * vn_f
    mass_b = state_properties(model, w_b, ns)[0] * _dot_n(w_b[:, 1:3], geom.b_normal)
    t_f = upwind_scalar_flux(mass_f[:, None], s_l, s_r)
    t_b = upwind_scalar_flux(mass_b[:, None], s_in, w_b[:, NK:tc])

    rho, nu, strain, f1, f2, nu_t, kgw = (
        cell_state(geom, w, grad)
        if model.closed_mass is not None
        else cell_state(model, geom, w, grad)
    )
    g_f = interior_face_gradient(geom, w, grad, model.viscous_gradient, model.alpha)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    mu_tf, dc_f = _face_coefficients(model, nu_t, f1, rho, nu, o, n)
    mu_tb, dc_b = _boundary_coefficients(model, bc, nu_t, f1, rho, nu, b)
    temp_f = 0.5 * (w[o, tc] + w[n, tc])
    f_flux = f_flux + energy_viscous_flux(
        model,
        g_f[:, 1:3],
        g_f[:, tc],
        temp_f,
        mu_tf,
        face_distance(geom),
        geom.f_normal,
        grad_y=g_f[:, tc + 1 :],
        y_f=0.5 * (w[o, tc + 1 :] + w[n, tc + 1 :]) if ns else None,
    )
    b_flux = b_flux + boundary_energy_viscous_flux(
        model,
        bc,
        g_b[:, 1:3],
        g_b[:, tc],
        w_b[:, tc],
        mu_tb,
        boundary_distance(geom, bc),
        geom.b_normal,
        grad_y=g_b[:, tc + 1 :],
        y_b=w_b[:, tc + 1 :] if ns else None,
    )
    dn_f = _dot_n(geom.f_delta, geom.f_normal)
    dn_b = _dot_n(geom.b_delta, geom.b_normal)
    t_f = t_f - dc_f * (w[n, NK:tc] - w[o, NK:tc]) / dn_f[:, None]
    t_b = t_b - dc_b * (w_b[:, NK:tc] - w[b, NK:tc]) / dn_b[:, None]

    r = divergence(geom, assemble(f_flux, t_f), assemble(b_flux, t_b))
    switch = 1.0 if geom.production_switch is None else geom.production_switch
    g_eff = None
    a_rho = geom.area * rho
    if model.transition == "lm":
        args = base._lm_arguments(model, geom, w, grad, strain, nu)
        smooth = model.lm_jacobian_smoothing
        g_eff = lm.effective_intermittency(*args[:9], smooth=smooth)
        s_g, s_t = _lm_sources_remat(
            *args, positive=model.lm_positive_jacobian, smooth=smooth
        )
        r = r.at[:, NGL].add(-a_rho * s_g)
        r = r.at[:, NRT].add(-a_rho * s_t)
    s_k, s_w = _sst_sources_remat(
        w[:, NK],
        _omega(w),
        strain,
        f1,
        f2,
        kgw,
        switch,
        g_eff,
        _divergence(model, w, grad),
        model.lm_jacobian_smoothing if model.transition == "lm" else 0.0,
    )
    r = r.at[:, NK].add(-a_rho * s_k)
    r = r.at[:, NW].add(-a_rho * s_w)
    r = r.at[:, NK:tc].multiply(_row_scale(model)[None, :])  # row scaling
    r = r.at[:, U : V + 1].add(-vd.cell_sources(model, geom, mean_state(w, ns)))
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


@partial(jax.jit, static_argnums=(0, 4))
def residual_linearize(model, geom, bc, w, order, beta2, psi=None, gamma=None):
    """Residual and its linearization at ``w`` (``jax.linearize``)."""

    def f(x):
        return residual(model, geom, bc, x, order, beta2, psi)

    return jax.linearize(f, w)


# -- first-order Jacobian ----------------------------------------------------------


@partial(jax.jit, static_argnums=0)
def jacobian_coefficients(model, geom, bc, w):
    """Face mu_t and diffusion coefficients (mu_tf, dc_f, mu_tb, dc_b) of ``w``."""
    model = vd.closed_model(model, geom, w)
    bc = vd.closed_boundary(model, bc)
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    w_b = boundary_state(bc, geom.b_normal, w[b])
    grad = lsq_gradient(geom, w, w_b)
    rho, nu, _, f1, _, nu_t, _ = _cell_state(model, geom, w, grad)
    mu_tf, dc_f = _face_coefficients(model, nu_t, f1, rho, nu, o, n)
    mu_tb, dc_b = _boundary_coefficients(model, bc, nu_t, f1, rho, nu, b)
    return mu_tf, dc_f, mu_tb, dc_b


def jacobian_arguments(model, geom, bc, w):
    """Per-face arguments (f_args, b_args) of the compact fluxes.

    The eddy viscosity and the turbulence diffusion coefficients are given
    values from the current state (as in :mod:`pyaeroflow.physics.rans_sst`).
    """
    mu_tf, dc_f, mu_tb, dc_b = jacobian_coefficients(model, geom, bc, w)
    b_args = vd.jacobian_boundary_args(bc)[:5] + (
        bc.turbulence,
        mu_tb,
        dc_b,
        boundary_distance(geom, bc),
    )
    return (mu_tf, dc_f, face_distance(geom)), b_args + vd.species_boundary_args(bc)


def interior_flux(model, beta2, w_o, w_n, mu_t, dc, distance, normal, delta):
    """Compact first-order flux through an interior face (for the Jacobian)."""
    ns = n_species(model)
    tc = t_column(w_o, ns)
    conv = vd.fds_flux(
        model,
        mean_state(w_o, ns)[None],
        mean_state(w_n, ns)[None],
        normal[None],
        beta2,
    )[0]
    dn = jnp.dot(delta, normal)
    if model.viscous_gradient == "alpha-damping":
        coeff = model.alpha / dn
    else:
        coeff = dn / jnp.dot(delta, delta)
    g = (coeff * (w_n - w_o))[:, None] * normal[None, :]
    t_f = 0.5 * (w_o[tc] + w_n[tc])
    visc = energy_viscous_flux(
        model,
        g[None, 1:3],
        g[None, tc],
        t_f[None],
        mu_t[None],
        distance[None],
        normal[None],
        grad_y=g[None, tc + 1 :],
        y_f=0.5 * (w_o[tc + 1 :] + w_n[tc + 1 :])[None] if ns else None,
    )[0]
    rho_f = 0.5 * (
        state_properties(model, w_o, ns)[0] + state_properties(model, w_n, ns)[0]
    )
    mass = rho_f * 0.5 * jnp.dot(w_o[1:3] + w_n[1:3], normal)
    flux = (
        upwind_scalar_flux(mass, w_o[NK:tc], w_n[NK:tc])
        - dc * (w_n[NK:tc] - w_o[NK:tc]) / dn
    )
    return assemble(conv + visc, flux * _row_scale(model))


def boundary_flux(
    model,
    beta2,
    w_o,
    bc_kind,
    bc_value,
    bc_thermal,
    bc_heat,
    bc_rho,
    bc_turb,
    mu_t,
    dc,
    distance,
    *args,
):
    """Compact first-order flux through a boundary face (for the Jacobian).

    ``args``: the species arrays of the face (with species only), the face
    normal and the vector from the cell centre to the face.
    """
    *species, normal, delta = args
    ns = n_species(model)
    tc = t_column(w_o, ns)
    bc = vd.BoundaryData(
        bc_kind[None],
        bc_value[None],
        bc_thermal[None],
        bc_heat[None],
        bc_rho[None],
        model.density,
        None,
        bc_turb[None],
        th.StaticThermo(model.thermo),
    )
    bc = vd.with_species_arguments(bc, tuple(species))
    w_b = boundary_state(bc, normal[None], w_o[None])[0]
    conv = vd.boundary_convective_flux(
        model, bc, normal[None], mean_state(w_o, ns)[None], beta2
    )[0]
    dn = jnp.dot(delta, normal)
    g = ((w_b - w_o) / dn)[:, None] * normal[None, :]
    visc = boundary_energy_viscous_flux(
        model,
        bc,
        g[None, 1:3],
        g[None, tc],
        w_b[tc][None],
        mu_t[None],
        distance[None],
        normal[None],
        grad_y=g[None, tc + 1 :],
        y_b=w_b[tc + 1 :][None] if ns else None,
    )[0]
    mass = state_properties(model, w_b, ns)[0] * jnp.dot(w_b[1:3], normal)
    flux = (
        upwind_scalar_flux(mass, w_o[NK:tc], w_b[NK:tc])
        - dc * (w_b[NK:tc] - w_o[NK:tc]) / dn
    )
    return assemble(conv + visc, flux * _row_scale(model))


def _lm_cell_sources(model, smooth):
    def cell(g, t, *rest):
        return jnp.stack(lm.sources(g, t, *rest, smooth=smooth))

    return cell


@partial(jax.jit, static_argnums=0)
def source_blocks(model, geom, bc, w):
    """Cell-local source blocks (n_cells, n_var, n_var) of the first-order Jacobian.

    Those of :func:`pyaeroflow.physics.rans_sst.source_blocks` times the
    local density, with the local nu, and the T derivative of the buoyancy.
    """
    model = vd.closed_model(model, geom, w)
    bc = vd.closed_boundary(model, bc)
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    rho, nu, strain, f1, _, _, _ = _cell_state(model, geom, w, grad)
    k = jnp.maximum(w[:, NK], 0.0)
    omega = _omega(w)
    beta = sst.blend(f1, sst.BETA_1, sst.BETA_2)
    a = geom.area * rho
    n_var = w.shape[1]
    d_fac = 1.0
    blocks = jnp.zeros((w.shape[0], n_var, n_var))
    if model.transition == "lm":
        args = base._lm_arguments(model, geom, w, grad, strain, nu)
        smooth = model.lm_jacobian_smoothing
        d_fac = lm.destruction_factor(lm.effective_intermittency(*args[:9]))
        cell = _lm_cell_sources(model, smooth)
        d_g = jax.vmap(jax.grad(lambda *x: cell(*x)[0], argnums=0))(*args)
        d_t = jax.vmap(jax.grad(lambda *x: cell(*x)[1], argnums=1))(*args)
        blocks = blocks.at[:, NGL, NGL].set(-a * jnp.minimum(d_g, 0.0))
        blocks = blocks.at[:, NRT, NRT].set(-a * RET_SCALE * jnp.minimum(d_t, 0.0))
    blocks = blocks.at[:, NK, NK].set(a * d_fac * sst.BETA_STAR * omega)
    blocks = blocks.at[:, NK, NW].set(a * d_fac * sst.BETA_STAR * k)
    nu_ref = base.kinematic_viscosity(model)
    blocks = blocks.at[:, NW, NW].set(nu_ref * a * 2.0 * beta * omega)
    return rvd.buoyancy_blocks(model, geom, w, blocks)


@partial(jax.jit, static_argnums=0)
def source_jacobian(model, geom, bc, w):
    """Cell-local d(-area row_scale rho src)/d(turbulence columns, T).

    Shape (n, n_t + 1, n_t + 1); as
    :func:`pyaeroflow.physics.rans_sst.source_jacobian` with the local rho
    and nu, the T row and column zero.
    """
    model = vd.closed_model(model, geom, w)
    bc = vd.closed_boundary(model, bc)
    tc = t_column(w, n_species(model))
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    rho, nu, strain, f1, f2, _, kgw = _cell_state(model, geom, w, grad)
    switch = 1.0 if geom.production_switch is None else geom.production_switch
    switch = jnp.broadcast_to(jnp.asarray(switch, dtype=float), strain.shape)
    vort, speed, du_ds = base._flow_inputs(w, grad)
    div = _divergence(model, w, grad)
    scale = _row_scale(model)
    lm_on = model.transition == "lm"

    def cell(t, s, a, b, c, sw, vo, d, sp, dd, nu_c, dv):
        k, om = t[0], jnp.maximum(t[1], OMEGA_MIN)
        g_eff, extra = None, []
        if lm_on:
            largs = (t[2], t[3], jnp.maximum(k, 0.0), om, nu_c, s, vo, d, sp, dd)
            g_eff = lm.effective_intermittency(*largs[:9])
            extra = list(lm.sources(*largs))
        s_k, s_w = sst.sources(k, om, s, a, b, c, sw, g_eff, dv)
        return -jnp.stack([s_k, s_w] + extra) * scale

    jac = jax.vmap(jax.jacfwd(cell))(
        w[:, NK:tc],
        strain,
        f1,
        f2,
        kgw,
        switch,
        vort,
        geom.wall_distance,
        speed,
        du_ds,
        nu,
        div,
    )
    jac = (geom.area * rho)[:, None, None] * jac
    pad = 1 + n_species(model)
    return jnp.pad(jac, ((0, 0), (0, pad), (0, pad)))


@partial(jax.jit, static_argnums=0)
def time_term_blocks(model, geom, w, beta2, cfl):
    """Pseudo-time blocks: Gamma on (p, u, v, T), rho row_scale on turbulence."""
    n_var = w.shape[1]
    ns = n_species(model)
    gamma, lam = rvd.mean_time_blocks(model, geom, w, beta2)
    idx = rvd.mean_index(n_var, ns)
    g = jnp.zeros((w.shape[0], n_var, n_var))
    g = g.at[:, idx[:, None], idx[None, :]].set(gamma)
    k = jnp.arange(NK, t_column(w, ns))
    rho = state_properties(vd.closed_model(model, geom, w), w, ns)[0]
    g = g.at[:, k, k].set(rho[:, None] * _row_scale(model)[None, :])
    return g * (lam / cfl)[:, None, None]


def segregated_mask(n_var, ns=0):
    """Block mask of the segregated coupling: (p, u, v, T, Y) and turbulence apart."""
    return rvd.mean_segregated_mask(n_var, NK, ns)


# -- output ------------------------------------------------------------------------


@partial(jax.jit, static_argnums=0)
def _fields(model, geom, bc, w):
    model = vd.closed_model(model, geom, w)
    bc = vd.closed_boundary(model, bc)
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    _, _, _, f1, f2, nu_t, _ = _cell_state(model, geom, w, grad)
    return nu_t, f1, f2


def eddy_viscosity(model, geom, bc, w):
    """Kinematic eddy viscosity per cell."""
    return _fields(model, geom, bc, w)[0]


def blending_f1(model, geom, bc, w):
    """Menter's first blending function per cell."""
    return _fields(model, geom, bc, w)[1]


@partial(jax.jit, static_argnums=0)
def intermittency_field(model, geom, bc, w):
    """Transported intermittency gamma per cell (ones without Langtry-Menter)."""
    if model.transition != "lm":
        return jnp.ones(w.shape[0])
    return w[:, NGL]


@partial(jax.jit, static_argnums=0)
def effective_intermittency_field(model, geom, bc, w):
    """gamma_eff = max(gamma, gamma_sep) per cell (Langtry-Menter)."""
    model = vd.closed_model(model, geom, w)
    bc = vd.closed_boundary(model, bc)
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    strain = strain_rate(grad[:, U : V + 1, :])
    _, nu = state_properties(model, w, n_species(model))
    return base.effective_intermittency(model, geom, w, grad, strain, nu)


class _SST:
    """Boundary pieces of SST for the wall heat flux and temperature."""

    boundary_state = staticmethod(boundary_state)

    @staticmethod
    def boundary_eddy_viscosity(model, geom, bc, w, w_b, grad):
        rho, nu, _, f1, _, nu_t, _ = _cell_state(model, geom, w, grad)
        return _boundary_coefficients(model, bc, nu_t, f1, rho, nu, geom.b_owner)[0]


@partial(jax.jit, static_argnums=0)
def wall_heat_flux(model, geom, bc, w):
    """Heat flux into the fluid per boundary face (n_bfaces,), k_eff dT/dn."""
    return rvd.heat_flux_and_temperature(model, geom, bc, w, _SST)[0]


@partial(jax.jit, static_argnums=0)
def wall_temperature(model, geom, bc, w):
    """Temperature per boundary face (n_bfaces,)."""
    return rvd.heat_flux_and_temperature(model, geom, bc, w, _SST)[1]


@partial(jax.jit, static_argnums=0)
def wall_species_flux(model, geom, bc, w):
    """Species mass fluxes and the net mass flux into the fluid per boundary face."""
    j, m, _ = rvd.species_wall_values(model, geom, bc, w, _SST)
    return j, m


@partial(jax.jit, static_argnums=0)
def wall_mass_fraction(model, geom, bc, w):
    """Mass fractions per boundary face, (n_bfaces, n_species)."""
    return rvd.species_wall_values(model, geom, bc, w, _SST)[2]

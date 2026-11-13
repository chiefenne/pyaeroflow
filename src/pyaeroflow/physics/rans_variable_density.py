"""RANS with the incompressible energy equation: variable density, SA family (V2).

State per cell: (p, u, v, nu_tilde[, gamma_tilde, n], T), the layout of
:func:`pyaeroflow.physics.layout.incompressible_layout` with the energy block
last, so that the turbulence and transition columns keep their numbers of
the constant-density solver (:mod:`pyaeroflow.physics.rans`).

Mean flow and energy: the fluxes of :mod:`pyaeroflow.physics.variable_density`
(rho(T), h(T), the 4x4 preconditioned flux-difference splitting, buoyancy),
with the effective viscosity mu(T) + mu_t in the stress and the effective
conductivity

    k_eff = k(T) + c_p(T) mu_t / Pr_t

in the heat flux, Pr_t from the pluggable law of
:func:`pyaeroflow.physics.thermo.turbulent_prandtl` (constant 0.9 by
default) evaluated at the faces with the face state and wall distance.

Turbulence: the conservative form of the incompressible equations with the
local density (that of the compressible SA of the NASA Turbulence Modeling
Resource, :mod:`pyaeroflow.physics.rans_compressible`):

    div(rho v nu_t) = rho (P - D) + (1/sigma) [div(rho (nu + nu_t f_n) grad nu_t)
                      + c_b2 rho |grad nu_t|^2]
                      - (1/sigma) (nu + nu_t f_n) grad rho . grad nu_t,

nu_t standing for nu_tilde and nu = mu(T) / rho(T) the local kinematic
viscosity, so that the row is rho times the kinematic equation. The SA-AFT
rows (gamma_tilde, n) are rho times their kinematic sources with the
diffusion div(rho D grad phi); the BCM intermittency uses the local nu.
grad rho = rho_T grad T.

Discretization as in :mod:`pyaeroflow.physics.rans`: the turbulence columns
are upwinded with the face mass flux rho_f v_n,f (rho_f the mean of the two
reconstructed face densities, v_n,f the mean normal velocity) and diffused
with rho_f times the kinematic coefficient of the face-mean nu_tilde and nu;
the face eddy viscosity mu_t is the mean of rho nu_t of the two cells. With
``density_model = constant`` at rho = 1 and constant properties the flow and
turbulence rows are those of :mod:`pyaeroflow.physics.rans` (to round-off),
and T is a passive scalar with the Reynolds-analogy conductivity.

Boundary values: those of :mod:`pyaeroflow.physics.rans` for the turbulence
columns, those of :mod:`pyaeroflow.physics.variable_density` for (p, u, v, T).
At boundary faces mu_t is the mean of the owner value and the boundary-state
value (zero at walls, as the momentum stress of the constant-density solver).

Pseudo time: Gamma of :mod:`pyaeroflow.physics.variable_density` on
(p, u, v, T) and rho on the turbulence rows (d(rho phi)/d phi), times the
spectral radius over CFL.
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
from pyaeroflow.physics import aft, rans
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import spalart_allmaras as sa
from pyaeroflow.physics import thermo as th
from pyaeroflow.physics import variable_density as vd
from pyaeroflow.physics.incompressible import (
    FARFIELD,
    SYMMETRY,
    TOTAL_PRESSURE_INLET,
    VELOCITY_INLET,
    WALL,
    U,
    V,
)
from pyaeroflow.physics.rans import _dot_n, _dot_n_cols, upwind_scalar_flux
from pyaeroflow.solver import divergence

FLOW = vd.FLOW
NT = rans.NT  # column of nu_tilde
NG = rans.NG  # log-intermittency (SA-AFT)
NN = rans.NN  # amplification factor (SA-AFT)
RESIDUAL_NAMES = vd.RESIDUAL_NAMES
CELL_SOURCES = True

build_boundary_data = vd.build_boundary_data
lsq_include = vd.lsq_include
scheme_parameters = vd.scheme_parameters


# -- shared pieces of the variable-density RANS modules (also used by SST) ---------


def t_column(w):
    """Column of T: the last one (static, from the shape)."""
    return w.shape[-1] - 1


def mean_state(w):
    """(p, u, v, T) of full states (..., n_var), the variable-density flow state."""
    return jnp.concatenate((w[..., :3], w[..., -1:]), axis=-1)


def assemble(mean, turb):
    """Full rows (..., n_var) from the (p, u, v, T) part and the turbulence part."""
    return jnp.concatenate((mean[..., :3], turb, mean[..., 3:]), axis=-1)


def mean_index(n_var):
    """Columns of (p, u, v, T) in a state of ``n_var`` columns."""
    return jnp.array([0, 1, 2, n_var - 1])


def cell_properties(model, t):
    """Density rho and nu = mu / rho at temperatures ``t`` (any shape)."""
    rho = th.density(model.thermo, t)
    return rho, th.viscosity(model.thermo, t) / rho


def mean_boundary_state(bc, normal, w_in):
    """(p, u, v, T) on boundary faces (variable-density boundary rules)."""
    return vd.boundary_state(bc, normal, mean_state(w_in))


def inflow_mask(bc, normal, w_b4):
    """Faces where the turbulence values are prescribed (inlets, inflow farfield)."""
    vn = w_b4[:, U] * normal[:, 0] + w_b4[:, V] * normal[:, 1]
    kind = bc.kind
    return (
        (kind == VELOCITY_INLET)
        | (kind == TOTAL_PRESSURE_INLET)
        | ((kind == FARFIELD) & (vn < 0.0))
    )


def energy_viscous_flux(model, grad_v, grad_t, t_f, mu_t, distance, normal, heat=True):
    """-(0, tau.n, k_eff grad T.n) with the eddy viscosity ``mu_t`` (dynamic).

    ``grad_v`` (n, 2, 2) and ``grad_t`` (n, 2) are face gradients, ``t_f``
    the face temperature, ``distance`` the face wall distance (an input of
    the Pr_t law). ``heat`` False drops the conduction (bool or array).
    """
    tm = model.thermo
    mu = th.viscosity(tm, t_f) + mu_t
    flow = inc.viscous_flux(model, grad_v, normal, mu)
    k = th.effective_conductivity(
        tm, model.turbulent_prandtl, t_f, th.density(tm, t_f), mu_t, distance
    )
    q = k * (grad_t[:, 0] * normal[:, 0] + grad_t[:, 1] * normal[:, 1])
    q = jnp.where(heat, q, 0.0)
    return jnp.concatenate((flow, -q[:, None]), axis=1)


def boundary_energy_viscous_flux(model, bc, grad_v, grad_t, t_b, mu_t, d_b, normal):
    """-F_v.n through boundary faces (n_bfaces, 4), thermal rules of V1.

    Conduction (with the eddy conductivity of ``mu_t``) at isothermal walls,
    inlets, farfields and outlets, the prescribed heat flux at heat-flux
    walls, nothing at adiabatic walls and symmetry faces.
    """
    k = bc.kind
    wall = k == WALL
    conduct = ~((wall & (bc.thermal != vd.ISOTHERMAL)) | (k == SYMMETRY))
    flux = energy_viscous_flux(
        model, grad_v, grad_t, t_b, mu_t, d_b, normal, heat=conduct
    )
    prescribed = wall & (bc.thermal == vd.HEAT_FLUX)
    return flux.at[:, 3].set(jnp.where(prescribed, -bc.heat_flux, flux[:, 3]))


def boundary_distance(geom, bc):
    """Wall distance at boundary faces: 0 at walls, the owner's elsewhere."""
    return jnp.where(bc.kind == WALL, 0.0, geom.wall_distance[geom.b_owner])


def face_distance(geom):
    """Wall distance at interior faces: mean of the two cells."""
    d = geom.wall_distance
    return 0.5 * (d[geom.f_owner] + d[geom.f_neigh])


def mean_time_blocks(model, geom, w, beta2):
    """Gamma on (p, u, v, T) and the spectral radius: ((n, 4, 4), (n,))."""
    m = mean_state(w)
    return (
        vd.preconditioner(model, m, beta2),
        vd.spectral_radius(model, geom, m, beta2),
    )


def mean_segregated_mask(n_var, first):
    """Segregated block mask: (p, u, v, T) apart from columns first..n_var-2."""
    idx = mean_index(n_var)
    mask = jnp.zeros((n_var, n_var))
    mask = mask.at[idx[:, None], idx[None, :]].set(1.0)
    return mask.at[first : n_var - 1, first : n_var - 1].set(1.0)


def buoyancy_blocks(model, geom, w, blocks):
    """Add d(-area (rho - rho_op) g)/dT to the (u, v; T) entries of ``blocks``."""
    tc = t_column(w)
    rho_t = th.density_derivative(model.thermo, w[:, tc])
    g = jnp.asarray(model.gravity, dtype=w.dtype)
    blocks = blocks.at[:, U, tc].add(-geom.area * rho_t * g[0])
    return blocks.at[:, V, tc].add(-geom.area * rho_t * g[1])


def wall_traction(model, grad, w_b, normal):
    """tau.n on boundary faces (n, 2) with mu of the boundary temperature."""
    mu = th.viscosity(model.thermo, w_b[:, -1])
    return -inc.viscous_flux(model, grad[:, U : V + 1], normal, mu)[:, 1:]


# -- SA family ---------------------------------------------------------------------


def boundary_state(bc, normal, w_in):
    """Boundary-face state (p, u, v, nu_tilde[, gamma_tilde, n], T)."""
    w_b4 = mean_boundary_state(bc, normal, w_in)
    nt = rans.boundary_nu_tilde(bc, normal, w_b4[:, :3], w_in[:, NT])
    cols = [nt[:, None]]
    if w_in.shape[1] > NT + 2:  # SA-AFT
        inflow = inflow_mask(bc, normal, w_b4)
        cols.append(jnp.where(inflow[:, None], 0.0, w_in[:, NT + 1 : -1]))
    return assemble(w_b4, jnp.concatenate(cols, axis=1))


def face_coefficients(model, t_a, t_b, rho_a, rho_b, nu_a, nu_b):
    """Eddy viscosity mu_t and the turbulence diffusion coefficients at faces.

    ``t_a``, ``t_b`` are the turbulence columns (..., n_t) on both sides,
    ``rho`` and ``nu`` the local density and kinematic viscosity. mu_t is
    the mean of rho nu_t of the two sides; the coefficients are rho_f times
    SA's (nu_f + nu_tilde_f)/sigma and, with AFT, nu_f + nu_t/sigma_y and
    sigma_n (nu_f + nu_t), of the face means.
    """
    nt_a, nt_b = t_a[..., 0], t_b[..., 0]
    nut_a = sa.eddy_viscosity(nt_a, nu_a)
    nut_b = sa.eddy_viscosity(nt_b, nu_b)
    mu_t = 0.5 * (rho_a * nut_a + rho_b * nut_b)
    nu_t = 0.5 * (nut_a + nut_b)
    nu_f = 0.5 * (nu_a + nu_b)
    rho_f = 0.5 * (rho_a + rho_b)
    coeffs = [sa.diffusion_coefficient(0.5 * (nt_a + nt_b), nu_f)]
    if t_a.shape[-1] > 1:
        coeffs += [nu_f + nu_t / aft.SIGMA_Y, aft.SIGMA_N * (nu_f + nu_t)]
    return mu_t, rho_f[..., None] * jnp.stack(coeffs, axis=-1)


def _cell_sources(model, geom, w, w_b, grad, gamma=None):
    """Turbulence sources per unit volume (n, n_t).

    rho times the kinematic sources, plus SA's density-gradient correction.
    """
    tc = t_column(w)
    t = w[:, tc]
    rho, nu = cell_properties(model, t)
    grad_v, grad_sq, switch, h_l = rans._source_inputs(model, geom, w, w_b, grad, nu)
    src = rans._turbulence_sources(
        model, w[:, NT:tc], grad_v, geom.wall_distance, grad_sq, switch, h_l, gamma, nu
    )
    rho_t = th.density_derivative(model.thermo, t)
    cross = rho_t * jnp.sum(grad[:, tc, :] * grad[:, NT, :], axis=1)
    src = rho[:, None] * src
    return src.at[:, 0].add(-sa.diffusion_coefficient(w[:, NT], nu) * cross)


@partial(jax.jit, static_argnums=0)
def limiter(model, geom, bc, w):
    """Limiter values psi of the current state, shape (n_cells, n_var)."""
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    return inc._limiter(model, geom, w, w_b, lsq_gradient(geom, w, w_b))


@partial(jax.jit, static_argnums=(0, 4))
def residual(model, geom, bc, w, order, beta2, psi=None, gamma=None):
    """Residual R(w), shape (n_cells, n_var): net outflow minus sources.

    Parameters as in :func:`pyaeroflow.physics.rans.residual`.
    """
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    tc = t_column(w)
    w_b = boundary_state(bc, geom.b_normal, w[b])
    grad = lsq_gradient(geom, w, w_b)
    sa_order = model.turbulence_spatial_order
    if order == 2 or sa_order == 2:
        if psi is None:
            psi = inc._limiter(model, geom, w, w_b, grad)
        w_l, w_r, w_in = reconstruct(geom, w, grad, psi)
    m_l, m_r, m_in = (
        (mean_state(w_l), mean_state(w_r), mean_state(w_in))
        if order == 2
        else (mean_state(w[o]), mean_state(w[n]), mean_state(w[b]))
    )
    s_l, s_r, s_in = (
        (w_l[:, NT:tc], w_r[:, NT:tc], w_in[:, NT:tc])
        if sa_order == 2
        else (w[o, NT:tc], w[n, NT:tc], w[b, NT:tc])
    )
    tm = model.thermo

    f_flux = vd.fds_flux(model, m_l, m_r, geom.f_normal, beta2)
    b_flux = vd.boundary_convective_flux(model, bc, geom.b_normal, m_in, beta2)
    vn_f = _dot_n(0.5 * (m_l[:, 1:3] + m_r[:, 1:3]), geom.f_normal)
    mass_f = 0.5 * (th.density(tm, m_l[:, 3]) + th.density(tm, m_r[:, 3])) * vn_f
    mass_b = th.density(tm, w_b[:, tc]) * _dot_n(w_b[:, 1:3], geom.b_normal)
    sa_f = upwind_scalar_flux(mass_f[:, None], s_l, s_r)
    sa_b = upwind_scalar_flux(mass_b[:, None], s_in, w_b[:, NT:tc])

    rho_c, nu_c = cell_properties(model, w[:, tc])
    rho_bs, nu_bs = cell_properties(model, w_b[:, tc])
    g_f = interior_face_gradient(geom, w, grad, model.viscous_gradient, model.alpha)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    mu_tf, dc_f = face_coefficients(
        model, w[o, NT:tc], w[n, NT:tc], rho_c[o], rho_c[n], nu_c[o], nu_c[n]
    )
    mu_tb, dc_b = face_coefficients(
        model, w[b, NT:tc], w_b[:, NT:tc], rho_c[b], rho_bs, nu_c[b], nu_bs
    )
    t_f = 0.5 * (w[o, tc] + w[n, tc])
    f_flux = f_flux + energy_viscous_flux(
        model, g_f[:, 1:3], g_f[:, tc], t_f, mu_tf, face_distance(geom), geom.f_normal
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
    )
    sa_f = sa_f - dc_f * _dot_n_cols(g_f[:, NT:tc], geom.f_normal)
    sa_b = sa_b - dc_b * _dot_n_cols(g_b[:, NT:tc], geom.b_normal)

    r = divergence(geom, assemble(f_flux, sa_f), assemble(b_flux, sa_b))
    src = _cell_sources(model, geom, w, w_b, grad, gamma)
    r = r.at[:, NT:tc].add(-geom.area[:, None] * src)
    r = r.at[:, U : V + 1].add(-vd.cell_sources(model, geom, mean_state(w)))
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


@partial(jax.jit, static_argnums=(0, 4))
def residual_linearize(model, geom, bc, w, order, beta2, psi=None, gamma=None):
    """Residual and its linearization at ``w`` (``jax.linearize``)."""

    def f(x):
        return residual(model, geom, bc, x, order, beta2, psi, gamma)

    return jax.linearize(f, w)


# -- first-order Jacobian ----------------------------------------------------------


def jacobian_arguments(model, geom, bc, w):
    """Per-face arguments (f_args, b_args) of the compact fluxes."""
    b_args = vd.jacobian_boundary_args(bc) + (bc.nu_tilde, boundary_distance(geom, bc))
    return (face_distance(geom),), b_args


def _compact_flux(model, conv, w_o, w_x, g, coeff, distance, normal, boundary=None):
    """Viscous and turbulence part of a compact face flux (one face).

    ``w_x`` is the neighbour or boundary state, ``g`` (n_var, 2) the face
    gradient, ``coeff`` the factor of the turbulence jump, ``boundary`` the
    one-face BoundaryData (None for an interior face).
    """
    tc = t_column(w_o)
    rho_o, nu_o = cell_properties(model, w_o[tc])
    rho_x, nu_x = cell_properties(model, w_x[tc])
    mu_t, dc = face_coefficients(
        model, w_o[NT:tc], w_x[NT:tc], rho_o, rho_x, nu_o, nu_x
    )
    if boundary is None:
        t_f = 0.5 * (w_o[tc] + w_x[tc])
        visc = energy_viscous_flux(
            model,
            g[None, 1:3],
            g[None, tc],
            t_f[None],
            mu_t[None],
            distance[None],
            normal[None],
        )[0]
        mass = 0.5 * (rho_o + rho_x) * 0.5 * jnp.dot(w_o[1:3] + w_x[1:3], normal)
    else:
        visc = boundary_energy_viscous_flux(
            model,
            boundary,
            g[None, 1:3],
            g[None, tc],
            w_x[tc][None],
            mu_t[None],
            distance[None],
            normal[None],
        )[0]
        mass = rho_x * jnp.dot(w_x[1:3], normal)
    flux = upwind_scalar_flux(mass, w_o[NT:tc], w_x[NT:tc]) - dc * coeff * (
        w_x[NT:tc] - w_o[NT:tc]
    )
    return assemble(conv + visc, flux)


def interior_flux(model, beta2, w_o, w_n, distance, normal, delta):
    """Compact first-order flux through an interior face (for the Jacobian)."""
    conv = vd.fds_flux(
        model, mean_state(w_o)[None], mean_state(w_n)[None], normal[None], beta2
    )[0]
    dn = jnp.dot(delta, normal)
    if model.viscous_gradient == "alpha-damping":
        coeff = model.alpha / dn
    else:
        coeff = dn / jnp.dot(delta, delta)
    g = (coeff * (w_n - w_o))[:, None] * normal[None, :]
    return _compact_flux(model, conv, w_o, w_n, g, coeff, distance, normal)


def boundary_flux(
    model,
    beta2,
    w_o,
    bc_kind,
    bc_value,
    bc_thermal,
    bc_heat,
    bc_rho,
    bc_nt,
    distance,
    normal,
    delta,
):
    """Compact first-order flux through a boundary face (for the Jacobian)."""
    bc = vd.BoundaryData(
        bc_kind[None],
        bc_value[None],
        bc_thermal[None],
        bc_heat[None],
        bc_rho[None],
        model.density,
        bc_nt[None],
    )
    w_b = boundary_state(bc, normal[None], w_o[None])[0]
    conv = vd.boundary_convective_flux(
        model, bc, normal[None], mean_state(w_o)[None], beta2
    )[0]
    dn = jnp.dot(delta, normal)
    g = ((w_b - w_o) / dn)[:, None] * normal[None, :]
    return _compact_flux(model, conv, w_o, w_b, g, 1.0 / dn, distance, normal, bc)


def _source_parts(model, geom, bc, w):
    """Frozen stencil inputs and the local nu, rho of the source Jacobians."""
    tc = t_column(w)
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    rho, nu = cell_properties(model, w[:, tc])
    grad_v, grad_sq, switch, h_l = rans._source_inputs(model, geom, w, w_b, grad, nu)

    def cell_source(t, gv, d, gs, sw, hl, nu_c):
        return rans._turbulence_sources(
            model,
            t[None],
            gv[None],
            d[None],
            gs[None],
            sw[None],
            hl[None],
            None,
            nu_c[None],
        )[0]

    jac = jax.vmap(jax.jacfwd(cell_source))(
        w[:, NT:tc], grad_v, geom.wall_distance, grad_sq, switch, h_l, nu
    )
    return rho[:, None, None] * jac


@partial(jax.jit, static_argnums=0)
def source_blocks(model, geom, bc, w):
    """Cell-local source blocks (n_cells, n_var, n_var) of the first-order Jacobian.

    The negative part of the diagonal derivatives of the turbulence sources
    (as :func:`pyaeroflow.physics.rans.source_blocks`, times rho, the
    density-gradient correction and the stencil inputs frozen) and the T
    derivative of the buoyancy.
    """
    tc = t_column(w)
    d_src = jnp.diagonal(_source_parts(model, geom, bc, w), axis1=1, axis2=2)
    n_var = w.shape[1]
    blocks = jnp.zeros((w.shape[0], n_var, n_var))
    k = jnp.arange(NT, tc)
    blocks = blocks.at[:, k, k].set(-geom.area[:, None] * jnp.minimum(d_src, 0.0))
    return buoyancy_blocks(model, geom, w, blocks)


@partial(jax.jit, static_argnums=0)
def source_jacobian(model, geom, bc, w):
    """Cell-local d(-area src)/d(turbulence columns, T), (n, n_t + 1, n_t + 1).

    Production included, stencil inputs frozen; the T row and column are
    zero (the energy equation has no source). For the source-term
    time-step limit of the driver.
    """
    jac = -geom.area[:, None, None] * _source_parts(model, geom, bc, w)
    return jnp.pad(jac, ((0, 0), (0, 1), (0, 1)))


@partial(jax.jit, static_argnums=0)
def time_term_blocks(model, geom, w, beta2, cfl):
    """Pseudo-time blocks: Gamma on (p, u, v, T), rho on the turbulence rows."""
    n_var = w.shape[1]
    gamma, lam = mean_time_blocks(model, geom, w, beta2)
    idx = mean_index(n_var)
    g = jnp.zeros((w.shape[0], n_var, n_var))
    g = g.at[:, idx[:, None], idx[None, :]].set(gamma)
    k = jnp.arange(NT, n_var - 1)
    rho = th.density(model.thermo, w[:, -1])
    g = g.at[:, k, k].set(rho[:, None])
    return g * (lam / cfl)[:, None, None]


def segregated_mask(n_var):
    """Block mask of the segregated coupling: (p, u, v, T) and turbulence apart."""
    return mean_segregated_mask(n_var, NT)


# -- output ------------------------------------------------------------------------


def eddy_viscosity(model, w):
    """Kinematic eddy viscosity per cell (local nu)."""
    _, nu = cell_properties(model, w[:, -1])
    return sa.eddy_viscosity(w[:, NT], nu)


@partial(jax.jit, static_argnums=0)
def intermittency_field(model, geom, bc, w):
    """Intermittency per cell for the output (ones without a transition model)."""
    if model.transition == "aft":
        return jnp.exp(w[:, NG])
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    vort = rans.vorticity(lsq_gradient(geom, w, w_b))
    _, nu = cell_properties(model, w[:, -1])
    gamma = rans.intermittency(model, w[:, NT], vort, geom.wall_distance, nu)
    return jnp.broadcast_to(jnp.asarray(gamma, dtype=float), w[:, NT].shape)


def _boundary_heat(model, geom, bc, w, coefficients):
    """Heat flux into the fluid per boundary face and the face conductivity.

    ``coefficients(w, w_b)`` returns mu_t at the boundary faces.
    """
    tc = t_column(w)
    w_b = coefficients.boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    mu_t = coefficients.boundary_eddy_viscosity(model, geom, bc, w, w_b, grad)
    d_b = boundary_distance(geom, bc)
    q = -boundary_energy_viscous_flux(
        model, bc, g_b[:, 1:3], g_b[:, tc], w_b[:, tc], mu_t, d_b, geom.b_normal
    )[:, 3]
    t_c = w[geom.b_owner, tc]
    k = th.effective_conductivity(
        model.thermo,
        model.turbulent_prandtl,
        t_c,
        th.density(model.thermo, t_c),
        mu_t,
        d_b,
    )
    return q, k, w_b[:, tc]


def heat_flux_and_temperature(model, geom, bc, w, coefficients):
    """(q_w, T_w) per boundary face with the eddy conductivity of the faces.

    T_w: the prescribed value at isothermal walls and fixed-T boundaries,
    elsewhere T_cell + q_w (d.n) / k_eff.
    """
    q, k, t_b = _boundary_heat(model, geom, bc, w, coefficients)
    t_c = w[geom.b_owner, t_column(w)]
    dn = jnp.einsum("fk,fk->f", geom.b_delta, geom.b_normal)
    walls = (bc.kind == WALL) & (bc.thermal != vd.ISOTHERMAL)
    return q, jnp.where(walls, t_c + q * dn / k, t_b)


class _SA:
    """Boundary pieces of the SA family for :func:`heat_flux_and_temperature`."""

    boundary_state = staticmethod(boundary_state)

    @staticmethod
    def boundary_eddy_viscosity(model, geom, bc, w, w_b, grad):
        b = geom.b_owner
        tc = t_column(w)
        rho_c, nu_c = cell_properties(model, w[b, tc])
        rho_b, nu_b = cell_properties(model, w_b[:, tc])
        return face_coefficients(
            model, w[b, NT:tc], w_b[:, NT:tc], rho_c, rho_b, nu_c, nu_b
        )[0]


@partial(jax.jit, static_argnums=0)
def wall_heat_flux(model, geom, bc, w):
    """Heat flux into the fluid per boundary face (n_bfaces,), k_eff dT/dn."""
    return heat_flux_and_temperature(model, geom, bc, w, _SA)[0]


@partial(jax.jit, static_argnums=0)
def wall_temperature(model, geom, bc, w):
    """Temperature per boundary face (n_bfaces,)."""
    return heat_flux_and_temperature(model, geom, bc, w, _SA)[1]

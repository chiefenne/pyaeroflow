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


def n_species(model):
    """Return the number of transported species of a model (or StaticThermo)."""
    return len(model.thermo.species)


def bc_species(bc):
    """Return the number of species of boundary data (static, from the shapes)."""
    return 0 if bc.species_flux is None else bc.species_flux.shape[-1]


def t_column(w, ns=0):
    """Column of T: in front of the ``ns`` mass fractions (static, from the shape)."""
    return w.shape[-1] - 1 - ns


def thermal_state(w, ns=0):
    """(T, Y) of full states (..., n_var); Y is None without species."""
    tc = t_column(w, ns)
    return w[..., tc], (w[..., tc + 1 :] if ns else None)


def mean_state(w, ns=0):
    """(p, u, v, T[, Y]) of full states (..., n_var), the variable-density state."""
    return jnp.concatenate((w[..., :3], w[..., t_column(w, ns) :]), axis=-1)


def assemble(mean, turb):
    """Full rows (..., n_var) of the (p, u, v, T[, Y]) and the turbulence parts."""
    return jnp.concatenate((mean[..., :3], turb, mean[..., 3:]), axis=-1)


def mean_index(n_var, ns=0):
    """Columns of (p, u, v, T[, Y]) in a state of ``n_var`` columns."""
    return jnp.array([0, 1, 2] + list(range(n_var - 1 - ns, n_var)))


def cell_properties(model, t, y=None):
    """Density rho and nu = mu / rho at temperatures ``t`` (mass fractions ``y``)."""
    rho = th.density(model.thermo, t, y)
    return rho, th.viscosity(model.thermo, t) / rho


def state_properties(model, w, ns=0):
    """Density rho and nu = mu / rho of full states (..., n_var)."""
    return cell_properties(model, *thermal_state(w, ns))


def mean_density(model, m):
    """Density of mean states (..., 4 + n_s) = (p, u, v, T[, Y])."""
    y = m[..., 4:] if n_species(model) else None
    return th.density(model.thermo, m[..., 3], y)


def mean_boundary_state(bc, normal, w_in):
    """(p, u, v, T[, Y]) on boundary faces (variable-density boundary rules)."""
    return vd.boundary_state(bc, normal, mean_state(w_in, bc_species(bc)))


def inflow_mask(bc, normal, w_b4):
    """Faces where the turbulence values are prescribed (inlets, inflow farfield)."""
    vn = w_b4[:, U] * normal[:, 0] + w_b4[:, V] * normal[:, 1]
    kind = bc.kind
    return (
        (kind == VELOCITY_INLET)
        | (kind == TOTAL_PRESSURE_INLET)
        | ((kind == FARFIELD) & (vn < 0.0))
    )


def _effective_conductivity(model, t_f, y_f, mu_t, distance):
    tm = model.thermo
    return th.effective_conductivity(
        tm, model.turbulent_prandtl, t_f, th.density(tm, t_f, y_f), mu_t, distance
    )


def _turbulent_diffusion(model, t_f, y_f, mu_t, distance):
    """mu_t / Sc_t,k at the faces (species)."""
    return th.turbulent_schmidt_coefficients(
        model.thermo, model.turbulent_schmidt, t_f, y_f, mu_t, distance
    )


def energy_viscous_flux(
    model,
    grad_v,
    grad_t,
    t_f,
    mu_t,
    distance,
    normal,
    heat=True,
    grad_y=None,
    y_f=None,
):
    """-(0, tau.n, k_eff grad T.n[, J.n]) with the eddy viscosity ``mu_t`` (dynamic).

    ``grad_v`` (n, 2, 2) and ``grad_t`` (n, 2) are face gradients, ``t_f``
    the face temperature, ``distance`` the face wall distance (an input of
    the Pr_t and Sc_t laws). ``heat`` False drops the conduction (bool or
    array). Species: ``grad_y`` (n, n_s, 2) and the face mass fractions
    ``y_f``; the diffusion fluxes with rho D_k + mu_t / Sc_t,k and their
    enthalpy flux are appended.
    """
    tm = model.thermo
    mu = th.viscosity(tm, t_f) + mu_t
    flow = inc.viscous_flux(model, grad_v, normal, mu)
    k = _effective_conductivity(model, t_f, y_f, mu_t, distance)
    q = k * (grad_t[:, 0] * normal[:, 0] + grad_t[:, 1] * normal[:, 1])
    q = jnp.where(heat, q, 0.0)
    flux = jnp.concatenate((flow, -q[:, None]), axis=1)
    if y_f is None:
        return flux
    grad_n = jnp.einsum("nsk,nk->ns", grad_y, normal)
    turbulent = _turbulent_diffusion(model, t_f, y_f, mu_t, distance)
    j = th.diffusion_flux(tm, t_f, y_f, grad_n, turbulent)
    return vd._with_species(model, flux, j, t_f)


def boundary_energy_viscous_flux(
    model,
    bc,
    grad_v,
    grad_t,
    t_b,
    mu_t,
    d_b,
    normal,
    grad_y=None,
    y_b=None,
    details=False,
):
    """-F_v.n through boundary faces (n_bfaces, 4[ + n_s]), thermal rules of V1.

    Conduction (with the eddy conductivity of ``mu_t``) at isothermal walls,
    inlets, farfields and outlets, the prescribed heat flux at heat-flux
    walls, nothing at adiabatic walls and symmetry faces. Species: the
    boundary rules of
    :func:`pyaeroflow.physics.variable_density.boundary_species_flux` with
    rho D_k + mu_t / Sc_t,k and the effective conductivity (``details``: also
    its dict).
    """
    k = bc.kind
    wall = k == WALL
    conduct = ~((wall & (bc.thermal != vd.ISOTHERMAL)) | (k == SYMMETRY))
    flux = energy_viscous_flux(
        model, grad_v, grad_t, t_b, mu_t, d_b, normal, heat=conduct
    )
    prescribed = wall & (bc.thermal == vd.HEAT_FLUX)
    flux = flux.at[:, 3].set(jnp.where(prescribed, -bc.heat_flux, flux[:, 3]))
    if y_b is None:
        return (flux, {"conduction": -flux[:, 3]}) if details else flux
    return vd.boundary_species_flux(
        model,
        bc,
        flux,
        grad_y,
        t_b,
        y_b,
        normal,
        turbulent=_turbulent_diffusion(model, t_b, y_b, mu_t, d_b),
        conductivity=_effective_conductivity(model, t_b, y_b, mu_t, d_b),
        details=details,
    )


def boundary_distance(geom, bc):
    """Wall distance at boundary faces: 0 at walls, the owner's elsewhere."""
    return jnp.where(bc.kind == WALL, 0.0, geom.wall_distance[geom.b_owner])


def face_distance(geom):
    """Wall distance at interior faces: mean of the two cells."""
    d = geom.wall_distance
    return 0.5 * (d[geom.f_owner] + d[geom.f_neigh])


def mean_time_blocks(model, geom, w, beta2):
    """Gamma on (p, u, v, T) and the spectral radius: ((n, 4, 4), (n,)).

    ``model`` is the static model (closed domains: p_th of ``w`` is applied
    here).
    """
    m = mean_state(w, n_species(model))
    return (
        vd.preconditioner(vd.closed_model(model, geom, m), m, beta2),
        vd.spectral_radius(model, geom, m, beta2),
    )


def mean_segregated_mask(n_var, first, ns=0):
    """Segregated block mask: (p, u, v, T, Y) apart from the turbulence columns."""
    idx = mean_index(n_var, ns)
    mask = jnp.zeros((n_var, n_var))
    mask = mask.at[idx[:, None], idx[None, :]].set(1.0)
    last = n_var - 1 - ns
    return mask.at[first:last, first:last].set(1.0)


def buoyancy_blocks(model, geom, w, blocks):
    """Add d(-area (rho - rho_op) g)/d(T, Y) to the (u, v; T, Y) blocks."""
    ns = n_species(model)
    tc = t_column(w, ns)
    t, y = thermal_state(w, ns)
    rho_t = th.density_derivative(model.thermo, t, y)
    g = jnp.asarray(model.gravity, dtype=w.dtype)
    blocks = blocks.at[:, U, tc].add(-geom.area * rho_t * g[0])
    blocks = blocks.at[:, V, tc].add(-geom.area * rho_t * g[1])
    if y is None:
        return blocks
    rho_y = th.density_species_derivative(model.thermo, t, y)
    blocks = blocks.at[:, U, tc + 1 :].add(-geom.area[:, None] * rho_y * g[0])
    return blocks.at[:, V, tc + 1 :].add(-geom.area[:, None] * rho_y * g[1])


def wall_traction(model, grad, w_b, normal):
    """tau.n on boundary faces (n, 2) with mu of the boundary temperature."""
    mu = th.viscosity(model.thermo, w_b[:, t_column(w_b, n_species(model))])
    return -inc.viscous_flux(model, grad[:, U : V + 1], normal, mu)[:, 1:]


# -- SA family ---------------------------------------------------------------------


def boundary_state(bc, normal, w_in):
    """Boundary-face state (p, u, v, nu_tilde[, gamma_tilde, n], T[, Y])."""
    w_b4 = mean_boundary_state(bc, normal, w_in)
    tc = t_column(w_in, bc_species(bc))
    nt = rans.boundary_nu_tilde(bc, normal, w_b4[:, :3], w_in[:, NT])
    cols = [nt[:, None]]
    if tc > NT + 1:  # SA-AFT
        inflow = inflow_mask(bc, normal, w_b4)
        cols.append(jnp.where(inflow[:, None], 0.0, w_in[:, NT + 1 : tc]))
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
    ns = n_species(model)
    tc = t_column(w, ns)
    t, y = thermal_state(w, ns)
    rho, nu = cell_properties(model, t, y)
    grad_v, grad_sq, switch, h_l = rans._source_inputs(model, geom, w, w_b, grad, nu)
    src = rans._turbulence_sources(
        model, w[:, NT:tc], grad_v, geom.wall_distance, grad_sq, switch, h_l, gamma, nu
    )
    rho_t = th.density_derivative(model.thermo, t, y)
    cross = rho_t * jnp.sum(grad[:, tc, :] * grad[:, NT, :], axis=1)
    if y is not None:  # grad rho = rho_T grad T + sum rho_Y grad Y
        rho_y = th.density_species_derivative(model.thermo, t, y)
        cross = cross + jnp.einsum(
            "ns,nsk,nk->n", rho_y, grad[:, tc + 1 :, :], grad[:, NT, :]
        )
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

    Parameters as in :func:`pyaeroflow.physics.rans.residual`. Closed
    domains: the density follows p_th(w).
    """
    model = vd.closed_model(model, geom, w)
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    ns = n_species(model)
    tc = t_column(w, ns)
    w_b = boundary_state(bc, geom.b_normal, w[b])
    grad = lsq_gradient(geom, w, w_b)
    sa_order = model.turbulence_spatial_order
    if order == 2 or sa_order == 2:
        if psi is None:
            psi = inc._limiter(model, geom, w, w_b, grad)
        w_l, w_r, w_in = reconstruct(geom, w, grad, psi)
    m_l, m_r, m_in = (
        (mean_state(w_l, ns), mean_state(w_r, ns), mean_state(w_in, ns))
        if order == 2
        else (mean_state(w[o], ns), mean_state(w[n], ns), mean_state(w[b], ns))
    )
    s_l, s_r, s_in = (
        (w_l[:, NT:tc], w_r[:, NT:tc], w_in[:, NT:tc])
        if sa_order == 2
        else (w[o, NT:tc], w[n, NT:tc], w[b, NT:tc])
    )
    f_flux = vd.fds_flux(model, m_l, m_r, geom.f_normal, beta2)
    b_flux = vd.boundary_convective_flux(model, bc, geom.b_normal, m_in, beta2)
    vn_f = _dot_n(0.5 * (m_l[:, 1:3] + m_r[:, 1:3]), geom.f_normal)
    mass_f = 0.5 * (mean_density(model, m_l) + mean_density(model, m_r)) * vn_f
    mass_b = state_properties(model, w_b, ns)[0] * _dot_n(w_b[:, 1:3], geom.b_normal)
    sa_f = upwind_scalar_flux(mass_f[:, None], s_l, s_r)
    sa_b = upwind_scalar_flux(mass_b[:, None], s_in, w_b[:, NT:tc])

    rho_c, nu_c = state_properties(model, w, ns)
    rho_bs, nu_bs = state_properties(model, w_b, ns)
    g_f = interior_face_gradient(geom, w, grad, model.viscous_gradient, model.alpha)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    mu_tf, dc_f = face_coefficients(
        model, w[o, NT:tc], w[n, NT:tc], rho_c[o], rho_c[n], nu_c[o], nu_c[n]
    )
    mu_tb, dc_b = face_coefficients(
        model, w[b, NT:tc], w_b[:, NT:tc], rho_c[b], rho_bs, nu_c[b], nu_bs
    )
    t_f = 0.5 * (w[o, tc] + w[n, tc])
    y_f = 0.5 * (w[o, tc + 1 :] + w[n, tc + 1 :]) if ns else None
    f_flux = f_flux + energy_viscous_flux(
        model,
        g_f[:, 1:3],
        g_f[:, tc],
        t_f,
        mu_tf,
        face_distance(geom),
        geom.f_normal,
        grad_y=g_f[:, tc + 1 :],
        y_f=y_f,
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
    sa_f = sa_f - dc_f * _dot_n_cols(g_f[:, NT:tc], geom.f_normal)
    sa_b = sa_b - dc_b * _dot_n_cols(g_b[:, NT:tc], geom.b_normal)

    r = divergence(geom, assemble(f_flux, sa_f), assemble(b_flux, sa_b))
    src = _cell_sources(model, geom, w, w_b, grad, gamma)
    r = r.at[:, NT:tc].add(-geom.area[:, None] * src)
    r = r.at[:, U : V + 1].add(-vd.cell_sources(model, geom, mean_state(w, ns)))
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
    b_args = vd.jacobian_boundary_args(bc)[:5] + (
        bc.nu_tilde,
        boundary_distance(geom, bc),
    )
    return (face_distance(geom),), b_args + vd.species_boundary_args(bc)


def _compact_flux(model, conv, w_o, w_x, g, coeff, distance, normal, boundary=None):
    """Viscous and turbulence part of a compact face flux (one face).

    ``w_x`` is the neighbour or boundary state, ``g`` (n_var, 2) the face
    gradient, ``coeff`` the factor of the turbulence jump, ``boundary`` the
    one-face BoundaryData (None for an interior face).
    """
    ns = n_species(model)
    tc = t_column(w_o, ns)
    rho_o, nu_o = state_properties(model, w_o, ns)
    rho_x, nu_x = state_properties(model, w_x, ns)
    mu_t, dc = face_coefficients(
        model, w_o[NT:tc], w_x[NT:tc], rho_o, rho_x, nu_o, nu_x
    )
    if boundary is None:
        t_f = 0.5 * (w_o[tc] + w_x[tc])
        y_f = 0.5 * (w_o[tc + 1 :] + w_x[tc + 1 :])[None] if ns else None
        visc = energy_viscous_flux(
            model,
            g[None, 1:3],
            g[None, tc],
            t_f[None],
            mu_t[None],
            distance[None],
            normal[None],
            grad_y=g[None, tc + 1 :],
            y_f=y_f,
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
            grad_y=g[None, tc + 1 :],
            y_b=w_x[tc + 1 :][None] if ns else None,
        )[0]
        mass = rho_x * jnp.dot(w_x[1:3], normal)
    flux = upwind_scalar_flux(mass, w_o[NT:tc], w_x[NT:tc]) - dc * coeff * (
        w_x[NT:tc] - w_o[NT:tc]
    )
    return assemble(conv + visc, flux)


def interior_flux(model, beta2, w_o, w_n, distance, normal, delta):
    """Compact first-order flux through an interior face (for the Jacobian)."""
    ns = n_species(model)
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
    *args,
):
    """Compact first-order flux through a boundary face (for the Jacobian).

    ``args``: the species arrays of the face (with species only), the face
    normal and the vector from the cell centre to the face.
    """
    *species, normal, delta = args
    bc = vd.BoundaryData(
        bc_kind[None],
        bc_value[None],
        bc_thermal[None],
        bc_heat[None],
        bc_rho[None],
        model.density,
        bc_nt[None],
    )
    bc = vd.with_species_arguments(bc, tuple(species))
    w_b = boundary_state(bc, normal[None], w_o[None])[0]
    conv = vd.boundary_convective_flux(
        model, bc, normal[None], mean_state(w_o, n_species(model))[None], beta2
    )[0]
    dn = jnp.dot(delta, normal)
    g = ((w_b - w_o) / dn)[:, None] * normal[None, :]
    return _compact_flux(model, conv, w_o, w_b, g, 1.0 / dn, distance, normal, bc)


def _source_parts(model, geom, bc, w):
    """Frozen stencil inputs and the local nu, rho of the source Jacobians."""
    ns = n_species(model)
    tc = t_column(w, ns)
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    rho, nu = state_properties(model, w, ns)
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
    model = vd.closed_model(model, geom, w)
    tc = t_column(w, n_species(model))
    d_src = jnp.diagonal(_source_parts(model, geom, bc, w), axis1=1, axis2=2)
    n_var = w.shape[1]
    blocks = jnp.zeros((w.shape[0], n_var, n_var))
    k = jnp.arange(NT, tc)
    blocks = blocks.at[:, k, k].set(-geom.area[:, None] * jnp.minimum(d_src, 0.0))
    return buoyancy_blocks(model, geom, w, blocks)


@partial(jax.jit, static_argnums=0)
def source_jacobian(model, geom, bc, w):
    """Cell-local d(-area src)/d(turbulence columns, T, Y), (n, n - 3, n - 3).

    Production included, stencil inputs frozen; the T and Y rows and
    columns are zero (no sources). For the source-term time-step limit of
    the driver.
    """
    model = vd.closed_model(model, geom, w)
    jac = -geom.area[:, None, None] * _source_parts(model, geom, bc, w)
    pad = 1 + n_species(model)
    return jnp.pad(jac, ((0, 0), (0, pad), (0, pad)))


@partial(jax.jit, static_argnums=0)
def time_term_blocks(model, geom, w, beta2, cfl):
    """Pseudo-time blocks: Gamma on (p, u, v, T, Y), rho on the turbulence rows."""
    n_var = w.shape[1]
    ns = n_species(model)
    gamma, lam = mean_time_blocks(model, geom, w, beta2)
    idx = mean_index(n_var, ns)
    g = jnp.zeros((w.shape[0], n_var, n_var))
    g = g.at[:, idx[:, None], idx[None, :]].set(gamma)
    k = jnp.arange(NT, t_column(w, ns))
    rho = state_properties(vd.closed_model(model, geom, w), w, ns)[0]
    g = g.at[:, k, k].set(rho[:, None])
    return g * (lam / cfl)[:, None, None]


def segregated_mask(n_var, ns=0):
    """Block mask of the segregated coupling: (p, u, v, T, Y) and turbulence apart."""
    return mean_segregated_mask(n_var, NT, ns)


# -- output ------------------------------------------------------------------------


def eddy_viscosity(model, w):
    """Kinematic eddy viscosity per cell (local nu).

    Closed domains: ``model`` must carry p_th of ``w``
    (:func:`pyaeroflow.physics.variable_density.at_pressure`).
    """
    _, nu = state_properties(model, w, n_species(model))
    return sa.eddy_viscosity(w[:, NT], nu)


@partial(jax.jit, static_argnums=0)
def intermittency_field(model, geom, bc, w):
    """Intermittency per cell for the output (ones without a transition model)."""
    if model.transition == "aft":
        return jnp.exp(w[:, NG])
    model = vd.closed_model(model, geom, w)
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    vort = rans.vorticity(lsq_gradient(geom, w, w_b))
    _, nu = state_properties(model, w, n_species(model))
    gamma = rans.intermittency(model, w[:, NT], vort, geom.wall_distance, nu)
    return jnp.broadcast_to(jnp.asarray(gamma, dtype=float), w[:, NT].shape)


def _boundary_heat(model, geom, bc, w, coefficients):
    """Heat flux into the fluid per boundary face and the face conductivity.

    ``coefficients(w, w_b)`` returns mu_t at the boundary faces. Closed
    domains: the density follows p_th(w).
    """
    model = vd.closed_model(model, geom, w)
    bc = vd.closed_boundary(model, bc)
    ns = n_species(model)
    tc = t_column(w, ns)
    w_b = coefficients.boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    mu_t = coefficients.boundary_eddy_viscosity(model, geom, bc, w, w_b, grad)
    d_b = boundary_distance(geom, bc)
    y_b = w_b[:, tc + 1 :] if ns else None
    flux, details = boundary_energy_viscous_flux(
        model,
        bc,
        g_b[:, 1:3],
        g_b[:, tc],
        w_b[:, tc],
        mu_t,
        d_b,
        geom.b_normal,
        grad_y=g_b[:, tc + 1 :],
        y_b=y_b,
        details=True,
    )
    # T_b is the cell value at adiabatic and heat-flux walls (the same
    # effective conductivity as the evaporating-wall balance).
    k = _effective_conductivity(model, w_b[:, tc], y_b, mu_t, d_b)
    return details["conduction"], k, w_b[:, tc], flux, details, mu_t, d_b


def heat_flux_and_temperature(model, geom, bc, w, coefficients):
    """(q_w, T_w) per boundary face with the eddy conductivity of the faces.

    q_w is the heat conducted into the fluid. T_w: the prescribed value at
    isothermal walls and fixed-T boundaries, elsewhere T_cell + q_w (d.n) /
    k_eff (at evaporating walls the surface temperature of the balance).
    """
    q, k, t_b, *_ = _boundary_heat(model, geom, bc, w, coefficients)
    t_c = w[geom.b_owner, t_column(w, n_species(model))]
    dn = jnp.einsum("fk,fk->f", geom.b_delta, geom.b_normal)
    walls = (bc.kind == WALL) & (bc.thermal != vd.ISOTHERMAL)
    return q, jnp.where(walls, t_c + q * dn / k, t_b)


class _SA:
    """Boundary pieces of the SA family for :func:`heat_flux_and_temperature`."""

    boundary_state = staticmethod(boundary_state)

    @staticmethod
    def boundary_eddy_viscosity(model, geom, bc, w, w_b, grad):
        b = geom.b_owner
        ns = n_species(model)
        tc = t_column(w, ns)
        rho_c, nu_c = state_properties(model, w[b], ns)
        rho_b, nu_b = state_properties(model, w_b, ns)
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


def species_wall_values(model, geom, bc, w, coefficients):
    """Species walls (RANS): (j, m, Y_w) per boundary face.

    The species fluxes and the net mass flux into the fluid, and the wall
    mass fractions.
    """
    *_, flux, details, mu_t, d_b = _boundary_heat(model, geom, bc, w, coefficients)
    model = vd.closed_model(model, geom, w)
    ns = n_species(model)
    j = -flux[:, 4:]
    w_c = w[geom.b_owner]
    t_c, y_c = thermal_state(w_c, ns)
    turbulent = _turbulent_diffusion(model, t_c, y_c, mu_t, d_b)
    # wall_fractions reads T and Y from the last columns of ``w``.
    y_w = vd.wall_fractions(model, geom, bc, w, j, details, turbulent)
    return j, details["mass_flux"], y_w


@partial(jax.jit, static_argnums=0)
def wall_species_flux(model, geom, bc, w):
    """Species mass fluxes and the net mass flux into the fluid per boundary face."""
    j, m, _ = species_wall_values(model, geom, bc, w, _SA)
    return j, m


@partial(jax.jit, static_argnums=0)
def wall_mass_fraction(model, geom, bc, w):
    """Mass fractions per boundary face, (n_bfaces, n_species)."""
    return species_wall_values(model, geom, bc, w, _SA)[2]

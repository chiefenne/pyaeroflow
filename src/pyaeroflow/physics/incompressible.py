"""Incompressible Navier-Stokes equations, SU2 INC formulation (D5, D8, D21).

Working variables per cell: V = (p, u, v) with constant density rho. The
steady equations are integrated with a preconditioned pseudo-time term

    Gamma dV/dtau + R(V) = 0,     Gamma = [[1/beta^2, 0,   0  ],
                                           [u/beta^2, rho, 0  ],
                                           [v/beta^2, 0,   rho]],

the density-based generalisation of artificial compressibility used by SU2
(Economon 2020). beta^2 = beta_factor * max|v|^2 over the domain. The
residual R(V) is the net outflow of the inviscid flux

    F.n = (rho v_n, rho u v_n + p n_x, rho v v_n + p n_y),   v_n = v.n,

minus the viscous flux (0, tau.n) with tau = mu (grad v + grad v^T)
- 2/3 mu (div v) I, minus body forces.

Convective flux: flux-difference splitting (FDS) with low-speed
preconditioning, F = 0.5 (F_L + F_R) - 0.5 Gamma |Gamma^-1 A| (V_R - V_L),
where A = dF/dV includes d(rho)/dp = 1/beta^2 as in SU2, so that
Gamma^-1 A has the eigenvalues v_n, v_n - beta, v_n + beta. |Gamma^-1 A| is
the exact matrix absolute value from the eigen-decomposition; it differs from
SU2's closed form, which carries twice the tangential-velocity dissipation
(verified numerically, see tests).

Boundary conditions (D38): wall (no-slip, optional wall velocity),
symmetry / slip wall, velocity inlet, pressure outlet, farfield. Periodic
boundaries are interior faces of the geometry and need no treatment here.
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
from pyaeroflow.solver import divergence

# Boundary-condition kinds (integer codes per boundary face).
WALL, SYMMETRY, VELOCITY_INLET, PRESSURE_OUTLET, FARFIELD, TOTAL_PRESSURE_INLET = range(
    6
)
KIND_NAMES = {
    "wall": WALL,
    "transpiration-wall": WALL,  # wall with a prescribed normal velocity
    "symmetry": SYMMETRY,
    "slip-wall": SYMMETRY,
    "velocity-inlet": VELOCITY_INLET,
    "pressure-outlet": PRESSURE_OUTLET,
    "farfield": FARFIELD,
    "total-pressure-inlet": TOTAL_PRESSURE_INLET,
}
P, U, V = 0, 1, 2


class IncompressibleModel(NamedTuple):
    """Model parameters (hashable; a static jit argument).

    Attributes
    ----------
    density : float
        rho (constant).
    viscosity : float
        Dynamic viscosity mu; 0 for the Euler equations.
    beta_factor : float
        epsilon^2 in beta^2 = epsilon^2 max|v|^2 (SU2 default 4.1).
    spatial_order : int
        1 or 2 (MUSCL reconstruction of the convective states).
    limiter : {"none", "venkatakrishnan-wang", "venkatakrishnan"}
    limiter_threshold : float
        Wang's epsilon'.
    viscous_gradient : {"alpha-damping", "edge-correction"}
    alpha : float
        Damping coefficient of the alpha-damping scheme.
    body_force : tuple of float
        Force per unit volume (f_x, f_y), e.g. minus a mean pressure gradient
        for periodic channels (D39 R4).
    pin : tuple or None
        ``(cell, pressure)``: the continuity residual of that cell is replaced
        by p_cell - pressure (D39 R2), used when no boundary fixes the
        pressure level.
    turbulence : {"none", "sa-neg"}
        Turbulence model (D11); "sa-neg" adds nu_tilde as fourth variable
        (see :mod:`pyaeroflow.physics.rans`).
    turbulence_spatial_order : int
        1 or 2: convective states of the turbulence equation.
    coupling : {"segregated", "coupled"}
        Mean-flow/turbulence coupling of the first-order Jacobian (D34):
        "segregated" drops the cross-coupling blocks.
    transition : {"none", "bcm", "aft"}
        Transition model (L3): algebraic BCM, or SA-AFT2019b, which adds the
        log-intermittency and the amplification factor as columns 4 and 5.
    turbulence_intensity : float
        Freestream Tu in percent.
    critical_amplification : float or None
        N_crit of SA-AFT; None derives it from ``turbulence_intensity``.
    """

    density: float = 1.0
    viscosity: float = 1.0e-3
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
    transition: str = "none"  # "none", "bcm" (algebraic) or "aft" (SA-AFT2019b)
    turbulence_intensity: float = 0.0  # freestream Tu in percent (BCM, AFT)
    critical_amplification: float = None  # AFT N_crit; None: from Tu

    @property
    def n_var(self):
        """Number of variables: 3, 4 with a turbulence model, 6 with AFT."""
        if self.turbulence == "none":
            return 3
        return 6 if self.transition == "aft" else 4


def _density(bc):
    return bc.density


class BoundaryData(NamedTuple):
    """Per-boundary-face condition (a JAX pytree).

    Attributes
    ----------
    kind : int array, shape (n_bfaces,)
        One of WALL, SYMMETRY, VELOCITY_INLET, PRESSURE_OUTLET, FARFIELD,
        TOTAL_PRESSURE_INLET.
    value : float array, shape (n_bfaces, 3)
        Prescribed (p, u, v) where the kind uses them: wall and velocity
        inlet use (u, v); pressure outlet uses p; farfield uses all three;
        total-pressure inlet stores (p_total, d_x, d_y) with the unit flow
        direction d.
    nu_tilde : float array, shape (n_bfaces,) or None
        Prescribed SA working variable at inflow faces (zero elsewhere).
    """

    kind: jnp.ndarray
    value: jnp.ndarray
    density: float = 1.0
    nu_tilde: jnp.ndarray = None


def boundary_data(disc, conditions, density=1.0):
    """Build :class:`BoundaryData` from a dict of boundary conditions.

    Parameters
    ----------
    disc : Discretization
    conditions : dict of str to dict
        ``{name: {"type": ..., "velocity": (u, v) or callable(x, y),
        "pressure": p or callable(x, y)}}`` for every non-periodic boundary.
        A ``"transpiration-wall"`` takes ``normal_velocity`` (positive =
        blowing into the domain, negative = suction); a
        ``"total-pressure-inlet"`` takes ``total_pressure`` and
        ``flow_direction``. Callables receive the boundary-face centres
        (arrays) and return per-face values (D41).

    Returns
    -------
    BoundaryData
    """
    nb = disc.boundary_face_ids.size
    kind = np.zeros(nb, dtype=np.int64)
    value = np.zeros((nb, 3))
    nu_tilde = np.zeros(nb)
    missing = set(disc.boundary_slices) - set(conditions)
    extra = set(conditions) - set(disc.boundary_slices)
    if missing or extra:
        raise ValueError(
            f"boundary conditions missing for {sorted(missing)}, "
            f"unknown boundaries {sorted(extra)}"
        )
    centres = np.asarray(disc.geom.b_centre)
    normals = np.asarray(disc.geom.b_normal)
    for name, sl in disc.boundary_slices.items():
        bc = conditions[name]
        kind[sl] = KIND_NAMES[bc["type"]]
        x, y = centres[sl, 0], centres[sl, 1]
        if bc["type"] == "transpiration-wall":
            vn = _evaluate(bc["normal_velocity"], x, y)
            vel = (-vn * normals[sl, 0], -vn * normals[sl, 1])  # blowing = inward
            pres = 0.0
        elif bc["type"] == "total-pressure-inlet":
            d = np.asarray(_evaluate(bc["flow_direction"], x, y), dtype=float)
            d = np.broadcast_to(
                d.reshape(2, -1) if d.ndim > 1 else d[:, None], (2, x.size)
            )
            norm = np.hypot(d[0], d[1])
            vel = (d[0] / norm, d[1] / norm)
            pres = bc["total_pressure"]
        else:
            vel = bc.get("velocity", (0.0, 0.0))
            if callable(vel):
                vel = vel(x, y)
            pres = bc.get("pressure", 0.0)
        value[sl, U] = np.broadcast_to(np.asarray(vel[0], dtype=float), x.shape)
        value[sl, V] = np.broadcast_to(np.asarray(vel[1], dtype=float), x.shape)
        value[sl, P] = np.broadcast_to(_evaluate(pres, x, y), x.shape)
        nu_tilde[sl] = np.broadcast_to(
            _evaluate(bc.get("nu_tilde", 0.0), x, y), x.shape
        )
    return BoundaryData(
        jnp.asarray(kind), jnp.asarray(value), float(density), jnp.asarray(nu_tilde)
    )


def _evaluate(spec, x, y):
    """Evaluate a constant or a callable of the face centres as a float array."""
    return np.asarray(spec(x, y) if callable(spec) else spec, dtype=float)


def lsq_include(conditions, n_var=3):
    """Return the ``lsq_include`` callback for :func:`build_geometry`.

    A boundary face takes part in a variable's gradient only where its value
    is known: velocity at walls, velocity inlets and total-pressure inlets,
    pressure at pressure outlets, everything at farfield and symmetry faces
    (symmetry values are the mirrored cell values). The turbulence and
    transition variables (columns 3 and up) follow the velocity.
    """

    def include(slices, n_bfaces):
        mask = np.ones((n_bfaces, n_var), dtype=bool)
        for name, sl in slices.items():
            k = KIND_NAMES[conditions[name]["type"]]
            if k in (WALL, VELOCITY_INLET, TOTAL_PRESSURE_INLET):
                mask[sl, P] = False
            elif k == PRESSURE_OUTLET:
                mask[sl, U:] = False
        return mask

    return include


def boundary_state(bc, normal, v_in):
    """Boundary-face state (p, u, v) from the interior state ``v_in``.

    Wall / inlet: prescribed velocity, interior pressure. Outlet: prescribed
    pressure, interior velocity. Farfield: prescribed state. Symmetry:
    interior pressure, interior velocity with the normal component removed.
    Total-pressure inlet: interior pressure p, velocity magnitude from
    p_total - p = rho |v|^2 / 2 along the prescribed direction (SU2's
    incompressible pressure inlet).
    """
    vn = v_in[:, U] * normal[:, 0] + v_in[:, V] * normal[:, 1]
    mirrored = v_in.at[:, U].add(-vn * normal[:, 0]).at[:, V].add(-vn * normal[:, 1])
    kind = bc.kind[:, None]
    out = jnp.where(kind == SYMMETRY, mirrored, v_in)
    out = jnp.where(kind == FARFIELD, bc.value, out)
    # Total-pressure inlet: |v| = sqrt(2 (p0 - p) / rho), floored at zero.
    dyn = jnp.maximum(bc.value[:, P] - v_in[:, P], 0.0)
    speed = jnp.sqrt(2.0 * dyn / _density(bc) + 1e-300)
    inlet = (
        v_in.at[:, U].set(speed * bc.value[:, U]).at[:, V].set(speed * bc.value[:, V])
    )
    out = jnp.where(kind == TOTAL_PRESSURE_INLET, inlet, out)
    prescribe_v = (kind == WALL) | (kind == VELOCITY_INLET)
    out = out.at[:, U].set(jnp.where(prescribe_v[:, 0], bc.value[:, U], out[:, U]))
    out = out.at[:, V].set(jnp.where(prescribe_v[:, 0], bc.value[:, V], out[:, V]))
    out = out.at[:, P].set(
        jnp.where(kind[:, 0] == PRESSURE_OUTLET, bc.value[:, P], out[:, P])
    )
    return out


def beta_squared(model, u, bc):
    """beta^2 = beta_factor * max|v|^2 over cells and boundary values."""
    v2 = np.concatenate(
        (
            np.sum(np.asarray(u)[:, 1:] ** 2, axis=1),
            np.sum(np.asarray(bc.value)[:, 1:] ** 2, axis=1),
        )
    )
    return float(model.beta_factor * max(v2.max(), 1e-10))


def inviscid_flux(model, w, normal):
    """F.n for states w (n, 3) and unit normals (n, 2)."""
    rho = model.density
    vn = w[:, U] * normal[:, 0] + w[:, V] * normal[:, 1]
    return jnp.stack(
        (
            rho * vn,
            rho * w[:, U] * vn + w[:, P] * normal[:, 0],
            rho * w[:, V] * vn + w[:, P] * normal[:, 1],
        ),
        axis=1,
    )


def preconditioner(model, w, beta2):
    """Gamma for states w (n, 3): shape (n, 3, 3)."""
    rho = model.density
    n = w.shape[0]
    g = jnp.zeros((n, 3, 3))
    g = g.at[:, 0, 0].set(1.0 / beta2)
    g = g.at[:, 1, 0].set(w[:, U] / beta2)
    g = g.at[:, 2, 0].set(w[:, V] / beta2)
    g = g.at[:, 1, 1].set(rho)
    g = g.at[:, 2, 2].set(rho)
    return g


def preconditioner_inverse(model, w, beta2):
    """Gamma^-1 for states w (n, 3): shape (n, 3, 3)."""
    rho = model.density
    n = w.shape[0]
    g = jnp.zeros((n, 3, 3))
    g = g.at[:, 0, 0].set(beta2)
    g = g.at[:, 1, 0].set(-w[:, U] / rho)
    g = g.at[:, 2, 0].set(-w[:, V] / rho)
    g = g.at[:, 1, 1].set(1.0 / rho)
    g = g.at[:, 2, 2].set(1.0 / rho)
    return g


def abs_preconditioned_jacobian(model, w, normal, beta2):
    """|Gamma^-1 A| for mean states w (n, 3) and unit normals: (n, 3, 3).

    Eigenvalues v_n (tangential), v_n - beta, v_n + beta with eigenvectors
    (0, t), (-beta rho, n), (beta rho, n) and left eigenvectors (0, t),
    (-1/(rho beta), n)/2, (1/(rho beta), n)/2.
    """
    rho = model.density
    beta = jnp.sqrt(beta2)
    nx, ny = normal[:, 0], normal[:, 1]
    vn = w[:, U] * nx + w[:, V] * ny
    l0, lm, lp = jnp.abs(vn), jnp.abs(vn - beta), jnp.abs(vn + beta)
    s, d = 0.5 * (lm + lp), 0.5 * (lp - lm)
    tx, ty = -ny, nx
    m = jnp.zeros((w.shape[0], 3, 3))
    m = m.at[:, 0, 0].set(s)
    m = m.at[:, 0, 1].set(beta * rho * nx * d)
    m = m.at[:, 0, 2].set(beta * rho * ny * d)
    m = m.at[:, 1, 0].set(nx * d / (beta * rho))
    m = m.at[:, 2, 0].set(ny * d / (beta * rho))
    m = m.at[:, 1, 1].set(s * nx * nx + l0 * tx * tx)
    m = m.at[:, 1, 2].set(s * nx * ny + l0 * tx * ty)
    m = m.at[:, 2, 1].set(s * ny * nx + l0 * ty * tx)
    m = m.at[:, 2, 2].set(s * ny * ny + l0 * ty * ty)
    return m


def fds_flux(model, w_l, w_r, normal, beta2):
    """Flux-difference-splitting flux per unit length, states (n, 3)."""
    w_m = 0.5 * (w_l + w_r)
    diss = jnp.einsum(
        "nij,njk,nk->ni",
        preconditioner(model, w_m, beta2),
        abs_preconditioned_jacobian(model, w_m, normal, beta2),
        w_r - w_l,
    )
    central = 0.5 * (
        inviscid_flux(model, w_l, normal) + inviscid_flux(model, w_r, normal)
    )
    return central - 0.5 * diss


def viscous_flux(model, grad_v, normal, mu=None):
    """-(0, tau.n) for velocity gradients grad_v (n, 2, 2) [var, dir].

    ``mu``: per-face effective viscosity (array), default ``model.viscosity``.
    """
    mu = model.viscosity if mu is None else jnp.asarray(mu)
    mu3 = mu if jnp.ndim(mu) == 0 else mu[:, None, None]
    div = grad_v[:, 0, 0] + grad_v[:, 1, 1]
    tau = mu3 * (grad_v + jnp.swapaxes(grad_v, 1, 2))
    tau = tau - (2.0 / 3.0 * mu * div)[:, None, None] * jnp.eye(2)[None]
    tn = jnp.einsum("nij,nj->ni", tau, normal)
    return -jnp.concatenate((jnp.zeros((tn.shape[0], 1)), tn), axis=1)


def boundary_convective_flux(model, bc, normal, w_in, beta2):
    """Convective flux through boundary faces from the interior state w_in."""
    w_out = boundary_state(bc, normal, w_in)
    upwind = fds_flux(model, w_in, w_out, normal, beta2)
    # Walls and symmetry: mass flux of the prescribed velocity (zero for
    # tangential walls), momentum flux of pressure (and transpiration).
    wall = w_in.at[:, U].set(bc.value[:, U]).at[:, V].set(bc.value[:, V])
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
    """Limiter values psi of the current state, shape (n_cells, 3)."""
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    return _limiter(model, geom, w, w_b, lsq_gradient(geom, w, w_b))


@partial(jax.jit, static_argnums=(0, 4))
def residual(model, geom, bc, w, order, beta2, psi=None):
    """Residual R(V), shape (n_cells, 3): net outflow minus body force.

    Parameters
    ----------
    model : IncompressibleModel
    geom : FVGeometry
    bc : BoundaryData
    w : jax array, shape (n_cells, 3)
    order : int
        Spatial order of the convective states (1 or 2); static.
    beta2 : float
        Artificial compressibility beta^2 (frozen for the evaluation).
    psi : jax array, shape (n_cells, 3), optional
        Frozen limiter values (D23); default: computed from ``w``.
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
    if model.viscosity > 0.0:
        g_f = interior_face_gradient(geom, w, grad, model.viscous_gradient, model.alpha)
        g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
        f_flux = f_flux + viscous_flux(model, g_f[:, 1:], geom.f_normal)
        b_flux = b_flux + viscous_flux(model, g_b[:, 1:], geom.b_normal)
    r = divergence(geom, f_flux, b_flux)
    force = jnp.asarray(model.body_force)
    r = r.at[:, 1:].add(-geom.area[:, None] * force[None, :])
    if model.pin is not None:
        cell, p0 = model.pin
        r = r.at[cell, P].set(w[cell, P] - p0)
    return r


@partial(jax.jit, static_argnums=(0, 5))
def residual_jvp(model, geom, bc, w, dw, order, beta2, psi=None):
    """Exact Jacobian-vector product dR/dw(w) @ dw by forward-mode AD (D25)."""

    def f(x):
        return residual(model, geom, bc, x, order, beta2, psi)

    return jax.jvp(f, (w,), (dw,))[1]


def interior_flux(model, beta2, w_o, w_n, normal, delta):
    """Compact first-order flux through an interior face (for the Jacobian)."""
    conv = fds_flux(model, w_o[None], w_n[None], normal[None], beta2)[0]
    if model.viscosity > 0.0:
        dn = jnp.dot(delta, normal)
        if model.viscous_gradient == "alpha-damping":
            coeff = model.alpha / dn
        else:
            coeff = dn / jnp.dot(delta, delta)
        # Two-point part of the face gradient: d(v)/dn * n
        g = (coeff * (w_n[1:] - w_o[1:]))[:, None] * normal[None, :]
        conv = conv + viscous_flux(model, g[None], normal[None])[0]
    return conv


def boundary_flux(model, beta2, w_o, bc_kind, bc_value, normal, delta):
    """Compact first-order flux through a boundary face (for the Jacobian)."""
    bc = BoundaryData(bc_kind[None], bc_value[None], model.density)
    conv = boundary_convective_flux(model, bc, normal[None], w_o[None], beta2)[0]
    if model.viscosity > 0.0:
        w_b = boundary_state(bc, normal[None], w_o[None])[0]
        dn = jnp.dot(delta, normal)
        g = ((w_b[1:] - w_o[1:]) / dn)[:, None] * normal[None, :]
        conv = conv + viscous_flux(model, g[None], normal[None])[0]
    return conv


@partial(jax.jit, static_argnums=0)
def spectral_radius(model, geom, w, beta2):
    """Sum over faces of (|v_n| + beta + 2 nu / (d.n)) * length, per cell."""
    nu = model.viscosity / model.density
    beta = jnp.sqrt(beta2)
    n = w.shape[0]
    o, nb, b = geom.f_owner, geom.f_neigh, geom.b_owner
    v_f = 0.5 * (w[o, 1:] + w[nb, 1:])
    dn_f = jnp.einsum("fk,fk->f", geom.f_delta, geom.f_normal)
    lam_f = (
        jnp.abs(jnp.einsum("fk,fk->f", v_f, geom.f_normal)) + beta + 2.0 * nu / dn_f
    ) * geom.f_length
    dn_b = jnp.einsum("fk,fk->f", geom.b_delta, geom.b_normal)
    lam_b = (
        jnp.abs(jnp.einsum("fk,fk->f", w[b, 1:], geom.b_normal))
        + beta
        + 2.0 * nu / dn_b
    ) * geom.b_length
    return (
        jax.ops.segment_sum(lam_f, o, num_segments=n)
        + jax.ops.segment_sum(lam_f, nb, num_segments=n)
        + jax.ops.segment_sum(lam_b, b, num_segments=n)
    )


@partial(jax.jit, static_argnums=0)
def explicit_update(model, geom, w, r, beta2, cfl):
    """Explicit increment CFL / Lambda_i * Gamma_i^-1 R_i, shape (n_cells, 3)."""
    lam = spectral_radius(model, geom, w, beta2)
    ginv = preconditioner_inverse(model, w, beta2)
    return jnp.einsum("nij,nj->ni", ginv, r) * (cfl / lam)[:, None]


def time_term_blocks(model, geom, w, beta2, cfl):
    """Pseudo-time blocks Gamma_i * Lambda_i / CFL, shape (n_cells, 3, 3)."""
    w = jnp.asarray(w)
    gamma = np.asarray(preconditioner(model, w, beta2))
    lam = np.asarray(spectral_radius(model, geom, w, beta2))
    return gamma * (lam / cfl)[:, None, None]

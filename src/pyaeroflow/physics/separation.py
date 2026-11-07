"""Pressure-gradient separation sensor for k-omega SST (L5, "SST-lambda_theta").

Griffin, Lee, Vijayakumar, Savino, Bornhoft, Shende & Whitmore (2026,
arXiv:2610.33268) lower the Bradshaw-limiter coefficient a1 of the eddy
viscosity where a local pressure-gradient parameter says that the boundary
layer is close to separation:

    nu_t = k / omega               if a1 omega > S F2,
    nu_t = a1' k / (S F2)          otherwise,
    a1'  = a1,APG                  if lambda_theta < lambda_theta,c, else a1.

lambda_theta is the local approximation of Menter et al. (2015),

    lambda_theta = clip(-7.57e-3 H_L + 0.0128, -1, 1),
    H_L = d^2 / nu  d(u . n)/dn,

with the unit wall-distance gradient n (the quantity of
:func:`pyaeroflow.physics.rans.local_shape_factor`), and the critical value
blends the laminar Falkner-Skan limit with the continuously critical
turbulent boundary layer (Buri; Allan) by the intermittency gamma,

    lambda_theta,c = (1 - gamma) (-0.0681) + gamma Gamma Re_theta^(3/4),
    Re_theta = d |u_s| / nu,  u_s = u - (u . n) n,

with Gamma = -5e-4 and a1,APG = 0.275 (calibrated by the authors on the S809
airfoil). The fully turbulent model has gamma = 1.

The switch is cell-local and enters the residual through nu_t and the omega
production only. It is evaluated from the current state and held fixed
(``stop_gradient``) in every linearisation, like the BCM intermittency: a
step has no useful derivative, and the derivatives of the second-derivative-
like H_L through the least-squares gradients would only stiffen the Newton
steps.

All functions are pure and cell-local; they work with kinematic viscosity, so
the same code serves incompressible and compressible flow (nu = mu / rho).
"""

import jax
import jax.numpy as jnp

A1_APG = 0.275  # a1 in regions of strong adverse pressure gradient
GAMMA_TURBULENT = -5.0e-4  # Gamma of the continuously critical theory
LAMBDA_LAMINAR = -0.0681  # Falkner-Skan laminar separation value
C_SLOPE = -7.57e-3  # lambda_theta = C_SLOPE H_L + C_OFFSET (Menter et al., 2015)
C_OFFSET = 0.0128
TINY = 1.0e-300


def _safe_sqrt(x):
    """sqrt(max(x, 0)) with a zero derivative for x <= 0."""
    positive = x > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def wall_normal_gradient(grad, normal):
    """d(u . n)/dn per cell for a constant unit normal n (the wall-distance gradient).

    ``grad`` has shape (n_cells, 2, 2) with grad[:, i, j] = du_i/dx_j, i.e. the
    rows of the velocity components u and v of the cell gradient.
    """
    nx, ny = normal[:, 0], normal[:, 1]
    gx = grad[:, 0, 0] * nx + grad[:, 0, 1] * ny  # du/dn
    gy = grad[:, 1, 0] * nx + grad[:, 1, 1] * ny  # dv/dn
    return gx * nx + gy * ny


def lambda_theta(shape_factor):
    """clip(-7.57e-3 H_L + 0.0128, -1, 1) with H_L = d^2/nu d(u.n)/dn."""
    return jnp.clip(C_SLOPE * shape_factor + C_OFFSET, -1.0, 1.0)


def reynolds_theta(u, v, normal, distance, nu):
    """Re_theta = d |u_s| / nu with u_s the velocity without its wall-normal part."""
    q = u * normal[:, 0] + v * normal[:, 1]
    return distance * _safe_sqrt(u * u + v * v - q * q) / nu


def critical_lambda(re_theta, gamma=1.0, gamma_coefficient=GAMMA_TURBULENT):
    """(1 - gamma) lambda_theta,l + gamma Gamma Re_theta^(3/4)."""
    return (1.0 - gamma) * LAMBDA_LAMINAR + gamma * gamma_coefficient * (
        jnp.maximum(re_theta, 0.0) ** 0.75
    )


def sensor(
    u,
    v,
    grad_uv,
    normal,
    distance,
    nu,
    gamma=1.0,
    gamma_coefficient=GAMMA_TURBULENT,
):
    """(lambda_theta, lambda_theta,c) per cell."""
    dq_dn = wall_normal_gradient(grad_uv, normal)
    lam = lambda_theta(distance**2 / nu * dq_dn)
    lam_c = critical_lambda(
        reynolds_theta(u, v, normal, distance, nu), gamma, gamma_coefficient
    )
    return lam, lam_c


def a1_field(
    u,
    v,
    grad_uv,
    normal,
    distance,
    nu,
    a1,
    a1_apg=A1_APG,
    gamma=1.0,
    gamma_coefficient=GAMMA_TURBULENT,
):
    """a1' per cell: ``a1_apg`` where lambda_theta < lambda_theta,c, else ``a1``.

    Held fixed in every linearisation (``stop_gradient``).
    """
    lam, lam_c = sensor(u, v, grad_uv, normal, distance, nu, gamma, gamma_coefficient)
    return jax.lax.stop_gradient(jnp.where(lam < lam_c, a1_apg, a1))

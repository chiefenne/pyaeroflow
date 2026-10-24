"""Menter k-omega SST turbulence model, version SST-2003m as in SU2 (L2, D12).

Kinematic form for incompressible flow (rho constant):

    dk/dt + div(v k) = P_k - beta* k omega + div((nu + sigma_k nu_t) grad k)
    domega/dt + div(v omega) = P_omega - beta omega^2 + CD
                               + div((nu + sigma_omega nu_t) grad omega)

with the eddy viscosity nu_t = a1 k / max(a1 omega, S F2) (S the strain-rate
magnitude), the limited production P_k = max(0, min(nu_t S^2, 10 beta* k
omega)), P_omega = gamma P_k / nu_t and the cross diffusion
CD = (1 - F1) 2 sigma_omega2 grad k . grad omega / omega. The constants
sigma_k, sigma_omega, beta and gamma blend set 1 (k-omega, near the wall) and
set 2 (k-epsilon, outer) with F1. Formulas and constants are those of SU2
(``CSourcePieceWise_TurbSST``, ``CTurbSSTVariable::SetBlendingFunc``,
``CTurbSSTSolver``): Menter, Kuntz & Langtry (2003), NASA TMR "SST-2003m"
(the cross diffusion as a source, SU2 up to 7.5).
"""

import jax.numpy as jnp

SIGMA_K1, SIGMA_K2 = 0.85, 1.0
SIGMA_W1, SIGMA_W2 = 0.5, 0.856
BETA_1, BETA_2 = 0.075, 0.0828
BETA_STAR = 0.09
A1 = 0.31
KAPPA = 0.41
GAMMA_1 = BETA_1 / BETA_STAR - SIGMA_W1 * KAPPA**2 / BETA_STAR**0.5
GAMMA_2 = BETA_2 / BETA_STAR - SIGMA_W2 * KAPPA**2 / BETA_STAR**0.5
PROD_LIMIT = 10.0  # production limiter constant (SU2: 10 for SST-2003)
CD_FLOOR = 1.0e-10  # 10^-PROD_LIMIT, as in SU2
OMEGA_WALL = 60.0  # omega_wall = 60 nu / (beta_1 d_1^2) (Menter, SU2)
TINY = 1.0e-300


def _safe_sqrt(x):
    """sqrt(max(x, 0)) with a zero derivative for x <= 0."""
    positive = x > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def blend(f1, inner, outer):
    """F1 inner + (1 - F1) outer."""
    return f1 * inner + (1.0 - f1) * outer


def cross_diffusion(k_grad_omega, omega):
    """CD_kw = max(2 sigma_omega2 grad k . grad omega / omega, 1e-10)."""
    return jnp.maximum(2.0 * SIGMA_W2 * k_grad_omega / omega, CD_FLOOR)


def blending(k, omega, nu, distance, k_grad_omega):
    """Menter's blending functions (F1, F2), SU2's ``SetBlendingFunc``.

    ``k_grad_omega`` is grad k . grad omega; ``k`` must be non-negative and
    ``omega`` positive.
    """
    d = distance
    sqrt_k = _safe_sqrt(k)
    arg2a = sqrt_k / (BETA_STAR * omega * d + TINY)
    arg2b = 500.0 * nu / (d * d * omega + TINY)
    arg2 = jnp.maximum(arg2a, arg2b)
    cd = cross_diffusion(k_grad_omega, omega)
    arg1 = jnp.minimum(arg2, 4.0 * SIGMA_W2 * k / (cd * d * d + TINY))
    f1 = jnp.tanh(arg1**4)
    f2 = jnp.tanh(jnp.maximum(2.0 * arg2a, arg2b) ** 2)
    return f1, f2


def zeta(omega, strain, f2):
    """max(omega, S F2 / a1): nu_t = k / zeta."""
    return jnp.maximum(omega, strain * f2 / A1)


def eddy_viscosity(k, omega, strain, f2):
    """nu_t = a1 k / max(a1 omega, S F2), zero for k <= 0."""
    return jnp.maximum(k, 0.0) / zeta(omega, strain, f2)


def sources(k, omega, strain, f1, f2, k_grad_omega, production_switch=1.0):
    """Return the sources per unit volume (s_k, s_omega) of SST-2003m.

    P_k = max(0, min(nu_t S^2, 10 beta* k omega)) times the production switch
    (prescribed laminar regions), P_omega = gamma P_k / nu_t written as
    gamma min(S^2, 10 beta* omega zeta) (no 0/0 where k = 0), destruction
    beta* k omega and beta omega^2, cross diffusion (1 - F1) 2 sigma_omega2
    grad k . grad omega / omega (unclipped, as the source of SU2 7.5).
    """
    k_pos = jnp.maximum(k, 0.0)
    z = zeta(omega, strain, f2)
    nu_t = k_pos / z
    s2 = strain**2
    p_k = production_switch * jnp.maximum(
        0.0, jnp.minimum(nu_t * s2, PROD_LIMIT * BETA_STAR * k_pos * omega)
    )
    gamma = blend(f1, GAMMA_1, GAMMA_2)
    beta = blend(f1, BETA_1, BETA_2)
    p_w = jnp.where(
        k > 0.0,
        production_switch
        * gamma
        * jnp.maximum(0.0, jnp.minimum(s2, PROD_LIMIT * BETA_STAR * omega * z)),
        0.0,
    )
    cd = (1.0 - f1) * 2.0 * SIGMA_W2 * k_grad_omega / omega
    s_k = p_k - BETA_STAR * k * omega
    s_w = p_w - beta * omega**2 + cd
    return s_k, s_w


def diffusion_coefficients(nu, nu_t, f1):
    """(nu + sigma_k nu_t, nu + sigma_omega nu_t) with the F1-blended sigmas."""
    return (
        nu + blend(f1, SIGMA_K1, SIGMA_K2) * nu_t,
        nu + blend(f1, SIGMA_W1, SIGMA_W2) * nu_t,
    )


def wall_omega(nu, d1):
    """Wall omega of a smooth wall: 60 nu / (beta_1 d1^2), d1 the first spacing."""
    return OMEGA_WALL * nu / (BETA_1 * d1**2)


def freestream(viscosity_ratio, kinetic_energy, nu):
    """(k, omega) from k and nu_t / nu: omega = k / nu_t."""
    return kinetic_energy, kinetic_energy / (viscosity_ratio * nu)

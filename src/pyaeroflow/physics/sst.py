"""Menter k-omega SST turbulence model, version SST-2003m (L2, D12).

Kinematic form for incompressible flow (rho constant):

    dk/dt + div(v k) = P_k - beta* k omega + div((nu + sigma_k nu_t) grad k)
    domega/dt + div(v omega) = P_omega - beta omega^2 + CD
                               + div((nu + sigma_omega nu_t) grad omega)

with the eddy viscosity nu_t = a1 k / max(a1 omega, S F2) (S the strain-rate
magnitude), the limited production P_k = max(0, min(nu_t S^2, 10 beta* k
omega)), P_omega = gamma P_k / nu_t and the cross diffusion
CD = (1 - F1) 2 sigma_omega2 grad k . grad omega / omega. The constants
sigma_k, sigma_omega, beta and gamma blend set 1 (k-omega, near the wall) and
set 2 (k-epsilon, outer) with F1. Formulas and constants are those of Menter, Kuntz &
Langtry (2003) and the NASA TMR page "SST-2003m" (the cross diffusion as a source).
"""

import jax.numpy as jnp

from pyaeroflow.physics.lm import destruction_factor

SIGMA_K1, SIGMA_K2 = 0.85, 1.0
SIGMA_W1, SIGMA_W2 = 0.5, 0.856
BETA_1, BETA_2 = 0.075, 0.0828
BETA_STAR = 0.09
A1 = 0.31
KAPPA = 0.41
GAMMA_1 = BETA_1 / BETA_STAR - SIGMA_W1 * KAPPA**2 / BETA_STAR**0.5
GAMMA_2 = BETA_2 / BETA_STAR - SIGMA_W2 * KAPPA**2 / BETA_STAR**0.5
PROD_LIMIT = 10.0  # production limiter constant (10 for SST-2003)
CD_FLOOR = 1.0e-10  # floor of CD_kw (Menter, Kuntz & Langtry 2003)
OMEGA_WALL = 60.0  # omega_wall = 60 nu / (beta_1 d_1^2) (Menter 1994)
TINY = 1.0e-300  # denominator floor (float64)
# Floor in single precision (float32 Jacobian products,
# newton_krylov.product_precision): 1e-300 is 0 in float32, and the floor
# appears squared in the tangent, so it must stay a normal float32 number.
TINY_SINGLE = 1.0e-18


def tiny(x):
    """Denominator floor for the precision of ``x`` (float64 unchanged)."""
    return TINY if jnp.result_type(x) == jnp.float64 else TINY_SINGLE


# tanh(x) = 1 to double precision for x > 20: the tanh arguments are clamped
# before the power so that the power cannot overflow in single precision (an
# infinite argument gives a 0 * inf = NaN tangent). Exact in double precision.
F1_ARG_MAX = 10.0
F2_ARG_MAX = 100.0


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
    """Menter's blending functions (F1, F2).

    ``k_grad_omega`` is grad k . grad omega; ``k`` must be non-negative and
    ``omega`` positive.
    """
    d = distance
    sqrt_k = _safe_sqrt(k)
    eps = tiny(omega)
    arg2a = sqrt_k / (BETA_STAR * omega * d + eps)
    arg2b = 500.0 * nu / (d * d * omega + eps)
    arg2 = jnp.maximum(arg2a, arg2b)
    cd = cross_diffusion(k_grad_omega, omega)
    arg1 = jnp.minimum(arg2, 4.0 * SIGMA_W2 * k / (cd * d * d + eps))
    f1 = jnp.tanh(jnp.minimum(arg1, F1_ARG_MAX) ** 4)
    f2 = jnp.tanh(jnp.minimum(jnp.maximum(2.0 * arg2a, arg2b), F2_ARG_MAX) ** 2)
    return f1, f2


def zeta(omega, strain, f2):
    """max(omega, S F2 / a1): nu_t = k / zeta."""
    return jnp.maximum(omega, strain * f2 / A1)


def eddy_viscosity(k, omega, strain, f2):
    """nu_t = a1 k / max(a1 omega, S F2), zero for k <= 0."""
    return jnp.maximum(k, 0.0) / zeta(omega, strain, f2)


def sources(
    k,
    omega,
    strain,
    f1,
    f2,
    k_grad_omega,
    production_switch=1.0,
    gamma_eff=None,
    divergence=None,
    gamma_smooth=0.0,
):
    """Return the sources per unit volume (s_k, s_omega) of SST-2003m.

    P_k = max(0, min(nu_t S^2, 10 beta* k omega)) times the production switch
    (prescribed laminar regions), P_omega = gamma P_k / nu_t written as
    gamma min(S^2, 10 beta* omega zeta) (no 0/0 where k = 0), destruction
    beta* k omega and beta omega^2, cross diffusion (1 - F1) 2 sigma_omega2
    grad k . grad omega / omega (unclipped, as a source term).

    ``gamma_eff`` (Langtry-Menter, :mod:`pyaeroflow.physics.lm`) multiplies
    P_k and min(max(gamma_eff, 0.1), 1) the k destruction; P_omega keeps the
    unmodified P_k. ``gamma_smooth`` > 0 gives that clip the smooth
    derivative of :func:`pyaeroflow.physics.lm.destruction_factor` (value
    unchanged).

    ``divergence`` (compressible flow, :mod:`pyaeroflow.physics.rans_sst_compressible`):
    div v; the production is then that of the full Boussinesq stress with
    its -2/3 rho k delta_ij part, P_k = nu_t S^2 - 2/3 k div v, and
    P_omega = gamma P_k / nu_t, i.e.
    S^2 - 2/3 zeta div v in place of S^2, both before the limiter.
    """
    k_pos = jnp.maximum(k, 0.0)
    z = zeta(omega, strain, f2)
    nu_t = k_pos / z
    s2 = strain**2
    p_k_raw = nu_t * s2
    s2_w = s2
    if divergence is not None:
        p_k_raw = p_k_raw - 2.0 / 3.0 * k_pos * divergence
        s2_w = s2 - 2.0 / 3.0 * z * divergence
    p_k = production_switch * jnp.maximum(
        0.0, jnp.minimum(p_k_raw, PROD_LIMIT * BETA_STAR * k_pos * omega)
    )
    gamma = blend(f1, GAMMA_1, GAMMA_2)
    beta = blend(f1, BETA_1, BETA_2)
    p_w = jnp.where(
        k > 0.0,
        production_switch
        * gamma
        * jnp.maximum(0.0, jnp.minimum(s2_w, PROD_LIMIT * BETA_STAR * omega * z)),
        0.0,
    )
    cd = (1.0 - f1) * 2.0 * SIGMA_W2 * k_grad_omega / omega
    d_k = BETA_STAR * k * omega
    if gamma_eff is not None:
        d_k = destruction_factor(gamma_eff, gamma_smooth) * d_k
        s_k = gamma_eff * p_k - d_k
    else:
        s_k = p_k - d_k
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

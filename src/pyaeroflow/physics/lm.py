"""Langtry-Menter gamma-Re_theta transition model on SST (SST-2003m-LM2009, L3).

Two transport equations join k-omega SST (:mod:`pyaeroflow.physics.sst`):

    D gamma/Dt    = P_gamma - E_gamma + div((nu + nu_t / sigma_f) grad gamma)
    D Re_thetat/Dt = P_thetat + div(sigma_thetat (nu + nu_t) grad Re_thetat)

with the intermittency gamma and the transported transition-onset momentum
thickness Reynolds number Re_thetat. The coupling to SST is

    P_k -> gamma_eff P_k,   D_k -> min(max(gamma_eff, 0.1), 1) D_k,
    gamma_eff = max(gamma, gamma_sep),   F1 -> max(F1, F3),

the omega equation is unchanged. Formulas, constants and limits are those of
the NASA TMR page "langtrymenter_4eqn" (SST-2003-LM2009: Langtry & Menter,
AIAA J. 47(12), 2009, with F_turb of Menter, Langtry & Voelker 2006) and of
SU2 (``CSourcePieceWise_TransLM``, ``TransLMCorrelations`` with the
MENTER_LANGTRY correlation, ``CTransLMSolver::Postprocessing``,
``CTurbSSTVariable::SetBlendingFunc``). Tu and dU/ds are local values (TMR,
SU2). All functions are pure and cell-local (kinematic form, constant
density).
"""

import jax
import jax.numpy as jnp

CA1, CA2 = 2.0, 0.06
CE1, CE2 = 1.0, 50.0
C_THETA = 0.03
S1 = 2.0
SIGMA_F = 1.0
SIGMA_THETA = 2.0
TU_MIN = 0.027  # percent
LAMBDA_LIMIT = 0.1
GAMMA_LIMITS = (1.0e-4, 5.0)  # solution limits of SU2 (CTransLMSolver)
RE_THETA_LIMITS = (1.0e-4, 1.0e15)
RE_THETA_EQ_MIN = 20.0
EQ_ITERATIONS = 12  # fixed-point iterations of Re_thetat^eq(lambda(theta))
TINY = 1.0e-30


def _safe_sqrt(x):
    """sqrt(max(x, 0)) with a zero derivative for x <= 0."""
    positive = x > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def turbulence_intensity(k, speed):
    """Local Tu = 100 sqrt(2k/3) / U in percent, at least 0.027."""
    tu = 100.0 * _safe_sqrt(2.0 * k / 3.0) / jnp.maximum(speed, TINY)
    return jnp.maximum(tu, TU_MIN)


def _re_theta_tu(tu):
    """Return the Tu part of Re_thetat^eq (F(lambda) = 1)."""
    tu_lo = jnp.minimum(tu, 1.3)
    tu_hi = jnp.maximum(tu, 1.3 + 1e-12)
    return jnp.where(
        tu <= 1.3,
        1173.51 - 589.428 * tu_lo + 0.2196 / tu_lo**2,
        331.5 * (tu_hi - 0.5658) ** (-0.671),
    )


def freestream_re_theta(tu):
    """Re_thetat at inflow faces: the correlation with F(lambda) = 1 (TMR).

    ``tu`` in percent; the limit Tu >= 0.027 applies.
    """
    return _re_theta_tu(jnp.maximum(tu, TU_MIN))


def pressure_gradient_factor(lam, tu):
    """F(lambda_theta) of the Re_thetat^eq correlation."""
    neg = 1.0 + (12.986 * lam + 123.66 * lam**2 + 405.689 * lam**3) * jnp.exp(
        -((tu / 1.5) ** 1.5)
    )
    pos = 1.0 + 0.275 * (1.0 - jnp.exp(-35.0 * lam)) * jnp.exp(-tu / 0.5)
    return jnp.where(lam <= 0.0, neg, pos)


def equilibrium_re_theta(tu, du_ds, nu, speed):
    """Re_thetat^eq(Tu, lambda_theta), solved by fixed-point iteration on theta.

    lambda_theta = theta^2 / nu dU/ds with theta = Re_thetat^eq nu / U,
    limited to [-0.1, 0.1]; Re_thetat^eq >= 20 (TMR, SU2).
    """
    base = _re_theta_tu(tu)
    scale = nu * du_ds / jnp.maximum(speed, TINY) ** 2  # lambda / Re_theta^2
    re = jnp.maximum(base, RE_THETA_EQ_MIN)
    for _ in range(EQ_ITERATIONS):
        lam = jnp.clip(re**2 * scale, -LAMBDA_LIMIT, LAMBDA_LIMIT)
        re = jnp.maximum(base * pressure_gradient_factor(lam, tu), RE_THETA_EQ_MIN)
    return re


def critical_re_theta(re_t):
    """Re_thetac(Re_thetat), Langtry-Menter 2009 (MENTER_LANGTRY in SU2)."""
    r = jnp.minimum(re_t, 1870.0)
    poly = (
        -396.035e-2
        + 10120.656e-4 * r
        - 868.230e-6 * r**2
        + 696.506e-9 * r**3
        - 174.105e-12 * r**4
    )
    return jnp.where(re_t <= 1870.0, poly, re_t - (593.11 + 0.482 * (re_t - 1870.0)))


def length_function(re_t):
    """F_length,1(Re_thetat), Langtry-Menter 2009."""
    f_a = 39.8189 - 119.270e-4 * re_t - 132.567e-6 * re_t**2
    f_b = 263.404 - 123.939e-2 * re_t + 194.548e-5 * re_t**2 - 101.695e-8 * re_t**3
    f_c = 0.5 - 3.0e-4 * (re_t - 596.0)
    return jnp.where(
        re_t < 400.0,
        f_a,
        jnp.where(re_t < 596.0, f_b, jnp.where(re_t < 1200.0, f_c, 0.3188)),
    )


def blending_f3(k, nu, distance):
    """F3 = exp(-(R_y/120)^8), R_y = d sqrt(k)/nu: F1 -> max(F1, F3)."""
    r_y = distance * _safe_sqrt(k) / nu
    return jnp.exp(-((r_y / 120.0) ** 8))


def _theta_blending(gamma, re_t, omega, nu, vorticity, distance, speed):
    """F_thetat of the Re_thetat production (and of gamma_sep)."""
    u2 = jnp.maximum(speed, TINY) ** 2
    re_omega = omega * distance**2 / nu
    f_wake = jnp.exp(-((re_omega / 1.0e5) ** 2))
    delta = 375.0 * vorticity * nu * re_t * distance / u2 + 1.0e-20
    var = (CE2 * gamma - 1.0) / (CE2 - 1.0)
    return jnp.minimum(
        jnp.maximum(f_wake * jnp.exp(-((distance / delta) ** 4)), 1.0 - var**2), 1.0
    )


def _reynolds_numbers(k, omega, nu, strain, distance):
    """(Re_V, R_T, Re_omega)."""
    return strain * distance**2 / nu, k / (nu * omega), omega * distance**2 / nu


def separation_intermittency(
    gamma, re_t, k, omega, nu, strain, vorticity, distance, speed
):
    """gamma_sep = min(s1 max(0, Re_V/(3.235 Re_thetac) - 1) F_reattach, 2) F_thetat."""
    re_v, r_t, _ = _reynolds_numbers(k, omega, nu, strain, distance)
    f_reattach = jnp.exp(-((r_t / 20.0) ** 4))
    f_theta = _theta_blending(gamma, re_t, omega, nu, vorticity, distance, speed)
    g = S1 * jnp.maximum(0.0, re_v / (3.235 * critical_re_theta(re_t)) - 1.0)
    return jnp.clip(jnp.minimum(g * f_reattach, 2.0) * f_theta, 0.0, 2.0)


def effective_intermittency(
    gamma, re_t, k, omega, nu, strain, vorticity, distance, speed
):
    """gamma_eff = max(gamma, gamma_sep), the factor on SST's P_k."""
    g_sep = separation_intermittency(
        gamma, re_t, k, omega, nu, strain, vorticity, distance, speed
    )
    return jnp.maximum(gamma, g_sep)


def destruction_factor(gamma_eff):
    """min(max(gamma_eff, 0.1), 1): the factor on SST's D_k."""
    return jnp.clip(gamma_eff, 0.1, 1.0)


def sources(
    gamma, re_t, k, omega, nu, strain, vorticity, distance, speed, du_ds, positive=False
):
    """Return the sources per unit volume (s_gamma, s_re_thetat), kinematic.

    Parameters
    ----------
    gamma, re_t : arrays
        Intermittency and transported Re_thetat.
    k, omega : arrays
        SST variables (k >= 0, omega > 0).
    nu : float
        Kinematic viscosity.
    strain, vorticity : arrays
        S = sqrt(2 S_ij S_ij) and Omega = sqrt(2 W_ij W_ij).
    distance : array
        Wall distance d.
    speed, du_ds : arrays
        Local velocity magnitude U and dU/ds = u_m u_n / U^2 du_m/dx_n.
    positive : bool
        Linearization of the gamma source; the values are the same. With
        True the derivatives with respect to gamma that would lower the
        diagonal of the gamma equation are left out (``stop_gradient``): the
        sqrt(gamma) of the production, whose derivative 1 / (2 sqrt gamma) is
        50 at the clip floor and outweighs the convective diagonal of a cell
        (the Newton update then goes against the residual), and the
        destruction derivative where it is negative (gamma < 1 / (2 c_e2)).
        Kept: the (1 - c_e1 gamma) factor of the production and the
        destruction derivative where it is positive, so d s_gamma / d gamma
        <= 0 everywhere. The derivatives with respect to the other variables
        stay exact.
    """
    g = jnp.maximum(gamma, 0.0)
    re_v, r_t, re_omega = _reynolds_numbers(k, omega, nu, strain, distance)
    re_c = critical_re_theta(re_t)
    f_sub = jnp.exp(-((re_omega / 200.0) ** 2))
    f_length = length_function(re_t) * (1.0 - f_sub) + 40.0 * f_sub
    f1 = re_v / (2.193 * re_c)
    f2 = jnp.minimum(jnp.maximum(f1, f1**4), 2.0)
    f3 = jnp.maximum(1.0 - (r_t / 2.5) ** 3, 0.0)
    f_onset = jnp.maximum(f2 - f3, 0.0)
    f_turb = jnp.exp(-((r_t / 4.0) ** 4))
    g_root = jax.lax.stop_gradient(g) if positive else g
    p_g = f_length * CA1 * strain * _safe_sqrt(g_root * f_onset) * (1.0 - CE1 * g)
    h = g * (CE2 * g - 1.0)  # h' = 2 c_e2 gamma - 1, negative below gamma = 0.01
    if positive:
        h = jnp.where(2.0 * CE2 * g > 1.0, h, jax.lax.stop_gradient(h))
    e_g = CA2 * vorticity * f_turb * h

    tu = turbulence_intensity(k, speed)
    re_eq = equilibrium_re_theta(tu, du_ds, nu, speed)
    inv_t = jnp.maximum(speed, TINY) ** 2 / (500.0 * nu)  # 1 / T
    f_theta = _theta_blending(gamma, re_t, omega, nu, vorticity, distance, speed)
    p_t = C_THETA * inv_t * (re_eq - re_t) * (1.0 - f_theta)
    return p_g - e_g, p_t


def diffusion_coefficients(nu, nu_t):
    """(nu + nu_t / sigma_f, sigma_thetat (nu + nu_t))."""
    return nu + nu_t / SIGMA_F, SIGMA_THETA * (nu + nu_t)

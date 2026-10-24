"""SA-AFT2019b transition model: amplification factor and intermittency (L3).

Coder's amplification factor transport model as documented on the NASA
Turbulence Modeling Resource (``aft_transition_3eqn``), with the 2019b
correlations of Coder, AIAA 2019-0039 (including the H12 limits of its
eq. 3). Two transport equations join SA:

    D n/Dt     = Omega F_crit F_growth dn/dRe_theta
                 + div(sigma_n (nu + nu_t) grad n)
    D g/Dt     = c_1 S F_onset (1 - exp(g)) - c_2 Omega F_turb (c_3 exp(g) - 1)
                 + div((nu + nu_t / sigma_y) grad g)

with n the amplification factor and g = ln(gamma) the log-intermittency.
The only change to SA is f_t2 = c_t3 (1 - exp(g)). All functions here are
pure and cell-local (kinematic form, constant density); the derivation and
the checks against the papers are in ``docs/aft_design.md``.
"""

import jax.numpy as jnp
import numpy as np

SIGMA_N = 1.0
SIGMA_Y = 1.0
C1 = 100.0
C2 = 0.06
C3 = 50.0
CT3 = 1.2
H12_MIN, H12_MAX = 2.2, 20.0


def critical_amplification(turbulence_intensity):
    """N_crit = -8.43 - 2.4 ln(tau/100), tau = 2.5 tanh(Tu/2.5), Tu in percent.

    Mack's relation with Drela's smooth limit; 8.15 at Tu = 0.1 %.
    """
    tau = 2.5 * np.tanh(float(turbulence_intensity) / 2.5)
    return -8.43 - 2.4 * np.log(tau / 100.0)


def shape_factor(local_shape_factor):
    """H12 = min(max(0.26 H_L + 2.4, 2.2), 20) (2019b, eq. 3)."""
    return jnp.clip(0.26 * local_shape_factor + 2.4, H12_MIN, H12_MAX)


def growth_rate(h12):
    """dn/dRe_theta = 0.028 (H12 - 1) - 0.0345 exp(-(3.87/(H12 - 1) - 2.52)^2)."""
    hm = h12 - 1.0
    return 0.028 * hm - 0.0345 * jnp.exp(-((3.87 / hm - 2.52) ** 2))


def growth_factor(h12):
    """F_growth = D (1 + m)/2 l with the 2019b D = 2.4 H12 / (H12 - 1)."""
    d = 2.4 * h12 / (h12 - 1.0)
    l_h = (6.54 * h12 - 14.07) / h12**2
    m = (0.058 * (h12 - 4.0) ** 2 / (h12 - 1.0) - 0.068) / l_h
    return d * (1.0 + m) / 2.0 * l_h


def critical_reynolds_theta(h12):
    """Re_theta0: log10 = 0.7 tanh(14/(H12-1) - 9.24) + 2.492/(H12-1)^0.43 + 0.62."""
    hm = h12 - 1.0
    log10 = 0.7 * jnp.tanh(14.0 / hm - 9.24) + 2.492 / hm**0.43 + 0.62
    return 10.0**log10


def critical_reynolds_vorticity(h12):
    """Re_V0 = k_V Re_theta0, k_V = 1/(0.4036 H12^2 - 2.5394 H12 + 4.3273)."""
    k_v = 1.0 / (0.4036 * h12**2 - 2.5394 * h12 + 4.3273)
    return k_v * critical_reynolds_theta(h12)


def onset_function(amplification, n_crit, r_t):
    """F_onset = max(min(n/N_crit, 2) - max(1 - (R_T/3.5)^3, 0), 0)."""
    f2 = jnp.minimum(amplification / n_crit, 2.0)
    f3 = jnp.maximum(1.0 - (r_t / 3.5) ** 3, 0.0)
    return jnp.maximum(f2 - f3, 0.0)


def turbulence_function(r_t):
    """F_turb = exp(-(R_T/2)^4)."""
    return jnp.exp(-((r_t / 2.0) ** 4))


def ft2(log_intermittency):
    """SA coupling f_t2 = c_t3 (1 - exp(g))."""
    return CT3 * (1.0 - jnp.exp(log_intermittency))


def sources(
    amplification,
    log_intermittency,
    nu,
    nu_t,
    strain,
    vorticity,
    distance,
    local_shape_factor,
    n_crit,
):
    """Return the sources (per unit volume, kinematic) of the n and g equations.

    Parameters
    ----------
    amplification, log_intermittency : arrays
        n and g = ln(gamma).
    nu, nu_t : float, array
        Molecular and eddy kinematic viscosity (nu_t from SA, >= 0).
    strain, vorticity : arrays
        S = sqrt(2 S_ij S_ij) and Omega = sqrt(2 Omega_ij Omega_ij).
    distance : array
        Wall distance d.
    local_shape_factor : array
        H_L = d^2/nu grad(u . grad d) . grad d.
    n_crit : float
        Critical amplification factor N_crit.

    Returns
    -------
    (src_n, src_g)
    """
    h12 = shape_factor(local_shape_factor)
    re_v = strain * distance**2 / (nu + nu_t)
    f_crit = jnp.where(re_v >= critical_reynolds_vorticity(h12), 1.0, 0.0)
    src_n = vorticity * f_crit * growth_factor(h12) * growth_rate(h12)
    r_t = nu_t / nu
    e = jnp.exp(log_intermittency)
    src_g = C1 * strain * onset_function(amplification, n_crit, r_t) * (
        1.0 - e
    ) - C2 * vorticity * turbulence_function(r_t) * (C3 * e - 1.0)
    return src_n, src_g

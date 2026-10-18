"""Spalart-Allmaras one-equation model, negative variant SA-neg (D11).

Formulation of the NASA Turbulence Modeling Resource ("SA-neg"): the
standard model (with f_t2) for nu_tilde >= 0 and the modified production,
destruction and diffusion of Allmaras, Johnson & Spalart (2012) for
nu_tilde < 0, so that negative values decay instead of growing. The
transport equation (kinematic form, incompressible)

    d nu_t/dt + div(v nu_t) = P - D + (1/sigma) [div((nu + nu_t f_n) grad nu_t)
                                                 + c_b2 |grad nu_t|^2]

with nu_t written for nu_tilde. The eddy viscosity is nu_tilde f_v1 for
nu_tilde >= 0 and 0 otherwise.

Transition (L3, first step): the algebraic BCM model of Cakmakcioglu, Bas,
Mura & Kaynak (2020 revision, as implemented in SU2) multiplies the
production term by an intermittency gamma_BC built from the vorticity
Reynolds number and the freestream turbulence intensity; it is used with
the f_t2 term switched off (SA-noft2-BCM).
"""

import jax.numpy as jnp
import numpy as np

CB1 = 0.1355
SIGMA = 2.0 / 3.0
CB2 = 0.622
KAPPA = 0.41
CW2 = 0.3
CW3 = 2.0
CV1 = 7.1
CT3 = 1.2
CT4 = 0.5
CN1 = 16.0
CW1 = CB1 / KAPPA**2 + (1.0 + CB2) / SIGMA
C2, C3 = 0.7, 0.9
R_MAX = 10.0
# BCM transition model (2020 revision)
BCM_CHI1 = 0.002
BCM_CHI2 = 50.0


def fv1(chi):
    """f_v1 = chi^3 / (chi^3 + c_v1^3)."""
    c3 = chi**3
    return c3 / (c3 + CV1**3)


def eddy_viscosity(nu_tilde, nu):
    """Kinematic eddy viscosity nu_t = nu_tilde f_v1 (zero for nu_tilde < 0)."""
    chi = nu_tilde / nu
    return jnp.where(nu_tilde > 0.0, nu_tilde * fv1(chi), 0.0)


def diffusion_coefficient(nu_tilde, nu):
    """(nu + nu_tilde f_n) / sigma with f_n = 1 for nu_tilde >= 0."""
    # f_n only matters for chi < 0; evaluating it at chi >= 0 would hit the
    # pole at chi^3 = c_n1 (NaN gradients through jnp.where).
    chi3 = jnp.minimum(nu_tilde / nu, 0.0) ** 3
    fn = (CN1 + chi3) / (CN1 - chi3)
    return (nu + nu_tilde * jnp.where(nu_tilde >= 0.0, 1.0, fn)) / SIGMA


def bcm_intermittency(nu_tilde, nu, vorticity, distance, turbulence_intensity):
    """Intermittency gamma_BC of the BCM transition model (2020 revision).

    gamma = 1 - exp(-sqrt(T1) - sqrt(T2)) with
    T1 = max(Re_theta - Re_theta_t, 0) / (chi_1 Re_theta_t),
    Re_theta = Re_v / 2.193, Re_v = d^2 |omega| / nu (vorticity Reynolds
    number), Re_theta_t = 803.73 (Tu + 0.6067)^-1.027 (Menter's correlation,
    Tu in percent), and T2 = chi_2 nu_t / nu with nu_t = nu_tilde f_v1.
    T1 switches production on once the laminar boundary layer is thick
    enough for its Tu; T2 keeps it on where eddy viscosity already exists.
    """
    re_v = distance**2 * vorticity / nu
    re_theta = re_v / 2.193
    re_theta_t = 803.73 * (turbulence_intensity + 0.6067) ** (-1.027)
    term1 = _safe_sqrt((re_theta - re_theta_t) / (BCM_CHI1 * re_theta_t))
    term2 = _safe_sqrt(BCM_CHI2 * nu_tilde * fv1(nu_tilde / nu) / nu)
    return 1.0 - jnp.exp(-(term1 + term2))


def _safe_sqrt(x):
    """sqrt(max(x, 0)) with a zero derivative for x <= 0.

    ``jnp.sqrt(jnp.maximum(x, 0))`` has the tangent 0 / 0 = NaN there, which
    would poison the Jacobian-vector products of every laminar cell.
    """
    positive = x > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def source(
    nu_tilde,
    nu,
    vorticity,
    distance,
    grad_sq,
    production_switch=1.0,
    intermittency=1.0,
    ft2=True,
):
    """Source per unit volume: production - destruction + c_b2 term.

    Parameters
    ----------
    nu_tilde, nu : arrays or floats
        Working variable and molecular kinematic viscosity.
    vorticity : array
        |omega| (shear strain magnitude S of the model).
    distance : array
        Wall distance d.
    grad_sq : array
        |grad nu_tilde|^2 (treated as given; the c_b2 term).
    production_switch : array or float
        1 where the model is active, 0 in prescribed laminar regions (D15):
        the production term is switched off there.
    intermittency : array or float
        Transition-model factor on the production term (BCM); 1 = fully
        turbulent.
    ft2 : bool
        Include the f_t2 term (standard SA-neg); False gives SA-noft2, the
        form the BCM model is paired with.
    """
    ct3 = CT3 if ft2 else 0.0
    chi = nu_tilde / nu
    s = vorticity
    kd2 = (KAPPA * distance) ** 2
    # nu_tilde >= 0: standard SA with f_t2.
    f_v2 = 1.0 - chi / (1.0 + chi * fv1(chi))
    s_bar = nu_tilde * f_v2 / kd2
    # Modified S (Allmaras et al. 2012); the unused branch is kept finite so
    # that derivatives through jnp.where stay finite.
    plain = s_bar >= -C2 * s
    denom = jnp.where(plain, 1.0, (C3 - 2.0 * C2) * s - s_bar)
    s_mod = jnp.where(plain, s + s_bar, s + s * (C2**2 * s + C3 * s_bar) / denom)
    # r = nu_tilde / (S kappa^2 d^2), clipped at 10; the floor of the
    # denominator (relative to nu) keeps r and its derivative finite where
    # both nu_tilde and S vanish (1e-300 would underflow in the derivative).
    floor = jnp.abs(nu_tilde) / R_MAX + 1e-12 * nu
    r = jnp.minimum(nu_tilde / jnp.maximum(s_mod * kd2, floor), R_MAX)
    g = r + CW2 * (r**6 - r)
    f_w = g * ((1.0 + CW3**6) / (g**6 + CW3**6)) ** (1.0 / 6.0)
    f_t2 = ct3 * jnp.exp(-CT4 * chi**2)
    production = (
        production_switch * intermittency * CB1 * (1.0 - f_t2) * s_mod * nu_tilde
    )
    destruction = (CW1 * f_w - CB1 / KAPPA**2 * f_t2) * (nu_tilde / distance) ** 2
    positive = production - destruction
    # nu_tilde < 0: modified terms (destruction becomes a sink of negative values).
    negative = CB1 * (1.0 - ct3) * s * nu_tilde + CW1 * (nu_tilde / distance) ** 2
    return jnp.where(nu_tilde >= 0.0, positive, negative) + CB2 / SIGMA * grad_sq


def nu_tilde_from_eddy_viscosity(ratio):
    """Solve chi f_v1(chi) = nu_t / nu for chi = nu_tilde / nu (Newton, D33).

    Solves chi f_v1(chi) = ratio; chi f_v1 is monotonic in chi.
    """
    ratio = np.asarray(ratio, dtype=float)
    chi = np.maximum(ratio, 1.0e-12) ** (1.0 / 4.0) * CV1 ** (3.0 / 4.0)
    chi = np.maximum(chi, ratio)
    for _ in range(50):
        c3 = chi**3
        f = chi * c3 / (c3 + CV1**3) - ratio
        df = c3 * (c3 + 4.0 * CV1**3) / (c3 + CV1**3) ** 2
        step = f / df
        chi = chi - step
        if np.all(np.abs(step) < 1e-14 * np.maximum(chi, 1.0)):
            break
    return chi


def eddy_viscosity_ratio(chi):
    """nu_t / nu = chi f_v1(chi) for chi = nu_tilde / nu."""
    chi = np.asarray(chi, dtype=float)
    return chi**4 / (chi**3 + CV1**3)

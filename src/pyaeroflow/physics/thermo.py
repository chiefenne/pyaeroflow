"""Thermodynamics and transport properties of the incompressible energy equation.

All functions work on non-dimensional temperatures T* = T / T_ref and return
non-dimensional values; :func:`property_law` converts the SI parameters of
the case file. Everything is written in JAX, so the properties and their
temperature derivatives enter the Newton Jacobian exactly.

Property laws (:class:`PropertyLaw`), selectable per property (mu, k, c_p):

    constant      phi = phi_ref
    sutherland    phi = phi_ref T^(3/2) (1 + S) / (T + S)      (S scaled by T_ref)
    power-law     phi = phi_ref T^n
    polynomial    phi = sum_i a_i T^i                           (a_i scaled)
    table         phi linear between (T_j, phi_j), constant outside
    prandtl       k = mu(T) c_p(T) / Pr                         (conductivity only)
    kinetic-viscosity  mu ~ sqrt(T) / Omega_mu(T / eps)    (Chapman-Enskog)
    kinetic-diffusion  D ~ T^1.5 / Omega_D(T / eps_jk)     (Chapman-Enskog)

The enthalpy h(T) = int_{T_ref}^{T} c_p dT is the exact integral of the
c_p law (Gauss-Legendre quadrature for Sutherland's form), so that
dh/dT = c_p and h(T_ref) = 0. The energy equation is written in the
conservative enthalpy form div(rho v h) = div(k grad T): c_p never stands
outside a derivative.

Density models (:class:`DensityModel`) are equations of state rho(p_th, T)
with the same residual code:

    constant      rho = rho_0
    boussinesq    rho = rho_0 (1 - beta (T - T_0))
    ideal_gas     rho = p_th / (R T)       (non-dimensional: rho = c / T)

Species (stage S1, :class:`Species`): the fluid of the case is the bulk
species (the carrier, not transported), ``thermo.species`` are the
transported ones with the mass fractions Y_k; the bulk has 1 - sum Y_k. All
functions take the mass fractions as the optional last argument ``y``
(shape (..., n_species)); without it they are those of the bulk fluid.

    molar mass    1 / W = 1 + sum_k Y_k (1 / W_k - 1)            (W_k / W_bulk)
    ideal_gas     rho = p_th W / T
    boussinesq    rho = rho_0 (1 - beta (T - T_0) - sum_k beta_k Y_k)
    c_p, h        c_p = c_p,bulk + sum_k Y_k (c_p,k - c_p,bulk), h likewise
    mu, k         those of the bulk fluid (mixture rules: stage S2)
    diffusion     J_k = -rho D_k grad Y_k + Y_k V_c  (Fick's law with a
                  correction velocity, so that the fluxes of all species sum
                  to zero; exact for a binary mixture)

D_k(T) is a property law at the operating pressure (rho D_k does not depend
on the thermodynamic pressure) or ``schmidt``: rho D_k = mu / Sc_k.

Multicomponent mixtures (stage S2, ``Thermo.diffusion_model`` etc.):

    diffusion     ``hirschfelder-curtiss``: J_k = -rho D_k,m (dY_k + Y_k dln(W))
                  - mu_t / Sc_t dY_k - Y_k sum_j (...) with the mixture-averaged
                  D_k,m = (1 - X_k) / sum_{j != k} X_j / D_jk
                  (the mole-fraction gradient form of Hirschfelder and Curtiss,
                  a correction velocity makes the fluxes of all species sum to
                  zero; the bulk species is index 0), D_jk binary coefficients
                  (explicit laws or Chapman-Enskog from Lennard-Jones data)
    mu            ``wilke``: sum_i X_i mu_i / sum_j X_j Phi_ij
    k             ``mathur``: (sum_i X_i k_i + 1 / sum_i X_i / k_i) / 2

Turbulent Prandtl number: one interface, :func:`turbulent_prandtl`, a JAX
function of the local state (:class:`TurbulentPrandtlInputs`); built in is
``constant`` (0.9). Further correlations are registered with
:func:`register_turbulent_prandtl`. The same interface will serve the
turbulent Schmidt number of the species equations.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

LAWS = ("constant", "sutherland", "power-law", "polynomial", "table")
CONDUCTIVITY_LAWS = LAWS + ("prandtl",)
DENSITY_MODELS = ("constant", "boussinesq", "ideal_gas")

# Gauss-Legendre nodes and weights on [0, 1] (8 points: exact for
# polynomials up to degree 15) for the enthalpy of Sutherland's form.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


class PropertyLaw(NamedTuple):
    """A temperature law of a property, non-dimensional (hashable).

    Attributes
    ----------
    kind : str
        One of :data:`CONDUCTIVITY_LAWS`.
    value : float
        Value at T = 1 (constant, sutherland, power-law); the Prandtl number
        for ``prandtl``.
    parameters : tuple
        sutherland: (S,); power-law: (n,); polynomial: (a_0, a_1, ...);
        table: ((T_0, T_1, ...), (phi_0, phi_1, ...)).
    """

    kind: str = "constant"
    value: float = 1.0
    parameters: tuple = ()


class DensityModel(NamedTuple):
    """Equation of state rho(p_th, T), non-dimensional (hashable).

    Attributes
    ----------
    kind : {"constant", "boussinesq", "ideal_gas"}
    density : float
        constant, boussinesq: rho_0.
    expansion : float
        boussinesq: beta T_ref (thermal expansion coefficient, scaled).
    temperature : float
        boussinesq: T_0 (scaled).
    pressure : float
        ideal_gas: p_th / (R T_ref rho_ref), so that rho = pressure / T.
    """

    kind: str = "constant"
    density: float = 1.0
    expansion: float = 0.0
    temperature: float = 1.0
    pressure: float = 1.0


class Species(NamedTuple):
    """A transported species of the mixture, non-dimensional (hashable).

    Attributes
    ----------
    name : str
    molar_mass : float
        W_k / W_bulk.
    diffusivity : PropertyLaw
        D_k(T) at the operating pressure, or ``schmidt`` (value Sc_k):
        rho D_k = mu(T) / Sc_k.
    specific_heat : PropertyLaw or None
        c_p,k(T); None: that of the bulk fluid.
    formation_enthalpy : float
        h_k - h_bulk at the reference temperature (zero for inert mixing;
        the hook of the heat release of chemistry).
    expansion : float
        boussinesq: beta_k = -(1 / rho_0) d(rho)/dY_k.
    latent_heat : float
        Evaporating walls: latent heat of the phase change, L / (c_p,ref T_ref).
    saturation : tuple or None
        Evaporating walls: saturation mole fraction X = p_sat(T) / p_op of the
        non-dimensional temperature, ("antoine", a, b, c) with
        log10 X = a - b / (c + T), or ("clausius-clapeyron", ln X_0, theta,
        1 / T_0) with ln X = ln X_0 - theta (1 / T - 1 / T_0).
    """

    name: str = "Y1"
    molar_mass: float = 1.0
    diffusivity: PropertyLaw = PropertyLaw("schmidt", 1.0)
    specific_heat: PropertyLaw = None
    formation_enthalpy: float = 0.0
    expansion: float = 0.0
    latent_heat: float = 0.0
    saturation: tuple = None
    viscosity: PropertyLaw = None
    conductivity: PropertyLaw = None


class Thermo(NamedTuple):
    """Density model and property laws of one fluid (hashable).

    ``species`` are the transported species of a mixture (the fluid is then
    the bulk species); ``pressure_ratio`` is p_th / p_op of a closed domain
    with species (D_k is given at the operating pressure).

    Stage S2: ``diffusion_model`` (``fick`` or ``hirschfelder-curtiss``),
    ``mixture_viscosity`` (``bulk`` or ``wilke``) and ``mixture_conductivity``
    (``bulk`` or ``mathur``); ``pair_diffusivity`` holds the binary
    coefficients D_jk as :class:`PropertyLaw` of the pairs (j < k) of all
    species, the bulk species being index 0, in the order (0,1), (0,2), ...,
    (1,2), ... (rho D is evaluated at the operating pressure).
    """

    density: DensityModel = DensityModel()
    viscosity: PropertyLaw = PropertyLaw("constant", 1.0e-3)
    conductivity: PropertyLaw = PropertyLaw("prandtl", 0.72)
    specific_heat: PropertyLaw = PropertyLaw("constant", 1.0)
    species: tuple = ()
    pressure_ratio: float = 1.0
    diffusion_model: str = "fick"
    mixture_viscosity: str = "bulk"
    mixture_conductivity: str = "bulk"
    pair_diffusivity: tuple = ()


@jax.tree_util.register_static
@dataclass(frozen=True)
class StaticThermo:
    """A :class:`Thermo` carried as static (non-traced) data inside a pytree.

    The variable-density SST keeps it in its boundary data, so that the wall
    value of omega can use nu(T_w) of the wall-face temperature.
    """

    thermo: Thermo


def property_law(kind, value_si=None, parameters=None, *, t_ref=1.0, scale=1.0):
    """Non-dimensional :class:`PropertyLaw` from SI input.

    Parameters
    ----------
    kind : str
    value_si : float, optional
        Property at the reference temperature (constant, sutherland,
        power-law), or the Prandtl number (prandtl).
    parameters : dict, optional
        sutherland: ``temperature`` (S in K, default 110.4); power-law:
        ``exponent``; polynomial: ``coefficients`` [a_0, a_1, ...] of
        phi(T) = sum a_i T^i, T in K; table: ``temperature`` (K, increasing)
        and ``value``.
    t_ref : float
        Reference temperature in K.
    scale : float
        SI value of one non-dimensional unit of the property.
    """
    parameters = dict(parameters or {})
    if kind not in CONDUCTIVITY_LAWS:
        raise ValueError(f"unknown property law '{kind}'")
    if kind == "prandtl":
        return PropertyLaw(kind, float(value_si))
    if kind == "polynomial":
        coeffs = parameters.get("coefficients")
        if not coeffs:
            raise ValueError("polynomial law needs coefficients")
        nd = tuple(float(a) * t_ref**i / scale for i, a in enumerate(coeffs))
        return PropertyLaw(kind, float(sum(nd)), nd)
    if kind == "table":
        t = np.asarray(parameters.get("temperature", ()), dtype=float)
        v = np.asarray(parameters.get("value", ()), dtype=float)
        if t.size < 2 or t.size != v.size or np.any(np.diff(t) <= 0.0):
            raise ValueError(
                "table law needs at least two increasing temperatures and as "
                "many values"
            )
        t_nd, v_nd = tuple(t / t_ref), tuple(v / scale)
        return PropertyLaw(kind, float(np.interp(1.0, t_nd, v_nd)), (t_nd, v_nd))
    if value_si is None:
        raise ValueError(f"{kind} law needs the value at the reference temperature")
    value = float(value_si) / scale
    if kind == "sutherland":
        s = float(parameters.get("temperature", 110.4))
        return PropertyLaw(kind, value, (s / t_ref,))
    if kind == "power-law":
        if "exponent" not in parameters:
            raise ValueError("power-law needs exponent")
        return PropertyLaw(kind, value, (float(parameters["exponent"]),))
    return PropertyLaw(kind, value)


def evaluate(law, t):
    """phi(T) of a law (not ``prandtl``), same shape as ``t``."""
    t = jnp.asarray(t)
    if law.kind == "constant":
        return law.value * jnp.ones_like(t)
    if law.kind == "sutherland":
        (s,) = law.parameters
        tp = jnp.maximum(t, 1e-12)
        return law.value * tp**1.5 * (1.0 + s) / (tp + s)
    if law.kind == "power-law":
        (n,) = law.parameters
        return law.value * jnp.maximum(t, 1e-12) ** n
    if law.kind == "polynomial":
        out = jnp.zeros_like(t)
        for a in reversed(law.parameters):  # Horner
            out = out * t + a
        return out
    if law.kind == "table":
        ts, vs = law.parameters
        return jnp.interp(t, jnp.asarray(ts), jnp.asarray(vs))
    if law.kind in KINETIC_LAWS:
        # Chapman-Enskog: value at T = 1, parameter eps / (k T_ref).
        (eps,) = law.parameters
        tp = jnp.maximum(t, 1e-12)
        if law.kind == "kinetic-viscosity":
            return (
                law.value
                * jnp.sqrt(tp)
                * collision_integral_viscosity(1.0 / eps)
                / collision_integral_viscosity(tp / eps)
            )
        return (
            law.value
            * tp**1.5
            * collision_integral_diffusion(1.0 / eps)
            / collision_integral_diffusion(tp / eps)
        )
    raise ValueError(f"law '{law.kind}' has no direct evaluation")


KINETIC_LAWS = ("kinetic-viscosity", "kinetic-diffusion")


def collision_integral_diffusion(t_star):
    """Omega_D(T*) of Neufeld, Janzen and Aziz (1972), Lennard-Jones 12-6."""
    return (
        1.06036 / t_star**0.15610
        + 0.19300 * jnp.exp(-0.47635 * t_star)
        + 1.03587 * jnp.exp(-1.52996 * t_star)
        + 1.76474 * jnp.exp(-3.89411 * t_star)
    )


def collision_integral_viscosity(t_star):
    """Omega_mu(T*) = Omega_(2,2)* of Neufeld, Janzen and Aziz (1972)."""
    return (
        1.16145 / t_star**0.14874
        + 0.52487 * jnp.exp(-0.77320 * t_star)
        + 2.16178 * jnp.exp(-2.43787 * t_star)
    )


def kinetic_viscosity(molar_mass, sigma, eps_over_k, t):
    """Chapman-Enskog viscosity in Pa s: 2.6693e-6 sqrt(M T) / (sigma^2 Omega).

    ``molar_mass`` in kg/kmol, ``sigma`` in Angstrom, ``eps_over_k`` and
    ``t`` in K (numpy or float).
    """
    t_star = t / eps_over_k
    omega = float(collision_integral_viscosity(t_star))
    return 2.6693e-6 * np.sqrt(molar_mass * t) / (sigma**2 * omega)


def kinetic_diffusivity(m1, m2, sigma, eps_over_k, t, pressure):
    """Chapman-Enskog binary diffusion coefficient D_12 in m^2/s.

    D = 0.00266 T^1.5 / (p sqrt(M_12) sigma_12^2 Omega_D) in cm^2/s with p in
    bar and M_12 = 2 / (1 / M_1 + 1 / M_2) (Poling, Prausnitz, O'Connell);
    ``sigma`` is sigma_12 in Angstrom, ``eps_over_k`` is eps_12 / k in K, ``t``
    in K, ``pressure`` in Pa.
    """
    m12 = 2.0 / (1.0 / m1 + 1.0 / m2)
    omega = float(collision_integral_diffusion(t / eps_over_k))
    d_cm2 = 0.00266 * t**1.5 / (pressure / 1.0e5 * np.sqrt(m12) * sigma**2 * omega)
    return d_cm2 * 1.0e-4


def integral(law, t):
    """int_1^T phi dT of a law (the enthalpy of a c_p law), shape of ``t``."""
    t = jnp.asarray(t)
    if law.kind == "constant":
        return law.value * (t - 1.0)
    if law.kind == "power-law":
        (n,) = law.parameters
        if abs(n + 1.0) < 1e-12:
            return law.value * jnp.log(jnp.maximum(t, 1e-12))
        return law.value * (jnp.maximum(t, 1e-12) ** (n + 1.0) - 1.0) / (n + 1.0)
    if law.kind == "polynomial":
        out = jnp.zeros_like(t)
        for i, a in enumerate(law.parameters):
            out = out + a * (t ** (i + 1) - 1.0) / (i + 1)
        return out
    if law.kind == "table":
        return _table_integral(law, t) - _table_integral(law, jnp.ones(()))
    if law.kind == "sutherland":
        x = jnp.asarray(_GL_X)
        w = jnp.asarray(_GL_W)
        nodes = 1.0 + (t[..., None] - 1.0) * x
        return (t - 1.0) * jnp.sum(w * evaluate(law, nodes), axis=-1)
    raise ValueError(f"law '{law.kind}' has no integral")


def _table_integral(law, t):
    """int_{T_0}^T of the piecewise-linear table, constant extrapolation."""
    ts = np.asarray(law.parameters[0])
    vs = np.asarray(law.parameters[1])
    seg = np.diff(ts)
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * seg * (vs[1:] + vs[:-1]))))
    slope = np.diff(vs) / seg
    tj, vj, sj, cj = map(jnp.asarray, (ts, vs, slope, cumulative))
    tc = jnp.clip(t, ts[0], ts[-1])
    i = jnp.clip(jnp.searchsorted(tj, tc, side="right") - 1, 0, ts.size - 2)
    dx = tc - tj[i]
    inside = cj[i] + vj[i] * dx + 0.5 * sj[i] * dx**2
    below = jnp.minimum(t - ts[0], 0.0) * vs[0]
    above = jnp.maximum(t - ts[-1], 0.0) * vs[-1]
    return inside + below + above


# -- fluid properties ---------------------------------------------------------------


def _species_array(thermo, attribute):
    return jnp.asarray([getattr(s, attribute) for s in thermo.species])


def inverse_molar_mass(thermo, y):
    """W_bulk / W of the mixture: 1 + sum_k Y_k (1 / W_k - 1), shape y[..., 0]."""
    c = jnp.asarray([1.0 / s.molar_mass - 1.0 for s in thermo.species])
    return jnp.maximum(1.0 + jnp.sum(jnp.asarray(y) * c, axis=-1), 1e-12)


def density(thermo, t, y=None):
    """rho(T, Y) of the density model (constant thermodynamic pressure)."""
    d = thermo.density
    t = jnp.asarray(t)
    if d.kind == "constant":
        return d.density * jnp.ones_like(t)
    if d.kind == "boussinesq":
        rho = d.density * (1.0 - d.expansion * (t - d.temperature))
        if y is not None:
            beta = _species_array(thermo, "expansion")
            rho = rho - d.density * jnp.sum(jnp.asarray(y) * beta, axis=-1)
        return rho
    if d.kind == "ideal_gas":
        rho = d.pressure / jnp.maximum(t, 1e-12)
        return rho if y is None else rho / inverse_molar_mass(thermo, y)
    raise ValueError(f"unknown density model '{d.kind}'")


def density_derivative(thermo, t, y=None):
    """d(rho)/dT at constant thermodynamic pressure and composition."""
    d = thermo.density
    t = jnp.asarray(t)
    if d.kind == "constant":
        return jnp.zeros_like(t)
    if d.kind == "boussinesq":
        return -d.density * d.expansion * jnp.ones_like(t)
    if d.kind == "ideal_gas":
        rho_t = -d.pressure / jnp.maximum(t, 1e-12) ** 2
        return rho_t if y is None else rho_t / inverse_molar_mass(thermo, y)
    raise ValueError(f"unknown density model '{d.kind}'")


def density_species_derivative(thermo, t, y):
    """d(rho)/dY_k at constant T and p_th, shape y.shape."""
    d = thermo.density
    y = jnp.asarray(y)
    if d.kind == "constant":
        return jnp.zeros_like(y)
    if d.kind == "boussinesq":
        return -d.density * _species_array(thermo, "expansion") * jnp.ones_like(y)
    if d.kind == "ideal_gas":
        c = jnp.asarray([1.0 / s.molar_mass - 1.0 for s in thermo.species])
        rho = density(thermo, t, y)
        return -(rho / inverse_molar_mass(thermo, y))[..., None] * c
    raise ValueError(f"unknown density model '{d.kind}'")


def mole_fractions(thermo, y, bounded=False):
    """Mole fractions X of all species, (..., n + 1); index 0 is the bulk.

    ``bounded``: mass fractions limited to >= 0 (the mixture rules and
    D_k,m of the transient overshoots of Y, which is not clipped itself).
    """
    y = jnp.asarray(y)
    yy = jnp.concatenate((1.0 - jnp.sum(y, axis=-1, keepdims=True), y), axis=-1)
    if bounded:
        yy = jnp.maximum(yy, 0.0)
    inv_w = jnp.asarray([1.0] + [1.0 / s.molar_mass for s in thermo.species])
    x = yy * inv_w
    return x / jnp.maximum(jnp.sum(x, axis=-1, keepdims=True), 1e-30)


def _molar_masses(thermo):
    return jnp.asarray([1.0] + [s.molar_mass for s in thermo.species])


def _pair_matrix(thermo, t):
    """D_jk(T) of all pairs, shape t.shape + (n + 1, n + 1), the diagonal 1."""
    n = len(thermo.species) + 1
    t = jnp.asarray(t)
    if len(thermo.pair_diffusivity) != n * (n - 1) // 2:
        raise ValueError(
            "diffusion_model hirschfelder-curtiss needs the binary diffusion "
            f"coefficients of all {n * (n - 1) // 2} pairs (Thermo.pair_diffusivity)"
        )
    one = jnp.ones_like(t)
    rows = [[one] * n for _ in range(n)]
    pair = iter(thermo.pair_diffusivity)
    for j in range(n):
        for k in range(j + 1, n):
            rows[j][k] = rows[k][j] = evaluate(next(pair), t)
    return jnp.stack([jnp.stack(r, axis=-1) for r in rows], axis=-2)


def mixture_diffusivity(thermo, t, y):
    """D_k,m of all species (bulk first), shape t.shape + (n + 1,), at p_op.

    D_k,m = sum_{j != k} X_j / sum_{j != k} X_j / D_jk = (1 - X_k) / sum_{j != k}
    X_j / D_jk (Hirschfelder and Curtiss; Bird, Stewart, Lightfoot). For two
    species it is the binary coefficient, for equal D_jk that coefficient. X_j
    gets a floor of 1e-12 so that a pure species has a finite limit.
    """
    t = jnp.asarray(t)
    x = mole_fractions(thermo, y, bounded=True) + 1e-12
    n = x.shape[-1]
    d = _pair_matrix(thermo, t)
    off = 1.0 - jnp.eye(n)
    den = jnp.einsum("...j,...kj->...k", x, off / d)
    num = jnp.sum(x, axis=-1, keepdims=True) - x
    return num / den


def _species_viscosities(thermo, t):
    """mu_i(T) of all species (bulk first), shape t.shape + (n + 1,)."""
    laws = [thermo.viscosity] + [s.viscosity for s in thermo.species]
    if any(law is None for law in laws):
        raise ValueError("mixture viscosity wilke needs a viscosity of every species")
    return jnp.stack([evaluate(law, t) for law in laws], axis=-1)


def _wilke_matrix(mu, w):
    """Phi_ij of Wilke's mixing rule for viscosities mu (..., N) and molar masses w."""
    ratio = jnp.sqrt(mu[..., :, None] / mu[..., None, :])
    wr = w[None, :] / w[:, None]  # W_j / W_i
    return (1.0 + ratio * wr**0.25) ** 2 / jnp.sqrt(8.0 * (1.0 + 1.0 / wr))


def viscosity(thermo, t, y=None):
    """Dynamic viscosity mu(T) (``wilke``: of the mixture of composition ``y``)."""
    if y is None or thermo.mixture_viscosity == "bulk" or not thermo.species:
        return evaluate(thermo.viscosity, t)
    t = jnp.asarray(t)
    mu = _species_viscosities(thermo, t)
    x = mole_fractions(thermo, y, bounded=True)
    phi = _wilke_matrix(mu, _molar_masses(thermo))
    return jnp.sum(x * mu / jnp.einsum("...j,...ij->...i", x, phi), axis=-1)


def specific_heat(thermo, t, y=None):
    """c_p(T, Y) (without ``y``: of the bulk fluid)."""
    cp = evaluate(thermo.specific_heat, t)
    if y is None:
        return cp
    return cp + jnp.sum(jnp.asarray(y) * species_specific_heat(thermo, t), axis=-1)


def enthalpy(thermo, t, y=None):
    """h(T, Y) = int_1^T c_p dT (bulk fluid: zero at the reference temperature)."""
    h = integral(thermo.specific_heat, t)
    if y is None:
        return h
    return h + jnp.sum(jnp.asarray(y) * species_enthalpy(thermo, t), axis=-1)


def species_specific_heat(thermo, t):
    """c_p,k(T) - c_p,bulk(T) of the transported species, shape t.shape + (n,)."""
    t = jnp.asarray(t)
    cp = evaluate(thermo.specific_heat, t)
    return jnp.stack(
        [
            jnp.zeros_like(t)
            if s.specific_heat is None
            else evaluate(s.specific_heat, t) - cp
            for s in thermo.species
        ],
        axis=-1,
    )


def species_enthalpy(thermo, t):
    """h_k(T) - h_bulk(T) of the transported species, shape t.shape + (n,).

    The partial enthalpy dh/dY_k of the mixture: zero for species with the
    c_p of the bulk fluid and no formation enthalpy.
    """
    t = jnp.asarray(t)
    h = integral(thermo.specific_heat, t)
    return jnp.stack(
        [
            s.formation_enthalpy
            + (
                jnp.zeros_like(t)
                if s.specific_heat is None
                else integral(s.specific_heat, t) - h
            )
            for s in thermo.species
        ],
        axis=-1,
    )


def density_diffusivity(thermo, t, y):
    """Rho D_k of the transported species, shape y.shape.

    ``schmidt``: mu(T) / Sc_k. Other laws: D_k(T) at the operating pressure
    times the density at the operating pressure (rho D_k of a gas does not
    depend on the pressure).
    """
    t = jnp.asarray(t)
    rho = density(thermo, t, y) / thermo.pressure_ratio
    if thermo.diffusion_model == "hirschfelder-curtiss":
        return rho[..., None] * mixture_diffusivity(thermo, t, y)[..., 1:]
    mu = viscosity(thermo, t, y)
    return jnp.stack(
        [
            mu / s.diffusivity.value
            if s.diffusivity.kind == "schmidt"
            else rho * evaluate(s.diffusivity, t)
            for s in thermo.species
        ],
        axis=-1,
    )


def saturation_mole_fraction(species, t, pressure_ratio=1.0):
    """X = p_sat(T) / p_th of an evaporating species at temperatures ``t``.

    ``pressure_ratio`` is p_th / p_op (closed domains).
    """
    t = jnp.maximum(jnp.asarray(t), 1e-12)
    kind, *par = species.saturation
    if kind == "antoine":
        a, b, c = par
        x = 10.0 ** (a - b / (c + t))
    elif kind == "clausius-clapeyron":
        ln_x0, theta, inv_t0 = par
        x = jnp.exp(ln_x0 - theta * (1.0 / t - inv_t0))
    else:
        raise ValueError(f"unknown saturation law '{kind}'")
    return x / pressure_ratio


def saturation_mass_fraction(thermo, t):
    """Saturation mass fraction of the (single) transported species at ``t``.

    Y = X W / (X W + 1 - X) with the molar mass ratio W = W_k / W_bulk; the
    saturation mole fraction is limited to 0.99.
    """
    sp = thermo.species[0]
    x = jnp.minimum(saturation_mole_fraction(sp, t, thermo.pressure_ratio), 0.99)
    return x * sp.molar_mass / (x * sp.molar_mass + 1.0 - x)


def turbulent_schmidt_coefficients(thermo, sct_law, t, y, mu_t, wall_distance=None):
    """mu_t / Sc_t,k of the transported species at the faces, shape y.shape.

    The Sc_t law has the interface of the turbulent Prandtl number laws
    (:func:`register_turbulent_schmidt`): the inputs' ``prandtl`` field holds
    the molecular Schmidt number mu / (rho D_k) of the species.
    """
    rho = density(thermo, t, y)
    mu = viscosity(thermo, t, y)
    rho_d = density_diffusivity(thermo, t, y)
    nu = mu / rho
    d = jnp.zeros_like(t) if wall_distance is None else wall_distance
    out = []
    for k in range(len(thermo.species)):
        inputs = TurbulentPrandtlInputs(t, nu, mu_t / rho, d, mu / rho_d[..., k])
        out.append(mu_t / turbulent_schmidt(sct_law, inputs))
    return jnp.stack(out, axis=-1)


def diffusion_flux(thermo, t, y, grad_n, turbulent=None):
    """Diffusion mass fluxes J_k.n of the transported species, shape y.shape.

    Fick's law with a correction velocity: J_k = -rho D_k dY_k/dn + Y_k S
    with S = sum over all species (the bulk included) of rho D_j dY_j/dn, so
    that the fluxes of all species sum to zero. The bulk species has
    dY/dn = -sum_k dY_k/dn and the mass-averaged diffusivity of the
    transported species (with one transported species its D: S = 0 and the
    flux is Fick's law of a binary mixture).

    ``grad_n`` is dY_k/dn, shape y.shape; ``turbulent`` (shape y.shape) is
    added to rho D_k (RANS: mu_t / Sc_t,k).
    """
    y = jnp.asarray(y)
    if thermo.diffusion_model == "hirschfelder-curtiss":
        return _hirschfelder_curtiss_flux(thermo, t, y, grad_n, turbulent)
    rho_d = density_diffusivity(thermo, t, y)
    if turbulent is not None:
        rho_d = rho_d + turbulent
    fick = rho_d * grad_n
    if len(thermo.species) == 1:
        return -fick
    y_sum = jnp.sum(jnp.abs(y), axis=-1) + 1e-30
    bulk = (jnp.sum(jnp.abs(y) * rho_d, axis=-1) + 1e-30 * rho_d[..., 0]) / y_sum
    correction = jnp.sum(fick, axis=-1) - bulk * jnp.sum(grad_n, axis=-1)
    return -fick + y * correction[..., None]


def _hirschfelder_curtiss_flux(thermo, t, y, grad_n, turbulent):
    """J_k.n with the mole-fraction gradient form and a correction velocity.

    j_k = -rho D_k,m (dY_k + Y_k dln W) - (mu_t / Sc_t,k) dY_k for all species
    (the bulk: dY_0 = -sum dY_k, D_0,m, and the Y-weighted mu_t / Sc_t), the
    flux J_k = j_k - Y_k sum_j j_j, so that the fluxes of all species sum to
    zero. dln W = -d(1 / W) W, 1 / W = sum_j Y_j / W_j: the gradient of the
    mole fractions is the gradient of the mass fractions plus the molar-mass
    term. For two species this is Fick's law with D_12 exactly.
    """
    rho = density(thermo, t, y) / thermo.pressure_ratio
    d_m = mixture_diffusivity(thermo, t, y)
    rho_d = rho[..., None] * d_m  # (..., n + 1), bulk first
    c = jnp.asarray([1.0 / s.molar_mass - 1.0 for s in thermo.species])
    inv_w = 1.0 + jnp.sum(y * c, axis=-1)
    dln_w = -jnp.sum(grad_n * c, axis=-1) / jnp.maximum(inv_w, 1e-12)
    y0 = 1.0 - jnp.sum(y, axis=-1)
    g0 = -jnp.sum(grad_n, axis=-1)
    yy = jnp.concatenate((y0[..., None], y), axis=-1)
    gg = jnp.concatenate((g0[..., None], grad_n), axis=-1)
    j = -rho_d * (gg + yy * dln_w[..., None])
    if turbulent is not None:
        ay = jnp.abs(y)
        t0 = jnp.sum(ay * turbulent, axis=-1) / (jnp.sum(ay, axis=-1) + 1e-30)
        tt = jnp.concatenate((t0[..., None], turbulent), axis=-1)
        j = j - tt * gg
    return j[..., 1:] - y * jnp.sum(j, axis=-1, keepdims=True)


def conductivity(thermo, t, y=None):
    """Thermal conductivity k(T); ``prandtl``: mu(T) c_p(T) / Pr.

    ``mathur`` with ``y``: (sum_i X_i k_i + 1 / sum_i X_i / k_i) / 2 of the
    species conductivities (``Species.conductivity``, default mu_i c_p,i / Pr
    of the fluid's Prandtl law).
    """
    law = thermo.conductivity
    if y is None or thermo.mixture_conductivity == "bulk" or not thermo.species:
        if law.kind == "prandtl":
            return viscosity(thermo, t) * specific_heat(thermo, t) / law.value
        return evaluate(law, t)
    t = jnp.asarray(t)
    cp = specific_heat(thermo, t)
    cps = [cp] + [cp + d for d in jnp.moveaxis(species_specific_heat(thermo, t), -1, 0)]
    ks = [conductivity(thermo, t)]
    for sp, c in zip(thermo.species, cps[1:], strict=True):
        if sp.conductivity is not None:
            ks.append(evaluate(sp.conductivity, t))
        elif law.kind == "prandtl" and sp.viscosity is not None:
            ks.append(evaluate(sp.viscosity, t) * c / law.value)
        else:
            raise ValueError(
                f"mixture conductivity mathur needs a conductivity of species {sp.name}"
            )
    k = jnp.stack(ks, axis=-1)
    x = mole_fractions(thermo, y, bounded=True)
    return 0.5 * (jnp.sum(x * k, axis=-1) + 1.0 / jnp.sum(x / k, axis=-1))


# -- turbulent Prandtl number --------------------------------------------------------


class TurbulentPrandtlInputs(NamedTuple):
    """Local state at the faces passed to a turbulent Prandtl number law.

    All arrays have the shape (n_faces,). The turbulent Peclet number is
    Pe_t = (nu_t / nu) Pr.
    """

    temperature: jnp.ndarray
    nu: jnp.ndarray
    nu_t: jnp.ndarray
    wall_distance: jnp.ndarray
    prandtl: jnp.ndarray

    @property
    def peclet(self):
        """Turbulent Peclet number (nu_t / nu) Pr."""
        return self.nu_t / self.nu * self.prandtl


class TurbulentPrandtl(NamedTuple):
    """Selected turbulent Prandtl number law (hashable).

    ``parameters`` is a tuple of (name, value) pairs passed to the law.
    """

    kind: str = "constant"
    parameters: tuple = (("value", 0.9),)


def _constant_prandtl(inputs, value=0.9):
    return value * jnp.ones_like(inputs.temperature)


TURBULENT_PRANDTL_LAWS = {"constant": _constant_prandtl}


def register_turbulent_prandtl(name, function):
    """Add a turbulent Prandtl number law.

    ``function(inputs, **parameters) -> Pr_t`` takes
    :class:`TurbulentPrandtlInputs` and returns an array of the faces; it
    must be written in JAX (it is differentiated for the Newton Jacobian).
    """
    TURBULENT_PRANDTL_LAWS[name] = function


def turbulent_prandtl(law, inputs):
    """Pr_t at the faces for the selected law."""
    try:
        function = TURBULENT_PRANDTL_LAWS[law.kind]
    except KeyError:
        raise ValueError(
            f"unknown turbulent Prandtl number law '{law.kind}' (known: "
            f"{sorted(TURBULENT_PRANDTL_LAWS)})"
        ) from None
    return function(inputs, **dict(law.parameters))


TURBULENT_SCHMIDT_LAWS = {"constant": _constant_prandtl}


def register_turbulent_schmidt(name, function):
    """Add a turbulent Schmidt number law.

    ``function(inputs, **parameters) -> Sc_t`` takes
    :class:`TurbulentPrandtlInputs` (``prandtl`` holds the molecular Schmidt
    number of the species) and returns an array of the faces; written in JAX.
    """
    TURBULENT_SCHMIDT_LAWS[name] = function


def turbulent_schmidt(law, inputs):
    """Sc_t at the faces for the selected law."""
    try:
        function = TURBULENT_SCHMIDT_LAWS[law.kind]
    except KeyError:
        raise ValueError(
            f"unknown turbulent Schmidt number law '{law.kind}' (known: "
            f"{sorted(TURBULENT_SCHMIDT_LAWS)})"
        ) from None
    return function(inputs, **dict(law.parameters))


def effective_conductivity(
    thermo, prt_law, t, rho, mu_t=None, wall_distance=None, y=None
):
    """Effective conductivity k + c_p mu_t / Pr_t at the faces (k without mu_t)."""
    k = conductivity(thermo, t, y)
    if mu_t is None:
        return k
    mu = viscosity(thermo, t, y)
    # (the bulk-fluid c_p stays in the S1 mixture rules; mixture c_p with mathur)
    cp = specific_heat(thermo, t, None if thermo.mixture_conductivity == "bulk" else y)
    nu = mu / rho
    inputs = TurbulentPrandtlInputs(
        t,
        nu,
        mu_t / rho,
        jnp.zeros_like(t) if wall_distance is None else wall_distance,
        mu * cp / k,
    )
    return k + cp * mu_t / turbulent_prandtl(prt_law, inputs)

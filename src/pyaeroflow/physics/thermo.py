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

Turbulent Prandtl number: one interface, :func:`turbulent_prandtl`, a JAX
function of the local state (:class:`TurbulentPrandtlInputs`); built in is
``constant`` (0.9). Further correlations are registered with
:func:`register_turbulent_prandtl`. The same interface will serve the
turbulent Schmidt number of the species equations.
"""

from typing import NamedTuple

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


class Thermo(NamedTuple):
    """Density model and property laws of one fluid (hashable)."""

    density: DensityModel = DensityModel()
    viscosity: PropertyLaw = PropertyLaw("constant", 1.0e-3)
    conductivity: PropertyLaw = PropertyLaw("prandtl", 0.72)
    specific_heat: PropertyLaw = PropertyLaw("constant", 1.0)


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
    raise ValueError(f"law '{law.kind}' has no direct evaluation")


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


def density(thermo, t):
    """rho(T) of the density model (constant thermodynamic pressure)."""
    d = thermo.density
    t = jnp.asarray(t)
    if d.kind == "constant":
        return d.density * jnp.ones_like(t)
    if d.kind == "boussinesq":
        return d.density * (1.0 - d.expansion * (t - d.temperature))
    if d.kind == "ideal_gas":
        return d.pressure / jnp.maximum(t, 1e-12)
    raise ValueError(f"unknown density model '{d.kind}'")


def density_derivative(thermo, t):
    """d(rho)/dT at constant thermodynamic pressure."""
    d = thermo.density
    t = jnp.asarray(t)
    if d.kind == "constant":
        return jnp.zeros_like(t)
    if d.kind == "boussinesq":
        return -d.density * d.expansion * jnp.ones_like(t)
    if d.kind == "ideal_gas":
        return -d.pressure / jnp.maximum(t, 1e-12) ** 2
    raise ValueError(f"unknown density model '{d.kind}'")


def viscosity(thermo, t):
    """Dynamic viscosity mu(T)."""
    return evaluate(thermo.viscosity, t)


def specific_heat(thermo, t):
    """c_p(T)."""
    return evaluate(thermo.specific_heat, t)


def enthalpy(thermo, t):
    """h(T) = int_1^T c_p dT (zero at the reference temperature)."""
    return integral(thermo.specific_heat, t)


def conductivity(thermo, t):
    """Thermal conductivity k(T); ``prandtl``: mu(T) c_p(T) / Pr."""
    law = thermo.conductivity
    if law.kind == "prandtl":
        return viscosity(thermo, t) * specific_heat(thermo, t) / law.value
    return evaluate(law, t)


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


def effective_conductivity(thermo, prt_law, t, rho, mu_t=None, wall_distance=None):
    """Effective conductivity k + c_p mu_t / Pr_t at the faces (k without mu_t)."""
    k = conductivity(thermo, t)
    if mu_t is None:
        return k
    mu = viscosity(thermo, t)
    cp = specific_heat(thermo, t)
    nu = mu / rho
    inputs = TurbulentPrandtlInputs(
        t,
        nu,
        mu_t / rho,
        jnp.zeros_like(t) if wall_distance is None else wall_distance,
        mu * cp / k,
    )
    return k + cp * mu_t / turbulent_prandtl(prt_law, inputs)

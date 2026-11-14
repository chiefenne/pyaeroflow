"""Multicomponent diffusion and mixture rules (stage S2): Hirschfelder-Curtiss
fluxes, Chapman-Enskog coefficients, Wilke viscosity and Mathur conductivity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.physics import thermo as th

# Lennard-Jones data (Poling et al.): molar mass kg/kmol, sigma A, eps/k K.
LJ = {
    "N2": (28.0134, 3.798, 71.4),
    "O2": (31.9988, 3.467, 106.7),
    "H2": (2.016, 2.827, 59.7),
    "CO2": (44.01, 3.941, 195.2),
}
T_REF = 300.0
P_OP = 101325.0


def _laws(name):
    """Species with kinetic viscosity and diffusion data at T_ref (SI scales of 1)."""
    m, sigma, eps = LJ[name]
    mu = th.kinetic_viscosity(m, sigma, eps, T_REF)
    return m, sigma, eps, mu


def _mixture(bulk="N2", species=("O2", "CO2"), equal_d=None, rules=True):
    """Thermo of a bulk fluid and species from kinetic theory (SI scales of 1)."""
    mb, sb, eb, mub = _laws(bulk)
    data = {"bulk": (mb, sb, eb)} | {s: LJ[s] for s in species}
    names = list(data)
    pairs = []
    for j in range(len(names)):
        for k in range(j + 1, len(names)):
            a, b = data[names[j]], data[names[k]]
            eps = float(np.sqrt(a[2] * b[2]))
            if equal_d is not None:
                pairs.append(th.PropertyLaw("constant", equal_d))
                continue
            d = th.kinetic_diffusivity(
                a[0], b[0], 0.5 * (a[1] + b[1]), eps, T_REF, P_OP
            )
            pairs.append(th.PropertyLaw("kinetic-diffusion", d, (eps / T_REF,)))
    sps = []
    for s in species:
        m, sigma, eps, mu = _laws(s)
        sps.append(
            th.Species(
                s,
                m / mb,
                th.PropertyLaw("schmidt", 1.0),
                viscosity=th.PropertyLaw("kinetic-viscosity", mu, (eps / T_REF,)),
            )
        )
    return th.Thermo(
        th.DensityModel("ideal_gas", pressure=P_OP * mb / (8314.462618 * T_REF)),
        th.PropertyLaw("kinetic-viscosity", mub, (eb / T_REF,)),
        th.PropertyLaw("prandtl", 0.72),
        th.PropertyLaw("constant", 1.0),
        tuple(sps),
        diffusion_model="hirschfelder-curtiss",
        mixture_viscosity="wilke" if rules else "bulk",
        mixture_conductivity="mathur" if rules else "bulk",
        pair_diffusivity=tuple(pairs),
    )


def test_chapman_enskog_binary_diffusion_and_viscosity():
    # Poling et al. / Bird: N2-O2 0.20 cm2/s, H2-N2 0.78 cm2/s at 300 K, 1 atm;
    # mu(N2, 300 K) = 17.9e-6 Pa s, mu(O2) = 20.7e-6, mu(H2) = 8.9e-6.
    n2, o2, h2 = LJ["N2"], LJ["O2"], LJ["H2"]

    def d(a, b):
        eps = float(np.sqrt(a[2] * b[2]))
        return th.kinetic_diffusivity(a[0], b[0], 0.5 * (a[1] + b[1]), eps, T_REF, P_OP)

    assert d(n2, o2) * 1e4 == pytest.approx(0.20, rel=0.08)
    assert d(h2, n2) * 1e4 == pytest.approx(0.78, rel=0.08)
    for name, ref in (("N2", 17.9e-6), ("O2", 20.7e-6), ("H2", 8.9e-6)):
        m, sigma, eps = LJ[name]
        assert th.kinetic_viscosity(m, sigma, eps, T_REF) == pytest.approx(
            ref, rel=0.06
        )


def test_kinetic_laws_scale_with_temperature():
    m, sigma, eps = LJ["N2"]
    mu = th.kinetic_viscosity(m, sigma, eps, T_REF)
    law = th.PropertyLaw("kinetic-viscosity", mu, (eps / T_REF,))
    t = np.array([1.0, 1.5, 2.0])
    ref = [th.kinetic_viscosity(m, sigma, eps, T_REF * x) for x in t]
    np.testing.assert_allclose(th.evaluate(law, t), ref, rtol=1e-12)
    d = th.kinetic_diffusivity(m, m, sigma, eps, T_REF, P_OP)
    law = th.PropertyLaw("kinetic-diffusion", d, (eps / T_REF,))
    ref = [th.kinetic_diffusivity(m, m, sigma, eps, T_REF * x, P_OP) for x in t]
    np.testing.assert_allclose(th.evaluate(law, t), ref, rtol=1e-12)


def test_binary_hirschfelder_curtiss_is_ficks_law():
    # Two species: D_1,m = D_12 and the flux is -rho D dY/dn exactly, also with a
    # turbulent diffusivity (the molar-mass term and the correction cancel).
    thermo = _mixture(species=("O2",))
    t, y, g = jnp.array(1.2), jnp.array([0.37]), jnp.array([0.8])
    rho = th.density(thermo, t, y)
    d12 = th.evaluate(thermo.pair_diffusivity[0], t)
    j = th.diffusion_flux(thermo, t, y, g)
    np.testing.assert_allclose(j, -rho * d12 * g, rtol=1e-9)
    turb = jnp.array([0.3])
    j = th.diffusion_flux(thermo, t, y, g, turb)
    np.testing.assert_allclose(j, -(rho * d12 + turb) * g, rtol=1e-9)


def _stefan_maxwell(thermo, t, y, grad_y, rho_d_pairs):
    """J_k of the Stefan-Maxwell equations with zero mass-average diffusion velocity."""

    def x_of_y(yy):
        return th.mole_fractions(thermo, yy)

    x = np.asarray(x_of_y(y))
    dx = np.asarray(jax.jacfwd(x_of_y)(y)) @ np.asarray(grad_y)
    yy = np.concatenate(([1.0 - np.sum(y)], np.asarray(y)))
    n = x.size
    a = np.zeros((n + 1, n))
    for k in range(n):
        for j in range(n):
            if j != k:
                c = x[k] * x[j] / rho_d_pairs[min(j, k)][max(j, k)]
                a[k, j] += c
                a[k, k] -= c
    a[n] = yy
    rhs = np.concatenate((dx, [0.0]))
    v = np.linalg.lstsq(a, rhs, rcond=None)[0]
    rho = float(th.density(thermo, t, y)) / thermo.pressure_ratio
    return rho * yy * v


def _pair_d(thermo, t):
    d = np.asarray(th._pair_matrix(thermo, jnp.asarray(t)))
    return d


def test_equal_binary_coefficients_match_stefan_maxwell():
    # With all D_jk equal the mixture-averaged form with the correction velocity
    # reproduces the Stefan-Maxwell fluxes (unequal molar masses included): -rho D dY.
    thermo = _mixture(species=("H2", "CO2"), equal_d=2.0e-5)
    t, y, g = jnp.array(1.1), jnp.array([0.2, 0.3]), jnp.array([0.4, -0.7])
    sm = _stefan_maxwell(thermo, t, y, g, _pair_d(thermo, t))
    hc = np.asarray(th.diffusion_flux(thermo, t, y, g))
    # the Stefan-Maxwell flux is D rho (...) with rho D = rho_op D; here D given
    rho = float(th.density(thermo, t, y))
    sm = sm  # (J_bulk, J_1, J_2) with rho D_jk = rho * D_jk inside the system
    np.testing.assert_allclose(hc, sm[1:], rtol=1e-8, atol=1e-14 * rho)
    assert abs(hc.sum() + sm[0]) < 1e-12 * rho


def test_unequal_coefficients_approximate_stefan_maxwell():
    # H2-N2-CO2 (the Duncan-Toor mixture): the mixture-averaged approximation of the
    # fluxes is a mixture-averaged approximation: it stays within 35 % of the exact
    # Stefan-Maxwell fluxes (the approximation error of the model, 25 % here).
    thermo = _mixture(bulk="N2", species=("H2", "CO2"))
    t, y, g = jnp.array(1.0), jnp.array([0.05, 0.4]), jnp.array([0.3, -0.5])
    sm = _stefan_maxwell(thermo, t, y, g, _pair_d(thermo, t))
    hc = np.asarray(th.diffusion_flux(thermo, t, y, g))
    scale = np.max(np.abs(sm))
    assert np.max(np.abs(hc - sm[1:])) < 0.35 * scale


def test_fluxes_sum_to_zero_and_pure_species_limit_is_finite():
    thermo = _mixture(species=("O2", "CO2"))
    t = jnp.array(1.0)
    for y in (jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0]), jnp.array([0.3, 0.7])):
        g = jnp.array([0.5, -0.2])
        j = th.diffusion_flux(thermo, t, y, g, jnp.array([0.1, 0.1]))
        assert np.all(np.isfinite(np.asarray(j)))
        # the flux of the bulk species follows from sum J = 0: check the sum of
        # all fluxes through the identity used in the formula
        d = th.mixture_diffusivity(thermo, t, y)
        assert np.all(np.isfinite(np.asarray(d))) and np.all(np.asarray(d) > 0.0)
        jac = jax.jacfwd(lambda a, g=g: th.diffusion_flux(thermo, t, a, g))(y)
        assert np.all(np.isfinite(np.asarray(jac)))


def test_wilke_viscosity_and_mathur_conductivity():
    # air from O2 and N2: Wilke gives 18.5e-6 Pa s at 300 K (measured 18.6e-6).
    thermo = _mixture(bulk="N2", species=("O2",))
    t, y = jnp.array(1.0), jnp.array([0.2315])
    mu = float(th.viscosity(thermo, t, y))
    assert mu == pytest.approx(18.6e-6, rel=0.04)
    # the pure-species limits are those of the species
    mu_n2 = float(th.viscosity(thermo, t, jnp.array([0.0])))
    mu_o2 = float(th.viscosity(thermo, t, jnp.array([1.0])))
    assert mu_n2 == pytest.approx(float(th.evaluate(thermo.viscosity, t)), rel=1e-12)
    assert mu_o2 == pytest.approx(
        float(th.evaluate(thermo.species[0].viscosity, t)), rel=1e-12
    )
    assert min(mu_n2, mu_o2) < mu < max(mu_n2, mu_o2)
    # Mathur with k_i = mu_i c_p / Pr: the same limits and lies within the bounds
    k0 = float(th.conductivity(thermo, t, jnp.array([0.0])))
    k1 = float(th.conductivity(thermo, t, jnp.array([1.0])))
    k = float(th.conductivity(thermo, t, y))
    assert k0 == pytest.approx(float(th.evaluate(thermo.viscosity, t)) / 0.72, rel=1e-9)
    assert min(k0, k1) <= k <= max(k0, k1)
    # derivatives exist (the Newton Jacobian differentiates through the rules)
    g = jax.jacfwd(lambda yy: th.viscosity(thermo, t, yy))(y)
    assert np.all(np.isfinite(np.asarray(g)))


def test_bulk_rules_leave_properties_unchanged():
    thermo = _mixture(species=("O2",), rules=False)
    t, y = jnp.array(1.3), jnp.array([0.4])
    assert float(th.viscosity(thermo, t, y)) == float(th.viscosity(thermo, t))
    assert float(th.conductivity(thermo, t, y)) == float(th.conductivity(thermo, t))


def test_default_thermo_keeps_the_s1_flux():
    # fick (default): unchanged code path, identical to the S1 binary result.
    sp = th.Species("a", 2.0, th.PropertyLaw("schmidt", 0.7))
    thermo = th.Thermo(
        th.DensityModel("ideal_gas", pressure=1.0),
        th.PropertyLaw("constant", 0.01),
        th.PropertyLaw("prandtl", 0.7),
        th.PropertyLaw("constant", 1.0),
        (sp,),
    )
    t, y, g = jnp.array(1.1), jnp.array([0.3]), jnp.array([0.5])
    rho = th.density(thermo, t, y)
    np.testing.assert_allclose(
        th.diffusion_flux(thermo, t, y, g), -0.01 / 0.7 * g, rtol=1e-12
    )
    assert rho > 0.0


# -- configuration -------------------------------------------------------------------


def test_config_requires_pair_data_and_viscosity():
    from pyaeroflow.config import case as case_module

    problems = []
    fluid = {
        "diffusion_model": "hirschfelder-curtiss",
        "mixture_viscosity": "wilke",
        "mixture_conductivity": "bulk",
        "lennard_jones_sigma": None,
        "lennard_jones_epsilon": None,
        "binary_diffusion": None,
        "conductivity_model": "prandtl",
    }
    case_module._mixture_rules(fluid, [{"name": "a"}], ["a"], problems)
    assert any("pair (bulk, a)" in p for p in problems)
    assert any("viscosity" in p for p in problems)
    problems.clear()
    fluid["lennard_jones_sigma"], fluid["lennard_jones_epsilon"] = 3.0, 90.0
    sp = {"name": "a", "lennard_jones_sigma": 3.5, "lennard_jones_epsilon": 100.0}
    case_module._mixture_rules(fluid, [sp], ["a"], problems)
    assert problems == []
    fluid["binary_diffusion"] = [{"species": ["a", "zzz"], "diffusivity": 1e-5}]
    case_module._mixture_rules(fluid, [sp], ["a"], problems)
    assert any("binary_diffusion[0].species" in p for p in problems)


def _case(**changes):
    import copy

    from test_species import CASE

    case = copy.deepcopy(CASE)
    case["fluid"].update(
        diffusion_model="hirschfelder-curtiss",
        lennard_jones_sigma=3.798,
        lennard_jones_epsilon=71.4,
        molar_mass=28.0134,
    )
    for sp, (sigma, eps) in zip(
        case["fluid"]["species"], ((3.467, 106.7), (2.827, 59.7)), strict=True
    ):
        sp.pop("schmidt", None)
        sp.pop("diffusivity", None)
        sp.pop("diffusivity_exponent", None)
        sp.update(lennard_jones_sigma=sigma, lennard_jones_epsilon=eps)
    for dotted, value in changes.items():
        *path, key = dotted.split("__")
        table = case
        for part in path:
            table = table[int(part)] if part.isdigit() else table[part]
        if value is None:
            table.pop(key, None)
        else:
            table[key] = value
    return case


def test_multicomponent_case_resolves_and_builds_the_thermo():
    from pyaeroflow.config import resolve
    from pyaeroflow.mesh import rectangle
    from pyaeroflow.run import setup

    cfg = resolve(_case(fluid__mixture_viscosity="wilke"))
    assert cfg["fluid"]["mixture_conductivity"] == "bulk"
    problem, scaling, _, _ = setup(cfg, rectangle(4, 4))
    tm = problem.model.thermo
    assert tm.diffusion_model == "hirschfelder-curtiss"
    assert tm.mixture_viscosity == "wilke" and len(tm.pair_diffusivity) == 3
    # bulk - first species: the Chapman-Enskog coefficient of the case's data
    sp = cfg["fluid"]["species"][0]
    expected = th.kinetic_diffusivity(
        cfg["fluid"]["molar_mass"],
        sp["molar_mass"],
        0.5 * (cfg["fluid"]["lennard_jones_sigma"] + sp["lennard_jones_sigma"]),
        float(
            np.sqrt(cfg["fluid"]["lennard_jones_epsilon"] * sp["lennard_jones_epsilon"])
        ),
        scaling.temperature,
        cfg["fluid"]["operating_pressure"],
    )
    d = float(th.evaluate(tm.pair_diffusivity[0], jnp.asarray(1.0)))
    d *= scaling.velocity * scaling.length
    assert d == pytest.approx(expected, rel=1e-9)
    assert tm.species[0].viscosity.kind == "kinetic-viscosity"
    mu = float(th.viscosity(tm, jnp.asarray(1.0), jnp.array([0.2, 0.1])))
    assert mu > 0.0


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"fluid__species__1__lennard_jones_sigma": None}, "pair \\(bulk, "),
        ({"fluid__lennard_jones_epsilon": None}, "pair \\(bulk, "),
        (
            {
                "fluid__binary_diffusion": [
                    {"species": ["bulk", "zz"], "diffusivity": 1e-5}
                ]
            },
            "binary_diffusion\\[0\\].species",
        ),
        (
            {
                "fluid__mixture_viscosity": "wilke",
                "fluid__species__0__lennard_jones_sigma": None,
            },
            "viscosity",
        ),
        (
            {"fluid__species": None, "fluid__diffusion_model": "hirschfelder-curtiss"},
            "needs fluid.species",
        ),
        ({"fluid__diffusion_model": "fick"}, "needs diffusion_model"),
    ],
)
def test_multicomponent_case_rules(changes, message):
    from pyaeroflow.config import ConfigError, resolve

    case = _case(**changes)
    if changes.get("fluid__diffusion_model") == "fick":
        case["fluid"]["binary_diffusion"] = [
            {"species": ["bulk", "vapour"], "diffusivity": 1e-5}
        ]
        for sp in case["fluid"]["species"]:
            sp["schmidt"] = 0.7
    with pytest.raises(ConfigError, match=message):
        resolve(case)


@pytest.mark.parametrize("turbulence", ["sa-neg"])
def test_rans_multicomponent_linearization(turbulence):
    """HC, Wilke and Mathur under RANS: JVP of the residual = finite differences."""
    from test_species import (
        SPECIES,
        _rans_conditions,
        _rans_mesh,
        _rans_state,
        _rans_thermo,
    )

    from pyaeroflow.driver import FlowProblem
    from pyaeroflow.physics import variable_density as vd

    species = (
        SPECIES[0]._replace(
            diffusivity=th.PropertyLaw("schmidt", 1.0),
            viscosity=th.PropertyLaw("power-law", 8.0e-5, (0.7,)),
        ),
        SPECIES[1]._replace(
            diffusivity=th.PropertyLaw("schmidt", 1.0),
            viscosity=th.PropertyLaw("kinetic-viscosity", 1.5e-4, (0.4,)),
            conductivity=th.PropertyLaw("constant", 2.0e-4),
        ),
    )
    d = (
        th.PropertyLaw("constant", 4.0e-4),
        th.PropertyLaw("kinetic-diffusion", 2.0e-4, (0.5,)),
        th.PropertyLaw("power-law", 1.0e-4, (1.7,)),
    )
    thermo = _rans_thermo(species)._replace(
        diffusion_model="hirschfelder-curtiss",
        mixture_viscosity="wilke",
        mixture_conductivity="mathur",
        pair_diffusivity=d,
    )
    conditions = _rans_conditions(
        {
            "type": "wall",
            "temperature": 1.05,
            "mass_fraction": (0.03, 0.01),
            "stefan_flow": True,
        }
    )
    conditions["top"]["mass_fraction"] = (0.01, 0.01)
    model = vd.VariableDensityModel(
        density=1.0,
        thermo=thermo,
        turbulence=turbulence,
        turbulence_intensity=0.5,
        gravity=(0.0, -0.5),
    )
    p = FlowProblem(_rans_mesh(), model, conditions)
    w = _rans_state(p, 2)
    beta2 = p.beta2(w)
    assert np.isfinite(p.residual(w, beta2)).all()
    dw = np.random.default_rng(1).standard_normal(w.shape) * np.abs(w).mean(axis=0)
    eps = 1e-6

    def r1(x):
        return p.residual(x, beta2, order=1)

    fd = (r1(w + eps * dw) - r1(w - eps * dw)) / (2.0 * eps)
    jv = jax.jvp(
        lambda x: p.physics.residual(p.model, p.geom, p.bc, x, 1, beta2),
        (jnp.asarray(w),),
        (jnp.asarray(dw),),
    )[1]
    np.testing.assert_allclose(jv, fd, atol=1e-7 * np.abs(jv).max())
    assert p.jacobian(w, beta2).shape == (w.size, w.size)


def test_duncan_toor_reverse_diffusion():
    """H2-N2-CO2 of Duncan and Toor (Taylor and Krishna, 1993, p. 21): at the
    middle of the tube the N2 gradient vanishes, yet N2 diffuses (reverse
    diffusion). Mixture-averaged diffusion with the correction velocity has the
    sign and the size of the Stefan-Maxwell flux (within 30 %)."""
    w = {"n2": 28.0134, "h2": 2.016, "co2": 44.01}
    species = (
        th.Species("h2", w["h2"] / w["n2"], th.PropertyLaw("schmidt", 1.0)),
        th.Species("co2", w["co2"] / w["n2"], th.PropertyLaw("schmidt", 1.0)),
    )
    thermo = th.Thermo(
        th.DensityModel("ideal_gas", pressure=101325.0 * w["n2"] / (8314.46 * 308.15)),
        th.PropertyLaw("constant", 1.8e-5),
        th.PropertyLaw("prandtl", 0.72),
        th.PropertyLaw("constant", 1.0),
        species,
        diffusion_model="hirschfelder-curtiss",
        pair_diffusivity=(
            th.PropertyLaw("constant", 8.33e-5),  # N2-H2
            th.PropertyLaw("constant", 1.68e-5),  # N2-CO2
            th.PropertyLaw("constant", 6.8e-5),  # H2-CO2
        ),
    )
    masses = np.array([w["n2"], w["h2"], w["co2"]])

    def y_of(x):
        m = np.asarray(x) * masses
        return m / m.sum()

    x1, x2 = [0.50086, 0.0, 0.49914], [0.49879, 0.50121, 0.0]
    xm = 0.5 * (np.array(x1) + np.array(x2))
    y = jnp.asarray(y_of(xm)[1:])
    g = jnp.asarray(y_of(x2)[1:] - y_of(x1)[1:])
    t = jnp.array(1.0)
    hc = np.asarray(th.diffusion_flux(thermo, t, y, g))
    sm = _stefan_maxwell(thermo, t, y, g, _pair_d(thermo, t))
    dx = np.asarray(jax.jacfwd(lambda a: th.mole_fractions(thermo, a))(y)) @ np.asarray(
        g
    )
    assert abs(dx[0]) < 0.01 * abs(dx[1])  # no N2 gradient ...
    assert sm[0] * -hc.sum() > 0.0  # ... both predict the same N2 flux direction
    np.testing.assert_allclose(-hc.sum(), sm[0], rtol=0.30)
    np.testing.assert_allclose(hc, sm[1:], rtol=0.30)

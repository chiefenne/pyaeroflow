"""Species transport in the incompressible energy solver (stage S1, laminar)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.config import ConfigError, resolve
from pyaeroflow.driver import FlowProblem
from pyaeroflow.io.fields import Capabilities, default_fields
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import thermo as th
from pyaeroflow.physics import variable_density as vd

SPECIES = (
    th.Species(
        "a",
        0.4,
        th.PropertyLaw("power-law", 0.03, (1.7,)),
        th.PropertyLaw("polynomial", 1.0, (2.0, 0.5)),
        0.3,
        0.2,
    ),
    th.Species("b", 2.5, th.PropertyLaw("schmidt", 0.7), None, 0.0, -0.1),
)


def _thermo(kind="ideal_gas", n_species=2, diffusion=True):
    density = (
        th.DensityModel("ideal_gas", pressure=1.1)
        if kind == "ideal_gas"
        else th.DensityModel(kind, density=1.3, expansion=0.8)
    )
    species = SPECIES[:n_species]
    if not diffusion:
        species = tuple(
            s._replace(diffusivity=th.PropertyLaw("constant", 0.0)) for s in species
        )
    return th.Thermo(
        density,
        th.PropertyLaw("sutherland", 0.02 if diffusion else 0.0, (0.37,)),
        th.PropertyLaw("prandtl", 0.71)
        if diffusion
        else th.PropertyLaw("constant", 0.0),
        th.PropertyLaw("polynomial", 1.0, (0.8, 0.2)),
        species,
    )


def _model(thermo, **kwargs):
    return vd.VariableDensityModel(
        density=float(th.density(thermo, jnp.array(1.0))), thermo=thermo, **kwargs
    )


@pytest.mark.parametrize("kind", ["constant", "boussinesq", "ideal_gas"])
def test_mixture_properties_and_derivatives(kind):
    """rho(T, Y): AD derivatives; c_p = dh/dT; pure bulk fluid at Y = 0."""
    thermo = _thermo(kind)
    t, y = jnp.array(1.15), jnp.array([0.2, 0.3])
    np.testing.assert_allclose(
        th.density_derivative(thermo, t, y),
        jax.grad(lambda x: th.density(thermo, x, y))(t),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        th.density_species_derivative(thermo, t, y),
        jax.grad(lambda x: th.density(thermo, t, x))(y),
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        th.specific_heat(thermo, t, y),
        jax.grad(lambda x: th.enthalpy(thermo, x, y))(t),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        th.species_enthalpy(thermo, t),
        jax.grad(lambda x: th.enthalpy(thermo, t, x))(y),
        rtol=1e-12,
    )
    zero = jnp.zeros(2)
    for with_y, bulk in (
        (th.density(thermo, t, zero), th.density(thermo, t)),
        (th.enthalpy(thermo, t, zero), th.enthalpy(thermo, t)),
        (th.specific_heat(thermo, t, zero), th.specific_heat(thermo, t)),
    ):
        assert float(with_y) == pytest.approx(float(bulk), rel=1e-14)


def test_ideal_gas_mixture_density():
    """rho = p W / T with 1 / W = sum Y_k / W_k (the bulk has W = 1)."""
    thermo = _thermo()
    y = np.array([0.2, 0.3])
    inverse = y[0] / 0.4 + y[1] / 2.5 + (1.0 - y.sum())
    assert float(th.density(thermo, 1.2, y)) == pytest.approx(1.1 / (1.2 * inverse))


def test_diffusion_fluxes():
    """One species: Fick's law. Several: all fluxes (with the bulk) sum to zero."""
    one = _thermo(n_species=1)
    t, y, g = jnp.array([1.1]), jnp.array([[0.2]]), jnp.array([[0.7]])
    rho_d = th.density_diffusivity(one, t, y)
    np.testing.assert_allclose(th.diffusion_flux(one, t, y, g), -rho_d * g)
    rho = th.density(one, t, y)
    np.testing.assert_allclose(rho_d[:, 0], rho * 0.03 * 1.1**1.7, rtol=1e-13)
    two = _thermo()
    y, g = jnp.array([[0.2, 0.3]]), jnp.array([[0.7, -0.2]])
    rho_d = np.asarray(th.density_diffusivity(two, t, y))[0]
    np.testing.assert_allclose(rho_d[1], th.viscosity(two, t)[0] / 0.7)
    j = np.asarray(th.diffusion_flux(two, t, y, g))[0]
    # The bulk species: the mass-averaged diffusivity, gradient -sum dY_k.
    y_np, g_np = np.asarray(y)[0], np.asarray(g)[0]
    d_bulk = np.sum(y_np * rho_d) / y_np.sum()
    fick = np.append(-rho_d * g_np, d_bulk * g_np.sum())
    y_all = np.append(y_np, 1.0 - y_np.sum())
    expected = fick - y_all * fick.sum()
    np.testing.assert_allclose(j, expected[:2], rtol=1e-12)
    assert expected.sum() == pytest.approx(0.0, abs=1e-15)
    # D at the operating pressure: rho D does not depend on p_th.
    closed = vd.at_pressure(_model(two), 1.1 * 1.3).thermo
    assert closed.pressure_ratio == pytest.approx(1.3)
    np.testing.assert_allclose(
        th.density_diffusivity(closed, t, y)[0], rho_d, rtol=1e-13
    )


@pytest.mark.parametrize("kind", ["ideal_gas", "boussinesq"])
@pytest.mark.parametrize("n_species", [1, 2])
def test_preconditioner_and_dissipation_with_species(kind, n_species):
    """Gamma = dU/dV and Gamma |Gamma^-1 A| of (p, u, v, T, Y)."""
    thermo = _thermo(kind, n_species)
    model = _model(thermo)
    nv = 4 + n_species
    beta2 = 1.7**2
    w = jnp.array([0.3, 0.6, -0.35, 1.15, 0.2, 0.3][:nv])
    n = jnp.array([np.cos(0.7), np.sin(0.7)])

    def rho_art(x):
        return th.density(thermo, x[3], x[4:]) + (x[0] - w[0]) / beta2

    def conservative(x):
        r = rho_art(x)
        h = th.enthalpy(thermo, x[3], x[4:])
        return jnp.concatenate((jnp.stack((r, r * x[1], r * x[2], r * h)), r * x[4:]))

    def flux(x):
        vn = x[1] * n[0] + x[2] * n[1]
        pressure = jnp.zeros(nv).at[1:3].set(x[0] * n)
        return conservative(x) * vn + pressure

    gamma = np.asarray(jax.jacfwd(conservative)(w))
    a = np.asarray(jax.jacfwd(flux)(w))
    ours_g = np.asarray(vd.preconditioner(model, w[None], beta2))[0]
    np.testing.assert_allclose(ours_g, gamma, atol=1e-13)
    lam, p = np.linalg.eig(np.linalg.solve(gamma, a))
    vn = float(w[1] * n[0] + w[2] * n[1])
    b = np.sqrt(beta2)
    np.testing.assert_allclose(
        np.sort(lam.real), sorted([vn - b, vn + b] + [vn] * (nv - 2))
    )
    abs_m = (p @ np.diag(np.abs(lam)) @ np.linalg.inv(p)).real
    ours = np.asarray(vd.abs_preconditioned_jacobian(model, w[None], n[None], beta2))
    np.testing.assert_allclose(ours[0], abs_m, atol=1e-12)
    dw = jnp.array([[0.1, -0.2, 0.05, 0.03, 0.02, -0.04][:nv]])
    np.testing.assert_allclose(
        vd.dissipation(model, w[None], dw, n[None], beta2)[0],
        ours_g @ abs_m @ np.asarray(dw[0]),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        vd.fds_flux(model, w[None], w[None], n[None], beta2),
        vd.inviscid_flux(model, w[None], n[None]),
        atol=1e-15,
    )


CONDITIONS = (
    {
        "bottom": {"type": "symmetry"},
        "top": {
            "type": "wall",
            "temperature": 1.1,
            "mass_fraction": (0.3, 0.1),
            "stefan_flow": True,
        },
        "left": {
            "type": "velocity-inlet",
            "velocity": (1.0, 0.1),
            "temperature": 0.9,
            "mass_fraction": (0.05, 0.2),
        },
        "right": {"type": "pressure-outlet", "pressure": 0.0},
    },
    {
        "bottom": {"type": "wall", "species_flux": (0.01, 0.02)},
        "top": {"type": "wall", "temperature": 1.1, "mass_fraction": (0.3, 0.1)},
        "left": {"type": "wall"},
        "right": {
            "type": "farfield",
            "velocity": (0.5, 0.2),
            "pressure": 0.0,
            "mass_fraction": (0.1, 0.1),
        },
    },
)


def _state(mesh, n_species, seed=0):
    rng = np.random.default_rng(seed)
    w = 0.5 + rng.uniform(-0.3, 0.3, (mesh.n_cells, 4 + n_species))
    w[:, 3] = 1.0 + rng.uniform(-0.1, 0.1, mesh.n_cells)
    w[:, 4:] = rng.uniform(0.05, 0.3, (mesh.n_cells, n_species))
    return w


@pytest.mark.parametrize("conditions", CONDITIONS, ids=["stefan", "flux"])
@pytest.mark.parametrize("kind", ["ideal_gas", "boussinesq"])
def test_first_order_jacobian_is_exact_without_diffusion(kind, conditions):
    """Convection, boundary data and the buoyancy blocks of the species columns."""
    mesh = rectangle(4, 3, cell_type="mixed", perturb=0.3, seed=4)
    model = _model(_thermo(kind, diffusion=False), gravity=(0.3, -2.0), spatial_order=1)
    prob = FlowProblem(mesh, model, conditions)
    w = _state(mesh, 2)
    beta2 = prob.beta2(w)
    sparse = prob.jacobian(w, beta2).toarray()
    dense = jax.jacfwd(
        lambda x: vd.residual(
            prob.model, prob.geom, prob.bc, x.reshape(w.shape), 1, beta2
        ).ravel()
    )(jnp.asarray(w.ravel()))
    np.testing.assert_allclose(sparse, np.asarray(dense), atol=1e-12)


def test_species_walls():
    """Zero-flux, flux, fixed and Stefan-flow walls; balances of the residual."""
    mesh = rectangle(6, 5, cell_type="mixed", perturb=0.3, seed=2)
    conditions = {
        "bottom": {"type": "wall", "species_flux": (0.01, 0.02)},
        "top": {
            "type": "wall",
            "temperature": 1.1,
            "mass_fraction": (0.3, 0.1),
            "stefan_flow": True,
        },
        "left": {"type": "wall"},
        "right": {"type": "wall", "mass_fraction": (0.1, 0.2)},
    }
    model = _model(_thermo())
    prob = FlowProblem(mesh, model, conditions)
    assert prob.model.n_var == 6
    assert prob.model.layout.names[-3:] == ("T", "Y1", "Y2")
    w = _state(mesh, 2, seed=3)
    w[:, :3] = 0.0
    j, m = prob.wall_species_flux(w)
    faces = {name: prob.boundary_faces([name]) for name in conditions}
    np.testing.assert_allclose(j[faces["bottom"]], [[0.01, 0.02]] * 6)
    assert np.all(j[faces["left"]] == 0.0)
    assert np.all(m[faces["bottom"]] == 0.0) and np.all(m[faces["right"]] == 0.0)
    y_w = prob.wall_mass_fraction(w)
    np.testing.assert_allclose(y_w[faces["right"]], [[0.1, 0.2]] * 5)
    # Stefan flow: the bulk species does not cross the wall, m (1 - sum Y) is
    # the sum of the diffusion fluxes, and j = diffusion + m Y.
    top = faces["top"]
    y_top = np.array([0.3, 0.1])
    diffusion = j[top] - m[top, None] * y_top
    np.testing.assert_allclose(m[top] * (1.0 - y_top.sum()), diffusion.sum(axis=1))
    np.testing.assert_allclose(j[top].sum(axis=1), m[top], rtol=1e-12)
    # At rest the residual sums to the boundary fluxes: the continuity rows
    # to the Stefan mass flux, the species rows to the wall fluxes.
    length = np.asarray(prob.geom.b_length)
    r = np.asarray(
        vd.residual(prob.model._replace(pin=None), prob.geom, prob.bc, w, 2, 1.0)
    )
    assert r[:, 0].sum() == pytest.approx(-np.sum(m * length), rel=1e-10)
    np.testing.assert_allclose(
        r[:, 4:].sum(axis=0), -np.sum(j * length[:, None], axis=0), rtol=1e-10
    )


def test_closed_domain_and_time_term_with_species():
    """p_th keeps the mass of the mixture; Q has rho Y behind rho h - c p_th."""
    mesh = rectangle(5, 4, cell_type="mixed", perturb=0.3, seed=1)
    thermo = _thermo()
    w = _state(mesh, 2, seed=5)
    area = np.asarray(mesh.cell_area)
    conditions = {n: {"type": "wall"} for n in ("bottom", "top", "left", "right")}
    model = _model(thermo, closed_mass=1.3 * area.sum(), pressure_work=0.25)
    prob = FlowProblem(mesh, model, conditions)
    assert prob.closed
    p_th = prob.thermodynamic_pressure(w)
    at_pth = vd.at_pressure(model, p_th).thermo
    rho = np.asarray(th.density(at_pth, w[:, 3], w[:, 4:]))
    assert np.sum(rho * area) == pytest.approx(model.closed_mass, rel=1e-13)
    q = np.asarray(
        vd.conservative_variables(prob.model, prob.geom, jnp.asarray(w), jnp.ones(6))
    )
    h = np.asarray(th.enthalpy(at_pth, w[:, 3], w[:, 4:]))
    pinned = prob.model.pin[0]
    keep = np.arange(mesh.n_cells) != pinned
    np.testing.assert_allclose(q[keep, 0], (area * rho)[keep], rtol=1e-13)
    np.testing.assert_allclose(q[:, 3], area * (rho * h - 0.25 * p_th), rtol=1e-12)
    np.testing.assert_allclose(q[:, 4:], (area * rho)[:, None] * w[:, 4:], rtol=1e-13)


CASE = {
    "mesh": {"file": "m.su2"},
    "physics": {"model": "laminar", "energy": True},
    "fluid": {
        "density_model": "ideal_gas",
        "viscosity": 1.8e-5,
        "species": [
            {"name": "vapour", "molar_mass": 18.015, "schmidt": 0.6},
            {
                "name": "helium",
                "molar_mass": 4.0026,
                "diffusivity": 7.0e-5,
                "diffusivity_exponent": 1.75,
                "specific_heat": 5193.0,
            },
        ],
    },
    "reference": {"velocity": 0.5, "length": 0.1, "temperature": 300.0},
    "initial": {"velocity": [0.0, 0.0], "mass_fraction": [0.0, 0.1]},
    "boundary": {
        "left": {"type": "wall", "mass_fraction": [0.2, 0.0], "stefan_flow": True},
        "right": {"type": "wall", "species_flux": [1.0e-4, 0.0]},
        "top": {"type": "wall"},
        "bottom": {"type": "wall", "temperature": 310.0},
    },
}


def _case(**changes):
    import copy

    case = copy.deepcopy(CASE)
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


def test_species_case_resolves():
    cfg = resolve(CASE)
    fluid = cfg["fluid"]
    assert fluid["molar_mass"] == pytest.approx(8314.462618 / 287.058)
    assert [s["name"] for s in fluid["species"]] == ["vapour", "helium"]
    caps = Capabilities.from_config(cfg.data)
    assert caps.species == 2
    names = default_fields(caps)
    assert names[names.index("temperature") + 1 :][:2] == [
        "mass_fraction_1",
        "mass_fraction_2",
    ]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"physics__energy": False}, "physics.energy = true"),
        ({"physics__model": "rans"}, "physics.model = laminar"),
        ({"fluid__species__0__schmidt": None}, "either diffusivity or schmidt"),
        ({"fluid__species__1__schmidt": 0.3}, "either diffusivity or schmidt"),
        ({"fluid__species__1__name": "vapour"}, "unique name"),
        ({"fluid__species__0__lewis": 1.0}, "unknown key"),
        ({"fluid__species__0__expansion": 0.5}, "density_model = boussinesq"),
        ({"boundary__left__mass_fraction": [0.2]}, "one per species"),
        ({"boundary__left__mass_fraction": [0.7, 0.4]}, "sum to more than 1"),
        ({"boundary__right__mass_fraction": [0.1, 0.1]}, "not both"),
        ({"boundary__top__stefan_flow": True}, "needs mass_fraction"),
        ({"fluid__species": None}, "needs fluid.species"),
    ],
)
def test_species_case_rules(changes, message):
    with pytest.raises(ConfigError, match=message):
        resolve(_case(**changes))

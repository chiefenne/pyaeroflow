"""Incompressible energy equation (variable density, V1): pieces and set-up."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.config import ConfigError, resolve
from pyaeroflow.driver import FlowProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import thermo as th
from pyaeroflow.physics import variable_density as vd
from pyaeroflow.physics.layout import incompressible_layout, map_columns

LAWS = [
    th.PropertyLaw("constant", 1.7),
    th.PropertyLaw("sutherland", 1.7, (0.37,)),
    th.PropertyLaw("power-law", 1.7, (0.7,)),
    th.PropertyLaw("polynomial", 1.7, (0.9, 0.5, 0.3)),
    th.PropertyLaw("table", 1.0, ((0.8, 1.0, 1.3), (0.6, 1.0, 1.9))),
]


@pytest.mark.parametrize("law", LAWS, ids=lambda law: law.kind)
def test_integral_derivative_is_the_law(law):
    """h(T) of every c_p law: dh/dT = c_p (AD) and h(1) = 0."""
    t = jnp.array([0.7, 0.85, 0.95, 1.0, 1.1, 1.25, 1.5])
    dh = jax.vmap(jax.grad(lambda x: th.integral(law, x)))(t)
    np.testing.assert_allclose(dh, th.evaluate(law, t), rtol=1e-6)
    assert float(th.integral(law, jnp.array(1.0))) == pytest.approx(0.0, abs=1e-14)


def test_property_law_from_si_input():
    """SI parameters become non-dimensional laws of T / T_ref."""
    t_ref, scale = 300.0, 2.0
    poly = th.property_law(
        "polynomial",
        None,
        {"coefficients": [1.0, 0.01, 1e-5]},
        t_ref=t_ref,
        scale=scale,
    )
    t = 330.0
    expected = (1.0 + 0.01 * t + 1e-5 * t**2) / scale
    assert float(th.evaluate(poly, t / t_ref)) == pytest.approx(expected)
    table = th.property_law(
        "table", None, {"temperature": [250, 350], "value": [2.0, 4.0]}, t_ref=t_ref
    )
    assert float(th.evaluate(table, 1.0)) == pytest.approx(3.0)
    assert table.value == pytest.approx(3.0)
    suth = th.property_law(
        "sutherland", 1.8e-5, {"temperature": 110.4}, t_ref=t_ref, scale=1e-5
    )
    s = 110.4 / t_ref
    tt = 1.2
    mu = 1.8 * tt**1.5 * (1 + s) / (tt + s)
    assert float(th.evaluate(suth, tt)) == pytest.approx(mu)
    power = th.property_law("power-law", 2.0, {"exponent": 0.7}, t_ref=t_ref)
    assert float(th.evaluate(power, 1.3)) == pytest.approx(2.0 * 1.3**0.7)
    with pytest.raises(ValueError):
        th.property_law("table", None, {"temperature": [300, 250], "value": [1, 2]})


@pytest.mark.parametrize(
    "model",
    [
        th.DensityModel("constant", density=1.3),
        th.DensityModel("boussinesq", density=1.3, expansion=0.8, temperature=1.0),
        th.DensityModel("ideal_gas", pressure=1.1),
    ],
    ids=lambda m: m.kind,
)
def test_density_derivative(model):
    thermo = th.Thermo(density=model)
    t = jnp.array([0.8, 1.0, 1.2])
    ad = jax.vmap(jax.grad(lambda x: th.density(thermo, x)))(t)
    np.testing.assert_allclose(th.density_derivative(thermo, t), ad, rtol=1e-12)


def test_prandtl_conductivity_and_turbulent_prandtl_interface():
    thermo = th.Thermo(
        viscosity=th.PropertyLaw("power-law", 0.01, (0.7,)),
        conductivity=th.PropertyLaw("prandtl", 0.7),
        specific_heat=th.PropertyLaw("polynomial", 1.0, (0.9, 0.1)),
    )
    t = jnp.array([0.9, 1.1])
    k = th.conductivity(thermo, t)
    np.testing.assert_allclose(
        k, th.viscosity(thermo, t) * th.specific_heat(thermo, t) / 0.7
    )
    rho, mu_t = jnp.ones(2), jnp.array([0.0, 0.05])
    k_eff = th.effective_conductivity(thermo, th.TurbulentPrandtl(), t, rho, mu_t)
    np.testing.assert_allclose(k_eff, k + th.specific_heat(thermo, t) * mu_t / 0.9)

    def peclet_law(inputs, a=0.85, b=0.1):
        return a + b / (1.0 + inputs.peclet)

    th.register_turbulent_prandtl("test-peclet", peclet_law)
    law = th.TurbulentPrandtl("test-peclet", (("a", 0.8), ("b", 0.2)))

    def total(mu_t):
        return jnp.sum(th.effective_conductivity(thermo, law, t, rho, mu_t))

    # Differentiable (it enters the Newton Jacobian).
    assert np.all(np.isfinite(np.asarray(jax.grad(total)(mu_t))))
    with pytest.raises(ValueError, match="unknown turbulent Prandtl"):
        th.turbulent_prandtl(th.TurbulentPrandtl("nope", ()), None)


@pytest.mark.parametrize(
    "turbulence, transition",
    [("none", "none"), ("sa-neg", "none"), ("sa-neg", "bcm"), ("sa-neg", "aft")]
    + [("sst", "none"), ("sst", "lm")],
)
def test_layout_matches_column_counts(turbulence, transition):
    model = inc.IncompressibleModel(turbulence=turbulence, transition=transition)
    layout = model.layout
    assert layout.n_var == model.n_var
    assert layout.names[:3] == ("p", "u", "v")
    energy = incompressible_layout(turbulence, transition, energy=True)
    # The energy block is appended: earlier columns keep their numbers.
    assert energy.names[: layout.n_var] == layout.names
    assert energy.column("T") == layout.n_var


def test_map_columns():
    assert map_columns(["p", "u", "v", "T"], ["p", "u", "v", "nu_tilde"]) == [
        (0, 0),
        (1, 1),
        (2, 2),
    ]
    assert map_columns(["p", "u", "v"], ["p", "u", "v", "T"]) == [
        (0, 0),
        (1, 1),
        (2, 2),
    ]


def _thermo(kind="ideal_gas"):
    dens = {
        "ideal_gas": th.DensityModel("ideal_gas", pressure=1.0),
        "boussinesq": th.DensityModel("boussinesq", density=1.0, expansion=1.0),
        "constant": th.DensityModel("constant", density=1.3),
    }[kind]
    return th.Thermo(
        dens,
        th.PropertyLaw("sutherland", 0.02, (0.37,)),
        th.PropertyLaw("prandtl", 0.71),
        th.PropertyLaw("polynomial", 1.0, (0.8, 0.2)),
    )


def test_preconditioner_and_dissipation_match_eigendecomposition():
    """Gamma = dU/dV and |Gamma^-1 A| of the 4x4 system (d(rho)/dp = 1/beta^2)."""
    model = vd.VariableDensityModel(thermo=_thermo())
    beta2 = 1.7**2
    w = jnp.array([0.3, 0.6, -0.35, 1.15])
    n = jnp.array([np.cos(0.7), np.sin(0.7)])

    def rho_art(x):  # density with the artificial pressure dependence
        return th.density(model.thermo, x[3]) + (x[0] - w[0]) / beta2

    def conservative(x):
        r = rho_art(x)
        return jnp.stack((r, r * x[1], r * x[2], r * th.enthalpy(model.thermo, x[3])))

    def flux(x):
        r = rho_art(x)
        vn = x[1] * n[0] + x[2] * n[1]
        h = th.enthalpy(model.thermo, x[3])
        return jnp.stack(
            (
                r * vn,
                r * x[1] * vn + x[0] * n[0],
                r * x[2] * vn + x[0] * n[1],
                r * h * vn,
            )
        )

    gamma = np.asarray(jax.jacfwd(conservative)(w))
    a = np.asarray(jax.jacfwd(flux)(w))
    ours_g = np.asarray(vd.preconditioner(model, w[None], beta2))[0]
    np.testing.assert_allclose(ours_g, gamma, atol=1e-13)
    lam, p = np.linalg.eig(np.linalg.solve(gamma, a))
    vn = float(w[1] * n[0] + w[2] * n[1])
    b = np.sqrt(beta2)
    np.testing.assert_allclose(np.sort(lam.real), sorted([vn - b, vn, vn, vn + b]))
    abs_m = (p @ np.diag(np.abs(lam)) @ np.linalg.inv(p)).real
    ours = np.asarray(vd.abs_preconditioned_jacobian(model, w[None], n[None], beta2))
    np.testing.assert_allclose(ours[0], abs_m, atol=1e-12)
    dw = jnp.array([[0.1, -0.2, 0.05, 0.03]])
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


def _cavity_conditions():
    return {
        "left": {"type": "wall", "temperature": 1.05},
        "right": {"type": "wall", "temperature": 0.95},
        "top": {"type": "wall", "velocity": (0.3, 0.0), "heat_flux": 0.02},
        "bottom": {"type": "wall"},
    }


def test_constant_density_reproduces_the_incompressible_residual():
    """density_model = constant: the flow rows are those of the INC solver."""
    mesh = rectangle(6, 5, cell_type="mixed", perturb=0.3, seed=2)
    mu, rho = 0.02, 1.3
    thermo = th.Thermo(
        th.DensityModel("constant", density=rho),
        th.PropertyLaw("constant", mu),
        th.PropertyLaw("prandtl", 0.71),
        th.PropertyLaw("constant", 1.0),
    )
    cond = _cavity_conditions()
    m_vd = vd.VariableDensityModel(
        density=rho, thermo=thermo, limiter="venkatakrishnan"
    )
    m_inc = inc.IncompressibleModel(
        density=rho, viscosity=mu, limiter="venkatakrishnan"
    )
    p_vd = FlowProblem(mesh, m_vd, cond)
    p_inc = FlowProblem(
        mesh,
        m_inc,
        {
            k: {
                "type": c["type"],
                **({"velocity": c["velocity"]} if "velocity" in c else {}),
            }
            for k, c in cond.items()
        },
    )
    assert p_vd.thermal and not p_inc.thermal
    rng = np.random.default_rng(1)
    w = rng.uniform(-0.3, 0.3, (mesh.n_cells, 4))
    w[:, 3] = 1.0 + 0.05 * w[:, 3]
    beta2 = p_inc.beta2(w[:, :3])
    r_vd = p_vd.residual(w, beta2)
    r_inc = p_inc.residual(w[:, :3], beta2)
    np.testing.assert_allclose(r_vd[:, :3], r_inc, atol=1e-13)


@pytest.mark.parametrize("kind", ["ideal_gas", "boussinesq"])
def test_first_order_jacobian_is_exact_without_diffusion(kind):
    """Convection, boundary data and the buoyancy source blocks are exact."""
    mesh = rectangle(4, 3, cell_type="mixed", perturb=0.3, seed=4)
    base = _thermo(kind)
    thermo = base._replace(
        viscosity=th.PropertyLaw("constant", 0.0),
        conductivity=th.PropertyLaw("constant", 0.0),
    )
    conditions = {
        "bottom": {"type": "symmetry"},
        "top": {"type": "wall", "temperature": 1.1},
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.1), "temperature": 0.9},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
    }
    model = vd.VariableDensityModel(
        density=float(th.density(thermo, jnp.array(1.0))),
        thermo=thermo,
        gravity=(0.3, -2.0),
        spatial_order=1,
    )
    prob = FlowProblem(mesh, model, conditions)
    rng = np.random.default_rng(0)
    w = 0.5 + rng.uniform(-0.3, 0.3, (mesh.n_cells, 4))
    w[:, 3] = 1.0 + rng.uniform(-0.1, 0.1, mesh.n_cells)
    beta2 = prob.beta2(w)
    sparse = prob.jacobian(w, beta2).toarray()
    dense = jax.jacfwd(
        lambda x: vd.residual(
            prob.model, prob.geom, prob.bc, x.reshape(w.shape), 1, beta2
        ).ravel()
    )(jnp.asarray(w.ravel()))
    np.testing.assert_allclose(sparse, np.asarray(dense), atol=1e-12)


def test_thermal_walls_and_beta_floor():
    """Heat-flux wall returns its value; isothermal walls fix T; beta^2 floor."""
    mesh = rectangle(8, 8)
    model = vd.VariableDensityModel(thermo=_thermo(), gravity=(0.0, -50.0))
    prob = FlowProblem(mesh, model, _cavity_conditions())
    w = prob.initial_state((0.0, 0.0), 0.0, temperature=1.0)
    assert w.shape[1] == 4 and np.all(w[:, 3] == 1.0)
    q = prob.wall_heat_flux(w)
    top = prob.boundary_faces(["top"])
    np.testing.assert_allclose(q[top], 0.02)
    assert np.all(q[prob.boundary_faces(["bottom"])] == 0.0)
    assert np.all(q[prob.boundary_faces(["left"])] > 0.0)  # hot wall heats
    t_w = prob.wall_temperature(w)
    np.testing.assert_allclose(t_w[prob.boundary_faces(["left"])], 1.05)
    assert np.all(t_w[top] > 1.0)  # heated by the prescribed flux
    # At rest, beta^2 follows the buoyancy velocity of the wall temperatures.
    ub2 = 50.0 * (1.0 / 0.95 - 1.0 / 1.05) / model.density
    assert prob.beta2(w) == pytest.approx(model.beta_factor * max(ub2, 0.3**2))
    # RANS (stage V2): the variable-density SA module, T in the last column.
    rans = FlowProblem(mesh, model._replace(turbulence="sa-neg"), _cavity_conditions())
    assert rans.physics.__name__.endswith("rans_variable_density")
    assert rans.initial_state((0.0, 0.0)).shape[1] == 5


CASE = {
    "mesh": {"file": "m.su2"},
    "physics": {"model": "laminar", "energy": True, "gravity": [0.0, -9.81]},
    "fluid": {"density_model": "ideal_gas", "viscosity": 1.8e-5},
    "reference": {"velocity": 0.05, "length": 0.03, "temperature": 300.0},
    "initial": {"velocity": [0.0, 0.0]},
    "boundary": {
        "left": {"type": "wall", "temperature": 301.5},
        "right": {"type": "wall", "temperature": 298.5},
        "top": {"type": "wall", "heat_flux": 5.0},
        "bottom": {"type": "wall"},
    },
}


def test_energy_case_resolves():
    cfg = resolve(CASE)
    fluid, ref = cfg["fluid"], cfg["reference"]
    assert fluid["conductivity_model"] == "prandtl"
    assert fluid["viscosity_model"] == "constant"
    assert ref["density"] == pytest.approx(101325.0 / (287.058 * 300.0))


def _problems(raw):
    with pytest.raises(ConfigError) as err:
        resolve(raw)
    return str(err.value)


def test_energy_case_rules():
    raw = {**CASE, "reference": {"velocity": 0.05, "length": 0.03}}
    assert "reference.temperature is required" in _problems(raw)
    raw = {**CASE, "physics": {"model": "euler", "energy": True}}
    assert "needs physics.model = laminar or rans" in _problems(raw)
    both = {
        **CASE["boundary"],
        "top": {"type": "wall", "heat_flux": 1.0, "temperature": 300.0},
    }
    assert "not both" in _problems({**CASE, "boundary": both})
    plain = {
        **CASE,
        "physics": {"model": "laminar", "gravity": [0.0, -9.81]},
        "fluid": {"density": 1.2, "viscosity": 1.8e-5},
    }
    msg = _problems(plain)
    assert "physics.gravity: needs physics.energy" in msg
    assert "boundary.left.temperature" in msg
    power = {**CASE, "fluid": {**CASE["fluid"], "viscosity_model": "power-law"}}
    assert "viscosity_parameters.exponent is required" in _problems(power)


def test_energy_setup_builds_the_model():
    from pyaeroflow.run import setup

    cfg = resolve(CASE)
    mesh = rectangle(4, 4, x=(0.0, 0.03), y=(0.0, 0.03))
    problem, scaling, w0, _ = setup(cfg, mesh)
    assert problem.thermal and isinstance(problem.model, vd.VariableDensityModel)
    assert scaling.temperature == 300.0
    assert scaling.specific_heat == pytest.approx(1.4 * 287.058 / 0.4)
    np.testing.assert_allclose(w0[:, 3], 1.0)
    assert problem.model.density == pytest.approx(1.0)  # rho(T_ref) = rho_ref
    assert problem.model.gravity[1] == pytest.approx(-9.81 * 0.03 / 0.05**2)
    hf = problem.bc.heat_flux[problem.boundary_faces(["top"])]
    np.testing.assert_allclose(hf, 5.0 / scaling.heat_flux)


CASE_TOML = """
[mesh]
file = "m.su2"
[physics]
model = "laminar"
energy = true
gravity = [0.0, -9.81]
[fluid]
density_model = "ideal_gas"
viscosity = 1.8e-5
[reference]
velocity = 0.05
length = 0.03
temperature = 300.0
temperature_difference = 3.0
[initial]
velocity = [0.0, 0.0]
[boundary.left]
type = "wall"
temperature = 301.5
[boundary.right]
type = "wall"
temperature = 298.5
[boundary.top]
type = "wall"
[boundary.bottom]
type = "wall"
[numerics]
limiter = "none"
[solver]
max_iterations = 200
residual_drop = 8.0
[output]
directory = "out"
"""


def test_run_case_writes_heat_transfer_and_restarts_by_name(tmp_path):
    """End to end: wall CSV with T, q_w, Nu, St; restart columns by name."""
    from pyaeroflow.mesh import write_su2
    from pyaeroflow.run import load_restart, mesh_fingerprint, run_case

    mesh = rectangle(
        12,
        12,
        x=(0.0, 0.03),
        y=(0.0, 0.03),
        cluster_x="both",
        cluster_y="both",
        beta=1.5,
    )
    write_su2(tmp_path / "m.su2", mesh)
    path = tmp_path / "cav.toml"
    path.write_text(CASE_TOML)
    result = run_case(path, console=False)
    assert result.converged
    wall = (tmp_path / "out" / "cav.wall_left.csv").read_text().splitlines()
    header = wall[0].split(",")
    assert {"T", "q_w", "nu", "st"} <= set(header)
    row = [float(x) for x in wall[1].split(",")]
    assert row[header.index("T")] == pytest.approx(301.5)
    assert row[header.index("q_w")] > 0.0
    log = (tmp_path / "out" / "cav.log").read_text()
    assert "heat 'left'" in log and "mean Nu" in log
    restart = result.restart_file
    # Into a constant-density case: p, u, v only (the file names its columns).
    w3, n = load_restart(restart, result.mesh, np.zeros((result.mesh.n_cells, 3)))
    assert n == 3
    np.testing.assert_allclose(w3, np.load(restart)["w"][:, :3])
    # Into an SA case: nu_tilde is not taken from T.
    w4 = np.full((result.mesh.n_cells, 4), 7.0)
    w4, n = load_restart(restart, result.mesh, w4)
    assert n == 3 and np.all(w4[:, 3] == 7.0)
    # Energy case from an older file without names: p, u, v only.
    w_old = np.load(restart)["w"]
    old = tmp_path / "old.npz"
    np.savez(
        old,
        w=np.column_stack((w_old[:, :3], np.full(len(w_old), 5.0))),
        fingerprint=mesh_fingerprint(result.mesh),
        flow="incompressible",
    )
    w5, n = load_restart(
        old,
        result.mesh,
        np.ones((result.mesh.n_cells, 4)),
        columns=("p", "u", "v", "T"),
    )
    assert n == 3 and np.all(w5[:, 3] == 1.0)

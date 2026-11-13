"""RANS with the incompressible energy equation (variable density, stage V2)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.driver import FlowProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import rans_sst_variable_density as rsvd
from pyaeroflow.physics import rans_variable_density as rvd
from pyaeroflow.physics import thermo as th
from pyaeroflow.physics import variable_density as vd

MODELS = [
    ("sa-neg", "none"),
    ("sa-neg", "bcm"),
    ("sa-neg", "aft"),
    ("sst", "none"),
    ("sst", "lm"),
]
IDS = ["sa", "sa-bcm", "sa-aft", "sst", "sst-lm"]


def _mesh():
    return rectangle(
        10, 6, x=(-0.2, 1.0), y=(0.0, 0.4), cell_type="mixed", perturb=0.25, seed=3
    )


def _conditions(wall_temperature=None):
    wall = {"type": "wall"}
    if wall_temperature is not None:
        wall["temperature"] = wall_temperature
    return {
        "bottom": wall,
        "top": {"type": "farfield", "velocity": (1.0, 0.02), "pressure": 0.0},
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.0)},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
    }


def _random_state(problem, seed=7):
    w = problem.initial_state((1.0, 0.0))
    rng = np.random.default_rng(seed)
    w = w * (1.0 + 0.2 * rng.uniform(-1.0, 1.0, w.shape))
    w[:, :3] += 0.1 * rng.standard_normal((w.shape[0], 3))
    return w


def _ideal_gas_thermo(mu=1.0e-4):
    return th.Thermo(
        th.DensityModel("ideal_gas", pressure=1.0),
        th.PropertyLaw("sutherland", mu, (0.37,)),
        th.PropertyLaw("prandtl", 0.71),
        th.PropertyLaw("polynomial", 1.0, (0.8, 0.2)),
    )


@pytest.mark.parametrize("turbulence, transition", MODELS, ids=IDS)
def test_constant_density_reproduces_the_rans_residual(turbulence, transition):
    """rho = 1, constant mu: flow and turbulence rows of the constant-density RANS."""
    mesh, mu = _mesh(), 1.0e-4
    thermo = th.Thermo(
        th.DensityModel("constant", density=1.0),
        th.PropertyLaw("constant", mu),
        th.PropertyLaw("prandtl", 0.71),
        th.PropertyLaw("constant", 1.0),
    )
    common = dict(
        turbulence=turbulence,
        transition=transition,
        turbulence_intensity=0.5,
        limiter="venkatakrishnan",
    )
    m_inc = inc.IncompressibleModel(density=1.0, viscosity=mu, **common)
    p_inc = FlowProblem(mesh, m_inc, _conditions())
    p_vd = FlowProblem(
        mesh,
        vd.VariableDensityModel(density=1.0, thermo=thermo, **common),
        _conditions(),
    )
    assert p_vd.physics in (rvd, rsvd)
    w = _random_state(p_inc)
    rng = np.random.default_rng(3)
    t = 1.0 + 0.05 * rng.uniform(-1.0, 1.0, (w.shape[0], 1))
    w_vd = np.concatenate((w, t), axis=1)
    n_var = w.shape[1]
    beta2 = p_inc.beta2(w)
    for order in (1, 2):
        r_inc = p_inc.residual(w, beta2, order=order)
        r_vd = p_vd.residual(w_vd, beta2, order=order)
        np.testing.assert_allclose(r_vd[:, :n_var], r_inc, rtol=0.0, atol=1e-15)
    # First-order Jacobian without the T row and column.
    cells = np.arange(w.shape[0])[:, None]
    idx = (cells * (n_var + 1) + np.arange(n_var)[None, :]).ravel()
    j_inc = p_inc.jacobian(w, beta2).toarray()
    j_vd = p_vd.jacobian(w_vd, beta2).toarray()[np.ix_(idx, idx)]
    np.testing.assert_allclose(j_vd, j_inc, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        p_vd.eddy_viscosity(w_vd), p_inc.eddy_viscosity(w), rtol=1e-14, atol=1e-20
    )


@pytest.mark.parametrize("turbulence, transition", MODELS, ids=IDS)
def test_variable_density_pieces(turbulence, transition):
    """Ideal gas, Sutherland: residual, JVP, source cap, limiter, output."""
    mesh = _mesh()
    model = vd.VariableDensityModel(
        density=1.0,
        thermo=_ideal_gas_thermo(),
        turbulence=turbulence,
        transition=transition,
        turbulence_intensity=0.5,
        lm_positive_jacobian=False,
        gravity=(0.0, -0.5),
    )
    p = FlowProblem(mesh, model, _conditions(1.3))
    assert model.layout.column("T") == model.n_var - 1
    w = _random_state(p)
    beta2 = p.beta2(w)
    r = p.residual(w, beta2)
    assert np.isfinite(r).all()
    rng = np.random.default_rng(1)
    dw = rng.standard_normal(w.shape) * np.abs(w).mean(axis=0)
    eps = 1e-6

    def r1(x):
        return p.residual(x, beta2, order=1)

    fd = (r1(w + eps * dw) - r1(w - eps * dw)) / (2.0 * eps)
    jv = jax.jvp(
        lambda x: p.physics.residual(p.model, p.geom, p.bc, x, 1, beta2),
        (jnp.asarray(w),),
        (jnp.asarray(dw),),
    )[1]
    if transition != "bcm":  # BCM: gamma is frozen in the linearization
        np.testing.assert_allclose(jv, fd, atol=1e-7 * np.abs(jv).max())
    assert p.jacobian(w, beta2).shape == (w.size, w.size)
    cap = p.source_cfl_cap(w, beta2, 0.9)
    assert cap.shape == (mesh.n_cells,)
    p.time_term(w, beta2, 10.0, transition_cfl_max=5.0, source_limit=0.9)
    # Step limiter: T stays positive also with the turbulence limiter.
    du = p._step_limiter(0.5)(w, -2.0 * w)
    assert (w + du)[:, -1].min() > 0.0
    t_w = p.wall_temperature(w)[p.boundary_faces(["bottom"])]
    np.testing.assert_allclose(t_w, 1.3)
    assert np.isfinite(p.wall_heat_flux(w)).all()
    assert np.all(p.kinematic_viscosity(w) > 0.0)


def test_turbulent_prandtl_law_sets_the_eddy_conductivity():
    """k_eff = k + c_p mu_t / Pr_t with a registered Pr_t law (wall distance input)."""

    def linear(inputs, a=0.85, b=1.0):
        return a + b * inputs.wall_distance

    th.register_turbulent_prandtl("test-linear", linear)
    model = vd.VariableDensityModel(
        thermo=_ideal_gas_thermo(),
        turbulent_prandtl=th.TurbulentPrandtl("test-linear", (("a", 0.5),)),
    )
    n = 3
    grad_v = jnp.zeros((n, 2, 2))
    grad_t = jnp.tile(jnp.array([0.0, 2.0]), (n, 1))
    t_f = jnp.array([0.9, 1.0, 1.2])
    mu_t = jnp.array([0.0, 1e-3, 2e-3])
    d = jnp.array([0.1, 0.2, 0.3])
    normal = jnp.tile(jnp.array([0.0, 1.0]), (n, 1))
    flux = rvd.energy_viscous_flux(model, grad_v, grad_t, t_f, mu_t, d, normal)
    tm = model.thermo
    k = th.conductivity(tm, t_f) + th.specific_heat(tm, t_f) * mu_t / (0.5 + d)
    np.testing.assert_allclose(flux[:, 3], -2.0 * k, rtol=1e-14)


def test_sst_wall_omega_uses_the_wall_temperature():
    """omega_w = 60 nu(T_w) / (beta_1 d1^2) at an isothermal wall."""
    mesh = _mesh()
    model = vd.VariableDensityModel(
        density=1.0, thermo=_ideal_gas_thermo(), turbulence="sst"
    )
    p = FlowProblem(mesh, model, _conditions(1.5))
    w = p.initial_state((1.0, 0.0))
    faces = p.boundary_faces(["bottom"])
    w_b = np.asarray(
        rsvd.boundary_state(p.bc, p.geom.b_normal, jnp.asarray(w)[p.geom.b_owner])
    )[faces]
    rho, nu = rvd.cell_properties(model, jnp.full(faces.size, 1.5))
    d1 = 2.0 * np.abs(
        np.einsum(
            "fk,fk->f",
            np.asarray(p.geom.b_delta)[faces],
            np.asarray(p.geom.b_normal)[faces],
        )
    )
    from pyaeroflow.physics import sst

    np.testing.assert_allclose(w_b[:, rsvd.NW], sst.wall_omega(np.asarray(nu), d1))
    np.testing.assert_allclose(w_b[:, -1], 1.5)


PLATE_TOML = """
[mesh]
file = "plate.su2"
[physics]
model = "rans"
turbulence = "{turbulence}"
energy = true
[fluid]
density_model = "ideal_gas"
viscosity = 1.846e-5
specific_heat = 1006.43
prandtl = 0.71
prandtl_turbulent = 0.85
[reference]
temperature = 300.0
length = 1.0
velocity = 15.7
[initial]
from_boundary = "left"
[boundary.left]
type = "velocity-inlet"
velocity = [15.7, 0.0]
temperature = 300.0
[boundary.right]
type = "pressure-outlet"
[boundary.top]
type = "farfield"
velocity = 15.7
temperature = 300.0
[boundary.symmetry]
type = "symmetry"
[boundary.plate]
type = "wall"
temperature = 330.0
[solver]
max_iterations = {iterations}
residual_drop = 8.0
[output]
directory = "out"
force_boundaries = ["plate"]
"""


@pytest.mark.parametrize("turbulence", ["sa-neg", "sst"])
def test_run_case_rans_energy(tmp_path, turbulence):
    """End to end: RANS energy case set-up, wall heat transfer, restarts by name."""
    from pyaeroflow.cases.tmr_flat_plate import plate_mesh
    from pyaeroflow.mesh import write_su2
    from pyaeroflow.run import load_restart, mesh_fingerprint, run_case

    mesh = plate_mesh(24, 8, 24, reynolds=1.0e6)
    write_su2(tmp_path / "plate.su2", mesh)
    path = tmp_path / "plate.toml"
    path.write_text(PLATE_TOML.format(turbulence=turbulence, iterations=12))
    result = run_case(path, console=False)
    problem = result.problem
    assert problem.thermal and problem.turbulent
    model = problem.model
    assert isinstance(model, vd.VariableDensityModel)
    assert model.turbulence == turbulence
    assert dict(model.turbulent_prandtl.parameters)["value"] == pytest.approx(0.85)
    names = model.layout.names
    first = "nu_tilde" if turbulence == "sa-neg" else "k"
    assert names[-1] == "T" and names[3] == first
    wall = (tmp_path / "out" / "plate.wall_plate.csv").read_text().splitlines()
    header = wall[0].split(",")
    assert {"T", "q_w", "st", "y_plus"} <= set(header)
    rows = np.array([[float(x) for x in r.split(",")] for r in wall[1:]])
    np.testing.assert_allclose(rows[:, header.index("T")], 330.0)
    assert np.all(rows[:, header.index("q_w")] > 0.0)
    # Restart into the same layout: all columns by name.
    restart = result.restart_file
    w_file = np.load(restart)["w"]
    w, n = load_restart(restart, result.mesh, np.zeros_like(w_file), columns=names)
    assert n == len(names)
    np.testing.assert_allclose(w, w_file)
    # A constant-density RANS file without names keeps its turbulence columns.
    old = tmp_path / "old.npz"
    np.savez(
        old,
        w=w_file[:, :-1],
        fingerprint=mesh_fingerprint(result.mesh),
        flow="incompressible",
    )
    w, n = load_restart(old, result.mesh, np.ones_like(w_file), columns=names)
    assert n == len(names) - 1
    np.testing.assert_allclose(w[:, :-1], w_file[:, :-1])
    np.testing.assert_allclose(w[:, -1], 1.0)

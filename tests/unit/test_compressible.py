"""Compressible solver (L1): Euler and laminar fluxes, boundary states, setup."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.cases.oblique_shock import oblique_shock, ramp_mesh
from pyaeroflow.config import ConfigError, resolve
from pyaeroflow.driver import FlowProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import compressible as comp

GAMMA = 1.4


def _states(n, seed=0, mach=0.6):
    """Random physical states (p, u, v, T) around a freestream of Mach ``mach``."""
    rng = np.random.default_rng(seed)
    r_gas = comp.gas_constant_for_mach(mach)
    w = np.column_stack(
        (
            r_gas * rng.uniform(0.7, 1.3, n),
            rng.uniform(-1.2, 1.2, n),
            rng.uniform(-1.2, 1.2, n),
            rng.uniform(0.7, 1.3, n),
        )
    )
    a = rng.uniform(0.0, 2.0 * np.pi, n)
    normal = np.column_stack((np.cos(a), np.sin(a)))
    return r_gas, jnp.asarray(w), jnp.asarray(normal)


def test_roe_flux_is_consistent():
    r_gas, w, normal = _states(20)
    f = comp.roe_flux(GAMMA, r_gas, 0.001, w, w, normal)
    np.testing.assert_allclose(
        f, comp.physical_flux(GAMMA, r_gas, w, normal), rtol=1e-13, atol=1e-14
    )


def test_roe_flux_is_conservative():
    r_gas, w, normal = _states(20, seed=1)
    w_r = w[::-1]
    f = comp.roe_flux(GAMMA, r_gas, 0.001, w, w_r, normal)
    g = comp.roe_flux(GAMMA, r_gas, 0.001, w_r, w, -normal)
    np.testing.assert_allclose(f, -g, rtol=1e-12, atol=1e-13)


def test_roe_flux_upwinds_supersonic_flow():
    """All eigenvalues positive: the flux is the left physical flux."""
    r_gas = comp.gas_constant_for_mach(2.0)
    w_l = jnp.asarray([[r_gas, 1.0, 0.1, 1.0]])
    w_r = jnp.asarray([[1.3 * r_gas, 0.9, 0.0, 1.1]])
    n = jnp.asarray([[1.0, 0.0]])
    f = comp.roe_flux(GAMMA, r_gas, 0.001, w_l, w_r, n)
    np.testing.assert_allclose(f, comp.physical_flux(GAMMA, r_gas, w_l, n), rtol=1e-12)


def test_roe_dissipation_matches_eigendecomposition():
    """|A| (U_R - U_L) of the Roe matrix for a small jump (linearised)."""
    r_gas, w, normal = _states(1, seed=3)
    eps = 1e-6
    dw = jnp.asarray([[0.3 * r_gas, 0.2, -0.1, 0.05]]) * eps
    w_l, w_r = w - 0.5 * dw, w + 0.5 * dw
    f = comp.roe_flux(GAMMA, r_gas, 0.0, w_l, w_r, normal)
    f_c = 0.5 * (
        comp.physical_flux(GAMMA, r_gas, w_l, normal)
        + comp.physical_flux(GAMMA, r_gas, w_r, normal)
    )
    diss = -2.0 * (f - f_c)[0] / eps

    def flux_of_u(u):
        rho, mx, my, e = u
        vel = jnp.stack((mx / rho, my / rho))
        p = (GAMMA - 1.0) * (e - 0.5 * rho * jnp.sum(vel**2))
        prim = jnp.stack((p, vel[0], vel[1], p / (rho * r_gas)))[None]
        return comp.physical_flux(GAMMA, r_gas, prim, normal)[0]

    u = comp.conservative(GAMMA, r_gas, w)[0]
    a = np.asarray(jax.jacfwd(flux_of_u)(u))
    lam, vec = np.linalg.eig(a)
    abs_a = (vec @ np.diag(np.abs(lam)) @ np.linalg.inv(vec)).real
    du = np.asarray(
        comp.conservative(GAMMA, r_gas, w_r)[0]
        - comp.conservative(GAMMA, r_gas, w_l)[0]
    )
    np.testing.assert_allclose(diss, abs_a @ du / eps, rtol=1e-4, atol=1e-8)


def test_transformation_is_du_dv():
    r_gas, w, _ = _states(5, seed=2)
    model = comp.CompressibleModel(gas_constant=r_gas)
    m = np.asarray(comp.transformation(model, w))
    p, u, v, t = np.asarray(w[0])
    rho = p / (r_gas * t)
    q2 = u * u + v * v
    expected = np.array(
        [
            [rho / p, 0.0, 0.0, -rho / t],
            [u * rho / p, rho, 0.0, -u * rho / t],
            [v * rho / p, 0.0, rho, -v * rho / t],
            [
                1.0 / (GAMMA - 1.0) + 0.5 * q2 * rho / p,
                rho * u,
                rho * v,
                -0.5 * q2 * rho / t,
            ],
        ]
    )
    np.testing.assert_allclose(m[0], expected, rtol=1e-12)


def _bc(kind, value, r_gas):
    return comp.BoundaryData(
        jnp.asarray([kind]), jnp.asarray([value], dtype=float), GAMMA, r_gas
    )


def test_outlet_boundary_state_is_isentropic_or_supersonic():
    r_gas = comp.gas_constant_for_mach(0.5)
    n = jnp.asarray([[1.0, 0.0]])
    w_in = jnp.asarray([[r_gas, 1.0, 0.0, 1.0]])
    out = comp.boundary_state(
        _bc(comp.PRESSURE_OUTLET, [0.9 * r_gas, 0, 0, 0], r_gas), n, w_in
    )
    np.testing.assert_allclose(out[0, comp.P], 0.9 * r_gas)
    np.testing.assert_allclose(out[0, comp.T], 0.9 ** (0.4 / 1.4))
    fast = jnp.asarray([[r_gas, 3.0, 0.0, 1.0]])  # Mach 3 outflow
    out = comp.boundary_state(
        _bc(comp.PRESSURE_OUTLET, [0.9 * r_gas, 0, 0, 0], r_gas), n, fast
    )
    np.testing.assert_allclose(out, fast)


def test_total_conditions_inlet_state():
    mach = 0.4
    r_gas = comp.gas_constant_for_mach(mach)
    t0 = 1.0 + 0.2 * mach**2
    p0 = r_gas * t0**3.5
    w_in = jnp.asarray([[r_gas, 0.5, 0.0, 1.0]])
    out = comp.boundary_state(
        _bc(comp.TOTAL_PRESSURE_INLET, [p0, 1.0, 0.0, t0], r_gas),
        jnp.asarray([[-1.0, 0.0]]),
        w_in,
    )
    np.testing.assert_allclose(out[0], [r_gas, 1.0, 0.0, 1.0], rtol=1e-10)


def _box_problem(mach=0.6, alpha=5.0, order=2):
    mesh = rectangle(6, 5, cell_type="mixed", perturb=0.3, seed=4)
    r_gas, ff = comp.freestream_conditions(mach, alpha)
    model = comp.CompressibleModel(gas_constant=r_gas, spatial_order=order)
    conditions = {
        "bottom": {"type": "slip-wall"},
        "top": ff,
        "left": ff,
        "right": {"type": "pressure-outlet", "pressure": r_gas},
    }
    return FlowProblem(mesh, model, conditions), ff


@pytest.mark.parametrize("order", [1, 2])
def test_freestream_is_preserved(order):
    mesh = rectangle(5, 4, cell_type="triangle", perturb=0.3, seed=2)
    r_gas, ff = comp.freestream_conditions(0.8, 3.0)
    model = comp.CompressibleModel(gas_constant=r_gas, spatial_order=order)
    names = ("bottom", "top", "left", "right")
    prob = FlowProblem(mesh, model, dict.fromkeys(names, ff))
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    r = prob.residual(w, prob.scheme_parameters(w))
    assert np.abs(r).max() < 1e-13


def test_first_order_jacobian_matches_dense_ad():
    prob, ff = _box_problem(order=1)
    rng = np.random.default_rng(0)
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    w = w * (1.0 + rng.uniform(-0.1, 0.1, w.shape))
    sparse = prob.jacobian(w, 0.0).toarray()
    dense = jax.jacfwd(
        lambda x: comp.residual(
            prob.model, prob.geom, prob.bc, x.reshape(w.shape), 1
        ).ravel()
    )(jnp.asarray(w.ravel()))
    np.testing.assert_allclose(sparse, np.asarray(dense), atol=1e-12)


def test_jvp_matches_finite_difference():
    prob, ff = _box_problem(order=2)
    rng = np.random.default_rng(1)
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    w = w * (1.0 + rng.uniform(-0.05, 0.05, w.shape))
    dw = rng.standard_normal(w.shape) * 1e-2
    psi = prob.limiter(w)  # frozen: the residual is smooth in w
    jv = prob.jvp(w, dw, 0.0, psi)
    eps = 1e-6
    fd = (
        prob.residual(w + eps * dw, 0.0, psi=psi)
        - prob.residual(w - eps * dw, 0.0, psi=psi)
    ) / (2 * eps)
    np.testing.assert_allclose(jv, fd, rtol=1e-6, atol=1e-9)
    lin = prob.linearize(w, 0.0, psi)
    np.testing.assert_allclose(lin(dw), jv, rtol=1e-12, atol=1e-14)


def test_non_physical_reconstruction_falls_back_to_first_order():
    prob, ff = _box_problem(order=2)
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    w[0, comp.P] *= 20.0  # a strong local jump: unlimited MUSCL goes negative
    psi = jnp.ones_like(jnp.asarray(w))
    assert (
        int(
            comp.non_physical_faces(
                prob.model, prob.geom, prob.bc, jnp.asarray(w), 2, psi
            )
        )
        > 0
    )
    r = prob.residual(w, 0.0, psi=psi)
    assert np.all(np.isfinite(r))


def test_positivity_limiter_keeps_pressure_and_temperature_positive():
    prob, ff = _box_problem()
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    du = np.zeros_like(w)
    du[0, comp.P] = -3.0 * w[0, comp.P]
    du[1, comp.T] = -0.5 * w[1, comp.T]
    du[2, comp.U] = 1.0
    out = prob._step_limiter(None)(w, du)
    assert np.all(w[:, [comp.P, comp.T]] + out[:, [comp.P, comp.T]] > 0.0)
    np.testing.assert_allclose(out[2], du[2])  # untouched cell
    np.testing.assert_allclose(out[1], du[1])  # a 50 % drop passes


def test_compressible_problem_rejects_unsteady():
    mesh = rectangle(3, 3)
    r_gas, ff = comp.freestream_conditions(0.5)
    names = ("bottom", "top", "left", "right")
    prob = FlowProblem(
        mesh, comp.CompressibleModel(gas_constant=r_gas), dict.fromkeys(names, ff)
    )
    with pytest.raises(NotImplementedError):
        prob.mass()


def test_oblique_shock_relations():
    s = oblique_shock(2.0, math.radians(10.0))
    assert math.degrees(s.beta) == pytest.approx(39.3139, abs=1e-3)
    assert s.pressure_ratio == pytest.approx(1.7066, abs=1e-4)
    assert s.mach == pytest.approx(1.6405, abs=1e-4)


def test_ramp_first_order_captures_the_shock():
    """First order on a coarse mesh: ramp pressure within 2 % of the exact."""
    from pyaeroflow.cases.oblique_shock import run
    from pyaeroflow.solver import PseudoTimeSettings

    r = run(
        30,
        20,
        spatial_order=1,
        settings=PseudoTimeSettings(residual_drop=8.0, max_iterations=60),
    )
    assert r.converged
    assert r.pressure_ratio == pytest.approx(r.exact.pressure_ratio, rel=0.02)
    assert ramp_mesh(4, 3).n_cells == 12


# -- configuration -------------------------------------------------------------


def _raw(**farfield):
    return {
        "physics": {"flow": "compressible", "model": "euler"},
        "mesh": {"file": "m.su2"},
        "boundary": {
            "airfoil": {"type": "wall"},
            "farfield": {"type": "farfield", "mach": 0.8, **farfield},
        },
    }


def test_compressible_case_resolves_reference_values():
    config = resolve(_raw(angle_of_attack=1.25))
    ref = config.data["reference"]
    ff = config.data["boundary"]["farfield"]
    assert ff["pressure"] == 101325.0 and ff["temperature"] == 288.15
    assert ref["density"] == pytest.approx(101325.0 / (287.058 * 288.15))
    assert ref["velocity"] == pytest.approx(0.8 * math.sqrt(1.4 * 287.058 * 288.15))
    assert ref["temperature"] == 288.15
    assert config.data["numerics"]["convective_flux"] == "roe"
    assert config.data["initial"]["from_boundary"] == "farfield"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda r: r["physics"].update(
                model="rans",
                turbulence="sst",
                transition={"model": "lm", "turbulence_intensity": 0.01},
            ),
            "incompressible only",
        ),
        (lambda r: r["boundary"]["farfield"].pop("mach"), "mach is required"),
        (lambda r: r["boundary"]["farfield"].update(velocity=1.0), "take mach"),
        (lambda r: r.update(numerics={"convective_flux": "fds"}), "convective_flux"),
        (
            lambda r: r.update(
                time={"mode": "unsteady", "time_step": 1.0, "end_time": 2.0}
            ),
            "unsteady",
        ),
    ],
)
def test_compressible_case_rules(change, message):
    raw = _raw()
    change(raw)
    with pytest.raises(ConfigError, match=message):
        resolve(raw)


def test_incompressible_case_keeps_its_rules():
    raw = {
        "physics": {"model": "euler"},
        "mesh": {"file": "m.su2"},
        "fluid": {"density": 1.0},
        "boundary": {
            "airfoil": {"type": "wall"},
            "farfield": {"type": "farfield", "velocity": 1.0},
        },
    }
    config = resolve(raw)
    assert config.data["boundary"]["farfield"]["pressure"] == 0.0
    assert config.data["numerics"]["convective_flux"] == "fds"
    raw["boundary"]["farfield"]["mach"] = 0.5
    with pytest.raises(ConfigError, match="needs physics.flow = compressible"):
        resolve(raw)
    del raw["boundary"]["farfield"]["mach"]
    del raw["fluid"]
    with pytest.raises(ConfigError, match="fluid.density is required"):
        resolve(raw)


def test_restart_of_another_flow_family_is_rejected(tmp_path):
    from pyaeroflow.run import load_restart, save_restart

    mesh = rectangle(3, 3)
    w = np.ones((mesh.n_cells, 4))
    save_restart(tmp_path / "c.npz", mesh, w, flow="compressible")
    w0, n = load_restart(
        tmp_path / "c.npz", mesh, np.zeros_like(w), flow="compressible"
    )
    assert n == 4 and np.all(w0 == 1.0)
    with pytest.raises(ValueError, match="compressible state"):
        load_restart(tmp_path / "c.npz", mesh, np.zeros((mesh.n_cells, 4)))


def test_explicit_mode_reduces_the_residual():
    from pyaeroflow.cases.oblique_shock import ramp_mesh

    mesh = ramp_mesh(16, 10)
    r_gas, ff = comp.freestream_conditions(2.0)
    model = comp.CompressibleModel(gas_constant=r_gas, spatial_order=1)
    names = ("inflow", "top", "outflow")
    prob = FlowProblem(
        mesh, model, {"wall": {"type": "slip-wall"}, **dict.fromkeys(names, ff)}
    )
    w0 = prob.initial_state(ff["velocity"], ff["pressure"])
    result = prob.solve_explicit(w0, cfl=0.8, max_iterations=400, residual_drop=2.0)
    assert result.converged


# -- laminar Navier-Stokes (C2) ---------------------------------------------------


def test_sutherland_viscosity():
    model = comp.CompressibleModel(viscosity=2e-4, sutherland_temperature=0.4)
    t = jnp.asarray([1.0, 2.0])
    mu = np.asarray(comp.laminar_viscosity(model, t))
    np.testing.assert_allclose(mu, [2e-4, 2e-4 * 2.0**1.5 * 1.4 / 2.4], rtol=1e-14)
    const = model._replace(viscosity_law="constant")
    np.testing.assert_allclose(comp.laminar_viscosity(const, t), 2e-4)


def test_viscous_flux_of_linear_fields():
    """Shear flow u = a y with dT/dx = b: tau_xy = mu a, heat flux k b."""
    r_gas = comp.gas_constant_for_mach(0.5)
    model = comp.CompressibleModel(
        gas_constant=r_gas, viscosity=1e-3, viscosity_law="constant"
    )
    a, b = 2.0, 0.3
    grad = jnp.zeros((2, 4, 2)).at[:, comp.U, 1].set(a).at[:, comp.T, 0].set(b)
    w_f = jnp.asarray([[r_gas, 0.5, 0.0, 1.0]] * 2)
    normal = jnp.asarray([[0.0, 1.0], [1.0, 0.0]])
    f = np.asarray(comp.viscous_flux(model, grad, w_f, normal))
    k = 1e-3 * comp.heat_capacity(model) / model.prandtl
    # Face normal to y: shear stress, its work u tau_xy, no heat flux.
    np.testing.assert_allclose(f[0], [0.0, -1e-3 * a, 0.0, -0.5 * 1e-3 * a])
    # Face normal to x: tau_yx on the y-momentum row, heat flux k b.
    np.testing.assert_allclose(f[1], [0.0, 0.0, -1e-3 * a, -k * b])
    adiabatic = comp.viscous_flux(model, grad, w_f, normal, heat=False)
    assert float(adiabatic[1, 3]) == 0.0


def test_wall_states_adiabatic_and_isothermal():
    r_gas = comp.gas_constant_for_mach(0.5)
    n = jnp.asarray([[0.0, -1.0]])
    w_in = jnp.asarray([[r_gas, 0.8, 0.1, 1.05]])
    adiabatic = comp.boundary_state(_bc(comp.WALL, [0, 0, 0, 0], r_gas), n, w_in)
    np.testing.assert_allclose(adiabatic[0], [r_gas, 0.0, 0.0, 1.05])
    hot = comp.boundary_state(_bc(comp.WALL, [0, 0, 0, 1.3], r_gas), n, w_in)
    np.testing.assert_allclose(hot[0], [r_gas, 0.0, 0.0, 1.3])


def _viscous_box(wall_temperature=None, order=2):
    mesh = rectangle(6, 5, cell_type="mixed", perturb=0.3, seed=4)
    r_gas, ff = comp.freestream_conditions(0.5, 2.0)
    model = comp.CompressibleModel(
        gas_constant=r_gas, viscosity=1e-2, spatial_order=order
    )
    wall = {"type": "wall"}
    if wall_temperature is not None:
        wall["temperature"] = wall_temperature
    conditions = {
        "bottom": wall,
        "top": ff,
        "left": ff,
        "right": {"type": "pressure-outlet", "pressure": r_gas},
    }
    return FlowProblem(mesh, model, conditions), ff


def test_viscous_freestream_is_preserved():
    mesh = rectangle(5, 4, cell_type="triangle", perturb=0.3, seed=2)
    r_gas, ff = comp.freestream_conditions(0.5, 3.0)
    model = comp.CompressibleModel(gas_constant=r_gas, viscosity=1e-3)
    prob = FlowProblem(
        mesh, model, dict.fromkeys(("bottom", "top", "left", "right"), ff)
    )
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    assert np.abs(prob.residual(w, 0.0)).max() < 1e-13


@pytest.mark.parametrize("wall_temperature", [None, 1.2])
def test_viscous_jvp_matches_finite_difference(wall_temperature):
    prob, ff = _viscous_box(wall_temperature)
    rng = np.random.default_rng(3)
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    w = w * (1.0 + rng.uniform(-0.05, 0.05, w.shape))
    dw = rng.standard_normal(w.shape) * 1e-2
    psi = prob.limiter(w)
    jv = prob.jvp(w, dw, 0.0, psi)
    eps = 1e-6
    fd = (
        prob.residual(w + eps * dw, 0.0, psi=psi)
        - prob.residual(w - eps * dw, 0.0, psi=psi)
    ) / (2 * eps)
    np.testing.assert_allclose(jv, fd, rtol=1e-6, atol=1e-9)


def test_viscous_first_order_jacobian_is_close_to_exact():
    """The compact viscous Jacobian keeps the two-point part of the face
    gradient and drops the LSQ-gradient terms: close to dense AD, not equal.
    """
    prob, ff = _viscous_box(order=1)
    rng = np.random.default_rng(0)
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    w = w * (1.0 + rng.uniform(-0.1, 0.1, w.shape))
    sparse = prob.jacobian(w, 0.0).toarray()
    dense = np.asarray(
        jax.jacfwd(
            lambda x: comp.residual(
                prob.model, prob.geom, prob.bc, x.reshape(w.shape), 1
            ).ravel()
        )(jnp.asarray(w.ravel()))
    )
    err = np.linalg.norm(sparse - dense) / np.linalg.norm(dense)
    assert err < 0.05, err


def test_adiabatic_wall_has_no_heat_flux():
    prob, ff = _viscous_box()
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    w[:, comp.T] *= 1.0 + 0.1 * np.sin(np.arange(w.shape[0]))
    faces = prob.boundary_faces(["bottom"])
    np.testing.assert_array_equal(prob.wall_heat_flux(w)[faces], 0.0)
    hot, _ = _viscous_box(wall_temperature=1.5)
    assert np.all(hot.wall_heat_flux(w)[faces] > 0.0)  # hot wall heats the fluid


def test_similarity_solution():
    from pyaeroflow.cases.flat_plate_compressible import similarity

    low = similarity(0.0, sutherland=None)
    assert low.cf_sqrt_re == pytest.approx(0.664115, abs=1e-5)
    ad = similarity(2.0)
    assert ad.recovery_factor == pytest.approx(math.sqrt(0.72), rel=0.01)
    assert ad.cf_sqrt_re < low.cf_sqrt_re
    cold = similarity(0.5, wall_temperature=1.0)
    assert cold.heat_flux_sqrt_re < 0.0  # below T_aw: heat into the wall


def test_compressible_flat_plate():
    from pyaeroflow.cases.flat_plate_compressible import solve

    r = solve(nx=48, ny=32)
    assert r.converged
    sel = (r.x > 0.3) & (r.x < 0.9)
    cf = r.cf[sel] * np.sqrt(1.0e5 * r.x[sel])
    assert np.mean(cf) == pytest.approx(r.reference.cf_sqrt_re, rel=0.03)
    assert np.mean(r.wall_temperature[sel]) == pytest.approx(
        r.reference.wall_temperature, rel=1e-3
    )


def _laminar_raw(**fluid):
    ff = {"type": "farfield", "mach": 0.5}
    return {
        "physics": {"flow": "compressible", "model": "laminar"},
        "mesh": {"file": "m.su2"},
        "fluid": fluid,
        "reference": {"length": 1.0},
        "boundary": {"bottom": {"type": "wall"}, "top": ff, "left": ff, "right": ff},
    }


def test_compressible_laminar_case_setup():
    from pyaeroflow.run import setup

    raw = _laminar_raw(reynolds_number=5000.0)
    raw["boundary"]["bottom"]["temperature"] = 300.0
    config = resolve(raw)
    mesh = rectangle(3, 3)
    problem, scaling, _, _ = setup(config, mesh)
    m = problem.model
    assert m.viscosity == pytest.approx(1.0 / 5000.0)
    assert m.sutherland_temperature == pytest.approx(110.4 / 288.15)
    assert problem.conditions["bottom"]["type"] == "wall"
    assert problem.conditions["bottom"]["temperature"] == pytest.approx(300 / 288.15)
    # Without viscosity or Re: Sutherland's law of air at T_ref.
    config = resolve(_laminar_raw())
    assert any("Sutherland" in n for n in config.notices)
    problem, scaling, _, _ = setup(config, mesh)
    mu = problem.model.viscosity * scaling.viscosity
    assert mu == pytest.approx(1.7894e-5, rel=1e-3)


def test_compressible_laminar_case_rules():
    with pytest.raises(ConfigError, match="exactly one"):
        resolve(_laminar_raw(reynolds_number=5000.0, viscosity=1.8e-5))
    raw = _raw()
    raw["boundary"]["airfoil"]["temperature"] = 300.0
    assert any("ignored" in n for n in resolve(raw).notices)  # Euler


def test_low_mach_correction():
    """Rieper's correction: no change at M >= 1, consistent, less dissipation."""
    r_gas = comp.gas_constant_for_mach(0.1)
    w_l = jnp.array([[r_gas, 0.1, 0.02, 1.0]])
    w_r = jnp.array([[r_gas * 1.001, 0.09, -0.01, 1.002]])
    a, b = comp.low_mach_states(1.4, r_gas, w_l, w_r)
    np.testing.assert_allclose(a[:, [0, 3]], w_l[:, [0, 3]])  # p, T unchanged
    mean = 0.5 * (w_l[:, 1:3] + w_r[:, 1:3])
    assert np.all(np.abs(a[:, 1:3] - mean) < np.abs(w_l[:, 1:3] - mean))
    fast = w_l.at[:, 1].set(15.0)  # c = sqrt(gamma R T) = 10: M = 1.5, z = 1
    a, b = comp.low_mach_states(1.4, r_gas, fast, w_r)
    np.testing.assert_allclose(a, fast)
    np.testing.assert_allclose(b, w_r)
    n = jnp.array([[1.0, 0.0]])
    model = comp.CompressibleModel(gas_constant=r_gas, low_mach_correction=True)
    f = comp.interior_convective_flux(model, w_l, w_l, n)
    np.testing.assert_allclose(f, comp.physical_flux(1.4, r_gas, w_l, n), rtol=1e-12)


def test_low_mach_correction_jacobian_and_jvp():
    prob, ff = _viscous_box(order=2)
    prob.model = prob.model._replace(low_mach_correction=True)
    prob._fi = partial(comp.interior_flux, prob.model)
    rng = np.random.default_rng(8)
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    w = w * (1.0 + rng.uniform(-0.05, 0.05, w.shape))
    dw = rng.standard_normal(w.shape) * 1e-2
    psi = prob.limiter(w)
    jv = prob.jvp(w, dw, 0.0, psi)
    eps = 1e-6
    fd = (
        prob.residual(w + eps * dw, 0.0, psi=psi)
        - prob.residual(w - eps * dw, 0.0, psi=psi)
    ) / (2 * eps)
    np.testing.assert_allclose(jv, fd, rtol=1e-6, atol=1e-9)

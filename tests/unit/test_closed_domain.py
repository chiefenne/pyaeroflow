"""Closed domains and unsteady flow with the energy equation (stage V3)."""

import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.config import ConfigError, resolve
from pyaeroflow.driver import FlowProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import thermo as th
from pyaeroflow.physics import variable_density as vd
from pyaeroflow.solver import ConservativeMass, DualTimeSettings, PseudoTimeSettings

MODELS = [
    ("none", "none"),
    ("sa-neg", "none"),
    ("sa-neg", "aft"),
    ("sst", "none"),
    ("sst", "lm"),
]
IDS = ["laminar", "sa", "sa-aft", "sst", "sst-lm"]


def _thermo():
    return th.Thermo(
        th.DensityModel("ideal_gas", pressure=1.0),
        th.PropertyLaw("sutherland", 2.0e-3, (0.37,)),
        th.PropertyLaw("prandtl", 0.71),
        th.PropertyLaw("polynomial", 1.0, (0.8, 0.2)),
    )


def _cavity():
    return {
        "left": {"type": "wall", "temperature": 1.4},
        "right": {"type": "wall", "temperature": 0.6},
        "top": {"type": "wall", "velocity": (0.3, 0.0)},
        "bottom": {"type": "wall", "heat_flux": 0.05},
    }


def _model(closed_mass, turbulence="none", transition="none", **kw):
    return vd.VariableDensityModel(
        density=1.0,
        thermo=_thermo(),
        gravity=(0.0, -2.0),
        turbulence=turbulence,
        transition=transition,
        turbulence_intensity=0.5,
        limiter="venkatakrishnan",
        closed_mass=closed_mass,
        pressure_work=0.4 / 1.4,
        **kw,
    )


def _state(problem, seed=5):
    w = problem.initial_state((0.0, 0.0))
    rng = np.random.default_rng(seed)
    w = w * (1.0 + 0.2 * rng.uniform(-1.0, 1.0, w.shape))
    w[:, :3] += 0.1 * rng.standard_normal((w.shape[0], 3))
    w[:, -1] = 1.0 + 0.35 * rng.uniform(-1.0, 1.0, w.shape[0])
    return w


def _pair(turbulence="none", transition="none", mass=0.83):
    """A closed problem and the open problem at its p_th (same pin)."""
    mesh = rectangle(7, 6, cell_type="mixed", perturb=0.25, seed=2)
    closed = FlowProblem(mesh, _model(mass, turbulence, transition), _cavity())
    w = _state(closed)
    p_th = closed.thermodynamic_pressure(w)
    open_model = vd.at_pressure(closed.model._replace(closed_mass=None), p_th)
    opened = FlowProblem(mesh, open_model, _cavity())
    return closed, opened, w, p_th


def test_closed_domain_conserves_the_mass():
    closed, _, w, p_th = _pair()
    rho = np.asarray(th.density(vd.at_pressure(closed.model, p_th).thermo, w[:, -1]))
    total = float(np.sum(np.asarray(closed.geom.area) * rho))
    assert total == pytest.approx(0.83, rel=1e-13)
    assert p_th != pytest.approx(1.0)


@pytest.mark.parametrize("turbulence, transition", MODELS, ids=IDS)
def test_closed_residual_is_the_open_residual_at_p_th(turbulence, transition):
    """Residual, pseudo-time, source blocks and Jacobian use p_th of the state."""
    closed, opened, w, _ = _pair(turbulence, transition)
    beta2 = closed.beta2(w)
    np.testing.assert_allclose(
        closed.residual(w, beta2), opened.residual(w, beta2), rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(
        closed.time_term(w, beta2, 7.0).toarray(),
        opened.time_term(w, beta2, 7.0).toarray(),
        rtol=1e-12,
    )
    jc = closed.jacobian(w, beta2)
    jo = opened.jacobian(w, beta2)
    np.testing.assert_allclose(jc.toarray(), jo.toarray(), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        closed.wall_heat_flux(w), opened.wall_heat_flux(w), rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        closed.kinematic_viscosity(w), opened.kinematic_viscosity(w), rtol=1e-13
    )
    if turbulence != "none":
        np.testing.assert_allclose(
            closed.eddy_viscosity(w), opened.eddy_viscosity(w), rtol=1e-12
        )


def test_newton_product_contains_the_global_pressure_coupling():
    """The JVP differentiates p_th(T): it matches a central difference."""
    closed, opened, w, _ = _pair()
    beta2 = closed.beta2(w)
    rng = np.random.default_rng(3)
    dw = np.zeros_like(w)
    dw[:, -1] = rng.standard_normal(w.shape[0])
    jv = closed.jvp(w, dw, beta2)
    h = 1e-6
    fd = (closed.residual(w + h * dw, beta2) - closed.residual(w - h * dw, beta2)) / (
        2 * h
    )
    np.testing.assert_allclose(jv, fd, rtol=1e-6, atol=1e-8)
    # the open problem at the same p_th misses the global term
    jo = opened.jvp(w, dw, beta2)
    assert np.max(np.abs(jo - fd)) > 1e-4 * np.max(np.abs(fd))


@pytest.mark.parametrize("turbulence, transition", MODELS[:4], ids=IDS[:4])
def test_conservative_time_term(turbulence, transition):
    """Q(w): area (rho, rho u, rho v, rho phi, rho h - c p_th), exact derivatives."""
    closed, _, w, p_th = _pair(turbulence, transition)
    mass = closed.mass()
    assert isinstance(mass, ConservativeMass)
    q = mass.q(w)
    area = np.asarray(closed.geom.area)
    rho = p_th / w[:, -1]
    cp = closed.model.thermo.specific_heat
    h = np.asarray(th.integral(cp, jnp.asarray(w[:, -1])))
    scale = closed.time_row_scale()
    expected = area[:, None] * rho[:, None] * w * scale
    expected[:, 0] = area * rho
    expected[:, -1] = area * (rho * h - closed.model.pressure_work * p_th)
    expected[closed.model.pin[0], 0] = 0.0
    np.testing.assert_allclose(q, expected, rtol=1e-12, atol=1e-15)
    if turbulence == "sst":
        assert scale[closed.turbulence_column + 1] == pytest.approx(
            closed.model.viscosity / closed.model.density
        )
    rng = np.random.default_rng(0)
    dw = rng.standard_normal(w.shape)
    eps = 1e-6
    fd = (mass.q(w + eps * dw) - mass.q(w - eps * dw)) / (2 * eps)
    np.testing.assert_allclose(mass.linearize(w)(dw), fd, rtol=1e-6, atol=1e-10)
    # the blocks are the cell-local part: exact for changes that keep T
    blocks = mass.jacobian(w)
    du = np.zeros_like(w)
    du[:, 1:3] = rng.standard_normal((w.shape[0], 2))
    np.testing.assert_allclose(
        (blocks @ du.ravel()).reshape(w.shape), mass.linearize(w)(du), rtol=1e-12
    )


def test_heated_closed_box_energy_balance():
    """Heat into a closed box raises p_th: (1 - c) A dp_th/dt = Q (BDF1).

    Constant c_p = 1 (h = T - 1, so sum A rho h = p_th A - M_0): the energy
    rows sum to the wall heat flux at the new time level, so the discrete
    balance holds to round-off, with the dp_th/dt work c of the low-Mach
    energy equation; the mass stays M_0.
    """
    mesh = rectangle(6, 6)
    thermo = th.Thermo(
        th.DensityModel("ideal_gas", pressure=1.0),
        th.PropertyLaw("constant", 1.0e-2),
        th.PropertyLaw("prandtl", 0.71),
        th.PropertyLaw("constant", 1.0),
    )
    c = 0.4 / 1.4
    model = vd.VariableDensityModel(
        density=1.0,
        thermo=thermo,
        gravity=(0.0, -1.0),
        closed_mass=1.0,
        pressure_work=c,
    )
    cond = {
        "left": {"type": "wall", "temperature": 1.2},
        "right": {"type": "wall", "heat_flux": 0.2},
        "top": {"type": "wall"},
        "bottom": {"type": "wall"},
    }
    prob = FlowProblem(mesh, model, cond)
    w0 = prob.initial_state((0.0, 0.0), temperature=1.0)
    dt = 0.05
    inner = PseudoTimeSettings(
        cfl_start=1.0e4,
        residual_drop=10.0,
        max_iterations=20,
        nk_switch_residual_drop=0.0,
    )
    res = prob.solve_unsteady(
        w0, DualTimeSettings(dt, 2 * dt, scheme="bdf1", inner=inner)
    )
    assert res.converged
    assert np.all(res.history["inner_iterations"] <= 8)
    length = np.asarray(prob.geom.b_length)
    previous = w0
    for w in (res.u_previous, res.u):
        dp = prob.thermodynamic_pressure(w) - prob.thermodynamic_pressure(previous)
        heat = float(np.sum(prob.wall_heat_flux(w) * length))
        assert heat > 0.1
        assert (1.0 - c) * dp / dt == pytest.approx(heat, rel=1e-10)
        previous = w
    rho = prob.thermodynamic_pressure(res.u) / res.u[:, -1]
    assert float(np.sum(prob.geom.area * rho)) == pytest.approx(1.0, rel=1e-12)


CASE = {
    "mesh": {"file": "m.su2"},
    "physics": {"model": "laminar", "energy": True, "gravity": [0.0, -9.81]},
    "fluid": {"density_model": "ideal_gas", "viscosity": 1.8e-5},
    "reference": {"velocity": 0.05, "length": 0.03, "temperature": 300.0},
    "initial": {"velocity": [0.0, 0.0]},
    "boundary": {
        "left": {"type": "wall", "temperature": 400.0},
        "right": {"type": "wall", "temperature": 200.0},
        "top": {"type": "wall"},
        "bottom": {"type": "symmetry"},
    },
}


def _setup(raw):
    from pyaeroflow.run import setup

    mesh = rectangle(4, 4, x=(0.0, 0.03), y=(0.0, 0.03))
    return setup(resolve(raw), mesh)


def test_thermodynamic_pressure_modes():
    """auto: closed without a mass-exchanging boundary; constant; rules."""
    problem, scaling, w0, _ = _setup(CASE)
    assert problem.closed
    area = 0.03**2 / 0.03**2
    assert problem.model.closed_mass == pytest.approx(area)
    assert problem.thermodynamic_pressure(w0) == pytest.approx(1.0)
    assert problem.model.pressure_work == pytest.approx(287.058 / scaling.specific_heat)
    hot = {**CASE, "initial": {"velocity": [0.0, 0.0], "temperature": 600.0}}
    problem, *_ = _setup(hot)
    assert problem.model.closed_mass == pytest.approx(0.5)  # rho(2 T_ref)
    const = {**CASE, "fluid": {**CASE["fluid"], "thermodynamic_pressure": "constant"}}
    assert not _setup(const)[0].closed
    outlet = {
        **CASE,
        "boundary": {
            **CASE["boundary"],
            "top": {"type": "pressure-outlet", "pressure": 0.0},
        },
    }
    assert not _setup(outlet)[0].closed
    forced = {
        **outlet,
        "fluid": {**CASE["fluid"], "thermodynamic_pressure": "closed"},
    }
    with pytest.raises(ConfigError, match="exchange mass"):
        resolve(forced)
    bous = {
        **CASE,
        "fluid": {
            "density_model": "boussinesq",
            "density": 1.2,
            "viscosity": 1.8e-5,
            "thermodynamic_pressure": "closed",
        },
    }
    with pytest.raises(ConfigError, match="needs density_model = ideal_gas"):
        resolve(bous)
    unsteady = {**CASE, "time": {"mode": "unsteady", "time_step": 0.1, "end_time": 1}}
    resolve(unsteady)  # stage V3: no longer rejected


def test_unsteady_mass_carries_the_sst_row_scaling():
    """Constant-density SST: the time term of omega has the residual's nu_ref.

    The omega row of the residual is scaled by nu_ref (rans_sst._row_scale);
    without the same factor on the mass the physical time derivative of
    omega would be weighted 1/nu_ref too strongly.
    """
    from pyaeroflow.physics import incompressible as inc
    from pyaeroflow.physics import rans_sst

    mesh = rectangle(4, 3)
    cond = {
        "bottom": {"type": "wall"},
        "top": {"type": "farfield", "velocity": (1.0, 0.0), "pressure": 0.0},
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.0)},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
    }
    for transition in ("none", "lm"):
        model = inc.IncompressibleModel(
            density=1.2,
            viscosity=1e-3,
            turbulence="sst",
            transition=transition,
            turbulence_intensity=0.5,
        )
        prob = FlowProblem(mesh, model, cond)
        mass = prob.mass()
        area = np.asarray(prob.geom.area)
        scale = np.asarray(rans_sst._row_scale(model))
        np.testing.assert_allclose(mass[:, 3:], area[:, None] * scale[None, :])
        np.testing.assert_allclose(mass[:, :3], area[:, None] * [0.0, 1.2, 1.2])


def test_extrapolated_start_keeps_the_temperature_positive():
    """The start of a time step changes T by at most 10 % per cell."""
    closed, _, w, _ = _pair()
    limit = closed._extrapolation_limiter()
    du = np.zeros_like(w)
    du[:, -1] = -0.8 * w[:, -1]  # would end at 0.2 T
    du[0, -1] = 0.01 * w[0, -1]  # small change: kept
    du[:, 1] = 0.3
    out = limit(w, du)
    np.testing.assert_allclose(out[1:, -1], -0.1 * w[1:, -1])
    np.testing.assert_allclose(out[1:, 1], 0.3 / 8.0)  # the whole cell scaled
    np.testing.assert_allclose(out[0], du[0])

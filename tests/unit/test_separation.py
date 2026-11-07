"""SST separation sensor (SST-lambda_theta, Griffin et al. 2026)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_rans import RANS_MINIMAL

from pyaeroflow.cases import tmr_flat_plate as tp
from pyaeroflow.config import resolve
from pyaeroflow.physics import separation, sst

jax.config.update("jax_enable_x64", True)


def _grad(dq_dn):
    """Velocity gradient of u = (1, 0) with d(v)/dy = dq_dn for n = (0, 1)."""
    g = np.zeros((1, 2, 2))
    g[0, 1, 1] = dq_dn
    return jnp.asarray(g)


def _a1(dq_dn, gamma=1.0, d=1.0, nu=1.0):
    return float(
        separation.a1_field(
            jnp.array([1.0]),
            jnp.array([0.0]),
            _grad(dq_dn),
            jnp.array([[0.0, 1.0]]),
            jnp.array([d]),
            jnp.array([nu]),
            sst.A1,
            0.275,
            gamma,
            -5.0e-4,
        )[0]
    )


def test_lambda_theta_formula_and_clip():
    f = separation.lambda_theta
    assert float(f(0.0)) == pytest.approx(0.0128)
    assert float(f(20.0)) == pytest.approx(-7.57e-3 * 20.0 + 0.0128)
    assert float(f(1e6)) == -1.0 and float(f(-1e6)) == 1.0


def test_reynolds_theta_uses_the_streamwise_velocity():
    n = jnp.array([[0.0, 1.0]])
    re = separation.reynolds_theta(
        jnp.array([3.0]), jnp.array([4.0]), n, jnp.array([2.0]), jnp.array([0.5])
    )
    assert float(re[0]) == pytest.approx(2.0 * 3.0 / 0.5)  # |u_s| = |u|, v is normal


def test_critical_value_limits():
    assert float(separation.critical_lambda(100.0, 0.0)) == pytest.approx(-0.0681)
    assert float(separation.critical_lambda(100.0, 1.0)) == pytest.approx(
        -5e-4 * 100.0**0.75
    )
    half = float(separation.critical_lambda(100.0, 0.5))
    assert half == pytest.approx(0.5 * (-0.0681 - 5e-4 * 100.0**0.75))


def test_switch_turbulent_and_laminar_criteria():
    # Re_theta = 1: lambda_c = -5e-4 (turbulent). lambda = 0.0128 - 7.57e-3 dq.
    assert _a1(0.0) == sst.A1  # zero pressure gradient: off
    assert _a1(1.0) == sst.A1  # lambda = 0.0052 > -5e-4
    assert _a1(2.0) == 0.275  # lambda = -0.0023 < -5e-4
    # Laminar criterion (gamma = 0): switch at lambda = -0.0681, dq ~ 10.7.
    assert _a1(8.0, gamma=0.0) == sst.A1
    assert _a1(12.0, gamma=0.0) == 0.275


def test_clip_makes_the_turbulent_branch_local_to_low_re_theta():
    # Large Re_theta: Gamma Re^(3/4) < -1 <= lambda: never active (free stream, wake).
    assert _a1(1e5, d=1e6) == sst.A1


def test_eddy_viscosity_with_a1_prime():
    k, omega = 1e-3, 10.0
    # Branch a1 omega > S F2: unchanged, whatever a1'.
    assert float(sst.eddy_viscosity(k, omega, 1.0, 1.0, 0.275)) == pytest.approx(
        k / omega
    )
    # Branch a1 omega <= S F2: nu_t = a1' k / (S F2).
    assert float(sst.eddy_viscosity(k, omega, 100.0, 1.0, 0.275)) == pytest.approx(
        0.275 * k / 100.0
    )
    # a1' = a1 reproduces the standard model, also in the sources.
    base = sst.sources(k, omega, 4.0, 0.7, 0.9, 0.3)
    same = sst.sources(k, omega, 4.0, 0.7, 0.9, 0.3, a1=sst.A1)
    np.testing.assert_allclose(base, same, rtol=1e-14)
    low = sst.sources(k, omega, 4.0, 0.7, 0.9, 0.3, a1=0.275)
    assert float(low[0]) < float(base[0])  # less production, same destruction


def _plate(separation_model, **kw):
    return tp.solve(
        12,
        4,
        12,
        reynolds=1.0e5,
        turbulence="sst",
        settings=tp.PseudoTimeSettings(max_iterations=3),
        separation=separation_model,
        **kw,
    ).problem


def test_zero_pressure_gradient_state_is_unchanged_and_jvp_exact():
    off, on = _plate("none"), _plate("lambda-theta")
    w = off.initial_state((1.0, 0.0))
    rng = np.random.default_rng(2)
    w = w * (1.0 + 0.1 * rng.random(w.shape))
    w[:, 1] = np.minimum(1.0, np.asarray(off.geom.wall_distance) / 0.01)
    b2 = off.beta2(w)
    lam, lam_c, a1 = on.separation_fields(w)
    assert np.all(np.isfinite(lam)) and np.all(np.isfinite(lam_c))
    # The sensor fires only close to the plate in the developing profile; where
    # it does not, the residual is the standard one.
    r_off, r_on = off.residual(w, b2), on.residual(w, b2)
    quiet = a1 == sst.A1
    np.testing.assert_allclose(r_on[quiet], r_off[quiet], rtol=1e-12, atol=1e-14)
    # JVP with respect to k and omega (the sensor depends on u, v only): exact.
    dw = np.zeros_like(w)
    dw[:, 3:] = rng.standard_normal(w[:, 3:].shape) * np.abs(w[:, 3:]) * 1e-3
    exact = on.jvp(w, dw, b2)
    eps = 1e-4
    fd = (on.residual(w + eps * dw, b2) - on.residual(w - eps * dw, b2)) / (2 * eps)
    for col in (3, 4):
        scale = np.abs(fd[:, col]).max()
        np.testing.assert_allclose(exact[:, col], fd[:, col], atol=1e-3 * scale)


def test_sensor_changes_the_residual_where_it_fires():
    on = _plate("lambda-theta")
    off = _plate("none")
    w = off.initial_state((1.0, 0.0))
    d = np.asarray(off.geom.wall_distance)
    # A strongly decelerating wall-normal profile: v grows linearly off the wall.
    w[:, 2] = 50.0 * d * (d < 0.01)
    w[:, 3] = 1e-4
    w[:, 4] = 1e2
    b2 = off.beta2(w)
    _, _, a1 = on.separation_fields(w)
    assert (a1 < sst.A1).any()
    assert np.abs(on.residual(w, b2) - off.residual(w, b2)).max() > 0.0
    # The off model reports the diagnostics with a1' = a1 everywhere.
    assert np.all(off.separation_fields(w)[2] == sst.A1)


def test_compressible_sst_with_separation_runs():
    p = tp.solve(
        12,
        4,
        12,
        reynolds=1.0e5,
        turbulence="sst",
        mach=0.2,
        settings=tp.PseudoTimeSettings(max_iterations=2),
        separation="lambda-theta",
    ).problem
    assert p.model.separation == "lambda-theta"
    w = p.initial_state((1.0, 0.0))
    w[:, 0] = p.model.gas_constant  # rho = T = 1
    lam, lam_c, a1 = p.separation_fields(w)
    assert np.all(np.isfinite(lam)) and np.all(np.isfinite(lam_c))
    assert np.all(a1 <= sst.A1)
    assert np.all(np.isfinite(p.residual(w, p.beta2(w))))


def test_config_rules():
    sst_physics = {"model": "rans", "turbulence": "sst"}
    cfg = resolve(
        {
            **RANS_MINIMAL,
            "physics": {**sst_physics, "separation": {"model": "lambda-theta"}},
        }
    )
    sep = cfg["physics"]["separation"]
    assert sep["model"] == "lambda-theta"
    assert sep["a1_apg"] == 0.275 and sep["gamma"] == -5.0e-4
    assert (
        resolve({**RANS_MINIMAL, "physics": sst_physics})["physics"]["separation"][
            "model"
        ]
        == "none"
    )
    raw = {
        **RANS_MINIMAL,
        "physics": {"model": "rans", "separation": {"model": "lambda-theta"}},
    }
    with pytest.raises(Exception, match="needs physics.model = rans with turbulence"):
        resolve(raw)

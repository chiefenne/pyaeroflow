"""Langtry-Menter gamma-Re_theta on SST (SST-2003m-LM2009, L3): functions, residual."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_rans import RANS_MINIMAL

from pyaeroflow.cases import tmr_flat_plate as tp
from pyaeroflow.config import resolve
from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import lm, rans_sst, sst
from pyaeroflow.run import _mode_rows, load_restart, mesh_fingerprint

jax.config.update("jax_enable_x64", True)


def test_correlations_hand_values():
    # Re_thetat^eq without pressure gradient (TMR): Tu 0.1 % and 3 %.
    assert float(lm.freestream_re_theta(0.1)) == pytest.approx(
        1173.51 - 58.9428 + 21.96
    )
    assert float(lm.freestream_re_theta(3.0)) == pytest.approx(
        331.5 * (3.0 - 0.5658) ** -0.671
    )
    # The Tu limit 0.027 %.
    assert float(lm.freestream_re_theta(0.001)) == pytest.approx(
        float(lm.freestream_re_theta(0.027))
    )
    # Re_thetac: polynomial below 1870, linear above (continuous there).
    r = 300.0
    poly = (
        -3.96035 + 1.0120656 * r - 868.230e-6 * r**2 + 696.506e-9 * r**3
    ) - 174.105e-12 * r**4
    assert float(lm.critical_re_theta(r)) == pytest.approx(poly)
    lo, hi = lm.critical_re_theta(1870.0 - 1e-9), lm.critical_re_theta(1870.0 + 1e-9)
    assert float(lo) == pytest.approx(float(hi), rel=2e-3)
    # F_length: the four ranges.
    assert float(lm.length_function(100.0)) == pytest.approx(
        39.8189 - 119.270e-4 * 100 - 132.567e-6 * 1e4
    )
    assert float(lm.length_function(800.0)) == pytest.approx(0.5 - 3e-4 * 204.0)
    assert float(lm.length_function(1500.0)) == pytest.approx(0.3188)


def test_equilibrium_re_theta_pressure_gradient():
    nu, speed, tu = 1e-5, 1.0, 0.5
    zpg = float(lm.equilibrium_re_theta(tu, 0.0, nu, speed))
    assert zpg == pytest.approx(float(lm.freestream_re_theta(tu)))
    # Favourable gradient raises Re_thetat, adverse lowers it.
    fav = float(lm.equilibrium_re_theta(tu, 0.05, nu, speed))
    adv = float(lm.equilibrium_re_theta(tu, -0.05, nu, speed))
    assert fav > zpg > adv >= lm.RE_THETA_EQ_MIN
    # The fixed point: re = base F(lambda(re)).
    lam = np.clip(fav**2 * nu * 0.05 / speed**2, -0.1, 0.1)
    base = float(lm.freestream_re_theta(tu))
    assert fav == pytest.approx(
        base * float(lm.pressure_gradient_factor(lam, tu)), rel=1e-6
    )


def test_sources_and_coupling_limits():
    nu = 1e-5
    # Freestream (no strain, gamma = 1): no gamma source; Re_thetat relaxes to
    # the equilibrium value.
    k, omega, d = 1.5e-4, 10.0, 1.0
    s_g, s_t = lm.sources(1.0, 100.0, k, omega, nu, 0.0, 0.0, d, 1.0, 0.0)
    assert float(s_g) == pytest.approx(0.0, abs=1e-12)
    tu = float(lm.turbulence_intensity(k, 1.0))
    assert tu == pytest.approx(1.0)
    re_eq = float(lm.freestream_re_theta(tu))
    assert float(s_t) == pytest.approx(lm.C_THETA / (500 * nu) * (re_eq - 100.0))
    # Laminar boundary layer (R_T small, Re_V below onset): gamma destroyed to
    # 1/c_e2, so gamma = 1/50 is a zero of the source.
    args = (1e-8, 1.0, nu, 10.0, 10.0, 1e-4, 0.5, 0.0)
    s_g, _ = lm.sources(1.0 / lm.CE2, 1000.0, *args)
    assert float(s_g) == pytest.approx(0.0, abs=1e-10)
    s_g, _ = lm.sources(0.5, 1000.0, *args)
    assert float(s_g) < 0.0
    # gamma_eff on SST: P_k times gamma_eff, D_k times clip(gamma_eff, 0.1, 1).
    s_k, s_w = sst.sources(1.0, 10.0, 1.0, 1.0, 0.0, 0.0, gamma_eff=0.02)
    s_k0, s_w0 = sst.sources(1.0, 10.0, 1.0, 1.0, 0.0, 0.0)
    assert float(s_k) == pytest.approx(0.02 * 0.1 - 0.1 * 0.9)
    assert float(s_w) == pytest.approx(float(s_w0))  # omega unchanged
    assert float(s_k0) == pytest.approx(0.1 - 0.9)
    # F3 keeps F1 = 1 in the viscous sublayer.
    assert float(lm.blending_f3(1e-6, nu, 1e-4)) == pytest.approx(1.0)
    assert float(lm.blending_f3(1e-2, nu, 0.1)) < 1e-12


def test_separation_intermittency():
    nu = 1e-5
    # Large Re_V (separated shear layer) with low R_T: gamma_sep up to 2.
    g = lm.separation_intermittency(0.02, 200.0, 1e-8, 1.0, nu, 1e3, 1e3, 1e-2, 0.5)
    assert 1.0 < float(g) <= 2.0
    # Turbulent (R_T large): F_reattach switches it off.
    g = lm.separation_intermittency(0.02, 200.0, 1.0, 1.0, nu, 1e3, 1e3, 1e-2, 0.5)
    assert float(g) == pytest.approx(0.0, abs=1e-12)


def _lm_plate(n=(12, 4, 12)):
    k = 1.5 * 0.01**2  # Tu 1 %
    mesh = tp.plate_mesh(*n, reynolds=1e5)
    turb = {"k": k, "omega": k / (10.0 * 1e-5)}
    cond = {
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.0), **turb},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
        "top": {"type": "farfield", "velocity": (1.0, 0.0), "pressure": 0.0, **turb},
        "symmetry": {"type": "symmetry"},
        "plate": {"type": "wall"},
    }
    model = inc.IncompressibleModel(viscosity=1e-5, turbulence="sst", transition="lm")
    return IncompressibleProblem(mesh, model, cond), (k, turb["omega"])


def test_inflow_values_initial_state_and_jvp():
    p, k_omega = _lm_plate()
    assert p.model.n_var == 7
    turb = np.asarray(p.bc.turbulence)
    left = p.boundary_faces(["left"])
    re_inf = float(lm.freestream_re_theta(1.0))
    np.testing.assert_allclose(turb[left, 2], 1.0)
    np.testing.assert_allclose(turb[left, 3], re_inf, rtol=1e-9)
    w = p.initial_state((1.0, 0.0), k_omega=k_omega)
    assert np.all(w[:, rans_sst.NGL] == 1.0)
    assert w[0, rans_sst.NRT] == pytest.approx(re_inf)
    # Walls: zero gradient of gamma and Re_thetat, prescribed k and omega.
    faces = p.boundary_faces(["plate"])
    w_in = w[np.asarray(p.geom.b_owner)[faces]] * 1.3
    w_b = rans_sst.boundary_state(
        p.bc._replace(
            kind=p.bc.kind[faces],
            value=p.bc.value[faces],
            turbulence=p.bc.turbulence[faces],
        ),
        p.geom.b_normal[faces],
        jnp.asarray(w_in),
    )
    np.testing.assert_allclose(np.asarray(w_b)[:, 5:], w_in[:, 5:])
    np.testing.assert_allclose(np.asarray(w_b)[:, 3], 0.0)
    # JVP against central differences on a perturbed developed-like state.
    rng = np.random.default_rng(2)
    w[:, 1] = np.minimum(1.0, np.asarray(p.geom.wall_distance) / 0.01)
    w = w * (1.0 + 0.1 * rng.random(w.shape))
    w[:, rans_sst.NGL] = 0.3 + 0.5 * rng.random(len(w))
    b2 = p.beta2(w)
    dw = rng.standard_normal(w.shape) * np.abs(w) * 1e-3
    exact = p.jvp(w, dw, b2)
    eps = 1e-4
    fd = (p.residual(w + eps * dw, b2) - p.residual(w - eps * dw, b2)) / (2 * eps)
    for col in range(7):
        scale = np.abs(fd[:, col]).max()
        np.testing.assert_allclose(exact[:, col], fd[:, col], atol=2e-3 * scale)
    # The first-order Jacobian assembles with 7 x 7 blocks.
    jac = p.jacobian(w, b2)
    assert jac.shape == (7 * len(w), 7 * len(w))


def test_step_limiter_groups():
    p, k_omega = _lm_plate()
    w = p.initial_state((1.0, 0.0), k_omega=k_omega)
    du = np.zeros_like(w)
    du[:, rans_sst.NGL] = -2.0 * w[:, rans_sst.NGL]
    du[:, rans_sst.NK] = 0.5 * w[:, rans_sst.NK]
    out = p._step_limiter(None)(w, du)
    np.testing.assert_allclose(out[:, rans_sst.NGL], -0.99 * w[:, rans_sst.NGL])
    np.testing.assert_allclose(out[:, rans_sst.NK], du[:, rans_sst.NK])  # apart


def test_step_limiter_clips_gamma_and_re_theta_like_su2():
    p, k_omega = _lm_plate()
    w = p.initial_state((1.0, 0.0), k_omega=k_omega)
    n = len(w)
    w[: n // 3, rans_sst.NGL] = 1e-142  # a cell that lost 99 % per iteration
    w[n // 3 :, rans_sst.NGL] = 4.9
    du = np.zeros_like(w)
    du[: n // 3, rans_sst.NGL] = -1e-60  # asks for a negative gamma
    du[n // 3 :, rans_sst.NGL] = 3.0  # asks for gamma = 7.9
    du[:, rans_sst.NRT] = -2.0 * w[:, rans_sst.NRT]
    out = w + p._step_limiter(None)(w, du)
    assert out[:, rans_sst.NGL].min() == pytest.approx(lm.GAMMA_LIMITS[0])
    assert out[:, rans_sst.NGL].max() == pytest.approx(lm.GAMMA_LIMITS[1])
    assert np.all(out[:, rans_sst.NRT] > 0.0)  # 99 % limit, then the floor
    np.testing.assert_allclose(out[:, rans_sst.NK], w[:, rans_sst.NK])


def test_lm_needs_sst_and_config_rules():
    mesh = tp.plate_mesh(6, 2, 6, reynolds=1e5)
    cond = {
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.0)},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
        "top": {"type": "farfield", "velocity": (1.0, 0.0), "pressure": 0.0},
        "symmetry": {"type": "symmetry"},
        "plate": {"type": "wall"},
    }
    model = inc.IncompressibleModel(
        viscosity=1e-5, turbulence="sa-neg", transition="lm"
    )
    with pytest.raises(ValueError, match="needs SST"):
        IncompressibleProblem(mesh, model, cond)
    raw = {**RANS_MINIMAL, "physics": {"model": "rans", "transition": {"model": "lm"}}}
    with pytest.raises(Exception, match="lm needs turbulence = sst"):
        resolve(raw)
    raw = {
        **RANS_MINIMAL,
        "physics": {
            "model": "rans",
            "turbulence": "sst",
            "transition": {"model": "lm", "turbulence_intensity": 0.01},
        },
    }
    cfg = resolve(raw)
    assert cfg["physics"]["transition"]["model"] == "lm"
    assert any("turbulence_intensity ignored" in n for n in cfg.notices)


def test_configuration_block_names_the_transition_clip():
    raw = {
        **RANS_MINIMAL,
        "physics": {"model": "rans", "turbulence": "sst", "transition": {"model": "lm"}},
    }
    rows = dict(_mode_rows(resolve(raw)))
    assert rows["gamma, Re_thetat"].startswith("clipped as in SU2: gamma [0.0001, 5]")
    raw["physics"]["transition"]["clip"] = False
    rows = dict(_mode_rows(resolve(raw)))
    assert rows["gamma, Re_thetat"].startswith("not clipped")


def test_restart_between_sst_and_sst_lm_keeps_k_omega(tmp_path):
    mesh = tp.plate_mesh(6, 2, 6, reynolds=1e5)
    np.savez(
        tmp_path / "r.npz",
        w=np.full((mesh.n_cells, 5), 7.0),
        fingerprint=mesh_fingerprint(mesh),
    )
    w0, n = load_restart(tmp_path / "r.npz", mesh, np.zeros((mesh.n_cells, 7)))
    assert n == 5 and np.all(w0[:, :5] == 7.0) and np.all(w0[:, 5:] == 0.0)
    w0, n = load_restart(tmp_path / "r.npz", mesh, np.zeros((mesh.n_cells, 6)))
    assert n == 3


def test_step_limiter_without_clip_leaves_gamma_unclipped():
    p, k_omega = _lm_plate()
    w = p.initial_state((1.0, 0.0), k_omega=k_omega)
    w[:, rans_sst.NGL] = 1e-142
    du = np.zeros_like(w)
    du[:, rans_sst.NGL] = -1e-60
    clipped = w + p._step_limiter(None, True)(w, du)
    free = w + p._step_limiter(None, False)(w, du)
    assert clipped[:, rans_sst.NGL].min() == pytest.approx(lm.GAMMA_LIMITS[0])
    assert free[:, rans_sst.NGL].max() < 1e-142  # the 99 % limit only

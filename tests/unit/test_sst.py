"""k-omega SST (SST-2003m, L2): model functions, residual, driver, case files."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_rans import PLATE_TOML, RANS_MINIMAL

from pyaeroflow.cases import tmr_flat_plate as tp
from pyaeroflow.config import resolve
from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import write_su2
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import rans_sst, sst
from pyaeroflow.run import load_restart, mesh_fingerprint, run_case, turbulence_values

jax.config.update("jax_enable_x64", True)


def test_constants_are_those_of_su2():
    assert sst.GAMMA_1 == pytest.approx(0.5532, abs=1e-4)
    assert sst.GAMMA_2 == pytest.approx(0.4404, abs=1e-4)
    assert sst.blend(1.0, sst.BETA_1, sst.BETA_2) == sst.BETA_1
    assert sst.blend(0.0, sst.SIGMA_W1, sst.SIGMA_W2) == sst.SIGMA_W2


def test_blending_functions_limits():
    nu = 1e-5
    # Next to a wall (d small, omega large): k-omega, F1 = F2 = 1.
    f1, f2 = sst.blending(1e-4, 1e5, nu, 1e-5, 0.0)
    assert float(f1) == pytest.approx(1.0) and float(f2) == pytest.approx(1.0)
    # Freestream far from walls: k-epsilon, F1 = F2 = 0.
    f1, f2 = sst.blending(1e-6, 1.0, nu, 50.0, 0.0)
    assert float(f1) < 1e-6 and float(f2) < 1e-6


def test_eddy_viscosity_limiter():
    k, omega = 1e-3, 10.0
    assert float(sst.eddy_viscosity(k, omega, 1.0, 1.0)) == pytest.approx(k / omega)
    # S F2 > a1 omega: nu_t = a1 k / (S F2).
    big = float(sst.eddy_viscosity(k, omega, 100.0, 1.0))
    assert big == pytest.approx(sst.A1 * k / 100.0)
    assert float(sst.eddy_viscosity(-1e-3, omega, 1.0, 1.0)) == 0.0


def test_sources_hand_values():
    # k = 1, omega = 10, S = 1, F1 = 1, F2 = 0: nu_t = 0.1, P_k = 0.1.
    s_k, s_w = sst.sources(1.0, 10.0, 1.0, 1.0, 0.0, 0.0)
    assert float(s_k) == pytest.approx(0.1 - 0.09 * 10.0)
    assert float(s_w) == pytest.approx(sst.GAMMA_1 - 0.075 * 100.0)
    # Production limiter: nu_t S^2 = 1000 > 10 beta* k omega = 9.
    s_k, s_w = sst.sources(1.0, 10.0, 100.0, 1.0, 0.0, 0.0)
    assert float(s_k) == pytest.approx(9.0 - 0.9)
    assert float(s_w) == pytest.approx(sst.GAMMA_1 * 90.0 - 7.5)
    # Cross diffusion acts in the outer layer only; no production without k.
    _, cd_out = sst.sources(1.0, 10.0, 0.0, 0.0, 0.0, 2.0)
    _, cd_in = sst.sources(1.0, 10.0, 0.0, 1.0, 0.0, 2.0)
    assert float(cd_out) - float(cd_in) == pytest.approx(
        2 * sst.SIGMA_W2 * 2.0 / 10.0 - (0.0828 - 0.075) * 100.0
    )
    _, s_w = sst.sources(0.0, 10.0, 1.0, 1.0, 0.0, 0.0)
    assert float(s_w) == pytest.approx(-0.075 * 100.0)
    # Laminar regions switch the production off.
    s_k, _ = sst.sources(1.0, 10.0, 1.0, 1.0, 0.0, 0.0, production_switch=0.0)
    assert float(s_k) == pytest.approx(-0.9)
    # Finite derivatives at k = 0.
    g = jax.grad(lambda k: sum(sst.sources(k, 10.0, 1.0, 0.5, 0.5, 0.0)))(0.0)
    assert np.isfinite(float(g))


def _plate_problem(n=(12, 4, 12)):
    return tp.solve(
        *n,
        reynolds=1.0e5,
        turbulence="sst",
        settings=tp.PseudoTimeSettings(max_iterations=3),
    ).problem


def test_wall_values_initial_state_and_jvp():
    p = _plate_problem()
    faces = p.boundary_faces(["plate"])
    turb = np.asarray(p.bc.turbulence)[faces]
    owner = np.asarray(p.geom.b_owner)[faces]
    d1 = np.asarray(p.geom.wall_distance)[owner]
    np.testing.assert_allclose(turb[:, 0], 0.0)
    np.testing.assert_allclose(
        turb[:, 1], 60.0 * 1e-5 / (0.075 * (2 * d1) ** 2), rtol=1e-6
    )
    w = p.initial_state((1.0, 0.0))
    assert w.shape[1] == 5
    assert w[0, 3] == pytest.approx(1.5e-6)  # Tu = 0.1 %
    assert w[0, 4] == pytest.approx(1.5e-6 / 1e-5)  # nu_t / nu = 1
    # A developed state: perturb the start so that gradients are non-trivial.
    rng = np.random.default_rng(1)
    w = w * (1.0 + 0.1 * rng.random(w.shape))
    b2 = p.beta2(w)
    dw = rng.standard_normal(w.shape) * np.abs(w) * 1e-3
    exact = p.jvp(w, dw, b2)
    eps = 1e-4
    fd = (p.residual(w + eps * dw, b2) - p.residual(w - eps * dw, b2)) / (2 * eps)
    for col in range(5):
        scale = np.abs(fd[:, col]).max()
        np.testing.assert_allclose(exact[:, col], fd[:, col], atol=1e-3 * scale)


def test_first_order_jacobian_matches_residual_diagonal():
    """The compact Jacobian's turbulence diagonal has the residual's magnitude.

    It uses the residual's face viscosities (SU2 practice); a k / omega
    estimate is orders of magnitude off in a laminar boundary layer.
    """
    p = _plate_problem()
    w = p.initial_state((1.0, 0.0))
    w[:, 1] = np.minimum(1.0, np.asarray(p.geom.wall_distance) / 0.01)
    b2 = p.beta2(w)
    jac = p.jacobian(w, b2).tocsr()
    r0 = p.residual(w, b2, order=1)
    cell = int(np.argmin(np.asarray(p.geom.wall_distance)))
    for col in (3, 4):
        e = np.zeros_like(w)
        e[cell, col] = 1e-6 * w[cell, col]
        fd = (p.residual(w + e, b2, order=1) - r0)[cell, col] / e[cell, col]
        j = jac[cell * 5 + col, cell * 5 + col]
        assert 1.0 / 3.0 < j / fd < 3.0


def test_step_limiter_keeps_k_and_omega_positive():
    p = _plate_problem()
    w = p.initial_state((1.0, 0.0))
    du = np.zeros_like(w)
    du[:, 3] = -10.0 * w[:, 3]
    du[:, 4] = 0.5 * w[:, 4]
    du[:, 1] = 1.0
    out = p._step_limiter(None)(w, du)
    assert np.all(w[:, 3:] + out[:, 3:] > 0.0)
    np.testing.assert_allclose(out[:, 3], -0.99 * w[:, 3])
    np.testing.assert_allclose(out[:, 4], 0.099 * 0.5 * w[:, 4])  # same cell factor
    np.testing.assert_allclose(out[:, 1], 1.0)  # mean flow untouched


def test_transition_models_need_sa():
    mesh = tp.plate_mesh(6, 2, 6, reynolds=1e5)
    cond = {
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.0)},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
        "top": {"type": "farfield", "velocity": (1.0, 0.0), "pressure": 0.0},
        "symmetry": {"type": "symmetry"},
        "plate": {"type": "wall"},
    }
    model = inc.IncompressibleModel(viscosity=1e-5, turbulence="sst", transition="aft")
    with pytest.raises(ValueError, match="SA only"):
        IncompressibleProblem(mesh, model, cond)


def test_sst_turbulence_values():
    tv = turbulence_values(
        {"turbulent_viscosity_ratio": 10.0, "intensity": 0.01}, 1.5e-5, 20.0, "sst"
    )
    assert tv.kinetic_energy == pytest.approx(1.5 * 0.2**2)
    assert tv.omega == pytest.approx(0.06 / 1.5e-4)
    assert tv.chi is None and "omega =" in tv.report()
    tv = turbulence_values(None, 1.5e-5, 20.0, "sst")
    assert tv.intensity == 0.001 and tv.viscosity_ratio == 1.0
    tv = turbulence_values(
        {"kinetic_energy": 0.06, "length_scale": 0.01}, 1.5e-5, 20.0, "sst"
    )
    assert tv.omega == pytest.approx(0.06 / (tv.viscosity_ratio * 1.5e-5))


def test_sst_config_rules():
    sst_physics = {"model": "rans", "turbulence": "sst"}
    cfg = resolve({**RANS_MINIMAL, "physics": sst_physics})
    assert cfg["physics"]["turbulence"] == "sst"
    raw = {
        **RANS_MINIMAL,
        "physics": {**sst_physics, "transition": {"model": "aft"}},
    }
    with pytest.raises(Exception, match="needs turbulence = sa-neg"):
        resolve(raw)
    raw = {
        **RANS_MINIMAL,
        "physics": sst_physics,
        "initial": {"velocity": [1.0, 0.0], "turbulence": {"nu_tilde_ratio": 3}},
    }
    with pytest.raises(Exception, match="SA only"):
        resolve(raw)
    raw["initial"]["turbulence"] = {"turbulent_viscosity_ratio": 3}
    with pytest.raises(Exception, match="intensity is required with turbulence"):
        resolve(raw)
    raw = {
        **RANS_MINIMAL,
        "physics": sst_physics,
        "solver": {"nu_tilde_step_limit": 1.0},
    }
    assert any("nu_tilde_step_limit ignored" in n for n in resolve(raw).notices)


def test_restart_between_sa_and_sst_keeps_the_mean_flow(tmp_path):
    mesh = tp.plate_mesh(6, 2, 6, reynolds=1e5)
    w_sa = np.full((mesh.n_cells, 4), 7.0)
    np.savez(tmp_path / "r.npz", w=w_sa, fingerprint=mesh_fingerprint(mesh))
    default = np.zeros((mesh.n_cells, 5))
    w0, n = load_restart(tmp_path / "r.npz", mesh, default)
    assert n == 3 and np.all(w0[:, :3] == 7.0) and np.all(w0[:, 3:] == 0.0)


SST_PLATE_TOML = (
    PLATE_TOML.replace(
        '[physics.transition]\nmode = "prescribed"\n'
        'laminar_region = [{ boundary = "plate", side = "upper", x_max = 0.05 }]\n',
        "",
    )
    .replace('model = "rans"\n', 'model = "rans"\nturbulence = "sst"\n')
    .replace(
        "[output]\n",
        '[output]\nfields = ["k", "omega", "blending_f1", "mu_t", "residual_omega"]\n',
    )
)


def test_run_case_sst(tmp_path):
    write_su2(tmp_path / "plate.su2", tp.plate_mesh(24, 6, 24, reynolds=1.0e5))
    (tmp_path / "plate.toml").write_text(SST_PLATE_TOML)
    result = run_case(tmp_path / "plate.toml", console=False)
    assert result.converged and result.w.shape[1] == 5
    assert np.all(result.w[:, 3] >= 0.0) and np.all(result.w[:, 4] > 0.0)
    log = (tmp_path / "out" / "plate.log").read_text()
    assert "omega =" in log and "log(R_om)" in log and "nu_tilde/nu" not in log
    vtu = (tmp_path / "out" / "plate.vtu").read_text(errors="ignore")
    for name in ("k", "omega", "blending_f1", "mu_t", "residual_omega"):
        assert f'Name="{name}"' in vtu
    # The turbulent boundary layer has developed: nu_t/nu well above 1.
    nu_t = result.w[:, 3] / result.w[:, 4]
    assert nu_t.max() / (10.0 * 1.0 / 1.0e5) > 10.0


def test_jit_functions_accept_jax_arrays():
    p = _plate_problem()
    w = jnp.asarray(p.initial_state((1.0, 0.0)))
    nu_t = rans_sst.eddy_viscosity(p.model, p.geom, p.bc, w)
    assert nu_t.shape == (p.mesh.n_cells,)

"""SA-AFT2019b transition model: correlations, residual, Jacobian, config."""

import jax
import numpy as np
import pytest
from test_rans import RANS_MINIMAL

from pyaeroflow.cases.tmr_flat_plate import plate_mesh
from pyaeroflow.config import resolve
from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.numerics.wall_distance import mesh_wall_distance, mesh_wall_normal
from pyaeroflow.physics import aft, rans
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import spalart_allmaras as sa


def test_critical_amplification_from_turbulence_intensity():
    """Mack with Drela's limit: 8.15 at Tu = 0.1 % (TMR page), bounded at high Tu."""
    assert aft.critical_amplification(0.1) == pytest.approx(8.15, abs=0.005)
    assert aft.critical_amplification(0.07) > aft.critical_amplification(0.15)
    assert aft.critical_amplification(100.0) == pytest.approx(
        -8.43 - 2.4 * np.log(0.025)
    )


def test_shape_factor_and_blasius_correlations():
    """H12 limits of 2019b, and the correlations at the Blasius H12 = 2.59."""
    np.testing.assert_allclose(
        aft.shape_factor(np.array([-10.0, 0.0, 100.0])), [2.2, 2.4, 20.0]
    )
    h = 2.59
    # log10 Re_theta0 = 0.7 tanh(14/1.59 - 9.24) + 2.492/1.59^0.43 + 0.62
    assert float(aft.critical_reynolds_theta(h)) == pytest.approx(237.06, rel=1e-4)
    assert float(aft.growth_rate(h)) == pytest.approx(0.010274, rel=1e-4)
    k_v = 1.0 / (0.4036 * h**2 - 2.5394 * h + 4.3273)
    assert float(aft.critical_reynolds_vorticity(h)) == pytest.approx(
        k_v * 237.06, rel=1e-4
    )
    d = 2.4 * h / (h - 1.0)
    l_h = (6.54 * h - 14.07) / h**2
    m = (0.058 * (h - 4.0) ** 2 / (h - 1.0) - 0.068) / l_h
    assert float(aft.growth_factor(h)) == pytest.approx(d * (1.0 + m) / 2.0 * l_h)
    # Growth stays positive down to the lower H12 limit.
    assert float(aft.growth_rate(aft.H12_MIN)) > 0.0
    assert float(aft.growth_factor(aft.H12_MIN)) > 0.0


def test_onset_turbulence_and_coupling_functions():
    # Laminar (R_T = 0): onset only once n exceeds N_crit, capped at 1 (2 - 1).
    assert float(aft.onset_function(4.0, 8.0, 0.0)) == 0.0
    assert float(aft.onset_function(12.0, 8.0, 0.0)) == pytest.approx(0.5)
    assert float(aft.onset_function(40.0, 8.0, 0.0)) == pytest.approx(1.0)
    # Turbulent (R_T >= 3.5): F_onset,3 vanishes.
    assert float(aft.onset_function(4.0, 8.0, 3.5)) == pytest.approx(0.5)
    assert float(aft.turbulence_function(0.0)) == 1.0
    assert float(aft.turbulence_function(4.0)) == pytest.approx(np.exp(-16.0))
    assert float(aft.ft2(0.0)) == 0.0
    assert float(aft.ft2(np.log(1.0 / aft.C3))) == pytest.approx(1.2 * (1.0 - 0.02))


def test_sources_fixed_points():
    """The gamma_tilde source vanishes at gamma = 1 when onset is complete and
    at gamma = 1/c_3 in a laminar layer; n grows only beyond Re_V0."""
    nu = 1e-6
    args = dict(nu=nu, nu_t=0.0, strain=1e3, vorticity=1e3, local_shape_factor=0.73)
    n_lam, g_lam = aft.sources(
        0.0, np.log(1.0 / aft.C3), distance=1e-6, n_crit=9.0, **args
    )
    assert float(g_lam) == pytest.approx(0.0, abs=1e-12)
    assert float(n_lam) == 0.0  # Re_V = 1e-3 far below Re_V0
    n_grow, _ = aft.sources(0.0, 0.0, distance=1e-3, n_crit=9.0, **args)
    assert float(n_grow) > 0.0  # Re_V = 1e3 above Re_V0 (about 518 at H12 = 2.59)
    _, g_turb = aft.sources(
        20.0,
        0.0,
        nu=nu,
        nu_t=100 * nu,
        strain=1e3,
        vorticity=1e3,
        distance=1e-3,
        local_shape_factor=0.73,
        n_crit=9.0,
    )
    assert float(g_turb) == pytest.approx(0.0, abs=1e-9)


def test_sa_source_with_given_ft2():
    """f_t2 given as a value: zero reproduces SA-noft2, c_t3 the chi -> 0 limit."""
    nu = 1e-6
    for nt in (1e-7, 1e-5, -1e-6):
        noft2 = sa.source(nt, nu, 1e3, 1e-3, 0.0, ft2=False)
        assert float(sa.source(nt, nu, 1e3, 1e-3, 0.0, ft2_value=0.0)) == pytest.approx(
            float(noft2)
        )
    neg = sa.source(-1e-6, nu, 1e3, 1e-3, 0.0)
    assert float(
        sa.source(-1e-6, nu, 1e3, 1e-3, 0.0, ft2_value=sa.CT3)
    ) == pytest.approx(float(neg))


def test_wall_normal_is_the_distance_gradient():
    mesh = plate_mesh(12, 4, 12, reynolds=1e5)
    faces = mesh.boundary_faces["plate"]
    d, nearest = mesh_wall_distance(mesh, faces)
    n = mesh_wall_normal(mesh, nearest)
    on_plate = mesh.cell_centroid[:, 0] > 0.0
    np.testing.assert_allclose(n[on_plate], [[0.0, 1.0]] * on_plate.sum(), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0)


def _aft_problem(n_crit=3.0):
    nu = 1e-6
    nt = 0.1 * nu
    conditions = {
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.0), "nu_tilde": nt},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
        "top": {
            "type": "farfield",
            "velocity": (1.0, 0.0),
            "pressure": 0.0,
            "nu_tilde": nt,
        },
        "symmetry": {"type": "symmetry"},
        "plate": {"type": "wall"},
    }
    model = inc.IncompressibleModel(
        density=1.0,
        viscosity=nu,
        turbulence="sa-neg",
        transition="aft",
        critical_amplification=n_crit,
    )
    return IncompressibleProblem(plate_mesh(16, 4, 16, reynolds=1e6), model, conditions)


def test_aft_residual_jvp_matches_finite_differences():
    """Six columns; the exact JVP (nothing frozen) agrees with central differences."""
    problem = _aft_problem()
    nu = problem.model.viscosity
    rng = np.random.default_rng(1)
    w = problem.initial_state((1.0, 0.0), nu_tilde=0.1 * nu)
    assert w.shape[1] == 6 and np.all(w[:, rans.NG :] == 0.0)
    w[:, :3] += 0.01 * rng.standard_normal((len(w), 3))
    w[:, rans.NT] = nu * (0.1 + rng.random(len(w)))
    w[:, rans.NG] = -3.0 * rng.random(len(w))
    w[:, rans.NN] = 6.0 * rng.random(len(w))
    beta2 = problem.beta2(w)
    r = problem.residual(w, beta2)
    assert r.shape == w.shape and np.all(np.isfinite(r))
    dw = rng.standard_normal(w.shape) * np.array([1e-2, 1e-2, 1e-2, nu, 0.1, 0.1])
    jv = problem.jvp(w, dw, beta2)
    eps = 1e-6
    fd = (
        problem.residual(w + eps * dw, beta2) - problem.residual(w - eps * dw, beta2)
    ) / (2.0 * eps)
    scale = np.abs(fd).max(axis=0)
    assert np.all(np.abs(jv - fd).max(axis=0) <= 1e-6 * scale)
    j = problem.jacobian(w, beta2)
    assert j.shape == (6 * len(w), 6 * len(w))
    assert np.allclose(problem.mass()[0, 3:], problem.geom.area[0])
    assert np.all(np.isfinite(problem.intermittency(w)))


def test_aft_config_rules():
    raw = {**RANS_MINIMAL, "physics": {"model": "rans", "transition": {"model": "aft"}}}
    with pytest.raises(Exception, match="or critical_amplification is set"):
        resolve(raw)
    raw["physics"]["transition"]["critical_amplification"] = 9.0
    assert resolve(raw)["physics"]["transition"]["critical_amplification"] == 9.0
    raw["physics"]["transition"]["relaxation"] = 0.5
    with pytest.raises(Exception, match="needs model = bcm"):
        resolve(raw)


def test_aft_source_is_differentiable_everywhere():
    """No NaN in the source derivatives at laminar, onset and turbulent states."""
    nu = 1e-6

    def f(n, g, nt):
        src = aft.sources(n, g, nu, sa.eddy_viscosity(nt, nu), 1e3, 1e3, 1e-3, 0.5, 8.0)
        return src[0] + src[1]

    for n, g, nt in ((0.0, 0.0, 0.1 * nu), (8.0, -2.0, nu), (20.0, -0.1, 50 * nu)):
        grads = jax.grad(f, argnums=(0, 1, 2))(n, g, nt)
        assert all(np.isfinite(float(x)) for x in grads)


def test_run_case_aft(tmp_path):
    """SA-AFT through the case file: log rows, residual columns, VTU fields,
    restart of a 6-column case from a 4-column (BCM) restart."""
    from test_rans import _bcm_plate

    result = _bcm_plate(tmp_path, "critical_amplification = 3.0\n", model="aft")
    assert result.converged
    log = (tmp_path / "out" / "plate.log").read_text()
    assert "aft (SA-AFT2019b), N_crit = 3 (given)" in log
    assert "SA-AFT recommends" not in log  # nu_tilde/nu = 0.1
    vtu = (tmp_path / "out" / "plate.vtu").read_text(errors="ignore")
    assert "amplification_factor" in vtu and "intermittency" in vtu
    history = np.genfromtxt(
        tmp_path / "out" / "plate.history.csv", delimiter=",", names=True
    )
    assert "gam" in "".join(history.dtype.names)

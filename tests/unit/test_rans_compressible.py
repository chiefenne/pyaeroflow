"""Compressible SA-neg RANS (L1 stage C3)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.config import ConfigError, resolve
from pyaeroflow.driver import FlowProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import compressible as comp
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import rans
from pyaeroflow.physics import rans_compressible as rc
from pyaeroflow.physics import spalart_allmaras as sa

jax.config.update("jax_enable_x64", True)

MU = 1.0e-3


def _box(mach=0.3, order=2, turb_order=1, mesh=None, wall_temperature=None):
    mesh = mesh or rectangle(6, 5, cell_type="mixed", perturb=0.3, seed=4)
    r_gas, ff = comp.freestream_conditions(mach, 2.0)
    ff = {**ff, "nu_tilde": 3.0 * MU}
    model = comp.CompressibleModel(
        gas_constant=r_gas,
        viscosity=MU,
        spatial_order=order,
        turbulence="sa-neg",
        turbulence_spatial_order=turb_order,
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


def _perturbed(prob, ff, seed, amount=0.05):
    rng = np.random.default_rng(seed)
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    w[:, rc.NT] = 3.0 * MU * (1.0 + 20.0 * rng.uniform(0.0, 1.0, w.shape[0]))
    return w * (1.0 + rng.uniform(-amount, amount, w.shape))


def test_model_has_five_variables_and_the_sa_module():
    prob, ff = _box()
    assert prob.model.n_var == 5
    assert prob.physics is rc
    assert prob.turbulence_column == rc.NT == 4
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    np.testing.assert_allclose(w[:, rc.NT], 3.0 * MU)


def test_boundary_values_of_nu_tilde():
    prob, ff = _box()
    w = _perturbed(prob, ff, 0)
    w_b = np.asarray(
        rc.boundary_state(
            prob.bc, prob.geom.b_normal, jnp.asarray(w)[prob.geom.b_owner]
        )
    )
    own = np.asarray(prob.geom.b_owner)
    for name, expected in (("bottom", 0.0), ("left", 3.0 * MU)):
        faces = prob.boundary_faces([name])
        np.testing.assert_allclose(w_b[faces, rc.NT], expected)
    faces = prob.boundary_faces(["right"])  # outflow: interior value
    np.testing.assert_allclose(w_b[faces, rc.NT], w[own[faces], rc.NT])


def test_turbulence_row_is_density_times_the_incompressible_row():
    """At uniform density (p and T constant) the compressible SA row is rho
    times the incompressible one; the mean flow then differs (Roe vs. FDS),
    so the velocity field is frozen and only the turbulence row compared.
    """
    mesh = rectangle(6, 5, cell_type="quad", perturb=0.2, seed=1)
    prob, ff = _box(mesh=mesh, order=1)
    rng = np.random.default_rng(5)
    w = prob.initial_state(ff["velocity"], ff["pressure"])
    w[:, rc.NT] = 3.0 * MU * (1.0 + 10.0 * rng.uniform(0.0, 1.0, w.shape[0]))
    w[:, comp.U] *= 1.0 + 0.2 * rng.uniform(-1.0, 1.0, w.shape[0])
    rho = 1.0
    # Incompressible problem with the same velocity and nu_tilde; nu = MU.
    model_i = inc.IncompressibleModel(density=rho, viscosity=MU, turbulence="sa-neg")
    ff_i = {
        "type": "farfield",
        "velocity": ff["velocity"],
        "pressure": 0.0,
        "nu_tilde": 3.0 * MU,
    }
    cond_i = {
        "bottom": {"type": "wall"},
        "top": ff_i,
        "left": ff_i,
        "right": {"type": "pressure-outlet", "pressure": 0.0},
    }
    prob_i = FlowProblem(mesh, model_i, cond_i)
    w_i = np.column_stack((np.zeros(w.shape[0]), w[:, 1:3], w[:, rc.NT]))
    # Same source and diffusion: compare the turbulence sources cell by cell.
    grad = comp.lsq_gradient(
        prob.geom,
        jnp.asarray(w),
        rc.boundary_state(
            prob.bc, prob.geom.b_normal, jnp.asarray(w)[prob.geom.b_owner]
        ),
    )
    src_c = np.asarray(
        rc._sources_of_state(prob.model, prob.geom, jnp.asarray(w), grad)
    )
    w_bi = rans.boundary_state(
        prob_i.bc, prob_i.geom.b_normal, jnp.asarray(w_i)[prob_i.geom.b_owner]
    )
    grad_i = comp.lsq_gradient(prob_i.geom, jnp.asarray(w_i), w_bi)
    src_i = np.asarray(
        rans._sources_of_state(
            prob_i.model, prob_i.geom, jnp.asarray(w_i), w_bi, grad_i
        )
    )[:, 0]
    # T = 1 and p = R: rho = 1, mu(1) = MU.
    np.testing.assert_allclose(src_c, rho * src_i, rtol=1e-10, atol=1e-14)


def test_density_gradient_term():
    """The cross term -(nu + nu_t)/sigma grad rho . grad nu_t of the source."""
    model = comp.CompressibleModel(viscosity=MU, turbulence="sa-neg")
    w = jnp.array([[1.0 / 1.4, 0.0, 0.0, 1.0, 5.0 * MU]])
    args = (jnp.array([0.0]), jnp.array([1.0]), jnp.array([0.0]))
    base = rc.cell_source(model, w, *args[:2], args[2], jnp.array([0.0]), 1.0)
    cross = 2.0
    with_cross = rc.cell_source(model, w, *args[:2], args[2], jnp.array([cross]), 1.0)
    nu = MU  # T = 1, rho = 1
    expected = -sa.diffusion_coefficient(5.0 * MU, nu) * cross
    np.testing.assert_allclose(with_cross - base, expected, rtol=1e-12)


@pytest.mark.parametrize("turb_order", [1, 2])
def test_jvp_matches_finite_difference(turb_order):
    prob, ff = _box(turb_order=turb_order)
    w = _perturbed(prob, ff, 1)
    rng = np.random.default_rng(2)
    dw = rng.standard_normal(w.shape) * 1e-2
    dw[:, rc.NT] *= MU
    psi = prob.limiter(w)
    jv = prob.jvp(w, dw, 0.0, psi)
    eps = 1e-6
    fd = (
        prob.residual(w + eps * dw, 0.0, psi=psi)
        - prob.residual(w - eps * dw, 0.0, psi=psi)
    ) / (2 * eps)
    np.testing.assert_allclose(jv, fd, rtol=1e-5, atol=1e-10)
    lin = prob.linearize(w, 0.0, psi)
    np.testing.assert_allclose(lin(dw), jv, rtol=1e-12, atol=1e-15)


def test_coupled_first_order_jacobian_is_close_to_exact():
    """Compact fluxes plus the frozen-input source diagonal: close to dense AD."""
    prob, ff = _box(order=1)
    prob.model = prob.model._replace(coupling="coupled")
    w = _perturbed(prob, ff, 3)
    sparse = prob.jacobian(w, 0.0).toarray()
    dense = np.asarray(
        jax.jacfwd(
            lambda x: rc.residual(
                prob.model, prob.geom, prob.bc, x.reshape(w.shape), 1
            ).ravel()
        )(jnp.asarray(w.ravel()))
    )
    err = np.linalg.norm(sparse - dense) / np.linalg.norm(dense)
    assert err < 0.05, err


def test_segregated_jacobian_has_no_cross_blocks():
    prob, ff = _box(order=1)
    w = _perturbed(prob, ff, 4)
    jac = prob.jacobian(w, 0.0).toarray()
    n = prob.model.n_var
    rows = np.arange(jac.shape[0]) % n
    cols = np.arange(jac.shape[1]) % n
    turb_rows, turb_cols = rows == rc.NT, cols == rc.NT
    assert np.all(jac[np.ix_(turb_rows, ~turb_cols)] == 0.0)
    assert np.all(jac[np.ix_(~turb_rows, turb_cols)] == 0.0)


def test_time_term_is_the_conservative_derivative():
    prob, ff = _box()
    w = _perturbed(prob, ff, 5)
    m = np.asarray(rc.transformation(prob.model, jnp.asarray(w)))
    rho = np.asarray(comp.density(prob.model.gas_constant, jnp.asarray(w)))
    np.testing.assert_allclose(m[:, rc.NT, rc.NT], rho)
    np.testing.assert_allclose(
        m[:, :4, :4], comp.transformation(prob.model, jnp.asarray(w[:, :4]))
    )


def test_eddy_viscosity_heats_and_stresses():
    """mu_t enters the stress and the conductivity c_p mu_t / Pr_t."""
    model = comp.CompressibleModel(viscosity=MU)
    grad = np.zeros((1, 4, 2))
    grad[0, comp.U, 1] = 2.0  # du/dy
    grad[0, comp.T, 1] = 3.0  # dT/dy
    w_f = jnp.array([[1.0 / 1.4, 0.0, 0.0, 1.0]])
    n = jnp.array([[0.0, 1.0]])
    lam = comp.viscous_flux(model, jnp.asarray(grad), w_f, n)
    turb = comp.viscous_flux(
        model, jnp.asarray(grad), w_f, n, mu_t=jnp.array([4.0 * MU])
    )
    np.testing.assert_allclose(turb[0, 1], 5.0 * lam[0, 1])  # mu + mu_t = 5 mu
    cp = comp.heat_capacity(model)
    expected_q = -cp * (MU / 0.72 + 4.0 * MU / 0.9) * 3.0
    np.testing.assert_allclose(turb[0, 3], expected_q)


def test_adiabatic_wall_has_no_heat_flux_with_rans():
    prob, ff = _box()
    w = _perturbed(prob, ff, 6)
    faces = prob.boundary_faces(["bottom"])
    np.testing.assert_array_equal(prob.wall_heat_flux(w)[faces], 0.0)
    hot, _ = _box(wall_temperature=1.5)
    assert np.all(hot.wall_heat_flux(w)[faces] > 0.0)


def test_step_limiter_caps_nu_tilde_and_keeps_positivity():
    prob, ff = _box()
    w = _perturbed(prob, ff, 7)
    du = np.zeros_like(w)
    du[0, rc.NT] = 100.0 * w[0, rc.NT]
    du[1, comp.P] = -3.0 * w[1, comp.P]
    out = prob._step_limiter(1.0)(w, du)
    assert out[0, rc.NT] == pytest.approx(w[0, rc.NT])
    assert w[1, comp.P] + out[1, comp.P] > 0.0


def _rans_raw(**physics):
    ff = {"type": "farfield", "mach": 0.15, "turbulence": {"nu_tilde_ratio": 3.0}}
    return {
        "physics": {"flow": "compressible", "model": "rans", **physics},
        "mesh": {"file": "m.su2"},
        "fluid": {"reynolds_number": 1.0e6},
        "reference": {"length": 1.0},
        "boundary": {"bottom": {"type": "wall"}, "top": ff, "left": ff, "right": ff},
    }


def test_compressible_rans_case_setup():
    from pyaeroflow.run import setup, state_scale

    config = resolve(_rans_raw())
    assert config.data["physics"]["turbulence"] == "sa-neg"
    mesh = rectangle(3, 3)
    problem, scaling, w0, _ = setup(config, mesh)
    assert problem.physics is rc
    nu_inf = problem.model.viscosity  # rho = T = 1 at the farfield
    assert problem.conditions["top"]["nu_tilde"] == pytest.approx(3.0 * nu_inf)
    np.testing.assert_allclose(w0[:, rc.NT], 3.0 * nu_inf)
    scale = state_scale(problem, scaling)
    assert scale.size == 5
    assert scale[rc.NT] == pytest.approx(scaling.viscosity / scaling.density)


def test_compressible_rans_case_rules():
    with pytest.raises(ConfigError, match="sst is planned"):
        resolve(_rans_raw(turbulence="sst"))
    with pytest.raises(ConfigError, match="incompressible only"):
        resolve(_rans_raw(transition={"model": "bcm", "turbulence_intensity": 0.01}))


def test_restart_between_laminar_and_rans(tmp_path):
    from pyaeroflow.run import load_restart, mesh_fingerprint, save_restart

    mesh = rectangle(3, 3)
    w4 = np.ones((mesh.n_cells, 4))
    path = tmp_path / "lam.npz"
    save_restart(path, mesh, w4, flow=comp.FLOW)
    default = np.full((mesh.n_cells, 5), 7.0)
    w, n = load_restart(path, mesh, default, flow=comp.FLOW)
    assert n == 4 and mesh_fingerprint(mesh)
    np.testing.assert_allclose(w[:, :4], 1.0)
    np.testing.assert_allclose(w[:, 4], 7.0)

"""Compressible k-omega SST RANS (L1 stage C4)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.config import resolve
from pyaeroflow.driver import FlowProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import compressible as comp
from pyaeroflow.physics import rans_sst_compressible as rs
from pyaeroflow.physics import sst

jax.config.update("jax_enable_x64", True)

MU = 1.0e-3
K_INF, OMEGA_INF = 1.0e-4, 1.0


def _box(mach=0.3, order=2, turb_order=1, mesh=None, coupling="segregated"):
    mesh = mesh or rectangle(6, 5, cell_type="mixed", perturb=0.3, seed=4)
    r_gas, ff = comp.freestream_conditions(mach, 2.0)
    ff = {**ff, "k": K_INF, "omega": OMEGA_INF}
    model = comp.CompressibleModel(
        gas_constant=r_gas,
        viscosity=MU,
        spatial_order=order,
        turbulence="sst",
        turbulence_spatial_order=turb_order,
        coupling=coupling,
    )
    conditions = {
        "bottom": {"type": "wall"},
        "top": ff,
        "left": ff,
        "right": {"type": "pressure-outlet", "pressure": r_gas},
    }
    return FlowProblem(mesh, model, conditions), ff


def _perturbed(prob, ff, seed, amount=0.05):
    rng = np.random.default_rng(seed)
    w = prob.initial_state(ff["velocity"], ff["pressure"], k_omega=(K_INF, OMEGA_INF))
    w[:, rs.NK] *= 1.0 + 20.0 * rng.uniform(0.0, 1.0, w.shape[0])
    w[:, rs.NW] *= 1.0 + 20.0 * rng.uniform(0.0, 1.0, w.shape[0])
    return w * (1.0 + rng.uniform(-amount, amount, w.shape))


def _boundary_state(prob, w):
    return np.asarray(
        rs.boundary_state(
            prob.bc, prob.geom.b_normal, jnp.asarray(w)[prob.geom.b_owner]
        )
    )


def test_model_has_six_variables_and_the_sst_module():
    prob, ff = _box()
    assert prob.model.n_var == 6
    assert prob.physics is rs
    assert prob.turbulence_column == rs.NK == 4
    w = prob.initial_state(ff["velocity"], ff["pressure"], k_omega=(K_INF, OMEGA_INF))
    np.testing.assert_allclose(w[:, rs.NK], K_INF)
    np.testing.assert_allclose(w[:, rs.NW], OMEGA_INF)


def test_boundary_values_of_k_and_omega():
    """Walls: k = 0, omega = 60 nu_w / (beta_1 d1^2) with nu_w of the wall state."""
    prob, ff = _box()
    w = _perturbed(prob, ff, 0)
    w_b = _boundary_state(prob, w)
    own = np.asarray(prob.geom.b_owner)
    wall = prob.boundary_faces(["bottom"])
    np.testing.assert_array_equal(w_b[wall, rs.NK], 0.0)
    delta, normal = np.asarray(prob.geom.b_delta), np.asarray(prob.geom.b_normal)
    d1 = 2.0 * np.abs(np.einsum("fk,fk->f", delta, normal))[wall]
    nu_w = np.asarray(rs.kinematic_viscosity(prob.model, jnp.asarray(w_b[wall])))
    np.testing.assert_allclose(w_b[wall, rs.NW], sst.wall_omega(nu_w, d1), rtol=1e-12)
    inflow = prob.boundary_faces(["left"])
    np.testing.assert_allclose(w_b[inflow, rs.NK], K_INF)
    np.testing.assert_allclose(w_b[inflow, rs.NW], OMEGA_INF)
    out = prob.boundary_faces(["right"])
    np.testing.assert_allclose(w_b[out, rs.NK :], w[own[out], rs.NK :])


def test_sources_are_density_times_the_kinematic_sources():
    prob, ff = _box()
    w = jnp.asarray(_perturbed(prob, ff, 1))
    w_b = rs.boundary_state(prob.bc, prob.geom.b_normal, w[prob.geom.b_owner])
    grad = comp.lsq_gradient(prob.geom, w, w_b)
    q = rs.cell_quantities(prob.model, prob.geom, w, grad)
    s_k, s_w = rs._sources(prob.model, prob.geom, w, grad, q)
    strain, f1, f2, _, kgw, rho, _ = q
    div = grad[:, comp.U, 0] + grad[:, comp.V, 1]
    k_ref, w_ref = sst.sources(
        w[:, rs.NK], w[:, rs.NW], strain, f1, f2, kgw, 1.0, divergence=div
    )
    np.testing.assert_allclose(s_k, rho * k_ref, rtol=1e-12)
    np.testing.assert_allclose(s_w, rho * w_ref, rtol=1e-12)


def test_divergence_part_of_the_production():
    """P_k = nu_t S^2 - 2/3 k div v and P_omega = gamma (S^2 - 2/3 zeta div v)."""
    k, omega = jnp.array([1.0e-3]), jnp.array([10.0])
    strain, f1, f2, kgw = jnp.array([2.0]), jnp.array([1.0]), jnp.array([1.0]), 0.0
    base_k, base_w = sst.sources(k, omega, strain, f1, f2, kgw)
    same_k, same_w = sst.sources(k, omega, strain, f1, f2, kgw, divergence=0.0)
    np.testing.assert_array_equal(base_k, same_k)
    np.testing.assert_array_equal(base_w, same_w)
    div = 0.5
    s_k, s_w = sst.sources(k, omega, strain, f1, f2, kgw, divergence=div)
    zeta = sst.zeta(omega, strain, f2)
    np.testing.assert_allclose(s_k - base_k, -2.0 / 3.0 * k * div, rtol=1e-12)
    np.testing.assert_allclose(
        s_w - base_w, -sst.GAMMA_1 * 2.0 / 3.0 * zeta * div, rtol=1e-12
    )


def test_tke_stress_acts_like_a_pressure():
    """The -2/3 rho k delta_ij stress gives the flux 2/3 rho k (0, n, v.n)."""
    model = comp.CompressibleModel(viscosity=MU, turbulence="sst")
    w = jnp.array([[1.0 / 1.4, 0.5, 0.2, 1.0, 0.01, 1.0]])
    n = jnp.array([[0.6, 0.8]])
    flux = rs._tke_flux(model, w, w, n)[0]
    q = 2.0 / 3.0 * 1.0 * 0.01  # rho = 1
    np.testing.assert_allclose(flux, [0.0, q * 0.6, q * 0.8, q * (0.3 + 0.16)])


@pytest.mark.parametrize("turb_order", [1, 2])
def test_jvp_matches_finite_difference(turb_order):
    prob, ff = _box(turb_order=turb_order)
    w = _perturbed(prob, ff, 2)
    rng = np.random.default_rng(3)
    dw = rng.standard_normal(w.shape) * 1e-2
    dw[:, rs.NK] *= K_INF
    psi = prob.limiter(w)
    jv = prob.jvp(w, dw, 0.0, psi)
    eps = 1e-6
    fd = (
        prob.residual(w + eps * dw, 0.0, psi=psi)
        - prob.residual(w - eps * dw, 0.0, psi=psi)
    ) / (2 * eps)
    np.testing.assert_allclose(jv, fd, rtol=1e-5, atol=1e-9)
    lin = prob.linearize(w, 0.0, psi)
    np.testing.assert_allclose(lin(dw), jv, rtol=1e-12, atol=1e-15)


def test_first_order_jacobian_diagonal_blocks_are_close_to_exact():
    """Mean-flow block and the k-k, omega-omega blocks: close to dense AD.

    The compact fluxes freeze mu_t and the diffusion coefficients, the source
    blocks keep only the destruction (SU2), so only these blocks are compared.
    """
    prob, ff = _box(order=1, coupling="coupled")
    w = _perturbed(prob, ff, 4)
    sparse = prob.jacobian(w, 0.0).toarray()
    dense = np.asarray(
        jax.jacfwd(
            lambda x: rs.residual(
                prob.model, prob.geom, prob.bc, x.reshape(w.shape), 1
            ).ravel()
        )(jnp.asarray(w.ravel()))
    )
    n = prob.model.n_var

    def block(a, rows, cols):
        sel_r = np.isin(np.arange(a.shape[0]) % n, rows)
        sel_c = np.isin(np.arange(a.shape[1]) % n, cols)
        return a[np.ix_(sel_r, sel_c)]

    for rows in ([0, 1, 2, 3], [rs.NK], [rs.NW]):
        d, s = block(dense, rows, rows), block(sparse, rows, rows)
        err = np.linalg.norm(s - d) / np.linalg.norm(d)
        assert err < 0.06, (rows, err)


def test_segregated_jacobian_has_no_cross_blocks():
    prob, ff = _box(order=1)
    w = _perturbed(prob, ff, 5)
    jac = prob.jacobian(w, 0.0).toarray()
    n = prob.model.n_var
    rows = np.arange(jac.shape[0]) % n
    cols = np.arange(jac.shape[1]) % n
    turb_rows, turb_cols = rows >= rs.NK, cols >= rs.NK
    assert np.all(jac[np.ix_(turb_rows, ~turb_cols)] == 0.0)
    assert np.all(jac[np.ix_(~turb_rows, turb_cols)] == 0.0)


def test_time_term_is_the_row_scaled_conservative_derivative():
    prob, ff = _box()
    w = _perturbed(prob, ff, 6)
    m = np.asarray(rs.transformation(prob.model, jnp.asarray(w)))
    rho = np.asarray(comp.density(prob.model.gas_constant, jnp.asarray(w)))
    np.testing.assert_allclose(m[:, rs.NK, rs.NK], rho)
    np.testing.assert_allclose(m[:, rs.NW, rs.NW], rho)
    np.testing.assert_allclose(
        m[:, :4, :4], comp.transformation(prob.model, jnp.asarray(w[:, :4]))
    )
    blocks = rs.time_term_blocks(prob.model, prob.geom, w, 0.0, 10.0)
    lam = np.asarray(rs.spectral_radius(prob.model, prob.geom, jnp.asarray(w)))
    np.testing.assert_allclose(blocks[:, rs.NK, :], m[:, rs.NK, :] * lam[:, None] / 10)
    np.testing.assert_allclose(
        blocks[:, rs.NW, :], MU * m[:, rs.NW, :] * lam[:, None] / 10
    )


def test_step_limiter_keeps_k_omega_and_p_positive():
    prob, ff = _box()
    w = _perturbed(prob, ff, 7)
    du = np.zeros_like(w)
    du[0, rs.NK] = -5.0 * w[0, rs.NK]
    du[1, rs.NW] = -5.0 * w[1, rs.NW]
    du[2, comp.P] = -3.0 * w[2, comp.P]
    out = prob._step_limiter(None)(w, du)
    new = w + out
    assert new[0, rs.NK] > 0.0 and new[1, rs.NW] > 0.0 and new[2, comp.P] > 0.0


def _sst_raw(**init):
    ff = {
        "type": "farfield",
        "mach": 0.15,
        "turbulence": {"turbulent_viscosity_ratio": 0.009, "intensity": 5.164e-4},
    }
    raw = {
        "physics": {"flow": "compressible", "model": "rans", "turbulence": "sst"},
        "mesh": {"file": "m.su2"},
        "fluid": {"reynolds_number": 1.0e6},
        "reference": {"length": 1.0},
        "boundary": {"bottom": {"type": "wall"}, "top": ff, "left": ff, "right": ff},
    }
    if init:
        raw["initial"] = init
    return raw


def test_compressible_sst_case_setup():
    from pyaeroflow.run import setup, state_scale

    config = resolve(_sst_raw())
    mesh = rectangle(3, 3)
    problem, scaling, w0, _ = setup(config, mesh)
    assert problem.physics is rs
    nu_inf = problem.model.viscosity  # rho = T = U = 1 at the farfield
    k = 1.5 * 5.164e-4**2
    top = problem.conditions["top"]
    assert top["k"] == pytest.approx(k)
    assert top["omega"] == pytest.approx(k / (0.009 * nu_inf))
    np.testing.assert_allclose(w0[:, rs.NK], k)
    np.testing.assert_allclose(w0[:, rs.NW], top["omega"])
    scale = state_scale(problem, scaling)
    assert scale.size == 6
    assert scale[rs.NK] == pytest.approx(scaling.velocity**2)
    assert scale[rs.NW] == pytest.approx(scaling.velocity / scaling.length)


def test_restart_between_sa_and_sst_keeps_the_mean_flow(tmp_path):
    from pyaeroflow.run import load_restart, save_restart

    mesh = rectangle(3, 3)
    path = tmp_path / "sa.npz"
    save_restart(path, mesh, np.ones((mesh.n_cells, 5)), flow=comp.FLOW)
    w, n = load_restart(path, mesh, np.full((mesh.n_cells, 6), 7.0), flow=comp.FLOW)
    assert n == 4
    np.testing.assert_allclose(w[:, :4], 1.0)
    np.testing.assert_allclose(w[:, 4:], 7.0)

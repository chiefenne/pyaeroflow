"""Float32 Newton-Krylov products (newton_krylov.product_precision)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_lm import _lm_plate
from test_newton_krylov import cavity
from test_rans import RANS_MINIMAL

import pyaeroflow.driver as dr
from pyaeroflow.config import resolve
from pyaeroflow.physics import lm, rans_sst, sst
from pyaeroflow.solver import PseudoTimeSettings


def _lm_state(seed=0):
    p, k_omega = _lm_plate()
    w = p.initial_state((1.0, 0.0), k_omega=k_omega)
    rng = np.random.default_rng(seed)
    w = w * (1.0 + 0.05 * rng.standard_normal(w.shape))
    w[:, rans_sst.NK :] = np.abs(w[:, rans_sst.NK :])
    return p, w, rng.standard_normal(w.shape)


def test_float32_product_matches_float64():
    p, w, dw = _lm_state()
    b2 = p.beta2(w)
    psi = p.limiter(w)
    exact = p.linearize(w, b2, psi)(dw)
    single = p.linearize(w, b2, psi, precision="float32")(dw)
    assert single.dtype == np.float64
    assert np.isfinite(single).all()
    err = np.linalg.norm(single - exact) / np.linalg.norm(exact)
    assert 0.0 < err < 1e-4  # really single precision, and accurate enough
    st = p.product_stats
    assert st["linearizations"] == 1 and st["products"] == 1
    assert st["fallback_products"] == 0


def test_nonfinite_product_falls_back_to_float64(monkeypatch):
    p, w, dw = _lm_state(1)
    b2 = p.beta2(w)
    exact = p.linearize(w, b2)(dw)
    plain = dr._apply_linear

    def broken(lin, x):  # every float32 product overflows
        out = plain(lin, x)
        return out * np.nan if out.dtype == jnp.float32 else out

    monkeypatch.setattr(dr, "_apply_linear", broken)
    apply = p.linearize(w, b2, precision="float32")
    np.testing.assert_allclose(apply(dw), exact, rtol=1e-12)
    np.testing.assert_allclose(apply(2.0 * dw), 2.0 * exact, rtol=1e-12)
    st = p.product_stats
    assert st["fallback_steps"] == 1 and st["fallback_products"] == 2
    # After FLOAT32_FALLBACK_LIMIT steps in a row with a fallback the run
    # stays in float64: no more float32 linearizations.
    for _ in range(dr.FLOAT32_FALLBACK_LIMIT - 1):
        p.linearize(w, b2, precision="float32")(dw)
    assert st["disabled"]
    n = st["linearizations"]
    np.testing.assert_allclose(
        p.linearize(w, b2, precision="float32")(dw), exact, rtol=1e-12
    )
    assert st["linearizations"] == n


def test_fallback_streak_resets_after_a_clean_step(monkeypatch):
    p, w, dw = _lm_state(2)
    b2 = p.beta2(w)
    plain = dr._apply_linear
    fail = {"on": True}

    def flaky(lin, x):
        out = plain(lin, x)
        return out * np.nan if fail["on"] and out.dtype == jnp.float32 else out

    monkeypatch.setattr(dr, "_apply_linear", flaky)
    for _ in range(dr.FLOAT32_FALLBACK_LIMIT - 1):
        p.linearize(w, b2, precision="float32")(dw)
    fail["on"] = False
    p.linearize(w, b2, precision="float32")(dw)  # clean step
    fail["on"] = True
    p.linearize(w, b2, precision="float32")(dw)
    assert not p.product_stats["disabled"]
    assert p.product_stats["fallback_streak"] == 1


def _blending_unclamped(k, omega, nu, d, kgw, tiny):
    arg2a = jnp.sqrt(k) / (sst.BETA_STAR * omega * d + tiny)
    arg2b = 500.0 * nu / (d * d * omega + tiny)
    cd = sst.cross_diffusion(kgw, omega)
    arg1 = jnp.minimum(
        jnp.maximum(arg2a, arg2b), 4.0 * sst.SIGMA_W2 * k / (cd * d * d + tiny)
    )
    return jnp.tanh(arg1**4), jnp.tanh(jnp.maximum(2.0 * arg2a, arg2b) ** 2)


def test_sst_blending_clamp_is_exact_and_float32_safe():
    # float64: the clamps only act where tanh is 1 to double precision and
    # the float64 floor is unchanged, so the result is bitwise the same.
    rng = np.random.default_rng(0)
    n = 4000
    k = 10.0 ** rng.uniform(-12, -1, n)
    omega = 10.0 ** rng.uniform(-2, 8, n)
    d = 10.0 ** rng.uniform(-6, 1.7, n)
    # one wall point: d = 0, k = 0
    k, omega, d = (
        np.concatenate(([v], a)) for v, a in ((0.0, k), (1e5, omega), (0.0, d))
    )
    kgw = rng.standard_normal(k.size) * 10.0 ** rng.uniform(-8, 2, k.size)
    new = sst.blending(k, omega, 1e-5, d, kgw)
    old = _blending_unclamped(k, omega, 1e-5, d, kgw, 1e-300)
    for a, b in zip(new, old, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert 0.05 < np.mean(np.asarray(new[0]) < 1.0) < 0.95  # both branches hit
    with jax.enable_x64(False):
        g = jax.vmap(jax.grad(lambda *x: sst.blending(*x)[0]), (0, 0, None, 0, 0))(
            *(np.asarray(a, np.float32) for a in (k, omega)),
            1e-5,
            *(np.asarray(a, np.float32) for a in (d, kgw)),
        )
    assert np.isfinite(np.asarray(g)).all()


def test_theta_blending_tangent_is_finite_at_zero_vorticity():
    """Freestream cell: the float32 vorticity cancels to exactly 0, so delta
    sits on its floor; 1 / delta^2 must not overflow into an inf * 0 = NaN."""
    args = (0.998, 668.0, 9.3e-7, 0.1085, 1.6667e-5, 0.194, 0.0, 0.1686, 0.9876)
    tang = (1.0, 1.0, 1e-6, 1e-2, 0.0, 1.0, 1.0, 0.0, 1.0)
    fn = lambda *a: lm.effective_intermittency(*a)  # noqa: E731
    with jax.enable_x64(False):
        _, t32 = jax.jvp(
            fn, tuple(np.float32(a) for a in args), tuple(np.float32(a) for a in tang)
        )
    _, t64 = jax.jvp(
        fn, tuple(np.float64(a) for a in args), tuple(np.float64(a) for a in tang)
    )
    assert np.isfinite(float(t32))
    assert float(t32) == pytest.approx(float(t64), rel=1e-3)


def test_float32_cavity_run_matches_float64():
    prob = cavity(12)
    s = {"max_iterations": 40, "residual_drop": 8.0, "nk_switch_iteration": 3}
    ref = prob.solve(settings=PseudoTimeSettings(**s))
    f32 = prob.solve(settings=PseudoTimeSettings(nk_product_precision="float32", **s))
    assert ref.converged and f32.converged
    assert abs(len(f32.history["iteration"]) - len(ref.history["iteration"])) <= 2
    np.testing.assert_allclose(f32.u, ref.u, rtol=1e-6, atol=1e-8)
    assert prob.product_stats["products"] > 0


def test_product_precision_option():
    cfg = resolve(RANS_MINIMAL).data
    assert cfg["solver"]["newton_krylov"]["product_precision"] == "float64"
    raw = {
        **RANS_MINIMAL,
        "solver": {"newton_krylov": {"product_precision": "float32"}},
    }
    assert (
        resolve(raw).data["solver"]["newton_krylov"]["product_precision"] == "float32"
    )
    with pytest.raises(Exception, match="not one of"):
        resolve(
            {**RANS_MINIMAL, "solver": {"newton_krylov": {"product_precision": "f16"}}}
        )

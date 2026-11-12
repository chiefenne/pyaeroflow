"""Viscous panel analysis (pyaeroflow.ibl): closures, sources, residuals, solver.

Reference numbers of the closure and residual tests come from mfoil
(Fidkowski, MIT licence), run as a black box on the same states.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyaeroflow.ibl import closures as cl
from pyaeroflow.ibl import residuals as res
from pyaeroflow.ibl import sources
from pyaeroflow.ibl.geometry import naca4, repanel

jax.config.update("jax_enable_x64", True)

NU = 1e-6
# theta, H, s, ue, then (H*, Cf, 2CD/H*) laminar, turbulent, wake,
# then dn/dxi, sqrt(Ctau_eq)
CLOSURE_REF = [
    (
        0.0002,
        2.6,
        3.0,
        1.2,
        1.5746884930555554,
        0.001760509606481481,
        0.00091685591559329,
        1.5675985663082437,
        0.001013713576909809,
        9.810407221055948,
        1.5675985663082437,
        0.0,
        19.620632536460732,
        2.710788672447251,
        0.07573381519958088,
    ),
    (
        0.0005,
        1.45,
        0.08,
        1.1,
        1.8393065337755101,
        0.003456749601113171,
        0.0010181081491624584,
        1.7715356293706295,
        0.004795363387520546,
        0.004824588127097079,
        1.7715356293706295,
        0.0,
        0.0070144202221090355,
        0.0,
        0.03750300348431755,
    ),
    (
        0.001,
        3.2,
        0.05,
        0.9,
        1.5388534426190477,
        0.00015622775132275124,
        0.00023066758455443444,
        1.5064534534534535,
        0.0001468518235192701,
        0.0032925399618546516,
        1.5064534534534535,
        0.0,
        0.006572842279653536,
        13.636996214994575,
        0.08727240535406762,
    ),
    (
        0.0003,
        5.5,
        7.0,
        1.0,
        1.5316068181818183,
        -0.00023333333333333336,
        0.0006785167464114832,
        1.5380088282336288,
        -0.00020897559148498542,
        67.85544852581947,
        1.5380088282336288,
        0.0,
        135.7108780538579,
        128.96180491828306,
        0.10669170473767488,
    ),
]


@pytest.mark.parametrize("row", CLOSURE_REF)
def test_closures_match_reference(row):
    th, h, sa, ue = row[:4]
    ds = th * h
    got = []
    for turb, wake in ((0, 0), (1, 0), (1, 1)):
        got += [
            cl.hstar(th, ds, ue, NU, turb, wake),
            cl.skin_friction(th, ds, ue, NU, turb, wake),
            cl.dissipation(th, ds, sa, ue, NU, turb, wake),
        ]
    got += [
        cl.amplification_rate(th, ds, sa, ue, NU, 9.0),
        cl.ctau_equilibrium(th, ds, ue, NU, False),
    ]
    np.testing.assert_allclose(
        np.array(got, dtype=float), row[4:], rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize(
    "turb, wake, ref",
    [
        (0, 0, [-0.06390421046657427, 0.032262497109435226, 0.3999999999999999]),
        (1, 0, [-0.07871777945062111, -0.08453880056552136, -0.0022854492187519915]),
        (1, 1, [0.011394857682743904, -0.21078788466605025, -0.0030231567586587477]),
    ],
)
def test_interval_residuals_match_reference(turb, wake, ref):
    u1 = np.array([4e-4, 4e-4 * 1.6, 0.06, 1.2])
    u2 = np.array([4.3e-4, 4.3e-4 * 1.65, 0.065, 1.18])
    if not turb:
        u1[2], u2[2] = 3.0, 3.4
    g1, g2 = (1e-4, 8e-5) if wake else (0.0, 0.0)
    r = res.interval(
        jnp.asarray(u1), jnp.asarray(u2), 0.3, 0.32, g1, g2, NU, turb, wake, 9.0
    )
    np.testing.assert_allclose(np.asarray(r), ref, rtol=1e-10, atol=1e-14)


def test_source_stream_functions_give_source_velocities():
    a, b = jnp.array([[0.3, 0.1]]), jnp.array([[1.1, 0.4]])
    pts = np.random.default_rng(0).uniform(-1.0, 2.0, (100, 2))

    def grad_velocity(fn, k=None):
        def f(p):
            v = fn(p[None, :], a, b)[0, 0]
            return v if k is None else v[k]

        g = np.asarray(jax.vmap(jax.grad(f))(jnp.asarray(pts)))
        return np.stack((g[:, 1], -g[:, 0]), axis=-1)

    x1, y, length, _ = (np.asarray(t) for t in sources._local(jnp.asarray(pts), a, b))
    outside_strip = ~((y[:, 0] < 0) & (x1[:, 0] > 0) & (x1[:, 0] < length[0, 0]))
    vc = np.asarray(sources.constant_velocity(jnp.asarray(pts), a, b))[:, 0]
    gc = grad_velocity(sources.constant_psi)
    np.testing.assert_allclose(gc[outside_strip], vc[outside_strip], atol=1e-13)
    vl = np.asarray(sources.linear_velocity(jnp.asarray(pts), a, b))[:, 0]
    for k in range(2):
        np.testing.assert_allclose(
            grad_velocity(sources.linear_psi, k), vl[:, k], atol=1e-13
        )
    # brute-force integral of the linear source at one point
    s = np.linspace(0.0, 1.0, 100001)
    src = np.asarray(a[0])[None] + np.asarray(b[0] - a[0])[None] * s[:, None]
    ds = float(np.linalg.norm(b[0] - a[0])) / (s.size - 1)
    p = np.array([0.5, -0.3])
    r = p - src
    r2 = np.sum(r * r, axis=1)
    for k, w in enumerate((1.0 - s, s)):
        f = w[:, None] * r / r2[:, None] / (2.0 * np.pi)
        integral = 0.5 * (f[:-1] + f[1:]).sum(axis=0) * ds
        got = np.asarray(sources.linear_velocity(jnp.asarray(p[None]), a, b))[0, 0, k]
        np.testing.assert_allclose(got, integral, rtol=1e-8)


def test_repanel_sharp_and_blunt_trailing_edge():
    blunt = repanel(naca4("2412", 101), 120)
    assert blunt.shape == (120, 2)
    assert np.linalg.norm(blunt[0] - blunt[-1]) > 1e-3
    assert blunt[0, 1] > blunt[-1, 1]  # upper trailing edge first
    sharp = repanel(naca4("0012", 101, closed_te=True), 120)
    np.testing.assert_allclose(sharp[0], sharp[-1])


@pytest.mark.slow
def test_naca2412_against_reference():
    """NACA 2412, Re 1e6, alpha 2: mfoil CL 0.4494, CD 0.00578, x_tr 0.526/0.966."""
    from pyaeroflow.ibl import analyse

    r = analyse(repanel(naca4("2412", 201), 200), 2.0, 1e6)
    assert r.converged
    assert abs(r.cl - 0.4494) < 0.006
    assert abs(r.cd - 0.00578) < 0.0001
    assert abs(r.x_transition[0] - 0.526) < 0.005
    assert abs(r.x_transition[1] - 0.966) < 0.01


def test_displacement_velocity_flux_equals_source_strength():
    """The source sheet of a mass defect sends its total strength through a circle."""
    from pyaeroflow.ibl import inviscid as inv_mod

    inv = inv_mod.build(repanel(naca4("0012", 121), 80), 2.0)
    n, nw = inv.s.size, inv.wake_s.size
    rng = np.random.default_rng(1)
    mass = rng.uniform(0.0, 1e-3, n + nw)
    sigma = inv_mod._sigma_matrix(inv.s, inv.wake_s) @ mass
    seg_a, seg_b, val_a, val_b = inv_mod._wake_segments(jnp.asarray(inv.wake), n)
    length = np.linalg.norm(np.asarray(seg_b) - np.asarray(seg_a), axis=1)
    total = float(np.sum(sigma[: n - 1] * np.diff(inv.s)))
    total += float(np.sum(0.5 * (val_a @ sigma + val_b @ sigma) * length))
    phi = np.linspace(0.0, 2.0 * np.pi, 2001)[:-1]
    centre = np.array([1.0, 0.0])
    radius = 20.0
    normal = np.column_stack((np.cos(phi), np.sin(phi)))
    v = inv_mod.displacement_velocity(inv, mass, centre + radius * normal)
    flux = float(np.mean(np.einsum("ij,ij->i", v, normal)) * 2.0 * np.pi * radius)
    assert flux == pytest.approx(total, rel=1e-3)


def test_foil_cli_max_iterations_option():
    import argparse

    from pyaeroflow.ibl.cli import add_parser

    sub = argparse.ArgumentParser().add_subparsers()
    p = add_parser(sub)
    args = p.parse_args(["naca0012", "--alpha", "-1", "2", "--re", "1e6"])
    assert args.max_iterations == 100 and args.alpha == [-1.0, 2.0]
    args = p.parse_args(
        ["e387.dat", "--alpha", "6", "--re", "2e5", "--max-iterations", "200"]
    )
    assert args.max_iterations == 200


def test_stall_needs_small_steps_and_a_level_residual():
    from pyaeroflow.ibl.solver import _stalled

    level = [10.0] * 40
    assert _stalled(level, [0.001] * 39)
    assert not _stalled(level, [0.001] * 30 + [1.0] + [0.001] * 8)  # a full step
    falling = [10.0 * 0.9**k for k in range(40)]
    assert not _stalled(falling, [0.001] * 39)
    assert not _stalled(level[:10], [0.001] * 9)  # too short


def test_cached_jit_stores_and_reloads_the_traced_program(tmp_path, monkeypatch):
    from pyaeroflow.ibl import cache

    monkeypatch.setenv("PYAEROFLOW_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PYAEROFLOW_JAX_CACHE", raising=False)

    def f(x, k):
        return jnp.sin(x) * k

    x = np.linspace(0.0, 1.0, 5)
    first = cache.cached_jit("unit_test_kernel", f)
    y = np.asarray(first(x, 2.0))
    files = list((tmp_path / "ibl").glob("unit_test_kernel-*.bin"))
    assert len(files) == 1
    second = cache.cached_jit("unit_test_kernel", f)  # new process: loads the file
    np.testing.assert_allclose(np.asarray(second(x, 3.0)), 1.5 * y, rtol=1e-14)
    assert len(list((tmp_path / "ibl").glob("*.bin"))) == 1
    np.testing.assert_allclose(y, 2.0 * np.sin(x), rtol=1e-14)
    # another shape is another program
    second(np.zeros(3), 1.0)
    assert len(list((tmp_path / "ibl").glob("*.bin"))) == 2
    monkeypatch.setenv("PYAEROFLOW_JAX_CACHE", "0")
    assert cache.cache_directory() is None


@pytest.mark.slow
def test_e387_alpha6_single_case_and_warm_start():
    """Cold start stalls (transition at the leading edge); continuation solves it."""
    from pathlib import Path

    from pyaeroflow.ibl import analyse
    from pyaeroflow.ibl.geometry import read_coordinates

    foil = Path(__file__).parents[2] / "examples" / "airfoils" / "e387.dat"
    nodes = repanel(read_coordinates(foil), 200)
    r = analyse(nodes, 6.0, 2e5, ncrit=8.15)
    assert r.converged and r.start.startswith("continuation")
    assert abs(r.cl - 1.0451) < 2e-3 and abs(r.cd - 0.01270) < 1e-4
    assert abs(r.x_transition[0] - 0.508) < 0.01
    cold = analyse(nodes, 6.0, 2e5, ncrit=8.15, continuation=False)
    assert not cold.converged
    r5 = analyse(nodes, 5.0, 2e5, ncrit=8.15)
    warm = analyse(nodes, 6.0, 2e5, ncrit=8.15, initial=r5)
    assert warm.converged and warm.start == "warm"
    assert abs(warm.cl - r.cl) < 1e-6 and abs(warm.cd - r.cd) < 1e-7

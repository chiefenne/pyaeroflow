"""Refinement indicators and marking (R3)."""

import numpy as np
import pytest

from pyaeroflow.mesh import rectangle, structured
from pyaeroflow.mesh.indicator import (
    IndicatorError,
    RefineSpec,
    apply,
    evaluate,
    geometry_variables,
    names_used,
    plan,
)
from pyaeroflow.mesh.refine import Refinement

V = {
    "p": np.array([1.0, 2.0, 4.0, 8.0]),
    "k": np.array([0.0, 1.0, 2.0, 3.0]),
    "vel": np.array([[3.0, 4.0], [0.0, 1.0], [6.0, 8.0], [0.0, 0.0]]),
}


def test_arithmetic_and_functions():
    np.testing.assert_allclose(evaluate("p * 2 + 1", V), [3, 5, 9, 17])
    np.testing.assert_allclose(evaluate("sqrt(2/3 * k)", V), np.sqrt(2 / 3 * V["k"]))
    np.testing.assert_allclose(evaluate("-p**2", V), -(V["p"] ** 2))
    np.testing.assert_allclose(evaluate("norm(vel)", V), [5, 1, 10, 0])
    np.testing.assert_allclose(evaluate("p * (k > 1)", V), [0, 0, 4, 8])
    np.testing.assert_allclose(evaluate("where(k > 1, p, 0)", V), [0, 0, 4, 8])
    np.testing.assert_allclose(
        evaluate("maximum(p, 3) / pi", V), np.maximum(V["p"], 3) / np.pi
    )
    np.testing.assert_allclose(evaluate("3", V), [3, 3, 3, 3])


def test_rank_makes_fields_comparable():
    r = evaluate("rank(p)", V)
    np.testing.assert_allclose(r, [0.0, 0.25, 0.5, 0.75])
    # ties share the lowest rank; a masked field ranks its zeros 0
    np.testing.assert_allclose(evaluate("rank(p * (k > 1))", V), [0, 0, 0.5, 0.75])
    both = evaluate("maximum(rank(p), rank(1000 * k))", V)
    np.testing.assert_allclose(both, [0.0, 0.25, 0.5, 0.75])
    nan = evaluate("rank(log(k - 1))", V)  # log of -1 and 0: nan, -inf
    assert np.isnan(nan[:2]).all() and list(nan[2:]) == [0.0, 0.5]
    with pytest.raises(IndicatorError, match="norm"):
        evaluate("rank(vel)", V)


def test_names_used_excludes_functions_and_constants():
    assert names_used("norm(vel) * sqrt(k) / pi + p") == ["k", "p", "vel"]


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('true')",
        "p.sum()",
        "p[0]",
        "(lambda: 1)()",
        "[p]",
        "'text'",
        "p if k else k",
        "p and k",
        "p < k < 3",
        "p == k",
        "p // 2",
        "p % 2",
        "sqrt(p, out=p)",
        "open('x')",
        "True",
        "",
        "p +",
    ],
)
def test_rejected_expressions(expr):
    with pytest.raises(IndicatorError):
        evaluate(expr, V)


def test_unknown_name_gets_hint_and_lists_available():
    with pytest.raises(IndicatorError, match="provides k, p, vel"):
        evaluate("zz * 2", V)
    with pytest.raises(IndicatorError, match="unknown name 'kk'.*did you mean 'k'"):
        evaluate("kk", V)


def test_vector_without_norm_is_an_error():
    with pytest.raises(IndicatorError, match="norm"):
        evaluate("vel * p", V)
    with pytest.raises(IndicatorError, match="norm"):
        evaluate("vel", V)


def step_mesh():
    """8 x 4 quads on [0, 8] x [0, 4], cell (i, j) at x = i + 0.5."""
    return structured(np.arange(9.0), np.arange(5.0))


def step_fields(ref):
    return {"q": (ref.mesh.cell_centroid[:, 0] > 4.0).astype(float)}


def test_threshold_marks_cells():
    ref = Refinement.from_mesh(step_mesh())
    spec = RefineSpec("q", threshold=0.5, direction="iso")
    m = plan(ref, step_fields(ref), spec)
    assert m.cells.size == 16
    assert np.all(ref.mesh.cell_centroid[m.cells, 0] > 4.0)
    assert np.all(m.nu == 2) and np.all(m.nv == 2)
    assert "marked 16 of 32" in m.summary()


def test_quantile_and_top_fraction():
    ref = Refinement.from_mesh(step_mesh())
    x = {"x2": ref.mesh.cell_centroid[:, 0] + 0.01 * ref.mesh.cell_centroid[:, 1]}
    top = plan(ref, x, RefineSpec("x2", top_fraction=0.25))
    assert top.cells.size == 8
    assert np.all(ref.mesh.cell_centroid[top.cells, 0] > 6.0)
    quant = plan(ref, x, RefineSpec("x2", quantile=0.75))
    np.testing.assert_array_equal(quant.cells, top.cells)
    everything = plan(ref, x, RefineSpec("x2", top_fraction=0.999))
    assert everything.cells.size == 32


def test_max_cells_keeps_largest_indicator():
    ref = Refinement.from_mesh(step_mesh())
    x = ref.mesh.cell_centroid
    f = {"s": x[:, 0] + 0.1 * x[:, 1]}
    m = plan(ref, f, RefineSpec("s", threshold=0.0, direction="iso", max_cells=9))
    # each 2 x 2 split adds 3 cells: three cells fit under the cap
    assert m.cells.size == 3 and m.n_capped == 29
    assert np.all(f["s"][m.cells] >= np.sort(f["s"])[-3])


def test_nonfinite_and_triangles_are_skipped():
    mesh = rectangle(4, 3, cell_type="mixed", quad_layers=1)
    ref = Refinement.from_mesh(mesh)
    ind = np.ones(ref.n_cells)
    quads = np.flatnonzero(mesh.cell_n_nodes == 4)
    ind[quads[0]] = np.nan
    m = plan(ref, {"i": ind}, RefineSpec("i", threshold=0.5))
    assert m.n_non_quad == np.count_nonzero(mesh.cell_n_nodes == 3)
    assert m.n_nonfinite == 1
    assert quads[0] not in m.cells
    assert set(m.cells) == set(quads) - {quads[0]}


def test_auto_direction_splits_thin_cells_along_their_long_side():
    xs = np.linspace(0.0, 4.0, 5)
    ys = np.array([0.0, 0.05, 0.4, 1.4])  # aspect 20, 2.9, 1
    ref = Refinement.from_mesh(structured(xs, ys))
    ext = ref.extents()
    aspect = ext.max(axis=1) / ext.min(axis=1)
    spec = RefineSpec("one", threshold=0.5, aspect_target=3.0, n_max=4)
    m = plan(ref, {"one": np.ones(ref.n_cells)}, spec)
    by_cell = dict(
        zip(
            m.cells.tolist(),
            zip(m.nu.tolist(), m.nv.tolist(), strict=True),
            strict=True,
        )
    )
    for i in range(ref.n_cells):
        long_u = ext[i, 0] >= ext[i, 1]
        if aspect[i] > 3.0:
            n = min(4, int(np.ceil(aspect[i] / 3.0)))
            assert by_cell[i] == ((n, 1) if long_u else (1, n))
        else:
            assert by_cell[i] == (2, 2)
    assert by_cell[0] == (4, 1)  # aspect 20: capped at n_max


def test_max_factor_stops_further_splits():
    ref = Refinement.from_mesh(step_mesh())
    spec = RefineSpec("q", threshold=0.5, direction="iso", max_factor=4)
    sizes = []
    for _ in range(4):
        source, m = apply(ref, step_fields(ref), spec)
        sizes.append(ref.n_cells)
        if source is None:
            break
    assert sizes[-1] == sizes[-2]  # stopped: nothing left below the factor cap
    assert ref.split_factors.max() == 4
    assert m.n_at_max_factor > 0 and m.cells.size == 0


def test_apply_refines_and_returns_parent_map():
    ref = Refinement.from_mesh(step_mesh())
    n0 = ref.n_cells
    spec = RefineSpec("q", threshold=0.5, direction="iso", max_hanging=1)
    source, m = apply(ref, step_fields(ref), spec)
    assert ref.n_cells >= n0 + 3 * m.cells.size
    assert source.size == ref.n_cells
    assert ref.mesh.cell_area.sum() == pytest.approx(32.0)
    assert ref.hanging_counts().max() <= 1
    none, m0 = apply(ref, {"q": np.zeros(ref.n_cells)}, spec)
    assert none is None and m0.cells.size == 0


def test_geometry_variables_and_indicator_with_h():
    ref = Refinement.from_mesh(step_mesh())
    ref.split([0], 2, 2)
    g = geometry_variables(ref)
    assert set(g) >= {"x", "y", "area", "h", "h_u", "h_v", "aspect", "factor_u"}
    ind = evaluate("q * h", {**g, "q": np.ones(ref.n_cells)})
    np.testing.assert_allclose(ind, np.sqrt(ref.mesh.cell_area))
    assert g["factor_u"].max() == 2


def test_field_size_mismatch_is_an_error():
    ref = Refinement.from_mesh(step_mesh())
    with pytest.raises(IndicatorError, match="values, the mesh has"):
        plan(ref, {"q": np.ones(3)}, RefineSpec("q", threshold=0.5))


def test_spec_validation():
    with pytest.raises(IndicatorError, match="exactly one"):
        RefineSpec("p")
    with pytest.raises(IndicatorError, match="exactly one"):
        RefineSpec("p", threshold=1.0, quantile=0.5)
    with pytest.raises(IndicatorError, match="quantile"):
        RefineSpec("p", quantile=1.5)
    with pytest.raises(IndicatorError, match="direction"):
        RefineSpec("p", threshold=1.0, direction="x")
    with pytest.raises(IndicatorError, match="syntax"):
        RefineSpec("p +", threshold=1.0)
    with pytest.raises(IndicatorError, match=r"did you mean 'top_fraction'"):
        RefineSpec.from_dict({"indicator": "p", "top_fractoin": 0.1})
    spec = RefineSpec.from_dict({"indicator": "p", "quantile": 0.9, "n_max": 3})
    assert spec.n_max == 3

"""C2: the one-pass structural-pattern assembly equals the dense-block assembly."""

import jax.numpy as jnp
import numpy as np

from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import rectangle
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.solver import (
    JacobianPattern,
    block_diagonal,
    first_order_jacobian,
    replace_rows,
)

LID = {
    "bottom": {"type": "wall"},
    "right": {"type": "wall"},
    "left": {"type": "wall"},
    "top": {"type": "wall", "velocity": (1.0, 0.0)},
}


def _problem():
    mesh = rectangle(7, 6, cell_type="quad", cluster_x="both", beta=1.0)
    model = inc.IncompressibleModel(density=1.0, viscosity=1e-2)
    return IncompressibleProblem(mesh, model, LID)


def _reference(prob, w, beta2):
    """Dense-block assembly + sparse add + replace_rows (the 1.1 path)."""
    j = first_order_jacobian(
        prob.geom,
        prob._fi,
        prob._fb,
        jnp.asarray(w),
        params=beta2,
        b_args=(prob.bc.kind, prob.bc.value),
        pattern=JacobianPattern(prob.geom, prob.model.n_var),
    )
    return replace_rows(j, [prob.model.pin[0] * prob.model.n_var + inc.P])


def test_structural_assembly_equals_dense_assembly_with_pin():
    prob = _problem()
    assert prob.model.pin is not None
    rng = np.random.default_rng(0)
    w = 0.3 + rng.uniform(-0.2, 0.2, (prob.mesh.n_cells, 3))
    beta2 = prob.beta2(w)
    new = prob.jacobian(w, beta2)
    ref = _reference(prob, w, beta2)
    dense = JacobianPattern(prob.geom, prob.model.n_var)
    assert prob._pattern.indices.size <= dense.indices.size
    np.testing.assert_allclose(new.toarray(), ref.toarray(), atol=1e-14)
    # a second state (the pattern is re-used)
    w2 = 0.5 + rng.uniform(-0.3, 0.3, w.shape)
    np.testing.assert_allclose(
        prob.jacobian(w2, prob.beta2(w2)).toarray(),
        _reference(prob, w2, prob.beta2(w2)).toarray(),
        atol=1e-14,
    )


def test_pattern_widens_when_an_entry_was_missed():
    prob = _problem()
    rng = np.random.default_rng(1)
    w = 0.3 + rng.uniform(-0.2, 0.2, (prob.mesh.n_cells, 3))
    beta2 = prob.beta2(w)
    ref = _reference(prob, w, beta2)
    n = prob.model.n_var
    empty = {k: np.zeros((n, n), bool) for k in "onbd"}
    empty["d"][0, 0] = True  # keeps the pressure pin row's diagonal in place
    prob._pattern = JacobianPattern(prob.geom, n, empty)
    new = prob.jacobian(w, beta2)
    np.testing.assert_allclose(new.toarray(), ref.toarray(), atol=1e-14)
    assert prob._pattern.masks["o"].any()  # widened


def test_block_diagonal_source_blocks_are_added_on_the_diagonal():
    prob = _problem()
    n = prob.model.n_var
    rng = np.random.default_rng(2)
    w = 0.3 + rng.uniform(-0.2, 0.2, (prob.mesh.n_cells, 3))
    beta2 = prob.beta2(w)
    base = prob.jacobian(w, beta2)
    blocks = rng.standard_normal((prob.mesh.n_cells, n, n))
    pat = prob._pattern
    pat.widen({"d": np.ones((n, n), bool)})
    with_src = first_order_jacobian(
        prob.geom,
        prob._fi,
        prob._fb,
        jnp.asarray(w),
        params=beta2,
        b_args=(prob.bc.kind, prob.bc.value),
        pattern=pat,
        diag_blocks=blocks,
    )
    expected = first_order_jacobian(
        prob.geom,
        prob._fi,
        prob._fb,
        jnp.asarray(w),
        params=beta2,
        b_args=(prob.bc.kind, prob.bc.value),
        pattern=JacobianPattern(prob.geom, n),
    ) + block_diagonal(blocks)
    np.testing.assert_allclose(with_src.toarray(), expected.toarray(), atol=1e-14)
    assert base.shape == with_src.shape

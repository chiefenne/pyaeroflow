"""PARDISO session: same answers as SuperLU, symbolic reuse, stale-handle guard."""

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("pypardiso")

from pyaeroflow.solver.linear import (  # noqa: E402
    DirectLU,
    LinearSolverError,
    PardisoLU,
)


def _matrix(shift, n=400):
    a = sp.random(n, n, density=0.01, random_state=3, format="csr")
    return (a + sp.eye(n) * (5.0 + shift)).tocsr()


def test_matches_superlu_over_refactorizations():
    rng = np.random.default_rng(0)
    for shift in (0.0, 1.0, 2.0):  # same pattern: phase 22 after the first
        a = _matrix(shift)
        b = rng.random(a.shape[0])
        x = PardisoLU(a).solve(b)
        assert np.allclose(x, DirectLU(a).solve(b), atol=1e-10)


def test_new_pattern_is_reanalysed():
    b = np.ones(400)
    PardisoLU(_matrix(0.0)).solve(b)
    a = _matrix(0.0) + sp.diags([np.ones(399)], [7], shape=(400, 400))
    a = a.tocsr()
    assert np.allclose(PardisoLU(a).solve(b), DirectLU(a).solve(b), atol=1e-10)


def test_replaced_factorization_is_refused():
    old = PardisoLU(_matrix(0.0))
    PardisoLU(_matrix(1.0))
    with pytest.raises(LinearSolverError):
        old.solve(np.ones(400))

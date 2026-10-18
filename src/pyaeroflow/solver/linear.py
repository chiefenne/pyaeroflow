"""Linear solvers for the implicit iterations (D32).

Solvers for the sparse system ``A x = b`` with ``A = T + J1``:

- ``"direct"``: sparse LU (SciPy SuperLU). Robust, no parameters, always
  available; single-threaded, memory and time grow faster than the mesh size.
- ``"pardiso"``: Intel MKL PARDISO through the optional ``pypardiso``
  package (multithreaded; x86 only).
- ``"umfpack"``: UMFPACK through the optional ``scikit-umfpack`` package
  (multifrontal, BLAS-threaded).
- ``"gmres-ilu"``: restarted GMRES preconditioned by an incomplete LU
  (SciPy ``spilu``). Measured no cheaper than the exact LU at any useful
  accuracy on a 100k-cell airfoil mesh; kept for experiments.

Both also serve as the preconditioner of the matrix-free Newton-Krylov
solve (:func:`solve_matrix_free`), where the operator is ``T + J`` with the
exact Jacobian applied through ``jax.jvp``.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

SOLVERS = ("direct", "pardiso", "umfpack", "gmres-ilu")


class LinearSolverError(RuntimeError):
    """Raised when the linear system cannot be solved."""


@dataclass(frozen=True)
class LinearSettings:
    """Settings of the linear solver.

    Attributes
    ----------
    type : {"direct", "pardiso", "umfpack", "gmres-ilu"}
    tolerance : float
        Relative residual tolerance of GMRES (``"gmres-ilu"`` only).
    restart : int
        GMRES restart length.
    max_iterations : int
        Maximum GMRES iterations, i.e. matrix-vector products, rounded up to
        whole restart cycles (``"gmres-ilu"`` only).
    ilu_drop_tolerance : float
        Drop tolerance of the incomplete LU.
    ilu_fill_factor : float
        Fill factor of the incomplete LU.
    ordering : {"mmd-at-plus-a", "colamd"}
        SuperLU column ordering. The minimum-degree ordering of A + A^T
        gives the least fill for the mean-flow Jacobian (symmetric pattern);
        with the coupled turbulence Jacobian its threshold pivoting inflates
        the fill by 100x, where COLAMD stays at about 20x (D34).

    The ILU defaults were set on the 64x64 cavity at Re 1000: drop 1e-5 with
    fill 20 needs about as many GMRES iterations as the exact LU (3 per
    approximate-Jacobian step, 20 per Newton-Krylov step), while drop 1e-4
    with fill 10 needs 70 and 170.
    """

    type: str = "direct"
    tolerance: float = 1.0e-6
    restart: int = 50
    max_iterations: int = 500
    ilu_drop_tolerance: float = 1.0e-5
    ilu_fill_factor: float = 20.0
    ordering: str = "mmd-at-plus-a"


# SuperLU options. The Jacobian has a symmetric sparsity pattern, for which
# the minimum-degree ordering of A^T + A gives about half the fill and
# factorization time of the default COLAMD (37k-cell mesh: fill 10.1x vs
# 19.7x of nnz(A), 0.9 s vs 1.8 s). With this ordering, row pivoting must be
# restricted to keep the fill: the default threshold (1.0, partial pivoting)
# let the fill grow to >100x on a converged Euler state, 0.1 keeps it at 7x
# (diagonal pivots are used unless smaller than 0.1 times the column maximum).
SUPERLU_OPTIONS = {"permc_spec": "MMD_AT_PLUS_A", "diag_pivot_thresh": 0.1}
ORDERINGS = {"mmd-at-plus-a": "MMD_AT_PLUS_A", "colamd": "COLAMD"}


def superlu_options(settings):
    """SuperLU keyword arguments for the given settings."""
    return {**SUPERLU_OPTIONS, "permc_spec": ORDERINGS[settings.ordering]}


class DirectLU:
    """Sparse LU factorization of a matrix."""

    def __init__(self, matrix, settings=None):
        try:
            opts = superlu_options(settings or LinearSettings())
            self._lu = spla.splu(matrix.tocsc(), **opts)
        except RuntimeError as exc:  # "Factor is exactly singular"
            raise LinearSolverError(str(exc)) from exc
        self.iterations = 0

    def solve(self, rhs):
        """Solve ``A x = rhs``."""
        return _finite(self._lu.solve(rhs))

    def precondition(self, rhs):
        """Apply the preconditioner (here: the exact inverse)."""
        return self._lu.solve(rhs)


class PardisoLU:
    """Sparse LU by Intel MKL PARDISO (optional dependency ``pypardiso``)."""

    def __init__(self, matrix, settings=None):
        try:
            import pypardiso
        except ImportError as exc:
            raise LinearSolverError(
                "linear_solver.type = 'pardiso' needs the pypardiso package "
                "(pip install pypardiso, or conda install -c conda-forge pypardiso)"
            ) from exc
        self._matrix = matrix.tocsr()
        self._matrix.sort_indices()
        self._solver = pypardiso.PyPardisoSolver()
        try:
            self._solver.factorize(self._matrix)
        except Exception as exc:  # pypardiso.PyPardisoError on a singular matrix
            self._solver.free_memory(everything=True)
            raise LinearSolverError(str(exc)) from exc
        self.iterations = 0

    def solve(self, rhs):
        """Solve ``A x = rhs``."""
        return _finite(self.precondition(rhs))

    def precondition(self, rhs):
        """Apply the factorization."""
        return self._solver.solve(self._matrix, rhs).reshape(rhs.shape)

    def __del__(self):
        """Release PARDISO's internal memory."""
        solver = getattr(self, "_solver", None)
        if solver is not None:
            try:
                solver.free_memory(everything=True)
            except Exception:  # noqa: BLE001 - interpreter shutdown
                pass


class UmfpackLU:
    """Sparse LU by UMFPACK (optional dependency ``scikit-umfpack``)."""

    def __init__(self, matrix, settings=None):
        try:
            import scikits.umfpack as umfpack
        except ImportError as exc:
            raise LinearSolverError(
                "linear_solver.type = 'umfpack' needs the scikit-umfpack package "
                "(conda install -c conda-forge scikit-umfpack)"
            ) from exc
        try:
            self._lu = umfpack.splu(matrix.tocsc())
        except Exception as exc:  # singular matrix
            raise LinearSolverError(str(exc)) from exc
        self.iterations = 0

    def solve(self, rhs):
        """Solve ``A x = rhs``."""
        return _finite(self._lu.solve(rhs))

    def precondition(self, rhs):
        """Apply the factorization."""
        return self._lu.solve(rhs)


class GmresILU:
    """GMRES with an incomplete-LU preconditioner."""

    def __init__(self, matrix, settings):
        self._matrix = matrix.tocsc()
        self._settings = settings
        try:
            self._ilu = spla.spilu(
                self._matrix,
                drop_tol=settings.ilu_drop_tolerance,
                fill_factor=settings.ilu_fill_factor,
                **superlu_options(settings),
            )
        except RuntimeError as exc:
            raise LinearSolverError(str(exc)) from exc
        self.iterations = 0

    def solve(self, rhs):
        """Solve ``A x = rhs`` with preconditioned GMRES."""
        s = self._settings
        x, self.iterations = solve_matrix_free(
            self._matrix.dot,
            rhs,
            self.precondition,
            tolerance=s.tolerance,
            restart=s.restart,
            max_iterations=s.max_iterations,
        )
        return x

    def precondition(self, rhs):
        """Apply the incomplete LU as a preconditioner."""
        return self._ilu.solve(rhs)


def factorize(matrix, settings=None):
    """Return a solver object for ``settings.type``."""
    s = settings or LinearSettings()
    if s.type == "direct":
        return DirectLU(matrix, settings)
    if s.type == "pardiso":
        return PardisoLU(matrix, s)
    if s.type == "umfpack":
        return UmfpackLU(matrix, s)
    if s.type == "gmres-ilu":
        return GmresILU(matrix, s)
    raise ValueError(f"linear solver type must be one of {SOLVERS}")


def solve_matrix_free(
    apply, rhs, precondition, tolerance=1.0e-2, restart=50, max_iterations=200
):
    """Right-preconditioned restarted GMRES for an operator given as a function.

    Parameters
    ----------
    apply : callable
        ``apply(x) -> A x``.
    rhs : ndarray
    precondition : callable
        ``precondition(r) -> M^-1 r``.
    tolerance : float
        Relative residual tolerance.
    restart : int
    max_iterations : int
        Maximum matrix-vector products, rounded up to whole restart cycles.

    Returns
    -------
    x : ndarray
    iterations : int
        Number of GMRES iterations (matrix-vector products).

    Raises
    ------
    LinearSolverError
        If the solution is not finite.
    """
    n = rhs.size
    count = {"n": 0}

    def counted(x):
        count["n"] += 1
        return apply(x)

    op = spla.LinearOperator((n, n), matvec=counted, dtype=float)
    pre = spla.LinearOperator((n, n), matvec=precondition, dtype=float)
    cycles = max(1, -(-max_iterations // restart))  # maxiter counts restart cycles
    x, info = spla.gmres(
        op, rhs, M=pre, rtol=tolerance, restart=restart, maxiter=cycles
    )
    if info < 0:
        raise LinearSolverError(f"GMRES breakdown (info = {info})")
    return _finite(x), count["n"]


def solve_direct(matrix, rhs):
    """Solve ``matrix @ x = rhs`` with a sparse LU factorization (convenience)."""
    return DirectLU(matrix).solve(rhs)


def _finite(x):
    if not np.all(np.isfinite(x)):
        raise LinearSolverError("non-finite solution")
    return x

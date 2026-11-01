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


class _PardisoSession:
    """One PARDISO instance shared by all factorizations of a run.

    Stock ``pypardiso`` hashes the whole matrix (SHA-1 over ``data``, ``indices``
    and ``indptr`` once the matrix has more than 5e7 non-zeros) and rebuilds the
    1-based, int32 ``indptr``/``indices`` copies on *every* solve; at 320k cells
    x 7 unknowns (78 M non-zeros) that is about 0.7 s per preconditioner
    application. The sparsity pattern is fixed during a run, so here the index
    arrays are converted once per pattern, the solve phase (33) is called
    directly, and a new factorization with an unchanged pattern re-uses the
    reordering and symbolic analysis (phase 22 instead of 12). Results are
    identical to the stock path. Needs ``pypardiso`` 0.4.x internals
    (``set_phase``, ``_call_pardiso`` pattern); :func:`_fast_path_available`
    falls back to the public API otherwise.
    """

    def __init__(self, pypardiso):
        self.solver = pypardiso.PyPardisoSolver()
        self.pattern = None  # (indptr, indices) of the analysed matrix
        self.ia = self.ja = None
        self.generation = 0

    def factorize(self, matrix):
        """Factorize ``matrix`` (CSR, sorted indices); returns the generation."""
        sv = self.solver
        sv._check_A(matrix)
        same = (
            self.pattern is not None
            and self.pattern[0].shape == matrix.indptr.shape
            and self.pattern[1].shape == matrix.indices.shape
            and np.array_equal(self.pattern[0], matrix.indptr)
            and np.array_equal(self.pattern[1], matrix.indices)
        )
        if not same:
            self.pattern = (matrix.indptr.copy(), matrix.indices.copy())
            self.ia = matrix.indptr.astype(np.int32) + 1
            self.ja = matrix.indices.astype(np.int32) + 1
        sv.factorized_A = None  # the stock hash check is bypassed
        sv.set_phase(22 if same else 12)
        self._call(matrix, np.zeros((matrix.shape[0], 1)))
        self.generation += 1
        return self.generation

    def solve(self, matrix, rhs):
        """Back-substitution with the current factorization (phase 33)."""
        self.solver.set_phase(33)
        b = np.asarray(rhs, dtype=float)
        x = self._call(matrix, b.reshape(-1, 1) if b.ndim == 1 else b)
        return x.reshape(b.shape)

    def _call(self, matrix, b):
        import ctypes

        sv = self.solver
        x = np.zeros_like(b)
        err = ctypes.c_int32(0)
        p32 = ctypes.POINTER(ctypes.c_int32)
        pf = ctypes.POINTER(ctypes.c_double)
        sv._mkl_pardiso(
            sv.pt.ctypes.data_as(ctypes.POINTER(sv._pt_type[0])),
            ctypes.byref(ctypes.c_int32(1)),
            ctypes.byref(ctypes.c_int32(1)),
            ctypes.byref(ctypes.c_int32(sv.mtype)),
            ctypes.byref(ctypes.c_int32(sv.phase)),
            ctypes.byref(ctypes.c_int32(matrix.shape[0])),
            matrix.data.ctypes.data_as(pf),
            self.ia.ctypes.data_as(p32),
            self.ja.ctypes.data_as(p32),
            sv.perm.ctypes.data_as(p32),
            ctypes.byref(ctypes.c_int32(1 if b.ndim == 1 else b.shape[1])),
            sv.iparm.ctypes.data_as(p32),
            ctypes.byref(ctypes.c_int32(sv.msglvl)),
            b.ctypes.data_as(pf),
            x.ctypes.data_as(pf),
            ctypes.byref(err),
        )
        if err.value != 0:
            raise RuntimeError(f"PARDISO error {err.value}")
        return np.ascontiguousarray(x)

    def free(self):
        """Release PARDISO's internal memory."""
        try:
            self.solver.free_memory(everything=True)
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass
        self.pattern = None


_SESSION = {"obj": None}


def _fast_path_available(pypardiso):
    """Tell whether the installed pypardiso has the internals the session relies on."""
    sv_cls = pypardiso.PyPardisoSolver
    return all(
        hasattr(sv_cls, a)
        for a in ("set_phase", "_call_pardiso", "_check_A", "free_memory")
    ) and pypardiso.__version__.startswith("0.4")


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
        self._session = None
        try:
            if _fast_path_available(pypardiso):
                session = _SESSION["obj"]
                if session is None:
                    session = _SESSION["obj"] = _PardisoSession(pypardiso)
                self._generation = session.factorize(self._matrix)
                self._session = session
            else:
                self._solver = pypardiso.PyPardisoSolver()
                self._solver.factorize(self._matrix)
        except Exception as exc:  # pypardiso.PyPardisoError on a singular matrix
            if self._session is None and hasattr(self, "_solver"):
                self._solver.free_memory(everything=True)
            elif _SESSION["obj"] is not None:
                _SESSION["obj"].free()
                _SESSION["obj"] = None
            raise LinearSolverError(str(exc)) from exc
        self.iterations = 0

    def solve(self, rhs):
        """Solve ``A x = rhs``."""
        return _finite(self.precondition(rhs))

    def precondition(self, rhs):
        """Apply the factorization."""
        if self._session is not None:
            if self._session.generation != self._generation:
                raise LinearSolverError("PARDISO factorization was replaced")
            return self._session.solve(self._matrix, rhs)
        return self._solver.solve(self._matrix, rhs).reshape(rhs.shape)

    def __del__(self):
        """Release PARDISO's internal memory (stock path only)."""
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
    """Restarted GMRES for an operator given as a function.

    SciPy's GMRES is left-preconditioned: it stops when the preconditioned
    residual and the true residual have both dropped by ``tolerance``.

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

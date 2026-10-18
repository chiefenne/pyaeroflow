"""Residual assembly, Jacobian, linear solvers, pseudo-time and dual time stepping."""

from pyaeroflow.solver.dual_time import (
    DualTimeSettings,
    UnsteadyResult,
    bdf_coefficients,
    solve_unsteady,
)
from pyaeroflow.solver.jacobian import (
    JacobianPattern,
    block_diagonal,
    first_order_jacobian,
    replace_rows,
)
from pyaeroflow.solver.linear import (
    LinearSettings,
    LinearSolverError,
    factorize,
    solve_direct,
    solve_matrix_free,
)
from pyaeroflow.solver.pseudo_time import (
    PseudoTimeSettings,
    SteadyResult,
    rms,
    rms_per_variable,
    solve_explicit,
    solve_steady,
)
from pyaeroflow.solver.residual import divergence

__all__ = [
    "DualTimeSettings",
    "LinearSettings",
    "LinearSolverError",
    "PseudoTimeSettings",
    "SteadyResult",
    "UnsteadyResult",
    "bdf_coefficients",
    "block_diagonal",
    "divergence",
    "factorize",
    "JacobianPattern",
    "first_order_jacobian",
    "replace_rows",
    "rms",
    "rms_per_variable",
    "solve_direct",
    "solve_explicit",
    "solve_matrix_free",
    "solve_steady",
    "solve_unsteady",
]

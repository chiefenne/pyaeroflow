"""Convergence history as CSV (D45)."""

import csv
from pathlib import Path


class HistoryWriter:
    """Append one CSV row per iteration.

    Parameters
    ----------
    path : Path
    residual_names : sequence of str
        One column ``residual_<name>`` per variable.
    """

    def __init__(self, path, residual_names, coefficient_names=()):
        self.path = Path(path)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "iteration",
                *(f"residual_{n}" for n in residual_names),
                "cfl",
                "accepted",
                "newton_krylov",
                "linear_iterations",
                *coefficient_names,
                "time_s",
            ]
        )

    def row(
        self,
        it,
        residuals,
        cfl,
        accepted,
        elapsed,
        newton_krylov=False,
        linear=0,
        coefficients=(),
    ):
        """Write one row."""
        self._writer.writerow(
            [
                it,
                *(f"{r:.6e}" for r in residuals),
                f"{cfl:.6e}",
                int(accepted),
                int(newton_krylov),
                int(linear),
                *(f"{c:.8e}" for c in coefficients),
                f"{elapsed:.3f}",
            ]
        )
        self._file.flush()

    def close(self):
        """Close the file."""
        self._file.close()


class StepHistoryWriter:
    """Append one CSV row per physical time step (unsteady runs, D45).

    Columns: step, time, residual per variable (inner residual at the end
    of the step), inner_iterations, inner_residual_drop, inner_converged,
    linear_iterations, the coefficients, time_s.
    """

    def __init__(self, path, residual_names, coefficient_names=()):
        self.path = Path(path)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "step",
                "time",
                *(f"residual_{n}" for n in residual_names),
                "inner_iterations",
                "inner_residual_drop",
                "inner_converged",
                "linear_iterations",
                *coefficient_names,
                "time_s",
            ]
        )

    def row(
        self,
        step,
        time,
        residuals,
        inner,
        drop,
        converged,
        linear,
        coefficients,
        elapsed,
    ):
        """Write one row."""
        self._writer.writerow(
            [
                step,
                f"{time:.10e}",
                *(f"{r:.6e}" for r in residuals),
                int(inner),
                f"{drop:.3f}",
                int(converged),
                int(linear),
                *(f"{c:.8e}" for c in coefficients),
                f"{elapsed:.3f}",
            ]
        )
        self._file.flush()

    def close(self):
        """Close the file."""
        self._file.close()

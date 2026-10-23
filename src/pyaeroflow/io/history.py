"""Convergence history as CSV (D45)."""

import csv
from pathlib import Path


def _open_history(path, header, resume_after):
    """Open the history CSV, new or continued; returns (file, writer).

    With ``resume_after`` (the iteration or step a restart continues after)
    and an existing file of the same columns, the file is kept and appended
    to; rows beyond ``resume_after`` (a restart from an older file) are moved
    to ``<name>.cut_<n>.csv`` first, so the iterations stay unique. A file
    with other columns is renamed to ``<name>.before_restart.csv``.
    """
    path = Path(path)
    head = ",".join(header)
    if resume_after is not None and path.exists():
        lines = path.read_text().splitlines()
        if lines and lines[0] == head:
            keep = [lines[0]]
            cut = []
            for line in lines[1:]:
                number = int(float(line.split(",")[0]))
                (keep if number <= resume_after else cut).append(line)
            if cut:
                path.with_name(f"{path.stem}.cut_{resume_after + 1}.csv").write_text(
                    "\n".join([lines[0], *cut]) + "\n"
                )
                path.write_text("\n".join(keep) + "\n")
            file = open(path, "a", newline="")
            return file, csv.writer(file)
        path.replace(path.with_name(f"{path.stem}.before_restart{path.suffix}"))
    file = open(path, "w", newline="")
    writer = csv.writer(file)
    writer.writerow(header)
    return file, writer


class HistoryWriter:
    """Append one CSV row per iteration.

    Parameters
    ----------
    path : Path
    residual_names : sequence of str
        One column ``residual_<name>`` per variable.
    diagnostic_names : sequence of str
        Extra columns after the coefficients (transition diagnostics).
    resume_after : int, optional
        Continue an existing file after this iteration (restart).
    """

    def __init__(
        self,
        path,
        residual_names,
        coefficient_names=(),
        diagnostic_names=(),
        resume_after=None,
    ):
        self.path = Path(path)
        self._file, self._writer = _open_history(
            self.path,
            [
                "iteration",
                *(f"residual_{n}" for n in residual_names),
                "cfl",
                "accepted",
                "newton_krylov",
                "linear_iterations",
                *coefficient_names,
                *diagnostic_names,
                "time_s",
            ],
            resume_after,
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
        diagnostics=(),
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
                *(f"{d:.6e}" if isinstance(d, float) else d for d in diagnostics),
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

    def __init__(
        self,
        path,
        residual_names,
        coefficient_names=(),
        diagnostic_names=(),
        resume_after=None,
    ):
        self.path = Path(path)
        self._file, self._writer = _open_history(
            self.path,
            [
                "step",
                "time",
                *(f"residual_{n}" for n in residual_names),
                "inner_iterations",
                "inner_residual_drop",
                "inner_converged",
                "linear_iterations",
                *coefficient_names,
                *diagnostic_names,
                "time_s",
            ],
            resume_after,
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
        diagnostics=(),
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
                *(f"{d:.6e}" if isinstance(d, float) else d for d in diagnostics),
                f"{elapsed:.3f}",
            ]
        )
        self._file.flush()

    def close(self):
        """Close the file."""
        self._file.close()

"""Human-readable run log (D45).

Written to the console and to ``<case>.log`` with fixed-width columns,
readable with ``tail -f`` or ``more``.

Layout for steady runs: one line per pseudo-time iteration. (Unsteady runs,
stage 4, write one line per physical time step.)
"""

import sys
import time
from pathlib import Path

import numpy as np


class RunLog:
    """Write messages and iteration lines to the console and a log file.

    Parameters
    ----------
    path : Path or None
        Log file; None writes to the console only.
    console : bool
        Echo to stdout.
    """

    def __init__(self, path=None, console=True):
        self.path = Path(path) if path is not None else None
        self._file = open(self.path, "w") if self.path is not None else None
        self._console = console
        self._t0 = time.perf_counter()
        self._columns = None

    def close(self):
        """Close the log file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        """Support ``with RunLog(...) as log``."""
        return self

    def __exit__(self, *exc):
        """Close the file when leaving the ``with`` block."""
        self.close()

    def write(self, text=""):
        """Write a line."""
        if self._console:
            print(text)
            sys.stdout.flush()
        if self._file is not None:
            self._file.write(text + "\n")
            self._file.flush()

    def section(self, title):
        """Write a section heading."""
        self.write("")
        self.write(title)
        self.write("-" * len(title))

    def table(self, rows, indent=2):
        """Write ``key: value`` rows aligned on the colon."""
        width = max((len(k) for k, _ in rows), default=0)
        for key, value in rows:
            self.write(" " * indent + f"{key:<{width}}  {value}")

    def iteration_header(self, residual_names, extra=()):
        """Write the column header of the iteration lines.

        ``extra`` names additional columns, each a name or a ``(name, width)``
        pair (default width 7), e.g. the Newton-Krylov phase flag, the GMRES
        iteration count and force coefficients.
        """
        self._columns = list(residual_names)
        self._extra = [e if isinstance(e, tuple) else (e, 7) for e in extra]
        cols = "".join(f"{'log(R_' + n + ')':>12}" for n in self._columns)
        more = "".join(f"{name:>{width}}" for name, width in self._extra)
        self.write("")
        self.write(
            f"{'iter':>6}{cols}{'drop':>7}{'CFL':>10}{more}{'wall[s]':>10}  note"
        )

    def iteration(
        self, it, residuals, cfl, accepted=True, extra=(), note="", drop=None
    ):
        """Write one iteration line.

        ``drop`` is the per-equation convergence measure in orders of
        magnitude (the worst equation's residual relative to its own peak).
        """
        with np.errstate(divide="ignore"):
            logs = np.log10(np.asarray(residuals, dtype=float))
        cols = "".join(f"{v:>12.4f}" for v in logs)
        cols += f"{drop:>7.2f}" if drop is not None else f"{'':>7}"
        widths = [w for _, w in self._extra] + [7] * len(extra)
        more = "".join(f"{str(e):>{w}}" for e, w in zip(extra, widths, strict=False))
        elapsed = time.perf_counter() - self._t0
        flag = note if accepted else ("rejected " + note).strip()
        self.write(f"{it:>6d}{cols}{cfl:>10.2e}{more}{elapsed:>10.1f}  {flag}".rstrip())

    def step_header(self, residual_names, extra=()):
        """Write the column header of the time-step lines (unsteady runs).

        One line per physical time step: step, time, inner iterations used,
        inner residual drop reached, residuals, ``extra`` columns (as in
        :meth:`iteration_header`), elapsed time.
        """
        self._columns = list(residual_names)
        self._extra = [e if isinstance(e, tuple) else (e, 7) for e in extra]
        cols = "".join(f"{'log(R_' + n + ')':>12}" for n in self._columns)
        more = "".join(f"{name:>{width}}" for name, width in self._extra)
        self.write("")
        self.write(
            f"{'step':>6}{'time':>12}{'inner':>6}{'drop':>6}{cols}{more}"
            f"{'wall[s]':>10}  note"
        )

    def step(self, step, t, inner, drop, residuals, extra=(), note=""):
        """Write one time-step line."""
        with np.errstate(divide="ignore"):
            logs = np.log10(np.asarray(residuals, dtype=float))
        cols = "".join(f"{v:>12.4f}" for v in logs)
        widths = [w for _, w in self._extra] + [7] * len(extra)
        more = "".join(f"{str(e):>{w}}" for e, w in zip(extra, widths, strict=False))
        elapsed = time.perf_counter() - self._t0
        self.write(
            f"{step:>6d}{t:>12.5e}{inner:>6d}{drop:>6.1f}{cols}{more}"
            f"{elapsed:>10.1f}  {note}".rstrip()
        )

    @property
    def elapsed(self):
        """Seconds since the log was opened."""
        return time.perf_counter() - self._t0

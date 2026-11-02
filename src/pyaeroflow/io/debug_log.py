"""Numerics debug log of steady implicit runs (``output.debug_log = true``).

One block per pseudo-time iteration in ``<case>.debug.log``, from the record
that :func:`pyaeroflow.solver.solve_steady` passes to its ``debug`` hook:

- the linear solve: Newton-Krylov or direct, GMRES iterations, the GMRES
  tolerance and the true linear residual |(T + J) du + R| / |R| (one extra
  Jacobian product per Newton step), whether the preconditioner was rebuilt;
- a re-evaluation of the reference residual after the operator changed
  (limiter release, farfield vortex update), with the RMS before and after;
- the reference residual: plain RMS (the norm the line search and the step
  acceptance use) and each variable's share of its square;
- the update: cells changed by the step limiter (clip, under-relaxation) per
  variable, the largest |du| per variable and where;
- every line-search trial: RMS and the ratio new / reference per variable;
- the outcome, and the cells with the largest |R| after the step for the
  variables that dominate the RMS, with their share of that variable's RMS².

The log costs one Jacobian product per Newton step and a few array passes;
it is meant for investigations, not for production runs.
"""

from pathlib import Path

import numpy as np

TOP_CELLS = 3  # cells listed per variable
TOP_SHARE = 10  # cells whose share of the variable's RMS^2 is reported
TOP_VARIABLES = 3  # variables with the largest share of RMS^2 that get cells


class NumericsDebugLog:
    """Write the per-iteration numerics record of :func:`solve_steady`.

    Parameters
    ----------
    path : Path
        Output file (overwritten, or appended to with ``append``).
    names : sequence of str
        Variable names (the residual columns of the run log).
    centroids : ndarray, shape (n_cells, 2)
        Cell centres for the locations (non-dimensional, as the solver's).
    length : float
        Reference length that turns ``centroids`` into metres.
    append : bool
        Continue an existing file (restart of the same case).
    """

    def __init__(self, path, names, centroids, length=1.0, append=False):
        self.path = Path(path)
        self.names = list(names)
        self.xy = np.asarray(centroids, dtype=float) * float(length)
        self._file = open(self.path, "a" if append else "w")  # noqa: SIM115
        self._file.write(
            "# PyAeroFlow numerics debug log: one block per iteration. RMS is the "
            "plain norm of the whole residual that the line search and the step "
            "acceptance use; 'share' is each variable's part of RMS^2.\n"
        )

    def close(self):
        """Close the file."""
        self._file.close()

    def _write(self, text):
        self._file.write(text + "\n")

    def _per_variable(self, values, fmt):
        return "  ".join(
            f"{n} {format(v, fmt)}" for n, v in zip(self.names, values, strict=True)
        )

    def _cells(self, r, j):
        """Lines with the largest |R| of variable ``j``."""
        a = np.abs(np.asarray(r)[:, j])
        total = float(np.sum(a**2))
        if total <= 0.0 or not np.isfinite(total):
            return f"    {self.names[j]}: no finite residual"
        k = min(TOP_SHARE, a.size)
        top = np.argpartition(a, -k)[-k:]
        top = top[np.argsort(-a[top])]
        share = float(np.sum(a[top] ** 2)) / total
        cells = ", ".join(
            f"cell {c} ({self.xy[c, 0]:.4f}, {self.xy[c, 1]:.4f}) |R| {a[c]:.2e}"
            for c in top[:TOP_CELLS]
        )
        return (
            f"    {self.names[j]}: top {k} cells hold {100 * share:.0f} % of its "
            f"RMS^2; {cells}"
        )

    def __call__(self, rec):
        """Write the block of one iteration."""
        w = self._write
        head = f"it {rec['it']}  CFL {rec['cfl']:.3e}"
        if "nk" in rec:
            if rec["nk"]:
                head += f"  Newton-Krylov, GMRES {rec['gmres']}"
                if "linear_residual" in rec:
                    head += (
                        f", tolerance {rec['tolerance']:.2e}, true linear "
                        f"residual {rec['linear_residual']:.3e}"
                    )
            else:
                head += "  direct solve (approximate Jacobian)"
            if rec.get("factorized"):
                head += "  [preconditioner rebuilt]"
        w(head)
        if "re_evaluated" in rec:
            before, after = rec["re_evaluated"]
            w(
                f"  reference re-evaluated (operator changed: limiter or farfield "
                f"vortex): RMS {before:.4e} -> {after:.4e} "
                f"(x{after / max(before, 1e-300):.4f})"
            )
        if "reference" in rec:
            ref = rms_per_variable(rec["reference"])
            sq = ref**2
            share = sq / max(float(np.sum(sq)), 1e-300)
            w(f"  reference RMS {rec['reference_rms']:.4e}")
            w(f"    per variable  {self._per_variable(ref, '.2e')}")
            w(f"    share of RMS^2  {self._per_variable(100 * share, '.1f')} (%)")
        if "du" in rec:
            du = np.abs(rec["du"])
            where = np.argmax(du, axis=0)
            w(
                "  update: cells changed by the step limiter  "
                + self._per_variable(rec["limited"], "d")
            )
            w(
                "    largest |du|  "
                + "  ".join(
                    f"{n} {du[c, j]:.2e} ({self.xy[c, 0]:.3f}, {self.xy[c, 1]:.3f})"
                    for j, (n, c) in enumerate(zip(self.names, where, strict=True))
                )
            )
        if "reference" in rec:
            ref = rms_per_variable(rec["reference"])
            for factor, total, per in rec.get("trials", []):
                label = "1" if factor >= 1.0 else f"1/{round(1.0 / factor)}"
                ratio = per / np.maximum(ref, 1e-300)
                verdict = (
                    "not finite"
                    if not np.isfinite(total)
                    else ("lower" if total <= rec["reference_rms"] else "HIGHER")
                )
                w(
                    f"  trial step {label:<4} RMS {total:.4e} "
                    f"(x{total / max(rec['reference_rms'], 1e-300):.4f}, {verdict}); "
                    f"new/ref  {self._per_variable(ratio, '.3f')}"
                )
        if "error" in rec:
            w(f"  linear solver failed: {rec['error']}")
        step = rec.get("step", 1.0)
        label = "1" if step >= 1.0 else f"1/{round(1.0 / step)}"
        w(f"  {'accepted' if rec['accepted'] else 'REJECTED'}, step {label}")
        if "new" in rec:
            new = np.asarray(rec["new"])
            per = rms_per_variable(new)
            if np.all(np.isfinite(per)):
                order = np.argsort(-(per**2))[:TOP_VARIABLES]
                w("  largest residuals after the step:")
                for j in order:
                    w(self._cells(new, j))
        self._file.flush()


def rms_per_variable(r):
    """RMS of each column."""
    r = np.asarray(r, dtype=float)
    with np.errstate(over="ignore"):
        return np.sqrt(np.mean(np.square(r), axis=0))

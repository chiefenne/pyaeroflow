"""``pyaeroflow plot log``: convergence plots of a run (successor of plot_log.py).

Input: ``<name>.log`` (all columns and the notes) or, without a log,
``<name>.history.csv`` (no notes; unsteady: the inner drop). Steady runs
plot over the iteration, unsteady runs over the physical time. Four PNG
files, each panel only if its columns exist:

- ``<prefix>_residuals.png``: flow residuals (left axis), turbulence and
  transition residuals (right axis, dashed).
- ``<prefix>_solver.png``: drop, CFL (log, capped), GMRES iterations, the
  line-search step (if a step was cut: notes "step 1/N"), wall time;
  unsteady: inner iterations instead of CFL.
- ``<prefix>_coeffs.png``: CL, CD, CM. Steady: y-limits fitted from 20
  iterations before "limiter released" (or set by ``[scale.KEY]`` of the
  targets file). Unsteady: plus the running time average from ``--average-from``.
- ``<prefix>_diffs.png``: consecutive differences of drop, CFL and GMRES.
"""

import re
from pathlib import Path

import numpy as np

from pyaeroflow.plotting.common import draw_target, fit_ylim, load_targets, pyplot, save
from pyaeroflow.plotting.readers import notes_for, read_history, read_log

FLOW = {"R_p", "R_u", "R_v", "R_rho", "R_rhou", "R_rhov", "R_rhoE"}
LIMITER_TAG = "limiter released"
ZOOM_LEAD = 20  # iterations before the limiter release in the coefficient window
CFL_MAX = 1e5
DCFL_MAX = 8e3
STEP_NOTE = re.compile(r",?\s*\bstep 1/(\d+)")
PANELS = ("residuals", "solver", "coeffs", "diffs")


def load_run(path):
    """Columns, notes and mode of a log or history CSV.

    A history CSV takes the notes of ``<name>.log`` next to it, if any.
    Steady logs get the line-search step fraction as column ``ls_step``
    (1 = full step) from the "step 1/N" notes, which leave the notes.
    """
    path = Path(path)
    if path.name.endswith(".history.csv"):
        run = read_history(path)
        log = path.with_name(path.name[: -len(".history.csv")] + ".log")
        key = "step" if run.mode == "unsteady" else "iter"
        if log.exists():
            notes = notes_for(log, run.columns[key], step=run.mode == "unsteady")
            run = run._replace(notes=notes)
    else:
        run = read_log(path)
    if run.mode == "steady":
        frac, notes = [], []
        for note in run.notes:
            cut = STEP_NOTE.search(note)
            frac.append(1.0 / int(cut.group(1)) if cut else 1.0)
            notes.append(STEP_NOTE.sub("", note).strip(" ,"))
        columns = dict(run.columns)
        if any(f < 1.0 for f in frac):
            columns["ls_step"] = np.array(frac)
        run = run._replace(columns=columns, notes=notes)
    return run


def _mark_notes(axes, x, notes):
    for xi, note in zip(x, notes, strict=True):
        if note:
            for a in axes:
                a.axvline(xi, color="k", ls=":", lw=0.7, alpha=0.6)
            axes[0].annotate(
                note,
                (xi, 1.0),
                xycoords=("data", "axes fraction"),
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=7,
            )


def _finish(fig, axes, x, notes, xlabel, show_notes):
    if show_notes:
        _mark_notes(axes, x, notes)
    axes[-1].set_xlabel(xlabel)
    if x.size:
        axes[-1].set_xlim(x.min(), x.max())


def _stack(n):
    plt = pyplot()
    fig, axes = plt.subplots(
        n, 1, sharex=True, figsize=(13, 2.4 * n + 1), gridspec_kw={"hspace": 0.08}
    )
    return fig, np.atleast_1d(axes)


def _plot_residuals(d, residuals, x, ctx):
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(13, 6))
    left = [k for k in residuals if k in FLOW]
    right = [k for k in residuals if k not in FLOW]
    lines = []
    for k, key in enumerate(left):
        lab = f"log({key})"
        lines += ax.plot(x, d[key], marker=".", ms=4, lw=1, color=f"C{k}", label=lab)
        t = draw_target(ax, key, ctx["targets"], f"C{k}")
        if t is not None:
            t.set_label(f"{lab} {t.get_label()}")
            lines.append(t)
    ax.set_ylabel(", ".join(f"log({k})" for k in left))
    if right:
        ax2 = ax.twinx()
        for k, key in enumerate(right):
            c, lab = f"C{len(left) + k}", f"log({key})"
            lines += ax2.plot(
                x, d[key], marker=".", ms=4, lw=1.2, ls="--", color=c,
                label=f"{lab}  [right axis]",
            )  # fmt: skip
            t = draw_target(ax2, key, ctx["targets"], c)
            if t is not None:
                t.set_label(f"{lab} {t.get_label()}")
                lines.append(t)
        ax2.set_ylabel(", ".join(f"log({k})" for k in right))
    ax.legend(
        lines, [ln.get_label() for ln in lines], loc="upper right", ncol=2, fontsize=9
    )
    _finish(fig, [ax], x, ctx["notes"], ctx["xlabel"], ctx["show_notes"])
    return save(fig, ctx["prefix"] + "_residuals.png", f"Residuals: {ctx['name']}")


def _plot_panels(d, x, specs, ctx, suffix, title, window=None, average_from=None):
    specs = [s for s in specs if s[0] in d and np.isfinite(d[s[0]]).any()]
    if not specs:
        print(f"skipped {ctx['prefix']}{suffix}: no data")
        return None
    fig, axes = _stack(len(specs))
    targets, scales = ctx["targets"], ctx["scales"]
    for k, (a, (key, lab, logy)) in enumerate(zip(axes, specs, strict=True)):
        y, c = d[key], f"C{k}"
        if key in ("GMRES", "ls_step", "inner"):
            a.step(x, y, where="mid", color=c, lw=1)
            a.plot(x, y, ".", ms=4, color=c)
        else:
            a.plot(x, y, marker=".", ms=4, lw=1, color=c)
        if key == "ls_step":  # 1, 1/2, 1/4 ... on a log2 axis
            n = int(round(-np.log2(np.nanmin(y))))
            a.set_yscale("log", base=2)
            a.set_ylim(2.0 ** -(n + 0.5), 1.4)
            a.set_yticks([2.0**-j for j in range(n + 1)])
            a.set_yticklabels(["1"] + [f"1/{2**j}" for j in range(1, n + 1)])
            a.minorticks_off()
        if logy:
            a.set_yscale("log")
            a.set_ylim(top=CFL_MAX)
        if average_from is not None and key in ("CL", "CD", "CM"):
            sel = x >= average_from
            if sel.any():
                mean = np.cumsum(y[sel]) / np.arange(1, sel.sum() + 1)
                a.plot(
                    x[sel], mean, "k--", lw=1.2, label=f"running mean {mean[-1]:.5g}"
                )
                a.legend(loc="upper right", fontsize=8)
        if draw_target(a, key, targets, c, logy=logy) is not None:
            a.legend(loc="upper right", fontsize=8)
        if not logy and key != "ls_step":
            if key in scales:
                v = targets[key][0]
                a.set_ylim(v - scales[key], v + scales[key])
            elif window is not None:
                sel = (x >= window[0]) & (x <= window[1])
                fit_ylim(a, y[sel], targets.get(key))
        a.set_ylabel(lab)
    _finish(fig, list(axes), x, ctx["notes"], ctx["xlabel"], ctx["show_notes"])
    return save(fig, ctx["prefix"] + suffix, f"{title}: {ctx['name']}")


def _plot_diffs(d, x, ctx):
    names = (("drop", "drop"), ("CFL", "CFL"), ("GMRES", "GMRES its"))
    specs = [(k, f"Δ {lab}") for k, lab in names if k in d]
    if not specs or x.size < 2:
        print(f"skipped {ctx['prefix']}_diffs.png: no data")
        return None
    fig, axes = _stack(len(specs))
    for k, (a, (key, lab)) in enumerate(zip(axes, specs, strict=True)):
        y = np.concatenate(([0.0], np.diff(d[key])))
        if key == "GMRES":
            a.step(x, y, where="mid", color=f"C{k}", lw=1)
            a.plot(x, y, ".", ms=4, color=f"C{k}")
        else:
            a.plot(x, y, marker=".", ms=4, lw=1, color=f"C{k}")
        a.axhline(0, color="k", lw=0.6)
        if key == "CFL":
            a.set_ylim(-DCFL_MAX, DCFL_MAX)
        a.set_ylabel(lab)
    _finish(fig, list(axes), x, ctx["notes"], ctx["xlabel"], ctx["show_notes"])
    return save(
        fig, ctx["prefix"] + "_diffs.png", f"Consecutive differences: {ctx['name']}"
    )


def plot_log(
    path,
    prefix=None,
    targets=None,
    panels=PANELS,
    show_notes=True,
    x_from=None,
    x_to=None,
    last=None,
    average_from=None,
):
    """Write the convergence plots of a log or history CSV; returns the files.

    ``x_from``, ``x_to`` (iterations, or seconds for unsteady runs) and
    ``last`` (number of rows) restrict the plotted range. ``average_from``
    (unsteady, seconds) adds the running mean of the coefficients.
    """
    path = Path(path)
    run = load_run(path)
    d, notes = run.columns, run.notes
    unsteady = run.mode == "unsteady"
    x = d["time"] if unsteady else d["iter"]
    sel = np.ones(x.size, bool)
    if x_from is not None:
        sel &= x >= x_from
    if x_to is not None:
        sel &= x <= x_to
    if last is not None:
        sel &= np.arange(x.size) >= x.size - last
    d = {k: v[sel] for k, v in d.items()}
    notes = [n for n, s in zip(notes, sel, strict=True) if s]
    x = x[sel]
    target_values, scales = load_targets(targets)
    stem = path.name.split(".")[0]
    ctx = {
        "prefix": str(prefix) if prefix else str(path.with_name(stem)),
        "name": path.name,
        "notes": notes,
        "show_notes": show_notes,
        "xlabel": "time [s]" if unsteady else "iteration",
        "targets": target_values,
        "scales": scales,
    }
    print(f"{path.name}: {run.mode} run, {x.size} rows")
    window = None
    if not unsteady:
        lim = next(
            (xi for xi, n in zip(x, notes, strict=True) if LIMITER_TAG in n), None
        )
        if lim is not None:
            window = (lim - ZOOM_LEAD, x.max())
    written = []
    if "residuals" in panels:
        written.append(_plot_residuals(d, run.residuals, x, ctx))
    if "solver" in panels:
        specs = [
            ("drop", "drop", False),
            ("inner", "inner its", False) if unsteady else ("CFL", "CFL", True),
            ("GMRES", "GMRES its (0 = none)", False),
            ("ls_step", "line-search step (1 = full)", False),
            ("wall", "wall [s]", False),
        ]
        written.append(_plot_panels(d, x, specs, ctx, "_solver.png", "Solver control"))
    if "coeffs" in panels:
        specs = [("CL", "CL", False), ("CD", "CD", False), ("CM", "CM", False)]
        written.append(
            _plot_panels(
                d, x, specs, ctx, "_coeffs.png", "Coefficients", window,
                average_from if unsteady else None,
            )
        )  # fmt: skip
    if "diffs" in panels:
        written.append(_plot_diffs(d, x, ctx))
    return [w for w in written if w is not None]

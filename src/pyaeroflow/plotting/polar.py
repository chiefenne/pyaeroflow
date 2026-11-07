"""``pyaeroflow plot polar``: CL-alpha, drag polar, cm-alpha, L/D of a polar CSV.

Points of runs that did not converge (column ``converged`` = 0) are drawn
hollow. The solver's CM is counter-clockwise positive, which is nose-down
for a flow in +x; the plot shows the usual nose-up positive cm = -CM unless
``cm_sign="solver"``. Reference data (``--ref``) with columns alpha, cl,
cd, cm are drawn as markers (their cm taken as nose-up positive).
"""

from pathlib import Path

import numpy as np

from pyaeroflow.plotting.common import pyplot, save
from pyaeroflow.plotting.readers import read_reference, read_table

PANELS = {
    "cl": ("alpha", "cl", "alpha [deg]", "CL"),
    "drag": ("cd", "cl", "CD", "CL"),
    "cm": ("alpha", "cm", "alpha [deg]", "cm (nose-up positive)"),
    "ld": ("alpha", "ld", "alpha [deg]", "L/D"),
    "cd": ("alpha", "cd", "alpha [deg]", "CD"),
}


def _columns(table, nose_up):
    c = dict(table.columns)
    if "cm" in c and nose_up:
        c["cm"] = -c["cm"]
    if "cl" in c and "cd" in c:
        c["ld"] = c["cl"] / np.where(c["cd"] != 0, c["cd"], np.nan)
    return c


def plot_polar(
    paths,
    prefix=None,
    labels=None,
    refs=(),
    panels=("cl", "drag", "cm", "ld"),
    cm_sign="nose-up",
    split_cd=False,
    title=None,
):
    """Plot one or more polar CSV files with references; returns the file."""
    paths = [Path(p) for p in paths]
    labels = list(labels or [])
    runs = []
    for i, p in enumerate(paths):
        table = read_table(p)[0]
        c = _columns(table, cm_sign == "nose-up")
        order = np.argsort(c["alpha"])
        runs.append(
            (
                {k: v[order] for k, v in c.items()},
                labels[i] if i < len(labels) else p.name.split(".")[0],
            )
        )
    references = [t for spec in refs for t in read_reference(spec)]
    plt = pyplot()
    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = (n + 1) // 2 if n > 1 else 1
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.2 * cols, 4.6 * rows), squeeze=False
    )
    axes = axes.ravel()
    for a in axes[n:]:
        a.set_visible(False)
    for a, panel in zip(axes, panels, strict=False):
        xkey, ykey, xlabel, ylabel = PANELS[panel]
        for k, (c, label) in enumerate(runs):
            if xkey not in c or ykey not in c:
                continue
            x, y, color = c[xkey], c[ykey], f"C{k}"
            a.plot(x, y, "-", color=color, lw=1.2, label=label)
            ok = c.get("converged", np.ones_like(x)) > 0.5
            a.plot(x[ok], y[ok], "o", color=color, ms=5)
            if (~ok).any():
                a.plot(x[~ok], y[~ok], "o", mfc="white", color=color, ms=5,
                       label=f"{label} (not converged)")  # fmt: skip
            if split_cd and panel in ("drag", "cd"):
                for part, ls in (("cd_pressure", ":"), ("cd_viscous", "-.")):
                    if part in c:
                        xs, ys = (
                            (c[part], c["cl"])
                            if panel == "drag"
                            else (c["alpha"], c[part])
                        )
                        a.plot(xs, ys, ls, color=color, lw=1, label=f"{label} {part}")
        for j, t in enumerate(references):
            c = _columns(t, False)
            if xkey not in c or ykey not in c:
                continue
            a.plot(c[xkey], c[ykey], ls="none", marker="sD^v<>p"[j % 7], mfc="none",
                   ms=5, color=f"C{len(runs) + j}", label=t.label)  # fmt: skip
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        a.legend(fontsize=8, loc="best")
    first = paths[0]
    prefix = str(prefix) if prefix else str(first.with_name(first.name.split(".")[0]))
    name = title or ", ".join(p.name for p in paths)
    fig.tight_layout()
    return save(fig, f"{prefix}_polar.png", f"Polar: {name}")

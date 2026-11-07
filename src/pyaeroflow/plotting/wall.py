"""``pyaeroflow plot wall``: Cp and Cf along a wall from ``<name>.wall_<b>.csv``.

Default (as in most of the airfoil literature): over x/c with the upper
surface solid and the lower surface dashed, Cp axis inverted. ``x="arc"``
plots over the wall arc length s/c instead, one continuous curve from the
start of the wall file (trailing edge, upper side, leading edge, lower
side), which keeps blunt trailing edges and bases readable.

Cf is flow-aligned by default: the wall CSV keeps the sign of the face
tangent (negative on the lower surface of an airfoil even where the flow is
attached), so the lower-surface values are negated; attached flow is then
positive on both sides and Cf < 0 marks separated flow. The zero crossings
(separation and reattachment) are printed and marked.
"""

from pathlib import Path

import numpy as np

from pyaeroflow.plotting.common import pyplot, save
from pyaeroflow.plotting.readers import read_reference, read_table

PANEL_LABELS = {
    "cp": "Cp",
    "cf": "Cf",
    "y_plus": "y+",
    "t": "T [K]",
    "q_w": "q_w [W/m^2]",
}


class Wall:
    """One wall CSV split into surfaces."""

    def __init__(self, path, label=None):
        self.path = Path(path)
        self.label = label or self.path.name.split(".")[0]
        self.c = read_table(self.path)[0].columns
        x, y = self.c["x"], self.c["y"]
        self.x0 = x.min()
        self.chord = (x.max() - x.min()) or 1.0
        span = np.hypot(x[0] - x[-1], y[0] - y[-1])
        self.closed = span < 0.05 * self.chord
        if "s" not in self.c:
            ds = np.hypot(np.diff(x), np.diff(y))
            self.c["s"] = np.concatenate(([0.0], np.cumsum(ds)))
        if self.closed:
            le = int(np.argmin(x))
            first, second = np.arange(le + 1), np.arange(le, x.size)
            upper_first = y[first].mean() >= y[second].mean()
            self.upper, self.lower = (first, second) if upper_first else (second, first)
        else:
            self.upper, self.lower = np.arange(x.size), np.array([], int)

    def values(self, key, flow_aligned=True):
        """Column ``key`` with the lower-surface Cf negated (flow-aligned)."""
        v = np.array(self.c[key], float)
        if key == "cf" and flow_aligned and self.lower.size:
            v[self.lower] *= -1.0
            if self.upper.size and self.lower.size:  # the shared LE point
                v[np.intersect1d(self.upper, self.lower)] *= -1.0
        return v

    def xc(self, idx):
        """x/c of the points ``idx``."""
        return (self.c["x"][idx] - self.x0) / self.chord


def zero_crossings(x, y):
    """Return the x where y changes sign (linear interpolation), sorted along x."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    s = np.sign(y)
    i = np.flatnonzero(s[:-1] * s[1:] < 0)
    return x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])


def plot_wall(
    paths,
    prefix=None,
    labels=None,
    refs=(),
    x="chord",
    panels=("cp", "cf"),
    surface="both",
    cf_sign="flow",
    x_range=None,
    title=None,
):
    """Plot Cp/Cf (and y+, T, q_w) of one or more wall CSV files; returns the file."""
    paths = [Path(p) for p in paths]
    labels = list(labels or [])
    walls = [
        Wall(p, labels[i] if i < len(labels) else None) for i, p in enumerate(paths)
    ]
    panels = [p.lower() for p in panels]
    panels = [p for p in panels if any(p in w.c for w in walls)]
    if not panels:
        raise ValueError(
            "no plottable column (cp, cf, y_plus, t, q_w) in the wall files"
        )
    plt = pyplot()
    fig, axes = plt.subplots(
        len(panels), 1, sharex=True, figsize=(11, 3.6 * len(panels) + 0.8),
        gridspec_kw={"hspace": 0.08},
    )  # fmt: skip
    axes = np.atleast_1d(axes)
    arc = x == "arc"
    for k, wall in enumerate(walls):
        color = f"C{k}"
        for a, key in zip(axes, panels, strict=True):
            if key not in wall.c:
                continue
            y = wall.values(key, cf_sign == "flow")
            if arc:
                s = wall.c["s"] / wall.chord
                a.plot(s, y, color=color, lw=1.2, label=wall.label)
                continue
            parts = [("upper", wall.upper, "-"), ("lower", wall.lower, "--")]
            for name, idx, ls in parts:
                if not idx.size or surface not in ("both", name):
                    continue
                order = np.argsort(wall.xc(idx))
                xs, ys = wall.xc(idx)[order], y[idx][order]
                lab = wall.label if not wall.lower.size else f"{wall.label} {name}"
                a.plot(xs, ys, color=color, ls=ls, lw=1.2, label=lab)
                if key == "cf" and cf_sign == "flow":
                    # Crossings at the very ends are the stagnation point and
                    # the trailing edge, not separation.
                    zs = zero_crossings(xs, ys)
                    for xz in zs[(zs > 0.005) & (zs < 0.995)]:
                        a.plot(xz, 0.0, "o", mfc="none", color=color, ms=6)
                        print(
                            f"{wall.label} {name}: Cf = 0 at x/c = {xz:.4f}"
                        )  # sep./reatt./stagn.
    for j, spec in enumerate(refs):
        for t in read_reference(spec):
            for a, key in zip(axes, panels, strict=True):
                xs, ys = t.get("x", "s"), t.get(key)
                if xs is None or ys is None:
                    continue
                a.plot(
                    xs, ys, ls="none", marker="osD^v<>"[j % 7], ms=4, mfc="none",
                    color=f"C{len(walls) + j}", label=t.label,
                    markevery=max(1, xs.size // 120),
                )  # fmt: skip
    for a, key in zip(axes, panels, strict=True):
        a.set_ylabel(PANEL_LABELS.get(key, key))
        if key == "cp":
            a.invert_yaxis()
        if key == "cf":
            a.axhline(0.0, color="k", lw=0.6)
        if key == "y_plus":
            a.axhline(1.0, color="k", lw=0.6, ls=":")
        a.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel(
        "s/c (arc length from the start of the wall file)" if arc else "x/c"
    )
    if x_range is not None:
        axes[-1].set_xlim(*x_range)
    first = paths[0]
    stem = first.name.split(".")[0]
    prefix = str(prefix) if prefix else str(first.with_name(stem))
    name = title or ", ".join(p.name for p in paths)
    return save(fig, f"{prefix}_wall.png", f"Wall: {name}")

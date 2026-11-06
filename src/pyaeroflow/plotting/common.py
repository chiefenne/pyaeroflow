"""Shared pieces of the plot tool: matplotlib import, style, targets, saving."""

import tomllib
from pathlib import Path

import numpy as np


def pyplot():
    """matplotlib.pyplot with the file backend (optional dependency, D46)."""
    try:
        import matplotlib

        if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting needs matplotlib: pip install pyaeroflow[plot]"
        ) from exc
    return plt


def thin_grid(ax):
    """Major and minor grid lines (minor ticks only on linear axes)."""
    from matplotlib.ticker import AutoMinorLocator

    if ax.get_xscale() != "log":
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    if ax.get_yscale() != "log":
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", lw=0.4, color="0.5", alpha=0.7)
    ax.grid(which="minor", lw=0.2, color="0.7", alpha=0.6)


def save(fig, path, title=None, dpi=150):
    """Grid on every axes, optional suptitle, write a PNG; returns the path."""
    plt = pyplot()
    for ax in fig.axes:
        thin_grid(ax)
    if title:
        # Above the rotated note labels that stick out of the top axes.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        top = max(ax.get_position().y1 for ax in fig.axes) + 0.01
        for ax in fig.axes:
            for text in ax.texts:
                box = text.get_window_extent(renderer).transformed(
                    fig.transFigure.inverted()
                )
                top = max(top, box.y1 + 0.01)
        fig.suptitle(title, fontsize=13, y=top, va="bottom")
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path


# -- targets ---------------------------------------------------------------------


def canonical_key(key):
    """Target key as the log reader names the column: 'log(R_p)', 'Rp' -> 'R_p'."""
    if key.startswith("log(R_") and key.endswith(")"):
        return "R_" + key[len("log(R_") : -1]
    if key.startswith("R") and not key.startswith("R_") and len(key) > 1:
        return "R_" + key[1:]
    return key


def _half_width(spec, value):
    return abs(float(spec.get("tol", 0.0)) + float(spec.get("rtol", 0.0)) * abs(value))


def load_targets(path):
    """Targets and scale windows of a TOML file.

    Returns ``(targets, scales)``: ``targets[key] = (value, half_width, band)``
    from ``[targets]`` (a number or a table with value, tol, rtol; the band
    label is "±x %" for rtol alone, else the absolute half-width) and
    ``scales[key] = half_width`` from ``[scale.KEY]`` (y-window centred on the
    target value). Keys: any column of the plotted file (CL, CD, CM, drop,
    CFL, GMRES, wall, R_p / Rp / log(R_p), ..., cl, cd at a polar).
    """
    if path is None:
        return {}, {}
    with open(path, "rb") as f:
        data = tomllib.load(f)
    raw_targets = data.get("targets", {k: v for k, v in data.items() if k != "scale"})
    targets = {}
    for raw, spec in raw_targets.items():
        key = canonical_key(raw)
        if isinstance(spec, dict):
            if "value" not in spec:
                print(f"targets: '{raw}' has no 'value', ignored")
                continue
            v = float(spec["value"])
            hw = _half_width(spec, v)
            if hw == 0:
                band = ""
            elif not spec.get("tol"):
                band = f" ±{round(float(spec['rtol']) * 100, 6):g} %"
            else:
                band = f" ±{hw:g}"
        else:
            v, hw, band = float(spec), 0.0, ""
        targets[key] = (v, hw, band)
    scales = {}
    for raw, spec in data.get("scale", {}).items():
        key = canonical_key(raw)
        if key not in targets:
            print(f"scale: '{raw}' has no target value, ignored")
            continue
        hw = _half_width(spec, targets[key][0])
        if hw == 0:
            print(f"scale: '{raw}' has neither tol nor rtol, ignored")
            continue
        scales[key] = hw
    return targets, scales


def draw_target(ax, key, targets, color, logy=False):
    """Dashed target line and optional tolerance band; returns the line or None."""
    if key not in targets:
        return None
    v, hw, band = targets[key]
    if hw > 0:
        lo = max(v - hw, abs(v) * 1e-6) if logy else v - hw
        ax.axhspan(lo, v + hw, color=color, alpha=0.15, lw=0)
    return ax.axhline(v, color=color, ls="--", lw=1.2, label=f"target {v:g}{band}")


def fit_ylim(ax, y, target=None, pad=0.05):
    """y-limits around the finite values of ``y`` (and a target band)."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return
    lo, hi = y.min(), y.max()
    if target is not None:
        v, hw, _ = target
        lo, hi = min(lo, v - hw), max(hi, v + hw)
    margin = pad * ((hi - lo) or abs(hi) or 1.0)
    ax.set_ylim(lo - margin, hi + margin)

#!/usr/bin/env python3
"""Plot a solver log -> four PNG files. Handles steady and transient runs.

Usage:  python plot_log.py <logfile> [-t targets.toml] [-o output_prefix]

Mode is taken from the header line:
  "iter ..."  -> steady   (x-axis = iter; CFL present)
  "step ..."  -> transient (x-axis = time [s]; no CFL column, so CFL panels
                            are skipped; "inner" is parsed but not plotted)

Writes (prefix defaults to the log file name without extension):
  <prefix>_residuals.png  R_p, R_u, R_v (left axis) + R_nu (right axis);
                          with SA-AFT also R_gam and R_n, with SST R_k and
                          R_om instead of R_nu (right axis)
  <prefix>_solver.png     drop / CFL / GMRES / wall[s] as stacked subplots
  <prefix>_coeffs.png     CL / CD / CM as stacked subplots
  <prefix>_diffs.png      consecutive differences of drop / CFL / GMRES
The input file name is printed in the title of every figure.

Scaling:
  - coeffs (steady only): y-limits fitted to the window from 20 iterations
    before the "limiter released" row to the last iteration (the target band,
    if any, is included). Without such a row the full range is used.
  - solver: CFL y-axis is capped at 1e5.
  - diffs: value(i) - value(i-1); the first point is set to 0. CFL delta
    is limited to +-DCFL_MAX.

Targets (optional TOML file, every entry optional, see targets.toml):
  Keys: CL CD CM drop CFL GMRES wall Rp Ru Rv Rnu Rk Rom  (also "log(R_p)" etc.)
  Each entry is either a number (target line only) or a table:
      [targets.CD]
      value = 0.0195
      tol   = 0.0005     # absolute half-width of the band
      rtol  = 0.02       # relative half-width (+-2 %); added to tol if both given
  The legend uses the unit of the TOML: rtol alone gives "±2 %", tol (or a
  mix of tol and rtol) gives the absolute half-width.
  Only targets that are present AND apply to the run are drawn.
  Optional [scale.KEY] tables set the y-window of a coefficient panel, centred
  on the target value of KEY, with the same tol/rtol method:
      [scale.CD]
      tol   = 0.004       # absolute half-width of the window
      rtol  = 0.10        # relative half-width (fraction of |value|); added to tol
  The window replaces the automatic fit (which follows the data and so is
  stretched by the first iterations). Values outside it are clipped, on
  purpose. Needs a target for KEY: without one the entry is ignored with a
  note. Log-scaled panels (CFL) and residuals keep their own scaling.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

try:
    import tomllib                      # Python >= 3.11
except ModuleNotFoundError:             # pragma: no cover
    try:
        import tomli as tomllib         # pip install tomli
    except ModuleNotFoundError:
        tomllib = None

LIMITER_TAG = "limiter released"
ZOOM_LEAD = 20       # iterations before limiter release used for coeff scaling
CFL_MAX = 1e5
DCFL_MAX = 8e3       # +- limit for the delta CFL panel

TARGET_ALIASES = {
    "log(R_p)": "Rp", "log(R_u)": "Ru", "log(R_v)": "Rv", "log(R_nu)": "Rnu",
    "log(R_gam)": "Rgam", "log(R_n)": "Rn", "log(R_k)": "Rk", "log(R_om)": "Rom",
}
TARGET_KEYS = {"CL", "CD", "CM", "drop", "CFL", "GMRES", "wall",
               "Rp", "Ru", "Rv", "Rnu", "Rgam", "Rn", "Rk", "Rom"}


def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _read_toml(path):
    if tomllib is None:
        raise SystemExit("reading TOML needs Python >= 3.11 or 'pip install tomli'")
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_scales(path, targets):
    """Read the optional [scale.KEY] tables -> {key: half_width}, centred on the target.

    half_width = tol + rtol*|value| with the target value of KEY; a key without a
    target is reported and ignored.
    """
    if path is None:
        return {}
    raw = _read_toml(path).get("scale", {})
    scales = {}
    for raw_key, spec in raw.items():
        key = TARGET_ALIASES.get(raw_key, raw_key)
        if key not in TARGET_KEYS:
            print(f"scale: unknown key '{raw_key}' ignored")
            continue
        if key not in targets:
            print(f"scale: '{raw_key}' has no target value, ignored")
            continue
        v = targets[key][0]
        hw = abs(float(spec.get("tol", 0.0)) + float(spec.get("rtol", 0.0)) * abs(v))
        if hw == 0:
            print(f"scale: '{raw_key}' has neither tol nor rtol, ignored")
            continue
        scales[key] = hw
    return scales


def load_targets(path):
    """Read the TOML target file -> {key: (value, half_width)}.

    Returns {key: (value, half_width, band_label)}: half_width = tol +
    rtol*|value| (0 if neither is given -> line only); band_label is " ±5 %"
    for a pure rtol, " ±<absolute>" otherwise, "" without a band.
    Unknown keys are reported and ignored.
    """
    if path is None:
        return {}
    data = _read_toml(path)
    data = data.get("targets", data)
    targets = {}
    for raw_key, spec in data.items():
        key = TARGET_ALIASES.get(raw_key, raw_key)
        if key not in TARGET_KEYS:
            print(f"targets: unknown key '{raw_key}' ignored")
            continue
        if isinstance(spec, dict):
            if "value" not in spec:
                print(f"targets: '{raw_key}' has no 'value', ignored")
                continue
            v = float(spec["value"])
            tol = float(spec.get("tol", 0.0))
            rtol = float(spec.get("rtol", 0.0))
            hw = abs(tol + rtol * abs(v))
            if hw == 0:
                band = ""
            elif tol == 0:  # percent tolerance in the TOML: percent in the legend
                band = f" ±{round(rtol * 100, 6):g} %"
            else:
                band = f" ±{hw:g}"
        else:
            v, hw, band = float(spec), 0.0, ""
        targets[key] = (v, hw, band)
    return targets


def draw_target(ax, key, targets, color, logy=False):
    """Dashed target line plus optional tolerance band. Returns the line."""
    if key not in targets:
        return None
    v, hw, band = targets[key]
    if hw > 0:
        lo, hi = v - hw, v + hw
        if logy:
            lo = max(lo, abs(v) * 1e-6)
        ax.axhspan(lo, hi, color=color, alpha=0.15, lw=0)
    lab = f"target {v:g}{band}"
    return ax.axhline(v, color=color, ls="--", lw=1.2, label=lab)


def parse_log(path):
    """Return (data dict, notes, mode). mode is 'steady' or 'transient'.

    The residual columns are taken from the header ("log(R_p)" ... ), so
    laminar (3), SA (4), SA-AFT (6: also R_gam, R_n) and SST (5: R_k, R_om
    instead of R_nu) logs all work.
    Steady: iter, log(R_*), drop, CFL, [NK], GMRES, CL, CD, CM, wall, note.
            NK is optional per row; GMRES is 0 when no NK step ran.
    Transient: step, time, inner, drop, log(R_*), GMRES, CL, CD, CM, wall, note.
            Mapped onto the same keys; 'iter' holds the step number.
    """
    header, mode, res_keys = None, None, []
    rows, notes = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if header is None and tok[0] in ("iter", "step"):
                header = tok
                mode = "steady" if tok[0] == "iter" else "transient"
                res_keys = ["R" + t[len("log(R_"):-1] for t in tok
                            if t.startswith("log(R_")]
                continue
            if header is None or not is_num(tok[0]):
                continue
            nr = len(res_keys)
            try:
                if mode == "steady":
                    it = int(tok[0])
                    res = [float(v) for v in tok[1:1 + nr]]
                    drop, cfl = float(tok[1 + nr]), float(tok[2 + nr])
                    rest = tok[3 + nr:]
                    gmres = 0
                    if rest and rest[0] == "NK":
                        gmres = int(rest[1])
                        rest = rest[2:]
                    cl, cd, cm, wall = map(float, rest[:4])
                    rows.append([it, *res, drop, cfl, gmres, cl, cd, cm, wall])
                    notes.append(" ".join(rest[4:]))
                else:
                    n = 4 + nr + 5
                    rows.append([float(v) for v in tok[:n]])
                    notes.append(" ".join(tok[n:]))
            except (ValueError, IndexError):
                continue  # incomplete row (e.g. still being written)

    if mode == "steady":
        cols = ["iter", *res_keys, "drop", "CFL", "GMRES", "CL", "CD", "CM", "wall"]
    else:
        cols = ["iter", "time", "inner", "drop", *res_keys,
                "GMRES", "CL", "CD", "CM", "wall"]
    data = np.array(rows, dtype=float).reshape(-1, len(cols))
    d = {c: data[:, i] for i, c in enumerate(cols)}
    return d, notes, mode or "steady"


def thin_grid(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    if ax.get_yscale() != "log":  # AutoMinorLocator is not valid on log axes
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", lw=0.4, color="0.5", alpha=0.7)
    ax.grid(which="minor", lw=0.2, color="0.7", alpha=0.6)


def mark_notes(axes, x, notes):
    """Dotted vertical line + label for rows with a note (rejected, limiter...)."""
    for i, n in enumerate(notes):
        if n:
            for a in axes:
                a.axvline(x[i], color="k", ls=":", lw=0.7, alpha=0.6)
            axes[0].annotate(n, (x[i], 1.0), xycoords=("data", "axes fraction"),
                             rotation=90, va="bottom", ha="right", fontsize=7)


def finish(fig, axes, x, notes, xlabel, title, path):
    mark_notes(axes, x, notes)
    for a in axes:
        thin_grid(a)
    axes[-1].set_xlabel(xlabel)
    axes[-1].set_xlim(x.min(), x.max())
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def limiter_iter(it, notes):
    """Iteration number of the 'limiter released' row, or None."""
    for i, n in enumerate(notes):
        if LIMITER_TAG in n:
            return it[i]
    return None


def consecutive_diff(x):
    """x[i] - x[i-1] with the first entry set to 0."""
    return np.concatenate(([0.0], np.diff(x)))


def plot_residuals(d, x, notes, xlabel, targets, title, path):
    fig, ax = plt.subplots(figsize=(13, 6))
    lines = []
    for k, (key, lab) in enumerate([("Rp", "log(R_p)"), ("Ru", "log(R_u)"),
                                    ("Rv", "log(R_v)")]):
        lines += ax.plot(x, d[key], marker=".", ms=4, lw=1, color=f"C{k}", label=lab)
        t = draw_target(ax, key, targets, f"C{k}")
        if t is not None:
            t.set_label(f"{lab} {t.get_label()}")
            lines.append(t)
    ax.set_ylabel("log(R_p), log(R_u), log(R_v)")

    right = [k for k in ("Rnu", "Rgam", "Rn", "Rk", "Rom") if k in d]
    if right:
        ax2 = ax.twinx()  # secondary y axis for the turbulence residuals
        for k, key in enumerate(right):
            lab = f"log(R_{key[1:]})"
            c = f"C{3 + k}"
            lines += ax2.plot(x, d[key], marker=".", ms=4, lw=1.2, color=c, ls="--",
                              label=f"{lab}  [right axis]")
            t = draw_target(ax2, key, targets, c)
            if t is not None:
                t.set_label(f"{lab} {t.get_label()}")
                lines.append(t)
        ylab = ", ".join(f"log(R_{key[1:]})" for key in right)
        ax2.set_ylabel(ylab, color="C3" if len(right) == 1 else "k")
        if len(right) == 1:
            ax2.tick_params(axis="y", colors="C3")
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right",
              ncol=2, fontsize=9)

    finish(fig, [ax], x, notes, xlabel, title, path)


def plot_stack(d, x, notes, xlabel, specs, title, path, targets=None, height=2.4,
               ylim_window=None, ymax=None, scales=None):
    """specs: list of (key, ylabel, logy). Keys missing from d are skipped.
    ylim_window: (lo, hi) on x -> y-limits fitted to that window (+ target band).
    ymax: optional upper y-limit applied to log-scaled panels.
    scales: {key: half_width} y-window centred on the target (see load_scales)."""
    targets, scales = targets or {}, scales or {}
    specs = [s for s in specs if s[0] in d]
    if not specs:
        print(f"skipped {path}: no data")
        return
    fig, axes = plt.subplots(len(specs), 1, sharex=True,
                             figsize=(13, height * len(specs) + 1),
                             gridspec_kw={"hspace": 0.08})
    axes = np.atleast_1d(axes)
    if ylim_window is not None:
        mask = (x >= ylim_window[0]) & (x <= ylim_window[1])
    for k, (a, (key, lab, logy)) in enumerate(zip(axes, specs)):
        if key == "GMRES":
            a.step(x, d[key], where="mid", color=f"C{k}", lw=1)
            a.plot(x, d[key], ".", ms=4, color=f"C{k}")
        else:
            a.plot(x, d[key], marker=".", ms=4, lw=1, color=f"C{k}")
        if logy:
            a.set_yscale("log")
        t = draw_target(a, key, targets, f"C{k}", logy=logy)
        if t is not None:
            a.legend(loc="upper right", fontsize=8)
        if logy:
            if ymax is not None:
                a.set_ylim(top=ymax)
        elif key in scales:
            v = targets[key][0]
            a.set_ylim(v - scales[key], v + scales[key])
        elif ylim_window is not None and mask.any():
            y = d[key][mask]
            ymin_, ymax_ = y.min(), y.max()
            if key in targets:  # keep the target band visible
                v, hw, _ = targets[key]
                ymin_, ymax_ = min(ymin_, v - hw), max(ymax_, v + hw)
            pad = 0.05 * ((ymax_ - ymin_) or 1.0)
            a.set_ylim(ymin_ - pad, ymax_ + pad)
        a.set_ylabel(lab)
    finish(fig, list(axes), x, notes, xlabel, title, path)


def plot_diffs(d, x, notes, xlabel, title, path):
    """Stacked consecutive differences of drop, CFL and GMRES (linear y)."""
    all_specs = [("drop", "Δ drop"), ("CFL", "Δ CFL"), ("GMRES", "Δ GMRES its")]
    specs = [s for s in all_specs if s[0] in d]
    if not specs:
        print(f"skipped {path}: no data")
        return
    fig, axes = plt.subplots(len(specs), 1, sharex=True,
                             figsize=(13, 2.4 * len(specs) + 1),
                             gridspec_kw={"hspace": 0.08})
    axes = np.atleast_1d(axes)
    for k, (a, (key, lab)) in enumerate(zip(axes, specs)):
        y = consecutive_diff(d[key])
        if key == "GMRES":
            a.step(x, y, where="mid", color=f"C{k}", lw=1)
            a.plot(x, y, ".", ms=4, color=f"C{k}")
        else:
            a.plot(x, y, marker=".", ms=4, lw=1, color=f"C{k}")
        a.axhline(0, color="k", lw=0.6)
        if key == "CFL":  # symmetric limit for the delta panel
            a.set_ylim(-DCFL_MAX, DCFL_MAX)
        a.set_ylabel(lab)
    finish(fig, list(axes), x, notes, xlabel, title, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("-t", "--targets", default=None,
                    help="optional TOML file with target values / tolerances")
    ap.add_argument("-o", "--output", default=None,
                    help="output file prefix (default: log file stem)")
    args = ap.parse_args()

    d, notes, mode = parse_log(args.logfile)
    targets = load_targets(args.targets)
    scales = load_scales(args.targets, targets)
    name = os.path.basename(args.logfile)
    prefix = args.output or os.path.splitext(args.logfile)[0]
    print(f"detected {mode} run")
    if targets:
        print("targets: " + ", ".join(sorted(targets)))
    if scales:
        print("scale windows: " + ", ".join(f"{k} ±{h:g}" for k, h in sorted(scales.items())))

    if mode == "transient":
        x, xlabel = d["time"], "time [s]"
    else:
        x, xlabel = d["iter"], "iter"

    window = None
    if mode == "steady":
        lim = limiter_iter(d["iter"], notes)
        if lim is not None:
            window = (lim - ZOOM_LEAD, x.max())
        else:
            print("no 'limiter released' row found: coefficient scaling uses full range")

    plot_residuals(d, x, notes, xlabel, targets, f"Residuals: {name}",
                   f"{prefix}_residuals.png")
    plot_stack(d, x, notes, xlabel,
               [("drop", "drop", False), ("CFL", "CFL", True),
                ("GMRES", "GMRES its (0 = none)", False), ("wall", "wall [s]", False)],
               f"Solver control: {name}", f"{prefix}_solver.png",
               targets=targets, ymax=CFL_MAX)
    plot_stack(d, x, notes, xlabel,
               [("CL", "CL", False), ("CD", "CD", False), ("CM", "CM", False)],
               f"Coefficients: {name}", f"{prefix}_coeffs.png",
               targets=targets, ylim_window=window, scales=scales)
    plot_diffs(d, x, notes, xlabel, f"Consecutive differences: {name}",
               f"{prefix}_diffs.png")


if __name__ == "__main__":
    main()

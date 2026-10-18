"""Stage 6 validation on the PyAero NACA 0012 meshes: polar and C_p vs experiment.

Runs the fully turbulent polar at Re = 6e6 (alpha 0, 4, 8, 10, 12 as a
restart chain) with ``pyaeroflow polar`` on the 113-points-per-side mesh
(100k cells, ~30 minutes on two cores) and on the 472-points-per-side mesh
(295k cells, several times longer), and plots both against Ladson's tripped
data (NASA TM 4074) and the TMR SA values, and the C_p at alpha = 10 against
Gregory & O'Reilly (NASA R&M 3726, Re 2.88e6). A mesh whose polar has not
been run yet is left out of the plot.

    python examples/stage6_pyaero_validation.py [--plot-only] [--coarse|--fine]

Output: ``examples/cases/output/<case>/`` (runs) and
``examples/output/stage6_pyaero_validation.png``.
"""

import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CASES = (  # case name, label, colour
    ("naca0012_pyaero_rans", "113 points/side, 100k cells", "#2a78d6"),
    ("naca0012_pyaero_rans_fine", "472 points/side, 295k cells", "#eb6834"),
)
OUT = HERE / "output"
ANGLES = (0.0, 4.0, 8.0, 10.0, 12.0)
TMR = {0.0: (0.0, 0.0081925), 10.0: (1.0909, 0.012311), 15.0: (1.5461, 0.021237)}


def data(name):
    """Return packaged experimental data as a structured array."""
    path = files("pyaeroflow.cases") / "data" / name
    # Two comment lines (source) precede the header.
    return np.genfromtxt(path, delimiter=",", names=True, skip_header=2)


def main():
    """Run the polars (unless --plot-only) and plot."""
    OUT.mkdir(exist_ok=True)
    selected = CASES
    if "--coarse" in sys.argv:
        selected = CASES[:1]
    elif "--fine" in sys.argv:
        selected = CASES[1:]
    results = []
    for name, label, colour in CASES:
        runs = HERE / "cases" / "output" / name
        if "--plot-only" not in sys.argv and (name, label, colour) in selected:
            case = HERE / "cases" / f"{name}.toml"
            args = [
                sys.executable,
                "-m",
                "pyaeroflow.cli",
                "polar",
                str(case),
                "--alpha",
            ]
            subprocess.run(args + [f"{a:g}" for a in ANGLES], check=True)
        csv = runs / f"{name}.polar.csv"
        if not csv.exists():
            print(f"{label}: no polar yet ({csv})")
            continue
        polar = np.genfromtxt(csv, delimiter=",", names=True)
        print(label)
        for row in polar:
            print(
                f"  alpha {row['alpha']:5.1f}: CL = {row['CL']:.4f}, "
                f"CD = {row['CD']:.5f} (pressure {row['CD_pressure']:.5f}, "
                f"viscous {row['CD_viscous']:.5f}), {int(row['iterations'])} iterations"
            )
        wall_csv = runs / "alpha_10" / f"{name}_alpha_10.wall_airfoil.csv"
        wall = (
            np.genfromtxt(wall_csv, delimiter=",", names=True)
            if wall_csv.exists()
            else None
        )
        results.append((label, colour, polar, wall))
    if results:
        plot(results)


def plot(results):
    """Polar vs Ladson/TMR, C_p vs Gregory & O'Reilly, one line set per mesh."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot")
        return

    ladson = data("naca0012_ladson_re6e6_tripped.csv")
    gregory = data("naca0012_gregory_cp.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for grit, marker in (("80", "s"), ("120", "^")):
        sel = (
            ladson["grit"] == int(grit)
            if ladson["grit"].dtype.kind != "U"
            else ladson["grit"] == grit
        )
        ax.plot(
            ladson["alpha"][sel],
            ladson["cl"][sel],
            marker,
            color="0.4",
            mfc="white",
            ms=6,
            label=f"Ladson, {grit} grit",
        )
    ax.plot(
        [a for a in TMR], [TMR[a][0] for a in TMR], "k*", ms=11, label="TMR (CFL3D, SA)"
    )
    for label, colour, polar, _ in results:
        a, cl = polar["alpha"], polar["CL"]
        ax.plot(a, cl, "o-", color=colour, lw=2, ms=6, label=label)
    ax.set(xlabel="α [deg]", ylabel="CL", title="NACA 0012, Re = 6e6, fully turbulent")

    ax = axes[1]
    for grit, marker in (("80", "s"), ("120", "^")):
        sel = (
            ladson["grit"] == int(grit)
            if ladson["grit"].dtype.kind != "U"
            else ladson["grit"] == grit
        )
        ax.plot(
            ladson["alpha"][sel],
            ladson["cd"][sel],
            marker,
            color="0.4",
            mfc="white",
            ms=6,
            label=f"Ladson, {grit} grit",
        )
    ax.plot(
        [a for a in TMR], [TMR[a][1] for a in TMR], "k*", ms=11, label="TMR (CFL3D, SA)"
    )
    for label, colour, polar, _ in results:
        a, cd = polar["alpha"], polar["CD"]
        ax.plot(a, cd, "o-", color=colour, lw=2, ms=6, label=label)
        ax.plot(
            polar["alpha"],
            polar["CD_viscous"],
            "--",
            color=colour,
            lw=1.2,
            label=f"{label}, viscous part",
        )
    ax.set(xlabel="α [deg]", ylabel="CD", title="drag", xlim=(-1, 16), ylim=(0, 0.03))

    ax = axes[2]
    for label, colour, _, wall in results:
        if wall is None:
            continue
        sides = ((wall["y"] >= 0, "-", "upper"), (wall["y"] < 0, "--", ""))
        for sel, ls, side in sides:
            o = np.argsort(wall["x"][sel])
            ax.plot(
                wall["x"][sel][o],
                wall["cp"][sel][o],
                ls,
                color=colour,
                lw=1.6,
                label=f"{label}, {side}" if side else None,
            )
    g = gregory[gregory["alpha"] == 10.0]
    ax.plot(
        g["x_c"],
        g["cp"],
        "o",
        color="0.3",
        mfc="white",
        ms=5,
        label="Gregory & O'Reilly, Re 2.88e6, upper",
    )
    ax.invert_yaxis()
    ax.set(xlabel="x / c", ylabel="Cp", title="α = 10°")
    for ax in axes:
        ax.grid(True, color="0.88", lw=0.8)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage6_pyaero_validation.png", dpi=130)
    print(f"plot: {OUT / 'stage6_pyaero_validation.png'}")


if __name__ == "__main__":
    main()

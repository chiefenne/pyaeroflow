"""Stage 6 example: NACA 0012 at Re = 6e6 (NASA TMR 2DN00), SA-neg, O-grid.

Run from the repository root (about three minutes):

    python examples/stage6_naca0012.py [--medium]

Output in ``examples/output/``: ``stage6_naca0012.png`` (C_p at alpha = 10,
convergence history) and ``stage6_naca0012.vtu``. ``--medium`` uses the
256 x 128 O-grid (33k cells, about 10 minutes).
"""

import sys
from pathlib import Path

import numpy as np

from pyaeroflow.cases import naca0012
from pyaeroflow.io import write_vtu
from pyaeroflow.physics import rans

OUT = Path(__file__).parent / "output"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def main():
    """Run alpha = 0 and 10, print the comparison and plot."""
    OUT.mkdir(exist_ok=True)
    n_surface, n_radial = (256, 128) if "--medium" in sys.argv else (128, 64)
    print(f"R2 NACA 0012, Re = 6e6, SA-neg, O-grid {n_surface}x{n_radial}, 500 chords")
    runs = {}
    for alpha in (0.0, 10.0):
        # Frozen limiter on the O-grid (see tests/verification/test_naca0012.py).
        r = naca0012.solve(alpha, n_surface, n_radial, release_limiter=False)
        runs[alpha] = r
        ref = r.reference
        its = len(r.history["iteration"]) - 1
        print(
            f"  alpha = {alpha:4.1f}: CL = {r.cl:.4f} (TMR {ref['cl']:.4f}), "
            f"CD = {r.cd:.5f} (TMR {ref['cd']:.5f}; pressure {r.cd_pressure:.5f}, "
            f"viscous {r.cd_viscous:.5f}), CM = {r.cm:.4f}, y+ max {r.y_plus_max:.2f}, "
            f"{its} iterations, converged: {r.converged}"
        )
    r = runs[10.0]
    w = r.w
    nu = r.problem.model.viscosity
    write_vtu(
        OUT / "stage6_naca0012.vtu",
        r.problem.mesh,
        cell_data={
            "pressure": w[:, 0],
            "velocity": w[:, 1:3],
            "nu_tilde": w[:, rans.NT],
            "eddy_viscosity_ratio": r.problem.eddy_viscosity(w) / nu,
        },
    )
    plot(runs)


def plot(runs):
    """C_p distribution and the convergence history."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for k, (alpha, r) in enumerate(runs.items()):
        upper = r.y >= 0.0
        for sel, ls in ((upper, "-"), (~upper, "--")):
            order = np.argsort(r.x[sel])
            ax.plot(
                r.x[sel][order],
                r.cp[sel][order],
                ls,
                color=COLORS[k],
                lw=1.5,
                label=f"α = {alpha:g}°" if ls == "-" else None,
            )
    ax.invert_yaxis()
    ax.set(
        xlabel="x / c", ylabel="Cp", title="R2: NACA 0012, Re = 6e6 (upper —, lower --)"
    )

    ax = axes[1]
    for k, (alpha, r) in enumerate(runs.items()):
        h = r.history
        ax.semilogy(
            h["iteration"],
            h["residual"] / h["residual"][0],
            color=COLORS[k],
            lw=1.5,
            label=f"α = {alpha:g}°",
        )
    ax.set(xlabel="iteration", ylabel="residual / initial", title="convergence")
    for ax in axes:
        ax.grid(True, color="0.88", lw=0.8)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "stage6_naca0012.png", dpi=130)
    print(f"plot: {OUT / 'stage6_naca0012.png'}")


if __name__ == "__main__":
    main()

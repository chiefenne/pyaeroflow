"""Stage 5 example: turbulent flat plate (SA-neg) on the NASA TMR domain.

Run from the repository root (about one minute):

    python examples/stage5_rans.py

Output in ``examples/output/``: ``stage5_rans.png`` (C_f(x) on two meshes vs
the TMR SA values, and the law of the wall at x = 0.97) and the field
``stage5_flat_plate_rans.vtu`` (pressure, velocity, nu_tilde, nu_t/nu).
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from pyaeroflow.cases import tmr_flat_plate as tp
from pyaeroflow.io import write_vtu
from pyaeroflow.physics import rans

OUT = Path(__file__).parent / "output"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def wall_profile(r, x_station=0.97):
    """u+ and y+ at a station from the cell values (interpolated)."""
    mesh, w = r.problem.mesh, r.w
    nu = r.problem.model.viscosity
    interp = LinearNDInterpolator(mesh.cell_centroid, w[:, 1])
    cf = float(np.interp(x_station, r.x, r.cf))
    u_tau = np.sqrt(0.5 * cf)
    y = np.geomspace(2e-6, 0.05, 80)
    u = interp(np.full_like(y, x_station), y)
    return y * u_tau / nu, u / u_tau


def main():
    """Run two meshes, print the comparison and plot."""
    OUT.mkdir(exist_ok=True)
    print("R1 turbulent flat plate, Re_L = 5e6, SA-neg, vs NASA TMR (SA)")
    runs = {}
    for nxp, nxf, ny in ((48, 12, 48), (96, 24, 64)):
        r = tp.solve(nxp, nxf, ny)
        runs[(nxp, ny)] = r
        ref = r.reference
        its = len(r.history["iteration"]) - 1
        print(
            f"  {r.problem.mesh.n_cells:6d} cells: Cf(0.97) = {r.cf_097:.6f} "
            f"({100 * (r.cf_097 / ref['cf_097'] - 1):+.1f} %), CD = {r.cd:.6f} "
            f"({100 * (r.cd / ref['cd'] - 1):+.1f} %), y+ max {r.y_plus_max:.2f}, "
            f"{its} iterations"
        )
    w = r.w
    nu = r.problem.model.viscosity / r.problem.model.density
    write_vtu(
        OUT / "stage5_flat_plate_rans.vtu",
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
    """C_f(x) and the law of the wall."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for k, ((nxp, ny), r) in enumerate(runs.items()):
        ax.plot(r.x, r.cf, color=COLORS[k], lw=2, label=f"PyAeroFlow {nxp}x{ny}")
    ref = r.reference
    ax.plot([0.97], [ref["cf_097"]], "k*", ms=12, label="TMR SA, x = 0.97")
    x = np.linspace(0.05, 2.0, 100)
    ax.plot(
        x,
        0.0576 * (5e6 * x) ** -0.2,
        "--",
        color="0.5",
        lw=1,
        label="0.0576 Re_x^-1/5 (correlation)",
    )
    ax.set(
        xlabel="x",
        ylabel="Cf",
        ylim=(0.0, 0.006),
        title="R1: skin friction, Re_L = 5e6",
    )

    ax = axes[1]
    r = runs[max(runs)]
    y_plus, u_plus = wall_profile(r)
    ax.semilogx(
        y_plus,
        u_plus,
        "o",
        color=COLORS[0],
        mfc="white",
        ms=5,
        label="PyAeroFlow, x = 0.97",
    )
    yp = np.geomspace(1.0, 3000.0, 100)
    ax.semilogx(yp[yp < 12], yp[yp < 12], "--", color="0.3", lw=1, label="u+ = y+")
    ax.semilogx(
        yp[yp > 8],
        np.log(yp[yp > 8]) / 0.41 + 5.0,
        "-",
        color="0.3",
        lw=1,
        label="log law (κ = 0.41, B = 5.0)",
    )
    ax.set(xlabel="y+", ylabel="u+", title="R1: law of the wall", ylim=(0, 30))
    for ax in axes:
        ax.grid(True, color="0.88", lw=0.8)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "stage5_rans.png", dpi=130)
    print(f"plot: {OUT / 'stage5_rans.png'}")


if __name__ == "__main__":
    main()

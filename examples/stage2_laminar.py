"""Stage 2 example: laminar incompressible flow.

Lid-driven cavity at Re = 100 (and Re = 1000 with ``--re1000``, about one
minute) compared with Ghia, Ghia & Shin (1982), and plane Poiseuille flow on
a perturbed triangle mesh compared with the exact profile. Run from the
repository root:

    python examples/stage2_laminar.py [--re1000]

Output in ``examples/output/``: ``stage2_laminar.png`` and
``stage2_cavity_re<Re>.vtu``.
"""

import sys
from pathlib import Path

import numpy as np

from pyaeroflow.cases import cavity, channel
from pyaeroflow.io import write_vtu

OUT = Path(__file__).parent / "output"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def main():
    """Run the cases, print the comparison and plot."""
    OUT.mkdir(exist_ok=True)
    reynolds = [100, 1000] if "--re1000" in sys.argv else [100]
    cavities = {}
    for re in reynolds:
        n, cluster = (64, False) if re == 100 else (128, True)
        r = cavity.solve(re, n, cluster=cluster)
        cavities[re] = r
        du = np.abs(r.u_centre - r.ghia_u).max()
        dv = np.abs(r.v_centre - r.ghia_v).max()
        its = len(r.history["iteration"]) - 1
        print(
            f"cavity Re={re}: {n}x{n} cells, converged={r.converged} in {its} "
            f"iterations; max |u - Ghia| = {du:.4f}, max |v - Ghia| = {dv:.4f}"
        )
        write_vtu(
            OUT / f"stage2_cavity_re{re}.vtu",
            r.problem.mesh,
            cell_data={"pressure": r.w[:, 0], "velocity": r.w[:, 1:]},
        )
    pois = channel.poiseuille(32, "triangle", perturb=0.4)
    print(
        f"poiseuille (triangles, perturbed): converged={pois.converged}, "
        f"RMS error u = {pois.error[1]:.2e}"
    )
    plot(cavities, pois)


def plot(cavities, pois):
    """Centreline profiles vs Ghia and the Poiseuille profile vs exact."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
    for k, (re, r) in enumerate(cavities.items()):
        mesh = r.problem.mesh
        y = np.linspace(0.0, 1.0, 201)
        u_line, _ = cavity.interpolate_profiles(
            r.problem,
            r.w,
            np.column_stack((np.full_like(y, 0.5), y)),
            np.column_stack((y, np.full_like(y, 0.5))),
        )
        _, v_line = cavity.interpolate_profiles(
            r.problem,
            r.w,
            np.column_stack((np.full_like(y, 0.5), y)),
            np.column_stack((y, np.full_like(y, 0.5))),
        )
        label = f"PyAeroFlow Re {re} ({mesh.n_cells} cells)"
        ax1.plot(u_line, y, color=COLORS[k], lw=2, label=label)
        ax1.plot(
            r.ghia_u,
            r.ghia_y,
            "o",
            color=COLORS[k],
            mfc="white",
            ms=7,
            label=f"Ghia et al. Re {re}",
        )
        ax2.plot(y, v_line, color=COLORS[k], lw=2, label=label)
        ax2.plot(
            r.ghia_x,
            r.ghia_v,
            "o",
            color=COLORS[k],
            mfc="white",
            ms=7,
            label=f"Ghia et al. Re {re}",
        )
    ax1.set(xlabel="u / U_lid", ylabel="y", title="Cavity: u on x = 0.5")
    ax2.set(xlabel="x", ylabel="v / U_lid", title="Cavity: v on y = 0.5")

    mesh = pois.problem.mesh
    yc = mesh.cell_centroid[:, 1]
    ax3.plot(
        pois.w[:, 1], yc, ".", color=COLORS[0], ms=4, label="PyAeroFlow, cell values"
    )
    ys = np.linspace(0.0, 1.0, 100)
    ax3.plot(
        1.0 / (2 * 0.1) * ys * (1 - ys), ys, "-", color="0.3", lw=1.5, label="exact"
    )
    ax3.set(
        xlabel="u", ylabel="y", title=f"Poiseuille, {mesh.n_cells} perturbed triangles"
    )
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="0.88", lw=0.8)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "stage2_laminar.png", dpi=130)
    print(f"plot: {OUT / 'stage2_laminar.png'}")


if __name__ == "__main__":
    main()

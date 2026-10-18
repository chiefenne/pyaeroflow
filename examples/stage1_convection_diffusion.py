"""Stage 1 example: order of accuracy of the convection-diffusion model problem.

Solves div(a u - nu grad u) = 0 on the unit square (a = (1, 0.5), nu = 0.25,
exact solution u = exp(a . (x - (1, 1)) / nu)) on perturbed meshes of all four
cell types, four refinement levels each. Run from the repository root:

    python examples/stage1_convection_diffusion.py

Output in ``examples/output/``: ``stage1_convection_diffusion.png`` (needs
matplotlib) and ``stage1_polygon_64.vtu`` (solution, exact solution and error).
"""

from pathlib import Path

import numpy as np

from pyaeroflow.cases import convection_diffusion as case
from pyaeroflow.io import write_vtu
from pyaeroflow.physics.convection_diffusion import ConvectionDiffusion

OUT = Path(__file__).parent / "output"
LEVELS = (8, 16, 32, 64)
CELL_TYPES = ("quad", "triangle", "polygon", "mixed")
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4
MARKERS = ("o", "s", "^", "D")


def main():
    """Run the refinement study, print a table, write VTU and plot."""
    OUT.mkdir(exist_ok=True)
    model = ConvectionDiffusion()
    studies = {}
    print(f"{'cells':>8} {'error':>10} {'order':>6} {'iterations':>10}")
    for cell_type in CELL_TYPES:
        results = [case.solve(model, cell_type, n, perturb=0.4) for n in LEVELS]
        orders = case.observed_orders(results)
        print(cell_type)
        for k, r in enumerate(results):
            order = f"{orders[k - 1]:6.2f}" if k else " " * 6
            its = len(r.history["iteration"]) - 1
            print(f"{r.mesh.n_cells:8d} {r.error:10.3e} {order} {its:10d}")
        studies[cell_type] = results

    finest = studies["polygon"][-1]
    write_vtu(
        OUT / "stage1_polygon_64.vtu",
        finest.mesh,
        cell_data={
            "u": finest.u[:, 0],
            "u_exact": finest.exact[:, 0],
            "error": finest.u[:, 0] - finest.exact[:, 0],
        },
    )
    plot(studies, case.solve(model, "polygon", 16, perturb=0.4))


def plot(studies, field):
    """Error vs mesh size, convergence history, and a solution field."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("matplotlib not installed: no plot")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
    for k, (cell_type, results) in enumerate(studies.items()):
        style = {"color": COLORS[k], "marker": MARKERS[k], "lw": 2, "ms": 8}
        h = [r.h for r in results]
        ax1.loglog(h, [r.error for r in results], label=cell_type, **style)
        hist = results[-1].history
        ax2.semilogy(
            hist["iteration"],
            hist["residual"] / hist["residual"][0],
            label=f"{cell_type} ({results[-1].mesh.n_cells} cells)",
            markevery=4,
            **style,
        )
    h_ref = np.array([studies["quad"][0].h, studies["quad"][-1].h])
    e0 = studies["quad"][0].error
    ax1.loglog(h_ref, 0.5 * e0 * (h_ref / h_ref[0]) ** 2, "--", color="0.5", lw=1.5)
    h_mid = np.sqrt(h_ref[0] * h_ref[1])
    ax1.text(h_mid, 0.3 * e0 * (h_mid / h_ref[0]) ** 2, "slope 2", color="0.4")
    ax1.set(xlabel="h = sqrt(area / cells)", ylabel="RMS error", title="Error")
    ax2.set(xlabel="iteration", ylabel="residual / initial residual")
    ax2.set_title("Convergence, finest meshes")
    for ax in (ax1, ax2):
        ax.grid(True, which="major", color="0.88", lw=0.8)
        ax.legend(frameon=False)

    blues = LinearSegmentedColormap.from_list(
        "blue_ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    )
    mesh = field.mesh
    cells = PolyCollection(
        [mesh.nodes[mesh.cell(i)] for i in range(mesh.n_cells)],
        array=field.u[:, 0],
        cmap=blues,
        edgecolors="white",
        linewidths=0.3,
    )
    ax3.add_collection(cells)
    ax3.autoscale_view()
    ax3.set_aspect("equal")
    ax3.set(xlabel="x", ylabel="y", title=f"u, polygon mesh ({mesh.n_cells} cells)")
    fig.colorbar(cells, ax=ax3, shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT / "stage1_convection_diffusion.png", dpi=130)
    print(f"plot: {OUT / 'stage1_convection_diffusion.png'}")


if __name__ == "__main__":
    main()

"""Stage 0 example: generate the test meshes, write VTU files and a mesh plot.

Run from the repository root:

    python examples/stage0_meshes.py

Output goes to ``examples/output/``: one VTU file per mesh (open in ParaView)
and ``stage0_meshes.png`` (needs matplotlib).
"""

from pathlib import Path

import numpy as np

from pyaeroflow.io import write_vtu
from pyaeroflow.mesh import rectangle

OUT = Path(__file__).parent / "output"
CELL_TYPES = ("quad", "triangle", "polygon", "mixed")
VARIANTS = {
    "regular": {"perturb": 0.0},
    "perturbed": {"perturb": 0.4, "diagonal": "random"},
    "clustered": {"cluster_y": "both", "beta": 2.0},
}


def main():
    """Generate, report and write all example meshes."""
    OUT.mkdir(exist_ok=True)
    meshes = {}
    for variant, options in VARIANTS.items():
        for cell_type in CELL_TYPES:
            mesh = rectangle(
                12, 8, y=(0.0, 0.5), cell_type=cell_type, seed=1, **options
            )
            name = f"{cell_type}_{variant}"
            meshes[name] = mesh
            write_vtu(
                OUT / f"{name}.vtu",
                mesh,
                cell_data={
                    "cell_area": mesh.cell_area,
                    "nodes_per_cell": mesh.cell_n_nodes,
                },
            )
            print(f"--- {name}\n{mesh.summary()}\n")
    plot(meshes)


def plot(meshes):
    """Plot all meshes in one figure, if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot")
        return
    from pyaeroflow.plotting import plot_mesh

    rows, cols = len(VARIANTS), len(CELL_TYPES)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.1 * rows))
    for ax, (name, mesh) in zip(np.ravel(axes), meshes.items(), strict=True):
        plot_mesh(mesh, ax=ax, title=f"{name}: {mesh.n_cells} cells")
        ax.get_legend().remove()
    handles, labels = np.ravel(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "stage0_meshes.png", dpi=130)
    print(f"plot: {OUT / 'stage0_meshes.png'}")


if __name__ == "__main__":
    main()

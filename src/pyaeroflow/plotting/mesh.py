"""Plot a mesh with its named boundary faces."""

import numpy as np


def plot_mesh(mesh, ax=None, title=None):
    """Draw cells and colour the boundary faces by set name.

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    ax : matplotlib.axes.Axes, optional
        Target axes; a new figure is created if omitted.
    title : str, optional
        Axes title; default is a cell-count summary.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection
    except ImportError as exc:  # optional dependency (D46)
        raise ImportError(
            "plotting needs matplotlib: pip install pyaeroflow[plot]"
        ) from exc

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    polygons = [mesh.nodes[mesh.cell(i)] for i in range(mesh.n_cells)]
    ax.add_collection(
        PolyCollection(polygons, facecolors="none", edgecolors="0.45", linewidths=0.4)
    )
    for k, (name, faces) in enumerate(mesh.boundary_faces.items()):
        segments = mesh.nodes[mesh.face_nodes[faces]]
        ax.add_collection(
            LineCollection(segments, colors=f"C{k}", linewidths=2.0, label=name)
        )
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    counts = np.bincount(mesh.cell_n_nodes)
    kinds = ", ".join(f"{n}×{k}-node" for k, n in enumerate(counts) if n)
    ax.set_title(title if title is not None else f"{mesh.n_cells} cells ({kinds})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    return ax

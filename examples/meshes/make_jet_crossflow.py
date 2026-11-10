"""Jet in crossflow over a flat plate: structured quad mesh with a slot marker.

Length unit: the slot width d = 1. The slot spans x = -0.5 .. 0.5 on the
bottom wall (y = 0). The domain runs from x = -XUP (inflow) to x = XDOWN
(outflow) and up to y = H. Cells are uniform across the slot (NSLOT faces),
grow geometrically upstream, stay fine over the first 10 d behind the slot
(jet and recirculation), and grow towards the outflow. Normal to the wall
the first cell is DY1 and the cells grow to the top.

Markers: ``plate`` (no-slip wall), ``slot`` (jet exit, a velocity-inlet in
``examples/cases/jet_crossflow*.toml``), ``inlet``, ``outlet``, ``top``.

Usage: python make_jet_crossflow.py [OUT.su2]   (default jet_crossflow.su2)
"""

import sys

import numpy as np

from pyaeroflow.mesh import geometric_growth, split_boundary, structured, write_su2

XUP, XDOWN, H = 10.0, 80.0, 20.0  # domain in slot widths
NSLOT = 32  # faces across the slot
DY1 = 0.01  # first cell height normal to the wall
NY = 110  # cells normal to the wall


def graded(length, first, n):
    """Geometrically growing widths of n cells from ``first``, summing to ``length``."""
    r = geometric_growth(length, first, n)
    widths = first * r ** np.arange(n)
    return widths * length / widths.sum()


def grid_lines():
    """Grid-line coordinates xs, ys."""
    dx = 1.0 / NSLOT
    slot = np.linspace(-0.5, 0.5, NSLOT + 1)
    up = -0.5 - np.cumsum(graded(XUP - 0.5, dx, 60))[::-1]
    near = 0.5 + np.cumsum(graded(10.0, dx, 120))  # x = 0.5 .. 10.5
    far = near[-1] + np.cumsum(graded(XDOWN - near[-1], near[-1] - near[-2], 60))
    xs = np.concatenate((up, slot, near, far))
    ys = np.concatenate(([0.0], np.cumsum(graded(H, DY1, NY))))
    return xs, ys


def jet_crossflow_mesh():
    """Mesh with the markers plate, slot, inlet, outlet, top."""
    xs, ys = grid_lines()
    mesh = structured(xs, ys, names=("bottom", "outlet", "top", "inlet"))

    def slot(x, y):
        return np.abs(x) < 0.5

    return split_boundary(
        mesh, "bottom", {"slot": slot, "plate": lambda x, y: ~slot(x, y)}
    )


def main(argv):
    """Command line: see the module docstring."""
    out = argv[1] if len(argv) > 1 else "jet_crossflow.su2"
    mesh = jet_crossflow_mesh()
    write_su2(out, mesh)
    n_slot = mesh.boundary_faces["slot"].size
    print(
        f"{out}: {mesh.n_cells} cells, slot {n_slot} faces, "
        f"markers {sorted(mesh.boundary_faces)}"
    )


if __name__ == "__main__":
    main(sys.argv)

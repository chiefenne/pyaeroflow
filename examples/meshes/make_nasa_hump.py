"""NASA wall-mounted hump grid (TMR "no plenum", PLOT3D 2D) -> .su2, with a slot marker.

The TMR grids hump2newtop_noplenumZ<ni>x<nj>.p2dfmt (CFDVAL2004 Case 3, chord 1,
hump from x = 0 to 1, inflow at x = -6.39, outflow at x = 4) are one block:
j = 1 is the lower (no-slip) wall, j = nj the contoured upper wall (slip),
i = 1 the inflow and i = ni the outflow. Marker names: wall, top, inlet,
outlet. With XMIN XMAX the wall faces with centres in that x range become the
marker ``slot``: the exit of the suction slot (x/c = 0.654 - 0.659 in the
CFDVAL2004 geometry with plenum), used with a ``transpiration-wall``. The
script prints the slot length L/c; the suction velocity follows from the
measured mass flow as v_n / U = -C_q / (L/c), C_q = 1.509e-3.

The grids come from the NASA zip https://www.nasa.gov/wp-content/uploads/2026/02/nasahump-grids.zip
(folder Nasahump_grids; gunzip the .p2dfmt.gz files first).

Usage: python make_nasa_hump.py IN.p2dfmt OUT.su2 [XMIN XMAX]

    python make_nasa_hump.py hump2newtop_noplenumZ409x109.p2dfmt \
        nasa_hump_409x109_slot.su2 0.6543 0.6584
"""

import sys

import numpy as np

from pyaeroflow.mesh import Mesh, split_boundary, write_su2

C_Q = 0.01518 / (0.5842 * 1.185 * 34.6 * 0.42)  # kg/s over 23 in span -> 1.509e-3


def read_p2d(path):
    """Read a formatted single-block 2D PLOT3D grid: x, y indexed [i, j]."""
    with open(path) as f:
        v = np.array(f.read().split(), dtype=float)
    if int(v[0]) != 1:
        raise ValueError(f"{path}: {int(v[0])} blocks, expected 1")
    ni, nj = int(v[1]), int(v[2])
    n = ni * nj
    x = v[3 : 3 + n].reshape(nj, ni).T
    y = v[3 + n : 3 + 2 * n].reshape(nj, ni).T
    return x, y


def hump_mesh(x, y):
    """Quadrilateral mesh with the markers wall, top, inlet, outlet."""
    ni, nj = x.shape
    k = np.arange(ni * nj).reshape(nj, ni).T  # node index of (i, j)
    nodes = np.column_stack((x.T.ravel(), y.T.ravel()))
    cells = np.column_stack(
        (k[:-1, :-1].ravel(), k[1:, :-1].ravel(), k[1:, 1:].ravel(), k[:-1, 1:].ravel())
    )

    def edges(line):
        return np.column_stack((line[:-1], line[1:]))

    boundary = {
        "wall": edges(k[:, 0]),
        "top": edges(k[:, -1]),
        "inlet": edges(k[0, :]),
        "outlet": edges(k[-1, :]),
    }
    return Mesh.from_cells(nodes, cells, boundary)


def main(argv):
    """Command line: see the module docstring."""
    if len(argv) not in (3, 5):
        sys.exit(__doc__)
    mesh = hump_mesh(*read_p2d(argv[1]))
    if len(argv) == 5:
        xmin, xmax = float(argv[3]), float(argv[4])

        def slot(x, y):
            return (x >= xmin) & (x <= xmax)

        mesh = split_boundary(
            mesh, "wall", {"slot": slot, "wall": lambda x, y: ~slot(x, y)}
        )
        faces = mesh.boundary_faces["slot"]
        if faces.size == 0:
            sys.exit(f"no wall faces with centres in x = [{xmin}, {xmax}]")
        length = mesh.face_length[faces].sum()
        xc = mesh.face_centre[faces, 0]
        print(
            f"slot: {faces.size} faces, x = {xc.min():.4f} .. {xc.max():.4f}, "
            f"L/c = {length:.5f}, suction v_n/U = {-C_Q / length:.5f} (C_q {C_Q:.4e})"
        )
    write_su2(argv[2], mesh)
    print(f"{argv[2]}: {mesh.n_cells} cells, markers {sorted(mesh.boundary_faces)}")


if __name__ == "__main__":
    main(sys.argv)

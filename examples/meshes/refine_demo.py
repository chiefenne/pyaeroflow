r"""Refine an airfoil mesh in fixed regions and write it for ParaView.

Stage R2 demo of local mesh refinement (hanging nodes) before the solver uses
it: wall cells with face centre x < ``--le`` are split along the wall into
``--n`` parts (the boundary-layer column rule refines the whole column of thin
cells above them), and the cells within ``--te`` of the trailing edge
(x = ``--x-te``) are split 2 x 2 and, where thin, along the wall. New wall
nodes are placed on a cubic through the neighbouring wall nodes.

    python examples/meshes/refine_demo.py sd7003_C10_W25_320k_cells.su2 \\
        sd7003_refined.vtu --walls airfoil --le 0.1 --te 0.03

Cell fields in the VTU: ``split_u``/``split_v`` (accumulated split factors),
``n_nodes`` (5 or more = cell with hanging nodes), ``aspect_ratio``,
``cell_area`` and ``origin`` (cell in the input mesh).
"""

import argparse
import time

import numpy as np

from pyaeroflow.io.vtu import write_vtu
from pyaeroflow.mesh import read_mesh
from pyaeroflow.mesh.refine import Refinement


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mesh")
    ap.add_argument("out", help="output .vtu")
    ap.add_argument("--walls", nargs="+", default=["airfoil"])
    ap.add_argument("--le", type=float, default=0.1, help="refine wall cells x < LE")
    ap.add_argument("--te", type=float, default=0.03, help="radius around the TE")
    ap.add_argument("--x-te", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=2, help="split factor along the wall")
    ap.add_argument("--column-aspect", type=float, default=5.0)
    ap.add_argument("--decay", type=float, default=20.0)
    ap.add_argument("--cycles", type=int, default=1)
    args = ap.parse_args(argv)

    mesh = read_mesh(args.mesh)
    print(f"input: {mesh.n_cells} cells")
    t0 = time.perf_counter()
    ref = Refinement.from_mesh(mesh, walls=args.walls, decay=args.decay)
    for cycle in range(args.cycles):
        m = ref.mesh
        wall = np.concatenate([m.boundary_faces[w] for w in args.walls])
        le = np.unique(m.owner[wall][m.face_centre[wall, 0] < args.le])
        nu, nv = ref.along_long_side(le, args.n)
        ref.split(le, nu, nv, max_hanging=1, column_aspect=args.column_aspect)

        m = ref.mesh
        c = m.cell_centroid
        near_te = np.hypot(c[:, 0] - args.x_te, c[:, 1]) < args.te
        cells = np.flatnonzero(near_te & _splittable(ref))
        ext = ref.extents()[cells]
        thin = np.maximum(ext[:, 0], ext[:, 1]) > args.column_aspect * np.minimum(
            ext[:, 0], ext[:, 1]
        )
        nu, nv = ref.along_long_side(cells, 2)
        nu = np.where(thin, nu, 2)
        nv = np.where(thin, nv, 2)
        ref.split(cells, nu, nv, max_hanging=1, column_aspect=args.column_aspect)
        print(f"cycle {cycle + 1}: {ref.n_cells} cells")
    m = ref.mesh
    print(f"refined in {time.perf_counter() - t0:.1f} s")
    print(m.summary())

    ext = ref.extents()
    factors = ref.split_factors
    write_vtu(
        args.out,
        m,
        cell_data={
            "split_u": factors[:, 0],
            "split_v": factors[:, 1],
            "n_nodes": m.cell_n_nodes,
            "aspect_ratio": ext.max(axis=1) / ext.min(axis=1),
            "cell_area": m.cell_area,
            "origin": ref.origin,
        },
    )
    print(f"written: {args.out}")


def _splittable(ref):
    return np.array([len(c.corners) == 4 for c in ref.cells])


if __name__ == "__main__":
    main()

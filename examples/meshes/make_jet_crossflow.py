"""Jet in crossflow over a flat plate: structured quad mesh with a slot marker.

Thin wrapper of ``pyaeroflow.mesh.library.jet_crossflow_mesh`` (see there for
the geometry and the markers); the same mesh comes from
``pyaeroflow make-mesh jet-crossflow -o jet_crossflow.su2``.

Usage: python make_jet_crossflow.py [OUT.su2]   (default jet_crossflow.su2)
"""

import sys

from pyaeroflow.mesh import write_su2
from pyaeroflow.mesh.library import jet_crossflow_mesh


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

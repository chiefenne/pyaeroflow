"""Split part of a boundary marker into a new marker, e.g. a suction or blowing slot.

The faces of MARKER whose centres lie in XMIN <= x <= XMAX on the chosen side
(``upper``: y > 0, ``lower``: y < 0, ``both``) move to the new marker NAME;
all other markers are copied unchanged. The new marker can then get its own
boundary condition, e.g. ``type = "transpiration-wall"``.

Usage:
    python split_marker.py IN.su2 OUT.su2 MARKER NAME XMIN XMAX [upper|lower|both]

Example (2.5 % chord slot on the NACA 0012 upper surface, x/c = 0.075 - 0.100):
    python split_marker.py n0012_449.su2 n0012_449_slot.su2 airfoil slot 0.075 0.1 upper
"""

import sys

import numpy as np

from pyaeroflow.mesh import read_su2, split_boundary, write_su2


def split_marker(mesh, marker, name, xmin, xmax, side="upper"):
    """Return the mesh with the faces of ``marker`` in [xmin, xmax] in ``name``."""
    if marker not in mesh.boundary_faces:
        raise ValueError(
            f"no marker {marker!r}; markers: {sorted(mesh.boundary_faces)}"
        )
    if name in mesh.boundary_faces:
        raise ValueError(f"marker {name!r} exists already")
    if side not in ("upper", "lower", "both"):
        raise ValueError(f"side must be upper, lower or both, not {side!r}")

    def slot(x, y):
        on_side = {"upper": y > 0.0, "lower": y < 0.0, "both": np.ones_like(y, bool)}
        return (x >= xmin) & (x <= xmax) & on_side[side]

    centre = mesh.face_centre[mesh.boundary_faces[marker]]
    if not slot(centre[:, 0], centre[:, 1]).any():
        raise ValueError(f"no faces of {marker!r} in x = [{xmin}, {xmax}] ({side})")
    return split_boundary(mesh, marker, {name: slot, marker: lambda x, y: ~slot(x, y)})


def main(argv):
    """Command line: see the module docstring."""
    if len(argv) not in (7, 8):
        sys.exit(__doc__)
    src, dst, marker, name = argv[1:5]
    xmin, xmax = float(argv[5]), float(argv[6])
    side = argv[7] if len(argv) == 8 else "upper"
    mesh = read_su2(src)
    new = split_marker(mesh, marker, name, xmin, xmax, side)
    write_su2(dst, new)
    picked = new.boundary_faces[name]
    xc = new.face_centre[picked, 0]
    length = new.face_length[picked].sum()
    print(
        f"{dst}: {picked.size} faces of {marker!r} -> {name!r}, "
        f"x = {xc.min():.4f} .. {xc.max():.4f}, slot length {length:.5f}"
    )


if __name__ == "__main__":
    main(sys.argv)

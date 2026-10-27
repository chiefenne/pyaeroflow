"""Mesh reading by file type."""

from pathlib import Path

from pyaeroflow.mesh.gmsh import read_msh
from pyaeroflow.mesh.mesh import MeshError
from pyaeroflow.mesh.su2 import read_su2


def read_mesh(path, scale=1.0):
    """Read a mesh file, choosing the reader from the extension.

    ``.su2`` is read by :func:`read_su2`, ``.msh`` by :func:`read_msh`.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".su2":
        return read_su2(path, scale)
    if suffix == ".msh":
        return read_msh(path, scale)
    raise MeshError(f"{path}: unknown mesh file type '{path.suffix}' (.su2 or .msh)")

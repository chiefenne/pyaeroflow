"""Reader and writer for 2D SU2 native mesh files (``.su2``), requirement D43.

Supported: ``NDIME= 2`` with triangles (type 5) and quadrilaterals (type 9);
boundary markers made of line elements (type 3). Keywords may appear in any
order; trailing element or point indices are ignored. Comments start with ``%``.
"""

from pathlib import Path

import numpy as np

from pyaeroflow.mesh.mesh import Mesh, MeshError

_NODES_PER_ELEMENT = {3: 2, 5: 3, 9: 4}


def read_su2(path, scale=1.0):
    """Read a 2D ``.su2`` mesh.

    Parameters
    ----------
    path : str or Path
        Mesh file.
    scale : float
        Factor applied to all coordinates (e.g. 0.001 for mm to m).

    Returns
    -------
    Mesh
        Boundary-face sets are named after the ``MARKER_TAG`` entries.
    """
    lines = _content_lines(Path(path))
    nodes = cells = None
    markers = {}
    pos = 0
    while pos < len(lines):
        keyword, value = _keyword(lines[pos])
        pos += 1
        if keyword == "NDIME":
            if int(value) != 2:
                raise MeshError(
                    f"{path}: only 2D meshes are supported (NDIME= {value})"
                )
        elif keyword == "NPOIN":
            n = int(value.split()[0])
            rows = [lines[pos + k].split() for k in range(n)]
            nodes = np.array([[float(r[0]), float(r[1])] for r in rows])
            pos += n
        elif keyword == "NELEM":
            n = int(value)
            cells = [_element(lines[pos + k], path, allowed=(5, 9)) for k in range(n)]
            pos += n
        elif keyword == "MARKER_TAG":
            name = value.strip()
            key, count = _keyword(lines[pos])
            if key != "MARKER_ELEMS":
                raise MeshError(
                    f"{path}: MARKER_ELEMS expected after MARKER_TAG= {name}"
                )
            n = int(count)
            markers[name] = np.array(
                [_element(lines[pos + 1 + k], path, allowed=(3,)) for k in range(n)]
            ).reshape(-1, 2)
            pos += 1 + n
        # NMARK and unknown keywords carry no data we need.
    if nodes is None or cells is None:
        raise MeshError(f"{path}: NPOIN or NELEM section missing")
    return Mesh.from_cells(scale * nodes, cells, markers)


def _content_lines(path):
    """Return the non-empty lines without comments."""
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("%", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _keyword(line):
    """Split ``KEY= value``; return (None, None) for data lines."""
    if "=" not in line:
        return None, None
    key, value = line.split("=", 1)
    return key.strip().upper(), value.strip()


def _element(line, path, allowed):
    """Parse one element line: type id, node indices, optional element index."""
    parts = line.split()
    etype = int(parts[0])
    if etype not in allowed:
        raise MeshError(f"{path}: unsupported element type {etype} in line '{line}'")
    k = _NODES_PER_ELEMENT[etype]
    return [int(p) for p in parts[1 : 1 + k]]


def write_su2(path, mesh):
    """Write a mesh as a 2D ``.su2`` file (triangles, quadrilaterals only).

    Boundary-face sets become markers. Polygon cells with more than four
    nodes have no SU2 element type and raise ``MeshError``.
    """
    counts = mesh.cell_n_nodes
    if np.any(counts > 4):
        raise MeshError("SU2 files support triangles and quadrilaterals only")
    lines = ["NDIME= 2", f"NELEM= {mesh.n_cells}"]
    for i in range(mesh.n_cells):
        nodes = mesh.cell(i)
        etype = 5 if nodes.size == 3 else 9
        lines.append(f"{etype} " + " ".join(str(k) for k in nodes) + f" {i}")
    lines.append(f"NPOIN= {mesh.n_nodes}")
    lines += [f"{x:.16e} {y:.16e} {k}" for k, (x, y) in enumerate(mesh.nodes)]
    lines.append(f"NMARK= {len(mesh.boundary_faces)}")
    for name, faces in mesh.boundary_faces.items():
        lines.append(f"MARKER_TAG= {name}")
        lines.append(f"MARKER_ELEMS= {faces.size}")
        lines += [f"3 {a} {b}" for a, b in mesh.face_nodes[faces]]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)

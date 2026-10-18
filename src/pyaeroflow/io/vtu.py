"""VTU (VTK XML unstructured grid) writer, requirement D45.

Writes 2D meshes (triangles, quadrilaterals, polygons) with cell data and
optional point data for ParaView. Arrays are stored inline as base64-encoded
binary (default) or as ASCII text.
"""

import base64
from pathlib import Path

import numpy as np

_VTK_TRIANGLE, _VTK_QUAD, _VTK_POLYGON = 5, 9, 7
_VTK_TYPE_NAMES = {
    np.dtype(np.float64): "Float64",
    np.dtype(np.int64): "Int64",
    np.dtype(np.uint8): "UInt8",
}


def write_vtu(path, mesh, cell_data=None, point_data=None, binary=True):
    """Write a mesh and data fields to a ``.vtu`` file.

    Parameters
    ----------
    path : str or Path
        Output file.
    mesh : Mesh
        The mesh.
    cell_data : dict of str to ndarray, optional
        Values per cell: shape (n_cells,) for scalars, (n_cells, 2) for
        vectors (written with a zero z-component).
    point_data : dict of str to ndarray, optional
        Values per node, same shapes as ``cell_data`` with n_nodes rows.
    binary : bool
        Inline base64 binary (smaller, faster) or ASCII (human-readable).

    Returns
    -------
    Path
        The written file.
    """
    path = Path(path)
    points = np.column_stack((mesh.nodes, np.zeros(mesh.n_nodes)))
    n_per_cell = mesh.cell_n_nodes
    types = np.full(mesh.n_cells, _VTK_POLYGON, dtype=np.uint8)
    types[n_per_cell == 3] = _VTK_TRIANGLE
    types[n_per_cell == 4] = _VTK_QUAD

    def array(name, values, components=1):
        values = np.ascontiguousarray(values)
        if values.dtype.kind == "f":
            values = values.astype(np.float64)
        elif values.dtype.kind in "iu" and values.dtype != np.uint8:
            values = values.astype(np.int64)
        fmt = "binary" if binary else "ascii"
        head = (
            f'<DataArray type="{_VTK_TYPE_NAMES[values.dtype]}" Name="{name}" '
            f'NumberOfComponents="{components}" format="{fmt}">'
        )
        if binary:
            raw = values.tobytes()
            body = base64.b64encode(np.uint64(len(raw)).tobytes() + raw).decode()
        else:
            body = " ".join(repr(v) for v in values.ravel().tolist())
        return f"{head}\n{body}\n</DataArray>"

    def data_section(tag, data, n_rows):
        if not data:
            return f"<{tag}/>"
        parts = []
        for name, values in data.items():
            values = np.asarray(values)
            if values.shape[0] != n_rows:
                raise ValueError(f"{tag} '{name}': expected {n_rows} rows")
            if values.ndim == 2 and values.shape[1] == 2:
                values = np.column_stack((values, np.zeros(n_rows)))
                parts.append(array(name, values, 3))
            elif values.ndim == 1:
                parts.append(array(name, values))
            else:
                raise ValueError(f"{tag} '{name}': shape must be (n,) or (n, 2)")
        return f"<{tag}>\n" + "\n".join(parts) + f"\n</{tag}>"

    xml = "\n".join(
        [
            '<?xml version="1.0"?>',
            '<VTKFile type="UnstructuredGrid" version="1.0" '
            'byte_order="LittleEndian" header_type="UInt64">',
            "<UnstructuredGrid>",
            f'<Piece NumberOfPoints="{mesh.n_nodes}" NumberOfCells="{mesh.n_cells}">',
            "<Points>",
            array("Points", points, 3),
            "</Points>",
            "<Cells>",
            array("connectivity", mesh.cell_nodes),
            array("offsets", mesh.cell_ptr[1:]),
            array("types", types),
            "</Cells>",
            data_section("CellData", cell_data, mesh.n_cells),
            data_section("PointData", point_data, mesh.n_nodes),
            "</Piece>",
            "</UnstructuredGrid>",
            "</VTKFile>",
            "",
        ]
    )
    path.write_text(xml)
    return path


def write_pvd(path, entries):
    """Write a ParaView data collection (.pvd) of a time series (D45).

    Parameters
    ----------
    path : str or Path
    entries : sequence of (float, str)
        ``(time, file)`` pairs; ``file`` is relative to the directory of the
        .pvd file.
    """
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    lines += [
        f'    <DataSet timestep="{t!r}" group="" part="0" file="{f}"/>'
        for t, f in entries
    ]
    lines += ["  </Collection>", "</VTKFile>", ""]
    Path(path).write_text("\n".join(lines))

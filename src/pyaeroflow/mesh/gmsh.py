"""Reader for 2D Gmsh meshes (``.msh``), requirement L6.

Supported file formats (``$MeshFormat``): 2.0 to 2.2 (Gmsh 2.x and 3.x, and
every later Gmsh that writes ``Mesh.MshFileVersion = 2.2``), 4.0 (ASCII only,
as Gmsh itself cannot write it binary) and 4.1 (Gmsh 4.1 up to the current
release), each in ASCII and binary form (both byte orders). Formats 1 and 3
are rejected with a clear message.

Cells are the 2D elements: triangles (type 2) and quadrangles (type 3); the
second-order types 9, 10 and 16 are read with their corner nodes. Boundary
faces are the line elements (types 1, 8, 26), named after the physical group
of the curve (``$PhysicalNames``); a curve without a name is called
``physical_<tag>``, or ``curve_<tag>`` when the mesh has no physical groups
at all. Point elements are ignored; 3D elements are an error. The nodes must
lie in the plane z = 0.
"""

import re
import struct
from pathlib import Path

import numpy as np

from pyaeroflow.mesh.mesh import Mesh, MeshError

# Nodes per element of the Gmsh element types that can appear in a file.
_NODES = {
    1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5, 8: 3, 9: 6, 10: 9, 11: 10, 12: 27,
    13: 18, 14: 14, 15: 1, 16: 8, 17: 20, 18: 15, 19: 13, 20: 9, 21: 10, 22: 12,
    23: 15, 24: 15, 25: 21, 26: 4, 27: 5, 28: 6, 29: 20, 30: 35, 31: 56,
}  # fmt: skip
# Corner nodes (the first ones in Gmsh's numbering) of the 2D and 1D types.
_LINES = {1: 2, 8: 2, 26: 2}
_CELLS = {2: 3, 3: 4, 9: 3, 10: 4, 16: 4}
_SOLIDS = {4, 5, 6, 7, 11, 12, 13, 14, 17, 18, 19, 29, 30, 31}

_NAME = re.compile(rb'^\s*(\d+)\s+(\d+)\s+"(.*)"\s*$')


def read_msh(path, scale=1.0):
    """Read a 2D Gmsh ``.msh`` mesh.

    Parameters
    ----------
    path : str or Path
        Mesh file (MSH format 2.x or 4.x, ASCII or binary).
    scale : float
        Factor applied to all coordinates (e.g. 0.001 for mm to m).

    Returns
    -------
    Mesh
        Boundary-face sets are named after the physical groups of the
        curves, see the module documentation.

    Raises
    ------
    MeshError
        For unsupported formats or element types, 3D meshes, meshes outside
        the plane z = 0, and meshes without 2D elements.
    """
    path = Path(path)
    cur = _Cursor(path.read_bytes(), path)
    names, entity_phys = {}, {}
    nodes = elements = None
    version = None
    while True:
        line = cur.readline()
        if line is None:
            break
        if not line.startswith(b"$") or line.startswith(b"$End"):
            continue
        section = line[1:].decode()
        if section == "MeshFormat":
            version = _mesh_format(cur)
        elif version is None:
            raise MeshError(f"{path}: not a Gmsh MSH file ($MeshFormat missing)")
        elif section == "PhysicalNames":
            names = _physical_names(cur)
        elif section == "Entities":
            entity_phys = _entities(cur)
        elif section == "Nodes":
            nodes = (_nodes_2 if cur.major == 2 else _nodes_4)(cur)
        elif section == "Elements":
            elements = (_elements_2 if cur.major == 2 else _elements_4)(
                cur, entity_phys
            )
        else:
            continue
        cur.skip_to_end(section)
    if version is None:
        raise MeshError(f"{path}: not a Gmsh MSH file ($MeshFormat missing)")
    if nodes is None or elements is None:
        raise MeshError(f"{path}: $Nodes or $Elements section missing")
    try:
        return _build(scale, nodes, elements, names)
    except MeshError as err:
        raise MeshError(f"{path}: {err}") from None


class _Cursor:
    """Position in the file bytes with line and binary reads."""

    def __init__(self, data, path):
        self.data, self.path, self.pos = data, path, 0
        self.major = 0
        self.v40 = False
        self.binary = False
        self.size_t = 8
        self.end = "<"

    def readline(self):
        if self.pos >= len(self.data):
            return None
        stop = self.data.find(b"\n", self.pos)
        stop = len(self.data) if stop < 0 else stop
        line = self.data[self.pos : stop].strip()
        self.pos = stop + 1
        return line

    def tokens(self):
        """Whitespace-separated tokens of the next non-empty line."""
        line = self.readline()
        while line == b"":
            line = self.readline()
        if line is None:
            raise MeshError(f"{self.path}: unexpected end of file")
        return line.split()

    def take(self, dtype, count):
        dtype = np.dtype(dtype).newbyteorder(self.end)
        out = np.frombuffer(self.data, dtype, count, self.pos)
        self.pos += dtype.itemsize * count
        return out

    def ints(self, n_lines, width=None):
        """Read ``n_lines`` ASCII lines as an int array (rows of equal width)."""
        rows = [self.readline().split() for _ in range(n_lines)]
        return np.array(rows, dtype=np.int64).reshape(
            n_lines, -1 if width is None else width
        )

    def skip_to_end(self, section):
        marker = b"$End" + section.encode()
        i = self.data.find(marker, self.pos)
        if i < 0:
            raise MeshError(f"{self.path}: ${section} section is not closed")
        self.pos = i + len(marker)


def _mesh_format(cur):
    parts = cur.tokens()
    version = parts[0].decode()
    try:
        major = int(version.split(".")[0])
    except ValueError:
        raise MeshError(f"{cur.path}: bad MSH version '{version}'") from None
    if major not in (2, 4):
        raise MeshError(
            f"{cur.path}: MSH format {version} is not supported; "
            "export the mesh as MSH 2.2 or 4.1"
        )
    cur.major = major
    cur.v40 = version == "4" or version.startswith("4.0")
    cur.binary = int(parts[1]) == 1
    cur.size_t = int(parts[2])
    if cur.binary:
        if cur.v40:
            raise MeshError(f"{cur.path}: binary MSH 4.0 is not supported (use 4.1)")
        # The int 1 after the header line reveals the byte order.
        if np.frombuffer(cur.data, "<i4", 1, cur.pos)[0] != 1:
            if np.frombuffer(cur.data, ">i4", 1, cur.pos)[0] != 1:
                raise MeshError(f"{cur.path}: binary MSH file with a bad marker")
            cur.end = ">"
        cur.pos += 4
    return version


def _physical_names(cur):
    n = int(cur.tokens()[0])
    names = {}
    for _ in range(n):
        m = _NAME.match(cur.readline())
        if m is None:
            raise MeshError(f"{cur.path}: bad line in $PhysicalNames")
        names[(int(m.group(1)), int(m.group(2)))] = m.group(3).decode()
    return names


def _entities(cur):
    """Physical tags of the curves: ``{curve tag: first physical tag}``."""
    out = {}
    if cur.binary:
        st = f"u{cur.size_t}"
        counts = cur.take(st, 4)
        for dim in range(4):
            for _ in range(int(counts[dim])):
                tag = int(cur.take("i4", 1)[0])
                cur.take("f8", 3 if dim == 0 else 6)
                nphys = int(cur.take(st, 1)[0])
                phys = cur.take("i4", nphys)
                if dim > 0:
                    cur.take("i4", int(cur.take(st, 1)[0]))
                if dim == 1 and nphys:
                    out[tag] = int(phys[0])
        return out
    counts = [int(t) for t in cur.tokens()]
    for dim in range(4):
        # Points carry a box (6 numbers) in 4.0 and only x y z in 4.1.
        k = 7 if dim > 0 or cur.v40 else 4
        for _ in range(counts[dim]):
            t = cur.tokens()
            nphys = int(t[k])
            if dim == 1 and nphys:
                out[int(t[0])] = int(t[k + 1])
    return out


def _nodes_2(cur):
    """Format 2: ``(tags, xyz)``."""
    n = int(cur.tokens()[0])
    if cur.binary:
        rec = cur.take([("id", "i4"), ("xyz", "f8", (3,))], n)
        return rec["id"].astype(np.int64), rec["xyz"].astype(float)
    rows = np.array([cur.readline().split() for _ in range(n)], dtype=float)
    return rows[:, 0].astype(np.int64), rows[:, 1:4]


def _nodes_4(cur):
    if cur.binary:
        n_blocks = int(cur.take(f"u{cur.size_t}", 4)[0])
    else:
        n_blocks = int(cur.tokens()[0])
    tags, xyz = [], []
    for _ in range(n_blocks):
        if cur.binary:
            parametric = int(cur.take("i4", 3)[2])
            n = int(cur.take(f"u{cur.size_t}", 1)[0])
            if parametric:
                raise MeshError("parametric nodes are not supported")
            tags.append(cur.take(f"u{cur.size_t}", n).astype(np.int64))
            xyz.append(cur.take("f8", 3 * n).reshape(n, 3))
            continue
        a, b, parametric, n = (int(v) for v in cur.tokens())
        dim = b if cur.v40 else a  # 4.0: entityTag entityDim, 4.1: dim tag
        extra = dim if parametric else 0
        if cur.v40:  # 4.0: one line "tag x y z" per node
            rows = _float_rows(cur, n, 4 + extra)
            tags.append(rows[:, 0].astype(np.int64))
            xyz.append(rows[:, 1:4])
        else:  # 4.1: all tags first, then all coordinates
            tags.append(cur.ints(n, 1)[:, 0])
            xyz.append(_float_rows(cur, n, 3 + extra)[:, :3])
    if not tags:
        return np.zeros(0, np.int64), np.zeros((0, 3))
    return np.concatenate(tags), np.concatenate(xyz)


def _float_rows(cur, n, width):
    rows = [cur.readline().split() for _ in range(n)]
    return np.array(rows, dtype=float).reshape(n, width)


def _elements_2(cur, entity_phys):
    n = int(cur.tokens()[0])
    out = []
    if cur.binary:
        groups, done = {}, 0
        while done < n:
            etype, count, ntags = struct.unpack_from(cur.end + "3i", cur.data, cur.pos)
            cur.pos += 12
            nn = _NODES.get(etype)
            if nn is None:
                raise MeshError(f"unknown element type {etype}")
            rec = cur.take(
                [("id", "i4"), ("tags", "i4", (ntags,)), ("nodes", "i4", (nn,))],
                count,
            )
            tags = rec["tags"]
            phys = tags[:, 0] if ntags > 0 else np.zeros(count, np.int64)
            ent = tags[:, 1] if ntags > 1 else np.zeros(count, np.int64)
            for p, e in set(zip(phys.tolist(), ent.tolist(), strict=True)):
                sel = (phys == p) & (ent == e)
                groups.setdefault((etype, p, e), []).append(rec["nodes"][sel])
            done += count
        return [
            (etype, p, e, np.concatenate(parts).astype(np.int64))
            for (etype, p, e), parts in groups.items()
        ]
    groups = {}
    for _ in range(n):
        t = cur.readline().split()
        etype, ntags = int(t[1]), int(t[2])
        phys = int(t[3]) if ntags > 0 else 0
        ent = int(t[4]) if ntags > 1 else 0
        groups.setdefault((etype, phys, ent), []).append(t[3 + ntags :])
    for (etype, p, e), rows in groups.items():
        out.append((etype, p, e, np.array(rows, dtype=np.int64)))
    return out


def _elements_4(cur, entity_phys):
    if cur.binary:
        n_blocks = int(cur.take(f"u{cur.size_t}", 4)[0])
    else:
        n_blocks = int(cur.tokens()[0])
    out = []
    for _ in range(n_blocks):
        if cur.binary:
            dim, tag, etype = (int(v) for v in cur.take("i4", 3))
            n = int(cur.take(f"u{cur.size_t}", 1)[0])
            nn = _NODES.get(etype)
            if nn is None:
                raise MeshError(f"{cur.path}: unknown element type {etype}")
            rec = cur.take(f"u{cur.size_t}", n * (1 + nn)).reshape(n, 1 + nn)
            conn = rec[:, 1:].astype(np.int64)
        else:
            a, b, c, n = (int(v) for v in cur.tokens())
            if cur.v40:
                tag, dim, etype = a, b, c  # 4.0: entityTag entityDim type
            else:
                dim, tag, etype = a, b, c
            conn = cur.ints(n)[:, 1:]
        phys = entity_phys.get(tag, 0) if dim == 1 else 0
        out.append((etype, phys, tag, conn))
    return out


def _build(scale, node_data, elements, names):
    tags, xyz = node_data
    blocks, lines = [], {}
    for etype, phys, ent, conn in elements:
        if etype == 15:
            continue
        if etype in _SOLIDS:
            raise MeshError("3D elements found; only 2D meshes are supported")
        if etype in _CELLS:
            blocks.append(conn[:, : _CELLS[etype]])
        elif etype in _LINES:
            if phys > 0:
                name = names.get((1, phys), f"physical_{phys}")
            else:
                name = f"curve_{ent}"
            lines.setdefault(name, []).append(conn[:, :2])
        else:
            raise MeshError(f"unsupported element type {etype}")
    if not blocks:
        raise MeshError(
            "no 2D elements (triangles, quadrangles) found; if the geometry has "
            "physical groups, add a Physical Surface, as Gmsh then writes only "
            "the elements in groups"
        )

    order = np.argsort(tags, kind="stable")
    sorted_tags = tags[order]

    def index(node_ids):
        i = np.minimum(np.searchsorted(sorted_tags, node_ids), tags.size - 1)
        if np.any(sorted_tags[i] != node_ids):
            raise MeshError("an element refers to a node that is not defined")
        return order[i]

    cells = [index(b) for b in blocks]
    used = np.unique(np.concatenate([c.ravel() for c in cells]))
    pts = xyz[used]
    extent = np.ptp(pts[:, :2], axis=0).max()
    if np.abs(pts[:, 2]).max() > 1e-8 * extent:
        raise MeshError("the mesh is not in the plane z = 0")

    def compact(idx):
        j = np.minimum(np.searchsorted(used, idx), used.size - 1)
        if np.any(used[j] != idx):
            raise MeshError("a boundary line has nodes that belong to no cell")
        return j

    cells = [compact(c) for c in cells]
    if len({c.shape[1] for c in cells}) == 1:
        cells = np.concatenate(cells)
    else:
        cells = [row for c in cells for row in c]
    markers = {
        name: compact(index(np.concatenate(parts))).reshape(-1, 2)
        for name, parts in lines.items()
    }
    return Mesh.from_cells(scale * pts[:, :2], cells, markers)

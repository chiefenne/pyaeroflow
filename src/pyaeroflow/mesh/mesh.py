"""Face-based mesh data structure for 2D cell-centred finite volumes.

Cells are arbitrary polygons (triangles, quadrilaterals, general polygons).
Every face (edge) knows its owner cell and, for interior faces, its neighbour
cell. Face normals point from the owner to the neighbour; on boundary faces
they point out of the domain.
"""

from dataclasses import dataclass

import numpy as np


class MeshError(ValueError):
    """Raised for invalid mesh topology, geometry or boundary-face sets."""


@dataclass(frozen=True)
class Mesh:
    """Immutable 2D mesh with cell-centred finite-volume geometry.

    Use :meth:`from_cells` to build a mesh; it derives faces and geometry and
    checks the boundary-face sets.

    Attributes
    ----------
    nodes : ndarray, shape (n_nodes, 2)
        Node coordinates.
    cell_ptr : ndarray, shape (n_cells + 1,)
        Offsets into ``cell_nodes`` (CSR layout).
    cell_nodes : ndarray
        Node indices of all cells, counter-clockwise per cell.
    face_nodes : ndarray, shape (n_faces, 2)
        Face end nodes, ordered counter-clockwise with respect to the owner.
    owner : ndarray, shape (n_faces,)
        Owner cell of each face (the lower cell index on interior faces).
    neighbour : ndarray, shape (n_faces,)
        Neighbour cell of each face; -1 on boundary faces.
    face_normal : ndarray, shape (n_faces, 2)
        Unit normal pointing from owner to neighbour (outward on boundaries).
    face_length : ndarray, shape (n_faces,)
        Face length (the face "area" in 2D).
    face_centre : ndarray, shape (n_faces, 2)
        Face midpoint.
    cell_area : ndarray, shape (n_cells,)
        Cell area (the cell "volume" in 2D).
    cell_centroid : ndarray, shape (n_cells, 2)
        Cell centroid.
    boundary_faces : dict of str to ndarray
        Face indices of each named set of boundary faces.
    """

    nodes: np.ndarray
    cell_ptr: np.ndarray
    cell_nodes: np.ndarray
    face_nodes: np.ndarray
    owner: np.ndarray
    neighbour: np.ndarray
    face_normal: np.ndarray
    face_length: np.ndarray
    face_centre: np.ndarray
    cell_area: np.ndarray
    cell_centroid: np.ndarray
    boundary_faces: dict

    @property
    def n_cells(self):
        """Number of cells."""
        return self.cell_ptr.size - 1

    @property
    def n_faces(self):
        """Number of faces."""
        return self.owner.size

    @property
    def n_nodes(self):
        """Number of nodes."""
        return self.nodes.shape[0]

    @property
    def cell_n_nodes(self):
        """Number of nodes (= faces) per cell."""
        return np.diff(self.cell_ptr)

    def cell(self, i):
        """Return the node indices of cell ``i`` (counter-clockwise)."""
        return self.cell_nodes[self.cell_ptr[i] : self.cell_ptr[i + 1]]

    def summary(self):
        """Return a short multi-line text description of the mesh."""
        n_per_cell = self.cell_n_nodes
        kinds = {3: "triangles", 4: "quadrilaterals"}
        lines = [f"nodes: {self.n_nodes}, cells: {self.n_cells}, faces: {self.n_faces}"]
        for k in np.unique(n_per_cell):
            name = kinds.get(int(k), f"{k}-gons")
            lines.append(f"  {name}: {np.count_nonzero(n_per_cell == k)}")
        lines.append(
            f"cell area: min {self.cell_area.min():.4e}, "
            f"max {self.cell_area.max():.4e}, total {self.cell_area.sum():.6e}"
        )
        for name, faces in self.boundary_faces.items():
            lines.append(f"boundary '{name}': {faces.size} faces")
        return "\n".join(lines)

    @classmethod
    def from_cells(cls, nodes, cells, boundary_edges):
        """Build a mesh from polygon cells and named boundary edges.

        Parameters
        ----------
        nodes : array_like, shape (n_nodes, 2)
            Node coordinates.
        cells : sequence of sequences of int, or ndarray (n_cells, k)
            Node indices per cell, in either orientation; cells are
            reoriented counter-clockwise.
        boundary_edges : dict of str to array_like, shape (m, 2)
            For each name, the node pairs of its boundary faces.

        Returns
        -------
        Mesh

        Raises
        ------
        MeshError
            For degenerate cells, edges shared by more than two cells, or
            boundary-face sets that are not a partition of the boundary faces.
        """
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise MeshError("nodes must have shape (n_nodes, 2)")
        cell_ptr, cell_nodes = _to_csr(cells)
        n_cells = cell_ptr.size - 1
        counts = np.diff(cell_ptr)
        if np.any(counts < 3):
            raise MeshError("every cell needs at least 3 nodes")

        # Edges of every cell: node a -> node b, following the stored order.
        cell_of_slot = np.repeat(np.arange(n_cells), counts)
        nxt = np.arange(cell_nodes.size) + 1
        nxt[cell_ptr[1:] - 1] = cell_ptr[:-1]  # wrap around to the first node
        a, b = cell_nodes, cell_nodes[nxt]

        # Orient every cell counter-clockwise (positive shoelace area).
        cross = nodes[a, 0] * nodes[b, 1] - nodes[b, 0] * nodes[a, 1]
        area2 = np.bincount(cell_of_slot, weights=cross, minlength=n_cells)
        if np.any(area2 == 0.0):
            bad = np.flatnonzero(area2 == 0.0)
            raise MeshError(f"{bad.size} cells with zero area, e.g. cell {bad[0]}")
        flip = area2 < 0.0
        if np.any(flip):
            cell_nodes = _reverse_cells(cell_ptr, cell_nodes, flip)
            a, b = cell_nodes, cell_nodes[nxt]
            cross = nodes[a, 0] * nodes[b, 1] - nodes[b, 0] * nodes[a, 1]
        cell_area = 0.5 * np.bincount(cell_of_slot, weights=cross, minlength=n_cells)

        # Centroid of a polygon: sum over edges of (p_a + p_b) * cross / (6 A).
        cx = np.bincount(
            cell_of_slot, weights=(nodes[a, 0] + nodes[b, 0]) * cross, minlength=n_cells
        )
        cy = np.bincount(
            cell_of_slot, weights=(nodes[a, 1] + nodes[b, 1]) * cross, minlength=n_cells
        )
        cell_centroid = np.column_stack((cx, cy)) / (6.0 * cell_area[:, None])

        # Faces: unique undirected edges. Owner = lower cell index.
        n_nodes = nodes.shape[0]
        key = np.minimum(a, b).astype(np.int64) * n_nodes + np.maximum(a, b)
        order = np.lexsort((cell_of_slot, key))
        key_sorted = key[order]
        first = np.ones(key_sorted.size, dtype=bool)
        first[1:] = key_sorted[1:] != key_sorted[:-1]
        starts = np.flatnonzero(first)
        multiplicity = np.diff(np.append(starts, key_sorted.size))
        if np.any(multiplicity > 2):
            raise MeshError("edges shared by more than two cells (non-manifold mesh)")
        own_slot = order[starts]
        owner = cell_of_slot[own_slot]
        neighbour = np.full(starts.size, -1, dtype=np.int64)
        interior = multiplicity == 2
        neighbour[interior] = cell_of_slot[order[starts[interior] + 1]]
        face_nodes = np.column_stack((a[own_slot], b[own_slot]))
        face_key = key_sorted[starts]

        # Face geometry; the owner's counter-clockwise edge gives an outward
        # normal (dy, -dx).
        d = nodes[face_nodes[:, 1]] - nodes[face_nodes[:, 0]]
        face_length = np.hypot(d[:, 0], d[:, 1])
        if np.any(face_length == 0.0):
            raise MeshError("faces with zero length (duplicate nodes)")
        face_normal = np.column_stack((d[:, 1], -d[:, 0])) / face_length[:, None]
        face_centre = 0.5 * (nodes[face_nodes[:, 0]] + nodes[face_nodes[:, 1]])

        boundary_faces = _assign_boundary_faces(
            boundary_edges, face_key, neighbour, n_nodes
        )
        _check_boundary_partition(boundary_faces, neighbour, face_centre)
        _check_orientation(owner, neighbour, face_normal, face_centre, cell_centroid)

        return cls(
            nodes=nodes,
            cell_ptr=cell_ptr,
            cell_nodes=cell_nodes,
            face_nodes=face_nodes,
            owner=owner,
            neighbour=neighbour,
            face_normal=face_normal,
            face_length=face_length,
            face_centre=face_centre,
            cell_area=cell_area,
            cell_centroid=cell_centroid,
            boundary_faces=boundary_faces,
        )


def _to_csr(cells):
    """Convert a cell list or a regular (n_cells, k) array to CSR arrays."""
    if isinstance(cells, np.ndarray) and cells.ndim == 2:
        n_cells, k = cells.shape
        cell_ptr = np.arange(0, n_cells * k + 1, k, dtype=np.int64)
        return cell_ptr, cells.astype(np.int64).ravel()
    lengths = np.array([len(c) for c in cells], dtype=np.int64)
    cell_ptr = np.concatenate(([0], np.cumsum(lengths)))
    cell_nodes = (
        np.concatenate([np.asarray(c, dtype=np.int64) for c in cells])
        if len(cells)
        else np.zeros(0, dtype=np.int64)
    )
    return cell_ptr, cell_nodes


def _reverse_cells(cell_ptr, cell_nodes, flip):
    """Return ``cell_nodes`` with the node order of flagged cells reversed."""
    out = cell_nodes.copy()
    for i in np.flatnonzero(flip):
        s, e = cell_ptr[i], cell_ptr[i + 1]
        out[s:e] = cell_nodes[s:e][::-1]
    return out


def _assign_boundary_faces(boundary_edges, face_key, neighbour, n_nodes):
    """Map named boundary node pairs to face indices."""
    result = {}
    for name, edges in boundary_edges.items():
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        key = np.minimum(edges[:, 0], edges[:, 1]) * n_nodes + np.maximum(
            edges[:, 0], edges[:, 1]
        )
        idx = np.searchsorted(face_key, key)
        idx_c = np.minimum(idx, face_key.size - 1)
        missing = face_key[idx_c] != key
        if np.any(missing):
            pair = edges[np.flatnonzero(missing)[0]]
            raise MeshError(
                f"boundary '{name}': {np.count_nonzero(missing)} node pairs are "
                f"not mesh faces, e.g. nodes ({pair[0]}, {pair[1]})"
            )
        result[name] = np.unique(idx_c)
    return result


def _check_boundary_partition(boundary_faces, neighbour, face_centre, n_show=5):
    """Check that the named sets partition the boundary faces exactly.

    Every boundary face must belong to exactly one set, and sets may contain
    boundary faces only (requirement D39, R1).
    """
    problems = []
    count = np.zeros(neighbour.size, dtype=np.int64)
    for name, faces in boundary_faces.items():
        inner = faces[neighbour[faces] >= 0]
        if inner.size:
            problems.append(
                f"boundary '{name}' contains {inner.size} interior faces, "
                f"e.g. at {_fmt_points(face_centre[inner[:n_show]])}"
            )
        np.add.at(count, faces, 1)
    on_boundary = neighbour < 0
    unassigned = np.flatnonzero(on_boundary & (count == 0))
    if unassigned.size:
        problems.append(
            f"{unassigned.size} boundary faces belong to no set, "
            f"e.g. at {_fmt_points(face_centre[unassigned[:n_show]])}"
        )
    multiple = np.flatnonzero(count > 1)
    if multiple.size:
        problems.append(
            f"{multiple.size} faces belong to more than one set, "
            f"e.g. at {_fmt_points(face_centre[multiple[:n_show]])}"
        )
    if problems:
        raise MeshError("invalid boundary-face sets:\n  " + "\n  ".join(problems))


def _check_orientation(owner, neighbour, face_normal, face_centre, centroid, n_show=5):
    """Check that every face normal points from its owner to its neighbour.

    The cell-centred discretisation needs ``(c_neighbour - c_owner) . n > 0``
    (the alpha-damping viscous flux divides by it). A self-intersecting
    (bow-tie) or folded cell violates it; the uniform state then has a
    residual and the pseudo-time iteration drifts instead of converging.
    """
    inner = neighbour >= 0
    far = np.where(inner[:, None], centroid[np.maximum(neighbour, 0)], face_centre)
    delta = far - centroid[owner]
    bad = np.flatnonzero(np.einsum("ij,ij->i", delta, face_normal) <= 0.0)
    if bad.size:
        cells = np.unique(np.concatenate((owner[bad], neighbour[bad][inner[bad]])))
        raise MeshError(
            f"{bad.size} faces have their neighbouring cell centroid on the wrong "
            f"side ({cells.size} self-intersecting or folded cells), e.g. at "
            f"{_fmt_points(face_centre[bad[:n_show]])}; fix the mesh in the generator"
        )


def _fmt_points(points):
    """Format face-centre coordinates for error messages."""
    return ", ".join(f"({x:.6g}, {y:.6g})" for x, y in points)

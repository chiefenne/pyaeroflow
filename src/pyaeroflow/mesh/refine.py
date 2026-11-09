"""Local mesh refinement by cell splitting with hanging nodes (R1).

A quadrilateral cell is split into ``nu x nv`` children along its own two
logical directions: ``u`` runs along its first side (nodes 0 -> 1), ``v`` along
its second side (nodes 1 -> 2). ``nu`` and ``nv`` are arbitrary positive
integers, so a thin near-wall cell can be split 1 x n along the wall only.

The neighbours of a split cell are not touched: the new nodes on the shared
side are hanging nodes, and the neighbour becomes a polygon with extra
vertices on that side. :class:`~pyaeroflow.mesh.mesh.Mesh` stores arbitrary
polygons, so the finite-volume operators need no special treatment.

Topology is kept exactly, independent of node coordinates:

* every straight segment of the original mesh, and every grid line created
  inside a split cell, is a *line* with a parameter ``t`` in [0, 1] from its
  first to its last node; the nodes on it are stored by exact rational
  parameter (:class:`fractions.Fraction`), so the node at ``t = 1/2`` created
  from one side is found again from the other;
* a logical side of a cell is a range ``(line, t0, t1)`` on one line;
* a cell's polygon is the corner of each side followed by every node of the
  line strictly inside the side's range (its hanging nodes).

Children keep the parent's orientation (``u`` of the children is ``u`` of the
parent), so a direction such as "along the wall" is inherited through all
refinement cycles. Triangles and other polygons of the original mesh are kept
as they are (they can gain hanging nodes but are never split).

Every call of :meth:`Refinement.split` returns the map from the new cells to
the cells before the call (for the state transfer by injection) and appends
one entry to :attr:`Refinement.log`; :meth:`Refinement.replay` repeats a log
on the original mesh, so a refined case can be restarted from the original
mesh file without a new mesh format.
"""

from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from pyaeroflow.mesh.mesh import Mesh, MeshError

_ZERO, _ONE = Fraction(0), Fraction(1)


class _Line:
    """Straight segment with the nodes on it, ordered by exact parameter."""

    __slots__ = ("first", "last", "params", "nodes", "boundary")

    def __init__(self, first, last, boundary=None):
        self.first = first
        self.last = last
        self.params = [_ZERO, _ONE]
        self.nodes = [first, last]
        self.boundary = boundary  # name of the boundary set, or None

    def inside(self, t0, t1):
        """Nodes strictly between t0 and t1, in the order t0 -> t1."""
        lo, hi = (t0, t1) if t0 < t1 else (t1, t0)
        i, j = bisect_right(self.params, lo), bisect_left(self.params, hi)
        out = self.nodes[i:j]
        return out if t0 < t1 else out[::-1]


@dataclass
class _Cell:
    """Leaf cell: corner nodes and one side range per corner (CCW)."""

    corners: list
    sides: list  # (line id, t0, t1) from corners[k] to corners[k + 1]
    origin: int  # cell index in the original mesh
    nu: int = 1  # accumulated split factors along u and v
    nv: int = 1

    @property
    def splittable(self):
        return len(self.corners) == 4


@dataclass
class Refinement:
    """Refinement state built on an original mesh.

    Use :meth:`from_mesh` to create it, :meth:`split` to refine and
    :attr:`mesh` for the current :class:`~pyaeroflow.mesh.mesh.Mesh`.

    Attributes
    ----------
    log : list of dict
        One entry per :meth:`split` call: ``cells`` (leaf indices at that
        time), ``nu``, ``nv`` and ``max_hanging``.
    """

    nodes: list
    lines: list
    cells: list
    boundary_names: tuple
    log: list = field(default_factory=list)
    curves: dict = field(default_factory=dict)  # wall line -> 4 control points
    chains: dict = field(default_factory=dict)  # node -> grid lines ending there
    decay: float = 20.0
    _moved: list = field(default_factory=list)
    _mesh: Mesh = None

    # -- construction --------------------------------------------------------

    @classmethod
    def from_mesh(cls, mesh, walls=(), corner_angle=30.0, decay=20.0):
        """Refinement state of an unrefined mesh (no split yet).

        Parameters
        ----------
        mesh : Mesh
            The original mesh.
        walls : sequence of str
            Boundary sets whose new nodes are placed on a curve instead of the
            straight face: a centripetal Catmull-Rom cubic through the face's
            two nodes and their neighbours along the wall.
        corner_angle : float
            Turning angle (degrees) between two wall faces above which their
            common node is a corner (e.g. a sharp or blunt trailing edge); the
            curve does not reach across a corner.
        decay : float
            A wall node moved by ``d`` onto the curve moves the nodes of the
            grid lines that start at it by ``d * (1 - s / (decay * |d|))``,
            ``s`` the distance from the wall, so thin near-wall cells move with
            the wall instead of folding.
        """
        nodes = [tuple(p) for p in mesh.nodes.tolist()]
        boundary_of_face = {}
        for name, faces in mesh.boundary_faces.items():
            for f in faces.tolist():
                boundary_of_face[f] = name
        lines = []
        line_of_key = {}
        for f, (a, b) in enumerate(mesh.face_nodes.tolist()):
            lo, hi = (a, b) if a < b else (b, a)
            line_of_key[(lo, hi)] = len(lines)
            lines.append(_Line(lo, hi, boundary_of_face.get(f)))
        cells = []
        ptr, cn = mesh.cell_ptr.tolist(), mesh.cell_nodes.tolist()
        for i in range(mesh.n_cells):
            corners = cn[ptr[i] : ptr[i + 1]]
            sides = []
            for k, a in enumerate(corners):
                b = corners[(k + 1) % len(corners)]
                lid = line_of_key[(a, b) if a < b else (b, a)]
                sides.append((lid, _ZERO, _ONE) if a < b else (lid, _ONE, _ZERO))
            cells.append(_Cell(list(corners), sides, i))
        ref = cls(nodes, lines, cells, tuple(mesh.boundary_faces), decay=decay)
        ref.curves = _wall_curves(mesh, walls, corner_angle, line_of_key)
        ref._mesh = mesh
        return ref

    # -- queries -------------------------------------------------------------

    @property
    def n_cells(self):
        """Number of leaf cells (= cells of :attr:`mesh`)."""
        return len(self.cells)

    @property
    def mesh(self):
        """The current mesh (cached until the next split)."""
        if self._mesh is None:
            self._mesh = self._build_mesh()
        return self._mesh

    @property
    def origin(self):
        """Original-mesh cell of every leaf cell."""
        return np.array([c.origin for c in self.cells], dtype=np.int64)

    @property
    def split_factors(self):
        """Accumulated split factors ``(nu, nv)`` of every leaf, shape (n, 2)."""
        return np.array([(c.nu, c.nv) for c in self.cells], dtype=np.int64)

    def hanging_counts(self):
        """Hanging nodes per logical side, shape (n_cells, 4); -1 for non-quads."""
        out = np.full((self.n_cells, 4), -1, dtype=np.int64)
        for i, c in enumerate(self.cells):
            if c.splittable:
                for k, (lid, t0, t1) in enumerate(c.sides):
                    out[i, k] = len(self.lines[lid].inside(t0, t1))
        return out

    def extents(self):
        """Mean lengths of the logical u and v sides of every leaf, shape (n, 2).

        For non-quadrilateral cells both entries are the square root of the
        cell area. Used for aspect ratios and size weights of indicators.
        """
        p = np.asarray(self.nodes)
        out = np.empty((self.n_cells, 2))
        mesh = self.mesh
        for i, c in enumerate(self.cells):
            if c.splittable:
                q = p[c.corners]
                d = np.hypot(*(np.roll(q, -1, axis=0) - q).T)
                out[i] = 0.5 * (d[0] + d[2]), 0.5 * (d[1] + d[3])
            else:
                out[i] = np.sqrt(mesh.cell_area[i])
        return out

    # -- refinement ----------------------------------------------------------

    def split(self, cells, nu=2, nv=2, max_hanging=1, column_aspect=None):
        """Split leaf cells and balance the result.

        Parameters
        ----------
        cells : array_like of int
            Leaf indices (cells of the current :attr:`mesh`) to split.
            Non-quadrilateral cells raise ``MeshError``.
        nu, nv : int or array_like of int
            Split factors along u and v, scalars or one per entry of ``cells``
            (1 = no split in that direction).
        max_hanging : int or None
            Balance rule: no logical side may carry more hanging nodes than
            this. A violating quad is split along that side into the fewest
            parts that satisfy the rule, repeated until no side violates it.
            ``None`` disables balancing.
        column_aspect : float or None
            Boundary-layer column rule: a split along one side is passed on
            to the quad across that side if that quad is thinner than
            ``1 / column_aspect`` across the side (its extent along the side
            over its extent across it exceeds ``column_aspect``), and so on.
            A tangential split of a wall cell then refines the whole
            wall-normal column of thin cells, and the hanging nodes end up
            where the cells are no longer thin. ``None`` disables the rule.

        Returns
        -------
        ndarray of int, shape (n_new_cells,)
            For every new leaf cell, its cell index before this call
            (children map to their parent, untouched cells to themselves).
        """
        cells = np.atleast_1d(np.asarray(cells, dtype=np.int64))
        nu = np.broadcast_to(np.asarray(nu, dtype=np.int64), cells.shape)
        nv = np.broadcast_to(np.asarray(nv, dtype=np.int64), cells.shape)
        if cells.size and (cells.min() < 0 or cells.max() >= self.n_cells):
            raise MeshError("split: cell index out of range")
        if np.any(nu < 1) or np.any(nv < 1):
            raise MeshError("split: factors must be positive integers")
        self.log.append(
            {
                "cells": cells.tolist(),
                "nu": nu.tolist(),
                "nv": nv.tolist(),
                "max_hanging": max_hanging,
                "column_aspect": column_aspect,
            }
        )
        factors = {}
        for i, a, b in zip(cells.tolist(), nu.tolist(), nv.tolist(), strict=True):
            fu, fv = factors.get(i, (1, 1))
            factors[i] = (max(fu, a), max(fv, b))
        source = np.arange(self.n_cells, dtype=np.int64)
        while factors:
            if column_aspect is not None:
                factors = self._columns(factors, column_aspect)
            source = source[self._apply(factors)]
            self._propagate()
            factors = {} if max_hanging is None else self._balance(max_hanging)
        self._mesh = None
        return source

    @classmethod
    def replay(cls, mesh, log, **options):
        """Repeat a refinement log on the original mesh.

        ``options`` are those of :meth:`from_mesh` (walls, corner_angle,
        decay) and must match the original run.
        """
        ref = cls.from_mesh(mesh, **options)
        for entry in log:
            ref.split(
                entry["cells"],
                entry["nu"],
                entry["nv"],
                entry["max_hanging"],
                entry.get("column_aspect"),
            )
        return ref

    def along_long_side(self, cells, n):
        """Split factors ``(nu, nv)`` that cut ``cells`` along their long side.

        A cell longer in u than in v gets ``(n, 1)``, otherwise ``(1, n)``:
        for a thin near-wall cell this is a split along the wall.
        """
        cells = np.atleast_1d(np.asarray(cells, dtype=np.int64))
        ext = self.extents()[cells]
        long_u = ext[:, 0] >= ext[:, 1]
        n = np.broadcast_to(np.asarray(n, dtype=np.int64), cells.shape)
        return np.where(long_u, n, 1), np.where(long_u, 1, n)

    def _apply(self, factors):
        """Split ``{leaf: (nu, nv)}``; return the old leaf of every new leaf."""
        new_cells, source = [], []
        for i, c in enumerate(self.cells):
            fu, fv = factors.get(i, (1, 1))
            if fu == 1 and fv == 1:
                new_cells.append(c)
                source.append(i)
                continue
            if not c.splittable:
                raise MeshError(
                    f"cell {i} has {len(c.corners)} corners; only quadrilaterals "
                    "can be split"
                )
            children = self._split_cell(c, fu, fv)
            new_cells.extend(children)
            source.extend([i] * len(children))
        self.cells = new_cells
        return np.asarray(source, dtype=np.int64)

    def _balance(self, max_hanging):
        """Quads that violate the hanging-node rule, with their split factors."""
        factors = {}
        for i, c in enumerate(self.cells):
            if not c.splittable:
                continue
            need = [1, 1]
            for k, (lid, t0, t1) in enumerate(c.sides):
                h = len(self.lines[lid].inside(t0, t1))
                if h > max_hanging:
                    n = -(-(h + 1) // (max_hanging + 1))  # ceil
                    need[k % 2] = max(need[k % 2], n)
            if need != [1, 1]:
                factors[i] = tuple(need)
        return factors

    def _node_on(self, lid, t):
        """Node of line ``lid`` at parameter ``t``, created if missing.

        A new node lies between its two neighbours on the line (linear in
        the parameter), or on the wall curve for wall lines.
        """
        line = self.lines[lid]
        j = bisect_left(line.params, t)
        if line.params[j] == t:
            return line.nodes[j]
        ta, tb = line.params[j - 1], line.params[j]
        (x0, y0), (x1, y1) = self.nodes[line.nodes[j - 1]], self.nodes[line.nodes[j]]
        s = float((t - ta) / (tb - ta))
        straight = (x0 + s * (x1 - x0), y0 + s * (y1 - y0))
        node = len(self.nodes)
        if lid in self.curves:
            # Wall lines are faces of the original mesh: the displacement is
            # taken from the straight face (end nodes never move), where the
            # nodes of the parallel grid lines above sit; measured from the
            # bracketing nodes it would be wrong once a neighbour on the same
            # face is already on the curve.
            p = _catmull_rom(self.curves[lid], float(t))
            (a0, b0), (a1, b1) = self.nodes[line.first], self.nodes[line.last]
            s = float(t)
            straight = (a0 + s * (a1 - a0), b0 + s * (b1 - b0))
            self.nodes.append(p)
            self._moved.append((node, (p[0] - straight[0], p[1] - straight[1])))
        else:
            self.nodes.append(straight)
        line.params.insert(j, t)
        line.nodes.insert(j, node)
        return node

    def _add_chain(self, chain):
        chain = list(chain)
        self.chains.setdefault(chain[0], []).append(chain)
        self.chains.setdefault(chain[-1], []).append(chain)

    def _propagate(self):
        """Carry the wall displacement of new wall nodes along their grid lines."""
        moved, self._moved = self._moved, []
        for start, d in moved:
            size = np.hypot(*d)
            if size == 0.0:
                continue
            reach = self.decay * size
            p0 = self.nodes[start]
            # Distances are measured on the positions before the move.
            stack, seen = [(start, 0.0, (p0[0] - d[0], p0[1] - d[1]))], {start}
            while stack:
                node, s, origin = stack.pop()
                for chain in self.chains.get(node, ()):
                    path = chain if chain[0] == node else chain[::-1]
                    if path[-1] in seen:
                        continue
                    prev = origin
                    dist = s
                    for k in path[1:]:
                        p = self.nodes[k]
                        dist += np.hypot(p[0] - prev[0], p[1] - prev[1])
                        prev = p
                        w = 1.0 - dist / reach
                        if w <= 0.0:
                            break
                        self.nodes[k] = (p[0] + w * d[0], p[1] + w * d[1])
                    else:
                        seen.add(path[-1])
                        stack.append((path[-1], dist, prev))

    def _columns(self, factors, aspect):
        """Pass splits on to thin quads across the split cell's sides."""
        across = {}
        for i, c in enumerate(self.cells):
            if c.splittable:
                for k, (lid, t0, t1) in enumerate(c.sides):
                    key = (lid, min(t0, t1), max(t0, t1))
                    across.setdefault(key, []).append((i, k))
        ext = self.extents()
        out = dict(factors)
        stack = list(factors)
        while stack:
            i = stack.pop()
            c = self.cells[i]
            if not c.splittable:
                continue
            for k, (lid, t0, t1) in enumerate(c.sides):
                f = out[i][k % 2]
                if f == 1:
                    continue
                for j, kk in across.get((lid, min(t0, t1), max(t0, t1)), ()):
                    if j == i:
                        continue
                    d = kk % 2
                    if ext[j, d] <= aspect * ext[j, 1 - d]:
                        continue
                    old = out.get(j, (1, 1))
                    if old[d] >= f:
                        continue
                    new = list(old)
                    new[d] = f
                    out[j] = tuple(new)
                    stack.append(j)
        return out

    def _new_line(self, first, last):
        self.lines.append(_Line(first, last))
        return len(self.lines) - 1

    def _split_cell(self, c, nu, nv):
        """Children of quad ``c`` split ``nu x nv``; row-major in (v, u)."""

        def at(side, s):  # parameter on the side's line at local s in [0, 1]
            lid, t0, t1 = side
            return lid, t0 + s * (t1 - t0)

        s0, s1, s2, s3 = c.sides
        fu = [Fraction(i, nu) for i in range(nu + 1)]
        fv = [Fraction(j, nv) for j in range(nv + 1)]
        # Grid nodes N[j][i] at (u, v) = (i/nu, j/nv).
        grid = [[None] * (nu + 1) for _ in range(nv + 1)]
        for i in range(nu + 1):
            grid[0][i] = self._node_on(*at(s0, fu[i]))
            grid[nv][i] = self._node_on(*at(s2, 1 - fu[i]))
        for j in range(nv + 1):
            grid[j][nu] = self._node_on(*at(s1, fv[j]))
            grid[j][0] = self._node_on(*at(s3, 1 - fv[j]))
        horiz = [None] * (nv + 1)  # line of grid row j (0 < j < nv), along u
        for j in range(1, nv):
            lid = self._new_line(grid[j][0], grid[j][nu])
            horiz[j] = lid
            for i in range(1, nu):
                grid[j][i] = self._node_on(lid, fu[i])
        vert = [None] * (nu + 1)  # line of grid column i (0 < i < nu), along v
        for i in range(1, nu):
            lid = self._new_line(grid[0][i], grid[nv][i])
            vert[i] = lid
            line = self.lines[lid]
            for j in range(1, nv):  # interior nodes already exist on the rows
                insort(line.params, fv[j])
                line.nodes.insert(line.params.index(fv[j]), grid[j][i])
        for i in range(1, nu):
            self._add_chain([grid[j][i] for j in range(nv + 1)])
        for j in range(1, nv):
            self._add_chain(grid[j])

        children = []
        for j in range(nv):
            for i in range(nu):
                bottom = (
                    (s0[0], *[at(s0, fu[k])[1] for k in (i, i + 1)])
                    if j == 0
                    else (horiz[j], fu[i], fu[i + 1])
                )
                right = (
                    (s1[0], *[at(s1, fv[k])[1] for k in (j, j + 1)])
                    if i + 1 == nu
                    else (vert[i + 1], fv[j], fv[j + 1])
                )
                top = (
                    (s2[0], *[at(s2, 1 - fu[k])[1] for k in (i + 1, i)])
                    if j + 1 == nv
                    else (horiz[j + 1], fu[i + 1], fu[i])
                )
                left = (
                    (s3[0], *[at(s3, 1 - fv[k])[1] for k in (j + 1, j)])
                    if i == 0
                    else (vert[i], fv[j + 1], fv[j])
                )
                corners = [
                    grid[j][i],
                    grid[j][i + 1],
                    grid[j + 1][i + 1],
                    grid[j + 1][i],
                ]
                children.append(
                    _Cell(
                        corners,
                        [bottom, right, top, left],
                        c.origin,
                        c.nu * nu,
                        c.nv * nv,
                    )
                )
        return children

    # -- output --------------------------------------------------------------

    def _build_mesh(self):
        polygons = []
        for c in self.cells:
            poly = []
            for corner, (lid, t0, t1) in zip(c.corners, c.sides, strict=True):
                poly.append(corner)
                poly.extend(self.lines[lid].inside(t0, t1))
            polygons.append(poly)
        nodes = np.asarray(self.nodes, dtype=float)
        ptr = np.concatenate(([0], np.cumsum([len(q) for q in polygons])))
        flat = np.concatenate([np.asarray(q, dtype=np.int64) for q in polygons])
        nxt = np.arange(flat.size) + 1
        nxt[ptr[1:] - 1] = ptr[:-1]
        a, b = nodes[flat], nodes[flat[nxt]]
        cross = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
        area = np.add.reduceat(cross, ptr[:-1])
        bad = np.flatnonzero(area <= 0.0)
        if bad.size:
            where = ", ".join(
                f"({x:.6g}, {y:.6g})"
                for x, y in nodes[[polygons[i][0] for i in bad[:5]]]
            )
            raise MeshError(
                f"refinement folded {bad.size} cells (non-positive area), e.g. at "
                f"{where}; increase the decay length or refine less near walls"
            )
        boundary = {name: [] for name in self.boundary_names}
        for line in self.lines:
            if line.boundary is not None:
                n = line.nodes
                boundary[line.boundary].extend(zip(n[:-1], n[1:], strict=True))
        return Mesh.from_cells(
            nodes,
            polygons,
            {
                k: np.asarray(v, dtype=np.int64).reshape(-1, 2)
                for k, v in boundary.items()
            },
        )


def _wall_curves(mesh, walls, corner_angle, line_of_key):
    """Catmull-Rom control points of every wall face, oriented low -> high node.

    All ``walls`` together form the wall chains, so a curve continues across
    the border of two wall sets (e.g. a slot cut out of an airfoil).
    """
    cos_max = np.cos(np.radians(corner_angle))
    p = mesh.nodes
    edges = []
    for name in walls:
        if name not in mesh.boundary_faces:
            raise MeshError(f"refinement wall '{name}' is not a boundary set")
        edges += mesh.face_nodes[mesh.boundary_faces[name]].tolist()
    nbrs = {}
    for a, b in edges:
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)

    def beyond(a, b):
        """Wall node before a when walking b -> a, or None at a corner."""
        others = [n for n in nbrs[a] if n != b]
        if len(others) != 1:
            return None
        c = others[0]
        t1, t2 = p[a] - p[c], p[b] - p[a]
        cos = t1 @ t2 / (np.hypot(*t1) * np.hypot(*t2))
        return c if cos >= cos_max else None

    curves = {}
    for a, b in edges:
        lo, hi = (a, b) if a < b else (b, a)
        p1, p2 = p[lo], p[hi]
        c0, c3 = beyond(lo, hi), beyond(hi, lo)
        p0 = p[c0] if c0 is not None else 2.0 * p1 - p2
        p3 = p[c3] if c3 is not None else 2.0 * p2 - p1
        curves[line_of_key[(lo, hi)]] = (p0, p1, p2, p3)
    return curves


def _catmull_rom(ctrl, s):
    """Centripetal Catmull-Rom point between ctrl[1] (s=0) and ctrl[2] (s=1)."""
    p0, p1, p2, p3 = ctrl
    t0 = 0.0
    t1 = t0 + np.sqrt(np.hypot(*(p1 - p0)))
    t2 = t1 + np.sqrt(np.hypot(*(p2 - p1)))
    t3 = t2 + np.sqrt(np.hypot(*(p3 - p2)))
    t = t1 + s * (t2 - t1)
    a1 = ((t1 - t) * p0 + (t - t0) * p1) / (t1 - t0)
    a2 = ((t2 - t) * p1 + (t - t1) * p2) / (t2 - t1)
    a3 = ((t3 - t) * p2 + (t - t2) * p3) / (t3 - t2)
    b1 = ((t2 - t) * a1 + (t - t0) * a2) / (t2 - t0)
    b2 = ((t3 - t) * a2 + (t - t1) * a3) / (t3 - t1)
    c = ((t2 - t) * b1 + (t - t1) * b2) / (t2 - t1)
    return float(c[0]), float(c[1])


def refine_cells(mesh, cells, nu=2, nv=2, max_hanging=1):
    """One-shot refinement of an unrefined mesh.

    Returns ``(new_mesh, source)`` with ``source`` the original cell of every
    new cell. See :meth:`Refinement.split`.
    """
    ref = Refinement.from_mesh(mesh)
    source = ref.split(cells, nu, nv, max_hanging)
    return ref.mesh, source


def inject(values, source):
    """Transfer cell values to the refined mesh (children copy the parent)."""
    return np.asarray(values)[np.asarray(source)]

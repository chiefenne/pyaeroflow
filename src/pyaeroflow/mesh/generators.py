"""Mesh generators for test cases (NumPy only, requirement D42).

Stage 0 provides rectangular domains with quadrilateral, triangular, polygonal
and mixed cells, optional wall-normal (y) clustering and optional random
perturbation of interior nodes.
"""

import numpy as np

from pyaeroflow.mesh.mesh import Mesh

CELL_TYPES = ("quad", "triangle", "polygon", "mixed")
DEFAULT_NAMES = ("bottom", "right", "top", "left")


def rectangle(
    nx,
    ny,
    *,
    x=(0.0, 1.0),
    y=(0.0, 1.0),
    cell_type="quad",
    diagonal="alternating",
    quad_layers=None,
    cluster_x="none",
    cluster_y="none",
    beta=2.0,
    perturb=0.0,
    seed=0,
    names=DEFAULT_NAMES,
):
    """Generate a mesh of a rectangle.

    Parameters
    ----------
    nx, ny : int
        Number of grid intervals in x and y. The cell count depends on
        ``cell_type``: quad ``nx*ny``; triangle ``2*nx*ny``; polygon
        ``(nx+1)*(ny+1)``.
    x, y : tuple of float
        Domain extent ``(min, max)``.
    cell_type : {"quad", "triangle", "polygon", "mixed"}
        - ``"quad"``: structured quadrilaterals.
        - ``"triangle"``: every quadrilateral split along one diagonal.
        - ``"polygon"``: centroid-dual of the triangle mesh, one polygon per
          grid node (mostly hexagons; boundary cells include the node).
        - ``"mixed"``: ``quad_layers`` rows of quadrilaterals at the bottom
          and top walls, triangles in between.
    diagonal : {"alternating", "random"}
        Diagonal pattern for triangles (also used by "polygon" and "mixed").
    quad_layers : int, optional
        Rows of quadrilaterals at each wall for "mixed"; default ``ny // 4``
        (at least 1).
    cluster_x : {"none", "left", "right", "both"}
        Clustering of grid lines towards the given side(s), tanh distribution.
    cluster_y : {"none", "bottom", "top", "both"}
        Clustering of grid lines towards the given wall(s), tanh distribution.
    beta : float
        Clustering strength (> 0); larger values cluster more strongly.
    perturb : float
        Random displacement of interior grid nodes, as a fraction of the local
        grid spacing, in [0, 0.8]. 0 gives a regular mesh.
    seed : int
        Seed of the random generator (perturbation, random diagonals).
    names : tuple of str
        Names of the boundary faces at bottom, right, top and left.

    Returns
    -------
    Mesh
    """
    if cell_type not in CELL_TYPES:
        raise ValueError(f"cell_type must be one of {CELL_TYPES}")
    if not 0.0 <= perturb <= 0.8:
        raise ValueError("perturb must be in [0, 0.8]")
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    rng = np.random.default_rng(seed)

    cx = {"none": "none", "left": "bottom", "right": "top", "both": "both"}
    if cluster_x not in cx:
        raise ValueError("cluster_x must be 'none', 'left', 'right' or 'both'")
    xs = x[0] + (x[1] - x[0]) * _cluster(
        np.linspace(0.0, 1.0, nx + 1), cx[cluster_x], beta
    )
    ys = y[0] + (y[1] - y[0]) * _cluster(np.linspace(0.0, 1.0, ny + 1), cluster_y, beta)
    nodes = _grid_nodes(xs, ys, perturb, rng)
    edges = _rectangle_boundary_edges(nx, ny)
    boundary = dict(zip(names, edges, strict=True))

    if cell_type == "quad":
        return Mesh.from_cells(nodes, _quads(nx, ny), boundary)

    split = _diagonals(nx, ny, diagonal, rng)
    if cell_type == "mixed":
        layers = max(1, ny // 4) if quad_layers is None else quad_layers
        rows = np.arange(ny)
        quad_rows = (rows < layers) | (rows >= ny - layers)
        cells = _mixed(nx, ny, split, quad_rows)
        return Mesh.from_cells(nodes, cells, boundary)

    triangles = _triangles(nx, ny, split)
    if cell_type == "triangle":
        return Mesh.from_cells(nodes, triangles, boundary)
    return _centroid_dual(nodes, triangles, boundary)


def structured(xs, ys, names=("bottom", "right", "top", "left")):
    """Quadrilateral mesh from grid-line coordinates ``xs`` (nx+1) and ``ys`` (ny+1).

    Boundary names in the order bottom, right, top, left.
    """
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    nx, ny = xs.size - 1, ys.size - 1
    nodes = _grid_nodes(xs, ys, 0.0, np.random.default_rng(0))
    boundary = dict(zip(names, _rectangle_boundary_edges(nx, ny), strict=True))
    return Mesh.from_cells(nodes, _quads(nx, ny), boundary)


def geometric_growth(length, first, n):
    """Growth ratio r so that n cells with first spacing ``first`` span ``length``."""
    if n * first >= length:
        return 1.0
    lo, hi = 1.0, 2.0
    while first * (hi**n - 1.0) / (hi - 1.0) < length:
        hi *= 2.0
    for _ in range(100):
        r = 0.5 * (lo + hi)
        if first * (r**n - 1.0) / (r - 1.0) < length:
            lo = r
        else:
            hi = r
    return 0.5 * (lo + hi)


def _cluster(s, mode, beta):
    """Map uniform s in [0, 1] to a clustered distribution (tanh)."""
    if mode == "none":
        return s
    if beta <= 0.0:
        raise ValueError("beta must be > 0")
    t = np.tanh(beta)
    if mode == "both":
        return 0.5 * (1.0 + np.tanh(beta * (2.0 * s - 1.0)) / t)
    if mode == "bottom":
        return 1.0 + np.tanh(beta * (s - 1.0)) / t
    if mode == "top":
        return np.tanh(beta * s) / t
    raise ValueError("cluster_y must be 'none', 'bottom', 'top' or 'both'")


def _grid_nodes(xs, ys, perturb, rng):
    """Tensor-product nodes, index j*(nx+1)+i, with optional perturbation."""
    gx, gy = np.meshgrid(xs, ys)
    if perturb > 0.0:
        # Local spacing: smaller of the two adjacent intervals.
        hx = np.minimum(np.diff(xs)[:-1], np.diff(xs)[1:])
        hy = np.minimum(np.diff(ys)[:-1], np.diff(ys)[1:])
        shape = (ys.size - 2, xs.size - 2)
        gx[1:-1, 1:-1] += perturb * hx[None, :] * rng.uniform(-0.5, 0.5, shape)
        gy[1:-1, 1:-1] += perturb * hy[:, None] * rng.uniform(-0.5, 0.5, shape)
    return np.column_stack((gx.ravel(), gy.ravel()))


def _node(i, j, nx):
    """Node index of grid point (i, j)."""
    return j * (nx + 1) + i


def _rectangle_boundary_edges(nx, ny):
    """Node pairs on the bottom, right, top and left sides."""
    i = np.arange(nx)
    j = np.arange(ny)
    bottom = np.column_stack((_node(i, 0, nx), _node(i + 1, 0, nx)))
    right = np.column_stack((_node(nx, j, nx), _node(nx, j + 1, nx)))
    top = np.column_stack((_node(i, ny, nx), _node(i + 1, ny, nx)))
    left = np.column_stack((_node(0, j, nx), _node(0, j + 1, nx)))
    return bottom, right, top, left


def _quads(nx, ny):
    """Counter-clockwise quadrilaterals, shape (nx*ny, 4)."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    i, j = i.ravel(), j.ravel()
    return np.column_stack(
        (
            _node(i, j, nx),
            _node(i + 1, j, nx),
            _node(i + 1, j + 1, nx),
            _node(i, j + 1, nx),
        )
    )


def _diagonals(nx, ny, pattern, rng):
    """Boolean (ny, nx): True splits a quad along (i,j)-(i+1,j+1)."""
    if pattern == "alternating":
        i, j = np.meshgrid(np.arange(nx), np.arange(ny))
        return (i + j) % 2 == 0
    if pattern == "random":
        return rng.random((ny, nx)) < 0.5
    raise ValueError("diagonal must be 'alternating' or 'random'")


def _split_quad(q, along_main):
    """Split quads q (n, 4) into two triangles each."""
    n0, n1, n2, n3 = q.T
    t_main = np.column_stack((n0, n1, n2)), np.column_stack((n0, n2, n3))
    t_anti = np.column_stack((n0, n1, n3)), np.column_stack((n1, n2, n3))
    first = np.where(along_main[:, None], t_main[0], t_anti[0])
    second = np.where(along_main[:, None], t_main[1], t_anti[1])
    return np.vstack((first, second))


def _triangles(nx, ny, split):
    """Triangles from splitting every quad, shape (2*nx*ny, 3)."""
    return _split_quad(_quads(nx, ny), split.ravel())


def _mixed(nx, ny, split, quad_rows):
    """Quads in rows flagged by ``quad_rows``, triangles elsewhere."""
    quads = _quads(nx, ny).reshape(ny, nx, 4)
    keep = quads[quad_rows].reshape(-1, 4)
    tris = _split_quad(quads[~quad_rows].reshape(-1, 4), split[~quad_rows].ravel())
    return [*keep, *tris]


def _centroid_dual(nodes, triangles, boundary):
    """Polygon mesh: one cell per node, built from incident triangle centroids.

    Interior nodes give a polygon through the centroids of the surrounding
    triangles. Boundary nodes additionally include the midpoints of their two
    boundary edges and the node itself, so the polygon mesh covers exactly the
    same domain.
    """
    n_nodes = nodes.shape[0]
    n_tri = triangles.shape[0]
    centroids = nodes[triangles].mean(axis=1)

    # Boundary edges: midpoints become new polygon nodes.
    b_edges = np.vstack(list(boundary.values()))
    b_names = np.repeat(list(boundary), [len(e) for e in boundary.values()])
    midpoints = 0.5 * (nodes[b_edges[:, 0]] + nodes[b_edges[:, 1]])
    b_nodes = np.unique(b_edges)

    # New node numbering: centroids, midpoints, boundary grid nodes.
    mid_id = n_tri + np.arange(b_edges.shape[0])
    node_id = np.full(n_nodes, -1, dtype=np.int64)
    node_id[b_nodes] = n_tri + b_edges.shape[0] + np.arange(b_nodes.size)
    new_nodes = np.vstack((centroids, midpoints, nodes[b_nodes]))

    # Incident triangles and boundary edges per grid node.
    tri_of = [[] for _ in range(n_nodes)]
    for t, tri in enumerate(triangles):
        for v in tri:
            tri_of[v].append(t)
    edge_of = [[] for _ in range(n_nodes)]
    for e, (p, q) in enumerate(b_edges):
        edge_of[p].append(e)
        edge_of[q].append(e)

    cells = []
    for v in range(n_nodes):
        ring = np.array(tri_of[v])
        if node_id[v] < 0:  # interior node: sort centroids by angle
            d = centroids[ring] - nodes[v]
            order = np.argsort(np.arctan2(d[:, 1], d[:, 0]))
            cells.append(ring[order])
            continue
        # Boundary node: sort centroids and edge midpoints by angle measured
        # from the inward direction, then close the polygon through the node.
        ids = np.concatenate((ring, mid_id[edge_of[v]]))
        d = new_nodes[ids] - nodes[v]
        inward = (centroids[ring] - nodes[v]).sum(axis=0)
        cross = inward[0] * d[:, 1] - inward[1] * d[:, 0]
        dot = inward @ d.T
        order = np.argsort(np.arctan2(cross, dot))
        cells.append(np.concatenate(([node_id[v]], ids[order])))

    # Each boundary edge (p, q) becomes the two faces (p, m) and (m, q).
    new_boundary = {}
    for name in boundary:
        e = np.flatnonzero(b_names == name)
        p, q = node_id[b_edges[e, 0]], node_id[b_edges[e, 1]]
        m = mid_id[e]
        new_boundary[name] = np.vstack(
            (np.column_stack((p, m)), np.column_stack((m, q)))
        )
    return Mesh.from_cells(new_nodes, cells, new_boundary)

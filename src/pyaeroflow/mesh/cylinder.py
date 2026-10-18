"""Structured meshes around a circular cylinder (NumPy only, D42).

- :func:`annulus`: O-grid between a cylinder and a circular farfield
  (verification case V4, inviscid cylinder).
- :func:`cylinder_in_channel`: block-structured quadrilateral mesh of the
  DFG benchmark geometry (Schaefer & Turek 1996): channel [0, 2.2] x [0, 0.41]
  with a cylinder of radius 0.05 at (0.2, 0.2). An O-grid block fills a square
  around the cylinder; eight rectangular blocks fill the rest.
- :func:`split_boundary`: rename parts of a boundary-face set by a predicate
  on the face centres (e.g. the plate of a flat-plate domain).

Blocks are built as global node arrays; coincident nodes on block interfaces
are merged by rounding the coordinates.
"""

import numpy as np

from pyaeroflow.mesh.generators import _split_quad
from pyaeroflow.mesh.mesh import Mesh


def geometric_spacing(n, ratio):
    """Return n+1 points in [0, 1] with interval growth ``ratio`` (1 = uniform)."""
    if abs(ratio - 1.0) < 1e-12:
        return np.linspace(0.0, 1.0, n + 1)
    h = np.cumprod(np.full(n, ratio)) / ratio
    s = np.concatenate(([0.0], np.cumsum(h)))
    return s / s[-1]


def annulus(
    n_theta,
    n_radial,
    *,
    radius=0.5,
    outer_radius=20.0,
    centre=(0.0, 0.0),
    growth=1.1,
    cell_type="quad",
    names=("cylinder", "farfield"),
):
    """O-grid between a cylinder and a circular outer boundary.

    Parameters
    ----------
    n_theta : int
        Cells around the circumference.
    n_radial : int
        Cells in the radial direction.
    radius, outer_radius : float
    centre : tuple of float
    growth : float
        Geometric growth of the radial spacing from the cylinder outward.
    cell_type : {"quad", "triangle"}
    names : tuple of str
        Boundary names of the cylinder and the outer boundary.

    Returns
    -------
    Mesh
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    s = geometric_spacing(n_radial, growth)
    r = radius + (outer_radius - radius) * s
    rr, tt = np.meshgrid(r, theta, indexing="ij")  # (n_radial+1, n_theta)
    nodes = np.column_stack(
        (centre[0] + (rr * np.cos(tt)).ravel(), centre[1] + (rr * np.sin(tt)).ravel())
    )

    def node(j, k):
        return j * n_theta + (k % n_theta)

    j, k = np.meshgrid(np.arange(n_radial), np.arange(n_theta), indexing="ij")
    j, k = j.ravel(), k.ravel()
    # Counter-clockwise: inner-k, inner-(k+1), outer-(k+1), outer-k.
    quads = np.column_stack(
        (node(j, k), node(j, k + 1), node(j + 1, k + 1), node(j + 1, k))
    )
    k = np.arange(n_theta)
    boundary = {
        names[0]: np.column_stack((node(0, k + 1), node(0, k))),
        names[1]: np.column_stack((node(n_radial, k), node(n_radial, k + 1))),
    }
    cells = _as_cells(quads, cell_type)
    return Mesh.from_cells(nodes, cells, boundary)


def cylinder_in_channel(
    m=16,
    n_radial=12,
    *,
    length=2.2,
    height=0.41,
    centre=(0.2, 0.2),
    radius=0.05,
    half_side=0.15,
    growth=1.15,
    cell_type="quad",
    names=("inlet", "outlet", "wall", "cylinder"),
):
    """Block-structured mesh of the DFG cylinder-in-channel geometry.

    Parameters
    ----------
    m : int
        Cells along each side of the square around the cylinder; sets the
        resolution everywhere (other block sizes follow from the spacing
        ``2 half_side / m``).
    n_radial : int
        Cells between the cylinder and the square.
    length, height, centre, radius, half_side : float
        Geometry; the square [cx - a, cx + a] x [cy - a, cy + a] with
        a = ``half_side`` must lie inside the channel.
    growth : float
        Geometric growth of the radial spacing from the cylinder outward.
    cell_type : {"quad", "triangle"}
    names : tuple of str
        Boundary names of inlet (x = 0), outlet (x = length), the channel
        walls, and the cylinder.

    Returns
    -------
    Mesh
    """
    cx, cy, a = centre[0], centre[1], half_side
    x0, x1, y0, y1 = cx - a, cx + a, cy - a, cy + a
    if not (0.0 < x0 and x1 < length and 0.0 < y0 and y1 < height):
        raise ValueError("the square around the cylinder must lie inside the channel")
    h = 2.0 * a / m
    # Grid lines of the outer blocks, spacing ~h, stretched away from the square.
    xs_left = x0 - _stretched(x0, h, 1.1)[::-1]
    xs_right = x1 + _stretched(length - x1, h, 1.1)
    ys_bottom = y0 - _stretched(y0, h, 1.1)[::-1]
    ys_top = y1 + _stretched(height - y1, h, 1.1)
    xs_mid = np.linspace(x0, x1, m + 1)
    ys_mid = np.linspace(y0, y1, m + 1)

    quads, nodes = [], []

    def add_block(bx, by):
        gx, gy = np.meshgrid(bx, by)
        pts = np.column_stack((gx.ravel(), gy.ravel()))
        off = sum(len(n) for n in nodes)
        nx, ny = bx.size - 1, by.size - 1
        i, j = np.meshgrid(np.arange(nx), np.arange(ny))
        i, j = i.ravel(), j.ravel()
        idx = lambda i, j: off + j * (nx + 1) + i  # noqa: E731
        quads.append(
            np.column_stack(
                (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1))
            )
        )
        nodes.append(pts)

    # Eight rectangular blocks around the square (3 x 3 layout minus centre).
    for bx in (xs_left, xs_mid, xs_right):
        for by in (ys_bottom, ys_mid, ys_top):
            if bx is xs_mid and by is ys_mid:
                continue
            add_block(bx, by)

    # O-grid block: square boundary points (counter-clockwise from the
    # right-bottom corner) and the matching cylinder points.
    side = np.linspace(-a, a, m + 1)[:-1]
    square = np.vstack(
        (
            np.column_stack((np.full(m, a), side)),  # right side, upward
            np.column_stack((-side, np.full(m, a))),  # top side, leftward
            np.column_stack((np.full(m, -a), -side)),  # left side, downward
            np.column_stack((side, np.full(m, -a))),  # bottom side, rightward
        )
    )
    theta = np.arctan2(square[:, 1], square[:, 0])
    circle = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    s = geometric_spacing(n_radial, growth)
    ring = circle[None] + s[:, None, None] * (square - circle)[None]  # (n_r+1, 4m, 2)
    ring = ring + np.array(centre)
    off = sum(len(n) for n in nodes)
    nodes.append(ring.reshape(-1, 2))
    n_theta = 4 * m

    def onode(j, k):
        return off + j * n_theta + (k % n_theta)

    j, k = np.meshgrid(np.arange(n_radial), np.arange(n_theta), indexing="ij")
    j, k = j.ravel(), k.ravel()
    quads.append(
        np.column_stack(
            (onode(j, k), onode(j, k + 1), onode(j + 1, k + 1), onode(j + 1, k))
        )
    )
    k = np.arange(n_theta)
    cyl_edges = np.column_stack((onode(0, k + 1), onode(0, k)))

    all_nodes = np.vstack(nodes)
    all_quads = np.vstack(quads)
    all_nodes, remap = _merge_nodes(all_nodes)
    all_quads = remap[all_quads]
    cyl_edges = remap[cyl_edges]

    # Channel boundaries from the merged node coordinates.
    boundary = {
        names[0]: _boundary_edges(
            all_nodes, all_nodes[:, 0] < 1e-12, axis=1, reverse=True
        ),
        names[1]: _boundary_edges(all_nodes, all_nodes[:, 0] > length - 1e-12, axis=1),
        names[2]: np.vstack(
            (
                _boundary_edges(all_nodes, all_nodes[:, 1] < 1e-12, axis=0),
                _boundary_edges(
                    all_nodes, all_nodes[:, 1] > height - 1e-12, axis=0, reverse=True
                ),
            )
        ),
        names[3]: cyl_edges,
    }
    return Mesh.from_cells(all_nodes, _as_cells(all_quads, cell_type), boundary)


def split_boundary(mesh, name, parts):
    """Split the boundary-face set ``name`` by predicates on the face centres.

    Parameters
    ----------
    mesh : Mesh
    name : str
    parts : dict of str to callable
        ``{new_name: predicate(x, y) -> bool array}``; every face of the set
        must match exactly one predicate.

    Returns
    -------
    Mesh
    """
    faces = mesh.boundary_faces[name]
    centres = mesh.face_centre[faces]
    boundary = {
        k: mesh.face_nodes[v] for k, v in mesh.boundary_faces.items() if k != name
    }
    assigned = np.zeros(faces.size, dtype=int)
    for new, pred in parts.items():
        mask = np.asarray(pred(centres[:, 0], centres[:, 1]), dtype=bool)
        boundary[new] = mesh.face_nodes[faces[mask]]
        assigned += mask
    if np.any(assigned != 1):
        raise ValueError(f"split of '{name}': every face must match exactly one part")
    cells = [mesh.cell(i) for i in range(mesh.n_cells)]
    return Mesh.from_cells(mesh.nodes, cells, boundary)


def _as_cells(quads, cell_type):
    if cell_type == "quad":
        return quads
    if cell_type == "triangle":
        along_main = (np.arange(quads.shape[0]) % 2) == 0
        return _split_quad(quads, along_main)
    raise ValueError("cell_type must be 'quad' or 'triangle'")


def _stretched(total, h0, growth):
    """Points from 0 to ``total`` starting with spacing h0, growing geometrically."""
    pts, x, h = [0.0], 0.0, h0
    while x + h < total - 0.5 * h0:
        x += h
        pts.append(x)
        h *= growth
    pts.append(total)
    return np.array(pts)


def _merge_nodes(nodes, decimals=10):
    key = np.round(nodes, decimals)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return nodes[first], inverse.ravel()


def _boundary_edges(nodes, mask, axis, reverse=False):
    """Consecutive node pairs along a straight boundary, sorted by ``axis``."""
    idx = np.flatnonzero(mask)
    idx = idx[np.argsort(nodes[idx, axis])]
    if reverse:
        idx = idx[::-1]
    return np.column_stack((idx[:-1], idx[1:]))

"""Weighted least-squares cell gradients (requirement D22).

The stencil of a cell consists of all cells sharing at least one node with it
(vertex neighbours) and all boundary faces sharing a node with it; boundary
faces enter with their face-centre values. The gradient of cell i minimises

    sum_k ( w_k * (phi_k - phi_i - grad(phi)_i . d_k) )^2,   w_k = |d_k|^(-q),

with d_k the vector from the cell centroid to the stencil point. This is the
weighting convention of Syrakos et al. (2017): q = 0 unweighted, q = 1 inverse
distance (as SU2's weighted least squares), q = 3/2 their recommendation for
second-order accuracy at boundary cells.

Boundary faces can be excluded per variable (``include`` mask): a face whose
value of a variable is not known (e.g. the pressure at a wall) then takes no
part in that variable's gradient. Periodic boundaries enter as cell-cell
entries with the neighbour centroid translated to the near side.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


def lsq_stencil(mesh, boundary_faces, weight_exponent=1.0, include=None, periodic=()):
    """Precompute stencil and coefficients of the least-squares gradient.

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    boundary_faces : ndarray
        Mesh indices of the boundary faces, in the order in which their values
        are passed to :func:`lsq_gradient` (periodic faces excluded).
    weight_exponent : float
        Exponent q of the weights |d|^(-q).
    include : ndarray of bool, shape (n_bfaces, n_var), optional
        Which boundary faces take part in the gradient of which variable.
        Default: all faces for all variables (one coefficient set).
    periodic : sequence of (node_a, node_b, translation)
        Periodic node pairs ``node_a`` ↔ ``node_b`` (integer arrays) with
        ``x_b = x_a + translation``; cells touching the ``a`` nodes see cells
        touching the ``b`` nodes at their position minus the translation and
        vice versa.

    Returns
    -------
    cell : ndarray, shape (n_entries,)
        Cell of each stencil entry.
    src : ndarray, shape (n_entries,)
        Source index: < n_cells for cells, n_cells + j for boundary face j.
    coeff : ndarray, shape (n_entries, n_sets, 2)
        Coefficients c such that grad_i = sum c * (phi_src - phi_i); one set
        per variable if ``include`` is given, else a single set.
    """
    n_cells, n_nodes = mesh.n_cells, mesh.n_nodes
    rows = np.repeat(np.arange(n_cells), mesh.cell_n_nodes)
    incidence = sp.csr_matrix(
        (np.ones(rows.size), (rows, mesh.cell_nodes)), shape=(n_cells, n_nodes)
    )
    nb = len(boundary_faces)
    b_rows = np.repeat(np.arange(nb), 2)
    b_cols = mesh.face_nodes[boundary_faces].ravel()
    b_inc = sp.csr_matrix((np.ones(2 * nb), (b_rows, b_cols)), shape=(nb, n_nodes))
    cell_cell = (incidence @ incidence.T).tocoo()
    keep = cell_cell.row != cell_cell.col
    cell_bnd = (incidence @ b_inc.T).tocoo()

    cells = [cell_cell.row[keep], cell_bnd.row]
    srcs = [cell_cell.col[keep], n_cells + cell_bnd.col]
    shifts = [np.zeros((keep.sum(), 2)), np.zeros((cell_bnd.nnz, 2))]
    for node_a, node_b, translation in periodic:
        # Cells touching the a-nodes and cells touching the matching b-nodes.
        sel_a = incidence[:, node_a].tocoo()
        sel_b = incidence[:, node_b].tocoo()
        # For every a-node (column k) pair its cells with the cells of b-node k.
        pair = sp.csr_matrix(
            (np.ones(sel_a.nnz), (sel_a.row, sel_a.col)), shape=(n_cells, len(node_a))
        ) @ sp.csr_matrix(
            (np.ones(sel_b.nnz), (sel_b.col, sel_b.row)), shape=(len(node_b), n_cells)
        )
        pair = pair.tocoo()
        t = np.asarray(translation, dtype=float)
        cells += [pair.row, pair.col]
        srcs += [pair.col, pair.row]
        shifts += [np.tile(-t, (pair.nnz, 1)), np.tile(t, (pair.nnz, 1))]
    cell = np.concatenate(cells)
    src = np.concatenate(srcs)
    shift = np.vstack(shifts)
    key = np.stack((cell, src, shift[:, 0], shift[:, 1]), axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    first = np.sort(first)
    cell, src, shift = cell[first], src[first], shift[first]

    points = np.vstack((mesh.cell_centroid, mesh.face_centre[boundary_faces]))
    d = points[src] + shift - mesh.cell_centroid[cell]
    w2 = np.hypot(d[:, 0], d[:, 1]) ** (-2.0 * weight_exponent)
    is_bnd = src >= n_cells
    if include is None:
        masks = [np.ones(cell.size, dtype=bool)]
    else:
        include = np.asarray(include, dtype=bool)
        masks = []
        for v in range(include.shape[1]):
            m = np.ones(cell.size, dtype=bool)
            m[is_bnd] = include[src[is_bnd] - n_cells, v]
            masks.append(m)
    coeff = np.zeros((cell.size, len(masks), 2))
    for k, m in enumerate(masks):
        wm = np.where(m, w2, 0.0)
        m11 = np.bincount(cell, wm * d[:, 0] ** 2, n_cells)
        m12 = np.bincount(cell, wm * d[:, 0] * d[:, 1], n_cells)
        m22 = np.bincount(cell, wm * d[:, 1] ** 2, n_cells)
        det = m11 * m22 - m12**2
        if np.any(det <= 1e-14 * (m11 * m22)):
            raise ValueError("degenerate least-squares stencil (collinear points)")
        inv11, inv12, inv22 = m22 / det, -m12 / det, m11 / det
        coeff[:, k, 0] = wm * (inv11[cell] * d[:, 0] + inv12[cell] * d[:, 1])
        coeff[:, k, 1] = wm * (inv12[cell] * d[:, 0] + inv22[cell] * d[:, 1])
    return cell, src, coeff


@jax.jit
def lsq_gradient(geom, u, u_b):
    """Least-squares gradients of cell values.

    Parameters
    ----------
    geom : FVGeometry
        Geometry with the precomputed stencil.
    u : jax array, shape (n_cells, n_var)
        Cell values.
    u_b : jax array, shape (n_bfaces, n_var)
        Boundary-face values, in boundary-face order.

    Returns
    -------
    jax array, shape (n_cells, n_var, 2)
    """
    values = jnp.concatenate((u, u_b))[geom.lsq_src]
    diff = values - u[geom.lsq_cell]
    coeff = geom.lsq_coeff  # (n_entries, 1 or n_var, 2)
    contrib = diff[:, :, None] * coeff
    return jax.ops.segment_sum(contrib, geom.lsq_cell, num_segments=u.shape[0])

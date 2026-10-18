"""Approximate (first-order) Jacobian by automatic differentiation (D25).

The Jacobian is assembled from compact face fluxes that depend only on the
two cells next to a face (interior faces) or on the owner cell (boundary
faces). JAX differentiates each face flux; the blocks are assembled into a
SciPy sparse matrix with block size n_var, unknowns ordered cell by cell
(index ``i * n_var + v``), matching ``u.ravel()``.
"""

from functools import partial

import jax
import numpy as np
import scipy.sparse as sp


@partial(jax.jit, static_argnums=(1, 2))
def _face_blocks(geom, interior_flux, boundary_flux, u, params, b_args):
    """Flux derivatives per face, scaled by the face length."""
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    d_inner = jax.vmap(
        jax.jacfwd(interior_flux, argnums=(1, 2)), in_axes=(None, 0, 0, 0, 0)
    )
    d_o, d_n = d_inner(params, u[o], u[n], geom.f_normal, geom.f_delta)
    n_extra = len(b_args)
    d_bnd = jax.vmap(
        jax.jacfwd(boundary_flux, argnums=1), in_axes=(None, 0, *([0] * n_extra), 0, 0)
    )
    d_b = d_bnd(params, u[b], *b_args, geom.b_normal, geom.b_delta)
    fl = geom.f_length[:, None, None]
    return d_o * fl, d_n * fl, d_b * geom.b_length[:, None, None]


class JacobianPattern:
    """Fixed sparsity of the first-order Jacobian of one geometry.

    The face blocks always land on the same matrix entries, so the CSR
    structure and the map from face-block entries to CSR positions are built
    once (``np.unique`` on the row/column keys) and every assembly is one
    ``np.bincount``; rebuilding a COO matrix per iteration cost about as much
    as a residual evaluation on a 100k-cell mesh.
    """

    def __init__(self, geom, n_var):
        o, n, b = (np.asarray(a) for a in (geom.f_owner, geom.f_neigh, geom.b_owner))
        n_cells = int(np.asarray(geom.area).shape[0])
        block_rows = np.concatenate((o, o, n, n, b))
        block_cols = np.concatenate((o, n, o, n, b))
        k = np.arange(n_var)
        rows = (block_rows[:, None, None] * n_var + k[None, :, None]).repeat(n_var, 2)
        cols = (block_cols[:, None, None] * n_var + k[None, None, :]).repeat(n_var, 1)
        size = n_cells * n_var
        key = rows.ravel().astype(np.int64) * size + cols.ravel()
        unique, inverse = np.unique(key, return_inverse=True)
        self.n_var = n_var
        self.shape = (size, size)
        self.position = inverse.astype(np.int32)  # face-block entry -> CSR entry
        self.indices = (unique % size).astype(np.int32)
        self.indptr = np.searchsorted(unique // size, np.arange(size + 1)).astype(
            np.int32
        )

    def assemble(self, blocks):
        """CSR matrix from the face blocks ordered (d_o, d_n, -d_o, -d_n, d_b)."""
        data = np.bincount(
            self.position, weights=blocks.ravel(), minlength=self.indices.size
        )
        return sp.csr_matrix((data, self.indices, self.indptr), shape=self.shape)


def first_order_jacobian(
    geom,
    interior_flux,
    boundary_flux,
    u,
    params=None,
    b_args=(),
    block_mask=None,
    pattern=None,
):
    """Assemble dR1/du of the compact first-order residual.

    Parameters
    ----------
    geom : FVGeometry
    interior_flux : callable
        ``f(params, u_o, u_n, normal, delta) -> (n_var,)``: flux density
        through an interior face (owner to neighbour); ``delta`` = owner-to-
        neighbour centroid vector. Must be hashable (a jit static argument),
        e.g. a module-level function or a ``functools.partial`` of one with
        hashable arguments.
    boundary_flux : callable
        ``f(params, u_o, *b_args_face, normal, delta) -> (n_var,)``: outward
        flux density through a boundary face; ``delta`` = owner centroid to
        face centre.
    u : array, shape (n_cells, n_var)
    params : pytree, optional
        Traced parameters passed unchanged to every flux call (e.g. a
        per-iteration constant); not differentiated.
    b_args : tuple of arrays
        Per-boundary-face arrays (first axis n_bfaces) passed to
        ``boundary_flux``; held fixed in the derivative.
    block_mask : array (n_var, n_var), optional
        Multiplies every flux block, e.g. to drop cross-coupling entries
        (segregated turbulence coupling, D34).
    pattern : JacobianPattern, optional
        Sparsity built once for this geometry; built on the fly otherwise.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_cells * n_var, n_cells * n_var)
    """
    n_var = u.shape[1]
    d_o, d_n, d_b = (
        np.asarray(a)
        for a in _face_blocks(geom, interior_flux, boundary_flux, u, params, b_args)
    )
    blocks = np.concatenate((d_o, d_n, -d_o, -d_n, d_b))
    if block_mask is not None:
        blocks = blocks * np.asarray(block_mask, dtype=float)[None]
    if pattern is None:
        pattern = JacobianPattern(geom, n_var)
    return pattern.assemble(blocks)


def block_diagonal(blocks):
    """Sparse block-diagonal matrix from blocks of shape (n, k, k)."""
    n, k, _ = blocks.shape
    indptr = np.arange(n + 1)
    indices = np.arange(n)
    return sp.bsr_matrix((blocks, indices, indptr), shape=(n * k, n * k)).tocsr()


def replace_rows(matrix, rows, diagonal=1.0):
    """Return a copy with the given rows set to ``diagonal`` on the diagonal."""
    matrix = matrix.tocsr()
    keep = np.ones(matrix.shape[0])
    keep[rows] = 0.0
    out = sp.diags(keep) @ matrix
    out = out + sp.coo_matrix(
        (np.full(len(rows), diagonal), (rows, rows)), shape=matrix.shape
    )
    return out.tocsr()

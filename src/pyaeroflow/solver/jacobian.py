"""Approximate (first-order) Jacobian by automatic differentiation (D25).

The Jacobian is assembled from compact face fluxes that depend only on the
two cells next to a face (interior faces) or on the owner cell (boundary
faces). JAX differentiates each face flux; the blocks are assembled into a
SciPy sparse matrix with block size n_var, unknowns ordered cell by cell
(index ``i * n_var + v``), matching ``u.ravel()``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


@partial(jax.jit, static_argnums=(1, 2))
def _face_blocks(geom, interior_flux, boundary_flux, u, params, b_args, f_args=()):
    """Flux derivatives per face, scaled by the face length."""
    o, n, b = geom.f_owner, geom.f_neigh, geom.b_owner
    n_f = len(f_args)
    d_inner = jax.vmap(
        jax.jacfwd(interior_flux, argnums=(1, 2)),
        in_axes=(None, 0, 0, *([0] * n_f), 0, 0),
    )
    d_o, d_n = d_inner(params, u[o], u[n], *f_args, geom.f_normal, geom.f_delta)
    n_extra = len(b_args)
    d_bnd = jax.vmap(
        jax.jacfwd(boundary_flux, argnums=1), in_axes=(None, 0, *([0] * n_extra), 0, 0)
    )
    d_b = d_bnd(params, u[b], *b_args, geom.b_normal, geom.b_delta)
    fl = geom.f_length[:, None, None]
    return d_o * fl, d_n * fl, d_b * geom.b_length[:, None, None]


@partial(jax.jit, static_argnums=(1, 2))
def _face_blocks_sel(
    geom, interior_flux, boundary_flux, u, params, b_args, f_args, block_mask, sel
):
    """Face-block entries on the structural pattern, plus the dropped mass.

    ``sel`` = ``(sel_o, sel_n, sel_b)``: flat indices (into the n_var x n_var
    block) of the entries kept per block kind. Returns the compressed blocks
    (n_faces, len(sel)) and the largest absolute value among the entries that
    are *not* kept (0 when the structure is complete).
    """
    d_o, d_n, d_b = _face_blocks(
        geom, interior_flux, boundary_flux, u, params, b_args, f_args
    )
    out, dropped = [], 0.0
    for d, idx in zip((d_o, d_n, d_b), sel, strict=True):
        d = (d * block_mask[None]).reshape(d.shape[0], -1)
        out.append(d[:, idx])
        kept = jnp.zeros(d.shape[1], bool).at[idx].set(True)
        dropped = jnp.maximum(dropped, jnp.max(jnp.where(kept[None], 0.0, jnp.abs(d))))
    return (*out, dropped)


class JacobianPattern:
    """Fixed sparsity of the first-order Jacobian of one geometry.

    The face blocks always land on the same matrix entries, so the CSR
    structure and the map from face-block entries to CSR positions are built
    once (``np.unique`` on the row/column keys) and every assembly is one
    ``np.bincount``; rebuilding a COO matrix per iteration cost about as much
    as a residual evaluation on a 100k-cell mesh.

    ``masks`` restricts the pattern to the structurally non-zero entries of
    the n_var x n_var blocks (keys ``"o"``, ``"n"``, ``"b"`` for the face
    blocks d/d(owner), d/d(neighbour), d/d(boundary owner) and ``"d"`` for
    the extra cell-diagonal blocks such as the sources): the coupled
    Langtry-Menter Jacobian then has about 30 % of the entries of the dense
    7 x 7 blocks, assembled directly and in one pass. Without ``masks`` the
    blocks are dense and there are no extra diagonal blocks.
    """

    KINDS = "onbd"

    def __init__(self, geom, n_var, masks=None):
        self.n_var = n_var
        self.geom = geom
        n_cells = int(np.asarray(geom.area).shape[0])
        self.n_cells = n_cells
        self.shape = (n_cells * n_var, n_cells * n_var)
        full = np.ones((n_var, n_var), bool)
        self.masks = {
            k: (
                np.array(masks[k], bool)
                if masks is not None and k in masks
                else (full.copy() if k != "d" else np.zeros_like(full))
            )
            for k in self.KINDS
        }
        self.sel = {k: np.flatnonzero(self.masks[k].ravel()) for k in self.KINDS}
        self.dense = masks is None
        self._build()

    def _build(self):
        g, n_var, size = self.geom, self.n_var, self.shape[0]
        o, n, b = (np.asarray(a) for a in (g.f_owner, g.f_neigh, g.b_owner))
        cells = np.arange(self.n_cells)
        parts = (  # (kind, block rows, block columns) in assembly order
            ("o", o, o),
            ("n", o, n),
            ("o", n, o),
            ("n", n, n),
            ("b", b, b),
            ("d", cells, cells),
        )
        keys, self._segments = [], []
        start = 0
        for kind, brow, bcol in parts:
            sel = self.sel[kind]
            a, c = sel // n_var, sel % n_var
            rows = brow[:, None] * n_var + a[None, :]
            cols = bcol[:, None] * n_var + c[None, :]
            keys.append((rows.astype(np.int64) * size + cols).ravel())
            self._segments.append((start, start + keys[-1].size))
            start += keys[-1].size
        key = np.concatenate(keys)
        unique, inverse = np.unique(key, return_inverse=True)
        self.position = inverse.astype(np.intp)  # block entry -> CSR entry
        self.indices = (unique % size).astype(np.int32)
        self.indptr = np.searchsorted(unique // size, np.arange(size + 1)).astype(
            np.int32
        )
        self._weights = np.empty(start)
        # data position of every diagonal entry (row r, column r)
        row_of = np.repeat(np.arange(size), np.diff(self.indptr))
        self._diag = np.flatnonzero(row_of == self.indices)
        self._diag_row = row_of[self._diag]

    def widen(self, masks):
        """Add entries to the structure (a rebuild); returns True if it grew."""
        grown = False
        for k, m in masks.items():
            m = np.asarray(m, bool)
            if (m & ~self.masks[k]).any():
                self.masks[k] = self.masks[k] | m
                grown = True
        if grown:
            self.sel = {k: np.flatnonzero(self.masks[k].ravel()) for k in self.KINDS}
            self._build()
        return grown

    def assemble(self, blocks):
        """CSR matrix from dense face blocks ordered (d_o, d_n, -d_o, -d_n, d_b)."""
        data = np.bincount(
            self.position[: np.asarray(blocks).size],
            weights=blocks.ravel(),
            minlength=self.indices.size,
        )
        return sp.csr_matrix((data, self.indices, self.indptr), shape=self.shape)

    def assemble_selected(self, c_o, c_n, c_b, c_d=None):
        """CSR matrix from the compressed face blocks (and diagonal blocks).

        One ``np.bincount`` over a re-used weight buffer; the signs of the
        neighbour contributions are applied while filling it.
        """
        w, seg = self._weights, self._segments
        w[seg[0][0] : seg[0][1]] = c_o.ravel()
        w[seg[1][0] : seg[1][1]] = c_n.ravel()
        np.negative(c_o.ravel(), out=w[seg[2][0] : seg[2][1]])
        np.negative(c_n.ravel(), out=w[seg[3][0] : seg[3][1]])
        w[seg[4][0] : seg[4][1]] = c_b.ravel()
        if c_d is None:
            w[seg[5][0] : seg[5][1]] = 0.0
        else:
            w[seg[5][0] : seg[5][1]] = c_d.ravel()
        data = np.bincount(self.position, weights=w, minlength=self.indices.size)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=self.shape)

    def replace_rows(self, matrix, rows, diagonal=1.0):
        """Set the given rows of ``matrix`` (this pattern) to ``diagonal`` I."""
        for r in rows:
            matrix.data[matrix.indptr[r] : matrix.indptr[r + 1]] = 0.0
            hit = self._diag[self._diag_row == r]
            matrix.data[hit] = diagonal
        return matrix


def first_order_jacobian(
    geom,
    interior_flux,
    boundary_flux,
    u,
    params=None,
    b_args=(),
    block_mask=None,
    pattern=None,
    f_args=(),
    diag_blocks=None,
    pin_rows=(),
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
    f_args : tuple of arrays
        Per-interior-face arrays (first axis n_ifaces) passed to
        ``interior_flux`` after the two states; held fixed in the derivative.
    diag_blocks : array (n_cells, n_var, n_var), optional
        Cell-local blocks added to the diagonal (the source Jacobian).
    pin_rows : sequence of int
        Matrix rows replaced by rows of the identity (pressure pin).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_cells * n_var, n_cells * n_var)
    """
    n_var = u.shape[1]
    if pattern is None:
        pattern = JacobianPattern(geom, n_var)
    if not pattern.dense:
        return _assemble_selected(
            geom,
            interior_flux,
            boundary_flux,
            u,
            params,
            b_args,
            f_args,
            block_mask,
            pattern,
            diag_blocks,
            pin_rows,
        )
    d_o, d_n, d_b = (
        np.asarray(a)
        for a in _face_blocks(
            geom, interior_flux, boundary_flux, u, params, b_args, tuple(f_args)
        )
    )
    blocks = np.concatenate((d_o, d_n, -d_o, -d_n, d_b))
    if block_mask is not None:
        blocks = blocks * np.asarray(block_mask, dtype=float)[None]
    matrix = pattern.assemble(blocks)
    if diag_blocks is not None:
        matrix = matrix + block_diagonal(np.asarray(diag_blocks))
    if len(pin_rows):
        matrix = replace_rows(matrix, list(pin_rows))
    return matrix


def structural_masks(
    geom,
    interior_flux,
    boundary_flux,
    states,
    params,
    b_args,
    f_args=(),
    block_mask=None,
    diag_samples=(),
):
    """Structurally non-zero entries of the blocks, from sample states.

    An entry counts when it is non-zero on any face (or cell, for
    ``diag_samples``) at any of the ``states``; with states that differ in
    sign and magnitude (upwind branches, wall values) this is the structure.
    Returns the masks for :class:`JacobianPattern`.
    """
    n_var = states[0].shape[1]
    masks = {k: np.zeros((n_var, n_var), bool) for k in "onbd"}
    bm = None if block_mask is None else np.asarray(block_mask, float)[None]
    for u in states:
        blocks = _face_blocks(
            geom, interior_flux, boundary_flux, u, params, b_args, tuple(f_args)
        )
        for kind, d in zip("onb", blocks, strict=True):
            d = np.asarray(d) if bm is None else np.asarray(d) * bm
            masks[kind] |= np.any(d != 0.0, axis=0)
    for d in diag_samples:
        masks["d"] |= np.any(np.asarray(d) != 0.0, axis=0)
    return masks


def _assemble_selected(
    geom,
    interior_flux,
    boundary_flux,
    u,
    params,
    b_args,
    f_args,
    block_mask,
    pattern,
    diag_blocks,
    pin_rows,
):
    """One-pass assembly on the structural pattern (C2)."""
    n_var = pattern.n_var
    bm = jnp.asarray(
        np.ones((n_var, n_var)) if block_mask is None else block_mask, dtype=float
    )
    if diag_blocks is not None:
        diag_blocks = np.asarray(diag_blocks)
        lost = np.abs(
            diag_blocks.reshape(len(diag_blocks), -1)[:, ~pattern.masks["d"].ravel()]
        )
        if lost.size and lost.max() > 0.0:
            pattern.widen({"d": diag_blocks.any(axis=0)})
    for _ in range(2):
        sel = tuple(jnp.asarray(pattern.sel[k]) for k in "onb")
        c_o, c_n, c_b, dropped = _face_blocks_sel(
            geom,
            interior_flux,
            boundary_flux,
            u,
            params,
            b_args,
            tuple(f_args),
            bm,
            sel,
        )
        if float(dropped) == 0.0:
            break
        # a structural entry the samples missed: widen the pattern and retry
        full = _face_blocks(
            geom, interior_flux, boundary_flux, u, params, b_args, tuple(f_args)
        )
        new = {
            k: np.any(np.asarray(d) * np.asarray(bm)[None] != 0.0, axis=0)
            for k, d in zip("onb", full, strict=True)
        }
        pattern.widen(new)
    c_d = None
    if diag_blocks is not None:
        c_d = diag_blocks.reshape(len(diag_blocks), -1)[:, pattern.sel["d"]]
    matrix = pattern.assemble_selected(
        np.asarray(c_o), np.asarray(c_n), np.asarray(c_b), c_d
    )
    if len(pin_rows):
        pattern.replace_rows(matrix, list(pin_rows))
    return matrix


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

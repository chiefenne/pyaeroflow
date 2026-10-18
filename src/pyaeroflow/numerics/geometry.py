"""Finite-volume geometry as JAX arrays.

:func:`build_geometry` converts a :class:`~pyaeroflow.mesh.Mesh` into the arrays
used by the jitted numerics: interior faces, boundary faces (grouped by
boundary name) and the least-squares gradient stencil. All preprocessing is
done once in NumPy.

Periodic boundaries (D38, D39 R4): a pair of boundary-face sets related by a
translation is turned into interior faces. The neighbour across such a face is
seen at its translated position, so all operators treat it like any other
interior face and no boundary-condition code is involved.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from pyaeroflow.mesh.mesh import MeshError
from pyaeroflow.numerics.gradients import lsq_stencil


class FVGeometry(NamedTuple):
    """Geometry arrays for jitted finite-volume operators (a JAX pytree).

    Interior faces are prefixed ``f_``, boundary faces ``b_``. Face normals
    point from owner to neighbour, and out of the domain on boundary faces.

    - ``f_delta``: vector from the owner centroid to the (translated)
      neighbour centroid.
    - ``f_dx_owner`` / ``f_dx_neigh``: vector from the owner / (translated)
      neighbour centroid to the face centre.
    - ``b_delta``: vector from the owner centroid to the boundary-face centre.

    Boundary faces are ordered by boundary name (see :class:`Discretization`).
    ``lsq_coeff`` has shape (n_entries, n_sets, 2) with one coefficient set
    per variable, or a single shared set.
    """

    area: jnp.ndarray
    centroid: jnp.ndarray
    f_owner: jnp.ndarray
    f_neigh: jnp.ndarray
    f_normal: jnp.ndarray
    f_length: jnp.ndarray
    f_centre: jnp.ndarray
    f_delta: jnp.ndarray
    f_dx_owner: jnp.ndarray
    f_dx_neigh: jnp.ndarray
    b_owner: jnp.ndarray
    b_normal: jnp.ndarray
    b_length: jnp.ndarray
    b_centre: jnp.ndarray
    b_delta: jnp.ndarray
    lsq_cell: jnp.ndarray
    lsq_src: jnp.ndarray
    lsq_coeff: jnp.ndarray
    wall_distance: jnp.ndarray = None
    production_switch: jnp.ndarray = None


class Discretization(NamedTuple):
    """Mesh, its JAX geometry, and the position of each boundary in ``b_*``.

    Attributes
    ----------
    mesh : Mesh
        The source mesh.
    geom : FVGeometry
        Arrays for the jitted operators.
    boundary_slices : dict of str to slice
        Range of each named set of boundary faces in the ``b_*`` arrays
        (periodic sets are absent: their faces are interior).
    boundary_face_ids : ndarray
        Mesh face index of every boundary face, in ``b_*`` order.
    periodic_faces : ndarray, shape (n_periodic, 2)
        Mesh face indices of the matched periodic pairs (side a, side b).
    """

    mesh: object
    geom: FVGeometry
    boundary_slices: dict
    boundary_face_ids: np.ndarray
    periodic_faces: np.ndarray


def build_geometry(mesh, weight_exponent=1.0, periodic=(), lsq_include=None):
    """Build the finite-volume geometry of a mesh.

    Parameters
    ----------
    mesh : Mesh
        The mesh.
    weight_exponent : float
        Exponent q of the least-squares weights 1/d^q (see
        :func:`~pyaeroflow.numerics.gradients.lsq_stencil`).
    periodic : sequence of (name_a, name_b, translation)
        Periodic pairs of boundary-face sets with ``x_b = x_a + translation``.
    lsq_include : callable, optional
        ``lsq_include(boundary_slices, n_bfaces) -> bool array (n_bfaces,
        n_var)``: which boundary faces take part in which variable's
        gradient. Default: all faces for all variables.

    Returns
    -------
    Discretization
    """
    periodic_names = set()
    for name_a, name_b, _ in periodic:
        periodic_names |= {name_a, name_b}
    slices, ids, start = {}, [], 0
    for name, faces in mesh.boundary_faces.items():
        if name in periodic_names:
            continue
        slices[name] = slice(start, start + faces.size)
        ids.append(faces)
        start += faces.size
    bnd = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)

    # Interior faces, then the periodic pairs as additional interior faces.
    interior = np.flatnonzero(mesh.neighbour >= 0)
    c = mesh.cell_centroid
    o, n = mesh.owner[interior], mesh.neighbour[interior]
    normal, length = mesh.face_normal[interior], mesh.face_length[interior]
    centre = mesh.face_centre[interior]
    delta = c[n] - c[o]
    pairs, node_pairs = [], []
    for name_a, name_b, translation in periodic:
        t = np.asarray(translation, dtype=float)
        fa, fb = _match_periodic(mesh, name_a, name_b, t)
        pairs.append(np.column_stack((fa, fb)))
        node_pairs.append((*_match_periodic_nodes(mesh, fa, fb, t), t))
        o = np.concatenate((o, mesh.owner[fa]))
        n = np.concatenate((n, mesh.owner[fb]))
        normal = np.vstack((normal, mesh.face_normal[fa]))
        length = np.concatenate((length, mesh.face_length[fa]))
        centre = np.vstack((centre, mesh.face_centre[fa]))
        delta = np.vstack((delta, c[mesh.owner[fb]] - t - c[mesh.owner[fa]]))
    periodic_faces = np.vstack(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)

    include = None if lsq_include is None else lsq_include(slices, bnd.size)
    cell, src, coeff = lsq_stencil(mesh, bnd, weight_exponent, include, node_pairs)
    ob = mesh.owner[bnd]
    dx_owner = centre - c[o]
    geom = FVGeometry(
        area=jnp.asarray(mesh.cell_area),
        centroid=jnp.asarray(c),
        f_owner=jnp.asarray(o),
        f_neigh=jnp.asarray(n),
        f_normal=jnp.asarray(normal),
        f_length=jnp.asarray(length),
        f_centre=jnp.asarray(centre),
        f_delta=jnp.asarray(delta),
        f_dx_owner=jnp.asarray(dx_owner),
        f_dx_neigh=jnp.asarray(dx_owner - delta),
        b_owner=jnp.asarray(ob),
        b_normal=jnp.asarray(mesh.face_normal[bnd]),
        b_length=jnp.asarray(mesh.face_length[bnd]),
        b_centre=jnp.asarray(mesh.face_centre[bnd]),
        b_delta=jnp.asarray(mesh.face_centre[bnd] - c[ob]),
        lsq_cell=jnp.asarray(cell),
        lsq_src=jnp.asarray(src),
        lsq_coeff=jnp.asarray(coeff),
    )
    return Discretization(mesh, geom, slices, bnd, periodic_faces)


def _match_periodic(mesh, name_a, name_b, translation):
    """Pair the faces of two boundary sets by translation (D39 R4)."""
    for name in (name_a, name_b):
        if name not in mesh.boundary_faces:
            raise MeshError(f"periodic boundary '{name}' does not exist in the mesh")
    fa, fb = mesh.boundary_faces[name_a], mesh.boundary_faces[name_b]
    if fa.size != fb.size:
        raise MeshError(
            f"periodic boundaries '{name_a}' ({fa.size} faces) and '{name_b}' "
            f"({fb.size} faces) differ in face count"
        )
    tree = cKDTree(mesh.face_centre[fb])
    dist, idx = tree.query(mesh.face_centre[fa] + translation)
    tol = 1e-6 * mesh.face_length[fa]
    bad = dist > tol
    if np.any(bad) or np.unique(idx).size != fa.size:
        k = np.flatnonzero(bad)[:1]
        where = mesh.face_centre[fa[k]] if k.size else mesh.face_centre[fa[:1]]
        raise MeshError(
            f"periodic boundaries '{name_a}' and '{name_b}': faces do not match "
            f"under translation {translation.tolist()}, e.g. face at "
            f"({where[0, 0]:.6g}, {where[0, 1]:.6g})"
        )
    lengths_differ = ~np.isclose(mesh.face_length[fa], mesh.face_length[fb[idx]])
    if np.any(lengths_differ):
        raise MeshError(
            f"periodic boundaries '{name_a}' and '{name_b}': matched faces differ "
            "in length"
        )
    return fa, fb[idx]


def _match_periodic_nodes(mesh, fa, fb, translation):
    """Node pairs (a, b) of matched periodic faces, ordered by position."""
    na = mesh.face_nodes[fa]  # (n, 2)
    nb = mesh.face_nodes[fb]
    # Node 0 of face a maps to whichever node of face b is nearer to x_a + t.
    xa = mesh.nodes[na[:, 0]] + translation
    d0 = np.linalg.norm(mesh.nodes[nb[:, 0]] - xa, axis=1)
    d1 = np.linalg.norm(mesh.nodes[nb[:, 1]] - xa, axis=1)
    swap = d1 < d0
    nb_ordered = np.where(swap[:, None], nb[:, ::-1], nb)
    pairs = np.unique(np.column_stack((na.ravel(), nb_ordered.ravel())), axis=0)
    return pairs[:, 0], pairs[:, 1]

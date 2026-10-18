"""Distance of every cell centre to the nearest wall face (SA, y+).

The exact distance to a set of straight boundary faces (segments) is found
by a candidate search: a KD-tree over the face midpoints and end nodes gives
the ``k`` nearest sample points of every cell centre, and the exact
point-segment distance is evaluated for the faces of those samples. With
``k`` = 16 this is exact unless the wall faces vary by more than about an
order of magnitude in length within a neighbourhood, which the mesh checks
of PyAero/the generators exclude in practice.
"""

import numpy as np
from scipy.spatial import cKDTree


def segment_distance(points, a, b):
    """Distance from ``points`` (n, 2) to the segments a-b, arrays (n, 2)."""
    ab = b - a
    t = np.einsum("ij,ij->i", points - a, ab) / np.maximum(
        np.einsum("ij,ij->i", ab, ab), 1e-300
    )
    t = np.clip(t, 0.0, 1.0)
    foot = a + t[:, None] * ab
    return np.linalg.norm(points - foot, axis=1)


def wall_distance(points, segments, k=16):
    """Minimum distance from ``points`` (n, 2) to ``segments`` (m, 2, 2).

    Returns the distances (n,) and the index of the nearest segment (n,);
    infinity and -1 if there are no segments.
    """
    points = np.asarray(points, dtype=float)
    segments = np.asarray(segments, dtype=float)
    if segments.shape[0] == 0:
        return np.full(points.shape[0], np.inf), np.full(points.shape[0], -1)
    a, b = segments[:, 0], segments[:, 1]
    samples = np.vstack((0.5 * (a + b), a, b))
    owner = np.tile(np.arange(segments.shape[0]), 3)
    k = min(k, samples.shape[0])
    _, idx = cKDTree(samples).query(points, k=k)
    idx = np.atleast_2d(idx).reshape(points.shape[0], k)
    faces = owner[idx]  # (n, k)
    rep = np.repeat(points, k, axis=0)
    d = segment_distance(rep, a[faces.ravel()], b[faces.ravel()]).reshape(-1, k)
    best = np.argmin(d, axis=1)
    rows = np.arange(points.shape[0])
    return d[rows, best], faces[rows, best]


def mesh_wall_distance(mesh, wall_faces, k=16):
    """Distance of the cell centroids to the given mesh faces.

    Returns the distances and the mesh face index of the nearest wall face
    (-1 without wall faces).
    """
    faces = np.asarray(wall_faces, dtype=int)
    segments = mesh.nodes[mesh.face_nodes[faces]]  # (m, 2, 2)
    d, nearest = wall_distance(mesh.cell_centroid, segments, k)
    return d, np.where(nearest >= 0, faces[np.maximum(nearest, 0)], -1)

"""Ordering of boundary faces along a wall contour."""

import numpy as np


def order_boundary_faces(face_nodes, nodes):
    """Order the faces of one wall boundary along its contour.

    Parameters
    ----------
    face_nodes : ndarray, shape (m, 2)
        Node indices of the boundary faces.
    nodes : ndarray, shape (n_nodes, 2)
        Node coordinates.

    Returns
    -------
    ndarray of int or None
        Positions into ``face_nodes`` in contour order, or ``None`` when the
        faces do not form one single chain (several walls in one boundary).

    The direction follows PyAero's airfoil contour: a closed contour starts
    at the trailing edge (largest x, upper corner of a blunt trailing edge)
    and runs over the upper side to the leading edge and back along the
    lower side; an open chain (a plate) starts at its smaller x.
    """
    face_nodes = np.asarray(face_nodes)
    m = face_nodes.shape[0]
    if m == 0:
        return np.zeros(0, dtype=np.int64)
    at_node = {}
    for f, (a, b) in enumerate(face_nodes):
        at_node.setdefault(int(a), []).append((f, int(b)))
        at_node.setdefault(int(b), []).append((f, int(a)))
    if any(len(v) > 2 for v in at_node.values()):
        return None
    ends = [n for n, v in at_node.items() if len(v) == 1]
    xy = np.asarray(nodes)
    if len(ends) == 2:
        start = min(ends, key=lambda n: (xy[n, 0], xy[n, 1]))
    elif not ends:
        x_max = max(xy[n, 0] for n in at_node)
        scale = np.ptp(xy[list(at_node)], axis=0).max()
        near = [n for n in at_node if xy[n, 0] >= x_max - 1e-9 * scale]
        start = max(near, key=lambda n: xy[n, 1])
    else:
        return None
    order, seen, node = [], set(), start
    while True:
        options = [(f, o) for f, o in at_node[node] if f not in seen]
        if not options:
            break
        if len(options) == 2:  # closed contour: leave on the side with larger y
            options.sort(key=lambda fo: xy[fo[1], 1], reverse=True)
        f, node = options[0]
        seen.add(f)
        order.append(f)
    if len(order) != m:
        return None
    return np.asarray(order, dtype=np.int64)

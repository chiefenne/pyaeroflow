"""Airfoil coordinates for the viscous panel analysis: reading and panelling."""

import numpy as np
from scipy.interpolate import CubicSpline


def read_coordinates(path):
    """Read an airfoil coordinate file (Selig or Lednicer format).

    Lines that do not hold two numbers (name, comments) are skipped. In the
    Lednicer format the two counts line ("61. 61.") is recognised by values
    above 1.5 and the two surfaces are joined trailing edge -> leading edge ->
    trailing edge.
    """
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    xy = np.array(rows)
    if xy.shape[0] > 2 and xy[0, 0] > 1.5 and xy[0, 1] > 1.5:  # Lednicer
        nu, nl = int(xy[0, 0]), int(xy[0, 1])
        upper, lower = xy[1 : 1 + nu], xy[1 + nu : 1 + nu + nl]
        xy = np.vstack((upper[::-1], lower[1:]))
    return xy


def naca4(code, n=161, closed_te=False):
    """NACA four-digit airfoil, trailing edge -> upper -> leading edge -> lower."""
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    a4 = -0.1036 if closed_te else -0.1015
    yt = (
        5.0
        * t
        * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)
    )
    if m > 0.0:
        yc = np.where(
            x < p,
            m / p**2 * (2 * p * x - x**2),
            m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2),
        )
        dy = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = np.zeros_like(x)
        dy = np.zeros_like(x)
    th = np.arctan(dy)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return np.vstack(
        (np.column_stack((xu[::-1], yu[::-1])), np.column_stack((xl[1:], yl[1:])))
    )


def repanel(xy, n_nodes=200, leading_edge_weight=0.6, trailing_edge_weight=2.0):
    """Redistribute the nodes of an airfoil contour for the panel analysis.

    Parameters
    ----------
    xy : array_like (m, 2)
        Contour from the trailing edge over one side to the leading edge and
        back along the other side (Selig order or reverse). A sharp trailing
        edge may or may not repeat its point.
    n_nodes : int
        Number of nodes of the new contour (both trailing-edge corners count).
    leading_edge_weight, trailing_edge_weight : float
        Clustering towards the leading edge (curvature) and the trailing edge.

    Returns
    -------
    ndarray (n_nodes, 2)
        Nodes from the upper trailing edge over the upper side round the
        leading edge to the lower trailing edge; a sharp trailing edge (gap
        below 1e-4 of the airfoil size) is closed and repeated as first and
        last node.

    The contour is a cubic spline in arc length through the given points;
    the nodes are placed at equal increments of the integrated density
    1 + w_le sqrt(c kappa) + w_te exp(-s_te / 0.02 c), where kappa is the
    spline curvature, c the chord and s_te the arc length to the nearer
    trailing edge.
    """
    pts = np.asarray(xy, dtype=float)
    keep = np.concatenate(
        ([True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-12)
    )
    pts = pts[keep]
    # orientation: upper side first (counter-clockwise)
    area = 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
    if area < 0.0:
        pts = pts[::-1]
    size = float(np.max(np.ptp(pts, axis=0)))
    sharp = np.linalg.norm(pts[0] - pts[-1]) < 1e-4 * size
    if sharp:  # close a (nearly) sharp trailing edge exactly
        pts[0] = pts[-1] = 0.5 * (pts[0] + pts[-1])
    t = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    sx, sy = CubicSpline(t, pts[:, 0]), CubicSpline(t, pts[:, 1])
    fine = np.linspace(0.0, t[-1], 20001)
    dx, dy = sx(fine, 1), sy(fine, 1)
    ddx, ddy = sx(fine, 2), sy(fine, 2)
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-30)
    te = 0.5 * (pts[0] + pts[-1])
    chord = float(np.max(np.hypot(pts[:, 0] - te[0], pts[:, 1] - te[1])))
    s_te = np.minimum(fine, t[-1] - fine)
    density = (
        1.0
        + leading_edge_weight * np.sqrt(chord * kappa)
        + trailing_edge_weight * np.exp(-s_te / (0.02 * chord))
    )
    cum = np.concatenate(
        ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(fine)))
    )
    targets = np.linspace(0.0, cum[-1], n_nodes)
    tn = np.interp(targets, cum, fine)
    nodes = np.column_stack((sx(tn), sy(tn)))
    nodes[0], nodes[-1] = pts[0], pts[-1]
    if sharp:
        nodes[-1] = nodes[0]
    return nodes

"""Airfoil geometry and an O-grid around it (D42; production meshes: PyAero).

- :func:`naca0012`: the sharp-trailing-edge NACA 0012 of the NASA TMR
  verification cases.
- :func:`airfoil_points`: surface nodes with cosine clustering.
- :func:`airfoil_ogrid`: structured O-grid by marching the surface outward
  along smoothed normals with geometric spacing; the farfield is the last
  layer (nearly circular at large distance). The sharp trailing edge opens a
  fan of cells that the normal smoothing spreads over a few nodes.
"""

import numpy as np

from pyaeroflow.mesh.generators import geometric_growth
from pyaeroflow.mesh.mesh import Mesh


def naca0012(x):
    """Half thickness of the TMR NACA 0012 (sharp trailing edge at x = 1)."""
    x = np.asarray(x, dtype=float)
    return 0.594689181 * (
        0.298222773 * np.sqrt(x)
        - 0.127125232 * x
        - 0.357907906 * x**2
        + 0.291984971 * x**3
        - 0.105174606 * x**4
    )


def airfoil_points(n, thickness=naca0012, cut=0.995, n_base=4):
    """Return the closed surface loop, counter-clockwise from the trailing edge.

    Cosine spacing along the chord; upper surface first (TE to LE), then the
    lower surface (LE to TE) and the base of the blunt trailing edge. The
    airfoil is cut at ``cut`` and scaled back to unit chord: a sharp trailing
    edge produces a fan of nearly flat cells in an O-grid (interior angle
    close to 180 degrees) on which the second-order reconstruction is
    unstable at high Reynolds numbers; the base (0.14 % chord for the NACA
    0012 at ``cut`` = 0.995) carries ``n_base`` faces. ``cut`` = 1 gives the
    sharp trailing edge. ``n`` counts the surface nodes without the base.
    """
    if n % 2:
        raise ValueError("n must be even")
    theta = np.linspace(0.0, np.pi, n // 2 + 1)
    x = cut * 0.5 * (1.0 - np.cos(theta))[::-1]  # cut -> 0, TE to LE
    y = thickness(x)
    upper = np.column_stack((x[:-1], y[:-1]))  # TE ... before LE
    lower = np.column_stack((x[::-1][:-1], -y[::-1][:-1]))  # LE ... before TE
    pts = np.vstack((upper, lower))
    if cut < 1.0 and y[0] > 0.0:
        base_y = np.linspace(-y[0], y[0], n_base + 1)[1:-1]
        pts = np.vstack((pts, np.column_stack((np.full(base_y.size, x[0]), base_y))))
    return pts / cut


def _loop_normals(loop):
    """Outward unit normals of a counter-clockwise closed loop (at the nodes)."""
    tangent = np.roll(loop, -1, axis=0) - np.roll(loop, 1, axis=0)
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    return normal / np.linalg.norm(normal, axis=1)[:, None]


def _smooth(vectors, passes, weight=0.5):
    for _ in range(passes):
        vectors = vectors + weight * (
            np.roll(vectors, 1, axis=0) + np.roll(vectors, -1, axis=0)
        )
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    return vectors


def airfoil_ogrid(
    surface,
    n_radial,
    *,
    first_spacing,
    farfield_radius=500.0,
    max_growth=1.35,
    names=("airfoil", "farfield"),
):
    """O-grid from a closed surface loop to a farfield at ``farfield_radius``.

    Parameters
    ----------
    surface : array (n, 2)
        Counter-clockwise closed loop of surface nodes (first node repeated
        nowhere).
    n_radial : int
        Layers between the surface and the farfield.
    first_spacing : float
        Wall-normal spacing of the first layer (from the target y+).
    farfield_radius : float
        Distance of the farfield from the surface along the marching
        direction, in chords; NASA TMR recommends ~500 without a vortex
        correction (P10).
    max_growth : float
        Upper bound of the geometric growth ratio (raises if n_radial is
        too small for the radius).

    Returns
    -------
    Mesh
    """
    surface = np.asarray(surface, dtype=float)
    n = surface.shape[0]
    r = geometric_growth(farfield_radius, first_spacing, n_radial)
    if r > max_growth:
        raise ValueError(f"growth ratio {r:.3f} exceeds max_growth; increase n_radial")
    spacing = first_spacing * r ** np.arange(n_radial)
    centre = 0.5 * (surface.min(axis=0) + surface.max(axis=0))
    chord = np.ptp(surface[:, 0])
    layers = [surface]
    normals = _smooth(_loop_normals(surface), 6)
    loop, distance = surface, 0.0
    for d in spacing:
        # Marching direction: smoothed surface normals near the wall, blended
        # into the radial direction from the centre with distance, so that the
        # grid is orthogonal at the wall and circular at the farfield.
        radial = loop - centre
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        b = min(1.0, distance / (2.0 * chord))
        direction = (1.0 - b) * normals + b * radial
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        loop = loop + d * direction
        distance += d
        # Light coordinate smoothing along the loop keeps nodes evenly spread.
        w = 0.25 * b
        loop = (1.0 - w) * loop + w * 0.5 * (
            np.roll(loop, 1, axis=0) + np.roll(loop, -1, axis=0)
        )
        layers.append(loop)
        normals = _smooth(_loop_normals(loop), 2 + int(20 * b))
    nodes = np.vstack(layers)

    def node(j, k):
        return j * n + (k % n)

    j, k = np.meshgrid(np.arange(n_radial), np.arange(n), indexing="ij")
    j, k = j.ravel(), k.ravel()
    quads = np.column_stack(
        (node(j, k), node(j + 1, k), node(j + 1, k + 1), node(j, k + 1))
    )
    _check_folds(nodes, quads)
    k = np.arange(n)
    boundary = {
        names[0]: np.column_stack((node(0, k + 1), node(0, k))),
        names[1]: np.column_stack((node(n_radial, k), node(n_radial, k + 1))),
    }
    return Mesh.from_cells(nodes, quads, boundary)


def _check_folds(nodes, quads):
    """Raise if any quad is folded (non-positive signed area)."""
    p = nodes[quads]  # (m, 4, 2)
    x, y = p[..., 0], p[..., 1]
    area = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    if np.any(area <= 0.0):
        raise ValueError(f"O-grid folded in {int(np.sum(area <= 0.0))} cells")


def naca0012_ogrid(
    n_surface=256, n_radial=128, *, reynolds=6.0e6, y_plus=1.0, farfield_radius=500.0
):
    """O-grid around the TMR NACA 0012 with the first spacing from y+ (C_f ~ 0.003)."""
    first = y_plus / (reynolds * np.sqrt(0.003 / 2.0))
    return airfoil_ogrid(
        airfoil_points(n_surface),
        n_radial,
        first_spacing=first,
        farfield_radius=farfield_radius,
    )

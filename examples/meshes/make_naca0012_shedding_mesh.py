"""O-grid around the NACA 0012 for laminar vortex shedding (Re = 1000).

Usage: python make_naca0012_shedding_mesh.py [OUT.su2] [n_surface]

Radial spacing (chord = 1): geometric from 0.002 at the wall up to 0.025,
constant 0.025 out to 5 chords (wake region of the video), then geometric
growth to the farfield at 30 chords. The marching follows
pyaeroflow.mesh.airfoil.airfoil_ogrid (smoothed normals blended into the
radial direction), with this spacing instead of a single growth ratio.
Boundary faces: airfoil, farfield.
"""

import sys

import numpy as np

from pyaeroflow.mesh import Mesh, write_su2
from pyaeroflow.mesh.airfoil import _check_folds, _loop_normals, _smooth, airfoil_points

out = sys.argv[1] if len(sys.argv) > 1 else "naca0012_shedding.su2"
n_surface = int(sys.argv[2]) if len(sys.argv) > 2 else 384
first, growth, plateau, r_plateau, r_far = 0.002, 1.08, 0.025, 5.0, 30.0

spacing, d = [], 0.0
s = first
while d < r_far:
    if d < r_plateau:
        s = min(s * growth, plateau) if spacing else first
    else:
        s = s * growth
    spacing.append(s)
    d += s
spacing = np.array(spacing) * (r_far / d)  # land exactly on the farfield radius

surface = airfoil_points(n_surface)
n = surface.shape[0]
n_radial = len(spacing)
centre = 0.5 * (surface.min(axis=0) + surface.max(axis=0))
chord = np.ptp(surface[:, 0])
layers = [surface]
normals = _smooth(_loop_normals(surface), 6)
loop, distance = surface, 0.0
for ds in spacing:
    radial = loop - centre
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    b = min(1.0, distance / (2.0 * chord))
    direction = (1.0 - b) * normals + b * radial
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    loop = loop + ds * direction
    distance += ds
    w = 0.25 * b
    loop = (1.0 - w) * loop + w * 0.5 * (
        np.roll(loop, 1, axis=0) + np.roll(loop, -1, axis=0)
    )
    layers.append(loop)
    normals = _smooth(_loop_normals(loop), 2 + int(20 * b))
nodes = np.vstack(layers)


def node(j, k):
    """Node index of radial layer j, surface point k (periodic)."""
    return j * n + (k % n)


j, k = np.meshgrid(np.arange(n_radial), np.arange(n), indexing="ij")
j, k = j.ravel(), k.ravel()
quads = np.column_stack(
    (node(j, k), node(j + 1, k), node(j + 1, k + 1), node(j, k + 1))
)
_check_folds(nodes, quads)
k = np.arange(n)
boundary = {
    "airfoil": np.column_stack((node(0, k + 1), node(0, k))),
    "farfield": np.column_stack((node(n_radial, k), node(n_radial, k + 1))),
}
mesh = Mesh.from_cells(nodes, quads, boundary)
write_su2(out, mesh)
print(
    f"{mesh.n_cells} cells, {n} surface nodes, {n_radial} layers, "
    f"first spacing {spacing[0]:.4f}"
)

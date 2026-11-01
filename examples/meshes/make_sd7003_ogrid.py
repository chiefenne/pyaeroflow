"""O-grid around the SD7003 contour of the user's PyAero mesh (from the AFT wall CSV).

The wall CSV of the SA-AFT alpha 4 run holds the wall-face centres in traversal
order. Upper and lower sides are resampled with cosine spacing in arc length,
cut at x = 0.995 with a 4-face base (as the NACA 0012 O-grid of the repo),
then the O-grid is marched to 100 chords (use the point-vortex correction).
Usage: python sd7003_mesh.py <wall_csv> <out.su2> [n_surface] [n_radial]
"""

import sys

import numpy as np

from pyaeroflow.mesh import write_su2
from pyaeroflow.mesh.airfoil import airfoil_ogrid

d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
p = np.column_stack((d["x"], d["y"]))
p = p[p[:, 0] < 0.9985]  # drop the blunt base faces of the PyAero mesh
ile = int(np.argmin(p[:, 0]))
a, b = p[: ile + 1], p[ile:]
upper, lower = (a, b) if a[:, 1].mean() > b[:, 1].mean() else (b, a)
if upper[0, 0] < upper[-1, 0]:
    upper = upper[::-1]  # TE -> LE
if lower[0, 0] > lower[-1, 0]:
    lower = lower[::-1]  # LE -> TE
n_side = (int(sys.argv[3]) if len(sys.argv) > 3 else 320) // 2
cut = 0.995


# Upper: TE -> LE; truncate where x drops below cut, then add the LE point.
def side_nodes(side, n, te_first):
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(side, axis=0), axis=1))))
    x = side[:, 0]
    if te_first:
        s_cut = np.interp(cut, x[::-1], s[::-1])
        t = s_cut + (s[-1] - s_cut) * 0.5 * (1 - np.cos(np.linspace(0, np.pi, n + 1)))
    else:
        s_cut = np.interp(cut, x, s)
        t = s_cut * 0.5 * (1 - np.cos(np.linspace(0, np.pi, n + 1)))
    return np.column_stack((np.interp(t, s, side[:, 0]), np.interp(t, s, side[:, 1])))


up = side_nodes(upper, n_side, True)  # x = cut ... LE
lo = side_nodes(lower, n_side, False)  # LE ... x = cut
base_y = np.linspace(lo[-1, 1], up[0, 1], 5)[1:-1]
base = np.column_stack((np.full(3, cut), base_y))
loop = np.vstack((up[:-1], lo[:-1], lo[-1:], base))
loop = loop / cut
area = 0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1])
assert area > 0, "loop must be counter-clockwise"
n_radial = int(sys.argv[4]) if len(sys.argv) > 4 else 120
# y+ ~ 0.5 at Re 6e4 with C_f up to 0.01.
first = 0.5 / (6.0e4 * np.sqrt(0.01 / 2.0))
mesh = airfoil_ogrid(loop, n_radial, first_spacing=first, farfield_radius=100.0)
write_su2(sys.argv[2], mesh)
print(mesh.n_cells, "cells,", loop.shape[0], "surface nodes, first spacing", first)

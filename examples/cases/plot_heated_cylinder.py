"""Temperature, isotherms and streamlines of the heated cylinder (gallery 9).

Usage: python plot_heated_cylinder.py FILE.vtu out.png
Needs matplotlib and meshio (pip install matplotlib meshio).
"""

import sys

import matplotlib
import meshio
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation

m = meshio.read(sys.argv[1])
q = m.cells[0].data
c = m.points[q][:, :, :2].mean(1)
T = m.cell_data["temperature"][0][:, 0] - 300.0
V = m.cell_data["velocity"][0][:, :2]
tri = Triangulation(c[:, 0], c[:, 1])
# mask triangles bridging the cylinder or far gaps
r = np.hypot(c[tri.triangles][:, :, 0], c[tri.triangles][:, :, 1])
edge = (
    np.ptp(c[tri.triangles][:, :, 0], axis=1) ** 2
    + np.ptp(c[tri.triangles][:, :, 1], axis=1) ** 2
)
tri.set_mask((r.min(1) < 0.52) & (edge > 0.04))
fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
lv = np.linspace(0, 3, 31)
cf = ax.tricontourf(tri, np.clip(T, 0, 3), levels=lv, cmap="magma")
ax.tricontour(
    tri,
    T,
    levels=[0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5],
    colors="white",
    linewidths=0.5,
    alpha=0.7,
)
# streamlines on a regular grid
gx, gy = np.meshgrid(np.linspace(-3, 9, 400), np.linspace(-3, 3, 200))
iu = LinearTriInterpolator(tri, V[:, 0])
iv = LinearTriInterpolator(tri, V[:, 1])
u = iu(gx, gy)
v = iv(gx, gy)
ax.streamplot(
    gx,
    gy,
    np.ma.filled(u, 0),
    np.ma.filled(v, 0),
    color=(1, 1, 1, 0.35),
    linewidth=0.5,
    density=1.6,
    arrowsize=0.5,
)
ax.add_patch(plt.Circle((0, 0), 0.5, color="#222", zorder=5))
ax.set_xlim(-3, 9)
ax.set_ylim(-3, 3)
ax.set_aspect("equal")
ax.set_xlabel("x / D")
ax.set_ylabel("y / D")
cb = fig.colorbar(cf, ax=ax, pad=0.015, shrink=0.85)
cb.set_label("T − T∞ (K)")
ax.set_title(
    "Heated cylinder, Re 40: temperature, isotherms and streamlines", fontsize=10
)
fig.tight_layout()
fig.savefig(sys.argv[2])

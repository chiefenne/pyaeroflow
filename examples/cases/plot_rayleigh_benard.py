"""Temperature, isotherms and streamlines of Rayleigh-Benard convection (gallery 10).

Usage: python plot_rayleigh_benard.py FILE.vtu out.png [periods]
Needs matplotlib and meshio (pip install matplotlib meshio). The periodic box
is drawn ``periods`` times side by side (default 2); the wall temperatures
come from the wall files next to the .vtu.
"""

import sys
from pathlib import Path

import matplotlib
import meshio
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation

vtu = Path(sys.argv[1])
periods = int(sys.argv[3]) if len(sys.argv) > 3 else 2
m = meshio.read(vtu)
q = m.cells[0].data
c = m.points[q][:, :, :2].mean(1)
height = m.points[:, 1].max()
width = m.points[:, 0].max()
c = c / height
width /= height
T = m.cell_data["temperature"][0][:, 0]
V = m.cell_data["velocity"][0][:, :2]
walls = {}
for name in ("bottom", "top"):
    data = np.genfromtxt(
        vtu.with_name(f"{vtu.stem}.wall_{name}.csv"), delimiter=",", names=True
    )
    walls[name] = data["T"].mean()
t_hot, t_cold = walls["bottom"], walls["top"]
t_mid = 0.5 * (t_hot + t_cold)

# periodic copies of the cells, plus the wall values on y = 0 and y = 1
px, py, pt, pu, pv = [], [], [], [], []
for k in range(-1, periods + 1):
    px.append(c[:, 0] + k * width)
    py.append(c[:, 1])
    pt.append(T)
    pu.append(V[:, 0])
    pv.append(V[:, 1])
    for y, tw in ((0.0, t_hot), (1.0, t_cold)):
        px.append(c[:, 0] + k * width)
        py.append(np.full(len(c), y))
        pt.append(np.full(len(c), tw))
        pu.append(np.zeros(len(c)))
        pv.append(np.zeros(len(c)))
px, py, pt, pu, pv = (np.concatenate(a) for a in (px, py, pt, pu, pv))
tri = Triangulation(px, py)

fig, ax = plt.subplots(figsize=(9.5, 3.0), dpi=200)
lv = np.linspace(t_cold, t_hot, 41)
cf = ax.tricontourf(tri, np.clip(pt, t_cold, t_hot), levels=lv, cmap="magma")
ax.tricontour(
    tri,
    pt,
    levels=np.linspace(t_cold, t_hot, 13)[1:-1],
    colors="white",
    linewidths=0.4,
    alpha=0.6,
)
gx, gy = np.meshgrid(
    np.linspace(0.0, periods * width, 80 * periods),
    np.linspace(0.0, 1.0, 80),
)
u = LinearTriInterpolator(tri, pu)(gx, gy)
v = LinearTriInterpolator(tri, pv)(gx, gy)
ax.streamplot(
    gx,
    gy,
    np.ma.filled(u, 0),
    np.ma.filled(v, 0),
    color=(1, 1, 1, 0.45),
    linewidth=0.6,
    density=1.3,
    arrowsize=0.6,
)
ax.set_xlim(0.0, periods * width)
ax.set_ylim(0.0, 1.0)
ax.set_aspect("equal")
ax.set_xlabel("x / h")
ax.set_ylabel("y / h")
cb = fig.colorbar(
    cf, ax=ax, pad=0.015, shrink=0.9, ticks=np.linspace(t_cold, t_hot, 7), format="%.1f"
)
cb.set_label("T (K)")
ax.set_title(
    f"Rayleigh-Benard convection, hot plate {t_hot - t_mid:+.1f} K, "
    f"cold plate {t_cold - t_mid:+.1f} K around {t_mid:.0f} K",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(sys.argv[2])

"""CO2 jet in an H2-N2 crossflow (gallery 11): CO2 and differential diffusion.

Usage: python plot_jet_species.py FILE.vtu out.png [x_max]
Needs matplotlib and meshio (pip install matplotlib meshio). Lengths in slot
widths. Top: CO2 mass fraction with streamlines. Bottom: the H2 that has
diffused relative to the flow, Y_H2 - Y_H2,inflow (1 - Y_CO2 / Y_CO2,jet): zero
if H2 were carried along with the jet fluid like a passive tracer, negative
where the fast H2 has left the jet's edge, positive where the light gas has
entered the jet fluid.
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
x_max = float(sys.argv[3]) if len(sys.argv) > 3 else 22.0
m = meshio.read(vtu)
d = 1.0  # slot width of the case file (mesh unit)
c = m.points[m.cells[0].data][:, :, :2].mean(1) / d
y_h2 = m.cell_data["mass_fraction_1"][0][:, 0]
y_co2 = m.cell_data["mass_fraction_2"][0][:, 0]
v = m.cell_data["velocity"][0][:, :2]
tri = Triangulation(c[:, 0], c[:, 1])
# drop the long thin triangles of the mesh corners and below the plate
xy = c[tri.triangles]
span = xy.max(1) - xy.min(1)
tri.set_mask((span[:, 0] > 3.0) | (span[:, 1] > 3.0))

fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), dpi=150, sharex=True)
y_top = 6.0
gx, gy = np.meshgrid(np.linspace(-4, x_max, 160), np.linspace(0.05, y_top, 60))
u = np.ma.filled(LinearTriInterpolator(tri, v[:, 0])(gx, gy), 0)
w = np.ma.filled(LinearTriInterpolator(tri, v[:, 1])(gx, gy), 0)
h2_in, co2_jet = 0.03, 0.5  # the case file: crossflow H2, jet CO2
excess = y_h2 - h2_in * (1.0 - y_co2 / co2_jet)
lim = float(np.percentile(np.abs(excess), 99.5))
panels = (
    (y_co2, "CO$_2$ mass fraction (jet gas) and streamlines", "inferno", None),
    (
        excess,
        "H$_2$ relative to a passive tracer (differential diffusion)",
        "RdBu_r",
        lim,
    ),
)
for ax, (field, title, cmap, sym) in zip(axes, panels, strict=True):
    if sym is None:
        levels = np.linspace(0, max(float(field.max()), 1e-3), 41)
    else:
        levels = np.linspace(-sym, sym, 41)
    cf = ax.tricontourf(
        tri, np.clip(field, levels[0], levels[-1]), levels=levels, cmap=cmap
    )
    ax.tricontour(
        tri,
        field,
        levels=levels[4:-3:4],
        colors="k" if sym else "w",
        linewidths=0.3,
        alpha=0.35,
    )
    ax.streamplot(
        gx,
        gy,
        u,
        w,
        color=(0, 0, 0, 0.35) if sym else (1, 1, 1, 0.5),
        linewidth=0.5,
        density=1.6,
        arrowsize=0.6,
    )
    ax.fill_between([-4, x_max], -1, 0, color="0.15")
    ax.fill_between([-0.5, 0.5], -1, 0, color="tab:red")
    ax.set_xlim(-4, x_max)
    ax.set_ylim(-0.4, y_top)
    ax.set_aspect("equal")
    ax.set_ylabel("y / d")
    ax.set_title(title, fontsize=10, loc="left")
    fig.colorbar(cf, ax=ax, pad=0.01, fraction=0.025)
axes[-1].set_xlabel("x / d")
fig.tight_layout()
fig.savefig(sys.argv[2])
print("wrote", sys.argv[2])

"""T3 flat-plate mesh for the Langtry-Menter runs: inlet 0.1 m upstream, plate 1.6 m.

Usage: python examples/cases/make_plate_lm.py [nx_plate ny out.su2]
(default 120 x 56 cells, plate_lm_120x56.su2, used by lm_t3a/lm_t3b/lm_t3am.toml)
"""

import sys

import numpy as np

from pyaeroflow.mesh import geometric_growth, split_boundary, structured
from pyaeroflow.mesh.su2 import write_su2

nxp = int(sys.argv[1]) if len(sys.argv) > 1 else 120
ny = int(sys.argv[2]) if len(sys.argv) > 2 else 56
out = sys.argv[3] if len(sys.argv) > 3 else f"plate_lm_{nxp}x{ny}.su2"
L, H, x0 = 1.6, 0.5, -0.1
y1 = 1.0e-5  # y+ < 1 up to Re/m 1.3e6 (T3A-)
r = geometric_growth(H, y1, ny)
ys = np.concatenate(([0.0], np.cumsum(y1 * r ** np.arange(ny))))
ys[-1] = H
# plate: dx 2 mm at the leading edge, growing geometrically to L
dx0 = 2.0e-3
g = geometric_growth(L, dx0, nxp)
xp = np.concatenate(([0.0], np.cumsum(dx0 * g ** np.arange(nxp))))
xp[-1] = L
# upstream: 12 cells, geometric toward the leading edge
nxf = 12
gf = geometric_growth(-x0, dx0, nxf)
xf = -np.concatenate(([0.0], np.cumsum(dx0 * gf ** np.arange(nxf))))[::-1]
xf[0] = x0
xs = np.concatenate((xf[:-1], xp))
m = structured(xs, ys)
m = split_boundary(
    m, "bottom", {"symmetry": lambda x, y: x < 0.0, "plate": lambda x, y: x >= 0.0}
)
write_su2(out, m)
print(m.summary())
print("plate growth", g, "dx max", np.diff(xp).max(), "y growth", r)

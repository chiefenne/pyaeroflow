"""T3 flat-plate mesh for the Langtry-Menter runs: inlet 0.1 m upstream, plate 1.6 m.

Thin wrapper of ``pyaeroflow.mesh.library.plate_lm_mesh``; the same mesh comes
from ``pyaeroflow make-mesh plate-lm``.

Usage: python examples/cases/make_plate_lm.py [nx_plate ny out.su2]
(default 120 x 56 cells, plate_lm_120x56.su2, used by lm_t3a/lm_t3b/lm_t3am.toml)
"""

import sys

from pyaeroflow.mesh.library import plate_lm_mesh
from pyaeroflow.mesh.su2 import write_su2

nxp = int(sys.argv[1]) if len(sys.argv) > 1 else 120
ny = int(sys.argv[2]) if len(sys.argv) > 2 else 56
out = sys.argv[3] if len(sys.argv) > 3 else f"plate_lm_{nxp}x{ny}.su2"
mesh = plate_lm_mesh(nxp, ny)
write_su2(out, mesh)
print(mesh.summary())

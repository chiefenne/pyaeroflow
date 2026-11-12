"""O-grid around the NACA 0012 for laminar vortex shedding (Re = 1000).

Thin wrapper of ``pyaeroflow.mesh.library.naca0012_shedding_mesh``; the same
mesh comes from ``pyaeroflow make-mesh naca0012-shedding``.

Usage: python make_naca0012_shedding_mesh.py [OUT.su2] [n_surface]
"""

import sys

from pyaeroflow.mesh import write_su2
from pyaeroflow.mesh.library import naca0012_shedding_mesh

out = sys.argv[1] if len(sys.argv) > 1 else "naca0012_shedding.su2"
n_surface = int(sys.argv[2]) if len(sys.argv) > 2 else 384
mesh = naca0012_shedding_mesh(n_surface)
write_su2(out, mesh)
print(f"{mesh.n_cells} cells, {n_surface} surface nodes")

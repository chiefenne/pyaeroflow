"""Optional matplotlib helpers (requirement D46).

matplotlib is imported only when a plotting function is called.
"""

from pyaeroflow.plotting.mesh import plot_mesh

__all__ = ["plot_mesh"]

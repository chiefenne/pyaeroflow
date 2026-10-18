"""Mesh data structure, generators and readers."""

from pyaeroflow.mesh.airfoil import (
    airfoil_ogrid,
    airfoil_points,
    naca0012,
    naca0012_ogrid,
)
from pyaeroflow.mesh.cylinder import annulus, cylinder_in_channel, split_boundary
from pyaeroflow.mesh.generators import geometric_growth, rectangle, structured
from pyaeroflow.mesh.mesh import Mesh, MeshError
from pyaeroflow.mesh.su2 import read_su2, write_su2

__all__ = [
    "Mesh",
    "MeshError",
    "airfoil_ogrid",
    "airfoil_points",
    "annulus",
    "cylinder_in_channel",
    "geometric_growth",
    "naca0012",
    "naca0012_ogrid",
    "read_su2",
    "rectangle",
    "split_boundary",
    "structured",
    "write_su2",
]

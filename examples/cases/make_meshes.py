"""Generate the .su2 meshes of the example cases (not committed, D44).

python examples/cases/make_meshes.py   (or: pyaeroflow make-mesh NAME)
"""

from pathlib import Path

from pyaeroflow.mesh import write_su2
from pyaeroflow.mesh.library import build

HERE = Path(__file__).parent


def main():
    """Write the example meshes next to the case files."""
    meshes = {
        "channel_quad_32x16.su2": build("channel-quad"),
        "channel_tri_32x16.su2": build("channel-tri"),
        "cavity_quad_64.su2": build("cavity"),
        "cylinder_ogrid_128x48.su2": build("cylinder-ogrid"),
        "plate_rans_96x64.su2": build("plate-rans", reynolds=5.0e6),
        "plate_transition_96x64.su2": build("plate-rans", reynolds=5.0e5),
        # Compressible Euler (L1): NACA 0012 O-grid, farfield at 100 chords,
        # and the Mach 2 compression ramp of 10 degrees.
        "naca0012_euler_ogrid_192x64.su2": build("naca0012-euler-ogrid"),
        "ramp_10deg_96x64.su2": build("ramp", nx=96, ny=64),
        "ramp_10deg_192x128.su2": build("ramp", nx=192, ny=128),
        "ramp_10deg_384x256.su2": build("ramp", nx=384, ny=256),
    }
    for name, mesh in meshes.items():
        write_su2(HERE / name, mesh)
        print(f"{name}: {mesh.n_cells} cells")


if __name__ == "__main__":
    main()

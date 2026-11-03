"""Generate the .su2 meshes of the example cases (not committed, D44).

python examples/cases/make_meshes.py
"""

from pathlib import Path

from pyaeroflow.cases.oblique_shock import ramp_mesh
from pyaeroflow.cases.tmr_flat_plate import plate_mesh
from pyaeroflow.mesh import annulus, rectangle, write_su2
from pyaeroflow.mesh.airfoil import airfoil_ogrid, airfoil_points

HERE = Path(__file__).parent


def main():
    """Write the example meshes next to the case files."""
    meshes = {
        "channel_quad_32x16.su2": rectangle(
            32,
            16,
            x=(0.0, 2.0),
            y=(0.0, 1.0),
            cell_type="quad",
            cluster_y="both",
            beta=1.5,
        ),
        "channel_tri_32x16.su2": rectangle(
            32,
            16,
            x=(0.0, 2.0),
            y=(0.0, 1.0),
            cell_type="triangle",
            perturb=0.3,
            seed=1,
        ),
        "cavity_quad_64.su2": rectangle(64, 64, cell_type="quad"),
        "cylinder_ogrid_128x48.su2": annulus(
            128, 48, radius=0.5, outer_radius=10.0, growth=1.0 + 6.0 / 48
        ),
        "plate_rans_96x64.su2": plate_mesh(96, 24, 64, reynolds=5.0e6),
        "plate_transition_96x64.su2": plate_mesh(96, 24, 64, reynolds=5.0e5),
        # Compressible Euler (L1): NACA 0012 O-grid, farfield at 100 chords,
        # and the Mach 2 compression ramp of 10 degrees.
        "naca0012_euler_ogrid_192x64.su2": airfoil_ogrid(
            airfoil_points(192, cut=1.0),
            64,
            first_spacing=1.0e-3,
            farfield_radius=100.0,
        ),
        "ramp_10deg_96x64.su2": ramp_mesh(96, 64, angle=10.0),
    }
    for name, mesh in meshes.items():
        write_su2(HERE / name, mesh)
        print(f"{name}: {mesh.n_cells} cells")


if __name__ == "__main__":
    main()

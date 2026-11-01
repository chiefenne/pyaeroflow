"""Generate the .su2 meshes of the example cases (not committed, D44).

python examples/cases/make_meshes.py
"""

from pathlib import Path

from pyaeroflow.cases.tmr_flat_plate import plate_mesh
from pyaeroflow.mesh import annulus, rectangle, write_su2

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
    }
    for name, mesh in meshes.items():
        write_su2(HERE / name, mesh)
        print(f"{name}: {mesh.n_cells} cells")


if __name__ == "__main__":
    main()

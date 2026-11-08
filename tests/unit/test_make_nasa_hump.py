"""examples/meshes/make_nasa_hump.py: PLOT3D hump grid to markers and a slot."""

import importlib.util
from pathlib import Path

import numpy as np

MESHES = Path(__file__).resolve().parents[2] / "examples" / "meshes"
SCRIPT = MESHES / "make_nasa_hump.py"
spec = importlib.util.spec_from_file_location("make_nasa_hump", SCRIPT)
hump = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hump)


def test_hump_mesh_markers_and_slot(tmp_path):
    xs, ys = np.linspace(-1.0, 2.0, 31), np.linspace(0.0, 1.0, 6)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    y = y + 0.1 * np.exp(-(((x - 0.5) / 0.3) ** 2)) * (1.0 - y)
    p2d = tmp_path / "g.p2dfmt"
    p2d.write_text(
        f"1\n{xs.size} {ys.size}\n"
        + " ".join(f"{v:.12e}" for v in x.T.ravel())
        + "\n"
        + " ".join(f"{v:.12e}" for v in y.T.ravel())
        + "\n"
    )
    mesh = hump.hump_mesh(*hump.read_p2d(p2d))
    sizes = {k: v.size for k, v in mesh.boundary_faces.items()}
    assert sizes == {"wall": 30, "top": 30, "inlet": 5, "outlet": 5}
    assert np.all(mesh.cell_area > 0)
    out = tmp_path / "g.su2"
    hump.main(["make_nasa_hump.py", str(p2d), str(out), "0.39", "0.61"])
    from pyaeroflow.mesh import read_su2

    m = read_su2(out)
    assert m.boundary_faces["slot"].size == 2
    assert m.boundary_faces["wall"].size == 28

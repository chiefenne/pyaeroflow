"""Mesh library: every entry builds, and the CLI writes a readable file."""

import pytest

from pyaeroflow.cli import main
from pyaeroflow.mesh import read_su2
from pyaeroflow.mesh.library import LIBRARY, build

# small parameters keep the test fast
SMALL = {
    "channel-quad": {},
    "channel-tri": {},
    "cavity": {"n": 8},
    "cylinder-ogrid": {"n_surface": 32, "n_radial": 8},
    "plate-rans": {"nx_plate": 16, "ny": 12},
    "plate-lm": {"nx_plate": 20, "ny": 12},
    "ramp": {"nx": 16, "ny": 12},
    "naca0012-ogrid": {"n_surface": 64, "n_radial": 64},
    "naca0012-euler-ogrid": {"n_surface": 48, "n_radial": 40},
    "naca0012-shedding": {"n_surface": 64},
    "jet-crossflow": {"ny": 20},
}


def test_every_library_mesh_is_tested():
    assert set(SMALL) == set(LIBRARY)


@pytest.mark.parametrize("name", sorted(SMALL))
def test_library_mesh_builds(name):
    mesh = build(name, **SMALL[name])
    assert mesh.n_cells > 0 and mesh.boundary_faces


def test_jet_crossflow_has_the_slot_marker():
    mesh = build("jet-crossflow", n_slot=8, ny=20)
    assert mesh.boundary_faces["slot"].size == 8
    assert {"plate", "inlet", "outlet", "top"} <= set(mesh.boundary_faces)


def test_make_mesh_command(tmp_path, capsys):
    out = tmp_path / "plate.su2"
    code = main(
        ["make-mesh", "plate-lm", "--set", "nx_plate=20", "ny=12", "-o", str(out)]
    )
    assert code == 0 and "cells" in capsys.readouterr().out
    assert read_su2(out).n_cells == (20 + 12) * 12
    assert main(["make-mesh", "--list"]) == 0
    assert "jet-crossflow" in capsys.readouterr().out
    assert main(["make-mesh", "nope"]) != 0

"""The example cases and meshes of the repository are consistent."""

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CASES = sorted((ROOT / "examples" / "cases").glob("*.toml"))
MESHES = ROOT / "examples" / "meshes"
# Meshes that make_meshes.py and make_plate_lm.py generate (not committed).
GENERATED = {
    "cavity_quad_64.su2",
    "channel_quad_32x16.su2",
    "channel_tri_32x16.su2",
    "cylinder_ogrid_128x48.su2",
    "plate_rans_96x64.su2",
    "plate_transition_96x64.su2",
    "plate_lm_120x56.su2",
}


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_example_case_mesh_exists_and_restart_is_not_a_private_path(case):
    data = tomllib.loads(case.read_text())
    mesh = Path(data["mesh"]["file"])
    if mesh.name in GENERATED:
        assert mesh.parent == Path("."), (
            f"{case.name}: generated meshes sit next to the case"
        )
    else:
        assert (case.parent / mesh).exists(), f"{case.name}: {mesh} is not committed"
    assert "restart" not in data.get("initial", {}), (
        f"{case.name} restarts from a file that is not in the repository"
    )


def test_every_committed_mesh_is_described():
    readme = (MESHES / "README.md").read_text()
    for mesh in MESHES.glob("*.su2"):
        assert mesh.name in readme, (
            f"{mesh.name} is missing in examples/meshes/README.md"
        )

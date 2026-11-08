"""examples/meshes/split_marker.py: a slot marker for a transpiration wall."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyaeroflow.mesh import rectangle

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "meshes" / "split_marker.py"
spec = importlib.util.spec_from_file_location("split_marker", SCRIPT)
split_marker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_marker)


def test_split_moves_only_the_faces_in_range():
    mesh = rectangle(10, 4, x=(0.0, 1.0), y=(-0.5, 0.5), cell_type="quad")
    new = split_marker.split_marker(mesh, "top", "slot", 0.25, 0.55, side="upper")
    x = new.face_centre[new.boundary_faces["slot"], 0]
    np.testing.assert_allclose(np.sort(x), [0.25, 0.35, 0.45, 0.55])
    assert new.boundary_faces["top"].size == 6
    assert new.boundary_faces["bottom"].size == 10


def test_split_rejects_an_empty_range_and_the_wrong_side():
    mesh = rectangle(10, 4, x=(0.0, 1.0), y=(-0.5, 0.5), cell_type="quad")
    with pytest.raises(ValueError, match="no faces"):
        split_marker.split_marker(mesh, "top", "slot", 0.25, 0.55, side="lower")
    with pytest.raises(ValueError, match="no marker"):
        split_marker.split_marker(mesh, "airfoil", "slot", 0.0, 1.0)

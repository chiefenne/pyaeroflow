"""Reader for SU2 native meshes."""

import numpy as np
import pytest

from pyaeroflow.mesh import MeshError, read_su2

# Same layout as PyAero's writer: NPOIN before NELEM, trailing indices.
MIXED_MESH = """\
%
% Problem dimension
%
NDIME= 2
%
% Node coordinates
%
NPOIN= 6
 0.00000000e+00  0.00000000e+00 0
 1.00000000e+00  0.00000000e+00 1
 2.00000000e+00  0.00000000e+00 2
 0.00000000e+00  1.00000000e+00 3
 1.00000000e+00  1.00000000e+00 4
 2.00000000e+00  1.00000000e+00 5
%
% Element connectivity
%
NELEM= 3
9 0 1 4 3 0
5 1 2 5 1
5 1 5 4 2
%
% Boundary tags
%
NMARK= 4
MARKER_TAG= bottom
MARKER_ELEMS= 2
3 0 1
3 1 2
MARKER_TAG= outlet
MARKER_ELEMS= 1
3 2 5
MARKER_TAG= top
MARKER_ELEMS= 2
3 5 4
3 4 3
MARKER_TAG= inlet
MARKER_ELEMS= 1
3 3 0
"""


def write(tmp_path, text, name="mesh.su2"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_read_mixed_mesh(tmp_path):
    mesh = read_su2(write(tmp_path, MIXED_MESH))
    assert mesh.n_nodes == 6 and mesh.n_cells == 3
    np.testing.assert_array_equal(np.bincount(mesh.cell_n_nodes), [0, 0, 0, 2, 1])
    assert mesh.cell_area.sum() == pytest.approx(2.0)
    assert {k: v.size for k, v in mesh.boundary_faces.items()} == {
        "bottom": 2,
        "outlet": 1,
        "top": 2,
        "inlet": 1,
    }
    outlet = mesh.boundary_faces["outlet"]
    np.testing.assert_allclose(mesh.face_normal[outlet], [[1.0, 0.0]])


def test_scale(tmp_path):
    mesh = read_su2(write(tmp_path, MIXED_MESH), scale=0.001)
    assert mesh.cell_area.sum() == pytest.approx(2.0e-6)


def test_keyword_order_and_no_indices(tmp_path):
    text = """\
NDIME= 2
NELEM= 1
9 0 1 2 3
NMARK= 1
MARKER_TAG= wall
MARKER_ELEMS= 4
3 0 1
3 1 2
3 2 3
3 3 0
NPOIN= 4
0 0
1 0
1 1
0 1
"""
    mesh = read_su2(write(tmp_path, text))
    assert mesh.n_cells == 1 and mesh.boundary_faces["wall"].size == 4


def test_missing_marker_is_reported(tmp_path):
    text = MIXED_MESH.replace("NMARK= 4", "NMARK= 3").split("MARKER_TAG= inlet")[0]
    with pytest.raises(MeshError, match="belong to no set"):
        read_su2(write(tmp_path, text))


def test_3d_rejected(tmp_path):
    with pytest.raises(MeshError, match="only 2D"):
        read_su2(write(tmp_path, MIXED_MESH.replace("NDIME= 2", "NDIME= 3")))

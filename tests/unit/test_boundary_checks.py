"""Face-level boundary checks (requirement D39, R1)."""

import numpy as np
import pytest

from pyaeroflow.mesh import Mesh, MeshError

# 2 x 1 quads on [0, 2] x [0, 1]; nodes 0-2 bottom row, 3-5 top row.
NODES = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
CELLS = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
BOUNDARY = {
    "bottom": [[0, 1], [1, 2]],
    "right": [[2, 5]],
    "top": [[5, 4], [4, 3]],
    "left": [[3, 0]],
}


def test_valid_sets():
    mesh = Mesh.from_cells(NODES, CELLS, BOUNDARY)
    assert mesh.n_faces == 7
    assert sorted(mesh.boundary_faces) == ["bottom", "left", "right", "top"]
    assert np.count_nonzero(mesh.neighbour < 0) == 6


def test_unassigned_face():
    boundary = dict(BOUNDARY, top=[[5, 4]])
    with pytest.raises(MeshError, match="1 boundary faces belong to no set") as err:
        Mesh.from_cells(NODES, CELLS, boundary)
    assert "(0.5, 1)" in str(err.value)  # location of the missing face


def test_face_in_two_sets():
    boundary = dict(BOUNDARY, wall=[[0, 1]])
    with pytest.raises(MeshError, match="1 faces belong to more than one set"):
        Mesh.from_cells(NODES, CELLS, boundary)


def test_interior_face_in_set():
    boundary = dict(BOUNDARY, cut=[[1, 4]])
    with pytest.raises(MeshError, match="'cut' contains 1 interior faces"):
        Mesh.from_cells(NODES, CELLS, boundary)


def test_pair_that_is_not_a_face():
    boundary = dict(BOUNDARY, bogus=[[0, 5]])
    with pytest.raises(MeshError, match="not mesh faces"):
        Mesh.from_cells(NODES, CELLS, boundary)


def test_clockwise_cells_are_reoriented():
    mesh = Mesh.from_cells(NODES, CELLS[:, ::-1], BOUNDARY)
    np.testing.assert_allclose(mesh.cell_area, [1.0, 1.0])


def test_non_manifold_edge():
    nodes = np.vstack((NODES, [[0.5, 2.0]]))
    cells = [[0, 1, 4, 3], [1, 2, 5, 4], [1, 4, 6]]
    with pytest.raises(MeshError, match="more than two cells"):
        Mesh.from_cells(nodes, cells, {})


def test_folded_cell_is_rejected():
    """A bow-tie quad (nodes listed 1-4-0-3, edges 4-0 and 3-1 cross) next to
    a valid one; they share the face 1-4."""
    nodes = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1.2, 1.5], [2, 1]], dtype=float)
    cells = [[1, 4, 0, 3], [1, 2, 5, 4]]
    boundary = {
        "bottom": [[1, 2]],
        "right": [[2, 5]],
        "top": [[5, 4]],
        "left": [[0, 3]],
        "diagonals": [[4, 0], [3, 1]],
    }
    with pytest.raises(MeshError, match="self-intersecting or folded") as err:
        Mesh.from_cells(nodes, cells, boundary)
    assert "(0, 0.5)" in str(err.value)  # the inverted edge 0-3 of the bow-tie

"""Geometry and topology of generated rectangle meshes (all cell types)."""

import numpy as np
import pytest

from pyaeroflow.mesh import rectangle

X, Y = (-0.5, 1.5), (0.0, 0.75)
CASES = [
    (cell_type, perturb, cluster)
    for cell_type in ("quad", "triangle", "polygon", "mixed")
    for perturb, cluster in ((0.0, "none"), (0.4, "none"), (0.3, "both"))
]


def face_sign_sum(mesh, values):
    """Sum face values into cells: +value for the owner, -value for the neighbour."""
    out = np.zeros((mesh.n_cells,) + values.shape[1:])
    np.add.at(out, mesh.owner, values)
    inner = mesh.neighbour >= 0
    np.add.at(out, mesh.neighbour[inner], -values[inner])
    return out


@pytest.fixture(params=CASES, ids=lambda c: f"{c[0]}-p{c[1]}-{c[2]}")
def mesh(request):
    cell_type, perturb, cluster = request.param
    return rectangle(
        8, 6, x=X, y=Y, cell_type=cell_type, perturb=perturb, cluster_y=cluster, seed=3
    )


def test_cells_closed(mesh):
    """Sum of n dS over each cell vanishes."""
    flux = face_sign_sum(mesh, mesh.face_normal * mesh.face_length[:, None])
    assert np.abs(flux).max() < 1e-14


def test_total_area(mesh):
    area = (X[1] - X[0]) * (Y[1] - Y[0])
    assert np.all(mesh.cell_area > 0.0)
    assert mesh.cell_area.sum() == pytest.approx(area, rel=1e-13)


def test_normals_outward_via_divergence_theorem(mesh):
    """Closed integral of x n_x and y n_y gives the cell area (Gauss)."""
    w = mesh.face_length
    gx = face_sign_sum(mesh, mesh.face_centre[:, 0] * mesh.face_normal[:, 0] * w)
    gy = face_sign_sum(mesh, mesh.face_centre[:, 1] * mesh.face_normal[:, 1] * w)
    np.testing.assert_allclose(gx, mesh.cell_area, rtol=1e-12)
    np.testing.assert_allclose(gy, mesh.cell_area, rtol=1e-12)


def test_centroid_via_divergence_theorem(mesh):
    """Closed integral of (x^2/2) n_x dl equals area times centroid x (Simpson)."""
    a = mesh.nodes[mesh.face_nodes[:, 0], 0]
    b = mesh.nodes[mesh.face_nodes[:, 1], 0]
    m = mesh.face_centre[:, 0]
    x2 = (a**2 + 4.0 * m**2 + b**2) / 6.0
    first_moment = face_sign_sum(
        mesh, 0.5 * x2 * mesh.face_normal[:, 0] * mesh.face_length
    )
    np.testing.assert_allclose(
        first_moment, mesh.cell_area * mesh.cell_centroid[:, 0], rtol=1e-11, atol=1e-15
    )


def test_euler_formula(mesh):
    """Faces = nodes + cells - 1 for a mesh of a simply connected domain."""
    used_nodes = np.unique(mesh.cell_nodes).size
    assert mesh.n_faces == used_nodes + mesh.n_cells - 1


def test_boundary_sets(mesh):
    lengths = {
        name: mesh.face_length[faces].sum()
        for name, faces in mesh.boundary_faces.items()
    }
    assert lengths["bottom"] == pytest.approx(X[1] - X[0], rel=1e-13)
    assert lengths["top"] == pytest.approx(X[1] - X[0], rel=1e-13)
    assert lengths["left"] == pytest.approx(Y[1] - Y[0], rel=1e-13)
    assert lengths["right"] == pytest.approx(Y[1] - Y[0], rel=1e-13)
    expected = {"bottom": (0, -1), "right": (1, 0), "top": (0, 1), "left": (-1, 0)}
    for name, normal in expected.items():
        np.testing.assert_allclose(
            mesh.face_normal[mesh.boundary_faces[name]],
            np.broadcast_to(normal, (1, 2))
            * np.ones((mesh.boundary_faces[name].size, 1)),
            atol=1e-14,
        )


@pytest.mark.parametrize(
    ("cell_type", "n_cells"),
    [("quad", 48), ("triangle", 96), ("polygon", 63), ("mixed", 2 * 8 + 2 * 8 * 4)],
)
def test_cell_counts(cell_type, n_cells):
    mesh = rectangle(8, 6, cell_type=cell_type)
    assert mesh.n_cells == n_cells


def test_mixed_layers():
    mesh = rectangle(8, 6, cell_type="mixed", quad_layers=2)
    counts = np.bincount(mesh.cell_n_nodes)
    assert counts[4] == 2 * 2 * 8
    assert counts[3] == 2 * 2 * 8


@pytest.mark.parametrize("mode", ["bottom", "top", "both"])
def test_clustering(mode):
    mesh = rectangle(1, 20, cell_type="quad", cluster_y=mode, beta=2.5)
    ys = np.unique(np.round(mesh.nodes[:, 1], 14))
    h = np.diff(ys)
    assert ys[0] == 0.0 and ys[-1] == 1.0 and np.all(h > 0.0)
    if mode in ("bottom", "both"):
        assert h[0] < h[len(h) // 2]
    if mode in ("top", "both"):
        assert h[-1] < h[len(h) // 2]


def test_perturbation_reproducible():
    a = rectangle(6, 6, perturb=0.5, seed=1)
    b = rectangle(6, 6, perturb=0.5, seed=1)
    c = rectangle(6, 6, perturb=0.5, seed=2)
    np.testing.assert_array_equal(a.nodes, b.nodes)
    assert not np.array_equal(a.nodes, c.nodes)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        rectangle(4, 4, cell_type="hexagon")
    with pytest.raises(ValueError):
        rectangle(4, 4, perturb=0.9)
    with pytest.raises(ValueError):
        rectangle(4, 4, cluster_y="left")

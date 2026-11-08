"""Local refinement by cell splitting with hanging nodes (R1)."""

import numpy as np
import pytest

from pyaeroflow.mesh import MeshError, annulus, rectangle, structured
from pyaeroflow.mesh.refine import Refinement, inject, refine_cells
from pyaeroflow.numerics.gradients import lsq_stencil

X, Y = (0.0, 2.0), (0.0, 1.0)


def base(perturb=0.0, cell_type="quad"):
    return rectangle(6, 4, x=X, y=Y, cell_type=cell_type, perturb=perturb, seed=2)


def closure(mesh):
    """Largest |sum n dS| over the cells (0 for closed polygons)."""
    s = mesh.face_normal * mesh.face_length[:, None]
    out = np.zeros((mesh.n_cells, 2))
    np.add.at(out, mesh.owner, s)
    inner = mesh.neighbour >= 0
    np.add.at(out, mesh.neighbour[inner], -s[inner])
    return np.abs(out).max()


def check(mesh, area):
    assert closure(mesh) < 1e-13
    assert mesh.cell_area.sum() == pytest.approx(area, rel=1e-13)
    assert np.all(mesh.cell_area > 0)
    bnd = np.concatenate(list(mesh.boundary_faces.values()))
    assert np.array_equal(np.sort(bnd), np.flatnonzero(mesh.neighbour < 0))


def test_single_split_makes_hanging_nodes():
    m0 = base()
    centre = 9  # interior cell, row 1, column 3
    m1, source = refine_cells(m0, [centre], 2, 2)
    check(m1, 2.0)
    assert m1.n_cells == m0.n_cells + 3
    assert np.count_nonzero(source == centre) == 4
    # The four edge neighbours become pentagons, nothing else changes.
    assert np.bincount(m1.cell_n_nodes).tolist()[4:] == [m0.n_cells - 1 - 4 + 4, 4]
    assert m1.n_nodes == m0.n_nodes + 5


@pytest.mark.parametrize("nu,nv", [(1, 4), (3, 1), (3, 2), (5, 5)])
def test_anisotropic_split(nu, nv):
    m0 = base(perturb=0.3)
    ref = Refinement.from_mesh(m0)
    source = ref.split([0, 9, 23], nu, nv, max_hanging=None)
    check(ref.mesh, 2.0)
    assert ref.n_cells == m0.n_cells + 3 * (nu * nv - 1)
    for c in (0, 9, 23):
        assert np.count_nonzero(source == c) == nu * nv
    children = np.flatnonzero(source == 9)
    np.testing.assert_allclose(
        ref.mesh.cell_area[children].sum(), m0.cell_area[9], rtol=1e-13
    )
    assert np.all(ref.split_factors[children] == (nu, nv))


def test_boundary_faces_follow_split():
    m0 = base()
    m1, _ = refine_cells(m0, [0], 3, 3)  # corner cell: bottom and left sides
    for name in m0.boundary_faces:
        total0 = m0.face_length[m0.boundary_faces[name]].sum()
        total1 = m1.face_length[m1.boundary_faces[name]].sum()
        assert total1 == pytest.approx(total0, rel=1e-14)
    assert m1.boundary_faces["bottom"].size == m0.boundary_faces["bottom"].size + 2
    assert m1.boundary_faces["left"].size == m0.boundary_faces["left"].size + 2


def test_children_inherit_direction_and_reuse_nodes():
    """A 1x2 split and then a 2x1 split of both children equal one 2x2 split."""
    m0 = base(perturb=0.3)
    a = Refinement.from_mesh(m0)
    a.split([9], 1, 2)
    kids = np.flatnonzero(a.origin == 9)
    a.split(kids, 2, 1)
    b = Refinement.from_mesh(m0)
    b.split([9], 2, 2)
    assert a.mesh.n_nodes == b.mesh.n_nodes
    np.testing.assert_allclose(np.sort(a.mesh.cell_area), np.sort(b.mesh.cell_area))
    assert np.all(a.split_factors[a.origin == 9] == (2, 2))


def test_balance_limits_hanging_nodes():
    m0 = base()
    ref = Refinement.from_mesh(m0)
    ref.split([9], 2, 2)
    # Refine one child of cell 9 twice more: neighbours must follow.
    for _ in range(2):
        child = int(np.flatnonzero(ref.origin == 9)[0])
        ref.split([child], 2, 2, max_hanging=1)
        assert ref.hanging_counts().max() <= 1
    check(ref.mesh, 2.0)
    # Without balancing the coarse neighbours collect 3 hanging nodes.
    raw = Refinement.from_mesh(m0)
    raw.split([9], 2, 2, max_hanging=None)
    for _ in range(2):
        child = int(np.flatnonzero(raw.origin == 9)[0])
        raw.split([child], 2, 2, max_hanging=None)
    assert raw.hanging_counts().max() == 3
    assert raw.n_cells < ref.n_cells


def test_one_by_n_balance_splits_neighbour_along_side():
    """A 1x4 split leaves 3 hanging nodes; max_hanging=1 splits the neighbour 2x."""
    m0 = base()
    ref = Refinement.from_mesh(m0)
    source = ref.split([9], 4, 1, max_hanging=1)
    assert ref.hanging_counts().max() <= 1
    # Cells above and below (v-neighbours) are split along u only.
    for nb in (3, 15):
        assert np.all(ref.split_factors[source == nb] == (2, 1))


def test_replay_reproduces_mesh():
    m0 = base(perturb=0.2)
    ref = Refinement.from_mesh(m0)
    ref.split([2, 9, 10], [2, 1, 3], [2, 4, 1])
    ref.split(np.flatnonzero(ref.origin == 9)[:2], 2, 2)
    again = Refinement.replay(m0, ref.log)
    np.testing.assert_array_equal(again.mesh.nodes, ref.mesh.nodes)
    np.testing.assert_array_equal(again.mesh.cell_nodes, ref.mesh.cell_nodes)


def test_uniform_split_matches_fine_mesh():
    m0 = rectangle(3, 2, x=X, y=Y)
    m1, _ = refine_cells(m0, np.arange(m0.n_cells), 2, 2)
    fine = rectangle(6, 4, x=X, y=Y)
    assert m1.n_nodes == fine.n_nodes
    assert np.all(m1.cell_n_nodes == 4)
    np.testing.assert_allclose(
        np.sort(m1.cell_centroid, axis=0),
        np.sort(fine.cell_centroid, axis=0),
        atol=1e-14,
    )


def test_lsq_gradient_exact_for_linear_field_on_hanging_mesh():
    m0 = base(perturb=0.3)
    ref = Refinement.from_mesh(m0)
    ref.split([8, 9, 14], 2, 2)
    ref.split(np.flatnonzero(ref.origin == 9)[:1], 1, 3)
    mesh = ref.mesh
    assert mesh.cell_n_nodes.max() > 4
    bfaces = np.flatnonzero(mesh.neighbour < 0)
    cell, src, coeff = lsq_stencil(mesh, bfaces)
    g = np.array([0.7, -1.3])
    phi = np.concatenate((mesh.cell_centroid @ g, mesh.face_centre[bfaces] @ g))
    grad = np.zeros((mesh.n_cells, 2))
    np.add.at(grad, cell, coeff[:, 0, :] * (phi[src] - phi[cell])[:, None])
    np.testing.assert_allclose(grad, np.tile(g, (mesh.n_cells, 1)), atol=1e-12)


def test_inject_copies_parent_values():
    m0 = base()
    values = np.arange(m0.n_cells, dtype=float)[:, None] * [1.0, 2.0]
    _, source = refine_cells(m0, [4, 9], 2, 3, max_hanging=None)
    out = inject(values, source)
    assert out.shape == (m0.n_cells + 10, 2)
    np.testing.assert_array_equal(out[source == 9], np.tile([9.0, 18.0], (6, 1)))


def test_triangles_are_kept_but_not_split():
    m0 = rectangle(4, 3, x=X, y=Y, cell_type="mixed", quad_layers=1)
    ref = Refinement.from_mesh(m0)
    tri = int(np.flatnonzero(m0.cell_n_nodes == 3)[0])
    with pytest.raises(MeshError, match="only quadrilaterals"):
        ref.split([tri], 2, 2)
    quad = int(np.flatnonzero(m0.cell_n_nodes == 4)[0])
    ref = Refinement.from_mesh(m0)
    ref.split([quad], 2, 2)
    check(ref.mesh, 2.0)


def test_extents_give_aspect_ratio():
    m0 = rectangle(2, 2, x=(0.0, 4.0), y=(0.0, 1.0))
    ref = Refinement.from_mesh(m0)
    np.testing.assert_allclose(ref.extents(), np.tile([2.0, 0.5], (4, 1)))
    ref.split([0], 4, 1)
    np.testing.assert_allclose(
        ref.extents()[ref.origin == 0], np.tile([0.5, 0.5], (4, 1))
    )


# -- R2: wall curves, boundary-layer columns, displacement ------------------


def ring(growth=1.3, n_radial=40):
    """O-grid around a unit-diameter cylinder with very thin wall cells."""
    return annulus(48, n_radial, radius=0.5, outer_radius=5.0, growth=growth)


def wall_radius_error(mesh, nodes):
    return np.abs(np.hypot(*mesh.nodes[nodes].T) - 0.5).max()


def test_new_wall_nodes_lie_on_curve():
    m0 = ring(growth=1.1, n_radial=10)
    wall = m0.boundary_faces["cylinder"]
    straight, _ = refine_cells(m0, m0.owner[wall], 2, 2, max_hanging=None)
    ref = Refinement.from_mesh(m0, walls=["cylinder"])
    ref.split(m0.owner[wall], 2, 2, max_hanging=None)
    new = np.arange(m0.n_nodes, ref.mesh.n_nodes)
    on_wall = np.intersect1d(
        new, ref.mesh.face_nodes[ref.mesh.boundary_faces["cylinder"]]
    )
    assert on_wall.size == wall.size
    sagitta = wall_radius_error(straight, on_wall)
    assert sagitta > 1e-3
    assert wall_radius_error(ref.mesh, on_wall) < 1e-2 * sagitta


def test_straight_wall_and_corner_stay_exact():
    xs = np.linspace(0.0, 1.0, 5)
    m0 = structured(xs, xs)
    ref = Refinement.from_mesh(m0, walls=["bottom", "right"])
    ref.split(np.arange(m0.n_cells), 2, 2)
    m = ref.mesh
    bottom = m.face_nodes[m.boundary_faces["bottom"]].ravel()
    right = m.face_nodes[m.boundary_faces["right"]].ravel()
    np.testing.assert_array_equal(m.nodes[bottom, 1], 0.0)
    np.testing.assert_allclose(m.nodes[right, 0], 1.0, atol=1e-15)
    np.testing.assert_allclose(m.cell_area.sum(), 1.0, rtol=1e-14)


def test_displacement_carried_up_prevents_folding():
    m0 = ring()
    wall = m0.boundary_faces["cylinder"]
    first = m0.cell_area[m0.owner[wall]].max() / m0.face_length[wall].min()
    cells = m0.owner[wall]
    # Sagitta (~1.7e-3) is far larger than the first cell height.
    assert first < 1e-4
    with pytest.raises(MeshError, match="folded"):
        ref = Refinement.from_mesh(m0, walls=["cylinder"], decay=1e-3)
        ref.split(cells, 2, 1, column_aspect=5.0)
        ref.mesh  # noqa: B018
    ref = Refinement.from_mesh(m0, walls=["cylinder"])
    ref.split(cells, *ref.along_long_side(cells, 2), column_aspect=5.0)
    check(ref.mesh, ref.mesh.cell_area.sum())


def test_column_rule_refines_thin_cells_above_the_wall():
    xs = np.linspace(0.0, 1.0, 5)
    ys = np.concatenate(([0.0], 1e-3 * 1.5 ** np.arange(14)))
    ys = np.cumsum(np.diff(ys, prepend=0.0))
    m0 = structured(xs, ys)
    ext = Refinement.from_mesh(m0).extents()
    thin = ext[:, 0] > 5.0 * ext[:, 1]
    ref = Refinement.from_mesh(m0, walls=["bottom"])
    source = ref.split([1], 3, 1, column_aspect=5.0)
    nx = xs.size - 1
    split = {int(s) for s in source if np.count_nonzero(source == s) == 3}
    expected = {1 + nx * j for j in range(ys.size - 1) if thin[1 + nx * j]}
    assert split == expected
    # Hanging nodes only where the column ends.
    h = ref.hanging_counts()
    top = max(expected) + nx
    assert h.max() <= 1 or set(np.flatnonzero(h.max(axis=1) > 1)) <= {top}
    check(ref.mesh, ref.mesh.cell_area.sum())


def test_replay_with_walls():
    m0 = ring(growth=1.2, n_radial=20)
    cells = m0.owner[m0.boundary_faces["cylinder"]][:10]
    ref = Refinement.from_mesh(m0, walls=["cylinder"])
    ref.split(cells, 2, 1, column_aspect=5.0)
    ref.split(np.flatnonzero(ref.origin == cells[0]), 2, 2)
    again = Refinement.replay(m0, ref.log, walls=["cylinder"])
    np.testing.assert_array_equal(again.mesh.nodes, ref.mesh.nodes)


def test_along_long_side():
    m0 = rectangle(2, 2, x=(0.0, 4.0), y=(0.0, 1.0))
    ref = Refinement.from_mesh(m0)
    nu, nv = ref.along_long_side([0, 1], 3)
    np.testing.assert_array_equal(nu, [3, 3])
    np.testing.assert_array_equal(nv, [1, 1])


def test_several_new_nodes_on_one_wall_face_do_not_fold():
    """A 4-way split puts three nodes on one wall face; each moves from the face."""
    m0 = ring(growth=1.3)
    cells = m0.owner[m0.boundary_faces["cylinder"]]
    ref = Refinement.from_mesh(m0, walls=["cylinder"])
    ref.split(cells, *ref.along_long_side(cells, 4), column_aspect=5.0)
    check(ref.mesh, ref.mesh.cell_area.sum())
    # The same wall node whether it comes from a 2-way or a 4-way split.
    two = Refinement.from_mesh(m0, walls=["cylinder"])
    two.split(cells, *two.along_long_side(cells, 2), column_aspect=5.0)
    wall = ref.mesh.face_nodes[ref.mesh.boundary_faces["cylinder"]].ravel()
    wall2 = two.mesh.face_nodes[two.mesh.boundary_faces["cylinder"]].ravel()
    p4 = {tuple(np.round(p, 12)) for p in ref.mesh.nodes[wall]}
    p2 = {tuple(np.round(p, 12)) for p in two.mesh.nodes[wall2]}
    assert p2 <= p4

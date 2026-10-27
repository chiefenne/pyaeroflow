"""Reader for Gmsh meshes (MSH 2.2, 4.0, 4.1; ASCII and binary)."""

import struct
from pathlib import Path

import numpy as np
import pytest
from test_run import COUETTE

from pyaeroflow.mesh import (
    MeshError,
    read_mesh,
    read_msh,
    read_su2,
    rectangle,
    write_su2,
)
from pyaeroflow.run import run_case

DATA = Path(__file__).parent.parent / "data" / "gmsh"
# Written by Gmsh 4.15.2: a 2 x 1 transfinite quad block (8 cells), physical
# groups walls (two curves), outlet, inlet and the surface "fluid".
QUAD = {
    "2.2 ascii": "quad_v2.2_ascii.msh",
    "2.2 binary": "quad_v2.2_bin.msh",
    "4.0 ascii": "quad_v4.0_ascii.msh",
    "4.1 ascii": "quad_v4.1_ascii.msh",
    "4.1 binary": "quad_v4.1_bin.msh",
}
# A channel with a hole, second-order triangles and quadrangles (types 9, 10).
MIXED = {
    "2.2 ascii": "mixed_o2_v2.2_ascii.msh",
    "2.2 binary": "mixed_o2_v2.2_bin.msh",
    "4.1 ascii": "mixed_o2_v4.1_ascii.msh",
    "4.1 binary": "mixed_o2_v4.1_bin.msh",
}


@pytest.mark.parametrize("name", QUAD.values(), ids=QUAD.keys())
def test_quad_block_all_formats(name):
    mesh = read_msh(DATA / name)
    assert mesh.n_cells == 8 and mesh.n_nodes == 15
    assert mesh.cell_area.sum() == pytest.approx(2.0)
    assert {k: v.size for k, v in mesh.boundary_faces.items()} == {
        "walls": 8,
        "outlet": 2,
        "inlet": 2,
    }
    outlet = mesh.boundary_faces["outlet"]
    np.testing.assert_allclose(mesh.face_normal[outlet], [[1, 0], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(mesh.face_centre[outlet][:, 0], 2.0, atol=1e-9)


@pytest.mark.parametrize("name", MIXED.values(), ids=MIXED.keys())
def test_second_order_mixed_mesh_uses_corner_nodes(name):
    mesh = read_msh(DATA / name)
    assert mesh.n_cells == 40
    assert set(mesh.cell_n_nodes) == {3, 4}
    # Mid-edge nodes are dropped: 40 cells, Euler's formula for one hole.
    assert mesh.n_nodes == 56
    assert {k: v.size for k, v in mesh.boundary_faces.items()} == {
        "cyl wall": 7,
        "far": 28,
    }
    assert mesh.cell_area.sum() == pytest.approx(11.3159, abs=1e-4)


def test_formats_give_the_same_mesh():
    ref = read_msh(DATA / MIXED["2.2 ascii"])
    for name in MIXED.values():
        mesh = read_msh(DATA / name)
        assert mesh.n_cells == ref.n_cells
        np.testing.assert_allclose(np.sort(mesh.cell_area), np.sort(ref.cell_area))
        np.testing.assert_allclose(
            np.sort(mesh.face_length[mesh.boundary_faces["far"]]),
            np.sort(ref.face_length[ref.boundary_faces["far"]]),
        )


def test_without_physical_groups_curves_are_named_by_tag():
    mesh = read_msh(DATA / "quad_nogroups_v4.1_ascii.msh")
    assert set(mesh.boundary_faces) == {"curve_1", "curve_2", "curve_3", "curve_4"}
    assert mesh.n_cells == 8


def test_scale_and_dispatch_by_extension():
    mesh = read_mesh(DATA / QUAD["4.1 ascii"], scale=0.001)
    assert mesh.cell_area.sum() == pytest.approx(2.0e-6)
    with pytest.raises(MeshError, match="unknown mesh file type"):
        read_mesh(DATA / "mesh.vtk")


def test_dispatch_su2(tmp_path):
    write_su2(tmp_path / "a.su2", rectangle(2, 2, cell_type="quad"))
    assert read_mesh(tmp_path / "a.su2").n_cells == read_su2(tmp_path / "a.su2").n_cells


def test_windows_line_endings(tmp_path):
    path = tmp_path / "m.msh"
    path.write_bytes((DATA / QUAD["2.2 ascii"]).read_bytes().replace(b"\n", b"\r\n"))
    assert read_msh(path).n_cells == 8


def _write_msh22(mesh, path, binary=False, big_endian=False):
    """Minimal MSH 2.2 writer for a quad mesh with physical curves."""
    e = ">" if big_endian else "<"
    names = list(mesh.boundary_faces)
    out = [b"$MeshFormat\n"]
    out.append(b"2.2 %d 8\n" % int(binary))
    if binary:
        out.append(struct.pack(e + "i", 1) + b"\n")
    out.append(b"$EndMeshFormat\n$PhysicalNames\n%d\n" % (len(names) + 1))
    for i, name in enumerate(names):
        out.append(b'1 %d "%s"\n' % (i + 1, name.encode()))
    out.append(b'2 100 "fluid"\n$EndPhysicalNames\n$Nodes\n%d\n' % mesh.n_nodes)
    # Non-contiguous, shuffled node numbers exercise the tag lookup.
    tags = 7 + 3 * np.arange(mesh.n_nodes)[::-1]
    for t, (x, y) in zip(tags, mesh.nodes, strict=True):
        out.append(
            struct.pack(e + "i3d", t, x, y, 0.0)
            if binary
            else b"%d %.16g %.16g 0\n" % (t, x, y)
        )
    out.append(b"\n$EndNodes\n" if binary else b"$EndNodes\n")
    lines = [
        (i + 1, tags[a], tags[b])
        for i, name in enumerate(names)
        for a, b in mesh.face_nodes[mesh.boundary_faces[name]]
    ]
    n_el = len(lines) + mesh.n_cells
    out.append(b"$Elements\n%d\n" % n_el)
    if binary:
        out.append(struct.pack(e + "3i", 1, len(lines), 2))
        for k, (phys, a, b) in enumerate(lines):
            out.append(struct.pack(e + "5i", k + 1, phys, phys, a, b))
        out.append(struct.pack(e + "3i", 3, mesh.n_cells, 2))
        for i in range(mesh.n_cells):
            out.append(struct.pack(e + "7i", 1000 + i, 100, 100, *tags[mesh.cell(i)]))
        out.append(b"\n")
    else:
        for k, (phys, a, b) in enumerate(lines):
            out.append(b"%d 1 2 %d %d %d %d\n" % (k + 1, phys, phys, a, b))
        for i in range(mesh.n_cells):
            n = tags[mesh.cell(i)]
            out.append(b"%d 3 2 100 100 %d %d %d %d\n" % (1000 + i, *n))
    out.append(b"$EndElements\n")
    Path(path).write_bytes(b"".join(out))


@pytest.mark.parametrize("binary, big", [(False, False), (True, False), (True, True)])
def test_written_msh22_roundtrip(tmp_path, binary, big):
    ref = rectangle(5, 3, x=(0.0, 2.0), y=(0.0, 1.0), cell_type="quad")
    _write_msh22(ref, tmp_path / "m.msh", binary=binary, big_endian=big)
    mesh = read_msh(tmp_path / "m.msh")
    assert mesh.n_cells == ref.n_cells and mesh.n_nodes == ref.n_nodes
    assert mesh.cell_area.sum() == pytest.approx(2.0)
    assert {k: v.size for k, v in mesh.boundary_faces.items()} == {
        k: v.size for k, v in ref.boundary_faces.items()
    }
    np.testing.assert_allclose(
        np.sort(mesh.nodes[:, 0]), np.sort(ref.nodes[:, 0]), atol=1e-14
    )


def test_run_case_with_msh_file(tmp_path):
    _write_msh22(
        rectangle(6, 6, x=(0.0, 1.0), y=(0.0, 0.5), cell_type="quad"),
        tmp_path / "channel.msh",
    )
    (tmp_path / "couette.toml").write_text(
        COUETTE.replace("channel.su2", "channel.msh")
    )
    result = run_case(tmp_path / "couette.toml", console=False)
    assert result.converged
    np.testing.assert_allclose(
        result.w[:, 1], 3.0 * result.mesh.cell_centroid[:, 1] / 0.5, atol=1e-8
    )


def test_unsupported_versions(tmp_path):
    for old_format in ("1", "3.0"):
        path = tmp_path / "v.msh"
        path.write_text(f"$MeshFormat\n{old_format} 0 8\n$EndMeshFormat\n")
        with pytest.raises(MeshError, match="not supported"):
            read_msh(path)


def test_not_a_msh_file(tmp_path):
    path = tmp_path / "x.msh"
    path.write_text("NDIME= 2\n")
    with pytest.raises(MeshError, match=r"\$MeshFormat missing"):
        read_msh(path)


def test_mesh_without_surface_elements(tmp_path):
    text = (DATA / QUAD["2.2 ascii"]).read_text()
    head, rest = text.split("$Elements\n")
    lines = rest.splitlines()
    kept = [ln for ln in lines[1:] if " 1 2 " in ln and not ln.startswith("$")]
    path = tmp_path / "m.msh"
    path.write_text(
        f"{head}$Elements\n{len(kept)}\n" + "\n".join(kept) + "\n$EndElements\n"
    )
    with pytest.raises(MeshError, match="Physical Surface"):
        read_msh(path)


def test_3d_mesh_rejected(tmp_path):
    text = (DATA / QUAD["2.2 ascii"]).read_text()
    head, rest = text.split("$Elements\n")
    path = tmp_path / "m.msh"
    path.write_text(f"{head}$Elements\n1\n1 4 2 1 1 1 2 3 4\n$EndElements\n")
    with pytest.raises(MeshError, match="3D elements"):
        read_msh(path)


def test_mesh_outside_the_plane_rejected(tmp_path):
    text = (DATA / QUAD["2.2 ascii"]).read_text().splitlines()
    i = text.index("$Nodes") + 2
    text[i] = text[i][:-1] + "0.5"
    path = tmp_path / "m.msh"
    path.write_text("\n".join(text) + "\n")
    with pytest.raises(MeshError, match="plane z = 0"):
        read_msh(path)


def test_undefined_node_reported(tmp_path):
    lines = (DATA / QUAD["2.2 ascii"]).read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.split()[1:2] == ["3"])
    lines[i] = lines[i].rsplit(" ", 1)[0] + " 999"
    path = tmp_path / "m.msh"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(MeshError, match="not defined"):
        read_msh(path)

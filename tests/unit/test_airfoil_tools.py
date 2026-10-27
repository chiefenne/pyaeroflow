"""Wall contour order, ``check-mesh`` and ``init-case`` for airfoil meshes."""

import re

import numpy as np
import pytest
from test_run import COUETTE

from pyaeroflow.cli import main
from pyaeroflow.config import load_case
from pyaeroflow.initcase import MODELS, make_case
from pyaeroflow.mesh import naca0012_ogrid, rectangle, write_su2
from pyaeroflow.mesh.ordering import order_boundary_faces
from pyaeroflow.meshcheck import check_mesh, first_cell_for_y_plus, flat_plate_cf
from pyaeroflow.run import run_case


@pytest.fixture(scope="module")
def ogrid():
    """A coarse O-grid (y+ about 16 at Re 2e5), boundaries airfoil and farfield."""
    return naca0012_ogrid(
        n_surface=64, n_radial=64, reynolds=2e5, y_plus=20.0, farfield_radius=50.0
    )


@pytest.fixture
def mesh_file(tmp_path, ogrid):
    path = tmp_path / "naca.su2"
    write_su2(path, ogrid)
    return path


def test_contour_order_closed_ring_starts_at_the_upper_trailing_edge(ogrid):
    faces = ogrid.boundary_faces["airfoil"]
    rng = np.random.default_rng(1)
    shuffled = rng.permutation(faces.size)
    nodes = ogrid.face_nodes[faces][shuffled]
    order = order_boundary_faces(nodes, ogrid.nodes)
    centre = ogrid.face_centre[faces][shuffled][order]
    assert order.size == faces.size and np.unique(order).size == faces.size
    assert centre[0, 0] > 0.99 and centre[1, 1] > 0  # trailing edge, upper side first
    le = np.argmin(centre[:, 0])
    assert np.all(centre[1:le, 1] > 0) and np.all(centre[le + 1 : -3, 1] < 0)
    steps = np.hypot(*np.diff(centre, axis=0).T)
    assert steps.max() < 0.1  # consecutive faces are neighbours


def test_contour_order_open_chain_runs_in_increasing_x():
    mesh = rectangle(5, 3, x=(0.0, 1.0), y=(0.0, 1.0), cell_type="quad")
    faces = mesh.boundary_faces["bottom"]
    perm = np.random.default_rng(0).permutation(faces.size)
    order = order_boundary_faces(mesh.face_nodes[faces][perm], mesh.nodes)
    x = mesh.face_centre[faces][perm][order][:, 0]
    assert np.all(np.diff(x) > 0)


def test_contour_order_rejects_two_separate_walls():
    nodes = np.array([[0, 0], [1, 0], [2, 0], [3, 0.0]])
    assert order_boundary_faces(np.array([[0, 1], [2, 3]]), nodes) is None


def test_wall_csv_is_ordered_and_has_arc_length(tmp_path):
    write_su2(tmp_path / "channel.su2", rectangle(6, 6, x=(0, 1), y=(0, 0.5)))
    (tmp_path / "couette.toml").write_text(COUETTE)
    run_case(tmp_path / "couette.toml", console=False)
    wall = np.genfromtxt(
        tmp_path / "out" / "couette_test.wall_bottom.csv", delimiter=",", names=True
    )
    assert wall.dtype.names[-1] == "s"
    assert np.all(np.diff(wall["x"]) > 0)
    np.testing.assert_allclose(wall["s"], wall["x"], atol=1e-12)


def test_check_mesh_y_plus_matches_the_documented_table(ogrid):
    rep = check_mesh(ogrid, reynolds=2e5)
    # Re 2e5: y1/c = 1.9e-4 for y+ = 1 (docs/pyaero.md, section 2)
    assert first_cell_for_y_plus(2e5) == pytest.approx(1.9e-4, rel=0.05)
    assert flat_plate_cf(6e6) == pytest.approx(0.0029, rel=0.05)
    assert 10 < rep.y_plus_max < 25
    text = rep.format()
    assert "too thick" in text and "y+ (cell centre)" in text
    assert rep.wall == "airfoil" and rep.chord == pytest.approx(1.0, abs=1e-3)


def test_check_mesh_without_reynolds_number_only_hints(ogrid):
    rep = check_mesh(ogrid)
    assert rep.y_plus_max is None
    assert any("--re" in text for _, text in rep.findings)


def test_check_mesh_flags_wrong_units_and_farfield(ogrid):
    rep = check_mesh(
        ogrid.__class__.from_cells(
            ogrid.nodes * 1000.0,
            [ogrid.cell(i) for i in range(ogrid.n_cells)],
            {k: ogrid.face_nodes[v] for k, v in ogrid.boundary_faces.items()},
        )
    )
    assert any("length unit" in text for _, text in rep.findings)
    near = naca0012_ogrid(
        n_surface=64, n_radial=64, reynolds=2e5, y_plus=20.0, farfield_radius=4.0
    )
    assert any("farfield only" in text for _, text in check_mesh(near).findings)


def test_check_mesh_needs_a_wall_name(ogrid):
    mesh = rectangle(4, 4, cell_type="quad")
    rep = check_mesh(mesh)
    assert rep.n_warnings == 1 and "--wall" in rep.format()
    assert check_mesh(ogrid, wall="nope").n_warnings == 1


def test_cli_check_mesh(mesh_file, capsys):
    assert main(["check-mesh", str(mesh_file), "--re", "2e5"]) == 0
    out = capsys.readouterr().out
    assert "boundary airfoil" in out and "warning:" in out
    assert main(["check-mesh", str(mesh_file / "missing.su2")]) == 1


@pytest.mark.parametrize("model", list(MODELS))
def test_init_case_writes_a_valid_case_for_every_model(mesh_file, model):
    case = mesh_file.with_name(f"c_{model}.toml")
    text, notes = make_case(mesh_file, case, model, reynolds=2e5, alpha=4.0)
    case.write_text(text)
    cfg = load_case(case).data
    assert cfg["mesh"]["file"] == "naca.su2"
    assert cfg["boundary"]["airfoil"]["type"] == (
        "slip-wall" if model == "euler" else "wall"
    )
    far = cfg["boundary"]["farfield"]
    assert far["angle_of_attack"] == 4.0
    assert cfg["output"]["force_boundaries"] == ["airfoil"]
    assert cfg["reference"]["moment_center"][0] == pytest.approx(0.25, abs=1e-3)
    if model != "euler":
        mu = cfg["fluid"]["viscosity"]
        assert mu == pytest.approx(cfg["reference"]["length"] / 2e5, rel=1e-6)
    if model in ("bcm", "aft", "lm"):
        assert cfg["physics"]["transition"]["model"] == model
    assert any("y+" in n for n in notes) == (model != "euler")


def test_init_case_models_use_the_validated_freestream_and_solver(mesh_file):
    def case(model):
        text, _ = make_case(mesh_file, mesh_file.with_name("x.toml"), model, 2e5)
        return text

    assert "nu_tilde_ratio = 3.0" in case("sa")
    assert "nu_tilde_ratio = 0.1" in case("aft")
    assert re.search(r"turbulent_viscosity_ratio = 1.0, intensity = 0.001", case("lm"))
    assert 'control = "growth"' in case("lm") and 'control = "ser"' in case("sst")


def test_init_case_errors(mesh_file, tmp_path):
    with pytest.raises(ValueError, match="--re is required"):
        make_case(mesh_file, tmp_path / "c.toml", "sa")
    with pytest.raises(ValueError, match="unknown model"):
        make_case(mesh_file, tmp_path / "c.toml", "k-epsilon", 1e6)
    with pytest.raises(ValueError, match="no airfoil boundary 'wing'"):
        make_case(mesh_file, tmp_path / "c.toml", "sa", 1e6, wall="wing")


def test_cli_init_case_then_run(mesh_file, capsys):
    assert (
        main(
            [
                "init-case",
                str(mesh_file),
                "--model",
                "sst",
                "--re",
                "2e5",
                "--alpha",
                "2",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    case = mesh_file.with_name("naca_sst.toml")
    assert "wrote" in out and "pyaeroflow polar" in out and case.exists()
    # Refuses to overwrite; --force does.
    assert main(["init-case", str(mesh_file), "--model", "sst", "--re", "2e5"]) == 1
    assert "exists" in capsys.readouterr().err
    assert (
        main(["init-case", str(mesh_file), "--model", "sst", "--re", "2e5", "--force"])
        == 0
    )
    capsys.readouterr()
    case.write_text(
        re.sub(r"max_iterations = \d+", "max_iterations = 3", case.read_text())
    )
    result = run_case(case, console=False)
    assert np.isfinite(result.coefficients.cl)
    wall = np.genfromtxt(
        case.parent / "output" / "naca_sst" / "naca_sst.wall_airfoil.csv",
        delimiter=",",
        names=True,
    )
    assert wall["x"][0] > 0.99 and wall["y"][1] > 0 and np.all(np.diff(wall["s"]) > 0)


def test_cli_init_case_reports_missing_reynolds_number(mesh_file, capsys):
    assert main(["init-case", str(mesh_file), "--model", "sa"]) == 1
    assert "--re is required" in capsys.readouterr().err

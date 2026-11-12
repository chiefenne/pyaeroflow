"""Case templates: packaged copies, init writes runnable cases."""

from pathlib import Path

import pytest

from pyaeroflow.cli import main
from pyaeroflow.config import load_case
from pyaeroflow.mesh import write_su2
from pyaeroflow.mesh.library import build
from pyaeroflow.templates import TEMPLATES, description, template_text, write_template

EXAMPLES = Path(__file__).parents[2] / "examples" / "cases"
FILE_BASED = sorted(n for n, t in TEMPLATES.items() if t.case_file)


def _without_mesh_line(text):
    return [line for line in text.splitlines() if not line.startswith("file = ")]


@pytest.mark.parametrize("name", FILE_BASED)
def test_packaged_template_equals_the_example_case(name):
    """The packaged copy may not drift from examples/cases (mesh path aside)."""
    t = TEMPLATES[name]
    example = (EXAMPLES / t.case_file).read_text()
    assert _without_mesh_line(template_text(name)) == _without_mesh_line(example)


@pytest.mark.parametrize("name", FILE_BASED)
def test_template_writes_a_valid_case(name, tmp_path):
    case, mesh, written = write_template(name, tmp_path)
    assert written and mesh.exists() and mesh.name == TEMPLATES[name].mesh_file
    config = load_case(case)
    assert config.resolve_path(config.data["mesh"]["file"]) == mesh.resolve()
    assert description(name)


@pytest.mark.parametrize("name", ["couette", "plate-sa", "ramp-m2"])
def test_template_runs(name, tmp_path):
    from pyaeroflow.run import run_case

    case, _, _ = write_template(name, tmp_path)
    config = load_case(case)
    config.data["solver"]["max_iterations"] = 3
    result = run_case(config, console=False)
    assert result.history["iteration"][-1] == 3


def test_new_case_name_and_existing_files(tmp_path):
    case, mesh, _ = write_template("plate-bcm", tmp_path, case="my_plate")
    assert case.name == "my_plate.toml"
    text = case.read_text()
    assert 'name = "my_plate"' in text and '"output/my_plate"' in text
    with pytest.raises(FileExistsError):
        write_template("plate-bcm", tmp_path, case="my_plate")
    # a second template on the same mesh keeps the file
    _, _, written = write_template("plate-bcm", tmp_path, case="other")
    assert not written


def test_cli_templates_and_init(tmp_path, capsys):
    assert main(["templates"]) == 0
    out = capsys.readouterr().out
    assert "plate-sa" in out and "airfoil" in out
    assert main(["templates", "jet"]) == 0
    assert main(["init", "cavity", "-o", str(tmp_path)]) == 0
    assert (tmp_path / "cavity_re100.toml").exists()
    assert (tmp_path / "cavity_quad_64.su2").exists()
    assert main(["init", "nope"]) == 1
    assert main(["init", "airfoil"]) == 1  # needs --mesh
    assert main(["init", "cavity", "-o", str(tmp_path), "--re", "1e5"]) == 1


def test_init_airfoil_with_band_and_boundary_layer_start(tmp_path):
    mesh = tmp_path / "foil.su2"
    write_su2(mesh, build("naca0012-ogrid", n_surface=128, n_radial=64))
    args = ["init", "airfoil", "--mesh", str(mesh), "--re", "6e6", "--alpha", "4"]
    assert main(args + ["--band", "--boundary-layer-start"]) == 0
    case = tmp_path / "foil_sa.toml"
    config = load_case(case)
    assert config.data["initial"]["boundary_layer"] == "airfoil"
    distances = config.data["refinement"]["band"]["distances"]
    assert distances == pytest.approx([0.05, 0.02, 0.008], rel=1e-3)
    assert config.data["numerics"]["limiter"] == "venkatakrishnan-wang"
    # init-case (the alias) writes the same file
    assert (
        main(
            [
                "init-case",
                str(mesh),
                "--re",
                "6e6",
                "--alpha",
                "4",
                "--band",
                "--boundary-layer-start",
                "-o",
                str(tmp_path / "b.toml"),
            ]
        )
        == 0
    )
    a = case.read_text().splitlines()
    b = (tmp_path / "b.toml").read_text().splitlines()
    assert a == b

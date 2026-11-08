"""Local mesh refinement inside a steady run ([refinement], R4)."""

import tomllib

import numpy as np
import pytest

from pyaeroflow.config import ConfigError, resolve
from pyaeroflow.mesh import annulus, read_mesh, write_su2
from pyaeroflow.refinement import read_refinement_info, replay_refinement
from pyaeroflow.run import mesh_fingerprint, run_case, run_polar

CYL = """
[case]
name = "cyl"
[mesh]
file = "annulus.su2"
[physics]
model = "euler"
[fluid]
density = 1.0
[reference]
length = 1.0
moment_center = [0.0, 0.0]
[boundary.cylinder]
type = "slip-wall"
[boundary.farfield]
type = "farfield"
velocity = 1.0
[numerics]
limiter = "none"
[solver]
residual_drop = 5.0
[refinement]
cycles = 1
indicator = "pressure_gradient_magnitude * h"
top_fraction = 0.15
[output]
directory = "out"
"""


@pytest.fixture
def cyl_dir(tmp_path):
    write_su2(tmp_path / "annulus.su2", annulus(32, 8, outer_radius=10.0, growth=1.3))
    (tmp_path / "cyl.toml").write_text(CYL)
    return tmp_path


def _raw(text):
    return tomllib.loads(text)


CHECKS = [
    (('indicator = "pressure_gradient_magnitude * h"\n', ""), "indicator is required"),
    (("top_fraction = 0.15\n", ""), "exactly one of threshold"),
    (("* h", "* hh"), "unknown name 'hh'"),
    (("pressure_gradient", "presure_gradient"), "did you mean"),
    (("pressure_gradient_magnitude", "residual_p"), "live residual"),
    (("pressure_gradient_magnitude", "nu_tilde"), "needs"),
    (("pressure_gradient_magnitude * h", "h[0]"), "not allowed"),
    (("cycles = 1\n", "cycles = 1\nwalls = ['farfield']\n"), "is a farfield"),
    (("cycles = 1\n", "cycles = 1\nwalls = ['wing']\n"), "no boundary 'wing'"),
]


@pytest.mark.parametrize("edit,message", CHECKS)
def test_refinement_config_is_checked(edit, message):
    text = CYL.replace(*edit)
    with pytest.raises(ConfigError, match=message):
        resolve(_raw(text))


def test_unsteady_runs_cannot_be_refined():
    text = CYL.replace(
        "[numerics]", '[time]\nmode = "unsteady"\ntime_step = 0.1\n[numerics]'
    )
    with pytest.raises(ConfigError, match="only steady runs"):
        resolve(_raw(text))


def test_walls_default_to_all_walls():
    config = resolve(_raw(CYL))
    assert config.data["refinement"]["walls"] == ["cylinder"]
    off = resolve(_raw(CYL.replace("cycles = 1", "cycles = 0")))
    assert off.data["refinement"]["cycles"] == 0


def test_refined_run_and_restart_replay(cyl_dir):
    result = run_case(cyl_dir / "cyl.toml", console=False)
    out = cyl_dir / "out"
    assert result.converged
    assert result.restart_file.parent == out / "cycle_1"
    rows = np.genfromtxt(out / "cyl.refinement.csv", delimiter=",", names=True)
    assert list(rows["cycle"]) == [0, 1]
    assert rows["cells"][1] == rows["cells_after"][0] > rows["cells"][0] == 256
    assert rows["marked"][0] == np.ceil(0.15 * 256)
    assert result.mesh.n_cells == rows["cells"][1]
    assert result.mesh.cell_n_nodes.max() > 4  # hanging nodes
    # Refinement lowers the spurious drag of the inviscid cylinder.
    assert rows["CD"][1] < rows["CD"][0]
    assert "marked" in (out / "cyl.refinement.log").read_text()

    # Every restart file of the refined cycle carries the log; the replay on
    # the original mesh gives the same mesh.
    info = read_refinement_info(result.restart_file)
    assert info["cycle"] == 1 and len(info["log"]) == 1
    ref = replay_refinement(read_mesh(cyl_dir / "annulus.su2"), info)
    assert mesh_fingerprint(ref.mesh) == mesh_fingerprint(result.mesh)
    assert read_refinement_info(next((out / "cycle_0").glob("*.npz"))) is None

    # A plain run from the refined file rebuilds the refined mesh itself.
    plain = CYL.replace("cycles = 1", "cycles = 0").replace(
        "[numerics]", f'[initial]\nrestart = "{result.restart_file}"\n[numerics]'
    )
    (cyl_dir / "plain.toml").write_text(plain.replace('"out"', '"out_plain"'))
    again = run_case(cyl_dir / "plain.toml", console=False)
    assert again.mesh.n_cells == result.mesh.n_cells
    assert again.history["iteration"][0] >= result.history["iteration"][-1]
    assert read_refinement_info(again.restart_file)["cycle"] == 1


def test_polar_with_refinement_is_refused(cyl_dir):
    with pytest.raises(ValueError, match="not available yet"):
        run_polar(cyl_dir / "cyl.toml", [0.0], console=False)

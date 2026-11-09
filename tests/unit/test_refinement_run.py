"""Local mesh refinement inside a steady run ([refinement], R4)."""

import tomllib

import numpy as np
import pytest

from pyaeroflow.config import ConfigError, resolve
from pyaeroflow.mesh import annulus, read_mesh, write_su2
from pyaeroflow.mesh.refine import Refinement
from pyaeroflow.refinement import (
    _cycle_config,
    _start_file,
    latest_chain_backup,
    own_chain,
    read_refinement_info,
    replay_refinement,
    restrict,
)
from pyaeroflow.run import (
    load_restart_state,
    mesh_fingerprint,
    run_case,
    run_polar,
    save_restart,
)

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


def test_limiter_frozen_in_the_cycles_after_a_refinement():
    text = CYL.replace('limiter = "none"', 'limiter = "venkatakrishnan-wang"')
    config = resolve(_raw(text))

    def freeze(cfg, k):
        c = _cycle_config(cfg, k, False, None, "continue", None)
        return c.data["numerics"]["limiter_freeze"]["after_residual_drop"]

    assert freeze(config, 0) is None
    assert freeze(config, 1) == 1.0
    assert config.data["numerics"]["limiter_freeze"]["after_residual_drop"] is None
    off = resolve(_raw(text.replace("cycles = 1", "cycles = 1\nlimiter_freeze = 0")))
    assert freeze(off, 1) is None
    own = text.replace(
        "[numerics]", "[numerics]\nlimiter_freeze = { after_iterations = 30 }"
    )
    assert freeze(resolve(_raw(own)), 1) is None


def test_force_stop_in_the_cycles_after_a_refinement():
    config = resolve(_raw(CYL))

    def conv(cfg, k, final=False):
        return _cycle_config(cfg, k, final, None, "continue", None).data["solver"][
            "convergence"
        ]

    assert conv(config, 0)["fields"] == []
    for final in (False, True):
        c = conv(config, 1, final)
        assert c["fields"] == ["cl", "cd"]
        assert (c["cauchy_eps"], c["cauchy_elements"]) == (1.0e-6, 20)
        assert c["min_residual_drop"] == 2.0
    assert config.data["solver"]["convergence"]["fields"] == []
    off = resolve(_raw(CYL.replace("cycles = 1", "cycles = 1\ncauchy_eps = 0")))
    assert conv(off, 1)["fields"] == []
    own = CYL.replace(
        "[numerics]",
        '[solver.convergence]\nfields = ["cm"]\ncauchy_eps = 1e-9\n[numerics]',
    )
    c = conv(resolve(_raw(own)), 1)
    assert c["fields"] == ["cm"] and c["cauchy_eps"] == 1e-9


@pytest.mark.parametrize("value,kept", [(0.5, False), (1.0, True)])
def test_transfer_drops_a_frozen_limiter(tmp_path, value, kept):
    mesh = annulus(8, 3)
    psi = np.full((mesh.n_cells, 3), value)
    file = tmp_path / "c_cycle_0.backup_000010.npz"
    state = {"kind": "steady", "cfl": 50.0, "nk_on": True, "psi": psi}
    save_restart(file, mesh, np.ones((mesh.n_cells, 3)), 10, None, state)
    source = np.repeat(np.arange(mesh.n_cells), 2)
    out = _start_file(file, mesh, lambda v: v[source], tmp_path / "s.npz", None, True)
    new = load_restart_state(out)
    assert new["cfl"] == 50.0 and new["nk_on"]
    assert ("psi" in new) == kept
    if kept:
        assert new["psi"].shape == (2 * mesh.n_cells, 3)


def test_restrict_undoes_injection():
    ref = Refinement.from_mesh(annulus(8, 3))
    n0 = ref.n_cells
    ref.split([0, 5], 3, 1)
    ref.split([1], 2, 2)
    origin = ref.origin
    v0 = np.arange(2.0 * n0).reshape(n0, 2)
    back = restrict(v0[origin], origin, ref.mesh.cell_area, n0)
    np.testing.assert_allclose(back, v0)
    # Children with different values: the area-weighted mean of the parent.
    leaf = np.arange(ref.n_cells, dtype=float)
    area = ref.mesh.cell_area
    i = origin == 1
    mean = (leaf[i] * area[i]).sum() / area[i].sum()
    assert restrict(leaf, origin, area, n0)[1] == pytest.approx(mean)


def test_own_chain_and_latest_chain_backup(tmp_path):
    assert own_chain("c_alpha_2_cycle_1.backup_000010.npz", "c_alpha_2")
    assert own_chain("c_alpha_2.cycle_2_start.npz", "c_alpha_2")
    assert not own_chain("c_alpha_2_cycle_1.backup_000010.npz", "c_alpha_4")
    assert not own_chain("c_alpha_20_cycle_1.backup_000010.npz", "c_alpha_2")
    assert not own_chain("c_alpha_2.restricted_start.npz", "c_alpha_2")
    for k, n in ((0, 30), (1, 12), (1, 20)):
        d = tmp_path / f"cycle_{k}"
        d.mkdir(exist_ok=True)
        (d / f"c_cycle_{k}.backup_{n:06d}.npz").touch()
    (tmp_path / "cycle_2").mkdir()  # started, no backup yet
    found = latest_chain_backup(tmp_path, "c")
    assert found.name == "c_cycle_1.backup_000020.npz"
    assert latest_chain_backup(tmp_path / "cycle_2", "c") is None


def test_refine_restart_skips_the_first_run(cyl_dir):
    plain = CYL.replace("cycles = 1", "cycles = 0").replace('"out"', '"out_plain"')
    (cyl_dir / "plain.toml").write_text(plain)
    done = run_case(cyl_dir / "plain.toml", console=False)
    text = CYL.replace("cycles = 1", "cycles = 1\nrefine_restart = true").replace(
        "[numerics]", f'[initial]\nrestart = "{done.restart_file}"\n[numerics]'
    )
    (cyl_dir / "again.toml").write_text(text)
    result = run_case(cyl_dir / "again.toml", console=False)
    out = cyl_dir / "out"
    rows = np.genfromtxt(out / "cyl.refinement.csv", delimiter=",", names=True)
    assert list(rows["cycle"]) == [0, 1] and rows["iterations"][0] == 0
    assert rows["marked"][0] == np.ceil(0.15 * 256)
    assert result.mesh.n_cells == rows["cells"][1] == rows["cells_after"][0]
    assert not (out / "cycle_0").exists() and result.converged
    assert "not run" in (out / "cyl.refinement.log").read_text()


def test_polar_refines_every_angle(cyl_dir):
    # refine_restart applies to the case's own initial.restart only: without
    # one every angle runs its cycle 0.
    text = CYL.replace("cycles = 1", "cycles = 1\nrefine_restart = true")
    (cyl_dir / "cyl.toml").write_text(text)
    results = run_polar(cyl_dir / "cyl.toml", [0.0, 10.0], console=False)
    out = cyl_dir / "out"
    assert len(results) == 2 and all(r.converged for r in results)
    polar = np.genfromtxt(out / "cyl.polar.csv", delimiter=",", names=True)
    assert list(polar["alpha"]) == [0.0, 10.0]
    for alpha, r in zip((0, 10), results, strict=True):
        a = out / f"alpha_{alpha}"
        rows = np.genfromtxt(
            a / f"cyl_alpha_{alpha}.refinement.csv", delimiter=",", names=True
        )
        assert list(rows["cycle"]) == [0, 1]
        assert rows["cells"][0] == 256 and r.mesh.n_cells == rows["cells"][1]
        assert r.restart_file.parent == a / "cycle_1"
        assert (a / "cycle_0").is_dir()
    # The second angle starts on the original mesh from the first angle's
    # refined state, averaged onto it, and refines for its own flow.
    log = (out / "alpha_10" / "cyl_alpha_10.refinement.log").read_text()
    assert "averaged onto the original mesh" in log
    start = out / "alpha_10" / "cyl_alpha_10.restricted_start.npz"
    assert read_refinement_info(start) is None
    with np.load(start) as data:
        assert data["w"].shape[0] == 256
    # A second polar run of an angle continues its own chain at the last cycle.
    again = run_polar(cyl_dir / "cyl.toml", [10.0], console=False)
    assert again[0].mesh.n_cells == results[1].mesh.n_cells
    log = (out / "alpha_10" / "cyl_alpha_10.refinement.log").read_text()
    assert "start at cycle 1" in log

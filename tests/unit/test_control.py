"""Control files of a running case: ``.stop`` and ``.result_N`` (L7)."""

import numpy as np
import pytest
from test_run import COUETTE, UNSTEADY

from pyaeroflow.control import STOP, ControlFiles, StopFileError, check_no_stop_file
from pyaeroflow.mesh import rectangle, write_su2
from pyaeroflow.run import run_case


@pytest.fixture
def case_dir(tmp_path):
    mesh = rectangle(6, 6, x=(0.0, 1.0), y=(0.0, 0.5), cell_type="quad")
    write_su2(tmp_path / "channel.su2", mesh)
    (tmp_path / "couette.toml").write_text(COUETTE)
    return tmp_path


def _touch_stop_at_check(monkeypatch, number):
    """Create the real ``.stop`` file when the run checks for it the n-th time."""
    original = ControlFiles.stop_requested
    calls = []

    def check(self):
        calls.append(1)
        if len(calls) == number:
            (self.out_dir / STOP).touch()
        return original(self)

    monkeypatch.setattr(ControlFiles, "stop_requested", check)


def test_result_numbers_notices_and_schedule(tmp_path):
    for name in (".result_7", ".result_2", ".result_x", ".results_9", ".stopped"):
        (tmp_path / name).touch()
    control = ControlFiles(tmp_path, after=4)
    assert control.result_numbers() == [2, 7]
    assert control.notices() == [
        "notice: control file .result_2 found, but the run starts after "
        "iteration 4: not written",
        "notice: control file .result_7 found, the fields are written at iteration 7",
    ]
    assert not control.stop_requested()
    assert control.due_results(5) == [] and control.due_results(6) == []
    # N = 7 falls on a rejected iteration: it waits for the next good one.
    assert control.due_results(7, good=False) == []
    assert control.due_results(8) == [7]
    assert control.due_results(9) == []  # written once
    # A file created after its number has passed, or at or before the start.
    (tmp_path / ".result_3").touch()
    (tmp_path / ".result_4").touch()
    assert control.due_results(10) == []
    (tmp_path / ".result_11").touch()
    assert control.due_results(11) == [11]
    (tmp_path / STOP).touch()
    assert control.stop_requested()


def test_deferred_result_file_removed_meanwhile_is_dropped(tmp_path):
    control = ControlFiles(tmp_path)
    (tmp_path / ".result_3").touch()
    assert control.due_results(3, good=False) == []
    (tmp_path / ".result_3").unlink()
    assert control.due_results(4) == []


def test_existing_stop_file_fails_before_the_mesh_is_read(case_dir):
    out = case_dir / "out"
    out.mkdir()
    (out / STOP).touch()
    with pytest.raises(StopFileError, match="rm .*out/.stop"):
        check_no_stop_file(out)
    # Not even the mesh is read: a missing mesh file does not matter.
    (case_dir / "channel.su2").unlink()
    with pytest.raises(StopFileError, match="stop file"):
        run_case(case_dir / "couette.toml", console=False)
    assert not list(out.glob("couette_test*"))


def test_result_file_writes_fields_at_n_and_the_run_goes_on(case_dir):
    out = case_dir / "out"
    out.mkdir()
    for n in (3, 5000):  # the second one is never reached
        (out / f".result_{n}").touch()
    result = run_case(case_dir / "couette.toml", console=False)
    assert result.converged and int(result.history["iteration"][-1]) > 3
    assert (out / "couette_test_000003.vtu").exists()
    assert not (out / "couette_test_005000.vtu").exists()
    assert (out / "couette_test.pvd").exists()
    log = (out / "couette_test.log").read_text()
    assert "notice: control file .result_3 found, the fields are written at" in log
    assert "control file .result_3: fields written for iteration 3" in log
    assert (out / ".result_3").exists()  # never deleted


def test_stop_file_ends_the_run_with_backup_and_results(case_dir, monkeypatch):
    out = case_dir / "out"
    _touch_stop_at_check(monkeypatch, 4)  # the 1st check is iteration 0
    result = run_case(case_dir / "couette.toml", console=False)
    assert not result.converged
    assert int(result.history["iteration"][-1]) == 3
    assert result.restart_file == out / "couette_test.backup_000003.npz"
    assert result.restart_file.exists() and (out / "couette_test.vtu").exists()
    log = (out / "couette_test.log").read_text()
    assert "control file .stop found: stopping after iteration 3" in log
    assert "stopped" in log.split("Summary")[1]
    assert (out / STOP).exists()  # never deleted
    # The next start fails until the file is removed; then the run continues.
    with pytest.raises(StopFileError):
        run_case(case_dir / "couette.toml", console=False)
    (out / STOP).unlink()
    monkeypatch.undo()
    text = COUETTE.replace(
        "[initial]\nvelocity = [0.0, 0.0]",
        f'[initial]\nrestart = "{result.restart_file}"',
    )
    (case_dir / "couette.toml").write_text(text)
    (out / ".result_2").touch()  # before the restart: a notice only
    second = run_case(case_dir / "couette.toml", console=False)
    assert second.converged and second.history["iteration"][0] == 3
    log = (out / "couette_test.log").read_text()
    assert "control file .result_2 found, but the run starts after iteration 3" in log


def test_stop_file_unsteady(tmp_path, monkeypatch):
    mesh = rectangle(4, 32, x=(0.0, 1.0), y=(0.0, 0.5), cell_type="quad")
    write_su2(tmp_path / "channel.su2", mesh)
    (tmp_path / "couette.toml").write_text(UNSTEADY)
    out = tmp_path / "out"
    out.mkdir()
    (out / ".result_2").touch()
    _touch_stop_at_check(monkeypatch, 3)
    result = run_case(tmp_path / "couette.toml", console=False)
    assert int(result.history["step"][-1]) == 3
    assert result.restart_file == out / "couette_unsteady.backup_000003.npz"
    assert result.restart_file.exists() and (out / "couette_unsteady.vtu").exists()
    assert (out / "couette_unsteady_000002.vtu").exists()
    assert np.load(result.restart_file)["time"] == pytest.approx(0.15)
    log = (out / "couette_unsteady.log").read_text()
    assert "control file .stop found: stopping after step 3" in log


def test_read_toml_numbers_and_result_field_names(tmp_path):
    for name in (".read_toml_7", ".read_toml_2", ".read_toml_x", ".read_toml_"):
        (tmp_path / name).touch()
    (tmp_path / ".result_4").write_text("vorticity, k  # debug\nwall_distance\n")
    (tmp_path / ".result_5").touch()
    control = ControlFiles(tmp_path, after=3)
    assert control.reload_numbers() == [2, 7]
    assert control.result_fields(4) == ["vorticity", "k", "wall_distance"]
    assert control.result_fields(5) == [] and control.result_fields(9) == []
    notes = control.notices("iteration")
    assert any(
        ".read_toml_7 found, the case file is read at iteration 7" in n for n in notes
    )
    assert any(
        ".read_toml_2 found, but the run starts after iteration 3" in n for n in notes
    )
    assert control.due_reloads(2) == [] and control.due_reloads(6) == []
    assert control.due_reloads(7, good=False) == []
    assert control.due_reloads(8) == [7]  # waits for the next good iteration
    assert control.due_reloads(9) == []


def test_result_file_with_field_names_adds_them(case_dir):
    out = case_dir / "out"
    out.mkdir()
    (out / ".result_2").write_text("vorticity wall_distance nope\n")
    (out / ".result_3").touch()
    run_case(case_dir / "couette.toml", console=False)
    with_extra = (out / "couette_test_000002.vtu").read_bytes()
    plain = (out / "couette_test_000003.vtu").read_bytes()
    assert b"vorticity" in with_extra and b"vorticity" not in plain
    log = (out / "couette_test.log").read_text()
    assert "control file .result_2: unknown field 'nope'" in log
    assert "skipped" in log and "fields written for iteration 2 (4 fields)" in log


def _edit_toml_at_first_check(monkeypatch, path, old, new):
    """Change the case file when the run first looks for ``.read_toml_N``."""
    original = ControlFiles.due_reloads
    done = []

    def due(self, number, good=True):
        if not done:
            done.append(1)
            path.write_text(path.read_text().replace(old, new))
        return original(self, number, good)

    monkeypatch.setattr(ControlFiles, "due_reloads", due)


def test_read_toml_applies_live_keys_and_reports_the_rest(case_dir, monkeypatch):
    out = case_dir / "out"
    out.mkdir()
    (out / ".read_toml_3").touch()
    toml = case_dir / "couette.toml"
    _edit_toml_at_first_check(
        monkeypatch,
        toml,
        '[output]\ndirectory = "out"',
        '[output]\ndirectory = "out"\nfields_interval = 2',
    )
    toml.write_text(toml.read_text().replace("viscosity = 0.3", "viscosity = 0.5"))
    toml.write_text(
        toml.read_text().replace("[solver]\n", "[solver]\nmax_iterations = 6\n")
    )
    result = run_case(toml, console=False)
    log = (out / "couette_test.log").read_text()
    assert "control file .read_toml_3: case file read at iteration 3" in log
    assert "changed output.fields_interval: 0 -> 2" in log
    assert "not applied (cannot change during a run)" not in log  # file is as at start
    assert int(result.history["iteration"][-1]) == 6
    assert (out / "couette_test_000004.vtu").exists()
    assert (out / "couette_test_000006.vtu").exists()
    snapshot = out / "couette_test.read_toml_000003.toml"
    assert "fields_interval = 2" in snapshot.read_text()


def test_read_toml_changing_the_physics_is_reported_not_applied(case_dir, monkeypatch):
    out = case_dir / "out"
    out.mkdir()
    (out / ".read_toml_2").touch()
    toml = case_dir / "couette.toml"
    _edit_toml_at_first_check(monkeypatch, toml, "viscosity = 0.3", "viscosity = 0.5")
    run_case(toml, console=False)
    log = (out / "couette_test.log").read_text()
    assert "not applied (cannot change during a run) fluid.viscosity: 0.3 -> 0.5" in log
    assert not list(out.glob("*.read_toml_*.toml"))  # nothing applied: no copy


def test_read_toml_with_an_invalid_case_file_keeps_the_run_going(case_dir, monkeypatch):
    out = case_dir / "out"
    out.mkdir()
    (out / ".read_toml_2").touch()
    toml = case_dir / "couette.toml"
    _edit_toml_at_first_check(monkeypatch, toml, "[numerics]", "[numerics\n")
    result = run_case(toml, console=False)
    assert result.converged
    log = (out / "couette_test.log").read_text()
    assert "not applied, the case file is invalid" in log


def test_fields_listing_of_a_case(case_dir):
    from pyaeroflow.config import load_case
    from pyaeroflow.io.fields import case_listing

    text = case_listing(load_case(case_dir / "couette.toml").data)
    assert "* velocity " in text and "  vorticity " in text
    assert text.index("provides (") < text.index("does not provide")
    assert "k  " in text.split("does not provide")[1]  # SST only

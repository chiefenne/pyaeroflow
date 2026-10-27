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

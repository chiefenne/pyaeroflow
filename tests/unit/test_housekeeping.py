"""Backup file names, restart-info and clean (D56, D60)."""

import json

import pytest
from test_run import COUETTE, POLAR

from pyaeroflow.cli import main
from pyaeroflow.housekeeping import (
    backup_name,
    backup_number,
    case_files,
    latest_backup,
    restart_info,
)
from pyaeroflow.mesh import annulus, rectangle, write_su2
from pyaeroflow.run import run_case, run_polar


@pytest.fixture
def run_dir(tmp_path):
    """A finished Couette run with rolling files, backups, a field series."""
    write_su2(tmp_path / "channel.su2", rectangle(6, 6, x=(0, 1), y=(0, 0.5)))
    text = COUETTE.replace(
        'directory = "out"',
        'directory = "out"\nrestart_interval = 3\nrestart_backup_interval = 5\n'
        "fields_interval = 4",
    )
    (tmp_path / "couette.toml").write_text(text)
    result = run_case(tmp_path / "couette.toml", console=False)
    return tmp_path, result


def test_backup_names_round_trip(tmp_path):
    assert backup_name("c", 645) == "c.backup_000645.npz"
    assert backup_number("out/c.backup_000645.npz") == 645
    assert backup_number("c.restart_000012.npz") == 12  # earlier versions
    assert backup_number("c.restart_a.npz") is None
    assert latest_backup(tmp_path, "c") is None
    for n in (100, 200, 45):
        (tmp_path / backup_name("c", n)).write_bytes(b"")
    (tmp_path / "c.restart.npz").write_bytes(b"")
    assert latest_backup(tmp_path, "c").name == "c.backup_000200.npz"
    (tmp_path / "other.backup_000999.npz").write_bytes(b"")
    assert latest_backup(tmp_path, "c").name == "c.backup_000200.npz"
    assert latest_backup(tmp_path, "d") is None
    # a run of an earlier version: only the final file without a number
    legacy = tmp_path / "old"
    legacy.mkdir()
    (legacy / "c.restart.npz").write_bytes(b"")
    assert latest_backup(legacy, "c").name == "c.restart.npz"


def test_the_final_state_is_a_backup(run_dir):
    tmp_path, result = run_dir
    out = result.output_dir
    n = int(result.history["iteration"][-1])
    assert result.restart_file == out / backup_name("couette_test", n)
    assert latest_backup(out, "couette_test") == result.restart_file
    assert not (out / "couette_test.restart.npz").exists()


def test_restart_info_reports_where_the_run_stopped(run_dir, capsys):
    _, result = run_dir
    n = int(result.history["iteration"][-1])
    info = restart_info(result.restart_file)
    assert info["kind"] == "steady" and info["iteration"] == n
    assert info["n_cells"] == 36 and info["n_variables"] == 3
    assert info["cfl"] > 0 and 0.0 < info["eta"] <= 0.5
    assert main(["restart-info", str(result.restart_file), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["iteration"] == n
    assert main(["restart-info", str(result.restart_file)]) == 0
    assert f"iteration    {n}" in capsys.readouterr().out
    rolling = restart_info(result.output_dir / "couette_test.restart_a.npz")
    assert rolling["iteration"] % 3 == 0


def test_clean_lists_by_default_and_deletes_by_category(run_dir, capsys):
    tmp_path, result = run_dir
    out, case = result.output_dir, str(tmp_path / "couette.toml")
    before = sorted(p.name for p in out.iterdir())
    assert main(["clean", case]) == 0  # nothing selected
    assert "nothing selected" in capsys.readouterr().out
    assert main(["clean", case, "--backup", "--fields"]) == 0  # dry run
    text = capsys.readouterr().out
    assert "dry run" in text and "would be deleted" in text
    assert sorted(p.name for p in out.iterdir()) == before
    found = case_files(out, "couette_test")
    assert len(found["restarts"]) == 2 and len(found["backup"]) >= 2
    assert main(["clean", case, "--backup", "--yes"]) == 0
    assert not list(out.glob("*.backup_*.npz")) and (out / "couette_test.vtu").exists()
    assert (out / "couette_test.restart_a.npz").exists()
    assert main(["clean", case, "--restarts", "--fields", "--yes"]) == 0
    assert not list(out.glob("*.npz")) and not list(out.glob("*.vtu"))
    assert not (out / "couette_test.pvd").exists()
    assert (out / "couette_test.log").exists()  # not selected
    assert (out / "couette_test.history.csv").exists()
    assert main(["clean", case, "--log", "--history", "--wall", "--yes"]) == 0
    assert [p.name for p in out.iterdir()] == ["couette_test.resolved.toml"]
    assert (tmp_path / "channel.su2").exists() and (tmp_path / "couette.toml").exists()


def test_clean_all_removes_everything_the_run_wrote(run_dir):
    tmp_path, result = run_dir
    assert main(["clean", str(tmp_path / "couette.toml"), "--all", "--yes"]) == 0
    assert not list(result.output_dir.iterdir())
    assert (tmp_path / "channel.su2").exists() and (tmp_path / "couette.toml").exists()


def test_clean_takes_the_old_names_as_backups(run_dir):
    tmp_path, result = run_dir
    out = result.output_dir
    for name in ("couette_test.restart.npz", "couette_test.restart_000007.npz"):
        (out / name).write_bytes(b"")
    found = case_files(out, "couette_test")["backup"]
    assert {"couette_test.restart.npz", "couette_test.restart_000007.npz"} <= {
        p.name for p in found
    }


def test_clean_a_polar_run(tmp_path):
    write_su2(tmp_path / "annulus.su2", annulus(32, 8, outer_radius=10.0, growth=1.3))
    (tmp_path / "cyl.toml").write_text(POLAR)
    run_polar(tmp_path / "cyl.toml", [0.0, 5.0], console=False)
    out = tmp_path / "out"
    assert len(case_files(out, "cyl")["backup"]) == 2  # one final state per angle
    assert main(["clean", str(tmp_path / "cyl.toml"), "--backup", "--yes"]) == 0
    assert not list(out.glob("alpha_*/*.backup_*.npz"))
    assert (out / "cyl.polar.csv").exists()
    assert main(["clean", str(tmp_path / "cyl.toml"), "--all", "--yes"]) == 0
    assert not list(out.iterdir())

"""Numerics debug log of steady implicit runs (output.debug_log)."""

from test_run import COUETTE

from pyaeroflow.mesh import rectangle, write_su2
from pyaeroflow.run import run_case


def test_debug_log_has_a_block_per_iteration(tmp_path):
    write_su2(tmp_path / "channel.su2", rectangle(6, 6, x=(0, 1), y=(0, 0.5)))
    text = COUETTE.replace(
        'directory = "out"', 'directory = "out"\ndebug_log = true'
    ).replace("residual_drop = 9.0", "residual_drop = 9.0\nmax_iterations = 12")
    (tmp_path / "couette.toml").write_text(text)
    result = run_case(tmp_path / "couette.toml", console=False)
    debug = (tmp_path / "out" / "couette_test.debug.log").read_text()
    n = int(result.history["iteration"][-1])
    assert debug.count("\nit ") == n
    assert "reference RMS" in debug and "share of RMS^2" in debug
    assert "trial step 1 " in debug
    assert "true linear residual" in debug  # Newton-Krylov steps
    assert "largest residuals after the step" in debug
    assert "cells changed by the step limiter" in debug
    log = (tmp_path / "out" / "couette_test.log").read_text()
    assert "numerics debug log" in log.split("Summary")[1]


def test_no_debug_log_by_default(tmp_path):
    write_su2(tmp_path / "channel.su2", rectangle(6, 6, x=(0, 1), y=(0, 0.5)))
    (tmp_path / "couette.toml").write_text(COUETTE)
    run_case(tmp_path / "couette.toml", console=False)
    assert not (tmp_path / "out" / "couette_test.debug.log").exists()

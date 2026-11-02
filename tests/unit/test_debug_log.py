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


def test_full_step_rise_names_the_cells():
    import numpy as np

    from pyaeroflow.io.debug_log import NumericsDebugLog

    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]])
    ref = np.array([[1e-3, 1e-3], [1e-3, 1e-3], [1e-3, 1e-3]])
    full = ref.copy()
    full[2, 1] = 5e-2  # the second variable rises in the third cell
    rec = {
        "it": 3,
        "cfl": 1e4,
        "nk": True,
        "gmres": 5,
        "tolerance": 0.5,
        "linear_residual": 0.1,
        "factorized": False,
        "reference": ref,
        "reference_rms": 1e-3,
        "du": np.zeros((3, 2)),
        "limited": [0, 0],
        "trials": [(1.0, 3e-2, np.array([1e-3, 3e-2]))],
        "full_step": full,
        "accepted": True,
        "step": 0.25,
        "new": ref,
    }
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        log = NumericsDebugLog(Path(d) / "x.log", ["a", "b"], xy)
        log(rec)
        log.close()
        text = (Path(d) / "x.log").read_text()
    assert "full step: where it raised the residual" in text
    assert "b: top 3 cells hold 100 % of the rise" in text
    assert "cell 2 (2.0000, 0.5000) |R| 1.00e-03 -> 5.00e-02" in text

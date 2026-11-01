"""Force-based (Cauchy) convergence of steady and unsteady runs."""

import numpy as np
import pytest
from test_run import COUETTE, UNSTEADY

from pyaeroflow.config import ConfigError, load_case
from pyaeroflow.mesh import rectangle, write_su2
from pyaeroflow.run import run_case
from pyaeroflow.solver.convergence import (
    CauchyCriterion,
    WindowedCauchyCriterion,
    window_weight,
)


def test_cauchy_needs_a_full_window_below_eps():
    c = CauchyCriterion(["cd"], elements=3, eps=1e-3)
    # (CL, CD, CM): CL moves a lot but is not monitored.
    assert not c.update(1, (0.0, 1.0, 0.0))  # no change yet
    assert not c.update(2, (5.0, 1.0004, 0.0))
    assert not c.update(3, (9.0, 1.0006, 0.0))  # window of two changes
    assert c.update(4, (1.0, 1.0008, 0.0))
    expected = (4e-4 / 1.0004 + 2e-4 / 1.0006 + 2e-4 / 1.0008) / 3
    assert c.value[0] == pytest.approx(expected, rel=1e-9)
    assert not c.update(5, (1.0, 1.1, 0.0))  # a jump leaves the window mean high


def test_cauchy_start_iteration_and_all_fields():
    c = CauchyCriterion(["cl", "cm"], elements=2, eps=1e-3, start=6)
    for it in range(1, 6):
        assert not c.update(it, (0.5, 0.0, 0.1))
    assert not c.update(6, (0.5, 0.0, 0.1))  # one change in the window
    assert c.update(7, (0.5, 0.0, 0.1))
    assert not c.update(8, (0.6, 0.0, 0.1))  # CL changed by 0.1


def test_cauchy_element_is_relative():
    c = CauchyCriterion(["cd"], elements=1, eps=1e-3)
    c.update(1, (0.0, 100.0, 0.0))
    # A change of 0.05 is 5e-4 of a coefficient of 100 (met) ...
    assert c.update(2, (0.0, 100.05, 0.0))
    # ... but 0.05 of a coefficient of 1 would not be.
    d = CauchyCriterion(["cd"], elements=1, eps=1e-3)
    d.update(1, (0.0, 1.0, 0.0))
    assert not d.update(2, (0.0, 1.05, 0.0))


def test_window_weights():
    assert window_weight("square", 3, 10) == 1.0
    for f in ("hann", "hann_square", "bump"):
        assert window_weight(f, 0, 11) == 0.0
        assert window_weight(f, 10, 11) == pytest.approx(0.0, abs=1e-12)
        assert window_weight(f, 5, 11) > 0.0
    assert window_weight("hann", 5, 11) == pytest.approx(1.0)
    assert window_weight("hann_square", 5, 11) == pytest.approx(1.0)


def test_windowed_cauchy_on_a_periodic_signal():
    # CD oscillates around 1; the time average settles, the signal does not.
    def cd(step):
        return (0.0, 1.0 + 0.2 * np.sin(0.5 * step), 0.0)

    c = WindowedCauchyCriterion(
        ["cd"], n_window=2000, elements=20, eps=1e-3, window_start=10, start=5
    )
    met = None
    for step in range(1, 2001):
        if c.update(step, cd(step)):
            met = step
            break
    assert met is not None and met > 10 + 5 + 20
    # No element before window_start + start.
    d = WindowedCauchyCriterion(
        ["cd"], n_window=100, elements=2, eps=1.0, window_start=10, start=5
    )
    for step in range(1, 15):
        assert not d.update(step, (0.0, 1.0, 0.0))
    assert not d.update(15, (0.0, 1.0, 0.0))  # first element, window not full
    assert d.update(16, (0.0, 1.0, 0.0))


def test_windowed_cauchy_hann_ignores_the_first_sample():
    c = WindowedCauchyCriterion(["cd"], n_window=11, function="hann", elements=2)
    assert not c.update(0, (0.0, 5.0, 0.0))  # weight 0: no average yet
    assert c.average is None
    c.update(5, (0.0, 3.0, 0.0))
    assert c.average[0] == pytest.approx(3.0)


@pytest.fixture
def case_dir(tmp_path):
    write_su2(tmp_path / "channel.su2", rectangle(6, 6, x=(0, 1), y=(0, 0.5)))
    return tmp_path


def test_cauchy_stops_a_run_before_the_residual_drop(case_dir):
    text = COUETTE.replace(
        "residual_drop = 9.0",
        "residual_drop = 14.0\n[solver.convergence]\n"
        'fields = ["cm"]\ncauchy_elements = 2\ncauchy_eps = 1e-4',
    )
    (case_dir / "couette.toml").write_text(text)
    result = run_case(case_dir / "couette.toml", console=False)
    assert result.converged
    log = (case_dir / "out" / "couette_test.log").read_text()
    assert "Cauchy criterion met on cm" in log
    assert "convergence criterion" in log.split("Summary")[1]


def test_convergence_field_must_be_a_coefficient(case_dir):
    (case_dir / "couette.toml").write_text(
        COUETTE + '\n[solver.convergence]\nfields = ["ct"]\n'
    )
    with pytest.raises(ConfigError, match="solver.convergence.fields"):
        load_case(case_dir / "couette.toml")


def test_windowed_cauchy_stops_an_unsteady_run(case_dir):
    text = UNSTEADY.replace("end_time = 0.5", "end_time = 5.0") + (
        '\n[time.convergence]\nfields = ["cm"]\ncauchy_elements = 3\n'
        "cauchy_eps = 1e-1\nwindow_start_step = 10\nstart_step = 2\n"
    )
    (case_dir / "couette.toml").write_text(text)
    run_case(case_dir / "couette.toml", console=False)
    log = (case_dir / "out" / "couette_unsteady.log").read_text()
    assert "Cauchy criterion met on the time average of cm" in log
    summary = log.split("Summary")[1]
    assert "convergence criterion" in summary
    assert "stopped" not in summary
    assert "time steps" in summary


def test_time_convergence_field_must_be_a_coefficient(case_dir):
    (case_dir / "couette.toml").write_text(
        UNSTEADY + '\n[time.convergence]\nfields = ["ct"]\n'
    )
    with pytest.raises(ConfigError, match="time.convergence.fields"):
        load_case(case_dir / "couette.toml")

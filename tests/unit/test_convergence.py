"""Force-based (Cauchy) convergence of steady runs."""

import pytest
from test_run import COUETTE

from pyaeroflow.config import ConfigError, load_case
from pyaeroflow.mesh import rectangle, write_su2
from pyaeroflow.run import run_case
from pyaeroflow.solver.convergence import CauchyCriterion


def test_cauchy_needs_a_full_window_below_eps():
    c = CauchyCriterion(["cd"], elements=3, eps=1e-3)
    # (CL, CD, CM): CL moves a lot but is not monitored.
    assert not c.update(1, (0.0, 1.0, 0.0))  # no change yet
    assert not c.update(2, (5.0, 1.0004, 0.0))
    assert not c.update(3, (9.0, 1.0006, 0.0))  # window of two changes
    assert c.update(4, (1.0, 1.0008, 0.0))
    assert c.value[0] == pytest.approx(8e-4 / 3, rel=1e-6)
    assert not c.update(5, (1.0, 1.1, 0.0))  # a jump leaves the window mean high


def test_cauchy_start_iteration_and_all_fields():
    c = CauchyCriterion(["cl", "cm"], elements=2, eps=1e-3, start=6)
    for it in range(1, 6):
        assert not c.update(it, (0.5, 0.0, 0.1))
    assert not c.update(6, (0.5, 0.0, 0.1))  # one change in the window
    assert c.update(7, (0.5, 0.0, 0.1))
    assert not c.update(8, (0.6, 0.0, 0.1))  # CL changed by 0.1


@pytest.fixture
def case_dir(tmp_path):
    write_su2(tmp_path / "channel.su2", rectangle(6, 6, x=(0, 1), y=(0, 0.5)))
    return tmp_path


def test_cauchy_stops_a_run_before_the_residual_drop(case_dir):
    text = COUETTE.replace(
        "residual_drop = 9.0",
        "residual_drop = 14.0\n[solver.convergence]\n"
        'fields = ["cd"]\ncauchy_elements = 2\ncauchy_eps = 1e-4',
    )
    (case_dir / "couette.toml").write_text(text)
    result = run_case(case_dir / "couette.toml", console=False)
    assert result.converged
    log = (case_dir / "out" / "couette_test.log").read_text()
    assert "Cauchy criterion met on cd" in log
    assert "convergence criterion" in log.split("Summary")[1]


def test_convergence_field_must_be_a_coefficient(case_dir):
    (case_dir / "couette.toml").write_text(
        COUETTE + '\n[solver.convergence]\nfields = ["ct"]\n'
    )
    with pytest.raises(ConfigError, match="solver.convergence.fields"):
        load_case(case_dir / "couette.toml")

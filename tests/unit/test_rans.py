"""SA-neg model functions, wall distance, laminar regions, RANS case files."""

import jax
import numpy as np
import pytest

from pyaeroflow.config import resolve
from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import rectangle, split_boundary
from pyaeroflow.numerics.wall_distance import mesh_wall_distance, wall_distance
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import spalart_allmaras as sa
from pyaeroflow.run import run_case, turbulence_values


def test_sa_eddy_viscosity_conversions():
    chi = np.array([0.5, 3.0, 50.0])
    ratio = sa.eddy_viscosity_ratio(chi)
    np.testing.assert_allclose(sa.nu_tilde_from_eddy_viscosity(ratio), chi, rtol=1e-10)
    assert sa.eddy_viscosity_ratio(3.0) == pytest.approx(0.2104, abs=5e-4)
    assert float(sa.eddy_viscosity(-1.0, 1.0)) == 0.0


def test_sa_source_and_derivative_are_finite():
    nu = 1e-6
    for nt in (0.0, 1e-7, 1e-5, 1e-3, -1e-6, -5e-6):
        for omega in (0.0, 1e-3, 1e3):
            val = sa.source(nt, nu, omega, 1e-4, 1e-6)
            der = jax.grad(sa.source)(nt, nu, omega, 1e-4, 1e-6)
            assert np.isfinite(val) and np.isfinite(der)
    # Negative branch: the destruction term becomes a sink of negative values.
    assert float(sa.source(-1e-6, nu, 0.0, 1e-4, 0.0)) > 0.0
    # Production off in laminar regions: only destruction and c_b2 term remain.
    full = float(sa.source(1e-5, nu, 1e3, 1e-3, 0.0))
    off = float(sa.source(1e-5, nu, 1e3, 1e-3, 0.0, production_switch=0.0))
    assert off < full


def test_wall_distance_rectangle():
    mesh = rectangle(8, 6, x=(0.0, 2.0), y=(0.0, 1.0), cell_type="quad")
    d, nearest = mesh_wall_distance(mesh, mesh.boundary_faces["bottom"])
    np.testing.assert_allclose(d, mesh.cell_centroid[:, 1])
    assert np.all(np.isin(nearest, mesh.boundary_faces["bottom"]))
    both = np.concatenate([mesh.boundary_faces[n] for n in ("bottom", "top")])
    d2, _ = mesh_wall_distance(mesh, both)
    y = mesh.cell_centroid[:, 1]
    np.testing.assert_allclose(d2, np.minimum(y, 1.0 - y))
    # Exact point-segment distance beyond the segment ends.
    d3, i3 = wall_distance(
        [[3.0, 1.0]], [[[0.0, 0.0], [1.0, 0.0]], [[2.0, 0.0], [2.5, 0.0]]]
    )
    assert d3[0] == pytest.approx(np.hypot(0.5, 1.0)) and i3[0] == 1


def test_laminar_region_switch():
    mesh = rectangle(10, 4, x=(-0.5, 1.5), y=(0.0, 1.0), cell_type="quad")
    mesh = split_boundary(
        mesh, "bottom", {"sym": lambda x, y: x < 0.0, "plate": lambda x, y: x >= 0.0}
    )
    conditions = {
        "left": {"type": "velocity-inlet", "velocity": (1.0, 0.0)},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
        "top": {"type": "farfield", "velocity": (1.0, 0.0), "pressure": 0.0},
        "sym": {"type": "symmetry"},
        "plate": {"type": "wall"},
    }
    model = inc.IncompressibleModel(density=1.0, viscosity=1e-3, turbulence="sa-neg")
    problem = IncompressibleProblem(
        mesh, model, conditions, laminar_regions=[{"boundary": "plate", "x_max": 0.5}]
    )
    switch = np.asarray(problem.geom.production_switch)
    x = mesh.cell_centroid[:, 0]
    assert np.all(switch[x > 0.6] == 1.0) and np.all(
        switch[(x > 0.0) & (x < 0.4)] == 0.0
    )
    ramped = IncompressibleProblem(
        mesh,
        model,
        conditions,
        laminar_regions=[{"boundary": "plate", "x_max": 0.5, "ramp": 0.4}],
    )
    sw = np.asarray(ramped.geom.production_switch)
    inside = (x > 0.0) & (x < 0.5)
    expected = 1.0 - np.clip((0.5 - x[inside]) / 0.4, 0.0, 1.0)
    np.testing.assert_allclose(sw[inside], expected)
    assert problem.initial_state((1.0, 0.0)).shape == (mesh.n_cells, 4)
    with pytest.raises(ValueError, match="unknown boundary"):
        IncompressibleProblem(
            mesh, model, conditions, laminar_regions=[{"boundary": "x"}]
        )


def test_turbulence_value_forms():
    nu = 1.5e-5
    b = turbulence_values(
        {"turbulent_viscosity_ratio": 0.2104, "intensity": 0.01}, nu, 10.0
    )
    assert b.chi == pytest.approx(3.0, abs=2e-3) and b.kinetic_energy == pytest.approx(
        0.015
    )
    a = turbulence_values({"kinetic_energy": 0.015, "length_scale": 0.001}, nu, 10.0)
    assert a.viscosity_ratio == pytest.approx(0.09**0.25 * np.sqrt(0.015) * 0.001 / nu)
    assert a.intensity == pytest.approx(0.01)
    c = turbulence_values({"nu_tilde_ratio": 5.0}, nu, 10.0)
    assert c.chi == 5.0 and turbulence_values(None, nu, 10.0).chi == 3.0


RANS_MINIMAL = {
    "mesh": {"file": "m.su2"},
    "physics": {"model": "rans"},
    "fluid": {"density": 1.0, "viscosity": 1e-6},
    "reference": {"velocity": 1.0},
    "initial": {"velocity": [1.0, 0.0]},
    "boundary": {
        "left": {"type": "velocity-inlet", "velocity": [1.0, 0.0]},
        "right": {"type": "pressure-outlet"},
        "top": {"type": "farfield", "velocity": 1.0},
        "bottom": {"type": "wall"},
    },
}


def test_transition_by_alpha_interpolates_to_the_case_angle():
    raw = {
        **RANS_MINIMAL,
        "boundary": {
            **RANS_MINIMAL["boundary"],
            "top": {"type": "farfield", "velocity": 1.0, "angle_of_attack": 5.0},
        },
        "physics": {
            "model": "rans",
            "transition": {
                "mode": "prescribed",
                "boundary": "bottom",
                "ramp": 0.02,
                "by_alpha": [
                    {"alpha": 0.0, "upper": 0.5, "lower": 0.5},
                    {"alpha": 10.0, "upper": 0.1, "lower": 0.7},
                ],
            },
        },
    }
    regions = resolve(raw)["physics"]["transition"]["laminar_region"]
    assert [r["side"] for r in regions] == ["upper", "lower"]
    assert regions[0]["x_max"] == pytest.approx(0.3) and regions[1][
        "x_max"
    ] == pytest.approx(0.6)
    assert regions[0]["ramp"] == 0.02 and regions[0]["boundary"] == "bottom"
    raw["physics"]["transition"]["laminar_region"] = [{"boundary": "bottom"}]
    with pytest.raises(Exception, match="either laminar_region or by_alpha"):
        resolve(raw)


def test_rans_config_rules():
    cfg = resolve(RANS_MINIMAL)
    assert cfg["physics"]["turbulence"] == "sa-neg"
    assert cfg["solver"]["turbulence_coupling"] == "segregated"
    raw = {
        **RANS_MINIMAL,
        "initial": {
            "velocity": [1.0, 0.0],
            "turbulence": {"nu_tilde_ratio": 3, "intensity": 0.1},
        },
    }
    with pytest.raises(Exception, match="exactly one form"):
        resolve(raw)
    raw = {
        **RANS_MINIMAL,
        "physics": {"model": "rans", "transition": {"mode": "prescribed"}},
    }
    with pytest.raises(Exception, match="needs laminar_region"):
        resolve(raw)
    raw = {**RANS_MINIMAL, "physics": {"model": "laminar", "turbulence": "sa-neg"}}
    assert any("ignored" in n for n in resolve(raw).notices)


PLATE_TOML = """
[case]
name = "plate"
[mesh]
file = "plate.su2"
[physics]
model = "rans"
[physics.transition]
mode = "prescribed"
laminar_region = [{ boundary = "plate", side = "upper", x_max = 0.05 }]
[fluid]
density = 1.2
reynolds_number = 1.0e5
[reference]
velocity = 10.0
length = 1.0
[initial]
from_boundary = "left"
[boundary.left]
type = "velocity-inlet"
velocity = [10.0, 0.0]
turbulence = { kinetic_energy = 0.0015, length_scale = 0.01 }
[boundary.right]
type = "pressure-outlet"
[boundary.top]
type = "farfield"
velocity = 10.0
[boundary.symmetry]
type = "symmetry"
[boundary.plate]
type = "wall"
[solver]
residual_drop = 4.0
[output]
directory = "out"
"""


def test_run_case_rans(tmp_path):
    from pyaeroflow.cases.tmr_flat_plate import plate_mesh
    from pyaeroflow.mesh import write_su2

    write_su2(tmp_path / "plate.su2", plate_mesh(24, 6, 24, reynolds=1.0e5))
    (tmp_path / "plate.toml").write_text(PLATE_TOML)
    result = run_case(tmp_path / "plate.toml", console=False)
    assert result.converged and result.w.shape[1] == 4
    log = (tmp_path / "out" / "plate.log").read_text()
    assert "nu_tilde/nu" in log and "cells without SA production" in log
    assert "outside the NASA TMR range" in log  # k, l form gives nu_tilde/nu > 5
    assert "log(R_nu)   drop" in log  # four residual columns, then the measure
    history = (tmp_path / "out" / "plate.history.csv").read_text().splitlines()
    assert len(history[0].split(",")) == len(history[-1].split(","))
    wall = np.genfromtxt(
        tmp_path / "out" / "plate.wall_plate.csv", delimiter=",", names=True
    )
    assert "y_plus" in wall.dtype.names and wall["y_plus"].max() < 2.0
    assert "eddy_viscosity_ratio" in (tmp_path / "out" / "plate.vtu").read_text(
        errors="ignore"
    )


def _bcm_plate(tmp_path, transition="", output="", solver="", tail="", model="bcm"):
    """Run the BCM flat-plate case; ``transition`` / ``output`` add case keys."""
    from pyaeroflow.cases.tmr_flat_plate import plate_mesh
    from pyaeroflow.mesh import write_su2

    toml = (
        PLATE_TOML.replace(
            '[physics.transition]\nmode = "prescribed"\n'
            'laminar_region = [{ boundary = "plate", side = "upper", x_max = 0.05 }]\n',
            f'[physics.transition]\nmodel = "{model}"\n{transition}',
        )
        .replace(
            "turbulence = { kinetic_energy = 0.0015, length_scale = 0.01 }",
            # nu_t/nu = 2.8e-7 is nu_tilde/nu = 0.1, the laminar freestream BCM needs
            "turbulence = { turbulent_viscosity_ratio = 2.8e-7, intensity = 0.033 }",
        )
        .replace("reynolds_number = 1.0e5", "reynolds_number = 5.0e5")
        .replace("[solver]\n", '[numerics]\nlimiter = "none"\n[solver]\n')
        .replace(
            '[boundary.top]\ntype = "farfield"\nvelocity = 10.0\n',
            '[boundary.top]\ntype = "farfield"\nvelocity = 10.0\n'
            "turbulence = { turbulent_viscosity_ratio = 2.8e-7, intensity = 0.033 }\n",
        )
    )
    assert toml.count("intensity = 0.033") == 2
    write_su2(tmp_path / "plate.su2", plate_mesh(48, 12, 48, reynolds=5.0e5))
    if output:
        toml = toml.replace("[output]\n", f"[output]\n{output}")
    if solver:
        toml = toml.replace("[solver]\n", f"[solver]\n{solver}")
    (tmp_path / "plate.toml").write_text(toml + tail)
    return run_case(tmp_path / "plate.toml", console=False)


def test_run_case_bcm_transition(tmp_path):
    """The BCM model through the case file: Tu from the inflow table, log
    rows, intermittency in the VTU and the wall CSV, transition location in
    the summary."""
    result = _bcm_plate(tmp_path)
    assert result.converged
    log = (tmp_path / "out" / "plate.log").read_text()
    assert "bcm (SA-noft2), Tu = 3.3 %" in log and "Re_theta_t = 198" in log
    assert "transition 'plate' upper" in log and "x = 0." in log
    assert "with the BCM transition model" not in log  # nu_tilde/nu = 0.1 is fine
    wall = np.genfromtxt(
        tmp_path / "out" / "plate.wall_plate.csv", delimiter=",", names=True
    )
    g = wall["nu_t_ratio_max"]
    assert g[wall["x"] < 0.2].max() < 0.01 and g[wall["x"] > 0.6].min() > 1.0
    assert "intermittency" in (tmp_path / "out" / "plate.vtu").read_text(
        errors="ignore"
    )


def test_bcm_relaxed_intermittency_and_diagnostics(tmp_path):
    """Under-relaxed gamma converges to the unrelaxed transition location and
    the diagnostics columns reach the history CSV."""
    ref = tmp_path / "ref"
    ref.mkdir()
    _bcm_plate(ref)
    run = tmp_path / "relaxed"
    run.mkdir()
    result = _bcm_plate(run, "relaxation = 0.3\n", "transition_diagnostics = true\n")
    assert result.converged
    log = (run / "out" / "plate.log").read_text()
    assert "under-relaxed, omega = 0.3" in log
    h = np.genfromtxt(run / "out" / "plate.history.csv", delimiter=",", names=True)
    for col in ("gamma_cells", "gamma_jumped", "gamma_dmax_x", "rnu_max_y"):
        assert col in h.dtype.names
    assert h["gamma_cells"][-1] > 0 and h["gamma_dmax"][-1] < 1e-3
    w_ref = np.genfromtxt(
        ref / "out" / "plate.wall_plate.csv", delimiter=",", names=True
    )
    w_rel = np.genfromtxt(
        run / "out" / "plate.wall_plate.csv", delimiter=",", names=True
    )
    np.testing.assert_allclose(
        w_rel["nu_t_ratio_max"], w_ref["nu_t_ratio_max"], rtol=2e-2, atol=1e-3
    )


def test_bcm_nu_tilde_step_limit(tmp_path):
    """A nu_tilde step cap of 1 still converges the BCM plate to the same state."""
    ref = tmp_path / "ref"
    ref.mkdir()
    _bcm_plate(ref)
    run = tmp_path / "limited"
    run.mkdir()
    result = _bcm_plate(run, solver="nu_tilde_step_limit = 1.0\n")
    assert result.converged
    assert "capped at 1 x" in (run / "out" / "plate.log").read_text()
    kw = {"delimiter": ",", "names": True}
    a = np.genfromtxt(ref / "out" / "plate.wall_plate.csv", **kw)
    b = np.genfromtxt(run / "out" / "plate.wall_plate.csv", **kw)
    np.testing.assert_allclose(
        b["nu_t_ratio_max"], a["nu_t_ratio_max"], rtol=2e-2, atol=1e-3
    )


def test_bcm_relaxation_config_rules():
    plate = {**RANS_MINIMAL, "physics": {"model": "rans", "transition": {}}}
    plate["physics"]["transition"] = {"relaxation": 0.5}
    with pytest.raises(Exception, match="needs model = bcm"):
        resolve(plate)
    plate["physics"]["transition"] = {"model": "bcm", "relaxation": 1.5}
    with pytest.raises(Exception, match=r"must be in \(0, 1\]"):
        resolve(plate)


def test_bcm_config_rules():
    raw = {**RANS_MINIMAL, "physics": {"model": "rans", "transition": {"model": "bcm"}}}
    with pytest.raises(Exception, match="turbulence_intensity is required"):
        resolve(raw)
    raw["physics"]["transition"]["turbulence_intensity"] = 0.001
    assert resolve(raw)["physics"]["transition"]["turbulence_intensity"] == 0.001
    raw = {
        **RANS_MINIMAL,
        "physics": {"model": "laminar", "transition": {"model": "bcm"}},
    }
    with pytest.raises(Exception, match="needs physics.model = rans"):
        resolve(raw)


def test_unsteady_nu_tilde_step_limit(tmp_path, monkeypatch):
    """solver.nu_tilde_step_limit reaches the inner iterations of a BDF2 run."""
    import pyaeroflow.driver as driver
    from pyaeroflow.cases.tmr_flat_plate import plate_mesh
    from pyaeroflow.mesh import write_su2

    seen = {}
    original = driver.solve_unsteady

    def spy(*args, **kwargs):
        seen["limit_step"] = kwargs.get("limit_step")
        return original(*args, **kwargs)

    monkeypatch.setattr(driver, "solve_unsteady", spy)
    write_su2(tmp_path / "plate.su2", plate_mesh(24, 6, 24, reynolds=1.0e5))
    toml = (
        PLATE_TOML.replace("[solver]\n", "[solver]\nnu_tilde_step_limit = 1.0\n")
        + '[time]\nmode = "unsteady"\ntime_step = 0.01\nend_time = 0.03\n'
    )
    (tmp_path / "plate.toml").write_text(toml)
    result = run_case(tmp_path / "plate.toml", console=False)
    assert np.isfinite(result.w).all()
    log = (tmp_path / "out" / "plate.log").read_text()
    assert "capped at 1 x max(|nu_tilde|, nu) per cell (inner iterations)" in log
    limit = seen["limit_step"]
    assert limit is not None
    # cap f max(|nu_tilde|, nu) per cell, nu = 1 / Re non-dimensional
    u = np.zeros((3, 4))
    u[:, 3] = [0.0, 1.0, -4.0]
    du = np.full((3, 4), 7.0)
    du[:, 3] = [1e-9, 5.0, -10.0]
    out = limit(u, du)
    np.testing.assert_array_equal(out[:, :3], 7.0)
    np.testing.assert_allclose(out[:, 3], [1e-9, 1.0, -4.0])


def test_unsteady_gamma_diagnostics(tmp_path):
    """transition_diagnostics adds the gamma statistics to the unsteady history."""
    tail = '[time]\nmode = "unsteady"\ntime_step = 0.01\nend_time = 0.03\n'
    _bcm_plate(tmp_path, output="transition_diagnostics = true\n", tail=tail)
    h = np.genfromtxt(tmp_path / "out" / "plate.history.csv", delimiter=",", names=True)
    assert len(h["step"]) == 3
    for col in ("gamma_cells", "gamma_area", "gamma_jumped", "gamma_dmax_y"):
        assert col in h.dtype.names
    assert "rnu_max" not in h.dtype.names
    assert (h["gamma_cells"] > 0).all() and (h["gamma_area"] > 0).all()
    assert (h["gamma_dmax"] >= 0).all() and (h["gamma_dmax"] <= 1).all()


def test_restart_log_marks_the_initial_turbulence_values(tmp_path):
    """With a restart file the initial turbulence values are not the start
    values: the log says so instead of listing (and warning about) the defaults."""
    _bcm_plate(tmp_path)
    case = tmp_path / "plate.toml"
    case.write_text(
        case.read_text().replace(
            '[initial]\nfrom_boundary = "left"',
            '[initial]\nfrom_boundary = "left"\nrestart = "out/plate.restart.npz"',
        )
    )
    run_case(case, console=False)
    log = (tmp_path / "out" / "plate.log").read_text()
    restarted = log[log.index("RESTART") :]
    assert "initial     from the restart file (default " in restarted
    assert "initial: nu_tilde/nu" not in restarted

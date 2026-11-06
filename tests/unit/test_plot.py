"""pyaeroflow plot: readers, plots and command line (matplotlib optional)."""

import numpy as np
import pytest

from pyaeroflow.cli import main
from pyaeroflow.plotting.readers import (
    builtin_names,
    read_history,
    read_log,
    read_reference,
    read_table,
)

STEADY = """\
pyaeroflow 1.4.0
  step control            unsteady residual (HANIM): quadratic under-relaxation
  iter  log(R_rho) log(R_rhou) log(R_rhov) log(R_rhoE)   drop       CFL     NK  GMRES         CL         CD         CM   wall[s]  note
     0     -3.7200     -3.7188     -4.5801     -3.0757   0.00  1.00e+01                 -0.00000   -0.00000    0.00000       5.2
     1     -3.8610     -3.9307     -3.8256     -3.1052   0.00  1.00e+01                  0.10279    0.19536    0.02347      20.6  limiter released
     2     -3.9344     -4.0806     -3.9606     -3.2234   0.13  2.00e+01                  0.10097    0.16424    0.02508      20.7  step 1/2
    58     -6.4440     -6.1161     -6.8396     -5.7633   2.40  1.00e+03     NK      8    0.33857    0.02236    0.03415      60.2
    59     -6.6211     -6.2896     -7.0658     -5.9249   2.57  2.54e+03     NK      5    0.33878    0.02238    0.03419      61.3
    60     -6.9685
"""  # noqa: E501

UNSTEADY = """\
  step        time inner  drop    log(R_p)    log(R_u)    log(R_v)  GMRES         CL         CD         CM   wall[s]  note
     1 5.00000e-02     3   4.7     -7.9131     -7.7630     -7.4476      9    0.00000   -5.48844    5.49802       4.8
  iter    log(R_p)    log(R_u)    log(R_v)   drop       CFL     NK  GMRES
     1     -3.0516     -0.3623     -3.0793   0.00  1.00e+01     NK      3
  step        time inner  drop    log(R_p)    log(R_u)    log(R_v)  GMRES         CL         CD         CM   wall[s]  note
     2 1.00000e-01     3   3.8     -7.3579     -7.2507     -7.1352      9    0.00000   -2.02190    2.05068       4.8  inner limit
"""  # noqa: E501


def test_read_log_fixed_width_columns(tmp_path):
    (tmp_path / "c.log").write_text(STEADY)
    log = read_log(tmp_path / "c.log")
    assert log.mode == "steady"
    assert log.residuals == ["R_rho", "R_rhou", "R_rhov", "R_rhoE"]
    c = log.columns
    assert list(c["iter"]) == [0, 1, 2, 58, 59]  # the cut row is left out
    assert list(c["NK"]) == [0, 0, 0, 1, 1] and list(c["GMRES"]) == [0, 0, 0, 8, 5]
    assert c["CL"][3] == pytest.approx(0.33857) and c["wall"][4] == 61.3
    assert log.notes == ["", "limiter released", "step 1/2", "", ""]


def test_read_log_unsteady_skips_inner_tables(tmp_path):
    (tmp_path / "u.log").write_text(UNSTEADY)
    log = read_log(tmp_path / "u.log")
    assert log.mode == "unsteady" and list(log.columns["step"]) == [1, 2]
    assert list(log.columns["time"]) == [0.05, 0.1] and log.notes[1] == "inner limit"


def test_read_history_as_log(tmp_path):
    (tmp_path / "u.history.csv").write_text(
        "step,time,residual_p,residual_u,inner_iterations,linear_iterations,CL,time_s\n"
        "1,0.05,1e-8,1e-6,3,9,0.1,4.8\n2,0.1,1e-7,1e-5,4,8,0.2,5.0\n"
    )
    run = read_history(tmp_path / "u.history.csv")
    assert run.mode == "unsteady" and run.residuals == ["R_p", "R_u"]
    assert np.allclose(run.columns["R_p"], [-8, -7]) and list(run.columns["GMRES"]) == [
        9,
        8,
    ]


def test_read_table_formats(tmp_path):
    (tmp_path / "t.dat").write_text(
        '# comment\nvariables="x/c","cp"\nzone, t="alpha=0"\n0 1\n1 0.2\n'
        'zone, t="alpha=10"\n0 -2\n0.5 -1\n1 0\n'
    )
    zones = read_table(tmp_path / "t.dat")
    assert [z.title for z in zones] == ["alpha=0", "alpha=10"]
    assert list(zones[1].get("x")) == [0, 0.5, 1]
    (tmp_path / "x.pol").write_text(
        " XFOIL Version 6.99\n\n Calculated polar for: E387\n\n"
        "  alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr\n"
        " ------ -------- --------- --------- -------- -------- --------\n"
        "  0.000   0.4000   0.00900   0.00300  -0.0800   0.6000   0.9000\n"
        "  2.000   0.6000   0.01000   0.00350  -0.0810   0.5000   1.0000\n"
    )
    pol = read_table(tmp_path / "x.pol")[0]
    assert list(pol.get("cl")) == [0.4, 0.6] and list(pol.get("xtr_up")) == [0.6, 0.5]
    (tmp_path / "cp.txt").write_text(
        "#    x        y        Cp\n 1.0 0.0 0.2\n 0.5 0.05 -0.4\n"
    )
    cp = read_table(tmp_path / "cp.txt")[0]
    assert list(cp.get("cp")) == [0.2, -0.4]


def test_builtin_references():
    names = builtin_names()
    assert "tmr_naca0012_cp_cfl3d_sa" in names and "e387_tm4062_re200k" in names
    for name in names:
        assert read_reference(f"builtin:{name}")
    zone = read_reference("builtin:tmr_naca0012_cp_cfl3d_sa:alpha=10@CFL3D")
    assert len(zone) == 1 and zone[0].label == "CFL3D alpha=10"
    with pytest.raises(ValueError, match="unknown built-in"):
        read_reference("builtin:nope")


# -- plots (need matplotlib) --------------------------------------------------------


def _wall_csv(path):
    """Closed contour TE -> lower -> LE -> upper with tangent-signed Cf."""
    t = np.linspace(0.0, 2.0 * np.pi, 81)
    x, y = 0.5 + 0.5 * np.cos(t), -0.06 * np.sin(t)  # lower half first
    cp = 1.0 - 4.0 * np.sin(t) ** 2
    cf = np.where(y < 0, -0.004, 0.004)  # attached flow, raw sign
    cf[(y > 0) & (x > 0.6)] = -0.001  # separated upper aft part
    rows = ["x,y,p,cp,tau_x,tau_y,cf"] + [
        f"{a},{b},0,{c},0,0,{f}" for a, b, c, f in zip(x, y, cp, cf, strict=True)
    ]
    path.write_text("\n".join(rows) + "\n")


def test_wall_flow_aligned_cf_and_separation(tmp_path):
    pytest.importorskip("matplotlib")
    from pyaeroflow.plotting.wall import Wall, plot_wall, zero_crossings

    _wall_csv(tmp_path / "c.wall_airfoil.csv")
    wall = Wall(tmp_path / "c.wall_airfoil.csv")
    assert wall.closed and wall.c["y"][wall.upper].mean() > 0
    cf = wall.values("cf")
    assert np.all(cf[wall.lower][1:-1] > 0)  # attached lower side made positive
    xs = wall.xc(wall.upper)
    order = np.argsort(xs)
    z = zero_crossings(xs[order], cf[wall.upper][order])
    z = z[z > 0.005]  # the leading-edge point itself
    assert z.size == 1 and 0.55 < z[0] < 0.65
    out = plot_wall(
        [tmp_path / "c.wall_airfoil.csv"],
        refs=["builtin:tmr_naca0012_cp_cfl3d_sa:alpha=0"],
    )
    assert out.name == "c_wall.png" and out.exists()
    assert plot_wall(
        [tmp_path / "c.wall_airfoil.csv"], tmp_path / "a", x="arc"
    ).exists()


def test_plot_log_and_cli(tmp_path):
    pytest.importorskip("matplotlib")
    (tmp_path / "c.log").write_text(STEADY)
    (tmp_path / "t.toml").write_text("[targets.CL]\nvalue = 0.34\nrtol = 0.01\n")
    assert main(["plot", str(tmp_path / "c.log"), "-t", str(tmp_path / "t.toml")]) == 0
    for suffix in ("residuals", "solver", "coeffs", "diffs"):
        assert (tmp_path / f"c_{suffix}.png").exists()
    (tmp_path / "u.log").write_text(UNSTEADY)
    assert main(["plot", "log", str(tmp_path / "u.log"), "--average-from", "0"]) == 0
    assert main(["plot", str(tmp_path / "t.toml")]) == 1  # unknown kind


def test_plot_polar_cm_sign(tmp_path):
    pytest.importorskip("matplotlib")
    from pyaeroflow.plotting.polar import _columns, plot_polar

    (tmp_path / "e.polar.csv").write_text(
        "alpha,CL,CD,CD_pressure,CD_viscous,CM,iterations,converged\n"
        "0,0.4,0.009,0.003,0.006,0.08,400,1\n2,0.6,0.010,0.004,0.006,0.079,2000,0\n"
    )
    c = _columns(read_table(tmp_path / "e.polar.csv")[0], True)
    assert np.allclose(c["cm"], [-0.08, -0.079]) and np.allclose(
        c["ld"], [0.4 / 0.009, 60]
    )
    out = plot_polar(
        [tmp_path / "e.polar.csv"],
        refs=["builtin:e387_xfoil_re200k:N_crit 9"],
        split_cd=True,
    )
    assert out.exists()


def test_probe_spectrum_and_strouhal(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from pyaeroflow.plotting.probes import dominant_frequency, plot_probes

    t = np.arange(0.0, 4.0, 0.01)
    f, _, _ = dominant_frequency(t, np.sin(2 * np.pi * 2.3 * t))
    assert f == pytest.approx(2.3, abs=0.05)
    rows = ["step,time,probe1_p,probe1_u,probe1_v"] + [
        f"{i + 1},{ti},0,1,{np.sin(2 * np.pi * 2.5 * ti)}" for i, ti in enumerate(t)
    ]
    (tmp_path / "c.probes.csv").write_text("\n".join(rows) + "\n")
    (tmp_path / "c.resolved.toml").write_text(
        "[reference]\nlength = 0.2\nvelocity = 10.0\n"
    )
    files = plot_probes(tmp_path / "c.probes.csv", spectrum=["v"], x_from=1.0)
    assert [p.name for p in files] == ["c_probes.png", "c_spectrum.png"]
    assert "St = 0.0500" in capsys.readouterr().out
    with pytest.raises(ValueError, match="no column"):
        plot_probes(tmp_path / "c.probes.csv", spectrum=["w"])


def test_cli_lists_builtin_references(capsys):
    assert main(["plot", "refs"]) == 0
    assert "tmr_naca0012_cp_gregory" in capsys.readouterr().out

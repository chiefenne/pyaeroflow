"""Configurable VTU field output (D59): registry, schedule, wall and residual fields."""

import base64
import re
from pathlib import Path

import numpy as np
import pytest
from test_rans import _bcm_plate
from test_run import COUETTE, UNSTEADY

from pyaeroflow.config import resolve
from pyaeroflow.io.fields import (
    FIELDS,
    Capabilities,
    available,
    catalogue_table,
    default_fields,
    resolve_fields,
)
from pyaeroflow.mesh import rectangle, write_su2
from pyaeroflow.run import FieldWriter, run_case

LAMINAR = Capabilities("laminar", "none", True, True, True)
SA = Capabilities("rans", "none", True, True, True)
AFT = Capabilities("rans", "aft", True, True, True)
EULER = Capabilities("euler", "none", True, True, True)
EXPLICIT = Capabilities("laminar", "none", True, False, True)
UNSTEADY_CAPS = Capabilities("laminar", "none", False, False, True)


def vtu_arrays(path):
    """Cell and mesh arrays of a binary VTU file written by ``write_vtu``."""
    text = Path(path).read_text(errors="ignore")
    out = {}
    pattern = (
        r'<DataArray type="(\w+)" Name="(\w+)" NumberOfComponents="(\d)" '
        r'format="binary">\n(.*?)\n</DataArray>'
    )
    for m in re.finditer(pattern, text, re.S):
        dtype = {"Float64": np.float64, "Int64": np.int64, "UInt8": np.uint8}[m[1]]
        a = np.frombuffer(base64.b64decode(m[4])[8:], dtype=dtype)
        out[m[2]] = a.reshape(-1, int(m[3])) if m[3] != "1" else a
    return out


def test_registry_names_are_unique_and_documented():
    names = [f.name for f in FIELDS]
    assert len(names) == len(set(names))
    assert all(f.unit and f.description and f.needs for f in FIELDS)


def test_default_selection_is_todays_output():
    assert default_fields(LAMINAR) == ["pressure", "velocity", "velocity_magnitude"]
    assert default_fields(SA) == [
        "pressure",
        "velocity",
        "velocity_magnitude",
        "nu_tilde",
        "eddy_viscosity_ratio",
    ]
    assert default_fields(AFT)[-2:] == ["intermittency", "amplification_factor"]


def test_availability_by_model_combination():
    assert "residual_gamma" in available(AFT) and "residual_gamma" not in available(SA)
    assert "eddy_viscosity" not in available(LAMINAR)
    assert "kinematic_viscosity" not in available(EULER)
    assert "cp_wall" in available(EULER) and "cf" not in available(EULER)
    assert "y_plus" in available(SA) and "y_plus" not in available(LAMINAR)
    assert not [n for n in available(EXPLICIT) if n.startswith("residual")]
    assert not [n for n in available(UNSTEADY_CAPS) if n.startswith("residual")]
    no_walls = SA._replace(walls=False)
    assert not {"cf", "cp_wall", "tau_wall_x", "y_plus"} & set(available(no_walls))


def test_resolve_fields_messages():
    names, problems = resolve_fields(["velocity", "pressure"], LAMINAR)
    assert names == ["pressure", "velocity"] and not problems  # registry order
    assert resolve_fields(["all"], SA)[0] == available(SA)
    _, problems = resolve_fields(["presure"], LAMINAR)
    assert "unknown field 'presure' (did you mean 'pressure'?)" in problems[0]
    _, problems = resolve_fields(["nu_tilde"], LAMINAR)
    assert "'nu_tilde' needs rans" in problems[0]
    _, problems = resolve_fields(["residual_gamma"], SA)
    assert "needs steady, implicit, transition.model = aft" in problems[0]
    assert "model rans, transition none, steady, implicit" in problems[0]


def test_config_interval_is_an_alias(tmp_path):
    base = {
        "mesh": {"file": "m.su2"},
        "physics": {"model": "laminar"},
        "fluid": {"density": 1.0, "viscosity": 1.0},
        "reference": {"velocity": 1.0, "length": 1.0},
        "initial": {"velocity": [0.0, 0.0]},
        "boundary": {
            "left": {"type": "periodic", "partner": "right", "translation": [1, 0]},
            "bottom": {"type": "wall"},
            "top": {"type": "wall"},
        },
    }
    out = resolve({**base, "output": {"interval": 5}})["output"]
    assert out["fields_interval"] == 5 and out["interval"] == 5
    out = resolve({**base, "output": {"fields_interval": 7}})["output"]
    assert out["fields_interval"] == 7 and out["interval"] == 7
    with pytest.raises(Exception, match="set only one"):
        resolve({**base, "output": {"interval": 5, "fields_interval": 7}})
    with pytest.raises(Exception, match="did you mean 'vorticity'"):
        resolve({**base, "output": {"fields": ["vortcity"]}})
    with pytest.raises(Exception, match="fields_at"):
        resolve({**base, "output": {"fields_at": [0]}})


def test_field_writer_schedule_waits_for_a_good_step(tmp_path):
    output = {
        "fields": ["pressure"],
        "fields_interval": 4,
        "fields_at": [3, 5, 50],
        "fields_final": True,
    }
    writer = FieldWriter(tmp_path, "c", None, None, None, None, output)
    due = []
    for number in range(1, 13):
        if writer.due(number, good=number != 8):  # step 8 is rejected
            due.append(number)
    assert due == [3, 4, 5, 9, 12]  # the multiple of 4 at 8 waits for 9
    resumed = FieldWriter(tmp_path, "c", None, None, None, None, output, after=4)
    assert resumed.at == [5, 50] and resumed.bucket == 1


def test_documentation_table_is_generated_from_the_registry():
    docs = (Path(__file__).parents[2] / "docs" / "configuration.md").read_text()
    assert catalogue_table() in docs


def test_laminar_default_fields_and_unsteady_schedule(tmp_path):
    mesh = rectangle(4, 8, x=(0.0, 1.0), y=(0.0, 0.5), cell_type="quad")
    write_su2(tmp_path / "channel.su2", mesh)
    (tmp_path / "couette.toml").write_text(
        UNSTEADY.replace(
            'directory = "out"',
            'directory = "out"\nfields_interval = 4\nfields_at = [3]',
        )
    )
    result = run_case(tmp_path / "couette.toml", console=False)
    out = tmp_path / "out"
    assert result.history["step"][-1] == 10
    series = sorted(p.name for p in out.glob("couette_unsteady_0*.vtu"))
    assert series == [f"couette_unsteady_{n:06d}.vtu" for n in (3, 4, 8, 10)]
    assert set(vtu_arrays(out / "couette_unsteady.vtu")) - {
        "Points",
        "connectivity",
        "offsets",
        "types",
    } == {"pressure", "velocity", "velocity_magnitude"}


def test_fields_final_false_writes_only_the_series(tmp_path):
    mesh = rectangle(6, 6, x=(0.0, 1.0), y=(0.0, 0.5), cell_type="quad")
    write_su2(tmp_path / "channel.su2", mesh)
    (tmp_path / "couette.toml").write_text(
        COUETTE.replace(
            'directory = "out"',
            'directory = "out"\nfields_at = [2]\nfields_final = false',
        )
    )
    run_case(tmp_path / "couette.toml", console=False)
    out = tmp_path / "out"
    assert (out / "couette_test_000002.vtu").exists()
    assert (out / "couette_test.pvd").exists()
    assert not (out / "couette_test.vtu").exists()


def test_couette_wall_fields_are_flow_aligned(tmp_path):
    """cf is positive when the wall shear acts along the flow (+x here): the
    bottom wall is dragged forward by the fluid, the moving top wall is held back."""
    mesh = rectangle(6, 6, x=(0.0, 1.0), y=(0.0, 0.5), cell_type="quad")
    write_su2(tmp_path / "channel.su2", mesh)
    (tmp_path / "couette.toml").write_text(
        COUETTE.replace(
            'directory = "out"',
            'directory = "out"\nfields = ["cf", "cp_wall", "tau_wall_x"]',
        )
    )
    result = run_case(tmp_path / "couette.toml", console=False)
    a = vtu_arrays(tmp_path / "out" / "couette_test.vtu")
    y = result.mesh.cell_centroid[:, 1]
    bottom, top = y < 0.5 / 6, y > 0.5 - 0.5 / 6
    assert np.all(a["cf"][bottom] > 0) and np.all(a["cf"][top] < 0)
    assert np.all(a["cf"][~bottom & ~top] == 0.0)
    assert np.allclose(a["tau_wall_x"][bottom], -a["tau_wall_x"][top])


def test_aft_plate_fields_residuals_and_header(tmp_path):
    """All fields of SA-AFT on the flat plate: wall fields only in the wall
    cells, cell residuals per cell volume, schedule files, size line in the log."""
    result = _bcm_plate(
        tmp_path,
        "critical_amplification = 3.0\n",
        output='fields = ["all"]\nfields_interval = 100\nfields_at = [3]\n',
        model="aft",
    )
    assert result.converged
    out = tmp_path / "out"
    a = vtu_arrays(out / "plate.vtu")
    assert set(available(AFT)) <= set(a)
    n_wall_faces = len(result.mesh.boundary_faces["plate"])
    for key in ("cf", "y_plus", "cp_wall", "tau_wall_x"):
        assert np.count_nonzero(a[key]) <= n_wall_faces
        assert np.isfinite(a[key]).all()
    assert a["cf"].max() > 0 and a["y_plus"].max() < 5.0
    for key in ("residual_p", "residual_nu_tilde", "residual_gamma", "residual_n"):
        assert np.isfinite(a[key]).all() and np.any(a[key] != 0.0)
    assert np.abs(a["residual_p"]).max() < 1e-4  # converged state
    assert (out / "plate_000003.vtu").exists() and (out / "plate_000100.vtu").exists()
    pvd = (out / "plate.pvd").read_text()
    assert 'timestep="3"' in pvd and 'timestep="100"' in pvd
    log = (out / "plate.log").read_text()
    row = next(line for line in log.splitlines() if "field files" in line)
    assert "every 100 iterations, at 3, final" in row
    megabytes = float(re.search(r"([\d.]+) MB each", row)[1])
    actual = (out / "plate.vtu").stat().st_size / 1e6
    assert abs(megabytes - actual) < 0.06 * actual


def test_explicit_solver_has_no_residual_fields(tmp_path):
    mesh = rectangle(6, 6, x=(0.0, 1.0), y=(0.0, 0.5), cell_type="quad")
    write_su2(tmp_path / "channel.su2", mesh)
    toml = COUETTE.replace(
        'directory = "out"', 'directory = "out"\nfields = ["residual_p"]'
    ).replace("[solver]\n", '[solver]\nmethod = "explicit-rk"\n')
    (tmp_path / "couette.toml").write_text(toml)
    with pytest.raises(Exception, match="'residual_p' needs steady, implicit"):
        run_case(tmp_path / "couette.toml", console=False)

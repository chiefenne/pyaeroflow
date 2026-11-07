"""Derived VTU fields (gradient based, freestream based) and point probes."""

import numpy as np
import pytest
from test_fields import vtu_arrays
from test_run import COUETTE, UNSTEADY

from pyaeroflow.config import resolve
from pyaeroflow.io.fields import FieldContext, Freestream, compute_fields
from pyaeroflow.mesh import rectangle, write_su2
from pyaeroflow.run import Scaling, run_case


class _Model:
    density = 1.0


class _Problem:
    """Stub with a prescribed velocity gradient (du/dx, du/dy, dv/dx, dv/dy)."""

    compressible = False
    model = _Model()

    def __init__(self, ux, uy, vx, vy, n=3):
        self.g = np.zeros((n, 3, 2))
        self.g[:, 1] = (ux, uy)
        self.g[:, 2] = (vx, vy)

    def gradient(self, w):
        return self.g

    def vorticity(self, w, grad):
        return grad[:, 2, 0] - grad[:, 1, 1]


def _fields(problem, names, w=None, freestream=None):
    w = np.zeros((problem.g.shape[0], 3)) if w is None else w
    ctx = FieldContext(problem, w, Scaling(1.0, 1.0, 1.0), freestream=freestream)
    return compute_fields(names, ctx)


GRADIENT_FIELDS = ["vorticity", "q_criterion", "strain_rate", "divergence"]


def test_solid_body_rotation():
    # u = -W y, v = W x: vorticity 2W, Q = W^2, no strain, no divergence
    f = _fields(_Problem(0.0, -2.0, 2.0, 0.0), GRADIENT_FIELDS)
    assert np.allclose(f["vorticity"], 4.0) and np.allclose(f["q_criterion"], 4.0)
    assert np.allclose(f["strain_rate"], 0.0) and np.allclose(f["divergence"], 0.0)


def test_pure_shear_and_straining():
    f = _fields(_Problem(0.0, 3.0, 0.0, 0.0), GRADIENT_FIELDS)  # u = 3 y
    assert np.allclose(f["q_criterion"], 0.0) and np.allclose(f["strain_rate"], 3.0)
    f = _fields(_Problem(1.0, 0.0, 0.0, -1.0), GRADIENT_FIELDS)  # stagnation flow
    assert np.allclose(f["q_criterion"], -1.0) and np.allclose(f["strain_rate"], 2.0)
    f = _fields(_Problem(1.0, 0.0, 0.0, 1.0), GRADIENT_FIELDS)  # source
    assert np.allclose(f["divergence"], 2.0)


def test_cp_and_total_pressure_loss_incompressible():
    w = np.array([[0.0, 1.0, 0.0], [-0.5, 0.0, 0.0], [0.2, 0.5, 0.0]])
    fs = Freestream(pressure=0.0, density=1.0, speed=1.0)
    f = _fields(_Problem(0, 0, 0, 0), ["cp", "total_pressure_loss"], w, fs)
    assert np.allclose(f["cp"], [0.0, -1.0, 0.4])
    # p0_inf = 0.5; p0 = 0.5, -0.5, 0.325
    assert np.allclose(f["total_pressure_loss"], [0.0, 2.0, 0.35])


def test_compressible_freestream_total_pressure():
    fs = Freestream(1.0 / 1.4, 1.0, 0.5, 1.0, 0.5, 1.4, True)  # c = 1, M = 0.5
    assert np.isclose(fs.mach, 0.5)
    assert np.isclose(fs.total_pressure / fs.pressure, (1 + 0.2 * 0.25) ** 3.5)


def _couette(tmp_path, text, n=(4, 16)):
    mesh = rectangle(*n, x=(0.0, 1.0), y=(0.0, 0.5), cell_type="quad")
    write_su2(tmp_path / "channel.su2", mesh)
    (tmp_path / "couette.toml").write_text(text)
    return run_case(tmp_path / "couette.toml", console=False)


def test_couette_derived_fields(tmp_path):
    """u = U y / H, U = 3 m/s, H = 0.5 m: strain 6 1/s, Q = 0, loss 1 - (u/U)^2."""
    text = COUETTE.replace(
        'directory = "out"',
        'directory = "out"\nfields = ["velocity", "cp", "total_pressure", '
        '"total_pressure_loss", "vorticity", "q_criterion", "strain_rate", '
        '"divergence"]',
    )
    _couette(tmp_path, text)
    a = vtu_arrays(tmp_path / "out" / "couette_test.vtu")
    u = a["velocity"][:, 0]
    assert np.allclose(a["strain_rate"], 6.0, rtol=1e-6)
    assert np.allclose(a["vorticity"], -6.0, rtol=1e-6)
    assert np.allclose(a["q_criterion"], 0.0, atol=1e-6)
    assert np.allclose(a["divergence"], 0.0, atol=1e-6)
    assert np.allclose(a["total_pressure"], 0.5 * 2.0 * u**2 + a["cp"] * 9.0)
    p0 = 0.5 * 2.0 * u**2 + a["cp"] * 9.0
    assert np.allclose(a["total_pressure_loss"], (9.0 - p0) / 9.0)


def test_steady_probes_interpolate_linear_profile(tmp_path):
    text = COUETTE.replace(
        'directory = "out"',
        'directory = "out"\nprobes = [[0.5, 0.25], [0.3, 0.1], [0.02, 0.01]]\n'
        "probe_interval = 2",
    )
    result = _couette(tmp_path, text)
    out = tmp_path / "out"
    rows = (out / "couette_test.probes.csv").read_text().splitlines()
    assert rows[0] == (
        "iteration," + ",".join(f"probe{i}_{c}" for i in (1, 2, 3) for c in "puv")
    )
    last = np.array(rows[-1].split(","), dtype=float)
    assert int(last[0]) % 2 == 0 and int(last[0]) <= result.history["iteration"][-1]
    assert np.allclose(last[[2, 5, 8]], [1.5, 0.6, 0.06], atol=1e-5)
    assert np.allclose(last[[3, 6, 9]], 0.0, atol=1e-8)
    log = (out / "couette_test.log").read_text()
    assert "probes: 3 points (p, u, v) every 2 iterations" in log


def test_unsteady_probes_and_outside_point(tmp_path):
    text = UNSTEADY.replace(
        'directory = "out"', 'directory = "out"\nprobes = [[0.5, 0.25]]'
    )
    _couette(tmp_path, text)
    rows = (tmp_path / "out" / "couette_unsteady.probes.csv").read_text().splitlines()
    assert rows[0] == "step,time,probe1_p,probe1_u,probe1_v" and len(rows) == 11
    assert np.isclose(float(rows[-1].split(",")[1]), 0.5)
    bad = UNSTEADY.replace(
        'directory = "out"', 'directory = "out"\nprobes = [[2.0, 0.25]]'
    )
    with pytest.raises(ValueError, match=r"outside the mesh: probe 1 \(2, 0.25\) m"):
        _couette(tmp_path, bad)


def test_probes_config_type():
    base = {
        "mesh": {"file": "m.su2"},
        "physics": {"model": "laminar"},
        "fluid": {"density": 1.0, "viscosity": 1.0},
        "reference": {"velocity": 1.0, "length": 1.0},
        "initial": {"velocity": [0.0, 0.0]},
        "boundary": {"bottom": {"type": "wall"}},
    }
    out = resolve({**base, "output": {"probes": [[1, 2]]}})["output"]
    assert out["probes"] == [[1.0, 2.0]] and out["probe_interval"] == 1
    with pytest.raises(Exception, match=r"output.probes: expected array of \[x, y\]"):
        resolve({**base, "output": {"probes": [1, 2]}})


def test_compressible_freestream_fields():
    """Uniform M 0.8 freestream: cp, losses, entropy and |grad rho| vanish."""
    from pyaeroflow.run import setup

    raw = {
        "physics": {"flow": "compressible", "model": "euler"},
        "mesh": {"file": "m.su2"},
        "boundary": {
            "bottom": {"type": "slip-wall"},
            "top": {"type": "farfield", "mach": 0.8},
            "left": {"type": "farfield", "mach": 0.8},
            "right": {"type": "farfield", "mach": 0.8},
        },
    }
    config = resolve(raw)
    problem, scaling, w0, _ = setup(config, rectangle(6, 4))
    fs = Freestream.from_config(config.data, scaling, problem.model, True)
    assert np.isclose(fs.mach, 0.8)
    names = ["cp", "total_pressure_loss", "entropy", "density_gradient", "mach"]
    f = compute_fields(names, FieldContext(problem, w0, scaling, freestream=fs))
    for name in names[:4]:
        assert np.allclose(f[name], 0.0, atol=1e-9), name
    assert np.allclose(f["mach"], 0.8)
    w = np.array(w0)
    w[:, 0] *= 2.0  # double p at the same T: rho doubles too
    f = compute_fields(["entropy"], FieldContext(problem, w, scaling, freestream=fs))
    assert np.allclose(f["entropy"], np.log(2.0) * (1.0 - 1.4))

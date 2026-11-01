"""Run a case from a TOML file: setup, non-dimensional solve, output.

Internally the solver works with rho_ref = U_ref = L_ref = 1 (D51): the mesh
is scaled by 1 / L_ref, velocities by 1 / U_ref, pressures by
1 / (rho_ref U_ref^2), the viscosity becomes 1 / Re. Results are scaled back
before output.
"""

import copy
import dataclasses
import hashlib
import math
import os
import tomllib
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

from pyaeroflow import __version__
from pyaeroflow.config import Resolved, load_case, resolve
from pyaeroflow.control import ControlFiles, check_no_stop_file
from pyaeroflow.driver import (
    GAMMA_DIAGNOSTICS,
    GAMMA_STEP_DIAGNOSTICS,
    INFLOW_TYPES,
    SST_DEFAULT_INTENSITY,
    SST_DEFAULT_VISCOSITY_RATIO,
    IncompressibleProblem,
)
from pyaeroflow.housekeeping import backup_name, latest_backup
from pyaeroflow.io import (
    HistoryWriter,
    RunLog,
    StepHistoryWriter,
    estimate_vtu_bytes,
    write_pvd,
    write_vtu,
)
from pyaeroflow.io.fields import (
    FIELDS,
    Capabilities,
    FieldContext,
    compute_fields,
    n_columns,
    resolve_fields,
)
from pyaeroflow.mesh import Mesh, read_mesh
from pyaeroflow.mesh.ordering import order_boundary_faces
from pyaeroflow.physics import forces as fz
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.physics import lm, rans
from pyaeroflow.physics import spalart_allmaras as sa
from pyaeroflow.reload import LIVE_STEADY, LIVE_UNSTEADY, reload_case
from pyaeroflow.solver import DualTimeSettings, LinearSettings, PseudoTimeSettings
from pyaeroflow.solver.convergence import CauchyCriterion, WindowedCauchyCriterion
from pyaeroflow.solver.profile import timers


class Scaling(NamedTuple):
    """Reference values of the non-dimensionalisation (D51)."""

    density: float
    velocity: float
    length: float

    @property
    def pressure(self):
        """rho_ref U_ref^2."""
        return self.density * self.velocity**2

    @property
    def viscosity(self):
        """rho_ref U_ref L_ref."""
        return self.density * self.velocity * self.length


class RunResult(NamedTuple):
    """Outcome of :func:`run_case`.

    Attributes
    ----------
    config : Resolved
    mesh : Mesh
        The mesh in physical units.
    w : ndarray, shape (n_cells, 3)
        (p, u, v) in physical units, plus nu_tilde (m^2/s) with SA or k
        (m^2/s^2) and omega (1/s) with SST.
    converged : bool
    history : dict
    output_dir : Path
    coefficients : Coefficients or None
        Final force coefficients over ``output.force_boundaries``.
    restart_file : Path or None
        The final backup ``<name>.backup_<iteration or step>.npz``.
    """

    config: Resolved
    mesh: Mesh
    w: np.ndarray
    converged: bool
    history: dict
    output_dir: Path
    coefficients: object = None
    restart_file: Path = None


class TurbulenceValues(NamedTuple):
    """Equivalent turbulence values of one input table (D33), SI units."""

    chi: float  # nu_tilde / nu (None for SST)
    viscosity_ratio: float  # nu_t / nu
    kinetic_energy: float = None  # m^2/s^2
    length_scale: float = None  # m
    intensity: float = None  # Tu as a fraction
    omega: float = None  # 1/s (SST)

    def report(self):
        """Log line: equivalent values."""
        text = f"nu_t/nu = {self.viscosity_ratio:.4g}"
        if self.chi is not None:
            text += f", nu_tilde/nu = {self.chi:.4g}"
        if self.kinetic_energy is not None:
            text += f", k = {self.kinetic_energy:.4g} m^2/s^2"
        if self.omega is not None:
            text += f", omega = {self.omega:.4g} 1/s"
        if self.length_scale is not None:
            text += f", l = {self.length_scale:.4g} m"
        if self.intensity is not None:
            text += f", Tu = {100 * self.intensity:.3g} %"
        return text


def sst_turbulence_values(table, nu, velocity):
    """Convert a turbulence table to SST's k and omega = k / nu_t (D33).

    None gives the default Tu = 0.1 %, nu_t / nu = 1; form b needs the
    intensity (checked with the configuration).
    """
    if table is None:
        table = {
            "turbulent_viscosity_ratio": SST_DEFAULT_VISCOSITY_RATIO,
            "intensity": SST_DEFAULT_INTENSITY,
        }
    if "kinetic_energy" in table:
        k, ell = table["kinetic_energy"], table["length_scale"]
        ratio = table.get("c_mu", 0.09) ** 0.25 * math.sqrt(k) * ell / nu
        tu = math.sqrt(2.0 * k / 3.0) / velocity if velocity > 0.0 else None
    else:
        ratio, tu, ell = table["turbulent_viscosity_ratio"], table["intensity"], None
        k = 1.5 * (tu * velocity) ** 2
    omega = k / (ratio * nu) if ratio > 0.0 else 0.0
    return TurbulenceValues(None, ratio, k, ell, tu, omega)


def turbulence_values(table, nu, velocity, model="sa-neg"):
    """Convert a turbulence table to the SA working variable (D33).

    Parameters
    ----------
    table : dict or None
        One of the three forms; None gives nu_tilde / nu = 3.
    nu : float
        Kinematic viscosity in m^2/s.
    velocity : float
        Local velocity magnitude in m/s (for Tu <-> k).
    model : str
        ``"sst"`` gives k and omega instead (:func:`sst_turbulence_values`).
    """
    if model == "sst":
        return sst_turbulence_values(table, nu, velocity)
    if table is None:
        chi = 3.0
        return TurbulenceValues(chi, float(sa.eddy_viscosity_ratio(chi)))
    if "nu_tilde_ratio" in table:
        chi = float(table["nu_tilde_ratio"])
        return TurbulenceValues(chi, float(sa.eddy_viscosity_ratio(chi)))
    if "kinetic_energy" in table:
        k, ell = table["kinetic_energy"], table["length_scale"]
        c_mu = table.get("c_mu", 0.09)
        ratio = c_mu**0.25 * math.sqrt(k) * ell / nu
        chi = float(sa.nu_tilde_from_eddy_viscosity(ratio))
        tu = math.sqrt(2.0 * k / 3.0) / velocity if velocity > 0.0 else None
        return TurbulenceValues(chi, ratio, k, ell, tu)
    ratio = table["turbulent_viscosity_ratio"]
    chi = float(sa.nu_tilde_from_eddy_viscosity(ratio))
    tu = table.get("intensity")
    k = 1.5 * (tu * velocity) ** 2 if tu is not None else None
    return TurbulenceValues(chi, ratio, k, None, tu)


def setup(config, mesh=None):
    """Build the non-dimensional problem of a resolved configuration.

    Parameters
    ----------
    config : Resolved
    mesh : Mesh, optional
        Mesh in physical units; default: read ``mesh.file``.

    Returns
    -------
    problem : IncompressibleProblem
    scaling : Scaling
    w0 : ndarray
        Non-dimensional initial state.
    mesh : Mesh
        The physical mesh.
    """
    cfg = config.data
    if mesh is None:
        mesh = read_mesh(config.resolve_path(cfg["mesh"]["file"]), cfg["mesh"]["scale"])
    fluid, ref, physics = cfg["fluid"], cfg["reference"], cfg["physics"]
    scaling = Scaling(
        float(ref["density"]), float(ref["velocity"]), float(ref["length"])
    )
    if physics["model"] == "euler":
        mu_nd = 0.0
    elif fluid["viscosity"] is not None:
        mu_nd = fluid["viscosity"] / scaling.viscosity
    else:
        mu_nd = 1.0 / fluid["reynolds_number"]
    rho_nd = fluid["density"] / scaling.density
    rans = physics["model"] == "rans"
    turb_model = physics["turbulence"] if rans else "none"
    nu_si = mu_nd * scaling.viscosity / fluid["density"]  # m^2/s
    turbulence = {}  # reported values per inflow boundary and "initial"

    conditions = {}
    for name, b in cfg["boundary"].items():
        kind = b["type"]
        if kind == "wall" and physics["model"] == "euler":
            kind = "slip-wall"
        entry = {"type": kind}
        if kind == "periodic":
            entry["partner"] = b["partner"]
            entry["translation"] = tuple(t / scaling.length for t in b["translation"])
        elif kind == "farfield":
            a = math.radians(b["angle_of_attack"])
            u = b["velocity"] / scaling.velocity
            entry["velocity"] = (u * math.cos(a), u * math.sin(a))
            entry["pressure"] = b["pressure"] / scaling.pressure
        elif kind in ("wall", "velocity-inlet"):
            entry["velocity"] = tuple(v / scaling.velocity for v in b["velocity"])
        elif kind == "pressure-outlet":
            entry["pressure"] = b["pressure"] / scaling.pressure
        elif kind == "total-pressure-inlet":
            entry["total_pressure"] = b["total_pressure"] / scaling.pressure
            entry["flow_direction"] = tuple(b["flow_direction"])
        elif kind == "transpiration-wall":
            entry["normal_velocity"] = b["normal_velocity"] / scaling.velocity
        if rans and kind in INFLOW_TYPES:
            if kind == "farfield":
                speed = b["velocity"]
            elif kind == "velocity-inlet":
                speed = math.hypot(*b["velocity"])
            else:
                speed = scaling.velocity
            tv = turbulence_values(b.get("turbulence"), nu_si, speed, turb_model)
            turbulence[name] = tv
            entry.update(_turbulence_entry(tv, mu_nd, rho_nd, scaling))
        conditions[name] = entry

    num = cfg["numerics"]
    force = (0.0, 0.0)
    if physics["mean_pressure_gradient"] is not None:
        g = physics["mean_pressure_gradient"]
        force = tuple(-gi * scaling.length / scaling.pressure for gi in g)
    tr_cfg = physics["transition"]
    model = inc.IncompressibleModel(
        density=rho_nd,
        viscosity=mu_nd,
        beta_factor=num["beta_factor"],
        spatial_order=num["spatial_order"],
        limiter=num["limiter"],
        limiter_threshold=num["limiter_threshold"],
        viscous_gradient=num["viscous_gradient"],
        alpha=num["alpha"],
        body_force=force,
        turbulence=physics["turbulence"] if rans else "none",
        turbulence_spatial_order=num["turbulence_spatial_order"],
        coupling=cfg["solver"]["turbulence_coupling"],
        transition=physics["transition"]["model"] if rans else "none",
        turbulence_intensity=100.0 * (tr_cfg["turbulence_intensity"] or 0.0),
        critical_amplification=(
            tr_cfg["critical_amplification"] if tr_cfg["model"] == "aft" else None
        ),
        lm_positive_jacobian=bool(
            rans and tr_cfg["model"] == "lm" and tr_cfg["positive_jacobian"]
        ),
    )
    regions = None
    if rans and physics["transition"]["mode"] == "prescribed":
        regions = [
            {
                k: (
                    v / scaling.length
                    if k in ("x_min", "x_max", "y_min", "y_max", "ramp")
                    else v
                )
                for k, v in region.items()
            }
            for region in physics["transition"]["laminar_region"]
        ]
    nd_mesh = _scaled_mesh(mesh, 1.0 / scaling.length)
    point = ref["pressure_point"]
    problem = IncompressibleProblem(
        nd_mesh,
        model,
        conditions,
        weight_exponent=num["gradient_weight_exponent"],
        pressure_point=None
        if point is None
        else tuple(p / scaling.length for p in point),
        pin_pressure=cfg["initial"]["pressure"] / scaling.pressure,
        laminar_regions=regions,
    )

    init = cfg["initial"]
    nu_tilde = k_omega = None
    if init["from_boundary"] is not None:
        src = conditions[init["from_boundary"]]
        velocity, pressure = src["velocity"], src.get("pressure", 0.0)
        nu_tilde = src.get("nu_tilde")
        if "k" in src:
            k_omega = (src["k"], src["omega"])
        if rans and init["turbulence"] is None and init["from_boundary"] in turbulence:
            turbulence["initial"] = turbulence[init["from_boundary"]]
    elif init["velocity"] is not None:
        velocity = tuple(v / scaling.velocity for v in init["velocity"])
        pressure = init["pressure"] / scaling.pressure
    else:  # restart without an explicit initial state: freestream of 1
        velocity, pressure = (1.0, 0.0), init["pressure"] / scaling.pressure
    if rans and "initial" not in turbulence:
        speed = math.hypot(*velocity) * scaling.velocity
        tv = turbulence_values(init["turbulence"], nu_si, speed, turb_model)
        turbulence["initial"] = tv
        entry = _turbulence_entry(tv, mu_nd, rho_nd, scaling)
        nu_tilde = entry.get("nu_tilde")
        k_omega = (entry["k"], entry["omega"]) if "k" in entry else None
    w0 = problem.initial_state(velocity, pressure, nu_tilde, k_omega)
    if init["restart"] is not None:
        restart = config.resolve_path(init["restart"])
        w0, n_read = load_restart(restart, mesh, w0)
        source = (
            f"boundary '{init['from_boundary']}'"
            if init["from_boundary"] is not None
            else "initial.velocity/pressure"
        )
        note = f"initial state from restart file {_display_path(restart)}"
        if n_read < w0.shape[1]:
            note += f" ({n_read} of {w0.shape[1]} variables; the rest from {source})"
        config.notices.append(note)
    problem.turbulence_values = turbulence
    return problem, scaling, w0, mesh


def _turbulence_entry(tv, mu_nd, rho_nd, scaling):
    """Non-dimensional turbulence values of a boundary or the initial state."""
    if tv.chi is not None:
        return {"nu_tilde": tv.chi * mu_nd / rho_nd}
    return {
        "k": tv.kinetic_energy / scaling.velocity**2,
        "omega": tv.omega * scaling.length / scaling.velocity,
    }


def mesh_fingerprint(mesh):
    """Hash of the cell centroids: identifies a mesh for restart files."""
    data = np.ascontiguousarray(np.round(mesh.cell_centroid, 9))
    return hashlib.sha1(data.tobytes()).hexdigest()[:16]


def save_restart(path, mesh, w_nd, step=None, time=None, state=None):
    """Write the non-dimensional state and the mesh fingerprint (D45).

    The file is written next to ``path`` and renamed, so an interrupted write
    never damages an existing restart file. ``step`` (iteration or time step)
    and ``time`` (physical seconds, unsteady runs) are stored with the
    ``state``: the solver state that a continuation needs (``kind`` "steady":
    CFL, residual peaks, Newton-Krylov switch, limiter, farfield vortex;
    "unsteady": the solution one step earlier and the time step), see
    :func:`load_restart_state`.
    """
    path = Path(path)
    extra = {}
    if step is not None:
        extra["step"] = step
    if time is not None:
        extra["time"] = time
    for key, value in (state or {}).items():
        if value is not None:
            extra[f"state_{key}"] = np.asarray(value)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as file:
        np.savez(
            file,
            w=np.asarray(w_nd),
            fingerprint=mesh_fingerprint(mesh),
            n_cells=mesh.n_cells,
            version=__version__,
            **extra,
        )
    os.replace(tmp, path)


class RestartWriter:
    """Restart and backup files written during a run (D56).

    ``restart_interval`` = n writes ``<name>.restart_a.npz`` and
    ``<name>.restart_b.npz`` alternately every n iterations (steady) or time
    steps (unsteady); each write replaces the older of the two, so the newer
    one is always complete. ``restart_backup_interval`` = m writes a backup
    ``<name>.backup_<number>.npz`` that is never replaced, and
    ``restart_backup_at`` adds such files at the listed iterations or steps.
    A write that falls on a rejected iteration or a time step that hit the
    inner limit waits for the next good one.
    """

    def __init__(self, out_dir, name, mesh, output, after=0, loaded=None):
        self.out_dir, self.name, self.mesh = out_dir, name, mesh
        self.every = output["restart_interval"]
        self.backup = output["restart_backup_interval"]
        self.backup_at = [
            n for n in sorted(set(output.get("restart_backup_at") or [])) if n > after
        ]
        self.written = 1 if str(loaded).endswith(".restart_a.npz") else 0
        self.buckets = {
            "rolling": after // self.every if self.every else 0,
            "backup": after // self.backup if self.backup else 0,
        }

    def update(self, output, number):
        """Take new ``restart_*`` settings (``.read_toml_N``) at ``number``."""
        self.every = output["restart_interval"]
        self.backup = output["restart_backup_interval"]
        self.backup_at = [
            n for n in sorted(set(output.get("restart_backup_at") or [])) if n >= number
        ]
        for key, every in (("rolling", self.every), ("backup", self.backup)):
            self.buckets[key] = (number - 1) // every if every else 0

    def __call__(self, number, w, good=True, time=None, state=None):
        """Write the files that are due at iteration or step ``number``.

        ``state`` (callable returning the dict of :func:`save_restart`) is
        only evaluated when a file is written.
        """
        if not good:
            return
        if self.every and number // self.every > self.buckets["rolling"]:
            self.buckets["rolling"] = number // self.every
            file = f"{self.name}.restart_{'ab'[self.written % 2]}.npz"
            self.written += 1
            save_restart(
                self.out_dir / file, self.mesh, w, number, time, state and state()
            )
        due = False
        if self.backup and number // self.backup > self.buckets["backup"]:
            self.buckets["backup"] = number // self.backup
            due = True
        if self.backup_at and self.backup_at[0] <= number:
            self.backup_at = [n for n in self.backup_at if n > number]
            due = True
        if due:
            file = backup_name(self.name, number)
            save_restart(
                self.out_dir / file, self.mesh, w, number, time, state and state()
            )


class FieldWriter:
    """VTU field files written during a run (``output.fields_*``, D59).

    ``fields_interval`` = n writes ``<name>_<number>.vtu`` every n iterations
    (steady) or time steps (unsteady), ``fields_at`` adds files at the listed
    numbers, ``fields_final`` the final ``<name>.vtu``. The schedule follows
    :class:`RestartWriter`: a write that falls on a rejected iteration or a
    time step that hit the inner limit waits for the next good one.
    """

    def __init__(
        self, out_dir, name, mesh, scaling, problem, monitor, output, after=0, caps=None
    ):
        self.capabilities = caps
        self.out_dir, self.name, self.mesh = out_dir, name, mesh
        self.scaling, self.problem, self.monitor = scaling, problem, monitor
        self.names = list(output["fields"])
        self.every = output["fields_interval"]
        self.final = output["fields_final"]
        self.at = [n for n in sorted(set(output["fields_at"])) if n > after]
        self.bucket = after // self.every if self.every else 0
        self.last = None  # number of the latest series file

    def update(self, output, number):
        """Take new ``fields_*`` settings (``.read_toml_N``) at ``number``."""
        self.names = list(output["fields"])
        self.every = output["fields_interval"]
        self.final = output["fields_final"]
        self.at = [n for n in sorted(set(output["fields_at"])) if n >= number]
        self.bucket = (number - 1) // self.every if self.every else 0

    def with_extra(self, requested):
        """Return the selected fields plus those of ``.result_N``, and problems."""
        extra, problems = resolve_fields(requested, self.capabilities)
        names = set(self.names) | set(extra)
        return [f.name for f in FIELDS if f.name in names], problems

    def write(self, path, w, cell_residual=None, names=None):
        """Write the selected fields (or ``names``) of the state ``w``."""
        ctx = FieldContext(self.problem, w, self.scaling, self.monitor, cell_residual)
        write_vtu(path, self.mesh, cell_data=compute_fields(names or self.names, ctx))

    def due(self, number, good=True):
        """Return True if a series file is due at ``number`` (updates the schedule)."""
        if not good:
            return False
        due = False
        if self.every and number // self.every > self.bucket:
            self.bucket = number // self.every
            due = True
        if self.at and self.at[0] <= number:
            self.at = [n for n in self.at if n > number]
            due = True
        return due

    def series_file(self, number):
        """Name of the series file of ``number`` (recorded as the latest)."""
        self.last = number
        return f"{self.name}_{number:06d}.vtu"

    def write_pvd(self):
        """Steady runs: collect the files of the series (also earlier runs)."""
        pattern = f"{self.name}_[0-9][0-9][0-9][0-9][0-9][0-9].vtu"
        files = sorted(self.out_dir.glob(pattern))
        write_pvd(
            self.out_dir / f"{self.name}.pvd",
            [(int(f.stem.rsplit("_", 1)[1]), f.name) for f in files],
        )


def _field_rows(cfg, mesh):
    """Build the configuration rows of the field output (fields, schedule, sizes)."""
    out = cfg["output"]
    names = out["fields"]
    n_scalar, n_vector = n_columns(names)
    size = estimate_vtu_bytes(mesh, n_scalar, n_vector)
    unsteady = cfg["time"]["mode"] == "unsteady"
    total = (
        int(round(cfg["time"]["end_time"] / cfg["time"]["time_step"]))
        if unsteady
        else cfg["solver"]["max_iterations"]
    )
    unit = "time steps" if unsteady else "iterations"
    every, at = out["fields_interval"], sorted(set(out["fields_at"]))
    parts, files = [], 0
    if every:
        parts.append(f"every {every} {unit}")
        files += total // every
    if at:
        parts.append(f"at {', '.join(str(n) for n in at)}")
        files += len([n for n in at if n <= total and (not every or n % every)])
    if out["fields_final"]:
        parts.append("final")
        files += 1
    schedule = ", ".join(parts) if parts else "none"
    rows = [("field output", f"{len(names)} fields: {', '.join(names)}")]
    rows.append(
        (
            "field files",
            f"{size / 1e6:.1f} MB each; {schedule}; at most {files} files, "
            f"{size * files / 1e6:.0f} MB",
        )
    )
    return rows


def load_restart(path, mesh, w_default):
    """Read a restart file for ``mesh``; missing variables keep ``w_default``.

    A laminar restart used for a RANS case keeps the default nu_tilde, a RANS
    restart used for a laminar case drops it. The column count tells the
    turbulence model (4 SA, 5 SST, 6 SA-AFT, 7 SST-LM): between SST and SA
    only p, u, v are taken over; SST <-> SST-LM keeps k and omega, SA <-> SA-AFT
    keeps nu_tilde.
    """
    with np.load(path) as data:
        if str(data["fingerprint"]) != mesh_fingerprint(mesh):
            raise ValueError(f"restart file {path} belongs to a different mesh")
        w = np.array(data["w"], dtype=float)
    n = min(w.shape[1], w_default.shape[1])
    sst_family = (5, 7)
    if (w.shape[1] in sst_family) != (w_default.shape[1] in sst_family):
        n = min(n, 3)
    w0 = np.array(w_default, dtype=float)
    w0[:, :n] = w[:, :n]
    return w0, n


def load_restart_state(path):
    """Solver state stored in a restart file, or None for an older file.

    Returns the ``state`` of :func:`save_restart` with the iteration or time
    step as ``iteration`` (steady) or ``step`` (unsteady) and the physical
    ``time``: the ``resume`` argument of ``IncompressibleProblem.solve`` and
    ``solve_unsteady``. A file written before the state was stored (no
    ``state_*`` entries) gives None: the run then starts as a new run.
    """
    with np.load(path) as data:
        keys = [k for k in data.files if k.startswith("state_")]
        if not keys or "step" not in data.files:
            return None
        state = {k[6:]: np.array(data[k]) for k in keys}
        step = int(data["step"])
        if "time" in data.files:
            state["time"] = float(data["time"])
    for key in ("cfl", "cfl_cap", "eta", "elapsed", "vortex_gamma", "time_step"):
        if key in state:
            state[key] = float(state[key])
    if "nk_on" in state:
        state["nk_on"] = bool(state["nk_on"])
    kind = str(state.pop("kind", "steady"))
    state["kind"] = kind
    state["step" if kind == "unsteady" else "iteration"] = step
    return state


def _display_path(path):
    """``path`` relative to the working directory if below it, else absolute."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _resume_state(config, scaling, restart_mode):
    """Solver state to continue from ``initial.restart``, or None; notices appended.

    ``restart_mode`` "continue" (the default of every run) continues the
    run that wrote the file: iteration or time step, CFL, residual peaks,
    Newton-Krylov switch, limiter and vortex, or the previous time level.
    "state" (the next angle of a polar) takes over only the solver state of a
    steady run (CFL, Newton-Krylov switch, limiter, vortex): the counter and
    the residual reference start again. A file without stored state (older
    version) or of the other kind (steady/unsteady) starts a new run.
    """
    cfg = config.data
    if cfg["initial"]["restart"] is None:
        return None
    path = config.resolve_path(cfg["initial"]["restart"])
    state = load_restart_state(path)
    unsteady = cfg["time"]["mode"] == "unsteady"
    if state is None:
        config.notices.append(
            "restart file has no solver state: new run from its fields"
        )
        return None
    if (state["kind"] == "unsteady") != unsteady:
        config.notices.append(
            f"restart file is a {state['kind']} state: new run from its fields"
        )
        return None
    if restart_mode == "state":
        if unsteady:
            return None
        keys = ["cfl", "cfl_cap", "nk_on", "psi", "gamma", "vortex_gamma"]
        if cfg["solver"]["cfl"]["polar_reset"]:
            keys = keys[2:]  # the CFL ramp starts again, Newton-Krylov stays on
        return {k: state[k] for k in keys if k in state}
    if unsteady:
        dt = cfg["time"]["time_step"] / (scaling.length / scaling.velocity)
        if abs(state.get("time_step", dt) - dt) > 1e-9 * dt:
            state.pop("u_previous", None)
            config.notices.append(
                "time step differs from the restart file: first step BDF1"
            )
        config.notices.append(
            f"continuing after time step {state['step']} (t = {state['step'] * dt * scaling.length / scaling.velocity:.6g} s)"
        )
    else:
        config.notices.append(f"continuing after iteration {state['iteration']}")
    state["file"] = str(path)
    return state


def run_case(path, console=True, mesh=None, restart_mode="continue"):
    """Run a case file end to end and write the results.

    Parameters
    ----------
    path : str, Path or Resolved
        TOML case file, or an already resolved configuration.
    console : bool
        Echo the log to stdout.
    mesh : Mesh, optional
        Use this mesh instead of reading ``mesh.file`` (tests).
    restart_mode : {"continue", "state"}
        What a restart file (``initial.restart``) carries over, see
        :func:`_resume_state`; ``run_polar`` uses "state".

    Returns
    -------
    RunResult
    """
    config = path if isinstance(path, Resolved) else load_case(path)
    cfg = config.data
    out_dir = config.resolve_path(cfg["output"]["directory"])
    check_no_stop_file(out_dir)  # before the mesh is read (L7)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = config.name
    (out_dir / f"{name}.resolved.toml").write_text(config.to_toml())

    # The log and history of a continued run are appended to (D58): a restart
    # that continues the counter writes one log from start to end.
    log_path = out_dir / f"{name}.log"
    problem, scaling, w0, mesh = setup(config, mesh)
    resume = _resume_state(config, scaling, restart_mode)
    continued = bool(resume) and ("iteration" in resume or "step" in resume)
    with RunLog(
        log_path,
        console,
        append=continued and log_path.exists(),
        elapsed_offset=resume.get("elapsed", 0.0) if continued else 0.0,
    ) as log:
        if continued and log_path.exists() and log_path.stat().st_size:
            number = resume.get("iteration", resume.get("step"))
            log.write("")
            log.write("=" * 100)
            log.write(
                f"RESTART {datetime.now():%Y-%m-%d %H:%M:%S}: continuing after "
                f"{'iteration' if 'iteration' in resume else 'step'} {number} "
                f"from {_display_path(resume['file'])}"
            )
            log.write("=" * 100)
        log.write(f"PyAeroFlow {__version__}: case '{name}' ({config.path})")
        _log_setup(log, config, problem, scaling, mesh)

        if cfg["time"]["mode"] == "unsteady":
            result, coefficients = _run_unsteady(
                log, config, problem, scaling, w0, mesh, out_dir, resume
            )
            state = {
                "kind": "unsteady",
                "u_previous": result.u_previous,
                "time_step": cfg["time"]["time_step"]
                / (scaling.length / scaling.velocity),
            }
            n_steps = (
                int(result.history["step"][-1])
                if result.history["step"].size
                else (resume or {}).get("step", 0)
            )
            t_phys = result.time * scaling.length / scaling.velocity
        else:
            result, coefficients = _run_steady(
                log, config, problem, scaling, w0, mesh, out_dir, resume
            )
            state = (
                {"kind": "steady", **result.state, "elapsed": log.elapsed}
                if result.state
                else None
            )
            n_steps = int(result.history["iteration"][-1])
            t_phys = None
    restart_file = out_dir / backup_name(name, n_steps)  # the final backup (D56)
    save_restart(restart_file, mesh, result.u, n_steps, t_phys, state)
    scale = [scaling.pressure, scaling.velocity, scaling.velocity]
    if problem.model.turbulence == "sst":
        scale += [scaling.velocity**2, scaling.velocity / scaling.length]
    elif problem.turbulent:
        scale.append(scaling.viscosity / scaling.density)
    scale += [1.0] * (problem.model.n_var - len(scale))  # AFT: dimensionless
    w = result.u * np.array(scale)
    return RunResult(
        config,
        mesh,
        w,
        result.converged,
        result.history,
        out_dir,
        coefficients,
        restart_file,
    )


def _pseudo_time_settings(cfg):
    solver = cfg["solver"]
    return PseudoTimeSettings(
        cfl_start=solver["cfl"]["start"],
        cfl_max=solver["cfl"]["max"],
        cfl_max_approximate=solver["cfl"]["max_approximate"],
        growth_factor=solver["cfl"]["growth_factor"],
        cut_factor=solver["cfl"]["cut_factor"],
        reject_factor=solver["cfl"]["reject_factor"],
        cfl_control=solver["cfl"]["control"],
        ser_exponent=solver["cfl"]["ser_exponent"],
        ser_growth_min=solver["cfl"]["ser_growth_min"],
        ser_ratio_min=solver["cfl"]["ser_ratio_min"],
        ser_ratio_max=solver["cfl"]["ser_ratio_max"],
        ser_gmres_hold=solver["cfl"]["ser_gmres_hold"],
        ser_gmres_reduce=solver["cfl"]["ser_gmres_reduce"],
        max_iterations=solver["max_iterations"],
        residual_drop=solver["residual_drop"],
        first_order_iterations=cfg["numerics"]["first_order_iterations"],
        newton_krylov=solver["newton_krylov"]["enabled"],
        nk_switch_residual_drop=solver["newton_krylov"]["switch_at_residual_drop"],
        nk_switch_iteration=solver["newton_krylov"]["switch_at_iteration"],
        nk_linear_tolerance=solver["newton_krylov"]["linear_tolerance"],
        nk_restart=solver["newton_krylov"]["restart"],
        nk_max_krylov=solver["newton_krylov"]["max_iterations"],
        nk_cfl_adapt_iterations=solver["newton_krylov"]["cfl_adapt_iterations"],
        refactor_interval=solver["refactor_interval"],
        linear=LinearSettings(**solver["linear_solver"]),
    )


def _residual_names(problem):
    """Names of the residual columns in the log and history."""
    names = ("p", "u", "v")
    if problem.model.turbulence == "sst":
        names += ("k", "om")
    elif problem.turbulent:
        names += ("nu",)
    if problem.model.transition == "aft":
        names += ("gam", "n")
    elif problem.model.transition == "lm":
        names += ("gam", "ret")
    return names


def _apply_reload(log, config, number, unit, live, out_dir, fields, restarts):
    """Handle ``.read_toml_<number>``: reload, apply, log, keep a copy."""
    change = reload_case(config, live)
    for line in change.lines(number, unit):
        log.write(line)
    if change.applied:
        if change.changed("output"):
            fields.update(config.data["output"], number)
            restarts.update(config.data["output"], number)
        copy_file = out_dir / f"{config.name}.read_toml_{number:06d}.toml"
        copy_file.write_text(config.to_toml())
        log.write(f"  settings in force written to {copy_file.name}")
    return change


def _result_names(log, control, fields, number):
    """Fields of ``.result_<number>``: the selected ones plus those named in it."""
    requested = control.result_fields(number)
    if not requested:
        return None
    names, problems = fields.with_extra(requested)
    for problem in problems:
        log.write(
            f"control file .result_{number}: "
            + problem.replace("output.fields: ", "", 1)
            + ": skipped"
        )
    return names


def _log_timing(log, wall):
    """Write the wall-time breakdown of the implicit iteration."""
    rows = timers.rows(wall)
    if rows:
        log.section("Timing")
        log.table(rows)


def _run_steady(log, config, problem, scaling, w0, mesh, out_dir, resume=None):
    """Steady solve with one log line per pseudo-time iteration."""
    cfg = config.data
    name = config.name
    names = _residual_names(problem)
    implicit = cfg["solver"]["method"] == "implicit"
    force_boundaries = cfg["output"]["force_boundaries"]
    monitor = Monitor(config, problem, scaling) if force_boundaries else None
    coeff_names = ("CL", "CD", "CM") if monitor else ()
    tr_cfg = cfg["physics"]["transition"]
    diagnose = cfg["output"]["transition_diagnostics"]
    after = (resume or {}).get("iteration")
    history = HistoryWriter(
        out_dir / f"{name}.history.csv",
        names,
        coeff_names,
        GAMMA_DIAGNOSTICS if diagnose else (),
        resume_after=after,
    )
    extra_names = (("NK", "GMRES") if implicit else ()) + tuple(
        (c, 11) for c in coeff_names
    )
    control = ControlFiles(out_dir, after or 0)
    for line in control.notices("iteration"):
        log.write(line)
    log.iteration_header(names, extra_names)
    fields = FieldWriter(
        out_dir,
        name,
        mesh,
        scaling,
        problem,
        monitor,
        cfg["output"],
        after=after or 0,
        caps=Capabilities.from_config(cfg),
    )
    restarts = RestartWriter(
        out_dir,
        name,
        mesh,
        cfg["output"],
        after=after or 0,
        loaded=(resume or {}).get("file"),
    )

    timing = {"setup": None}
    timers.reset()
    offset = (resume or {}).get("elapsed", 0.0)  # wall time of the earlier runs

    def restart_state():
        info = callback.info
        return {
            "kind": "steady",
            **info["solver_state"](),
            **info["driver_state"](),
            "elapsed": log.elapsed,
        }

    def callback(it, w, residuals, cfl, accepted):
        if timing["setup"] is None:
            timing["setup"] = log.elapsed - offset
        info = callback.info
        nk, lin = info.get("newton_krylov", False), info.get("linear_iterations", 0)
        extra = ("NK" if nk else "", lin if lin else "") if implicit else ()
        coeffs = monitor.coefficients(w) if monitor else ()
        extra += tuple(f"{c:.5f}" for c in coeffs)
        drop = max(0.0, -np.log10(max(info.get("measure", 1.0), np.finfo(float).tiny)))
        log.iteration(it, residuals, cfl, accepted, extra, info.pop("note", ""), drop)
        history.row(
            it,
            residuals,
            cfl,
            accepted,
            log.elapsed,
            nk,
            lin,
            coeffs,
            info.pop("transition_diagnostics", ("",) * len(GAMMA_DIAGNOSTICS))
            if diagnose
            else (),
        )
        for n in control.due_reloads(it, bool(accepted)):
            change = _apply_reload(
                log, config, n, "iteration", LIVE_STEADY, out_dir, fields, restarts
            )
            if change.changed("solver.convergence"):
                make_cauchy()
            for key in ("max_iterations", "residual_drop"):
                if change.changed(f"solver.{key}"):
                    info[key] = cfg["solver"][key]
        written = fields.due(it, accepted)
        if written:
            fields.write(out_dir / fields.series_file(it), w, info.get("cell_residual"))
            fields.write_pvd()
        for n in control.due_results(it, bool(accepted)):
            names = _result_names(log, control, fields, n)
            if names or not (written and n == it):
                fields.write(
                    out_dir / fields.series_file(n),
                    w,
                    info.get("cell_residual"),
                    names,
                )
                fields.write_pvd()
            extra = f" ({len(names)} fields)" if names else ""
            log.write(
                f"control file .result_{n}: fields written for iteration {n}{extra}"
            )
        restarts(it, w, good=bool(accepted), state=restart_state)
        if cauchy and accepted and cauchy.update(it, coeffs):
            log.write(
                f"Cauchy criterion met on {', '.join(cauchy.names)}: mean relative "
                f"change below {cauchy.eps:g} over {cauchy.elements} iterations"
            )
            info["converged"] = True
        if control.stop_requested():
            log.write(
                f"control file .stop found: stopping after iteration {it}, "
                "writing the backup and the result files"
            )
            info["stop"] = True

    callback.info = {}

    solver = cfg["solver"]
    cauchy = None

    def make_cauchy():
        """(Re)build the Cauchy criterion from ``solver.convergence``."""
        nonlocal cauchy
        conv = solver["convergence"]
        cauchy = (
            CauchyCriterion(
                conv["fields"],
                conv["cauchy_elements"],
                conv["cauchy_eps"],
                conv["start_iteration"],
            )
            if conv["fields"] and monitor and implicit
            else None
        )

    make_cauchy()
    vortex = None
    if solver["method"] == "implicit":
        settings = _pseudo_time_settings(cfg)
        freeze = cfg["numerics"]["limiter_freeze"]
        if cfg["reference"]["vortex_correction"]:
            vortex = problem.farfield_vortex(
                tuple(c / scaling.length for c in cfg["reference"]["moment_center"]),
                force_boundaries,
            )
        result = problem.solve(
            w0,
            settings,
            callback=callback,
            freeze_limiter_after_iterations=freeze["after_iterations"],
            freeze_limiter_after_residual_drop=freeze["after_residual_drop"],
            vortex=vortex,
            release_limiter=freeze["release"],
            gamma_relaxation=tr_cfg["relaxation"],
            gamma_diagnostics=diagnose,
            nu_tilde_step_limit=solver["nu_tilde_step_limit"],
            transition_clip=tr_cfg["clip"],
            gamma_guard=tr_cfg["newton_guard"],
            resume=resume,
        )
    else:
        result = problem.solve_explicit(
            w0,
            cfl=solver["cfl"]["explicit"],
            max_iterations=solver["max_iterations"],
            residual_drop=solver["residual_drop"],
            callback=callback,
        )
    history.close()
    if timing["setup"] is None:  # a restart that is converged already
        timing["setup"] = log.elapsed - offset

    stopped = bool(callback.info.get("stop"))
    write_final = fields.final or stopped
    if write_final:
        fields.write(
            out_dir / f"{name}.vtu", result.u, callback.info.get("cell_residual")
        )
    h = result.history
    log.section("Summary")
    nk_steps = int(np.count_nonzero(h["newton_krylov"]))
    rows = [
        ("converged", "yes" if result.converged else "NO"),
        *(
            [
                (
                    "convergence criterion",
                    f"Cauchy on {', '.join(cauchy.names)} (mean change < "
                    f"{cauchy.eps:g} over {cauchy.elements} iterations)",
                )
            ]
            if cauchy and callback.info.get("converged")
            else []
        ),
        ("iterations", f"{int(h['iteration'][-1])} ({nk_steps} Newton-Krylov)"),
        ("rejected updates", str(int(np.count_nonzero(~h["accepted"])))),
        *(
            [
                (
                    "gamma descent guard",
                    f"fired in {g['fired']} of {g['updates']} updates, at most "
                    f"{g['cells_max']} cells, {g['cells_last']} in the last",
                )
            ]
            if (g := callback.info.get("gamma_guard"))
            else []
        ),
        (
            "residual drop",
            f"{-np.log10(max(h['measure'][-1], np.finfo(float).tiny)):.1f} orders"
            f" (RMS {h['residual'][-1] / h['residual'][0]:.1e} x initial)",
        ),
        ("setup and compilation", f"{timing['setup']:.1f} s"),
        ("wall time", f"{log.elapsed:.1f} s"),
        *([("fields", str(out_dir / f"{name}.vtu"))] if write_final else []),
        *([("field series", str(out_dir / f"{name}.pvd"))] if fields.last else []),
        ("history", str(history.path)),
    ]
    coefficients = None
    if monitor:
        coefficients = monitor.final(result.u)
        c = coefficients
        rows[4:4] = (
            [
                ("force boundaries", ", ".join(force_boundaries)),
                ("CL, CD, CM", f"{c.cl:.6f}, {c.cd:.6f}, {c.cm:.6f}"),
            ]
            + _vortex_rows(vortex, scaling)
            + [
                ("CD pressure, viscous", f"{c.cd_pressure:.6f}, {c.cd_viscous:.6f}"),
                ("Cx, Cy", f"{c.cx:.6f}, {c.cy:.6f}"),
            ]
            + _transition_rows(monitor, result.u)
        )
        for bname in force_boundaries:
            path = out_dir / f"{name}.wall_{bname}.csv"
            monitor.write_wall(result.u, bname, path)
            rows.append((f"wall '{bname}'", str(path)))
    if stopped:
        rows.insert(1, ("stopped", "by the .stop file"))
    log.table(rows)
    _log_timing(log, log.elapsed - offset)
    return result, coefficients


def _transition_rows(monitor, w):
    """Summary rows with the transition locations (transition model only)."""
    if monitor.problem.model.transition == "none":
        return []
    rows = []
    for (boundary, side), x in monitor.transition_locations(w).items():
        where = "none (laminar)" if x is None else f"x = {x:.4g} m"
        rows.append((f"transition '{boundary}' {side}", where + " (nu_t/nu > 1)"))
    return rows


def _vortex_rows(vortex, scaling):
    if vortex is None:
        return []
    gamma = vortex.gamma * scaling.velocity * scaling.length
    return [("farfield vortex", f"circulation {gamma:.6g} m^2/s")]


def _run_unsteady(log, config, problem, scaling, w0, mesh, out_dir, resume=None):
    """Dual time stepping with one log line per physical time step (D45)."""
    cfg = config.data
    name = config.name
    tm = cfg["time"]
    t_scale = scaling.length / scaling.velocity  # physical seconds per unit time
    names = _residual_names(problem)
    force_boundaries = cfg["output"]["force_boundaries"]
    monitor = Monitor(config, problem, scaling) if force_boundaries else None
    coeff_names = ("CL", "CD", "CM") if monitor else ()
    diagnose = cfg["output"]["transition_diagnostics"]
    after = (resume or {}).get("step")
    history = StepHistoryWriter(
        out_dir / f"{name}.history.csv",
        names,
        coeff_names,
        GAMMA_STEP_DIAGNOSTICS if diagnose else (),
        resume_after=after,
    )
    tconv = tm["convergence"]
    n_steps_total = int(round(tm["end_time"] / tm["time_step"]))
    window_start = max(tconv["window_start_step"], (after or 0) + 1)
    tavg = (
        WindowedCauchyCriterion(
            tconv["fields"],
            n_steps_total - window_start + 1,
            tconv["window_function"],
            tconv["cauchy_elements"],
            tconv["cauchy_eps"],
            tconv["start_step"],
            window_start,
        )
        if tconv["fields"] and monitor
        else None
    )
    gamma = {"previous": None}
    if diagnose:  # the first step is compared with the restart state
        _, gamma["previous"] = problem.gamma_step_diagnostics(w0)
    detailed = cfg["output"]["verbosity"] == "detailed"
    control = ControlFiles(out_dir, after or 0)
    for line in control.notices("step"):
        log.write(line)
    log.step_header(names, ("GMRES",) + tuple((c, 11) for c in coeff_names))
    series = []
    fields = FieldWriter(
        out_dir,
        name,
        mesh,
        scaling,
        problem,
        monitor,
        cfg["output"],
        after=after or 0,
        caps=Capabilities.from_config(cfg),
    )
    restarts = RestartWriter(
        out_dir,
        name,
        mesh,
        cfg["output"],
        after=after or 0,
        loaded=(resume or {}).get("file"),
    )
    timing = {"setup": None}
    timers.reset()
    offset = (resume or {}).get("elapsed", 0.0)  # wall time of the earlier runs

    def restart_state():
        return {
            "kind": "unsteady",
            "u_previous": callback.info.get("u_previous"),
            "time_step": callback.info["time_step"],
            "elapsed": log.elapsed,
        }

    def write_series(step, t, w, names=None):
        file = fields.series_file(step)
        fields.write(out_dir / file, w, names=names)
        series.append((t * t_scale, file))
        write_pvd(out_dir / f"{name}.pvd", series)

    def callback(step, t, w, residuals, inner, drop, converged):
        if timing["setup"] is None:
            timing["setup"] = log.elapsed - offset
        lin = callback.info.get("linear_iterations", 0)
        coeffs = monitor.coefficients(w) if monitor else ()
        extra = (lin if lin else "",) + tuple(f"{c:.5f}" for c in coeffs)
        note = "" if converged else "inner limit"
        log.step(step, t * t_scale, inner, drop, residuals, extra, note)
        stats = ()
        if diagnose:
            stats, gamma["previous"] = problem.gamma_step_diagnostics(
                w, gamma["previous"]
            )
        history.row(
            step,
            t * t_scale,
            residuals,
            inner,
            drop,
            converged,
            lin,
            coeffs,
            log.elapsed,
            stats,
        )
        for n in control.due_reloads(step, bool(converged)):
            _apply_reload(
                log, config, n, "step", LIVE_UNSTEADY, out_dir, fields, restarts
            )
        written = fields.due(step, converged)
        if written:
            write_series(step, t, w)
        for n in control.due_results(step, bool(converged)):
            names = _result_names(log, control, fields, n)
            if names or not (written and n == step):
                write_series(n, t, w, names)
            extra = f" ({len(names)} fields)" if names else ""
            log.write(f"control file .result_{n}: fields written for step {n}{extra}")
        restarts(step, w, good=bool(converged), time=t * t_scale, state=restart_state)
        if tavg and tavg.update(step, coeffs):
            log.write(
                f"Cauchy criterion met on the time average of "
                f"{', '.join(tavg.names)}: mean change below {tavg.eps:g} over "
                f"{tavg.elements} steps"
            )
            callback.info["stop"] = True
            callback.info["time_converged"] = True
        if control.stop_requested():
            log.write(
                f"control file .stop found: stopping after step {step}, "
                "writing the backup and the result files"
            )
            callback.info["stop"] = True
        if detailed:
            log.iteration_header(names, ("NK", "GMRES"))
            log.step_header(names, ("GMRES",) + tuple((c, 11) for c in coeff_names))

    callback.info = {}

    def inner_callback(it, w, residuals, cfl, accepted):
        info = inner_callback.info
        nk, lin = info.get("newton_krylov", False), info.get("linear_iterations", 0)
        drop = max(0.0, -np.log10(max(info.get("measure", 1.0), np.finfo(float).tiny)))
        log.iteration(
            it, residuals, cfl, accepted, ("NK" if nk else "", lin or ""), drop=drop
        )

    inner_callback.info = {}

    inner = dataclasses.replace(
        _pseudo_time_settings(cfg),
        cfl_start=tm["inner_cfl"],
        max_iterations=tm["inner_iterations"],
        residual_drop=tm["inner_residual_drop"],
        first_order_iterations=0,
        nk_switch_residual_drop=0.0,
    )
    settings = DualTimeSettings(
        tm["time_step"] / t_scale, tm["end_time"] / t_scale, tm["scheme"], inner
    )
    result = problem.solve_unsteady(
        w0,
        settings,
        callback=callback,
        inner_callback=inner_callback if detailed else None,
        nu_tilde_step_limit=cfg["solver"]["nu_tilde_step_limit"],
        transition_clip=cfg["physics"]["transition"]["clip"],
        gamma_guard=cfg["physics"]["transition"]["newton_guard"],
        resume=resume,
    )
    history.close()
    if timing["setup"] is None:  # a restart at the end time already
        timing["setup"] = log.elapsed - offset

    n_steps = (
        int(result.history["step"][-1])
        if result.history["step"].size
        else (resume or {}).get("step", 0)
    )
    stopped = bool(callback.info.get("stop")) and not callback.info.get(
        "time_converged"
    )
    write_final = fields.final or bool(callback.info.get("stop"))
    if write_final:
        if fields.last != n_steps:
            write_series(n_steps, result.time, result.u)
        fields.write(out_dir / f"{name}.vtu", result.u)
    h = result.history
    log.section("Summary")
    rows = [
        *(
            [
                (
                    "convergence criterion",
                    f"Cauchy on the time average of {', '.join(tavg.names)} "
                    f"(mean change < {tavg.eps:g} over {tavg.elements} steps)",
                )
            ]
            if tavg and callback.info.get("time_converged")
            else []
        ),
        ("time steps", f"{n_steps} ({tm['scheme']}, dt = {tm['time_step']} s)"),
        ("final time", f"{result.time * t_scale:.6g} s"),
        ("inner iterations", f"{h['inner_iterations'].mean():.1f} per step (mean)"),
        ("steps at inner limit", str(int(np.count_nonzero(~h["converged"])))),
        ("setup and compilation", f"{timing['setup']:.1f} s"),
        ("wall time", f"{log.elapsed:.1f} s"),
        *([("fields", str(out_dir / f"{name}.vtu"))] if write_final else []),
        *([("field series", str(out_dir / f"{name}.pvd"))] if series else []),
        ("history", str(history.path)),
    ]
    coefficients = None
    if monitor:
        coefficients = monitor.final(result.u)
        c = coefficients
        rows[4:4] = [
            ("force boundaries", ", ".join(force_boundaries)),
            ("CL, CD, CM (final)", f"{c.cl:.6f}, {c.cd:.6f}, {c.cm:.6f}"),
        ]
        for bname in force_boundaries:
            path = out_dir / f"{name}.wall_{bname}.csv"
            monitor.write_wall(result.u, bname, path)
            rows.append((f"wall '{bname}'", str(path)))
    if stopped:
        rows.insert(1, ("stopped", "by the .stop file"))
    log.table(rows)
    _log_timing(log, log.elapsed - offset)
    return result, coefficients


class Monitor:
    """Force coefficients of the non-dimensional solution (D45).

    With rho_ref = U_ref = L_ref = 1 internally, q = rho_nd / 2 with the
    non-dimensional reference density and the reference length is 1.
    """

    def __init__(self, config, problem, scaling):
        cfg = config.data
        self.problem = problem
        self.scaling = scaling
        self.boundaries = cfg["output"]["force_boundaries"]
        self.q = 0.5 * cfg["reference"]["density"] / scaling.density
        self.alpha = cfg["reference"]["angle_of_attack"]
        self.moment_center = tuple(
            c / scaling.length for c in cfg["reference"]["moment_center"]
        )
        pressures = {
            b["pressure"] for b in cfg["boundary"].values() if b["type"] == "farfield"
        }
        self.p_ref = pressures.pop() / scaling.pressure if len(pressures) == 1 else 0.0

    def coefficients(self, w):
        """(CL, CD, CM) of the state ``w``."""
        c = self.final(w)
        return (c.cl, c.cd, c.cm)

    def final(self, w):
        """Full :class:`~pyaeroflow.physics.forces.Coefficients`."""
        f = self.problem.forces(w, self.boundaries, self.moment_center)
        return fz.coefficients(f, self.q, 1.0, self.alpha)

    def transition_locations(self, w):
        """Transition location per side of each force boundary.

        The chordwise position (SI) where the boundary layer's maximum
        nu_t/nu first exceeds 1 on the upper (y >= 0) and the lower side,
        or None if it never does (laminar to the trailing edge).
        """
        geom = self.problem.geom
        out = {}
        for boundary in self.boundaries:
            faces = self.problem.boundary_faces([boundary])
            centre = np.asarray(geom.b_centre)[faces]
            g = self.problem.wall_eddy_viscosity_ratio(w, faces)
            for side, sel in (
                ("upper", centre[:, 1] >= 0.0),
                ("lower", centre[:, 1] < 0.0),
            ):
                if not sel.any():
                    continue
                order = np.argsort(centre[sel, 0])
                turbulent = np.flatnonzero(g[sel][order] >= 1.0)
                x = centre[sel, 0][order]
                out[(boundary, side)] = (
                    float(x[turbulent[0]]) * self.scaling.length
                    if turbulent.size
                    else None
                )
        return out

    def write_wall(self, w, boundary, path):
        """Write x, y, p, Cp, tau_x, tau_y, Cf of one boundary (SI units)."""
        loads = self.problem.wall_loads(w)
        faces = self.problem.boundary_faces([boundary])
        d = fz.wall_distribution(loads, self.problem.geom, faces, self.q, self.p_ref)
        s = self.scaling
        columns = {
            "x": d["x"] * s.length,
            "y": d["y"] * s.length,
            "p": d["p"] * s.pressure,
            "cp": d["cp"],
            "tau_x": d["tau_x"] * s.pressure,
            "tau_y": d["tau_y"] * s.pressure,
            "cf": d["cf"],
        }
        if self.problem.turbulent:
            # y+ of the first cell centre: d sqrt(|tau_w| / rho) / nu (non-dim.).
            m = self.problem.model
            owner = np.asarray(self.problem.geom.b_owner)[faces]
            dist = np.asarray(self.problem.geom.wall_distance)[owner]
            tau_w = np.hypot(d["tau_x"], d["tau_y"])
            columns["y_plus"] = (
                dist * np.sqrt(tau_w / m.density) / (m.viscosity / m.density)
            )
            # Maximum nu_t/nu over the boundary-layer column above each face:
            # laminar ~1e-6-1e-2, turbulent 10-100.
            columns["nu_t_ratio_max"] = self.problem.wall_eddy_viscosity_ratio(w, faces)
        order = self._contour_order(boundary, d)
        if order is not None:
            columns = {key: col[order] for key, col in columns.items()}
            # Arc length from the start of the contour to the face centre.
            length = np.asarray(self.problem.mesh.face_length)[
                self.problem.mesh.boundary_faces[boundary]
            ][order]
            columns["s"] = (np.cumsum(length) - 0.5 * length) * s.length
        header = ",".join(columns)
        rows = np.column_stack(list(columns.values()))
        np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.10e")

    def _contour_order(self, boundary, d):
        """Face order along the wall contour (PyAero's airfoil direction), or None."""
        mesh = self.problem.mesh
        local = mesh.boundary_faces[boundary]
        centre = np.column_stack((d["x"], d["y"]))
        if local.size != centre.shape[0] or not np.allclose(
            mesh.face_centre[local], centre, rtol=1e-9, atol=1e-12
        ):
            return None
        return order_boundary_faces(mesh.face_nodes[local], mesh.nodes)


def _scaled_mesh(mesh, factor):
    if factor == 1.0:
        return mesh
    boundary = {
        name: mesh.face_nodes[faces] for name, faces in mesh.boundary_faces.items()
    }
    cells = [mesh.cell(i) for i in range(mesh.n_cells)]
    return Mesh.from_cells(mesh.nodes * factor, cells, boundary)


def _mode_rows(cfg):
    """Build the configuration rows for the time mode and iteration control."""
    solver, tm, num = cfg["solver"], cfg["time"], cfg["numerics"]
    cfl, nk = solver["cfl"], solver["newton_krylov"]
    unsteady = tm["mode"] == "unsteady"
    if unsteady:
        n_steps = int(round(tm["end_time"] / tm["time_step"]))
        rows = [
            (
                "mode",
                f"unsteady, {tm['scheme'].upper()}, time step {tm['time_step']} s, "
                f"end time {tm['end_time']} s ({n_steps} steps)",
            ),
            (
                "inner iterations",
                f"per step up to {tm['inner_iterations']}, stop after "
                f"{tm['inner_residual_drop']} orders, CFL {tm['inner_cfl']:.0e}, "
                "Newton-Krylov from the first inner iteration",
            ),
        ]
    else:
        rows = [
            ("mode", "steady (pseudo-time iterations; no physical time)"),
            (
                "pseudo-time",
                f"CFL start {cfl['start']:g}, max {cfl['max_approximate']:.0e} "
                f"before Newton-Krylov and {cfl['max']:.0e} with it, "
                + (
                    f"SER (RMS ratio)^{cfl['ser_exponent']:g}, at least "
                    f"x{cfl['ser_growth_min']:g} while falling, in "
                    f"[{cfl['ser_ratio_min']:g}, {cfl['ser_ratio_max']:g}], hold from "
                    f"{cfl['ser_gmres_hold']} and halve above "
                    f"{cfl['ser_gmres_reduce']} GMRES iterations"
                    if cfl["control"] == "ser"
                    else f"growth x{cfl['growth_factor']:g}"
                )
                + f", cut x{cfl['cut_factor']:g}; stop after "
                f"{solver['residual_drop']:g} orders or {solver['max_iterations']} "
                "iterations",
            ),
        ]
        if nk["enabled"] and solver["method"] == "implicit":
            when = f"after {nk['switch_at_residual_drop']:g} orders"
            if nk["switch_at_iteration"] is not None:
                when += f" or from iteration {nk['switch_at_iteration']}"
            rows.append(("Newton-Krylov", when))
        else:
            rows.append(("Newton-Krylov", "off"))
    conv = solver["convergence"]
    if conv["fields"] and not unsteady and solver["method"] == "implicit":
        rows.append(
            (
                "Cauchy convergence",
                f"{', '.join(conv['fields'])}: mean relative change < "
                f"{conv['cauchy_eps']:g} over {conv['cauchy_elements']} "
                f"iterations from iteration {conv['start_iteration']}",
            )
        )
    tconv = cfg["time"]["convergence"]
    if tconv["fields"] and unsteady:
        rows.append(
            (
                "Cauchy convergence (time)",
                f"time average of {', '.join(tconv['fields'])}: mean relative "
                f"change < {tconv['cauchy_eps']:g} over {tconv['cauchy_elements']} "
                f"steps, {tconv['window_function']} window from step "
                f"{tconv['window_start_step']}",
            )
        )
    tr = cfg["physics"]["transition"]
    if tr["model"] == "lm":
        if tr["clip"]:
            (g_lo, g_hi), (r_lo, r_hi) = lm.GAMMA_LIMITS, lm.RE_THETA_LIMITS
            text = (
                f"clipped as in SU2: gamma [{g_lo:g}, {g_hi:g}], "
                f"Re_thetat [{r_lo:g}, {r_hi:g}]"
            )
        else:
            text = "not clipped (physics.transition.clip = false)"
        rows.append(("gamma, Re_thetat", text))
        rows.append(
            (
                "gamma Newton step",
                "descent guard: an update the clip floor would discard against "
                "the residual becomes the explicit local step at CFL 1 "
                "(physics.transition.newton_guard = true)"
                if tr["newton_guard"]
                else "plain Newton (no descent guard)",
            )
        )
        rows.append(
            (
                "gamma linearization",
                "positive: the sqrt(gamma) production derivative and the negative "
                "destruction derivative are left out of the Newton-Krylov "
                "Jacobian, the residual is unchanged"
                if tr["positive_jacobian"]
                else "exact: Newton-Krylov differentiates the sources fully "
                "(physics.transition.positive_jacobian = false)",
            )
        )
    if (
        solver["nu_tilde_step_limit"] is not None
        and cfg["physics"]["turbulence"] != "sst"
        and (unsteady or solver["method"] == "implicit")
    ):
        rows.append(
            (
                "nu_tilde step",
                f"capped at {solver['nu_tilde_step_limit']:g} x "
                "max(|nu_tilde|, nu) per cell"
                + (" (inner iterations)" if unsteady else ""),
            )
        )
    if num["first_order_iterations"]:
        n_first = num["first_order_iterations"]
        rows.append(("first-order start-up", f"{n_first} iterations"))
    freeze = num["limiter_freeze"]
    if num["limiter"] != "none" and num["spatial_order"] == 2:
        when = []
        if freeze["after_residual_drop"] is not None:
            when.append(f"after {freeze['after_residual_drop']:g} orders")
        if freeze["after_iterations"] is not None:
            when.append(f"at iteration {freeze['after_iterations']}")
        if when:
            action = "released" if freeze["release"] else "frozen"
            rows.append(("limiter", f"{action} {' or '.join(when)}"))
    return rows


def _log_setup(log, config, problem, scaling, mesh):
    cfg = config.data
    log.section("Configuration")
    rows = [
        ("model", cfg["physics"]["model"]),
        ("density", f"{cfg['fluid']['density']} kg/m^3"),
    ]
    if cfg["fluid"]["viscosity"] is not None:
        rows.append(("viscosity", f"{cfg['fluid']['viscosity']} Pa s"))
    if cfg["fluid"]["reynolds_number"] is not None:
        rows.append(("Reynolds number", f"{cfg['fluid']['reynolds_number']}"))
    rows += [
        (
            "reference",
            f"rho {scaling.density} kg/m^3, U {scaling.velocity} m/s, "
            f"L {scaling.length} m",
        ),
        ("viscosity (non-dim.)", f"{problem.model.viscosity:.4e} (= 1/Re)"),
        (
            "convective flux",
            f"{cfg['numerics']['convective_flux']}, order "
            f"{cfg['numerics']['spatial_order']}, limiter "
            f"{cfg['numerics']['limiter']}",
        ),
        ("viscous gradient", cfg["numerics"]["viscous_gradient"]),
        ("solver", cfg["solver"]["method"]),
    ]
    rows += _mode_rows(cfg)
    rows += _field_rows(cfg, mesh)
    rows.append(("resolved configuration", f"{config.name}.resolved.toml"))
    log.table(rows)
    log.section("Mesh")
    scale = cfg["mesh"]["scale"]
    log.write(
        f"  file: {cfg['mesh']['file']}" + (f" (scale {scale})" if scale != 1 else "")
    )
    for line in mesh.summary().splitlines():
        log.write("  " + line)
    log.section("Boundary conditions")
    log.table([(name, _describe_boundary(b)) for name, b in cfg["boundary"].items()])
    if problem.turbulent:
        log.section("Turbulence")
        tv = problem.turbulence_values
        rows = [("model", cfg["physics"]["turbulence"])]
        restarted = cfg["initial"]["restart"] is not None
        for name in tv:
            if name == "initial" and restarted:
                # The restart file overrides these; they only fill variables
                # that the file lacks (see the notice on the initial state).
                rows.append(
                    (name, f"from the restart file (default {tv[name].report()})")
                )
            else:
                rows.append((name, tv[name].report()))
        if cfg["physics"]["transition"]["mode"] == "prescribed":
            n_off = int(
                np.count_nonzero(np.asarray(problem.geom.production_switch) == 0)
            )
            rows.append(
                ("laminar regions", f"{n_off} cells without turbulence production")
            )
        if problem.model.transition == "aft":
            tu = problem.model.turbulence_intensity
            n_crit = rans.critical_amplification(problem.model)
            given = problem.model.critical_amplification is not None
            rows.append(
                (
                    "transition",
                    f"aft (SA-AFT2019b), N_crit = {n_crit:.3g}"
                    + (" (given)" if given else f" from Tu = {tu:.3g} %"),
                )
            )
        elif problem.model.transition == "lm":
            turb = np.asarray(problem.bc.turbulence)
            kind = np.asarray(problem.bc.kind)
            inflow = np.isin(
                kind, (inc.VELOCITY_INLET, inc.FARFIELD, inc.TOTAL_PRESSURE_INLET)
            )
            if inflow.any():
                re_t = turb[inflow, 3]
                text = (
                    f"lm (SST-2003m-LM2009), inflow Re_thetat = {re_t.min():.0f}"
                    + (f"-{re_t.max():.0f}" if re_t.max() > 1.001 * re_t.min() else "")
                    + " from the inflow Tu (local Tu from k in the field)"
                )
            else:
                text = "lm (SST-2003m-LM2009)"
            rows.append(("transition", text))
        elif problem.model.transition != "none":
            tu = problem.model.turbulence_intensity
            re_theta_t = 803.73 * (tu + 0.6067) ** (-1.027)
            rows.append(
                (
                    "transition",
                    f"{problem.model.transition} (SA-noft2), Tu = {tu:.3g} %, "
                    f"Re_theta_t = {re_theta_t:.0f}",
                )
            )
            omega = config["physics"]["transition"]["relaxation"]
            if omega < 1.0:
                rows.append(
                    (
                        "intermittency",
                        f"under-relaxed, omega = {omega:g} (steady runs)",
                    )
                )
        log.table(rows)
        bcm = problem.model.transition == "bcm"
        aft_model = problem.model.transition == "aft"
        for name, v in tv.items():
            if (name == "initial" and restarted) or v.chi is None:
                continue
            if aft_model:
                if not 0.05 <= v.chi <= 0.5:
                    config.notices.append(
                        f"{name}: nu_tilde/nu = {v.chi:.3g}; SA-AFT recommends about "
                        "0.1 in the freestream (Coder 2019)"
                    )
            elif bcm and v.chi > 0.5:
                config.notices.append(
                    f"{name}: nu_tilde/nu = {v.chi:.3g} with the BCM transition model: "
                    "its second term reads eddy viscosity as turbulence (nu_tilde/nu "
                    "= 3 gives gamma = 0.96 in the freestream); use about 0.1"
                )
            elif not bcm and not 0.99 * 3.0 <= v.chi <= 1.01 * 5.0:
                config.notices.append(
                    f"{name}: nu_tilde/nu = {v.chi:.3g} outside the NASA TMR range 3-5"
                )
    if problem.pinned_cell is not None:
        c = mesh.cell_centroid[problem.pinned_cell]
        config.notices.append(
            f"pressure pinned in cell {problem.pinned_cell} at ({c[0]:.4g}, {c[1]:.4g})"
        )
    if config.notices:
        log.section("Notices")
        for note in config.notices:
            log.write("  - " + note)


def _describe_boundary(b):
    parts = [b["type"]]
    for key, value in b.items():
        if key != "type":
            parts.append(f"{key} = {value}")
    return ", ".join(parts)


def _polar_config(raw, path, base, alpha, previous):
    """Resolve the case of one polar angle: ``raw`` with angle and restart set."""
    case = copy.deepcopy(raw)
    for b in case.get("boundary", {}).values():
        if b.get("type") == "farfield":
            b["angle_of_attack"] = float(alpha)
    case.setdefault("reference", {})["angle_of_attack"] = float(alpha)
    case.setdefault("case", {})["name"] = f"{base.name}_alpha_{alpha:g}"
    case.setdefault("output", {})["directory"] = str(
        Path(base.data["output"]["directory"]) / f"alpha_{alpha:g}"
    )
    if previous is not None:
        case.setdefault("initial", {})["restart"] = str(previous)
    return resolve(case, path)


def run_polar(path, angles, console=True):
    """Run a case at several angles of attack, restarting from the previous one.

    Every farfield boundary and ``reference.angle_of_attack`` take the angle;
    each run writes into ``<output.directory>/alpha_<angle>`` and the
    coefficients are collected in ``<output.directory>/<name>.polar.csv``
    (merged with earlier angles). The first angle restarts from the nearest
    angle computed before, if any.

    Returns
    -------
    list of RunResult
    """
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    base = load_case(path)
    out_dir = base.resolve_path(base.data["output"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    polar_csv = out_dir / f"{base.name}.polar.csv"
    rows = _read_polar(polar_csv)
    results, previous = [], None
    for alpha in angles:
        if previous is None:  # restart from the nearest angle computed before
            previous = _nearest_restart(out_dir, base.name, alpha)
        config = _polar_config(raw, path, base, alpha, previous)
        config.reload = lambda a=alpha, p=previous: _polar_config(
            tomllib.loads(path.read_text()), path, base, a, p
        )
        result = run_case(config, console=console, restart_mode="state")
        previous = result.restart_file
        c = result.coefficients
        rows = [r for r in rows if abs(r[0] - alpha) > 1e-9]
        rows.append(
            (
                alpha,
                c.cl,
                c.cd,
                c.cd_pressure,
                c.cd_viscous,
                c.cm,
                int(result.history["iteration"][-1])
                if "iteration" in result.history
                else int(result.history["step"][-1]),
                int(result.converged),
            )
        )
        results.append(result)
        rows.sort(key=lambda r: r[0])
        np.savetxt(
            polar_csv,
            np.array(rows, dtype=float),
            delimiter=",",
            header="alpha,CL,CD,CD_pressure,CD_viscous,CM,iterations,converged",
            comments="",
            fmt=["%.4f", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e", "%d", "%d"],
        )
    return results


def _read_polar(path):
    if not path.exists():
        return []
    data = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    return [tuple(row) for row in data] if data.size else []


def _nearest_restart(out_dir, name, alpha):
    """Newest backup of the previously computed angle nearest to ``alpha``.

    On a tie the lower angle wins (continuation upwards, as the restart
    chain of a polar is built); the angle itself is never a candidate.
    """
    best = None
    for d in out_dir.glob("alpha_*"):
        try:
            a = float(d.name[6:])
        except ValueError:
            continue
        file = latest_backup(d, f"{name}_alpha_{a:g}")
        if file is None or a == alpha:
            continue
        rank = (abs(a - alpha), a > alpha)
        if best is None or rank < best[0]:
            best = (rank, file)
    return None if best is None else best[1]

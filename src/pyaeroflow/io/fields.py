"""Field registry of the VTU output (D59).

Every cell field that ``output.fields`` can select is one :class:`Field`:
its name, unit, description, the model combinations it exists for and the
function that computes it. The registry drives the validation of
``output.fields``, the default selection, the writer and the field table of
docs/configuration.md (:func:`catalogue_table`), so the code and the
documentation cannot drift apart.
"""

import difflib
from collections.abc import Callable
from typing import NamedTuple

import numpy as np


class Capabilities(NamedTuple):
    """What a case provides: decides which fields exist."""

    model: str  # "euler", "laminar" or "rans"
    transition: str  # "none", "bcm", "aft" or "lm" (only meaningful with rans)
    steady: bool
    implicit: bool  # steady with the implicit solver
    walls: bool  # output.force_boundaries is not empty
    turbulence: str = "sa-neg"  # "sa-neg" or "sst" (only meaningful with rans)

    @classmethod
    def from_config(cls, data):
        """Capabilities of a resolved configuration (``Resolved.data``)."""
        physics = data["physics"]
        rans = physics["model"] == "rans"
        steady = data["time"]["mode"] == "steady"
        return cls(
            physics["model"],
            physics["transition"]["model"] if rans else "none",
            steady,
            steady and data["solver"]["method"] == "implicit",
            bool(data["output"]["force_boundaries"]),
            physics["turbulence"] if rans else "none",
        )


class Field(NamedTuple):
    """One selectable output field."""

    name: str
    unit: str
    description: str
    needs: str  # short text of the availability rule (documentation, errors)
    available: Callable  # Capabilities -> bool
    compute: Callable  # FieldContext -> ndarray, (n_cells,) or (n_cells, 2)


class FieldContext:
    """State a field is computed from; wall quantities are computed once."""

    def __init__(self, problem, w, scaling, monitor=None, cell_residual=None):
        self.problem, self.w, self.scaling = problem, np.asarray(w), scaling
        self.monitor = monitor
        self.cell_residual = cell_residual
        self._wall = None

    @property
    def nu_si(self):
        """Laminar kinematic viscosity in m^2/s."""
        m = self.problem.model
        return m.viscosity / m.density * self.scaling.viscosity / self.scaling.density

    def wall(self):
        """Per-cell wall quantities (non-dimensional), 0 away from walls.

        A cell with several wall faces gets the length-weighted mean.
        """
        if self._wall is None:
            self._wall = _wall_cells(self)
        return self._wall


def _wall_cells(ctx):
    problem, monitor = ctx.problem, ctx.monitor
    geom = problem.geom
    faces = problem.boundary_faces(monitor.boundaries)
    d = _fz().wall_distribution(
        problem.wall_loads(ctx.w), geom, faces, monitor.q, monitor.p_ref
    )
    normal = np.asarray(geom.b_normal)[faces]
    tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
    a = np.radians(monitor.alpha)
    # Attached flow positive: orient the tangent along the freestream.
    sign = np.where(tangent @ np.array([np.cos(a), np.sin(a)]) >= 0.0, 1.0, -1.0)
    values = {
        "cf": d["cf"] * sign,
        "cp_wall": d["cp"],
        "tau_wall_x": d["tau_x"],
        "tau_wall_y": d["tau_y"],
    }
    owner = np.asarray(geom.b_owner)[faces]
    length = np.asarray(geom.b_length)[faces]
    if problem.turbulent:
        m = problem.model
        dist = np.asarray(geom.wall_distance)[owner]
        tau = np.hypot(d["tau_x"], d["tau_y"])
        values["y_plus"] = dist * np.sqrt(tau / m.density) / (m.viscosity / m.density)
    weight = np.zeros(problem.mesh.n_cells)
    np.add.at(weight, owner, length)
    out = {}
    for key, face_values in values.items():
        cell = np.zeros(problem.mesh.n_cells)
        np.add.at(cell, owner, face_values * length)
        out[key] = np.divide(cell, weight, out=np.zeros_like(cell), where=weight > 0)
    return out


def _fz():
    from pyaeroflow.physics import forces

    return forces


# -- compute functions --------------------------------------------------------


def _pressure(c):
    return c.w[:, 0] * c.scaling.pressure


def _velocity(c):
    return c.w[:, 1:3] * c.scaling.velocity


def _speed(c):
    return np.hypot(c.w[:, 1], c.w[:, 2]) * c.scaling.velocity


def _vorticity(c):
    return c.problem.vorticity(c.w) * c.scaling.velocity / c.scaling.length


def _kinematic_viscosity(c):
    return np.full(c.w.shape[0], c.nu_si)


def _nu_tilde(c):
    return c.w[:, 3] * c.scaling.viscosity / c.scaling.density


def _eddy_viscosity(c):
    return c.problem.eddy_viscosity(c.w) * c.scaling.viscosity / c.scaling.density


def _eddy_viscosity_ratio(c):
    m = c.problem.model
    return c.problem.eddy_viscosity(c.w) / (m.viscosity / m.density)


def _effective_viscosity(c):
    nu_t = _eddy_viscosity(c) if c.problem.turbulent else 0.0
    return c.nu_si + nu_t


def _dynamic_eddy_viscosity(c):
    nu_t = c.problem.eddy_viscosity(c.w) * c.scaling.viscosity / c.scaling.density
    return c.problem.model.density * c.scaling.density * nu_t


def _turbulent_kinetic_energy(c):
    return c.w[:, 3] * c.scaling.velocity**2


def _specific_dissipation(c):
    return c.w[:, 4] * c.scaling.velocity / c.scaling.length


def _blending_f1(c):
    return c.problem.blending_f1(c.w)


def _wall_distance(c):
    return np.asarray(c.problem.geom.wall_distance) * c.scaling.length


def _intermittency(c):
    return c.problem.intermittency(c.w)


def _amplification_factor(c):
    return c.w[:, 5]


def _re_theta_t(c):
    return c.w[:, 6]


def _effective_intermittency(c):
    return c.problem.effective_intermittency(c.w)


def _residual(column, scale=None):
    def compute(c):
        # Residual per unit cell volume (area in 2D), non-dimensional; ``scale``
        # undoes a row scaling of the residual.
        r = np.asarray(c.cell_residual())[:, column] / np.asarray(c.problem.geom.area)
        return r if scale is None else r / scale(c)

    return compute


def _nu_nd(c):
    # The SST omega residual is solved multiplied by nu (rans_sst).
    return c.problem.model.viscosity / c.problem.model.density


def _ret_scale(c):
    # The Langtry-Menter Re_thetat residual is solved multiplied by RET_SCALE.
    return 1.0e-3


def _wall(key):
    return lambda c: c.wall()[key]


# -- availability rules -------------------------------------------------------


def _always(caps):
    return True


def _viscous(caps):
    return caps.model != "euler"


def _rans(caps):
    return caps.model == "rans"


def _sa(caps):
    return caps.model == "rans" and caps.turbulence != "sst"


def _sst(caps):
    return caps.model == "rans" and caps.turbulence == "sst"


def _implicit_sa(caps):
    return caps.implicit and _sa(caps)


def _implicit_sst(caps):
    return caps.implicit and _sst(caps)


def _transition(caps):
    return caps.model == "rans" and caps.transition != "none"


def _aft(caps):
    return caps.model == "rans" and caps.transition == "aft"


def _implicit(caps):
    return caps.implicit


def _implicit_aft(caps):
    return caps.implicit and _aft(caps)


def _lm(caps):
    return caps.model == "rans" and caps.transition == "lm"


def _implicit_lm(caps):
    return caps.implicit and _lm(caps)


def _wall_any(caps):
    return caps.walls


def _wall_viscous(caps):
    return caps.walls and caps.model != "euler"


def _wall_rans(caps):
    return caps.walls and caps.model == "rans"


FIELDS = (
    Field("pressure", "Pa", "Static pressure", "always", _always, _pressure),
    Field("velocity", "m/s", "Velocity vector", "always", _always, _velocity),
    Field("velocity_magnitude", "m/s", "Speed", "always", _always, _speed),
    Field(
        "vorticity",
        "1/s",
        "dv/dx - du/dy from the least-squares gradient (signed)",
        "always",
        _always,
        _vorticity,
    ),
    Field(
        "kinematic_viscosity",
        "m^2/s",
        "Laminar kinematic viscosity (constant)",
        "laminar, rans",
        _viscous,
        _kinematic_viscosity,
    ),
    Field(
        "eddy_viscosity",
        "m^2/s",
        "Kinematic eddy viscosity nu_t",
        "rans",
        _rans,
        _eddy_viscosity,
    ),
    Field(
        "eddy_viscosity_ratio",
        "1",
        "nu_t / nu",
        "rans",
        _rans,
        _eddy_viscosity_ratio,
    ),
    Field(
        "effective_viscosity",
        "m^2/s",
        "nu + nu_t (the laminar value without a turbulence model)",
        "laminar, rans",
        _viscous,
        _effective_viscosity,
    ),
    Field(
        "mu_t",
        "Pa s",
        "Dynamic eddy viscosity rho nu_t",
        "rans",
        _rans,
        _dynamic_eddy_viscosity,
    ),
    Field(
        "nu_tilde",
        "m^2/s",
        "SA working variable",
        "rans with turbulence = sa-neg",
        _sa,
        _nu_tilde,
    ),
    Field(
        "k",
        "m^2/s^2",
        "Turbulent kinetic energy (SST)",
        "rans with turbulence = sst",
        _sst,
        _turbulent_kinetic_energy,
    ),
    Field(
        "omega",
        "1/s",
        "Specific dissipation rate (SST)",
        "rans with turbulence = sst",
        _sst,
        _specific_dissipation,
    ),
    Field(
        "blending_f1",
        "1",
        "Menter's blending function F1 (1: k-omega near the wall, 0: k-epsilon)",
        "rans with turbulence = sst",
        _sst,
        _blending_f1,
    ),
    Field(
        "wall_distance",
        "m",
        "Distance to the nearest wall used by the turbulence model",
        "rans",
        _rans,
        _wall_distance,
    ),
    Field(
        "intermittency",
        "1",
        "Transition intermittency gamma",
        "rans with a transition model",
        _transition,
        _intermittency,
    ),
    Field(
        "amplification_factor",
        "1",
        "Amplification factor n of SA-AFT",
        "rans with transition.model = aft",
        _aft,
        _amplification_factor,
    ),
    Field(
        "re_theta_t",
        "1",
        "Transported transition-onset momentum-thickness Reynolds number "
        "Re_thetat of Langtry-Menter",
        "rans with transition.model = lm",
        _lm,
        _re_theta_t,
    ),
    Field(
        "intermittency_effective",
        "1",
        "Effective intermittency max(gamma, gamma_sep) on the SST production "
        "(gamma_sep > 1 marks separation-induced transition)",
        "rans with transition.model = lm",
        _lm,
        _effective_intermittency,
    ),
    Field(
        "residual_p",
        "1",
        "Pressure (continuity) residual per cell volume",
        "steady, implicit",
        _implicit,
        _residual(0),
    ),
    Field(
        "residual_u",
        "1",
        "x-momentum residual per cell volume",
        "steady, implicit",
        _implicit,
        _residual(1),
    ),
    Field(
        "residual_v",
        "1",
        "y-momentum residual per cell volume",
        "steady, implicit",
        _implicit,
        _residual(2),
    ),
    Field(
        "residual_nu_tilde",
        "1",
        "nu_tilde residual per cell volume",
        "steady, implicit, turbulence = sa-neg",
        _implicit_sa,
        _residual(3),
    ),
    Field(
        "residual_k",
        "1",
        "k residual per cell volume",
        "steady, implicit, turbulence = sst",
        _implicit_sst,
        _residual(3),
    ),
    Field(
        "residual_omega",
        "1",
        "omega residual per cell volume",
        "steady, implicit, turbulence = sst",
        _implicit_sst,
        _residual(4, _nu_nd),
    ),
    Field(
        "residual_gamma",
        "1",
        "log-intermittency residual per cell volume",
        "steady, implicit, transition.model = aft",
        _implicit_aft,
        _residual(4),
    ),
    Field(
        "residual_n",
        "1",
        "Amplification-factor residual per cell volume",
        "steady, implicit, transition.model = aft",
        _implicit_aft,
        _residual(5),
    ),
    Field(
        "residual_intermittency",
        "1",
        "Intermittency residual per cell volume",
        "steady, implicit, transition.model = lm",
        _implicit_lm,
        _residual(5),
    ),
    Field(
        "residual_re_theta_t",
        "1",
        "Re_thetat residual per cell volume",
        "steady, implicit, transition.model = lm",
        _implicit_lm,
        _residual(6, _ret_scale),
    ),
    Field(
        "cf",
        "1",
        "Skin-friction coefficient with attached flow positive on every side "
        "(the wall CSV keeps the sign of the wall traversal); wall cells only",
        "wall in output.force_boundaries, laminar or rans",
        _wall_viscous,
        _wall("cf"),
    ),
    Field(
        "cp_wall",
        "1",
        "Pressure coefficient at the wall; wall cells only",
        "wall in output.force_boundaries",
        _wall_any,
        _wall("cp_wall"),
    ),
    Field(
        "tau_wall_x",
        "1",
        "Wall shear stress x component (non-dimensional, rho U^2); wall cells only",
        "wall in output.force_boundaries, laminar or rans",
        _wall_viscous,
        _wall("tau_wall_x"),
    ),
    Field(
        "tau_wall_y",
        "1",
        "Wall shear stress y component (non-dimensional, rho U^2); wall cells only",
        "wall in output.force_boundaries, laminar or rans",
        _wall_viscous,
        _wall("tau_wall_y"),
    ),
    Field(
        "y_plus",
        "1",
        "y+ of the first cell centre; wall cells only",
        "wall in output.force_boundaries, rans",
        _wall_rans,
        _wall("y_plus"),
    ),
)

BY_NAME = {f.name: f for f in FIELDS}

# The fields written without ``output.fields`` (what the VTU always held).
DEFAULT_NAMES = (
    "pressure",
    "velocity",
    "velocity_magnitude",
    "nu_tilde",
    "k",
    "omega",
    "eddy_viscosity_ratio",
    "intermittency",
    "amplification_factor",
    "re_theta_t",
    "intermittency_effective",
)


def available(caps):
    """Names of the fields that exist for ``caps``, in registry order."""
    return [f.name for f in FIELDS if f.available(caps)]


def default_fields(caps):
    """Default selection: pressure, velocity, speed and the turbulence fields."""
    return [n for n in DEFAULT_NAMES if BY_NAME[n].available(caps)]


def resolve_fields(requested, caps):
    """Return the selected field names in registry order and the problems found.

    ``requested`` is the ``output.fields`` array; ``None`` selects the
    default set and ``["all"]`` every available field. An unknown name gets a
    "did you mean" hint, a known field that the case does not provide names
    what it needs.
    """
    problems = []
    if requested is None:
        return default_fields(caps), problems
    if "all" in requested:
        return available(caps), problems
    chosen = set()
    for name in requested:
        field = BY_NAME.get(name)
        if field is None:
            close = difflib.get_close_matches(str(name), list(BY_NAME), n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            problems.append(f"output.fields: unknown field '{name}'{hint}")
        elif not field.available(caps):
            problems.append(
                f"output.fields: '{name}' needs {field.needs} "
                f"(this case: model {caps.model}"
                + (
                    f", turbulence {caps.turbulence}, transition {caps.transition}"
                    if caps.model == "rans"
                    else ""
                )
                + f", {'steady' if caps.steady else 'unsteady'}"
                + (", implicit" if caps.implicit else "")
                + ")"
            )
        else:
            chosen.add(name)
    return [f.name for f in FIELDS if f.name in chosen], problems


def compute_fields(names, ctx):
    """``{name: array}`` of the named fields for the write."""
    return {n: np.asarray(BY_NAME[n].compute(ctx)) for n in names}


def n_columns(names):
    """(scalar fields, vector fields) among ``names``."""
    vectors = sum(1 for n in names if n == "velocity")
    return len(names) - vectors, vectors


def catalogue_table():
    """Markdown table of all fields for docs/configuration.md."""
    cases = (
        ("euler", Capabilities("euler", "none", True, True, True)),
        ("laminar", Capabilities("laminar", "none", True, True, True)),
        ("SA", Capabilities("rans", "none", True, True, True)),
        ("SA + BCM", Capabilities("rans", "bcm", True, True, True)),
        ("SA + AFT", Capabilities("rans", "aft", True, True, True)),
        ("SST", Capabilities("rans", "none", True, True, True, "sst")),
        ("SST + LM", Capabilities("rans", "lm", True, True, True, "sst")),
    )
    head = "| Field | Unit | Description | Needs | " + " | ".join(c[0] for c in cases)
    lines = [head + " |", "|" + "---|" * (4 + len(cases))]
    for f in FIELDS:
        marks = " | ".join("x" if f.available(c[1]) else "" for c in cases)
        lines.append(
            f"| `{f.name}` | {f.unit} | {f.description} | {f.needs} | {marks} |"
        )
    return "\n".join(lines)

"""Reading, validating and resolving a TOML case file (D27-D33, D38, D39)."""

import difflib
import tomllib
from pathlib import Path

import numpy as np

from pyaeroflow.config.schema import (
    BOUNDARY_TYPES,
    PLANNED,
    PLANNED_BOUNDARY_TYPES,
    REQUIRED,
    SCHEMA,
    TURBULENCE_FORMS,
    Opt,
)

PRESSURE_FIXING = {"pressure-outlet", "farfield", "total-pressure-inlet"}


class ConfigError(ValueError):
    """Invalid case configuration; the message lists all problems found."""


class Resolved:
    """A validated configuration with all defaults filled in.

    Attributes
    ----------
    data : dict
        Nested values, ``data[section][key]`` and ``data["boundary"][name][key]``.
    explicit : set of str
        Dotted paths of the keys the user set.
    path : Path
        Case file (for relative paths).
    notices : list of str
        Informational messages produced while resolving.
    """

    def __init__(self, data, explicit, path, notices):
        self.data = data
        self.explicit = explicit
        self.path = Path(path)
        self.notices = notices

    def __getitem__(self, section):
        """Return the values of a section."""
        return self.data[section]

    @property
    def name(self):
        """Case name."""
        return self.data["case"]["name"]

    def resolve_path(self, relative):
        """Return an absolute path for one given relative to the case file."""
        return (self.path.parent / relative).resolve()

    def to_toml(self):
        """Serialise with a comment per key: description, unit, set/default."""
        lines = [f"# PyAeroFlow resolved configuration of {self.path.name}", ""]
        for section, options in SCHEMA.items():
            lines.append(f"[{section}]")
            lines += self._table_lines(section, options, self.data[section])
            lines.append("")
        for name, values in self.data["boundary"].items():
            lines.append(f'[boundary."{name}"]')
            opts = {"type": Opt("string", REQUIRED, "Boundary-condition type")}
            opts.update(BOUNDARY_TYPES[values["type"]])
            lines += self._table_lines(f"boundary.{name}", opts, values)
            lines.append("")
        return "\n".join(lines)

    def _table_lines(self, prefix, options, values):
        """Write scalar keys first, then the sub-tables.

        TOML attaches keys to the most recent table header.
        """
        lines = []
        for key, opt in options.items():
            if isinstance(opt, dict):
                continue
            value = values.get(key)
            if value is None:
                continue
            origin = "set" if f"{prefix}.{key}" in self.explicit else "default"
            unit = f" [{opt.unit}]" if opt.unit else ""
            lines.append(
                f"{key} = {_toml_value(value)}  # {opt.description}{unit} ({origin})"
            )
        for key, opt in options.items():
            if isinstance(opt, dict):
                sub = self._table_lines(f"{prefix}.{key}", opt, values[key])
                if sub:
                    lines += ["", f"[{prefix}.{key}]", *sub]
        return lines


def load_case(path):
    """Read and validate a case file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    Resolved

    Raises
    ------
    ConfigError
        With every problem found, one per line.
    """
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return resolve(raw, path)


def resolve(raw, path="case.toml"):
    """Validate a parsed case dictionary and fill in defaults."""
    problems, notices, explicit = [], [], set()
    data = {}
    known_sections = set(SCHEMA) | {"boundary"}
    for section in raw:
        if section not in known_sections:
            problems.append(_unknown("section", section, known_sections))
    for section, options in SCHEMA.items():
        values = raw.get(section, {})
        if not isinstance(values, dict):
            problems.append(f"[{section}] must be a table")
            values = {}
        data[section] = _resolve_table(section, options, values, explicit, problems)

    data["boundary"] = {}
    boundaries = raw.get("boundary", {})
    if not isinstance(boundaries, dict) or any(
        not isinstance(v, dict) for v in boundaries.values()
    ):
        problems.append("[boundary.<name>] tables expected")
        boundaries = {}
    for name, values in boundaries.items():
        data["boundary"][name] = _resolve_boundary(name, values, explicit, problems)

    if problems:
        raise ConfigError("invalid case configuration:\n  " + "\n  ".join(problems))
    _consistency(data, explicit, problems, notices, path)
    if problems:
        raise ConfigError("invalid case configuration:\n  " + "\n  ".join(problems))
    if data["case"]["name"] is None:
        data["case"]["name"] = Path(path).stem
    return Resolved(data, explicit, path, notices)


def _resolve_table(prefix, options, values, explicit, problems):
    out = {}
    for key in values:
        if key not in options:
            dotted = f"{prefix}.{key}"
            if dotted in PLANNED:
                problems.append(
                    f"{dotted}: not available yet (planned for {PLANNED[dotted]})"
                )
            else:
                problems.append(
                    _unknown("key", dotted, [f"{prefix}.{k}" for k in options])
                )
    for key, opt in options.items():
        dotted = f"{prefix}.{key}"
        if isinstance(opt, dict):
            sub = values.get(key, {})
            if not isinstance(sub, dict):
                problems.append(f"{dotted} must be a table")
                sub = {}
            out[key] = _resolve_table(dotted, opt, sub, explicit, problems)
            continue
        if key in values:
            explicit.add(dotted)
            out[key] = _check_value(dotted, opt, values[key], problems)
        elif opt.default is REQUIRED:
            problems.append(f"{dotted} is required")
            out[key] = None
        else:
            out[key] = (
                list(opt.default) if isinstance(opt.default, list) else opt.default
            )
    return out


def _resolve_boundary(name, values, explicit, problems):
    prefix = f"boundary.{name}"
    kind = values.get("type")
    if kind is None:
        problems.append(f"{prefix}.type is required")
        return {"type": None}
    if kind in PLANNED_BOUNDARY_TYPES:
        problems.append(
            f"{prefix}.type = {kind!r}: not available yet "
            f"(planned for {PLANNED_BOUNDARY_TYPES[kind]})"
        )
        return {"type": None}
    if kind not in BOUNDARY_TYPES:
        problems.append(_unknown(f"{prefix}.type", kind, BOUNDARY_TYPES))
        return {"type": None}
    explicit.add(f"{prefix}.type")
    options = BOUNDARY_TYPES[kind]
    rest = {k: v for k, v in values.items() if k != "type"}
    out = _resolve_table(prefix, options, rest, explicit, problems)
    out = {"type": kind, **out}
    return out


LAMINAR_REGION_KEYS = ("boundary", "side", "x_min", "x_max", "y_min", "y_max", "ramp")


def _transition_by_alpha(tr, boundaries, farfields, ref, problems):
    """Generate laminar_region from the per-angle transition table (D15)."""
    dotted = "physics.transition"
    if tr["laminar_region"]:
        problems.append(f"{dotted}: give either laminar_region or by_alpha")
        return
    if tr["mode"] != "prescribed":
        problems.append(f"{dotted}.by_alpha needs mode = prescribed")
        return
    if tr["boundary"] is None:
        problems.append(f"{dotted}.boundary is required with by_alpha")
        return
    if tr["boundary"] not in boundaries:
        problems.append(_unknown(dotted + ".boundary", tr["boundary"], boundaries))
        return
    table = tr["by_alpha"]
    for i, row in enumerate(table):
        for key in ("alpha", "upper", "lower"):
            if not isinstance(row.get(key), int | float) or isinstance(
                row.get(key), bool
            ):
                problems.append(f"{dotted}.by_alpha[{i}]: {key} must be a number")
                return
    alpha = ref["angle_of_attack"]
    if alpha is None:
        angles = {boundaries[n].get("angle_of_attack", 0.0) for n in farfields}
        if len(angles) != 1:
            problems.append(
                f"{dotted}.by_alpha: the angle of attack is not defined (give "
                "reference.angle_of_attack)"
            )
            return
        alpha = angles.pop()
    rows = sorted(table, key=lambda r: r["alpha"])
    a = np.array([r["alpha"] for r in rows], dtype=float)
    regions = []
    for side in ("upper", "lower"):
        x = float(np.interp(alpha, a, [r[side] for r in rows]))
        regions.append(
            {"boundary": tr["boundary"], "side": side, "x_max": x, "ramp": tr["ramp"]}
        )
    tr["laminar_region"] = regions


def _turbulence_table(holder, dotted, problems):
    """Check a turbulence table (D33): exactly one of the three forms."""
    table = holder.get("turbulence")
    if table is None:
        return
    dotted += ".turbulence"
    forms = [f for f, keys in TURBULENCE_FORMS.items() if any(k in table for k in keys)]
    allowed = tuple(k for keys in TURBULENCE_FORMS.values() for k in keys)
    for key in table:
        if key not in allowed:
            problems.append(_unknown(dotted + " key", key, allowed))
            return
    if len(forms) != 1:
        problems.append(
            f"{dotted}: give exactly one form, {{turbulent_viscosity_ratio, "
            "intensity}, {kinetic_energy, length_scale, c_mu} or {nu_tilde_ratio}"
        )
        return
    form = forms[0]
    required = {
        "a": ("kinetic_energy", "length_scale"),
        "b": ("turbulent_viscosity_ratio",),
        "c": ("nu_tilde_ratio",),
    }[form]
    for key in required:
        if key not in table:
            problems.append(f"{dotted}: {key} is required")
    for key, value in table.items():
        if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
            problems.append(f"{dotted}.{key}: expected a number >= 0, got {value!r}")


def _check_value(dotted, opt, value, problems):
    ok = True
    if opt.type == "float":
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        value = float(value) if ok else value
    elif opt.type == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif opt.type == "bool":
        ok = isinstance(value, bool)
    elif opt.type == "string":
        ok = isinstance(value, str)
    elif opt.type == "float2":
        ok = (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(v, int | float) and not isinstance(v, bool) for v in value
            )
        )
        value = [float(v) for v in value] if ok else value
    elif opt.type == "string_list":
        ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
    elif opt.type == "int_list":
        ok = isinstance(value, list) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        )
    elif opt.type == "table":
        ok = isinstance(value, dict)
    elif opt.type == "table_list":
        ok = isinstance(value, list) and all(isinstance(v, dict) for v in value)
    if not ok:
        kind = {
            "float2": "array of two numbers",
            "string_list": "array of strings",
            "int_list": "array of integers",
            "table": "table",
            "table_list": "array of tables",
        }
        problems.append(
            f"{dotted}: expected {kind.get(opt.type, opt.type)}, got {value!r}"
        )
        return value
    if opt.choices and value not in opt.choices:
        problems.append(f"{dotted}: {value!r} is not one of {list(opt.choices)}")
    if opt.minimum is not None and (
        min(value, default=opt.minimum) < opt.minimum
        if opt.type in ("float2", "int_list")
        else value < opt.minimum
    ):
        problems.append(f"{dotted}: must be >= {opt.minimum}, got {value!r}")
    return value


def _consistency(data, explicit, problems, notices, path):
    """Cross-key rules (D31, D28, D39 R1/R5/R6, configuration.md)."""
    fluid, physics, ref = data["fluid"], data["physics"], data["reference"]
    boundaries = data["boundary"]
    types = {name: b["type"] for name, b in boundaries.items()}
    farfields = [n for n, t in types.items() if t == "farfield"]
    # Several farfield boundaries with the same freestream (e.g. PyAero's
    # inlet/outlet/top/bottom) count as one for the defaults below.
    streams = {
        (
            boundaries[n].get("velocity"),
            boundaries[n].get("angle_of_attack"),
            boundaries[n].get("pressure"),
            repr(boundaries[n].get("turbulence")),
        )
        for n in farfields
    }
    if len(streams) == 1:
        farfields = farfields[:1]

    # Periodic pairs: partner exists and has no own condition.
    for name, b in boundaries.items():
        if b["type"] == "periodic":
            partner = b["partner"]
            if partner in boundaries:
                problems.append(
                    f"boundary.{partner}: is the periodic partner of '{name}' and "
                    "must not have its own entry"
                )
            if partner == name:
                problems.append(f"boundary.{name}: periodic partner is itself")

    # RANS: turbulence model, transition regions, turbulence tables.
    rans = physics["model"] == "rans"
    if rans and physics["turbulence"] is None:
        physics["turbulence"] = "sa-neg"
    elif not rans and physics["turbulence"] is not None:
        notices.append("physics.turbulence ignored: physics.model is not rans")
    tr = physics["transition"]
    if tr["by_alpha"]:
        _transition_by_alpha(tr, boundaries, farfields, ref, problems)
    if tr["mode"] == "prescribed":
        if not rans:
            problems.append("physics.transition needs physics.model = rans")
        if not tr["laminar_region"]:
            problems.append("physics.transition.mode = prescribed needs laminar_region")
        for i, region in enumerate(tr["laminar_region"] or []):
            dotted = f"physics.transition.laminar_region[{i}]"
            for key in region:
                if key not in LAMINAR_REGION_KEYS:
                    problems.append(_unknown(dotted + " key", key, LAMINAR_REGION_KEYS))
            if "boundary" not in region:
                problems.append(f"{dotted}: boundary is required")
            elif region["boundary"] not in boundaries:
                problems.append(
                    _unknown(dotted + ".boundary", region["boundary"], boundaries)
                )
            if region.get("side") not in (None, "upper", "lower"):
                problems.append(f"{dotted}.side: must be 'upper' or 'lower'")
    elif tr["laminar_region"]:
        notices.append("physics.transition.laminar_region ignored: mode = none")
    if data["output"]["transition_diagnostics"] and tr["model"] == "none":
        problems.append("output.transition_diagnostics needs a transition model")
    if not 0.0 < tr["relaxation"] <= 1.0:
        problems.append("physics.transition.relaxation must be in (0, 1]")
    elif tr["relaxation"] < 1.0:
        if tr["model"] == "none":
            problems.append("physics.transition.relaxation needs a transition model")
        if data["time"]["mode"] == "unsteady":
            problems.append("physics.transition.relaxation is for steady runs only")
    if tr["model"] != "none":
        if not rans:
            problems.append("physics.transition.model needs physics.model = rans")
        if tr["turbulence_intensity"] is None:
            tu = [
                b["turbulence"]["intensity"]
                for b in boundaries.values()
                if isinstance(b.get("turbulence"), dict)
                and b["turbulence"].get("intensity") is not None
            ]
            if tu:
                tr["turbulence_intensity"] = float(tu[0])
            else:
                problems.append(
                    "physics.transition.turbulence_intensity is required for "
                    f"model = {tr['model']} (fraction, e.g. 0.001 = 0.1 %) unless an "
                    "inflow turbulence table gives intensity"
                )
    elif tr["turbulence_intensity"] is not None:
        notices.append("physics.transition.turbulence_intensity ignored: model = none")
    if rans:
        _turbulence_table(data["initial"], "initial", problems)
        for name, b in boundaries.items():
            if "turbulence" in b:
                _turbulence_table(b, f"boundary.{name}", problems)
    else:
        for holder, dotted in [(data["initial"], "initial")] + [
            (b, f"boundary.{n}") for n, b in boundaries.items()
        ]:
            if holder.get("turbulence") is not None:
                notices.append(
                    f"{dotted}.turbulence ignored: physics.model is not rans"
                )

    if ref["vortex_correction"] and not farfields:
        problems.append("reference.vortex_correction needs a farfield boundary")

    # Unsteady runs need the time step and the final time.
    tm = data["time"]
    if tm["mode"] == "unsteady":
        for key in ("time_step", "end_time"):
            if tm[key] is None:
                problems.append(f"time.{key} is required with time.mode = unsteady")
        if tm["time_step"] and tm["end_time"] and tm["end_time"] < tm["time_step"]:
            problems.append("time.end_time must not be smaller than time.time_step")
    elif any(tm[k] is not None for k in ("time_step", "end_time")):
        notices.append("time.time_step / end_time ignored in steady mode")

    # Viscosity: exactly one of viscosity / reynolds_number, unless Euler.
    has_mu, has_re = (
        fluid["viscosity"] is not None,
        fluid["reynolds_number"] is not None,
    )
    if physics["model"] == "euler":
        if has_mu or has_re:
            notices.append("physics.model = euler: fluid viscosity ignored")
    elif has_mu == has_re:
        problems.append(
            "fluid: exactly one of viscosity and reynolds_number is required"
        )

    # Reference velocity: from the farfield if there is exactly one.
    if ref["velocity"] is None:
        if len(farfields) == 1:
            ref["velocity"] = boundaries[farfields[0]]["velocity"]
            notices.append(
                f"reference.velocity taken from farfield boundary '{farfields[0]}'"
            )
        elif has_re or data["initial"]["from_boundary"] is None and not farfields:
            problems.append(
                "reference.velocity is required (no single farfield boundary to "
                "take it from)"
            )
    if has_re and "reference.length" not in explicit:
        problems.append("reference.length is required with fluid.reynolds_number")
    if ref["density"] is None:
        ref["density"] = fluid["density"]
    if ref["angle_of_attack"] is None:
        ref["angle_of_attack"] = (
            boundaries[farfields[0]]["angle_of_attack"] if len(farfields) == 1 else 0.0
        )
    if ref["moment_center"] is None:
        ref["moment_center"] = [0.25 * ref["length"], 0.0]

    # Force boundaries: default all walls (slip walls too in Euler mode);
    # names must exist and be wall-type boundaries.
    body = ("wall", "transpiration-wall", "slip-wall")
    walls = [
        n
        for n, t in types.items()
        if t in body and (t != "slip-wall" or physics["model"] == "euler")
    ]
    out = data["output"]
    if out["force_boundaries"] is None:
        out["force_boundaries"] = walls
    for n in out["force_boundaries"]:
        if n not in boundaries:
            problems.append(_unknown("output.force_boundaries entry", n, boundaries))
        elif types[n] not in body:
            problems.append(
                f"output.force_boundaries: '{n}' is not a wall (type {types[n]!r})"
            )

    # Initial state: explicit values or from_boundary (default: the farfield).
    init = data["initial"]
    if init["from_boundary"] is not None and (
        init["velocity"] is not None or init["pressure"] is not None
    ):
        problems.append(
            "initial: give either from_boundary or explicit values, not both"
        )
    if (
        init["restart"] is None
        and init["from_boundary"] is None
        and init["velocity"] is None
    ):
        if len(farfields) == 1:
            init["from_boundary"] = farfields[0]
            notices.append(
                f"initial state taken from farfield boundary '{farfields[0]}'"
            )
        else:
            problems.append(
                "[initial] is required: give velocity (and pressure) or from_boundary"
            )
    if init["from_boundary"] is not None:
        src = init["from_boundary"]
        if src not in boundaries:
            problems.append(_unknown("initial.from_boundary", src, boundaries))
        elif boundaries[src]["type"] not in ("farfield", "velocity-inlet"):
            problems.append(
                f"initial.from_boundary = '{src}': must be a farfield or velocity "
                "inlet boundary"
            )
    if init["pressure"] is None:
        init["pressure"] = 0.0

    # Pressure level and periodic driving force.
    if not any(t in PRESSURE_FIXING for t in types.values()):
        notices.append(
            "no boundary fixes the pressure level: pressure pinned to "
            "initial.pressure in the cell nearest reference.pressure_point"
        )
    if (
        physics["mean_pressure_gradient"] is not None
        and "periodic" not in types.values()
    ):
        notices.append("physics.mean_pressure_gradient set without periodic boundaries")

    # Euler: walls become slip walls (D39 R5).
    if physics["model"] == "euler":
        for name, t in types.items():
            if t == "wall":
                notices.append(
                    f"physics.model = euler: wall '{name}' treated as slip-wall"
                )

    if data["output"]["formats"] and set(data["output"]["formats"]) - {"vtu"}:
        problems.append("output.formats: only 'vtu' is available")


def _unknown(what, name, candidates):
    close = difflib.get_close_matches(str(name), [str(c) for c in candidates], n=1)
    hint = f" (did you mean '{close[0]}'?)" if close else ""
    return f"unknown {what} '{name}'{hint}"


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list | tuple | np.ndarray):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items())
        return "{ " + items + " }"
    raise TypeError(f"cannot serialise {value!r}")

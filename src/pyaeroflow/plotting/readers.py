"""Readers for the plot tool: solver logs, CSV outputs and reference data.

A reader returns :class:`Table` objects: named columns (lower-case names)
plus a zone title. Text tables are read in one format-tolerant reader that
covers the solver's CSV files (history, wall, polar, probes), plain CSV or
whitespace columns with a header, Tecplot ASCII (``variables=``, ``zone
t=``, as the NASA TMR files) and XFOIL output (polar files with a dashed
line under the header, ``CPWR`` / ``DUMP`` files with a ``#`` header).
"""

import re
from importlib import resources
from pathlib import Path
from typing import NamedTuple

import numpy as np


class Table(NamedTuple):
    """Columns of one data block."""

    columns: dict  # lower-case name -> ndarray
    title: str = ""
    source: str = ""

    def get(self, *names):
        """Return the first of ``names`` that exists (case-insensitive), or None."""
        for name in names:
            if name.lower() in self.columns:
                return self.columns[name.lower()]
        return None

    @property
    def label(self):
        """Legend text: source and zone title."""
        return f"{self.source} {self.title}".strip()


ALIASES = {
    "x/c": "x",
    "alpha, deg": "alpha",
    "top_xtr": "xtr_up",
    "bot_xtr": "xtr_lo",
}


def _number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def _split(line):
    return [t for t in re.split(r"[,\s]+", line.strip()) if t]


def _names(text):
    """Column names of a header line (quoted Tecplot names or CSV/whitespace)."""
    quoted = re.findall(r'"([^"]*)"', text)
    names = quoted or [t for t in re.split(r",|\s{2,}|\t", text.strip()) if t]
    if not quoted and len(names) == 1:
        names = text.split()
    names = [n.strip().lower() for n in names]
    return [ALIASES.get(n, n) for n in names]


def read_table(path, source=None):
    """Read a text table file; returns a list of :class:`Table` (one per zone)."""
    path = Path(path)
    source = path.stem if source is None else source
    names, comment_names, title = None, None, ""
    zones, rows = [], []

    def close():
        if rows and names:
            n = len(names)
            data = np.array([r[:n] + [np.nan] * (n - len(r)) for r in rows], float)
            zones.append(
                Table({k: data[:, i] for i, k in enumerate(names)}, title, source)
            )
        rows.clear()

    lines = path.read_text(errors="replace").splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if line.startswith("#"):
            tokens = _split(line[1:])
            if tokens and not any(_number(t) for t in tokens):
                comment_names = _names(line[1:])  # XFOIL CPWR / DUMP header
            continue
        if low.startswith("variables"):
            close()
            names = _names(line.split("=", 1)[1])
            continue
        if low.startswith("zone"):
            close()
            m = re.search(r't\s*=\s*"([^"]*)"', line, re.I)
            title = m.group(1) if m else ""
            continue
        if set(line) <= set("- "):  # XFOIL polar: dashes under the header
            continue
        tokens = _split(line)
        if all(_number(t) for t in tokens):
            if names is None and comment_names is not None:
                names = comment_names
            if names is None:
                names = [f"c{j}" for j in range(len(tokens))]
            rows.append([float(t) for t in tokens])
            continue
        # A text line: a header if the next non-empty line is numeric or dashes.
        nxt = next((x.strip() for x in lines[i + 1 :] if x.strip()), "")
        if nxt and (set(nxt) <= set("- ") or all(_number(t) for t in _split(nxt))):
            close()
            names = _names(line)
    close()
    return zones


# -- solver log -------------------------------------------------------------------


class Log(NamedTuple):
    """Rows of the iteration (steady) or time-step (unsteady) table of a log."""

    columns: dict  # key -> ndarray; residuals as "R_p", "R_rho", ...
    notes: list
    mode: str  # "steady" or "unsteady"
    residuals: list  # residual keys in header order


_KEY = {"wall[s]": "wall", "iter": "iter", "step": "step"}


def _is_header(tokens):
    return (
        tokens
        and tokens[0] in ("iter", "step")
        and "drop" in tokens
        and any(t.startswith("log(R_") for t in tokens)
    )


def read_log(path):
    """Read the iteration or time-step table of a ``<name>.log``.

    The columns are cut at the right edge of each header label (the table is
    right-aligned, fixed width), so empty cells (no NK, no GMRES, no
    coefficients) are read as such. The note column is the rest of the row.
    Unsteady logs with ``verbosity = "detailed"`` repeat inner-iteration
    tables; only the rows of the first table kind are read.
    """
    spans, mode, kind = None, None, None
    rows, notes = [], []
    keys, residuals = [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        tokens = line.split()
        if _is_header(tokens):
            if mode is None:
                mode = "steady" if tokens[0] == "iter" else "unsteady"
                kind = tokens[0]
            if tokens[0] != kind:
                spans = None  # an inner-iteration table: skip its rows
                continue
            spans, keys = [], []
            for m in re.finditer(r"\S+", line):
                spans.append(m.end())
                name = m.group()
                if name.startswith("log(R_"):
                    name = "R_" + name[len("log(R_") : -1]
                keys.append(_KEY.get(name, name))
            residuals = [k for k in keys if k.startswith("R_")]
            continue
        if spans is None or not tokens or not tokens[0].isdigit():
            continue
        values, start = {}, 0
        try:
            for key, end in zip(keys, spans, strict=True):
                if key == "note":
                    break
                cell = line[start:end].strip()
                start = end
                if key == "NK":
                    values[key] = 1.0 if cell == "NK" else 0.0
                elif cell == "":
                    values[key] = 0.0 if key == "GMRES" else np.nan
                else:
                    values[key] = float(cell)
            if any(np.isnan(values.get(r, np.nan)) for r in residuals):
                continue  # incomplete row (still being written)
        except ValueError:
            continue
        rows.append(values)
        notes.append(line[start:].strip() if "note" in keys else "")
    if mode is None:
        raise ValueError(f"{path}: no iteration or time-step table found")
    names = [k for k in keys if k != "note"]
    columns = {k: np.array([r.get(k, np.nan) for r in rows], float) for k in names}
    return Log(columns, notes, mode, residuals)


def read_history(path):
    """Read a ``<name>.history.csv`` as a :class:`Log` (no notes, no drop).

    Residuals are converted to log10 to match the log; ``linear_iterations``
    becomes GMRES, ``time_s`` the wall time.
    """
    table = read_table(path)[0].columns
    unsteady = "step" in table
    out, residuals = {}, []
    for name, values in table.items():
        if name.startswith("residual_"):
            key = "R_" + name[len("residual_") :]
            out[key] = np.log10(np.maximum(values, 1e-300))
            residuals.append(key)
    rename = {
        "iteration": "iter",
        "step": "step",
        "time": "time",
        "cfl": "CFL",
        "newton_krylov": "NK",
        "linear_iterations": "GMRES",
        "inner_iterations": "inner",
        "inner_residual_drop": "drop",
        "cl": "CL",
        "cd": "CD",
        "cm": "CM",
        "time_s": "wall",
    }
    for name, key in rename.items():
        if name in table:
            out[key] = table[name]
    n = len(next(iter(table.values())))
    return Log(out, [""] * n, "unsteady" if unsteady else "steady", residuals)


def notes_for(log_path, numbers, step=False):
    """Return the notes of a log for the iteration or step ``numbers``."""
    try:
        log = read_log(log_path)
    except (OSError, ValueError):
        return [""] * len(numbers)
    key = "step" if step else "iter"
    by_number = {
        int(n): note for n, note in zip(log.columns[key], log.notes, strict=True)
    }
    return [by_number.get(int(n), "") for n in numbers]


# -- reference data -------------------------------------------------------------------


def builtin_names():
    """Names of the built-in reference data sets."""
    folder = resources.files("pyaeroflow.plotting") / "reference"
    return sorted(
        Path(p.name).stem for p in folder.iterdir() if p.name.endswith((".dat", ".csv"))
    )


def builtin_path(name):
    """Path of a built-in reference data set."""
    folder = resources.files("pyaeroflow.plotting") / "reference"
    for suffix in (".dat", ".csv"):
        p = folder / f"{name}{suffix}"
        if p.is_file():
            return Path(str(p))
    raise ValueError(
        f"unknown built-in reference '{name}'; available: {', '.join(builtin_names())}"
    )


def read_reference(spec):
    """Read ``--ref`` data: ``file[:zone][@label]`` or ``builtin:name[:zone][@label]``.

    ``zone`` selects the zones whose title contains the text (case-insensitive),
    e.g. ``builtin:tmr_naca0012_cp_cfl3d_sa:alpha=10``; ``@label`` sets the
    legend text, e.g. ``penn_state.csv@Penn State 2007``.
    """
    spec, _, label = spec.partition("@")
    zone = ""
    if spec.startswith("builtin:"):
        name, _, zone = spec[len("builtin:") :].partition(":")
        path, source = builtin_path(name), name
    else:
        path = Path(spec)
        if not path.exists() and ":" in spec:
            head, _, zone = spec.rpartition(":")
            path = Path(head)
        source = path.stem
    tables = read_table(path, label or source)
    if zone:
        tables = [t for t in tables if zone.lower() in t.title.lower()]
        if not tables:
            raise ValueError(f"{spec}: no zone with '{zone}' in its title")
    return tables

"""Local mesh refinement inside a steady run (``[refinement]``, stage R4).

A refined run is a chain of ordinary runs, one per cycle, like the angles of
``pyaeroflow polar``:

* cycle ``k`` runs in ``<output.directory>/cycle_<k>`` with the case name
  ``<name>_cycle_<k>``; every cycle but the last ends after
  ``refinement.residual_drop`` orders or ``refinement.iterations``
  iterations, the last one uses the ``solver`` settings;
* after a cycle the indicator (``refinement.indicator``) is computed from the
  final state, the marked cells are split
  (:mod:`pyaeroflow.mesh.indicator`, :mod:`pyaeroflow.mesh.refine`), every
  new cell takes the state of its parent (injection) and the next cycle
  starts from it with the solver state of the previous cycle (CFL,
  Newton-Krylov switch, released limiter, vortex; like the next angle of a
  polar);
* a frozen limiter is not passed on: every cycle after a refinement freezes
  its own, by ``numerics.limiter_freeze`` or else after
  ``refinement.limiter_freeze`` orders of the cycle (a captured shock on the
  refined mesh makes the limiter chatter: NACA 0012 at M 0.8 stalled at 3
  orders of cycle 2 and converges 10 orders with it frozen);
* the problem is built anew on the refined mesh (geometry, gradient stencils,
  wall distance, Jacobian pattern, linear-solver analysis).

Every restart and backup file of a refined cycle stores the refinement log
(the splits, the wall options and the cycle number) as JSON. ``pyaeroflow
run`` with such a file as ``initial.restart`` replays the splits on the
original mesh file, so no new mesh format is needed. With ``[refinement]`` a
file of the case's own chain (``<name>_cycle_<k>`` or a cycle start file)
continues the chain from that cycle; a file of another case (the previous
angle of ``pyaeroflow polar``) gives its state averaged onto the original
mesh (area-weighted over the children of each original cell) and the chain
starts again at cycle 0, so every angle refines for its own flow.

``<output.directory>/<name>.refinement.csv`` lists the cycles (cells,
iterations, residual drop, coefficients, marked cells) and
``<name>.refinement.log`` the marking of each refinement.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pyaeroflow.config import Resolved
from pyaeroflow.config.case import refine_spec_args
from pyaeroflow.control import STOP, check_no_stop_file
from pyaeroflow.io.fields import BY_NAME, FieldContext, Freestream, compute_fields
from pyaeroflow.mesh import read_mesh
from pyaeroflow.mesh.indicator import RefineSpec, apply, names_used
from pyaeroflow.mesh.refine import Refinement

CSV_HEADER = (
    "cycle,cells,iterations,converged,residual_drop,CL,CD,CM,marked,cells_after"
)


def read_refinement_info(path):
    """Refinement log stored in a restart file, or None for an unrefined mesh.

    Returns
    -------
    dict or None
        ``log`` (list of split records), ``options`` (walls, corner_angle,
        decay), ``cycle`` and ``json`` (the stored text).
    """
    with np.load(path) as data:
        if "refinement" not in data.files:
            return None
        text = str(data["refinement"])
    info = json.loads(text)
    info["json"] = text
    return info


def replay_refinement(mesh, info):
    """Rebuild the refined mesh of ``info`` from the original ``mesh``."""
    return Refinement.replay(mesh, info["log"], **info["options"])


def _info_json(ref, options, cycle):
    return json.dumps({"log": ref.log, "options": options, "cycle": cycle})


def restrict(values, origin, area, n_cells):
    """Area-weighted mean of leaf ``values`` over each original cell.

    The inverse of the injection: ``origin`` maps every leaf cell to its
    original cell (:attr:`~pyaeroflow.mesh.refine.Refinement.origin`), ``area``
    is the leaf cell area and ``n_cells`` the number of original cells. A
    constant field is kept exactly.
    """
    values = np.asarray(values, dtype=float)
    w = np.asarray(area, dtype=float)
    shape = (n_cells,) + values.shape[1:]
    total = np.zeros(n_cells)
    np.add.at(total, origin, w)
    out = np.zeros(shape)
    np.add.at(out, origin, values * w.reshape((-1,) + (1,) * (values.ndim - 1)))
    return out / total.reshape((-1,) + (1,) * (values.ndim - 1))


def own_chain(path, name):
    """Tell whether ``path`` is a file of the refinement chain of case ``name``."""
    file = Path(path).name
    return file.startswith(f"{name}_cycle_") or file.startswith(f"{name}.cycle_")


def latest_chain_backup(directory, name):
    """Newest backup of a refined run of ``name`` in ``directory``, or None.

    The backups of the highest cycle in ``<directory>/cycle_<k>`` win; a run
    without refinement (backups of ``name`` itself) is the fallback.
    """
    from pyaeroflow.housekeeping import latest_backup

    directory = Path(directory)
    cycles = []
    for d in directory.glob("cycle_*"):
        try:
            cycles.append((int(d.name[6:]), d))
        except ValueError:
            continue
    for k, d in sorted(cycles, reverse=True):
        found = latest_backup(d, f"{name}_cycle_{k}")
        if found is not None:
            return found
    return latest_backup(directory, name)


def _cycle_config(base, k, final, previous, mode, extra, refined=True):
    """Return the resolved configuration of cycle ``k`` (a copy of ``base``)."""
    data = copy.deepcopy(base.data)
    rc = data["refinement"]
    name = f"{base.name}_cycle_{k}"
    data["case"]["name"] = name
    data["output"]["directory"] = str(
        Path(base.data["output"]["directory"]) / f"cycle_{k}"
    )
    if previous is not None:
        data["initial"]["restart"] = str(previous)
    if not final:
        if rc["iterations"] is not None:
            data["solver"]["max_iterations"] = rc["iterations"]
        data["solver"]["residual_drop"] = rc["residual_drop"]
    freeze = data["numerics"]["limiter_freeze"]
    if (
        k > 0
        and rc["limiter_freeze"]
        and freeze["after_iterations"] is None
        and freeze["after_residual_drop"] is None
    ):
        # A shock captured on the refined mesh: the limiter chatters and the
        # line search cuts every step (NACA 0012 M 0.8 stalled at 3 orders);
        # frozen after an order the cycle converges.
        freeze["after_residual_drop"] = rc["limiter_freeze"]
    if mode == "state" and refined:
        # After a refinement the polar option decides whether the CFL ramp
        # starts again; a start from another case keeps solver.cfl.polar_reset.
        data["solver"]["cfl"]["polar_reset"] = rc["cfl_reset"]
    config = Resolved(data, set(base.explicit), base.path, list(base.notices))
    config.refinement_cycle = True
    config.restart_extra = extra
    if base.reload is not None:
        config.reload = lambda: _cycle_config(
            base.reload(), k, final, previous, mode, extra, refined
        )
    return config


def indicator_fields(result, config, names):
    """Output fields ``names`` of the final state of a cycle, in SI units."""
    from pyaeroflow.run import Monitor

    problem, scaling = result.problem, result.scaling
    with np.load(result.restart_file) as data:
        u = np.array(data["w"])
    cfg = config.data
    monitor = (
        Monitor(config, problem, scaling) if cfg["output"]["force_boundaries"] else None
    )
    freestream = Freestream.from_config(
        cfg, scaling, problem.model, problem.compressible
    )
    ctx = FieldContext(problem, u, scaling, monitor, None, freestream)
    return compute_fields(names, ctx)


def _refine_state(config, mesh, ref, file, names, spec):
    """Mark and split ``ref`` by the state of restart ``file`` on ``mesh``.

    Returns the parent map (None if nothing was marked), the marking and
    the number of cells before the split.
    """
    from pyaeroflow.run import setup

    problem, scaling, _, _ = setup(config, mesh)
    state = SimpleNamespace(problem=problem, scaling=scaling, restart_file=file)
    fields = indicator_fields(state, config, names)
    n0 = ref.n_cells
    source, marking = apply(ref, fields, spec)
    return source, marking, n0


def _start_file(file, mesh, mapping, path, extra, drop_frozen_limiter=False):
    """Write the state of restart ``file`` as a start file on ``mesh``.

    ``mapping`` takes an array over the cells of ``file`` (first axis) to
    one over the cells of ``mesh``; it is applied to the state and to every
    per-cell array of the solver state (limiter, lagged intermittency).
    ``drop_frozen_limiter`` leaves out a frozen limiter (not a released one).
    """
    from pyaeroflow.run import load_restart_state, save_restart

    with np.load(file) as data:
        u = np.array(data["w"])
        step = int(data["step"]) if "step" in data.files else 0
        flow = str(data["flow"]) if "flow" in data.files else "incompressible"
    n_old = u.shape[0]
    state = load_restart_state(file) or {}
    state.pop("iteration", None)
    state.pop("file", None)
    if drop_frozen_limiter and "psi" in state and not np.all(state["psi"] == 1.0):
        # A limiter frozen on the coarser mesh does not fit the split cells:
        # the next cycle freezes its own (a released limiter, all ones, stays).
        del state["psi"]
    for key, value in list(state.items()):
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == n_old:
            state[key] = mapping(value)
    save_restart(path, mesh, mapping(u), step, None, state or None, flow, extra)
    return path


def _transfer(result, ref, source, out_dir, name, k, options):
    """Write the injected start file of cycle ``k`` on the refined mesh."""
    return _start_file(
        result.restart_file,
        ref.mesh,
        lambda v: v[source],
        out_dir / f"{name}.cycle_{k}_start.npz",
        {"refinement": _info_json(ref, options, k)},
        drop_frozen_limiter=True,
    )


def _restricted_start(file, info, mesh, out_dir, name):
    """Write the state of a refined restart ``file`` averaged onto ``mesh``."""
    old = replay_refinement(mesh, info)
    origin, area = old.origin, old.mesh.cell_area
    return _start_file(
        file,
        mesh,
        lambda v: restrict(v, origin, area, mesh.n_cells),
        out_dir / f"{name}.restricted_start.npz",
        None,
    )


def run_refined(config, console=True, mesh=None):
    """Run a steady case with ``refinement.cycles`` refinements.

    Parameters
    ----------
    config : Resolved
    console : bool
    mesh : Mesh, optional
        The original mesh (default: ``mesh.file``).

    Returns
    -------
    RunResult
        The result of the last cycle run.
    """
    from pyaeroflow.run import _is_state_of, run_case

    cfg = config.data
    rc = cfg["refinement"]
    out_dir = config.resolve_path(cfg["output"]["directory"])
    check_no_stop_file(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = config.name
    if mesh is None:
        mesh = read_mesh(config.resolve_path(cfg["mesh"]["file"]), cfg["mesh"]["scale"])
    options = {
        "walls": list(rc["walls"]),
        "corner_angle": rc["corner_angle"],
        "decay": rc["decay"],
    }
    spec = RefineSpec(**refine_spec_args(rc))
    names = [n for n in names_used(rc["indicator"]) if n in BY_NAME]

    previous, start, ref, restricted = None, 0, None, None
    if cfg["initial"]["restart"] is not None:
        previous = config.resolve_path(cfg["initial"]["restart"])
        info = read_refinement_info(previous)
        if info is not None and own_chain(previous, name):
            ref = replay_refinement(mesh, info)  # continue the chain
            start = int(info["cycle"])
            options = info["options"]
        elif info is not None:  # another case: its state on the original mesh
            restricted = previous
            previous = _restricted_start(previous, info, mesh, out_dir, name)
    if ref is None:
        ref = Refinement.from_mesh(mesh, **options)

    log_path = out_dir / f"{name}.refinement.log"
    csv_path = out_dir / f"{name}.refinement.csv"
    rows = [r for r in _read_rows(csv_path) if r[0] < start]

    def note(text):
        with open(log_path, "a") as file:
            file.write(text + "\n")
        if console:
            print(f"[refinement] {text}", flush=True)

    indicator = " ".join(rc["indicator"].split())  # one line in the log
    note(
        f"case '{name}': {rc['cycles']} refinements, indicator '{indicator}', "
        f"start at cycle {start} with {ref.n_cells} cells"
    )
    if restricted is not None:
        note(
            f"start state from {Path(restricted).name} (a refined mesh of "
            f"another case), averaged onto the original mesh: {previous.name}"
        )
    k = start
    if rc["refine_restart"] and previous is not None and not own_chain(previous, name):
        # A converged state of another case: refine it before any run.
        first = _cycle_config(config, 0, False, previous, "state", None, False)
        source, marking, n0 = _refine_state(first, mesh, ref, previous, names, spec)
        note(f"cycle 0: {marking.summary()} (state of the restart file, not run)")
        rows.append([0, n0, 0, 0, np.nan, np.nan, np.nan, np.nan])
        rows[-1] += [int(marking.cells.size), ref.n_cells]
        _write_rows(csv_path, rows)
        if source is not None:
            k = 1
            previous = _start_file(
                previous,
                ref.mesh,
                lambda v: v[source],
                out_dir / f"{name}.cycle_1_start.npz",
                {"refinement": _info_json(ref, options, 1)},
                drop_frozen_limiter=True,
            )
            note(f"cycle 1: {ref.n_cells} cells, start file {previous.name}")
    result = None
    while True:
        final = k >= rc["cycles"]
        cycle_name = f"{name}_cycle_{k}"
        mode = "continue" if _is_state_of(previous, cycle_name) else "state"
        if previous is None:
            mode = "continue"  # a fresh start: nothing to take over
        extra = {"refinement": _info_json(ref, options, k)} if ref.log else None
        refined = previous is not None and own_chain(previous, name)
        cycle_config = _cycle_config(config, k, final, previous, mode, extra, refined)
        result = run_case(
            cycle_config, console=console, mesh=ref.mesh, restart_mode=mode
        )
        h = result.history
        drop = -np.log10(max(float(h["measure"][-1]), np.finfo(float).tiny))
        c = result.coefficients
        row = [
            k,
            ref.n_cells,
            int(h["iteration"][-1]),
            int(result.converged),
            drop,
            *((c.cl, c.cd, c.cm) if c is not None else (np.nan,) * 3),
            0,
            ref.n_cells,
        ]
        cycle_dir = out_dir / f"cycle_{k}"
        stopped = (cycle_dir / STOP).exists() or (out_dir / STOP).exists()
        if final or stopped:
            rows.append(row)
            _write_rows(csv_path, rows)
            if stopped:
                note(f"cycle {k}: .stop file found, no further refinement")
            break

        fields = indicator_fields(result, cycle_config, names)
        source, marking = apply(ref, fields, spec)
        note(f"cycle {k}: {marking.summary()}")
        row[8] = int(marking.cells.size)
        row[9] = ref.n_cells
        rows.append(row)
        _write_rows(csv_path, rows)
        if source is None:
            note(f"cycle {k}: nothing to refine, the last cycle runs on this mesh")
            k = rc["cycles"]
            previous = result.restart_file
            continue
        k += 1
        previous = _transfer(result, ref, source, out_dir, name, k, options)
        note(
            f"cycle {k}: {ref.n_cells} cells "
            f"({ref.mesh.cell_n_nodes.max()} nodes in the largest polygon), "
            f"start file {previous.name}"
        )
        result = result._replace(problem=None)  # free the old problem
    return result


def _read_rows(path):
    if not path.exists():
        return []
    data = np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1))
    return [list(r) for r in data if r.size]


def _write_rows(path, rows):
    np.savetxt(
        path,
        np.array(rows, dtype=float),
        delimiter=",",
        header=CSV_HEADER,
        comments="",
        fmt=["%d", "%d", "%d", "%d", "%.3f", "%.8e", "%.8e", "%.8e", "%d", "%d"],
    )

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
  Newton-Krylov switch, limiter, vortex; like the next angle of a polar);
* the problem is built anew on the refined mesh (geometry, gradient stencils,
  wall distance, Jacobian pattern, linear-solver analysis).

Every restart and backup file of a refined cycle stores the refinement log
(the splits, the wall options and the cycle number) as JSON. ``pyaeroflow
run`` with such a file as ``initial.restart`` replays the splits on the
original mesh file, so no new mesh format is needed; with ``[refinement]``
the chain continues from that cycle.

``<output.directory>/<name>.refinement.csv`` lists the cycles (cells,
iterations, residual drop, coefficients, marked cells) and
``<name>.refinement.log`` the marking of each refinement.
"""

import copy
import json
from pathlib import Path

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


def _cycle_config(base, k, final, previous, mode, extra):
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
    if mode == "state":
        # The polar option decides whether the next run restarts the CFL ramp.
        data["solver"]["cfl"]["polar_reset"] = rc["cfl_reset"]
    config = Resolved(data, set(base.explicit), base.path, list(base.notices))
    config.refinement_cycle = True
    config.restart_extra = extra
    if base.reload is not None:
        config.reload = lambda: _cycle_config(
            base.reload(), k, final, previous, mode, extra
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


def _transfer(result, ref, source, out_dir, name, k, options):
    """Write the injected start file of cycle ``k`` on the refined mesh."""
    from pyaeroflow.run import load_restart_state, save_restart

    with np.load(result.restart_file) as data:
        u = np.array(data["w"])
        step = int(data["step"]) if "step" in data.files else 0
        flow = str(data["flow"]) if "flow" in data.files else "incompressible"
    n_old = u.shape[0]
    state = load_restart_state(result.restart_file) or {}
    state.pop("iteration", None)
    state.pop("file", None)
    for key, value in list(state.items()):
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == n_old:
            state[key] = value[source]
    path = out_dir / f"{name}.cycle_{k}_start.npz"
    save_restart(
        path,
        ref.mesh,
        u[source],
        step,
        None,
        state or None,
        flow,
        {"refinement": _info_json(ref, options, k)},
    )
    return path


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

    previous, start = None, 0
    if cfg["initial"]["restart"] is not None:
        previous = config.resolve_path(cfg["initial"]["restart"])
        info = read_refinement_info(previous)
        if info is not None:
            ref = replay_refinement(mesh, info)
            start = int(info["cycle"])
            options = info["options"]
        else:
            ref = Refinement.from_mesh(mesh, **options)
    else:
        ref = Refinement.from_mesh(mesh, **options)

    log_path = out_dir / f"{name}.refinement.log"
    csv_path = out_dir / f"{name}.refinement.csv"
    rows = [r for r in _read_rows(csv_path) if r[0] < start]

    def note(text):
        with open(log_path, "a") as file:
            file.write(text + "\n")
        if console:
            print(f"[refinement] {text}", flush=True)

    note(
        f"case '{name}': {rc['cycles']} refinements, indicator "
        f"'{rc['indicator']}', start at cycle {start} with {ref.n_cells} cells"
    )
    result = None
    k = start
    while True:
        final = k >= rc["cycles"]
        cycle_name = f"{name}_cycle_{k}"
        mode = "continue" if _is_state_of(previous, cycle_name) else "state"
        if previous is None:
            mode = "continue"  # a fresh start: nothing to take over
        extra = {"refinement": _info_json(ref, options, k)} if ref.log else None
        cycle_config = _cycle_config(config, k, final, previous, mode, extra)
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

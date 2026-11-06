"""The ``pyaeroflow plot`` subcommand."""

import sys
from pathlib import Path

KINDS = ("log", "wall", "polar", "probes", "refs")

HELP = """kinds (detected from the file name when left out):
  log     <name>.log or <name>.history.csv: residuals, solver, coeffs, diffs
  wall    <name>.wall_<boundary>.csv (several: overlay): Cp, Cf, y+
  polar   <name>.polar.csv (several: overlay): CL-alpha, CL-CD, cm, L/D
  probes  <name>.probes.csv: time series, --spectrum for frequencies and St
  refs    list the built-in reference data (use as --ref builtin:<name>)

reference data: --ref FILE[:zone][@label] or --ref builtin:NAME[:zone][@label],
  e.g. --ref builtin:tmr_naca0012_cp_cfl3d_sa:alpha=10@CFL3D
"""


def add_parser(sub):
    """Register ``plot`` on the subparsers of the main command."""
    import argparse

    p = sub.add_parser(
        "plot",
        help="plot logs, wall Cp/Cf, polars and probes (needs matplotlib)",
        epilog=HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("items", nargs="+", metavar="[kind] file", help="kind and files")
    p.add_argument(
        "-o", "--output", help="output file prefix (default: the input stem)"
    )
    p.add_argument("--label", action="append", default=[], help="legend label per file")
    p.add_argument("--ref", action="append", default=[], help="reference data (repeat)")
    p.add_argument("--title", help="figure title (default: the file names)")
    p.add_argument(
        "--panels",
        help="comma list; log: residuals,solver,coeffs,diffs; "
        "wall: cp,cf,y_plus,t,q_w; polar: cl,drag,cm,ld,cd; probes: p,u,v,t",
    )
    g = p.add_argument_group("log")
    g.add_argument("-t", "--targets", help="TOML with [targets] and [scale.KEY]")
    g.add_argument("--no-notes", action="store_true", help="no note lines and labels")
    g.add_argument("--last", type=int, help="only the last N rows")
    g.add_argument("--to", type=float, dest="x_to", help="end of the range")
    g.add_argument(
        "--average-from", type=float, help="unsteady: running mean of CL, CD, CM from t"
    )
    g = p.add_argument_group("log and probes")
    g.add_argument(
        "--from", type=float, dest="x_from",
        help="start of the range (iteration, or time in s); probes: spectrum window",
    )  # fmt: skip
    g = p.add_argument_group("wall")
    g.add_argument(
        "--x", choices=("chord", "arc"), default="chord",
        help="x/c with upper/lower surfaces (default) or arc length s/c",
    )  # fmt: skip
    g.add_argument("--surface", choices=("both", "upper", "lower"), default="both")
    g.add_argument(
        "--cf-sign", choices=("flow", "raw"), default="flow",
        help="flow: attached flow positive on both sides (default); raw: as in the CSV",
    )  # fmt: skip
    g.add_argument("--x-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    g = p.add_argument_group("polar")
    g.add_argument(
        "--cm-sign", choices=("nose-up", "solver"), default="nose-up",
        help="nose-up positive cm = -CM (default) or the solver's CM",
    )  # fmt: skip
    g.add_argument(
        "--split-cd", action="store_true", help="also pressure and viscous CD"
    )
    g = p.add_argument_group("probes")
    g.add_argument(
        "--spectrum", action="append", default=[],
        help="column (probe1_v), quantity (v) or 'all'; repeat for more",
    )  # fmt: skip
    g.add_argument("--length", type=float, help="St reference length (default: case)")
    g.add_argument(
        "--velocity", type=float, help="St reference velocity (default: case)"
    )
    return p


def detect(path):
    """Plot kind of a file from its name."""
    name = Path(path).name
    if name.endswith((".log", ".history.csv")):
        return "log"
    if ".wall_" in name:
        return "wall"
    if name.endswith(".polar.csv"):
        return "polar"
    if name.endswith(".probes.csv"):
        return "probes"
    raise ValueError(
        f"cannot tell what {name} is; give the kind: "
        f"pyaeroflow plot {{log,wall,polar,probes}} {name}"
    )


def run(args):
    """Execute ``pyaeroflow plot``; returns the exit code."""
    items = list(args.items)
    kind = items.pop(0) if items[0] in KINDS else detect(items[0])
    panels = [s.strip() for s in args.panels.split(",")] if args.panels else None
    if kind == "refs":
        from pyaeroflow.plotting.readers import builtin_names, read_reference

        for name in builtin_names():
            zones = read_reference(f"builtin:{name}")
            cols = ", ".join(zones[0].columns)
            titles = "; ".join(z.title for z in zones if z.title)
            print(f"{name:<30} [{cols}]" + (f"  zones: {titles}" if titles else ""))
        return 0
    if not items:
        print("error: no input file", file=sys.stderr)
        return 1
    for item in items:
        if not Path(item).exists():
            print(f"error: {item} not found", file=sys.stderr)
            return 1
    if kind == "log":
        from pyaeroflow.plotting.log import PANELS, plot_log

        for item in items:
            plot_log(
                item,
                prefix=args.output if len(items) == 1 else None,
                targets=args.targets,
                panels=panels or PANELS,
                show_notes=not args.no_notes,
                x_from=args.x_from,
                x_to=args.x_to,
                last=args.last,
                average_from=args.average_from,
            )
    elif kind == "wall":
        from pyaeroflow.plotting.wall import plot_wall

        plot_wall(
            items, args.output, args.label, args.ref, args.x, panels or ("cp", "cf"),
            args.surface, args.cf_sign, args.x_range, args.title,
        )  # fmt: skip
    elif kind == "polar":
        from pyaeroflow.plotting.polar import plot_polar

        plot_polar(
            items, args.output, args.label, args.ref,
            panels or ("cl", "drag", "cm", "ld"), args.cm_sign, args.split_cd,
            args.title,
        )  # fmt: skip
    else:
        from pyaeroflow.plotting.probes import plot_probes

        for item in items:
            plot_probes(
                item, args.output if len(items) == 1 else None, panels, args.spectrum,
                args.x_from, args.length, args.velocity, args.title,
            )  # fmt: skip
    return 0

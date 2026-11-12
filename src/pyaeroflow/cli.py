"""Command line: run, check, polar, refine, fields, init-case, check-mesh, foil, ..."""

import argparse
import sys

from pyaeroflow import __version__
from pyaeroflow.config import ConfigError, load_case
from pyaeroflow.mesh import MeshError


def main(argv=None):
    """Entry point of the ``pyaeroflow`` command."""
    parser = argparse.ArgumentParser(
        prog="pyaeroflow",
        description="2D unstructured finite-volume Navier-Stokes solver",
    )
    parser.add_argument(
        "--version", action="version", version=f"pyaeroflow {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a case")
    run.add_argument("case", help="TOML case file")
    run.add_argument("--quiet", action="store_true", help="no console output")
    check = sub.add_parser("check", help="validate a case and print the resolved file")
    check.add_argument("case", help="TOML case file")
    polar = sub.add_parser(
        "polar", help="run a case at several angles of attack (restart chain)"
    )
    polar.add_argument("case", help="TOML case file with farfield boundaries")
    polar.add_argument("--alpha", nargs="+", type=float, required=True, help="deg")
    polar.add_argument("--quiet", action="store_true", help="no console output")
    polar.add_argument(
        "--predictor",
        choices=("none", "tangent"),
        default="none",
        help="tangent: start each angle from w + d_alpha dw/dalpha and write "
        "the slopes dCL/dalpha ... to <name>.sensitivity.csv",
    )
    init = sub.add_parser(
        "init-case",
        help="write a case file for an airfoil mesh (.su2 from PyAero, or .msh); "
        "same as 'init airfoil'",
    )
    init.add_argument("mesh", help="mesh file (.su2 or .msh)")
    _airfoil_arguments(init)
    init.add_argument("--name", help="case name (default <mesh>_<model>)")
    init.add_argument("-o", "--output", help="case file (default next to the mesh)")
    init.add_argument("--force", action="store_true", help="overwrite the case file")
    tpl = sub.add_parser("templates", help="list the case templates of 'init'")
    tpl.add_argument("group", nargs="?", help="only this group (channel, plate, ...)")
    ini = sub.add_parser(
        "init",
        help="write a ready-to-run case from a template (case file and mesh); "
        "'pyaeroflow templates' lists them",
    )
    ini.add_argument("template", help="template name, e.g. plate-sa or airfoil")
    ini.add_argument(
        "-o",
        "--directory",
        help="folder for the case and mesh (default: current folder; airfoil: "
        "the folder of the mesh)",
    )
    ini.add_argument("--name", help="case name (file name and output folder)")
    ini.add_argument("--force", action="store_true", help="overwrite existing files")
    ini.add_argument("--mesh", help="airfoil template: the airfoil mesh (.su2, .msh)")
    _airfoil_arguments(ini)
    cmesh = sub.add_parser(
        "check-mesh", help="check an airfoil mesh: y+, surface points, farfield"
    )
    cmesh.add_argument("mesh", help="mesh file (.su2 or .msh)")
    cmesh.add_argument("--re", type=float, help="chord Reynolds number (for y+)")
    cmesh.add_argument("--wall", help="airfoil boundary (default 'airfoil')")
    cmesh.add_argument("--chord", type=float, help="m; default from the mesh")
    cmesh.add_argument("--scale", type=float, default=1.0, help="mesh.scale")
    rfn = sub.add_parser(
        "refine",
        help="apply [refinement.band] to the case mesh and write it as VTU "
        "(no run), to check it in ParaView",
    )
    rfn.add_argument("case", help="TOML case file with [refinement.band]")
    rfn.add_argument(
        "-o", "--output", help="VTU file (default <output.directory>/<name>.band.vtu)"
    )
    flds = sub.add_parser(
        "fields", help="list the output fields a case provides (* = selected)"
    )
    flds.add_argument("case", help="TOML case file")
    info = sub.add_parser(
        "restart-info",
        help="iteration or step, time and solver state of a restart or backup file",
    )
    info.add_argument("file", help="<name>.backup_<n>.npz or <name>.restart_a.npz")
    info.add_argument("--json", action="store_true", help="machine-readable output")
    clean = sub.add_parser(
        "clean",
        help="list (default) or delete the files a case wrote; nothing is deleted "
        "without --yes",
    )
    clean.add_argument("case", help="TOML case file")
    clean.add_argument(
        "--restarts", action="store_true", help="rolling <name>.restart_a/b.npz"
    )
    clean.add_argument(
        "--backups",
        action="store_true",
        help="<name>.backup_<n>.npz, including the final state of each run",
    )
    clean.add_argument(
        "--fields", action="store_true", help="VTU files, series and .pvd"
    )
    clean.add_argument(
        "--history", action="store_true", help="history CSV and cut copies"
    )
    clean.add_argument("--wall", action="store_true", help="wall CSV files")
    clean.add_argument("--log", action="store_true", help="the .log file")
    clean.add_argument(
        "--all",
        action="store_true",
        help="everything above plus resolved configuration and polar CSV",
    )
    clean.add_argument("--yes", action="store_true", help="delete (default: only list)")
    from pyaeroflow.plotting.cli import add_parser as add_plot

    add_plot(sub)
    from pyaeroflow.ibl.cli import add_parser as add_foil

    add_foil(sub)
    mm = sub.add_parser(
        "make-mesh",
        help="write a mesh of the library (no input file needed); --list shows them",
    )
    mm.add_argument("name", nargs="?", help="library mesh, e.g. jet-crossflow")
    mm.add_argument("-o", "--output", help="SU2 file (default <name>.su2)")
    mm.add_argument(
        "--set",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="builder parameters, e.g. nx_plate=200 reynolds=5e6",
    )
    mm.add_argument("--list", action="store_true", help="list the library meshes")
    args = parser.parse_args(argv)

    try:
        if args.command == "make-mesh":
            return _make_mesh(args)
        if args.command == "check":
            config = load_case(args.case)
            print(config.to_toml())
            for note in config.notices:
                print(f"# notice: {note}")
            return 0
        if args.command == "fields":
            from pyaeroflow.io.fields import case_listing

            print(case_listing(load_case(args.case).data))
            return 0
        if args.command == "refine":
            from pyaeroflow.refinement import refine_case_mesh

            refine_case_mesh(load_case(args.case), args.output)
            return 0
        if args.command == "init-case":
            return _init_case(args)
        if args.command == "templates":
            return _list_templates(args)
        if args.command == "init":
            return _init(args)
        if args.command == "check-mesh":
            return _check_mesh(args)
        if args.command == "restart-info":
            from pyaeroflow.housekeeping import restart_info

            info = restart_info(args.file)
            if args.json:
                import json

                print(json.dumps(info, indent=2))
            else:
                for key, value in info.items():
                    print(f"{key:<12} {value}")
            return 0
        if args.command == "clean":
            return _clean(args)
        if args.command == "foil":
            from pyaeroflow.ibl.cli import run as run_foil

            return run_foil(args)
        if args.command == "plot":
            from pyaeroflow.plotting.cli import run as run_plot

            return run_plot(args)
        if args.command == "polar":
            from pyaeroflow.run import run_polar

            results = run_polar(
                args.case, args.alpha, console=not args.quiet, predictor=args.predictor
            )
            return 0 if all(r.converged for r in results) else 2
        from pyaeroflow.run import run_case

        result = run_case(args.case, console=not args.quiet)
        return 0 if result.converged else 2
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (OSError, ValueError, MeshError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _models():
    from pyaeroflow.initcase import MODELS

    return MODELS


def _init_case(args):
    """Write and validate a case file for an airfoil mesh."""
    from pathlib import Path

    from pyaeroflow.initcase import make_case

    mesh = Path(args.mesh)
    model = args.model
    name = args.name or f"{mesh.stem}_{model}"
    path = Path(args.output) if args.output else mesh.parent / f"{name}.toml"
    if path.exists() and not args.force:
        print(f"error: {path} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    text, notes = make_case(
        mesh,
        path,
        model,
        reynolds=args.re,
        alpha=args.alpha,
        chord=args.chord,
        wall=args.wall,
        turbulence_intensity=args.tu,
        scale=args.scale,
        name=name,
        band=DEFAULT_BAND if args.band == [] else args.band,
        boundary_layer_start=args.boundary_layer_start,
    )
    path.write_text(text)
    print(f"wrote {path}  ({_models()[model]})")
    for note in notes:
        print(f"  {note}")
    config = load_case(path)  # validates the file just written
    for note in config.notices:
        print(f"  notice: {note}")
    print("next:")
    if args.band:
        print(f"  pyaeroflow refine {path}      # band mesh as VTU, to check it")
    print(f"  pyaeroflow run {path}")
    print(f"  pyaeroflow polar {path} --alpha 0 2 4 6 8")
    return 0


DEFAULT_BAND = [0.05, 0.02, 0.008]  # chords: the validated NACA 0012 band case


def _airfoil_arguments(p):
    """Options of the airfoil case file (init-case and init airfoil)."""
    p.add_argument(
        "--model",
        default="sa",
        choices=list(_models()),
        help="sa (default), sst, bcm, aft, lm, laminar or euler",
    )
    p.add_argument("--re", type=float, help="chord Reynolds number")
    p.add_argument("--alpha", type=float, default=0.0, help="deg (default 0)")
    p.add_argument("--chord", type=float, help="m; default from the mesh")
    p.add_argument("--wall", help="airfoil boundary (default 'airfoil')")
    p.add_argument(
        "--tu", type=float, default=0.001, help="freestream Tu, fraction (0.001)"
    )
    p.add_argument("--scale", type=float, default=1.0, help="mesh.scale")
    p.add_argument(
        "--band",
        nargs="*",
        type=float,
        metavar="D",
        help="refinement band around the wall, distances in chords "
        "(no values: 0.05 0.02 0.008)",
    )
    p.add_argument(
        "--boundary-layer-start",
        action="store_true",
        help="start from the integral boundary-layer solution (initial.boundary_layer)",
    )


def _list_templates(args):
    """``pyaeroflow templates``: the templates of ``init``, grouped."""
    from pyaeroflow.templates import TEMPLATES, description

    groups = {}
    for name, t in TEMPLATES.items():
        groups.setdefault(t.group, []).append(name)
    if args.group and args.group not in groups:
        print(f"error: no group '{args.group}' ({', '.join(groups)})", file=sys.stderr)
        return 1
    width = max(len(n) for n in TEMPLATES)
    for group, names in groups.items():
        if args.group and group != args.group:
            continue
        print(f"{group}:")
        for name in names:
            print(f"  {name:<{width}}  {description(name)}")
    print("write one: pyaeroflow init NAME [-o FOLDER]")
    return 0


def _init(args):
    """``pyaeroflow init``: case file and mesh from a template."""
    from pathlib import Path

    from pyaeroflow.templates import TEMPLATES, write_template

    if args.template not in TEMPLATES:
        names = ", ".join(sorted(TEMPLATES))
        print(f"error: unknown template '{args.template}' ({names})", file=sys.stderr)
        return 1
    if args.template == "airfoil":
        if not args.mesh:
            print("error: the airfoil template needs --mesh FILE", file=sys.stderr)
            return 1
        mesh = Path(args.mesh)
        name = args.name or f"{mesh.stem}_{args.model}"
        folder = Path(args.directory) if args.directory else mesh.parent
        folder.mkdir(parents=True, exist_ok=True)
        args.output = str(folder / f"{name}.toml")
        args.name = name
        return _init_case(args)
    airfoil_only = [
        flag
        for flag, value in (
            ("--mesh", args.mesh),
            ("--re", args.re),
            ("--band", args.band is not None),
            ("--boundary-layer-start", args.boundary_layer_start),
        )
        if value
    ]
    if airfoil_only:
        print(
            f"error: {', '.join(airfoil_only)} only with the airfoil template; "
            "edit the case file of this template instead",
            file=sys.stderr,
        )
        return 1
    try:
        case, mesh, written = write_template(
            args.template, args.directory or ".", case=args.name, force=args.force
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {case}")
    print(f"{'wrote' if written else 'kept '} {mesh}")
    config = load_case(case)  # validates the file just written
    for note in config.notices:
        print(f"  notice: {note}")
    print("next:")
    print(f"  pyaeroflow run {case}")
    return 0


def _check_mesh(args):
    """Print the mesh check; invalid meshes fail with the reader's message."""
    from pyaeroflow.mesh import read_mesh
    from pyaeroflow.meshcheck import check_mesh

    mesh = read_mesh(args.mesh, args.scale)
    report = check_mesh(mesh, wall=args.wall, reynolds=args.re, chord=args.chord)
    print(report.format())
    return 0


def _clean(args):
    """List or delete the files of a case (D60)."""
    from pyaeroflow.housekeeping import CATEGORIES, clean_case

    config = load_case(args.case)
    out_dir = config.resolve_path(config.data["output"]["directory"])
    chosen = [k for k in CATEGORIES if args.all or getattr(args, k)]
    if args.all:
        chosen.append("other")
    inventory, removed = clean_case(out_dir, config.name, chosen, delete=args.yes)
    print(f"{out_dir}  (case {config.name})")
    total = 0
    for key, files in inventory.items():
        size = 0 if args.yes else sum(f.stat().st_size for f in files)
        mark = "*" if key in chosen else " "
        print(f" {mark} {key:<9} {len(files):4d} files  {_size(size)}")
        if key in chosen:
            total += size
    if not chosen:
        print("nothing selected: add --restarts, --backups, --fields, --history,")
        print("--wall, --log or --all (and --yes to delete)")
    elif args.yes:
        print(f"deleted {len(removed)} files")
    else:
        for key in chosen:
            for path in inventory[key]:
                print(f"   {path.relative_to(out_dir)}")
        print(f"dry run: {_size(total)} would be deleted; add --yes to delete")
    return 0


def _size(n):
    return f"{n / 1e6:9.1f} MB" if n >= 1e5 else f"{n / 1e3:9.1f} kB"


if __name__ == "__main__":
    sys.exit(main())


def _make_mesh(args):
    """``pyaeroflow make-mesh``: write a library mesh to an SU2 file."""
    from pyaeroflow.mesh import write_su2
    from pyaeroflow.mesh.library import LIBRARY, build

    if args.list or not args.name:
        width = max(len(n) for n in LIBRARY)
        for name, (_, text) in sorted(LIBRARY.items()):
            print(f"{name:<{width}}  {text}")
        return 0
    parameters = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
        for kind in (int, float, str):
            try:
                parameters[key] = kind(value)
                break
            except ValueError:
                continue
    mesh = build(args.name, **parameters)
    out = args.output or f"{args.name}.su2"
    write_su2(out, mesh)
    print(f"{out}: {mesh.n_cells} cells, boundaries {sorted(mesh.boundary_faces)}")
    return 0

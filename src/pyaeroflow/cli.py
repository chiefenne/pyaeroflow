"""Command line: ``run``, ``check``, ``polar``, ``restart-info`` and ``clean``."""

import argparse
import sys

from pyaeroflow import __version__
from pyaeroflow.config import ConfigError, load_case


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
        "--backup",
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
    args = parser.parse_args(argv)

    try:
        if args.command == "check":
            config = load_case(args.case)
            print(config.to_toml())
            for note in config.notices:
                print(f"# notice: {note}")
            return 0
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
        if args.command == "polar":
            from pyaeroflow.run import run_polar

            results = run_polar(args.case, args.alpha, console=not args.quiet)
            return 0 if all(r.converged for r in results) else 2
        from pyaeroflow.run import run_case

        result = run_case(args.case, console=not args.quiet)
        return 0 if result.converged else 2
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


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
        print("nothing selected: add --restarts, --backup, --fields, --history,")
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

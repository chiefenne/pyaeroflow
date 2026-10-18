"""Command line: ``pyaeroflow run``, ``pyaeroflow check`` and ``pyaeroflow polar``."""

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
    args = parser.parse_args(argv)

    try:
        if args.command == "check":
            config = load_case(args.case)
            print(config.to_toml())
            for note in config.notices:
                print(f"# notice: {note}")
            return 0
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


if __name__ == "__main__":
    sys.exit(main())

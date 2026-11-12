"""``pyaeroflow foil``: viscous panel analysis of an airfoil (polar to CSV)."""

import sys
from pathlib import Path


def add_parser(sub):
    """Add the ``foil`` subcommand to the argument parser."""
    p = sub.add_parser(
        "foil",
        help="viscous panel analysis of an airfoil (panels + coupled boundary layer)",
    )
    p.add_argument(
        "airfoil", help="coordinate file (Selig or Lednicer) or naca<4 digits>"
    )
    p.add_argument("--alpha", nargs="+", type=float, required=True, help="deg")
    p.add_argument("--re", type=float, required=True, help="chord Reynolds number")
    p.add_argument(
        "--ncrit", type=float, default=9.0, help="critical amplification (9)"
    )
    p.add_argument(
        "--panels", type=int, default=200, help="number of panel nodes (200)"
    )
    p.add_argument("--output", help="polar CSV (default <airfoil>_Re<re>.polar.csv)")
    p.add_argument(
        "--cp", action="store_true", help="also write <output>_a<alpha>.wall.csv"
    )
    return p


def run(args):
    """Run the ``foil`` subcommand: a polar and optional wall distributions."""
    import numpy as np

    from . import analyse
    from .geometry import naca4, read_coordinates, repanel

    name = args.airfoil
    if name.lower().startswith("naca") and len(name) == 8 and name[4:].isdigit():
        xy = naca4(name[4:], 201)
        stem = name.lower()
    else:
        xy = read_coordinates(name)
        stem = Path(name).stem
    nodes = repanel(xy, args.panels)
    out = (
        Path(args.output) if args.output else Path(f"{stem}_Re{args.re:.3g}.polar.csv")
    )
    rows = []
    print(
        f"{'alpha':>7} {'CL':>9} {'CD':>9} {'CDf':>9} {'CM':>9} "
        f"{'xtr_up':>7} {'xtr_lo':>7}  conv"
    )
    ok = True
    for alpha in args.alpha:
        r = analyse(nodes, alpha, args.re, ncrit=args.ncrit)
        ok &= r.converged
        xu, xl = r.x_transition
        rows.append(
            (alpha, r.cl, r.cd, r.cdf, r.cm, xu, xl, int(r.converged), r.iterations)
        )
        print(
            f"{alpha:7.2f} {r.cl:9.5f} {r.cd:9.6f} {r.cdf:9.6f} {r.cm:9.5f} "
            f"{xu:7.4f} {xl:7.4f}  {'yes' if r.converged else 'NO'} ({r.iterations})",
            flush=True,
        )
        if args.cp:
            n = r.nodes.shape[0]
            wall = out.with_name(
                f"{out.stem.replace('.polar', '')}_a{alpha:g}.wall.csv"
            )
            np.savetxt(
                wall,
                np.column_stack(
                    (r.nodes, r.cp, r.cf[:n], r.theta[:n], r.delta_star[:n], r.ue[:n])
                ),
                delimiter=",",
                header="x,y,cp,cf,theta,delta_star,ue",
                comments="",
            )
    header = "alpha,cl,cd,cdf,cm,xtr_upper,xtr_lower,converged,iterations"
    np.savetxt(
        out, np.array(rows), delimiter=",", header=header, comments="", fmt="%.8g"
    )
    print(f"polar written to {out}", file=sys.stderr)
    return 0 if ok else 2

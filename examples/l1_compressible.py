"""L1 (C1) example: compressible Euler verification and validation.

Run from the repository root (about 15 minutes with PARDISO on 4 cores for
the coarse level; ``--fine`` adds the 384 x 128 O-grids):

    python examples/l1_compressible.py [--fine]

Cases:

- Mach 2 flow over a 10 degree compression ramp against the exact
  oblique-shock solution (``pyaeroflow.cases.oblique_shock``).
- NACA 0012 (TMR definition, sharp trailing edge), alpha = 1.25 deg, at
  M 0.5 (shock-free: the drag is discretization error) and M 0.8 (shocks on
  both sides; AGARD-AR-211, Vassberg & Jameson 2010), on O-grids with the
  farfield at 100 chords.

Output in ``examples/output/l1/``: the run directories of the NACA cases
and ``l1_naca0012_cp.png`` (wall Cp at M 0.5 and 0.8).
"""

import argparse
import importlib.util
import tomllib
from pathlib import Path

import numpy as np

from pyaeroflow.cases import oblique_shock
from pyaeroflow.config import resolve
from pyaeroflow.mesh.airfoil import airfoil_ogrid, airfoil_points
from pyaeroflow.run import run_case

OUT = Path(__file__).parent / "output" / "l1"
# PARDISO is much faster than SuperLU at this size; fall back if not installed
LINEAR = "pardiso" if importlib.util.find_spec("pypardiso") else "direct"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

CASE = """
[case]
name = "{name}"
[physics]
flow = "compressible"
model = "euler"
[mesh]
file = "generated.su2"
[numerics]
limiter = "venkatakrishnan-wang"
[solver]
max_iterations = 200
residual_drop = 10
linear_solver = {{ type = "{linear}" }}
[output]
directory = "{directory}"
[boundary.airfoil]
type = "wall"
[boundary.farfield]
type = "farfield"
mach = {mach}
angle_of_attack = 1.25
"""


def naca(mach, n_surface, n_radial):
    """Run the NACA 0012 Euler case; return (name, cells, coefficients, out dir)."""
    name = f"naca0012_m{round(10 * mach):02d}_{n_surface}"
    mesh = airfoil_ogrid(
        airfoil_points(n_surface, cut=1.0),
        n_radial,
        first_spacing=1.0e-3,
        farfield_radius=100.0,
        max_growth=1.3,
    )
    raw = tomllib.loads(
        CASE.format(name=name, directory=OUT / name, mach=mach, linear=LINEAR)
    )
    result = run_case(resolve(raw, OUT / f"{name}.toml"), console=False, mesh=mesh)
    return name, mesh.n_cells, result.coefficients, result.output_dir


def plot(runs, path):
    """Wall Cp of the NACA runs (-Cp up)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, mach in zip(axes, (0.5, 0.8), strict=True):
        k = 0
        for name, cells, _, out_dir in runs:
            if f"_m{round(10 * mach):02d}_" not in name:
                continue
            data = np.genfromtxt(
                Path(out_dir) / f"{name}.wall_airfoil.csv", delimiter=",", names=True
            )
            ax.plot(
                data["x"], -data["cp"], color=COLORS[k], lw=1.2, label=f"{cells} cells"
            )
            k += 1
        ax.set_title(f"NACA 0012, M = {mach}, alpha = 1.25 deg (Euler)")
        ax.set_xlabel("x / c")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("-Cp")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    """Run the cases and print the comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine", action="store_true", help="add 384 x 128 grids")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ramp = oblique_shock.run()
    print(
        f"ramp M 2, 10 deg: p2/p1 {ramp.pressure_ratio:.4f} (exact "
        f"{ramp.exact.pressure_ratio:.4f}), M2 {ramp.mach:.4f} (exact "
        f"{ramp.exact.mach:.4f}), converged {ramp.converged}"
    )
    levels = [(192, 64)] + ([(384, 128)] if args.fine else [])
    runs = [naca(m, ns, nr) for m in (0.5, 0.8) for ns, nr in levels]
    for name, cells, c, _ in runs:
        print(f"{name}: {cells} cells, CL {c.cl:.5f}, CD {c.cd:.6f}, CM {c.cm:.5f}")
    plot(runs, OUT / "l1_naca0012_cp.png")


if __name__ == "__main__":
    main()

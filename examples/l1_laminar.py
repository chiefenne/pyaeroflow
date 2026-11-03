"""L1 (C2) example: compressible laminar Navier-Stokes verification and validation.

Run from the repository root (about 4 minutes with PARDISO):

    python examples/l1_laminar.py

Cases:

- Laminar flat plate at M 0.5, Re_L 1e5 (160 x 80 cells), adiabatic and
  isothermal (T_w = T_inf), against the compressible similarity solution
  (``pyaeroflow.cases.flat_plate_compressible``).
- NACA 0012 at M 0.5, alpha 0, Re 5000 (``examples/cases/
  naca0012_laminar_m05.toml``, PyAero C-mesh 100k cells) against Swanson &
  Langer (NASA/TM-2016-219003): CD 0.05557, separation at x/c 0.808.

Output in ``examples/output/l1_laminar/``: the NACA run directory and
``l1_laminar.png`` (plate Cf and heat flux, airfoil Cp and Cf).
"""

import importlib.util
import tomllib
from pathlib import Path

import numpy as np

from pyaeroflow.cases.flat_plate_compressible import solve
from pyaeroflow.config import resolve
from pyaeroflow.run import run_case
from pyaeroflow.solver import LinearSettings, PseudoTimeSettings

HERE = Path(__file__).parent
OUT = HERE / "output" / "l1_laminar"
LINEAR = "pardiso" if importlib.util.find_spec("pypardiso") else "direct"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
REYNOLDS = 1.0e5
SEPARATION_REF = 0.808  # Swanson & Langer 2016, finest grids
CD_REF = (0.05557, 0.02279, 0.03279)  # total, pressure, friction


def plates():
    """Adiabatic and isothermal plates; print the comparison."""
    settings = PseudoTimeSettings(
        residual_drop=8.0, max_iterations=200, linear=LinearSettings(type=LINEAR)
    )
    out = {}
    for label, tw in (("adiabatic", None), ("isothermal T_w = T_inf", 1.0)):
        r = solve(wall_temperature=tw, settings=settings)
        sel = (r.x > 0.2) & (r.x < 0.95)
        root = np.sqrt(REYNOLDS * r.x)
        cf = np.mean(r.cf[sel] * root[sel])
        t_w = np.mean(r.wall_temperature[sel])
        q = np.mean(r.heat_flux[sel] * root[sel])
        ref = r.reference
        print(
            f"plate {label}: converged {r.converged}, Cf sqrt(Re_x) {cf:.4f} "
            f"(similarity {ref.cf_sqrt_re:.4f}), T_w {t_w:.5f} "
            f"({ref.wall_temperature:.5f}), q sqrt(Re_x) {q:.5f} "
            f"({ref.heat_flux_sqrt_re:.5f})"
        )
        out[label] = r
    return out


def airfoil():
    """NACA 0012 case file, output redirected; returns the wall CSV path."""
    path = HERE / "cases" / "naca0012_laminar_m05.toml"
    raw = tomllib.loads(path.read_text())
    raw["solver"]["linear_solver"] = {"type": LINEAR}
    raw["output"]["directory"] = str(OUT / "naca0012_laminar_m05")
    result = run_case(resolve(raw, path), console=False)
    c = result.coefficients
    print(
        f"NACA 0012 M 0.5 Re 5000: converged {result.converged}, CD {c.cd:.5f} "
        f"(ref {CD_REF[0]}), pressure {c.cd_pressure:.5f} ({CD_REF[1]}), "
        f"friction {c.cd_viscous:.5f} ({CD_REF[2]})"
    )
    return result.output_dir / "naca0012_laminar_m05.wall_airfoil.csv"


def upper_surface(wall_csv):
    """x, Cp, Cf (positive for attached flow) of the upper surface, by x."""
    d = np.genfromtxt(wall_csv, delimiter=",", names=True)
    up = d["y"] > 0.0
    order = np.argsort(d["x"][up])
    return d["x"][up][order], d["cp"][up][order], d["cf"][up][order]


def separation(x, cf, x_min=0.3):
    """First zero crossing of Cf downstream of x_min (linear interpolation)."""
    k = np.flatnonzero((x[:-1] > x_min) & (cf[:-1] > 0.0) & (cf[1:] <= 0.0))
    if not k.size:
        return None
    i = k[0]
    return float(x[i] - cf[i] * (x[i + 1] - x[i]) / (cf[i + 1] - cf[i]))


def plot(plate, wall_csv, path):
    """Figure: plate Cf sqrt(Re_x) and heat flux, airfoil Cp and Cf."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    ax = axes[0]
    for (label, r), color in zip(plate.items(), COLORS, strict=False):
        root = np.sqrt(REYNOLDS * r.x)
        ax.plot(r.x, r.cf * root, color=color, lw=2, label=f"PyAeroFlow, {label}")
        ax.axhline(r.reference.cf_sqrt_re, color=color, lw=1, ls="--")
    ax.set(
        xlabel="x / L",
        ylabel=r"$C_f \sqrt{Re_x}$",
        title="Flat plate, M 0.5, Re$_L$ 10$^5$ (dashed: similarity)",
        xlim=(0, 1),
        ylim=(0.6, 0.7),
    )
    ax.legend(frameon=False, fontsize=9)

    x, cp, cf = upper_surface(wall_csv)
    ax = axes[1]
    ax.plot(x, cp, color=COLORS[0], lw=2)
    ax.invert_yaxis()
    ax.set(xlabel="x / c", ylabel="$C_p$", title="NACA 0012, M 0.5, α 0, Re 5000")
    ax = axes[2]
    ax.plot(x, cf, color=COLORS[0], lw=2, label="PyAeroFlow, 100k C-mesh")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(
        SEPARATION_REF,
        color=COLORS[1],
        lw=1,
        ls="--",
        label=f"separation, Swanson & Langer ({SEPARATION_REF})",
    )
    x_sep = separation(x, cf)
    if x_sep is not None:
        ax.plot([x_sep], [0.0], "o", color=COLORS[0], ms=8)
        ax.annotate(
            f"x/c = {x_sep:.3f}", (x_sep, 0.0), (x_sep - 0.35, 0.02), fontsize=9
        )
    ax.set(
        xlabel="x / c",
        ylabel="$C_f$",
        title="Upper-surface skin friction",
        ylim=(-0.01, 0.06),
    )
    ax.legend(frameon=False, fontsize=9)
    for a in axes:
        a.grid(color="0.9", lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)
    print(f"figure: {path}")


def main():
    """Run the plates and the airfoil and draw the figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    plate = plates()
    wall_csv = airfoil()
    x, _, cf = upper_surface(wall_csv)
    print(f"separation x/c {separation(x, cf):.3f} (ref {SEPARATION_REF})")
    plot(plate, wall_csv, OUT / "l1_laminar.png")


if __name__ == "__main__":
    main()

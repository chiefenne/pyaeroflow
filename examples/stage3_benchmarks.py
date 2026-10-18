"""Stage 3 example: inviscid cylinder, Blasius flat plate, DFG cylinder in a channel.

Run from the repository root (about two minutes):

    python examples/stage3_benchmarks.py

Output in ``examples/output/``: ``stage3_benchmarks.png`` and the VTU
fields ``stage3_cylinder_euler.vtu``, ``stage3_dfg_2d1.vtu``,
``stage3_flat_plate.vtu``.
"""

from pathlib import Path

import numpy as np

from pyaeroflow.cases import cylinder, flat_plate
from pyaeroflow.io import write_vtu

OUT = Path(__file__).parent / "output"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def fields(result):
    """Cell data for the VTU output of a case result."""
    w = result.w
    return {
        "pressure": w[:, 0],
        "velocity": w[:, 1:],
        "speed": np.hypot(w[:, 1], w[:, 2]),
    }


def main():
    """Run the three cases, print the comparison and plot."""
    OUT.mkdir(exist_ok=True)

    print("V4 inviscid cylinder (Euler) vs potential flow")
    euler = {}
    for nt, nr in ((64, 32), (128, 48)):
        r = cylinder.inviscid(nt, nr)
        euler[(nt, nr)] = r
        err = np.abs(r.cp - r.cp_exact)
        its = len(r.history["iteration"]) - 1
        print(
            f"  O-grid {nt}x{nr}: max |Cp - exact| = {err.max():.4f}, "
            f"cD = {r.cd:.5f}, cL = {r.cl:.2e}, {its} iterations"
        )
    write_vtu(OUT / "stage3_cylinder_euler.vtu", r.problem.mesh, cell_data=fields(r))

    print(
        "B2 DFG benchmark 2D-1 (Re = 20), reference cD 5.5795, cL 0.01062, dp 0.11752"
    )
    dfg = []
    for m, nr, ct in (
        (16, 12, "quad"),
        (32, 20, "quad"),
        (64, 32, "quad"),
        (32, 20, "triangle"),
    ):
        r = cylinder.dfg_2d1(m, nr, ct)
        dfg.append((r.problem.mesh.n_cells, ct, r))
        ref = r.reference
        d_cd = 100 * (r.cd / ref["cd"] - 1)
        d_dp = 100 * (r.dp / ref["dp"] - 1)
        its = len(r.history["iteration"]) - 1
        print(
            f"  {ct:8s} {r.problem.mesh.n_cells:6d} cells: cD = {r.cd:.4f} "
            f"({d_cd:+.2f} %), cL = {r.cl:.5f}, dp = {r.dp:.5f} ({d_dp:+.2f} %), "
            f"{its} iterations"
        )
    write_vtu(
        OUT / "stage3_dfg_2d1.vtu", dfg[2][2].problem.mesh, cell_data=fields(dfg[2][2])
    )

    print("B1 flat plate, Re_L = 1e5, vs Blasius")
    plate = flat_plate.solve()
    sel = (plate.x > 0.2) & (plate.x < 0.9)
    ratio = plate.cf[sel] / plate.cf_blasius[sel]
    d_profile = np.abs(plate.u_profile - plate.u_blasius).max()
    print(
        f"  {plate.problem.mesh.n_cells} cells: Cf / Cf_Blasius in "
        f"[{ratio.min():.3f}, {ratio.max():.3f}] for 0.2 < x < 0.9; "
        f"max |u/U - f'| = {d_profile:.4f} at x = 0.5"
    )
    write_vtu(
        OUT / "stage3_flat_plate.vtu", plate.problem.mesh, cell_data=fields(plate)
    )
    plot(euler, dfg, plate)


def plot(euler, dfg, plate):
    """Cp on the cylinder, DFG convergence, Blasius Cf and profile."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for k, ((nt, nr), r) in enumerate(euler.items()):
        order = np.argsort(r.theta)
        ax.plot(
            np.degrees(r.theta[order]),
            r.cp[order],
            color=COLORS[k],
            lw=2,
            label=f"PyAeroFlow O-grid {nt}x{nr}",
        )
    th = np.linspace(-np.pi, np.pi, 361)
    ax.plot(
        np.degrees(th),
        1 - 4 * np.sin(th) ** 2,
        "--",
        color="0.3",
        lw=1.5,
        label="potential flow",
    )
    ax.set(xlabel="θ [deg]", ylabel="Cp", title="V4: inviscid cylinder")

    ax = axes[0, 1]
    quads = [(n, r) for n, ct, r in dfg if ct == "quad"]
    cells = np.array([n for n, _ in quads])
    ax.plot(
        cells,
        [r.cd for _, r in quads],
        "o-",
        color=COLORS[0],
        lw=2,
        ms=7,
        label="cD, quad meshes",
    )
    tri = [(n, r) for n, ct, r in dfg if ct == "triangle"]
    ax.plot(
        [n for n, _ in tri],
        [r.cd for _, r in tri],
        "s",
        color=COLORS[1],
        ms=8,
        label="cD, triangles",
    )
    ax.axhline(
        cylinder.DFG_REFERENCE["cd"],
        color="0.3",
        ls="--",
        lw=1.5,
        label="reference 5.5795",
    )
    ax.set_xscale("log")
    ax.set(xlabel="cells", ylabel="cD", title="B2: DFG 2D-1, drag coefficient")

    ax = axes[1, 0]
    x = plate.x
    ax.plot(x, plate.cf, color=COLORS[0], lw=2, label="PyAeroFlow")
    ax.plot(
        x[x > 0.02],
        plate.cf_blasius[x > 0.02],
        "--",
        color="0.3",
        lw=1.5,
        label="Blasius 0.664/√Re_x",
    )
    ax.set(
        xlabel="x",
        ylabel="Cf",
        title="B1: flat plate skin friction, Re_L = 1e5",
        ylim=(0, 0.02),
    )

    ax = axes[1, 1]
    ax.plot(
        plate.u_profile,
        plate.eta,
        "o",
        color=COLORS[0],
        mfc="white",
        ms=6,
        label=f"PyAeroFlow, x = {plate.x_profile}",
    )
    ax.plot(plate.u_blasius, plate.eta, "-", color="0.3", lw=1.5, label="Blasius f'(η)")
    ax.set(xlabel="u / U", ylabel="η = y √(U / (ν x))", title="B1: velocity profile")
    for ax in axes.ravel():
        ax.grid(True, color="0.88", lw=0.8)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "stage3_benchmarks.png", dpi=130)
    print(f"plot: {OUT / 'stage3_benchmarks.png'}")


if __name__ == "__main__":
    main()

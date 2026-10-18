"""Stage 4 example: Taylor-Green vortex (time order) and DFG 2D-2 vortex shedding.

Run from the repository root (about two minutes):

    python examples/stage4_unsteady.py [--medium]

Output in ``examples/output/``: ``stage4_unsteady.png``, the final field
``stage4_dfg_2d2.vtu`` and the series ``stage4_dfg_2d2.pvd`` (one period of
shedding, open the .pvd in ParaView). ``--medium`` runs the DFG case on the
4.4k-cell mesh with dt = 0.01 (the verification-test resolution, about ten
minutes).
"""

import sys
from pathlib import Path

import numpy as np

from pyaeroflow.cases import cylinder, taylor_green
from pyaeroflow.io import write_pvd, write_vtu

OUT = Path(__file__).parent / "output"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def fields(w):
    """Cell data for the VTU output."""
    return {
        "pressure": w[:, 0],
        "velocity": w[:, 1:],
        "speed": np.hypot(w[:, 1], w[:, 2]),
    }


def main():
    """Run the two cases, print the comparison and plot."""
    OUT.mkdir(exist_ok=True)

    print("V3 Taylor-Green vortex, 32x32, nu = 0.05, t = 1: temporal order")
    n, nu, t_end = 32, 0.05, 1.0
    inner = taylor_green.inner_settings(6.0)  # tight: isolates the temporal error
    ref = taylor_green.solve(n, nu, 0.0125, t_end, inner=inner)
    print(
        f"  reference dt = 0.0125: velocity error vs exact {ref.velocity_error:.4f}, "
        f"pressure error {ref.pressure_error:.4f}"
    )
    area = ref.problem.mesh.cell_area
    steps = (0.2, 0.1, 0.05)
    errors = {}
    for scheme in ("bdf2", "bdf1"):
        errors[scheme] = []
        for dt in steps:
            r = taylor_green.solve(n, nu, dt, t_end, scheme=scheme, inner=inner)
            dv = r.w[:, 1:] - ref.w[:, 1:]
            errors[scheme].append(np.sqrt(np.sum(area[:, None] * dv**2) / np.sum(area)))
        e = errors[scheme]
        orders = np.log2(e[:-1]) - np.log2(e[1:])
        print(
            f"  {scheme}: errors vs reference {e[0]:.2e} {e[1]:.2e} {e[2]:.2e}, "
            f"orders {orders[0]:.2f} {orders[1]:.2f}"
        )

    medium = "--medium" in sys.argv
    n_s, n_r, dt = (32, 20, 0.01) if medium else (16, 12, 0.02)
    cells = "4.4k cells, dt = 0.01" if medium else "1.5k cells, dt = 0.02"
    print(f"B2 DFG benchmark 2D-2 (Re = 100), {cells}, t = 8")
    series = []
    t_end = 8.0
    n_steps = int(round(t_end / dt))

    def snapshot(step, t, w, problem):
        if step > n_steps - int(round(0.34 / dt)):  # about one shedding period
            file = f"stage4_dfg_2d2_{step:06d}.vtu"
            write_vtu(OUT / file, problem.mesh, cell_data=fields(w))
            series.append((t, file))

    r = cylinder.dfg_2d2(n_s, n_r, time_step=dt, end_time=t_end, callback=snapshot)
    ref2 = r.reference
    print(
        f"  St = {r.strouhal:.4f} (ref {ref2['strouhal']}), cD max = {r.cd_max:.4f} "
        f"(ref {ref2['cd_max']}), cL max = {r.cl_max:.4f} (ref {ref2['cl_max']}), "
        f"dp = {r.dp_half_period:.4f} (ref {ref2['dp']}), periodic: {r.periodic}, "
        f"{r.history['inner_iterations'].mean():.1f} inner iterations per step"
    )
    write_vtu(OUT / "stage4_dfg_2d2.vtu", r.problem.mesh, cell_data=fields(r.w))
    write_pvd(OUT / "stage4_dfg_2d2.pvd", series)
    plot(steps, errors, r)


def plot(steps, errors, dfg):
    """Temporal convergence and the lift/drag history."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for k, (scheme, e) in enumerate(errors.items()):
        ax.loglog(steps, e, "o-", color=COLORS[k], lw=2, ms=7, label=scheme.upper())
    s = np.array(steps)
    ax.loglog(
        s, errors["bdf2"][0] * (s / s[0]) ** 2, "--", color="0.3", lw=1, label="order 2"
    )
    ax.loglog(
        s, errors["bdf1"][0] * (s / s[0]), ":", color="0.3", lw=1, label="order 1"
    )
    ax.set(
        xlabel="Δt",
        ylabel="velocity error vs Δt = 0.0125",
        title="V3: Taylor-Green, temporal order",
    )

    ax = axes[1]
    ax.plot(dfg.time, dfg.cl, color=COLORS[0], lw=1.5, label="cL")
    ax.plot(dfg.time, dfg.cd, color=COLORS[1], lw=1.5, label="cD")
    ax.axhline(
        np.mean(dfg.reference["cl_max"]),
        color=COLORS[0],
        ls="--",
        lw=1,
        label="cL max, reference",
    )
    ax.axhline(
        np.mean(dfg.reference["cd_max"]),
        color=COLORS[1],
        ls="--",
        lw=1,
        label="cD max, reference",
    )
    ax.set(
        xlabel="t [s]",
        ylabel="coefficient",
        ylim=(-1.5, 4.0),
        title=f"B2: DFG 2D-2, St = {dfg.strouhal:.3f}",
    )
    for ax in axes:
        ax.grid(True, color="0.88", lw=0.8)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "stage4_unsteady.png", dpi=130)
    print(f"plot: {OUT / 'stage4_unsteady.png'}")


if __name__ == "__main__":
    main()

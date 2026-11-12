"""Validation of the viscous panel method (``pyaeroflow foil``).

Run from the repository root:

    python examples/ibl_validation.py            # everything, about 2 min
    python examples/ibl_validation.py e387       # one part: e387, sd7003, field

Parts:

* ``e387``: Eppler 387, Re 2e5, N_crit 8.15 (Tu 0.1 %), alpha -2 to 10, against
  NASA TM 4062 (McGhee et al. 1988), Penn State 2007 (Somers & Maughmer) and
  XFOIL; C_p, C_f and shape factor at alpha 4 with the oil-flow bubble.
* ``sd7003``: SD7003, Re 6e4, N_crit 8.15, alpha 0 to 10, against XFOIL and
  the CFD polars of docs/validation.md section 10 (SST-LM, SA-AFT).
* ``field``: the boundary-layer field that ``initial.boundary_layer`` builds,
  on the NACA 0012 TMR 225 x 65 mesh at Re 6e6, alpha 10 (VTU and picture).

Output in ``examples/output/ibl/``: polar CSVs, wall CSVs, PNGs and
``naca0012_ibl_field.vtu``. The figures in docs/figures/ are copies of the PNGs.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pyaeroflow.ibl import analyse  # noqa: E402
from pyaeroflow.ibl.geometry import read_coordinates, repanel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "output" / "ibl"
REF = ROOT / "src" / "pyaeroflow" / "plotting" / "reference"
FOILS = ROOT / "examples" / "airfoils"

# Somers & Maughmer 2007, table IV: Penn State oil flow, upper surface,
# laminar separation and turbulent reattachment in x/c.
E387_BUBBLE = {
    -2: (0.52, 0.78),
    0: (0.48, 0.72),
    2: (0.44, 0.67),
    4: (0.40, 0.61),
    5: (0.35, 0.59),
    6: (0.34, 0.55),
    7: (0.33, 0.47),
}
# docs/validation.md section 10 table: XFOIL 6.99 (N_crit 8.15) and SA-AFT on
# the 320k PyAero mesh, SD7003 Re 6e4.
SD7003_XFOIL = {
    "alpha": [0, 2, 4, 6, 8, 9, 10],
    "cl": [0.058, 0.424, 0.620, 0.813, 0.992, 1.065, 1.109],
    "cd": [0.0148, 0.0165, 0.0188, 0.0243, 0.0342, 0.0429, 0.0557],
}
SD7003_AFT = {
    "alpha": [4, 6, 8, 9, 10],
    "cl": [0.589, 0.783, 0.954, 1.013, 0.740],
    "cd": [0.0176, 0.0218, 0.0303, 0.0390, 0.1260],
}


def csv(path):
    """Read a CSV with '#' comment lines into a structured array."""
    lines = [ln for ln in open(path) if not ln.startswith("#")]
    return np.atleast_1d(np.genfromtxt(lines, delimiter=",", names=True))


def xfoil_zone(path, zone):
    """One zone of the built-in XFOIL reference file (alpha cl cd cm ...)."""
    rows, take = [], False
    for line in open(path):
        if line.startswith("zone"):
            take = f'"{zone}"' in line
        elif take and line[:1] not in "#v":
            rows.append([float(v) for v in line.split()])
    a = np.array(rows)
    return {"alpha": a[:, 0], "cl": a[:, 1], "cd": a[:, 2], "cm": a[:, 3]}


def polar(name, nodes, alphas, re, ncrit):
    """Run the angles one by one; write and return the polar rows."""
    rows, walls = [], {}
    previous = None
    for a in alphas:
        r = analyse(nodes, a, re, ncrit=ncrit, initial=previous)
        previous = r if r.converged else previous
        xu, xl = r.x_transition
        print(
            f"{name} alpha {a:5.1f}: CL {r.cl:.4f} CD {r.cd:.5f} CM {r.cm:.4f} "
            f"x_tr {xu:.3f}/{xl:.3f} {'ok' if r.converged else 'NOT converged'} "
            f"({r.iterations}, {r.start})",
            flush=True,
        )
        rows.append((a, r.cl, r.cd, r.cdf, r.cm, xu, xl, r.converged, r.iterations))
        walls[a] = r
    head = "alpha,cl,cd,cdf,cm,xtr_upper,xtr_lower,converged,iterations"
    np.savetxt(
        OUT / f"{name}.polar.csv",
        np.array(rows, dtype=float),
        delimiter=",",
        header=head,
        comments="",
        fmt="%.8g",
    )
    return np.array(rows, dtype=float), walls


def wall_csv(path, r):
    """Write the surface distributions of one result."""
    n = r.nodes.shape[0]
    h = r.delta_star[:n] / r.theta[:n]
    np.savetxt(
        path,
        np.column_stack((r.nodes, r.cp, r.cf[:n], r.theta[:n], r.delta_star[:n], h)),
        delimiter=",",
        header="x,y,cp,cf,theta,delta_star,H",
        comments="",
    )


def surfaces(r):
    """Index arrays of the upper and lower surface (split at the stagnation)."""
    n = r.nodes.shape[0]
    s = r.stagnation
    upper = np.arange(s + 1, n)
    lower = np.arange(s, -1, -1)
    if r.nodes[upper, 1].mean() < r.nodes[lower, 1].mean():
        upper, lower = lower, upper
    return upper, lower


def e387():
    """E387 polar against the experiments and XFOIL; distributions at alpha 4."""
    nodes = repanel(read_coordinates(FOILS / "e387.dat"), 200)
    alphas = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    run, walls = polar("e387_re2e5_n8.15", nodes, alphas, 2e5, 8.15)
    wall_csv(OUT / "e387_re2e5_a4.wall.csv", walls[4])
    tm = csv(REF / "e387_tm4062_re200k.csv")
    penn = csv(ROOT / "examples" / "cases" / "penn_state_2007_re200k.csv")
    xf = xfoil_zone(REF / "e387_xfoil_re200k.dat", "N_crit 8.15")

    a, cl, cd, cm = run[:, 0], run[:, 1], run[:, 2], run[:, 4]
    head = f"{'alpha':>5} {'CL':>7} {'CD':>8} {'CM':>8}"
    print(f"\n{head}   dCL/dCD% TM4062, Penn, XFOIL")
    for i, al in enumerate(a):
        cells = []
        for ref in (tm, penn, xf):
            if ref["alpha"].min() <= al <= ref["alpha"].max():
                rl = np.interp(al, ref["alpha"], ref["cl"])
                rd = np.interp(al, ref["alpha"], ref["cd"])
                cells.append(f"{cl[i] - rl:+.3f}/{100 * (cd[i] / rd - 1):+5.1f}")
            else:
                cells.append("      -      ")
        print(f"{al:5.1f} {cl[i]:7.4f} {cd[i]:8.5f} {cm[i]:8.4f}   " + "  ".join(cells))
    print("\nalpha  x_tr upper   oil flow LS / TR (Penn State)")
    for i, al in enumerate(a):
        if int(al) in E387_BUBBLE:
            ls, tr = E387_BUBBLE[int(al)]
            print(f"{al:5.1f}  {run[i, 5]:.3f}        {ls:.2f} / {tr:.2f}")

    refs = (
        (tm, "NASA TM 4062", "C3", "^"),
        (penn, "Penn State 2007", "C2", "s"),
        (xf, "XFOIL 6.99, N 8.15", "0.4", None),
    )
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for ref, lab, col, mk in refs:
        sty = dict(color=col, label=lab)
        if mk:
            sty.update(ls="none", marker=mk, mfc="none")
        else:
            sty.update(ls="--")
        ax[0].plot(ref["alpha"], ref["cl"], **sty)
        ax[1].plot(ref["cd"], ref["cl"], **sty)
        ax[2].plot(ref["alpha"], ref["cm"], **sty)
    ok = run[:, 7] > 0
    sty = dict(color="C0", marker="o", ms=4, label="pyaeroflow foil, N 8.15")
    bad = dict(color="C3", marker="x", ls="none", ms=7, label="not converged")
    ax[0].plot(a[ok], cl[ok], **sty)
    ax[1].plot(cd[ok], cl[ok], **sty)
    ax[2].plot(a[ok], cm[ok], **sty)
    if not ok.all():
        ax[0].plot(a[~ok], cl[~ok], **bad)
        ax[1].plot(cd[~ok], cl[~ok], **bad)
        ax[2].plot(a[~ok], cm[~ok], **bad)
    ax[0].set(xlabel="α (deg)", ylabel="$C_L$")
    ax[1].set(xlabel="$C_D$", ylabel="$C_L$", xlim=(0, 0.04))
    ax[2].set(xlabel="α (deg)", ylabel="$C_M$ (c/4, nose-up positive)", ylim=(-0.12, 0))
    for x in ax:
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("Eppler 387, Re = 2·10⁵: viscous panel method against experiments")
    fig.tight_layout()
    fig.savefig(OUT / "ibl_e387_polar.png", dpi=110)
    plt.close(fig)

    r = walls[4]
    up, lo = surfaces(r)
    x = r.nodes[:, 0]
    h = r.delta_star[: len(x)] / r.theta[: len(x)]
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax[0].plot(x[up], r.cp[up], "C0", label="upper")
    ax[0].plot(x[lo], r.cp[lo], "C1", label="lower")
    ax[0].invert_yaxis()
    ax[0].set_ylabel("$C_p$")
    ax[1].plot(x[up], r.cf[up], "C0")
    ax[1].plot(x[lo], r.cf[lo], "C1")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(ylabel="$C_f$", ylim=(-0.002, 0.01))
    ax[2].plot(x[up], h[up], "C0")
    ax[2].plot(x[lo], h[lo], "C1")
    ax[2].set(ylabel="$H = δ^*/θ$", xlabel="x/c", ylim=(1, 8))
    ls, tr = E387_BUBBLE[4]
    for x_ in ax:
        x_.axvspan(ls, tr, color="0.85", zorder=0)
        x_.axvline(r.x_transition[0], color="C0", ls=":", lw=1)
        x_.grid(alpha=0.3)
    ax[0].plot([], [], color="0.85", lw=8, label="oil flow bubble (Penn State)")
    ax[0].plot([], [], "C0:", label="computed transition")
    ax[0].legend(fontsize=8)
    ax[0].set_title(
        f"E387, Re 2·10⁵, α 4°, N 8.15: CL {r.cl:.3f}, CD {r.cd:.5f}, "
        f"x_tr {r.x_transition[0]:.3f}"
    )
    fig.tight_layout()
    fig.savefig(OUT / "ibl_e387_a4_cp_cf.png", dpi=110)
    plt.close(fig)


def sd7003():
    """SD7003 polar against XFOIL and the CFD polars of the transition models."""
    nodes = repanel(read_coordinates(FOILS / "sd7003.dat"), 200)
    alphas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    run, walls = polar("sd7003_re6e4_n8.15", nodes, alphas, 6e4, 8.15)
    wall_csv(OUT / "sd7003_re6e4_a4.wall.csv", walls[4])
    lm = csv(ROOT / "docs" / "data" / "lm_sd7003_320k.polar.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for ref, lab, sty in (
        (SD7003_XFOIL, "XFOIL 6.99, N 8.15", dict(color="0.4", ls="--")),
        (
            SD7003_AFT,
            "SA-AFT, 320k CFD",
            dict(color="C2", ls="none", marker="s", mfc="none"),
        ),
        (
            {"alpha": lm["alpha"], "cl": lm["CL"], "cd": lm["CD"]},
            "SST-LM, 320k CFD",
            dict(color="C3", ls="none", marker="^", mfc="none"),
        ),
    ):
        ax[0].plot(ref["alpha"], ref["cl"], label=lab, **sty)
        ax[1].plot(ref["cd"], ref["cl"], label=lab, **sty)
    ok = run[:, 7] > 0
    sty = dict(color="C0", marker="o", ms=4, label="pyaeroflow foil, N 8.15")
    bad = dict(color="C3", marker="x", ls="none", ms=7, label="not converged")
    ax[0].plot(run[ok, 0], run[ok, 1], **sty)
    ax[1].plot(run[ok, 2], run[ok, 1], **sty)
    if not ok.all():
        ax[0].plot(run[~ok, 0], run[~ok, 1], **bad)
        ax[1].plot(run[~ok, 2], run[~ok, 1], **bad)
    ax[0].set(xlabel="α (deg)", ylabel="$C_L$")
    ax[1].set(xlabel="$C_D$", ylabel="$C_L$", xlim=(0, 0.08))
    for x in ax:
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("SD7003, Re = 6·10⁴: viscous panel method, XFOIL and CFD")
    fig.tight_layout()
    fig.savefig(OUT / "ibl_sd7003_polar.png", dpi=110)
    plt.close(fig)


def field():
    """IBL initial field on the NACA 0012 TMR mesh: VTU and a picture."""
    from matplotlib.collections import PolyCollection

    from pyaeroflow.ibl.field import mesh_ibl_field
    from pyaeroflow.io.vtu import write_vtu
    from pyaeroflow.mesh.reader import read_mesh

    mesh = read_mesh(ROOT / "examples" / "meshes" / "naca0012_tmr_225x65.su2")
    alpha, re = 10.0, 6e6
    vel = (np.cos(np.radians(alpha)), np.sin(np.radians(alpha)))
    f = mesh_ibl_field(mesh, "airfoil", vel, 1.0 / re, ncrit=9.0)
    r = f.result
    print(
        f"NACA 0012 alpha 10, Re 6e6, N 9: CL {r.cl:.4f} CD {r.cd:.5f} "
        f"x_tr {r.x_transition[0]:.3f}/{r.x_transition[1]:.3f}"
    )
    speed = np.hypot(f.velocity[:, 0], f.velocity[:, 1])
    cp = 2.0 * f.dynamic
    nut = f.eddy_viscosity * re
    write_vtu(
        OUT / "naca0012_ibl_field.vtu",
        mesh,
        cell_data={"velocity": f.velocity, "cp": cp, "nu_t_over_nu": nut},
    )
    off, conn = mesh.cell_ptr, mesh.cell_nodes
    polys = [mesh.nodes[conn[off[i] : off[i + 1]]] for i in range(len(off) - 1)]
    fig, ax = plt.subplots(3, 1, figsize=(10, 11))
    zoom = dict(xlim=(0.55, 1.35), ylim=(-0.09, 0.11))
    full = dict(xlim=(-0.2, 1.6), ylim=(-0.35, 0.45))
    for x_, val, lab, lim, box in (
        (ax[0], speed, "|V| / U∞", (0, 1.8), full),
        (ax[1], speed, "|V| / U∞", (0, 1.3), zoom),
        (ax[2], nut, "ν_t / ν", (0, np.percentile(nut, 99.9)), zoom),
    ):
        pc = PolyCollection(
            polys, array=val, cmap="viridis", edgecolors="face", linewidths=0.2
        )
        pc.set_clim(*lim)
        x_.add_collection(pc)
        x_.set(aspect="equal", **box)
        fig.colorbar(pc, ax=x_, label=lab, shrink=0.8)
    ax[0].set_title(
        "Boundary-layer start field, NACA 0012, Re 6·10⁶, α 10° "
        f"(CL {r.cl:.3f}, CD {r.cd:.5f})"
    )
    ax[1].set_title("trailing edge: boundary layers and wake defect", fontsize=10)
    ax[2].set_title("eddy viscosity of the turbulent layers", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ibl_naca0012_field.png", dpi=110)
    plt.close(fig)


def main():
    """Run the parts named on the command line (all by default)."""
    OUT.mkdir(parents=True, exist_ok=True)
    parts = sys.argv[1:] or ["e387", "sd7003", "field"]
    for p in parts:
        {"e387": e387, "sd7003": sd7003, "field": field}[p]()
    print(f"output in {OUT}")


if __name__ == "__main__":
    main()

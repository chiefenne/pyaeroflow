"""Wall Cp and Cf of the NASA hump runs against the experiment and CFL3D SA.

Usage: python nasa_hump_plot.py NOFLOW.wall_wall.csv SUCTION.wall_wall.csv OUT.png

The CSVs are the wall files of nasa_hump_sa_noflow.toml and
nasa_hump_sa_suction.toml. Prints separation and reattachment (Cf = 0 on
0.3 < x/c < 2). Reference data: built-in tmr_hump_* (NASA TMR, CC0-1.0).
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pyaeroflow.plotting.readers import read_reference  # noqa: E402

EXP_REATTACH = {"noflow": 1.11, "suction": 0.94}


def wall(path):
    """x, Cp, Cf of a wall CSV, sorted by x."""
    w = np.genfromtxt(path, delimiter=",", names=True)
    o = np.argsort(w["x"])
    return w["x"][o], w["cp"][o], w["cf"][o]


def zero_crossings(x, cf, lo=0.3, hi=2.0):
    """Positions where Cf changes sign on lo < x < hi (linear interpolation)."""
    m = (x > lo) & (x < hi)
    x, cf = x[m], cf[m]
    k = np.where(np.diff(np.sign(cf)) != 0)[0]
    return x[k] - cf[k] * (x[k + 1] - x[k]) / (cf[k + 1] - cf[k])


def ref(name):
    """First table of a built-in reference: (x, y)."""
    t = read_reference(f"builtin:{name}")[0]
    return t.columns["x"], t.columns[name.split("_")[2]]


def main(argv):
    """Command line: see the module docstring."""
    if len(argv) != 4:
        sys.exit(__doc__)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, (case, path) in enumerate((("noflow", argv[1]), ("suction", argv[2]))):
        x, cp, cf = wall(path)
        z = zero_crossings(x, cf)
        print(
            f"{case}: separation x/c = {z[0]:.4f}, reattachment x/c = {z[-1]:.4f} "
            f"(experiment {EXP_REATTACH[case]})"
        )
        ax[0, col].plot(*ref(f"tmr_hump_cp_exp_{case}"), "ko", ms=3, label="experiment")
        ax[0, col].plot(x, cp, "C0-", label="PyAeroFlow SA-neg")
        if case == "noflow":
            ax[0, col].plot(
                *ref("tmr_hump_cp_cfl3d_sa_noflow"), "C1--", label="CFL3D SA 817"
            )
            ax[1, col].plot(
                *ref("tmr_hump_cf_exp_noflow"), "ko", ms=3, label="experiment"
            )
            ax[1, col].plot(
                *ref("tmr_hump_cf_cfl3d_sa_noflow"), "C1--", label="CFL3D SA 817"
            )
        ax[1, col].plot(x, cf, "C0-", label="PyAeroFlow SA-neg")
        ax[1, col].axvline(EXP_REATTACH[case], color="k", lw=0.8, ls=":")
        ax[1, col].axhline(0.0, color="0.6", lw=0.6)
        ax[0, col].set_title(
            f"{'no flow control' if case == 'noflow' else 'steady suction'}: "
            f"reattachment {z[-1]:.2f} (exp. {EXP_REATTACH[case]})"
        )
        ax[0, col].invert_yaxis()
        ax[0, col].set_ylabel("Cp")
        ax[1, col].set_ylabel("Cf")
        ax[1, col].set_xlabel("x/c")
        ax[1, col].set_ylim(-0.004, 0.008)
        for a in ax[:, col]:
            a.set_xlim(-0.5, 2.0)
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(argv[3], dpi=130)
    print(argv[3])


if __name__ == "__main__":
    main(sys.argv)

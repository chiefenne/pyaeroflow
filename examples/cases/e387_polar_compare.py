#!/usr/bin/env python3
"""Compare a PyAeroFlow E387 polar (Re 2e5) with Penn State 2007, TM 4062 and XFOIL.

Usage:  python e387_polar_compare.py <name>.polar.csv [-o prefix] [--label AFT]

Input: the CSV written by `pyaeroflow polar` (alpha, CL, CD, ..., CM,
iterations, converged). Reference data: reference/e387/ next to tools/.
The solver's CM is nose-down positive, so -CM is compared with c_m.
Prints CL, CD, c_m of the run against each reference at the run's angles
(references interpolated linearly in alpha) and writes <prefix>_polar.png.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference", "e387")


def load(name):
    with open(os.path.join(REF, name)) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return np.genfromtxt(lines, delimiter=",", names=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("polar")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--label", default="PyAeroFlow")
    a = ap.parse_args()
    run = np.genfromtxt(a.polar, delimiter=",", names=True)
    run = np.atleast_1d(run)
    alpha, cl, cd, cm = run["alpha"], run["CL"], run["CD"], -run["CM"]
    penn, tm = load("penn_state_2007_re200k.csv"), load("tm4062_re200k.csv")
    xf = json.load(open(os.path.join(REF, "xfoil_polar_ncrit_e387_re200k.json")))
    xfd = {}
    for n in ("8.15", "10.07"):
        k = sorted(xf[n], key=float)
        xfd[n] = {m: np.array([xf[n][i][m] for i in k]) for m in ("cl", "cd", "cm")}
        xfd[n]["alpha"] = np.array([float(i) for i in k])
    refs = {
        "Penn State 2007": penn,
        "NASA TM 4062": tm,
        "XFOIL N=8.15": xfd["8.15"],
        "XFOIL N=10.07": xfd["10.07"],
    }

    def at(ref, key):
        return np.interp(alpha, ref["alpha"], ref[key])

    print(f"{'alpha':>6} {'CL':>7} {'CD':>8} {'cm':>8}  " + "  ".join(f"{n:>26}" for n in refs))
    for i, al in enumerate(alpha):
        cells = []
        for ref in refs.values():
            lo, hi = ref["alpha"].min(), ref["alpha"].max()
            if not lo <= al <= hi:
                cells.append(f"{'-':>26}")
                continue
            dl, dd = cl[i] - at(ref, "cl")[i], 100 * (cd[i] / at(ref, "cd")[i] - 1)
            cells.append(f"dCL {dl:+.3f} dCD {dd:+5.1f}%".rjust(26))
        print(f"{al:6.2f} {cl[i]:7.4f} {cd[i]:8.5f} {cm[i]:8.4f}  " + "  ".join(cells))
    if "converged" in run.dtype.names and not run["converged"].all():
        bad = alpha[run["converged"] == 0]
        print("not flagged converged:", ", ".join(f"{b:g}" for b in bad))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    sty = {"Penn State 2007": ("C2", "s"), "NASA TM 4062": ("C3", "^"),
           "XFOIL N=8.15": ("C1", "-"), "XFOIL N=10.07": ("C1", "--")}
    for n, ref in refs.items():
        c, m = sty[n]
        kw = dict(color=c, label=n)
        kw.update(dict(ls="none", marker=m, ms=4) if m in "s^" else dict(ls=m, lw=1.2))
        ax[0].plot(ref["alpha"], ref["cl"], **kw)
        ax[1].plot(ref["cd"], ref["cl"], **kw)
        ax[2].plot(ref["alpha"], ref["cm"], **kw)
    for axi, (x, y) in zip(ax, ((alpha, cl), (cd, cl), (alpha, cm))):
        axi.plot(x, y, "C0o-", label=a.label, lw=1.5, ms=5)
        axi.grid(alpha=0.3)
    ax[0].set_xlabel("alpha [deg]"); ax[0].set_ylabel("CL"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("CD"); ax[1].set_ylabel("CL"); ax[1].set_xlim(0.006, 0.03)
    ax[2].set_xlabel("alpha [deg]"); ax[2].set_ylabel("cm (nose up +)")
    ax[0].set_xlim(-3, 10); ax[2].set_xlim(-3, 10); ax[0].set_ylim(-0.1, 1.4)
    fig.suptitle("E387, Re 2e5: " + os.path.basename(a.polar))
    out = (a.out or os.path.splitext(a.polar)[0]) + "_polar.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()

"""St of a heated-plate wall CSV against Kays & Crawford.

Usage: python stanton_check.py output/plate_heated_sa/plate_heated_sa.wall_plate.csv
"""

import sys

import numpy as np

d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
rho, U, mu, pr = 101325 / (287.058 * 300), 78.4, 1.846e-5, 0.71
for x in [0.1, 0.25, 0.5, 1.0, 1.5, 1.9]:
    i = np.argmin(abs(d["x"] - x))
    rex = rho * U * d["x"][i] / mu
    kc = 0.0287 * rex**-0.2 * pr**-0.4
    st, cf = d["st"][i], d["cf"][i]
    print(
        f"x={d['x'][i]:.3f} Re_x={rex:.3e} St={st:.5e} KC={kc:.5e} "
        f"ratio={st / kc:.4f}  2St/cf={2 * st / cf:.3f}"
    )

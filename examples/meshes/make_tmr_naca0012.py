"""TMR NACA 0012 C-grid (PLOT3D 2D, formatted) -> SU2, with optional refinement.

The TMR family (farfield at 500 chords) is nested: each grid is every other
point of the next finer one. Only the 225x65 and 113x33 grids are real files
in the public tmbwg/turbmodels repository (the finer ones are Git-LFS
objects), so the finer levels are rebuilt from 225x65 by inserting midpoints:
shape-preserving (PCHIP) interpolation in index space, then the new wall
nodes are put back on the TMR NACA 0012 formula and the same shift is applied
to the inner nodes of that grid line, fading out away from the wall.
Coarsening the result by every other point returns the 225x65 grid exactly.
The refined grids are close to the TMR grids but not identical to them.

Usage: python make_tmr_naca0012.py <in.p2dfmt> <out.su2> [levels]

    python make_tmr_naca0012.py naca0012_tmr_225x65.p2dfmt n0012_449.su2 1

gives 449x129 (57 344 cells), levels 2 gives 897x257 (229 376 cells); level 0
writes the committed naca0012_tmr_225x65.su2.
"""

import sys

import numpy as np
from scipy.interpolate import PchipInterpolator


def read_p2d(path):
    """Read a formatted single-block 2D PLOT3D grid: x, y indexed [i, j]."""
    with open(path) as f:
        v = np.array(f.read().split(), dtype=float)
    ni, nj = int(v[1]), int(v[2])
    n = ni * nj
    x = v[3 : 3 + n].reshape(nj, ni).T
    y = v[3 + n : 3 + 2 * n].reshape(nj, ni).T
    return x, y  # [i, j]


def naca0012_tmr(x):
    """Upper-surface y of the TMR NACA 0012 (closed sharp trailing edge)."""
    x = np.clip(x, 0.0, 1.0)
    return 0.594689181 * (
        0.298222773 * np.sqrt(x)
        - 0.127125232 * x
        - 0.357907906 * x**2
        + 0.291984971 * x**3
        - 0.105174606 * x**4
    )


def wake_split(x, y):
    """Return the wake points per side at j = 0 (nodes i and ni-1-i coincide)."""
    ni = x.shape[0]
    k = 0
    while np.hypot(x[k, 0] - x[ni - 1 - k, 0], y[k, 0] - y[ni - 1 - k, 0]) < 1e-10:
        k += 1
    return k  # i = 0..k-1 wake (lower copy); TE node is i = k-1 ... check below


def refine(x, y):
    """Insert midpoints in i and j; return x, y and the largest wall shift."""
    ni, nj = x.shape
    i0, j0 = np.arange(ni), np.arange(nj)
    i1, j1 = np.arange(2 * ni - 1) / 2, np.arange(2 * nj - 1) / 2
    # shape-preserving (PCHIP) along i, then along j: no overshoot at the
    # trailing-edge kink or in the strongly stretched wall-normal direction
    xf = PchipInterpolator(j0, PchipInterpolator(i0, x, axis=0)(i1), axis=1)(j1)
    yf = PchipInterpolator(j0, PchipInterpolator(i0, y, axis=0)(i1), axis=1)(j1)
    xf[::2, ::2], yf[::2, ::2] = x, y  # exact at the old nodes
    nw = wake_split(x, y)  # wake points per side on the coarse grid (incl. TE)
    te_lo, te_up = nw - 1, ni - nw  # TE node indices (coarse)
    te_lo_f, te_up_f = 2 * te_lo, 2 * te_up
    # wall: move each new node to the nearest point of the exact contour
    th = np.linspace(0.0, np.pi, 400001)
    xc = 0.5 * (1.0 - np.cos(th))
    yc = naca0012_tmr(xc)
    dx = np.zeros(xf.shape[0])
    dy = np.zeros(xf.shape[0])
    for i in range(te_lo_f + 1, te_up_f, 2):  # new wall nodes only
        sign = 1.0 if yf[i, 0] >= 0 else -1.0
        k = np.argmin((xc - xf[i, 0]) ** 2 + (sign * yc - yf[i, 0]) ** 2)
        dx[i] = xc[k] - xf[i, 0]
        dy[i] = sign * yc[k] - yf[i, 0]
    # wake cut: average the two copies (they are equal by symmetry of the spline)
    for i in range(te_lo_f):
        m = 0.5 * (yf[i, 0] + yf[-1 - i, 0])
        yf[i, 0] = yf[-1 - i, 0] = m
        m = 0.5 * (xf[i, 0] + xf[-1 - i, 0])
        xf[i, 0] = xf[-1 - i, 0] = m
    # fade the wall shift out over the j-line (distance from the wall)
    d = np.hypot(xf - xf[:, :1], yf - yf[:, :1])
    w = np.exp(-((d / 0.02) ** 2))
    xf += dx[:, None] * w
    yf += dy[:, None] * w
    return xf, yf, np.max(np.hypot(dx, dy))


def to_su2(x, y, path):
    """Write the C-grid as SU2 (wake cut merged); markers airfoil, farfield."""
    ni, nj = x.shape
    nw = wake_split(x, y)
    te_lo = nw - 1
    nid = np.arange(ni * nj).reshape(nj, ni).T.copy()  # node id [i, j]
    # merge the upper wake copies (j = 0, i > ni-1-nw ... ) into the lower ones
    for i in range(nw):
        nid[ni - 1 - i, 0] = nid[i, 0]
    used = np.unique(nid)
    remap = -np.ones(ni * nj, dtype=int)
    remap[used] = np.arange(used.size)
    nid = remap[nid]
    pts = np.zeros((used.size, 2))
    pts[nid.ravel(), 0] = x.ravel()
    pts[nid.ravel(), 1] = y.ravel()
    quads = []
    for j in range(nj - 1):
        for i in range(ni - 1):
            q = [nid[i, j], nid[i + 1, j], nid[i + 1, j + 1], nid[i, j + 1]]
            quads.append(q)
    quads = np.array(quads)
    p = pts[quads]
    px, py = p[:, :, 0], p[:, :, 1]
    area = 0.5 * np.sum(px * np.roll(py, -1, 1) - np.roll(px, -1, 1) * py, 1)
    if np.median(area) < 0:
        quads = quads[:, ::-1]
        area = -area
    assert area.min() > 0, f"folded cells: {np.sum(area <= 0)}"
    wall = [(nid[i, 0], nid[i + 1, 0]) for i in range(te_lo, ni - nw)]
    far = [(nid[i, nj - 1], nid[i + 1, nj - 1]) for i in range(ni - 1)]
    far += [(nid[0, j], nid[0, j + 1]) for j in range(nj - 1)]
    far += [(nid[ni - 1, j], nid[ni - 1, j + 1]) for j in range(nj - 1)]
    out = ["NDIME= 2", f"NELEM= {len(quads)}"]
    out += [f"9 {a} {b} {c} {d} {k}" for k, (a, b, c, d) in enumerate(quads)]
    out.append(f"NPOIN= {len(pts)}")
    out += [f"{px:.16e} {py:.16e} {k}" for k, (px, py) in enumerate(pts)]
    out += ["NMARK= 2", "MARKER_TAG= airfoil", f"MARKER_ELEMS= {len(wall)}"]
    out += [f"3 {a} {b}" for a, b in wall]
    out += ["MARKER_TAG= farfield", f"MARKER_ELEMS= {len(far)}"]
    out += [f"3 {a} {b}" for a, b in far]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return len(quads), len(wall)


if __name__ == "__main__":
    x, y = read_p2d(sys.argv[1])
    for _ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 0):
        xc, yc = x, y
        x, y, shift = refine(x, y)
        assert np.allclose(x[::2, ::2], xc) and np.allclose(y[::2, ::2], yc)
        print(f"refined to {x.shape[0]}x{x.shape[1]}, max wall shift {shift:.2e}")
    nc, nwall = to_su2(x, y, sys.argv[2])
    nw = wake_split(x, y)
    first = np.min(np.hypot(x[:, 1] - x[:, 0], y[:, 1] - y[:, 0])[nw:-nw])
    print(
        f"{x.shape[0]}x{x.shape[1]}: {nc} cells, {nwall} wall faces, "
        f"wake points/side {nw}, first wall spacing {first:.2e}"
    )

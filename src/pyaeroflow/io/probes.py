"""Point probes: the solution at fixed points per iteration or time step.

``output.probes`` lists points [x, y] in metres. Each probe is linearly
interpolated from the cell centres of the cell that contains it, its
neighbours and the neighbours of these (and their boundary faces, with the
boundary state), by a Delaunay triangulation of that local point cloud, or
a weighted least-squares plane where the triangulation does not cover the
point (next to a periodic boundary). The weights are computed once, a
sample costs one weighted sum. Rows go to
``<name>.probes.csv``, continued after a restart like the history CSV.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay, cKDTree

from pyaeroflow.io.history import _open_history


def _inside(nodes, point, tol=1e-12):
    """Return True if ``point`` lies in the convex polygon ``nodes``."""
    edge = np.roll(nodes, -1, axis=0) - nodes
    rel = point - nodes
    cross = edge[:, 0] * rel[:, 1] - edge[:, 1] * rel[:, 0]
    scale = tol * max(np.abs(edge).max(), 1e-300) ** 2
    return bool(np.all(cross >= -scale) or np.all(cross <= scale))


def containing_cell(mesh, point, tree=None, k=32):
    """Index of the cell that contains ``point``, or None (outside the mesh)."""
    tree = cKDTree(mesh.cell_centroid) if tree is None else tree
    _, near = tree.query(point, k=min(k, mesh.n_cells))
    for c in np.atleast_1d(near):
        if _inside(mesh.nodes[mesh.cell(int(c))], point):
            return int(c)
    return None


def _weights(cloud, index, xy):
    """Stencil and barycentric weights of ``xy`` in the local Delaunay mesh."""
    try:
        tri = Delaunay(cloud)
        simplex = int(tri.find_simplex(xy))
    except Exception:  # degenerate cloud (collinear points)
        simplex = -1
    if simplex >= 0:
        t = tri.transform[simplex]
        b = t[:2] @ (xy - t[2])
        return index[tri.simplices[simplex]], np.append(b, 1.0 - b.sum())
    # Not covered (near a periodic boundary or a corner): a weighted
    # least-squares plane through the cloud, exact for linear fields.
    rel = cloud - xy
    d2 = np.maximum(np.sum(rel**2, axis=1), 1e-24 * max(np.sum(rel**2).max(), 1e-300))
    a = np.column_stack((np.ones(len(cloud)), rel))
    aw = a / d2[:, None]
    return index, np.linalg.solve(a.T @ aw, aw.T)[0]


class ProbeSampler:
    """Interpolation weights of the probes, computed once.

    Parameters
    ----------
    problem : FlowProblem
        Non-dimensional problem (lengths divided by the reference length).
    points : sequence of (x, y)
        Probe positions in metres.
    scaling : Scaling
    """

    def __init__(self, problem, points, scaling):
        self.problem, self.scaling = problem, scaling
        self.points = np.asarray(points, dtype=float).reshape(-1, 2)
        mesh, geom = problem.mesh, problem.geom
        b_owner = np.asarray(geom.b_owner)
        b_centre = np.asarray(geom.b_centre)
        tree = cKDTree(mesh.cell_centroid)
        self.cells, self.stencils, self.weights = [], [], []
        outside = []
        for i, xy in enumerate(self.points / scaling.length):
            cell = containing_cell(mesh, xy, tree)
            if cell is None:
                outside.append(i)
                continue
            ring = {cell}
            for _ in range(2):  # neighbours and their neighbours
                faces = np.flatnonzero(
                    np.isin(mesh.owner, list(ring))
                    | np.isin(mesh.neighbour, list(ring))
                )
                ring |= set(mesh.owner[faces]) | set(mesh.neighbour[faces])
                ring.discard(-1)
            cells = np.array(sorted(ring))
            bfaces = np.flatnonzero(np.isin(b_owner, cells))
            cloud = np.vstack((mesh.cell_centroid[cells], b_centre[bfaces]))
            # Stencil index: cells as they are, boundary faces after n_cells.
            index = np.concatenate((cells, mesh.n_cells + bfaces))
            stencil, weights = _weights(cloud, index, xy)
            self.cells.append(cell)
            self.stencils.append(stencil)
            self.weights.append(weights)
        if outside:
            listed = ", ".join(
                f"probe {i + 1} ({self.points[i, 0]:g}, {self.points[i, 1]:g}) m"
                for i in outside
            )
            raise ValueError(f"output.probes: outside the mesh: {listed}")
        self.columns = ("p", "u", "v", "T") if problem.compressible else ("p", "u", "v")

    def sample(self, w):
        """Probe values (n_probes, len(columns)) in SI units."""
        problem, geom = self.problem, self.problem.geom
        w = np.asarray(w)
        w_b = np.asarray(
            problem.physics.boundary_state(
                problem.bc, geom.b_normal, jnp.asarray(w)[geom.b_owner]
            )
        )
        values = np.vstack((w, w_b))
        n = len(self.columns)
        pairs = zip(self.stencils, self.weights, strict=True)
        out = np.array([wt @ values[st, :n] for st, wt in pairs])
        s = self.scaling
        factor = np.array([s.pressure, s.velocity, s.velocity, s.temperature][:n])
        return out * factor

    def describe(self):
        """Log lines: position and containing cell of each probe."""
        return [
            f"probe {i + 1}: ({x:g}, {y:g}) m, cell {c}"
            for i, ((x, y), c) in enumerate(zip(self.points, self.cells, strict=True))
        ]


class ProbeWriter:
    """``<name>.probes.csv``: one row per sample.

    Columns: ``iteration`` (steady) or ``step, time`` (unsteady), then
    ``probe<i>_<column>`` per probe and column (SI units).
    """

    def __init__(self, path, sampler, every=1, unsteady=False, resume_after=None):
        self.path, self.sampler, self.every = Path(path), sampler, every
        self.unsteady = unsteady
        header = ["step", "time"] if unsteady else ["iteration"]
        header += [
            f"probe{i + 1}_{c}"
            for i in range(len(sampler.points))
            for c in sampler.columns
        ]
        self._file, self._writer = _open_history(self.path, header, resume_after)

    def row(self, number, w, time=None):
        """Write a row if ``number`` is a multiple of the interval."""
        if number % self.every:
            return
        values = self.sampler.sample(w).ravel()
        lead = [number, f"{time:.8e}"] if self.unsteady else [number]
        self._writer.writerow(lead + [f"{v:.8e}" for v in values])
        self._file.flush()

    def close(self):
        """Close the file."""
        self._file.close()

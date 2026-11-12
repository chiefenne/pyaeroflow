"""Inviscid panel method for airfoils: potential flow around closed walls.

Formulation after Drela (1989, "XFOIL: An Analysis and Design System for Low
Reynolds Number Airfoils", Lecture Notes in Engineering 54), written from
the published equations: the flow is the freestream plus a vortex sheet of
linearly varying strength gamma on the straight panels between the wall
nodes. Its stream function

    psi(x) = u_inf y - v_inf x + 1/(2 pi) int gamma(s) ln r(s; x) ds

is required to take one constant value psi_0 at every node of a body, which
makes the body a streamline. The Kutta condition closes the system:
gamma_first + gamma_last = 0 at the trailing edge. A blunt trailing edge is
closed by one base panel with a constant source and vortex strength set by
the mean of the two trailing-edge sheet strengths, so the flow leaves the
base along the trailing-edge bisector; at a sharp trailing edge, where the
first and last node coincide, the duplicate equation is replaced by an
extrapolation of gamma to the trailing edge from both sides.

Conventions: nodes run from the upper trailing-edge corner over the upper
side to the leading edge and back along the lower side (counter-clockwise),
gamma > 0 is clockwise, and the velocity of the outer flow along the node
direction is -gamma (the body interior is at rest). The velocity anywhere is
u = d psi / dy, v = -d psi / dx, evaluated by automatic differentiation.

Several bodies (multi-element airfoils) are solved together, each with its
own psi_0 and Kutta condition.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

_TWO_PI = 2.0 * np.pi
_TINY = 1e-300


class Body(NamedTuple):
    """One closed wall contour prepared for the panel method.

    Attributes
    ----------
    nodes : ndarray, shape (n, 2)
        Panel nodes from the upper trailing-edge corner around the leading
        edge to the lower trailing-edge corner. At a sharp trailing edge the
        first and the last node coincide.
    blunt : bool
        Whether a base panel closes the contour from the last to the first
        node.
    """

    nodes: np.ndarray
    blunt: bool


class PanelSolution(NamedTuple):
    """Solved vortex sheet of one or more bodies.

    Attributes
    ----------
    bodies : tuple of Body
    gamma : tuple of ndarray
        Sheet strength at the nodes of each body.
    psi0 : ndarray
        Stream-function value on each body.
    velocity : ndarray, shape (2,)
        Freestream velocity.
    """

    bodies: tuple
    gamma: tuple
    psi0: np.ndarray
    velocity: np.ndarray

    def circulation(self, body=0):
        """Clockwise circulation of a body: integral of gamma over its sheet."""
        b, g = self.bodies[body], self.gamma[body]
        length = np.linalg.norm(np.diff(b.nodes, axis=0), axis=1)
        total = float(np.sum(0.5 * (g[:-1] + g[1:]) * length))
        if b.blunt:
            total += _base_strengths(b)[1] * (g[0] - g[-1]) * _base_length(b)
        return total

    def lift_coefficient(self, chord=1.0):
        """Kutta-Joukowski lift coefficient of all bodies, 2 Gamma / (U c)."""
        speed = float(np.hypot(*self.velocity))
        gamma = sum(self.circulation(k) for k in range(len(self.bodies)))
        return 2.0 * gamma / (speed * chord)


def contour_body(nodes, direction=(1.0, 0.0), corner_angle=45.0):
    """Prepare a closed wall contour: find the trailing edge and its base.

    Parameters
    ----------
    nodes : array_like, shape (n, 2)
        Nodes of the closed contour in order, without a repeated end node,
        in either direction and from any start.
    direction : (float, float)
        Flow direction; the trailing edge is the downstream end of the body.
    corner_angle : float
        Turning angle in degrees above which a node counts as a corner.

    Returns
    -------
    Body

    The two most downstream corners bound a blunt base if they are closer
    than 10 % of the body size; the nodes between them are dropped. With one
    corner it is a sharp trailing edge, and without any (a round trailing
    edge) the most downstream node is taken as a sharp trailing edge.
    """
    xy = np.asarray(nodes, dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 5:
        raise ValueError("a panel body needs at least 5 contour nodes")
    area = 0.5 * np.sum(
        xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]
    )
    if area < 0.0:  # make the loop counter-clockwise
        xy = xy[::-1]
    n = xy.shape[0]
    before = xy - np.roll(xy, 1, axis=0)
    after = np.roll(xy, -1, axis=0) - xy
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    dot = np.einsum("ij,ij->i", before, after)
    turn = np.degrees(np.abs(np.arctan2(cross, dot)))
    d = np.asarray(direction, dtype=float)
    along = xy @ (d / np.linalg.norm(d))
    size = float(np.max(np.linalg.norm(xy - xy.mean(axis=0), axis=1))) * 2.0
    corners = np.flatnonzero(turn > corner_angle)
    corners = corners[np.argsort(along[corners])[::-1][:2]]
    if (
        corners.size == 2
        and np.linalg.norm(xy[corners[0]] - xy[corners[1]]) < 0.1 * size
    ):
        a, b = int(corners[0]), int(corners[1])
        # Counter-clockwise the base runs from the lower to the upper corner
        # and is the shorter of the two arcs between them: make it a -> b.
        if (b - a) % n > (a - b) % n:
            a, b = b, a
        idx = (b + np.arange((a - b) % n + 1)) % n  # upper corner ... lower corner
        return Body(xy[idx], True)
    start = int(corners[0]) if corners.size else int(np.argmax(along))
    idx = (start + np.arange(n + 1)) % n
    return Body(xy[idx], False)


def _base_length(body):
    return float(np.linalg.norm(body.nodes[0] - body.nodes[-1]))


def _base_strengths(body):
    """Factors (c_sigma, c_gamma) of the base panel of a blunt trailing edge.

    The base panel from the last to the first node carries the constant
    source strength c_sigma (gamma_first - gamma_last) and vortex strength
    c_gamma (gamma_first - gamma_last): with the mean leaving speed
    q = (gamma_first - gamma_last) / 2 along the trailing-edge bisector s,
    the flux through the panel is q |s x t| and the velocity along the panel
    direction t is q (s . t), which a sheet with a body at rest behind it
    produces with the vortex strength -q (s . t).
    """
    x = body.nodes
    t = x[0] - x[-1]
    t = t / np.linalg.norm(t)
    upper = x[0] - x[1]
    lower = x[-1] - x[-2]
    s = upper / np.linalg.norm(upper) + lower / np.linalg.norm(lower)
    s = s / np.linalg.norm(s)
    return 0.5 * abs(s[0] * t[1] - s[1] * t[0]), -0.5 * float(s @ t)


def _local(points, a, b):
    """Panel coordinates of points (m, 2) for panels a -> b (n, 2).

    Returns x1, x2 (distances along the panel from its two ends), y (normal
    distance, positive to the left of a -> b) and the panel lengths, shapes
    (m, n) and (n,).
    """
    d = b - a
    length = jnp.sqrt(jnp.sum(d * d, axis=1))
    t = d / length[:, None]
    r = points[:, None, :] - a[None, :, :]
    x1 = r[..., 0] * t[None, :, 0] + r[..., 1] * t[None, :, 1]
    y = r[..., 1] * t[None, :, 0] - r[..., 0] * t[None, :, 1]
    return x1, x1 - length[None, :], y, length


def _vortex_integrals(x1, x2, y, length):
    """Integrals of ln r and of (s / L) ln r over a panel, s from its start."""
    r1, r2 = x1 * x1 + y * y, x2 * x2 + y * y
    ln1 = 0.5 * jnp.log(jnp.maximum(r1, _TINY))
    ln2 = 0.5 * jnp.log(jnp.maximum(r2, _TINY))
    ay = jnp.abs(y)
    # F(a) = a ln r - a + |y| atan(a / |y|), G(a) = r^2 ln r / 2 - a^2 / 4
    i0 = x1 * ln1 - x2 * ln2 - length + ay * (jnp.arctan2(x1, ay) - jnp.arctan2(x2, ay))
    g = 0.5 * (r1 * ln1 - r2 * ln2) - 0.25 * (x1 * x1 - x2 * x2)
    i1 = (x1 * i0 - g) / length
    return i0, i1


def _source_integral(x1, x2, y):
    """Integral of the polar angle over a panel (stream function of a source).

    The angle is measured from the panel direction; its branch cut runs from
    the panel start against the panel direction.
    """
    r1, r2 = x1 * x1 + y * y, x2 * x2 + y * y
    ln1 = 0.5 * jnp.log(jnp.maximum(r1, _TINY))
    ln2 = 0.5 * jnp.log(jnp.maximum(r2, _TINY))
    yp = y + 0.0  # -0.0 -> +0.0: a point on the cut takes the +pi side
    return x1 * jnp.arctan2(yp, x1) - x2 * jnp.arctan2(yp, x2) + y * (ln1 - ln2)


def _influence(points, body):
    """Matrix a (m, n): stream function at points per unit gamma at the nodes."""
    x = jnp.asarray(body.nodes)
    x1, x2, y, length = _local(points, x[:-1], x[1:])
    i0, i1 = _vortex_integrals(x1, x2, y, length[None, :])
    n = x.shape[0]
    a = jnp.zeros((points.shape[0], n))
    a = a.at[:, :-1].add(i0 - i1)
    a = a.at[:, 1:].add(i1)
    if body.blunt:
        c_sigma, c_gamma = _base_strengths(body)
        x1, x2, y, length = _local(points, x[-1:], x[:1])
        i0, _ = _vortex_integrals(x1, x2, y, length[None, :])
        base = (c_gamma * i0 + c_sigma * _source_integral(x1, x2, y))[:, 0]
        a = a.at[:, 0].add(base)
        a = a.at[:, -1].add(-base)
    return a / _TWO_PI


def solve(bodies, velocity):
    """Solve the potential flow around one or more bodies.

    Parameters
    ----------
    bodies : Body or sequence of Body
        From :func:`contour_body`.
    velocity : (float, float)
        Freestream velocity.

    Returns
    -------
    PanelSolution
    """
    if isinstance(bodies, Body):
        bodies = (bodies,)
    bodies = tuple(bodies)
    u = np.asarray(velocity, dtype=float)
    sizes = [b.nodes.shape[0] for b in bodies]
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    n_gamma = int(offsets[-1])
    n = n_gamma + len(bodies)
    matrix = np.zeros((n, n))
    rhs = np.zeros(n)
    for k, body in enumerate(bodies):
        rows = slice(offsets[k], offsets[k + 1])
        points = jnp.asarray(body.nodes)
        for j, other in enumerate(bodies):
            cols = slice(offsets[j], offsets[j + 1])
            matrix[rows, cols] = np.asarray(_influence(points, other))
        matrix[rows, n_gamma + k] = -1.0
        rhs[rows] = -(u[0] * body.nodes[:, 1] - u[1] * body.nodes[:, 0])
        first, last = offsets[k], offsets[k + 1] - 1
        if not body.blunt:
            # The last node repeats the first: extrapolate gamma to the
            # trailing edge from both sides instead.
            matrix[last, :] = 0.0
            matrix[last, [first, first + 1, first + 2]] = (1.0, -2.0, 1.0)
            matrix[last, [last, last - 1, last - 2]] = (-1.0, 2.0, -1.0)
            rhs[last] = 0.0
        kutta = n_gamma + k
        matrix[kutta, first] = matrix[kutta, last] = 1.0
    x = np.linalg.solve(matrix, rhs)
    gamma = tuple(x[offsets[k] : offsets[k + 1]] for k in range(len(bodies)))
    return PanelSolution(bodies, gamma, x[n_gamma:], u)


def _stream_function(point, solution):
    psi = solution.velocity[0] * point[1] - solution.velocity[1] * point[0]
    for body, gamma in zip(solution.bodies, solution.gamma, strict=True):
        psi = psi + (_influence(point[None, :], body) @ jnp.asarray(gamma))[0]
    return psi


def stream_function(solution, points):
    """Stream function at points (m, 2)."""
    f = jax.vmap(lambda p: _stream_function(p, solution))
    return np.asarray(f(jnp.asarray(points, dtype=float)))


def velocity(solution, points, chunk=20000):
    """Velocity (u, v) at points (m, 2) outside the bodies.

    Evaluated in chunks of ``chunk`` points: the work is points x panels.
    """
    grad = jax.jit(jax.vmap(jax.grad(lambda p: _stream_function(p, solution))))
    pts = np.asarray(points, dtype=float)
    out = np.empty_like(pts)
    for start in range(0, pts.shape[0], chunk):
        g = np.asarray(grad(jnp.asarray(pts[start : start + chunk])))
        out[start : start + chunk, 0] = g[:, 1]
        out[start : start + chunk, 1] = -g[:, 0]
    return out


def surface_speed(solution, body=0):
    """Signed outer-flow speed along the node direction at the nodes: -gamma."""
    return -solution.gamma[body]

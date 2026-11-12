"""Inviscid side of the coupled airfoil analysis.

Panel solution of the airfoil (linear vorticity, Kutta condition; see
:mod:`pyaeroflow.numerics.panel`), a wake traced along the inviscid trailing
streamline, and the linear map from the mass defect m = ue delta* of the
boundary layer and wake to the edge speeds (Drela 1989):

    v_airfoil = v_inv(alpha) + D_a m,      ue_wake = ue_inv(alpha) + D_w m.

At the airfoil nodes v is the signed outer-flow speed along the node
direction (upper trailing edge -> leading edge -> lower trailing edge), so
the signed mass defect v delta* gives the source strength dm/ds on both sides
of the stagnation point without any sign switch.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pyaeroflow.numerics import panel

from . import sources


class Inviscid(NamedTuple):
    """Geometry and inviscid operators of one airfoil and its wake.

    Attributes
    ----------
    body : panel.Body
    s : ndarray (n,)
        Arc length at the airfoil nodes from the upper trailing-edge node.
    wake : ndarray (nw, 2)
        Wake nodes, the first just behind the trailing-edge midpoint.
    wake_s : ndarray (nw,)
        Arc length along the wake from its first node.
    wake_t : ndarray (nw, 2)
        Unit tangents of the wake.
    v0, v90 : ndarray (n,)
        Signed airfoil speeds for unit freestream at 0 and 90 degrees.
    w0, w90 : ndarray (nw,)
        Wake speeds along the wake tangent for the same two freestreams.
    dmat : ndarray (n + nw, n + nw)
        Sensitivity of [v_airfoil, ue_wake] to [v delta* (airfoil), ue delta* (wake)].
    gap : float
        Trailing-edge gap (zero at a sharp trailing edge).
    wake_gap : ndarray (nw,)
        Thickness of the dead-air region behind a blunt trailing edge, part of
        the wake delta*.
    chord : float
    leading_edge : ndarray (2,)
    """

    body: panel.Body
    s: np.ndarray
    wake: np.ndarray
    wake_s: np.ndarray
    wake_t: np.ndarray
    v0: np.ndarray
    v90: np.ndarray
    w0: np.ndarray
    w90: np.ndarray
    dmat: np.ndarray
    gap: float
    wake_gap: np.ndarray
    chord: float
    leading_edge: np.ndarray

    def speeds(self, alpha_deg):
        """Inviscid airfoil (signed) and wake speeds at the angle of attack."""
        a = np.radians(alpha_deg)
        return (
            np.cos(a) * self.v0 + np.sin(a) * self.v90,
            np.cos(a) * self.w0 + np.sin(a) * self.w90,
        )


def _unit(v):
    return v / np.linalg.norm(v)


def _trailing_edge(body):
    x = body.nodes
    mid = 0.5 * (x[0] + x[-1])
    upper = _unit(x[0] - x[1])
    lower = _unit(x[-1] - x[-2])
    bisector = _unit(upper + lower)
    gap = float(np.linalg.norm(x[0] - x[-1])) if body.blunt else 0.0
    return mid, bisector, gap


def _wake_spacing(first, length, count):
    """Count - 1 geometric steps starting with ``first`` that sum to ``length``."""
    steps = count - 1
    if first * steps >= length:
        return np.full(steps, length / steps)
    lo, hi = 1.0, 2.0
    while first * (hi**steps - 1.0) / (hi - 1.0) < length:
        hi *= 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        if first * (r**steps - 1.0) / (r - 1.0) < length:
            lo = r
        else:
            hi = r
    return first * r ** np.arange(steps)


def trace_wake(solution, count, length):
    """Wake nodes along the inviscid streamline leaving the trailing edge."""
    body = solution.bodies[0]
    mid, bisector, _ = _trailing_edge(body)
    chord = float(np.max(np.linalg.norm(body.nodes - mid, axis=1)))
    x = body.nodes
    first = 0.5 * (np.linalg.norm(x[1] - x[0]) + np.linalg.norm(x[-1] - x[-2]))
    steps = _wake_spacing(first, length * chord, count)
    gamma = jnp.asarray(solution.gamma[0])
    u_inf = np.asarray(solution.velocity)

    @jax.jit
    def vel(p):
        g = jax.grad(lambda q: panel._influence(q[None, :], body)[0] @ gamma)(p)
        return jnp.stack((g[1], -g[0]))

    def velocity(p):
        return np.asarray(vel(jnp.asarray(p))) + u_inf

    pts = [mid + 1e-5 * chord * bisector]
    direction = bisector
    for k, ds in enumerate(steps):
        p = pts[-1]
        if k == 0:
            pts.append(p + ds * direction)
            continue
        v = velocity(p)
        half = p + 0.5 * ds * _unit(v)
        v = velocity(half)
        direction = _unit(v)
        pts.append(p + ds * direction)
    return np.array(pts)


def _tangents(pts, first):
    t = np.empty_like(pts)
    t[0] = first
    t[1:-1] = pts[2:] - pts[:-2]
    t[-1] = pts[-1] - pts[-2]
    return t / np.linalg.norm(t, axis=1)[:, None]


def _gap_profile(gap, wake_s, slope):
    """Dead-air thickness behind a blunt trailing edge, cubic decay over 2.5 gaps."""
    if gap <= 0.0:
        return np.zeros_like(wake_s)
    flen = 2.5
    slope = float(np.clip(slope, -3.0 / flen, 3.0 / flen))
    z = wake_s / (flen * gap)
    w = gap * (1.0 + (2.0 + flen * slope) * z) * (1.0 - z) ** 2
    return np.where(z < 1.0, w, 0.0)


def _check_strips(nodes):
    """Raise if a node lies in the outward strip of an airfoil panel."""
    a, b = jnp.asarray(nodes[:-1]), jnp.asarray(nodes[1:])
    x1, y, length, _ = sources._local(jnp.asarray(nodes), a, b)
    x1, y, length = np.asarray(x1), np.asarray(y), np.asarray(length)
    inside = (y < -1e-9 * length) & (x1 > 1e-9 * length) & (x1 < length * (1 - 1e-9))
    if np.any(inside):
        raise ValueError(
            "airfoil node in the outward strip of a panel: geometry too wavy"
        )


def build(nodes, alpha_deg=0.0, wake_length=1.0, wake_count=None):
    """Prepare the inviscid operators of an airfoil contour.

    Parameters
    ----------
    nodes : array_like (n, 2)
        Panel nodes of the closed contour (any start and direction; see
        :func:`pyaeroflow.numerics.panel.contour_body`).
    alpha_deg : float
        Angle of attack that sets the wake trajectory.
    wake_length : float
        Wake length in chords.
    wake_count : int, optional
        Number of wake nodes (default n/10 + 10 per chord of wake).
    """
    pts = np.asarray(nodes, dtype=float)
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-12:  # repeated sharp trailing edge
        pts = pts[:-1]
    body = panel.contour_body(pts)
    x = body.nodes
    n = x.shape[0]
    mid, bisector, gap = _trailing_edge(body)
    le = x[np.argmax(np.linalg.norm(x - mid, axis=1))]
    chord = float(np.linalg.norm(le - mid))
    _check_strips(x)

    sol0 = panel.solve(body, (1.0, 0.0))
    sol90 = panel.solve(body, (0.0, 1.0))
    nw = wake_count or int(np.ceil(n / 10 + 10 * wake_length))
    a = np.radians(alpha_deg)
    wake = trace_wake(panel.solve(body, (np.cos(a), np.sin(a))), nw, wake_length)
    wake_s = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(wake, axis=0), axis=1)))
    )
    wake_t = _tangents(wake, bisector)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))))

    def wake_speed(sol):
        return np.einsum("ij,ij->i", panel.velocity(sol, wake), wake_t)

    # Source parameters: sigma = dm/ds on every airfoil panel and every wake
    # panel. Airfoil panels carry them as constant sources. On the wake the
    # distribution is piecewise linear between the panel midpoints (value
    # sigma_j) and the nodes (mean of the two neighbouring panels; at the first
    # wake node the sum of the two trailing-edge panel values), constant ghost panel
    # beyond the last node.
    p_air, p = n - 1, n - 1 + nw - 1
    w = jnp.asarray(wake)
    pts = jnp.asarray(x)
    mids = 0.5 * (w[:-1] + w[1:])
    node_val = np.zeros((nw, p))
    for j in range(nw):
        left, right = j - 1, j
        if j == 0:
            # continuity through the trailing edge: sum of the two last panels
            node_val[j, 0] = node_val[j, n - 2] = 1.0
        elif j == nw - 1:
            node_val[j, p_air + left] = 1.0
        else:
            node_val[j, p_air + left] = node_val[j, p_air + right] = 0.5
    mid_val = np.zeros((nw - 1, p))
    mid_val[np.arange(nw - 1), p_air + np.arange(nw - 1)] = 1.0
    seg_a = jnp.concatenate((w[:-1], mids, w[-1:]))
    seg_b = jnp.concatenate((mids, w[1:], 2.0 * w[-1:] - mids[-1:]))
    val_a = np.vstack((node_val[:-1], mid_val, node_val[-1:]))
    val_b = np.vstack((mid_val, node_val[1:], node_val[-1:]))
    # In the airfoil boundary condition the first half panel starts from zero
    # strength (the trailing-edge panels themselves are next to it).
    val_a_psi = val_a.copy()
    val_a_psi[0] = 0.0

    # Stream function of the sources at the airfoil nodes.
    matrix, _, _ = panel.assemble((body,), (1.0, 0.0))
    smat = np.zeros((n + 1, p))
    smat[:n, :p_air] = np.asarray(sources.constant_psi(pts, pts[:-1], pts[1:]))
    lin = np.asarray(sources.linear_psi(pts, seg_a, seg_b))
    smat[:n] += lin[..., 0] @ val_a_psi + lin[..., 1] @ val_b
    if not body.blunt:
        smat[n - 1, :] = 0.0  # trailing-edge extrapolation row
    dgamma = -np.linalg.solve(matrix, smat)[:n]  # (n, p)

    # Wake velocities.
    t = wake_t
    vg = np.einsum("mnk,mk->mn", np.asarray(sources.vortex_velocity(w, body)), t)
    vsig = np.zeros((nw, p))
    vsig[:, :p_air] = np.einsum(
        "mnk,mk->mn", np.asarray(sources.constant_velocity(w, pts[:-1], pts[1:])), t
    )
    lv = np.einsum(
        "mnjk,mk->mnj", np.asarray(sources.linear_velocity(w, seg_a, seg_b)), t
    )
    vsig += lv[..., 0] @ val_a + lv[..., 1] @ val_b
    dw = vg @ dgamma + vsig

    # sigma from the nodal mass defect
    sw = np.concatenate((s, s[-1] + 1.0 + wake_s))  # wake block is separate
    gmat = np.zeros((p, n + nw))
    for k in range(n - 1):
        d = s[k + 1] - s[k]
        gmat[k, k], gmat[k, k + 1] = -1.0 / d, 1.0 / d
    for j in range(nw - 1):
        d = wake_s[j + 1] - wake_s[j]
        gmat[p_air + j, n + j], gmat[p_air + j, n + j + 1] = -1.0 / d, 1.0 / d
    del sw
    dmat = np.vstack((-dgamma, dw)) @ gmat

    slope_u = (x[0, 1] - x[1, 1]) / (x[0, 0] - x[1, 0]) if x[0, 0] != x[1, 0] else 0.0
    slope_l = (
        (x[-1, 1] - x[-2, 1]) / (x[-1, 0] - x[-2, 0]) if x[-1, 0] != x[-2, 0] else 0.0
    )
    wake_gap = _gap_profile(gap, wake_s, slope_u - slope_l)

    return Inviscid(
        body=body,
        s=s,
        wake=wake,
        wake_s=wake_s,
        wake_t=wake_t,
        v0=-sol0.gamma[0],
        v90=-sol90.gamma[0],
        w0=wake_speed(sol0),
        w90=wake_speed(sol90),
        dmat=dmat,
        gap=gap,
        wake_gap=wake_gap,
        chord=chord,
        leading_edge=le,
    )

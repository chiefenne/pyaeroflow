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


_TRACERS = {}


def _tracer(bodies):
    """Compiled streamline march for one geometry (gamma and freestream are inputs)."""
    grad = panel._velocity_kernel(tuple(bodies))
    tracer = _TRACERS.get(grad)
    if tracer is None:

        def velocity(p, gamma, vinf):
            g = grad(p[None, :], gamma, vinf)[0]
            return jnp.stack((g[1], -g[0]))

        def unit(v):
            return v / jnp.linalg.norm(v)

        def march(start, steps, gamma, vinf):
            def step(p, ds):
                half = p + 0.5 * ds * unit(velocity(p, gamma, vinf))
                q = p + ds * unit(velocity(half, gamma, vinf))
                return q, q

            return jax.lax.scan(step, start, steps)[1]

        tracer = _TRACERS[grad] = jax.jit(march)
        if len(_TRACERS) > 16:
            _TRACERS.clear()
            _TRACERS[grad] = tracer
    return tracer


def trace_wake(solution, count, length):
    """Wake nodes along the inviscid streamline leaving the trailing edge.

    Midpoint rule along the local velocity direction, the first step along
    the trailing-edge bisector; compiled once per geometry.
    """
    body = solution.bodies[0]
    mid, bisector, _ = _trailing_edge(body)
    chord = float(np.max(np.linalg.norm(body.nodes - mid, axis=1)))
    x = body.nodes
    first = 0.5 * (np.linalg.norm(x[1] - x[0]) + np.linalg.norm(x[-1] - x[-2]))
    steps = _wake_spacing(first, length * chord, count)
    p0 = mid + 1e-5 * chord * bisector
    p1 = p0 + steps[0] * bisector
    rest = _tracer(solution.bodies)(
        jnp.asarray(p1),
        jnp.asarray(steps[1:]),
        tuple(jnp.asarray(g, dtype=float) for g in solution.gamma),
        jnp.asarray(solution.velocity, dtype=float),
    )
    return np.vstack((p0, p1, np.asarray(rest)))


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


def _wake_segments(w, n):
    """Linear source segments of the wake and their strengths.

    The distribution is piecewise linear between the wake panel midpoints
    (value sigma_j) and the nodes (mean of the two neighbouring panels; at
    the first wake node the sum of the two trailing-edge panel values), with
    a constant ghost panel beyond the last node. Returns the segment ends
    (k, 2) and the maps (k, p) from the source parameters to the strength at
    the start and the end of every segment.
    """
    nw = w.shape[0]
    p_air, p = n - 1, n - 1 + nw - 1
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
    return seg_a, seg_b, val_a, val_b


def _sigma_matrix(s, wake_s):
    """Map (p, n + nw) from the nodal mass defect to the source parameters."""
    n, nw = s.size, wake_s.size
    p_air, p = n - 1, n - 1 + nw - 1
    gmat = np.zeros((p, n + nw))
    for k in range(n - 1):
        d = s[k + 1] - s[k]
        gmat[k, k], gmat[k, k + 1] = -1.0 / d, 1.0 / d
    for j in range(nw - 1):
        d = wake_s[j + 1] - wake_s[j]
        gmat[p_air + j, n + j], gmat[p_air + j, n + j + 1] = -1.0 / d, 1.0 / d
    return gmat


def displacement_velocity(inv, mass, points, chunk=4000):
    """Velocity (m, 2) of the source sheet of a mass defect at field points.

    ``mass`` (n + nw,) is the nodal mass defect (signed speed times delta* on
    the airfoil, ue delta* in the wake), as in :attr:`Inviscid.dmat`. The
    change of the airfoil vorticity it causes is not included: the caller
    evaluates the vortex sheet of the viscous solution separately.
    """
    x = jnp.asarray(inv.body.nodes)
    n = x.shape[0]
    sigma = _sigma_matrix(inv.s, inv.wake_s) @ np.asarray(mass, dtype=float)
    seg_a, seg_b, val_a, val_b = _wake_segments(jnp.asarray(inv.wake), n)
    sig_air = jnp.asarray(sigma[: n - 1])
    sa, sb = jnp.asarray(val_a @ sigma), jnp.asarray(val_b @ sigma)
    pts = np.asarray(points, dtype=float)
    out = np.empty_like(pts)
    for start in range(0, pts.shape[0], chunk):
        q = jnp.asarray(pts[start : start + chunk])
        cv = sources.constant_velocity(q, x[:-1], x[1:])
        v = jnp.einsum("mnk,n->mk", cv, sig_air)
        lv = sources.linear_velocity(q, seg_a, seg_b)
        v = v + jnp.einsum("mnk,n->mk", lv[..., 0, :], sa)
        v = v + jnp.einsum("mnk,n->mk", lv[..., 1, :], sb)
        out[start : start + chunk] = np.asarray(v)
    return out


def _wake_influences(w, body):
    """Influences of the airfoil and wake singularities that involve the wake.

    Returns the source stream function at the airfoil nodes per wake segment
    end value, and at the wake nodes the velocities of the airfoil sources,
    the wake sources and the airfoil vortex sheet.
    """
    x = jnp.asarray(body.nodes)
    seg_a, seg_b, _, _ = _wake_segments(w, x.shape[0])
    return (
        sources.linear_psi(x, seg_a, seg_b),
        sources.constant_velocity(w, x[:-1], x[1:]),
        sources.linear_velocity(w, seg_a, seg_b),
        sources.vortex_velocity(w, body),
    )


class _Airfoil(NamedTuple):
    """Operators of an airfoil that do not depend on the angle of attack."""

    body: panel.Body
    s: np.ndarray
    bisector: np.ndarray
    gap: float
    chord: float
    leading_edge: np.ndarray
    sol0: panel.PanelSolution
    sol90: panel.PanelSolution
    lu: tuple  # LU factors of the panel matrix
    smat_air: np.ndarray  # (n + 1, n - 1) stream function of the airfoil sources
    slope: float  # trailing-edge wedge slope, upper minus lower
    wake_kernels: object  # compiled source and vortex influences at wake nodes


_AIRFOILS = {}


def _airfoil(pts):
    """Per-airfoil part of :func:`build`, kept for the airfoils of a polar.

    The airfoil solution at any angle is the superposition cos(alpha) sol0 +
    sin(alpha) sol90; the panel matrix is factored once.
    """
    key = (pts.shape, pts.tobytes())
    hit = _AIRFOILS.get(key)
    if hit is not None:
        return hit
    import scipy.linalg

    body = panel.contour_body(pts)
    x = body.nodes
    n = x.shape[0]
    mid, bisector, gap = _trailing_edge(body)
    le = x[np.argmax(np.linalg.norm(x - mid, axis=1))]
    chord = float(np.linalg.norm(le - mid))
    _check_strips(x)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(x, axis=0), axis=1))))
    matrix, _, _ = panel.assemble((body,), (1.0, 0.0))
    xj = jnp.asarray(x)
    smat_air = np.zeros((n + 1, n - 1))
    smat_air[:n] = np.asarray(sources.constant_psi(xj, xj[:-1], xj[1:]))
    slope_u = (x[0, 1] - x[1, 1]) / (x[0, 0] - x[1, 0]) if x[0, 0] != x[1, 0] else 0.0
    slope_l = (
        (x[-1, 1] - x[-2, 1]) / (x[-1, 0] - x[-2, 0]) if x[-1, 0] != x[-2, 0] else 0.0
    )
    foil = _Airfoil(
        body=body,
        s=s,
        bisector=bisector,
        gap=gap,
        chord=chord,
        leading_edge=le,
        sol0=panel.solve(body, (1.0, 0.0)),
        sol90=panel.solve(body, (0.0, 1.0)),
        lu=scipy.linalg.lu_factor(matrix),
        smat_air=smat_air,
        slope=slope_u - slope_l,
        wake_kernels=jax.jit(lambda w: _wake_influences(w, body)),
    )
    if len(_AIRFOILS) >= 8:
        _AIRFOILS.clear()
    _AIRFOILS[key] = foil
    return foil


def _superpose(foil, alpha_deg):
    """Panel solution at alpha from the 0 and 90 degree solutions."""
    a = np.radians(alpha_deg)
    ca, sa = np.cos(a), np.sin(a)
    s0, s90 = foil.sol0, foil.sol90
    return panel.PanelSolution(
        s0.bodies,
        tuple(ca * g0 + sa * g9 for g0, g9 in zip(s0.gamma, s90.gamma, strict=True)),
        ca * s0.psi0 + sa * s90.psi0,
        np.array([ca, sa]),
    )


def build(nodes, alpha_deg=0.0, wake_length=1.0, wake_count=None):
    """Prepare the inviscid operators of an airfoil contour.

    The airfoil part (panel matrix and its factors, the 0 and 90 degree
    solutions, the source stream function of the airfoil panels) is computed
    once per contour; only the wake, which follows the trailing streamline
    at alpha, and its couplings are built per call.

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
    import scipy.linalg

    pts = np.asarray(nodes, dtype=float)
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-12:  # repeated sharp trailing edge
        pts = pts[:-1]
    foil = _airfoil(pts)
    body = foil.body
    x = body.nodes
    n = x.shape[0]
    sol0, sol90 = foil.sol0, foil.sol90
    nw = wake_count or int(np.ceil(n / 10 + 10 * wake_length))
    wake = trace_wake(_superpose(foil, alpha_deg), nw, wake_length)
    wake_s = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(wake, axis=0), axis=1)))
    )
    wake_t = _tangents(wake, foil.bisector)
    s = foil.s

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
    seg_a, seg_b, val_a, val_b = _wake_segments(w, n)
    # In the airfoil boundary condition the first half panel starts from zero
    # strength (the trailing-edge panels themselves are next to it).
    val_a_psi = val_a.copy()
    val_a_psi[0] = 0.0

    # Stream function of the sources at the airfoil nodes.
    smat = np.zeros((n + 1, p))
    smat[:, :p_air] = foil.smat_air
    lin, cv, lv, vv = (np.asarray(t) for t in foil.wake_kernels(w))
    smat[:n] += lin[..., 0] @ val_a_psi + lin[..., 1] @ val_b
    if not body.blunt:
        smat[n - 1, :] = 0.0  # trailing-edge extrapolation row
    dgamma = -scipy.linalg.lu_solve(foil.lu, smat)[:n]  # (n, p)

    # Wake velocities.
    t = wake_t
    vg = np.einsum("mnk,mk->mn", vv, t)
    vsig = np.zeros((nw, p))
    vsig[:, :p_air] = np.einsum("mnk,mk->mn", cv, t)
    lv = np.einsum("mnjk,mk->mnj", lv, t)
    vsig += lv[..., 0] @ val_a + lv[..., 1] @ val_b
    dw = vg @ dgamma + vsig

    # sigma from the nodal mass defect
    gmat = _sigma_matrix(s, wake_s)
    dmat = np.vstack((-dgamma, dw)) @ gmat

    wake_gap = _gap_profile(foil.gap, wake_s, foil.slope)

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
        gap=foil.gap,
        wake_gap=wake_gap,
        chord=foil.chord,
        leading_edge=foil.leading_edge,
    )

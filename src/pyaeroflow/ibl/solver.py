"""Coupled viscous-inviscid airfoil analysis solved by Newton's method.

Unknowns at every airfoil and wake node: theta, delta*, s (amplification
factor n or sqrt(C_tau)) and the edge speed (signed speed v on the airfoil,
ue in the wake). Equations at every node: three boundary-layer equations
(similarity at the first node of each side, the interval equations of
:mod:`.residuals` elsewhere, the trailing-edge merge at the first wake node)
and the edge-speed equation of :mod:`.inviscid`. The Jacobian of the whole
system comes from ``jax.jacfwd``, including the dependence of the arc length
xi on the stagnation-point location.

The laminar / turbulent flags and the stagnation-point panel are discrete;
they are updated after every Newton step (transition from a march of the
amplification equation on the current state).
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from . import closures as cl
from . import inviscid as inv_mod
from . import residuals as res

jax.config.update("jax_enable_x64", True)

LAM, TURB, TRAN, WAKE, SIMI, TE = range(6)


@dataclass
class IblResult:
    """Result of :func:`analyse`.

    Coefficients refer to the chord and the freestream speed; distributions
    are per node (airfoil nodes from the upper trailing edge round the
    leading edge, then the wake).
    """

    alpha: float
    reynolds: float
    cl: float
    cd: float
    cdf: float
    cdp: float
    cm: float
    converged: bool
    iterations: int
    residual: float
    x_transition: tuple  # (upper, lower) x/c, nan if none on the airfoil
    nodes: np.ndarray
    wake: np.ndarray
    theta: np.ndarray
    delta_star: np.ndarray
    ue: np.ndarray
    cp: np.ndarray
    cf: np.ndarray
    turbulent: np.ndarray
    history: list = field(default_factory=list)
    state: np.ndarray = None  # Newton unknowns (N, 4); airfoil speeds signed
    stagnation: int = 0  # stagnation-point panel: between nodes a and a + 1
    inviscid: object = None  # the :class:`.inviscid.Inviscid` operators


def _local(u1, u2, x1, x2, g1, g2, kind, nu, nc):
    """Boundary-layer residuals (3,) of one node of any kind."""
    is_tran = kind == TRAN
    r_lam = res.interval(u1, u2, x1, x2, g1, g2, nu, False, False, nc)
    r_turb = res.interval(u1, u2, x1, x2, g1, g2, nu, True, False, nc)
    r_wake = res.interval(u1, u2, x1, x2, g1, g2, nu, True, True, nc)
    # keep the transition branch finite where it is not used
    u2t = jnp.where(is_tran, u2, u1 * jnp.array([1.1, 1.2, 1.0, 1.0]))
    x2t = jnp.where(is_tran, x2, x1 * 1.1 + 1e-6)
    r_tran, _ = res.transition_interval(u1, u2t, x1, x2t, nu, nc)
    # SIMI: u1, x1 hold the next station downstream
    x1s = jnp.where(kind == SIMI, x1, x2 * 2.0 + 1e-6)
    r_simi = res.stagnation(u2, x2, u1, x1s, nu)
    return jnp.select(
        [kind == LAM, kind == TURB, is_tran, kind == WAKE, kind == SIMI],
        [r_lam, r_turb, r_tran, r_wake, r_simi],
        jnp.zeros(3),
    )


def _unpack(U, sgn, a, s, wake_s):
    n = s.shape[0]
    th, ds, sa, q = U[:, 0], U[:, 1], U[:, 2], U[:, 3]
    v = q[:n]
    va, vb = v[a], v[a + 1]
    sst = s[a] + (s[a + 1] - s[a]) * (-va) / (vb - va)
    xi_air = sgn * (s - sst)
    xi = jnp.concatenate((xi_air, xi_air[n - 1] + wake_s))
    ue = jnp.concatenate((sgn * v, q[n:]))
    return th, ds, sa, q, ue, xi


def _te(uu, ul, uw, tu, tl, hte, nu):
    """Trailing-edge merge of the two sides into the first wake node."""
    cu = jnp.where(tu, uu[2], cl.ctau_transition(uu[0], uu[1], uu[3], nu))
    cl_ = jnp.where(tl, ul[2], cl.ctau_transition(ul[0], ul[1], ul[3], nu))
    return jnp.stack(
        (
            uw[0] - (uu[0] + ul[0]),
            uw[1] - (uu[1] + ul[1] + hte),
            uw[2] - (cu * uu[0] + cl_ * ul[0]) / (uu[0] + ul[0]),
        )
    )


def _residual(U, kind, prev, sgn, a, turbf, vinv, winv, geo):
    s, wake_s, gap, dmat, hte, nu, nc = geo
    n = s.shape[0]
    th, ds, sa, q, ue, xi = _unpack(U, sgn, a, s, wake_s)
    st = jnp.stack((th, ds, sa, ue), axis=1)
    r = jax.vmap(_local, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))(
        st[prev], st, xi[prev], xi, gap[prev], gap, kind, nu, nc
    )
    r = r.at[n].set(_te(st[0], st[n - 1], st[n], turbf[0], turbf[n - 1], hte, nu))
    r_ue = q - (jnp.concatenate((vinv, winv)) + dmat @ (q * ds))
    r_ue = r_ue.at[n].set(q[n] - 0.5 * (q[n - 1] - q[0]))
    return jnp.concatenate((r, r_ue[:, None]), axis=1).ravel()


# Compiled once per process (per problem size), shared by all analyses.
_residual_jit = jax.jit(_residual)
_unpack_jit = jax.jit(_unpack)
_local_jac_jit = jax.jit(
    jax.vmap(
        jax.jacfwd(_local, argnums=(0, 1, 2, 3)),
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
    )
)
_te_jac_jit = jax.jit(jax.jacfwd(_te, argnums=(0, 1, 2)))


def _station(kind):
    def f(u1, u2, x1, x2, g1, g2, nu, nc):
        if kind == SIMI:
            return res.similarity(u2, x2, nu)
        if kind == TRAN:
            return res.transition_interval(u1, u2, x1, x2, nu, nc)[0]
        return res.interval(u1, u2, x1, x2, g1, g2, nu, kind != LAM, kind == WAKE, nc)

    return jax.jit(f), jax.jit(jax.jacfwd(f, argnums=1))


_STATION = {k: _station(k) for k in (LAM, TURB, TRAN, WAKE, SIMI)}


class _System:
    """Discrete system of one airfoil at one angle of attack."""

    def __init__(self, inv, reynolds, ncrit):
        self.inv = inv
        self.n = n = inv.s.size
        self.nw = inv.wake_s.size
        self.N = n + self.nw
        self.nu = 1.0 / reynolds
        self.ncrit = float(ncrit)
        gap = np.concatenate((np.zeros(n), inv.wake_gap))
        self.geo = (
            jnp.asarray(inv.s),
            jnp.asarray(inv.wake_s),
            jnp.asarray(gap),
            jnp.asarray(inv.dmat),
            float(inv.gap),
            self.nu,
            self.ncrit,
        )
        self._dmat = np.asarray(inv.dmat)
        self._gap = gap
        self._s = np.asarray(inv.s)

    def _residual(self, U, kind, prev, sgn, a, turbf, vinv, winv):
        return _residual_jit(U, kind, prev, sgn, a, turbf, vinv, winv, self.geo)

    def _unpack(self, U, sgn, a):
        return _unpack_jit(U, sgn, a, self.geo[0], self.geo[1])

    def local_fn(self, kind):
        f, jac = _STATION[kind]
        nu, nc = self.nu, self.ncrit
        return (
            lambda u1, u2, x1, x2, g1, g2: f(u1, u2, x1, x2, g1, g2, nu, nc),
            lambda u1, u2, x1, x2, g1, g2: jac(u1, u2, x1, x2, g1, g2, nu, nc),
        )

    def jacobian(self, U, kind, prev, sgn, a, turb):
        """Jacobian of the residual, assembled from local blocks.

        The boundary-layer rows couple a node to its upstream neighbour and,
        through the arc length xi, to the two speeds that locate the
        stagnation point; the edge-speed rows are dense (mass-defect map).
        """
        n, N = self.n, self.N
        th, ds, sa, q, ue, xi = (
            np.asarray(t) for t in self._unpack(jnp.asarray(U), jnp.asarray(sgn), a)
        )
        st = np.stack((th, ds, sa, ue), axis=1)
        j1, j2, jx1, jx2 = (
            np.asarray(t)
            for t in _local_jac_jit(
                jnp.asarray(st[prev]),
                jnp.asarray(st),
                jnp.asarray(xi[prev]),
                jnp.asarray(xi),
                jnp.asarray(self._gap[prev]),
                jnp.asarray(self._gap),
                jnp.asarray(kind),
                self.nu,
                self.ncrit,
            )
        )
        c = np.concatenate((sgn, np.ones(N - n)))  # d ue / d q
        scale = np.ones((N, 4))
        scale[:, 3] = c
        J = np.zeros((N, 4, N, 4))
        idx = np.arange(N)
        J[idx, :3, idx, :] += j2 * scale[:, None, :]
        J[idx, :3, prev, :] += j1 * scale[prev][:, None, :]
        # stagnation point: xi_i = sgn_i (s_i - s_st), wake xi = xi_{n-1} + s_w
        dxi = np.concatenate((-sgn, -np.ones(N - n)))
        dr = jx1 * dxi[prev][:, None] + jx2 * dxi[:, None]  # (N, 3) d r / d s_st
        va, vb = q[a], q[a + 1]
        dsa = self._s[a + 1] - self._s[a]
        J[:, :3, a, 3] += dr * (-dsa * vb / (vb - va) ** 2)
        J[:, :3, a + 1, 3] += dr * (dsa * va / (vb - va) ** 2)
        # trailing-edge merge
        ju, jl, jw = (
            np.asarray(t)
            for t in _te_jac_jit(
                jnp.asarray(st[0]),
                jnp.asarray(st[n - 1]),
                jnp.asarray(st[n]),
                bool(turb[0]),
                bool(turb[n - 1]),
                self.geo[4],
                self.nu,
            )
        )
        J[n, :3] = 0.0
        J[n, :3, 0] = ju * scale[0]
        J[n, :3, n - 1] = jl * scale[n - 1]
        J[n, :3, n] = jw * scale[n]
        # edge speeds
        J[:, 3, :, 3] = np.eye(N) - self._dmat * ds[None, :]
        J[:, 3, :, 1] = -self._dmat * q[None, :]
        J[n, 3] = 0.0
        J[n, 3, n, 3] = 1.0
        J[n, 3, n - 1, 3] = -0.5
        J[n, 3, 0, 3] = 0.5
        return J.reshape(4 * N, 4 * N)

    # ----------------------------------------------------------- structure
    def kinds(self, a, turb):
        n, N = self.n, self.N
        kind = np.empty(N, dtype=np.int32)
        prev = np.arange(N, dtype=np.int32)

        def interval_kind(i, p):
            if turb[i] and turb[p]:
                return TURB
            return TRAN if turb[i] else LAM

        for j in range(a + 1):
            if j == a:
                kind[j] = SIMI
                prev[j] = j - 1  # next station downstream
            else:
                prev[j] = j + 1
                kind[j] = interval_kind(j, j + 1)
        for j in range(a + 1, n):
            if j == a + 1:
                kind[j] = SIMI
                prev[j] = j + 1  # next station downstream
            else:
                prev[j] = j - 1
                kind[j] = interval_kind(j, j - 1)
        kind[n] = TE
        for j in range(n + 1, N):
            prev[j] = j - 1
            kind[j] = WAKE
        sgn = np.where(np.arange(n) <= a, -1.0, 1.0)
        return kind, prev, sgn

    def surfaces(self, a):
        """Node indices in marching order: upper, lower, wake."""
        n, N = self.n, self.N
        return (
            list(range(a, -1, -1)),
            list(range(a + 1, n)),
            list(range(n, N)),
        )


def _hk(u, gap):
    return (u[1] - gap) / u[0]


def _solve_station(system, kind, u1, u2, x1, x2, g1, g2, hmax):
    """Newton solve of one station: theta, delta*, s (direct) or with ue (inverse)."""
    f, jac = system.local_fn(kind)
    turb = kind in (TURB, TRAN, WAKE)
    direct = True
    target = None
    for it in range(40):
        r = np.asarray(f(u1, u2, x1, x2, g1, g2))
        if direct and np.linalg.norm(r) < 1e-10:
            return u2, True
        j = np.asarray(jac(u1, u2, x1, x2, g1, g2))
        if direct:
            du = np.append(np.linalg.solve(j[:, :3], -r), 0.0)
        else:
            hk = _hk(u2, g2)
            dh = np.array([-hk / u2[0], 1.0 / u2[0], 0.0, 0.0])
            a = np.vstack((j, dh))
            rr = np.append(r, hk - target)
            if np.linalg.norm(rr) < 1e-10:
                return u2, True
            du = np.linalg.solve(a, -rr)
        dm = max(abs(du[0] / u1[0]), abs(du[1] / u1[1]))
        if not direct:
            dm = max(dm, abs(du[3] / u1[3]))
        if turb:
            dm = max(dm, abs(du[2] / max(u1[2], 1e-6)))
        elif direct:
            dm = max(dm, abs(du[2]) / 10.0)
        omega = 0.3 / dm if dm > 0.3 else 1.0
        new = u2 + omega * du
        if turb:
            new[2] = min(max(new[2], 1e-7), 0.3)
        new[0] = max(new[0], 1e-3 * u1[0])
        new[1] = max(new[1], g2 + 1.0005 * new[0])
        if direct and (_hk(new, g2) > hmax or it > 12):
            direct = False
            hk1 = _hk(u1, g1)
            hkr = (x2 - x1) / u1[0]
            if kind == WAKE:
                h2 = hk1
                for _ in range(6):
                    h2 -= (h2 + 0.03 * hkr * (h2 - 1.0) ** 3 - hk1) / (
                        1.0 + 0.09 * hkr * (h2 - 1.0) ** 2
                    )
                target = max(h2, 1.01)
            elif kind == TURB:
                target = max(hk1 - 0.15 * hkr, hmax)
            else:
                target = max(hk1 + 0.03 * hkr, hmax)
            if it > 12:
                u2 = u1.copy()
                u2[3] = new[3]
                continue
        else:
            u2 = new
    return u2, False


def _initial_state(system, vinv, winv):
    """March the boundary layer and wake on the inviscid edge speeds."""
    inv, n, N = system.inv, system.n, system.N
    a = int(np.flatnonzero((vinv[:-1] < 0.0) & (vinv[1:] >= 0.0))[0])
    sst = inv.s[a] + (inv.s[a + 1] - inv.s[a]) * (-vinv[a]) / (vinv[a + 1] - vinv[a])
    sgn = np.where(np.arange(n) <= a, -1.0, 1.0)
    xi_air = sgn * (inv.s - sst)
    xi = np.concatenate((xi_air, xi_air[-1] + inv.wake_s))
    ue = np.concatenate((np.maximum(sgn * vinv, 1e-8), winv))
    gap = np.concatenate((np.zeros(n), inv.wake_gap))
    U = np.zeros((N, 4))
    turb = np.zeros(N, dtype=bool)
    nu = system.nu
    upper, lower, wake = system.surfaces(a)
    for nodes in (upper, lower):
        i0, i1 = nodes[0], nodes[1]
        x0, x1 = xi[i0], xi[i1]
        k = ue[i0] * x1 / (x0 * (x1 - x0)) - ue[i1] * x0 / (x1 * (x1 - x0))
        th = np.sqrt(0.45 * nu / (6.0 * k))
        xst = 1e-6
        u = np.array([th, 2.2 * th, 0.0, k * xst])
        u, _ = _solve_station(system, SIMI, u, u, xst, xst, 0.0, 0.0, 3.8)
        U[i0] = u
        U[i0, 3] = ue[i0]
        tran = False
        m = 1
        while m < len(nodes):
            i, p = nodes[m], nodes[m - 1]
            kind = TRAN if tran else (TURB if turb[p] else LAM)
            u2 = U[p].copy()
            u2[3] = ue[i]
            if tran:
                u2[2] = float(cl.ctau_transition(u2[0], u2[1], u2[3], nu))
            hmax = 2.5 if kind == TURB else 3.8
            u2, ok = _solve_station(
                system, kind, U[p], u2, xi[p], xi[i], 0.0, 0.0, hmax
            )
            if not ok:
                u2 = U[p].copy()
                u2[3] = ue[i]
                u2[0] *= np.sqrt(xi[i] / xi[p])
                u2[1] *= np.sqrt(xi[i] / xi[p])
            if kind == LAM and u2[2] > system.ncrit:
                tran = True
                continue
            U[i] = u2
            turb[i] = kind in (TURB, TRAN)
            tran = False
            m += 1
    # wake
    uu, ul = U[0], U[n - 1]
    cu = uu[2] if turb[0] else float(cl.ctau_transition(uu[0], uu[1], uu[3], nu))
    cw = ul[2] if turb[n - 1] else float(cl.ctau_transition(ul[0], ul[1], ul[3], nu))
    U[n] = [
        uu[0] + ul[0],
        uu[1] + ul[1] + inv.gap,
        (cu * uu[0] + cw * ul[0]) / (uu[0] + ul[0]),
        0.5 * (uu[3] + ul[3]),
    ]
    turb[n:] = True
    for m in range(1, len(wake)):
        i, p = wake[m], wake[m - 1]
        u2 = U[p].copy()
        u2[3] = ue[i]
        u2, ok = _solve_station(
            system, WAKE, U[p], u2, xi[p], xi[i], gap[p], gap[i], 2.5
        )
        if not ok:
            u2 = U[p].copy()
            u2[3] = ue[i]
        U[i] = u2
    # signed speeds on the airfoil
    U[:n, 3] *= sgn
    return U, a, turb


def _march_amplification(system, U, xi, nodes):
    """N along one side on the current state; index of the first node above n_crit."""
    nu, nc = system.nu, system.ncrit
    ncur = 0.0
    out = {nodes[0]: 0.0}
    for m in range(1, len(nodes)):
        i, p = nodes[m], nodes[m - 1]
        th1, ds1, ue1 = U[p, 0], U[p, 1], abs(U[p, 3])
        th2, ds2, ue2 = U[i, 0], U[i, 1], abs(U[i, 3])
        r1 = float(cl.amplification_rate(th1, ds1, ncur, ue1, nu, nc))
        n2 = ncur
        for _ in range(4):
            r2 = float(cl.amplification_rate(th2, ds2, n2, ue2, nu, nc))
            n2 = ncur + 0.5 * (r1 + r2) * (xi[i] - xi[p])
        if n2 > nc:
            return m, out
        out[i] = n2
        ncur = n2
    return None, out


def _update_transition(system, U, a, turb, xi, step_limit=True):
    changed = False
    nu = system.nu
    for nodes in system.surfaces(a)[:2]:
        first_turb = next((m for m, i in enumerate(nodes) if turb[i]), None)
        m_new, amp = _march_amplification(system, U, xi, nodes)
        if m_new == first_turb:
            continue
        changed = True
        old = first_turb if first_turb is not None else len(nodes)
        new = m_new if m_new is not None else len(nodes)
        # move by at most one node per Newton step: a node switched to
        # laminar keeps its turbulent profile for a step and would let the
        # transition point run away downstream
        new = old + int(np.sign(new - old)) if step_limit else new
        if new == old:
            continue
        if new < old:
            for m in range(new, old):
                i = nodes[m]
                turb[i] = True
                U[i, 2] = float(cl.ctau_transition(U[i, 0], U[i, 1], abs(U[i, 3]), nu))
        else:
            for m in range(old, new):
                i = nodes[m]
                turb[i] = False
                U[i, 2] = amp.get(i, system.ncrit)
    return changed


def _update_stagnation(system, U, a, turb):
    v = U[: system.n, 3]
    if v[a] < 0.0 < v[a + 1]:
        return a
    cand = np.flatnonzero((v[:-1] < 0.0) & (v[1:] >= 0.0))
    if cand.size == 0:
        raise RuntimeError("stagnation point lost")
    new = int(cand[np.argmin(np.abs(cand - a))])
    lo, hi = sorted((a, new))
    for i in range(lo, hi + 2):
        if turb[i]:
            turb[i] = False
            U[i, 2] = 0.0
    return new


def _limit(system, U, dU, turb):
    """Under-relaxation factor of a Newton step and the limit that set it."""
    limits = [(1.0, "none")]
    for k, name in ((0, "theta"), (1, "delta*")):
        f = np.min(dU[:, k] / U[:, k])
        if f < -0.5:
            limits.append((0.5 / abs(f), name))
    lam = ~turb
    if np.any(lam):
        dmax = np.max(np.abs(dU[lam, 2]))
        if dmax > 2.0:
            limits.append(
                (
                    2.0 / dmax,
                    f"n at {int(np.flatnonzero(lam)[np.argmax(np.abs(dU[lam, 2]))])}",
                )
            )
    if np.any(turb):
        dmax = np.max(np.abs(dU[turb, 2]))
        if dmax > 0.05:
            limits.append(
                (
                    0.05 / dmax,
                    f"ctau at {int(np.flatnonzero(turb)[np.argmax(np.abs(dU[turb, 2]))])}",  # noqa: E501
                )
            )
        ctmax = np.max(U[turb, 2])
        big = turb & (U[:, 2] >= 0.1 * ctmax) & (U[:, 2] + dU[:, 2] < 0.0)
        if np.any(big):
            limits.append(
                (float(np.min(0.8 * np.abs(U[big, 2] / dU[big, 2]))), "ctau sign")
            )
    dmax = np.max(np.abs(dU[:, 3]))
    if dmax > 0.2:
        limits.append((0.2 / dmax, f"ue at {int(np.argmax(np.abs(dU[:, 3])))}"))
    return min(limits)


def _forces(inv, U, alpha_deg, system, a, turb):
    n = system.n
    x = inv.body.nodes
    v = U[:n, 3]
    cp = 1.0 - v**2
    ca, sa = np.cos(np.radians(alpha_deg)), np.sin(np.radians(alpha_deg))
    xref = inv.leading_edge + 0.25 * (0.5 * (x[0] + x[-1]) - inv.leading_edge)
    pts = x if not inv.body.blunt else np.vstack((x, x[:1]))
    cps = cp if not inv.body.blunt else np.append(cp, cp[0])
    d = np.diff(pts, axis=0)
    cpm = 0.5 * (cps[:-1] + cps[1:])
    f = -cpm[:, None] * np.column_stack((d[:, 1], -d[:, 0]))  # outward normal * ds
    mid = 0.5 * (pts[:-1] + pts[1:]) - xref
    lift = float(np.sum(f[:, 1] * ca - f[:, 0] * sa)) / inv.chord
    moment = -float(np.sum(mid[:, 0] * f[:, 1] - mid[:, 1] * f[:, 0])) / inv.chord**2
    # drag from the wake end (Squire-Young)
    th, ds, ue = U[-1, 0], U[-1, 1], U[-1, 3]
    cd = 2.0 * th * ue ** ((5.0 + ds / th) / 2.0) / inv.chord
    # friction drag
    th_, ds_, sa_, q, ue_all, xi = (
        np.asarray(t)
        for t in system._unpack(
            jnp.asarray(U), jnp.asarray(np.where(np.arange(n) <= a, -1.0, 1.0)), a
        )
    )
    cf = np.zeros(system.N)
    for i in range(n):
        cf[i] = float(
            cl.skin_friction(th_[i], ds_[i], ue_all[i], system.nu, turb[i], False)
        )
    sst = inv.s[a] + (inv.s[a + 1] - inv.s[a]) * (-v[a]) / (v[a + 1] - v[a])
    w = (sst - inv.s[a]) / (inv.s[a + 1] - inv.s[a])
    xst = (1 - w) * x[a] + w * x[a + 1]
    df = 0.0
    for nodes in system.surfaces(a)[:2]:
        p_x, p_f = xst, 0.0
        for i in nodes:
            fi = cf[i] * ue_all[i] ** 2
            dxv = x[i] - p_x
            df += 0.25 * (p_f + fi) * (dxv[0] * ca + dxv[1] * sa)
            p_x, p_f = x[i], fi
    cdf = df / (0.5 * inv.chord)
    return lift, cd, cdf, moment, cp, cf, xi


def _transition_x(system, U, a, turb, xi):
    x = system.inv.body.nodes
    out = []
    for nodes in system.surfaces(a)[:2]:
        m = next((m for m, i in enumerate(nodes) if turb[i]), None)
        if m is None or m == 0:
            out.append(float("nan"))
            continue
        p, i = nodes[m - 1], nodes[m]
        u1 = U[p].copy()
        u2 = U[i].copy()
        u1[3], u2[3] = abs(u1[3]), abs(u2[3])
        xt = float(
            res.transition_location(
                jnp.asarray(u1), jnp.asarray(u2), xi[p], xi[i], system.nu, system.ncrit
            )
        )
        w = (xt - xi[p]) / (xi[i] - xi[p])
        xx = (1 - w) * x[p] + w * x[i]
        le = system.inv.leading_edge
        out.append(float((xx - le) @ (0.5 * (x[0] + x[-1]) - le) / system.inv.chord**2))
    return tuple(out)


def analyse(
    nodes,
    alpha,
    reynolds,
    ncrit=9.0,
    wake_length=1.0,
    max_iterations=60,
    tolerance=1e-10,
    verbose=False,
):
    """Viscous analysis of an airfoil at one angle of attack.

    Parameters
    ----------
    nodes : array_like (n, 2)
        Panel nodes of the airfoil contour (see :func:`.geometry.repanel`).
    alpha : float
        Angle of attack in degrees.
    reynolds : float
        Chord Reynolds number (unit chord and freestream speed assumed).
    ncrit : float
        Critical amplification factor of the e^N method.
    wake_length : float
        Wake length in chords.
    max_iterations, tolerance : int, float
        Newton limits; converged when the residual norm drops below tolerance.
    verbose : bool
        Print the Newton history.

    Returns
    -------
    IblResult
    """
    inv = inv_mod.build(nodes, alpha, wake_length)
    system = _System(inv, reynolds, ncrit)
    vinv, winv = inv.speeds(alpha)
    U, a, turb = _initial_state(system, vinv, winv)
    jv, jw = jnp.asarray(vinv), jnp.asarray(winv)
    history = []
    converged = False
    it = 0
    rnorm = np.inf
    for it in range(1, max_iterations + 1):
        kind, prev, sgn = system.kinds(a, turb)
        args = (
            jnp.asarray(kind),
            jnp.asarray(prev),
            jnp.asarray(sgn),
            a,
            jnp.asarray(turb),
            jv,
            jw,
        )
        R = np.asarray(system._residual(jnp.asarray(U), *args))
        rnorm = float(np.linalg.norm(R))
        history.append(rnorm)
        if verbose:
            rr = np.abs(R.reshape(-1, 4))
            k = np.unravel_index(np.argmax(rr), rr.shape)
            print(
                f"  Newton {it:3d}  |R| = {rnorm:.3e}  "
                f"max at node {k[0]} eq {k[1]} kind {kind[k[0]]}"
            )
        if not np.isfinite(rnorm):
            break
        if rnorm < tolerance:
            converged = True
            break
        J = system.jacobian(U, kind, prev, sgn, a, turb)
        dU = np.linalg.solve(J, -R).reshape(system.N, 4)
        omega, why = _limit(system, U, dU, turb)
        if verbose:
            ft = [
                next((m for m, i in enumerate(nd) if turb[i]), None)
                for nd in system.surfaces(a)[:2]
            ]
            print(
                f"      omega {omega:.3g} ({why})  stag {a}  "
                f"first turbulent (upper, lower) {ft}"
            )
        U = U + omega * dU
        # keep Hk above its floor
        gap = np.concatenate((np.zeros(system.n), inv.wake_gap))
        hmin = np.where(np.arange(system.N) < system.n, 1.02, 1.00005)
        U[:, 1] = np.maximum(U[:, 1], gap + hmin * U[:, 0])
        if np.any(turb):
            ctmax = np.max(U[turb, 2])
            neg = turb & (U[:, 2] <= 0.0)
            U[neg, 2] = 0.1 * ctmax
        a = _update_stagnation(system, U, a, turb)
        _, _, _, _, _, xi = (
            np.asarray(t)
            for t in system._unpack(
                jnp.asarray(U),
                jnp.asarray(np.where(np.arange(system.n) <= a, -1.0, 1.0)),
                a,
            )
        )
        _update_transition(system, U, a, turb, xi)
    lift, cd, cdf, cm, cp, cf, xi = _forces(inv, U, alpha, system, a, turb)
    ue = np.abs(U[:, 3])
    return IblResult(
        alpha=float(alpha),
        reynolds=float(reynolds),
        cl=lift,
        cd=cd,
        cdf=cdf,
        cdp=cd - cdf,
        cm=cm,
        converged=converged,
        iterations=it,
        residual=rnorm,
        x_transition=_transition_x(system, U, a, turb, xi),
        nodes=inv.body.nodes,
        wake=inv.wake,
        theta=U[:, 0],
        delta_star=U[:, 1],
        ue=ue,
        cp=cp,
        cf=cf,
        turbulent=turb.copy(),
        history=history,
        state=U.copy(),
        stagnation=int(a),
        inviscid=inv,
    )

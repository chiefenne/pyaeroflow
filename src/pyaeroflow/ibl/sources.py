"""Source panels for the displacement effect of the boundary layer.

The boundary layer and wake displace the outer flow by the mass defect
m = ue delta*; the panel method sees it as a source sheet of strength
sigma = dm/dxi (Drela 1989). Airfoil panels carry constant sources, wake
panels linear ones (continuous strength, so the edge speed at a wake node
is finite).

Two kinds of influence are needed:

* stream function at the airfoil nodes, in the airfoil boundary condition.
  A source makes the stream function multi-valued; each formula here uses a
  branch that is single-valued inside the airfoil. For an airfoil panel the
  angle is measured from the inward normal, so the jump sits in the strip
  outside the panel; for a wake panel it is measured from the upstream
  direction, so the jump runs downstream along the wake.
* velocity at the wake nodes, from the closed-form integrals.

Panel coordinates: x1 along the panel from its start, y normal (positive to
the left of start -> end, the inside of a counter-clockwise contour).
"""

import jax
import jax.numpy as jnp
import numpy as np

_TWO_PI = 2.0 * np.pi
_TINY = 1e-30


def _local(points, a, b):
    d = b - a
    length = jnp.sqrt(jnp.sum(d * d, axis=1))
    t = d / length[:, None]
    r = points[:, None, :] - a[None, :, :]
    x1 = r[..., 0] * t[None, :, 0] + r[..., 1] * t[None, :, 1]
    y = r[..., 1] * t[None, :, 0] - r[..., 0] * t[None, :, 1]
    return x1, y, length[None, :], t


def constant_psi(points, a, b):
    """Stream function (m, n) at points per unit constant source on panels a -> b.

    Branch with its jump in the strip on the right of the panel (outside a
    counter-clockwise body): psi = 1/(2 pi) int atan2(s - x, y) ds.
    """
    x1, y, length, _ = _local(points, a, b)
    yp = y + 0.0

    def f(u):
        return u * jnp.arctan2(u, yp) - 0.5 * yp * jnp.log(
            jnp.maximum(u * u + yp * yp, _TINY)
        )

    return (f(length - x1) - f(-x1)) / _TWO_PI


def linear_psi(points, a, b):
    """Stream function (m, n, 2) per unit linear source at the start / end node.

    Angle measured from the upstream direction -t: the branch jump runs
    downstream of each source point along the panel line.
    """
    x1, y, length, _ = _local(points, a, b)
    yy = -y + 0.0
    ysafe = jnp.where(jnp.abs(yy) > _TINY, yy, _TINY)

    def f0(u):
        return u * jnp.arctan2(yy, u) + 0.5 * yy * jnp.log(
            jnp.maximum(u * u + yy * yy, _TINY)
        )

    def f1(u):
        return 0.5 * u * u * jnp.arctan2(yy, u) + 0.5 * yy * (
            u - yy * jnp.arctan(u / ysafe)
        )

    u1, u2 = -x1, length - x1
    d0 = f0(u2) - f0(u1)
    d1 = f1(u2) - f1(u1)
    pa = (1.0 - x1 / length) * d0 - d1 / length
    pb = (x1 / length) * d0 + d1 / length
    return jnp.stack((pa, pb), axis=-1) / _TWO_PI


def _subtended(points, a, b):
    ra = a[None, :, :] - points[:, None, :]
    rb = b[None, :, :] - points[:, None, :]
    cross = ra[..., 0] * rb[..., 1] - ra[..., 1] * rb[..., 0]
    dot = ra[..., 0] * rb[..., 0] + ra[..., 1] * rb[..., 1]
    beta = jnp.arctan2(cross, dot)
    lr1 = 0.5 * jnp.log(jnp.maximum(jnp.sum(ra * ra, axis=-1), _TINY))
    lr2 = 0.5 * jnp.log(jnp.maximum(jnp.sum(rb * rb, axis=-1), _TINY))
    return beta, lr1, lr2


def _to_global(ut, un, t):
    n = jnp.stack((-t[:, 1], t[:, 0]), axis=-1)  # left normal
    return ut[..., None] * t[None, :, None, :] + un[..., None] * n[None, :, None, :]


def constant_velocity(points, a, b):
    """Velocity (m, n, 2) at points per unit constant source on panels a -> b."""
    x1, y, length, t = _local(points, a, b)
    beta, lr1, lr2 = _subtended(points, a, b)
    ut = (lr1 - lr2) / _TWO_PI
    un = beta / _TWO_PI
    return _to_global(ut[..., None], un[..., None], t)[..., 0, :]


def linear_velocity(points, a, b):
    """Velocity (m, n, 2, 2): [..., k, :] per unit source at the start (k=0) / end."""
    x1, y, length, t = _local(points, a, b)
    beta, lr1, lr2 = _subtended(points, a, b)
    w = x1 / length
    ln12 = lr1 - lr2
    lin = (length - y * beta) / length
    ut = jnp.stack(((1.0 - w) * ln12 + lin, w * ln12 - lin), axis=-1)
    un = jnp.stack(
        ((1.0 - w) * beta + (y / length) * ln12, w * beta - (y / length) * ln12),
        axis=-1,
    )
    return _to_global(ut, un, t) / _TWO_PI


def vortex_velocity(points, body):
    """Velocity (m, n, 2) at points per unit sheet strength gamma at the nodes."""
    from pyaeroflow.numerics import panel

    def psi_row(p):
        return panel._influence(p[None, :], body)[0]

    jac = jax.vmap(jax.jacfwd(psi_row))(
        jnp.asarray(points)
    )  # (m, n, 2): dpsi/dx, dpsi/dy
    return jnp.stack((jac[..., 1], -jac[..., 0]), axis=-1)

"""Discrete boundary-layer equations between two stations.

State of a station: u = (theta, delta*, s, ue) with s the amplification factor
(laminar) or sqrt(C_tau) (turbulent and wake); delta* includes the wake gap in
the wake. For an interval [xi1, xi2] the three residuals are the logarithmic
forms of

    momentum:      d ln theta + (2 + H) d ln ue = (C_f/2) dxi / theta
    shape (k.e.):  d ln H* + (1 - H) d ln ue = (C_f/2 - 2 C_D/H*) dxi / theta
    laminar:       dn = (dn/dxi) dxi
    turbulent:     2 delta d ln s = K/(B(1+Us)) (s_eq - s) dxi
                                    + 2 delta (uq dxi - d ln ue)

(Drela & Giles 1987; Drela 1989), with dxi/theta terms in the form
ln(xi2/xi1) xi/theta so that the stations near the stagnation point keep
their accuracy. Friction is averaged with the midpoint (1/4, 1/2, 1/4); the
source terms of the shape and lag equations are weighted towards the
downstream station when Hk changes quickly (Fidkowski 2022, Sec. III.B).
"""

import jax
import jax.numpy as jnp

from . import closures as cl


def _upwind_weight(hk1, hk2, wake):
    """0.5 for smooth Hk, towards 1 where Hk changes quickly."""
    c = jnp.where(wake, 1.0, 5.0)
    huc = c / hk2**2
    la = jnp.log(jnp.abs((hk2 - 1.0) / (hk1 - 1.0)))
    hls = jnp.minimum(la**2, 15.0)
    return 1.0 - 0.5 * jnp.exp(-hls * huc)


def _point(u, gap, xi, nu, turb, wake):
    """Closure quantities of one station (delta* without the wake gap)."""
    th, ds, sa, ue = u[0], u[1] - gap, u[2], u[3]
    hk = cl.kinematic_shape(th, ds, wake)
    return dict(
        th=th,
        h=ds / th,
        hw=gap / th,
        hk=hk,
        hs=cl.hstar(th, ds, ue, nu, turb, wake),
        ret=cl.reynolds_theta(th, ue, nu),
        cfx=cl.skin_friction(th, ds, ue, nu, turb, wake) * xi / th,
        cf=cl.skin_friction(th, ds, ue, nu, turb, wake),
        dix=cl.dissipation(th, ds, sa, ue, nu, turb, wake) * xi / th,
        ds=ds,
    )


def interval(u1, u2, x1, x2, gap1, gap2, nu, turb, wake, ncrit):
    """Residuals (3,) of the interval from station 1 to station 2.

    ``turb`` and ``wake`` apply to the whole interval.
    """
    p1 = _point(u1, gap1, x1, nu, turb, wake)
    p2 = _point(u2, gap2, x2, nu, turb, wake)
    um = 0.5 * (u1 + u2)
    xm = 0.5 * (x1 + x2)
    pm = _point(um, 0.5 * (gap1 + gap2), xm, nu, turb, wake)
    upw = _upwind_weight(p1["h"], p2["h"], wake)

    def up(a, b):
        return (1.0 - upw) * a + upw * b

    thlog = jnp.log(u2[0] / u1[0])
    uelog = jnp.log(u2[3] / u1[3])
    xlog = jnp.log(x2 / x1)
    dx = x2 - x1
    h = 0.5 * (p1["h"] + p2["h"])
    hw = 0.5 * (p1["hw"] + p2["hw"])

    cfx = 0.25 * p1["cfx"] + 0.5 * pm["cfx"] + 0.25 * p2["cfx"]
    r_mom = thlog + (2.0 + h + hw) * uelog - 0.5 * xlog * cfx

    hslog = jnp.log(p2["hs"] / p1["hs"])
    r_shape = (
        hslog
        + (1.0 - h - hw) * uelog
        + xlog * (0.5 * up(p1["cfx"], p2["cfx"]) - up(p1["dix"], p2["dix"]))
    )

    # amplification (laminar)
    th1, ds1, n1, ue1 = u1[0], u1[1] - gap1, u1[2], u1[3]
    th2, ds2, n2, ue2 = u2[0], u2[1] - gap2, u2[2], u2[3]
    rate = 0.5 * (
        cl.amplification_rate(th1, ds1, n1, ue1, nu, ncrit)
        + cl.amplification_rate(th2, ds2, n2, ue2, nu, ncrit)
    )
    r_amp = n2 - n1 - rate * dx

    # shear lag (turbulent, wake)
    s1 = jnp.maximum(u1[2], 1e-12)
    s2 = jnp.maximum(u2[2], 1e-12)
    salog = jnp.log(s2 / s1)
    de = 0.5 * (cl.bl_thickness(th1, ds1, wake) + cl.bl_thickness(th2, ds2, wake))
    us = 0.5 * (
        cl.slip_velocity(th1, ds1, ue1, nu, True, wake)
        + cl.slip_velocity(th2, ds2, ue2, nu, True, wake)
    )
    hk = up(p1["h"], p2["h"])
    ret = 0.5 * (p1["ret"] + p2["ret"])
    cf = up(p1["cf"], p2["cf"])
    dsa = 0.5 * (p1["ds"] + p2["ds"])
    uq = cl.equilibrium_gradient(dsa, cf, hk, ret, wake)
    cteq = up(
        cl.ctau_equilibrium(th1, ds1, ue1, nu, wake),
        cl.ctau_equilibrium(th2, ds2, ue2, nu, wake),
    )
    saa = up(u1[2], u2[2])
    clag = cl.SHEAR_LAG_K / cl.GB / (1.0 + us)
    ald = jnp.where(wake, cl.DISSIPATION_LENGTH_RATIO, 1.0)
    r_lag = (
        clag * (cteq - ald * saa) * dx
        - 2.0 * de * salog
        + 2.0 * de * (uq * dx - uelog) * cl.CUQ
    )
    r3 = jnp.where(turb, r_lag, r_amp)
    return jnp.stack((r_mom, r_shape, r3))


def similarity(u, xi, nu):
    """Residuals (3,) of the first laminar station next to the stagnation point.

    Stagnation-point (Hiemenz) flow: ue proportional to xi, theta and H
    constant, so the equations reduce to (2 + H) = (C_f/2) xi/theta and
    (1 - H) = (2 C_D/H* - C_f/2) xi/theta, and n = 0.
    """
    p = _point(u, 0.0, xi, nu, False, False)
    r_mom = (2.0 + p["h"]) - 0.5 * p["cfx"]
    r_shape = (1.0 - p["h"]) + (0.5 * p["cfx"] - p["dix"])
    return jnp.stack((r_mom, r_shape, u[2]))


def transition_location(u1, u2, x1, x2, nu, ncrit, iterations=12):
    """Where n reaches n_crit in the interval; state linear in xi.

    Solves n_crit = n1 + (xt - x1) (rate(u1) + rate(ut)) / 2 by Newton's
    method (a fixed number of steps, so that autodiff gives its sensitivities).
    """
    dx = x2 - x1

    def state(xt):
        w = (xt - x1) / dx
        ut = (1.0 - w) * u1 + w * u2
        return ut.at[2].set(ncrit)

    def rate(u):
        return cl.amplification_rate(u[0], u[1], u[2], u[3], nu, ncrit)

    r1 = rate(u1)

    def g(xt):
        return ncrit - u1[2] - 0.5 * (xt - x1) * (r1 + rate(state(xt)))

    def body(k, xt):
        val, slope = jax.value_and_grad(g)(xt)
        step = -val / jnp.where(jnp.abs(slope) > 0.0, slope, -1.0)
        dmax = 0.2 * dx * (1.1 - k / iterations)
        step = jnp.clip(step, -dmax, dmax)
        return jnp.clip(xt + step, x1 + 1e-6 * dx, x2 - 1e-6 * dx)

    xt = jax.lax.fori_loop(0, iterations, body, x1 + 0.5 * dx)
    return xt


def transition_interval(u1, u2, x1, x2, nu, ncrit):
    """Residuals (3,) of an interval with transition inside.

    Laminar from xi1 to the transition point xt (n = n_crit there), turbulent
    from xt to xi2 starting with sqrt(C_tau) = C exp(-E/(Hk - 1)) sqrt(C_tau,eq).
    The two sets of residuals are added.
    """
    xt = transition_location(u1, u2, x1, x2, nu, ncrit)
    w = (xt - x1) / (x2 - x1)
    ut = (1.0 - w) * u1 + w * u2
    ut_lam = ut.at[2].set(ncrit)
    ut_turb = ut.at[2].set(cl.ctau_transition(ut[0], ut[1], ut[3], nu))
    r_lam = interval(u1, ut_lam, x1, xt, 0.0, 0.0, nu, False, False, ncrit)
    r_turb = interval(ut_turb, u2, xt, x2, 0.0, 0.0, nu, True, False, ncrit)
    return r_lam + r_turb, xt


def stagnation(u0, x0, u1, x1, nu, xst=1e-6):
    """Residuals (3,) of the first station of a side, next to the stagnation point.

    The stagnation-point state is extrapolated from the first two stations:
    theta and delta* linearly to xi = 0, and ue = K xi with K the slope at
    xi = 0 of the parabola through (0, 0), (xi0, ue0), (xi1, ue1). The
    similarity equations are applied to that state (they do not depend on
    xi_st, because C_f xi / theta and C_D xi / theta scale with 1/(K theta^2)).
    This keeps the equations regular when the stagnation point comes close
    to a node. The third residual sets n = 0 at the first station.
    """
    dx = x1 - x0
    w0, w1 = x1 / dx, -x0 / dx
    k = u0[3] * x1 / (x0 * dx) - u1[3] * x0 / (x1 * dx)
    ust = jnp.stack(
        (w0 * u0[0] + w1 * u1[0], w0 * u0[1] + w1 * u1[1], 0.0 * u0[2], k * xst)
    )
    r = similarity(ust, xst, nu)
    return r.at[2].set(u0[2])

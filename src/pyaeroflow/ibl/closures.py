"""Closure relations of the integral boundary layer (incompressible).

All functions take the station state

    th  momentum thickness theta
    ds  displacement thickness delta* without the trailing-edge wake gap
    sa  amplification factor n (laminar) or sqrt(C_tau) (turbulent, wake)
    ue  edge speed

and the kinematic viscosity ``nu``; ``turb`` and ``wake`` are 0/1 flags (floats
or booleans, so that the same code serves every kind of station under
``jax.vmap``). Each branch is written so that it stays finite for any state,
because ``jnp.where`` differentiates both branches.

Sources: laminar closures (H*, C_f, C_D, amplification rate) from Drela & Giles
(1987) and Drela (1989); turbulent closures (Swafford C_f, H*, slip velocity,
G-beta equilibrium locus, shear lag) from Drela (1989) and the constants listed
in Fidkowski (2022, Table 1).
"""

import jax.numpy as jnp

# Shear-lag and G-beta locus constants (Drela 1989; Fidkowski 2022, Table 1).
GA = 6.7  # G-beta A
GB = 0.75  # G-beta B
GC = 18.0  # G-beta C
SHEAR_LAG_K = 5.6
DISSIPATION_LENGTH_RATIO = 0.9  # wall / wake
CUQ = 1.0
CTAU_C = 1.8  # initial C_tau after transition: C exp(-E / (Hk - 1)) C_tau,eq
CTAU_E = 3.3


def _hk_floor(wake):
    return jnp.where(wake, 1.00005, 1.05)


def shape_factor(th, ds):
    """Shape factor H = delta* / theta."""
    return ds / th


def kinematic_shape(th, ds, wake):
    """Hk (= H for incompressible flow), floored as the closures require."""
    return jnp.maximum(ds / th, _hk_floor(wake))


def reynolds_theta(th, ue, nu):
    """Momentum-thickness Reynolds number."""
    return ue * th / nu


def hstar(th, ds, ue, nu, turb, wake):
    """Kinetic-energy shape factor H*."""
    hk = kinematic_shape(th, ds, wake)
    # laminar
    a = hk - 4.35
    lam_lo = (
        (0.0111 * a**2 - 0.0278 * a**3) / (hk + 1.0) + 1.528 - 0.0002 * (a * hk) ** 2
    )
    lam_hi = 0.015 * a**2 / hk + 1.528
    lam = jnp.where(hk < 4.35, lam_lo, lam_hi)
    # turbulent
    ret = reynolds_theta(th, ue, nu)
    ho = jnp.where(ret > 400.0, 3.0 + 400.0 / jnp.maximum(ret, 1e-30), 4.0)
    reb = jnp.maximum(ret, 200.0)
    hr = (ho - hk) / (ho - 1.0)
    att = 1.5 + 4.0 / reb + (2.0 - 1.5 - 4.0 / reb) * hr**2 * 1.5 / (hk + 0.5)
    lrb = jnp.log(reb)
    sep = (
        1.5
        + 4.0 / reb
        + (hk - ho) ** 2 * (0.007 * lrb / (hk - ho + 4.0 / lrb) ** 2 + 0.015 / hk)
    )
    tur = jnp.where(hk < ho, att, sep)
    return jnp.where(turb, tur, lam)


def skin_friction(th, ds, ue, nu, turb, wake):
    """Skin-friction coefficient C_f (zero in the wake)."""
    hk = ds / th
    ret = reynolds_theta(th, ue, nu)
    lo = 0.0727 * jnp.maximum(5.5 - hk, 0.0) ** 3 / (hk + 1.0) - 0.07
    hks = jnp.maximum(hk, 5.5)
    hi = 0.015 * (1.0 - 1.0 / (hks - 4.5)) ** 2 - 0.07
    lam = jnp.where(hk < 5.5, lo, hi) / ret
    aa = -1.33 * hk
    aa = jnp.where(aa < -17.0, -20.0 + 3.0 * jnp.exp((aa + 17.0) / 3.0), aa)
    bb = jnp.maximum(jnp.log(jnp.maximum(ret, 1e-30)), 3.0) / jnp.log(10.0)
    cc = -1.74 - 0.31 * hk
    dd = jnp.tanh(4.0 - hk / 0.875)
    tur = 0.3 * jnp.exp(aa) * bb**cc + 1.1e-4 * (dd - 1.0)
    cf = jnp.where(turb, tur, lam)
    return jnp.where(wake, 0.0, cf)


def slip_velocity(th, ds, ue, nu, turb, wake):
    """Normalised slip velocity Us of the turbulent layer."""
    hs = hstar(th, ds, ue, nu, turb, wake)
    hk = kinematic_shape(th, ds, wake)
    h = ds / th
    us = 0.5 * hs * (1.0 - (hk - 1.0) / (GB * h))
    us = jnp.where(jnp.logical_and(jnp.logical_not(wake), us > 0.95), 0.98, us)
    return us


def bl_thickness(th, ds, wake):
    """Boundary-layer thickness delta, at most 12 theta."""
    hk = kinematic_shape(th, ds, wake)
    de = th * (3.15 + 1.72 / (hk - 1.0)) + ds
    return jnp.minimum(de, 12.0 * th)


def _laminar_dissipation(hk, ret):
    """2 C_D / H* of the laminar wall layer."""
    lo = 0.00205 * jnp.maximum(4.0 - hk, 0.0) ** 5.5 + 0.207
    h1 = hk - 4.0
    hi = -0.0016 * h1**2 / (1.0 + 0.02 * h1**2) + 0.207
    return jnp.where(hk < 4.0, lo, hi) / ret


def dissipation(th, ds, sa, ue, nu, turb, wake):
    """Dissipation coefficient in the form 2 C_D / H*.

    Laminar wall layer: Drela-Giles fit. Turbulent: wall part (C_f Us / 2,
    blended off towards Hk = 1), outer-layer part from the shear stress
    C_tau (1 - Us), and a laminar-stress part; not lower than the laminar
    value. The wake has no wall part, the laminar wake value as floor and
    counts both halves.
    """
    hk = ds / th
    ret = reynolds_theta(th, ue, nu)
    hs = hstar(th, ds, ue, nu, turb, wake)
    lam_wall = _laminar_dissipation(hk, ret)
    hs_lam = hstar(th, ds, ue, nu, False, wake)
    lam_wake = 2.0 * 1.1 * (1.0 - 1.0 / hk) ** 2 / hk / (hs_lam * ret)
    cf = skin_friction(th, ds, ue, nu, turb, wake)
    us = slip_velocity(th, ds, ue, nu, turb, wake)
    lr = jnp.log(jnp.maximum(ret, 1.0 + 1e-6))
    hmin = 1.0 + 2.1 / lr
    fac = 0.5 + 0.5 * jnp.tanh((hk - 1.0) / (hmin - 1.0))
    wall = jnp.where(wake, 0.0, 0.5 * cf * us * (2.0 / hs) * fac)
    outer = sa**2 * (0.995 - us) * 2.0 / hs
    lam_stress = 0.15 * (0.995 - us) ** 2 * 2.0 / (hs * ret)
    tur = wall + outer + lam_stress
    floor = jnp.where(wake, lam_wake, lam_wall)
    tur = jnp.maximum(tur, floor)
    tur = jnp.where(wake, 2.0 * tur, tur)
    lam = jnp.where(wake, lam_wake, lam_wall)
    return jnp.where(turb, tur, lam)


def ctau_equilibrium(th, ds, ue, nu, wake):
    """Sqrt of the equilibrium shear-stress coefficient (G-beta locus)."""
    hk = kinematic_shape(th, ds, wake)
    ret = reynolds_theta(th, ue, nu)
    hs = hstar(th, ds, ue, nu, 1.0, wake)
    us = slip_velocity(th, ds, ue, nu, 1.0, wake)
    h = ds / th
    hkc = jnp.where(wake, hk - 1.0, jnp.maximum(hk - 1.0 - GC / ret, 0.01))
    cc = 0.5 / (GA**2 * GB)
    num = cc * hs * (hk - 1.0) * hkc**2
    den = (1.0 - us) * h * hk**2
    return jnp.sqrt(num / den)


def ctau_transition(th, ds, ue, nu):
    """sqrt(C_tau) set just downstream of transition."""
    hk = kinematic_shape(th, ds, False)
    return (
        CTAU_C * jnp.exp(-CTAU_E / (hk - 1.0)) * ctau_equilibrium(th, ds, ue, nu, False)
    )


def equilibrium_gradient(ds, cf, hk, ret, wake):
    """Term uq of the shear-lag equation: (C_f/2 - (Hkc/(A Hk))^2) / (B delta*)."""
    a = jnp.where(wake, GA * DISSIPATION_LENGTH_RATIO, GA)
    hkc = jnp.where(wake, hk - 1.0, hk - 1.0 - GC / ret)
    return (0.5 * cf - (hkc / (a * hk)) ** 2) / (GB * ds)


def amplification_rate(th, ds, sa, ue, nu, ncrit):
    """Envelope e^N amplification rate dn/dxi (Drela & Giles 1987).

    dn/dxi = dn/dRe_theta * (m + 1)/2 * l / theta, switched on smoothly over
    log10 Re_theta = log10 Re_theta,0 +- 0.1, plus a small term that pushes n
    through n_crit (it switches on around n_crit only).
    """
    hk = kinematic_shape(th, ds, False)
    ret = reynolds_theta(th, ue, nu)
    hmi = 1.0 / (hk - 1.0)
    lrc = 2.492 * hmi**0.43 + 0.7 * (jnp.tanh(14.0 * hmi - 9.24) + 1.0)
    lr = jnp.log10(jnp.maximum(ret, 1e-30))
    dl = 0.1
    rn = jnp.clip((lr - (lrc - dl)) / (2.0 * dl), 0.0, 1.0)
    rf = 3.0 * rn**2 - 2.0 * rn**3
    ar = 3.87 * hmi - 2.52
    da = 0.028 * (hk - 1.0) - 0.0345 * jnp.exp(-(ar**2))
    af = -0.05 + 2.7 * hmi - 5.5 * hmi**2 + 3.0 * hmi**3 + 0.1 * jnp.exp(-20.0 * hmi)
    rate = rf * af * da / th
    push = (1.0 + jnp.tanh(5.0 * (sa - ncrit))) * 0.001 / th
    return rate + push

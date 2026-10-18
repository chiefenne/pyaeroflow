"""Wall loads: pressure and skin friction, forces, moments, coefficients.

On a boundary face the outward normal n points from the fluid into the wall
or body. The Cauchy traction sigma . n with sigma = -p I + tau is the force
per area the wall exerts on the fluid, so the force the fluid exerts on the
body is

    F = sum_faces (p n - tau . n) L,

with the pressure of the boundary state and the viscous stress from the
boundary-face gradient (D24). The moment is taken about a reference point,
positive counter-clockwise.

Coefficients use q = rho_ref U_ref^2 / 2 and the reference length:
C_F = F / (q L_ref), C_M = M / (q L_ref^2), C_p = (p - p_ref) / q,
C_f = tau_w . t / q with the wall shear stress tau_w = -(tau . n) the fluid
exerts on the wall and the unit tangent t of the face, oriented
counter-clockwise with respect to the fluid (for a body in the domain this
runs clockwise around the body; on the bottom wall of a channel or a flat
plate it points in +x, so C_f > 0 for flow in +x). Lift and drag follow from
the angle of attack alpha: C_D = C_x cos(alpha) + C_y sin(alpha),
C_L = -C_x sin(alpha) + C_y cos(alpha).
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pyaeroflow.numerics import boundary_face_gradient, lsq_gradient
from pyaeroflow.physics.incompressible import P, boundary_state, viscous_flux


class WallLoads(NamedTuple):
    """Per boundary face: pressure and traction (JAX arrays)."""

    pressure: jnp.ndarray  # (n_bfaces,)
    traction: jnp.ndarray  # (n_bfaces, 2): tau . n


@partial(jax.jit, static_argnums=(0, 4))
def wall_loads(model, geom, bc, w, boundary_state=boundary_state):
    """Pressure and viscous traction on every boundary face.

    ``w`` carries all variables of the model (3, or 4 with a turbulence
    model, with the matching ``boundary_state``); the traction uses the
    molecular viscosity (the eddy viscosity vanishes at walls).
    """
    w_b = boundary_state(bc, geom.b_normal, w[geom.b_owner])
    grad = lsq_gradient(geom, w, w_b)
    g_b = boundary_face_gradient(geom, w, w_b, grad, model.viscous_gradient)
    traction = -viscous_flux(model, g_b[:, 1:3], geom.b_normal)[:, 1:]
    return WallLoads(w_b[:, P], traction)


class Forces(NamedTuple):
    """Integrated loads on a set of boundary faces.

    Attributes
    ----------
    fx, fy : float
        Force components (pressure plus viscous).
    fx_pressure, fy_pressure, fx_viscous, fy_viscous : float
        The two contributions.
    mz : float
        Moment about the reference point, counter-clockwise positive.
    """

    fx: float
    fy: float
    fx_pressure: float
    fy_pressure: float
    fx_viscous: float
    fy_viscous: float
    mz: float


def integrate(loads, geom, faces, moment_center=(0.0, 0.0)):
    """Integrate the loads over the given boundary-face indices.

    Parameters
    ----------
    loads : WallLoads
    geom : FVGeometry
    faces : ndarray of int
        Indices into the boundary-face arrays.
    moment_center : tuple of float

    Returns
    -------
    Forces
    """
    p = np.asarray(loads.pressure)[faces]
    tau = np.asarray(loads.traction)[faces]
    n = np.asarray(geom.b_normal)[faces]
    length = np.asarray(geom.b_length)[faces]
    centre = np.asarray(geom.b_centre)[faces]
    f_p = p[:, None] * n * length[:, None]
    f_v = -tau * length[:, None]
    f = f_p + f_v
    r = centre - np.asarray(moment_center, dtype=float)
    mz = float(np.sum(r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]))
    return Forces(
        float(f[:, 0].sum()),
        float(f[:, 1].sum()),
        float(f_p[:, 0].sum()),
        float(f_p[:, 1].sum()),
        float(f_v[:, 0].sum()),
        float(f_v[:, 1].sum()),
        mz,
    )


class Coefficients(NamedTuple):
    """Force and moment coefficients."""

    cl: float
    cd: float
    cm: float
    cx: float
    cy: float
    cd_pressure: float
    cd_viscous: float


def coefficients(forces, q, length, angle_of_attack=0.0):
    """Coefficients of integrated forces.

    Parameters
    ----------
    forces : Forces
    q : float
        Dynamic pressure rho_ref U_ref^2 / 2.
    length : float
        Reference length.
    angle_of_attack : float
        In degrees; defines the drag (along the flow) and lift directions.

    Returns
    -------
    Coefficients
    """
    a = np.radians(angle_of_attack)
    ca, sa = np.cos(a), np.sin(a)
    cx, cy = forces.fx / (q * length), forces.fy / (q * length)
    cxp, cyp = forces.fx_pressure / (q * length), forces.fy_pressure / (q * length)
    cxv, cyv = forces.fx_viscous / (q * length), forces.fy_viscous / (q * length)
    return Coefficients(
        cl=-cx * sa + cy * ca,
        cd=cx * ca + cy * sa,
        cm=forces.mz / (q * length**2),
        cx=cx,
        cy=cy,
        cd_pressure=cxp * ca + cyp * sa,
        cd_viscous=cxv * ca + cyv * sa,
    )


def wall_distribution(loads, geom, faces, q, p_ref=0.0):
    """Per-face wall quantities for output.

    Returns
    -------
    dict of str to ndarray
        ``x``, ``y`` (face centre), ``p``, ``cp``, ``tau_x``, ``tau_y`` (wall
        shear stress exerted by the fluid on the wall), ``cf`` (its
        tangential component / q, tangent oriented counter-clockwise with
        respect to the fluid), in the face order of the mesh.
    """
    p = np.asarray(loads.pressure)[faces]
    tau_w = -np.asarray(loads.traction)[faces]
    n = np.asarray(geom.b_normal)[faces]
    tangent = np.column_stack((-n[:, 1], n[:, 0]))
    centre = np.asarray(geom.b_centre)[faces]
    return {
        "x": centre[:, 0],
        "y": centre[:, 1],
        "p": p,
        "cp": (p - p_ref) / q,
        "tau_x": tau_w[:, 0],
        "tau_y": tau_w[:, 1],
        "cf": np.einsum("fk,fk->f", tau_w, tangent) / q,
    }

"""Initial flow field of a CFD case from the coupled boundary-layer solution.

The integral method gives, along the airfoil and the wake, the edge speed
ue, the displacement and momentum thicknesses delta* and theta, the skin
friction and the laminar / turbulent state. It does not give a field. The
field is built here in three parts:

* outer flow at every cell centre: the vortex sheet of the viscous solution
  (its circulation is the viscous lift) plus the source sheet of the mass
  defect ue delta* on the airfoil and the wake (the displacement effect);
* boundary layer: within the layer thickness delta the speed follows a
  power-law profile u / ue = (y / delta)^(1/N) with N = 2 / (H - 1), which
  has the shape factor H = delta* / theta of the solution; the wall distance
  y is the mesh's, the boundary-layer state that of the nearest wall point;
* wake: a cosine-squared defect whose depth and width give the delta* and
  theta of the wake solution; beyond the computed wake the last profile
  continues along the last wake direction.

The pressure is constant across the layer: p - p_inf = rho (U_inf^2 - |V|^2) / 2
of the outer flow. The eddy viscosity of a turbulent layer, for nu_tilde of
Spalart-Allmaras, is kappa u_tau y near the wall (the log-layer solution of
the model) and the Clauser value 0.0168 ue delta* further out, with
Klebanoff's intermittency; laminar parts keep the freestream value.
"""

from typing import NamedTuple

import numpy as np

from pyaeroflow.mesh.mesh import MeshError
from pyaeroflow.numerics import panel
from pyaeroflow.numerics.wall_distance import _foot, mesh_wall_distance, wall_distance

from . import inviscid as inv_mod
from .geometry import repanel
from .solver import analyse

KAPPA = 0.41
CLAUSER = 0.0168
SPEED_LIMIT = 3.0  # largest outer speed of the field / freestream speed


class IblField(NamedTuple):
    """Initial field of :func:`ibl_field` at the points (all in case units).

    Attributes
    ----------
    velocity : ndarray (m, 2)
    dynamic : ndarray (m,)
        (U_inf^2 - |V_outer|^2) / 2: pressure difference to the freestream per
        unit density.
    eddy_viscosity : ndarray (m,)
        Kinematic eddy viscosity of the turbulent layers (0 elsewhere).
    result : IblResult
        The boundary-layer solution (unit chord and freestream speed).
    chord : float
    """

    velocity: np.ndarray
    dynamic: np.ndarray
    eddy_viscosity: np.ndarray
    result: object
    chord: float


def _interp_nodes(values, k, t):
    return (1.0 - t) * values[k] + t * values[k + 1]


def _project(points, poly):
    """Nearest segment k and parameter t in [0, 1] on the polyline poly."""
    segs = np.stack((poly[:-1], poly[1:]), axis=1)
    d, k = wall_distance(points, segs)
    a, b = poly[k], poly[k + 1]
    ab = b - a
    t = np.einsum("ij,ij->i", points - a, ab) / np.einsum("ij,ij->i", ab, ab)
    return d, k, np.clip(t, 0.0, 1.0), t


def ibl_field(
    contour,
    points,
    wall_distance_,
    wall_foot,
    velocity,
    nu,
    ncrit=9.0,
    n_panels=200,
    wake_extension=100.0,
    fully_turbulent=False,
):
    """Build the initial velocity, pressure and eddy viscosity from the IBL.

    Parameters
    ----------
    contour : ndarray (n, 2)
        Closed airfoil contour (any start and direction).
    points : ndarray (m, 2)
        Cell centres.
    wall_distance_ : ndarray (m,)
        Distance of the points to the wall (the mesh wall distance).
    wall_foot : ndarray (m, 2)
        Nearest wall point of every point.
    velocity : (float, float)
        Freestream velocity.
    nu : float
        Kinematic viscosity.
    ncrit : float
        Critical amplification factor of the boundary-layer transition.
    n_panels : int
        Panel nodes of the boundary-layer analysis.
    wake_extension : float
        Length (chords) over which the last wake profile is continued.
    fully_turbulent : bool
        Give the laminar parts of the layer the turbulent eddy viscosity too
        (for a CFD model without transition, which is turbulent from the
        leading edge).

    Returns
    -------
    IblField

    Raises
    ------
    RuntimeError
        If the boundary-layer solution does not converge.
    """
    u_inf = np.asarray(velocity, dtype=float)
    speed = float(np.hypot(*u_inf))
    alpha = float(np.degrees(np.arctan2(u_inf[1], u_inf[0])))
    body = panel.contour_body(np.asarray(contour, dtype=float), u_inf)
    loop = body.nodes  # first == last node at a sharp trailing edge
    te = 0.5 * (loop[0] + loop[-1])
    chord = float(np.max(np.linalg.norm(loop - te, axis=1)))
    reynolds = speed * chord / nu
    nodes = repanel(loop / chord, n_panels)
    r = analyse(nodes, alpha, reynolds, ncrit=ncrit)
    if not r.converged:
        raise RuntimeError(
            f"boundary-layer solution not converged ({r.iterations} Newton steps, "
            f"residual {r.residual:.2e})"
        )
    inv = r.inviscid
    U = r.state
    n = inv.s.size
    x = inv.body.nodes
    q = U[:, 3]  # signed speed on the airfoil, ue in the wake
    th, ds = U[:, 0], U[:, 1]
    pts = np.asarray(points, dtype=float) / chord
    y = np.asarray(wall_distance_, dtype=float) / chord
    foot = np.asarray(wall_foot, dtype=float) / chord

    # outer flow: viscous vortex sheet + displacement sources (unit speed)
    sol = panel.PanelSolution(
        bodies=(inv.body,),
        gamma=(-q[:n],),
        psi0=np.zeros(1),
        velocity=np.array([np.cos(np.radians(alpha)), np.sin(np.radians(alpha))]),
    )
    v_out = panel.velocity(sol, pts) + inv_mod.displacement_velocity(inv, q * ds, pts)
    mag = np.linalg.norm(v_out, axis=1)
    v_out *= np.minimum(1.0, SPEED_LIMIT / np.maximum(mag, 1e-300))[:, None]

    # airfoil boundary layer at the nearest wall point
    poly = np.vstack((x, x[:1])) if inv.body.blunt else x
    _, k, t, _ = _project(foot, poly)
    k = np.minimum(k, n - 2)
    tan = x[k + 1] - x[k]
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    q_w = _interp_nodes(q[:n], k, t)
    th_w = _interp_nodes(th[:n], k, t)
    ds_w = _interp_nodes(ds[:n], k, t)
    cf_w = _interp_nodes(r.cf[:n], k, t)
    turb_w = np.where(t < 0.5, r.turbulent[k], r.turbulent[k + 1])
    h = np.clip(ds_w / th_w, 1.1, 8.0)
    expo = 2.0 / (h - 1.0)  # N of the power law
    delta = ds_w * (expo + 1.0)
    eta = y / delta
    f = np.where(eta < 1.0, np.maximum(eta, 0.0) ** (1.0 / expo), 1.0)
    blend = np.clip(eta, 0.0, 1.0) ** 2
    v_edge = q_w[:, None] * tan
    v_bl = (1.0 - blend)[:, None] * v_edge + blend[:, None] * v_out
    vel = np.where((eta < 1.0)[:, None], f[:, None] * v_bl, v_out)
    q2 = np.sum(np.where((eta < 1.0)[:, None], v_bl, v_out) ** 2, axis=1)
    ue_w = np.abs(q_w)
    u_tau = ue_w * np.sqrt(0.5 * np.abs(cf_w))
    nut_bl = np.minimum(KAPPA * u_tau * y, CLAUSER * ue_w * ds_w) / (
        1.0 + 5.5 * np.minimum(eta, 10.0) ** 6
    )
    nut = np.where(turb_w | fully_turbulent, nut_bl, 0.0)

    # wake: cosine-squared defect about the wake line
    wk = inv.wake
    end_dir = (wk[-1] - wk[-2]) / np.linalg.norm(wk[-1] - wk[-2])
    wpoly = np.vstack((wk, wk[-1] + wake_extension * end_dir))
    uw = np.append(q[n:], q[-1])
    thw = np.append(th[n:], th[-1])
    dsw = np.append(ds[n:], ds[-1])
    gap = np.append(inv.wake_gap, inv.wake_gap[-1])
    dw, kw, tw, traw = _project(pts, wpoly)
    in_wake = (dw < y) & ~((kw == 0) & (traw <= 0.0))
    ue_k = _interp_nodes(uw, kw, tw)
    ds_k = _interp_nodes(dsw - gap, kw, tw)
    th_k = _interp_nodes(thw, kw, tw)
    hw = np.clip(ds_k / th_k, 1.0001, 10.0)
    depth = np.clip((1.0 - 1.0 / hw) / 0.75, 0.0, 0.95)
    half = np.minimum(ds_k / np.maximum(depth, 1e-6), 50.0 * ds_k)
    zeta = dw / half
    shape = np.where(zeta < 1.0, np.cos(0.5 * np.pi * np.minimum(zeta, 1.0)) ** 2, 0.0)
    defect = depth * shape
    wake_vel = (1.0 - defect)[:, None] * v_out
    vel = np.where(in_wake[:, None], wake_vel, vel)
    q2 = np.where(in_wake, np.sum(v_out**2, axis=1), q2)
    nut_w = CLAUSER * np.abs(ue_k) * ds_k * shape
    nut = np.where(in_wake, nut_w, nut)

    return IblField(
        velocity=vel * speed,
        dynamic=0.5 * speed**2 * (1.0 - q2),
        eddy_viscosity=nut * speed * chord,
        result=r,
        chord=chord,
    )


def wall_contours(mesh, name):
    """Return the closed node loops (contour order) of the boundary ``name``."""
    if name not in mesh.boundary_faces:
        raise MeshError(f"initial.boundary_layer: no boundary '{name}'")
    faces = mesh.face_nodes[mesh.boundary_faces[name]]
    at_node = {}
    for a, b in faces:
        at_node.setdefault(int(a), []).append(int(b))
        at_node.setdefault(int(b), []).append(int(a))
    if any(len(v) != 2 for v in at_node.values()):
        raise MeshError(
            f"initial.boundary_layer: boundary '{name}' is not made of closed "
            "contours (every node must belong to two of its faces)"
        )
    loops = []
    todo = set(at_node)
    while todo:
        start = min(todo)
        loop, prev, node = [start], None, start
        while True:
            a, b = at_node[node]
            nxt = a if a != prev else b
            if nxt == start:
                break
            loop.append(nxt)
            prev, node = node, nxt
        todo -= set(loop)
        loops.append(mesh.nodes[loop])
    return loops


def mesh_ibl_field(mesh, name, velocity, nu, ncrit=9.0, fully_turbulent=False):
    """:func:`ibl_field` at the cell centres of a mesh around one airfoil wall."""
    loops = wall_contours(mesh, name)
    if len(loops) != 1:
        raise MeshError(
            f"initial.boundary_layer: boundary '{name}' has {len(loops)} closed "
            "contours; the boundary-layer start needs a single airfoil"
        )
    faces = np.asarray(mesh.boundary_faces[name], dtype=int)
    d, nearest = mesh_wall_distance(mesh, faces)
    seg = mesh.nodes[mesh.face_nodes[nearest]]
    foot = _foot(mesh.cell_centroid, seg[:, 0], seg[:, 1])
    return ibl_field(
        loops[0],
        mesh.cell_centroid,
        d,
        foot,
        velocity,
        nu,
        ncrit=ncrit,
        fully_turbulent=fully_turbulent,
    )

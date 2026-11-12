"""Initial field from the potential flow around the walls of a case.

``initial.potential_flow`` names wall boundaries (airfoils). Their contours
are the bodies of the panel method (:mod:`pyaeroflow.numerics.panel`); the
velocity of that flow at the cell centres, with the pressure (and, for
compressible flow, the temperature) of an isentropic flow from the
freestream, replaces the uniform initial field. The lift and the stagnation
and suction regions are then present from the first iteration; the boundary
layers still have to form.
"""

import numpy as np

from pyaeroflow.mesh.mesh import MeshError
from pyaeroflow.numerics import panel
from pyaeroflow.physics import compressible as comp
from pyaeroflow.physics import incompressible as inc

# Largest speed of the initial field in units of the freestream speed: the
# potential flow is unbounded at convex corners.
SPEED_LIMIT = 3.0


def wall_contours(mesh, names):
    """Return the closed node loops of the named boundaries.

    Returns
    -------
    list of ndarray, shape (n, 2)
        One array of node coordinates per closed loop, in contour order.
    """
    loops = []
    for name in names:
        if name not in mesh.boundary_faces:
            raise MeshError(f"initial.potential_flow: no boundary '{name}'")
        faces = mesh.face_nodes[mesh.boundary_faces[name]]
        at_node = {}
        for a, b in faces:
            at_node.setdefault(int(a), []).append(int(b))
            at_node.setdefault(int(b), []).append(int(a))
        if any(len(v) != 2 for v in at_node.values()):
            raise MeshError(
                f"initial.potential_flow: boundary '{name}' is not made of closed "
                "contours (every node must belong to two of its faces)"
            )
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


def potential_velocity(mesh, names, velocity):
    """Potential-flow velocity at the cell centres.

    Parameters
    ----------
    mesh : Mesh
    names : sequence of str
        Wall boundaries whose closed contours are the bodies.
    velocity : (float, float)
        Freestream velocity.

    Returns
    -------
    ndarray, shape (n_cells, 2)
    PanelSolution
    """
    u = np.asarray(velocity, dtype=float)
    speed = float(np.hypot(*u))
    if speed == 0.0:
        raise ValueError("initial.potential_flow needs a freestream velocity")
    bodies = [panel.contour_body(loop, u) for loop in wall_contours(mesh, names)]
    solution = panel.solve(bodies, u)
    vel = panel.velocity(solution, mesh.cell_centroid)
    mag = np.linalg.norm(vel, axis=1)
    vel *= np.minimum(1.0, SPEED_LIMIT * speed / np.maximum(mag, 1e-300))[:, None]
    return vel, solution


def potential_state(problem, w0, names, velocity):
    """Replace the flow columns of the uniform state ``w0`` by potential flow.

    ``w0`` carries the freestream pressure (and temperature); turbulence
    columns are kept.
    """
    vel, solution = potential_velocity(problem.mesh, names, velocity)
    w = np.array(w0, dtype=float)
    q_inf = float(np.dot(velocity, velocity))
    q = np.sum(vel * vel, axis=1)
    if problem.compressible:
        gamma, r_gas = problem.model.gamma, problem.model.gas_constant
        cp = gamma * r_gas / (gamma - 1.0)
        t_inf = w[:, comp.T]
        t = t_inf + 0.5 * (q_inf - q) / cp
        w[:, comp.P] = w[:, comp.P] * (t / t_inf) ** (gamma / (gamma - 1.0))
        w[:, comp.T] = t
    else:
        w[:, inc.P] = w[:, inc.P] + 0.5 * problem.model.density * (q_inf - q)
    w[:, inc.U] = vel[:, 0]
    w[:, inc.V] = vel[:, 1]
    return w, solution

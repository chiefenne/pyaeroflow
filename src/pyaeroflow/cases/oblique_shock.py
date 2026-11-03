"""Verification case: supersonic flow over a compression ramp (L1, Euler).

A uniform supersonic stream at Mach M along a flat wall meets a ramp of
angle theta. The attached oblique shock from the ramp corner turns the flow
parallel to the ramp; behind it the state is uniform, given by the
theta-beta-M relation and the Rankine-Hugoniot conditions:

    tan(theta) = 2 cot(beta) (M^2 sin^2 beta - 1)
                 / (M^2 (gamma + cos 2 beta) + 2),
    p2 / p1 = 1 + 2 gamma / (gamma + 1) (M^2 sin^2 beta - 1).

The domain is x in [0, L], the wall y = 0 up to the corner at x0 and the
ramp behind it, the top at y = H; inflow and top are farfield boundaries
(freestream), the outflow at x = L is a farfield boundary through which the
supersonic flow leaves (the Roe flux then takes the interior state).
"""

import math
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq

from pyaeroflow.driver import FlowProblem
from pyaeroflow.mesh.generators import _quads, _rectangle_boundary_edges
from pyaeroflow.mesh.mesh import Mesh
from pyaeroflow.physics import compressible as comp
from pyaeroflow.solver import PseudoTimeSettings


def shock_angle(mach, theta, gamma=1.4):
    """Weak oblique-shock angle beta (radians) for deflection theta (radians)."""

    def f(beta):
        m2 = (mach * math.sin(beta)) ** 2
        return 2.0 / math.tan(beta) * (m2 - 1.0) / (
            mach**2 * (gamma + math.cos(2 * beta)) + 2.0
        ) - math.tan(theta)

    mu = math.asin(1.0 / mach)
    # The weak solution lies between the Mach angle and the angle of
    # maximum deflection; scan for the first sign change.
    betas = np.linspace(mu + 1e-9, 0.5 * math.pi - 1e-9, 2000)
    values = np.array([f(b) for b in betas])
    k = int(np.argmax(values > 0.0))
    if values[k] <= 0.0:
        raise ValueError("detached shock: theta exceeds the maximum deflection")
    return brentq(f, betas[k - 1], betas[k])


class ObliqueShock(NamedTuple):
    """Exact state behind the shock, relative to the freestream."""

    beta: float  # shock angle (radians)
    pressure_ratio: float
    density_ratio: float
    temperature_ratio: float
    mach: float  # downstream Mach number


def oblique_shock(mach, theta, gamma=1.4):
    """Rankine-Hugoniot state behind the weak oblique shock."""
    beta = shock_angle(mach, theta, gamma)
    mn2 = (mach * math.sin(beta)) ** 2
    p = 1.0 + 2.0 * gamma / (gamma + 1.0) * (mn2 - 1.0)
    rho = (gamma + 1.0) * mn2 / ((gamma - 1.0) * mn2 + 2.0)
    mn_down2 = (mn2 + 2.0 / (gamma - 1.0)) / (2.0 * gamma / (gamma - 1.0) * mn2 - 1.0)
    m_down = math.sqrt(mn_down2) / math.sin(beta - theta)
    return ObliqueShock(beta, p, rho, p / rho, m_down)


def ramp_mesh(nx, ny, *, angle=10.0, corner=0.25, length=1.25, height=1.0):
    """Structured quadrilateral mesh over a flat wall and a ramp of ``angle`` deg.

    Boundaries: wall (bottom, flat part and ramp), outflow (x = length),
    top (y = height) and inflow (x = 0).
    """
    xs = np.linspace(0.0, length, nx + 1)
    bottom = np.where(xs > corner, (xs - corner) * math.tan(math.radians(angle)), 0.0)
    s = np.linspace(0.0, 1.0, ny + 1)
    x = np.tile(xs, ny + 1)
    y = (bottom[None, :] + (height - bottom[None, :]) * s[:, None]).ravel()
    nodes = np.column_stack((x, y))
    names = ("wall", "outflow", "top", "inflow")
    boundary = dict(zip(names, _rectangle_boundary_edges(nx, ny), strict=True))
    return Mesh.from_cells(nodes, _quads(nx, ny), boundary)


class RampResult(NamedTuple):
    """Solution of the ramp case and its comparison with the exact shock."""

    problem: object
    w: np.ndarray
    exact: ObliqueShock
    pressure_ratio: float  # mean wall pressure on the ramp / p_inf
    mach: float  # mean Mach number of the ramp wall cells
    converged: bool
    history: dict


def run(
    nx=96,
    ny=64,
    mach=2.0,
    angle=10.0,
    spatial_order=2,
    limiter="venkatakrishnan-wang",
    settings=None,
    freeze_limiter_after_residual_drop=None,
):
    """Solve the ramp case and compare the ramp wall pressure with the exact one.

    The comparison uses the wall faces with 0.3 L_ramp < x - x0 < 0.9 L_ramp,
    clear of the corner (where the captured shock is a few cells thick) and
    of the outflow.
    """
    mesh = ramp_mesh(nx, ny, angle=angle)
    r_gas, farfield = comp.freestream_conditions(mach)
    model = comp.CompressibleModel(
        gas_constant=r_gas, spatial_order=spatial_order, limiter=limiter
    )
    conditions = {
        "wall": {"type": "slip-wall"},
        "inflow": farfield,
        "top": farfield,
        "outflow": farfield,
    }
    problem = FlowProblem(mesh, model, conditions)
    w0 = problem.initial_state(farfield["velocity"], farfield["pressure"])
    result = problem.solve(
        w0,
        settings or PseudoTimeSettings(residual_drop=10.0, max_iterations=200),
        freeze_limiter_after_residual_drop=freeze_limiter_after_residual_drop,
    )
    w = result.u
    faces = problem.boundary_faces(["wall"])
    xc = np.asarray(problem.geom.b_centre)[faces, 0]
    corner, length = 0.25, 1.25
    ramp = (xc - corner > 0.3 * (length - corner)) & (
        xc - corner < 0.9 * (length - corner)
    )
    loads = problem.wall_loads(w)
    p_wall = np.asarray(loads.pressure)[faces][ramp]
    owner = np.asarray(problem.geom.b_owner)[faces][ramp]
    mach_cells = np.asarray(comp.mach_number(model.gamma, model.gas_constant, w[owner]))
    return RampResult(
        problem,
        w,
        oblique_shock(mach, math.radians(angle), model.gamma),
        float(p_wall.mean() / farfield["pressure"]),
        float(mach_cells.mean()),
        bool(result.converged),
        result.history,
    )

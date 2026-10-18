"""Cylinder cases: V4 inviscid cylinder and B2 DFG benchmark 2D-1.

V4: potential flow around a cylinder of radius R in a uniform stream U,
C_p = 1 - 4 sin^2(theta), zero drag and lift. The Euler equations on an
O-grid with a slip wall on the cylinder and a farfield boundary approach this
solution under mesh refinement.

B2: laminar flow around a cylinder in a channel (Schaefer & Turek 1996,
benchmark 2D-1): channel 2.2 x 0.41, cylinder of diameter D = 0.1 at
(0.2, 0.2), parabolic inflow with U_max = 0.3 (mean velocity U_m = 0.2),
nu = 1e-3, Re = U_m D / nu = 20. Reference values (featflow, Nabh 1998):
c_D = 5.57953523384, c_L = 0.010618948146, Delta p = p(0.15, 0.2) -
p(0.25, 0.2) = 0.11752016697, coefficients with q = rho U_m^2 / 2 and D.
"""

from typing import NamedTuple

import numpy as np

from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import annulus, cylinder_in_channel
from pyaeroflow.physics import forces as fz
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.solver import DualTimeSettings, PseudoTimeSettings

DFG_REFERENCE = {"cd": 5.57953523384, "cl": 0.010618948146, "dp": 0.11752016697}
# 2D-2 (Re = 100, periodic shedding): intervals of Schaefer & Turek 1996 and
# their midpoints; dp at t0 + 1 / (2 f) with t0 a time of maximum lift.
DFG_2D2_REFERENCE = {
    "cd_max": (3.22, 3.24),
    "cl_max": (0.99, 1.01),
    "strouhal": (0.295, 0.305),
    "dp": (2.46, 2.50),
}


class InviscidResult(NamedTuple):
    """V4 result."""

    problem: object
    w: np.ndarray
    converged: bool
    history: dict
    theta: np.ndarray
    cp: np.ndarray
    cp_exact: np.ndarray
    cd: float
    cl: float


def inviscid(n_theta=64, n_radial=32, outer_radius=20.0, settings=None):
    """Euler flow around a cylinder on an O-grid (V4).

    Parameters
    ----------
    n_theta, n_radial : int
        O-grid resolution.
    outer_radius : float
        Farfield radius in cylinder radii (R = 0.5, so D = 1).
    settings : PseudoTimeSettings, optional

    Returns
    -------
    InviscidResult
        ``theta`` and ``cp`` per cylinder face, ``cp_exact`` = 1 - 4 sin^2.
    """
    radius = 0.5
    mesh = annulus(
        n_theta,
        n_radial,
        radius=radius,
        outer_radius=outer_radius * radius,
        growth=1.0 + 6.0 / n_radial,
    )
    conditions = {
        "cylinder": {"type": "slip-wall"},
        "farfield": {"type": "farfield", "velocity": (1.0, 0.0), "pressure": 0.0},
    }
    model = inc.IncompressibleModel(density=1.0, viscosity=0.0)
    problem = IncompressibleProblem(mesh, model, conditions)
    result = problem.solve(
        problem.initial_state((1.0, 0.0)),
        settings or PseudoTimeSettings(residual_drop=8.0, max_iterations=400),
    )
    faces = problem.boundary_faces(["cylinder"])
    q = 0.5
    d = fz.wall_distribution(problem.wall_loads(result.u), problem.geom, faces, q)
    theta = np.arctan2(d["y"], d["x"])
    c = fz.coefficients(problem.forces(result.u, ["cylinder"]), q, 2.0 * radius)
    return InviscidResult(
        problem,
        result.u,
        result.converged,
        result.history,
        theta,
        d["cp"],
        1.0 - 4.0 * np.sin(theta) ** 2,
        c.cd,
        c.cl,
    )


class DFGResult(NamedTuple):
    """B2 result with the reference values of benchmark 2D-1."""

    problem: object
    w: np.ndarray
    converged: bool
    history: dict
    cd: float
    cl: float
    dp: float
    reference: dict


def dfg_2d1(m=32, n_radial=20, cell_type="quad", settings=None):
    """DFG benchmark 2D-1 (Re = 20) on the block-structured mesh.

    Parameters
    ----------
    m : int
        Cells per side of the square around the cylinder (mesh resolution).
    n_radial : int
        Cells between cylinder and square.
    cell_type : {"quad", "triangle"}
    settings : PseudoTimeSettings, optional

    Returns
    -------
    DFGResult
    """
    height, u_max, nu, diameter = 0.41, 0.3, 1.0e-3, 0.1
    centre = (0.2, 0.2)
    mesh = cylinder_in_channel(m, n_radial, cell_type=cell_type, growth=1.12)

    def inflow(x, y):
        return 4.0 * u_max * y * (height - y) / height**2, 0.0 * y

    conditions = {
        "inlet": {"type": "velocity-inlet", "velocity": inflow},
        "outlet": {"type": "pressure-outlet", "pressure": 0.0},
        "wall": {"type": "wall"},
        "cylinder": {"type": "wall"},
    }
    model = inc.IncompressibleModel(density=1.0, viscosity=nu)
    problem = IncompressibleProblem(mesh, model, conditions)
    result = problem.solve(
        problem.initial_state((2.0 * u_max / 3.0, 0.0)),
        settings or PseudoTimeSettings(residual_drop=8.0, max_iterations=300),
    )
    u_mean = 2.0 * u_max / 3.0
    q = 0.5 * u_mean**2
    forces = problem.forces(result.u, ["cylinder"], centre)
    c = fz.coefficients(forces, q, diameter)
    loads = problem.wall_loads(result.u)
    faces = problem.boundary_faces(["cylinder"])
    front = surface_pressure(loads, problem.geom, faces, (centre[0] - 0.05, centre[1]))
    back = surface_pressure(loads, problem.geom, faces, (centre[0] + 0.05, centre[1]))
    return DFGResult(
        problem,
        result.u,
        result.converged,
        result.history,
        c.cd,
        c.cl,
        front - back,
        DFG_REFERENCE,
    )


def surface_pressure(loads, geom, faces, point):
    """Pressure at a point on a wall: inverse-distance of the two nearest faces."""
    centres = np.asarray(geom.b_centre)[faces]
    p = np.asarray(loads.pressure)[faces]
    d = np.linalg.norm(centres - np.asarray(point), axis=1)
    k = np.argsort(d)[:2]
    w = 1.0 / np.maximum(d[k], 1e-300)
    return float(np.sum(p[k] * w) / np.sum(w))


class DFGUnsteadyResult(NamedTuple):
    """B2 2D-2 result: time series and the periodic-state quantities."""

    problem: object
    w: np.ndarray
    converged: bool
    history: dict
    time: np.ndarray
    cd: np.ndarray
    cl: np.ndarray
    dp: np.ndarray
    cd_max: float
    cl_max: float
    strouhal: float
    dp_half_period: float
    periodic: bool
    reference: dict


def dfg_2d2(
    m=16,
    n_radial=12,
    cell_type="quad",
    time_step=0.02,
    end_time=8.0,
    inner=None,
    perturbation=0.1,
    callback=None,
):
    """DFG benchmark 2D-2 (Re = 100, vortex shedding) by dual time stepping.

    Parabolic inflow with U_max = 1.5 (U_m = 1), nu = 1e-3, D = 0.1. The run
    starts from the uniform mean velocity plus a small transverse blob behind
    the cylinder (``perturbation`` times U_m) so that the shedding, whose
    periodic state does not depend on the start, sets in early.

    ``callback(step, t, w, problem)`` is called after every time step (e.g.
    to write a field series).

    Returns
    -------
    DFGUnsteadyResult
        Lift and drag coefficients and the pressure difference per time
        step; ``strouhal`` f D / U_m from the lift maxima of the last
        periods, ``cd_max``, ``cl_max`` over the last period, ``dp_half_period``
        = dp(t0 + 1 / (2 f)) with t0 the last-but-one lift maximum;
        ``periodic`` if the last two lift maxima agree within 1 %.
    """
    height, u_max, nu, diameter = 0.41, 1.5, 1.0e-3, 0.1
    centre = (0.2, 0.2)
    mesh = cylinder_in_channel(m, n_radial, cell_type=cell_type, growth=1.12)

    def inflow(x, y):
        return 4.0 * u_max * y * (height - y) / height**2, 0.0 * y

    conditions = {
        "inlet": {"type": "velocity-inlet", "velocity": inflow},
        "outlet": {"type": "pressure-outlet", "pressure": 0.0},
        "wall": {"type": "wall"},
        "cylinder": {"type": "wall"},
    }
    model = inc.IncompressibleModel(density=1.0, viscosity=nu)
    problem = IncompressibleProblem(mesh, model, conditions)
    u_mean = 2.0 * u_max / 3.0
    w0 = problem.initial_state((u_mean, 0.0))
    xc, yc = mesh.cell_centroid.T
    w0[:, inc.V] = (
        perturbation
        * u_mean
        * np.exp(-((xc - 0.35) ** 2 + (yc - 0.25) ** 2) / (2.0 * 0.05**2))
    )
    q = 0.5 * u_mean**2
    faces = problem.boundary_faces(["cylinder"])
    front, back = (centre[0] - 0.05, centre[1]), (centre[0] + 0.05, centre[1])
    series = {"time": [], "cd": [], "cl": [], "dp": []}

    def record(step, t, w, *_):
        loads = problem.wall_loads(w)
        c = fz.coefficients(problem.forces(w, ["cylinder"], centre, loads), q, diameter)
        series["time"].append(t)
        series["cd"].append(c.cd)
        series["cl"].append(c.cl)
        series["dp"].append(
            surface_pressure(loads, problem.geom, faces, front)
            - surface_pressure(loads, problem.geom, faces, back)
        )
        if callback is not None:
            callback(step, t, w, problem)

    settings = DualTimeSettings(
        time_step, end_time, "bdf2", inner or DualTimeSettings._field_defaults["inner"]
    )
    result = problem.solve_unsteady(w0, settings, callback=record)
    t, cd, cl, dp = (np.asarray(series[k]) for k in ("time", "cd", "cl", "dp"))
    st, cd_max, cl_max, dp_half, periodic = shedding_quantities(
        t, cd, cl, dp, u_mean, diameter
    )
    return DFGUnsteadyResult(
        problem,
        result.u,
        result.converged,
        result.history,
        t,
        cd,
        cl,
        dp,
        cd_max,
        cl_max,
        st,
        dp_half,
        periodic,
        DFG_2D2_REFERENCE,
    )


def shedding_quantities(t, cd, cl, dp, velocity, diameter):
    """Strouhal number, cd_max, cl_max, dp(t0 + 1/(2f)) from the time series.

    Lift maxima are located by quadratic interpolation through the samples
    around each local maximum of ``cl``; the frequency is the inverse of the
    mean spacing of the last three maxima. Returns NaN values (and
    ``periodic = False``) with fewer than three maxima.
    """
    i = np.flatnonzero((cl[1:-1] > cl[:-2]) & (cl[1:-1] >= cl[2:])) + 1
    i = i[cl[i] > 0.0]
    nan = float("nan")
    if i.size < 3:
        return nan, nan, nan, nan, False
    i = i[-3:]
    # Quadratic interpolation of the peak time and value.
    d = 0.5 * (cl[i - 1] - cl[i + 1]) / (cl[i - 1] - 2.0 * cl[i] + cl[i + 1])
    t_peak = t[i] + d * (t[1] - t[0])
    cl_peak = cl[i] - 0.25 * (cl[i - 1] - cl[i + 1]) * d
    period = float(np.mean(np.diff(t_peak)))
    f = 1.0 / period
    periodic = abs(cl_peak[-1] - cl_peak[-2]) < 0.01 * abs(cl_peak[-1])
    last = (t >= t_peak[-2]) & (t <= t_peak[-1])
    cd_max = float(cd[last].max())
    dp_half = float(np.interp(t_peak[-2] + 0.5 * period, t, dp))
    return f * diameter / velocity, cd_max, float(cl_peak[-1]), dp_half, periodic

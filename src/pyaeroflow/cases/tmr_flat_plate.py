"""R1: turbulent flat plate (NASA TMR 2D zero-pressure-gradient case), SA-neg.

Domain of the TMR case: plate from x = 0 to x = 2 (reference length L = 1),
inflow at x = -1/3, top at y = 1, symmetry ahead of the plate, Re_L = 5e6,
farfield nu_tilde = 3 nu. TMR runs it at M = 0.2; the incompressible values
are expected to deviate slightly (compressibility lowers C_f by about 1 %
at M = 0.2). Compared: C_f at x = 0.97 and the drag coefficient of the plate
(one wetted side, normalised with q and the plate length 2, see
``TMR_REFERENCE``).

The structured mesh is wall-clustered geometrically (first spacing from
the target y+ with C_f ~ 0.003) and clustered towards the leading edge in x.
"""

from typing import NamedTuple

import numpy as np

from pyaeroflow.driver import IncompressibleProblem
from pyaeroflow.mesh import geometric_growth, split_boundary, structured
from pyaeroflow.physics import forces as fz
from pyaeroflow.physics import incompressible as inc
from pyaeroflow.solver import PseudoTimeSettings

# NASA TMR, flatplate_sa.html, finest grid 545 x 385 (N = 208896), M = 0.2,
# Re = 5e6 per unit length (data files FlatPlate/SA/cf_convergence.dat and
# drag_convergence.dat): C_f at x = 0.970084 is 0.0027056 (CFL3D) and
# 0.0027054 (FUN3D); C_D, reference area 2 (the whole plate), is 0.0028599
# (CFL3D) and 0.0028525 (FUN3D). The means of the two codes are used.
TMR_REFERENCE = {"cf_097": 0.0027055, "cd": 0.0028562}
# NASA TMR (now tmbwg.github.io/turbmodels), flatplate_sst.html, SST-Vm (on
# this plate the same as SST-2003m: vorticity = strain), finest grid, data
# files FlatPlate/SST/cf_convergence_sstv.dat and drag_convergence_sstv.dat:
# C_f(0.970084) 0.0026909 (CFL3D) and 0.0026905 (FUN3D), C_D 0.0028533 and
# 0.0028442; the means are used.
TMR_REFERENCE_SST = {"cf_097": 0.0026907, "cd": 0.0028488}


def plate_mesh(nx_plate=96, nx_front=24, ny=64, reynolds=5.0e6, y_plus=1.0, height=1.0):
    """Structured mesh of the TMR flat-plate domain.

    First wall spacing y1 = y_plus / (Re sqrt(C_f/2)) with C_f = 0.003;
    geometric growth to the top; x clustered towards the leading edge on both
    sides (tanh), uniform-ish far downstream.
    """
    u_tau = np.sqrt(0.003 / 2.0)
    y1 = y_plus / (reynolds * u_tau)
    r = geometric_growth(height, y1, ny)
    ys = np.concatenate(([0.0], np.cumsum(y1 * r ** np.arange(ny))))
    ys[-1] = height

    def tanh_cluster(n, length, beta, side):
        s = np.linspace(0.0, 1.0, n + 1)
        t = np.tanh(beta * s) / np.tanh(beta)  # clustered towards s = 0
        return length * (t if side == "start" else 1.0 - t[::-1])

    xs_front = -1.0 / 3.0 + tanh_cluster(nx_front, 1.0 / 3.0, 2.5, "end")
    xs_plate = tanh_cluster(nx_plate, 2.0, 2.5, "start")
    xs = np.concatenate((xs_front[:-1], xs_plate))
    mesh = structured(xs, ys)
    return split_boundary(
        mesh,
        "bottom",
        {"symmetry": lambda x, y: x < 0.0, "plate": lambda x, y: x >= 0.0},
    )


class TurbulentPlateResult(NamedTuple):
    """R1 result."""

    problem: object
    w: np.ndarray
    converged: bool
    history: dict
    x: np.ndarray
    cf: np.ndarray
    cf_097: float
    cd: float
    y_plus_max: float
    reference: dict


def solve(
    nx_plate=96,
    nx_front=24,
    ny=64,
    reynolds=5.0e6,
    coupling="segregated",
    settings=None,
    transition="none",
    turbulence_intensity=0.0,
    nu_tilde_ratio=3.0,
    critical_amplification=None,
    turbulence="sa-neg",
    k_omega=None,
):
    """Run the turbulent flat plate and evaluate C_f(x) and the drag.

    ``transition = "bcm"`` with ``turbulence_intensity`` (percent) runs the
    transitional plate (T3-series style) instead of the fully turbulent one;
    the BCM model needs a low freestream ``nu_tilde_ratio`` (0.1 instead of
    the fully turbulent 3), its second term reads existing eddy viscosity as
    turbulence. ``transition = "aft"`` runs SA-AFT2019b with N_crit from
    ``critical_amplification`` or, if None, from ``turbulence_intensity``.

    ``turbulence = "sst"`` runs k-omega SST with the freestream ``k_omega`` =
    (k, omega); default the NASA TMR SST values for this case, k = 9e-9 a^2
    at M = 0.2 (k = 2.25e-7 U^2, Tu = 0.039 %) and mu_t / mu = 0.009.
    """
    u_inf, nu = 1.0, 1.0 / reynolds
    nt_inf = nu_tilde_ratio * nu
    if turbulence == "sst":
        if k_omega is None:
            k_inf = 9.0e-9 / 0.2**2 * u_inf**2
            k_omega = (k_inf, k_inf / (0.009 * nu))
        turb = {"k": k_omega[0], "omega": k_omega[1]}
    else:
        turb = {"nu_tilde": nt_inf}
    mesh = plate_mesh(nx_plate, nx_front, ny, reynolds)
    conditions = {
        "left": {"type": "velocity-inlet", "velocity": (u_inf, 0.0), **turb},
        "right": {"type": "pressure-outlet", "pressure": 0.0},
        "top": {
            "type": "farfield",
            "velocity": (u_inf, 0.0),
            "pressure": 0.0,
            **turb,
        },
        "symmetry": {"type": "symmetry"},
        "plate": {"type": "wall"},
    }
    model = inc.IncompressibleModel(
        density=1.0,
        viscosity=nu,
        turbulence=turbulence,
        coupling=coupling,
        transition=transition,
        turbulence_intensity=turbulence_intensity,
        critical_amplification=critical_amplification,
    )
    problem = IncompressibleProblem(mesh, model, conditions)
    result = problem.solve(
        problem.initial_state((u_inf, 0.0), nu_tilde=nt_inf, k_omega=k_omega),
        settings or PseudoTimeSettings(residual_drop=8.0, max_iterations=400),
    )
    faces = problem.boundary_faces(["plate"])
    q = 0.5 * u_inf**2
    loads = problem.wall_loads(result.u)
    d = fz.wall_distribution(loads, problem.geom, faces, q)
    order = np.argsort(d["x"])
    x, cf = d["x"][order], d["cf"][order]
    cf_097 = float(np.interp(0.97, x, cf))
    forces = problem.forces(result.u, ["plate"], loads=loads)
    cd = fz.coefficients(forces, q, 2.0).cd
    # y+ of the first cell centres: y u_tau / nu with u_tau = sqrt(|tau_w| / rho).
    owner = np.asarray(problem.geom.b_owner)[faces]
    dist = np.asarray(problem.geom.wall_distance)[owner]
    tau = np.abs(np.asarray(loads.traction)[faces])
    tau_w = np.hypot(tau[:, 0], tau[:, 1])
    y_plus = dist * np.sqrt(tau_w) / nu
    return TurbulentPlateResult(
        problem,
        result.u,
        result.converged,
        result.history,
        x,
        cf,
        cf_097,
        cd,
        float(y_plus.max()),
        TMR_REFERENCE_SST if turbulence == "sst" else TMR_REFERENCE,
    )

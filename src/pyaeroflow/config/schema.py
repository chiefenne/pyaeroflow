"""Declarative schema of the case file (docs/configuration.md).

Every option has a type, a default (or ``REQUIRED``), a description and an
optional unit; the resolved configuration is written with these as comments
(D27, D30). Keys that exist in the documentation but are not implemented in
the current stage are listed in ``PLANNED`` and rejected with a message.
"""

from typing import NamedTuple

REQUIRED = object()

TURBULENCE_DESCRIPTION = (
    "Turbulence values (D33), one form: {turbulent_viscosity_ratio, intensity}, "
    "{kinetic_energy, length_scale, c_mu} or {nu_tilde_ratio}; "
    "default nu_tilde/nu = 3"
)
# Keys of the three input forms of a turbulence table.
TURBULENCE_FORMS = {
    "b": ("turbulent_viscosity_ratio", "intensity"),
    "a": ("kinetic_energy", "length_scale", "c_mu"),
    "c": ("nu_tilde_ratio",),
}


class Opt(NamedTuple):
    """One configuration option."""

    type: str  # "float", "int", "bool", "string", "float2", "string_list"
    default: object
    description: str
    unit: str = ""
    choices: tuple = ()
    minimum: float = None


SCHEMA = {
    "case": {
        "name": Opt("string", None, "Case name; default: file name without extension"),
        "description": Opt("string", "", "Free text"),
    },
    "mesh": {
        "file": Opt("string", REQUIRED, "Mesh file (.su2), relative to the case file"),
        "scale": Opt("float", 1.0, "Factor applied to all coordinates", minimum=0.0),
    },
    "physics": {
        "model": Opt(
            "string",
            REQUIRED,
            "Governing equations",
            choices=("euler", "laminar", "rans"),
        ),
        "turbulence": Opt(
            "string",
            None,
            "Turbulence model (rans); default sa-neg",
            choices=("sa-neg",),
        ),
        "transition": {
            "mode": Opt(
                "string",
                "none",
                "none = fully turbulent; prescribed = laminar regions",
                choices=("none", "prescribed"),
            ),
            "model": Opt(
                "string",
                "none",
                "Transition model: bcm = algebraic intermittency on the SA production "
                "(Cakmakcioglu et al., 2020 revision, with SA-noft2); combines with "
                "mode = prescribed",
                choices=("none", "bcm"),
            ),
            "turbulence_intensity": Opt(
                "float",
                None,
                "Freestream turbulence intensity Tu for the transition model, as a "
                "fraction (0.001 = 0.1 %); default: the intensity of the inflow "
                "turbulence table",
                minimum=0.0,
            ),
            "laminar_region": Opt(
                "table_list",
                None,
                "Regions without turbulence production: tables with boundary, "
                "optional side (upper/lower), x_min, x_max, y_min, y_max and "
                "ramp (blending length at the bounds)",
                "m",
            ),
            "boundary": Opt(
                "string", None, "Airfoil boundary of the by_alpha transition table"
            ),
            "ramp": Opt(
                "float",
                0.0,
                "Blending length of the by_alpha transition locations",
                "m",
                minimum=0.0,
            ),
            "by_alpha": Opt(
                "table_list",
                None,
                "Transition locations per angle of attack: tables {alpha, upper, "
                "lower} (deg, m from the leading edge), interpolated linearly in "
                "alpha to the case's angle; generates laminar_region",
            ),
        },
        "mean_pressure_gradient": Opt(
            "float2",
            None,
            "Driving force for periodic channels: mean pressure gradient "
            "(dp/dx, dp/dy), applied as the body force -grad p",
            "Pa/m",
        ),
    },
    "fluid": {
        "density": Opt("float", REQUIRED, "Density rho", "kg/m^3", minimum=0.0),
        "viscosity": Opt("float", None, "Dynamic viscosity mu", "Pa s", minimum=0.0),
        "reynolds_number": Opt(
            "float",
            None,
            "Re = rho U_ref L_ref / mu; alternative to viscosity",
            minimum=0.0,
        ),
    },
    "initial": {
        "velocity": Opt("float2", None, "Initial velocity (u, v)", "m/s"),
        "pressure": Opt("float", None, "Initial gauge pressure", "Pa"),
        "from_boundary": Opt(
            "string", None, "Uniform initial field equal to this boundary's state"
        ),
        "turbulence": Opt("table", None, TURBULENCE_DESCRIPTION),
        "restart": Opt(
            "string",
            None,
            "Restart file (<name>.restart.npz of a previous run on the same "
            "mesh), relative to the case file; overrides velocity/pressure",
        ),
    },
    "reference": {
        "length": Opt(
            "float", 1.0, "Reference length (chord, height)", "m", minimum=0.0
        ),
        "velocity": Opt(
            "float",
            None,
            "Reference velocity; default: farfield velocity if exactly one "
            "farfield boundary exists",
            "m/s",
            minimum=0.0,
        ),
        "density": Opt(
            "float", None, "Reference density; default fluid.density", "kg/m^3"
        ),
        "angle_of_attack": Opt(
            "float",
            None,
            "Flow direction for lift and drag; default: farfield angle of attack "
            "if exactly one farfield boundary exists, else 0",
            "deg",
        ),
        "moment_center": Opt(
            "float2", None, "Moment reference point; default (0.25 length, 0)", "m"
        ),
        "vortex_correction": Opt(
            "bool",
            False,
            "Point-vortex farfield correction: freestream plus the velocity "
            "induced by the lift on output.force_boundaries, vortex at "
            "moment_center (for farfields closer than ~500 chords)",
        ),
        "pressure_point": Opt(
            "float2",
            None,
            "Without a pressure-fixing boundary: pressure is pinned in the cell "
            "nearest this point; default domain centroid",
            "m",
        ),
    },
    "numerics": {
        "convective_flux": Opt("string", "fds", "Convective flux", choices=("fds",)),
        "spatial_order": Opt(
            "int", 2, "1: piecewise constant, 2: MUSCL", choices=(1, 2)
        ),
        "limiter": Opt(
            "string",
            "venkatakrishnan-wang",
            "Slope limiter",
            choices=("venkatakrishnan-wang", "venkatakrishnan", "none"),
        ),
        "limiter_threshold": Opt(
            "float",
            0.05,
            "Wang's epsilon' (suggested 0.01-0.20), or K of the local "
            "Venkatakrishnan threshold eps^2 = (K h)^3",
            minimum=0.0,
        ),
        "gradient": Opt(
            "string",
            "weighted-least-squares",
            "Gradient reconstruction",
            choices=("weighted-least-squares",),
        ),
        "gradient_weight_exponent": Opt(
            "float", 1.0, "q in the least-squares weights 1/d^q", minimum=0.0
        ),
        "viscous_gradient": Opt(
            "string",
            "alpha-damping",
            "Face-gradient scheme for viscous fluxes",
            choices=("alpha-damping", "edge-correction"),
        ),
        "alpha": Opt("float", 4.0 / 3.0, "alpha-damping coefficient", minimum=0.0),
        "turbulence_spatial_order": Opt(
            "int",
            1,
            "Spatial order of the turbulence equation (1: upwind)",
            choices=(1, 2),
        ),
        "first_order_iterations": Opt(
            "int",
            0,
            "Iterations at first order before switching to spatial_order",
            minimum=0,
        ),
        "limiter_freeze": {
            "after_iterations": Opt(
                "int", None, "Freeze the limiter after this many iterations", minimum=0
            ),
            "after_residual_drop": Opt(
                "float",
                None,
                "Freeze the limiter after this residual drop (orders)",
                minimum=0.0,
            ),
            "release": Opt(
                "bool",
                False,
                "Release the limiter (unlimited second order) instead of "
                "freezing it: the standard for subsonic flows after a limited "
                "start-up",
            ),
        },
        "beta_factor": Opt(
            "float",
            4.1,
            "Artificial compressibility: beta^2 = beta_factor * max|v|^2",
            minimum=0.0,
        ),
    },
    "solver": {
        "method": Opt(
            "string",
            "implicit",
            "Pseudo-time method",
            choices=("implicit", "explicit-rk"),
        ),
        "max_iterations": Opt(
            "int", 10000, "Maximum pseudo-time iterations", minimum=1
        ),
        "refactor_interval": Opt(
            "int",
            3,
            "Rebuild the factorized first-order system only every n-th iteration "
            "(lagged pseudo-time matrix, Newton-Krylov preconditioner); earlier "
            "after a rejected update, when the CFL has grown fourfold, or after a "
            "Newton step with more than 15 GMRES iterations",
            minimum=1,
        ),
        "residual_drop": Opt(
            "float",
            8.0,
            "Convergence: orders of magnitude of residual reduction",
            minimum=0.0,
        ),
        "cfl": {
            "start": Opt("float", 10.0, "Initial CFL number", minimum=0.0),
            "max": Opt("float", 1.0e10, "Maximum CFL number", minimum=0.0),
            "max_approximate": Opt(
                "float",
                1.0e3,
                "Maximum CFL number while the Newton-Krylov switch is pending "
                "(approximate first-order Jacobian); without Newton-Krylov, max "
                "applies throughout",
                minimum=0.0,
            ),
            "growth_factor": Opt(
                "float", 2.0, "Minimum CFL growth per accepted iteration", minimum=1.0
            ),
            "cut_factor": Opt(
                "float", 0.1, "CFL reduction after a rejected update", minimum=0.0
            ),
            "reject_factor": Opt(
                "float",
                10.0,
                "Reject an update that raises the residual by more",
                minimum=1.0,
            ),
            "explicit": Opt(
                "float", 1.0, "CFL number of the explicit Runge-Kutta mode", minimum=0.0
            ),
        },
        "newton_krylov": {
            "enabled": Opt(
                "bool",
                True,
                "Switch to Newton-Krylov (exact Jacobian-vector products, "
                "preconditioned by the first-order Jacobian)",
            ),
            "switch_at_residual_drop": Opt(
                "float",
                2.0,
                "Residual drop (orders of magnitude) at which to switch",
                minimum=0.0,
            ),
            "switch_at_iteration": Opt(
                "int", None, "Switch at this iteration at the latest", minimum=1
            ),
            "linear_tolerance": Opt(
                "float", 1.0e-2, "Relative GMRES tolerance per Newton step", minimum=0.0
            ),
            "restart": Opt("int", 50, "GMRES restart length", minimum=1),
            "max_iterations": Opt(
                "int", 200, "Maximum GMRES iterations per Newton step", minimum=1
            ),
        },
        "linear_solver": {
            "type": Opt(
                "string",
                "direct",
                "Sparse solver of the start-up phase, preconditioner of Newton-Krylov: "
                "SuperLU (direct), Intel MKL PARDISO (pardiso, needs pypardiso), "
                "UMFPACK (umfpack, needs scikit-umfpack), GMRES with incomplete LU",
                choices=("direct", "pardiso", "umfpack", "gmres-ilu"),
            ),
            "tolerance": Opt(
                "float", 1.0e-6, "Relative GMRES tolerance (gmres-ilu)", minimum=0.0
            ),
            "restart": Opt("int", 50, "GMRES restart length (gmres-ilu)", minimum=1),
            "max_iterations": Opt(
                "int", 500, "Maximum GMRES iterations (gmres-ilu)", minimum=1
            ),
            "ilu_drop_tolerance": Opt(
                "float", 1.0e-5, "Drop tolerance of the incomplete LU", minimum=0.0
            ),
            "ilu_fill_factor": Opt(
                "float", 20.0, "Fill factor of the incomplete LU", minimum=1.0
            ),
            "ordering": Opt(
                "string",
                "mmd-at-plus-a",
                "SuperLU column ordering; colamd for turbulence_coupling = coupled",
                choices=("mmd-at-plus-a", "colamd"),
            ),
        },
        "turbulence_coupling": Opt(
            "string",
            "segregated",
            "First-order Jacobian of RANS: segregated drops the mean-flow/"
            "turbulence coupling blocks (Newton-Krylov is coupled in both)",
            choices=("segregated", "coupled"),
        ),
    },
    "time": {
        "mode": Opt("string", "steady", "Time mode", choices=("steady", "unsteady")),
        "scheme": Opt(
            "string",
            "bdf2",
            "Physical time discretisation (the first step uses BDF1)",
            choices=("bdf2", "bdf1"),
        ),
        "time_step": Opt("float", None, "Physical time step", "s", minimum=0.0),
        "end_time": Opt("float", None, "Final time", "s", minimum=0.0),
        "inner_iterations": Opt(
            "int", 50, "Maximum pseudo-time iterations per time step", minimum=1
        ),
        "inner_residual_drop": Opt(
            "float",
            3.0,
            "Inner convergence per time step: orders of magnitude relative to "
            "the unsteady residual of the previous solution",
            minimum=0.0,
        ),
        "inner_cfl": Opt(
            "float",
            1.0e4,
            "Starting pseudo-time CFL number of the inner iterations",
            minimum=0.0,
        ),
    },
    "output": {
        "directory": Opt(
            "string", "results", "Output directory, relative to the case file"
        ),
        "formats": Opt("string_list", ["vtu"], "Field output formats"),
        "interval": Opt(
            "int",
            0,
            "Write fields every n iterations (steady) or time steps (unsteady); "
            "0 = only at the end",
            minimum=0,
        ),
        "history": Opt("bool", True, "Write the convergence history (CSV and .log)"),
        "force_boundaries": Opt(
            "string_list",
            None,
            "Boundaries over which forces and moments are integrated; "
            "default: all walls",
        ),
        "verbosity": Opt(
            "string", "normal", "Log detail", choices=("normal", "detailed")
        ),
    },
}

# Boundary-condition types and their parameters (D38).
BOUNDARY_TYPES = {
    "wall": {
        "velocity": Opt("float2", [0.0, 0.0], "Wall velocity (moving wall)", "m/s"),
    },
    "symmetry": {},
    "slip-wall": {},
    "velocity-inlet": {
        "velocity": Opt("float2", REQUIRED, "Inlet velocity (u, v)", "m/s"),
        "turbulence": Opt("table", None, TURBULENCE_DESCRIPTION),
    },
    "pressure-outlet": {
        "pressure": Opt("float", 0.0, "Static gauge pressure", "Pa"),
    },
    "total-pressure-inlet": {
        "total_pressure": Opt("float", REQUIRED, "Total gauge pressure", "Pa"),
        "flow_direction": Opt(
            "float2", REQUIRED, "Flow direction (d_x, d_y), normalised internally"
        ),
        "turbulence": Opt("table", None, TURBULENCE_DESCRIPTION),
    },
    "transpiration-wall": {
        "normal_velocity": Opt(
            "float",
            REQUIRED,
            "Velocity normal to the wall: positive = blowing into the domain, "
            "negative = suction",
            "m/s",
        ),
        "turbulence": Opt("table", None, TURBULENCE_DESCRIPTION),
    },
    "farfield": {
        "velocity": Opt("float", REQUIRED, "Freestream velocity magnitude", "m/s"),
        "angle_of_attack": Opt("float", 0.0, "Angle of attack", "deg"),
        "pressure": Opt("float", 0.0, "Freestream gauge pressure", "Pa"),
        "turbulence": Opt("table", None, TURBULENCE_DESCRIPTION),
    },
    "periodic": {
        "partner": Opt("string", REQUIRED, "Name of the matching boundary faces"),
        "translation": Opt(
            "float2", REQUIRED, "Translation from this boundary to the partner", "m"
        ),
    },
}

# Documented keys of later stages: rejected with a pointer to the stage.
PLANNED = {}
PLANNED_BOUNDARY_TYPES = {}

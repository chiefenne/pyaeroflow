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
    "{kinetic_energy, length_scale, c_mu} or {nu_tilde_ratio} (SA only); "
    "default nu_tilde/nu = 3 (SA), Tu = 0.1 % and nu_t/nu = 1 (SST)"
)
# Keys of the three input forms of a turbulence table.
TURBULENCE_FORMS = {
    "b": ("turbulent_viscosity_ratio", "intensity"),
    "a": ("kinetic_energy", "length_scale", "c_mu"),
    "c": ("nu_tilde_ratio",),
}


class Opt(NamedTuple):
    """One configuration option."""

    type: str  # "float", "int", "bool", "string", "float2", "string_list", "int_list"
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
        "file": Opt("string", REQUIRED, "Mesh file (.su2 or .msh), relative to case"),
        "scale": Opt("float", 1.0, "Factor applied to all coordinates", minimum=0.0),
    },
    "physics": {
        "flow": Opt(
            "string",
            "incompressible",
            "Flow family: incompressible (artificial compressibility, "
            "constant density) or compressible (density-based, ideal gas; "
            "physics.model = euler or laminar so far)",
            choices=("incompressible", "compressible"),
        ),
        "model": Opt(
            "string",
            REQUIRED,
            "Governing equations",
            choices=("euler", "laminar", "rans"),
        ),
        "turbulence": Opt(
            "string",
            None,
            "Turbulence model (rans): sa-neg (Spalart-Allmaras, negative "
            "variant) or sst (Menter k-omega SST-2003m, NASA TMR); default sa-neg",
            choices=("sa-neg", "sst"),
        ),
        "transition": {
            "clip": Opt(
                "bool",
                True,
                "model = lm: clip gamma to [1e-4, 5] and Re_thetat to [1e-4, "
                "1e15] after every update; false leaves them "
                "unclipped (a cell can then run to gamma = 1e-142 and stall "
                "the gamma residual)",
            ),
            "newton_guard": Opt(
                "bool",
                False,
                "model = lm: descent guard. A Newton update of gamma that would "
                "end below the clip floor while the residual asks for a larger "
                "gamma is replaced by the explicit local step -R dt/area at CFL 1 "
                "(at the floor the sqrt(gamma) derivative of the production turns "
                "the diagonal of the cell negative and Newton steps away from the "
                "root). Treats the symptom; on the T3A start-up it was worse than "
                "plain Newton and positive_jacobian alone was better, so it is off "
                "by default",
            ),
            "cfl_max": Opt(
                "float",
                None,
                "model = lm: largest pseudo-time CFL number of the gamma and "
                "Re_thetat equations (implicit modes); the flow and k-omega rows "
                "keep the solver CFL. Like a segregated transition solver with "
                "its own CFL: at a high CFL the Newton update of gamma "
                "overshoots (the full step raised the gamma residual 10-50x on "
                "the 320k SD7003 at CFL >= 2560) and the line search then cuts "
                "the good flow update to a quarter as well. Default: no cap",
                minimum=0.0,
            ),
            "positive_jacobian": Opt(
                "bool",
                True,
                "model = lm: the Newton-Krylov Jacobian leaves out the "
                "derivatives of the gamma source that lower the diagonal of the "
                "gamma equation (the sqrt(gamma) of the production, the "
                "destruction below gamma = 0.01); the residual is unchanged, the "
                "step is a quasi-Newton step. The first-order Jacobian always does "
                "this. false: exact linearization (the state before 1.1)",
            ),
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
                "(Cakmakcioglu et al., 2020 revision, with SA-noft2); aft = "
                "SA-AFT2019b (Coder: transport equations for the amplification "
                "factor and the log-intermittency); lm = Langtry-Menter gamma-Re_theta "
                "on SST (SST-2003m-LM2009, NASA TMR, Tu from the inflow k); "
                "combines with mode = prescribed",
                choices=("none", "bcm", "aft", "lm"),
            ),
            "turbulence_intensity": Opt(
                "float",
                None,
                "Freestream turbulence intensity Tu for the transition model, as a "
                "fraction (0.001 = 0.1 %); default: the intensity of the inflow "
                "turbulence table",
                minimum=0.0,
            ),
            "critical_amplification": Opt(
                "float",
                None,
                "N_crit of model = aft (e^N method); default: from "
                "turbulence_intensity by Mack's relation with Drela's limit "
                "(8.15 at 0.1 %)",
                minimum=0.0,
            ),
            "relaxation": Opt(
                "float",
                1.0,
                "Under-relaxation omega in (0, 1] of the BCM intermittency "
                "in steady runs: gamma += omega (gamma_model - gamma) per accepted "
                "iteration, so onset ramps in over ~1/omega iterations; 1 = off. "
                "The converged state is that of the unrelaxed model",
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
        "density": Opt(
            "float",
            None,
            "Density rho (incompressible: required; compressible: from the "
            "farfield pressure and temperature)",
            "kg/m^3",
            minimum=0.0,
        ),
        "gamma": Opt(
            "float", 1.4, "Ratio of specific heats (compressible)", minimum=1.0
        ),
        "gas_constant": Opt(
            "float",
            287.058,
            "Specific gas constant R (compressible)",
            "J/(kg K)",
            minimum=0.0,
        ),
        "viscosity": Opt(
            "float",
            None,
            "Dynamic viscosity mu (compressible: at the reference temperature)",
            "Pa s",
            minimum=0.0,
        ),
        "reynolds_number": Opt(
            "float",
            None,
            "Re = rho U_ref L_ref / mu; alternative to viscosity (compressible: "
            "of the reference state; without both, mu follows from Sutherland's "
            "law at the reference temperature)",
            minimum=0.0,
        ),
        "viscosity_model": Opt(
            "string",
            "sutherland",
            "Compressible: mu(T) by Sutherland's law (scaled to fluid.viscosity "
            "or reynolds_number at the reference temperature) or constant",
            choices=("sutherland", "constant"),
        ),
        "sutherland_temperature": Opt(
            "float",
            110.4,
            "Sutherland's constant S (compressible)",
            "K",
            minimum=0.0,
        ),
        "prandtl": Opt(
            "float",
            0.72,
            "Laminar Prandtl number (compressible: conductivity mu c_p / Pr)",
            minimum=0.0,
        ),
        "prandtl_turbulent": Opt(
            "float",
            0.9,
            "Turbulent Prandtl number (compressible RANS: eddy conductivity "
            "mu_t c_p / Pr_t)",
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
            "Restart or backup file (<name>.backup_<n>.npz or "
            "<name>.restart_a.npz of a previous run on the same mesh), relative "
            "to the case file; overrides velocity/pressure",
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
            "float",
            None,
            "Reference density; default fluid.density (compressible: the "
            "farfield density)",
            "kg/m^3",
        ),
        "temperature": Opt(
            "float",
            None,
            "Reference temperature (compressible); default: the farfield "
            "temperature if exactly one farfield stream exists",
            "K",
            minimum=0.0,
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
        "convective_flux": Opt(
            "string",
            None,
            "Convective flux: fds (incompressible, flux-difference splitting "
            "with low-speed preconditioning), or compressible: roe (Roe with "
            "Harten's entropy fix) or slau2 (Kitamura and Shima 2013, "
            "all-speed, no tuning parameter); default by physics.flow",
            choices=("fds", "roe", "slau2"),
        ),
        "entropy_fix": Opt(
            "float",
            0.001,
            "Roe flux: Harten's entropy fix, delta = entropy_fix (|v_n| + c) "
            "(Harten 1983); 0 = off",
            minimum=0.0,
        ),
        "jacobian_flux": Opt(
            "string",
            "auto",
            "Flux of the first-order Jacobian (compressible): auto = the "
            "convective flux (slau2 or roe), roe = always plain Roe. Only "
            "changes the preconditioner and the Newton step, not the "
            "converged solution",
            choices=("auto", "roe"),
        ),
        "low_mach_correction": Opt(
            "bool",
            False,
            "Roe flux (compressible; not used by slau2): Rieper's low-Mach "
            "correction of the face velocities, z = min(max(M_L, M_R), 1) "
            "(Rieper 2011); reduces the excess pressure dissipation of the "
            "Roe scheme at low Mach numbers (about M < 0.3)",
        ),
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
        "whole_cell_relaxation": Opt(
            "bool",
            False,
            "SST: a cell whose k-omega update the positivity limiter scales "
            "(update would lower k or omega by more than 99 %) gets the same "
            "scale on all its variables, so that its coupled update stays "
            "consistent; false scales k and omega only",
        ),
        "line_search_merit": Opt(
            "string",
            "all",
            "Residuals whose RMS the line search, the step acceptance "
            "(cfl.reject_factor) and the SER CFL control judge: all, or "
            "no_transition (leaves out the Langtry-Menter gamma and Re_thetat "
            "rows; their raw residual is largest in the big outlet and "
            "farfield cells and vetoes good flow steps). The convergence "
            "measure (residual_drop) always covers all equations. Default "
            "no_transition with physics.transition.model = lm",
            choices=("all", "no_transition"),
        ),
        "step_control": Opt(
            "string",
            "merit",
            "Step acceptance (implicit): merit (halve a step that raises the "
            "steady residual RMS, reject above cfl.reject_factor) or unsteady "
            "(HANIM, Wang et al. 2021: a step must lower the pseudo-time "
            "residual a T du + R(u + a du); quadratic under-relaxation a, a "
            "partial step holds the CFL, a failed step is retried at "
            "cfl.cut_factor with the existing factorization)",
            choices=("merit", "unsteady"),
        ),
        "step_alpha_min": Opt(
            "float",
            0.1,
            "step_control = unsteady: smallest under-relaxation a step is "
            "accepted with",
            minimum=0.0,
        ),
        "source_time_step": {
            "limit": Opt(
                "float",
                None,
                "RANS: source-term time-step limit (Piotrowski & Zingg 2020): "
                "the local pseudo-time step of every cell obeys "
                "lambda_max(dS/dQ) dt <= limit, lambda_max the largest growth "
                "rate of the cell-local turbulence/transition source Jacobian "
                "(production included); 0.9 in the paper. Off by default",
                minimum=0.0,
            ),
            "release_after": Opt(
                "int",
                5,
                "Switch the limit off after this many consecutive full "
                "Newton-Krylov steps; any cut or rejected step arms it again",
                minimum=1,
            ),
        },
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
        "convergence": {
            "fields": Opt(
                "string_list",
                [],
                "Cauchy convergence on force coefficients: "
                "the run also stops (converged) when every listed coefficient "
                "(cl, cd, cm) has changed by less than cauchy_eps (relative "
                "change between iterations) on average over the last "
                "cauchy_elements iterations; empty = residual drop only",
            ),
            "cauchy_elements": Opt(
                "int",
                100,
                "Iterations in the Cauchy window",
                minimum=2,
            ),
            "cauchy_eps": Opt(
                "float",
                1.0e-10,
                "Cauchy tolerance on the mean relative change of a coefficient "
                "per iteration, |C_n - C_(n-1)| / |C_n|",
                minimum=0.0,
            ),
            "start_iteration": Opt(
                "int",
                0,
                "Cauchy values count only from this iteration",
                minimum=0,
            ),
            "min_residual_drop": Opt(
                "float",
                0.1,
                "Guard of the Cauchy criterion: an iteration counts in the "
                "window only if its residual drop (the drop column of the log: "
                "the worst equation relative to its own peak, in orders) is at "
                "least this; forces that stand still while the residual sits at "
                "its peak do not stop the run. 0 = the plain Cauchy criterion",
                minimum=0.0,
            ),
        },
        "cfl": {
            "start": Opt("float", 10.0, "Initial CFL number", minimum=0.0),
            "max": Opt(
                "float",
                1.0e10,
                "Maximum CFL number (Langtry-Menter: default 2000)",
                minimum=0.0,
            ),
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
            "floor_exponent": Opt(
                "float",
                None,
                "SER floor: after an accepted step, and with no failed step in "
                "the last 5 iterations, CFL >= start (R_0 / R)^p (R the residual "
                "RMS, R_0 that of the start), bounded by max (max_approximate "
                "before the Newton-Krylov switch); 0.5 is a common choice. Off "
                "by default",
                minimum=0.0,
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
            "control": Opt(
                "string",
                "growth",
                "CFL control: growth (SER with at least growth_factor, adaptive "
                "cap from the GMRES effort) or ser (proportional SER on the RMS "
                "norm, bounded per step, GMRES-modulated)",
                choices=("growth", "ser"),
            ),
            "polar_reset": Opt(
                "bool",
                False,
                "pyaeroflow polar: every angle after the first restarts the CFL "
                "ramp at start (the Newton-Krylov switch and limiter are kept) "
                "instead of continuing at the previous angle's CFL; for cases "
                "where the new angle moves a laminar bubble",
            ),
            "ser_exponent": Opt(
                "float", 1.0, "Exponent of the SER ratio (control = ser)", minimum=0.0
            ),
            "ser_growth_min": Opt(
                "float",
                1.2,
                "Minimum CFL factor while the residual falls (ser)",
                minimum=1.0,
            ),
            "ser_ratio_min": Opt(
                "float",
                0.3,
                "Lower bound of the CFL factor per accepted iteration (ser)",
                minimum=0.0,
            ),
            "ser_ratio_max": Opt(
                "float",
                2.0,
                "Upper bound of the CFL factor per accepted iteration (ser)",
                minimum=1.0,
            ),
            "ser_gmres_hold": Opt(
                "int",
                10,
                "Newton steps with this many GMRES iterations or more hold the CFL (ser)",
                minimum=1,
            ),
            "ser_gmres_reduce": Opt(
                "int",
                30,
                "Newton steps with more GMRES iterations than this halve the CFL (ser)",
                minimum=1,
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
            "variable_scaling": Opt(
                "bool",
                False,
                "Measure the GMRES error of each variable relative to its size "
                "(unknowns du_j / s_j: s = 1 for p, u, v, the field RMS for the "
                "turbulence and transition variables). The left-preconditioned "
                "GMRES otherwise minimizes the error in raw units and leaves the "
                "small variables unsolved: on the 320k SD7003 LM case 95 % of the "
                "gamma residual stayed in the linear model while Re_thetat (~1e3) "
                "was solved. Default true with physics.transition.model = lm",
            ),
            "linearize": Opt(
                "bool",
                True,
                "Linearize the residual once per Newton step (jax.linearize) and "
                "apply the stored tangent in every GMRES product, instead of a "
                "full jax.jvp (primal and tangent) per product: products cost "
                "about a quarter; the stored intermediates take about 1.4 GB at "
                "320k cells with Langtry-Menter during the GMRES solve. false "
                "uses jax.jvp per product (no extra memory, slower)",
            ),
            "cfl_adapt_iterations": Opt(
                "int",
                60,
                "CFL control (growth) in the Newton-Krylov phase: a step that "
                "needs this many GMRES iterations cuts the CFL cap to a quarter, "
                "a step that needs a quarter of it or fewer lets the cap grow "
                "(x4, up to cfl.max); in between the cap is held. A larger "
                "value allows a higher CFL where GMRES works moderately hard "
                "(stalled steps of 20-50 iterations hold the CFL near 1e3 with "
                "the default). Default 150 with physics.transition.model = lm",
                minimum=4,
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
        "nu_tilde_step_limit": Opt(
            "float",
            None,
            "SA, implicit and unsteady (inner iterations): cap every update "
            "of nu_tilde to this multiple of max(|nu_tilde|, nu) per cell "
            "before the line search (e.g. 1 = at most double); off by default. "
            "Stops a Newton step from multiplying nu_tilde in one "
            "boundary-layer cell",
            minimum=0.0,
        ),
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
        "convergence": {
            "fields": Opt(
                "string_list",
                [],
                "Cauchy convergence of the windowed time average of force "
                "coefficients (cl, cd, cm): the run stops after the current step when "
                "every listed time-averaged coefficient has changed by less "
                "than cauchy_eps on average over the last cauchy_elements "
                "steps; empty = off (the run goes to end_time)",
            ),
            "cauchy_elements": Opt(
                "int",
                100,
                "Steps in the Cauchy window",
                minimum=2,
            ),
            "cauchy_eps": Opt(
                "float",
                1.0e-3,
                "Cauchy tolerance on the mean relative change of the time "
                "average per step",
                minimum=0.0,
            ),
            "start_step": Opt(
                "int",
                0,
                "Steps to wait after window_start_step before the Cauchy "
                "elements count",
                minimum=0,
            ),
            "window_start_step": Opt(
                "int",
                0,
                "Time step from which the coefficients are averaged",
                minimum=0,
            ),
            "window_function": Opt(
                "string",
                "square",
                "Weights of the time average over the steps from "
                "window_start_step to the last step",
                choices=("square", "hann", "hann_square", "bump"),
            ),
        },
    },
    "output": {
        "directory": Opt(
            "string", "results", "Output directory, relative to the case file"
        ),
        "formats": Opt("string_list", ["vtu"], "Field output formats"),
        "fields": Opt(
            "string_list",
            None,
            "Cell fields written to the VTU files (D59, field table in "
            'docs/configuration.md); "all" = every field the case provides; '
            "default: pressure, velocity, velocity_magnitude and, with a "
            "turbulence model, nu_tilde, eddy_viscosity_ratio and the "
            "transition fields",
        ),
        "fields_interval": Opt(
            "int",
            0,
            "Write the fields every n iterations (steady) or time steps "
            "(unsteady), to <name>_<number>.vtu; 0 = only at the end",
            minimum=0,
        ),
        "fields_at": Opt(
            "int_list",
            [],
            "Also write the fields at these iterations (steady) or time steps "
            "(unsteady), in addition to fields_interval",
            minimum=1,
        ),
        "fields_final": Opt(
            "bool", True, "Write the final fields to <name>.vtu at the end of the run"
        ),
        "interval": Opt(
            "int",
            0,
            "Same as fields_interval (kept for older case files); do not set both",
            minimum=0,
        ),
        "history": Opt("bool", True, "Write the convergence history (CSV and .log)"),
        "restart_interval": Opt(
            "int",
            0,
            "Write a restart file every n iterations (steady) or time steps "
            "(unsteady), alternately to <name>.restart_a.npz and "
            "<name>.restart_b.npz (each write replaces the older one); "
            "0 = none (the final state is always written as a backup)",
            minimum=0,
        ),
        "restart_backup_interval": Opt(
            "int",
            0,
            "Write a backup <name>.backup_<number>.npz that is never "
            "replaced every n iterations (steady) or time steps (unsteady); "
            "0 = none (the final state is always written as a backup)",
            minimum=0,
        ),
        "restart_backup_at": Opt(
            "int_list",
            [],
            "Also write a backup <name>.backup_<number>.npz at "
            "these iterations (steady) or time steps (unsteady), in addition "
            "to restart_backup_interval",
            minimum=1,
        ),
        "force_boundaries": Opt(
            "string_list",
            None,
            "Boundaries over which forces and moments are integrated; "
            "default: all walls",
        ),
        "transition_diagnostics": Opt(
            "bool",
            False,
            "Add intermittency statistics to the history CSV (cells with "
            "gamma > 0.01, cells that switched, largest change with its cell "
            "centre; steady runs also |R_nu|), counted within 0.05 chords of "
            "a wall; needs physics.transition.model",
        ),
        "verbosity": Opt(
            "string", "normal", "Log detail", choices=("normal", "detailed")
        ),
        "debug_log": Opt(
            "bool",
            False,
            "Steady implicit runs: write <name>.debug.log with one numerics "
            "block per iteration (GMRES tolerance and true linear residual, "
            "step-limiter counts, largest updates, every line-search trial per "
            "variable, reference re-evaluations, cells with the largest "
            "residuals); costs one Jacobian product per Newton step",
        ),
    },
}

# Boundary-condition types and their parameters (D38).
BOUNDARY_TYPES = {
    "wall": {
        "velocity": Opt("float2", [0.0, 0.0], "Wall velocity (moving wall)", "m/s"),
        "temperature": Opt(
            "float",
            None,
            "Wall temperature (compressible laminar: isothermal wall; default "
            "adiabatic)",
            "K",
            minimum=0.0,
        ),
    },
    "symmetry": {},
    "slip-wall": {},
    "velocity-inlet": {
        "velocity": Opt("float2", REQUIRED, "Inlet velocity (u, v)", "m/s"),
        "temperature": Opt(
            "float", None, "Static temperature (compressible, required)", "K"
        ),
        "turbulence": Opt("table", None, TURBULENCE_DESCRIPTION),
    },
    "pressure-outlet": {
        "pressure": Opt(
            "float",
            None,
            "Static pressure: gauge (incompressible, default 0) or absolute "
            "(compressible, required)",
            "Pa",
        ),
    },
    "total-pressure-inlet": {
        "total_pressure": Opt(
            "float",
            REQUIRED,
            "Total pressure: gauge (incompressible) or absolute (compressible)",
            "Pa",
        ),
        "total_temperature": Opt(
            "float", None, "Total temperature (compressible, required)", "K"
        ),
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
        "velocity": Opt(
            "float",
            None,
            "Freestream velocity magnitude (incompressible, required)",
            "m/s",
        ),
        "mach": Opt(
            "float",
            None,
            "Freestream Mach number (compressible, required)",
            minimum=0.0,
        ),
        "angle_of_attack": Opt("float", 0.0, "Angle of attack", "deg"),
        "pressure": Opt(
            "float",
            None,
            "Freestream pressure: gauge (incompressible, default 0) or absolute "
            "(compressible, default 101325)",
            "Pa",
        ),
        "temperature": Opt(
            "float",
            None,
            "Freestream static temperature (compressible, default 288.15)",
            "K",
            minimum=0.0,
        ),
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

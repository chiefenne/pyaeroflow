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

    # "float", "int", "bool", "string", "float2", "float2_list", "string_list",
    # "int_list", "float_list", "table", "table_list"
    type: str
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
            "jacobian_smoothing": Opt(
                "float",
                0.0,
                "model = lm: width w of smooth switches in the Newton-Krylov "
                "Jacobian and the preconditioner. The max / min / clip of the "
                "onset (F_onset, F_onset1-3), separation (gamma_sep), blending "
                "(F_thetat, gamma_eff, k destruction) and Re_thetat^eq "
                "functions get the slope of 0.5 (a + b +- sqrt((a-b)^2 + w^2)), "
                "and sqrt(gamma F_onset) that of sqrt(max(x, 0) + w^2), bounded "
                "by 1/(2w). The residual and the converged solution are "
                "unchanged; the step is a quasi-Newton step. Try 0.05-0.2; "
                "0 = exact switches",
                minimum=0.0,
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
        "energy": Opt(
            "bool",
            False,
            "Incompressible: solve the energy equation, strongly coupled "
            "(temperature column, density from fluid.density_model, mu, k and "
            "c_p of T; laminar or RANS, steady or unsteady). false: the "
            "constant-density solver",
        ),
        "gravity": Opt(
            "float2",
            None,
            "Gravitational acceleration (g_x, g_y) of the buoyancy source "
            "(rho - rho_ref) g, rho_ref the density at reference.temperature "
            "(physics.energy); the pressure is then p - rho_ref g.x",
            "m/s^2",
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
            None,
            "mu(T): compressible sutherland (default, scaled to fluid.viscosity "
            "or reynolds_number at the reference temperature) or constant; "
            "incompressible with physics.energy also power-law, polynomial or "
            "table (default constant), parameters in viscosity_parameters",
            choices=("sutherland", "constant", "power-law", "polynomial", "table"),
        ),
        "viscosity_parameters": Opt(
            "table",
            None,
            "Parameters of viscosity_model (physics.energy): sutherland "
            "{temperature = S} (default sutherland_temperature), power-law "
            "{exponent = n} (mu = mu_ref (T/T_ref)^n), polynomial {coefficients "
            "= [a0, a1, ...]} (mu = sum a_i T^i, T in K), table {temperature = "
            "[...], value = [...]} (linear, constant outside)",
        ),
        "density_model": Opt(
            "string",
            "constant",
            "Incompressible with physics.energy: equation of state rho(p_th, T) "
            "(p_th see thermodynamic_pressure): constant (fluid.density; passive "
            "temperature), boussinesq (fluid.density (1 - beta (T - T_ref)) in "
            "every term) or ideal_gas (operating_pressure / (R T))",
            choices=("constant", "boussinesq", "ideal_gas"),
        ),
        "species": Opt(
            "table_list",
            None,
            "physics.energy (laminar or RANS): transported species of a mixture (the "
            "fluid of the case is the bulk species with 1 - sum Y_k). One table "
            "per species: name; molar_mass (kg/kmol, default that of the fluid); "
            "diffusivity (m^2/s in the fluid at the reference temperature and "
            "operating pressure) with the optional diffusivity_exponent n (D ~ "
            "T^n), or schmidt (rho D = mu / Sc); optional specific_heat (J/(kg "
            "K), default that of the fluid), formation_enthalpy (J/kg, relative "
            "to the fluid) and expansion (density_model = boussinesq: "
            "-(1/rho) d(rho)/dY); evaporating walls: latent_heat (J/kg) and "
            "saturation_model = antoine (antoine_a, antoine_b, antoine_c: "
            "log10(p_sat / bar) = A - B / (T + C), T in K, as the NIST WebBook) "
            "or clausius-clapeyron (saturation_pressure in Pa at "
            "saturation_temperature in K, with latent_heat and molar_mass)",
        ),
        "molar_mass": Opt(
            "float",
            None,
            "Molar mass of the fluid (the bulk species of fluid.species); "
            "default 8314.46 / gas_constant",
            "kg/kmol",
            minimum=0.0,
        ),
        "thermal_expansion": Opt(
            "float",
            None,
            "density_model = boussinesq: expansion coefficient beta; default "
            "1 / reference.temperature",
            "1/K",
        ),
        "operating_pressure": Opt(
            "float",
            101325.0,
            "density_model = ideal_gas: thermodynamic pressure p_th (the "
            "initial value in a closed domain, see thermodynamic_pressure)",
            "Pa",
            minimum=0.0,
        ),
        "thermodynamic_pressure": Opt(
            "string",
            "auto",
            "density_model = ideal_gas: constant (p_th = operating_pressure) or "
            "closed (closed domain: p_th follows from the total mass of the gas, "
            "filled at operating_pressure and initial.temperature, so that the "
            "mass is conserved); auto: closed when no boundary exchanges mass "
            "(walls, slip walls, symmetry, periodic, transpiration walls at zero "
            "velocity), else constant",
            choices=("auto", "constant", "closed"),
        ),
        "specific_heat": Opt(
            "float",
            None,
            "physics.energy: c_p at the reference temperature; default "
            "gamma R / (gamma - 1) of fluid.gamma and gas_constant",
            "J/(kg K)",
            minimum=0.0,
        ),
        "specific_heat_model": Opt(
            "string",
            "constant",
            "physics.energy: c_p(T) law (as viscosity_model); the enthalpy is "
            "its exact integral",
            choices=("constant", "sutherland", "power-law", "polynomial", "table"),
        ),
        "specific_heat_parameters": Opt(
            "table", None, "Parameters of specific_heat_model (as viscosity_parameters)"
        ),
        "conductivity": Opt(
            "float",
            None,
            "physics.energy: thermal conductivity k at the reference temperature; "
            "default k = mu c_p / Pr from fluid.prandtl",
            "W/(m K)",
            minimum=0.0,
        ),
        "conductivity_model": Opt(
            "string",
            None,
            "physics.energy: k(T) law (as viscosity_model) or prandtl (k = "
            "mu(T) c_p(T) / Pr); default prandtl without fluid.conductivity, "
            "else constant",
            choices=(
                "constant",
                "sutherland",
                "power-law",
                "polynomial",
                "table",
                "prandtl",
            ),
        ),
        "conductivity_parameters": Opt(
            "table", None, "Parameters of conductivity_model (as viscosity_parameters)"
        ),
        "prandtl_turbulent_model": Opt(
            "string",
            "constant",
            "Turbulent Prandtl number law Pr_t(T, nu, nu_t, wall distance, Pe_t) "
            "of the incompressible energy equation (RANS, stage V2): constant "
            "(prandtl_turbulent) or a law registered with "
            "pyaeroflow.physics.thermo.register_turbulent_prandtl",
        ),
        "prandtl_turbulent_parameters": Opt(
            "table", None, "Parameters of prandtl_turbulent_model (keyword arguments)"
        ),
        "schmidt_turbulent": Opt(
            "float",
            0.7,
            "Turbulent Schmidt number Sc_t of the species (RANS, stage S1b): "
            "eddy diffusivity mu_t / (rho Sc_t)",
            minimum=0.0,
        ),
        "schmidt_turbulent_model": Opt(
            "string",
            "constant",
            "Turbulent Schmidt number law of the species (RANS): constant "
            "(schmidt_turbulent) or a law registered with "
            "pyaeroflow.physics.thermo.register_turbulent_schmidt",
        ),
        "schmidt_turbulent_parameters": Opt(
            "table", None, "Parameters of schmidt_turbulent_model (keyword arguments)"
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
        "temperature": Opt(
            "float",
            None,
            "Initial temperature (physics.energy); default the temperature of "
            "from_boundary, else reference.temperature",
            "K",
            minimum=0.0,
        ),
        "mass_fraction": Opt(
            "float_list",
            None,
            "Initial mass fractions of fluid.species (in their order); default "
            "those of from_boundary, else 0",
            minimum=0.0,
        ),
        "from_boundary": Opt(
            "string", None, "Uniform initial field equal to this boundary's state"
        ),
        "turbulence": Opt("table", None, TURBULENCE_DESCRIPTION),
        "boundary_layer": Opt(
            "string",
            None,
            "Airfoil wall (one closed contour) whose integral boundary-layer "
            "solution (pyaeroflow foil) sets the initial velocity, pressure and "
            "nu_tilde instead of the uniform state; incompressible flow, ignored "
            "with restart",
        ),
        "boundary_layer_ncrit": Opt(
            "float",
            None,
            "Critical amplification factor of that boundary-layer solution "
            "(default 9 with a transition model or laminar flow, 1 for a fully "
            "turbulent RANS model)",
            minimum=0.0,
        ),
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
            "Reference temperature (compressible; physics.energy: zero of the "
            "enthalpy, rho_ref of the buoyancy and the fluid temperature of Nu "
            "and St); default: the farfield temperature if exactly one farfield "
            "stream exists (physics.energy: or a single inlet temperature)",
            "K",
            minimum=0.0,
        ),
        "temperature_difference": Opt(
            "float",
            None,
            "physics.energy: Delta T of the wall Nusselt and Stanton numbers "
            "Nu = q_w L / (k Delta T), St = q_w / (rho c_p U Delta T); default "
            "the local T_w - reference.temperature",
            "K",
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
            "unsteady",
            "Step acceptance (implicit): unsteady (default since 1.3; HANIM, "
            "Wang et al. 2021: a step must lower the pseudo-time residual "
            "a T du + R(u + a du); quadratic under-relaxation a, a partial "
            "step holds the CFL, a failed step is retried at cfl.cut_factor "
            "with the existing factorization) or merit (the 1.2 loop: halve a "
            "step that raises the steady residual RMS, reject above "
            "cfl.reject_factor)",
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
                0.9,
                "RANS: source-term time-step limit (Piotrowski & Zingg 2020): "
                "the local pseudo-time step of every cell obeys "
                "lambda_max(dS/dQ) dt <= limit, lambda_max the largest growth "
                "rate of the cell-local turbulence/transition source Jacobian "
                "(production included); 0.9 in the paper (default since 1.3). "
                "0 switches it off (the 1.2 behaviour)",
                minimum=0.0,
            ),
            "release_after": Opt(
                "int",
                5,
                "Switch the limit off after this many consecutive full "
                "Newton-Krylov steps; any cut or rejected step arms it again",
                minimum=1,
            ),
            "release_ramp": Opt(
                "float",
                1.0,
                "Gradual release: after release_after clean steps the limit is "
                "multiplied by this factor on every further clean step (off "
                "above 1e8 times the base), and a cut or rejected step divides "
                "it by the factor instead of re-arming it fully. 1 = release "
                "and re-arm at once (default); 10 is the value to try",
                minimum=1.0,
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
                25,
                "Newton steps with this many GMRES iterations or more hold the CFL "
                "(ser)",
                minimum=1,
            ),
            "ser_gmres_reduce": Opt(
                "int",
                60,
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
            "gmres": Opt(
                "string",
                "right",
                "GMRES of the Newton steps: scipy (left-preconditioned inner "
                "iteration; when the true residual misses the tolerance it "
                "restarts from scratch with a tighter inner tolerance) or right "
                "(right-preconditioned, minimizes the true linear residual that "
                "the forcing term refers to; no wasted restarts)",
                choices=("scipy", "right"),
            ),
            "unconverged": Opt(
                "string",
                "reject",
                "Newton step whose GMRES stops at max_iterations above the "
                "forcing tolerance: reject (default since 1.6.1; rejected like a "
                "failed linear solve: CFL x cfl.cut_factor and a fresh "
                "factorization, log note 'GMRES unconverged') or accept (1.6.0: "
                "the update is tried as it is)",
                choices=("reject", "accept"),
            ),
            "forcing_max": Opt(
                "float",
                0.5,
                "Upper bound of the Eisenstat-Walker forcing term (GMRES "
                "tolerance of a Newton step)",
                minimum=0.0,
            ),
            "forcing_safeguard": Opt(
                "bool",
                False,
                "Eisenstat-Walker (1996) safeguards: the forcing term does not "
                "fall below 0.9 eta_prev^2 when that exceeds 0.1, nor below "
                "0.5 target / residual measure (no over-solving of the last "
                "steps)",
            ),
            "approximate_products_until": Opt(
                "float",
                None,
                "Approximate (R1) Jacobian products with the limiter frozen at "
                "the current state until the residual has dropped by this many "
                "orders (Yildirim et al. 2019: 4-5), exact products afterwards; "
                "none = exact products throughout. No effect without a limiter "
                "or once the limiter is frozen",
                minimum=0.0,
            ),
            "product_precision": Opt(
                "string",
                "float64",
                "Precision of the Jacobian-vector products of GMRES: float32 "
                "evaluates the linearized residual in single precision (about "
                "half the time per product, accurate to ~1e-5, enough for the "
                "forcing tolerance); a product with non-finite entries is "
                "recomputed in float64 for the rest of that Newton step, and "
                "after 5 such steps in a row the run stays in float64. The "
                "residual, GMRES and the update stay float64",
                choices=("float64", "float32"),
            ),
            "preconditioner_lagging": Opt(
                "string",
                "adaptive",
                "Refactorization of the preconditioner in the Newton phase: "
                "fixed (refactor_interval, CFL x4 and the 15-iteration rule) or "
                "adaptive (keep it while the GMRES count stays below lag_growth "
                "x (and 3 above) the count right after the factorization, at "
                "most lag_max steps)",
                choices=("fixed", "adaptive"),
            ),
            "lag_max": Opt(
                "int",
                10,
                "Most Newton steps per factorization (adaptive lagging)",
                minimum=1,
            ),
            "lag_growth": Opt(
                "float",
                1.5,
                "GMRES growth that triggers a refactorization (adaptive lagging)",
                minimum=1.0,
            ),
            "lag_cfl_ratio": Opt(
                "float",
                1.0e9,
                "Adaptive lagging refactorizes when the CFL has grown more than "
                "this factor since the factorization. Default 1e9 (off): with 4 the "
                "320k SD7003 LM alpha 0 needed 147 iterations / 1400 s instead of "
                "104 / 1049 s, because the CFL doubles on most steps while it ramps",
                minimum=1.0,
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
            "before the line search (e.g. 1 = at most double); default 0.5 for "
            "plain incompressible SA (sa-neg without a transition model), off "
            "otherwise; 0 = "
            "off. Stops a Newton step from multiplying nu_tilde in one "
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
    "refinement": {
        "cycles": Opt(
            "int",
            0,
            "Local mesh refinement (hanging nodes): number of refinements. "
            "The steady run is split into cycles 0..cycles in "
            "<output.directory>/cycle_<k>; after each cycle but the last the "
            "marked cells are split, the state is copied to the new cells and "
            "the next cycle continues on the refined mesh; 0 = off",
            minimum=0,
        ),
        "indicator": Opt(
            "string",
            None,
            "Indicator expression over output fields and the geometric names "
            "x, y, area, h, h_u, h_v, aspect, factor_u, factor_v, e.g. "
            '"pressure_gradient_magnitude * h"; operators + - * / ** and '
            "< <= > >= (0 or 1), functions sqrt abs log log10 exp tanh sign "
            "minimum maximum clip where norm (vector fields), constant pi",
        ),
        "threshold": Opt(
            "float", None, "Mark cells whose indicator exceeds this value"
        ),
        "quantile": Opt(
            "float",
            None,
            "Mark cells above this quantile (0..1) of the indicator; give "
            "exactly one of threshold, quantile, top_fraction",
        ),
        "top_fraction": Opt(
            "float",
            None,
            "Mark this fraction (0..1) of the cells with the largest indicator",
        ),
        "max_cells": Opt(
            "int",
            None,
            "Cap on the cells added per refinement by the marked splits "
            "(balancing and boundary-layer columns add more); the largest "
            "indicator values are kept",
            minimum=1,
        ),
        "max_factor": Opt(
            "int",
            8,
            "No further split along a direction in which the accumulated split "
            "factor of a cell reached this value",
            minimum=2,
        ),
        "direction": Opt(
            "string",
            "auto",
            "auto: cells with aspect ratio above aspect_target are split along "
            "their long side only (along the wall in a boundary layer), the "
            "others 2 x 2; iso: always 2 x 2",
            choices=("auto", "iso"),
        ),
        "aspect_target": Opt(
            "float",
            3.0,
            "Aspect ratio above which direction = auto splits along the long "
            "side only, into ceil(aspect / aspect_target) parts",
            minimum=1.0,
        ),
        "n_max": Opt(
            "int", 4, "Largest split factor of one cell along its long side", minimum=2
        ),
        "max_hanging": Opt(
            "int",
            1,
            "Balance rule: at most this many hanging nodes per cell side; "
            "neighbours are split until it holds",
            minimum=1,
        ),
        "column_aspect": Opt(
            "float",
            5.0,
            "Boundary-layer column rule: a split along a cell side is passed on "
            "to the quad across it while that quad's aspect ratio (along the "
            "side over across it) exceeds this value; 0 = off",
            minimum=0.0,
        ),
        "walls": Opt(
            "string_list",
            None,
            "Boundaries whose new nodes are placed on a cubic through the "
            "neighbouring wall nodes instead of the straight face; default: "
            "all wall, slip-wall and transpiration-wall boundaries",
        ),
        "corner_angle": Opt(
            "float",
            30.0,
            "Turning angle between two wall faces above which their common node "
            "is a corner (trailing edge); the wall curve does not reach across "
            "it",
            "deg",
            minimum=0.0,
        ),
        "wall_curve": Opt(
            "string",
            "spline",
            "Where new nodes on the walls go: spline = on the cubic through the "
            "neighbouring wall nodes (a smooth contour); straight = on the "
            "straight face between the original wall nodes, so the refined "
            "mesh keeps the faceted contour of the input mesh and resolves its "
            "kinks (for studies of the contour itself)",
            choices=("spline", "straight"),
        ),
        "band": {
            "distances": Opt(
                "float_list",
                [],
                "Geometric refinement before cycle 0, no solution needed: level "
                "i splits every cell whose centre is closer than distances[i] "
                "to the band boundaries (decreasing values, e.g. [0.05, 0.02, "
                "0.008]); the band wraps the whole wall, trailing edge and "
                "near wake included, so no hanging node sits on the wall; "
                "empty = off",
                "m",
                minimum=0.0,
            ),
            "boundaries": Opt(
                "string_list",
                None,
                "Boundaries the band distance is measured from; default: all "
                "wall, slip-wall and transpiration-wall boundaries",
            ),
            "n_max": Opt(
                "int",
                2,
                "Split factor of a thin cell along its long side per level "
                "(other cells 2 x 2)",
                minimum=2,
            ),
            "sequence_residual_drop": Opt(
                "float",
                0.0,
                "Optional mesh sequencing for a start without initial.restart: the run "
                "first solves on the original mesh until the residual dropped "
                "this many orders (at most refinement.iterations, default "
                "solver.max_iterations), then injects that state onto the band "
                "mesh and continues there; 0 (default) = start on the band mesh "
                "from the initial state (converges with the default SA step cap)",
                minimum=0.0,
            ),
        },
        "placement": Opt(
            "string",
            "smooth",
            "Position of new nodes: smooth = on PCHIP curves along the old "
            "grid lines (Coons patch inside a cell), so grid lines keep their "
            "curvature and stretching; linear = midpoints of straight faces",
            choices=("smooth", "linear"),
        ),
        "decay": Opt(
            "float",
            10000.0,
            "A wall node moved by d onto the wall curve moves the grid nodes "
            "above it by d (1 - s / (decay |d|)), s = distance from the wall, "
            "so thin wall cells do not fold; large (the default) keeps the "
            "first-cell height smooth along the wall",
            minimum=0.0,
        ),
        "residual_drop": Opt(
            "float",
            3.0,
            "A cycle before the last ends when its residual dropped this many "
            "orders (the last cycle uses solver.residual_drop)",
            minimum=0.0,
        ),
        "iterations": Opt(
            "int",
            None,
            "A cycle before the last ends after at most this many iterations; "
            "default: solver.max_iterations",
            minimum=1,
        ),
        "cfl_reset": Opt(
            "bool",
            False,
            "The CFL ramp starts again after a refinement (the Newton-Krylov "
            "switch and the limiter are kept); otherwise the CFL of the "
            "previous cycle is continued",
        ),
        "refine_restart": Opt(
            "bool",
            False,
            "Mark and refine the state of initial.restart (a converged run of "
            "another case) before any run: cycle 0 is not run, the chain starts "
            "with cycle 1 on the refined mesh. pyaeroflow polar: first angle only",
        ),
        "cauchy_eps": Opt(
            "float",
            1.0e-6,
            "Force stop of every cycle after a refinement: the cycle also ends "
            "(converged) when CL and CD changed by less than this (mean "
            "relative change per iteration) over cauchy_elements iterations, "
            "each with at least cauchy_min_residual_drop orders. The residual "
            "drop of such a cycle counts from the transferred state, which is "
            "already close to converged, so its orders cost iterations that "
            "no longer move the forces; 0 = off. Not used when "
            "solver.convergence.fields is set",
            minimum=0.0,
        ),
        "cauchy_elements": Opt(
            "int",
            20,
            "Iterations in the Cauchy window of refinement.cauchy_eps",
            minimum=2,
        ),
        "cauchy_min_residual_drop": Opt(
            "float",
            2.0,
            "Residual drop in orders (drop column of the log) every iteration "
            "of the refinement.cauchy_eps window needs",
            minimum=0.0,
        ),
        "limiter_freeze": Opt(
            "float",
            None,
            "Freeze the limiter in every cycle after a refinement once the "
            "residual of the cycle fell by this many orders (0 = off); not "
            "used when numerics.limiter_freeze is set. Default: 1 for "
            "compressible flow (the limiter of a captured shock on a refined "
            "mesh can stall the Newton iteration), off for incompressible "
            "flow (a limiter frozen early in the cycle raises the drag and "
            "can make the cycle diverge later)",
            minimum=0.0,
        ),
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
        "probes": Opt(
            "float2_list",
            [],
            "Point probes [[x, y], ...] (m, mesh coordinates after mesh.scale): "
            "p, u, v (compressible also T) interpolated at each point, one row "
            "per iteration (steady) or time step (unsteady) in "
            "<name>.probes.csv; empty = none",
            "m",
        ),
        "probe_interval": Opt(
            "int",
            1,
            "Write a probe row every n iterations (steady) or time steps (unsteady)",
            minimum=1,
        ),
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
            "Wall temperature (compressible laminar and physics.energy: "
            "isothermal wall; default adiabatic)",
            "K",
            minimum=0.0,
        ),
        "heat_flux": Opt(
            "float",
            None,
            "physics.energy: prescribed heat flux from the wall into the fluid "
            "(instead of temperature)",
            "W/m^2",
        ),
        "mass_fraction": Opt(
            "float_list",
            None,
            "fluid.species: fixed mass fractions at the wall (in the order of "
            "fluid.species); default zero flux",
            minimum=0.0,
        ),
        "species_flux": Opt(
            "float_list",
            None,
            "fluid.species: prescribed diffusion mass fluxes from the wall into "
            "the fluid (instead of mass_fraction)",
            "kg/(m^2 s)",
        ),
        "stefan_flow": Opt(
            "bool",
            False,
            "fluid.species with mass_fraction: the fluid (bulk species) does not "
            "cross the wall, so that the diffusion fluxes drive a net mass flux "
            "sum J_k / (1 - sum Y_k) through it (evaporation, sublimation, "
            "condensation)",
        ),
        "evaporation": Opt(
            "bool",
            False,
            "fluid.species (one species with saturation_model and latent_heat): "
            "evaporating (or condensing) liquid surface; the surface holds the "
            "saturation mass fraction at its temperature, the Stefan flow "
            "j / (1 - Y_s) leaves it and the latent heat enters its energy "
            "balance (temperature: the surface temperature; heat_flux or "
            "adiabatic: the surface temperature follows from the balance)",
        ),
    },
    "symmetry": {},
    "slip-wall": {},
    "velocity-inlet": {
        "velocity": Opt("float2", REQUIRED, "Inlet velocity (u, v)", "m/s"),
        "temperature": Opt(
            "float",
            None,
            "Static temperature (compressible, required; physics.energy: "
            "default reference.temperature)",
            "K",
        ),
        "mass_fraction": Opt(
            "float_list",
            None,
            "fluid.species: mass fractions of the inflow (in the order of "
            "fluid.species); default 0",
            minimum=0.0,
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
        "mass_fraction": Opt(
            "float_list",
            None,
            "fluid.species: mass fractions of the inflow (in the order of "
            "fluid.species); default 0",
            minimum=0.0,
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
        "temperature": Opt(
            "float",
            None,
            "physics.energy: wall temperature (isothermal; also the temperature "
            "of the blown fluid); default adiabatic",
            "K",
            minimum=0.0,
        ),
        "heat_flux": Opt(
            "float",
            None,
            "physics.energy: heat flux from the wall into the fluid",
            "W/m^2",
        ),
        "mass_fraction": Opt(
            "float_list",
            None,
            "fluid.species: fixed mass fractions at the wall (in the order of "
            "fluid.species); default zero flux",
            minimum=0.0,
        ),
        "species_flux": Opt(
            "float_list",
            None,
            "fluid.species: prescribed diffusion mass fluxes from the wall into "
            "the fluid (instead of mass_fraction)",
            "kg/(m^2 s)",
        ),
        "stefan_flow": Opt(
            "bool",
            False,
            "fluid.species with mass_fraction: the fluid (bulk species) does not "
            "cross the wall, so that the diffusion fluxes drive a net mass flux "
            "sum J_k / (1 - sum Y_k) through it (evaporation, sublimation, "
            "condensation)",
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
            "Freestream static temperature (compressible, default 288.15; "
            "physics.energy: default reference.temperature)",
            "K",
            minimum=0.0,
        ),
        "mass_fraction": Opt(
            "float_list",
            None,
            "fluid.species: mass fractions of the inflow (in the order of "
            "fluid.species); default 0",
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

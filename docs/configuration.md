# PyAeroFlow – Case configuration

**Status:** implemented for steady and unsteady laminar, Euler and RANS (SA-neg, k-ω SST) cases (stage 6). Keys of
later stages are listed here for completeness and rejected by the validator
with a pointer to the stage. The schema of record is
`src/pyaeroflow/config/schema.py`; `pyaeroflow check case.toml` prints the resolved
configuration with every default.


## 1. Principles

- **Format:** TOML 1.0, read with the standard library (`tomllib`).
- **Names:** literature terminology and established acronyms (D26). SU2
  option names are not copied.
  - Keys are `snake_case`.
  - String values are lowercase with hyphens, e.g. `"sa-neg"`,
    `"venkatakrishnan-wang"`.
  - Acronyms keep their usual meaning: `cfl`, `muscl`, `fds`, `bdf2`,
    `gmres`, `ilu`.
- **Units:** SI and angles in degrees. Units are documented and shown in the
  example comments, not encoded in key names. *Open: O12
  (non-dimensionalization).*
- **Structure:** initial conditions and boundary conditions are separate
  sections. The farfield is a boundary condition (D28).
- **Defaults:** every key that has a sensible default is optional, so a
  minimal case file stays short.
- **Validation** before the run starts:
  - Unknown keys are rejected with a "did you mean …?" suggestion.
  - Types and value ranges are checked.
  - Combinations are checked for consistency, e.g. `turbulence` requires
    `model = "rans"`.
  - Every set of boundary faces in the mesh must have exactly one boundary
    condition, and every boundary face must belong to exactly one set (no
    unassigned faces, no face in several sets). Further rules for boundary
    conditions: D39.
- **Reproducibility:** the resolved configuration, including all defaults, is
  written to the output directory. Each key carries a comment with its
  description and unit, and says whether the value was set explicitly or is a
  default.
- **Examples:** every test case ships with a commented example case file
  (D30).
- **Python API:** exposes the same structure as the case file (D41). There,
  boundary values may also be functions u(x, y, t); the case file supports
  constant values (profiles: P7).


## 2. Sections

| Section | Content |
|---------|---------|
| `[case]` | Name, description |
| `[mesh]` | Mesh file, scaling |
| `[physics]` | Flow model, turbulence model, transition |
| `[fluid]` | Fluid properties |
| `[initial]` | Initial conditions |
| `[boundary.<name>]` | Boundary condition for the boundary faces called `<name>` in the mesh |
| `[reference]` | Reference values for force and moment coefficients |
| `[numerics]` | Spatial discretization |
| `[solver]` | Pseudo-time / nonlinear solver, convergence |
| `[time]` | Steady or unsteady, physical time stepping |
| `[output]` | Result files, monitoring |


## 3. Boundary faces

A *set of boundary faces* is a named group of mesh faces on the domain
boundary, e.g. `airfoil`, `inlet`, `outer`. The mesh file defines the names.
The case file assigns one boundary condition to each name:

```toml
[boundary.airfoil]        # "airfoil" = name of the boundary faces in the mesh
type = "wall"
```

Names containing characters other than letters, digits, `_` and `-` must be
quoted: `[boundary."wall 1"]`.

The same concept in other tools:

| Tool | Term |
|------|------|
| SU2 | marker (`MARKER_TAG`) |
| OpenFOAM | patch |
| Ansys Fluent | boundary zone |
| Gmsh | physical group (physical curve in 2D) |
| CGNS | boundary condition (`BC_t`) / family |


## 4. Examples

### 4.1 Laminar plane Poiseuille flow (channel)

```toml
# Laminar plane Poiseuille flow, Re = 100 (based on channel height and mean velocity)

[mesh]
file = "channel_quad_L2.su2"     # boundary faces: inlet, outlet, upper, lower

[physics]
model = "laminar"                # "euler" | "laminar" | "rans"

[fluid]
density = 1.0                    # kg/m^3
viscosity = 0.01                 # dynamic viscosity, Pa s

[initial]                        # fluid at rest
velocity = [0.0, 0.0]            # m/s
pressure = 0.0                   # Pa, gauge

[boundary.inlet]
type = "velocity-inlet"
velocity = [1.0, 0.0]            # m/s, uniform profile

[boundary.outlet]
type = "pressure-outlet"
pressure = 0.0                   # Pa, gauge; fixes the pressure level

[boundary.upper]
type = "wall"                    # no-slip

[boundary.lower]
type = "wall"
```

### 4.2 Airfoil in free flight

```toml
# NACA 0012, alpha = 4 deg, fully turbulent

[mesh]
file = "naca0012.su2"            # boundary faces: airfoil, outer

[physics]
model = "rans"
turbulence = "sa-neg"

[fluid]
density = 1.225                  # kg/m^3
viscosity = 1.789e-5             # dynamic viscosity, Pa s

[initial]
from_boundary = "outer"          # uniform field = farfield state.
                                 # May be omitted: with exactly one farfield
                                 # boundary this is the default.

[boundary.airfoil]
type = "wall"

[boundary.outer]
type = "farfield"                # imposes the freestream state
velocity = 14.6                  # m/s, magnitude
angle_of_attack = 4.0            # deg
pressure = 0.0                   # Pa, gauge
turbulence = { turbulent_viscosity_ratio = 0.21 }   # nu_t/nu; for SA equal
                                 # to nu_tilde/nu = 3 (NASA TMR). Same as the
                                 # default, shown here for illustration.

[reference]
length = 1.0                     # m, chord; velocity, density and flow
                                 # direction are taken from boundary "outer"
```


## 5. Reference

Notation: **req.** = required; **–** = no default. The `#` column links an
entry to a decision (D) or an open step (O) in `requirements.md`.

### `[case]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `name` | string | file name without extension | Case name, used for output files | |
| `description` | string | `""` | Free text | |

### `[mesh]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `file` | path | req. | Mesh file (`.su2`), path relative to the case file | D43 |
| `scale` | float | `1.0` | Factor applied to all coordinates (e.g. `0.001` for mm → m) | |

Test-case meshes generated in code are passed through the Python API, not the
case file (D42).

### `[physics]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `model` | `"euler"` \| `"laminar"` \| `"rans"` | req. | Governing equations | D10 |
| `turbulence` | `"sa-neg"` \| `"sst"` | `"sa-neg"` if `model = "rans"` | Turbulence model: `"sa-neg"` Spalart–Allmaras, negative variant (NASA TMR); `"sst"` Menter k-ω SST, version SST-2003m as in SU2 (see [k-ω SST](#k-ω-sst)). The transition models `"bcm"` and `"aft"` need `"sa-neg"`, `"lm"` needs `"sst"` | D11, D12, L2 |
| `mean_pressure_gradient` | [float, float] | – | Driving force for periodic channels: constant mean pressure gradient (dp/dx, dp/dy) in Pa/m, applied as a momentum source | D39 |

`[physics.transition]`, only for `model = "rans"`:

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `mode` | `"none"` \| `"prescribed"` | `"none"` | `"none"` = fully turbulent | D15 |
| `model` | `"none"` \| `"bcm"` \| `"aft"` \| `"lm"` | `"none"` | Transition model (L3). `"lm"`: Langtry–Menter γ-Re_θ on SST (SST-2003m-LM2009 as in SU2), see [Langtry–Menter γ-Re_θ](#langtrymenter-γ-re_θ); needs `turbulence = "sst"`. `"aft"`: SA-AFT2019b (Coder; NASA TMR `aft_transition_3eqn`, equations in `docs/aft_design.md` §5): two more transport equations, for the log-intermittency γ̃ = ln γ and the amplification factor ñ (the transported e^N envelope), so 6 unknowns per cell. ñ grows once the vorticity Reynolds number exceeds its critical value from the local shape factor; onset where ñ passes N_crit; the only change to SA is f_t2 = 1.2 (1 − e^γ̃). Freestream and inflow ñ = 0, γ̃ = 0; zero gradient at walls. Needs a laminar freestream, ν̃/ν ≈ 0.1 (`turbulence = { nu_tilde_ratio = 0.1 }`; a notice outside 0.05–0.5). Output: `intermittency` (= e^γ̃) and `amplification_factor` in the VTU, residual columns `gam` and `n` in the log. `"bcm"`: the algebraic intermittency of Cakmakcioglu, Bas, Mura & Kaynak (2020 revision, as in SU2) multiplies the SA production: γ = 1 − exp(−√T₁ − √T₂) with T₁ from the vorticity Reynolds number d²Ω/ν against Menter's critical Re_θ(Tu) and T₂ = 50 ν_t/ν; SA runs without the f_t2 term (SA-noft2-BCM). Combines with `mode = "prescribed"` (both factors multiply). Needs a laminar freestream: inflow `turbulence = { nu_tilde_ratio = 0.1 }` or `{ turbulent_viscosity_ratio = 2.8e-7, intensity = … }` — with the fully turbulent value 3 the second term makes γ = 0.96 everywhere (a notice says so). Output: `intermittency` field in the VTU, `nu_t_ratio_max` in the wall CSV, transition locations in the summary | L3 |
| `clip` | bool | `true` | `model = "lm"`: after every update clip γ to [1e-4, 5] and Re_θt to [1e-4, 1e15], as SU2 does (`CTransLMSolver`, `AddClippedSolution`). `false` leaves both unclipped: one cell can then fall by 99 % per iteration to γ = 1e-142, where the derivative of √γ in the onset term is about 1e70, and the γ residual stalls (320k SD7003, `docs/validation.md` §10). Ignored (with a notice) for other models | L3 |
| `turbulence_intensity` | float | inflow table | Freestream turbulence intensity Tu as a fraction (0.001 = 0.1 %) for the transition model (for `"aft"` only used for N_crit; ignored by `"lm"`, which takes Tu from the local k); default: the `intensity` of an inflow turbulence table. Free flight 0.0005–0.001, wind tunnels 0.001–0.01, the T3A plate 0.033. The model's critical Re_θ = 803.73 (100 Tu + 0.6067)^−1.027 is printed in the log | L3 |
| `critical_amplification` | float | from `turbulence_intensity` | N_crit of `model = "aft"`. Default: Mack's relation with Drela's limit, N_crit = −8.43 − 2.4 ln(τ/100), τ = 2.5 tanh(100 Tu / 2.5) (8.15 at Tu = 0.1 %). Set it directly to match an XFOIL run (Coder used 10.3 for the Schubauer–Klebanoff plate, 9 for the S809, 7.2 for the NLF(1)-0416); `turbulence_intensity` is then not needed | L3 |
| `relaxation` | float | `1` | Under-relaxation ω in (0, 1] of the BCM intermittency in steady runs: the residual uses a lagged γ that moves by ω (γ_model − γ) per accepted iteration, so a cell crossing the onset threshold ramps its production on over about 1/ω iterations instead of in one step (the step in R_ν seen as a force cycle in E387 α = 8, SER). The converged state is that of the unrelaxed model. `1` = off, 0.2–0.3 typical; not for unsteady runs | L3 |
| `boundary`, `ramp`, `by_alpha` | string, float (m), array of tables | – | Transition locations per angle of attack: `by_alpha = [{ alpha = 0, upper = 0.45, lower = 0.45 }, …]` (deg; metres from the leading edge, XFOIL's `xtr` times the chord), interpolated linearly to the case's angle (`reference.angle_of_attack`, by default the farfield value; end rows held beyond the table) and turned into two `laminar_region` entries on `boundary` with the given `ramp`. Used by `pyaeroflow polar` per angle | D15 |
| `laminar_region` | array of tables | – | Regions where SA production is switched off, e.g. `[{ boundary = "airfoil", side = "upper", x_max = 0.35 }]`: cells whose nearest wall face belongs to `boundary` (required), optionally only on one `side` (`"upper"`: the fluid lies above the wall, `"lower"`: below) and within `x_min`, `x_max`, `y_min`, `y_max` (m); `ramp` (m) blends the production in linearly over that distance inside each bound instead of a hard cut, which keeps the SA source differentiable at the transition front | D15 |

### `[fluid]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `density` | float | req. | ρ in kg/m³ (constant, D5) | |
| `viscosity` | float | – | Dynamic viscosity μ in Pa·s | |
| `reynolds_number` | float | – | Re = ρ U_ref L_ref / μ. μ is derived from it, using `[reference]` velocity and length | D31 |

Exactly one of `viscosity` and `reynolds_number` is required, except for
`model = "euler"`. Non-dimensional setup: `density = 1.0`, reference velocity
and length `1.0`, `reynolds_number = Re` → μ = 1/Re.

### `[initial]`

Either explicit values or `from_boundary`, not both.

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `velocity` | [float, float] | – | (u, v) in m/s | D16 |
| `pressure` | float | – | Gauge pressure in Pa | D16 |
| `turbulence` | table | ν̃/ν = 3 | See [Turbulence values](#turbulence-values) | D16, D33 |
| `from_boundary` | string | name of the farfield boundary, if exactly one exists | Uniform field equal to the state of this boundary | D28 |
| `restart` | path | – | Backup or restart file of a previous run (`<name>.backup_<n>.npz`, `<name>.restart_a.npz`; `pyaeroflow restart-info FILE` shows its iteration) on the same mesh (relative to the case file); overrides `velocity`/`pressure`/`from_boundary` for the field. The mesh is checked by a fingerprint; a laminar restart used for a RANS case keeps the default ν̃. **A restart continues the run that wrote the file** (D58): the iteration or time step counter goes on (`solver.max_iterations` and `time.end_time` are totals), the CFL, the residual peaks that `residual_drop` refers to, the Newton–Krylov switch, the frozen or released limiter, the lagged intermittency and the farfield vortex are restored, so the start-up phase is not repeated; an unsteady run continues with BDF2 at once (previous time level stored; a changed `time_step` gives a BDF1 first step). With the same case name and `output.directory` the log is appended (a `RESTART` marker, then the usual header and the continued lines, the wall column continues) and the history CSV is continued (rows beyond the restart iteration, from a restart from an older file, move to `<name>.history.cut_<n>.csv`). In the digits the continuation differs from an uninterrupted run only by the Jacobian refactorisation schedule. A restart file written before 0.9 has no solver state: the run starts as a new run from its fields (the log says so). `pyaeroflow polar case.toml --alpha 0 4 8` runs an angle sweep as a restart chain and writes `<name>.polar.csv`; the next angle takes over the CFL, Newton–Krylov switch, limiter and vortex of the previous one but starts its own counter and residual reference | D45, D58 |

If `[initial]` is missing and the case has no single farfield boundary, the
case is rejected with a message asking for `[initial]`.

### `[boundary.<name>]`

One table per set of boundary faces. `<name>` is the name in the mesh file.

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `type` | string | req. | Boundary condition type, see below | D38 |

Boundary-condition types (D38). All values are constant (profiles and time
dependence: P7). Types marked "inflow" take a `turbulence` table
([Turbulence values](#turbulence-values)):

| `type` | Parameters |
|--------|-----------------------|
| `"wall"` | No-slip wall. Optional `velocity = [u, v]` for a moving wall (e.g. cavity lid) |
| `"slip-wall"` | Identical to `"symmetry"`: zero normal velocity, no shear stress |
| `"symmetry"` | Symmetry plane: zero normal velocity, no shear stress |
| `"farfield"` | `velocity` (magnitude, m/s), `angle_of_attack` (deg, default 0), `pressure` (Pa, default 0); inflow: `turbulence`. Several farfield boundaries with the same values (e.g. PyAero's `inlet`, `outlet`, `top`, `bottom`) count as one freestream for the defaults of `reference.velocity`, `reference.angle_of_attack`, `initial.from_boundary` and the Cp reference pressure |
| `"velocity-inlet"` | `velocity = [u, v]` in m/s; inflow: `turbulence` |
| `"total-pressure-inlet"` | `total_pressure` (Pa, gauge), `flow_direction = [d_x, d_y]` (normalised internally); the velocity magnitude follows from p_total − p = ½ρ|v|² with the interior pressure; inflow: `turbulence` |
| `"pressure-outlet"` | `pressure` (Pa, default 0) |
| `"periodic"` | `partner` (name of the matching boundary faces; gets no entry of its own), `translation = [dx, dy]` (m); driving force: `physics.mean_pressure_gradient` |
| `"transpiration-wall"` | `normal_velocity` (m/s; + blowing into the domain, − suction), zero tangential velocity; time dependence: P7 (D18); inflow (blowing): `turbulence` |

#### k-ω SST

`turbulence = "sst"` solves two equations, for the turbulent kinetic energy k
and the specific dissipation rate ω (5 unknowns per cell; log residual
columns `k` and `om`). Formulation and constants are SU2's SST-2003m
(Menter, Kuntz & Langtry 2003; NASA TMR "SST-2003m"):

- ν_t = a₁k / max(a₁ω, S F₂), S the strain-rate magnitude;
- P_k = min(ν_t S², 10 β* k ω), P_ω = γ P_k / ν_t; destruction β* k ω and
  β ω²; cross diffusion (1 − F₁) 2σ_ω2 ∇k·∇ω / ω as a source;
- σ_k, σ_ω, β, γ blended with F₁ between (0.85, 0.5, 0.075, γ₁) and
  (1.0, 0.856, 0.0828, γ₂), γᵢ = βᵢ/β* − σ_ωᵢ κ²/√β*, β* = 0.09, a₁ = 0.31;
- walls: k = 0, ω = 60 ν / (β₁ d₁²) with d₁ the height of the wall cell
  (twice the wall distance of its centroid; SU2's first mesh spacing);
  inflow: the prescribed k and ω; outflow, symmetry: zero gradient.

Numerics that differ from SA: k and ω diffuse with the two-point normal
derivative (the least-squares gradient of the 1/y² profile of ω makes the
α-damping face gradient reverse its sign next to the wall); the ω equation
is solved multiplied by ν so that its residual has the scale of the k
residual (the log column `om` and the `residual_omega` field are ν times
and 1 times the ω residual); every update of k and ω is scaled per cell so
that neither drops by more than 99 % (SU2's under-relaxation), so
`solver.nu_tilde_step_limit` does not apply; the first-order Jacobian uses the
residual's face viscosities and SU2's implicit source part (destruction
only), Newton–Krylov the exact Jacobian-vector product.

Turbulence input: form b needs `intensity` with SST (k = 1.5 (Tu U)²,
ω = k/ν_t), form a gives k and ν_t, form c (ν̃/ν) is SA only. Default
without a table: Tu = 0.1 %, ν_t/ν = 1. Fields: `k`, `omega`, `blending_f1`,
`mu_t`, `residual_k`, `residual_omega` in addition to the eddy-viscosity
fields. A restart between SA and SST takes over p, u, v only.
Validation: `docs/validation.md` §9.

#### Langtry–Menter γ-Re_θ

`turbulence = "sst"` with `[physics.transition] model = "lm"` adds the
Langtry–Menter transport equations for the intermittency γ and the
transition-onset momentum-thickness Reynolds number Re_θt (7 unknowns per
cell; log residual columns `gam` and `ret`). Formulation, constants and
limits are those of the NASA TMR page "langtrymenter_4eqn" (SST-2003-LM2009,
Langtry & Menter 2009, F_turb of Menter, Langtry & Völker 2006) and of SU2
(`CSourcePieceWise_TransLM` with the `MENTER_LANGTRY` correlations, SU2's
default for SST); on the SST-2003m base the model is SST-2003m-LM2009:

- P_γ = F_length c_a1 S (γ F_onset)^½ (1 − c_e1 γ), E_γ = c_a2 Ω γ F_turb
  (c_e2 γ − 1), onset from Re_V = d² S/ν against Re_θc(Re_θt), F_length and
  Re_θc the 2009 correlations, R_T = k/(νω);
- P_θt = c_θt U²/(500ν) (Re_θt^eq − Re_θt)(1 − F_θt), Re_θt^eq the
  Tu–λ_θ correlation with the local Tu = 100 √(2k/3)/U (≥ 0.027 %) and the
  local dU/ds, iterated on θ_t (λ_θ in ±0.1, Re_θt^eq ≥ 20);
- coupling to SST: P_k × γ_eff, D_k × min(max(γ_eff, 0.1), 1),
  γ_eff = max(γ, γ_sep) with the separation-induced γ_sep ≤ 2, and
  F₁ = max(F₁, F₃), F₃ = exp(−(R_y/120)⁸); the ω equation is unchanged;
- boundaries: inflow γ = 1 and Re_θt from the inflow Tu of that boundary
  (the log prints it), zero gradient at walls and outflow.

Numerics: γ and Re_θt are convected and diffused like k and ω (two-point
normal derivative); the Re_θt equation is solved multiplied by 10⁻³ (the
log column `ret` and the `residual_re_theta_t` field are 10⁻³ times and
1 times its residual); their updates are limited like k and ω (at most
99 % decrease per update, as a separate group); the first-order Jacobian
adds the negative part of the diagonal source derivatives, Newton–Krylov
differentiates everything exactly (nothing frozen).

The freestream turbulence decays between the inflow and the body (SST
without sustaining terms): set the inflow `intensity` and
`turbulent_viscosity_ratio` so that the Tu at the body is the intended one
(for the T3 plates: fitted to the measured decay, `docs/validation.md`
§10). Fields: `intermittency` (γ), `intermittency_effective` (γ_eff),
`re_theta_t`, `residual_intermittency`, `residual_re_theta_t`. A restart
between SST and SST-LM keeps p, u, v, k, ω (γ = 1 and the freestream Re_θt
fill the rest).

### Turbulence values

Used in `[initial]` and at inflow boundaries (D33). The SA working variable ν̃
(SST: k and ω) is needed wherever flow enters the domain and in the initial
field. Walls (ν̃ = 0; SST k = 0 and the wall ω), outflow and symmetry need no
input.

Exactly one of three input forms:

| Form | Keys | Units | Purpose |
|------|------|-------|---------|
| **b (primary)** | `turbulent_viscosity_ratio`, `intensity` | –, – | ν_t/ν (turbulent over molecular viscosity) and Tu as a fraction (0.001 = 0.1 %). `intensity` is required only by models that need k or Tu (SST, transition models); SA reports it as unused |
| a | `kinetic_energy`, `length_scale`, optional `c_mu` | m²/s², m, – | k and the k-ε length scale ℓ (size of the energy-carrying eddies). `c_mu` defaults to 0.09 |
| c | `nu_tilde_ratio` | – | SA only: ν̃/ν, the notation of NASA TMR |

Default, if no `turbulence` table is given: ν̃/ν = 3 (NASA TMR, SU2 default);
SST: Tu = 0.1 %, ν_t/ν = 1.
The resolved configuration writes it in form b as
`turbulent_viscosity_ratio = 0.2104`.

Conversions (ν = μ/ρ, U = local inflow velocity magnitude):

- Form a → ν_t: ε = C_μ^{3/4} k^{3/2} / ℓ, so ν_t = C_μ k²/ε = C_μ^{1/4} √k ℓ
- Form b → k (when needed): k = 1.5 (Tu U)²; for SST, ω = k/ν_t
- ν_t → SA's ν̃: solve ν_t/ν = χ f_v1(χ) for χ = ν̃/ν, with
  f_v1 = χ³/(χ³ + c_v1³) and c_v1 = 7.1. The left side is monotonic in χ, so
  the solution is unique.

Reported for every inflow and for the initial state, in the log and in the
resolved configuration: the equivalent ν_t/ν, ν̃/ν and, where defined, k, ℓ
and Tu. A warning is issued when ν̃/ν lies outside the NASA TMR range of 3–5
for SA.

### `[reference]`

Needed for force and moment coefficients, and for `fluid.reynolds_number`.

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `length` | float | `1.0`; req. with `fluid.reynolds_number` | Reference length (e.g. chord, channel height) in m | D31 |
| `moment_center` | [float, float] | `[0.25 · length, 0.0]` | Moment reference point in m; C_M positive counter-clockwise, which is nose-down for flow in +x (the usual aerodynamic c_m, nose-up positive, is −C_M) | |
| `vortex_correction` | bool | `false` | Point-vortex farfield correction (P10): the farfield state is the freestream plus the velocity induced by a vortex at `moment_center` whose circulation follows from the lift on `output.force_boundaries` (Γ = L/(ρU), updated with under-relaxation 0.5 every iteration), pressure from Bernoulli. NASA TMR recommends it for farfields nearer than ~500 chords; on a 10-chord O-grid it removes the farfield error of the lift to 0.1 % (Euler, 10°), on PyAero's 3.5-chord C-mesh it recovers about half of the gap to a 500-chord domain |
| `velocity` | float | farfield velocity, if exactly one farfield boundary; otherwise req. with `fluid.reynolds_number` | Reference velocity in m/s | D28, D31 |
| `angle_of_attack` | float | farfield angle of attack if exactly one farfield boundary, else 0 | Defines the drag (along the flow) and lift directions | D28 |
| `density` | float | `fluid.density` | Reference density in kg/m³ | |
| `pressure_point` | [float, float] | domain centroid | Only without a pressure-fixing boundary: the pressure in the cell nearest this point is fixed to `initial.pressure` | D39 |

### `[numerics]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `convective_flux` | `"fds"` | `"fds"` | Flux-difference splitting with low-speed preconditioning | D21 |
| `spatial_order` | `1` \| `2` | `2` | 1: piecewise constant; 2: MUSCL reconstruction | D21 |
| `limiter` | `"venkatakrishnan-wang"` \| `"venkatakrishnan"` \| `"none"` | `"venkatakrishnan-wang"` | Slope limiter. `"venkatakrishnan"`: the classic form with the local threshold ε² = (K h)³, h = √(cell area) — use it for high-Reynolds-number boundary layers on curved walls (airfoils), where the global-range threshold of `"venkatakrishnan-wang"` leaves the thin wall cells unlimited and the reconstruction diverges | D23 |
| `limiter_threshold` | float | `0.05` | Wang's ε′ (suggested range 0.01–0.20), or K of the local threshold | D23 |
| `limiter_freeze` | table | – | `{ after_iterations = n }` and/or `{ after_residual_drop = orders }` (orders of the per-equation convergence measure, as for the Newton–Krylov switch): freeze the limiter values once either condition holds, so that Newton–Krylov converges to machine precision. `release = true` sets the limiter to one instead (unlimited second order after the limited start-up) — the standard for subsonic flows, D54; the airfoil cases use `{ after_residual_drop = 1.5, after_iterations = 40, release = true }` with `limiter = "venkatakrishnan"` | D23, D54 |
| `gradient` | `"weighted-least-squares"` | `"weighted-least-squares"` | Vertex-neighbour stencil | D22 |
| `gradient_weight_exponent` | float | `1.0` | q in weights 1/d^q; default to be confirmed by verification | D22 |
| `viscous_gradient` | `"alpha-damping"` \| `"edge-correction"` | `"alpha-damping"` | Face-gradient scheme | D24 |
| `alpha` | float | `4/3` | α-damping coefficient | D24 |
| `turbulence_spatial_order` | `1` \| `2` | `1` | SA convection (SU2 default: first order) | |
| `first_order_iterations` | int | `0` | Iterations at first order before switching to `spatial_order` | D36 |
| `beta_factor` | float | `4.1` | Artificial compressibility β² = `beta_factor`·max|v|² (SU2 default) | D5 |

### `[solver]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `method` | `"implicit"` \| `"explicit-rk"` | `"implicit"` | `"explicit-rk"`: reference/debugging mode | D25 |
| `max_iterations` | int | `10000` | Pseudo-time iterations (steady) | |
| `refactor_interval` | int | `3` | Reuse the factorized first-order system (lagged pseudo-time matrix, Newton–Krylov preconditioner) for up to n iterations; it is rebuilt earlier after a rejected update, when the CFL has grown fourfold since the factorization, or after a Newton step that needed more than 15 GMRES iterations. The factorization is 70 % of an iteration at 100k cells; 1 = every iteration | D25 |
| `residual_drop` | float | `8.0` | Convergence: orders of magnitude of residual reduction, measured per equation relative to each residual's own peak and applied to the worst equation (D54); equations whose peak is below 10⁻¹² of the largest (round-off) are ignored | D54 |
| `convergence` | table | `{ fields = [], cauchy_elements = 100, cauchy_eps = 1e-10, start_iteration = 0 }` | Force-based convergence as SU2's `CONV_FIELD` (Cauchy criterion), steady implicit runs: with `fields = ["cd"]` (any of `cl`, `cd`, `cm`; needs `output.force_boundaries`) the run also ends as converged when, for every listed coefficient, the mean absolute change per accepted iteration over the last `cauchy_elements` iterations is below `cauchy_eps`; changes count from `start_iteration`, and a restart starts a fresh window. Whichever comes first, this or `residual_drop`, ends the run; the log notes the Cauchy case (`Cauchy criterion met on …`) and the summary names the criterion. Empty `fields` = residual drop only (default). Not used by unsteady runs | L3 |
| `cfl` | table | `{ start = 10.0, max = 1.0e10, max_approximate = 1.0e3, growth_factor = 2.0, cut_factor = 0.1, reject_factor = 10.0, explicit = 1.0, control = "growth", ser_exponent = 1.0, ser_growth_min = 1.2, ser_ratio_min = 0.3, ser_ratio_max = 2.0, ser_gmres_hold = 10, ser_gmres_reduce = 30 }` | Pseudo-time CFL number: SER control with at least `growth_factor` per accepted iteration. `max_approximate` caps the CFL while the Newton–Krylov switch is pending: a pseudo-time step at a very large CFL is a Newton step with the approximate first-order Jacobian and diverges for the second-order residual; `max` applies once Newton–Krylov is on (and throughout when it is disabled). An update is rejected (and the CFL multiplied by `cut_factor`) if the linear solve fails, the update is not finite, or the residual grows by more than `reject_factor`. `explicit` is the CFL of the explicit Runge–Kutta mode. `control = "ser"` switches to proportional SER (D55): the CFL is multiplied by (‖R_old‖/‖R_new‖)^`ser_exponent` of the RMS norm, at least `ser_growth_min` while it falls, clipped to [`ser_ratio_min`, `ser_ratio_max`]; in the Newton–Krylov phase a step with `ser_gmres_hold` or more GMRES iterations does not raise the CFL and one with more than `ser_gmres_reduce` halves it (this replaces the adaptive cap); the `ser_*` keys are ignored with `control = "growth"` | D35, D55 |
| `newton_krylov` | table | `{ enabled = true, switch_at_residual_drop = 2.0, switch_at_iteration = none, linear_tolerance = 1e-2, restart = 50, max_iterations = 200 }` | Switch from approximate-Jacobian iterations to Newton–Krylov once the residual has dropped by the given orders (the switch latches), or at `switch_at_iteration` at the latest (the approximate-Jacobian iteration stalls one to three orders below the start, so a relative switch alone can miss; airfoil cases use 40); GMRES on the exact Jacobian (`jax.jvp`) preconditioned by the factorized first-order system. Newton safeguards (D54): a step that raises the residual is halved up to twice before it is rejected (line search), the GMRES tolerance follows the nonlinear progress (Eisenstat–Walker forcing, between `linear_tolerance` and 0.5), and the CFL is capped adaptively — a step needing 60 or more GMRES iterations quarters the cap, one needing 15 or fewer lets it grow (Python API `nk_forcing`, `line_search_steps`, `nk_cfl_adapt_iterations`) | D25 |
| `linear_solver` | table | `{ type = "direct", tolerance = 1e-6, restart = 50, max_iterations = 500, ilu_drop_tolerance = 1e-5, ilu_fill_factor = 20, ordering = "mmd-at-plus-a" }` | `"direct"`: sparse LU (SciPy SuperLU, single-threaded, always available). `"pardiso"`: Intel MKL PARDISO through the optional `pypardiso` package (multithreaded, x86 only). `"umfpack"`: UMFPACK through the optional `scikit-umfpack` package. Both optional solvers are installed with `conda install -c conda-forge pypardiso scikit-umfpack`; a missing package is reported at the first factorization. Measured on a 295k-cell airfoil mesh (8-core Intel): factorization 20 s (SuperLU), 13.5 s (UMFPACK), 5.8 s (PARDISO), the whole run 1.9× faster with PARDISO, identical results. `"gmres-ilu"`: GMRES with an incomplete LU; the ILU options set its accuracy and memory. Measured on a 100k-cell airfoil mesh, SciPy's ILU is not cheaper than the exact LU at any useful accuracy (drop 1e-4 costs the same; fill 5 / drop 1e-3 leaves a relative residual of 0.8), so `"direct"` stays the choice. Both also precondition Newton–Krylov. `ordering`: SuperLU column ordering, `"mmd-at-plus-a"` (least fill for the mean-flow Jacobian) or `"colamd"` | D32 |
| `nu_tilde_step_limit` | float | – | SA (ignored with SST, whose updates are always under-relaxed), implicit and unsteady (every inner iteration): cap every update of ν̃ to this multiple of max(\|ν̃\|, ν) in each cell before the line search (`1` = at most double the value per step); the mean-flow update is untouched. Off by default. Stops a Newton step from multiplying ν̃ in one boundary-layer cell, which the line search cannot see because ν̃ is a small part of the overall residual norm (E387 α = 8, SER: ν̃ residual jumps of 2 orders every 100 to 150 iterations) | D54, D57 |
| `turbulence_coupling` | `"segregated"` \| `"coupled"` | `"segregated"` | First-order Jacobian of RANS runs: `"segregated"` drops the mean-flow/turbulence coupling blocks (mean flow and ν̃ updated from the same state, as in SU2's segregated solver), `"coupled"` keeps them. The Newton–Krylov phase uses the exact coupled Jacobian-vector products in both. On the flat plate the coupled Jacobian needs more iterations and a 100× larger LU fill with the default ordering: use `linear_solver.ordering = "colamd"` with it | D34 |

### `[time]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `mode` | `"steady"` \| `"unsteady"` | `"steady"` | `"unsteady"`: dual time stepping from the initial state to `end_time` | D9 |
| `scheme` | `"bdf2"` \| `"bdf1"` | `"bdf2"` | Physical time discretisation; the first time step always uses BDF1 | D9, D37 |
| `time_step` | float | req. if unsteady | Δt in s | D9 |
| `end_time` | float | req. if unsteady | in s; the number of steps is `end_time / time_step` rounded | D9 |
| `inner_iterations` | int | `50` | Maximum pseudo-time iterations per time step | D37 |
| `inner_residual_drop` | float | `3.0` | Inner convergence per time step: orders of magnitude relative to the unsteady residual of the previous solution | D37 |
| `inner_cfl` | float | `1e4` | Starting pseudo-time CFL of the inner iterations (the physical time term stabilises the inner system; the SER control of `[solver].cfl` applies from there) | D37 |

The other `[solver]` settings (CFL control, Newton–Krylov, linear solver) apply to the inner iterations as well; Newton–Krylov is used from the first inner iteration. Unsteady runs write one log line and one history row per time step (step, time, inner iterations used, inner residual drop reached, residuals, GMRES iterations, coefficients); `output.verbosity = "detailed"` adds the inner iteration lines. Fields are written every `output.interval` time steps as `<name>_<step>.vtu`, collected in `<name>.pvd` (ParaView time series), plus the final `<name>.vtu`.

### `[output]`

| Key | Type | Default | Description | # |
|-----|------|---------|-------------|---|
| `directory` | path | `"results"` | Relative to the case file | |
| `formats` | array | `["vtu"]` | Field output format; only `"vtu"` (series as PVD). The fields are selected with `fields` | D45, D59 |
| `fields` | array of strings | pressure, velocity, velocity_magnitude and, with a turbulence model, nu_tilde, eddy_viscosity_ratio and the transition fields | Cell fields written to the VTU files, from the table in [Field output](#field-output); `["all"]` selects every field the case provides. The default is the set written before D59. An unknown name gets a "did you mean" hint, a field that the case does not provide is an error that names what it needs | D59 |
| `fields_interval` | int | `0` | Write the fields to `<name>_<number>.vtu` every n iterations (steady) or time steps (unsteady); 0 = only at the end. The number is the iteration or step, 6 digits; a write that falls on a rejected iteration or an unsteady step at the inner limit waits for the next good one (as the restart files, D56). Steady runs also write `<name>.pvd` (ParaView series over the iterations) | D59 |
| `fields_at` | int array | `[]` | Also write the fields at these iterations / time steps, e.g. `[120, 147, 211]`; adds to `fields_interval` (or works alone) | D59 |
| `fields_final` | bool | `true` | Write the final fields to `<name>.vtu` (unsteady runs also add the last step to the series) | D59 |
| `interval` | int | `0` | The older name of `fields_interval`, kept for existing case files; setting both to different values is an error | D59 |
| `restart_interval` | int | `0` | Write a restart file every n iterations (steady) or time steps (unsteady), alternately to `<name>.restart_a.npz` and `<name>.restart_b.npz`: each write replaces the older of the two and goes through a temporary file, so after a crash or a stopped run the newer file (check the modification time) is complete. A write that falls on a rejected iteration or an unsteady step that hit the inner limit waits for the next good one. 0 = none during the run; the final state is always written as a backup (`<name>.backup_<last number>.npz`). Restart from one of them with `initial.restart`; the file carries the iteration or step, the time and the solver state, so the run continues where it stopped (see `initial.restart`) | |
| `restart_backup_interval` | int | `0` | Write a backup `<name>.backup_<iteration or step, 6 digits>.npz` every n iterations / time steps that is never replaced (same content as the rolling files); 0 = none. The two intervals are independent | D56 |
| `restart_backup_at` | int array | `[]` | Also write a backup `<name>.backup_<number>.npz` at these iterations / time steps, e.g. `[120, 147, 211]`; adds to `restart_backup_interval` (or works alone when that is 0); a number due from both gives one file. A listed number that falls on a rejected iteration or an unsteady step at the inner limit is written at the next good one | D56 |
| `transition_diagnostics` | bool | `false` | Runs with a transition model: adds the columns `gamma_cells` (cells with γ > 0.01), `gamma_area` (Σ γ·area), `gamma_jumped` (cells whose γ changed by more than 0.05 since the last accepted iteration), `gamma_dmax` and its cell centre `gamma_dmax_x`, `gamma_dmax_y`, and `rnu_max` (largest \|R_ν\|) with `rnu_max_x`, `rnu_max_y` to the history CSV (coordinates divided by the length scale). Counts only cells within 0.05 chords of a wall. Costs one extra residual evaluation per iteration. Unsteady runs: one row per time step with the first six columns (no `rnu_max*`), changes counted relative to the previous step (the first one relative to the restart state); costs one intermittency evaluation per step | L3 |
| `verbosity` | `"normal"` \| `"detailed"` | `"normal"` | `"detailed"` also logs the inner iterations of unsteady runs | D45 |
| `force_boundaries` | array of strings | all wall boundaries | Boundary faces over which forces and moments are integrated (walls, transpiration walls; slip walls allowed, and included by default in Euler mode); C_L, C_D, C_M per iteration in the log and history, and a wall CSV (x, y, p, Cp, τ_x, τ_y, Cf, and y⁺ of the first cell centre with a turbulence model) per boundary at the end. Forces are those of the fluid on the wall; Cf uses the face tangent oriented counter-clockwise with respect to the fluid (+x on a bottom wall) | D45 |
| `history` | bool | `true` | Residuals and coefficients as CSV and in the `.log` file: per pseudo-time iteration (steady) or per physical time step (unsteady) | D45 |


### Field output

`output.fields` selects the cell fields of the VTU files; the schedule is `fields_interval`, `fields_at` and `fields_final`. The log header lists the selected fields and the size of one file, the number of files at most written and their total size, so a wrong setting can be seen and corrected at the start of a run.

All values are in SI units except the residuals and the wall coefficients (non-dimensional, marked `1`). Fields that exist only at the wall (`cf`, `cp_wall`, `tau_wall_x`, `tau_wall_y`, `y_plus`) are written in the cells next to the walls in `output.force_boundaries` and are 0 in all other cells; a cell with several wall faces gets the length-weighted mean. The residual fields are the residual of the last accepted iteration divided by the cell volume (area in 2D), non-dimensional and signed, with the sign of the residual vector `R(w)` (zero for a converged cell); they exist for steady runs with the implicit solver, not for unsteady runs and not for the explicit mode.

`cf` is flow-aligned: the wall shear stress component along the wall tangent oriented with the freestream direction (`reference.angle_of_attack`), divided by the dynamic pressure, so attached flow gives a positive value on every side of an airfoil and separated flow a negative one. The wall CSV keeps its own sign convention, that of the wall traversal (negative on the lower surface of an airfoil even where the flow is attached). The orientation by the freestream direction is not meaningful where the wall is perpendicular to the freestream (a leading edge, a base).

The table is generated from the field registry (`pyaeroflow.io.fields`); a column marks the model combinations for which a field exists. A field also needs the conditions of the column "Needs".

| Field | Unit | Description | Needs | euler | laminar | SA | SA + BCM | SA + AFT | SST | SST + LM |
|---|---|---|---|---|---|---|---|---|---|---|
| `pressure` | Pa | Static pressure | always | x | x | x | x | x | x | x |
| `velocity` | m/s | Velocity vector | always | x | x | x | x | x | x | x |
| `velocity_magnitude` | m/s | Speed | always | x | x | x | x | x | x | x |
| `vorticity` | 1/s | dv/dx - du/dy from the least-squares gradient (signed) | always | x | x | x | x | x | x | x |
| `kinematic_viscosity` | m^2/s | Laminar kinematic viscosity (constant) | laminar, rans |  | x | x | x | x | x | x |
| `eddy_viscosity` | m^2/s | Kinematic eddy viscosity nu_t | rans |  |  | x | x | x | x | x |
| `eddy_viscosity_ratio` | 1 | nu_t / nu | rans |  |  | x | x | x | x | x |
| `effective_viscosity` | m^2/s | nu + nu_t (the laminar value without a turbulence model) | laminar, rans |  | x | x | x | x | x | x |
| `mu_t` | Pa s | Dynamic eddy viscosity rho nu_t | rans |  |  | x | x | x | x | x |
| `nu_tilde` | m^2/s | SA working variable | rans with turbulence = sa-neg |  |  | x | x | x |  |  |
| `k` | m^2/s^2 | Turbulent kinetic energy (SST) | rans with turbulence = sst |  |  |  |  |  | x | x |
| `omega` | 1/s | Specific dissipation rate (SST) | rans with turbulence = sst |  |  |  |  |  | x | x |
| `blending_f1` | 1 | Menter's blending function F1 (1: k-omega near the wall, 0: k-epsilon) | rans with turbulence = sst |  |  |  |  |  | x | x |
| `wall_distance` | m | Distance to the nearest wall used by the turbulence model | rans |  |  | x | x | x | x | x |
| `intermittency` | 1 | Transition intermittency gamma | rans with a transition model |  |  |  | x | x |  | x |
| `amplification_factor` | 1 | Amplification factor n of SA-AFT | rans with transition.model = aft |  |  |  |  | x |  |  |
| `re_theta_t` | 1 | Transported transition-onset momentum-thickness Reynolds number Re_thetat of Langtry-Menter | rans with transition.model = lm |  |  |  |  |  |  | x |
| `intermittency_effective` | 1 | Effective intermittency max(gamma, gamma_sep) on the SST production (gamma_sep > 1 marks separation-induced transition) | rans with transition.model = lm |  |  |  |  |  |  | x |
| `residual_p` | 1 | Pressure (continuity) residual per cell volume | steady, implicit | x | x | x | x | x | x | x |
| `residual_u` | 1 | x-momentum residual per cell volume | steady, implicit | x | x | x | x | x | x | x |
| `residual_v` | 1 | y-momentum residual per cell volume | steady, implicit | x | x | x | x | x | x | x |
| `residual_nu_tilde` | 1 | nu_tilde residual per cell volume | steady, implicit, turbulence = sa-neg |  |  | x | x | x |  |  |
| `residual_k` | 1 | k residual per cell volume | steady, implicit, turbulence = sst |  |  |  |  |  | x | x |
| `residual_omega` | 1 | omega residual per cell volume | steady, implicit, turbulence = sst |  |  |  |  |  | x | x |
| `residual_gamma` | 1 | log-intermittency residual per cell volume | steady, implicit, transition.model = aft |  |  |  |  | x |  |  |
| `residual_n` | 1 | Amplification-factor residual per cell volume | steady, implicit, transition.model = aft |  |  |  |  | x |  |  |
| `residual_intermittency` | 1 | Intermittency residual per cell volume | steady, implicit, transition.model = lm |  |  |  |  |  |  | x |
| `residual_re_theta_t` | 1 | Re_thetat residual per cell volume | steady, implicit, transition.model = lm |  |  |  |  |  |  | x |
| `cf` | 1 | Skin-friction coefficient with attached flow positive on every side (the wall CSV keeps the sign of the wall traversal); wall cells only | wall in output.force_boundaries, laminar or rans |  | x | x | x | x | x | x |
| `cp_wall` | 1 | Pressure coefficient at the wall; wall cells only | wall in output.force_boundaries | x | x | x | x | x | x | x |
| `tau_wall_x` | 1 | Wall shear stress x component (non-dimensional, rho U^2); wall cells only | wall in output.force_boundaries, laminar or rans |  | x | x | x | x | x | x |
| `tau_wall_y` | 1 | Wall shear stress y component (non-dimensional, rho U^2); wall cells only | wall in output.force_boundaries, laminar or rans |  | x | x | x | x | x | x |
| `y_plus` | 1 | y+ of the first cell centre; wall cells only | wall in output.force_boundaries, rans |  |  | x | x | x | x | x |

### Saved-state files and housekeeping

| File | Written | Content |
|---|---|---|
| `<name>.backup_000100.npz`, `…000200.npz` | every `restart_backup_interval`, at `restart_backup_at` | solution and solver state, never replaced |
| `<name>.backup_000645.npz` | once at the end of every run | final state; the highest number is where the run stopped |
| `<name>.restart_a.npz`, `<name>.restart_b.npz` | every `restart_interval`, alternately | the same content, the older one replaced |

All of them are valid `initial.restart` files. A polar run writes them in
`alpha_<α>/` with the case name `<name>_alpha_<α>`. Files of earlier versions
(`<name>.restart.npz`, `<name>.restart_<n>.npz`) are still read.

```
pyaeroflow restart-info out/case.backup_000645.npz [--json]
pyaeroflow clean case.toml                      # inventory, nothing deleted
pyaeroflow clean case.toml --backups --fields    # dry run of two categories
pyaeroflow clean case.toml --log --history --yes
pyaeroflow clean case.toml --all --yes          # everything the case wrote
```

`clean` options: `--restarts`, `--backups`, `--fields`, `--history`, `--wall`,
`--log`, `--all`; without `--yes` it only lists (D60).

### Control files of a running case

Two empty files in the output directory (`output.directory`) steer a running
case. They are checked after every iteration (steady) or time step (unsteady)
and never deleted by the solver.

| File | Effect |
|---|---|
| `.stop` | The run writes the final backup `<name>.backup_<number>.npz`, the final `<name>.vtu` (even with `fields_final = false`) and the wall files of that iteration or step, and ends. The summary says `stopped`. |
| `.result_629` | At iteration or step 629 the run writes `<name>_000629.vtu` with the `output.fields` of the case, also when 629 is not in `fields_interval` or `fields_at`, and goes on. |

```
touch out/case/.stop            # stop gracefully
touch out/case/.result_629      # fields of iteration 629
rm out/case/.stop               # needed before the next start
```

A start or restart with an existing `.stop` fails right away with an error,
before the mesh is read. An existing `.result_N` is only a notice in the log:
the fields are written when N is reached, or, if N is not after the iteration
or step the run starts from, not at all. A `.result_N` created after N has
passed is ignored. A number that falls on a rejected iteration (or a time step
at the inner limit) is written at the next good one.

## 6. Open questions

None.

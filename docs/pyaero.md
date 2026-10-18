# From a PyAero mesh to a PyAeroFlow result

This page is the workflow for airfoil cases: mesh settings in PyAero that the
solver depends on, the case file, what to look at in the log and the output,
and the current limits. The reference values quoted come from the NACA 0012
runs of stage 6 (`docs/requirements.md`, D40 and D54).

## 1. What the solver needs from the mesh

- An `.su2` file in **metres** (PyAero's export). Quadrilaterals, triangles
  and mixed cells are all fine; PyAero's C-type quad meshes are the intended
  input.
- Boundary names as PyAero writes them: `airfoil`, `inlet` (the C-arc),
  `outlet`, `top`, `bottom`. Nothing is renamed; the case file assigns a
  condition to each name. Every boundary face must belong to exactly one
  named set (the reader checks this).
- Valid cells. The reader rejects self-intersecting ("bow-tie") or folded
  cells, i.e. faces whose neighbouring cell centroid lies on the wrong side,
  and prints their positions. PyAero can produce such cells where the
  C-block around the airfoil meets a tall tunnel extension (seen on a 150k
  mesh with a 50-chord tunnel: a band of slivers above and below the airfoil
  at mid-chord, out to |y| ≈ 6 c). Such a mesh does not converge: the
  uniform initial field already has a residual there and the pseudo-time
  iteration drifts (CD climbing while the CFL falls). Check the mesh in
  PyAero around the block boundaries and regenerate.
- The chord length in metres goes into `reference.length`; the Reynolds
  number is set either through `fluid.reynolds_number` (with `reference.
  velocity` and `reference.length`) or through `fluid.viscosity`.

## 2. PyAero settings that matter

**First layer thickness (the one that decides whether a RANS result is
valid).** The SA model is wall-resolved: the first cell centre must sit at
y⁺ ≈ 1. With the first cell height y₁ (the PyAero value, in metres), the
chord c and the skin-friction coefficient C_f,

    y⁺ = (y₁ / 2) · (Re / c) · √(C_f / 2)        (cell centre at y₁ / 2)

so for y⁺ = 1 at the centre:

| Re (chord) | C_f (mid-chord, turbulent) | y₁ / c |
|---|---|---|
| 1·10⁵ | 0.0066 | 3.5·10⁻⁴ |
| 2·10⁵ | 0.0058 | 1.9·10⁻⁴ |
| 5·10⁵ | 0.0048 | 8·10⁻⁵ |
| 1·10⁶ | 0.0042 | 4.5·10⁻⁵ |
| 6·10⁶ | 0.0029 | 9·10⁻⁶ |

Halve these for y⁺ ≈ 0.5, which is what the stage-6 mesh used at Re 6·10⁶
(y₁ = 5·10⁻⁶ m, y⁺ = 0.6 at α = 0 and 2.5 at the 10° suction peak). After a
run, the wall CSV column `y_plus` shows the actual values; the maximum should
stay below about 2–3. A mesh with y⁺ ≈ 12 gave the viscous drag 40 % too low
and the lift 7 % too low — the error is silent, so check the column.

**Growth ratio and number of layers.** Growth ≤ 1.15–1.2 normal to the
wall, and enough layers to put ~50 cells across the boundary layer, which is
about 0.02–0.03 c thick at the trailing edge of a fully turbulent airfoil
at Re 6·10⁶: from y₁ = 5·10⁻⁶ c with growth 1.15, 50 layers reach 0.036 c.

**Cell aspect ratio.** Not a concern: wall cells with aspect ratios of
1000–3000 are standard in wall-resolved RANS meshes (TMR's grids reach
~2000); the scheme handles stretching, not skewness. Keep the layers
orthogonal to the surface, which PyAero does.

**Surface points.** The lift is insensitive (113 points per side give it
within 1 % of TMR), the pressure drag is not: at Re 6·10⁶ and 10°, 113
points per side give CD_pressure 13 % above TMR, 472 points per side 3 %
(section 8). For drag use 400–500 points per side with clustering at the
leading and trailing edges — TMR's fine grids have 449 — and accept the
cell count that comes with it (295k cells with y⁺ = 1, 27 s per iteration
on an 8-core desktop). PyAero's blunt trailing edge is fine (the solver's own O-grid
uses a 0.14 % base for the same reason: a sharp trailing edge produces a
fan of nearly flat cells).

**Farfield distance.** The farfield condition is a freestream state; its
distance biases the lift. Measured with the inviscid NACA 0012 at 10°:
C-arc at 3.5 chords −6.4 % against a 500-chord domain, −2.9 % with the
point-vortex correction; a 10-chord O-grid with the correction is within
0.1 %. Recommendation: C-arc and top/bottom at 20–50 chords, outlet at
25–50 chords, and `reference.vortex_correction = true`. At 5 chords (the
stage-6 mesh) the correction leaves under 0.5 % on the lift against the
25-chord mesh of section 8, and it does not affect the drag.

**Wake block.** The refined wake block follows the chord; at angle of
attack the wake leaves it after one or two chords, which is acceptable
because the wake is thin only near the trailing edge. Keep the airfoil
unrotated so that one mesh serves all angles.

## 3. The case file

RANS at Re 6·10⁶, α = 10°, with the stage-6 numerics (D54). This is
`examples/cases/naca0012_pyaero_rans.toml` with the comments reduced:

```toml
[case]
name = "naca0012_pyaero_rans"

[mesh]
file = "../meshes/naca0012_pyaero_100k_y1.su2"

[physics]
model = "rans"                    # "laminar" or "euler" for the other models
turbulence = "sa-neg"

[fluid]
density = 1.0                     # kg/m^3
viscosity = 1.6666667e-7          # Pa s: Re = 6e6 with U = 1 m/s, c = 1 m
# or: reynolds_number = 6.0e6 together with reference.velocity and length

[reference]
velocity = 1.0                    # m/s
length = 1.0                      # m, chord; CL, CD, CM use q c
moment_center = [0.25, 0.0]       # m, quarter chord (also the vortex location)
vortex_correction = true

[initial]
from_boundary = "inlet"           # uniform freestream

[boundary.airfoil]
type = "wall"

[boundary.inlet]                  # PyAero's C-arc; outlet, top, bottom identical
type = "farfield"
velocity = 1.0                    # m/s
angle_of_attack = 10.0            # deg
pressure = 0.0                    # Pa
turbulence = { nu_tilde_ratio = 3.0 }

[boundary.outlet]
type = "farfield"
velocity = 1.0
angle_of_attack = 10.0
pressure = 0.0
turbulence = { nu_tilde_ratio = 3.0 }

[boundary.top]
type = "farfield"
velocity = 1.0
angle_of_attack = 10.0
pressure = 0.0
turbulence = { nu_tilde_ratio = 3.0 }

[boundary.bottom]
type = "farfield"
velocity = 1.0
angle_of_attack = 10.0
pressure = 0.0
turbulence = { nu_tilde_ratio = 3.0 }

[numerics]
limiter = "venkatakrishnan"       # local threshold: stabilises the start-up
limiter_freeze = { after_residual_drop = 1.5, after_iterations = 40, release = true }

[solver]
residual_drop = 7.0
max_iterations = 400
newton_krylov = { switch_at_residual_drop = 1.5, switch_at_iteration = 40 }
linear_solver = { type = "pardiso" }   # optional; default "direct" = SuperLU

[output]
directory = "output/naca0012_pyaero_rans"
force_boundaries = ["airfoil"]
```

Notes on the choices:

- All four outer boundaries are `farfield`; the condition switches between
  inflow and outflow per face from the sign of the normal velocity, so it is
  also right for the outlet. Identical farfield tables count as one
  freestream (reference velocity, angle of attack, initial state and the Cp
  reference pressure are taken from them).
- The numerics block is the subsonic RANS standard (D54): the local
  Venkatakrishnan limiter only stabilises the first iterations, after 1.5
  orders or 40 iterations it is released (unlimited second order) and the
  Newton–Krylov phase takes over. Do not freeze the limiter instead of
  releasing it on this kind of mesh: the frozen limiter gave C_f 70 % too
  high on the stage-6 mesh.
- `linear_solver = { type = "pardiso" }` uses Intel MKL PARDISO through the
  `pypardiso` package (`conda install -c conda-forge pypardiso`; x86 only):
  factorization 5.8 s instead of 20 s on the 295k-cell mesh, the whole run
  1.9× faster, identical results. Without the package remove the line —
  the default `"direct"` (SuperLU) needs nothing — or use `"umfpack"`
  (`scikit-umfpack`, 1.5× over SuperLU, also on Apple silicon).
- `nu_tilde_ratio = 3` is NASA TMR's farfield value; the equivalent ν_t/ν,
  k and Tu are printed in the log's Turbulence section (D33 has the other
  input forms).
- For a laminar case drop `[physics].turbulence`, the `turbulence` tables
  and `limiter_freeze`, set `model = "laminar"`; for an inviscid case
  `model = "euler"` (walls become slip walls, no viscosity needed).
- Prescribed transition (laminar region without SA production), e.g. from
  an XFOIL transition location:

  ```toml
  [physics.transition]
  mode = "prescribed"
  laminar_region = [
    { boundary = "airfoil", side = "upper", x_max = 0.35 },
    { boundary = "airfoil", side = "lower", x_max = 0.55 },
  ]
  ```

  `side = "upper"` means the fluid lies above the wall; `x_min`, `x_max`,
  `y_min`, `y_max` are in metres.

## 4. Running and reading the log

```
pyaeroflow check case.toml      # validates, prints every setting with its default
pyaeroflow run case.toml        # writes <name>.log, .history.csv, .vtu, .wall_<boundary>.csv
```

The log (also readable with `tail -f`) has these parts:

- **Configuration, Mesh, Boundary conditions, Turbulence**: the resolved
  setup, including the time mode (`steady` = pseudo-time iterations without
  physical time, or `unsteady` with scheme, time step and step count), the
  CFL control and stopping rule, when Newton–Krylov takes over and when the
  limiter is released; the Turbulence section lists ν_t/ν and ν̃/ν per inflow
  boundary and for the initial state, and warns outside TMR's range
  ν̃/ν = 3–5.
- **Iteration lines**: `log(R_p) log(R_u) log(R_v) log(R_nu)` are the RMS
  residuals of continuity, momentum and the SA equation, `drop` the
  convergence measure in orders (the worst equation's residual relative to
  its own peak — the run stops when it reaches `residual_drop`), `CFL` the
  pseudo-time CFL (it sets each cell's local pseudo-time step; it is not a
  physical time), `NK` marks Newton–Krylov iterations with the GMRES count
  next to it, then `CL CD CM`, and `wall[s]` is the wall-clock time since
  the start of the run. The note column shows `limiter released`
  (or `frozen`) and `rejected` updates. Expected pattern on a 100k-cell
  mesh: 40 start-up iterations that bring the flow residuals down about
  three orders while the turbulence field develops, then 4–10 Newton steps
  with quadratic convergence (residuals drop one to two orders per step).
  About 7–10 s per iteration on two cores at 100k cells; 44 iterations for
  the stage-6 cases.
- **Summary**: iterations, residual drop, CL/CD/CM with the pressure and
  viscous drag split, the vortex circulation if the correction is on, and
  the output paths.

Convergence is judged per equation (each residual relative to its own
peak, the worst equation must drop `residual_drop` orders, the SA equation
included — its residual is about 1/Re times the momentum residuals, which
is why the columns differ by six orders); with `residual_drop = 7` the
coefficients are converged to all printed digits.

## 5. Output

- `<name>.vtu` (ParaView): pressure, velocity, velocity magnitude, and with
  a turbulence model `nu_tilde` and `eddy_viscosity_ratio` (ν_t/ν). A
  converged airfoil at Re 6·10⁶ reaches ν_t/ν ≈ 800 in the wake.
- `<name>.wall_airfoil.csv`: x, y, p, Cp, τ_x, τ_y, Cf, y⁺ per wall face —
  the file for Cp and Cf plots and for the y⁺ check.
- `<name>.history.csv`: residuals, CFL, Newton/GMRES counts and
  coefficients per iteration.
- `<name>.resolved.toml`: every setting with its value and source.

## 6. Polars and restarts

Every run writes `<name>.restart.npz`; `initial.restart = "<file>"` starts
from it (same mesh, checked by a fingerprint). An angle-of-attack sweep is
one command:

```
pyaeroflow polar examples/cases/naca0012_pyaero_rans.toml --alpha 0 4 8 10 12
```

Each angle restarts from the previous one (about 35 instead of 44
iterations), the runs land in `output/.../alpha_<α>/`, and
`<name>.polar.csv` collects α, CL, CD, CDp, CDv, CM. Angles added later are
merged into the file and start from the nearest angle already computed.

## 7. Transition

The SA model is fully turbulent. Two ways out: the BCM transition model
(7.1), or prescribed laminar regions with the location from XFOIL's e^N
method or from experiment (7.2). Both can be combined.

### 7.1 Transition model (BCM)

```toml
[physics.transition]
model = "bcm"
turbulence_intensity = 0.001      # Tu as a fraction: 0.1 % (free flight ~0.05-0.1 %)

[boundary.inlet]                  # and every other farfield/inflow table
turbulence = { nu_tilde_ratio = 0.1 }   # laminar freestream, see below
```

The BCM model (Cakmakcioglu, Bas, Mura & Kaynak, 2020 revision, the form
SU2 implements) multiplies the SA production by an intermittency
γ = 1 − exp(−√T₁ − √T₂): T₁ compares the local vorticity Reynolds number
d²Ω/ν — a proxy for the momentum-thickness Reynolds number of the boundary
layer — with Menter's critical value Re_θ,t = 803.73 (Tu + 0.6067)^−1.027
(Tu in percent), T₂ = 50 ν_t/ν keeps production on where eddy viscosity
already exists. No extra transport equation, no convergence cost beyond
SA; the f_t2 term of SA is switched off with it (SA-noft2-BCM). What it
gives: laminar flow until the boundary layer reaches the critical
thickness for the given Tu, then a short transition; separation-induced
transition follows from the vorticity criterion firing quickly in a
separated shear layer. What it does not give: the N-factor sensitivity of
natural transition at low Tu that XFOIL has — the correlation saturates
(Re_θ,t = 1300 as Tu → 0).

Two inputs matter. The turbulence intensity (free flight 0.05–0.1 %, wind
tunnels 0.1–1 %) sets the critical Reynolds number. The freestream ν̃ must
be laminar: the fully turbulent TMR value ν̃/ν = 3 gives ν_t/ν = 0.21 and
therefore γ = 0.96 everywhere through T₂, i.e. turbulent flow from the
leading edge at any Tu; ν̃/ν = 0.1 (ν_t/ν = 3·10⁻⁷) is right, and the log
prints a notice above 0.5. The transition locations per side are in the
summary (where the boundary layer's maximum ν_t/ν first exceeds 1), the
wall CSV has `nu_t_ratio_max` (laminar 10⁻⁶–10⁻², turbulent 10–100) and
the VTU the `intermittency` field.

Verification on the flat plate (`docs/validation.md`, section 7): at Tu
0.1 % the plate stays laminar (Blasius C_f within 5 %), at Tu 3.3 % (the
ERCOFTAC T3A level) it is laminar at Re_x = 0.75·10⁵, transitions at
1.2·10⁵, where the experiment transitions, and reaches the turbulent C_f
by 2.5·10⁵. Because γ feeds back on itself through T₂, the discrete
problem can have more than one steady state when the laminar boundary
layer is under-resolved (seen on the 48-cell plate, not on the 96-cell
one): resolve the laminar part — y⁺ ≈ 1 does — and, when a transition
location matters, confirm it with a second start-up (e.g. a different
`cfl.start`). Airfoil validation at model-aircraft Reynolds numbers
(Eppler 387, NASA TM 4062) is the next step; until then treat BCM results
as the right trend, not as XFOIL-grade polars.

### 7.2 Prescribed transition

SA production is switched off in laminar regions given per surface side
and chordwise bound, with the location from XFOIL's e^N method or from
experiment (`examples/cases/naca0012_pyaero_transition.toml`):

```toml
[physics.transition]
mode = "prescribed"
boundary = "airfoil"
ramp = 0.02                       # m
by_alpha = [                      # alpha in deg, upper/lower in m from the LE
  { alpha = 0.0, upper = 0.45, lower = 0.45 },
  { alpha = 4.0, upper = 0.30, lower = 0.60 },
  { alpha = 8.0, upper = 0.10, lower = 0.75 },
]
```

How to fill the table:

1. Run XFOIL (or another e^N method) for the airfoil at the Reynolds number
   of the case, with the critical amplification N for the environment (9 for
   a wind tunnel or a clean free flight, 11–12 for a sailplane in calm air,
   lower for a turbulent environment), at the angles of the polar. XFOIL
   reports the transition points as `xtr` top/bottom in x/c.
2. Enter them as `upper` and `lower` in metres from the leading edge, i.e.
   `xtr` times the chord in metres (`reference.length`). `upper` is the
   side on which the fluid lies above the wall (the suction side at positive
   angle of attack).
3. The case's angle of attack (`reference.angle_of_attack`, by default the
   farfield value) picks the row, with linear interpolation between rows and
   the end rows held beyond the table. `pyaeroflow polar` re-resolves the case
   per angle, so one table serves the whole sweep; the generated regions
   appear in `<name>.resolved.toml`.

Fixed regions instead of a table (one angle):

```toml
[physics.transition]
mode = "prescribed"
laminar_region = [
  { boundary = "airfoil", side = "upper", x_max = 0.45, ramp = 0.02 },
  { boundary = "airfoil", side = "lower", x_max = 0.45, ramp = 0.02 },
]
```

`ramp` blends the production in over the given length (metres) instead of a
hard cut; with the hard cut the Newton phase stalled at a residual of 10⁻⁷
on the stage-6 mesh (a discontinuous source), with the coefficients
converged all the same. A laminar separation bubble (low Reynolds numbers)
is not represented by a prescribed location; that needs a transition model
(L3). At Re 6·10⁶, α = 0, transition at 0.45 gave
CD = 0.0053 (pressure 0.0010, viscous 0.0043) against 0.0083 fully
turbulent, Abbott & von Doenhoff's smooth-airfoil value of 0.0058 and
Ladson's tripped 0.0081 — the drag is as sensitive to the transition
location as expected, so the location must come from a boundary-layer
method, not from a guess.

## 8. Validation on the stage-6 mesh (fully turbulent, Re 6·10⁶)

`python examples/stage6_pyaero_validation.py` runs the polars on the 113-
and 472-points-per-side meshes (`--coarse` / `--fine` for one of them,
`--plot-only` to plot existing results) and plots them against Ladson's
tripped data (NASA TM 4074) and the TMR SA values, with
the C_p at 10° against Gregory & O'Reilly (NASA R&M 3726, Re 2.88·10⁶):

| α | CL | Ladson (80/120 grit) | CD | Ladson | CD pressure / viscous |
|---|---|---|---|---|---|
| 0° | 0.0000 | −0.007 / −0.012 | 0.00828 | 0.00809 / 0.00807 | 0.00133 / 0.00694 |
| 4° | 0.4447 | 0.427 / 0.428 | 0.00893 | 0.00823 / 0.00876 | 0.00206 / 0.00687 |
| 8° | 0.8763 | 0.856 / 0.862 | 0.01116 | 0.01028 / 0.00989 | 0.00460 / 0.00656 |
| 10° | 1.0807 | 1.059 / 1.067 | 0.01317 | 0.01191 / 0.01166 | 0.00687 / 0.00630 |
| 12° | 1.2710 | 1.249 / 1.259 | 0.01612 | 0.01321 / 0.01277 | 0.01016 / 0.00596 |

Lift within 1–4 % of the experiment (TMR's SA itself is +2.6 % at 10°),
C_p at 10° within 0.02 of Gregory & O'Reilly for x/c > 0.05, the viscous
drag within 2 % of TMR; the pressure drag grows above the data with α
(+12 % total at 10°, +24 % at 12°).

**Farfield distance is not the cause.** `naca0012_pyaero_rans_far.toml`
runs the same setup on `examples/meshes/naca0012_pyaero_150k_y1.su2`: the
same airfoil points and first-cell height, but a tunnel 25 chords above and
below the airfoil (inlet arc 19 chords upstream, outlet 30 chords behind
the trailing edge), 149k cells:

| α | CL (5c) | CL (25c) | CD (5c) | CD (25c) | CD pressure (5c) | (25c) | iterations (25c) |
|---|---|---|---|---|---|---|---|
| 0° | 0.0000 | 0.0000 | 0.00828 | 0.00838 | 0.00133 | 0.00139 | 39 |
| 4° | 0.4447 | 0.4449 | 0.00893 | 0.00932 | 0.00206 | 0.00241 | 31 |
| 8° | 0.8763 | 0.8741 | 0.01116 | 0.01233 | 0.00460 | 0.00572 | 34 |
| 10° | 1.0807 | 1.0764 | 0.01317 | 0.01487 | 0.00687 | 0.00852 | 34 |
| 12° | 1.2710 | 1.2646 | 0.01612 | 0.01839 | 0.01016 | 0.01238 | 35 |

The lift changes by less than 0.5 % (the vortex correction covers the
5-chord tunnel), the viscous drag by 1 %, but the pressure drag is 24 %
*higher* on the large tunnel at 10°. The excess therefore comes from the
near-field discretisation, which differs between the two C-meshes although
surface points and first layer are the same.

**Chordwise resolution is the cause.** `naca0012_pyaero_rans_fine.toml`
uses a 5-chord mesh with 472 points per airfoil side instead of 113 (same
first layer, 295k cells). At α = 10°:

| mesh | points/side | CL | CD | CD pressure | CD viscous | iterations, s/iteration |
|---|---|---|---|---|---|---|
| 100k, 5c | 113 | 1.0807 | 0.01317 | 0.00687 | 0.00630 | 44, 9 |
| 150k, 25c | 113 | 1.0764 | 0.01487 | 0.00852 | 0.00635 | 34, 9 |
| 295k, 5c | 472 | 1.0827 | 0.01258 | 0.00625 | 0.00632 | 71, 20–27 |
| TMR CFL3D, 897×257 | 449 | 1.0909 | 0.01231 | 0.00606 | 0.00620 | |

The pressure drag goes from +13 % to +3 % against TMR, the total drag from
+7 % to +2 %; the remaining 0.8 % on the lift is TMR's M = 0.15
(Prandtl–Glauert ≈ 1 %). The viscous drag is unchanged by any of this.
The full polar on the fine mesh (restart chain, 41–74 iterations per angle,
about 25 minutes each on an 8-core Intel iMac):

| α | CL fine | CL coarse | Ladson (80/120) | CD fine | CD coarse | Ladson | CD pressure / viscous, fine |
|---|---|---|---|---|---|---|---|
| 0° | 0.0000 | 0.0000 | −0.007 / −0.012 | 0.00825 | 0.00828 | 0.00809 / 0.00807 | 0.00133 / 0.00692 |
| 4° | 0.4448 | 0.4447 | 0.427 / 0.428 | 0.00883 | 0.00893 | 0.00823 / 0.00876 | 0.00198 / 0.00684 |
| 8° | 0.8771 | 0.8763 | 0.856 / 0.862 | 0.01081 | 0.01116 | 0.01028 / 0.00989 | 0.00424 / 0.00656 |
| 10° | 1.0827 | 1.0807 | 1.059 / 1.067 | 0.01258 | 0.01317 | 0.01191 / 0.01166 | 0.00625 / 0.00632 |
| 12° | 1.2758 | 1.2710 | 1.249 / 1.259 | 0.01516 | 0.01612 | 0.01321 / 0.01277 | 0.00916 / 0.00600 |

Against Ladson the fine mesh is within 2 % on CD at 0°, 5–8 % at 8–10°
and 15–19 % at 12°; the lift is 1–2 % high throughout, as TMR's SA is.
The growing drag gap towards 12° is shared with the SA model itself (TMR's
own grid-converged CD is above Ladson there) and with the fully turbulent
assumption against a tripped experiment, so it is not a mesh item any more.

## 9. Reference points from stage 6

PyAero C-meshes, y₁ = 5·10⁻⁶ m, farfield 5 chords, outlet 25 chords,
Re 6·10⁶, SA-neg, vortex correction; 113 points per side (100k cells) and
472 points per side (295k cells):

| α | CL 113 / 472 | CD 113 / 472 | CD pressure 113 / 472 | CD viscous 113 / 472 | TMR (CFL3D) CL / CD |
|---|---|---|---|---|---|
| 0° | 0.0000 / 0.0000 | 0.00828 / 0.00825 | 0.00133 / 0.00133 | 0.00694 / 0.00692 | 0 / 0.00819 |
| 10° | 1.0807 / 1.0827 | 0.01317 / 0.01258 | 0.00687 / 0.00625 | 0.00630 / 0.00632 | 1.0909 / 0.01231 |

TMR's values are compressible at M = 0.15, which raises the lift by about
1 % (Prandtl–Glauert); the pressure-drag excess at 10° is the chordwise
resolution (section 8: 472 points per side bring it to +3 %), not the
farfield distance.

## 10. Current limits

- Transition: the algebraic BCM model (section 7.1) is verified on the
  flat plate, not yet validated on an airfoil at model-aircraft Reynolds
  numbers; the e^N-based SA-AFT model is the planned second step (L3).
- Steady cases are verified; unsteady RANS (shedding at high Re) has not
  been exercised.
- Drag needs 400–500 points per airfoil side (section 8): with 113 the
  pressure drag is 13 % high at 10°. That means ~300k cells with y⁺ = 1
  and about 30 s per iteration on an 8-core desktop; the linear solver is the
  bottleneck (P13).
- Run time is the sparse LU factorization of the first-order system:
  5.5 s per factorization at 100k cells on two cores (70 % of a pseudo-time
  iteration), 20 s and several GB at 295k cells. The factorization is
  reused for up to three iterations (`solver.refactor_interval`), which
  makes the start-up phase about three times cheaper; the Newton phase is
  dominated by the GMRES residual evaluations. SciPy's incomplete LU
  (`linear_solver.type = "gmres-ilu"`) is no shortcut: at a useful accuracy
  it costs as much as the exact factorization. A faster direct solver
  (UMFPACK) or multigrid (P13) is the way beyond 300k cells.

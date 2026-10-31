# Changelog

## 1.1 – Positive γ linearization for Langtry–Menter (2026-10-31)

- Langtry–Menter: the γ residual of the 320k SD7003 runs stalled at 10⁻⁵·⁴
  because of eight cells at the clip floor γ = 10⁻⁴ whose exact Newton update
  pointed away from the root (the √γ derivative of the production, 50 at the
  floor, turns the diagonal of a shear-layer cell negative); the clip
  discarded the update every iteration and the cells kept a residual of
  hundreds, which also held the Newton–Krylov forcing term at its cap. Two
  new options of `[physics.transition]`: `positive_jacobian` (default true,
  a change of the LM Newton step) leaves the diagonal-lowering derivatives
  of the γ source out of the Newton–Krylov Jacobian, the residual is
  unchanged (T3A start-up: 2.4 orders in 60 iterations against 1.2; 320k SD7003 α = 4°
  restart: converged in 13 Newton–Krylov iterations at unchanged forces,
  where the exact linearization stalled for 98 iterations);
  `newton_guard` (default false) replaces such an update by the explicit
  local step −R Δt/A at CFL 1 of the cell and the log Summary counts the
  firings (`gamma descent guard`). Both are named in the Configuration block
  of the log; `positive_jacobian = false` restores the 1.0 behaviour. The
  step limiter hook of the solver now receives the residual and the
  pseudo-time diagonal at CFL 1: `limit_step(u, du, r, tau)`.

## 1.0 – Gmsh meshes, airfoil workflow, control files, LM polar (2026-10-30)

- `.read_toml_N` control file: at iteration or step N the run reads the case
  file again and applies the output and restart settings, and (steady)
  `solver.max_iterations`, `solver.residual_drop` and `solver.convergence`;
  everything else is reported as not applied. Each reload is logged and a copy
  of the settings in force goes to `<name>.read_toml_<N>.toml`.
- Field names inside a `.result_N` file are written in addition to
  `output.fields` for that file. New command `pyaeroflow fields case.toml`
  lists the fields a case provides.

- `solver.newton_krylov.cfl_adapt_iterations` (default 60, as before): the
  GMRES effort at which the adaptive CFL cap is quartered; a quarter of it is
  the effort below which the cap grows. A larger value lets the CFL grow where
  GMRES needs 20–50 iterations per step.

- Documentation: SD7003 SST-LM polar on the 320k PyAero mesh (`docs/validation.md` §10,
  limits in §11); 1.0 list item 2 closed.

- The Configuration block of the log names the SU2 clip of gamma and Re_thetat
  for Langtry–Menter cases also when it is on (the default); before, only
  `physics.transition.clip = false` was logged.

- `solver.cfl.polar_reset = true`: `pyaeroflow polar` restarts the CFL ramp at each
  new angle (Newton–Krylov switch and limiter kept) instead of carrying the CFL
  of the previous angle. Default off.

- `pyaeroflow init-case MESH --model sa|bcm|aft|sst|lm|laminar|euler --re R
  --alpha A` writes a validated case file for a PyAero or Gmsh airfoil mesh
  (L4); `pyaeroflow check-mesh MESH --re R` reports y⁺, the first layer,
  faces per side, farfield distance and cell-size jumps with warnings
  (`pyaeroflow.initcase`, `pyaeroflow.meshcheck`, `docs/pyaero.md` section 0).

- The wall CSV files now run along the contour (trailing edge, upper side,
  leading edge, lower side, as PyAero's airfoil) and have an arc-length column
  `s` at the end; the face order of the mesh was the order before.

- Gmsh reader (L6): `mesh.file` may be a Gmsh `.msh` file (`read_msh`, and
  `read_mesh`, which picks the reader from the extension). Formats 2.2 (also
  2.0, 2.1), 4.0 (ASCII) and 4.1, ASCII and binary, either byte order; read
  and tested with files written by Gmsh 4.15.2. Triangles and quadrangles,
  second-order types with their corner nodes; the curves' physical groups
  become the boundary names. See `docs/configuration.md` ("Gmsh meshes").

## 0.9 – Transition models, SST, control files (2026-10-27)

Covers transition steps 1–4 (SA-BCM, SA-AFT2019b, k-ω SST, Langtry–Menter);
the 0.8 section below was never released separately.

### Transition, step 2: SA-AFT2019b

- Force-based convergence (`solver.convergence`): `fields = ["cd"]` adds
  SU2's Cauchy criterion (mean change of the coefficient per iteration below
  `cauchy_eps` over `cauchy_elements` iterations, from `start_iteration`) to
  the residual drop of steady runs. `physics.transition.clip = false` turns
  the γ/Re_θt clip off (default on, as SU2).

- Langtry–Menter: after every update γ is clipped to [1e-4, 5] and Re_θt to
  [1e-4, 1e15], as in SU2 (`CTransLMSolver`). Without the floor one cell of
  the 320k SD7003 run fell to γ = 1e-142 (99 % per iteration for 70
  iterations), kept the γ residual at 10⁻⁵·⁴ and the run "not converged".

- Control files of a running case (L7): `touch .stop` in the output directory
  ends the run after the current iteration or step with a final backup,
  `<name>.vtu` and the wall files; `touch .result_629` writes the VTU file of
  iteration or step 629 and the run goes on. A start with an existing `.stop`
  fails before the mesh is read; an existing `.result_N` is a notice.
  See `docs/configuration.md` ("Control files of a running case").

- Langtry–Menter γ-Re_θ transition on SST (L3 step 4):
  `physics.transition.model = "lm"` with `turbulence = "sst"`, the TMR
  SST-2003-LM2009 equations on the SST-2003m base with SU2's
  `MENTER_LANGTRY` correlations (local Tu from k, local dU/ds,
  separation-induced γ_sep, F₁ = max(F₁, F₃)). Two transport equations for
  γ and Re_θt (7 unknowns per cell, log columns `gam`, `ret`), exact
  Newton–Krylov JVP. New fields `re_theta_t`, `intermittency_effective`,
  `residual_intermittency`, `residual_re_theta_t`. A restart between SST and
  SST-LM keeps k and ω. ERCOFTAC T3A, T3B, T3A− in `docs/validation.md` §10
  (T3A− onset at Re_x 1.35·10⁶ against 1.44·10⁶ measured); example cases
  `examples/cases/lm_*.toml`.

- k-ω SST turbulence model (D12, L2): `physics.turbulence = "sst"`, Menter's
  SST-2003m with SU2's formulation and constants (strain-rate ν_t limiter,
  production limiter, F1/F2 blending, cross diffusion, wall ω = 60ν/(β₁d₁²)).
  Two transport equations for k and ω (5 unknowns per cell), exact
  Newton–Krylov JVP, SU2's update under-relaxation. Turbulence tables give
  k and ω (form b needs `intensity`; default Tu 0.1 %, ν_t/ν = 1). New VTU
  fields `k`, `omega`, `blending_f1`, `residual_k`, `residual_omega`, and
  `mu_t` for every RANS model; `nu_tilde` and `residual_nu_tilde` are SA
  only. Flat plate: C_f(0.97) and C_D grid-converged, 0.1 % and 0.5 % from the
  TMR SST values;
  SD7003 comparison in `docs/validation.md` §9.

- Saved-state files renamed (D56, D60). Backups are `<name>.backup_000100.npz`
  and so on; the final state of every run is the backup with the last
  iteration or step (`<name>.backup_000645.npz`), and `<name>.restart.npz` is
  no longer written. Rolling files stay `<name>.restart_a/b.npz`. Old names
  are still read. New commands: `pyaeroflow restart-info FILE [--json]` and
  `pyaeroflow clean CASE` (inventory by default; `--restarts`, `--backups`,
  `--fields`, `--history`, `--wall`, `--log`, `--all`; `--yes` deletes).

- A restart also takes over the Newton–Krylov forcing term (the relative GMRES
  tolerance that the solver adapts to the residual reduction). Before, the
  first step after a restart used the tight default and solved the linear
  system far more accurately than the run before, so the residual dropped by
  more than the iterations just before the restart (α 10 fine mesh: log R_p
  −11.4 to −13.8 in one iteration). The initial turbulence values in the log
  are marked as defaults when a restart file supplies the state. Restart files
  of earlier versions start with the default forcing term.
- Configurable field output (D59). `output.fields` selects the VTU cell
  fields by name (`"all"` for everything the case provides): vorticity,
  kinematic, eddy and effective viscosity, wall distance, the cell residuals
  per variable divided by the cell volume (steady, implicit), and the wall
  fields `cf` (attached flow positive on every side), `cp_wall`,
  `tau_wall_x`, `tau_wall_y` and `y_plus` in the wall cells (0 elsewhere), in
  addition to the earlier fields. The schedule is `fields_interval` (the old
  `interval` stays as an alias), `fields_at` (extra iterations or steps) and
  `fields_final`; steady runs write a PVD series. The log header lists the
  fields, the size of one file and the largest total size. The field table is
  generated from `pyaeroflow.io.fields` and checked against
  docs/configuration.md.
- A restart now continues the run (D58). The restart files (rolling, backup
  and final) store the iteration or time step, the time and the solver state:
  CFL, per-variable residual peaks (the reference of `residual_drop`), the
  Newton–Krylov switch, the frozen or released limiter, the lagged
  intermittency, the farfield circulation and the wall time; unsteady runs
  store the previous time level, so BDF2 continues at once. With
  `initial.restart` the counting goes on (`max_iterations` and `end_time` are
  totals, no repeated start-up phase), and a run with the same name and output
  directory appends to its log (with a `RESTART` marker) and history CSV.
  `pyaeroflow polar` keeps the solver state from angle to angle but starts the
  counter and the residual reference again. Files from older versions start a
  new run from their fields.
- SA-AFT2019b transition model (Coder; NASA TMR `aft_transition_3eqn`):
  `physics.transition.model = "aft"` adds transport equations for the
  log-intermittency γ̃ and the amplification factor ñ (6 unknowns per cell).
  N_crit from `turbulence_intensity` (Mack/Drela, 8.15 at 0.1 %) or directly
  from the new `physics.transition.critical_amplification`. SA keeps its
  production; f_t2 = 1.2 (1 − e^γ̃) is the only coupling. Exact JVP (nothing
  frozen). Output: `intermittency` and `amplification_factor` in the VTU,
  residual columns `gam` and `n`. A restart from a 4-column run starts the
  new columns at 0. Needs ν̃/ν ≈ 0.1 in the freestream (notice otherwise).
- Flat plate: transition at the e^N location (N_crit 3 and 5), converged to
  machine zero (`docs/validation.md` §8).
- `physics.transition.relaxation` is now BCM only.

### Transition, step 1: SA-BCM (formerly 0.8)

- Restart files during a run: `output.restart_interval` writes two files
  alternately (`<name>.restart_a.npz`, `<name>.restart_b.npz`), and
  `output.restart_backup_interval` writes permanent numbered copies
  (`<name>.restart_<number>.npz`), counted in iterations (steady) or time steps
  (unsteady). All restart files are written through a temporary file and
  carry the iteration or step (and the time of an unsteady run). The file at
  the end of the run is unchanged. `output.restart_backup_at = [120, 147, 211]`
  adds backups at the listed iterations or steps, on top of
  `restart_backup_interval` or alone.
- `solver.nu_tilde_step_limit` now also caps the inner Newton updates of an
  unsteady (dual-time) run. `output.transition_diagnostics` also works in
  unsteady runs: per time step, the near-wall γ statistics (columns
  `gamma_cells` to `gamma_dmax_y`) in the history CSV.
- Renamed from `aerojax` to `pyaeroflow`: repository
  github.com/chiefenne/pyaeroflow (the old URL redirects), package, import,
  command and conda environment, all lower case. PyAeroFlow is the written
  name in titles, prose and the log header. Case files, meshes and result
  files are unchanged. Reinstall with
  `pip uninstall aerojax && pip install -e .`.
- BCM transition model (Cakmakcioglu et al., 2020 revision, SU2 form):
  `physics.transition.model = "bcm"` with `turbulence_intensity` (fraction,
  defaults to the inflow table's `intensity`). Algebraic intermittency
  γ = 1 − exp(−√T₁ − √T₂) from the vorticity Reynolds number and ν_t/ν
  multiplies the SA production; f_t2 is switched off (SA-noft2-BCM). Needs a
  laminar freestream (ν̃/ν ≈ 0.1; the log warns above 0.5).
- Transition reporting for every RANS run: transition location per side in
  the summary (first x where the boundary layer's max ν_t/ν exceeds 1), wall
  CSV column `nu_t_ratio_max`, VTU field `intermittency`.
- The SA source Jacobian in the first-order system keeps only its negative
  (damping) part; the positive production derivative made the 96 × 64 BCM
  plate diverge in ν̃ at CFL 10³.
- The BCM intermittency is frozen in every linearisation (Picard on γ, as
  SU2): the residual is unchanged, but neither the Newton-Krylov JVP nor the
  first-order source Jacobian differentiates γ. Both square roots of the BCM
  formula have unbounded slope (at onset and as ν_t → 0, dγ/d(ν_t/ν) ≈ 6500
  at ν_t/ν = 3·10⁻⁷), which made the E387 linear systems stiff (GMRES
  counts near 200) and the Newton steps erratic.
- `physics.transition.relaxation` (ω in (0, 1], steady runs): the residual uses
  a lagged BCM intermittency advanced by ω (γ_model − γ) per accepted
  iteration, which turns the one-step R_ν jumps at the onset threshold (E387
  α = 8 with SER: 2-order jumps every ~125 iterations) into a ramp. Same
  converged state. `output.transition_diagnostics` adds γ statistics and the
  location of the largest γ change and |R_ν| to the history CSV (near-wall
  cells only). `solver.nu_tilde_step_limit` caps each Newton update of ν̃ per
  cell relative to max(|ν̃|, ν) (off by default).
- Verified on the TMR plate at Re_L = 5·10⁵: laminar at Tu 0.1 %, transition
  at Re_x = 1.2·10⁵ at Tu 3.3 % (T3A); `docs/validation.md` section 7,
  example `examples/cases/flat_plate_transition.toml`. The discrete problem
  can have more than one steady state on a coarse laminar boundary layer
  (noted in the docs). Planned next: SA-AFT, SST, Langtry–Menter γ-Re_θ.
- Eppler 387 validation case `examples/cases/e387_transition.toml` on the
  PyAero mesh `examples/meshes/e387_pyaero_y1.su2` (209k cells, 481 points per
  side, first cell 2e-4 c) against NASA TM 4062, Re 2·10⁵, Tu 0.1 %. Results
  in `docs/validation.md` section 7: α = 0–6° converge and agree with the
  measurement to 0.015–0.028 in C_L, 5.3 % in C_D, 3.8 % in |c_m|; α = −2°
  and 8° do not converge in 400 iterations.
- Proportional SER CFL control, opt-in with `solver.cfl.control = "ser"`
  (D55): factor (|R_old|/|R_new|)^p of the RMS norm, at least ×1.2 while
  it falls, clipped to [0.3, 2]; in the Newton–Krylov phase hold from 10 and
  halve above 30 GMRES iterations instead of the adaptive cap. Used by the
  E387 case; the default stays `growth`.
- The log names the restart file a run starts from (Notices), and how many
  variables it supplies; the misleading notice "initial.restart given:
  velocity/from_boundary only set the pin" is gone.
- Documentation fix: C_M is positive counter-clockwise, which is nose-down
  for flow in +x; the docs said nose-up. The solver output is unchanged.

## 0.7.2 – Speed: lagged factorization, PARDISO/UMFPACK, Jacobian pattern (2026-10-17)

- Fixed: after the limiter was released (or frozen) the acceptance test still
  compared against the residual of the limited operator; a restart from a
  converged solution converged the limited problem three orders further than
  the unlimited one and every Newton step after the release was rejected. The
  reference residual is now re-evaluated when the operator changes.
- `pyaeroflow polar` restarts from the lower angle when two computed angles are
  equally near.
- Measured on the 295k-cell mesh (alpha 4, 65 iterations): SuperLU 1059 s,
  UMFPACK ~13.5 s per factorization instead of 20 s, PARDISO 570 s with a
  factorization of 5.8 s and faster triangular solves — identical results.
- Optional direct solvers: `solver.linear_solver.type = "pardiso"` (Intel MKL
  PARDISO via `pypardiso`, multithreaded) and `"umfpack"` (`scikit-umfpack`);
  SuperLU (`"direct"`) stays the default and the only built-in one.
  The PyAero example cases set `pardiso` with a comment on how to drop it.
- Speed: the factorized first-order system is reused for up to three
  iterations (`solver.refactor_interval`, moved from `newton_krylov`), rebuilt
  earlier after a rejection, a fourfold CFL growth or a hard GMRES solve;
  pseudo-time iterations without a factorization cost ~1 s instead of 7 s at
  100k cells. The Jacobian sparsity pattern is built once per mesh and the
  assembly is a single `bincount` (was ~1 s per iteration).
- Profile at 100k cells: SuperLU factorization 70 % of an iteration, Jacobian
  assembly 19 %, residual 6 %. SciPy's incomplete LU is not cheaper at any
  useful accuracy; the docs no longer recommend `gmres-ilu` for large meshes.
- The slow stretch of the Newton phase on airfoil start-ups is the SA
  transient (the nu_tilde residual rises 2.5 orders while the eddy viscosity
  establishes itself; CD halves meanwhile), not the GMRES tolerance.

## 0.7.1 – Convergence fixes, mesh validity, PyAero mesh study (2026-10-16)

- `docs/validation.md`: all verification and validation cases with their
  references, numbers and figures (`docs/figures/`); `stage4_unsteady.py
  --medium` runs the DFG 2D-2 case at the verification-test resolution.
- `solver.cfl.max_approximate` (default 1e3) caps the CFL while the
  approximate first-order Jacobian is used; without the cap a restart on the
  295k-cell mesh drove the CFL to 3e4 before Newton-Krylov was on, diverged
  and ran the LU factorisation out of memory.
- The limiter freeze/release uses the per-equation measure, as the
  Newton-Krylov switch does (the total RMS released it at 0.2 orders of the
  measure, in the middle of the transient).
- `examples/stage6_pyaero_validation.py` runs and plots both PyAero meshes
  (`--coarse`, `--fine`, `--plot-only`).
- Examples: `naca0012_pyaero_rans_fine.toml` with the 295k-cell PyAero mesh
  `naca0012_pyaero_300k_y1_s450.su2` (472 points per airfoil side), and
  `naca0012_pyaero_rans_far.toml` with the 149k-cell PyAero mesh
  `naca0012_pyaero_150k_y1.su2` (25-chord tunnel) for the farfield-distance
  study: the pressure-drag excess at high angles is not a farfield effect
  but the chordwise resolution — 472 points per airfoil side bring the
  pressure drag at 10 deg from +13 % to +3 % of TMR (docs/pyaero.md,
  section 8; mesh-setting advice in section 2).
- The Newton-Krylov switch latches: once on it stays on when the measure
  rises again while the turbulence field develops (a 150k-cell run spent 19
  iterations back in PTC before this).
- Mesh reader rejects self-intersecting or folded cells (neighbouring cell
  centroid on the wrong side of a face) and prints their positions. Such a
  PyAero mesh gave a non-zero residual of the uniform state and a drifting
  pseudo-time iteration.
- Fixed: Newton-Krylov switched on at iteration 1 whenever the absolute
  residual was already below 10^-switch_at_residual_drop (the start-up branch
  overwrote the per-equation measure). On the 25-chord PyAero mesh this failed the
  first GMRES solve and left the solution stuck at CFL ~ 1.
- Summary line `residual drop` reports orders of the per-equation measure.
- Convergence measure: an equation is excluded only when its peak residual is
  below 10^-12 of the largest (was 10^-6, which dropped the SA equation from
  the check at Re > ~2e6 because its residual scales with 1/Re).
- Log: `drop` column (orders reached by the per-equation measure, the number
  compared with `residual_drop`) after the residuals of every iteration line.
- Log: the Configuration section states the time mode (steady pseudo-time or
  unsteady with scheme, step and count), CFL control, stopping rule,
  Newton-Krylov switch and limiter release; the elapsed-time column is
  labelled `wall[s]`.
- RANS log and history header: four residual columns (`log(R_nu)` added), so
  the CFL, NK, GMRES and coefficient columns line up again.

## 0.7 – Restarts, polars, validation on the PyAero mesh (2026-10-15)

- Restart files (`<name>.restart.npz`, `initial.restart`) and `pyaeroflow polar`
  (angle sweep as a restart chain, merged `<name>.polar.csv`).
- Prescribed transition per angle of attack (`physics.transition.by_alpha`,
  interpolated to the case's angle, used by `pyaeroflow polar`); `ramp` blends
  the SA production in over a length.
- Validation of the fully turbulent NACA 0012 polar on the PyAero y⁺ = 1
  mesh against Ladson (NASA TM 4074) and of the C_p at 10° against Gregory
  & O'Reilly; packaged data `naca0012_ladson_re6e6_tripped.csv`,
  `naca0012_gregory_cp.csv`, `naca0012_abbott_re6e6_smooth.csv`; example
  `examples/stage6_pyaero_validation.py`; prescribed-transition case
  `examples/cases/naca0012_pyaero_transition.toml`; `docs/pyaero.md`
  sections on polars, transition and validation.

## 0.6 – Stage 6: NACA 0012 (2026-10-15)

- Airfoil O-grid generator (`pyaeroflow.mesh.airfoil`): TMR NACA 0012
  definition, cosine surface spacing with a small blunt trailing edge
  (0.14 % chord: the fan of nearly flat cells of a sharp trailing edge is
  unstable at high Reynolds numbers), marching along smoothed normals
  blended into the radial direction, farfield at 500 chords, fold check.
- Classic Venkatakrishnan limiter with the local threshold ε² = (K h)³
  (`limiter = "venkatakrishnan"`): the global-range threshold of the
  Venkatakrishnan–Wang limiter leaves the thin wall cells of a high-Re
  boundary layer unlimited, and the reconstruction diverges on the curved
  nose. Newton–Krylov switch at a fixed iteration
  (`newton_krylov.switch_at_iteration`) in addition to the residual drop.
- Case R2 (`pyaeroflow.cases.naca0012`): C_L, C_D (pressure/viscous), C_M, C_p
  and C_f at Re 6e6 vs the TMR SA values. 256 x 128 O-grid (33k cells):
  C_L -0.008 / 1.076 / 1.536 at α = 0 / 10 / 15° (TMR 0 / 1.0909 / 1.5461),
  viscous drag close to TMR, pressure drag too high on the coarse O-grid
  wake (see docs/requirements.md D40). PyAero meshes (boundary faces
  `airfoil`, `farfield`) can be used directly.
- Numerics standard for subsonic RANS (D54): limiter released after the
  start-up (`limiter_freeze.release`), per-equation convergence measure
  (each residual relative to its own peak, worst equation decides),
  backtracking line search, Eisenstat–Walker forcing and an adaptive CFL
  cap in the Newton phase. On the PyAero y⁺ = 1 mesh of the NACA 0012
  (100k cells, Re 6e6, α = 0) this converges 13 orders in 44 iterations
  with quadratic Newton convergence; C_D = 0.00828 vs TMR 0.00819, and at
  α = 10° with the vortex correction C_L = 1.0807 vs 1.0909 (TOML case
  `examples/cases/naca0012_pyaero_rans.toml`).
- Point-vortex farfield correction (P10, `reference.vortex_correction`):
  freestream plus the lift-induced velocity of a vortex at the moment
  centre, pressure from Bernoulli; removes the farfield error of the lift on
  a 10-chord domain (Euler, 10°). Several farfield boundaries with the same
  freestream count as one (PyAero's inlet/outlet/top/bottom).
- PyAero mesh of the NACA 0012 (`examples/meshes/naca0012_pyaero_50k.su2`,
  C-type, 52k cells) read and run as is; its wall spacing suits laminar
  flow (y⁺ ≈ 900 at Re 6e6, see docs/requirements.md).
- Documentation: `docs/pyaero.md`, the PyAero-to-PyAeroFlow workflow (first
  layer from y⁺, farfield distance, case file, log, limits).
- Examples: `examples/stage6_naca0012.py`, TOML cases
  `examples/cases/naca0012_rans.toml` and `naca0012_pyaero_euler.toml`.

## 0.5 – Stage 5: RANS with the SA-neg model (2026-10-14)

- Spalart–Allmaras, negative variant (NASA TMR SA-neg, with f_t2 and the
  modified S̃), as fourth variable ν̃ of the coupled state (D11, D13):
  upwind convection with the face-average velocity, α-damping diffusion,
  cell sources with AD-safe formulation; eddy viscosity in the mean-flow
  viscous flux; ν̃ = 0 at walls, prescribed at inflow faces.
- Wall distance of every cell centre (KD-tree candidates, exact
  point-segment distance) with the nearest wall face.
- Segregated/coupled first-order Jacobian (D34), Newton–Krylov coupled in
  both; `linear_solver.ordering` option. Default segregated confirmed on the
  flat plate (coupled: 2× the iterations, 10× the LU time).
- Prescribed transition (D15): `physics.transition.laminar_region` by
  boundary, side and x/y bounds; SA production switched off there.
- Turbulence inputs (D33): `turbulence` tables in `[initial]` and at inflow
  boundaries in the three forms (ν_t/ν + Tu, k + ℓ, ν̃/ν), equivalent values
  reported in the log, warning outside the TMR range ν̃/ν = 3–5; y⁺ in the
  wall CSV, ν̃ and ν_t/ν in the VTU output.
- Verification R1: turbulent flat plate on the TMR domain, C_f(0.97) +1.3 %
  and C_D +1.8 % vs the TMR SA values on 7.7k cells (+1.2 / +1.5 % on 23k),
  u⁺(y⁺) follows the log law. GMRES-ILU verified on the 23k-cell plate.
- Examples: `examples/stage5_rans.py`, TOML case
  `examples/cases/flat_plate_rans.toml`.

## 0.4 – Stage 4: unsteady flow (2026-10-14)

- Dual time stepping (D9, D37): BDF2 (BDF1 for the first step, or
  throughout) with the steady solver as inner iteration — extrapolated start
  2uⁿ − uⁿ⁻¹, large starting pseudo-CFL, Newton–Krylov from the first inner
  iteration, inner residual drop relative to R*(uⁿ). 1–2 inner iterations per
  step in smooth flow. `IncompressibleProblem.solve_unsteady`,
  `pyaeroflow.solver.solve_unsteady`, `DualTimeSettings`.
- Case file: `[time] mode = "unsteady"`, `scheme`, `time_step`, `end_time`,
  `inner_iterations`, `inner_residual_drop`, `inner_cfl`.
- Output (D45): unsteady log and history layouts (one line per time step with
  inner iterations and drop; inner lines with `verbosity = "detailed"`), field
  series `<name>_<step>.vtu` collected in a `.pvd` file.
- Verification: V3 Taylor–Green vortex (doubly periodic, exact pressure pin):
  observed temporal orders 2.02/2.08 (BDF2) and 0.99/1.00 (BDF1), velocity
  error 0.5 % on 32²; impulsively started Couette flow vs the series solution
  from TOML. Benchmark B2 DFG 2D-2 (Re 100): periodic shedding with St
  0.278, cD,max 3.20, cL,max 1.06, Δp 2.44 on 4.4k cells (reference 0.295–0.305,
  3.22–3.24, 0.99–1.01, 2.46–2.50).
- Examples: `examples/stage4_unsteady.py`, TOML case
  `examples/cases/couette_unsteady.toml`.

## 0.3 – Stage 3: Newton–Krylov, boundary conditions, Euler and benchmark cases (2026-10-14)

- Newton–Krylov phase (D25): once the residual has dropped by
  `switch_at_residual_drop` orders, each iteration solves the exact
  linearisation (Jacobian-vector products by `jax.jvp` of the full
  second-order residual) with matrix-free GMRES, right-preconditioned by the
  factorized first-order system. Cavity Re 1000 on 64²: 123 → 20 iterations.
  Optional reuse of the preconditioner (`refactor_interval`).
- Linear solver options (D32): `direct` (SuperLU with the MMD_AT_PLUS_A
  ordering and threshold pivoting, 5–10× less fill than COLAMD on the
  coupled system) or `gmres-ilu` (GMRES with an incomplete LU, options for
  drop tolerance and fill).
- Limiter freezing (`numerics.limiter_freeze`) after a number of iterations
  or a residual drop, so limited cases converge to machine precision.
- Boundary conditions (D38): total-pressure inlet and transpiration wall
  (blowing/suction); the full set of stage 3 is now available from TOML.
- Forces and wall loads (D45): C_L, C_D, C_M (with `reference.angle_of_attack`
  and `moment_center`) per iteration in the log and history, pressure and
  viscous drag in the summary, and a wall CSV (x, y, p, Cp, τ_x, τ_y, Cf) per
  force boundary (`output.force_boundaries`).
- Mesh generators (D42): O-grid `annulus`, block-structured
  `cylinder_in_channel` (DFG geometry), `split_boundary`.
- Verification and benchmarks: V4 inviscid cylinder (Cp vs potential flow,
  error halves under refinement, zero lift), B1 Blasius flat plate at
  Re_L = 1e5 (Cf within 1 %, profile within 0.006), B2 DFG benchmark 2D-1
  (cD within 0.3 %, Δp within 0.5 % of the reference on 13k cells).
- First profiling pass (D52): per-iteration cost at 37k cells dominated by
  the sparse LU (≈1 s) and the Jacobian assembly (0.2 s); residual 33 ms,
  Jacobian-vector product 46 ms.
- Examples: `examples/stage3_benchmarks.py`, TOML case
  `examples/cases/cylinder_euler.toml`.

## 0.2 – Stage 2: steady laminar incompressible flow (2026-10-14)

- Incompressible Navier–Stokes in the SU2 INC formulation (D5, D8): working
  variables (p, u, v), preconditioned pseudo-time with β² = 4.1·max|v|²,
  FDS convective flux (D21) with the exact |Γ⁻¹A| (SU2's closed form has
  twice the tangential dissipation), viscous flux with α-damping gradients
  (D24), body force, Euler mode.
- Boundary conditions (D38): wall (moving), symmetry/slip wall, velocity
  inlet, pressure outlet, farfield; periodic pairs as translated interior
  faces (D39 R4); pressure pin without pressure boundary (R2); mass-balance
  check (R3); walls as slip walls in Euler mode (R5).
- Solver: block pseudo-time term, SER with minimum CFL growth factor (D35),
  first-order start (D36), explicit Runge–Kutta reference mode (D25).
- Python API: `pyaeroflow.driver.IncompressibleProblem` with function-valued
  boundary values (D41).
- TOML case files (D27–D33): schema with defaults, validation with
  suggestions, resolved configuration with comments, dimensional input and
  internal non-dimensionalisation (D51), Reynolds-number input (D31).
- Output: `.log` (console and file), history CSV, VTU fields (D45).
- CLI: `pyaeroflow run case.toml`, `pyaeroflow check case.toml`.
- Mesh: `.su2` writer, x-clustering in the rectangle generator.
- Verification: V1 Couette to round-off (quad, triangle, mixed), V2
  Poiseuille order 1.9–2.1 on four mesh types, B3 cavity vs Ghia et al.
  (Re 100 and 1000), explicit = implicit solution.
- Examples: `examples/stage2_laminar.py`, TOML cases in `examples/cases/`.

## 0.1 – Stage 1: operators and implicit machinery (2026-10-14)

- Finite-volume geometry as JAX arrays (`pyaeroflow.numerics.build_geometry`).
- Weighted least-squares gradients on the vertex-neighbour stencil with
  boundary-face values, weights 1/d^q (D22).
- MUSCL reconstruction and the Venkatakrishnan–Wang limiter (D23).
- Face gradients: α-damping with the damping term at the edge midpoint, and
  the edge-correction fallback (D24).
- Residual assembly from face fluxes; first-order Jacobian from AD of compact
  face fluxes, assembled as a sparse block matrix (D25).
- Direct sparse LU (D32) and implicit pseudo-time stepping with SER CFL
  control and rejection safeguards (D35).
- Model problem: steady scalar convection–diffusion with an exact exponential
  solution (`pyaeroflow.physics.convection_diffusion`, `pyaeroflow.cases`).
- Verification: observed order 1.9–2.1 on the two finest of four levels for
  quadrilateral, triangle, polygon and mixed meshes (regular and perturbed);
  convergence by 11 orders of magnitude in 17–29 iterations.
- Example `examples/stage1_convection_diffusion.py`.

## 0.0 – Stage 0: foundations (2026-10-13)

- Face-based mesh data structure for cell-centred finite volumes: arbitrary
  polygon cells, faces with owner and neighbour cells, outward unit normals,
  face lengths and centres, cell areas and centroids.
- Face-level boundary checks (D39 R1): every boundary face in exactly one
  named set, sets contain boundary faces only. Errors report face counts and
  coordinates.
- Rectangle mesh generator (D42): quadrilateral, triangle, polygon
  (centroid dual) and mixed cells, tanh clustering in y, random perturbation
  of interior nodes.
- Reader for 2D SU2 meshes (`.su2`, D43), compatible with PyAero's export.
- VTU writer with cell and point data, binary or ASCII (D45).
- Optional matplotlib mesh plot (D46).
- Unit tests and the example `examples/stage0_meshes.py`.

# SA-AFT transition model: design note

Status: implemented (step (2) of L3 in `requirements.md`), first verification
on the flat plate (`docs/validation.md` §8). This note holds the structure,
the transcribed model definition (section 5) and the implementation choices
(section 7).

## 1. Model and source

SA-AFT2019b (Coder), as documented on the NASA Turbulence Modeling Resource
(TMR): Spalart-Allmaras plus two transport equations, one for the amplification
factor ñ (the envelope of linearly amplified instabilities, the transported
counterpart of the e^N method) and one for the natural logarithm of the
intermittency, γ̃ = ln γ. The TMR page is `aft_transition_3eqn.html`, now
hosted at https://tmbwg.github.io/turbmodels/ (source repository
github.com/TMBWG/turbmodels; transcribed from commit 20a39f5, page last
updated 10/04/2022 with typo fixes in the γ̃ equation and in c_2). AFT2019b has
the same form as SA-AFT2017b, with improved correlations for H12, the local
shape factor H_L and F_growth. Primary references: Coder, Pulliam and Jensen,
AIAA 2018-1039 (2017b); Coder, AIAA 2019-0039 (2019b; its γ̃ equation has a
sign typo on the c_2 term, the TMR form is the correct one); γ̃ equation after
Menter et al., Flow Turbul. Combust. 95 (2015); N_crit after Mack (1977);
limiter after Drela (MISES). Other papers: Coder and Maughmer (2014),
Stefanski et al. (2019, SA-neg coupling), Denison et al. (NASA LAVA, NTRS
20190028873).

## 2. Intended structure

- **State.** n_var grows from 4 to 6 for `physics.transition.model = "aft"`:
  columns (p, u, v, ν̃, γ̃, Ñ). New column constants next to `NT` in
  `physics/rans.py`; BCM and no-transition cases keep 4 columns.
- **Physics module.** A new `physics/aft.py` with pure functions: auxiliary
  quantities and correlations (shape factor, growth rate, critical Reynolds
  number, onset), sources of γ̃ and Ñ, and the coupling term for SA. Each
  function has a unit test against hand-evaluated values from the page.
- **Transport.** γ̃ and Ñ use the same upwind convection and diffusion
  machinery as ν̃ (`upwind_scalar_flux`, face diffusion coefficients as the page
  specifies), added to the residual in `rans.py`.
- **Coupling.** Only f_t2 changes: f_t2 = c_t3 (1 - exp(γ̃)) replaces the
  default f_t2 in the SA equation; the production term is not multiplied by an
  intermittency (unlike BCM). `spalart_allmaras.source` already has the f_t2
  term (`ft2` flag, `CT3`); it needs a way to take f_t2 as an argument.
  The page's SA equation is the standard SA with the SA-neg continuation left
  to the implementation (the code uses `sa-neg`).
- **Inputs from the flow.** Strain-rate magnitude S and vorticity Ω (full
  tensors, not only |∂v/∂x - ∂u/∂y| as in BCM), wall distance d, and the wall
  distance gradient ∂d/∂x_j for H_L (a geometric field, precomputed once per
  mesh, like `wall_distance`). R_T = μ_t/μ and Re_V use the current ν̃.
- **Configuration.** `physics.transition.model = "aft"`; `turbulence_intensity`
  reused (as for BCM); N_crit from the page's relation (section 5), so Tu
  fixes it. Inflow and wall values as in section 5; ν̃/ν = 0.1 in the
  freestream, as for BCM.

## 3. Numerics

- **Jacobian.** Differentiate the new sources by AD in the JVP, no frozen γ̃
  (the BCM stop_gradient was the source of the per-cell switching). The
  first-order Jacobian keeps the negative part of the source derivative, as for
  SA. Coupling blocks follow `turbulence_coupling` (segregated or coupled);
  Newton-Krylov is coupled either way.
- **Positivity.** Ñ and γ̃ clipped to their physical range after each update;
  the per-step cap of `solver.nu_tilde_step_limit` extended to the two new
  variables, with the same relative form.
- **Start-up.** From a laminar state (ν̃/ν ≈ 0.1, Ñ = 0) and from a restart; a
  restart of a BCM or no-transition run keeps the default values for the new
  columns (as `load_restart` already does for ν̃).
- **Cost.** About 2x the BCM iteration (6 instead of 4 unknowns, larger LU).
  Restart files gain two columns; old files remain readable.
- **Known risk.** Gosin et al. (2024) report difficulty reaching asymptotic
  solutions with the model, traced to the local shape factor. If Newton stalls,
  first look at that term (smoothing, clipping), before changing the
  time-marching.

## 4. Verification and validation

1. Unit tests of the auxiliary functions (page values).
2. Flat plates with free-stream turbulence, the ERCOFTAC T3A, T3A- and T3B
   family: transition onset against the measured skin friction, same mesh
   family as the existing BCM plate (`docs/validation.md` §7).
3. E387 at Re 2·10^5 (the case from the BCM work, NASA TM 4062 data) and the
   α polar; SD7003 at Re 6·10^4 (L3).
4. **XFOIL comparison.** Same Re and α, N_crit from Tu by the page's relation
   (section 5): 8.15 at Tu = 0.1 %.
   Compare transition and separation/reattachment locations, Cp, CL, CD. Use
   published XFOIL results where available, otherwise run XFOIL locally.
5. Time-accurate run at the E387 α = 8 case, to see whether the model removes
   the force oscillation seen with BCM (`analysis/e387_alpha8_ser_stall.md`).

## 5. Model definition (transcribed from the TMR page)

Notation: ρ density, μ molecular and μ_t eddy dynamic viscosity (μ_t from SA,
ν_t = ν̃ f_v1 as usual), d wall distance, S = sqrt(2 S_ij S_ij) the strain-rate
and Ω = sqrt(2 Ω_ij Ω_ij) the vorticity magnitude (S_ij, Ω_ij the symmetric and
antisymmetric parts of ∂u_i/∂x_j). Linear eddy-viscosity model. The page treats
variable density (ρ inside the derivatives); for the incompressible solver ρ is
constant. Typographical fixes of 10/04/2022 are included.

### 5.1 Transport equations (both versions)

```
Dν̃/Dt = c_b1 S̃ ν̃ (1 - f_t2) - (c_w1 f_w - c_b1/κ² f_t2) (ν̃/d)²
         + 1/σ [ ∂/∂x_j((ν + ν̃) ∂ν̃/∂x_j) + c_b2 ∂ν̃/∂x_j ∂ν̃/∂x_j ]

∂(ρñ)/∂t + ∂(ρ u_j ñ)/∂x_j = ρ Ω F_crit F_growth dñ/dRe_θ
                              + ∂/∂x_j [ σ_n (μ + μ_t) ∂ñ/∂x_j ]

∂(ργ̃)/∂t + ∂(ρ u_j γ̃)/∂x_j = c_1 ρ S F_onset [1 - exp(γ̃)]
                               - c_2 ρ Ω F_turb [c_3 exp(γ̃) - 1]
                               + ∂/∂x_j [ (μ + μ_t/σ_y) ∂γ̃/∂x_j ]

f_t2 = c_t3 [1 - exp(γ̃)]          (coupling to SA)
```

γ̃ = ln γ (γ = exp(γ̃) is the intermittency; the change of variable keeps γ
positive). The page notes that the reference paper combines F_onset,1 and
F_onset,2 into one function; they are separated here.

### 5.2 Auxiliary functions, SA-AFT2017b

```
H_L    = d²/μ [ ∂/∂x_i ( ρ u_j ∂d/∂x_j ) ∂d/∂x_i ]      (local shape factor)
H12    = 0.376960 + sqrt( (H_L + 2.453432) / 0.653181 )

Re_V   = ρ S d² / (μ + μ_t)
Re_V0  = k_V Re_θ0
k_V    = 0.246175 H12² - 0.141831 H12 + 0.008886
log10(Re_θ0) = 0.7 tanh(14/(H12 - 1) - 9.24) + 2.492/(H12 - 1)^0.43 + 0.62
F_crit = 0 for Re_V < Re_V0,   1 for Re_V >= Re_V0

F_growth = D(H12) (1 + m(H12))/2 l(H12)
D(H12) = H12 / (0.5482 H12 - 0.5185)
l(H12) = (6.54 H12 - 14.07) / H12²
m(H12) = 1/l(H12) [ 0.058 (H12 - 4)²/(H12 - 1) - 0.068 ]
dñ/dRe_θ = 0.028 (H12 - 1) - 0.0345 exp[ -(3.87/(H12 - 1) - 2.52)² ]

F_onset   = max(F_onset,2 - F_onset,3, 0)
F_onset,1 = ñ / N_crit
F_onset,2 = min(F_onset,1, 2)
F_onset,3 = max(1 - (R_T/3.5)³, 0)
F_turb    = exp[ -(R_T/2)⁴ ]
R_T       = μ_t / μ

N_crit = -8.43 - 2.4 ln(τ/100),    τ = 2.5 tanh( Tu(%) / 2.5 )
```

Tu(%) is the freestream turbulence intensity in percent; τ is Drela's limiter
(τ = 0.0999 for Tu = 0.1 %, so N_crit = 8.15).

### 5.3 SA-AFT2019b: replaced correlations

The form of 2017b is kept; these replace the 2017b ones (TMR page, confirmed
by Coder, AIAA 2019-0039, eqs. 2 to 14):

```
H_L    = d²/μ [ ∂/∂x_i ( ρ u_j ∂d/∂x_j ) ∂d/∂x_i ]      (the TMR page also gives the constant-density form d²/ν [...])
H12    = min[ max(0.26 H_L + 2.4, 2.2), 20.0 ]
D(H12) = 2.4 H12 / (H12 - 1)
k_V    = 1 / (0.4036 H12² - 2.5394 H12 + 4.3273)
```

Unchanged from 2017b: l(H12), m(H12), F_growth = D (1 + m)/2 l, dñ/dRe_θ,
log10(Re_θ0), F_crit, the onset and turbulence functions, constants. The
clipping of H12 is in the 2019 paper (eq. 3) but not on the TMR page: the lower
limit 2.2 keeps H12 physically reasonable and prevents unintended destruction
of ñ; the upper limit 20 prevents spurious production through shocks and near
stagnation points. 2017b (AIAA 2018-1039, Appendix, eq. A3 to A4) instead limits
-0.25 <= H_L <= 200 "for numerical reasons". The "B = 1" in the TMR equation
file does not appear in the page or the papers and is ignored.

### 5.4 Constants and boundary conditions

```
σ_n = 1.0   c_1 = 100   c_2 = 0.06   c_3 = 50   σ_y = 1.0   c_t3 = 1.2
(c_2 was listed as 0.6 before 10/04/2022: 0.06 is correct; the sign in front
 of the c_2 term of the γ̃ equation is minus, as written above)
SA constants (c_b1, c_b2, σ, κ, c_w1..3, c_v1) as in the standard model.

Freestream: ñ = 0,  γ̃ = 0 (γ = 1);   ν̃/ν = 0.1 suggested (ν_t/ν = 2.79e-7)
Wall:       ∂ñ/∂y = 0,  ∂γ̃/∂y = 0,  ν̃ = 0
```

Transition location: Spalart's turbulence index i_t = (1/(κ sqrt(νΩ))) ∂ν̃/∂n,
which is about 0 in laminar and 1 in turbulent boundary layers (the page cites
Spalart and Allmaras, 1994). The page notes that some implementations limit ñ
and γ̃ to prevent runaway behaviour.

### 5.5 Findings from the papers

Sources: Coder, AIAA 2019-0039 (AFT2019) and Coder, Pulliam, Jensen, AIAA
2018-1039 (AFT2017b, Appendix); copies in the project files under
`uploads/papers/`.

- **F_growth in 2019b:** only D changes (eq. 9); l and m are as in 2017b
  (eqs. 6 and 7). This settles the open reading.
- **H12 limits:** [2.2, 20] in 2019b, none on the TMR page; implement them.
- **Intermittency coupling:** f_t2 = c_t3 [1 - exp(γ̃)] with c_t3 = 1.2 is the
  only change to SA ("all functions are defined as in the standard SA model");
  f_t2 stays in the destruction term, where it cancels the production effect in
  turbulent boundary layers. The paper's γ̃ equation (eq. 15) has the c_2 sign
  typo that the TMR page corrects. The nonlinear diffusion term that the
  ln γ change of variable would add is omitted on purpose.
- **N_crit:** can be set directly (positive) or from Tu by eqs. 23 and 24
  (Mack with Drela's smooth maximum, limit 2.5 %). Used values: 10.3 for the
  Schubauer-Klebanoff flat plate, 7.2 for NLF(1)-0416 (Tu 0.15 %), 9 for the
  S809 (Tu 0.07 %).
- **Free-stream ν̃:** recommended ν̃/ν = 0.1 (ν_t/ν = 2.79e-7); the paper reports
  surprising sensitivity of the drag to very low values (0.01), no sensitivity
  of the skin friction up to ν̃/ν = 10.
- **Convergence (the open risk):** transition starts at the trailing edge and
  moves upstream during the iteration (flat plate: laminar for about 10 000
  iterations, transition moves upstream until about 18 000, 50 000 in total,
  with a structured third-order upwind solver, "this can limit the CFL
  condition"). On the TMR plate grid family the finest grid stalled
  (mean-flow residual not below 1e-7), no asymptotic grid convergence
  could be shown. Expect the same upstream-moving transition in our
  pseudo-time Newton-Krylov iteration; continuation from a laminar solution
  and restarts are the natural remedy.
- **Validation data in the paper:** TMR flat plate (N_crit 10.3), NLF(1)-0416
  (Re 4e6, M 0.1, polar and Cp), S809 (Re 2e6, laminar bucket and laminar
  separation bubbles; the model slightly underpredicts the bubble length),
  6:1 prolate spheroid, Braunschweig sickle wing. No E387 or SD7003 case: those
  remain our own validation against TM 4062 and XFOIL.
- **Still open:** how the F_crit step (and the F_onset max/min functions, all
  non-smooth) behave in the exact Jacobian; the paper uses a pseudo-time
  scheme without Jacobian differentiation of the transition terms to our
  knowledge, so this is untested for Newton-Krylov. Plan a first-order Jacobian
  without the step, as for BCM, and the exact JVP, and compare.

## 6. Out of scope

Crossflow transition (the model has no deliberate treatment),
compressible forms, and 3D.

## 7. Implementation (as built)

- `physics/aft.py`: N_crit, H12 (with the 2019b limits), growth rate,
  F_growth, Re_θ0, Re_V0, F_onset, F_turb, f_t2 and the two sources, all pure
  JAX functions. Columns NG = 4 (γ̃) and NN = 5 (ñ) in `physics/rans.py`.
- `spalart_allmaras.source(..., ft2_value=...)` takes the AFT f_t2 in both
  SA-neg branches (the negative branch uses c_b1 (1 − f_t2) S ν̃, the χ → 0
  limit of standard SA-neg).
- S = sqrt(2(u_x² + v_y²) + (u_y + v_x)²), Ω = |v_x − u_y| from the
  least-squares velocity gradients. H_L = d²/ν ∇q · n_d with q = u · n_d and
  n_d the unit vector from the nearest wall point to the centroid
  (`numerics.wall_distance.mesh_wall_normal`, stored as `geom.wall_normal`);
  ∇q is the least-squares gradient with the velocity's coefficient set.
- Boundary values: γ̃ = ñ = 0 on velocity inlets, total-pressure inlets and
  inflowing farfield faces, the interior value elsewhere (walls included).
- Diffusion: face-averaged ν_t, coefficients ν + ν_t/σ_y and σ_n (ν + ν_t).
- Jacobian: the JVP is exact (no stop_gradient; F_crit is a jnp.where step
  with zero derivative). The first-order Jacobian has the convective and
  diffusive blocks of all six columns, the segregated mask keeps the 3 × 3
  mean-flow and the 3 × 3 turbulence blocks, and the source part is the
  diagonal of the local source Jacobian clipped to its negative part.
- Not implemented: clipping of ñ and γ̃, the extension of
  `nu_tilde_step_limit` to them, γ relaxation (BCM only); none was needed on
  the plate.


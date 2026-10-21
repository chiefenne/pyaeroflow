# SA-AFT transition model: design note (planned, not started)

Status: planned step (2) of L3 in `requirements.md`. No code yet. This note
fixes the intended structure so that the work can start once the equations are
transcribed. The model switch that precedes the implementation is the owner's
decision.

## 1. Model and source

AFT2019b (Coder, with Stefanski et al. for the SA-neg coupling), as documented
on the NASA Turbulence Modeling Resource: Spalart-Allmaras plus two transport
equations, one for a modified intermittency and one for the amplification
factor Ñ (the envelope of linearly amplified instabilities, the transported
counterpart of the e^N method). Page named in the Flow360 documentation:
`turbmodels.larc.nasa.gov/aft_transition_3eqn.html`. Papers: Coder and
Maughmer (2014), Coder (2017, 2019), Stefanski et al. (2019, SA-neg coupling),
Denison et al. (NASA LAVA, NTRS 20190028873).

**The equations are not transcribed yet.** The page and NTRS could not be
reached from the cloud session (network policy). Nothing below depends on
values that were not retrieved. Section 5 lists what the transcription must
supply.

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
- **Coupling.** SA production/destruction modified as specified by the page,
  compatible with SA-neg (the code uses `sa-neg`).
- **Inputs from the flow.** Vorticity, wall distance (already used by BCM) and
  whichever velocity-gradient quantity the local shape factor needs: to be
  confirmed from the page.
- **Configuration.** `physics.transition.model = "aft"`; `turbulence_intensity`
  reused (as for BCM); N_crit follows from Tu by the page's relation unless the
  page offers an override. Freestream and wall values of γ̃ and Ñ as the page
  specifies, derived from the inflow Tu.

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
4. **XFOIL comparison.** Same Re and α, N_crit from Tu by Mack's relation
   (N_crit = -8.43 - 2.4 ln Tu, Tu as a fraction: about 8.1 at 0.1 %).
   Compare transition and separation/reattachment locations, Cp, CL, CD. Use
   published XFOIL results where available, otherwise run XFOIL locally.
5. Time-accurate run at the E387 α = 8 case, to see whether the model removes
   the force oscillation seen with BCM (`analysis/e387_alpha8_ser_stall.md`).

## 5. To transcribe from the page before coding

All transport equations (γ̃, Ñ), the SA coupling, every auxiliary function and
constant (shape factor and its local parameter, growth rate, critical
Reynolds number, onset and production/destruction terms), the N_crit relation,
free-stream and wall boundary values, and the page's own verification cases
with reference data.

## 6. Out of scope

Crossflow transition (the model has no deliberate treatment, 2019 paper),
compressible forms, and 3D.

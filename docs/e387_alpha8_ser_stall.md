# E387 α = 8, steady SER run: stall and force cycle

> Analysis note from the 0.7.2 to 0.9 development (BCM, SER CFL control). Paths such as
> `/mnt/project-files/…` and the cases `e387_alpha8_unsteady*.toml` refer to the
> project folder where the runs were made; those cases and run outputs are not
> in the repository.

Source: the uploaded log, history CSV and wall CSV of `e387_transition_alpha_8` (PyAeroFlow 0.7.2, main at 4150c03, read from chiefenne/pyaeroflow).

## What the data shows

### 1. The "2.3 orders" is the ν̃ equation measured against its own spikes
The log's `drop` is `max_k RMS_k / peak_k` (D54, `solver/pseudo_time.py:291`), where `peak_k` is the running maximum of each variable's RMS. I reproduced the column exactly from the history CSV. ν̃ sets it at every iteration, and its peak ratchets up at each spike: 10^-8.18 (it 13) → 10^-7.51 (it 87) → 10^-7.21 (it 219). Each spike therefore resets the reference.

At iteration 400 the momentum equations are 3.5 to 4.0 orders below their start (p 3.98, u 3.46, v 4.05). ν̃ sits at about 10^-9.5 between spikes, roughly the level it started from.

### 2. Three one-step ν̃ jumps, momentum untouched
| it | log R_ν before → after | Δlog R_p | CFL | GMRES |
|---|---|---|---|---|
| 87 | −9.28 → −7.51 | +0.05 | 446 → 422 | 11 |
| 219 | −8.90 → −7.21 (precursor at 215: −9.07 → −8.38) | +0.18 | 527 → 633 | 8 |
| 340 | −9.47 → −7.71 | −0.17 | 767 → 767 | 11 |

Spacing is 132 and 121 iterations. Each jump comes with a smaller rise one step earlier (86, 215, 339). The CFL is moderate, GMRES effort is normal, and the momentum residuals do not move. That rules out a linear-solver or CFL blow-up. Something in the ν̃ source changes discontinuously while the state barely moves.

### 3. Why the controller does not react (0 rejections)
ν̃ contributes 0.4 % to 3 % of the plain RMS² at the spike iterations. Acceptance, line search and SER use the plain RMS (`pseudo_time.py` docstring, D54), so a two-order ν̃ jump is invisible to them.

### 4. The cycle after each jump
After the jump, ν̃ decays back over about 10 to 15 iterations. CD rises at the same time (0.0208 → 0.0253 after it 87), with CL flat. About 25 to 45 iterations later CL falls (1.20 → 1.12, 1.21 → 1.166, 1.20 → 1.168). CL then recovers towards 1.20 to 1.21 until the next jump. The CL dip is shrinking (0.08, then 0.045, then 0.035), but the cycle has not died out in 400 iterations.

### 5. Wall state at iteration 400 (a snapshot inside the recovery phase, not a converged state)
- Upper surface: a small leading-edge bubble at x = 0.004 to 0.014, then a long laminar bubble with cf < 0 from x = 0.022 to 0.104 (40 faces). It reattaches at x ≈ 0.106, which is where ν_t/ν first exceeds 1, the reported transition point. ν_t/ν starts growing at x ≈ 0.076, inside the bubble.
- Wall spacing in the bubble is Δx ≈ 0.0022 c. The onset front therefore moves one wall-normal column per cell.
- Lower surface: laminar.

## Mechanism: consistent with γ-onset switching
- The BCM onset term is `sqrt(max(Reθ − Reθt, 0) / (0.002 Reθt))` (`physics/spalart_allmaras.py:78`). A cell needs only 0.1 % excess over Reθt = 1148 to reach γ = 0.5, and 0.2 % to reach γ = 0.63. In practice γ is a per-cell step function.
- γ is wrapped in `stop_gradient` (`physics/rans.py:99`), so Newton never sees it change. When a cell (or column) crosses the threshold, production switches on as an un-linearised explicit jump. That is the one-step R_ν spike with untouched momentum.
- The flow then adjusts (bubble and front move, CL/CD cycle) until another cell crosses, about 120 to 130 iterations later.

This is consistent with the data but not proven, because the history has no γ diagnostics. The direct test is in the next steps.

## What to try (in order)
1. **Confirm (cheap, diagnostic only).** At each iteration, log the number of cells with γ > 0.01, Σγ·area, and the cell index or x of max |ΔR_ν|. If the spike iterations (87, 219, 340 when restarted the same way) coincide with jumps in the γ count at a cell near the bubble, the hypothesis is confirmed.
2. **Relax γ (model-preserving fix).** Lag the intermittency: γⁿ⁺¹ = γⁿ + ω(γ_BCM(uⁿ) − γⁿ), with ω around 0.1 to 0.3. At a steady state this is identical to BCM, and it turns each step into a short ramp. A limit cycle can survive this if a cell sits exactly on the threshold, so step 1 is still worth doing.
3. **Smooth the onset (changes the model slightly).** Replace `max(x, 0)` by a softplus of width ε (for example 1e-3 in units of Reθt), and `sqrt` by `sqrt(x + δ²) − δ`. γ is then C¹ and could even go back into the JVP. Revalidate at α = 4 against the existing result.
4. **Let the controller see ν̃.** Use the per-equation measure (or ν̃ scaled by Re) in the SER ratio or acceptance, so a ν̃ jump cuts the CFL. This does not remove the cycle, but stops it being silent.
5. **Use the running time-accurate run to separate physics from numerics.** If BDF2 settles to steady forces, the cycle is numerical (do 1 to 3). If it oscillates with a physical period, the bubble is genuinely unsteady at α = 8, and a steady solve cannot converge regardless.

## Side observation
Laminar separation at x ≈ 0.02 c with reattachment at 0.106 c is far forward for α = 8. Compare it with the TM 4062 oil-flow separation and reattachment locations once the solution settles. I have not checked those values here.

## Update 2026-10-20: relaxed run (ω = 0.2, branch claude/gamma-diagnostics-relaxation at 488fceb)

Data: uploaded history CSV with the diagnostics columns, 400 iterations, restart from α = 6.

- The relaxation did not damp the ν̃ jumps. There are 18 jumps above 0.7 orders, against 3 in the baseline, and their size is unchanged (0.7 to 3.2 orders; baseline 1.7 to 1.8). After iteration 138 they recur at one place: x ≈ 0.21 to 0.23 c on the upper surface, inside the boundary layer (wall distance 1e-4 to 3.6e-3 c). At every spike the *model* γ flips by 1.0 in a cell there. That is T₁ (the Re_v threshold) switching in the boundary layer or shear layer, aft of the x ≈ 0.106 seen in the baseline snapshot.
- The logged R_ν at iteration i uses the lagged γ of i−1. The spike and the γ flip therefore come from the same Newton update of the state in that cell, not from γ acting through the residual. That is why lagging γ cannot remove them. Working hypothesis: Newton takes a large local ν̃ step that the line search and acceptance do not see, because they use the plain RMS, of which ν̃ is 0.4 % to 3 %.
- The two rejections (iterations 107 and 135) are ordinary high-CFL rejections: CFL 2366 and 1758, with 50 and 88 GMRES iterations, followed by recovery. The reference-residual re-evaluation after each γ update is required for a consistent acceptance test and should stay. The earlier "step 1" (remove it) was wrong.
- The γ statistics are swamped by far-field cells. Re_v = d²|ω|/ν is large far from the wall, so γ flips at x = 7 to 25 c and on the tunnel walls (y = ±4.98). gamma_cells (about 50 000) and gamma_jumped (hundreds to thousands per iteration) mostly count those far-field cells. The diagnostics should be restricted to near-wall cells.
- The relaxed run is not converged at 400 iterations: CL is 1.20 to 1.22 against 1.17 to 1.20 in the baseline, and the median CFL is 312 against 1165.

Candidate fixes:
1. Limit the ν̃ update per Newton step, relative to max(|ν̃|, ν), as SU2 and FUN3D do for the turbulence variable.
2. Make the line search and acceptance see ν̃ (per-equation scaled norm).
3. Restrict the γ diagnostics to near-wall cells.

Decisive data still missing: the relaxed run's wall CSV. Does cf cross zero near x ≈ 0.21 (reattachment or shear layer)?

## Update 2026-10-20 (later): ν̃ step cap, nu_tilde_step_limit = 1.0 (same restart, α = 6; polar run)

- The cap removes the ν̃ spikes: 0 jumps above 0.7 orders in 400 iterations (baseline 3, relaxed ω = 0.2 18), 0 rejections.
- It does not converge: final drop 1.13 orders (baseline 2.32), R_u −6.69 (baseline −7.31). CL still cycles 1.17 to 1.21 over the last 100 iterations, CD 0.017 to 0.022.
- The solution changes: the upper-surface transition (max ν_t/ν > 1) moves forward to x = 0.038 (baseline 0.106), and the laminar bubble spans x = 0.003 to 0.059. This is not yet a steady state, so the location is not a result.
- Likely cause of the stall: the cap is relative to max(|ν̃|, ν). Where ν̃/ν ≲ 1 it is ≈ ν per step, so ν̃/ν can grow by at most ≈ 1 per iteration. That slows early transition growth and may hold the iteration back.
- Next: test a looser cap (3 or 5), and a longer run (800 iterations) at cap 1, to tell slow convergence from a stall.

## Update 2026-10-20 (evening): cap 1 / 800 iterations and cap 3 / 400 iterations

Data: /mnt/project-files/uploads/e387_alpha8_cap_runs/ (all runs renamed by case; cap1_800_coefficients.png is the user's plot with a ±5 % band). File mapping checked from the logs ("capped at 1 x" / "capped at 3 x") and the row counts (801 and 401).

| Run (α = 8, from α = 6) | ν̃ jumps > 0.7 | Rejections | log R_u at end | CL range after it 200 | CL minima (period) |
|---|---|---|---|---|---|
| Baseline | 3 | 0 | −7.31 | 1.166–1.210 | 132, 267, 387 (≈ 120–135) |
| Cap 1, 400 | 0 | 0 | −6.69 | 1.164–1.215 | 144, 291 |
| Cap 1, 800 | 0 | 0 | −6.52 | 1.164–1.207 | 144, 291, 419, 546, 695 (128–149) |
| Cap 3, 400 | 0 | 0 | −7.22 | 1.167–1.210 | 130, 243, 367 (113–124) |

- Cap 1/800 reproduces cap 1/400 exactly over the first 400 iterations (max |ΔCL| = 0), so the runs are deterministic.
- **The force cycle is a limit cycle, not slow convergence.** From the second cycle on, the CL amplitude stays at 1.164–1.207 (about ±1.8 % around 1.186) and does not decay over 800 iterations. Momentum residuals stay at 10^-6.5 to 10^-7.
- **The cycle is the same with and without the cap.** Baseline, cap 1 and cap 3 all have the same CL band and a period of 115 to 150 iterations. The ν̃ spikes were a symptom; removing them did not change the cycle.
- **What starts each cycle (cap 1/800, it 651–663):** a burst of γ switching. gamma_jumped rises from 25–40 to 600–840 near-wall cells per iteration, gamma_area rises 0.022 → 0.033, and the largest γ change (≈ 0.8–1.0) sits at x ≈ 0.028–0.042, y ≈ 0.031–0.038, near the leading-edge bubble on the upper surface. CD and CM rise first, CL falls about 20–30 iterations later (1.206 → 1.165), then recovers slowly over about 100 iterations until the next burst. With the cap, ν̃ no longer spikes, but γ still switches in hundreds of cells at once.
- **Wall snapshots differ by phase, not by cap:** transition (ν_t/ν > 1) at x = 0.106 (baseline), 0.038 (cap 1/400), 0.040 (cap 1/800), 0.168 (cap 3/400); upper reattachment 0.06 to 0.14. The transition point moves over x ≈ 0.04 to 0.17 within one cycle, so no single snapshot is a result.
- **Against the user's ±5 % targets** (CL 1.182, CD 0.0174, CM 0.0751): the cycle mean CL 1.19 is inside the band, CD 0.0195 is about 12 % high and CM 0.068 about 10 % low, outside the band in both cases.

Conclusion: the remaining non-convergence is the discrete BCM onset switching a whole patch of cells near the leading-edge bubble at once (or a genuinely unsteady bubble). Next:
1. Run the BDF2 case (`e387_alpha8_unsteady.toml`) from the cap 1/800 α = 8 restart. Steady forces mean the cycle is numerical; a periodic signal means the bubble is physically unsteady and the steady solve cannot converge.
2. If numerical: smooth the onset (softplus on max(Reθ − Reθt, 0), C¹ sqrt), keep cap 1 to 3, and revalidate α = 4.

## Update 2026-10-20: BDF2 run without the cap (plot only)

Source: the user's plot `unsteady_precap_coefficients.png` in the cap-runs folder (Δt 0.02, 5 transits, before the ν̃ cap, from the old α = 8 restart). The uploaded log and history from that message are a different, newer run: 15 steps to t = 0.3 with CL ≈ 1.196 to 1.198, and the log holds only its header line.
- The forces do not settle within 5 transits. There is one large excursion at t ≈ 1.3–2.5 (CL up to 1.256, CD 0.026, CM 0.090), then a slower drift (CL 1.21 → 1.149, CD 0.013 → 0.022 at t = 3.5–5). There is no repeated period yet. The timescale is about 2 convective units, which is slow for laminar-bubble shedding.
- The inner iterations hit their limit at about 10 steps between t ≈ 1.25 and 1.65, exactly where the excursion starts. Those steps are not time-accurate, so the excursion may be numerical.
- The run is not conclusive. Repeat it with the cap (1 to 3), from the cap 1/800 α = 8 restart, for at least 10 to 15 transits, and check that no step hits the inner limit.

## Update 2026-10-21: BDF2 run from the cap 1/800 restart (log pasted in the thread, stopped at step 381, t = 7.62)

Setup: Δt = 0.02, 30 inner iterations, 3 orders per step, no limiter, no ν̃ cap (the cap is not wired into the unsteady solver), restart `alpha_8_iter800_cap1`. The run was stopped by hand at 5.2 h (18 685 s) of 750 steps, with no restart file.

- **Steps 1 to 277 (t ≤ 5.54): smooth and cheap.** 3 to 6 inner iterations, 30 to 60 GMRES iterations, about 30 s per step. There is no 120-iteration cycle in physical time. CL rises from 1.211 to 1.217 (t = 0.72), then falls slowly to a minimum of 1.1496 at t ≈ 5.15 and rises again. CD runs 0.0159 (t = 1.6) to 0.0222 (t = 4.5). CM 0.0759 → 0.0654. That is a slow drift over about 4 chord lengths of flow, not a repeating cycle.
- **From step 313 (t = 6.26) the inner solver loses convergence.** Inner iterations 15 to 30, GMRES 200 to 740, 250 to 300 s per step, and 12 steps hit the inner limit (316, 317, 328, 329, 332, 333, 343, 347, 349, 372, 380, 381; the last two reach only 0.8 and 2.2 orders, log R_p = −6.0 at step 380). Precursors: single steps with log R_u up by about 1 order (262, 273, 297 to 298) and a first hard step at 278 to 279 (18 and 13 inner iterations).
- **The force excursion starts together with the failing inner iterations.** CL 1.17 → 1.230, CD 0.026 → 0.038, CM 0.073 → 0.097 up to t = 7.28 (step 364). Then it collapses within 7 steps (0.14 time units): CL 1.132, CD 0.0194, CM 0.0488 at step 371, well below anything seen in the steady runs (CM 0.066 to 0.073). It rises again by CL +0.10 within 0.18 time units, so the oscillation period is at most about 0.3 chord lengths of flow. That is very fast for a laminar separation bubble.
- **Same pattern as the earlier BDF2 run without the cap:** its excursion at t ≈ 1.3 to 2.5 also started at steps that hit the inner limit. Here the excursion is later (t = 6.3, not 1.3) and larger (CD 0.038, not 0.026).
- **Reading:** the forces are not trustworthy once steps hit the inner limit, so the run does not show that the bubble is physically unsteady. The unconverged inner solves are consistent with the same discrete γ switching, now inside the inner Newton iteration, but this is not proven. The BDF2 test is therefore still open.

Next: wire `nu_tilde_step_limit` into the unsteady solver (the cap acts on the inner Newton update), rerun with the new restart files (below) so a stop does not lose the run, and compare the step at which inner convergence is first lost.

## Restart files during a run (branch claude/gamma-diagnostics-relaxation, commit d6351ca)
- `output.restart_interval = n`: every n iterations (steady) or steps (unsteady), alternating `<name>.restart_a.npz` and `<name>.restart_b.npz`.
- `output.restart_backup_interval = m`: `<name>.restart_<number>.npz`, never replaced.
- Writes go through a temporary file; a write that falls on a rejected iteration or an unconverged unsteady step waits for the next good one. Files store the step and time; an unsteady restart uses the state only (first step BDF1).

## Update 2026-10-21: cap in the unsteady solver, listed backups, rerun case

Branch claude/gamma-diagnostics-relaxation, commit e1f157c.
- `solver.nu_tilde_step_limit` now also caps every inner Newton update of the dual-time solver (`solve_unsteady` passes `limit_step` to each inner `solve_steady`). The log shows "capped at f x max(|nu_tilde|, nu) per cell (inner iterations)".
- New `output.restart_backup_at = [n1, n2, ...]`: backups `<name>.restart_<n>.npz` at listed iterations or steps, additive to `restart_backup_interval` (or alone). A listed number on a rejected iteration or an unconverged step is written at the next good one.
- Requirements: D56 (restart files during a run), D57 (ν̃ step cap). configuration.md and CHANGELOG updated.
- Rerun case: /mnt/project-files/cases/e387_alpha8_unsteady_cap1.toml (cap 1, end_time 15, restart_interval 25, restart_backup_interval 100, restart_backup_at [260, 275, 300, 310]). The restart path is a placeholder.

### Physical or BCM switching? Current reading (not decided)
For switching / numerical:
- The steady cycle has the same band and period (115–150 iterations) with and without the cap and with γ relaxation. Pseudo-time iterations have no time scale, and a pseudo-time period that does not change with the CFL path is not a physical period.
- Every cycle starts with a γ burst (hundreds of near-wall cells switch at once near x ≈ 0.03–0.04).
- In BDF2 the inner Newton iteration fails exactly where the forces jump (step 313 on), and γ is stop_gradient in the Jacobian: a cell that flips γ inside the inner iteration is a change that Newton cannot see, the same mechanism as in the steady run.
- The fast part of the BDF2 excursion (period ≲ 0.3 c/U) only exists on steps at the inner limit.
- URANS with an algebraic (local) transition model normally gives a steady laminar bubble at this Re; bubble shedding needs scale-resolving simulation.

For a physical (or model-physical) slow mode:
- The converged BDF2 part (t ≤ 5.5) is not steady: CL 1.217 → 1.150 → rising over about 4–5 c/U, a slow drift with an amplitude similar to the steady cycle band. That could be a slow bubble "breathing" of the URANS-BCM solution, which a steady solver would see as a cycle.

Decisive tests:
1. The rerun with the cap: if every step converges and the forces settle → steady solution exists, cycle is numerical. If a slow oscillation persists with all steps converged, repeat a stretch with Δt = 0.01 (from a backup): an unchanged period means a real (URANS-BCM) unsteadiness.
2. Smooth BCM onset (softplus on max(Reθ − Reθt, 0), C¹ sqrt) in the steady solver. Cycle gone → it is the switching.

## Update 2026-10-21 (interim): BDF2 rerun with nu_tilde cap 1, log up to step 312 (t = 6.24)

Data: /mnt/project-files/uploads/e387_alpha8_cap_runs/unsteady_cap1_interim_312.log (run still going).
- No step at the inner limit so far. Maximum 11 inner iterations (step 279), against 30 and 12 steps at the limit from step 313 on in the run without the cap. Steps 313 onwards are not in the log yet.
- Steps 1 to 277 reproduce the uncapped run to 3–4 digits (CL max 1.2167 at t = 0.72, CL min 1.1496 at t = 5.14–5.16, CD min 0.01587, CM min 0.0653). The cap is inactive in the smooth part, as expected.
- The same events occur at the same steps with and without the cap: log R_u (end of inner solve) jumps by about 1 order at steps 262, 273, 282, and the hard step is at 278–279 (8 and 11 inner iterations, GMRES 116 and 127; uncapped 18 and 13). The cap softens the hard step but does not remove it, and the events are tied to the state, not to the nu_tilde step.
- The forces are already on the excursion path of the uncapped run (t = 6.24: CL 1.172, CD 0.0254, CM 0.0718, all rising, CD rate +0.0002 per step), while the inner iterations are still easy (3 per step). Log R_u and R_nu rise together over steps 310–312 (−11.57 → −10.45 and −13.58 → −12.56).
- Decisive: do steps 313 onwards converge (inner iterations well below 30) while CL rises towards 1.23 and CD towards 0.038? Yes → the excursion is time-accurate (physical within URANS-BCM); no → the same inner failure as before, and the gamma-switching hypothesis gains weight.

## Update 2026-10-21 (later): BDF2 rerun with cap 1, log up to step 387 (t = 7.74)

Data: /mnt/project-files/uploads/e387_alpha8_cap_runs/unsteady_cap1_interim_latest.log (run still going, about 85 s per step from step 313 on, 27 s before).
- **The inner iterations converge.** Only 2 of 75 steps since step 313 hit the inner limit (313 and 357), against 12 steps at the limit without the cap. Mean 10.6 inner iterations (3.8 before step 313), maximum 30, GMRES mean 152 (maximum 450).
- **The force excursion and collapse are reproduced.** Peak at step 364 (t = 7.28): CL 1.2300, CD 0.0382, CM 0.0970; collapse to step 371: CL 1.1326, CD 0.0195, CM 0.0489. The uncapped run gave 1.230 / 0.038 / 0.097 and 1.132 / 0.0194 / 0.0488. Agreement to 3 digits, with 12 failed inner solves in one run and 2 in the other.
  **Correction of the earlier readings** (BDF2 update of 2026-10-21 and the interim note): the excursion is not an artefact of unconverged inner iterations. It is a property of the time-accurate solution of this discretisation and model.
- **It is a periodic oscillation.** CL extrema: peaks at t = 7.04 (1.2270), 7.28 (1.2300), 7.60 (1.2307); troughs at 7.14 (1.2028), 7.42 (1.1326), 7.72 (1.1385). Period about 0.3 chord lengths of flow; CL 1.13–1.23, CD 0.0195–0.0406, CM 0.049–0.098. It grows out of the smooth state after t ≈ 5.2, like a self-excited mode, and has not died out after about 2.5 cycles. The 130-iteration steady cycle has no direct relation to it (no time scale in pseudo-time); more likely the steady solver has no stable steady state to reach.
- **Still open: physical bubble dynamics or BCM threshold switching as the driver.** Both are deterministic and would reproduce identically. Tests: (1) time step 0.01 from restart_000300 (case /mnt/project-files/cases/e387_alpha8_unsteady_cap1_dt001.toml): unchanged period and amplitude means no time-step artefact; (2) smoothed BCM onset (softplus, C1 sqrt) in the unsteady run: oscillation unchanged means bubble dynamics; (3) near-wall gamma statistics per step, to see whether gamma flips line up with the hard steps 313 and 357.

## Update 2026-10-21 (evening): BDF2 with dt = 0.01 from restart_000300 (t = 6.0), 180 steps to t = 7.8

Data: /mnt/project-files/uploads/e387_alpha8_cap_runs/unsteady_cap1_dt001.log (6198 s, 0 steps at the inner limit, mean 5.5 inner iterations, maximum 12).
- **Onset is time-step independent up to t = 7.0.** CL/CD/CM at t = 7.0: 1.2177 / 0.0346 / 0.0909 (dt 0.01) against 1.2156 / 0.0342 / 0.0894 at 6.96 (dt 0.02).
- **The oscillation exists at dt = 0.01, but it is quantitatively different.** Peaks at t = 7.18 (CL 1.2441, CD 0.0403, CM 0.1048), 7.41 (1.2264), 7.68 (1.2301); troughs at 7.29 (1.1104, CD 0.0136, CM 0.0385), 7.50 (1.1237), 7.76 (1.1667). Peak-to-peak intervals 0.23 and 0.27, trough-to-trough 0.21 and 0.26, against 0.32 and 0.30 at dt = 0.02. The dt = 0.02 run had CL 1.13–1.23, CD >= 0.0195 and CM >= 0.049, so it damped the collapses. The period is about 0.25 (12 steps per period at dt = 0.02, 25 at dt = 0.01), irregular rather than a clean cycle.
- **The ν̃ and R_u events sit at fixed physical times.** R_u and R_nu rise at t = 6.22–6.23 in both runs (dt 0.02: steps 311–312; dt 0.01: steps 22–23, log R_nu −13.0 → −12.0), so they are tied to the state, not to the time discretisation.
- **Open question unchanged.** The oscillation is a relaxation-type cycle (slow rise of CL/CD, collapse within about 0.1, recovery): the bubble shrinks (CD below the steady-cycle minimum) and regrows, consistent with the BCM threshold feeding back on the bubble. Bubble breathing in the model without a switching artefact is not excluded.
- **Next test:** per-step near-wall gamma statistics (output.transition_diagnostics now works in unsteady runs, branch commit cb1ac97), case /mnt/project-files/cases/e387_alpha8_unsteady_cap1_dt001_gamma.toml (restart_000100 of the dt = 0.01 run, 80 steps). Compare gamma_area and gamma_jumped with CL/CD: if gamma_area/jumped lead the CL collapse by a fixed lag, the threshold switching drives the cycle; a smoothed onset or a lagged gamma in the unsteady solver would then be the next change.

# Choosing a model

Which physics to select with `init-case --model` (or `[physics]` in the case
file) for an airfoil, and what to expect from each. Every statement here is
backed by a case in [`validation.md`](validation.md); where a model has a known
weakness it is listed in [§11](validation.md#sec-11) there.

## Decision table

| your flow | model | `init-case --model` | notes |
|---|---|---|---|
| Re ≳ 10⁶, fully turbulent or transition fixed by a trip | SA, fully turbulent | `sa` | cheapest turbulent option; NACA 0012 at Re 6·10⁶ within a few percent of NASA TMR ([§6](validation.md#sec-6)) |
| Same, with separation or pressure-gradient effects where SST is preferred | k-ω SST (SST-2003m) | `sst` | two transport equations instead of one; on a flat plate within the TMR band ([§9](validation.md#sec-9)) |
| model-aircraft or glider Reynolds numbers, Tu ≲ 0.5 %, laminar separation bubble | SA-AFT2019b | `aft` | transition from an e^N envelope, N_crit from Tu (8.15 at 0.1 %), comparable with XFOIL; E387 α 4° bubble within 0.003 c of XFOIL, SD7003 reattachment 0.644 ([§8](validation.md#sec-8), [§9](validation.md#sec-9)) |
| elevated freestream turbulence (Tu ≳ 0.5 %), bypass transition | SST-LM γ-Re_θ | `lm` | T3A−, T3A, T3B ([§10](validation.md#sec-10)); needs `cfl.control = "growth"` (written by `init-case`); on the SD7003 polar the drag is too high from α = 4° and the stall comes early, use `aft` for drag and stall ([§10](validation.md#sec-10), [§11](validation.md#sec-11)) |
| low Re, quick look without a transition model | BCM | `bcm` | algebraic intermittency; the E387 α 4° run converges, α 8° oscillates ([§7](validation.md#sec-7-e387)) |
| no viscous effects | Euler | `euler` | lift from the potential flow with the vortex correction |
| fully laminar, low Re | laminar | `laminar` | no turbulence or transition model at all (cavity, DFG cylinder, plates, [§2](validation.md#sec-2) to [§4](validation.md#sec-4)) |

## Reading the table

- **Tu decides between AFT and LM.** AFT is an e^N model fitted for Tollmien–
  Schlichting transition: it is right for Tu up to about 0.5 % (flight, quiet
  tunnels) and about three times too early at Tu 0.87 % (T3A−). LM covers
  that case (rise at 1.35·10⁶ against 1.44·10⁶ measured) and bypass transition
  (T3B). AFT cannot transition before Re_x = 1.3·10⁵ by construction.
- **Laminar separation bubbles at low Re:** SA-AFT gives the better bubble
  (SD7003 α 4°: reattachment 0.644, LES/DNS 0.60–0.67); SST-LM gets C_L and C_M
  inside the LES/DNS bands but the bubble is too long (0.71) and C_D 3 % above
  the highest LES/DNS value.
- **Resolution:** the drag of the NACA 0012 needed 400–500 points per airfoil
  side (113 gave 13 % too much pressure drag), and the wall needs y⁺ ≈ 1
  ([PyAero guide](pyaero.md) §2 and §10; `check-mesh` warns).
- **Cost:** SA-AFT carries six unknowns per cell (about twice BCM), SST-LM
  seven. On the 320k-cell SD7003 mesh an LM iteration takes 20 to 90 s
  (the GMRES effort varies with the CFL, [§10](validation.md#sec-10)) and a run
  from the freestream converged in about 2000 iterations; with the 1.1
  default `positive_jacobian` a restart of the converged α = 4° state needed
  13 Newton–Krylov iterations.
- **Convergence:** `solver.cfl.control = "growth"` is the setting for LM and
  recommended for any transition model if the `ser` control of the AFT and BCM example cases creeps (the default is `growth`);
  `solver.convergence = { fields = ["cd"] }` ends a run when the force has
  settled (SU2's Cauchy criterion). Steady runs can fail near a bubble burst:
  E387 α −2° and 8° with SA-AFT did not settle, see §11 of the validation page
  for the cause and the likely remedy (a time-accurate run).

## Known limits in one place

[`validation.md` §11](validation.md#sec-11) lists them with causes: AFT on
T3A−/T3B, LM on the SD7003 bubble length, the low turbulent C_f after a bubble,
and the E387 α −2°/8° convergence.

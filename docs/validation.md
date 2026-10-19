# Verification and validation

What the solver has been checked against, in the order of the stage plan
(`requirements.md`). *Verification* cases have an exact or manufactured
solution and test the discretisation; *validation* cases compare with
benchmark computations or experiments. Every figure is produced by one
example script from the repository root; the numbers are from the runs
behind the figures (iMac 2019, 3.6 GHz 8-core Intel Core i9, 32 GB,
October 2026, v0.7.1; section 7 from the 0.8 development).

Click a case to jump to its section; each section ends with a link back here.

<a id="stage-table"></a>

| stage | case | type | reference | script |
|---|---|---|---|---|
| 1 | [convection–diffusion](#sec-1) | verification | manufactured exact solution | `examples/stage1_convection_diffusion.py` |
| 2 | [Poiseuille flow](#sec-2) | verification | exact profile | `examples/stage2_laminar.py --re1000` |
| 2 | [lid-driven cavity](#sec-2) | validation | Ghia, Ghia & Shin 1982 | same |
| 3 | [inviscid cylinder](#sec-3) | verification | potential flow | `examples/stage3_benchmarks.py` |
| 3 | [laminar flat plate](#sec-3) | validation | Blasius | same |
| 3 | [DFG 2D-1 cylinder](#sec-3) | validation | Schäfer & Turek 1996 | same |
| 4 | [Taylor–Green vortex](#sec-4) | verification | exact decaying solution | `examples/stage4_unsteady.py --medium` |
| 4 | [DFG 2D-2 shedding](#sec-4) | validation | Schäfer & Turek 1996 | same |
| 5 | [turbulent flat plate](#sec-5) | validation | NASA TMR, SA | `examples/stage5_rans.py` |
| 6 | [NACA 0012, Re 6·10⁶](#sec-6) | validation | Ladson 1988, Gregory & O'Reilly 1970, NASA TMR | `examples/stage6_pyaero_validation.py` |
| 7 (0.8) | [BCM transition, flat plate](#sec-7-plate) | verification/validation | Blasius, ERCOFTAC T3A | `tests/verification/test_rans_plate.py`, `pyaeroflow run examples/cases/flat_plate_transition.toml` |
| 7 (0.8) | [BCM transition, Eppler 387](#sec-7-e387) | validation | McGhee, Walker & Millard 1988 (NASA TM 4062) | `pyaeroflow polar examples/cases/e387_transition.toml --alpha 4 6 8 2 0 -2` |

The figures are committed under `docs/figures/`; `examples/output/` (where
the scripts write) is not tracked.

<a id="sec-1"></a>

## 1. Order of accuracy: convection–diffusion

div(a u − ν ∇u) = 0 on the unit square with a = (1, 0.5), ν = 0.25 and the
exact solution u = exp(a · (x − (1, 1)) / ν) as Dirichlet data, on perturbed
meshes of all four cell types, four refinement levels each. The RMS error
falls with second order on every type; the last refinement step gives
orders 2.08 (quadrilaterals), 1.94 (triangles), 1.91 (polygons), 2.11
(mixed). The residual drops eleven orders in 21–28 pseudo-time iterations.

![convection–diffusion](figures/stage1_convection_diffusion.png)

[↑ back to table](#stage-table)

<a id="sec-2"></a>

## 2. Laminar flow: Poiseuille and lid-driven cavity

Plane Poiseuille flow on 2048 perturbed triangles: RMS error of u against
the exact parabola 2.1·10⁻⁴ (profile at the right of the figure).

Lid-driven cavity against Ghia, Ghia & Shin (1982), centre-line profiles:

| Re | mesh | iterations | max \|u − Ghia\| | max \|v − Ghia\| |
|---|---|---|---|---|
| 100 | 64 × 64 | 22 | 0.0038 | 0.0076 |
| 1000 | 128 × 128 | 59 | 0.0055 | 0.0168 |

![laminar](figures/stage2_laminar.png)

[↑ back to table](#stage-table)

<a id="sec-3"></a>

## 3. Inviscid cylinder, Blasius plate, DFG 2D-1

**Inviscid cylinder (Euler mode)** against potential flow, O-grids of
64 × 32 and 128 × 48 cells: max |C_p − exact| 0.043 and 0.016, drag
coefficient 0.0176 and 0.0031 (exact 0), lift 10⁻⁹.

**Laminar flat plate**, Re_L = 10⁵, 12 800 cells: C_f within 0.7–0.8 % of
Blasius for 0.2 < x < 0.9; the velocity profile at x = 0.5 deviates from
f′(η) by at most 0.006.

**DFG 2D-1** (cylinder in a channel, Re = 20, Schäfer & Turek 1996;
reference c_D 5.5795, c_L 0.01062, Δp 0.11752):

| mesh | cells | c_D | c_L | Δp | iterations |
|---|---|---|---|---|---|
| quad | 1 480 | 5.5153 (−1.15 %) | 0.01014 | 0.11175 (−4.9 %) | 18 |
| quad | 4 434 | 5.5425 (−0.66 %) | 0.01195 | 0.11524 (−1.9 %) | 17 |
| quad | 13 087 | 5.5644 (−0.27 %) | 0.01032 | 0.11690 (−0.5 %) | 17 |
| triangles | 8 868 | 5.5541 (−0.45 %) | 0.01050 | 0.11564 (−1.6 %) | 33 |

![benchmarks](figures/stage3_benchmarks.png)

[↑ back to table](#stage-table)

<a id="sec-4"></a>

## 4. Unsteady: Taylor–Green vortex and DFG 2D-2 shedding

**Taylor–Green vortex** (32 × 32 cells, ν = 0.05, t = 1), temporal order
of the dual-time scheme against a reference run with Δt = 0.0125:

| scheme | Δt = 0.2, 0.1, 0.05 | orders |
|---|---|---|
| BDF2 | 2.10·10⁻⁴, 5.18·10⁻⁵, 1.23·10⁻⁵ | 2.02, 2.08 |
| BDF1 | 7.21·10⁻⁴, 3.63·10⁻⁴, 1.82·10⁻⁴ | 0.99, 1.00 |

Three inner Newton–Krylov iterations per time step reach the three-order
inner drop.

**DFG 2D-2** (Re = 100, periodic shedding, Schäfer & Turek 1996; reference
St 0.295–0.305, c_D,max 3.22–3.24, c_L,max 0.99–1.01, Δp 2.46–2.50):

| mesh, Δt | St | c_D,max | c_L,max | Δp |
|---|---|---|---|---|
| 1.5k cells, 0.02 (default) | 0.244 | 3.13 | 0.82 | 2.31 |
| 4.4k cells, 0.01 (`--medium`, figure) | 0.278 (−7 %) | 3.199 (−1 %) | 1.059 (+5 %) | 2.439 (−1.5 %) |

The coarse mesh resolves the wake poorly and the shedding frequency and
amplitude come out low; the medium mesh is within the spread the DFG
participants reported. Both are deliberately small cases (two and ten
minutes); three inner Newton–Krylov iterations per step in both.

![unsteady](figures/stage4_unsteady.png)

[↑ back to table](#stage-table)

<a id="sec-5"></a>

## 5. Turbulent flat plate (SA-neg) against NASA TMR

The TMR "2DZP" flat plate, Re_L = 5·10⁶, fully turbulent, SA-neg,
against the TMR SA values (C_f at x = 0.97: 0.0027055, C_D: 0.0028562):

| mesh | cells | C_f(0.97) | C_D | y⁺ max | iterations |
|---|---|---|---|---|---|
| 48 × 48 | 2 880 | 0.002747 (+1.5 %) | 0.002918 (+2.2 %) | 0.64 | 71 |
| 96 × 64 | 7 680 | 0.002742 (+1.3 %) | 0.002908 (+1.8 %) | 0.68 | 87 |

The law of the wall at x = 0.97 (right panel) follows u⁺ = y⁺ in the
sublayer and the log law with κ = 0.41, B = 5.0 up to y⁺ ≈ 10³.

![flat plate](figures/stage5_rans.png)

[↑ back to table](#stage-table)

<a id="sec-6"></a>

## 6. NACA 0012 at Re 6·10⁶

Fully turbulent SA-neg on two PyAero C-meshes (first cell 5·10⁻⁶ c,
tunnel 5 chords, outlet 25 chords, point-vortex farfield correction): 113
points per airfoil side (100k cells) and 472 (295k cells). Lift and drag
against Ladson's tripped measurements (NASA TM 4074, 80 and 120 grit) and
NASA TMR's CFL3D/SA values; C_p at 10° against Gregory & O'Reilly (NASA
R&M 3726, Re 2.88·10⁶).

| α | CL 472 | CL Ladson 80/120 | CD 472 | CD 113 | CD Ladson 80/120 | TMR CL / CD |
|---|---|---|---|---|---|---|
| 0° | 0.0000 | −0.007 / −0.012 | 0.00825 | 0.00828 | 0.00809 / 0.00807 | 0 / 0.00819 |
| 4° | 0.4448 | 0.427 / 0.428 | 0.00883 | 0.00893 | 0.00823 / 0.00876 | |
| 8° | 0.8771 | 0.856 / 0.862 | 0.01081 | 0.01116 | 0.01028 / 0.00989 | |
| 10° | 1.0827 | 1.059 / 1.067 | 0.01258 | 0.01317 | 0.01191 / 0.01166 | 1.0909 / 0.01231 |
| 12° | 1.2758 | 1.249 / 1.259 | 0.01516 | 0.01612 | 0.01321 / 0.01277 | |

Lift within 1–2 % of the experiment on both meshes (TMR's SA is +2.6 % at
10°); the drag needs the chordwise resolution: with 472 points per side it
is +0.7 % of TMR at 0° and +2.2 % at 10° (pressure drag +3 %, viscous
+2 %). The farfield distance has been ruled out as a drag item with a
25-chord tunnel (`pyaero.md`, section 8). The remaining gap to Ladson at
12° is shared with the SA model and the fully turbulent assumption against
a tripped experiment.

![NACA 0012](figures/stage6_pyaero_validation.png)

The self-contained O-grid case (`examples/stage6_naca0012.py`, generated
mesh, 500-chord farfield, frozen limiter) reproduces the same lift (1.076
at 10° on 256 × 128) and serves the test suite; its drag at 0° is 13 %
high for the same reason as the 113-point PyAero mesh.

[↑ back to table](#stage-table)

<a id="sec-7"></a>

## 7. BCM transition: flat plate and Eppler 387

<a id="sec-7-plate"></a>

### Flat plate

The SA-noft2-BCM model (`physics.transition.model = "bcm"`, section 7.1 of
`docs/pyaero.md`) on the TMR plate domain at Re_L = 5·10⁵, laminar
freestream ν̃/ν = 0.1 (`tests/verification/test_rans_plate.py`,
`examples/cases/flat_plate_transition.toml`):

| Tu | mesh | result |
|---|---|---|
| 0.1 % | 48 × 48 | laminar over the whole plate: C_f within 5 % of Blasius for 2.5·10⁵ ≤ Re_x ≤ 9.5·10⁵ |
| 3.3 % (ERCOFTAC T3A) | 96 × 64 | laminar at Re_x = 0.75·10⁵ (C_f −4 % vs Blasius), transition at Re_x = 1.2·10⁵ (boundary-layer ν_t/ν first above 1 at x = 0.236), turbulent from Re_x = 2.5·10⁵ on: C_f = 0.00433 / 0.00355 at Re_x = 2.5·10⁵ / 9.5·10⁵ against 0.00480 / 0.00367 from the 0.0576 Re_x^−0.2 correlation |

In the T3A experiment C_f leaves the laminar curve at Re_x ≈ 1.3·10⁵ and
reaches the turbulent one near 3·10⁵, which the model reproduces with the
2020 correlation constants. One property to know: the intermittency feeds
back on itself through ν_t (the T₂ term), so the discrete problem can have
more than one steady state. On the 48-cell plate at Tu 3.3 % the converged
transition location depends on the start-up path (Re_x = 0.75·10⁵ or
1.2·10⁵ depending on the Jacobian used in the pseudo-time phase); on the
96-cell plate it does not. Resolve the laminar part of the boundary layer
(as PyAero's y⁺ ≈ 1 meshes do) and compare two start-ups when a transition
location matters.

[↑ back to table](#stage-table)

<a id="sec-7-e387"></a>

### Eppler 387 at Re 2·10⁵, Tu 0.1 %

`examples/cases/e387_transition.toml` on PyAero's C-mesh
`examples/meshes/e387_pyaero_y1.su2` (209k quads, first cell 2·10⁻⁴ c,
farfield 5 c above and below, point-vortex correction), run with
`pyaeroflow polar`: α = 4° from freestream, the other angles restarted from
the neighbouring one. Reference: Langley LTPT, McGhee, Walker & Millard
(1988), NASA TM 4062: c_l, c_d (wake rake) and c_m about c/4 from fig. 17(c)
(tunnel-wall corrections applied; the angles are the corrected ones),
turbulence level 0.06–0.18 % (the case uses 0.1 %), and the oil-flow
laminar separation (LS) and turbulent reattachment (TR) of table III. The
measurements give no transition location. c_m is nose-up positive, i.e. the
solver's C_M with the opposite sign.

| α | C_L | ref. | C_D | ref. | c_m | ref. | ν_t/ν > 1 at x | oil flow LS / TR |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.3805 | 0.352 | 0.00996 | 0.0105 | −0.0796 | −0.0782 | 0.669 | 0.48 / 0.74 |
| 2 | 0.5956 | 0.574 | 0.01118 | 0.0118 | −0.0787 | −0.0794 | 0.609 | 0.43 / 0.67 |
| 4 | 0.8075 | 0.785 | 0.01293 | 0.0133 | −0.0778 | −0.0803 | 0.555 | 0.40 / 0.62 |
| 6 | 1.0136 | 0.999 | 0.01509 | 0.0144 | −0.0760 | −0.0790 | 0.497 | 0.37 / 0.55 |
| −2 | 0.173–0.178 \* | 0.156 | 0.0114–0.0125 \* | 0.0133 | – | −0.0814 | 0.58, lower 0.15 \* | 0.53 / 0.80 |
| 8 | 1.10–1.20 \* | 1.182 | 0.019–0.035 \* | 0.0174 | – | −0.0751 | 0.04 \* | natural transition at 0.32 |

\* not converged in 400 iterations; ranges over iterations 150–400, the
location from the last iteration. The reference angles are −1.99, 0.01,
2.04, 3.99, 6.04 and 8.03°; the difference to the computed angles (at most
0.04°, about 0.004 in C_L) is not removed.

For the converged angles C_L is 0.015–0.028 above the measurement (lift
slope 0.107 against 0.109 per degree between 0° and 4°), C_D is within
5.3 % and |c_m| within 3.8 %. The location where ν_t/ν first exceeds 1 lies
0.05–0.07 c upstream of the measured reattachment at every angle. At 4° the
wall C_f changes sign at x = 0.401 (laminar separation, measured 0.40) and
at x = 0.686 (reattachment, measured 0.62). This is one grid and one
farfield distance; neither was varied.

At α = −2° and 8° the iteration does not converge: the transition location
keeps moving and the forces drift slowly (8°: C_D 0.019–0.035; −2°: a
lower-surface transition appears at x = 0.15). Whether the discrete problem
has no stable steady state there, or the linearisation with frozen
intermittency (CHANGELOG 0.8) cannot reach it, is open.

[↑ back to table](#stage-table)

## References

- Ghia, U., Ghia, K. N., Shin, C. T. (1982): High-Re solutions for
  incompressible flow using the Navier–Stokes equations and a multigrid
  method. J. Comput. Phys. 48, 387–411.
- Schäfer, M., Turek, S. (1996): Benchmark computations of laminar flow
  around a cylinder. In: Flow Simulation with High-Performance Computers
  II, Vieweg, 547–566.
- NASA Langley Turbulence Modeling Resource (TMR), 2D zero-pressure-gradient
  flat plate and 2D NACA 0012 airfoil cases, https://turbmodels.larc.nasa.gov.
- Ladson, C. L. (1988): Effects of independent variation of Mach and
  Reynolds numbers on the low-speed aerodynamic characteristics of the NACA
  0012 airfoil section. NASA TM 4074.
- Gregory, N., O'Reilly, C. L. (1970): Low-speed aerodynamic
  characteristics of NACA 0012 aerofoil section, including the effects of
  upper-surface roughness simulating hoar frost. NPL Aero Report 1308 /
  ARC R&M 3726.
- McGhee, R. J., Walker, B. S., Millard, B. F. (1988): Experimental results
  for the Eppler 387 airfoil at low Reynolds numbers in the Langley
  low-turbulence pressure tunnel. NASA TM 4062.

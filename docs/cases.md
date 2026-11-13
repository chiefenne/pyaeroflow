# Case catalogue

The case files in `examples/cases/` (48) with what each one is, the mesh it
needs and where it is discussed. Start a case with the command in the last
column; a case with a template or a library mesh runs after a plain
`pip install`, the others need a clone of the repository (their meshes are
30 MB files or come from external grids).

How to read the **Mesh** column:

* **template** – `pyaeroflow init NAME` writes the case file and builds the
  mesh (`pyaeroflow templates` lists them).
* **library** – the mesh is built by `pyaeroflow make-mesh NAME`
  (`--list` shows the names); the case file is in the repository.
* **repo** – the mesh file is in `examples/meshes/` of the repository.

Validation sections are in [Validation](validation.md), pictures in the
[Example gallery](gallery.md), command sequences in the
[Recipes](recipes.md). To start from your own airfoil mesh use
`pyaeroflow init airfoil --mesh FILE --model sa --re 6e6` (recipe 18).

## Channel and cavity flows (laminar, incompressible)

| Case file | Physics | Mesh | Shows | Where |
|---|---|---|---|---|
| `couette.toml` | laminar | template `couette` | plane Couette flow against the exact linear profile (requirement V1) | unit tests |
| `couette_unsteady.toml` | laminar, unsteady | template `couette-unsteady` | impulsively started Couette flow, time accuracy | unit tests |
| `poiseuille.toml` | laminar | template `poiseuille` | plane Poiseuille flow on perturbed triangles | validation §2 |
| `cavity_re100.toml` | laminar | template `cavity` | lid-driven cavity, Re 100, against Ghia et al. | validation §2 |
| `cavity_heated_ra1e5.toml` | laminar, energy | template `cavity-heated` | natural convection at Ra 1e5, mean Nu against de Vahl Davis (recipe 19) | validation §19 |

## Flat plates

| Case file | Physics | Mesh | Shows | Where |
|---|---|---|---|---|
| `flat_plate_rans.toml` | SA-neg | template `plate-sa` | turbulent plate, NASA TMR zero-pressure-gradient case | validation §5 |
| `plate_heated_sa.toml` | SA-neg, energy | template `plate-heated-sa` | the same plate heated 3 K: Stanton number against Kays and Crawford (recipes 19, 20) | validation §19 |
| `flat_plate_transition.toml` | SA + BCM | template `plate-bcm` | transitional plate at Tu 3.3 % | validation §7 |
| `lm_t3a.toml` | SST + Langtry–Menter | template `plate-lm-t3a` | ERCOFTAC T3A | validation §10 |
| `lm_t3am.toml` | SST + Langtry–Menter | template `plate-lm-t3am` | ERCOFTAC T3A− | validation §10 |
| `lm_t3b.toml` | SST + Langtry–Menter | template `plate-lm-t3b` | ERCOFTAC T3B | validation §10 |

## Jets and bluff bodies

| Case file | Physics | Mesh | Shows | Where |
|---|---|---|---|---|
| `jet_crossflow.toml` | laminar | template `jet-crossflow` | normal jet in crossflow, Re_d 100, velocity ratio 2 | gallery 5 |
| `jet_crossflow_inclined.toml` | laminar | template `jet-crossflow-inclined` | the same jet inclined at 60° | gallery 5 |
| `heated_cylinder_re40.toml` | laminar, energy | library `cylinder-ogrid` (n_radial 72, 60 radii) | heated cylinder at Re 40, mean Nu against Lange et al. | gallery 9, validation §19 |
| `cylinder_euler.toml` | Euler | template `cylinder-euler` | inviscid cylinder, pressure against the potential solution | validation §3 |
| `naca0012_re1000_a20_spinup.toml` | laminar, unsteady | template `naca0012-shedding` | NACA 0012 at Re 1000, α 20°: vortex shedding, first phase | gallery 1 |
| `naca0012_re1000_a20_video.toml` | laminar, unsteady | library `naca0012-shedding` | second phase of the shedding run: restart and write frames | gallery 1, recipe 7 |

## NACA 0012, incompressible

| Case file | Physics | Mesh | Shows | Where |
|---|---|---|---|---|
| `naca0012_rans.toml` | SA-neg, α 10° | template `naca0012-sa` | NASA TMR 2DN00 case on a 33k O-grid (the smallest airfoil RANS case) | validation §6 |
| `naca0012_sa_a10_band.toml` | SA-neg, α 10°, refinement band | repo `naca0012_tmr_225x65.su2` | band around the airfoil, cold start from freestream | gallery 7, recipes 10, 11 |
| `naca0012_pyaero_euler.toml` | Euler, α 10° | repo (PyAero 100k) | PyAero mesh, inviscid | [PyAero meshes](pyaero.md) |
| `naca0012_pyaero_rans.toml` | SA-neg, α 10° | repo (PyAero 100k) | TMR case on the PyAero C-mesh | validation §6 |
| `naca0012_pyaero_rans_far.toml` | SA-neg, α 10° | repo (PyAero 150k) | larger tunnel: the farfield distance is not the cause of the CD difference | [PyAero meshes](pyaero.md) |
| `naca0012_pyaero_rans_fine.toml` | SA-neg, α 10° | repo (PyAero 300k) | chordwise-refined mesh: chordwise resolution is the cause | [PyAero meshes](pyaero.md) |
| `naca0012_pyaero_transition.toml` | SA-neg, α 0°, prescribed transition | repo (PyAero 100k) | transition at a prescribed position | [PyAero meshes](pyaero.md) |
| `naca0012_slot_base.toml` | SA-neg, α 12° | repo (TMR 449 slot) | slot on the upper side, closed baseline | gallery 4 |
| `naca0012_slot_suction.toml` | SA-neg, α 12° | repo (TMR 449 slot) | suction through the slot | gallery 4 |
| `naca0012_slot_blowing.toml` | SA-neg, α 12° | repo (TMR 449 slot) | blowing through the slot | gallery 4 |

## NACA 0012, compressible

| Case file | Physics | Mesh | Shows | Where |
|---|---|---|---|---|
| `naca0012_euler_m05.toml` | Euler, M 0.5, α 1.25° | template `naca0012-euler-m05` | subsonic compressible Euler | validation §12 |
| `naca0012_euler_m08.toml` | Euler, M 0.8, α 1.25° | template `naca0012-euler-m08` | transonic, shocks (Roe flux) | validation §12, gallery 3 |
| `naca0012_euler_m08_slau2.toml` | Euler, M 0.8, α 1.25° | library `naca0012-euler-ogrid` | the same with the SLAU2 flux | validation §16 |
| `naca0012_laminar_m05.toml` | laminar, M 0.5, Re 5000 | repo (PyAero 100k) | Swanson–Langer case 1, separation at the trailing edge | validation §13 |
| `naca0012_sa_m015.toml` | SA-neg, M 0.15, α 10° | repo (TMR 225) | TMR 2DN00 case, compressible | validation §14 |
| `naca0012_sst_m015.toml` | SST, M 0.15, α 10° | repo (TMR 225) | TMR 2DN00 case with k-ω SST | validation §15 |

## Supersonic and transonic

| Case file | Physics | Mesh | Shows | Where |
|---|---|---|---|---|
| `ramp_10deg_m2.toml` | Euler, M 2 | template `ramp-m2` | oblique shock on a 10° ramp against the exact solution | validation §12, gallery 6 |
| `ramp_10deg_m2_fine.toml` | Euler, M 2 | library `ramp` (`--set nx=192 ny=128`) | the same on a finer mesh | validation §12 |
| `rae2822_case6_roe.toml` | SST, M 0.729, α 2.31° | repo (RAE 2822 320k) | transonic RAE 2822 case 6, Roe flux | validation §15 |
| `rae2822_case6_slau2.toml` | SST, M 0.729, α 2.31° | repo (RAE 2822 320k) | the same with SLAU2 | validation §16 |

## Low-Reynolds-number airfoils and transition

| Case file | Physics | Mesh | Shows | Where |
|---|---|---|---|---|
| `sst_sd7003.toml` | SST, α 4° | repo (SD7003 320k) | fully turbulent SD7003, Re 6·10⁴ | validation §9 |
| `sst_sd7003_ogrid.toml` | SST, α 4° | repo (SD7003 O-grid 39k) | the same on a small O-grid | validation §9, §10 |
| `aft_sd7003.toml` | SA-AFT, α 4° | repo (SD7003 320k) | transition bubble with the AFT model | validation §9 |
| `aft_sd7003_tu05.toml` | SA-AFT, α 4° | repo (SD7003 320k) | Tu of the Herbst et al. 2017 baseline | validation §9 |
| `aft_sd7003_tu05_blunt.toml` | SA-AFT, α 4° | repo (SD7003 blunt TE) | blunt trailing edge | validation §9 |
| `lm_sd7003.toml` | SST + Langtry–Menter, α 4° | repo (SD7003 320k) | LM bubble, Re 6·10⁴ | validation §10, gallery 2 |
| `lm_sd7003_polar.toml` | SST + Langtry–Menter | repo (SD7003 320k) | polar `--alpha 0 2 4 6 8` | validation §10, recipe 3 |
| `e387_transition.toml` | SA + BCM, α 4° | repo (E387 209k) | Eppler 387, Re 2·10⁵ | validation §7 |
| `e387_aft.toml` | SA-AFT, α 4° | repo (E387 209k) | Eppler 387 with the AFT model | validation §8 |

## Wall suction

| Case file | Physics | Mesh | Shows | Where |
|---|---|---|---|---|
| `nasa_hump_sa_noflow.toml` | SA-neg | repo (hump 409x109) | NASA wall-mounted hump, no flow control | validation §17 |
| `nasa_hump_sa_suction.toml` | SA-neg | repo (hump 409x109) | steady suction through the slot | validation §17 |

## Plot and analysis scripts

`e387_polar_compare.py` and `nasa_hump_plot.py` in the same folder compare
finished runs with measurements; `make_meshes.py` and `make_plate_lm.py`
write the library meshes next to the case files (the same meshes as
`pyaeroflow make-mesh`).

Airfoil coordinate files for `pyaeroflow foil` are in `examples/airfoils/`
(E387, SD7003; validation §18).

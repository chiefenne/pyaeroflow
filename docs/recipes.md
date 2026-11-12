# Recipes

Short command sequences for everyday tasks. Each recipe names the case-file
keys it uses; the full reference is [Case configuration](configuration.md).
Pictures of complete cases are in the [Example gallery](gallery.md).

1. [From a PyAero mesh to a result](#pyaero-to-result)
2. [Plot the convergence and the wall Cp and Cf](#plot)
3. [Run a polar](#polar)
4. [Continue or restart a run](#restart)
5. [Stop a run or write fields while it runs](#control-files)
6. [Unsteady run: field series, probes and the shedding frequency](#unsteady)
7. [Make a video of an unsteady run](#video)
8. [Use a Gmsh mesh](#gmsh)
9. [Refine the mesh during a run](#refinement)
10. [Put a refinement band around an airfoil](#band)
11. [Run a refined mesh from freestream](#refined-cold-start)
12. [Speed up the Newton phase](#speed)
13. [Clean up the output of a case](#clean)
14. [Estimate an airfoil in seconds, before meshing](#foil)
15. [Start a run from the boundary-layer solution](#ibl-start)
16. [Polar: exact slopes and a faster angle sweep](#polar-tangent)
17. [Avoid recompiling at every start](#compile-cache)
18. [Start a case from a template](#templates)

---

<a id="pyaero-to-result"></a>
## 1. From a PyAero mesh to a result

```
pyaeroflow check-mesh naca.su2 --re 6e6        # y+, surface points, farfield distance
pyaeroflow init-case naca.su2 --model sst --re 6e6 --alpha 4   # writes naca_sst.toml
pyaeroflow check naca_sst.toml                 # validate, print the resolved case
pyaeroflow run naca_sst.toml
```

`--model` is one of `sa` (default), `sst`, `bcm`, `aft`, `lm`, `laminar`,
`euler`. The run writes `<name>.log`, `<name>.history.csv`, the wall CSV
`<name>.wall_airfoil.csv` and `<name>.vtu` (ParaView) to `output.directory`.
`pyaeroflow fields naca_sst.toml` lists the fields the case can write. Mesh
settings for PyAero (y⁺, farfield) are in [docs/pyaero.md](pyaero.md).

[↑ back to top](#recipes)

---

<a id="plot"></a>
## 2. Plot the convergence and the wall Cp and Cf

```
pip install "pyaeroflow[plot]"
pyaeroflow plot output/case/case.log
pyaeroflow plot output/case/case.wall_airfoil.csv \
    --ref builtin:tmr_naca0012_cp_cfl3d_sa:alpha=10@CFL3D
pyaeroflow plot refs                            # list the built-in reference data
```

Several wall files overlay in one figure; `--label` names each one.
`--show` opens interactive windows instead of writing PNG files. All options:
[docs/plotting.md](plotting.md).

[↑ back to top](#recipes)

---

<a id="polar"></a>
## 3. Run a polar

```
pyaeroflow polar examples/cases/lm_sd7003_polar.toml --alpha 0 2 4 6 8
pyaeroflow plot examples/cases/output/lm_sd7003_polar/lm_sd7003_polar.polar.csv
```

The angles run as a restart chain, each one in its own `alpha_<α>/` folder,
and the coefficients go to `<name>.polar.csv`. Running the same command again
continues an interrupted angle from its newest backup and skips the converged
ones, so a polar can be extended by adding angles. `pyaeroflow clean
case.toml --backups --yes` makes every angle start over. Overlay measured
data with `--ref`, e.g. `--ref builtin:e387_tm4062_re200k@"NASA TM 4062"`.

[↑ back to top](#recipes)

---

<a id="restart"></a>
## 4. Continue or restart a run

```toml
[initial]
restart = "output/case/case.backup_000400.npz"
```

```
pyaeroflow restart-info output/case/case.backup_000400.npz   # iteration, time, solver state
pyaeroflow run case.toml
```

A restart of the same case continues the run: the counter, CFL, residual
reference and log go on, and `solver.max_iterations` and `time.end_time` are
totals. A backup of another case on the same mesh (for example a laminar or
steady run as the start of a RANS or unsteady one) starts a new run from its
fields. Backups are written every `output.restart_backup_interval`
iterations or steps.

[↑ back to top](#recipes)

---

<a id="control-files"></a>
## 5. Stop a run or write fields while it runs

Control files go into the output directory of the running case:

```
touch output/case/.stop                 # final backup, fields and wall files, then stop
touch output/case/.result_629           # fields at iteration (or step) 629
echo "vorticity q_criterion" > output/case/.result_629   # with extra fields
touch output/case/.read_toml_380        # re-read the case file at iteration 380
rm output/case/.stop                    # before the next start
```

What `.read_toml_N` may change during a run is listed in
[configuration: control files](configuration.md).

[↑ back to top](#recipes)

---

<a id="unsteady"></a>
## 6. Unsteady run: field series, probes and the shedding frequency

```toml
[time]
mode = "unsteady"
time_step = 0.02
end_time = 80.0

[output]
fields = ["pressure", "velocity", "vorticity", "q_criterion"]
fields_interval = 10                    # a VTU every 10 steps, plus <name>.pvd
probes = [[2.0, -0.4], [3.0, -0.6]]     # points in the wake
```

```
pyaeroflow run case.toml
pyaeroflow plot output/case/case.probes.csv --spectrum v --from 20
```

Open `<name>.pvd` in ParaView for the animation; [recipe 7](#video) makes
a video of it. The spectrum gives the shedding frequency; `--length` and `--velocity` make it a Strouhal number.
`time.convergence` ends the run once the time-averaged forces have settled.
A complete case is [gallery example 1](gallery.md#shedding).

[↑ back to top](#recipes)

---

<a id="video"></a>
## 7. Make a video of an unsteady run

Two runs: a spin-up until the shedding is periodic, then a restart that
writes a field file every few steps (the frames). The complete case is
[gallery example 1](gallery.md#shedding); the commands below use its files.

```
python examples/meshes/make_naca0012_shedding_mesh.py examples/cases/naca0012_shedding.su2
cd examples/cases
pyaeroflow run naca0012_re1000_a20_spinup.toml          # ends when the averaged C_L has settled
ls output/naca0012_re1000_a20/*.backup_*.npz             # the last backup is the start of the video
pyaeroflow restart-info output/naca0012_re1000_a20/naca0012_re1000_a20.backup_001775.npz
```

In the video case set `initial.restart` to that backup and `time.end_time`
to its time plus the length of the video (`end_time` is a total, the step
counter continues):

```toml
[initial]
restart = "output/naca0012_re1000_a20/naca0012_re1000_a20.backup_001775.npz"

[time]
end_time = 47.7                 # backup time 35.7 + 12 c/U, about 6 shedding periods

[output]
fields = ["vorticity", "q_criterion", "cp", "velocity_magnitude"]  # scalars only: smaller files
fields_interval = 2             # one frame every 2 steps (about 50 frames per period)
```

```
pyaeroflow run naca0012_re1000_a20_video.toml
```

The run writes `<name>_<step>.vtu` and the series file `<name>.pvd`
(300 frames of about 14 MB here, so about 4 GB).

In ParaView:

1. Open `<name>.pvd` (not the single VTU files) and press Apply.
2. Colour by `vorticity`, set a fixed range (Rescale to Custom Range, e.g.
   -5 to 5) and switch off automatic rescaling, otherwise the colours jump
   from frame to frame.
3. Set the view, then File > Save Animation: PNG frames (`frame.0000.png`, ...)
   or directly an `.avi`/`.ogv` file.

From PNG frames to a video or GIF with ffmpeg (25 frames per second):

```
ffmpeg -framerate 25 -i frame.%04d.png -c:v libx264 -pix_fmt yuv420p -crf 20 shedding.mp4
ffmpeg -i shedding.mp4 -vf "fps=15,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" shedding.gif
```

The second command makes a GIF with its own colour palette (sharper than the
default). Delete the frames afterwards with
`pyaeroflow clean naca0012_re1000_a20_video.toml --fields --yes`
([recipe 13](#clean)).

[↑ back to top](#recipes)

---

<a id="gmsh"></a>
## 8. Use a Gmsh mesh

```
pyaeroflow check-mesh wing.msh --re 1e6
pyaeroflow init-case wing.msh --model sa --re 1e6 --alpha 2
```

or, in a case file, `file = "wing.msh"` under `[mesh]`. The boundary names
are the physical curves of the 2D mesh. Supported formats and limits:
[configuration: Gmsh meshes](configuration.md#gmsh-meshes).

[↑ back to top](#recipes)

---

<a id="refinement"></a>
## 9. Refine the mesh during a run

```toml
[refinement]
cycles = 2
indicator = "rank(abs(vorticity) * h)"
top_fraction = 0.05                     # split the 5 % of cells with the largest indicator
```

Each cycle converges, marks cells by the indicator (any output field and
the cell sizes `h`, `h_u`, `h_v`), splits them with hanging nodes and goes on
from the transferred state. `<name>.refinement.csv` lists cells and
coefficients per cycle. Only quadrilaterals are split; steady runs only. All
keys: [configuration: refinement](configuration.md#refinement).

A band around the airfoil made before the run is [recipe 10](#band).

[↑ back to top](#recipes)

---

<a id="band"></a>
## 10. Put a refinement band around an airfoil

```toml
[refinement]
wall_curve = "spline"                   # "straight": keep the faceted input contour
[refinement.band]
distances = [0.05, 0.02, 0.008]         # m, one level each, strictly decreasing
# boundaries = ["airfoil"]              # default: all walls
# n_max = 2                             # split factor of a thin cell per level
```

```
pyaeroflow refine case.toml      # writes <output.directory>/<name>.band.vtu, no run
pyaeroflow run case.toml         # makes the same band, then runs (cycles = 0 is fine)
```

Each level splits the cells whose centre is closer to the wall than its
distance, thin cells along the wall and the others 2 × 2, so three levels
give 2, 4 and 8 times the wall resolution; the trailing edge and the near
wake are inside the band, so no hanging node sits on the wall. No solution
is needed. Choose the distances in chords: the outer one about the
boundary-layer thickness near the trailing edge, each further one about
half or less of the one before. Open the `.band.vtu` in ParaView with
Surface With Edges and colour it by `factor_u` (split factor along the
wall) or `band_distance` before you start the run; the `refine` command
also prints the cells per level, the wall faces and the largest aspect
ratio. A complete case with a before/after picture is
[gallery example 7](gallery.md#band); keys in
[configuration: refinement](configuration.md#refinement).

[↑ back to top](#recipes)

---

<a id="refined-cold-start"></a>
## 11. Run a refined mesh from freestream

```toml
[numerics]
limiter = "venkatakrishnan-wang"        # active, never frozen
limiter_threshold = 0.2

[solver]
cfl = { control = "ser" }
newton_krylov = { switch_at_residual_drop = 1.5, switch_at_iteration = 40 }
# nu_tilde_step_limit = 0.5             # default for incompressible sa-neg
```

```
pyaeroflow run case.toml
pyaeroflow plot output/case/case.log    # residuals, CFL and forces per iteration
```

A band mesh, with hanging nodes inside the boundary layer, needs no warm
start: the NACA 0012 225 band (54k cells) converges
7 orders from freestream in about 100 iterations with these settings. The
key is the SA step cap `solver.nu_tilde_step_limit`, which limits each update
of ν̃ in a cell to half its value; it is on by default for incompressible `sa-neg`
without a transition model. With `0` (off) the same run stalls at C_D 0.085
because ν̃ blows up in single Newton steps at the cells between two band
levels. A stall shows in the log as a flat residual with C_D far above the
expected value. If a mesh still stalls, solve on the original mesh first:

```toml
[refinement.band]
sequence_residual_drop = 3              # original mesh to 3 orders, then the band
```

The cap is off by default for SST, SA with transition and compressible SA;
these have not been run from freestream on a band mesh.

[↑ back to top](#recipes)

---

<a id="speed"></a>
## 12. Speed up the Newton phase

```toml
[solver]
linear_solver = { type = "pardiso" }    # pip install pypardiso (Intel MKL)
newton_krylov = { product_precision = "float32" }
```

PARDISO factorizes much faster than the default SuperLU and is the first
choice for meshes above about 50k cells. Single-precision Jacobian products
save about 9 % on the 320k SD7003 LM case with identical forces.

[↑ back to top](#recipes)

---

<a id="clean"></a>
## 13. Clean up the output of a case

```
pyaeroflow clean case.toml                    # list what the case wrote
pyaeroflow clean case.toml --fields --yes     # delete the VTU files
pyaeroflow clean case.toml --all --yes        # delete everything
```

Nothing is deleted without `--yes`.

[↑ back to top](#recipes)

---

<a id="foil"></a>
## 14. Estimate an airfoil in seconds, before meshing

```
pyaeroflow foil naca2412 --alpha 0 2 4 --re 1e6 --cp
pyaeroflow foil sd7003.dat --alpha 2 4 6 --re 6e4 --ncrit 9
```

`foil` is a panel method coupled to an integral boundary layer (the
viscous-inviscid approach of XFOIL-type codes, written from the published
papers). It takes a NACA 4-digit name or a coordinate file (Selig or
Lednicer) and prints CL, CD, CDf, CM and the transition positions per
angle. `<airfoil>_Re<re>.polar.csv` holds the table and `--cp` also writes a
wall file per angle. Use `--ncrit` for the freestream turbulence (9 is a
quiet wind tunnel, 1 fully turbulent) and `--panels` for the resolution
(200). Typical use: judge where the boundary layer transitions, and which
angles have a laminar bubble, before building a mesh. It is an estimate
for attached and mildly separated flow, not a replacement for the CFD
solution. NACA 2412, Re 10⁶, α 2: CL 0.452, CD 0.00578. The first angle
takes about 6 s (40 s in the very first run, which compiles; see
[recipe 17](#compile-cache)), the next ones about 1 s. In a polar each angle
starts from the previous one (`--cold`: every angle from scratch). Checks against mfoil, the E387 wind-tunnel data and
XFOIL: [validation §18](validation.md#sec-18); picture:
[gallery 8](gallery.md#foil). The last column says how each angle
was started: `warm`, `cold`, or `continuation from 5` when the cold start
stalled and `foil` came from a neighbouring angle on its own (E387 α 6°). An
angle marked `NO` did not converge even so.

[↑ back to top](#recipes)

---

<a id="ibl-start"></a>
## 15. Start a run from the boundary-layer solution

```toml
[initial]
boundary_layer = "airfoil"        # the wall boundary of the airfoil (one closed contour)
# boundary_layer_ncrit = 9        # default: 9 with transition or laminar flow, 1 for RANS only
```

The initial field is built from the same boundary-layer solution as
`pyaeroflow foil` at the Reynolds number and angle of the case: outer flow
with the displacement sources, a boundary-layer profile with the right
shape factor and a wake defect, and for Spalart–Allmaras a ν̃ from the
turbulent layers. The run starts closer to the answer; the log says if the
boundary-layer solution did not converge and the run starts from the uniform
state instead. NACA 0012, 100k cells, SA, Re 6·10⁶: 38 instead of 60
iterations at α 10 and 26 instead of 45 at α 0, same CL and CD. Building the
field takes about 60 s (mostly compilation), so it pays on large meshes
and in repeated runs. Incompressible flow only, ignored with
`initial.restart`, and in a polar only the first angle uses it. Not needed
for a [refined mesh from freestream](#refined-cold-start).

[↑ back to top](#recipes)

---

<a id="polar-tangent"></a>
## 16. Polar: exact slopes and a faster angle sweep

```
pyaeroflow polar case.toml --alpha 0 2 4 6 8 --predictor tangent
```

At every converged angle one extra linear solve (15–17 Jacobian products,
a few seconds on 33k cells) gives dw/dα. `<name>.sensitivity.csv` then holds
the exact slopes dCL/dα, dCD/dα, dCM/dα per degree (NACA 0012: dCL/dα
0.109 per degree, close to the thin-airfoil value 0.110). The next angle
starts from w + Δα·dw/dα if that start has the lower residual; the log
prints both residuals. Steady runs without `[refinement]` and without a
farfield vortex only.

For the speed of the sweep, the CFL restart is the stronger lever on the NACA
0012 test (33k cells, SA, 0/4/8°; iterations at 4° and 8°):

| Setting | α 4 | α 8 |
|---|---|---|
| default | 165 | 364 |
| `--predictor tangent` | 49 | 400, not converged |
| `[solver] cfl = { polar_reset = true }` | 43 | 345 |
| both | 134 | 400, not converged |

`polar_reset` restarts the CFL ramp at each angle (the Newton–Krylov switch
stays on). Use it when a new angle moves a laminar bubble or separation; the
predictor is for the slopes and for smooth, attached steps. Newton–Krylov
paths vary, so compare on your case before trusting one count. Neither
option was run on the 320k SD7003 LM case yet.

[↑ back to top](#recipes)

---

<a id="compile-cache"></a>
## 17. Avoid recompiling at every start

Compiled kernels are cached on disk, in `~/.cache/pyaeroflow/jax`, so a new
run, polar angle or refinement cycle with the same mesh and model does not
compile them again (33k SA case: setup 1.1 s to 0.4 s; larger models and
the Newton phase gain more). Nothing to switch on. Environment variables:

```
export PYAEROFLOW_CACHE_DIR=/scratch/pyaeroflow   # other place, e.g. a fast disk
export PYAEROFLOW_JAX_CACHE=0                     # switch the cache off
```

An own `JAX_COMPILATION_CACHE_DIR` is respected. The folder grows with
every new mesh size; delete it any time. A new JAX version starts a fresh
cache.

[↑ back to top](#recipes)

---

<a id="templates"></a>
## 18. Start a case from a template

```
pyaeroflow templates                          # list them (pyaeroflow templates plate: one group)
pyaeroflow init plate-sa -o my_plate          # case file and mesh in my_plate/
pyaeroflow run my_plate/flat_plate_rans.toml
pyaeroflow init jet-crossflow --name my_jet   # own case name and output folder
pyaeroflow make-mesh --list                   # the meshes alone
```

A template is one of the validated example cases (channel, cavity, flat
plates with SA, BCM and Langtry–Menter, jet in crossflow, cylinder, NACA 0012
shedding, SA and Euler, Mach 2 ramp) together with the mesh it runs on. `init`
writes the case file and builds the mesh next to it, so it works after a
plain `pip install`, without the repository. Edit the case file from there;
every key is in [Case configuration](configuration.md). The [Case catalogue](cases.md) lists
all 45 example cases with their physics, mesh and where they are discussed.

For your own airfoil mesh use the `airfoil` template, the same as
`pyaeroflow init-case`:

```
pyaeroflow init airfoil --mesh my_foil.su2 --model sa --re 1e6 --alpha 4
pyaeroflow init airfoil --mesh my_foil.su2 --model sa --re 1e6 --band --boundary-layer-start
```

`--model` is one of sa, sst, bcm, aft, lm, laminar, euler. `--band` adds a
[refinement band](#band) (bare: 0.05, 0.02, 0.008 chords; or give the
distances) with the limiter of the validated band case for SA.
`--boundary-layer-start` starts from the [boundary-layer
solution](#ibl-start). The case files of `init airfoil` use the farfield
vortex correction, so `polar --predictor tangent` falls back to the plain
chain there; use `cfl = { polar_reset = true }` for the polar instead.

[↑ back to top](#recipes)

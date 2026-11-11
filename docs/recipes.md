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
7. [Use a Gmsh mesh](#gmsh)
8. [Refine the mesh during a run](#refinement)
9. [Put a refinement band around an airfoil](#band)
10. [Run a refined mesh from freestream](#refined-cold-start)
11. [Speed up the Newton phase](#speed)
12. [Clean up the output of a case](#clean)

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

Open `<name>.pvd` in ParaView for the animation. The spectrum gives the
shedding frequency; `--length` and `--velocity` make it a Strouhal number.
`time.convergence` ends the run once the time-averaged forces have settled.
A complete case is [gallery example 1](gallery.md#shedding).

[↑ back to top](#recipes)

---

<a id="gmsh"></a>
## 7. Use a Gmsh mesh

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
## 8. Refine the mesh during a run

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

A band around the airfoil made before the run is [recipe 9](#band).

[↑ back to top](#recipes)

---

<a id="band"></a>
## 9. Put a refinement band around an airfoil

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
## 10. Run a refined mesh from freestream

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
## 11. Speed up the Newton phase

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
## 12. Clean up the output of a case

```
pyaeroflow clean case.toml                    # list what the case wrote
pyaeroflow clean case.toml --fields --yes     # delete the VTU files
pyaeroflow clean case.toml --all --yes        # delete everything
```

Nothing is deleted without `--yes`.

[↑ back to top](#recipes)

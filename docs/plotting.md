# Plotting: `pyaeroflow plot`

`pyaeroflow plot` makes PNG figures from the files a run writes. It needs matplotlib
(`pip install pyaeroflow[plot]`). The kind of plot is detected from the file name,
or given first: `pyaeroflow plot [log|wall|polar|probes|refs] FILE ...`.
`pyaeroflow plot --help` lists every option. Each figure is written next to the
input as `<name>_<what>.png`, or under the prefix given with `-o`.

| kind | input | figures |
|---|---|---|
| `log` | `<name>.log` (or `<name>.history.csv`) | `_residuals`, `_solver`, `_coeffs`, `_diffs` |
| `wall` | `<name>.wall_<boundary>.csv`, several to overlay | `_wall`: Cp, Cf (and y+, T, q_w with `--panels`) |
| `polar` | `<name>.polar.csv`, several to overlay | `_polar`: CL-α, CL-CD, cm-α, L/D-α |
| `probes` | `<name>.probes.csv` | `_probes` (time series), `_spectrum` with `--spectrum` |
| `refs` | none | lists the built-in reference data |

```
pyaeroflow plot out/n0012_sa_a10.log
pyaeroflow plot out/n0012_sa_a10.wall_airfoil.csv \
    --ref builtin:tmr_naca0012_cp_cfl3d_sa:alpha=10@CFL3D \
    --ref builtin:tmr_naca0012_cp_gregory:alpha=10@Gregory
pyaeroflow plot out/e387.polar.csv --ref builtin:e387_tm4062_re200k@"NASA TM 4062" --split-cd
pyaeroflow plot out/cylinder.probes.csv --spectrum v --from 20
```

## Log

The log holds every column (residuals, drop, CFL, NK, GMRES, coefficients, wall
time) and the notes (limiter released, step 1/N, src-dt, rejected ...). The history CSV
has the same numbers at full precision, but no drop column and no notes. So
`plot log` reads the log; given a history CSV, it takes the notes from the log next to
it, if there is one. The log columns are read by their position under the
header labels, so empty cells (no NK, no GMRES, no forces) are handled. This works
for incompressible and compressible runs and for every turbulence and transition model.

- Residuals: flow residuals on the left axis, turbulence and transition residuals
  dashed on the right axis.
- Solver: drop, CFL (log axis, capped at 1e5; unsteady runs show the inner
  iterations instead), GMRES iterations, the line-search step fraction (if a step
  was cut), wall time.
- Coefficients: steady runs fit the y-limits from 20 iterations before
  "limiter released". `--average-from t` (unsteady) adds the running mean.
- `--from`, `--to`, `--last N` restrict the range; `--no-notes` drops the note lines.
- `-t targets.toml`: target lines and bands, and y-windows centred on a target:

```toml
[targets.CD]
value = 0.0123
rtol = 0.02          # band ±2 %; tol = absolute half-width
[scale.CD]
tol = 0.004          # y-window CD ±0.004 around the target
```

Target keys are the column names: `CL`, `CD`, `CM`, `drop`, `CFL`, `GMRES`, `wall`,
`R_p` (also `Rp`, `log(R_p)`), and so on.

`plot_log.py` in the repository root is kept for existing commands and calls `plot log`.

## Wall

- Default `--x chord`: the curves are plotted over x/c, as in most of the airfoil
  literature. The upper surface is solid, the lower surface dashed, and the Cp axis
  is inverted. The contour is split at the smallest x.
- `--x arc`: one curve over the arc length s/c from the start of the wall file.
  This is better for blunt trailing edges and bases.
- Cf is flow-aligned by default (`--cf-sign flow`): the lower-surface values of the
  CSV, which carry the sign of the face tangent, are negated. Attached flow is then
  positive on both sides. The zero crossings are marked and printed: separation,
  reattachment, and the stagnation point on the lower surface.
- `--surface upper|lower`, `--x-range`, `--label` (one per file).

## Polar

- Points that did not converge are drawn hollow.
- cm is shown nose-up positive (cm = -CM; the solver's CM is counter-clockwise
  positive). `--cm-sign solver` shows the solver's CM instead.
- `--split-cd` adds the pressure and viscous drag. `--panels cl,drag,cm,ld,cd`
  selects the panels.

## Probes

- Time series per quantity (p, u, v, T), with all probes overlaid.
- `--spectrum probe1_v` (a column), `v` (every probe) or `all`: the amplitude spectrum
  from `--from` on (mean removed, Hann window). It prints the dominant frequency and
  the Strouhal number St = f L_ref / U_ref. The reference length and velocity come
  from `<name>.resolved.toml` next to the file, or from `--length`, `--velocity`.

## Reference data

Use `--ref FILE[:zone][@label]` or `--ref builtin:NAME[:zone][@label]`. `zone` picks
the zones whose title contains the text, for example `alpha=10` or `N_crit 9`.
Readable formats:
- CSV or whitespace columns with a header line
- Tecplot ASCII (`variables=`, `zone t=`, as the NASA TMR files)
- XFOIL polar files
- XFOIL `CPWR` / `DUMP` files

Columns are found by name: `x` (or `x/c`), `cp`, `cf`, `alpha`, `cl`, `cd`, `cm`.

Built-in data (`pyaeroflow plot refs` lists it, with its zones): NASA TMR NACA 0012
Cp, Cf, CL/CD of CFL3D SA and SST, Gregory Cp, Ladson CL/CD, TMR flat-plate Cf
(CC0-1.0), the E387 polar of NASA TM 4062 (public domain), and our XFOIL E387 polars.
Sources and licences are in `src/pyaeroflow/plotting/reference/README.md`. Data whose
redistribution terms are unclear (Penn State 2007, ERCOFTAC, Selig) is not included;
pass such files with `--ref`.

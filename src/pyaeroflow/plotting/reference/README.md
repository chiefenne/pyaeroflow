# Built-in reference data of `pyaeroflow plot`

Use with `--ref builtin:<name>[:zone]`, list with `pyaeroflow plot refs`.
Only data that may be redistributed is kept here.

| name | content | source | licence |
|---|---|---|---|
| `tmr_naca0012_cp_cfl3d_sa`, `_sst` | NACA 0012 Cp, CFL3D 897x257, alpha 0/10/15 (zones) | NASA Turbulence Modeling Resource, github.com/tmbwg/turbmodels (fetched 2026-10-24) | CC0-1.0 |
| `tmr_naca0012_cf_cfl3d_sa`, `_sst` | NACA 0012 Cf, upper surface, alpha 0/10/15 | same | CC0-1.0 |
| `tmr_naca0012_clcd_cfl3d_sa`, `_sst` | NACA 0012 CL, CD over alpha | same | CC0-1.0 |
| `tmr_naca0012_cp_gregory` | Gregory & O'Reilly, NASA R&M 3726 (1970), upper surface Cp, Re 2.88e6, digitised (approximate) | TMR, same | CC0-1.0 |
| `tmr_naca0012_clcd_ladson` | Ladson, NASA TM 4074 (1988), Re 6e6, tripped, three grit sizes | TMR, same | CC0-1.0 |
| `tmr_flatplate_cf_sst` | Zero-pressure-gradient flat plate Cf, SST-V, CFL3D and FUN3D | TMR, same | CC0-1.0 |
| `e387_tm4062_re200k` | E387 Re 2e5 polar, McGhee, Walker & Millard, NASA TM 4062 (1988) | NASA report | US government work, public domain |
| `e387_xfoil_re200k` | E387 Re 2e5 XFOIL polars, N_crit 8.15/9/10.07/11 | generated for pyaeroflow | MIT, as this package |

Not included (redistribution not clear): Penn State 2007 E387 data, ERCOFTAC T3 tables,
SD7003 data of Selig et al. and Herbst. Pass such files with `--ref <file>`.

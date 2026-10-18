# PyAeroFlow

2D unstructured cell-centred finite-volume Navier–Stokes/RANS solver in
JAX, developed as a companion to
[PyAero](https://github.com/chiefenne/PyAero).
Command and package: `pyaeroflow`. Formerly named *aerojax* (renamed in 0.8;
not related to other projects of that name).

**Status:** stage 6 of the [stage plan](docs/requirements.md#7-stage-plan-d53):
steady and unsteady laminar, inviscid and RANS (Spalart–Allmaras, SA-neg)
incompressible flow with the SU2 INC formulation, a Newton–Krylov solver and
BDF2 dual time stepping, verified on Couette and Poiseuille flow, the
lid-driven cavity (Ghia et al.), the inviscid cylinder, the Blasius flat
plate, the Taylor–Green vortex, the DFG cylinder-in-channel benchmarks
(steady 2D-1 and vortex shedding 2D-2), the NASA TMR turbulent flat plate
and the NACA 0012 at Re 6·10⁶ (generated O-grid; PyAero meshes can be used
directly).

## Documentation

- [Requirements](docs/requirements.md): decisions, planned features, stage plan
- [Case configuration](docs/configuration.md): TOML case-file reference
- [From a PyAero mesh to a result](docs/pyaero.md): mesh settings (y⁺, farfield), case file, log, limits
- [Verification and validation](docs/validation.md): every case with its reference, numbers and figure

## Installation

PyAeroFlow needs a conda environment from conda-forge. On Intel Macs this is the
only source of `jaxlib`.

```
conda env create -f environment.yml
conda activate pyaeroflow
pip install -e . --no-deps
```

## Running a case

```
pyaeroflow check case.toml        # validate, print the resolved configuration
pyaeroflow run case.toml          # solve; writes <name>.log, .history.csv, .vtu
```

Example case files are in `examples/cases/` (generate their meshes first with
`python examples/cases/make_meshes.py`; `examples/meshes/` holds three PyAero
meshes of the NACA 0012: 5- and 25-chord tunnels with 113 points per
airfoil side, and a 5-chord tunnel with 472). The case-file reference is
[docs/configuration.md](docs/configuration.md); the same setup is available
from Python through `pyaeroflow.driver.IncompressibleProblem`.

## Tests and examples

```
pytest                         # fast tests
pytest -m slow                 # order-of-accuracy studies, benchmarks
python examples/stage0_meshes.py
python examples/stage1_convection_diffusion.py
python examples/stage2_laminar.py [--re1000]
python examples/stage3_benchmarks.py
python examples/stage4_unsteady.py
python examples/stage5_rans.py
python examples/stage6_naca0012.py [--medium]
```

The examples write VTU files (for ParaView) and plots to `examples/output/`.

## License

MIT, see [LICENSE](LICENSE).

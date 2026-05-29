# Current Status

## Must-pass commands

- python -m compileall src tests examples
- python -m pytest

## Passing

- Baseline inspection found no pre-existing Python package, tests, examples, notebooks, or docs.
- PlanarSystem numerical core has been implemented in `src/ssmtoolpy/systems/planar.py`.
- PlanarSystem regression and JAX transform tests have been added.
- `python -m compileall src tests examples` passes.
- `python -m pytest` passes with 6 tests.
- `python examples/planar_system.py` runs and prints the active graph coefficients.

## Failing

- None.

## Active target

- MATLAB example or `.mlx` workflow: `SSMTool/examples/PlanarSystem/demo.mlx`
- Python example: `examples/planar_system.py`
- Jupyter notebook: `notebooks/planar_system.ipynb`
- Required MATLAB files:
  - `SSMTool/examples/PlanarSystem/build_model.m`
  - `SSMTool/examples/PlanarSystem/demo.mlx`
- Required Python modules:
  - `src/ssmtoolpy/systems/planar.py`
- Acceptance criteria:
  - Reproduce the graph SSM coefficients stated in the MATLAB live script.
  - Verify the polynomial vector field implied by MATLAB sparse tensors.
  - Verify the graph invariance equation for the nonzero terms.
  - Exercise representative JAX transforms on public numerical functions.

## Notes for next run

- Continue from the `Next recommended batch` section in `docs/migration_report.md`.
- JAX x64 is enabled in the PlanarSystem module to preserve the tight numerical tolerances used by the source-derived coefficient tests.

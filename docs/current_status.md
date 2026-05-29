# Current Status

## Must-pass commands

- python -m compileall src tests examples
- python -m pytest

## Passing

- Baseline for this batch passed:
  - `python -m compileall src tests examples`
  - `python -m pytest` with 6 tests
- PlanarSystem numerical core now computes graph coefficients through a minimal scalar first-order graph solver.
- New core modules have been implemented:
  - `src/ssmtoolpy/core/multiindex.py`
  - `src/ssmtoolpy/core/polynomial.py`
  - `src/ssmtoolpy/core/invariance.py`
- PlanarSystem and core regression/JAX transform tests pass.
- `python -m compileall src tests examples` passes.
- `python -m pytest` passes with 14 tests.
- `python examples/planar_system.py` runs and reports zero difference from the `demo.mlx` coefficient formula.

## Failing

- None.

## Active target

- MATLAB example or `.mlx` workflow: `SSMTool/examples/PlanarSystem/demo.mlx`
- Python example: `examples/planar_system.py`
- Jupyter notebook: `notebooks/planar_system.ipynb`
- Required MATLAB files:
  - `SSMTool/examples/PlanarSystem/build_model.m`
  - `SSMTool/examples/PlanarSystem/demo.mlx`
  - `SSMTool/src/@Manifold/private/Aut_1stOrder_SSM.m`
  - `SSMTool/src/@Manifold/private/Aut_1stOrder_RedDyn.m`
  - `SSMTool/src/@Manifold/private/coeffs_setup.m`
  - `SSMTool/src/@Manifold/private/multi_nsumk.m`
- Required Python modules:
  - `src/ssmtoolpy/systems/planar.py`
  - `src/ssmtoolpy/core/multiindex.py`
  - `src/ssmtoolpy/core/polynomial.py`
  - `src/ssmtoolpy/core/invariance.py`
- Acceptance criteria:
  - Produce PlanarSystem coefficients by a minimal general solver, not only by a closed-form helper.
  - Keep existing PlanarSystem tests passing.
  - Cover the smallest polynomial and multi-index functionality used by that solver.

## Notes for next run

- Continue from the `Next recommended batch` section in `docs/migration_report.md`.
- JAX x64 is enabled at package import to preserve tight source-derived coefficient tolerances.

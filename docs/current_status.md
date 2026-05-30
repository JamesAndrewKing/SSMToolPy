# Current Status

## Must-pass commands

- python -m compileall src tests examples
- python -m pytest

## Passing

- Baseline for the previous solver batch passed:
  - `python -m compileall src tests examples`
  - `python -m pytest` with 6 tests
- Baseline for the current Benchmark batch passed:
  - `python -m compileall src tests examples`
  - `python -m pytest` with 14 tests
- PlanarSystem numerical core now computes graph coefficients through a minimal scalar first-order graph solver.
- New core modules have been implemented:
  - `src/ssmtoolpy/core/multiindex.py`
  - `src/ssmtoolpy/core/polynomial.py`
  - `src/ssmtoolpy/core/invariance.py`
- PlanarSystem and core regression/JAX transform tests pass.
- `BenchamrkSSM1stOrder` is reproduced as a source-confirmed duplicate of PlanarSystem.
- `python examples/benchmark_ssm_1st_order.py` runs and reports zero difference from the analytical coefficients.
- A minimal fixed-structure parameter-to-loss gradient smoke test exists in `tests/test_parameter_to_loss.py`.
  It differentiates `decay -> PlanarSystem graph coefficients -> graph prediction -> scalar loss`.
- `Lorenz1stOrder` first bounded numerical core is implemented:
  - source model matrices and nonlinear tensor terms
  - MATLAB `lorenz.m` vector field
  - standard-parameter linear eigenvalue regression
  - parameter-to-output loss gradient smoke test
- `python examples/lorenz_1st_order.py` runs and prints the expected vector field and sorted eigenvalues.
- `python -m compileall src tests examples` passes.
- `python -m pytest` passes with 25 tests.
- `python examples/planar_system.py` runs and reports zero difference from the `demo.mlx` coefficient formula.

## Failing

- None.
- No current test differentiates through a full MATLAB-faithful pipeline with adaptive mode selection, full SSM construction, reduced dynamics prediction, and loss. The existing parameter-to-loss test freezes those discrete choices.

## Active target

- MATLAB example or `.mlx` workflow: `SSMTool/examples/Lorenz1stOrder/demo.mlx`
- Python example: `examples/lorenz_1st_order.py`
- Jupyter notebook: `notebooks/lorenz_1st_order.ipynb`
- Required MATLAB files:
  - `SSMTool/examples/Lorenz1stOrder/build_model.m`
  - `SSMTool/examples/Lorenz1stOrder/lorenz.m`
  - `SSMTool/examples/Lorenz1stOrder/demo.mlx`
- Required Python modules:
  - `src/ssmtoolpy/systems/lorenz.py`
- Acceptance criteria:
  - Lorenz source model and vector field are implemented with docstrings and differentiability classifications.
  - Standard-parameter eigenvalues match `demo.mlx`.
  - MATLAB `lorenz.m` vector field formula is covered by regression tests.
  - A parameter-to-output loss gradient smoke test covers a Lorenz system parameter.

## Notes for next run

- Continue from the `Next recommended batch` section in `docs/migration_report.md`.
- JAX x64 is enabled at package import to preserve tight source-derived coefficient tolerances.
- The Benchmark source directory intentionally uses the spelling `BenchamrkSSM1stOrder`.
- Parameter-to-loss differentiability is now an explicit migration objective. Current coverage is only the minimal PlanarSystem fixed-structure smoke test.
- Lorenz currently covers parameter-to-output differentiability, not an SSM-reduction loss.

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
- `python examples/benchmark_ssm_1st_order/example.py` runs and reports zero difference from the analytical coefficients.
- A minimal fixed-structure parameter-to-loss gradient smoke test exists in `tests/test_parameter_to_loss.py`.
  It differentiates `decay -> PlanarSystem graph coefficients -> graph prediction -> scalar loss`.
- `Lorenz1stOrder` first bounded numerical core is implemented:
  - source model matrices and nonlinear tensor terms
  - MATLAB `lorenz.m` vector field
  - standard-parameter linear eigenvalue regression
  - direct fixed-step trajectory computation
  - parameter-to-output loss gradient smoke test
- `python examples/lorenz_1st_order/example.py` runs and prints the expected vector field and sorted eigenvalues.
- `python -m compileall src tests examples` passes.
- `python -m pytest` passes with 28 tests.
- `python examples/planar_system/example.py` runs and reports zero difference from the `demo.mlx` coefficient formula.

## Failing

- None.
- No current test differentiates through a full MATLAB-faithful pipeline with adaptive mode selection, full SSM construction, reduced dynamics prediction, and loss. The existing parameter-to-loss test freezes those discrete choices.
- Current notebooks are not complete `.mlx` reproductions unless explicitly
  stated. In particular, `examples/lorenz_1st_order/lorenz_1st_order.ipynb`
  is incomplete because SSM graph computation, reduced prediction,
  SSM/full-trajectory visualization, and `reduced_to_full_traj` are not yet
  implemented.

## Active target

- MATLAB example or `.mlx` workflow: `SSMTool/examples/Lorenz1stOrder/demo.mlx`
- Python example: `examples/lorenz_1st_order/example.py`
- Jupyter notebook: `examples/lorenz_1st_order/lorenz_1st_order.ipynb`
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
- Lorenz direct trajectory computation is implemented with fixed-step RK4, but
  SSM-reduced trajectory mapping and SSM/full visualization are still missing.
- Example layout has been normalized for all currently reproduced examples:
  - `examples/planar_system/`
  - `examples/benchmark_ssm_1st_order/`
  - `examples/lorenz_1st_order/`
- Each reproduced example directory contains `README.md`, `example.py`, and its colocated notebook.
- Notebook completion now requires reproducing the meaningful MATLAB `.mlx`
  workflow, not only source setup or eigenvalue checks.

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
  - fixed-choice unstable SSM graph coefficients through order 5
  - SSM graph invariance residual checks
  - linear reduced trajectory computation
  - reduced-to-full SSM lifting
  - two-sided SSM curve and full trajectory arrays for visualization
  - short reduced/full trajectory comparison
  - parameter-to-output loss gradient smoke test
- Example-specific system/model helpers have been moved out of
  `src/ssmtoolpy/`:
  - `examples/planar_system/planar.py`
  - `examples/benchmark_ssm_1st_order/benchmark.py`
  - `examples/lorenz_1st_order/lorenz.py`
- `src/ssmtoolpy/` now contains only reusable core kernels and no package
  imports from `examples/`; the obsolete `src/ssmtoolpy/systems/` directory has
  been removed entirely.
- Revalidated after the layout correction:
  - Searching under `src/ssmtoolpy` for systems paths returns no files or
    directories.
  - Searches for obsolete package imports return no active source, example,
    test, README, or planning-doc hits.
- `python examples/lorenz_1st_order/example.py` runs and prints the expected
  vector field, sorted eigenvalues, SSM graph coefficient summary, invariance
  residual, reduced-to-full trajectory summary, reduced/full comparison, and
  small-amplitude trajectory final state.
- `examples/lorenz_1st_order/lorenz_1st_order.ipynb` executes end-to-end with
  `python -m jupyter nbconvert --to notebook --execute ...` and writes an
  executed notebook to `/tmp/lorenz_1st_order.executed.ipynb`.
- `python -m compileall src tests examples` passes.
- `python -m pytest` passes with 38 tests.
- `python examples/planar_system/example.py` runs and reports zero difference from the `demo.mlx` coefficient formula.
- `python examples/benchmark_ssm_1st_order/example.py` runs and reports zero
  difference from the analytical coefficients.
- `python examples/lorenz_1st_order/example.py` runs and reports fixed-choice
  SSM graph coefficients, invariance residual, and direct trajectory output.
- All current examples are classified against the same substantive workflow
  standard:
  - `PlanarSystem`: substantive scalar SSM graph coefficient subproblem
    implemented and tested; full MATLAB class workflow still incomplete.
  - `BenchamrkSSM1stOrder`: substantive duplicate coefficient comparison
    implemented and tested; full MATLAB class workflow still incomplete.
  - `Lorenz1stOrder`: fixed-choice live-script workflow reproduced and tested,
    including reduced-to-full trajectory comparison and visualization.

## Failing

- None.
- No current test differentiates through a full MATLAB-faithful pipeline with adaptive mode selection, full SSM construction, reduced dynamics prediction, and loss. The existing parameter-to-loss test freezes those discrete choices.
- Current notebooks are not complete `.mlx` reproductions unless explicitly
  stated. `examples/lorenz_1st_order/lorenz_1st_order.ipynb` is now complete
  for the tested fixed-choice Python/JAX reproduction of the Lorenz live-script
  workflow.

## Active target

- MATLAB example or `.mlx` workflow: `SSMTool/examples/TwoOscillators/demo.mlx`
- Python example: `examples/two_oscillators/example.py`
- Jupyter notebook: `examples/two_oscillators/two_oscillators.ipynb` only if
  a meaningful tested workflow section is implemented.
- Required MATLAB files:
  - `SSMTool/examples/TwoOscillators/build_model.m`
  - `SSMTool/examples/TwoOscillators/demo.mlx`
- Required Python modules:
  - `examples/two_oscillators/two_oscillators.py`
  - reusable kernels under `src/ssmtoolpy/core/`
- Acceptance criteria:
  - Model matrices from `build_model.m` match source-derived references.
  - At least one substantive TwoOscillators numerical workflow step beyond
    setup/eigenvalues is implemented and tested.
  - A Python example prints deterministic outputs from the tested core.
  - JAX transform tests cover any differentiable public helpers introduced.

## Notes for next run

- Continue from the `Next recommended batch` section in
  `docs/migration_report.md`, which now targets `TwoOscillators/demo.mlx`.
- JAX x64 is enabled at package import to preserve tight source-derived coefficient tolerances.
- The Benchmark source directory intentionally uses the spelling `BenchamrkSSM1stOrder`.
- Parameter-to-loss differentiability is now an explicit migration objective. Current coverage is only the minimal PlanarSystem fixed-structure smoke test.
- Lorenz now covers a fixed-choice differentiable SSM graph coefficient solve
  and lifted reduced prediction, but not a full adaptive SSM-reduction loss
  through mode selection or resonance classification.
- `Lorenz1stOrder` is complete for the tested fixed-choice Python/JAX
  live-script reproduction; remaining Lorenz limitations are generic/adaptive
  MATLAB class-stack limitations rather than missing live-script cells.
- Example layout has been normalized for all currently reproduced examples:
  - `examples/planar_system/`
  - `examples/benchmark_ssm_1st_order/`
  - `examples/lorenz_1st_order/`
- Example-specific model/helper code now lives beside each example; tests use
  explicit path handling to import those helpers.
- Each reproduced example directory contains `README.md`, `example.py`, and its colocated notebook.
- Notebook completion now requires reproducing the meaningful MATLAB `.mlx`
  workflow, not only source setup or eigenvalue checks.

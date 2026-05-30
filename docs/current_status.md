# Current Status

## Must-pass commands

- python -m compileall src tests examples
- python -m pytest

## Passing

- Fidelity audit baseline passed:
  - `python -m compileall src tests examples`
  - `python -m pytest` with 48 tests
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
- Reusable numerical kernels originally introduced for the Lorenz workflow have
  been moved out of `examples/lorenz_1st_order/lorenz.py` into core modules:
  - `src/ssmtoolpy/core/integrators.py`
  - `src/ssmtoolpy/core/graph.py`
  - `src/ssmtoolpy/core/trajectories.py`
  - `src/ssmtoolpy/core/invariance.py::solve_autonomous_quadratic_graph_coefficients`
  - `src/ssmtoolpy/core/invariance.py::univariate_graph_invariance_residual`
- Lorenz-specific code now stays in `examples/lorenz_1st_order/lorenz.py` as
  model definitions, the Lorenz quadratic term, eigenpair setup, and
  Lorenz-specific trajectory wrappers over reusable core kernels.
- Removed redundant renamed graph wrappers. Current examples now call
  `evaluate_univariate_graph`, `linear_reduced_trajectory`, and
  `two_sided_graph_curve` directly for generic graph/lifting/reduced-dynamics
  work.
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
  executed notebook to `/tmp/lorenz_1st_order.executed.ipynb`; nbformat emits a
  non-fatal missing-cell-id warning.
- `python -m compileall src tests examples` passes.
- `python -m pytest` passes with 48 tests.
- `python examples/planar_system/example.py` runs and reports zero difference from the `demo.mlx` coefficient formula.
- `python examples/benchmark_ssm_1st_order/example.py` runs and reports zero
  difference from the analytical coefficients.
- `python examples/lorenz_1st_order/example.py` runs and reports fixed-choice
  SSM graph coefficients, invariance residual, and direct trajectory output.
- All current examples are classified against the same substantive workflow
  standard:
  - `PlanarSystem`: `partial`; substantive scalar SSM graph coefficient
    subproblem implemented and tested; full MATLAB class workflow still
    incomplete; no MATLAB plot exists.
  - `BenchamrkSSM1stOrder`: `partial`; substantive duplicate coefficient
    comparison implemented and tested; full MATLAB class workflow still
    incomplete; no MATLAB plot exists.
  - `Lorenz1stOrder`: `plot-incomplete`; fixed-choice live-script workflow
    reproduced and tested, including reduced-to-full trajectory comparison and
    visualization, but full MATLAB object workflow and adaptive `ode45`
    sampling are not exactly reproduced.
- Fidelity checklists now exist in every migrated example README.
- Lorenz notebook visualization now uses the MATLAB live-script grid
  `t = linspace(0,1,100)`, includes the `x`, `y`, `z` labels, grid, view
  equivalent to `[15,35]`, and legend entries `SSM` and `Full`.
- Final audit checks passed:
  - `python -m compileall src tests examples`
  - `python -m pytest` with 48 tests
  - `python examples/planar_system/example.py`
  - `python examples/benchmark_ssm_1st_order/example.py`
  - `python examples/lorenz_1st_order/example.py`
  - all three migrated notebooks executed with `nbconvert` and only non-fatal
    missing-cell-id warnings.

## Migrated example fidelity

| Example | MATLAB source | Python example | Notebook | Numerical fidelity | Plot fidelity | Missing workflow steps | Tests |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PlanarSystem | `SSMTool/examples/PlanarSystem/build_model.m`, `demo.mlx` | `examples/planar_system/example.py` | `examples/planar_system/planar_system.ipynb` | `partial` | not applicable: MATLAB source has no plot | full `DynamicalSystem`/`SSM`/`compute_whisker` object workflow | `tests/test_planar_system.py`, `tests/test_core_graph_solver.py`, `tests/test_parameter_to_loss.py` |
| BenchamrkSSM1stOrder | `SSMTool/examples/BenchamrkSSM1stOrder/build_model.m`, `demo.mlx` | `examples/benchmark_ssm_1st_order/example.py` | `examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb` | `partial` | not applicable: MATLAB source has no plot | full `DynamicalSystem`/`SSM`/`compute_whisker` object workflow and exact `W0/R0` layout | `tests/test_benchmark_ssm_1st_order.py`, shared core tests |
| Lorenz1stOrder | `SSMTool/examples/Lorenz1stOrder/build_model.m`, `lorenz.m`, `demo.mlx` | `examples/lorenz_1st_order/example.py` | `examples/lorenz_1st_order/lorenz_1st_order.ipynb` | `plot-incomplete` | mostly matched: 3D SSM/full plot reproduced, exact `ode45` sampling and MATLAB styling not exact | full adaptive `compute_whisker` object workflow and adaptive `ode45` trajectory sampling | `tests/test_lorenz_1st_order.py`, core graph/integrator/invariance tests |

## Failing

- None.
- No current test differentiates through a full MATLAB-faithful pipeline with adaptive mode selection, full SSM construction, reduced dynamics prediction, and loss. The existing parameter-to-loss test freezes those discrete choices.
- No migrated notebook is currently marked `complete` after the fidelity audit.
  Lorenz is `plot-incomplete` because the visual workflow is close but does not
  exactly reproduce MATLAB `ode45` sampling or the full MATLAB object workflow.

## Active target

- MATLAB example or `.mlx` workflow: migrated-example fidelity audit
- Python example: existing `examples/<example_name>/example.py` scripts
- Jupyter notebook: existing migrated notebooks under `examples/<example_name>/`
- Required MATLAB files:
  - `SSMTool/examples/PlanarSystem/build_model.m`
  - `SSMTool/examples/PlanarSystem/demo.mlx`
  - `SSMTool/examples/BenchamrkSSM1stOrder/build_model.m`
  - `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx`
  - `SSMTool/examples/Lorenz1stOrder/build_model.m`
  - `SSMTool/examples/Lorenz1stOrder/lorenz.m`
  - `SSMTool/examples/Lorenz1stOrder/demo.mlx`
- Required Python modules:
  - reusable kernels under `src/ssmtoolpy/core/`
- Acceptance criteria:
  - Every migrated example README has a fidelity checklist.
  - Every migrated example is classified as `smoke`, `partial`,
    `plot-incomplete`, or `complete`.
  - At least one already migrated example is corrected toward MATLAB numerical
    or plot fidelity.
  - Existing tests and example scripts pass.

## Notes for next run

- Continue from the `Next recommended batch` section in
  `docs/migration_report.md`, which now targets the largest remaining fidelity
  gap among already migrated examples.
- JAX x64 is enabled at package import to preserve tight source-derived coefficient tolerances.
- The Benchmark source directory intentionally uses the spelling `BenchamrkSSM1stOrder`.
- Parameter-to-loss differentiability is now an explicit migration objective. Current coverage is only the minimal PlanarSystem fixed-structure smoke test.
- Lorenz now covers a fixed-choice differentiable SSM graph coefficient solve
  and lifted reduced prediction, but not a full adaptive SSM-reduction loss
  through mode selection or resonance classification.
- The Lorenz reusable-kernel cleanup compared the example helper code against
  `SSMTool/src/misc/reduced_to_full.m`,
  `SSMTool/src/misc/reduced_to_full_traj.m`,
  `SSMTool/src/misc/transient_traj_on_auto_ssm.m`,
  `SSMTool/src/@DynamicalSystem/odefun.m`, and
  `SSMTool/src/@Manifold/private/Aut_1stOrder_SSM.m`.
- Reusable Lorenz kernels are now covered by dedicated core tests:
  `tests/test_core_graph.py`, `tests/test_core_integrators.py`, and the
  quadratic graph-solver tests in `tests/test_core_graph_solver.py`.
- Search confirmed no remaining active source/example/test/README references
  to removed wrappers; only documentation history mentions the removed names:
  `evaluate_graph_trajectory`, `evaluate_planar_ssm_graph`,
  `evaluate_lorenz_ssm_graph`, `lorenz_reduced_trajectory`,
  `lorenz_reduced_to_full_trajectory`, or `lorenz_unstable_ssm_curve`.
- All colocated notebooks execute with `python -m jupyter nbconvert --execute`
  after adding project `src/` bootstrap paths to PlanarSystem and Benchmark;
  nbformat emits non-fatal missing-cell-id warnings.
- `Lorenz1stOrder` is now classified as `plot-incomplete`: the tested
  fixed-choice Python/JAX live-script reproduction is close, but exact MATLAB
  `ode45` sampling/styling and the adaptive MATLAB class stack remain missing.
- Example layout has been normalized for all currently reproduced examples:
  - `examples/planar_system/`
  - `examples/benchmark_ssm_1st_order/`
  - `examples/lorenz_1st_order/`
- Example-specific model/helper code now lives beside each example; tests use
  explicit path handling to import those helpers.
- Each reproduced example directory contains `README.md`, `example.py`, and its colocated notebook.
- Notebook completion now requires reproducing the meaningful MATLAB `.mlx`
  workflow, not only source setup or eigenvalue checks.

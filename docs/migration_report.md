# Migration Report

## Implemented modules

- `src/ssmtoolpy/core/multiindex.py`
  - `multiindices_of_total_degree`
- `src/ssmtoolpy/core/polynomial.py`
  - `evaluate_monomial_polynomial`
  - `collect_univariate_coefficients`
- `src/ssmtoolpy/core/invariance.py`
  - `solve_scalar_graph_coefficients`
- `src/ssmtoolpy/systems/planar.py`
  - `build_planar_system`
  - `planar_vector_field`
  - `planar_nonlinear_exponents`
  - `planar_nonlinear_coefficients`
  - `planar_ssm_graph_coefficients`
  - `evaluate_planar_ssm_graph`
- `src/ssmtoolpy/systems/lorenz.py`
  - `standard_lorenz_parameters`
  - `build_lorenz_system`
  - `lorenz_nonlinear_exponents`
  - `lorenz_nonlinear_coefficients`
  - `lorenz_vector_field`
  - `lorenz_linear_eigenvalues`

## Reproduced examples

- `PlanarSystem` numerical subproblem from `SSMTool/examples/PlanarSystem/demo.mlx`:
  graph coefficients `a_2` through `a_5` for the first-mode SSM are now
  computed by the minimal scalar first-order graph solver and checked against
  the formula stated in the live script.
- `BenchamrkSSM1stOrder` from
  `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx`: reproduced as a
  source-confirmed duplicate of PlanarSystem, with a named Python example and
  tests covering the misspelled source path.
- `Lorenz1stOrder` first bounded numerical core from
  `SSMTool/examples/Lorenz1stOrder/demo.mlx`: source model, vector field,
  standard-parameter linear eigenvalues, and parameter-to-output gradient smoke
  test.

## Reproduced notebooks

- `examples/planar_system/planar_system.ipynb` calls the tested PlanarSystem numerical API.
- `examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb` mirrors the Benchmark coefficient
  comparison and calls the same tested numerical API.
- `examples/lorenz_1st_order/lorenz_1st_order.ipynb` mirrors the tested Lorenz numerical core.

## Example layout

- Reorganized all currently reproduced examples into colocated directories:
  - `examples/planar_system/`
  - `examples/benchmark_ssm_1st_order/`
  - `examples/lorenz_1st_order/`
- Each directory now contains `README.md`, `example.py`, and the matching
  notebook when applicable.
- The former top-level `notebooks/` directory was removed because the notebooks
  now live beside their examples.
- Reusable numerical code remains under `src/ssmtoolpy/`; `ssmtoolpy` does not
  import from `examples`.

## Skipped or deferred items

- Full MATLAB `DynamicalSystem`, `SSM`, and `Manifold` class behavior.
- General multi-dimensional autonomous coefficient solving.
- Resonant reduced-dynamics extraction from `Aut_1stOrder_RedDyn.m`.
- Lorenz unstable SSM graph computation, `reduced_to_full_traj`, and ODE
  trajectory comparison from the second half of `demo.mlx`.
- Continuation, FRC/FRS, plotting, and external finite-element workflows.
- Notebook execution checks; no notebook execution tooling was configured before
  this batch.

## Known limitations

- This batch reproduces a one-master, one-transverse nonresonant scalar graph
  subproblem, not the full `compute_whisker` MATLAB implementation.
- The first target has no MATLAB-generated fixture file; reference values are
  source-derived from `demo.mlx`, `build_model.m`, and the inspected
  homological solve in `Aut_1stOrder_SSM.m`.
- `BenchamrkSSM1stOrder` intentionally reuses the PlanarSystem implementation
  because the MATLAB model source is equivalent.
- The current parameter-to-loss differentiability coverage is a minimal
  fixed-structure PlanarSystem smoke test. It does not yet include adaptive
  mode selection, a full MATLAB-faithful SSM construction, or nonlinear reduced
  dynamics prediction.
- Lorenz currently covers parameter-to-output differentiability only, not an
  SSM-reduction loss.

## Exact commands run

- `pwd`
- `ls`
- `sed -n '1,260p' AGENTS.md`
- `git status --short`
- `sed -n '1,260p' AGENTS.md`
- `sed -n '1,240p' docs/current_status.md`
- `sed -n '1,280p' docs/migration_plan.md`
- `sed -n '1,300p' docs/migration_inventory.md`
- `sed -n '/^## Next recommended batch/,$p' docs/migration_report.md`
- `git status --short`
- `python -m compileall src tests examples`
- `python -m pytest`
- `find examples -maxdepth 3 -type f | sort`
- `git status --short`
- `sed -n '1,420p' AGENTS.md`
- `find examples notebooks src/ssmtoolpy tests docs -maxdepth 3 -type f | sort`
- `git status --short`
- `sed -n '1,220p' examples/planar_system.py`
- `python -m compileall src tests examples`
- `python -m pytest`
- `mkdir -p examples/planar_system`
- `mkdir -p examples/benchmark_ssm_1st_order examples/lorenz_1st_order`
- `rg "examples/|notebooks/|planar_system.py|benchmark_ssm_1st_order.py|lorenz_1st_order.py|planar_system.ipynb|benchmark_ssm_1st_order.ipynb|lorenz_1st_order.ipynb" README.md docs tests src AGENTS.md`
- `mv examples/planar_system.py examples/planar_system/example.py`
- `mv examples/benchmark_ssm_1st_order.py examples/benchmark_ssm_1st_order/example.py`
- `mv examples/lorenz_1st_order.py examples/lorenz_1st_order/example.py`
- `mv notebooks/planar_system.ipynb examples/planar_system/planar_system.ipynb`
- `mv notebooks/benchmark_ssm_1st_order.ipynb examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb`
- `mv notebooks/lorenz_1st_order.ipynb examples/lorenz_1st_order/lorenz_1st_order.ipynb`
- `rmdir notebooks`
- `rg "notebooks/|examples/[a-z_]+\\.py|examples/<target_example>\\.py|examples/planar_system.py|examples/benchmark_ssm_1st_order.py|examples/lorenz_1st_order.py" AGENTS.md README.md docs tests src examples`
- `python examples/planar_system/example.py`
- `python examples/benchmark_ssm_1st_order/example.py`
- `python examples/lorenz_1st_order/example.py`
- `python -m compileall src tests examples`
- `python -m pytest`
- `sed -n '1,380p' AGENTS.md`
- `sed -n '1,280p' docs/current_status.md`
- `sed -n '1,360p' docs/migration_plan.md`
- `sed -n '1,380p' docs/migration_inventory.md`
- `sed -n '/^## Next recommended batch/,$p' docs/migration_report.md`
- `git status --short`
- `python -m compileall src tests examples`
- `python -m pytest`
- `sed -n '1,220p' SSMTool/examples/Lorenz1stOrder/build_model.m`
- `sed -n '1,220p' SSMTool/examples/Lorenz1stOrder/lorenz.m`
- `unzip -p SSMTool/examples/Lorenz1stOrder/demo.mlx matlab/document.xml`
- `sed -n '1,160p' src/ssmtoolpy/systems/__init__.py`
- `python examples/lorenz_1st_order/example.py`
- `python -m compileall src tests examples`
- `python -m pytest tests/test_lorenz_1st_order.py`
- `python -m compileall src tests examples`
- `python -m pytest`
- `sed -n '1,220p' SSMTool/examples/BenchamrkSSM1stOrder/build_model.m`
- `sed -n '1,220p' SSMTool/examples/PlanarSystem/build_model.m`
- `unzip -p SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx matlab/document.xml`
- `unzip -p SSMTool/examples/PlanarSystem/demo.mlx matlab/document.xml`
- `python examples/benchmark_ssm_1st_order/example.py`
- `python -m compileall src tests examples`
- `python -m pytest`
- `sed -n '1,360p' AGENTS.md`
- `sed -n '1,260p' docs/current_status.md`
- `sed -n '1,320p' docs/migration_plan.md`
- `sed -n '1,340p' docs/migration_inventory.md`
- `sed -n '1,360p' docs/migration_report.md`
- `git status --short`
- `python -m compileall src tests examples`
- `python -m pytest`
- `python -m pytest tests/test_parameter_to_loss.py`
- `python -m compileall src tests examples`
- `python -m pytest`
- `find SSMTool -maxdepth 3 -type d`
- `rg --files SSMTool`
- `find docs src tests examples notebooks -maxdepth 3 -type f`
- `sed -n '1,220p' pyproject.toml`
- `find SSMTool/examples -type f \\( -name '*.m' -o -name '*.mlx' \\)`
- `find SSMTool/examples -maxdepth 2 -type f \\( -name '*.m' -o -name '*.mlx' \\)`
- `find SSMTool/src -type f -name '*.m'`
- `file SSMTool/examples/PlanarSystem/demo.mlx SSMTool/examples/Lorenz1stOrder/demo.mlx SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx`
- `sed -n '1,220p' SSMTool/examples/PlanarSystem/build_model.m`
- `sed -n '1,240p' SSMTool/examples/Lorenz1stOrder/build_model.m`
- `sed -n '1,220p' SSMTool/examples/Lorenz1stOrder/lorenz.m`
- `sed -n '1,220p' SSMTool/examples/BenchamrkSSM1stOrder/build_model.m`
- `sed -n '1,220p' SSMTool/examples/TwoOscillators/build_model.m`
- `unzip -l SSMTool/examples/PlanarSystem/demo.mlx`
- `unzip -l SSMTool/examples/Lorenz1stOrder/demo.mlx`
- `unzip -p SSMTool/examples/PlanarSystem/demo.mlx matlab/document.xml`
- `unzip -p SSMTool/examples/Lorenz1stOrder/demo.mlx matlab/document.xml`
- `python -c "import jax; print(jax.__version__)"`
- `python --version`
- `mkdir -p docs src/ssmtoolpy/core src/ssmtoolpy/systems tests examples notebooks`
- `python -m compileall src tests examples`
- `python -m pytest`
- `python examples/planar_system/example.py`
- `git status --short`
- `find docs src tests examples notebooks -maxdepth 3 -type f`
- `sed -n '1,260p' AGENTS.md`
- `sed -n '1,220p' docs/current_status.md`
- `sed -n '1,260p' docs/migration_plan.md`
- `sed -n '1,260p' docs/migration_inventory.md`
- `sed -n '/^## Next recommended batch/,$p' docs/migration_report.md`
- `git status --short`
- `python -m compileall src tests examples`
- `python -m pytest`
- `sed -n '1,260p' SSMTool/src/@Manifold/private/Aut_1stOrder_SSM.m`
- `sed -n '1,260p' SSMTool/src/@Manifold/private/Aut_1stOrder_RedDyn.m`
- `sed -n '1,260p' SSMTool/src/@Manifold/private/coeffs_setup.m`
- `sed -n '1,220p' SSMTool/src/@Manifold/private/multi_nsumk.m`
- `sed -n '1,240p' src/ssmtoolpy/systems/planar.py`
- `sed -n '1,240p' tests/test_planar_system.py`
- `sed -n '1,200p' examples/planar_system/example.py`
- `sed -n '1,120p' src/ssmtoolpy/__init__.py`
- `python -m compileall src tests examples`
- `python -m pytest`
- `python examples/planar_system/example.py`
- `sed -n '1,260p' docs/migration_report.md`
- `sed -n '1,220p' README.md`
- `git status --short`

## Failures and debugging notes

- Initial `find docs src tests examples notebooks -maxdepth 3 -type f` failed
  because this clean restart had not created those directories yet.
- First `python -m pytest` run failed because JAX defaulted to float32 and the
  invariance test evaluated the derivative polynomial with unshifted powers.
  Fixed by enabling JAX x64 for this numerical module and correcting the
  derivative evaluation in the test.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 6 tests.
- `python examples/planar_system/example.py` passed and printed
  `[0.34494897, 0.52659863, 1.11237244, -9.89897949]` for `a2..a5`.
- Baseline for this batch passed before edits:
  `python -m compileall src tests examples` and `python -m pytest` with 6 tests.
- First post-implementation `python -m pytest` run failed because the test
  expected gradient for `3*x**2 - 2*x*y` was hand-computed as `[4, -1]`; the
  correct value at `(0.5, -0.25)` is `[3.5, -1]`. The test expectation was
  corrected.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 14 tests.
- `python examples/planar_system/example.py` passed and reported zero maximum
  difference from the `demo.mlx` coefficient formula.
- Baseline for the Benchmark batch passed before edits:
  `python -m compileall src tests examples` and `python -m pytest` with 14 tests.
- `BenchamrkSSM1stOrder/build_model.m` was confirmed source-equivalent to
  `PlanarSystem/build_model.m` after whitespace normalization.
- `python examples/benchmark_ssm_1st_order/example.py` passed and reported zero
  difference from the analytical coefficients.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 18 tests.
- Retrospective adaptation baseline passed before edits:
  `python -m compileall src tests examples` and `python -m pytest` with 18 tests.
- Added `tests/test_parameter_to_loss.py`, a minimal fixed-structure
  PlanarSystem gradient smoke test for
  `decay -> graph coefficients -> graph prediction -> scalar loss`.
- `python -m pytest tests/test_parameter_to_loss.py` passed: 1 test.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 19 tests.
- Baseline for the Lorenz batch passed before edits:
  `python -m compileall src tests examples` and `python -m pytest` with 19 tests.
- `python examples/lorenz_1st_order/example.py` passed and printed vector field
  `[10, 23, -6]` at `[1, 2, 3]` with sorted eigenvalues
  `[-22.82772345, -2.66666667, 11.82772345]`.
- `python -m pytest tests/test_lorenz_1st_order.py` passed: 6 tests.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 25 tests.
- Final example tree inspection confirmed the colocated directories contain
  `README.md`, `example.py`, and their notebooks.
- Baseline for the example layout batch passed before edits:
  `python -m compileall src tests examples` and `python -m pytest` with 25 tests.
- Reorganized all currently reproduced examples into colocated directories under
  `examples/<example_name>/` and removed the empty top-level `notebooks/`
  directory.
- `python examples/planar_system/example.py` passed and reported zero maximum
  difference from the `demo.mlx` coefficient formula.
- `python examples/benchmark_ssm_1st_order/example.py` passed and reported zero
  difference from the analytical coefficients.
- `python examples/lorenz_1st_order/example.py` passed and printed vector field
  `[10, 23, -6]` at `[1, 2, 3]` with sorted eigenvalues
  `[-22.82772345, -2.66666667, 11.82772345]`.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 25 tests.

## Next recommended batch

### Target

- Extend `Lorenz1stOrder/demo.mlx` with the smallest fixed-choice unstable SSM
  graph coefficient subproblem: select the positive real eigenvalue/eigenvector
  externally, solve nonresonant scalar graph coefficients through order 2 or 3,
  and verify the invariance residual on deterministic small amplitudes.

### MATLAB files involved

- `SSMTool/examples/Lorenz1stOrder/build_model.m`
- `SSMTool/examples/Lorenz1stOrder/lorenz.m`
- `SSMTool/examples/Lorenz1stOrder/demo.mlx`
- `SSMTool/src/@Manifold/private/Aut_1stOrder_SSM.m`
- `SSMTool/src/@Manifold/private/Aut_1stOrder_RedDyn.m`
- `SSMTool/src/@Manifold/private/coeffs_setup.m`

### MATLAB examples or `.mlx` workflows involved

- `SSMTool/examples/Lorenz1stOrder/demo.mlx`

### Planned Python modules

- `src/ssmtoolpy/systems/lorenz.py`
- Possibly `src/ssmtoolpy/core/invariance.py` for a small fixed-choice vector
  homological solve if the current scalar graph helper is insufficient.
- Do not implement adaptive mode selection, resonance classification, or full
  `compute_whisker`.

### Planned examples or notebooks

- Update `examples/lorenz_1st_order/example.py` to print the fixed-choice graph
  coefficient residual or coefficient summary.
- Update `examples/lorenz_1st_order/lorenz_1st_order.ipynb` only if the tested numerical core
  remains concise.

### Expected tests

- Fixed eigenpair selection picks the positive eigenvalue near `11.8277`.
- Computed graph coefficients satisfy the autonomous invariance equation
  through the chosen order on small deterministic amplitudes.
- A finite scalar fixed-choice Lorenz graph loss has a finite gradient with
  respect to one continuous parameter if the solve is kept differentiable.
- Existing Lorenz vector-field/eigenvalue, PlanarSystem, Benchmark, and core
  tests remain passing.

### Known risks

- The full Lorenz `.mlx` workflow computes a 1D unstable SSM and compares
  trajectories; that is larger than the first bounded target.
- Eigenvector normalization/sign conventions can change discontinuously; freeze
  deterministic normalization outside any differentiable loss.
- Inner-resonance logic is deferred; this batch should document that the chosen
  standard-parameter case is nonresonant for the tested orders.

### Differentiability concerns

- Differentiability should be through fixed choices: selected eigenpair,
  truncation order, normalization convention, and nonresonant solve structure.
- Do not claim differentiability through eigenvalue sorting or adaptive mode
  selection.
- Homological linear systems are differentiable only while nonsingular and
  reasonably conditioned.

### Acceptance criteria

- A minimal fixed-choice Lorenz graph coefficient computation is implemented
  and tested without adding broad MATLAB compatibility wrappers.
- Invariance residual tests pass for the chosen order.
- Any differentiable public function added has a JAX transform or
  parameter-to-loss-style test.
- `python -m compileall src tests examples` and `python -m pytest` pass.

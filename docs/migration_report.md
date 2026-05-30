# Migration Report

## Implemented modules

- `src/ssmtoolpy/core/multiindex.py`
  - `multiindices_of_total_degree`
- `src/ssmtoolpy/core/polynomial.py`
  - `evaluate_monomial_polynomial`
  - `collect_univariate_coefficients`
- `src/ssmtoolpy/core/invariance.py`
  - `solve_scalar_graph_coefficients`
- `examples/planar_system/planar.py`
  - `build_planar_system`
  - `planar_vector_field`
  - `planar_nonlinear_exponents`
  - `planar_nonlinear_coefficients`
  - `planar_ssm_graph_coefficients`
  - `evaluate_planar_ssm_graph`
- `examples/benchmark_ssm_1st_order/benchmark.py`
  - `benchmark_ssm_graph_coefficients`
- `examples/lorenz_1st_order/lorenz.py`
  - `standard_lorenz_parameters`
  - `build_lorenz_system`
  - `lorenz_nonlinear_exponents`
  - `lorenz_nonlinear_coefficients`
  - `lorenz_vector_field`
  - `lorenz_linear_eigenvalues`
  - `lorenz_unstable_eigenpair`
  - `solve_lorenz_unstable_graph_coefficients`
  - `lorenz_unstable_ssm_graph_coefficients`
  - `evaluate_lorenz_ssm_graph`
  - `lorenz_ssm_invariance_residual`
  - `lorenz_rk4_trajectory`
  - `lorenz_reduced_trajectory`
  - `lorenz_reduced_to_full_trajectory`
  - `lorenz_unstable_ssm_curve`
  - `lorenz_full_unstable_trajectories`

## Reproduced examples

- `PlanarSystem` numerical subproblem from `SSMTool/examples/PlanarSystem/demo.mlx`:
  graph coefficients `a_2` through `a_5` for the first-mode SSM are now
  computed by the minimal scalar first-order graph solver and checked against
  the formula stated in the live script.
- `BenchamrkSSM1stOrder` from
  `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx`: reproduced as a
  source-confirmed duplicate of PlanarSystem, with a named Python example and
  tests covering the misspelled source path.
- `Lorenz1stOrder` fixed-choice Python/JAX workflow from
  `SSMTool/examples/Lorenz1stOrder/demo.mlx`: source model, vector field,
  standard-parameter linear eigenvalues, direct fixed-step trajectory
  computation, fixed-choice unstable SSM graph coefficients through order 5,
  invariance residual checks, linear reduced dynamics, reduced-to-full lifting,
  reduced/full trajectory comparison, 3D notebook visualization, and
  parameter-to-output/fixed-lifted-trajectory gradient smoke tests.

## Notebook migration status

- `examples/planar_system/planar_system.ipynb`: partial workflow migration;
  coefficient core is tested, full MATLAB class workflow equivalence remains
  limited to the reproduced subproblem.
- `examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb`: partial
  workflow migration; source-confirmed duplicate coefficient comparison is
  tested.
- `examples/lorenz_1st_order/lorenz_1st_order.ipynb`: complete for the tested
  fixed-choice Python/JAX Lorenz workflow. It covers setup, eigenvalues,
  fixed-choice SSM graph computation, reduced dynamics, reduced-to-full
  trajectory lifting, direct full trajectories, reduced/full comparison, and
  the MATLAB-style 3D SSM/full visualization. It executed successfully via
  `python -m jupyter nbconvert --to notebook --execute`.
- The same standard applies to all current and future examples: PlanarSystem
  and BenchamrkSSM1stOrder are substantive partial reproductions because they
  implement tested SSM graph coefficient workflows; Lorenz1stOrder is a
  substantive partial reproduction after the fixed-choice SSM graph solve, but
  remains incomplete as a full live-script migration.

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
- Example-specific model and workflow helpers now live beside their examples:
  - `examples/planar_system/planar.py`
  - `examples/benchmark_ssm_1st_order/benchmark.py`
  - `examples/lorenz_1st_order/lorenz.py`
- `src/ssmtoolpy/` now exposes only reusable core kernels and does not export
  PlanarSystem, Benchmark, or Lorenz example helpers.

## Skipped or deferred items

- Full MATLAB `DynamicalSystem`, `SSM`, and `Manifold` class behavior.
- General multi-dimensional autonomous coefficient solving.
- Resonant reduced-dynamics extraction from `Aut_1stOrder_RedDyn.m`.
- Adaptive/general Lorenz `SSM`, `DynamicalSystem`, and `Manifold` object
  workflow internals beyond the fixed-choice tested live-script path.
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
- Lorenz now covers a fixed-choice differentiable graph coefficient solve and
  lifted reduced prediction, but not gradients through adaptive eigenpair
  selection, resonance classification, or generic MATLAB object workflows.
- Setup-only notebooks are now explicitly classified as incomplete. A notebook
  is complete only when it reproduces the meaningful numerical and visual
  MATLAB `.mlx` workflow or documents a hard blocker.

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
- `sed -n '1,260p' SSMTool/src/misc/reduced_to_full_traj.m`
- `unzip -p SSMTool/examples/Lorenz1stOrder/demo.mlx matlab/document.xml`
- `sed -n '1,380p' examples/lorenz_1st_order/lorenz.py`
- `sed -n '1,280p' tests/test_lorenz_1st_order.py`
- `sed -n '1,220p' examples/lorenz_1st_order/example.py`
- `python -m pytest tests/test_lorenz_1st_order.py`
- `python examples/lorenz_1st_order/example.py`
- `python -m json.tool examples/lorenz_1st_order/lorenz_1st_order.ipynb`
- `python -m jupyter nbconvert --to notebook --execute examples/lorenz_1st_order/lorenz_1st_order.ipynb --output /tmp/lorenz_1st_order.executed.ipynb`
- `sed -n '1,180p' README.md`
- `sed -n '1,220p' examples/lorenz_1st_order/README.md`
- `sed -n '1,220p' docs/current_status.md`
- `sed -n '1,260p' docs/migration_report.md`
- `sed -n '1,240p' SSMTool/examples/TwoOscillators/build_model.m`
- `unzip -p SSMTool/examples/TwoOscillators/demo.mlx matlab/document.xml`
- `python -m compileall src tests examples`
- `python -m pytest`
- `sed -n '150,260p' AGENTS.md`
- `sed -n '1,260p' examples/lorenz_1st_order/lorenz_1st_order.ipynb`
- `sed -n '1,360p' tests/test_lorenz_1st_order.py`
- `git status --short`
- `python -m compileall src tests examples`
- `python -m pytest tests/test_lorenz_1st_order.py`
- `python examples/lorenz_1st_order/example.py`
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
- `find src/ssmtoolpy -maxdepth 3 -type f`
- `find examples -maxdepth 3 -type f`
- `rg "ssmtoolpy\\.systems|systems\\.planar|systems\\.lorenz|from ssmtoolpy import|import ssmtoolpy" -n src tests examples docs README.md`
- `python -m compileall src tests examples`
- `python -m pytest`
- `sed -n '1,260p' src/ssmtoolpy/__init__.py`
- `sed -n '1,260p' src/ssmtoolpy/systems/__init__.py`
- `sed -n '1,320p' src/ssmtoolpy/systems/planar.py`
- `sed -n '1,360p' src/ssmtoolpy/systems/lorenz.py`
- `sed -n '1,220p' pyproject.toml`
- `sed -n '1,220p' src/ssmtoolpy/core/__init__.py`
- `sed -n '1,220p' tests/test_planar_system.py`
- `sed -n '1,180p' tests/test_benchmark_ssm_1st_order.py`
- `sed -n '1,180p' tests/test_parameter_to_loss.py`
- `sed -n '1,180p' examples/planar_system/example.py`
- `sed -n '1,160p' examples/benchmark_ssm_1st_order/example.py`
- `rg "ssmtoolpy\\.systems|from ssmtoolpy import|import ssmtoolpy" examples tests src docs README.md`
- `rg "planar_ssm_graph_coefficients|benchmark_ssm_graph_coefficients|ssmtoolpy\\.systems" examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb examples/planar_system/planar_system.ipynb examples/lorenz_1st_order/lorenz_1st_order.ipynb`
- `rg "ssmtoolpy\\.systems|systems/planar|systems/lorenz|src/ssmtoolpy/systems|systems\\.planar|systems\\.lorenz" -n src tests examples docs README.md AGENTS.md`
- `sed -n '1,130p' docs/migration_inventory.md`
- `sed -n '1,230p' docs/migration_plan.md`
- `sed -n '1,120p' docs/current_status.md`
- `sed -n '1,130p' examples/planar_system/README.md`
- `sed -n '1,130p' examples/benchmark_ssm_1st_order/README.md`
- `python examples/planar_system/example.py`
- `python examples/benchmark_ssm_1st_order/example.py`
- `python examples/lorenz_1st_order/example.py`
- `python -m compileall src tests examples`
- `python -m pytest`
- `find src/ssmtoolpy/systems -type f -name '*.py'`
- `python -c "import ssmtoolpy; print(ssmtoolpy.__all__)"`
- `git status --short`
- `rg "src/ssmtoolpy/systems|ssmtoolpy\\.systems" -n README.md AGENTS.md docs/current_status.md docs/migration_plan.md docs/migration_inventory.md examples tests src`
- `find src/ssmtoolpy -maxdepth 3 -type f | sort`
- `find src/ssmtoolpy -path '*systems*' -type f | sort`
- `rg "ssmtoolpy\\.systems|src/ssmtoolpy/systems|systems/planar|systems/lorenz" -n README.md AGENTS.md docs examples tests src`
- `python -m compileall src tests examples`
- `python -m pytest`
- `sed -n '/^## Next recommended batch/,$p' docs/migration_report.md`
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
- `git status --short`
- `sed -n '1,240p' docs/current_status.md`
- `sed -n '1,260p' docs/migration_plan.md`
- `sed -n '1,320p' docs/migration_inventory.md`
- `sed -n '1,320p' docs/migration_report.md`
- `sed -n '320,460p' docs/migration_report.md`
- `sed -n '1,440p' AGENTS.md`
- `sed -n '1,260p' README.md`
- `sed -n '1,260p' src/ssmtoolpy/systems/lorenz.py`
- `sed -n '1,220p' tests/test_lorenz_1st_order.py`
- `sed -n '220,340p' tests/test_lorenz_1st_order.py`
- `sed -n '260,420p' src/ssmtoolpy/systems/lorenz.py`
- `python examples/planar_system/example.py`
- `python examples/benchmark_ssm_1st_order/example.py`
- `python examples/lorenz_1st_order/example.py`
- `python -m compileall src tests examples`
- `python -m pytest`

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
- Implemented the next substantive `Lorenz1stOrder` workflow step: a
  fixed-choice unstable SSM graph coefficient solve through order 3, graph
  evaluation, and invariance residual checks. This moves Lorenz beyond setup
  and smoke trajectory work, while still leaving reduced-to-full trajectory
  comparison and visualization incomplete.
- Updated AGENTS and migration docs so the substantive-workflow rule applies to
  all examples, demos, and `.mlx` workflows, not only Lorenz.
- `python -m pytest tests/test_lorenz_1st_order.py` passed: 13 tests.
- `python examples/planar_system/example.py` passed and reported zero maximum
  difference from the `demo.mlx` coefficient formula.
- `python examples/benchmark_ssm_1st_order/example.py` passed and reported zero
  difference from the analytical coefficients.
- `python examples/lorenz_1st_order/example.py` passed and printed SSM graph
  coefficients through order 3, maximum invariance residual
  `5.421010862427522e-20`, and the small-amplitude RK4 final state.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 32 tests.
- Revalidated the layout on the repeated layout request:
  `src/ssmtoolpy/systems/` contained no `.py` source modules, only generated
  `__pycache__` files. Removed that generated leftover directory entirely. The
  package source imports no example-local code.
- Verified that searching under `src/ssmtoolpy` for systems paths returns no
  files or directories.
- Revalidation `python -m compileall src tests examples` passed.
- Revalidation `python -m pytest` passed: 32 tests.
- Baseline before completing Lorenz passed:
  `python -m compileall src tests examples` and `python -m pytest` with 32
  tests.
- Added reduced dynamics, reduced-to-full lifting, two-sided SSM curve, and
  full unstable trajectory helpers for `Lorenz1stOrder`.
- `python -m pytest tests/test_lorenz_1st_order.py` passed: 19 tests.
- First notebook execution attempt failed because Jupyter could start only after
  escalated local-kernel permissions and the notebook did not add `src/` to
  `sys.path`. The notebook bootstrap was fixed to add both the example
  directory and project `src/`.
- `python -m jupyter nbconvert --to notebook --execute
  examples/lorenz_1st_order/lorenz_1st_order.ipynb --output
  /tmp/lorenz_1st_order.executed.ipynb` passed after the bootstrap fix.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 38 tests.
- Layout restructuring baseline passed before edits:
  `python -m compileall src tests examples` and `python -m pytest` with 32
  tests.
- Moved example-specific model/helper source out of `src/ssmtoolpy/systems/`
  into colocated example directories:
  `examples/planar_system/planar.py`,
  `examples/benchmark_ssm_1st_order/benchmark.py`, and
  `examples/lorenz_1st_order/lorenz.py`.
- Removed the `ssmtoolpy.systems` package source exports; `ssmtoolpy` now
  imports only reusable core kernels and does not import from `examples/`.
- Tests now use explicit path handling for example-local helpers.
- `find src/ssmtoolpy/systems -type f -name '*.py'` produced no output,
  confirming no source modules remain in the systems folder.
- `python examples/planar_system/example.py` passed and reported zero maximum
  difference from the `demo.mlx` coefficient formula.
- `python examples/benchmark_ssm_1st_order/example.py` passed and reported zero
  difference from the analytical coefficients.
- `python examples/lorenz_1st_order/example.py` passed and printed SSM graph
  coefficients through order 3, maximum invariance residual
  `5.421010862427522e-20`, and the small-amplitude RK4 final state.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 32 tests.
- Final example tree inspection confirmed the colocated directories contain
  `README.md`, `example.py`, and their notebooks.
- Corrected the notebook migration standard globally: `.mlx` notebooks are not
  complete unless they reproduce the meaningful numerical and visual workflow,
  including SSM graph/reduced dynamics/trajectory/visualization where present.
- Added `lorenz_rk4_trajectory` as the next small missing Lorenz step toward
  the trajectory-computation portion of `Lorenz1stOrder/demo.mlx`.
- `python -m pytest tests/test_lorenz_1st_order.py` passed: 9 tests.
- `python examples/lorenz_1st_order/example.py` passed and printed a
  small-amplitude RK4 final state.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 28 tests.
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

- Start the next substantive example workflow: `TwoOscillators/demo.mlx`.
  Implement the model setup and the smallest tested forced-response numerical
  subproblem that moves beyond setup, preferably the second-order linear modal
  analysis plus deterministic forced linear response data needed before the
  later SSM continuation/FRC workflow.

### MATLAB files involved

- `SSMTool/examples/TwoOscillators/build_model.m`
- `SSMTool/examples/TwoOscillators/demo.mlx`
- MATLAB second-order system setup and linear spectral-analysis code needed to
  derive the first deterministic modal reference.

### MATLAB examples or `.mlx` workflows involved

- `SSMTool/examples/TwoOscillators/demo.mlx`

### Planned Python modules

- `examples/two_oscillators/two_oscillators.py`
- Reuse `src/ssmtoolpy/core/` only for genuinely shared numerical kernels.
- Add a reusable core helper only if the same operation is immediately tested
  outside the TwoOscillators example.

### Planned examples or notebooks

- Create `examples/two_oscillators/README.md`.
- Create `examples/two_oscillators/example.py`.
- Create `examples/two_oscillators/two_oscillators.ipynb` only if the batch
  implements a meaningful executable workflow section beyond setup.

### Expected tests

- Model matrices from `build_model.m` match source-derived references for the
  default parameters in `demo.mlx`.
- Linearized second-order eigenvalues/frequencies match deterministic
  source-derived references.
- The first non-setup forced-response or modal-response numerical subproblem
  has shape/correctness tests.
- JAX transform tests cover any public differentiable helper introduced.
- Existing PlanarSystem, BenchamrkSSM1stOrder, Lorenz, and core tests remain
  passing.

### Known risks

- The full `TwoOscillators/demo.mlx` workflow is a continuation/FRC example with
  SSM continuation methods; do not attempt the whole workflow in one jump.
- Setup-only model/eigenvalue work is not a valid endpoint unless a precise
  blocker is reached. The batch must include a substantive response,
  continuation, or SSM-related numerical step.
- Internal resonance and forcing conventions must be documented before any
  reduced dynamics claim.

### Differentiability concerns

- Differentiate only through fixed continuous parameters and fixed modal/forcing
  choices.
- Do not claim differentiability through mode sorting, resonance
  classification, continuation branch selection, bifurcation detection, or
  plotting.
- Linear solves/eigenvalue calculations are differentiable only under
  nondegeneracy assumptions.

### Acceptance criteria

- A `TwoOscillators` example-local helper exists only under
  `examples/two_oscillators/`.
- At least one substantive `TwoOscillators` numerical workflow step beyond
  setup/eigenvalues is implemented and tested.
- A Python example prints deterministic numerical outputs from the tested core.
- A notebook is added only if it can present tested numerical workflow content,
  not placeholder setup.
- All current examples remain classified against the substantive workflow
  standard; no setup-only or smoke-only example is marked complete.
- Any differentiable public function added has a JAX transform or
  parameter-to-loss-style test.
- `python -m compileall src tests examples` and `python -m pytest` pass.

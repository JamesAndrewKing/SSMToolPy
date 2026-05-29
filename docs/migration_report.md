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

## Reproduced examples

- `PlanarSystem` numerical subproblem from `SSMTool/examples/PlanarSystem/demo.mlx`:
  graph coefficients `a_2` through `a_5` for the first-mode SSM are now
  computed by the minimal scalar first-order graph solver and checked against
  the formula stated in the live script.

## Reproduced notebooks

- `notebooks/planar_system.ipynb` calls the tested PlanarSystem numerical API.

## Skipped or deferred items

- Full MATLAB `DynamicalSystem`, `SSM`, and `Manifold` class behavior.
- General multi-dimensional autonomous coefficient solving.
- Resonant reduced-dynamics extraction from `Aut_1stOrder_RedDyn.m`.
- Continuation, FRC/FRS, plotting, and external finite-element workflows.
- Notebook execution checks; no notebook execution tooling was configured before
  this batch.

## Known limitations

- This batch reproduces a one-master, one-transverse nonresonant scalar graph
  subproblem, not the full `compute_whisker` MATLAB implementation.
- The first target has no MATLAB-generated fixture file; reference values are
  source-derived from `demo.mlx`, `build_model.m`, and the inspected
  homological solve in `Aut_1stOrder_SSM.m`.

## Exact commands run

- `pwd`
- `ls`
- `sed -n '1,260p' AGENTS.md`
- `git status --short`
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
- `python examples/planar_system.py`
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
- `sed -n '1,200p' examples/planar_system.py`
- `sed -n '1,120p' src/ssmtoolpy/__init__.py`
- `python -m compileall src tests examples`
- `python -m pytest`
- `python examples/planar_system.py`
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
- `python examples/planar_system.py` passed and printed
  `[0.34494897, 0.52659863, 1.11237244, -9.89897949]` for `a2..a5`.
- Baseline for this batch passed before edits:
  `python -m compileall src tests examples` and `python -m pytest` with 6 tests.
- First post-implementation `python -m pytest` run failed because the test
  expected gradient for `3*x**2 - 2*x*y` was hand-computed as `[4, -1]`; the
  correct value at `(0.5, -0.25)` is `[3.5, -1]`. The test expectation was
  corrected.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 14 tests.
- `python examples/planar_system.py` passed and reported zero maximum
  difference from the `demo.mlx` coefficient formula.

## Next recommended batch

### Target

- Reproduce the duplicate `BenchamrkSSM1stOrder/demo.mlx` workflow as a named
  regression target by confirming it is mathematically identical to
  PlanarSystem, adding a Python example or test alias only where useful, and
  documenting it as reproduced without adding new numerical abstractions.

### MATLAB files involved

- `SSMTool/examples/BenchamrkSSM1stOrder/build_model.m`
- `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx`
- `SSMTool/examples/PlanarSystem/build_model.m`
- `SSMTool/examples/PlanarSystem/demo.mlx`

### MATLAB examples or `.mlx` workflows involved

- `SSMTool/examples/PlanarSystem/demo.mlx`
- `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx`

### Planned Python modules

- Prefer no new core modules.
- Possibly add a small example-specific alias or metadata in
  `src/ssmtoolpy/systems/planar.py` only if tests or example clarity require it.

### Planned examples or notebooks

- Add `examples/benchmark_ssm_1st_order.py` only if it adds a clear reproduced
  example entry without duplicating logic.
- Add `notebooks/benchmark_ssm_1st_order.ipynb` only if the `.mlx` source has
  presentation content distinct from PlanarSystem.

### Expected tests

- Source comparison test showing the Benchmark and PlanarSystem MATLAB
  `build_model.m` files define the same `A`, `B`, and nonlinear terms.
- Regression test showing the Benchmark target obtains the same solver-derived
  coefficients as PlanarSystem.
- Existing PlanarSystem and core tests remain passing.

### Known risks

- The directory name is misspelled as `BenchamrkSSM1stOrder`; preserve the
  source path spelling in docs and tests.
- The `.mlx` workflow may contain little or no content beyond the duplicate
  model; avoid creating duplicate user-facing files if documentation plus tests
  are cleaner.

### Differentiability concerns

- No new differentiable numerical function is expected.
- If a new public wrapper is added, it should inherit the existing
  PlanarSystem differentiability classifications and be covered by a JAX
  transform test if marked differentiable.

### Acceptance criteria

- `BenchamrkSSM1stOrder` is recorded as a reproduced workflow or explicitly
  documented as a duplicate of PlanarSystem with source-derived proof.
- At least one regression test covers the Benchmark workflow name/path.
- No broad APIs or duplicated numerical implementation are introduced.
- `python -m compileall src tests examples` and `python -m pytest` pass.

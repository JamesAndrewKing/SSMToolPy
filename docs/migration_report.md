# Migration Report

## Implemented modules

- `src/ssmtoolpy/systems/planar.py`
  - `build_planar_system`
  - `planar_vector_field`
  - `planar_ssm_graph_coefficients`
  - `evaluate_planar_ssm_graph`

## Reproduced examples

- `PlanarSystem` numerical subproblem from `SSMTool/examples/PlanarSystem/demo.mlx`:
  graph coefficients `a_2` through `a_5` for the first-mode SSM.

## Reproduced notebooks

- `notebooks/planar_system.ipynb` calls the tested PlanarSystem numerical API.

## Skipped or deferred items

- Full MATLAB `DynamicalSystem`, `SSM`, and `Manifold` class behavior.
- General autonomous coefficient solving.
- Continuation, FRC/FRS, plotting, and external finite-element workflows.
- Notebook execution checks; no notebook execution tooling was configured before
  this batch.

## Known limitations

- This batch reproduces a stated closed-form subproblem, not the full
  `compute_whisker` MATLAB implementation.
- The first target has no MATLAB-generated fixture file; reference values are
  source-derived from `demo.mlx` and `build_model.m`.

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

## Next recommended batch

### Target

- Extend from the PlanarSystem closed-form coefficient check to the smallest
  general first-order autonomous graph coefficient solver needed to reproduce
  PlanarSystem through order 5 without hard-coding the answer.

### MATLAB files involved

- `SSMTool/examples/PlanarSystem/build_model.m`
- `SSMTool/examples/PlanarSystem/demo.mlx`
- `SSMTool/src/@Manifold/private/Aut_1stOrder_SSM.m`
- `SSMTool/src/@Manifold/private/Aut_1stOrder_RedDyn.m`
- `SSMTool/src/@Manifold/private/coeffs_setup.m`
- `SSMTool/src/@Manifold/private/multi_nsumk.m`

### MATLAB examples or `.mlx` workflows involved

- `SSMTool/examples/PlanarSystem/demo.mlx`
- Optionally compare `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx` if it is
  confirmed to be the same model.

### Planned Python modules

- `src/ssmtoolpy/core/multiindex.py`
- `src/ssmtoolpy/core/polynomial.py`
- Possibly `src/ssmtoolpy/core/invariance.py`

### Planned examples or notebooks

- Update `examples/planar_system.py` to compute coefficients via the general
  solver and compare to the closed-form formula.
- Keep `notebooks/planar_system.ipynb` focused on the tested API.

### Expected tests

- Multi-index generation for one-dimensional and two-dimensional small orders.
- Polynomial tensor evaluation against deterministic source-derived cases.
- PlanarSystem coefficient solve through order 5.
- JAX transform tests for differentiable polynomial evaluation.

### Known risks

- MATLAB tensor normalization and ordering conventions must be verified before
  broadening beyond the one-dimensional graph case.
- General coefficient solving may introduce linear solves with resonance or
  near-resonance denominators.

### Differentiability concerns

- Coefficient solves are differentiable only under nonresonance and nonsingular
  homological-equation assumptions.
- Multi-index enumeration and truncation are not differentiable with respect to
  order or index choices.

### Acceptance criteria

- PlanarSystem coefficients are produced by a minimal general solver, not only
  by the closed-form helper.
- The existing PlanarSystem tests still pass.
- New tests cover the smallest polynomial/multi-index functionality used by
  that solver.

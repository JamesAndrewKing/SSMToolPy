# Migration Report

## Implemented modules

- `src/ssmtoolpy/core/multiindex.py`
  - `multiindices_of_total_degree`
- `src/ssmtoolpy/core/polynomial.py`
  - `evaluate_monomial_polynomial`
  - `collect_univariate_coefficients`
- `src/ssmtoolpy/core/invariance.py`
  - `solve_scalar_graph_coefficients`
  - `solve_autonomous_quadratic_graph_coefficients`
  - `univariate_graph_invariance_residual`
- `src/ssmtoolpy/core/integrators.py`
  - `fixed_step_rk4`
- `src/ssmtoolpy/core/graph.py`
  - `evaluate_univariate_graph`
  - `linear_reduced_trajectory`
  - `two_sided_graph_curve`
- `src/ssmtoolpy/core/trajectories.py`
  - `integrate_two_sided_branches`
- `examples/planar_system/planar.py`
- `examples/benchmark_ssm_1st_order/benchmark.py`
- `examples/lorenz_1st_order/lorenz.py`

## Reproduced examples

- `PlanarSystem`: coefficient subproblem from
  `SSMTool/examples/PlanarSystem/demo.mlx`, including `a_2..a_5`.
- `BenchamrkSSM1stOrder`: source-confirmed duplicate coefficient comparison
  from `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx`.
- `Lorenz1stOrder`: fixed-choice workflow from
  `SSMTool/examples/Lorenz1stOrder/demo.mlx`, including source model,
  eigenvalues, unstable graph coefficients through order 5, reduced dynamics,
  reduced-to-full lifting, full trajectory comparison, and 3D notebook plot.

## Fidelity audit results

| Example | Fidelity status | MATLAB source inspected | Corrections made | Largest remaining gap |
| --- | --- | --- | --- | --- |
| PlanarSystem | `partial` | `build_model.m`, `demo.mlx` | Added README fidelity checklist; confirmed MATLAB has no figure | Full `DynamicalSystem`/`SSM`/`compute_whisker` object workflow |
| BenchamrkSSM1stOrder | `partial` | `build_model.m`, `demo.mlx` | Added README fidelity checklist; confirmed MATLAB has no figure and duplicates PlanarSystem model | Full `DynamicalSystem`/`SSM`/`compute_whisker` object workflow and exact `W0/R0` layout |
| Lorenz1stOrder | `plot-incomplete` | `build_model.m`, `lorenz.m`, `demo.mlx` | Added README fidelity checklist; changed visualization grid from 501 to MATLAB's 100 samples; added grid to 3D plot | Exact adaptive `ode45` trajectory sampling and full adaptive MATLAB `compute_whisker` workflow |

## Notebook migration status

- `examples/planar_system/planar_system.ipynb`: partial; coefficient core is
  tested and MATLAB has no plot.
- `examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb`: partial;
  duplicate coefficient comparison is tested and MATLAB has no plot.
- `examples/lorenz_1st_order/lorenz_1st_order.ipynb`: `plot-incomplete`; the
  SSM/full 3D visualization is present with matching variables, labels, legend,
  view, branches, and MATLAB time range/grid, but exact `ode45` adaptive
  sampling and MATLAB styling are not reproduced.

## Skipped or deferred items

- Full MATLAB `DynamicalSystem`, `SSM`, and `Manifold` class behavior.
- Exact `W0/R0` object layout from MATLAB `compute_whisker`.
- General multi-dimensional autonomous coefficient solving.
- Adaptive MATLAB `ode45` trajectory sampling for Lorenz validation.
- Continuation, FRC/FRS, forcing, finite-element, and external-model workflows.

## Known limitations

- Current examples are fixed-choice reproductions of bounded workflow slices,
  not full MATLAB class-stack ports.
- The PlanarSystem and Benchmark examples have no MATLAB plots; plot fidelity is
  not applicable there.
- Lorenz uses fixed-step RK4 for differentiable trajectory tests. This is
  useful for JAX but is not identical to MATLAB's adaptive `ode45` output.
- No current test differentiates through adaptive mode selection, resonance
  classification, or a full MATLAB-faithful SSM construction.

## Exact commands run

- `sed -n '1,460p' AGENTS.md`
- `find examples -maxdepth 2 -type f | sort`
- `git status --short`
- `sed -n '1,260p' docs/current_status.md`
- `sed -n '1,320p' docs/migration_plan.md`
- `sed -n '1,260p' docs/migration_inventory.md`
- `sed -n '/^## Next recommended batch/,$p' docs/migration_report.md`
- `python -m compileall src tests examples`
- `python -m pytest`
- `unzip -p SSMTool/examples/PlanarSystem/demo.mlx matlab/document.xml`
- `unzip -p SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx matlab/document.xml`
- `unzip -p SSMTool/examples/Lorenz1stOrder/demo.mlx matlab/document.xml`
- `python examples/planar_system/example.py`
- `python examples/benchmark_ssm_1st_order/example.py`
- `python examples/lorenz_1st_order/example.py`
- `python -m jupyter nbconvert --to notebook --execute examples/planar_system/planar_system.ipynb --output /tmp/planar_system.executed.ipynb`
- `python -m jupyter nbconvert --to notebook --execute examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb --output /tmp/benchmark_ssm_1st_order.executed.ipynb`
- `python -m jupyter nbconvert --to notebook --execute examples/lorenz_1st_order/lorenz_1st_order.ipynb --output /tmp/lorenz_1st_order.executed.ipynb`

## Failures and debugging notes

- None in the required checks.
- Notebook execution emitted non-fatal nbformat warnings about missing cell IDs.
- Final `python -m compileall src tests examples` passed.
- Final `python -m pytest` passed: 48 tests.
- All migrated example scripts passed explicitly.
- All migrated notebooks executed with `nbconvert` and wrote outputs under
  `/tmp/`.
- The fidelity audit intentionally reclassified Lorenz from previously described
  fixed-choice complete status to `plot-incomplete`, because exact MATLAB
  `ode45` sampling and the full adaptive object workflow are not reproduced.
- The highest-priority safe correction made in this batch was the Lorenz plot
  time grid: the notebook and example now use `t = linspace(0,1,100)` for the
  visualization data, matching the MATLAB live script.

## Next recommended batch

### Target

- Close the largest remaining fidelity gap among already migrated examples by
  improving `Lorenz1stOrder` plot/numerical fidelity before starting any new
  MATLAB example. Focus on the validation trajectory overlay in
  `SSMTool/examples/Lorenz1stOrder/demo.mlx`.

### MATLAB files involved

- `SSMTool/examples/Lorenz1stOrder/demo.mlx`
- `SSMTool/examples/Lorenz1stOrder/lorenz.m`
- `SSMTool/examples/Lorenz1stOrder/build_model.m`
- MATLAB trajectory helper behavior from `ode45` and the plotted SSM/full
  validation figure.

### MATLAB examples or `.mlx` workflows involved

- `SSMTool/examples/Lorenz1stOrder/demo.mlx`

### Planned Python modules

- `examples/lorenz_1st_order/lorenz.py` only for Lorenz-specific assembly.
- Reuse existing `src/ssmtoolpy/core/graph.py`, `integrators.py`,
  `trajectories.py`, and `invariance.py` kernels.
- Add a reusable non-differentiable reference integration helper only if needed
  and tested; keep SciPy or adaptive solver use outside differentiable core.

### Planned examples or notebooks

- Update `examples/lorenz_1st_order/example.py` only if it needs to report the
  new fidelity comparison.
- Update `examples/lorenz_1st_order/lorenz_1st_order.ipynb` to make the
  MATLAB-style validation plot closer in sampling, axes, legend, and visual
  grouping.
- Keep `examples/lorenz_1st_order/README.md` fidelity checklist current.

### Expected tests

- Tests comparing the fixed SSM curve branch assembly with the MATLAB
  `z = [z2(:,end:-1:1) z1]` structure.
- Tests covering the full trajectory overlay data and initial conditions
  `±V(:,1)*1e-4`.
- If adaptive/reference integration is added, tests must classify it as
  non-differentiable or setup/reference-only and keep differentiable RK4 tests.
- Existing PlanarSystem, BenchamrkSSM1stOrder, Lorenz, and core tests remain
  passing.

### Known risks

- Exact MATLAB `ode45` sample points are adaptive and may not match without a
  MATLAB-generated fixture or a carefully chosen SciPy reference path.
- Full `compute_whisker` object fidelity is still a larger algorithmic gap than
  can be closed by plot cleanup alone.
- Do not mark Lorenz complete unless exact remaining discrepancies are either
  fixed or explicitly justified as not applicable.

### Differentiability concerns

- Keep adaptive/reference trajectory integration outside differentiable core.
- Continue differentiating only through fixed choices: selected eigenpair,
  truncation order, graph coefficients, reduced trajectory, and fixed-step
  prediction.
- Do not claim differentiability through eigenpair selection, adaptive solver
  step selection, or MATLAB object workflow decisions.

### Acceptance criteria

- `examples/lorenz_1st_order/README.md` reflects any new plot or trajectory
  fidelity correction.
- The Lorenz notebook plot matches the MATLAB intent as closely as reasonably
  possible in plotted variables, branches, time range, initial conditions,
  labels, legend, and view.
- Any remaining Lorenz discrepancy names the precise missing dependency or
  numerical ambiguity.
- All migrated example scripts run explicitly.
- All migrated notebooks execute with `nbconvert`, or any execution blocker is
  documented precisely.
- `python -m compileall src tests examples` and `python -m pytest` pass.

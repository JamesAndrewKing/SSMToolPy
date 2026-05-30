# Migration Plan

## Strategy

This is a clean-room, example-first restart. `SSMTool/` is the only
implementation reference. The Python port should grow only through examples,
not through speculative MATLAB API coverage.

The implemented starting target is the smallest meaningful numerical subproblem
found during inspection: the PlanarSystem graph SSM coefficients explicitly
stated in `SSMTool/examples/PlanarSystem/demo.mlx`. The current implementation
derives those coefficients through a one-master, one-transverse first-order
graph homological solve rather than hard-coding the closed-form formula.

## Repository Inspection Summary

- Core MATLAB code lives under `SSMTool/src/`.
- Main MATLAB example workflows live under `SSMTool/examples/`.
- Third-party or external code lives under `SSMTool/ext/` and is deferred unless
  an accepted example requires it.
- MATLAB live scripts are OOXML `.mlx` files. Their document XML can be read
  with `unzip -p <file>.mlx matlab/document.xml`.

## Example Ranking

Estimated complexity from smallest to largest useful regression targets:

1. `PlanarSystem/demo.mlx`: 2D first-order polynomial system with closed-form
   graph coefficients in the live script.
2. `BenchamrkSSM1stOrder/demo.mlx`: source-confirmed duplicate of
   PlanarSystem; reproduced as a named regression target.
3. `Lorenz1stOrder/demo.mlx`: 3D first-order polynomial vector field with a
   1D unstable SSM and trajectory comparison; source model, direct trajectory,
   and fixed-choice unstable SSM graph coefficient subproblems are reproduced.
4. `TwoOscillators/demo.mlx`: small 2DOF second-order oscillator with forcing.
5. `ThreeOscillators/ThreeOscillatorsBook.mlx`: small but broader nonlinear
   oscillator workflow.
6. `OscillatorChain/*`: moderate-size chain with custom nonlinear force and
   derivatives.
7. Beam and shell examples (`BernoulliBeam`, `vonKarmanBeam`,
   `vonKarmanPlate`, `vonKarmanShell`, IR variants): finite-element and tensor
   dependencies.
8. FRS examples: require ridge/trench extraction, reduced response surfaces,
   and broader continuation/plotting support.
9. COMSOL, NACA wing, pipe conveying fluid, DAE, and parametric resonance
   workflows: high dependency closure with external model data or advanced
   continuation.

## `.mlx` to Notebook Ordering

1. `PlanarSystem/demo.mlx` -> `examples/planar_system/planar_system.ipynb`
2. `BenchamrkSSM1stOrder/demo.mlx` -> `examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb`
3. `Lorenz1stOrder/demo.mlx` -> `examples/lorenz_1st_order/lorenz_1st_order.ipynb`
4. `TwoOscillators/demo.mlx`
5. Remaining low-dimensional oscillator workflows
6. FE, COMSOL, FRS, DAE, and parametric resonance workflows

## Notebook Reproduction Standard

Notebook migration is not a summary of preliminary setup cells. A notebook may
be marked complete only when it reproduces the meaningful numerical and visual
workflow of the corresponding MATLAB `.mlx` file as closely as reasonably
possible in Python/JAX.

For SSMTool workflows this generally includes:

- source model definition,
- parameter setup,
- SSM reduction or SSM graph computation,
- reduced dynamics or reduced prediction,
- trajectory computation or simulation,
- visualization corresponding to the MATLAB output,
- numerical tests for the computational core outside the notebook.

SSM graph computation, reduced dynamics, trajectory computation, and
visualization must not be skipped unless there is a documented hard blocker
with a precise next fix. Setup-only notebooks are incomplete.

This standard applies to every MATLAB example, demo, and `.mlx` workflow in
the inventory. PlanarSystem and BenchamrkSSM1stOrder currently count as
substantive partial reproductions because they implement and test SSM graph
coefficient computations from the live scripts. Lorenz1stOrder currently counts
as a substantive partial reproduction because it implements and tests direct
trajectory computation plus a fixed-choice unstable SSM graph coefficient solve;
it remains incomplete until reduced-to-full trajectory comparison and
visualization are implemented.

## Completed PlanarSystem Dependency Closure

- `SSMTool/examples/PlanarSystem/build_model.m`
- `SSMTool/examples/PlanarSystem/demo.mlx`

The MATLAB live script also calls `DynamicalSystem`, `SSM`,
`linear_spectral_analysis`, `choose_E`, and `compute_whisker`. The implemented
dependency closure remains intentionally reduced to the scalar graph coefficient
subproblem, with the homological-equation form checked against:

- `SSMTool/src/@Manifold/private/Aut_1stOrder_SSM.m`
- `SSMTool/src/@Manifold/private/Aut_1stOrder_RedDyn.m`
- `SSMTool/src/@Manifold/private/coeffs_setup.m`
- `SSMTool/src/@Manifold/private/multi_nsumk.m`

The live script states the expected formula:

`a_k = 1 / (sqrt(24) - k)` for `k = 2, 3, 4, 5`; all higher coefficients used
in the script are zero.

## Completed Benchmark Duplicate Closure

- `SSMTool/examples/BenchamrkSSM1stOrder/build_model.m`
- `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx`
- `SSMTool/examples/PlanarSystem/build_model.m`
- `SSMTool/examples/PlanarSystem/demo.mlx`

`BenchamrkSSM1stOrder/build_model.m` is source-equivalent to
`PlanarSystem/build_model.m`. Its live script includes the same vector field,
the same analytical coefficients, and an explicit coefficient-difference check
for `coeffs(2,2:5)`.

## Completed Lorenz First Subproblem Closure

- `SSMTool/examples/Lorenz1stOrder/build_model.m`
- `SSMTool/examples/Lorenz1stOrder/lorenz.m`
- `SSMTool/examples/Lorenz1stOrder/demo.mlx`

The implemented Lorenz closure covers the source model
`B z_dot = A z + F(z)`, the MATLAB vector-field formula, the standard
parameters `sigma=10`, `rho=28`, `beta=8/3`, the linear eigenvalues stated in
the live script, direct fixed-step Lorenz trajectory computation, and a
fixed-choice unstable SSM graph coefficient solve through order 3 with
invariance residual checks. Reduced dynamics trajectory generation,
`reduced_to_full_traj`, reduced/full trajectory comparison, and corresponding
SSM/full trajectory visualization remain incomplete.

## Current Python Skeleton

- `src/ssmtoolpy/__init__.py`
- `src/ssmtoolpy/core/multiindex.py`
- `src/ssmtoolpy/core/polynomial.py`
- `src/ssmtoolpy/core/invariance.py`
- `examples/planar_system/planar.py`
- `examples/benchmark_ssm_1st_order/benchmark.py`
- `examples/lorenz_1st_order/lorenz.py`
- `examples/planar_system/example.py`
- `examples/benchmark_ssm_1st_order/example.py`
- `examples/lorenz_1st_order/example.py`
- `tests/test_planar_system.py`
- `tests/test_core_graph_solver.py`
- `tests/test_benchmark_ssm_1st_order.py`
- `tests/test_lorenz_1st_order.py`
- `tests/test_parameter_to_loss.py`
- `examples/planar_system/planar_system.ipynb`
- `examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb`
- `examples/lorenz_1st_order/lorenz_1st_order.ipynb`

## Example Layout

Each reproduced MATLAB example or `.mlx` workflow should live under
`examples/<example_name>/` with its executable script, README, and notebook
colocated:

```text
examples/<example_name>/
  README.md
  example.py
  <example_name>.ipynb
```

Reusable differentiable numerical code remains under `src/ssmtoolpy/`.
Example-local helpers, model definitions, plotting, fixtures, and notebook
support code live inside the example directory when they are only used by that
example. `ssmtoolpy` must not import from `examples`.

Current classification:

- `examples/planar_system/planar.py`: PlanarSystem-specific model and graph
  helper code.
- `examples/benchmark_ssm_1st_order/benchmark.py`: BenchamrkSSM1stOrder-specific
  duplicate coefficient helper.
- `examples/lorenz_1st_order/lorenz.py`: Lorenz1stOrder-specific model,
  trajectory, and fixed-choice graph helper code.
- `src/ssmtoolpy/core/`: reusable polynomial, multi-index, and graph-solve
  kernels used by example-local helpers and tests.

## Testing Strategy

- Import and shape tests for the PlanarSystem matrices.
- Correctness tests for the polynomial vector field from MATLAB sparse tensor
  entries.
- Regression tests for solver-derived graph coefficients against the closed-form formula.
- Multi-index generation tests for the one- and two-dimensional cases used near term.
- Sparse monomial polynomial evaluation and coefficient collection tests.
- Invariance residual test for the graph polynomial.
- JAX transform tests using `jax.jacfwd`, `jax.grad`, and `jax.jit`.
- Source-derived duplicate workflow tests for `BenchamrkSSM1stOrder`.
- Lorenz vector-field, model-matrix, nonlinear-term, eigenvalue, trajectory,
  fixed-choice SSM graph coefficient, invariance residual, and differentiability
  tests.

## Differentiability Strategy

- Keep numerical functions pure and array-oriented.
- Mark polynomial vector fields and polynomial evaluation as differentiable.
- Mark coefficient formulas with possible resonant denominators as
  differentiable under nondegeneracy assumptions.
- Avoid SciPy and plotting in differentiable core code.
- Prioritize differentiability of fixed-structure parameter-to-loss workflows:
  `system parameter -> system definition -> fixed SSM coefficient computation ->
  reduced prediction -> scalar loss -> gradient`.
- Separate non-differentiable setup/classification from differentiable numerical
  phases. Mode selection, truncation order, multi-index sets, resonance
  classification, and normalization conventions should be chosen outside the
  differentiated path until a representative example proves a safe
  piecewise-differentiable treatment.
- For each differentiable workflow target, add a pytest case that returns a
  finite scalar loss, runs `jax.grad`, checks gradient shape and finiteness, and
  compares with a finite-difference gradient where the problem is small and
  deterministic.
- The current minimal smoke test is the PlanarSystem fixed-structure path:
  transverse decay parameter -> scalar graph coefficients -> graph prediction
  -> mean-square loss. It does not yet cover the full adaptive SSM workflow.

## Parameter-to-Loss Roadmap

1. Keep the PlanarSystem fixed-structure graph loss as the smallest
   differentiability sentinel.
2. `Lorenz1stOrder` now has a parameterized source model, vector field, direct
   trajectory helper, and fixed-choice SSM graph coefficient solve so system
   parameters flow into deterministic numerical outputs and a fixed SSM graph.
3. Next, add reduced-coordinate trajectory lifting so a small Lorenz
   parameter-to-loss test can differentiate through
   `parameter -> fixed graph coefficients -> lifted reduced prediction -> loss`.
4. Only after fixed choices are tested, document which adaptive choices remain
   setup-only or piecewise differentiable.

## Known Risks

- Full MATLAB SSM coefficient computation is not yet ported.
- `.mlx` files may contain output-only context not present in source `.m` files.
- Tensor coefficient normalization conventions must be checked carefully for
  future examples with symmetric tensors or repeated indices.
- Full parameter-to-loss gradients can be invalid or discontinuous when
  eigenvalues collide, selected modes change, resonances appear/disappear, or
  homological-equation denominators become singular.

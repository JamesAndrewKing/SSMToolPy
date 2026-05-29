# Migration Plan

## Strategy

This is a clean-room, example-first restart. `SSMTool/` is the only
implementation reference. The Python port should grow only through examples,
not through speculative MATLAB API coverage.

The initial implementation target is the smallest meaningful numerical
subproblem found during inspection: the PlanarSystem graph SSM coefficients
explicitly stated in `SSMTool/examples/PlanarSystem/demo.mlx`.

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
2. `BenchamrkSSM1stOrder/demo.mlx`: appears to use the same model as
   PlanarSystem; useful as a duplicate-name regression after PlanarSystem.
3. `Lorenz1stOrder/demo.mlx`: 3D first-order polynomial vector field with a
   1D unstable SSM and trajectory comparison.
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

1. `PlanarSystem/demo.mlx` -> `notebooks/planar_system.ipynb`
2. `BenchamrkSSM1stOrder/demo.mlx`
3. `Lorenz1stOrder/demo.mlx`
4. `TwoOscillators/demo.mlx`
5. Remaining low-dimensional oscillator workflows
6. FE, COMSOL, FRS, DAE, and parametric resonance workflows

## Current Target Dependency Closure

- `SSMTool/examples/PlanarSystem/build_model.m`
- `SSMTool/examples/PlanarSystem/demo.mlx`

The MATLAB live script also calls `DynamicalSystem`, `SSM`,
`linear_spectral_analysis`, `choose_E`, and `compute_whisker`. For this first
batch, the dependency closure is reduced to the explicit graph coefficient
subproblem because the live script states the expected formula:

`a_k = 1 / (sqrt(24) - k)` for `k = 2, 3, 4, 5`; all higher coefficients used
in the script are zero.

## Minimal Python Skeleton

- `src/ssmtoolpy/__init__.py`
- `src/ssmtoolpy/systems/planar.py`
- `examples/planar_system.py`
- `tests/test_planar_system.py`
- `notebooks/planar_system.ipynb`

## Testing Strategy

- Import and shape tests for the PlanarSystem matrices.
- Correctness tests for the polynomial vector field from MATLAB sparse tensor
  entries.
- Regression tests for the closed-form graph coefficients.
- Invariance residual test for the graph polynomial.
- JAX transform tests using `jax.jacfwd`, `jax.grad`, and `jax.jit`.

## Differentiability Strategy

- Keep numerical functions pure and array-oriented.
- Mark polynomial vector fields and polynomial evaluation as differentiable.
- Mark coefficient formulas with possible resonant denominators as
  differentiable under nondegeneracy assumptions.
- Avoid SciPy and plotting in differentiable core code.

## Known Risks

- Full MATLAB SSM coefficient computation is not yet ported.
- `.mlx` files may contain output-only context not present in source `.m` files.
- Tensor coefficient normalization conventions must be checked carefully for
  future examples with symmetric tensors or repeated indices.

# SSMToolPy

Clean-room Python/JAX port of selected numerical kernels from MATLAB SSMTool.

This restart is example-driven. The current reproduced target is the numerical
core of `SSMTool/examples/PlanarSystem/demo.mlx`: graph-style SSM coefficients
for a two-dimensional first-order polynomial system, now computed by a minimal
first-order graph homological-equation solver.

## Install

```bash
python -m pip install -e ".[test]"
```

## Quickstart

```python
from ssmtoolpy.systems.planar import planar_ssm_graph_coefficients

coefficients = planar_ssm_graph_coefficients(order=8)
print(coefficients[2:6])
```

## Reproduced Examples

- `PlanarSystem`: numerical core from `demo.mlx`, covered by
  `examples/planar_system/example.py`, `tests/test_planar_system.py`, and
  `tests/test_core_graph_solver.py`.
- `BenchamrkSSM1stOrder`: source-confirmed duplicate of `PlanarSystem`, covered
  by `examples/benchmark_ssm_1st_order/example.py` and
  `tests/test_benchmark_ssm_1st_order.py`.
- `Lorenz1stOrder`: partial workflow from `demo.mlx`, now including source
  model, vector field, direct trajectory computation, and a fixed-choice
  unstable SSM graph coefficient subproblem covered by
  `examples/lorenz_1st_order/example.py` and `tests/test_lorenz_1st_order.py`.

## Notebook Status

- `examples/planar_system/planar_system.ipynb`: partial workflow migration,
  backed by tests for the coefficient core.
- `examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb`: partial
  workflow migration, backed by tests for the coefficient comparison.
- `examples/lorenz_1st_order/lorenz_1st_order.ipynb`: incomplete workflow
  migration. It currently covers setup, eigenvalues, fixed-choice SSM graph
  coefficients, invariance residual checks, and direct trajectory computation,
  but not reduced-to-full trajectory mapping or the corresponding SSM/full
  visualization from the MATLAB live script.

## Testing

```bash
python -m compileall src tests examples
python -m pytest
```

## Differentiability Summary

- `planar_vector_field`: differentiable polynomial JAX function.
- `planar_ssm_graph_coefficients`: differentiable under nonresonance
  assumptions, excluding active denominators `sqrt(24) - k = 0`.
- `evaluate_planar_ssm_graph`: differentiable polynomial evaluation.
- `evaluate_monomial_polynomial`: differentiable for fixed exponents.
- `solve_scalar_graph_coefficients`: differentiable under nonresonance
  assumptions.
- `lorenz_vector_field`: differentiable polynomial JAX function.
- `lorenz_rk4_trajectory`: differentiable fixed-time direct trajectory helper.
- `solve_lorenz_unstable_graph_coefficients`: differentiable under fixed
  eigenpair, fixed truncation, and nonresonance assumptions.
- `evaluate_lorenz_ssm_graph`: differentiable polynomial graph evaluation.
- `lorenz_linear_eigenvalues`: differentiable under eigenvalue nondegeneracy
  assumptions; currently used only for regression checks.

## Known Limitations

Only bounded, example-driven numerical subproblems have been ported. The
generic MATLAB class stack, continuation workflows, forcing workflows,
finite-element examples, and most `.mlx` notebooks remain unported or
incomplete until their substantive SSM/reduced-dynamics workflow steps are
implemented and tested.

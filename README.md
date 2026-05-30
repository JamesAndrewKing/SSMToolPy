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
  `examples/planar_system.py`, `tests/test_planar_system.py`, and
  `tests/test_core_graph_solver.py`.
- `BenchamrkSSM1stOrder`: source-confirmed duplicate of `PlanarSystem`, covered
  by `examples/benchmark_ssm_1st_order.py` and
  `tests/test_benchmark_ssm_1st_order.py`.
- `Lorenz1stOrder`: first bounded numerical core from `demo.mlx`, covered by
  `examples/lorenz_1st_order.py` and `tests/test_lorenz_1st_order.py`.

## Reproduced Notebooks

- `notebooks/planar_system.ipynb` mirrors the tested PlanarSystem numerical
  core.
- `notebooks/benchmark_ssm_1st_order.ipynb` mirrors the Benchmark coefficient
  comparison from the MATLAB live script.
- `notebooks/lorenz_1st_order.ipynb` mirrors the tested Lorenz vector-field and
  eigenvalue checks.

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
- `lorenz_linear_eigenvalues`: differentiable under eigenvalue nondegeneracy
  assumptions; currently used only for regression checks.

## Known Limitations

Only the first bounded PlanarSystem numerical subproblem has been ported. The
solver is intentionally limited to a one-master, one-transverse graph case. The
MATLAB class stack, continuation workflows, forcing workflows, finite-element
examples, and most `.mlx` notebooks remain unported.

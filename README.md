# SSMToolPy

Clean-room Python/JAX port of selected numerical kernels from MATLAB SSMTool.

This restart is example-driven. The current reproduced target is the numerical
core of `SSMTool/examples/PlanarSystem/demo.mlx`: graph-style SSM coefficients
for a two-dimensional first-order polynomial system.

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
  `examples/planar_system.py` and `tests/test_planar_system.py`.

## Reproduced Notebooks

- `notebooks/planar_system.ipynb` mirrors the tested PlanarSystem numerical
  core.

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

## Known Limitations

Only the first bounded PlanarSystem numerical subproblem has been ported. The
MATLAB class stack, continuation workflows, forcing workflows, finite-element
examples, and most `.mlx` notebooks remain unported.

# SSMToolPy

Python/JAX migration of selected low-level kernels from the MATLAB `SSMTool/`
reference implementation.

The MATLAB tree is kept immutable. Python code lives under `src/ssmtoolpy/`,
tests under `tests/`, and migration notes under `docs/`.

## Install

```bash
python -m pip install -e ".[test]"
```

## Quickstart

```python
import jax.numpy as jnp
from ssmtoolpy import MultiIndexPolynomial, expand_multiindex

poly = MultiIndexPolynomial(
    coeffs=jnp.array([[2.0, -1.0]]),
    ind=jnp.array([[2, 0], [1, 1]]),
)
points = jnp.array([[2.0], [3.0]])
value = expand_multiindex(poly, points)
```

## Tests

```bash
pytest
```

Representative tests cover correctness plus `jax.jit`, `jax.grad`,
`jax.jacfwd`, and `jax.vmap` on ported differentiable APIs.

## Current Coverage

Ported:

- multi-index enumeration, conversion, addition/subtraction, and ordering
- multi-index partition enumeration for Manifold coefficient assembly
- coefficient ordering/reconstruction helpers used by Manifold coefficient assembly
- multi-index polynomial evaluation and derivatives
- dense tensor polynomial evaluation and derivatives
- Khatri-Rao product
- FRC `frc_ab`
- basic reduced-to-full reconstruction maps

Not yet ported:

- high-level `DynamicalSystem`, `Manifold`, and `SSM` classes
- COCO continuation workflows
- sparse Tensor Toolbox-compatible storage
- most examples and plotting/solution-reader utilities

See `docs/migration_report.md` and `docs/differentiability.md` for details.

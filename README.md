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
- standalone FRC algebra, phase, stability, contour parsing, and grid fixed-point helpers
- functional 2D and 2mD SSM reduced-dynamics kernels in polar and Cartesian coordinates
- miscellaneous reconstruction, output, reduced-dynamics, projection-objective, block-diagonal, and linear-solve helpers
- functional dynamical-system nonlinear force, forcing, residual, ODE RHS, and mechanical first-order conversion kernels
- dynamical-system private helper equivalents for mechanical matrices, nonlinear/forcing metadata, callable input probing, and second-order to first-order nonlinear/forcing conversion
- lex/revlex Manifold coefficient composition and mixed-term algebra
- Manifold computation-type classification and autonomous/non-autonomous resonance detection helpers
- functional Manifold master-subspace selection and inner/outer resonance analysis
- autonomous Manifold invariance residual and 2D/4D sampling-error helpers
- non-autonomous Manifold harmonic conjugacy reduction and coefficient-structure setup helpers
- non-autonomous Manifold coefficient insertion/update helper
- non-autonomous Manifold mixed `W1 R0 + W0 R1` coefficient algebra
- revlex Manifold non-intrusive and semi-intrusive force-composition helpers
- revlex Manifold non-autonomous Jacobian force-composition helpers
- intrusive Manifold multi-index force-composition and Jacobian-action helpers
- non-autonomous Manifold first-order leading forcing and one-order invariance
  solve helpers
- MATLAB option defaults and validation as Python dataclasses
- multi-index polynomial evaluation and derivatives
- dense tensor polynomial evaluation and derivatives
- dense and JAX `BCOO` sparse tensor Tucker-style products and row-pattern tensor composition
- first- and second-order linear frequency-response kernels
- autonomous reduced-dynamics assembly, fixed-step transient SSM trajectories, and autonomous polar symbolic rendering
- autonomous first- and second-order one-order SSM cohomological solve kernels
- MATLAB `.mat` solution readers for SSM EP/PO/torus payloads and saved
  numerical-integration/periodic-orbit initial-condition fixtures
- Khatri-Rao product
- FRC `frc_ab`
- basic reduced-to-full reconstruction maps

Not yet ported:

- high-level `DynamicalSystem`, `Manifold`, and `SSM` classes
- COCO continuation workflows
- external COCO `po_read_solution`/`tor_read_solution` payload readers
- full sparse Tensor Toolbox-compatible storage beyond the current JAX `BCOO`
  tensor-composition adapters
- most examples, plotting utilities, and external COCO solution-reader payloads

See `docs/migration_report.md` and `docs/differentiability.md` for details.

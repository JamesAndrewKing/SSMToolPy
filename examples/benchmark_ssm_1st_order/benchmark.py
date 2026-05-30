"""BenchmarkSSM1stOrder example-local helpers.

MATLAB reference files:
- ``SSMTool/examples/BenchamrkSSM1stOrder/build_model.m``
- ``SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx``
"""

from __future__ import annotations

import jax.numpy as jnp

from ssmtoolpy.core.invariance import solve_scalar_graph_coefficients
from ssmtoolpy.core.polynomial import collect_univariate_coefficients


Array = jnp.ndarray


def benchmark_ssm_graph_coefficients(
    order: int, decay: Array | float = jnp.sqrt(24.0)
) -> Array:
    """Return the benchmark graph coefficients stated in the MATLAB live script.

    ``BenchamrkSSM1stOrder`` is source-equivalent to ``PlanarSystem`` but keeps
    its own example-local helper so example-specific model code does not live in
    ``src/ssmtoolpy``.

    Differentiability: differentiable under nondegeneracy assumptions. The
    denominator requires ``decay`` not to equal any active degree 2 through 5.
    """

    exponents = jnp.array([[2], [3], [4], [5]], dtype=jnp.int32)
    forcing = collect_univariate_coefficients(
        exponents, jnp.ones(4, dtype=jnp.asarray(decay).dtype), order
    )
    return solve_scalar_graph_coefficients(-1.0, -decay, forcing)

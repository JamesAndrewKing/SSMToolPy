"""PlanarSystem example from the MATLAB SSMTool live script.

MATLAB reference files:
- ``SSMTool/examples/PlanarSystem/build_model.m``
- ``SSMTool/examples/PlanarSystem/demo.mlx``
"""

from __future__ import annotations

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp


Array = jnp.ndarray


def build_planar_system() -> tuple[Array, Array]:
    """Return the linear matrices for the PlanarSystem example.

    The MATLAB model is ``B z_dot = A z + F(z)`` with ``B = I`` and
    ``A = diag(-1, -sqrt(24))``.

    Differentiability: differentiable. The returned constants are JAX arrays.
    """

    a = jnp.array([[-1.0, 0.0], [0.0, -jnp.sqrt(24.0)]])
    b = jnp.eye(2)
    return a, b


def planar_vector_field(z: Array) -> Array:
    """Evaluate the PlanarSystem vector field.

    For ``z = [x, y]``, the MATLAB tensors define
    ``x_dot = -x`` and
    ``y_dot = -sqrt(24) y + x**2 + x**3 + x**4 + x**5``.

    Differentiability: differentiable. This is a polynomial JAX function.
    """

    z = jnp.asarray(z)
    x = z[0]
    y = z[1]
    return jnp.array(
        [
            -x,
            -jnp.sqrt(24.0) * y + x**2 + x**3 + x**4 + x**5,
        ]
    )


def planar_ssm_graph_coefficients(
    order: int, decay: Array | float = jnp.sqrt(24.0)
) -> Array:
    """Return graph SSM coefficients ``a_k`` for ``y = sum a_k x**k``.

    The PlanarSystem demo states that the first mode is used as master mode
    and the reduced dynamics is linear. The invariance equation gives
    ``a_k = 1 / (decay - k)`` for ``k = 2, 3, 4, 5`` and ``a_k = 0`` for
    all other degrees in this example.

    The returned array has length ``order + 1`` so index ``k`` stores ``a_k``.

    Differentiability: differentiable under nondegeneracy assumptions. The
    denominator requires ``decay`` not to equal any active degree 2 through 5.
    """

    if order < 0:
        raise ValueError("order must be nonnegative")

    degrees = jnp.arange(order + 1, dtype=jnp.float64)
    active = (degrees >= 2.0) & (degrees <= 5.0)
    return jnp.where(active, 1.0 / (decay - degrees), 0.0)


def evaluate_planar_ssm_graph(x: Array, coefficients: Array) -> Array:
    """Evaluate ``y = sum_k coefficients[k] * x**k``.

    Differentiability: differentiable. This is a finite polynomial evaluation
    with fixed coefficient length.
    """

    x = jnp.asarray(x)
    coefficients = jnp.asarray(coefficients)
    degrees = jnp.arange(coefficients.shape[0], dtype=coefficients.dtype)
    return jnp.sum(coefficients * x[..., None] ** degrees, axis=-1)

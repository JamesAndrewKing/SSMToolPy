"""PlanarSystem example from the MATLAB SSMTool live script.

MATLAB reference files:
- ``SSMTool/examples/PlanarSystem/build_model.m``
- ``SSMTool/examples/PlanarSystem/demo.mlx``
"""

from __future__ import annotations

import jax.numpy as jnp

from ssmtoolpy.core.invariance import solve_scalar_graph_coefficients
from ssmtoolpy.core.polynomial import (
    collect_univariate_coefficients,
    evaluate_monomial_polynomial,
)


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
    a, _ = build_planar_system()
    nonlinear = evaluate_monomial_polynomial(
        z, planar_nonlinear_exponents(), planar_nonlinear_coefficients()
    )
    return a @ z + nonlinear


def planar_nonlinear_exponents() -> Array:
    """Return sparse nonlinear exponents from MATLAB ``build_model.m``.

    Each row is an exponent vector for ``[x, y]``. The MATLAB sparse tensors
    set only powers ``x**2`` through ``x**5`` in the second output equation.

    Differentiability: not differentiable. This returns integer exponents.
    """

    return jnp.array([[2, 0], [3, 0], [4, 0], [5, 0]], dtype=jnp.int32)


def planar_nonlinear_coefficients() -> Array:
    """Return sparse nonlinear coefficients from MATLAB ``build_model.m``.

    The shape is ``(outputs, terms)``. The first output has no nonlinear terms;
    the second output has unit coefficients for ``x**2`` through ``x**5``.

    Differentiability: differentiable. The returned constants are JAX arrays.
    """

    return jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
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

    a, _ = build_planar_system()
    exponents = planar_nonlinear_exponents()[:, :1]
    forcing = collect_univariate_coefficients(
        exponents, planar_nonlinear_coefficients()[1], order
    )
    return solve_scalar_graph_coefficients(a[0, 0], -decay, forcing)


def evaluate_planar_ssm_graph(x: Array, coefficients: Array) -> Array:
    """Evaluate ``y = sum_k coefficients[k] * x**k``.

    Differentiability: differentiable. This is a finite polynomial evaluation
    with fixed coefficient length.
    """

    x = jnp.asarray(x)
    coefficients = jnp.asarray(coefficients)
    degrees = jnp.arange(coefficients.shape[0], dtype=coefficients.dtype)
    return jnp.sum(coefficients * x[..., None] ** degrees, axis=-1)

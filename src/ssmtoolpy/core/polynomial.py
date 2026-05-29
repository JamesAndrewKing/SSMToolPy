"""Sparse monomial polynomial utilities for source-derived examples."""

from __future__ import annotations

import jax.numpy as jnp


Array = jnp.ndarray


def evaluate_monomial_polynomial(
    x: Array, exponents: Array, coefficients: Array
) -> Array:
    """Evaluate a vector polynomial represented by sparse monomial terms.

    Parameters
    ----------
    x:
        Point with shape ``(dimension,)``.
    exponents:
        Integer exponent matrix with shape ``(terms, dimension)``.
    coefficients:
        Coefficient matrix with shape ``(outputs, terms)``.

    Returns
    -------
    Array
        Polynomial value with shape ``(outputs,)``.

    Differentiability: differentiable with respect to ``x`` and
    ``coefficients`` for fixed integer exponents.
    """

    x = jnp.asarray(x)
    exponents = jnp.asarray(exponents)
    coefficients = jnp.asarray(coefficients)
    monomials = jnp.prod(x[None, :] ** exponents, axis=1)
    return coefficients @ monomials


def collect_univariate_coefficients(
    exponents: Array, coefficients: Array, order: int
) -> Array:
    """Collect sparse one-variable monomial terms into degree coefficients.

    ``exponents`` has shape ``(terms, 1)`` and ``coefficients`` has shape
    ``(terms,)``. The returned array has length ``order + 1`` so index ``k``
    stores the total coefficient multiplying ``x**k``.

    Differentiability: differentiable with respect to ``coefficients`` for
    fixed exponents and order.
    """

    if order < 0:
        raise ValueError("order must be nonnegative")

    exponents = jnp.asarray(exponents)
    coefficients = jnp.asarray(coefficients)
    if exponents.ndim != 2 or exponents.shape[1] != 1:
        raise ValueError("exponents must have shape (terms, 1)")
    if coefficients.ndim != 1:
        raise ValueError("coefficients must have shape (terms,)")
    if exponents.shape[0] != coefficients.shape[0]:
        raise ValueError("exponents and coefficients must have the same number of terms")

    degrees = jnp.arange(order + 1, dtype=exponents.dtype)
    return jnp.sum(jnp.where(exponents[:, 0] == degrees[:, None], coefficients, 0.0), axis=1)

"""Graph-parameterization helpers for reduced SSM workflows."""

from __future__ import annotations

import jax.numpy as jnp


Array = jnp.ndarray


def evaluate_univariate_graph(reduced_coordinate: Array, coefficients: Array) -> Array:
    """Evaluate ``sum_k coefficients[k] * p**k`` for a 1D reduced coordinate.

    ``coefficients[k]`` may be scalar or vector-valued. If coefficients have
    shape ``(order + 1, n)``, the result has shape ``(..., n)``.

    Differentiability: differentiable with respect to reduced coordinates and
    coefficients for fixed coefficient shape.
    """

    reduced_coordinate = jnp.asarray(reduced_coordinate)
    coefficients = jnp.asarray(coefficients)
    degrees = jnp.arange(coefficients.shape[0], dtype=coefficients.dtype)
    powers = reduced_coordinate[..., None] ** degrees
    return powers @ coefficients


def linear_reduced_trajectory(
    initial_reduced_coordinate: Array | float,
    times: Array,
    eigenvalue: Array | float,
) -> Array:
    """Evaluate linear reduced dynamics ``p(t) = p(0) exp(lambda t)``.

    Differentiability: differentiable with respect to the initial coordinate
    and eigenvalue for fixed ``times``.
    """

    times = jnp.asarray(times)
    if times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")
    return jnp.asarray(initial_reduced_coordinate) * jnp.exp(eigenvalue * times)


def two_sided_graph_curve(
    times: Array,
    amplitude: Array | float,
    eigenvalue: Array | float,
    coefficients: Array,
) -> Array:
    """Assemble a two-sided graph curve from positive and negative amplitudes.

    The negative branch is reversed and concatenated with the positive branch,
    matching the curve assembly pattern in ``Lorenz1stOrder/demo.mlx``.

    Differentiability: differentiable for fixed ``times`` and coefficient
    shape.
    """

    positive_reduced = linear_reduced_trajectory(amplitude, times, eigenvalue)
    negative_reduced = linear_reduced_trajectory(-amplitude, times, eigenvalue)
    positive = evaluate_univariate_graph(positive_reduced, coefficients)
    negative = evaluate_univariate_graph(negative_reduced, coefficients)
    return jnp.concatenate([negative[::-1], positive], axis=0)

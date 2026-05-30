"""Minimal autonomous invariance-equation solvers."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import vmap

from ssmtoolpy.core.graph import evaluate_univariate_graph


Array = jnp.ndarray


def solve_scalar_graph_coefficients(
    master_eigenvalue: Array | float,
    transverse_eigenvalue: Array | float,
    forcing_coefficients: Array,
) -> Array:
    """Solve scalar graph coefficients for a 1D autonomous SSM graph.

    For a first-order system

    ``x_dot = lambda x``
    ``y_dot = mu y + sum f_k x**k``

    and graph ``y = sum a_k x**k``, the nonresonant homological equation is

    ``(k * lambda - mu) a_k = f_k``.

    This is the one-master, one-transverse specialization of the matrix solve
    in MATLAB ``Aut_1stOrder_SSM.m``.

    Differentiability: differentiable under nondegeneracy assumptions. The
    denominators ``k * lambda - mu`` must be nonzero for active forcing terms.
    """

    forcing_coefficients = jnp.asarray(forcing_coefficients)
    degrees = jnp.arange(forcing_coefficients.shape[0], dtype=forcing_coefficients.dtype)
    denominators = degrees * master_eigenvalue - transverse_eigenvalue
    return forcing_coefficients / denominators


def solve_autonomous_quadratic_graph_coefficients(
    linear_matrix: Array,
    eigenvalue: Array | float,
    eigenvector: Array,
    order: int,
    quadratic_term: Callable[[Array, Array], Array],
) -> Array:
    """Solve fixed-choice vector graph coefficients with quadratic nonlinearity.

    The parameterization is ``W(p) = sum_k W[k] p**k`` with fixed scalar
    reduced dynamics ``p_dot = eigenvalue * p``. Coefficients solve

    ``(k * eigenvalue * I - A) W[k] = RHS[k]``

    where ``RHS[k]`` is the degree-``k`` coefficient assembled from the
    supplied bilinear ``quadratic_term``. This is a small reusable slice of the
    nonresonant first-order autonomous solve in MATLAB
    ``Aut_1stOrder_SSM.m``.

    Differentiability: differentiable under nondegeneracy assumptions for fixed
    eigenvalue, eigenvector, order, quadratic term, and nonsingular homological
    systems.
    """

    if order < 1:
        raise ValueError("order must be at least 1")

    linear_matrix = jnp.asarray(linear_matrix)
    eigenvector = jnp.asarray(eigenvector)
    dimension = linear_matrix.shape[0]
    coefficients = [jnp.zeros(dimension, dtype=linear_matrix.dtype), eigenvector]
    identity = jnp.eye(dimension, dtype=linear_matrix.dtype)

    for degree in range(2, order + 1):
        rhs = jnp.zeros(dimension, dtype=linear_matrix.dtype)
        for left_degree in range(1, degree):
            right_degree = degree - left_degree
            rhs = rhs + quadratic_term(
                coefficients[left_degree], coefficients[right_degree]
            )
        matrix = degree * eigenvalue * identity - linear_matrix
        coefficients.append(jnp.linalg.solve(matrix, rhs))

    return jnp.stack(coefficients)


def univariate_graph_invariance_residual(
    reduced_coordinate: Array,
    eigenvalue: Array | float,
    coefficients: Array,
    vector_field: Callable[[Array], Array],
) -> Array:
    """Evaluate ``DW(p) * eigenvalue * p - f(W(p))`` for a 1D graph.

    ``coefficients[k]`` stores the full-space coefficient multiplying
    ``p**k`` in ``W(p)``. The supplied ``vector_field`` must be a pure function
    from full states to full vector-field values.

    Differentiability: differentiable with respect to reduced coordinates,
    eigenvalue, coefficients, and continuous closed-over vector-field
    parameters for fixed coefficient shape and fixed vector-field structure.
    """

    reduced_coordinate = jnp.asarray(reduced_coordinate)
    coefficients = jnp.asarray(coefficients)
    degrees = jnp.arange(coefficients.shape[0], dtype=coefficients.dtype)
    derivative_coefficients = degrees[:, None] * coefficients
    derivative = jnp.sum(
        derivative_coefficients[1:]
        * reduced_coordinate[..., None, None] ** (degrees[1:, None] - 1.0),
        axis=-2,
    )
    state = evaluate_univariate_graph(reduced_coordinate, coefficients)
    if state.ndim == 1:
        field_value = vector_field(state)
    else:
        flat_state = state.reshape((-1, state.shape[-1]))
        field_value = vmap(vector_field)(flat_state).reshape(state.shape)
    return derivative * (eigenvalue * reduced_coordinate)[..., None] - field_value

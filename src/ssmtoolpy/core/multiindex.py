"""Multi-index helpers for small example-driven coefficient solves."""

from __future__ import annotations

import jax.numpy as jnp


Array = jnp.ndarray


def multiindices_of_total_degree(dimension: int, degree: int) -> Array:
    """Return nonnegative multi-indices whose entries sum to ``degree``.

    The ordering is reverse lexicographic in the MATLAB SSMTool sense used by
    the inspected setup code: for two coordinates and degree two this gives
    ``[[2, 0], [1, 1], [0, 2]]``.

    Differentiability: not differentiable. The output is an integer index set
    determined by Python integer inputs.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if degree < 0:
        raise ValueError("degree must be nonnegative")

    def build(prefix: tuple[int, ...], remaining_dimension: int, remaining_degree: int):
        if remaining_dimension == 1:
            return [prefix + (remaining_degree,)]

        rows = []
        for value in range(remaining_degree, -1, -1):
            rows.extend(build(prefix + (value,), remaining_dimension - 1, remaining_degree - value))
        return rows

    return jnp.asarray(build((), dimension, degree), dtype=jnp.int32)

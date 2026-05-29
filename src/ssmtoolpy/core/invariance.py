"""Minimal autonomous invariance-equation solvers."""

from __future__ import annotations

import jax.numpy as jnp


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

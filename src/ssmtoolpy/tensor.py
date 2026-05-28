"""Dense tensor polynomial utilities."""

from __future__ import annotations

from string import ascii_lowercase

import jax
import jax.numpy as jnp


Array = jnp.ndarray


def khatri_rao_product(a: Array, b: Array) -> Array:
    """Compute the column-wise Kronecker product of two matrices.

    Differentiability
    -----------------
    Differentiable. Representative tests exercise ``jax.jacfwd``.
    """

    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("Both matrices must have the same number of columns")
    return (a[:, None, :] * b[None, :, :]).reshape(a.shape[0] * b.shape[0], a.shape[1])


def expand_tensor(tensor: Array, point: Array) -> Array:
    """Evaluate ``T`` contracted with ``point`` along all non-output modes.

    Differentiability
    -----------------
    Differentiable. Representative tests exercise ``jax.grad`` and ``jax.jit``.
    """

    tensor = jnp.asarray(tensor)
    point = jnp.asarray(point)
    if tensor.size == 0:
        return jnp.asarray(0, dtype=point.dtype)
    degree = tensor.ndim - 1
    if degree == 0:
        return tensor
    if degree + 1 > len(ascii_lowercase):
        raise ValueError("tensor degree too high for einsum implementation")
    tensor_labels = ascii_lowercase[: degree + 1]
    expr = f"{tensor_labels}," + ",".join(tensor_labels[1:]) + f"->{tensor_labels[0]}"
    return jnp.einsum(expr, tensor, *([point] * degree))


def expand_tensor_derivative(tensor: Array, point: Array) -> Array:
    """Evaluate the Jacobian of a tensor polynomial at ``point``.

    Differentiability
    -----------------
    Differentiable wherever the tensor polynomial derivative is defined.
    Representative tests compare against ``jax.jacfwd``.
    """

    tensor = jnp.asarray(tensor)
    point = jnp.asarray(point)
    output_dim = tensor.shape[0]
    input_dim = point.shape[0]
    degree = tensor.ndim - 1
    if tensor.size == 0:
        return jnp.zeros((output_dim, input_dim), dtype=point.dtype)
    if degree == 0:
        return jnp.zeros((output_dim, input_dim), dtype=tensor.dtype)
    del output_dim, input_dim
    return jax.jacfwd(lambda x: expand_tensor(tensor, x))(point)

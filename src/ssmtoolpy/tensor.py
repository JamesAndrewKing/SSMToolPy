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


def tensor_product(tensor: Array, factors: tuple[Array, ...]) -> Array:
    """Contract ``tensor`` with one factor along each non-output mode.

    This is the dense JAX counterpart of the private ``tensor_product`` helper
    inside MATLAB ``misc/tensor_composition.m``.  Given an ``(n+1)``-mode
    tensor ``A`` and ``n`` tensors ``B_k``, it contracts mode ``k+1`` of
    ``A`` with mode 1 of ``B_k`` and appends the remaining modes of each
    ``B_k`` to the output.  The resulting axes are ordered as
    ``(A output, B_1 trailing axes, B_2 trailing axes, ...)``.

    Differentiability
    -----------------
    Differentiable with respect to ``tensor`` and factor values for fixed
    tensor ranks and compatible shapes. Representative tests exercise
    ``jax.jacfwd``.
    """

    result = jnp.asarray(tensor)
    expected_factors = result.ndim - 1
    if len(factors) != expected_factors:
        raise ValueError(f"Expected {expected_factors} factors, received {len(factors)}")
    for factor in factors:
        factor = jnp.asarray(factor)
        if factor.ndim < 1:
            raise ValueError("Each factor must have at least one mode")
        if result.shape[1] != factor.shape[0]:
            raise ValueError(
                "Tensor contraction dimensions do not match: "
                f"{result.shape[1]} != {factor.shape[0]}"
            )
        result = jnp.tensordot(result, factor, axes=((1,), (0,)))
    return result


def tensor_composition(
    tensor: Array,
    factors: tuple[Array, ...],
    pattern: Array,
    size: tuple[int, ...] | None = None,
    *,
    index_base: int = 0,
) -> Array:
    """Sum Tucker-style tensor products selected by ``pattern`` rows.

    MATLAB ``misc/tensor_composition.m`` computes

    ``sum_j A x_2 B[p[j, 1]] x_3 ... x_{n+1} B[p[j, n]]``

    using sparse tensors. This dense JAX port accepts a tuple of dense factor
    tensors and a row-wise integer ``pattern`` selecting which factors to use
    for each contraction. Python callers should use zero-based indices; pass
    ``index_base=1`` for MATLAB-derived pattern matrices.

    ``size`` mirrors the MATLAB ``SIZE`` argument. When provided, it is checked
    against the inferred dense result shape and used to construct the zero
    output for empty patterns.

    Differentiability
    -----------------
    Differentiable with respect to ``tensor`` and selected factor values for a
    fixed ``pattern`` and fixed shapes. The integer pattern, ``size`` and
    ``index_base`` are discrete and not differentiable. Representative tests
    exercise ``jax.jit`` and ``jax.grad``.
    """

    base_tensor = jnp.asarray(tensor)
    pattern = jnp.asarray(pattern)
    if pattern.ndim != 2:
        raise ValueError("pattern must be a two-dimensional integer array")
    expected_width = base_tensor.ndim - 1
    if pattern.shape[1] != expected_width:
        raise ValueError(f"Pattern width must equal tensor order {expected_width}")
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")

    selected_rows = tuple(tuple(int(index) - index_base for index in row) for row in pattern.tolist())
    if any(index < 0 or index >= len(factors) for row in selected_rows for index in row):
        raise IndexError("pattern contains a factor index outside the available factors")

    if selected_rows:
        terms = tuple(tensor_product(base_tensor, tuple(factors[index] for index in row)) for row in selected_rows)
        result = sum(terms[1:], terms[0]) if len(terms) > 1 else terms[0]
        if size is not None and tuple(result.shape) != tuple(size):
            raise ValueError(f"Inferred result shape {result.shape} does not match requested size {size}")
        return result

    if size is None:
        raise ValueError("size is required when pattern has no rows")
    return jnp.zeros(size, dtype=base_tensor.dtype)

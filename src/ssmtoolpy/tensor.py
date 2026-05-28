"""Dense tensor polynomial utilities."""

from __future__ import annotations

from string import ascii_lowercase

import jax
from jax.experimental import sparse as jax_sparse
import jax.numpy as jnp


Array = jnp.ndarray
SparseArray = jax_sparse.BCOO


def _is_sparse_tensor(value: object) -> bool:
    return isinstance(value, jax_sparse.BCOO)


def sparse_tensor_from_dense(tensor: Array, *, nse: int | None = None) -> SparseArray:
    """Convert a dense tensor to JAX ``BCOO`` sparse storage.

    This is the Python/JAX adapter for MATLAB ``sptensor`` inputs in tensor
    composition code. It keeps sparse storage explicit at API boundaries while
    using JAX's transformable sparse array type internally.

    Differentiability
    -----------------
    Piecewise differentiable for fixed sparsity capacity. The discrete index
    structure is not differentiable; sparse contraction value paths are tested
    with ``jax.jit`` and forward-mode ``jax.jacfwd``.
    """

    return jax_sparse.BCOO.fromdense(jnp.asarray(tensor), nse=nse)


def tensor_to_dense(tensor: Array | SparseArray) -> Array:
    """Return a dense JAX array from either dense or ``BCOO`` tensor storage.

    Differentiability
    -----------------
    Differentiable with respect to stored values for fixed sparse indices.
    """

    if _is_sparse_tensor(tensor):
        return tensor.todense()
    return jnp.asarray(tensor)


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


def tensor_product(
    tensor: Array | SparseArray,
    factors: tuple[Array | SparseArray, ...],
    *,
    sparse: bool | None = None,
) -> Array | SparseArray:
    """Contract ``tensor`` with one factor along each non-output mode.

    This is the dense JAX counterpart of the private ``tensor_product`` helper
    inside MATLAB ``misc/tensor_composition.m``.  Given an ``(n+1)``-mode
    tensor ``A`` and ``n`` tensors ``B_k``, it contracts mode ``k+1`` of
    ``A`` with mode 1 of ``B_k`` and appends the remaining modes of each
    ``B_k`` to the output.  The resulting axes are ordered as
    ``(A output, B_1 trailing axes, B_2 trailing axes, ...)``.

    When ``sparse=True``, operands are converted to JAX ``BCOO`` and the result
    stays sparse. When ``sparse=None``, the result is sparse only if all operands
    already use ``BCOO`` storage.

    Differentiability
    -----------------
    Dense path: differentiable with respect to ``tensor`` and factor values for
    fixed tensor ranks and compatible shapes. Sparse path: JAX-transformable for
    fixed sparsity structure; forward-mode differentiation is supported by the
    JAX ``BCOO`` primitives used here, while reverse-mode transposes are not
    implemented for sparse-sparse ``bcoo_spdot_general`` in JAX 0.10.1.
    Representative tests exercise dense ``jax.jacfwd`` and sparse
    ``jax.jit``/``jax.jacfwd``.
    """

    use_sparse = (
        bool(sparse)
        if sparse is not None
        else _is_sparse_tensor(tensor) and all(_is_sparse_tensor(f) for f in factors)
    )
    result = sparse_tensor_from_dense(tensor) if use_sparse and not _is_sparse_tensor(tensor) else tensor
    result = result if _is_sparse_tensor(result) else jnp.asarray(result)
    expected_factors = result.ndim - 1
    if len(factors) != expected_factors:
        raise ValueError(f"Expected {expected_factors} factors, received {len(factors)}")
    for factor in factors:
        factor = sparse_tensor_from_dense(factor) if use_sparse and not _is_sparse_tensor(factor) else factor
        factor = factor if _is_sparse_tensor(factor) else jnp.asarray(factor)
        if factor.ndim < 1:
            raise ValueError("Each factor must have at least one mode")
        if result.shape[1] != factor.shape[0]:
            raise ValueError(
                "Tensor contraction dimensions do not match: "
                f"{result.shape[1]} != {factor.shape[0]}"
            )
        if use_sparse:
            result = jax_sparse.bcoo_dot_general(
                result,
                factor,
                dimension_numbers=(((1,), (0,)), ((), ())),
            )
        else:
            result = jnp.tensordot(result, factor, axes=((1,), (0,)))
    return result


def tensor_composition(
    tensor: Array | SparseArray,
    factors: tuple[Array | SparseArray, ...],
    pattern: Array,
    size: tuple[int, ...] | None = None,
    *,
    index_base: int = 0,
    sparse: bool | None = None,
) -> Array | SparseArray:
    """Sum Tucker-style tensor products selected by ``pattern`` rows.

    MATLAB ``misc/tensor_composition.m`` computes

    ``sum_j A x_2 B[p[j, 1]] x_3 ... x_{n+1} B[p[j, n]]``

    using sparse tensors. This JAX port accepts dense arrays or JAX ``BCOO``
    sparse tensors. Python callers should use zero-based indices; pass
    ``index_base=1`` for MATLAB-derived pattern matrices.

    Pass ``sparse=True`` to convert operands to ``BCOO`` and preserve sparse
    output. With ``sparse=None``, the output remains sparse when the base tensor
    and all selected factor tensors already use ``BCOO`` storage.

    ``size`` mirrors the MATLAB ``SIZE`` argument. When provided, it is checked
    against the inferred dense result shape and used to construct the zero
    output for empty patterns.

    Differentiability
    -----------------
    Dense path: differentiable with respect to ``tensor`` and selected factor
    values for a fixed ``pattern`` and fixed shapes. Sparse path:
    JAX-transformable for fixed sparsity structure; forward-mode
    differentiation is supported, but reverse-mode differentiation through
    sparse-sparse contractions is limited by JAX's current ``BCOO`` transpose
    support. The integer pattern, ``size`` and ``index_base`` are discrete and
    not differentiable. Representative tests exercise dense ``jax.jit``,
    ``jax.grad``, ``jax.vmap`` and sparse ``jax.jit``/``jax.jacfwd``.
    """

    base_tensor = tensor if _is_sparse_tensor(tensor) else jnp.asarray(tensor)
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
        use_sparse = bool(sparse) if sparse is not None else _is_sparse_tensor(base_tensor) and all(
            _is_sparse_tensor(factors[index]) for row in selected_rows for index in row
        )
        terms = tuple(
            tensor_product(base_tensor, tuple(factors[index] for index in row), sparse=use_sparse) for row in selected_rows
        )
        result = sum(terms[1:], terms[0]) if len(terms) > 1 else terms[0]
        if size is not None and tuple(result.shape) != tuple(size):
            raise ValueError(f"Inferred result shape {result.shape} does not match requested size {size}")
        return result

    if size is None:
        raise ValueError("size is required when pattern has no rows")
    if sparse:
        return sparse_tensor_from_dense(jnp.zeros(size, dtype=tensor_to_dense(base_tensor).dtype), nse=0)
    return jnp.zeros(size, dtype=base_tensor.dtype)

"""Manifold coefficient algebra kernels."""

from __future__ import annotations

from math import comb
from typing import Literal

import jax.numpy as jnp

from ssmtoolpy.multi_index import MultiIndexPolynomial, multi_addition, multi_index_2_ordering, multi_subtraction, nsumk


Array = jnp.ndarray
Ordering = Literal["lex", "revlex"]


def _multi_indices(dim: int, order: int, ordering: Ordering) -> Array:
    if dim == 1:
        return jnp.asarray([[order]], dtype=jnp.int32)
    indices = nsumk(dim, order, "nonnegative")
    rows = sorted([tuple(int(value) for value in row) for row in indices.tolist()])
    matrix = jnp.asarray(rows, dtype=jnp.int32).T
    if ordering == "revlex":
        matrix = jnp.flip(matrix, axis=1)
    return matrix


def coeffs_composition(
    w0: tuple[MultiIndexPolynomial, ...],
    h: tuple[Array, ...],
    order: int,
    ordering: Ordering = "revlex",
) -> Array:
    """Compute power-series composition coefficients for one order.

    This ports the lexicographic/reverse-lexicographic branches of
    ``@Manifold/private/coeffs_composition.m``. The conjugate-order branch is
    intentionally left for the cohomological-solver layer.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values for fixed order and
    ordering. The combinatorial indexing is discrete preprocessing.
    """

    if ordering not in {"lex", "revlex"}:
        raise NotImplementedError("Only lex and revlex composition are ported")
    dim = h[0].shape[1]
    z_k = comb(order + dim - 1, dim - 1)
    output_dim = h[0].shape[0]
    k_indices = _multi_indices(dim, order, ordering)
    h_k = jnp.zeros((output_dim, z_k, order), dtype=jnp.result_type(*(item.coeffs for item in w0), *h))

    shifted = k_indices + (order + 1) * (k_indices == 0)
    kj = jnp.min(shifted, axis=0)
    ikj = jnp.argmin(shifted, axis=0)

    for ord_value in range(1, order):
        w_coeffs = w0[ord_value - 1].coeffs
        if w_coeffs.size == 0:
            continue
        g = _multi_indices(dim, order - ord_value, ordering)
        gmink, i_k, i_g = multi_subtraction(k_indices, g, "Arbitrary")
        if gmink.shape[1] == 0:
            continue
        i_gmk = multi_index_2_ordering(gmink, ordering) - 1
        pos = ikj[i_k]
        gmk_j = gmink[pos, jnp.arange(i_k.shape[0])]
        h_prev = h[order - ord_value - 1]
        if h_prev.ndim == 2:
            h_prev = h_prev[:, :, None]
        coeff = (w_coeffs[:, i_gmk] * gmk_j[None, :])[:, :, None] * h_prev[:, i_g, :]
        for target in sorted(set(int(value) for value in i_k.tolist())):
            mask = i_k == target
            update = jnp.sum(coeff[:, mask, :], axis=1)
            h_k = h_k.at[:, target, 1 : order - ord_value + 1].add(update)

    scale = (1.0 / kj)[:, None] * jnp.arange(1, order + 1, dtype=h_k.dtype)[None, :]
    h_k = jnp.transpose(jnp.transpose(h_k, (1, 2, 0)) * scale[:, :, None], (2, 0, 1))
    if len(w0) >= order and w0[order - 1].coeffs.size:
        h_k = h_k.at[:, :, 0].set(w0[order - 1].coeffs)
    return h_k


def coeffs_mixed_terms(
    order: int,
    m_order: int,
    w: tuple[MultiIndexPolynomial, ...],
    r: tuple[MultiIndexPolynomial, ...],
    *,
    dim: int,
    output_dim: int,
    mix: Literal["R1", "W1", "aut"] = "aut",
    ordering: Ordering = "revlex",
    explicit_indices: bool = False,
) -> Array:
    """Compute mixed coefficient products in lex/revlex ordering.

    This ports the reverse-lexicographic branch of
    ``@Manifold/private/coeffs_mixed_terms.m`` and also supports lexicographic
    ordering by using the same formula with lex-ordered multi-indices.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values for fixed index
    structure. The index assembly and column matching are discrete.
    """

    if ordering not in {"lex", "revlex"}:
        raise NotImplementedError("Only lex and revlex mixed terms are ported")
    z_k = comb(order + dim - 1, dim - 1)
    result = jnp.zeros((output_dim, z_k), dtype=jnp.result_type(*(item.coeffs for item in w), *(item.coeffs for item in r)))

    if mix == "R1":
        u_order = order + 1 - m_order + 1
    elif mix == "W1":
        u_order = order + 1 - (m_order - 1)
    elif mix == "aut":
        u_order = order - m_order + 1
    else:
        raise ValueError("mix must be 'R1', 'W1', or 'aut'")

    w_poly = w[m_order - 1]
    r_poly = r[u_order - 1]
    if w_poly.coeffs.size == 0 or r_poly.coeffs.size == 0:
        return result

    k_indices = _multi_indices(dim, order, ordering)
    if explicit_indices:
        m_indices = jnp.asarray(w_poly.ind, dtype=jnp.int32).T
        u_indices = jnp.asarray(r_poly.ind, dtype=jnp.int32).T
    else:
        m_indices = _multi_indices(dim, m_order, ordering)
        u_indices = _multi_indices(dim, u_order, ordering)

    summed, i_m, i_u = multi_addition(m_indices, u_indices)
    one = jnp.eye(dim, dtype=jnp.int32)
    k_lookup = {tuple(int(value) for value in k_indices[:, col].tolist()): col for col in range(k_indices.shape[1])}

    for row in range(dim):
        shifted = summed - one[:, row : row + 1]
        keep = jnp.all(shifted >= 0, axis=0)
        shifted_keep = shifted[:, keep]
        i_m_keep = i_m[keep]
        i_u_keep = i_u[keep]
        if shifted_keep.shape[1] == 0:
            continue
        w_values = w_poly.coeffs[:, i_m_keep]
        r_factor = r_poly.coeffs[row, i_u_keep] * m_indices[row, i_m_keep]
        wr = w_values * r_factor[None, :]
        for col in range(shifted_keep.shape[1]):
            key = tuple(int(value) for value in shifted_keep[:, col].tolist())
            target = k_lookup.get(key)
            if target is not None:
                result = result.at[:, target].add(wr[:, col])
    return result

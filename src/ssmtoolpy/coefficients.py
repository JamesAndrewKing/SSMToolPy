"""Coefficient ordering and output helpers ported from Manifold private code."""

from __future__ import annotations

from math import comb
from typing import NamedTuple

import jax.numpy as jnp

from ssmtoolpy.multi_index import MultiIndexPolynomial, nsumk


Array = jnp.ndarray


class ConjugateOrdering(NamedTuple):
    """Index data for conjugate multi-index ordering.

    Indices are zero-based, unlike MATLAB's one-based arrays.

    Differentiability
    -----------------
    Not differentiable. This is discrete combinatorial preprocessing.
    """

    conj2revlex: tuple[Array, ...]
    revlex2conj: tuple[Array, ...]
    z_cci: Array
    l_i: int
    l_r: int


def number_of_multis(dim: int, max_order: int) -> Array:
    """Return the number of ``dim``-variable multi-indices at each order.

    Differentiability
    -----------------
    Not differentiable. This is discrete combinatorial bookkeeping.
    """

    return jnp.asarray([comb(order + dim - 1, dim - 1) for order in range(1, max_order + 1)], dtype=jnp.int32)


def conjugate_flip(l_i: int, l_r: int) -> Array:
    """Return the coordinate flip used by SSMTool conjugate ordering.

    Indices are zero-based in Python.

    Differentiability
    -----------------
    Not differentiable. This is a discrete index permutation.
    """

    paired = []
    for pair in range(l_i):
        paired.extend([2 * pair + 1, 2 * pair])
    paired.extend(range(2 * l_i, 2 * l_i + l_r))
    return jnp.asarray(paired, dtype=jnp.int32)


def _sort_rows(array: Array) -> Array:
    rows = [tuple(int(value) for value in row) for row in jnp.asarray(array).tolist()]
    order = sorted(range(len(rows)), key=lambda index: rows[index])
    return jnp.asarray([rows[index] for index in order], dtype=jnp.int32)


def _unique_columns_stable(array: Array) -> Array:
    columns = []
    seen = set()
    for col in range(array.shape[1]):
        key = tuple(int(value) for value in array[:, col].tolist())
        if key not in seen:
            seen.add(key)
            columns.append(array[:, col])
    if not columns:
        return jnp.zeros((array.shape[0], 0), dtype=array.dtype)
    return jnp.stack(columns, axis=1)


def _column_positions(source: Array, target: Array) -> Array:
    positions = []
    target_map = {tuple(int(value) for value in target[:, col].tolist()): col for col in range(target.shape[1])}
    for col in range(source.shape[1]):
        key = tuple(int(value) for value in source[:, col].tolist())
        positions.append(target_map[key])
    return jnp.asarray(positions, dtype=jnp.int32)


def conjugate_ordering(max_order: int, l_r: int, l_i: int) -> ConjugateOrdering:
    """Construct conjugate ordering maps from ``coeffs_setup.m``.

    For a reverse-lexicographic multi-index matrix ``K`` and conjugate ordered
    matrix ``Z``, ``K[:, revlex2conj[k]] == Z`` and
    ``Z[:, conj2revlex[k]] == K`` for order ``k + 1``.

    Differentiability
    -----------------
    Not differentiable. This is discrete ordering logic.
    """

    conj2revlex = []
    revlex2conj = []
    z_cci = []
    dim = l_r + 2 * l_i
    flip_idx = conjugate_flip(l_i, l_r)
    for order in range(1, max_order + 1):
        k_matrix = jnp.flip(_sort_rows(nsumk(dim, order, "nonnegative")).T, axis=1)
        y_matrix = k_matrix[flip_idx, :]
        exempt = jnp.all(k_matrix - y_matrix == 0, axis=0)
        y_nonexempt = y_matrix[:, ~exempt]
        nonexempt_cols = [index for index, value in enumerate(exempt.tolist()) if not value]

        interleaved = []
        for col in range(y_nonexempt.shape[1]):
            interleaved.append(k_matrix[:, nonexempt_cols[col]])
            interleaved.append(y_nonexempt[:, col])
        z_matrix = _unique_columns_stable(jnp.stack(interleaved, axis=1)) if interleaved else jnp.zeros((dim, 0), dtype=jnp.int32)

        idx_1 = list(range(0, y_nonexempt.shape[1], 2))
        idx_2 = [y_nonexempt.shape[1] - index - 1 for index in idx_1]
        center = k_matrix[:, exempt]
        left = z_matrix[:, idx_1] if idx_1 else jnp.zeros((dim, 0), dtype=jnp.int32)
        right = z_matrix[:, idx_2] if idx_2 else jnp.zeros((dim, 0), dtype=jnp.int32)
        z_matrix = jnp.concatenate([left, center, right], axis=1)

        z_cci.append(center.shape[1] + len(idx_1))
        conj2revlex.append(_column_positions(k_matrix, z_matrix))
        revlex2conj.append(_column_positions(z_matrix, k_matrix))

    return ConjugateOrdering(
        conj2revlex=tuple(conj2revlex),
        revlex2conj=tuple(revlex2conj),
        z_cci=jnp.asarray(z_cci, dtype=jnp.int32),
        l_i=l_i,
        l_r=l_r,
    )


def coeffs_conj2full(
    coeff_part: MultiIndexPolynomial | Array,
    row_idx: Array | None,
    col_idx: int,
    ordering: Array,
    kind: str,
) -> MultiIndexPolynomial | Array:
    """Reconstruct full coefficients from conjugate-ordered partial data.

    ``row_idx`` and ``ordering`` are zero-based. ``col_idx`` is a count of
    conjugate-mirrored columns, matching MATLAB's ``1:col_idx`` semantics.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values for fixed index arrays.
    Representative tests exercise value reconstruction; transform tests are not
    required because the index structure is discrete.
    """

    ordering = jnp.asarray(ordering, dtype=jnp.int32)
    if kind == "TaylorCoeffs":
        if not isinstance(coeff_part, MultiIndexPolynomial):
            raise TypeError("TaylorCoeffs expects a MultiIndexPolynomial")
        coeffs = jnp.asarray(coeff_part.coeffs)
        if row_idx is None:
            mirrored = jnp.flip(jnp.conj(coeffs[:, :col_idx, ...]), axis=1)
        else:
            mirrored = jnp.flip(jnp.conj(coeffs[jnp.asarray(row_idx, dtype=jnp.int32), :col_idx]), axis=1)
        full = jnp.concatenate([coeffs, mirrored], axis=1)
        return MultiIndexPolynomial(coeffs=full[:, ordering, ...], ind=coeff_part.ind)
    if kind == "CompCoeffs":
        coeffs = jnp.asarray(coeff_part)
        mirrored = jnp.flip(jnp.conj(coeffs[:, :col_idx, :]), axis=1)
        full = jnp.concatenate([coeffs, mirrored], axis=1)
        return full[:, ordering, :]
    raise ValueError("kind must be 'TaylorCoeffs' or 'CompCoeffs'")


def coeffs_lex2revlex(coeffs: tuple[MultiIndexPolynomial | Array, ...], kind: str) -> tuple[MultiIndexPolynomial | Array, ...]:
    """Flip coefficient columns between lexicographic and reverse-lexicographic order.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values for fixed structures.
    """

    if kind == "CompCoeff":
        return tuple(jnp.flip(jnp.asarray(item), axis=1) for item in coeffs)
    if kind == "TaylorCoeff":
        result = []
        for item in coeffs:
            if isinstance(item, MultiIndexPolynomial):
                result.append(MultiIndexPolynomial(coeffs=jnp.flip(item.coeffs, axis=1), ind=jnp.flip(item.ind, axis=0)))
            else:
                result.append(jnp.flip(jnp.asarray(item), axis=1))
        return tuple(result)
    raise ValueError("kind must be 'CompCoeff' or 'TaylorCoeff'")


def coeffs_conj2lex(
    ordering: ConjugateOrdering,
    max_order: int,
    w_0: tuple[MultiIndexPolynomial, ...],
    r_0: tuple[MultiIndexPolynomial, ...],
) -> tuple[tuple[MultiIndexPolynomial, ...], tuple[MultiIndexPolynomial, ...]]:
    """Reconstruct lexicographic autonomous SSM and reduced-dynamics coefficients.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values under fixed ordering data.
    """

    dim = ordering.l_r + 2 * ordering.l_i
    counts = number_of_multis(dim, max_order)
    row_idx = conjugate_flip(ordering.l_i, ordering.l_r)
    w_out = []
    r_out = []
    for order in range(max_order):
        col_idx = int(counts[order] - ordering.z_cci[order])
        conj2lex = jnp.flip(ordering.conj2revlex[order])
        w_out.append(coeffs_conj2full(w_0[order], None, col_idx, conj2lex, "TaylorCoeffs"))
        r_out.append(coeffs_conj2full(r_0[order], row_idx, col_idx, conj2lex, "TaylorCoeffs"))
    return tuple(w_out), tuple(r_out)


def coeffs_output(
    w_0: tuple[MultiIndexPolynomial, ...],
    r_0: tuple[MultiIndexPolynomial, ...],
    max_order: int,
) -> tuple[tuple[MultiIndexPolynomial, ...], tuple[MultiIndexPolynomial, ...]]:
    """Attach lexicographic multi-indices and drop zero coefficient columns.

    This ports ``coeffs_output.m``.

    Differentiability
    -----------------
    Not differentiable as a whole because it selects columns based on exact zero
    tests. The retained coefficient values remain JAX arrays.
    """

    dim = w_0[0].coeffs.shape[1]
    w_out = []
    r_out = []
    for order in range(1, max_order + 1):
        if dim > 1:
            indices = _sort_rows(nsumk(dim, order, "nonnegative"))
        else:
            indices = jnp.asarray([[order]], dtype=jnp.int32)

        w_coeffs = jnp.asarray(w_0[order - 1].coeffs)
        w_keep = ~jnp.all(w_coeffs == 0, axis=0)
        w_out.append(MultiIndexPolynomial(coeffs=w_coeffs[:, w_keep], ind=indices[w_keep, :]))

        r_coeffs = jnp.asarray(r_0[order - 1].coeffs)
        r_keep = jnp.any(r_coeffs != 0, axis=0)
        r_out.append(MultiIndexPolynomial(coeffs=r_coeffs[:, r_keep], ind=indices[r_keep, :]))

    return tuple(w_out), tuple(r_out)
